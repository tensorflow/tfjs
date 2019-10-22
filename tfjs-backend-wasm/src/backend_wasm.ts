/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {backend_util, DataStorage, DataType, engine, KernelBackend, registerBackend, TensorInfo, util} from '@tensorflow/tfjs-core';

import wasmFactory from '../wasm-out/tfjs-backend-wasm';
import {BackendWasmModule} from '../wasm-out/tfjs-backend-wasm';

const WASM_PRIORITY = 2;

interface TensorData {
  id: number;
  memoryOffset: number;
  shape: number[];
  dtype: DataType;
  /** Only used for string tensors, storing encoded bytes. */
  stringBytes?: Uint8Array[];
}

export type DataId = object;  // object instead of {} to force non-primitive.

export class BackendWasm extends KernelBackend {
  private dataIdNextNumber = 0;
  dataIdMap: DataStorage<TensorData>;

  constructor(public wasm: BackendWasmModule) {
    super();
    this.wasm.tfjs.init();
    this.dataIdMap = new DataStorage(this, engine());
  }

  write(values: backend_util.BackendValues, shape: number[], dtype: DataType):
      DataId {
    const dataId = {};
    this.move(dataId, values, shape, dtype);
    return dataId;
  }

  numDataIds(): number {
    return this.dataIdMap.numDataIds();
  }

  move(
      dataId: DataId, values: backend_util.BackendValues, shape: number[],
      dtype: DataType): void {
    const id = this.dataIdNextNumber++;
    if (dtype === 'string') {
      const stringBytes = values as Uint8Array[];
      this.dataIdMap.set(
          dataId, {id, stringBytes, shape, dtype, memoryOffset: null});
      return;
    }
    const numBytes = util.sizeFromShape(shape) * util.bytesPerElement(dtype);
    const memoryOffset = this.wasm._malloc(numBytes);
    this.dataIdMap.set(dataId, {id, memoryOffset, shape, dtype});
    const shapeBytes = new Uint8Array(new Int32Array(shape).buffer);
    this.wasm.tfjs.registerTensor(
        id, shapeBytes, shape.length, dtypeToEnumValue(dtype), memoryOffset);
    if (values != null) {
      this.wasm.HEAPU8.set(
          new Uint8Array((values as backend_util.TypedArray).buffer),
          memoryOffset);
    }
  }

  async read(dataId: DataId): Promise<backend_util.BackendValues> {
    return this.readSync(dataId);
  }

  readSync(dataId: DataId): backend_util.BackendValues {
    const {memoryOffset, dtype, shape, stringBytes} =
        this.dataIdMap.get(dataId);
    if (dtype === 'string') {
      return stringBytes;
    }
    const bytes = this.wasm.HEAPU8.slice(
        memoryOffset,
        memoryOffset + util.sizeFromShape(shape) * util.bytesPerElement(dtype));
    return typedArrayFromBuffer(bytes.buffer, dtype);
  }

  disposeData(dataId: DataId) {
    const data = this.dataIdMap.get(dataId);
    this.wasm._free(data.memoryOffset);
    this.wasm.tfjs.disposeData(data.id);
    this.dataIdMap.delete(dataId);
  }

  floatPrecision(): 32 {
    return 32;
  }

  // Returns the memory offset of a tensor. Useful for debugging and unit
  // testing.
  getMemoryOffset(dataId: DataId): number {
    return this.dataIdMap.get(dataId).memoryOffset;
  }

  dispose() {
    this.wasm.tfjs.dispose();
    this.wasm = null;
  }

  makeOutput(shape: number[], dtype: DataType): TensorInfo {
    const dataId = this.write(null /* values */, shape, dtype);
    return {dataId, shape, dtype};
  }

  typedArrayFromHeap(offset: number, dtype: DataType, size: number):
      backend_util.TypedArray {
    const buffer = this.wasm.HEAPU8.buffer;
    switch (dtype) {
      case 'float32':
        return new Float32Array(buffer, offset, size);
      case 'int32':
        return new Int32Array(buffer, offset, size);
      case 'bool':
        return new Uint8Array(buffer, offset, size);
      default:
        throw new Error(`Uknown dtype ${dtype}`);
    }
  }
}

registerBackend('wasm', async () => {
  const {wasm} = await init();
  return new BackendWasm(wasm);
}, WASM_PRIORITY);

/**
 * Initializes the wasm module and creates the js <--> wasm bridge.
 *
 * NOTE: We wrap the wasm module in a object with property 'wasm' instead of
 * returning Promise<BackendWasmModule> to avoid freezing Chrome (last tested in
 * Chrome 76).
 */
async function init(): Promise<{wasm: BackendWasmModule}> {
  return new Promise(resolve => {
    const wasm = wasmFactory();
    const voidReturnType: string = null;
    // Using the tfjs namespace to avoid conflict with emscripten's API.
    wasm.tfjs = {
      init: wasm.cwrap('init', null, []),
      registerTensor: wasm.cwrap(
          'register_tensor', null,
          [
            'number',  // dataId
            'array',   // shape[]
            'number',  // shapeLength
            'number',  // dtype
            'number',  // memoryOffset
          ]),
      disposeData: wasm.cwrap('dispose_data', voidReturnType, ['number']),
      dispose: wasm.cwrap('dispose', voidReturnType, []),
    };
    wasm.onRuntimeInitialized = () => resolve({wasm});
  });
}

function dtypeToEnumValue(dtype: DataType): number {
  switch (dtype) {
    case 'float32':
      return 0;
    case 'int32':
      return 1;
    case 'bool':
      return 2;
    default:
      throw new Error(`Uknown dtype ${dtype}`);
  }
}

function typedArrayFromBuffer(
    buffer: ArrayBuffer, dtype: DataType): backend_util.TypedArray {
  switch (dtype) {
    case 'float32':
      return new Float32Array(buffer);
    case 'int32':
      return new Int32Array(buffer);
    case 'bool':
      return new Uint8Array(buffer);
    default:
      throw new Error(`Uknown dtype ${dtype}`);
  }
}
