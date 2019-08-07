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

import {backend_util, DataType, KernelBackend, Rank, registerBackend, ShapeMap, Tensor, util} from '@tensorflow/tfjs-core';

import wasmFactory from '../wasm-out/tfjs-backend-wasm';
import {BackendWasmModule} from '../wasm-out/tfjs-backend-wasm';

const WASM_PRIORITY = 2;

interface TensorInfo {
  id: number;
  memoryOffset: number;
  shape: number[];
  dtype: DataType;
}

export type DataId = object;  // object instead of {} to force non-primitive.

export class BackendWasm extends KernelBackend {
  private dataIdNextNumber = 0;
  private dataIdMap: WeakMap<DataId, TensorInfo> = new WeakMap();

  constructor(private wasm: BackendWasmModule) {
    super();
  }

  register(dataId: DataId, shape: number[], dtype: DataType) {
    const memoryOffset = this.wasm._malloc(
        util.sizeFromShape(shape) * util.bytesPerElement(dtype));
    const id = this.dataIdNextNumber++;
    this.dataIdMap.set(dataId, {id, memoryOffset, shape, dtype});

    const shapeBytes = new Uint8Array(new Int32Array(shape).buffer);
    this.wasm.tfjs.registerTensor(
        id, shapeBytes, shape.length, dtypeToEnumValue(dtype), memoryOffset);
  }

  write(dataId: DataId, values: backend_util.TypedArray) {
    const {memoryOffset} = this.dataIdMap.get(dataId);
    this.wasm.HEAPU8.set(new Uint8Array(values.buffer), memoryOffset);
  }

  async read(dataId: DataId): Promise<backend_util.BackendValues> {
    return this.readSync(dataId);
  }

  readSync(dataId: DataId): backend_util.BackendValues {
    const {memoryOffset, dtype, shape} = this.dataIdMap.get(dataId);
    const bytes = this.wasm.HEAPU8.slice(
        memoryOffset,
        memoryOffset + util.sizeFromShape(shape) * util.bytesPerElement(dtype));
    return typedArrayFromBuffer(bytes.buffer, dtype);
  }

  disposeData(dataId: DataId) {
    const data = this.dataIdMap.get(dataId);
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

  // Kernels.

  add(a: Tensor, b: Tensor): Tensor {
    const aId = this.dataIdMap.get(a.dataId).id;
    const bId = this.dataIdMap.get(b.dataId).id;

    const newShape = backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const aBroadcastDims = backend_util.getBroadcastDims(a.shape, newShape);
    const bBroadcastDims = backend_util.getBroadcastDims(b.shape, newShape);
    const loopsOverAllOfA = aBroadcastDims.every((v, i) => v === i);
    const loopsOverAllOfB = bBroadcastDims.every((v, i) => v === i);

    const out = this.makeOutput(newShape, a.dtype);
    const outId = this.dataIdMap.get(out.dataId).id;

    // Short-circuit zero-sized tensors.
    if (out.size === 0) {
      return out;
    }

    if (loopsOverAllOfA && loopsOverAllOfB) {
      this.wasm.tfjs.add(aId, bId, outId);
      return out;
    } else {
      throw new Error('Broadcasting along inner dims is not yet supported');
    }
  }

  reshape<T extends Tensor, R extends Rank>(x: T, newShape: ShapeMap[R]):
      Tensor<R> {
    return Tensor.make(newShape, {dataId: x.dataId}, x.dtype);
  }

  cast<T extends Tensor>(x: T, dtype: DataType): T {
    const out = this.makeOutput(x.shape, dtype);
    const {memoryOffset: inOffset} = this.dataIdMap.get(x.dataId);
    const {memoryOffset: outOffset} = this.dataIdMap.get(out.dataId);
    const inVals = this.typedArrayFromHeap(inOffset, x.dtype, x.size);
    const outVals = this.typedArrayFromHeap(outOffset, dtype, out.size);
    outVals.set(inVals);
    return out as T;
  }

  private typedArrayFromHeap(offset: number, dtype: DataType, size: number):
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

  private makeOutput(shape: number[], dtype: DataType): Tensor {
    return Tensor.make(shape, {}, dtype, this);
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
      add: wasm.cwrap('add', voidReturnType, ['number, number, number']),
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
