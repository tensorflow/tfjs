/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import './flags_wasm';

import {backend_util, BackendTimingInfo, DataStorage, DataType, engine, env, KernelBackend, registerBackend, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasmModule, WasmFactoryConfig} from '../wasm-out/tfjs-backend-wasm';
import wasmFactorySimd from '../wasm-out/tfjs-backend-wasm-simd.js';
import wasmFactory from '../wasm-out/tfjs-backend-wasm.js';

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
  // 0 is reserved for null data ids.
  private dataIdNextNumber = 1;
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

  async time(f: () => void): Promise<BackendTimingInfo> {
    const start = util.now();
    f();
    const kernelMs = util.now() - start;
    return {kernelMs};
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

    const size = util.sizeFromShape(shape);
    const numBytes = size * util.bytesPerElement(dtype);
    const memoryOffset = this.wasm._malloc(numBytes);

    this.dataIdMap.set(dataId, {id, memoryOffset, shape, dtype});

    this.wasm.tfjs.registerTensor(id, size, memoryOffset);

    if (values != null) {
      this.wasm.HEAPU8.set(
          new Uint8Array(
              (values as backend_util.TypedArray).buffer,
              (values as backend_util.TypedArray).byteOffset, numBytes),
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

  memory() {
    return {unreliable: false};
  }

  /**
   * Make a tensor info for the output of an op. If `memoryOffset` is not
   * present, this method allocates memory on the WASM heap. If `memoryOffset`
   * is present, the memory was allocated elsewhere (in c++) and we just record
   * the pointer where that memory lives.
   */
  makeOutput(shape: number[], dtype: DataType, memoryOffset?: number):
      TensorInfo {
    let dataId: {};
    if (memoryOffset == null) {
      dataId = this.write(null /* values */, shape, dtype);
    } else {
      dataId = {};
      const id = this.dataIdNextNumber++;
      this.dataIdMap.set(dataId, {id, memoryOffset, shape, dtype});
      const size = util.sizeFromShape(shape);
      this.wasm.tfjs.registerTensor(id, size, memoryOffset);
    }
    return {dataId, shape, dtype};
  }

  typedArrayFromHeap({shape, dtype, dataId}: TensorInfo):
      backend_util.TypedArray {
    const buffer = this.wasm.HEAPU8.buffer;
    const {memoryOffset} = this.dataIdMap.get(dataId);
    const size = util.sizeFromShape(shape);
    switch (dtype) {
      case 'float32':
        return new Float32Array(buffer, memoryOffset, size);
      case 'int32':
        return new Int32Array(buffer, memoryOffset, size);
      case 'bool':
        return new Uint8Array(buffer, memoryOffset, size);
      default:
        throw new Error(`Uknown dtype ${dtype}`);
    }
  }
}

registerBackend('wasm', async () => {
  const {wasm} = await init();
  return new BackendWasm(wasm);
}, WASM_PRIORITY);

function createInstantiateWasmFunc(path: string) {
  // tslint:disable-next-line:no-any
  return (imports: any, callback: any) => {
    util.fetch(path, {credentials: 'same-origin'}).then((response) => {
      if (!response['ok']) {
        imports.env.a(`failed to load wasm binary file at '${path}'`);
      }
      response.arrayBuffer().then(binary => {
        WebAssembly.instantiate(binary, imports).then(output => {
          callback(output.instance);
        });
      });
    });
    return {};
  };
}

/**
 * Initializes the wasm module and creates the js <--> wasm bridge.
 *
 * NOTE: We wrap the wasm module in a object with property 'wasm' instead of
 * returning Promise<BackendWasmModule> to avoid freezing Chrome (last tested
 * in Chrome 76).
 */
export async function init(): Promise<{wasm: BackendWasmModule}> {
  const simdSupported = await env().getAsync('WASM_HAS_SIMD_SUPPORT');
  return new Promise((resolve, reject) => {
    const factoryConfig: WasmFactoryConfig = {};
    if (wasmPath != null) {
      factoryConfig.locateFile = (path, prefix) => {
        if (path.endsWith('.wasm')) {
          return wasmPath;
        }
        return prefix + path;
      };
      // use wasm instantiateWasm override when system fetch is not available.
      // For detail references
      // https://github.com/emscripten-core/emscripten/blob/2bca083cbbd5a4133db61fbd74d04f7feecfa907/tests/manual_wasm_instantiate.html#L170
      if (customFetch) {
        factoryConfig.instantiateWasm = createInstantiateWasmFunc(wasmPath);
      }
    }
    const wasm = simdSupported ? wasmFactorySimd(factoryConfig) :
                                 wasmFactory(factoryConfig);
    const voidReturnType: string = null;
    // Using the tfjs namespace to avoid conflict with emscripten's API.
    wasm.tfjs = {
      init: wasm.cwrap('init', null, []),
      registerTensor: wasm.cwrap(
          'register_tensor', null,
          [
            'number',  // id
            'number',  // size
            'number',  // memoryOffset
          ]),
      disposeData: wasm.cwrap('dispose_data', voidReturnType, ['number']),
      dispose: wasm.cwrap('dispose', voidReturnType, []),
    };
    let initialized = false;
    wasm.onRuntimeInitialized = () => {
      initialized = true;
      initAborted = false;
      resolve({wasm});
    };
    wasm.onAbort = () => {
      if (initialized) {
        // Emscripten already called console.warn so no need to double log.
        return;
      }
      if (initAborted) {
        // Emscripten calls `onAbort` twice, resulting in double error
        // messages.
        return;
      }
      initAborted = true;
      const rejectMsg =
          'Make sure the server can serve the `.wasm` file relative to the ' +
          'bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers';
      reject({message: rejectMsg});
    };
  });
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
      throw new Error(`Unknown dtype ${dtype}`);
  }
}

let wasmPath: string = null;
let initAborted = false;
let customFetch = false;
/**
 * Sets the path to the `.wasm` file which will be fetched when the wasm
 * backend is initialized. See
 * https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers
 * for more details.
 * @param path wasm file path or url
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 */
/** @doc {heading: 'Environment', namespace: 'wasm'} */
export function setWasmPath(path: string, usePlatformFetch = false): void {
  if (initAborted) {
    throw new Error(
        'The WASM backend was already initialized. Make sure you call ' +
        '`setWasmPath()` before you call `tf.setBackend()` or `tf.ready()`');
  }
  wasmPath = path;
  customFetch = usePlatformFetch;
}

/** Used in unit tests. */
export function resetWasmPath(): void {
  wasmPath = null;
  customFetch = false;
  initAborted = false;
}
