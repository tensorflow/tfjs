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

import {backend_util, BackendTimingInfo, DataStorage, DataType, deprecationWarn, engine, env, KernelBackend, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasmModule, WasmFactoryConfig} from '../wasm-out/tfjs-backend-wasm';
import {BackendWasmThreadedSimdModule} from '../wasm-out/tfjs-backend-wasm-threaded-simd';
import wasmFactoryThreadedSimd from '../wasm-out/tfjs-backend-wasm-threaded-simd.js';
// @ts-ignore
import {wasmWorkerContents} from '../wasm-out/tfjs-backend-wasm-threaded-simd.worker.js';
import wasmFactory from '../wasm-out/tfjs-backend-wasm.js';

interface TensorData {
  id: number;
  memoryOffset: number;
  shape: number[];
  dtype: DataType;
  refCount: number;
  /** Only used for string tensors, storing encoded bytes. */
  stringBytes?: Uint8Array[];
}

export type DataId = object;  // object instead of {} to force non-primitive.

export class BackendWasm extends KernelBackend {
  // 0 is reserved for null data ids.
  private dataIdNextNumber = 1;
  dataIdMap: DataStorage<TensorData>;

  constructor(public wasm: BackendWasmModule | BackendWasmThreadedSimdModule) {
    super();
    this.wasm.tfjs.init();
    this.dataIdMap = new DataStorage(this, engine());
  }

  write(values: backend_util.BackendValues, shape: number[], dtype: DataType):
      DataId {
    const dataId = {id: this.dataIdNextNumber++};
    this.move(dataId, values, shape, dtype, 1);
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
      dtype: DataType, refCount: number): void {
    const id = this.dataIdNextNumber++;
    if (dtype === 'string') {
      const stringBytes = values as Uint8Array[];
      this.dataIdMap.set(
          dataId,
          {id, stringBytes, shape, dtype, memoryOffset: null, refCount});
      return;
    }

    const size = util.sizeFromShape(shape);
    const numBytes = size * util.bytesPerElement(dtype);
    const memoryOffset = this.wasm._malloc(numBytes);

    this.dataIdMap.set(dataId, {id, memoryOffset, shape, dtype, refCount});

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

  /**
   * Dispose the memory if the dataId has 0 refCount. Return true if the memory
   * is released, false otherwise.
   * @param dataId
   * @oaram force Optional, remove the data regardless of refCount
   */
  disposeData(dataId: DataId, force = false): boolean {
    if (this.dataIdMap.has(dataId)) {
      const data = this.dataIdMap.get(dataId);
      data.refCount--;
      if (!force && data.refCount > 0) {
        return false;
      }

      this.wasm._free(data.memoryOffset);
      this.wasm.tfjs.disposeData(data.id);
      this.dataIdMap.delete(dataId);
    }
    return true;
  }

  /** Return refCount of a `TensorData`. */
  refCount(dataId: DataId): number {
    if (this.dataIdMap.has(dataId)) {
      const tensorData = this.dataIdMap.get(dataId);
      return tensorData.refCount;
    }
    return 0;
  }

  incRef(dataId: DataId) {
    const data = this.dataIdMap.get(dataId);
    if (data != null) {
      data.refCount++;
    }
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
    if ('PThread' in this.wasm) {
      this.wasm.PThread.terminateAllThreads();
    }
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
      const id = this.dataIdNextNumber++;
      dataId = {id};
      this.dataIdMap.set(dataId, {id, memoryOffset, shape, dtype, refCount: 1});
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
        throw new Error(`Unknown dtype ${dtype}`);
    }
  }
}

function createInstantiateWasmFunc(path: string) {
  // this will be replace by rollup plugin patchWechatWebAssembly in 
  // minprogram's output.
  // tslint:disable-next-line:no-any
  return (imports: any, callback: any) => {
    util.fetch(path, {credentials: 'same-origin'}).then((response) => {
      if (!response['ok']) {
        imports.env.a(`failed to load wasm binary file at '${path}'`);
      }
      response.arrayBuffer().then(binary => {
        WebAssembly.instantiate(binary, imports).then(output => {
          callback(output.instance, output.module);
        });
      });
    });
    return {};
  };
}

/**
 * Returns the path of the WASM binary.
 * @param simdSupported whether SIMD is supported
 * @param threadsSupported whether multithreading is supported
 * @param wasmModuleFolder the directory containing the WASM binaries.
 */
function getPathToWasmBinary(
    simdSupported: boolean, threadsSupported: boolean,
    wasmModuleFolder: string) {
  if (wasmPath != null) {
    // If wasmPath is defined, the user has supplied a full path to
    // the vanilla .wasm binary.
    return wasmPath;
  }

  let path: WasmBinaryName = 'tfjs-backend-wasm.wasm';
  if (simdSupported && threadsSupported) {
    path = 'tfjs-backend-wasm-threaded-simd.wasm';
  } else if (simdSupported) {
    path = 'tfjs-backend-wasm-simd.wasm';
  }

  if (wasmFileMap != null) {
    if (wasmFileMap[path] != null) {
      return wasmFileMap[path];
    }
  }

  return wasmModuleFolder + path;
}

/**
 * Initializes the wasm module and creates the js <--> wasm bridge.
 *
 * NOTE: We wrap the wasm module in a object with property 'wasm' instead of
 * returning Promise<BackendWasmModule> to avoid freezing Chrome (last tested
 * in Chrome 76).
 */
export async function init(): Promise<{wasm: BackendWasmModule}> {
  const [simdSupported, threadsSupported] = await Promise.all([
    env().getAsync('WASM_HAS_SIMD_SUPPORT'),
    env().getAsync('WASM_HAS_MULTITHREAD_SUPPORT')
  ]);

  return new Promise((resolve, reject) => {
    const factoryConfig: WasmFactoryConfig = {};

    /**
     * This function overrides the Emscripten module locateFile utility.
     * @param path The relative path to the file that needs to be loaded.
     * @param prefix The path to the main JavaScript file's directory.
     */
    factoryConfig.locateFile = (path, prefix) => {
      if (path.endsWith('.worker.js')) {
        const response = wasmWorkerContents;
        const blob = new Blob([response], {type: 'application/javascript'});
        return URL.createObjectURL(blob);
      }

      if (path.endsWith('.wasm')) {
        return getPathToWasmBinary(
            simdSupported as boolean, threadsSupported as boolean,
            wasmPathPrefix != null ? wasmPathPrefix : prefix);
      }
      return prefix + path;
    };

    // Use the instantiateWasm override when system fetch is not available.
    // Reference:
    // https://github.com/emscripten-core/emscripten/blob/2bca083cbbd5a4133db61fbd74d04f7feecfa907/tests/manual_wasm_instantiate.html#L170
    if (customFetch) {
      factoryConfig.instantiateWasm =
          createInstantiateWasmFunc(getPathToWasmBinary(
              simdSupported as boolean, threadsSupported as boolean,
              wasmPathPrefix != null ? wasmPathPrefix : ''));
    }

    let initialized = false;
    factoryConfig.onAbort = () => {
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

    let wasm: Promise<BackendWasmModule>;
    // If `wasmPath` has been defined we must initialize the vanilla module.
    if (threadsSupported && simdSupported && wasmPath == null) {
      factoryConfig.mainScriptUrlOrBlob = new Blob(
          [`var WasmBackendModuleThreadedSimd = ` +
           wasmFactoryThreadedSimd.toString()],
          {type: 'text/javascript'});
      wasm = wasmFactoryThreadedSimd(factoryConfig);
    } else {
      // The wasmFactory works for both vanilla and SIMD binaries.
      wasm = wasmFactory(factoryConfig);
    }

    // The WASM module has been successfully created by the factory.
    // Any error will be caught by the onAbort callback defined above.
    wasm.then((module) => {
      initialized = true;
      initAborted = false;

      const voidReturnType: string = null;
      // Using the tfjs namespace to avoid conflict with emscripten's API.
      module.tfjs = {
        init: module.cwrap('init', null, []),
        registerTensor: module.cwrap(
            'register_tensor', null,
            [
              'number',  // id
              'number',  // size
              'number',  // memoryOffset
            ]),
        disposeData: module.cwrap('dispose_data', voidReturnType, ['number']),
        dispose: module.cwrap('dispose', voidReturnType, []),
      };

      resolve({wasm: module});
    });
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

const wasmBinaryNames = [
  'tfjs-backend-wasm.wasm', 'tfjs-backend-wasm-simd.wasm',
  'tfjs-backend-wasm-threaded-simd.wasm'
] as const ;
type WasmBinaryName = typeof wasmBinaryNames[number];

let wasmPath: string = null;
let wasmPathPrefix: string = null;
let wasmFileMap: {[key in WasmBinaryName]?: string} = {};
let initAborted = false;
let customFetch = false;

/**
 * @deprecated Use `setWasmPaths` instead.
 * Sets the path to the `.wasm` file which will be fetched when the wasm
 * backend is initialized. See
 * https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers
 * for more details.
 * @param path wasm file path or url
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 *
 * @doc {heading: 'Environment', namespace: 'wasm'}
 */
export function setWasmPath(path: string, usePlatformFetch = false): void {
  deprecationWarn(
      'setWasmPath has been deprecated in favor of setWasmPaths and' +
      ' will be removed in a future release.');
  if (initAborted) {
    throw new Error(
        'The WASM backend was already initialized. Make sure you call ' +
        '`setWasmPath()` before you call `tf.setBackend()` or `tf.ready()`');
  }
  wasmPath = path;
  customFetch = usePlatformFetch;
}

/**
 * Configures the locations of the WASM binaries.
 *
 * ```js
 * setWasmPaths({
 *  'tfjs-backend-wasm.wasm': 'renamed.wasm',
 *  'tfjs-backend-wasm-simd.wasm': 'renamed-simd.wasm',
 *  'tfjs-backend-wasm-threaded-simd.wasm': 'renamed-threaded-simd.wasm'
 * });
 * tf.setBackend('wasm');
 * ```
 *
 * @param prefixOrFileMap This can be either a string or object:
 *  - (string) The path to the directory where the WASM binaries are located.
 *     Note that this prefix will be used to load each binary (vanilla,
 *     SIMD-enabled, threading-enabled, etc.).
 *  - (object) Mapping from names of WASM binaries to custom
 *     full paths specifying the locations of those binaries. This is useful if
 *     your WASM binaries are not all located in the same directory, or if your
 *     WASM binaries have been renamed.
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 *
 * @doc {heading: 'Environment', namespace: 'wasm'}
 */
export function setWasmPaths(
    prefixOrFileMap: string|{[key in WasmBinaryName]?: string},
    usePlatformFetch = false): void {
  if (initAborted) {
    throw new Error(
        'The WASM backend was already initialized. Make sure you call ' +
        '`setWasmPaths()` before you call `tf.setBackend()` or ' +
        '`tf.ready()`');
  }

  if (typeof prefixOrFileMap === 'string') {
    wasmPathPrefix = prefixOrFileMap;
  } else {
    wasmFileMap = prefixOrFileMap;
    const missingPaths =
        wasmBinaryNames.filter(name => wasmFileMap[name] == null);
    if (missingPaths.length > 0) {
      throw new Error(
          `There were no entries found for the following binaries: ` +
          `${missingPaths.join(',')}. Please either call setWasmPaths with a ` +
          `map providing a path for each binary, or with a string indicating ` +
          `the directory where all the binaries can be found.`);
    }
  }

  customFetch = usePlatformFetch;
}

/** Used in unit tests. */
export function resetWasmPath(): void {
  wasmPath = null;
  wasmPathPrefix = null;
  wasmFileMap = {};
  customFetch = false;
  initAborted = false;
}
