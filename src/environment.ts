/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as device_util from './device_util';
import {Engine, MemoryInfo, ProfileInfo, ScopeFn, TimingInfo} from './engine';
import {Features, getFeaturesFromURL, getMaxTexturesInShader, getNumMBBeforePaging, getWebGLDisjointQueryTimerVersion, getWebGLMaxTextureSize, isChrome, isDownloadFloatTextureEnabled, isRenderToFloatTextureEnabled, isWebGLFenceEnabled, isWebGLVersionEnabled} from './environment_util';
import {KernelBackend} from './kernels/backend';
import {DataId, setDeprecationWarningFn, setTensorTracker, Tensor} from './tensor';
import {TensorContainer} from './tensor_types';
import {getTensorsInContainer} from './tensor_util';

export const EPSILON_FLOAT16 = 1e-4;
const TEST_EPSILON_FLOAT16 = 1e-1;

export const EPSILON_FLOAT32 = 1e-7;
const TEST_EPSILON_FLOAT32 = 1e-3;

export class Environment {
  private features: Features = {};
  private globalEngine: Engine;
  private registry:
      {[id: string]: {backend: KernelBackend, priority: number}} = {};
  private registryFactory: {[id: string]: () => KernelBackend} = {};
  backendName: string;

  constructor(features?: Features) {
    if (features != null) {
      this.features = features;
    }

    if (this.get('DEBUG')) {
      console.warn(
          'Debugging mode is ON. The output of every math call will ' +
          'be downloaded to CPU and checked for NaNs. ' +
          'This significantly impacts performance.');
    }
  }

  /**
   * Sets the backend (cpu, webgl, etc) responsible for creating tensors and
   * executing operations on those tensors.
   *
   * Note this disposes the current backend, if any, as well as any tensors
   * associated with it. A new backend is initialized, even if it is of the
   * same type as the previous one.
   *
   * @param backendName The name of the backend. Currently supports
   *     `'webgl'|'cpu'` in the browser, and `'tensorflow'` under node.js
   *     (requires tfjs-node).
   * @param safeMode Defaults to false. In safe mode, you are forced to
   *     construct tensors and call math operations inside a `tidy()` which
   *     will automatically clean up intermediate tensors.
   */
  /** @doc {heading: 'Environment'} */
  static setBackend(backendName: string, safeMode = false) {
    if (!(backendName in ENV.registry)) {
      throw new Error(`Backend name '${backendName}' not found in registry`);
    }
    ENV.engine.backend = ENV.findBackend(backendName);
    ENV.backendName = backendName;
  }

  /**
   * Returns the current backend name (cpu, webgl, etc). The backend is
   * responsible for creating tensors and executing operations on those tensors.
   */
  /** @doc {heading: 'Environment'} */
  static getBackend(): string {
    ENV.initEngine();
    return ENV.backendName;
  }

  /**
   * Dispose all variables kept in backend engine.
   */
  /** @doc {heading: 'Environment'} */
  static disposeVariables(): void {
    ENV.engine.disposeVariables();
  }

  /**
   * Returns memory info at the current time in the program. The result is an
   * object with the following properties:
   *
   * - `numBytes`: Number of bytes allocated (undisposed) at this time.
   * - `numTensors`: Number of unique tensors allocated.
   * - `numDataBuffers`: Number of unique data buffers allocated
   *   (undisposed) at this time, which is ≤ the number of tensors
   *   (e.g. `a.reshape(newShape)` makes a new Tensor that shares the same
   *   data buffer with `a`).
   * - `unreliable`: True if the memory usage is unreliable. See `reasons` when
   *    `unrealible` is true.
   * - `reasons`: `string[]`, reasons why the memory is unreliable, present if
   *    `unreliable` is true.
   */
  /** @doc {heading: 'Performance', subheading: 'Memory'} */
  static memory(): MemoryInfo {
    return ENV.engine.memory();
  }

  /**
   * Executes the provided function `f()` and returns a promise that resolves
   * with information about the function's memory use:
   * - `newBytes`: tne number of new bytes allocated
   * - `newTensors`: the number of new tensors created
   * - `peakBytes`: the peak number of bytes allocated
   * - `kernels`: an array of objects for each kernel involved that reports
   * their input and output shapes, number of bytes used, and number of new
   * tensors created.
   *
   * ```js
   * const profile = await tf.profile(() => {
   *   const x = tf.tensor1d([1, 2, 3]);
   *   let x2 = x.square();
   *   x2.dispose();
   *   x2 = x.square();
   *   x2.dispose();
   *   return x;
   * });
   *
   * console.log(`newBytes: ${profile.newBytes}`);
   * console.log(`newTensors: ${profile.newTensors}`);
   * console.log(`byte usage over all kernels: ${profile.kernels.map(k =>
   * k.totalBytesSnapshot)}`);
   * ```
   *
   */
  /** @doc {heading: 'Performance', subheading: 'Profile'} */
  static profile(f: () => TensorContainer): Promise<ProfileInfo> {
    return ENV.engine.profile(f);
  }

  /**
   * Executes the provided function `fn` and after it is executed, cleans up all
   * intermediate tensors allocated by `fn` except those returned by `fn`.
   * `fn` must not return a Promise (async functions not allowed). The returned
   * result can be a complex object.
   *
   * Using this method helps avoid memory leaks. In general, wrap calls to
   * operations in `tf.tidy` for automatic memory cleanup.
   *
   * When in safe mode, you must enclose all `tf.Tensor` creation and ops
   * inside a `tf.tidy` to prevent memory leaks.
   *
   * ```js
   * // y = 2 ^ 2 + 1
   * const y = tf.tidy(() => {
   *   // a, b, and one will be cleaned up when the tidy ends.
   *   const one = tf.scalar(1);
   *   const a = tf.scalar(2);
   *   const b = a.square();
   *
   *   console.log('numTensors (in tidy): ' + tf.memory().numTensors);
   *
   *   // The value returned inside the tidy function will return
   *   // through the tidy, in this case to the variable y.
   *   return b.add(one);
   * });
   *
   * console.log('numTensors (outside tidy): ' + tf.memory().numTensors);
   * y.print();
   * ```
   *
   * @param nameOrFn The name of the closure, or the function to execute.
   *     If a name is provided, the 2nd argument should be the function.
   *     If debug mode is on, the timing and the memory usage of the function
   *     will be tracked and displayed on the console using the provided name.
   * @param fn The function to execute.
   */
  /** @doc {heading: 'Performance', subheading: 'Memory'} */
  static tidy<T extends TensorContainer>(
      nameOrFn: string|ScopeFn<T>, fn?: ScopeFn<T>): T {
    return ENV.engine.tidy(nameOrFn, fn);
  }

  /**
   * Disposes any `tf.Tensor`s found within the provided object.
   *
   * @param container an object that may be a `tf.Tensor` or may directly
   *     contain `tf.Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. If
   *     the object is not a `tf.Tensor` or does not contain `Tensors`, nothing
   *     happens. In general it is safe to pass any object here, except that
   *     `Promise`s are not supported.
   */
  /** @doc {heading: 'Performance', subheading: 'Memory'} */
  static dispose(container: TensorContainer) {
    const tensors = getTensorsInContainer(container);
    tensors.forEach(tensor => tensor.dispose());
  }

  /**
   * Keeps a `tf.Tensor` generated inside a `tf.tidy` from being disposed
   * automatically.
   *
   * ```js
   * let b;
   * const y = tf.tidy(() => {
   *   const one = tf.scalar(1);
   *   const a = tf.scalar(2);
   *
   *   // b will not be cleaned up by the tidy. a and one will be cleaned up
   *   // when the tidy ends.
   *   b = tf.keep(a.square());
   *
   *   console.log('numTensors (in tidy): ' + tf.memory().numTensors);
   *
   *   // The value returned inside the tidy function will return
   *   // through the tidy, in this case to the variable y.
   *   return b.add(one);
   * });
   *
   * console.log('numTensors (outside tidy): ' + tf.memory().numTensors);
   * console.log('y:');
   * y.print();
   * console.log('b:');
   * b.print();
   * ```
   *
   * @param result The tensor to keep from being disposed.
   */
  /** @doc {heading: 'Performance', subheading: 'Memory'} */
  static keep<T extends Tensor>(result: T): T {
    return ENV.engine.keep(result);
  }

  /**
   * Executes `f()` and returns a promise that resolves with timing
   * information.
   *
   * The result is an object with the following properties:
   *
   * - `wallMs`: Wall execution time.
   * - `kernelMs`: Kernel execution time, ignoring data transfer.
   * - On `WebGL` The following additional properties exist:
   *   - `uploadWaitMs`: CPU blocking time on texture uploads.
   *   - `downloadWaitMs`: CPU blocking time on texture downloads (readPixels).
   *
   * ```js
   * const x = tf.randomNormal([20, 20]);
   * const time = await tf.time(() => x.matMul(x));
   *
   * console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);
   * ```
   *
   * @param f The function to execute and time.
   */
  /** @doc {heading: 'Performance', subheading: 'Timing'} */
  static time(f: () => void): Promise<TimingInfo> {
    return ENV.engine.time(f);
  }

  get<K extends keyof Features>(feature: K): Features[K] {
    if (feature in this.features) {
      return this.features[feature];
    }

    this.features[feature] = this.evaluateFeature(feature);

    return this.features[feature];
  }

  getFeatures(): Features {
    return this.features;
  }

  set<K extends keyof Features>(feature: K, value: Features[K]): void {
    this.features[feature] = value;
  }

  private getBestBackendName(): string {
    if (Object.keys(this.registry).length === 0) {
      throw new Error('No backend found in registry.');
    }
    const sortedBackends = Object.keys(this.registry)
                               .map(name => {
                                 return {name, entry: this.registry[name]};
                               })
                               .sort((a, b) => {
                                 // Highest priority comes first.
                                 return b.entry.priority - a.entry.priority;
                               });
    return sortedBackends[0].name;
  }

  private evaluateFeature<K extends keyof Features>(feature: K): Features[K] {
    if (feature === 'DEBUG') {
      return false;
    } else if (feature === 'IS_BROWSER') {
      return typeof window !== 'undefined';
    } else if (feature === 'IS_NODE') {
      return (typeof process !== 'undefined') &&
          (typeof process.versions !== 'undefined') &&
          (typeof process.versions.node !== 'undefined');
    } else if (feature === 'IS_CHROME') {
      return isChrome();
    } else if (feature === 'WEBGL_CPU_FORWARD') {
      return true;
    } else if (feature === 'WEBGL_PACK') {
      return this.get('WEBGL_VERSION') === 0 ? false : true;
    } else if (feature === 'WEBGL_PACK_BATCHNORMALIZATION') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_PACK_CLIP') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_PACK_DEPTHWISECONV') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_PACK_BINARY_OPERATIONS') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_PACK_ARRAY_OPERATIONS') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_PACK_IMAGE_OPERATIONS') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_PACK_REDUCE') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_LAZILY_UNPACK') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_CONV_IM2COL') {
      return this.get('WEBGL_PACK');
    } else if (feature === 'WEBGL_NUM_MB_BEFORE_PAGING') {
      if (this.get('PROD') || !this.get('IS_BROWSER')) {
        return Number.POSITIVE_INFINITY;
      }
      return getNumMBBeforePaging();
    } else if (feature === 'WEBGL_MAX_TEXTURE_SIZE') {
      return getWebGLMaxTextureSize(this.get('WEBGL_VERSION'));
    } else if (feature === 'WEBGL_MAX_TEXTURES_IN_SHADER') {
      return getMaxTexturesInShader(this.get('WEBGL_VERSION'));
    } else if (feature === 'IS_TEST') {
      return false;
    } else if (feature === 'BACKEND') {
      return this.getBestBackendName();
    } else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') {
      const webGLVersion = this.get('WEBGL_VERSION');

      if (webGLVersion === 0) {
        return 0;
      }
      return getWebGLDisjointQueryTimerVersion(webGLVersion);
    } else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') {
      return this.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0 &&
          !device_util.isMobile();
    } else if (feature === 'HAS_WEBGL') {
      return this.get('WEBGL_VERSION') > 0;
    } else if (feature === 'WEBGL_VERSION') {
      if (isWebGLVersionEnabled(2)) {
        return 2;
      } else if (isWebGLVersionEnabled(1)) {
        return 1;
      }
      return 0;
    } else if (feature === 'WEBGL_RENDER_FLOAT32_ENABLED') {
      return isRenderToFloatTextureEnabled(this.get('WEBGL_VERSION'));
    } else if (feature === 'WEBGL_DOWNLOAD_FLOAT_ENABLED') {
      return isDownloadFloatTextureEnabled(this.get('WEBGL_VERSION'));
    } else if (feature === 'WEBGL_FENCE_API_ENABLED') {
      return isWebGLFenceEnabled(this.get('WEBGL_VERSION'));
    } else if (feature === 'WEBGL_SIZE_UPLOAD_UNIFORM') {
      // Use uniform uploads only when 32bit floats are supported. In 16bit
      // environments there are problems with comparing a 16bit texture value
      // with a 32bit uniform value.
      const useUniforms = this.get('WEBGL_RENDER_FLOAT32_ENABLED');
      return useUniforms ? 4 : 0;
    } else if (feature === 'TEST_EPSILON') {
      return this.backend.floatPrecision() === 32 ? TEST_EPSILON_FLOAT32 :
                                                    TEST_EPSILON_FLOAT16;
    } else if (feature === 'EPSILON') {
      return this.backend.floatPrecision() === 32 ? EPSILON_FLOAT32 :
                                                    EPSILON_FLOAT16;
    } else if (feature === 'PROD') {
      return false;
    } else if (feature === 'TENSORLIKE_CHECK_SHAPE_CONSISTENCY') {
      return !this.get('PROD');
    } else if (feature === 'DEPRECATION_WARNINGS_ENABLED') {
      return true;
    }
    throw new Error(`Unknown feature ${feature}.`);
  }

  setFeatures(features: Features) {
    this.features = Object.assign({}, features);
  }

  reset() {
    this.features = getFeaturesFromURL();
    if (this.globalEngine != null) {
      this.globalEngine = null;
    }
  }

  get backend(): KernelBackend {
    return this.engine.backend;
  }

  /**
   * Finds the backend registered under the provided name. Returns null if the
   * name is not in the registry.
   */
  findBackend(name: string): KernelBackend {
    if (!(name in this.registry)) {
      return null;
    }
    return this.registry[name].backend;
  }

  /**
   * Finds the backend factory registered under the provided name. Returns a
   * function that produces a new backend when called. Returns null if the name
   * is not in the registry.
   */
  findBackendFactory(name: string): () => KernelBackend {
    if (!(name in this.registryFactory)) {
      return null;
    }
    return this.registryFactory[name];
  }

  /**
   * Registers a global backend. The registration should happen when importing
   * a module file (e.g. when importing `backend_webgl.ts`), and is used for
   * modular builds (e.g. custom tfjs bundle with only webgl support).
   *
   * @param factory The backend factory function. When called, it should
   * return an instance of the backend.
   * @param priority The priority of the backend (higher = more important).
   *     In case multiple backends are registered, the priority is used to find
   *     the best backend. Defaults to 1.
   * @return False if the creation/registration failed. True otherwise.
   */
  registerBackend(name: string, factory: () => KernelBackend, priority = 1):
      boolean {
    if (name in this.registry) {
      console.warn(
          `${name} backend was already registered. Reusing existing backend`);
      return false;
    }
    try {
      const backend = factory();
      backend.setDataMover(
          {moveData: (dataId: DataId) => this.engine.moveData(dataId)});
      this.registry[name] = {backend, priority};
      this.registryFactory[name] = factory;
      return true;
    } catch (err) {
      console.warn(`Registration of backend ${name} failed`);
      console.warn(err.stack || err.message);
      return false;
    }
  }

  removeBackend(name: string): void {
    if (!(name in this.registry)) {
      throw new Error(`${name} backend not found in registry`);
    }
    this.registry[name].backend.dispose();
    delete this.registry[name];
  }

  get engine(): Engine {
    this.initEngine();
    return this.globalEngine;
  }

  private initEngine() {
    if (this.globalEngine == null) {
      this.backendName = this.get('BACKEND');
      const backend = this.findBackend(this.backendName);
      this.globalEngine =
          new Engine(backend, false /* safeMode */, () => this.get('DEBUG'));
    }
  }

  // tslint:disable-next-line:no-any
  get global(): any {
    return getGlobalNamespace();
  }
}

let _global: {ENV: Environment};
function getGlobalNamespace(): {ENV: Environment} {
  if (_global == null) {
    // tslint:disable-next-line:no-any
    let ns: any;
    if (typeof (window) !== 'undefined') {
      ns = window;
    } else if (typeof (global) !== 'undefined') {
      ns = global;
    } else if (typeof (process) !== 'undefined') {
      ns = process;
    } else {
      throw new Error('Could not find a global object');
    }
    _global = ns;
  }
  return _global;
}

function getOrMakeEnvironment(): Environment {
  const ns = getGlobalNamespace();
  if (ns.ENV == null) {
    ns.ENV = new Environment(getFeaturesFromURL());
  }
  // Tell the current tensor interface that the global engine is responsible for
  // tracking.
  setTensorTracker(() => ns.ENV.engine);
  return ns.ENV;
}

/**
 * Enables production mode which disables correctness checks in favor of
 * performance.
 */
/** @doc {heading: 'Environment'} */
export function enableProdMode(): void {
  ENV.set('PROD', true);
}

/**
 * Enables debug mode which will log information about all executed kernels:
 * the ellapsed time of the kernel execution, as well as the rank, shape, and
 * size of the output tensor.
 *
 * Debug mode will significantly slow down your application as it will
 * download the result of every operation to the CPU. This should not be used in
 * production. Debug mode does not affect the timing information of the kernel
 * execution as we do not measure download time in the kernel execution time.
 *
 * See also: `tf.profile`, `tf.memory`.
 */
/** @doc {heading: 'Environment'} */
export function enableDebugMode(): void {
  ENV.set('DEBUG', true);
}

/** Globally disables deprecation warnings */
export function disableDeprecationWarnings(): void {
  ENV.set('DEPRECATION_WARNINGS_ENABLED', false);
  console.warn(`TensorFlow.js deprecation warnings have been disabled.`);
}

/** Warn users about deprecated functionality. */
export function deprecationWarn(msg: string) {
  if (ENV.get('DEPRECATION_WARNINGS_ENABLED')) {
    console.warn(
        msg + ' You can disable deprecation warnings with ' +
        'tf.disableDeprecationWarnings().');
  }
}
setDeprecationWarningFn(deprecationWarn);

export let ENV = getOrMakeEnvironment();
