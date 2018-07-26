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
import {Engine, MemoryInfo, ScopeFn, TimingInfo} from './engine';
import {Features, getFeaturesFromURL, getWebGLDisjointQueryTimerVersion, isChrome, isDownloadFloatTextureEnabled, isRenderToFloatTextureEnabled, isWebGLFenceEnabled, isWebGLVersionEnabled} from './environment_util';
import {KernelBackend} from './kernels/backend';
import {setTensorTracker, Tensor, TensorTracker} from './tensor';
import {TensorContainer} from './tensor_types';
import {getTensorsInContainer} from './tensor_util';

const TEST_EPSILON_FLOAT32_ENABLED = 1e-3;
const TEST_EPSILON_FLOAT32_DISABLED = 1e-1;

export class Environment {
  private features: Features = {};
  private globalEngine: Engine;
  private registry:
      {[id: string]: {backend: KernelBackend, priority: number}} = {};
  backendName: string;
  backend: KernelBackend;

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
   * associated with it.  A new backend is initialized, even if it is of the
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
    ENV.initBackend(backendName, safeMode);
  }

  /**
   * Returns the current backend name (cpu, webgl, etc). The backend is
   * responsible for creating tensors and executing operations on those tensors.
   */
  /** @doc {heading: 'Environment'} */
  static getBackend(): string {
    ENV.initDefaultBackend();
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
   * - `unreliable`: `Optional` `boolean`:
   *    - On WebGL, not present (always reliable).
   *    - On CPU, true. Due to automatic garbage collection, these numbers
   *     represent undisposed tensors, i.e. not wrapped in `tidy()`, or
   *     lacking a call to `tensor.dispose()`.
   */
  /** @doc {heading: 'Performance', subheading: 'Memory'} */
  static memory(): MemoryInfo {
    return ENV.engine.memory();
  }

  /**
   * Executes the provided function `fn` and after it is executed, cleans up all
   * intermediate tensors allocated by `fn` except those returned by `fn`.
   * `f` must not return a Promise (async functions not allowed).
   * The returned result can be a complex object, however tidy only walks the
   * top-level properties (depth 1) of that object to search for tensors, or
   * lists of tensors that need to be tracked in the parent scope.
   *
   * Using this method helps avoid memory leaks. In general, wrap calls to
   * operations in `tidy` for automatic memory cleanup.
   *
   * When in safe mode, you must enclose all `Tensor` creation and ops
   * inside a `tidy` to prevent memory leaks.
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
      nameOrFn: string|ScopeFn<T>, fn?: ScopeFn<T>, gradMode = false): T {
    return ENV.engine.tidy(nameOrFn, fn, gradMode);
  }

  /**
   * Disposes any `Tensor`s found within the provided object.
   *
   * @param container an object that may be a `Tensor` or may directly contain
   *     `Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`.  If the
   *     object is not a `Tensor` or does not contain `Tensors`, nothing
   *     happens. In general it is safe to pass any object here, except that
   *     `Promise`s are not supported.
   */
  /** @doc {heading: 'Performance', subheading: 'Memory'} */
  static dispose(container: TensorContainer) {
    const tensors = getTensorsInContainer(container);
    tensors.forEach(tensor => tensor.dispose());
  }

  /**
   * Keeps a `Tensor` generated inside a `tidy` from being disposed
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
          (typeof process.versions.node !== 'undefined');
    } else if (feature === 'IS_CHROME') {
      return isChrome();
    } else if (feature === 'IS_TEST') {
      return false;
    } else if (feature === 'BACKEND') {
      return this.getBestBackendName();
    } else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') {
      const webGLVersion = this.get('WEBGL_VERSION');

      if (webGLVersion === 0) {
        return 0;
      }
      // Remove this and reenable this extension when the
      // EXT_disjoint_query_timer extension is reenabled in chrome.
      // https://github.com/tensorflow/tfjs/issues/544
      if (webGLVersion > 0) {
        return 0;
      }
      return getWebGLDisjointQueryTimerVersion(
          webGLVersion, this.get('IS_BROWSER'));
    } else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') {
      return this.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0 &&
          !device_util.isMobile();
    } else if (feature === 'HAS_WEBGL') {
      return this.get('WEBGL_VERSION') > 0;
    } else if (feature === 'WEBGL_VERSION') {
      if (isWebGLVersionEnabled(2, this.get('IS_BROWSER'))) {
        return 2;
      } else if (isWebGLVersionEnabled(1, this.get('IS_BROWSER'))) {
        return 1;
      }
      return 0;
    } else if (feature === 'WEBGL_RENDER_FLOAT32_ENABLED') {
      return isRenderToFloatTextureEnabled(
          this.get('WEBGL_VERSION'), this.get('IS_BROWSER'));
    } else if (feature === 'WEBGL_DOWNLOAD_FLOAT_ENABLED') {
      return isDownloadFloatTextureEnabled(
          this.get('WEBGL_VERSION'), this.get('IS_BROWSER'));
    } else if (feature === 'WEBGL_FENCE_API_ENABLED') {
      return isWebGLFenceEnabled(
          this.get('WEBGL_VERSION'), this.get('IS_BROWSER'));
    } else if (feature === 'TEST_EPSILON') {
      if (this.get('WEBGL_RENDER_FLOAT32_ENABLED')) {
        return TEST_EPSILON_FLOAT32_ENABLED;
      }
      return TEST_EPSILON_FLOAT32_DISABLED;
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

  private initBackend(backendName?: string, safeMode = false) {
    this.backendName = backendName;
    this.backend = this.findBackend(backendName);
    this.globalEngine =
        new Engine(this.backend, safeMode, () => this.get('DEBUG'));
  }

  findBackend(name: string): KernelBackend {
    if (!(name in this.registry)) {
      return null;
    }
    return this.registry[name].backend;
  }

  /**
   * Registers a global backend. The registration should happen when importing
   * a module file (e.g. when importing `backend_webgl.ts`), and is used for
   * modular builds (e.g. custom tfjs bundle with only webgl support).
   *
   * @param factory: The backend factory function. When called, it should
   * return an instance of the backend.
   * @param priority The priority of the backend (higher = more important).
   *     In case multiple backends are registered, the priority is used to find
   *     the best backend. Defaults to 1.
   * @return False if the creation/registration failed. True otherwise.
   */
  registerBackend(
      name: string, factory: () => KernelBackend, priority = 1,
      setTensorTrackerFn?: (f: () => TensorTracker) => void): boolean {
    if (name in this.registry) {
      console.warn(
          `${name} backend was already registered. Reusing existing backend`);
      if (setTensorTrackerFn != null) {
        setTensorTrackerFn(() => this.engine);
      }
      return false;
    }
    try {
      const backend = factory();
      this.registry[name] = {backend, priority};
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
    this.initDefaultBackend();
    return this.globalEngine;
  }

  private initDefaultBackend() {
    if (this.globalEngine == null) {
      this.initBackend(this.get('BACKEND'), false /* safeMode */);
    }
  }
}

function getGlobalNamespace(): {ENV: Environment} {
  // tslint:disable-next-line:no-any
  let ns: any;
  if (typeof (window) !== 'undefined') {
    ns = window;
  } else if (typeof (process) !== 'undefined') {
    ns = process;
  } else {
    throw new Error('Could not find a global object');
  }
  return ns;
}

function getOrMakeEnvironment(): Environment {
  const ns = getGlobalNamespace();
  if (ns.ENV == null) {
    ns.ENV = new Environment(getFeaturesFromURL());
    setTensorTracker(() => ns.ENV.engine);
  }
  return ns.ENV;
}

export let ENV = getOrMakeEnvironment();
