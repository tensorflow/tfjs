/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {KernelBackend} from './backends/backend';
import {ENGINE, Engine, MemoryInfo, ProfileInfo, ScopeFn, TimingInfo} from './engine';
import {env} from './environment';

import {Platform} from './platforms/platform';
import {setDeprecationWarningFn, Tensor} from './tensor';
import {TensorContainer} from './tensor_types';
import {getTensorsInContainer} from './tensor_util';

/**
 * Enables production mode which disables correctness checks in favor of
 * performance.
 */
/** @doc {heading: 'Environment'} */
export function enableProdMode(): void {
  env().set('PROD', true);
}

/**
 * Enables debug mode which will log information about all executed kernels:
 * the elapsed time of the kernel execution, as well as the rank, shape, and
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
  env().set('DEBUG', true);
}

/** Globally disables deprecation warnings */
export function disableDeprecationWarnings(): void {
  env().set('DEPRECATION_WARNINGS_ENABLED', false);
  console.warn(`TensorFlow.js deprecation warnings have been disabled.`);
}

/** Warn users about deprecated functionality. */
export function deprecationWarn(msg: string) {
  if (env().getBool('DEPRECATION_WARNINGS_ENABLED')) {
    console.warn(
        msg + ' You can disable deprecation warnings with ' +
        'tf.disableDeprecationWarnings().');
  }
}
setDeprecationWarningFn(deprecationWarn);

/**
 * Dispose all variables kept in backend engine.
 */
/** @doc {heading: 'Environment'} */
export function disposeVariables(): void {
  ENGINE.disposeVariables();
}

/**
 * It returns the global engine that keeps track of all tensors and backends.
 */
/** @doc {heading: 'Environment'} */
export function engine(): Engine {
  return ENGINE;
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
 *    `unreliable` is true.
 * - `reasons`: `string[]`, reasons why the memory is unreliable, present if
 *    `unreliable` is true.
 *
 * WebGL Properties:
 * - `numBytesInGPU`: Number of bytes allocated (undisposed) in the GPU only at
 *     this time.
 */
/** @doc {heading: 'Performance', subheading: 'Memory'} */
export function memory(): MemoryInfo {
  return ENGINE.memory();
}

/**
 * Executes the provided function `f()` and returns a promise that resolves
 * with information about the function's memory use:
 * - `newBytes`: the number of new bytes allocated
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
export function profile(f: () => (TensorContainer | Promise<TensorContainer>)):
    Promise<ProfileInfo> {
  return ENGINE.profile(f);
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
 * NOTE: Variables do *not* get cleaned up when inside a tidy(). If you want to
 * dispose variables, please use `tf.disposeVariables` or call dispose()
 * directly on variables.
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
export function tidy<T extends TensorContainer>(
    nameOrFn: string|ScopeFn<T>, fn?: ScopeFn<T>): T {
  return ENGINE.tidy(nameOrFn, fn);
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
export function dispose(container: TensorContainer) {
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
export function keep<T extends Tensor>(result: T): T {
  return ENGINE.keep(result);
}

/**
 * Executes `f()` and returns a promise that resolves with timing
 * information.
 *
 * The result is an object with the following properties:
 *
 * - `wallMs`: Wall execution time.
 * - `kernelMs`: Kernel execution time, ignoring data transfer. If using the
 * WebGL backend and the query timer extension is not available, this will
 * return an error object.
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
export function time(f: () => void): Promise<TimingInfo> {
  return ENGINE.time(f);
}

/**
 * Sets the backend (cpu, webgl, wasm, etc) responsible for creating tensors and
 * executing operations on those tensors. Returns a promise that resolves
 * to a boolean if the backend initialization was successful.
 *
 * Note this disposes the current backend, if any, as well as any tensors
 * associated with it. A new backend is initialized, even if it is of the
 * same type as the previous one.
 *
 * @param backendName The name of the backend. Currently supports
 *     `'webgl'|'cpu'` in the browser, `'tensorflow'` under node.js
 *     (requires tfjs-node), and `'wasm'` (requires tfjs-backend-wasm).
 */
/** @doc {heading: 'Backends'} */
export function setBackend(backendName: string): Promise<boolean> {
  return ENGINE.setBackend(backendName);
}

/**
 * Returns a promise that resolves when the currently selected backend (or the
 * highest priority one) has initialized. Await this promise when you are using
 * a backend that has async initialization.
 */
/** @doc {heading: 'Backends'} */
export function ready(): Promise<void> {
  return ENGINE.ready();
}

/**
 * Returns the current backend name (cpu, webgl, etc). The backend is
 * responsible for creating tensors and executing operations on those tensors.
 */
/** @doc {heading: 'Backends'} */
export function getBackend(): string {
  return ENGINE.backendName;
}

/**
 * Removes a backend and the registered factory.
 */
/** @doc {heading: 'Backends'} */
export function removeBackend(name: string): void {
  ENGINE.removeBackend(name);
}

/**
 * Finds the backend registered under the provided name. Returns null if the
 * name is not in the registry, or the registration hasn't finished yet.
 */
export function findBackend(name: string): KernelBackend {
  return ENGINE.findBackend(name);
}

/**
 * Finds the backend factory registered under the provided name. Returns a
 * function that produces a new backend when called. Returns null if the name
 * is not in the registry.
 */
export function findBackendFactory(name: string): () =>
    KernelBackend | Promise<KernelBackend> {
  return ENGINE.findBackendFactory(name);
}

/**
 * Registers a global backend. The registration should happen when importing
 * a module file (e.g. when importing `backend_webgl.ts`), and is used for
 * modular builds (e.g. custom tfjs bundle with only webgl support).
 *
 * @param factory The backend factory function. When called, it should
 * return a backend instance, or a promise of an instance.
 * @param priority The priority of the backend (higher = more important).
 *     In case multiple backends are registered, the priority is used to find
 *     the best backend. Defaults to 1.
 * @return False if there is already a registered backend under this name, true
 *     if not.
 */
/** @doc {heading: 'Backends'} */
export function registerBackend(
    name: string, factory: () => KernelBackend | Promise<KernelBackend>,
    priority = 1): boolean {
  return ENGINE.registerBackend(name, factory, priority);
}

/**
 * Gets the current backend. If no backends have been initialized, this will
 * attempt to initialize the best backend. Will throw an error if the highest
 * priority backend has async initialization, in which case, you should call
 * 'await tf.ready()' before running other code.
 */
/** @doc {heading: 'Backends'} */
export function backend(): KernelBackend {
  return ENGINE.backend;
}

/**
 * Sets the global platform.
 *
 * @param platformName The name of this platform.
 * @param platform A platform implementation.
 */
export function setPlatform(platformName: string, platform: Platform) {
  env().setPlatform(platformName, platform);
}
