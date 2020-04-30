/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (global = global || self, factory(global.tf = global.tf || {}));
}(this, (function (exports) { 'use strict';

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
  // Expects flags from URL in the format ?tfjsflags=FLAG1:1,FLAG2:true.
  const TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
  /**
   * The environment contains evaluated flags as well as the registered platform.
   * This is always used as a global singleton and can be retrieved with
   * `tf.env()`.
   */
  /** @doc {heading: 'Environment'} */
  class Environment {
      // tslint:disable-next-line: no-any
      constructor(global) {
          this.global = global;
          this.flags = {};
          this.flagRegistry = {};
          this.urlFlags = {};
          this.populateURLFlags();
      }
      setPlatform(platformName, platform) {
          if (this.platform != null) {
              console.warn(`Platform ${this.platformName} has already been set. ` +
                  `Overwriting the platform with ${platform}.`);
          }
          this.platformName = platformName;
          this.platform = platform;
      }
      registerFlag(flagName, evaluationFn, setHook) {
          this.flagRegistry[flagName] = { evaluationFn, setHook };
          // Override the flag value from the URL. This has to happen here because the
          // environment is initialized before flags get registered.
          if (this.urlFlags[flagName] != null) {
              const flagValue = this.urlFlags[flagName];
              console.warn(`Setting feature override from URL ${flagName}: ${flagValue}.`);
              this.set(flagName, flagValue);
          }
      }
      get(flagName) {
          if (flagName in this.flags) {
              return this.flags[flagName];
          }
          this.flags[flagName] = this.evaluateFlag(flagName);
          return this.flags[flagName];
      }
      getNumber(flagName) {
          return this.get(flagName);
      }
      getBool(flagName) {
          return this.get(flagName);
      }
      getFlags() {
          return this.flags;
      }
      // For backwards compatibility.
      get features() {
          return this.flags;
      }
      set(flagName, value) {
          if (this.flagRegistry[flagName] == null) {
              throw new Error(`Cannot set flag ${flagName} as it has not been registered.`);
          }
          this.flags[flagName] = value;
          if (this.flagRegistry[flagName].setHook != null) {
              this.flagRegistry[flagName].setHook(value);
          }
      }
      evaluateFlag(flagName) {
          if (this.flagRegistry[flagName] == null) {
              throw new Error(`Cannot evaluate flag '${flagName}': no evaluation function found.`);
          }
          return this.flagRegistry[flagName].evaluationFn();
      }
      setFlags(flags) {
          this.flags = Object.assign({}, flags);
      }
      reset() {
          this.flags = {};
          this.urlFlags = {};
          this.populateURLFlags();
      }
      populateURLFlags() {
          if (typeof this.global === 'undefined' ||
              typeof this.global.location === 'undefined' ||
              typeof this.global.location.search === 'undefined') {
              return;
          }
          const urlParams = getQueryParams(this.global.location.search);
          if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
              const keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
              keyValues.forEach(keyValue => {
                  const [key, value] = keyValue.split(':');
                  this.urlFlags[key] = parseValue(key, value);
              });
          }
      }
  }
  function getQueryParams(queryString) {
      const params = {};
      queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (s, ...t) => {
          decodeParam(params, t[0], t[1]);
          return t.join('=');
      });
      return params;
  }
  function decodeParam(params, name, value) {
      params[decodeURIComponent(name)] = decodeURIComponent(value || '');
  }
  function parseValue(flagName, value) {
      value = value.toLowerCase();
      if (value === 'true' || value === 'false') {
          return value === 'true';
      }
      else if (`${+value}` === value) {
          return +value;
      }
      throw new Error(`Could not parse value flag value ${value} for flag ${flagName}.`);
  }
  /**
   * Returns the current environment (a global singleton).
   *
   * The environment object contains the evaluated feature values as well as the
   * active platform.
   */
  /** @doc {heading: 'Environment'} */
  function env() {
      return exports.ENV;
  }
  exports.ENV = null;
  function setEnvironmentGlobal(environment) {
      exports.ENV = environment;
  }

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  // Note that the identifier globalNameSpace is scoped to this module, but will
  // always resolve to the same global object regardless of how the module is
  // resolved.
  // tslint:disable-next-line:no-any
  let globalNameSpace;
  // tslint:disable-next-line:no-any
  function getGlobalNamespace() {
      if (globalNameSpace == null) {
          // tslint:disable-next-line:no-any
          let ns;
          if (typeof (window) !== 'undefined') {
              ns = window;
          }
          else if (typeof (global) !== 'undefined') {
              ns = global;
          }
          else if (typeof (process) !== 'undefined') {
              ns = process;
          }
          else if (typeof (self) !== 'undefined') {
              ns = self;
          }
          else {
              throw new Error('Could not find a global object');
          }
          globalNameSpace = ns;
      }
      return globalNameSpace;
  }
  // tslint:disable-next-line:no-any
  function getGlobalMap() {
      const ns = getGlobalNamespace();
      if (ns._tfGlobals == null) {
          ns._tfGlobals = new Map();
      }
      return ns._tfGlobals;
  }
  /**
   * Returns a globally accessible 'singleton' object.
   *
   * @param key the name of the object
   * @param init a function to initialize to initialize this object
   *             the first time it is fetched.
   */
  function getGlobal(key, init) {
      const globalMap = getGlobalMap();
      if (globalMap.has(key)) {
          return globalMap.get(key);
      }
      else {
          const singleton = init();
          globalMap.set(key, singleton);
          return globalMap.get(key);
      }
  }

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
  const kernelRegistry = getGlobal('kernelRegistry', () => new Map());
  const gradRegistry = getGlobal('gradRegistry', () => new Map());
  /**
   * Returns the kernel function (code) associated with the provided names.
   *
   * @param kernelName The official name of the kernel.
   * @param backendName The official name of the backend.
   */
  function getKernel(kernelName, backendName) {
      const key = makeKey(kernelName, backendName);
      return kernelRegistry.get(key);
  }
  /**
   * Returns the registered gradient info associated with the provided kernel.
   * @param kernelName The official TF kernel name.
   */
  function getGradient(kernelName) {
      return gradRegistry.get(kernelName);
  }
  function getKernelsForBackend(backendName) {
      const it = kernelRegistry.entries();
      const result = [];
      while (true) {
          const { done, value } = it.next();
          if (done) {
              break;
          }
          const [key, config] = value;
          const [backend,] = key.split('_');
          if (backend === backendName) {
              result.push(config);
          }
      }
      return result;
  }
  /**
   * Registers the function (forward pass) for the kernel in a global registry.
   *
   * @param config A config object with the following properties:
   * - `kernelName` The official name of the kernel.
   * - `backendName` The official name of the backend.
   * - `kernelFunc` The function to run during the forward pass of the kernel.
   * - `setupFunc` Optional. Gets called once, after the backend initializes.
   * - `disposeFunc` Optional. Gets called once, right before the backend is
   * disposed.
   */
  function registerKernel(config) {
      const { kernelName, backendName } = config;
      const key = makeKey(kernelName, backendName);
      if (kernelRegistry.has(key)) {
          throw new Error(`The kernel '${kernelName}' for backend ` +
              `'${backendName}' is already registered`);
      }
      kernelRegistry.set(key, config);
  }
  /**
   * Registers a gradient function for a given kernel in the global registry,
   * to be used during the back-propagation of that kernel.
   *
   * @param config An object with the following properties:
   * - `kernelName` The name of the kernel that the gradient function is for.
   * - `gradFunc` The function to run during back-propagation.
   */
  function registerGradient(config) {
      const { kernelName } = config;
      if (gradRegistry.has(kernelName)) {
          console.warn(`Overriding the gradient for '${kernelName}'`);
      }
      gradRegistry.set(kernelName, config);
  }
  /**
   * Removes the kernel function from the registry.
   *
   * @param kernelName The official name of the kernel.
   * @param backendName The official name of the backend.
   *
   */
  function unregisterKernel(kernelName, backendName) {
      const key = makeKey(kernelName, backendName);
      if (!kernelRegistry.has(key)) {
          throw new Error(`The kernel '${kernelName}' for backend ` +
              `'${backendName}' is not registered`);
      }
      kernelRegistry.delete(key);
  }
  /** Removes the registered gradient from the global registry. */
  function unregisterGradient(kernelName) {
      if (!gradRegistry.has(kernelName)) {
          throw new Error(`The gradient '${kernelName}' for backend is not registered`);
      }
      gradRegistry.delete(kernelName);
  }
  function makeKey(kernelName, backendName) {
      return `${backendName}_${kernelName}`;
  }

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
  /**
   * Shuffles the array in-place using Fisher-Yates algorithm.
   *
   * ```js
   * const a = [1, 2, 3, 4, 5];
   * tf.util.shuffle(a);
   * console.log(a);
   * ```
   *
   * @param array The array to shuffle in-place.
   */
  /** @doc {heading: 'Util', namespace: 'util'} */
  // tslint:disable-next-line:no-any
  function shuffle(array) {
      let counter = array.length;
      let temp = 0;
      let index = 0;
      // While there are elements in the array
      while (counter > 0) {
          // Pick a random index
          index = (Math.random() * counter) | 0;
          // Decrease counter by 1
          counter--;
          // And swap the last element with it
          temp = array[counter];
          array[counter] = array[index];
          array[index] = temp;
      }
  }
  /** Clamps a value to a specified range. */
  function clamp(min, x, max) {
      return Math.max(min, Math.min(x, max));
  }
  function nearestLargerEven(val) {
      return val % 2 === 0 ? val : val + 1;
  }
  function sum(arr) {
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
          sum += arr[i];
      }
      return sum;
  }
  /**
   * Returns a sample from a uniform [a, b) distribution.
   *
   * @param a The minimum support (inclusive).
   * @param b The maximum support (exclusive).
   * @return A pseudorandom number on the half-open interval [a,b).
   */
  function randUniform(a, b) {
      const r = Math.random();
      return (b * r) + (1 - r) * a;
  }
  /** Returns the squared Euclidean distance between two vectors. */
  function distSquared(a, b) {
      let result = 0;
      for (let i = 0; i < a.length; i++) {
          const diff = Number(a[i]) - Number(b[i]);
          result += diff * diff;
      }
      return result;
  }
  /**
   * Asserts that the expression is true. Otherwise throws an error with the
   * provided message.
   *
   * ```js
   * const x = 2;
   * tf.util.assert(x === 2, 'x is not 2');
   * ```
   *
   * @param expr The expression to assert (as a boolean).
   * @param msg A function that returns the message to report when throwing an
   *     error. We use a function for performance reasons.
   */
  /** @doc {heading: 'Util', namespace: 'util'} */
  function assert(expr, msg) {
      if (!expr) {
          throw new Error(typeof msg === 'string' ? msg : msg());
      }
  }
  function assertShapesMatch(shapeA, shapeB, errorMessagePrefix = '') {
      assert(arraysEqual(shapeA, shapeB), () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
  }
  function assertNonNull(a) {
      assert(a != null, () => `The input to the tensor constructor must be a non-null value.`);
  }
  // NOTE: We explicitly type out what T extends instead of any so that
  // util.flatten on a nested array of number doesn't try to infer T as a
  // number[][], causing us to explicitly type util.flatten<number>().
  /**
   *  Flattens an arbitrarily nested array.
   *
   * ```js
   * const a = [[1, 2], [3, 4], [5, [6, [7]]]];
   * const flat = tf.util.flatten(a);
   * console.log(flat);
   * ```
   *
   *  @param arr The nested array to flatten.
   *  @param result The destination array which holds the elements.
   *  @param skipTypedArray If true, avoids flattening the typed arrays. Defaults
   *      to false.
   */
  /** @doc {heading: 'Util', namespace: 'util'} */
  function flatten(arr, result = [], skipTypedArray = false) {
      if (result == null) {
          result = [];
      }
      if (Array.isArray(arr) || isTypedArray(arr) && !skipTypedArray) {
          for (let i = 0; i < arr.length; ++i) {
              flatten(arr[i], result, skipTypedArray);
          }
      }
      else {
          result.push(arr);
      }
      return result;
  }
  /**
   * Returns the size (number of elements) of the tensor given its shape.
   *
   * ```js
   * const shape = [3, 4, 2];
   * const size = tf.util.sizeFromShape(shape);
   * console.log(size);
   * ```
   */
  /** @doc {heading: 'Util', namespace: 'util'} */
  function sizeFromShape(shape) {
      if (shape.length === 0) {
          // Scalar.
          return 1;
      }
      let size = shape[0];
      for (let i = 1; i < shape.length; i++) {
          size *= shape[i];
      }
      return size;
  }
  function isScalarShape(shape) {
      return shape.length === 0;
  }
  function arraysEqual(n1, n2) {
      if (n1 === n2) {
          return true;
      }
      if (n1 == null || n2 == null) {
          return false;
      }
      if (n1.length !== n2.length) {
          return false;
      }
      for (let i = 0; i < n1.length; i++) {
          if (n1[i] !== n2[i]) {
              return false;
          }
      }
      return true;
  }
  function isInt(a) {
      return a % 1 === 0;
  }
  function tanh(x) {
      // tslint:disable-next-line:no-any
      if (Math.tanh != null) {
          // tslint:disable-next-line:no-any
          return Math.tanh(x);
      }
      if (x === Infinity) {
          return 1;
      }
      else if (x === -Infinity) {
          return -1;
      }
      else {
          const e2x = Math.exp(2 * x);
          return (e2x - 1) / (e2x + 1);
      }
  }
  function sizeToSquarishShape(size) {
      const width = Math.ceil(Math.sqrt(size));
      return [width, Math.ceil(size / width)];
  }
  /**
   * Creates a new array with randomized indicies to a given quantity.
   *
   * ```js
   * const randomTen = tf.util.createShuffledIndices(10);
   * console.log(randomTen);
   * ```
   *
   * @param number Quantity of how many shuffled indicies to create.
   */
  /** @doc {heading: 'Util', namespace: 'util'} */
  function createShuffledIndices(n) {
      const shuffledIndices = new Uint32Array(n);
      for (let i = 0; i < n; ++i) {
          shuffledIndices[i] = i;
      }
      shuffle(shuffledIndices);
      return shuffledIndices;
  }
  function rightPad(a, size) {
      if (size <= a.length) {
          return a;
      }
      return a + ' '.repeat(size - a.length);
  }
  function repeatedTry(checkFn, delayFn = (counter) => 0, maxCounter) {
      return new Promise((resolve, reject) => {
          let tryCount = 0;
          const tryFn = () => {
              if (checkFn()) {
                  resolve();
                  return;
              }
              tryCount++;
              const nextBackoff = delayFn(tryCount);
              if (maxCounter != null && tryCount >= maxCounter) {
                  reject();
                  return;
              }
              setTimeout(tryFn, nextBackoff);
          };
          tryFn();
      });
  }
  /**
   * Given the full size of the array and a shape that may contain -1 as the
   * implicit dimension, returns the inferred shape where -1 is replaced.
   * E.g. For shape=[2, -1, 3] and size=24, it will return [2, 4, 3].
   *
   * @param shape The shape, which may contain -1 in some dimension.
   * @param size The full size (number of elements) of the array.
   * @return The inferred shape where -1 is replaced with the inferred size.
   */
  function inferFromImplicitShape(shape, size) {
      let shapeProd = 1;
      let implicitIdx = -1;
      for (let i = 0; i < shape.length; ++i) {
          if (shape[i] >= 0) {
              shapeProd *= shape[i];
          }
          else if (shape[i] === -1) {
              if (implicitIdx !== -1) {
                  throw Error(`Shapes can only have 1 implicit size. ` +
                      `Found -1 at dim ${implicitIdx} and dim ${i}`);
              }
              implicitIdx = i;
          }
          else if (shape[i] < 0) {
              throw Error(`Shapes can not be < 0. Found ${shape[i]} at dim ${i}`);
          }
      }
      if (implicitIdx === -1) {
          if (size > 0 && size !== shapeProd) {
              throw Error(`Size(${size}) must match the product of shape ${shape}`);
          }
          return shape;
      }
      if (shapeProd === 0) {
          throw Error(`Cannot infer the missing size in [${shape}] when ` +
              `there are 0 elements`);
      }
      if (size % shapeProd !== 0) {
          throw Error(`The implicit shape can't be a fractional number. ` +
              `Got ${size} / ${shapeProd}`);
      }
      const newShape = shape.slice();
      newShape[implicitIdx] = size / shapeProd;
      return newShape;
  }
  function parseAxisParam(axis, shape) {
      const rank = shape.length;
      // Normalize input
      axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);
      // Check for valid range
      assert(axis.every(ax => ax >= -rank && ax < rank), () => `All values in axis param must be in range [-${rank}, ${rank}) but ` +
          `got axis ${axis}`);
      // Check for only integers
      assert(axis.every(ax => isInt(ax)), () => `All values in axis param must be integers but ` +
          `got axis ${axis}`);
      // Handle negative axis.
      return axis.map(a => a < 0 ? rank + a : a);
  }
  /** Reduces the shape by removing all dimensions of shape 1. */
  function squeezeShape(shape, axis) {
      const newShape = [];
      const keptDims = [];
      const isEmptyArray = axis != null && Array.isArray(axis) && axis.length === 0;
      const axes = (axis == null || isEmptyArray) ?
          null :
          parseAxisParam(axis, shape).sort();
      let j = 0;
      for (let i = 0; i < shape.length; ++i) {
          if (axes != null) {
              if (axes[j] === i && shape[i] !== 1) {
                  throw new Error(`Can't squeeze axis ${i} since its dim '${shape[i]}' is not 1`);
              }
              if ((axes[j] == null || axes[j] > i) && shape[i] === 1) {
                  newShape.push(shape[i]);
                  keptDims.push(i);
              }
              if (axes[j] <= i) {
                  j++;
              }
          }
          if (shape[i] !== 1) {
              newShape.push(shape[i]);
              keptDims.push(i);
          }
      }
      return { newShape, keptDims };
  }
  function getTypedArrayFromDType(dtype, size) {
      let values = null;
      if (dtype == null || dtype === 'float32') {
          values = new Float32Array(size);
      }
      else if (dtype === 'int32') {
          values = new Int32Array(size);
      }
      else if (dtype === 'bool') {
          values = new Uint8Array(size);
      }
      else {
          throw new Error(`Unknown data type ${dtype}`);
      }
      return values;
  }
  function getArrayFromDType(dtype, size) {
      let values = null;
      if (dtype == null || dtype === 'float32') {
          values = new Float32Array(size);
      }
      else if (dtype === 'int32') {
          values = new Int32Array(size);
      }
      else if (dtype === 'bool') {
          values = new Uint8Array(size);
      }
      else if (dtype === 'string') {
          values = new Array(size);
      }
      else {
          throw new Error(`Unknown data type ${dtype}`);
      }
      return values;
  }
  function checkConversionForErrors(vals, dtype) {
      for (let i = 0; i < vals.length; i++) {
          const num = vals[i];
          if (isNaN(num) || !isFinite(num)) {
              throw Error(`A tensor of type ${dtype} being uploaded contains ${num}.`);
          }
      }
  }
  /** Returns true if the dtype is valid. */
  function isValidDtype(dtype) {
      return dtype === 'bool' || dtype === 'complex64' || dtype === 'float32' ||
          dtype === 'int32' || dtype === 'string';
  }
  /**
   * Returns true if the new type can't encode the old type without loss of
   * precision.
   */
  function hasEncodingLoss(oldType, newType) {
      if (newType === 'complex64') {
          return false;
      }
      if (newType === 'float32' && oldType !== 'complex64') {
          return false;
      }
      if (newType === 'int32' && oldType !== 'float32' && oldType !== 'complex64') {
          return false;
      }
      if (newType === 'bool' && oldType === 'bool') {
          return false;
      }
      return true;
  }
  function isTypedArray(a) {
      return a instanceof Float32Array || a instanceof Int32Array ||
          a instanceof Uint8Array;
  }
  function bytesPerElement(dtype) {
      if (dtype === 'float32' || dtype === 'int32') {
          return 4;
      }
      else if (dtype === 'complex64') {
          return 8;
      }
      else if (dtype === 'bool') {
          return 1;
      }
      else {
          throw new Error(`Unknown dtype ${dtype}`);
      }
  }
  /**
   * Returns the approximate number of bytes allocated in the string array - 2
   * bytes per character. Computing the exact bytes for a native string in JS is
   * not possible since it depends on the encoding of the html page that serves
   * the website.
   */
  function bytesFromStringArray(arr) {
      if (arr == null) {
          return 0;
      }
      let bytes = 0;
      arr.forEach(x => bytes += x.length);
      return bytes;
  }
  /** Returns true if the value is a string. */
  function isString(value) {
      return typeof value === 'string' || value instanceof String;
  }
  function isBoolean(value) {
      return typeof value === 'boolean';
  }
  function isNumber(value) {
      return typeof value === 'number';
  }
  function inferDtype(values) {
      if (Array.isArray(values)) {
          return inferDtype(values[0]);
      }
      if (values instanceof Float32Array) {
          return 'float32';
      }
      else if (values instanceof Int32Array || values instanceof Uint8Array) {
          return 'int32';
      }
      else if (isNumber(values)) {
          return 'float32';
      }
      else if (isString(values)) {
          return 'string';
      }
      else if (isBoolean(values)) {
          return 'bool';
      }
      return 'float32';
  }
  function isFunction(f) {
      return !!(f && f.constructor && f.call && f.apply);
  }
  function nearestDivisor(size, start) {
      for (let i = start; i < size; ++i) {
          if (size % i === 0) {
              return i;
          }
      }
      return size;
  }
  function computeStrides(shape) {
      const rank = shape.length;
      if (rank < 2) {
          return [];
      }
      // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
      // strides.
      const strides = new Array(rank - 1);
      strides[rank - 2] = shape[rank - 1];
      for (let i = rank - 3; i >= 0; --i) {
          strides[i] = strides[i + 1] * shape[i + 1];
      }
      return strides;
  }
  function toTypedArray(a, dtype, debugMode) {
      if (dtype === 'string') {
          throw new Error('Cannot convert a string[] to a TypedArray');
      }
      if (Array.isArray(a)) {
          a = flatten(a);
      }
      if (debugMode) {
          checkConversionForErrors(a, dtype);
      }
      if (noConversionNeeded(a, dtype)) {
          return a;
      }
      if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
          return new Float32Array(a);
      }
      else if (dtype === 'int32') {
          return new Int32Array(a);
      }
      else if (dtype === 'bool') {
          const bool = new Uint8Array(a.length);
          for (let i = 0; i < bool.length; ++i) {
              if (Math.round(a[i]) !== 0) {
                  bool[i] = 1;
              }
          }
          return bool;
      }
      else {
          throw new Error(`Unknown data type ${dtype}`);
      }
  }
  function createNestedArray(offset, shape, a) {
      const ret = new Array();
      if (shape.length === 1) {
          const d = shape[0];
          for (let i = 0; i < d; i++) {
              ret[i] = a[offset + i];
          }
      }
      else {
          const d = shape[0];
          const rest = shape.slice(1);
          const len = rest.reduce((acc, c) => acc * c);
          for (let i = 0; i < d; i++) {
              ret[i] = createNestedArray(offset + i * len, rest, a);
          }
      }
      return ret;
  }
  // Provide a nested array of TypedArray in given shape.
  function toNestedArray(shape, a) {
      if (shape.length === 0) {
          // Scalar type should return a single number.
          return a[0];
      }
      const size = shape.reduce((acc, c) => acc * c);
      if (size === 0) {
          // A tensor with shape zero should be turned into empty list.
          return [];
      }
      if (size !== a.length) {
          throw new Error(`[${shape}] does not match the input size.`);
      }
      return createNestedArray(0, shape, a);
  }
  function noConversionNeeded(a, dtype) {
      return (a instanceof Float32Array && dtype === 'float32') ||
          (a instanceof Int32Array && dtype === 'int32') ||
          (a instanceof Uint8Array && dtype === 'bool');
  }
  function makeOnesTypedArray(size, dtype) {
      const array = makeZerosTypedArray(size, dtype);
      for (let i = 0; i < array.length; i++) {
          array[i] = 1;
      }
      return array;
  }
  function makeZerosTypedArray(size, dtype) {
      if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
          return new Float32Array(size);
      }
      else if (dtype === 'int32') {
          return new Int32Array(size);
      }
      else if (dtype === 'bool') {
          return new Uint8Array(size);
      }
      else {
          throw new Error(`Unknown data type ${dtype}`);
      }
  }
  /**
   * Returns the current high-resolution time in milliseconds relative to an
   * arbitrary time in the past. It works across different platforms (node.js,
   * browsers).
   *
   * ```js
   * console.log(tf.util.now());
   * ```
   */
  /** @doc {heading: 'Util', namespace: 'util'} */
  function now() {
      return env().platform.now();
  }
  function assertNonNegativeIntegerDimensions(shape) {
      shape.forEach(dimSize => {
          assert(Number.isInteger(dimSize) && dimSize >= 0, () => `Tensor must have a shape comprised of positive integers but got ` +
              `shape [${shape}].`);
      });
  }
  /**
   * Returns a platform-specific implementation of
   * [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
   *
   * If `fetch` is defined on the global object (`window`, `process`, etc.),
   * `tf.util.fetch` returns that function.
   *
   * If not, `tf.util.fetch` returns a platform-specific solution.
   *
   * ```js
   * const resource = await tf.util.fetch('https://unpkg.com/@tensorflow/tfjs');
   * // handle response
   * ```
   */
  /** @doc {heading: 'Util'} */
  function fetch$1(path, requestInits) {
      return env().platform.fetch(path, requestInits);
  }
  /**
   * Encodes the provided string into bytes using the provided encoding scheme.
   *
   * @param s The string to encode.
   * @param encoding The encoding scheme. Defaults to utf-8.
   *
   */
  /** @doc {heading: 'Util'} */
  function encodeString(s, encoding = 'utf-8') {
      encoding = encoding || 'utf-8';
      return env().platform.encode(s, encoding);
  }
  /**
   * Decodes the provided bytes into a string using the provided encoding scheme.
   * @param bytes The bytes to decode.
   *
   * @param encoding The encoding scheme. Defaults to utf-8.
   */
  /** @doc {heading: 'Util'} */
  function decodeString(bytes, encoding = 'utf-8') {
      encoding = encoding || 'utf-8';
      return env().platform.decode(bytes, encoding);
  }
  /**
   * Computes flat index for a given location (multidimentionsal index) in a
   * Tensor/multidimensional array.
   *
   * @param locs Location in the tensor.
   * @param rank Rank of the tensor.
   * @param strides Tensor strides.
   */
  function locToIndex(locs, rank, strides) {
      if (rank === 0) {
          return 0;
      }
      else if (rank === 1) {
          return locs[0];
      }
      let index = locs[locs.length - 1];
      for (let i = 0; i < locs.length - 1; ++i) {
          index += strides[i] * locs[i];
      }
      return index;
  }
  /**
   * Computes the location (multidimensional index) in a tensor/multidimentional
   * array for a given flat index.
   *
   * @param index Index in flat array.
   * @param rank Rank of tensor.
   * @param strides Strides of tensor.
   */
  function indexToLoc(index, rank, strides) {
      if (rank === 0) {
          return [];
      }
      else if (rank === 1) {
          return [index];
      }
      const locs = new Array(rank);
      for (let i = 0; i < locs.length - 1; ++i) {
          locs[i] = Math.floor(index / strides[i]);
          index -= locs[i] * strides[i];
      }
      locs[locs.length - 1] = index;
      return locs;
  }

  var util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    shuffle: shuffle,
    clamp: clamp,
    nearestLargerEven: nearestLargerEven,
    sum: sum,
    randUniform: randUniform,
    distSquared: distSquared,
    assert: assert,
    assertShapesMatch: assertShapesMatch,
    assertNonNull: assertNonNull,
    flatten: flatten,
    sizeFromShape: sizeFromShape,
    isScalarShape: isScalarShape,
    arraysEqual: arraysEqual,
    isInt: isInt,
    tanh: tanh,
    sizeToSquarishShape: sizeToSquarishShape,
    createShuffledIndices: createShuffledIndices,
    rightPad: rightPad,
    repeatedTry: repeatedTry,
    inferFromImplicitShape: inferFromImplicitShape,
    parseAxisParam: parseAxisParam,
    squeezeShape: squeezeShape,
    getTypedArrayFromDType: getTypedArrayFromDType,
    getArrayFromDType: getArrayFromDType,
    checkConversionForErrors: checkConversionForErrors,
    isValidDtype: isValidDtype,
    hasEncodingLoss: hasEncodingLoss,
    isTypedArray: isTypedArray,
    bytesPerElement: bytesPerElement,
    bytesFromStringArray: bytesFromStringArray,
    isString: isString,
    isBoolean: isBoolean,
    isNumber: isNumber,
    inferDtype: inferDtype,
    isFunction: isFunction,
    nearestDivisor: nearestDivisor,
    computeStrides: computeStrides,
    toTypedArray: toTypedArray,
    toNestedArray: toNestedArray,
    makeOnesTypedArray: makeOnesTypedArray,
    makeZerosTypedArray: makeZerosTypedArray,
    now: now,
    assertNonNegativeIntegerDimensions: assertNonNegativeIntegerDimensions,
    fetch: fetch$1,
    encodeString: encodeString,
    decodeString: decodeString,
    locToIndex: locToIndex,
    indexToLoc: indexToLoc
  });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class Profiler {
      constructor(backendTimer, logger) {
          this.backendTimer = backendTimer;
          this.logger = logger;
          if (logger == null) {
              this.logger = new Logger();
          }
      }
      profileKernel(kernelName, inputs, f) {
          let outputs;
          const holdResultWrapperFn = () => {
              outputs = f();
          };
          const timer = this.backendTimer.time(holdResultWrapperFn);
          outputs.forEach(r => {
              // Dangling promise here because we don't want to propagate up
              // asynchronicity.
              r.data().then(vals => {
                  checkComputationForErrors(vals, r.dtype, kernelName);
                  timer.then(timing => {
                      let extraInfo = '';
                      if (timing.getExtraProfileInfo != null) {
                          extraInfo = timing.getExtraProfileInfo();
                      }
                      this.logger.logKernelProfile(kernelName, r, vals, timing.kernelMs, inputs, extraInfo);
                  });
              });
          });
          return outputs;
      }
  }
  function checkComputationForErrors(vals, dtype, kernelName) {
      if (dtype !== 'float32') {
          // Only floating point computations will generate NaN values
          return false;
      }
      for (let i = 0; i < vals.length; i++) {
          const num = vals[i];
          if (isNaN(num) || !isFinite(num)) {
              // Throwing custom exception so behavior is testable.
              console.warn(`Found ${num} in the result of '${kernelName}'`);
              return true;
          }
      }
      return false;
  }
  class Logger {
      logKernelProfile(name, result, vals, timeMs, inputs, extraInfo) {
          const time = typeof timeMs === 'number' ? rightPad(`${timeMs}ms`, 9) :
              timeMs['error'];
          const paddedName = rightPad(name, 25);
          const rank = result.rank;
          const size = result.size;
          const shape = rightPad(result.shape.toString(), 14);
          let inputShapesDescription = '';
          for (const name in inputs) {
              const input = inputs[name];
              // The input might be a non-tensor (e.g HTMLImageElement), in which case
              // we claim the output shape as input shape.
              const inputShape = input.shape || result.shape;
              const inputRank = inputShape.length;
              inputShapesDescription +=
                  `${name}: ${inputRank}D ${inputRank > 0 ? inputShape : ''} `;
          }
          console.log(`%c${paddedName}\t%c${time}\t%c${rank}D ${shape}\t%c${size}\t%c${inputShapesDescription}\t%c${extraInfo}`, 'font-weight:bold', 'color:red', 'color:blue', 'color: orange', 'color: green', 'color: steelblue');
      }
  }

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
  /**
   * Computes a list of TapeNodes that connect x to y, filtering everything else
   * out and preserving the order of the original tape elements.
   *
   * @param tape The tape elements to filter.
   * @param xs The input Tensors.
   * @param y The output Tensor.
   */
  function getFilteredNodesXToY(tape, xs, y) {
      // Forward pass to compute all the nodes and Tensors that are transitively a
      // function of x.
      const tensorsFromX = {};
      const nodesFromX = {};
      for (let i = 0; i < xs.length; i++) {
          tensorsFromX[xs[i].id] = true;
      }
      for (let i = 0; i < tape.length; i++) {
          const node = tape[i];
          const nodeInputs = node.inputs;
          for (const inputName in nodeInputs) {
              const input = nodeInputs[inputName];
              let anyInputFromX = false;
              for (let j = 0; j < xs.length; j++) {
                  if (tensorsFromX[input.id]) {
                      node.outputs.forEach(output => tensorsFromX[output.id] = true);
                      anyInputFromX = true;
                      nodesFromX[node.id] = true;
                      break;
                  }
              }
              if (anyInputFromX) {
                  break;
              }
          }
      }
      // Backward pass to find all of the nodes and Tensors that lead to y.
      const tensorsLeadToY = {};
      tensorsLeadToY[y.id] = true;
      const nodesToY = {};
      for (let i = tape.length - 1; i >= 0; i--) {
          const node = tape[i];
          const nodeInputs = node.inputs;
          // If any of the outputs lead to y, mark all of the inputs as leading to y.
          for (let j = 0; j < node.outputs.length; j++) {
              if (tensorsLeadToY[node.outputs[j].id]) {
                  for (const inputName in nodeInputs) {
                      tensorsLeadToY[nodeInputs[inputName].id] = true;
                      nodesToY[node.id] = true;
                  }
                  break;
              }
          }
      }
      // Return the paths that come from x and lead to y.
      const filteredTape = [];
      for (let i = 0; i < tape.length; i++) {
          const node = tape[i];
          if (nodesFromX[node.id] && nodesToY[node.id]) {
              // Prune the inputs from the node that aren't a function of x.
              const prunedInputs = {};
              for (const inputName in node.inputs) {
                  const nodeInput = node.inputs[inputName];
                  if (tensorsFromX[nodeInput.id]) {
                      prunedInputs[inputName] = nodeInput;
                  }
              }
              // Copy the node and overwrite inputsAndArgs to the pruned version.
              const prunedNode = Object.assign({}, node);
              prunedNode.inputs = prunedInputs;
              prunedNode.outputs = node.outputs;
              filteredTape.push(prunedNode);
          }
      }
      return filteredTape;
  }
  /**
   * Backpropagate gradients through the filtered TapeNodes.
   *
   * @param tensorAccumulatedGradientMap A map of Tensor to its gradient. This map
   * is mutated by this method.
   * @param filteredTape The filtered TapeNodes to backprop through.
   */
  function backpropagateGradients(tensorAccumulatedGradientMap, filteredTape, tidy) {
      // Walk the tape backward and keep a map of Tensor to its gradient.
      for (let i = filteredTape.length - 1; i >= 0; i--) {
          const node = filteredTape[i];
          const dys = [];
          node.outputs.forEach(o => {
              const gradTensor = tensorAccumulatedGradientMap[o.id];
              if (gradTensor != null) {
                  dys.push(gradTensor);
              }
              else {
                  // This particular output is not in the back-propagation subgraph, so it
                  // does not affect the final output, thus we put null for its dy.
                  dys.push(null);
              }
          });
          if (node.gradient == null) {
              throw new Error(`Cannot compute gradient: gradient function not found ` +
                  `for ${node.kernelName}.`);
          }
          // Backprop dy through this node and accumulate gradients over the inputs.
          const inputGradients = node.gradient(dys);
          for (const inputName in node.inputs) {
              if (!(inputName in inputGradients)) {
                  throw new Error(`Cannot backprop through input ${inputName}. ` +
                      `Available gradients found: ${Object.keys(inputGradients)}.`);
              }
              // Call the gradient function.
              const dx = tidy(() => inputGradients[inputName]());
              if (dx.dtype !== 'float32') {
                  throw new Error(`Error in gradient for op ${node.kernelName}. The gradient of input ` +
                      `${inputName} must have 'float32' dtype, but has '${dx.dtype}'`);
              }
              const x = node.inputs[inputName];
              if (!arraysEqual(dx.shape, x.shape)) {
                  throw new Error(`Error in gradient for op ${node.kernelName}. The gradient of input ` +
                      `'${inputName}' has shape '${dx.shape}', which does not match ` +
                      `the shape of the input '${x.shape}'`);
              }
              if (tensorAccumulatedGradientMap[x.id] == null) {
                  tensorAccumulatedGradientMap[x.id] = dx;
              }
              else {
                  const curGradient = tensorAccumulatedGradientMap[x.id];
                  tensorAccumulatedGradientMap[x.id] = curGradient.add(dx);
                  curGradient.dispose();
              }
          }
      }
  }

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
  // Maximum number of values before we decide to show ellipsis.
  const FORMAT_LIMIT_NUM_VALS = 20;
  // Number of first and last values to show when displaying a, b,...,y, z.
  const FORMAT_NUM_FIRST_LAST_VALS = 3;
  // Number of significant digits to show.
  const FORMAT_NUM_SIG_DIGITS = 7;
  function tensorToString(vals, shape, dtype, verbose) {
      const strides = computeStrides(shape);
      const padPerCol = computeMaxSizePerColumn(vals, shape, dtype, strides);
      const rank = shape.length;
      const valsLines = subTensorToString(vals, shape, dtype, strides, padPerCol);
      const lines = ['Tensor'];
      if (verbose) {
          lines.push(`  dtype: ${dtype}`);
          lines.push(`  rank: ${rank}`);
          lines.push(`  shape: [${shape}]`);
          lines.push(`  values:`);
      }
      lines.push(valsLines.map(l => '    ' + l).join('\n'));
      return lines.join('\n');
  }
  function computeMaxSizePerColumn(vals, shape, dtype, strides) {
      const n = sizeFromShape(shape);
      const numCols = strides[strides.length - 1];
      const padPerCol = new Array(numCols).fill(0);
      const rank = shape.length;
      const valuesOrTuples = dtype === 'complex64' ? createComplexTuples(vals) : vals;
      if (rank > 1) {
          for (let row = 0; row < n / numCols; row++) {
              const offset = row * numCols;
              for (let j = 0; j < numCols; j++) {
                  padPerCol[j] = Math.max(padPerCol[j], valToString(valuesOrTuples[offset + j], 0, dtype).length);
              }
          }
      }
      return padPerCol;
  }
  function valToString(val, pad, dtype) {
      let valStr;
      if (Array.isArray(val)) {
          valStr = `${parseFloat(val[0].toFixed(FORMAT_NUM_SIG_DIGITS))} + ` +
              `${parseFloat(val[1].toFixed(FORMAT_NUM_SIG_DIGITS))}j`;
      }
      else if (isString(val)) {
          valStr = `'${val}'`;
      }
      else if (dtype === 'bool') {
          valStr = boolNumToString(val);
      }
      else {
          valStr = parseFloat(val.toFixed(FORMAT_NUM_SIG_DIGITS)).toString();
      }
      return rightPad(valStr, pad);
  }
  function boolNumToString(v) {
      return v === 0 ? 'false' : 'true';
  }
  function subTensorToString(vals, shape, dtype, strides, padPerCol, isLast = true) {
      const storagePerElement = dtype === 'complex64' ? 2 : 1;
      const size = shape[0];
      const rank = shape.length;
      if (rank === 0) {
          if (dtype === 'complex64') {
              const complexTuple = createComplexTuples(vals);
              return [valToString(complexTuple[0], 0, dtype)];
          }
          if (dtype === 'bool') {
              return [boolNumToString(vals[0])];
          }
          return [vals[0].toString()];
      }
      if (rank === 1) {
          if (size > FORMAT_LIMIT_NUM_VALS) {
              const firstValsSize = FORMAT_NUM_FIRST_LAST_VALS * storagePerElement;
              let firstVals = Array.from(vals.slice(0, firstValsSize));
              let lastVals = Array.from(vals.slice((size - FORMAT_NUM_FIRST_LAST_VALS) * storagePerElement, size * storagePerElement));
              if (dtype === 'complex64') {
                  firstVals = createComplexTuples(firstVals);
                  lastVals = createComplexTuples(lastVals);
              }
              return [
                  '[' +
                      firstVals.map((x, i) => valToString(x, padPerCol[i], dtype))
                          .join(', ') +
                      ', ..., ' +
                      lastVals
                          .map((x, i) => valToString(x, padPerCol[size - FORMAT_NUM_FIRST_LAST_VALS + i], dtype))
                          .join(', ') +
                      ']'
              ];
          }
          const displayVals = dtype === 'complex64' ? createComplexTuples(vals) :
              Array.from(vals);
          return [
              '[' +
                  displayVals.map((x, i) => valToString(x, padPerCol[i], dtype))
                      .join(', ') +
                  ']'
          ];
      }
      // The array is rank 2 or more.
      const subshape = shape.slice(1);
      const substrides = strides.slice(1);
      const stride = strides[0] * storagePerElement;
      const lines = [];
      if (size > FORMAT_LIMIT_NUM_VALS) {
          for (let i = 0; i < FORMAT_NUM_FIRST_LAST_VALS; i++) {
              const start = i * stride;
              const end = start + stride;
              lines.push(...subTensorToString(vals.slice(start, end), subshape, dtype, substrides, padPerCol, false /* isLast */));
          }
          lines.push('...');
          for (let i = size - FORMAT_NUM_FIRST_LAST_VALS; i < size; i++) {
              const start = i * stride;
              const end = start + stride;
              lines.push(...subTensorToString(vals.slice(start, end), subshape, dtype, substrides, padPerCol, i === size - 1 /* isLast */));
          }
      }
      else {
          for (let i = 0; i < size; i++) {
              const start = i * stride;
              const end = start + stride;
              lines.push(...subTensorToString(vals.slice(start, end), subshape, dtype, substrides, padPerCol, i === size - 1 /* isLast */));
          }
      }
      const sep = rank === 2 ? ',' : '';
      lines[0] = '[' + lines[0] + sep;
      for (let i = 1; i < lines.length - 1; i++) {
          lines[i] = ' ' + lines[i] + sep;
      }
      let newLineSep = ',\n';
      for (let i = 2; i < rank; i++) {
          newLineSep += '\n';
      }
      lines[lines.length - 1] =
          ' ' + lines[lines.length - 1] + ']' + (isLast ? '' : newLineSep);
      return lines;
  }
  function createComplexTuples(vals) {
      const complexTuples = [];
      for (let i = 0; i < vals.length; i += 2) {
          complexTuples.push([vals[i], vals[i + 1]]);
      }
      return complexTuples;
  }

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
  /**
   * A mutable object, similar to `tf.Tensor`, that allows users to set values
   * at locations before converting to an immutable `tf.Tensor`.
   *
   * See `tf.buffer` for creating a tensor buffer.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  class TensorBuffer {
      constructor(shape, dtype, values) {
          this.dtype = dtype;
          this.shape = shape.slice();
          this.size = sizeFromShape(shape);
          if (values != null) {
              const n = values.length;
              assert(n === this.size, () => `Length of values '${n}' does not match the size ` +
                  `inferred by the shape '${this.size}'.`);
          }
          if (dtype === 'complex64') {
              throw new Error(`complex64 dtype TensorBuffers are not supported. Please create ` +
                  `a TensorBuffer for the real and imaginary parts separately and ` +
                  `call tf.complex(real, imag).`);
          }
          this.values = values || getArrayFromDType(dtype, this.size);
          this.strides = computeStrides(shape);
      }
      /**
       * Sets a value in the buffer at a given location.
       *
       * @param value The value to set.
       * @param locs  The location indices.
       */
      /** @doc {heading: 'Tensors', subheading: 'Creation'} */
      set(value, ...locs) {
          if (locs.length === 0) {
              locs = [0];
          }
          assert(locs.length === this.rank, () => `The number of provided coordinates (${locs.length}) must ` +
              `match the rank (${this.rank})`);
          const index = this.locToIndex(locs);
          this.values[index] = value;
      }
      /**
       * Returns the value in the buffer at the provided location.
       *
       * @param locs The location indices.
       */
      /** @doc {heading: 'Tensors', subheading: 'Creation'} */
      get(...locs) {
          if (locs.length === 0) {
              locs = [0];
          }
          let i = 0;
          for (const loc of locs) {
              if (loc < 0 || loc >= this.shape[i]) {
                  const msg = `Requested out of range element at ${locs}. ` +
                      `  Buffer shape=${this.shape}`;
                  throw new Error(msg);
              }
              i++;
          }
          let index = locs[locs.length - 1];
          for (let i = 0; i < locs.length - 1; ++i) {
              index += this.strides[i] * locs[i];
          }
          return this.values[index];
      }
      locToIndex(locs) {
          if (this.rank === 0) {
              return 0;
          }
          else if (this.rank === 1) {
              return locs[0];
          }
          let index = locs[locs.length - 1];
          for (let i = 0; i < locs.length - 1; ++i) {
              index += this.strides[i] * locs[i];
          }
          return index;
      }
      indexToLoc(index) {
          if (this.rank === 0) {
              return [];
          }
          else if (this.rank === 1) {
              return [index];
          }
          const locs = new Array(this.shape.length);
          for (let i = 0; i < locs.length - 1; ++i) {
              locs[i] = Math.floor(index / this.strides[i]);
              index -= locs[i] * this.strides[i];
          }
          locs[locs.length - 1] = index;
          return locs;
      }
      get rank() {
          return this.shape.length;
      }
      /**
       * Creates an immutable `tf.Tensor` object from the buffer.
       */
      /** @doc {heading: 'Tensors', subheading: 'Creation'} */
      toTensor() {
          return trackerFn().makeTensor(this.values, this.shape, this.dtype);
      }
  }
  // For tracking tensor creation and disposal.
  let trackerFn = null;
  // Used by chaining methods to call into ops.
  let opHandler = null;
  // Used to warn about deprecated methods.
  let deprecationWarningFn = null;
  /**
   * An external consumer can register itself as the tensor tracker. This way
   * the Tensor class can notify the tracker for every tensor created and
   * disposed.
   */
  function setTensorTracker(fn) {
      trackerFn = fn;
  }
  /**
   * An external consumer can register itself as the op handler. This way the
   * Tensor class can have chaining methods that call into ops via the op
   * handler.
   */
  function setOpHandler(handler) {
      opHandler = handler;
  }
  /**
   * Sets the deprecation warning function to be used by this file. This way the
   * Tensor class can be a leaf but still use the environment.
   */
  function setDeprecationWarningFn(fn) {
      deprecationWarningFn = fn;
  }
  /**
   * A `tf.Tensor` object represents an immutable, multidimensional array of
   * numbers that has a shape and a data type.
   *
   * See `tf.tensor` for details on how to create a `tf.Tensor`.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  class Tensor {
      constructor(shape, dtype, dataId, id) {
          /** Whether this tensor has been globally kept. */
          this.kept = false;
          this.isDisposedInternal = false;
          this.shape = shape.slice();
          this.dtype = dtype || 'float32';
          this.size = sizeFromShape(shape);
          this.strides = computeStrides(shape);
          this.dataId = dataId;
          this.id = id;
          this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
      }
      /** Flatten a Tensor to a 1D array. */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      flatten() {
          this.throwIfDisposed();
          return this.as1D();
      }
      /** Converts a size-1 `tf.Tensor` to a `tf.Scalar`. */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      asScalar() {
          this.throwIfDisposed();
          assert(this.size === 1, () => 'The array must have only 1 element.');
          return this.reshape([]);
      }
      /** Converts a `tf.Tensor` to a `tf.Tensor1D`. */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      as1D() {
          this.throwIfDisposed();
          return this.reshape([this.size]);
      }
      /**
       * Converts a `tf.Tensor` to a `tf.Tensor2D`.
       *
       * @param rows Number of rows in `tf.Tensor2D`.
       * @param columns Number of columns in `tf.Tensor2D`.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      as2D(rows, columns) {
          this.throwIfDisposed();
          return this.reshape([rows, columns]);
      }
      /**
       * Converts a `tf.Tensor` to a `tf.Tensor3D`.
       *
       * @param rows Number of rows in `tf.Tensor3D`.
       * @param columns Number of columns in `tf.Tensor3D`.
       * @param depth Depth of `tf.Tensor3D`.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      as3D(rows, columns, depth) {
          this.throwIfDisposed();
          return this.reshape([rows, columns, depth]);
      }
      /**
       * Converts a `tf.Tensor` to a `tf.Tensor4D`.
       *
       * @param rows Number of rows in `tf.Tensor4D`.
       * @param columns Number of columns in `tf.Tensor4D`.
       * @param depth Depth of `tf.Tensor4D`.
       * @param depth2 4th dimension of `tf.Tensor4D`.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      as4D(rows, columns, depth, depth2) {
          this.throwIfDisposed();
          return this.reshape([rows, columns, depth, depth2]);
      }
      /**
       * Converts a `tf.Tensor` to a `tf.Tensor5D`.
       *
       * @param rows Number of rows in `tf.Tensor5D`.
       * @param columns Number of columns in `tf.Tensor5D`.
       * @param depth Depth of `tf.Tensor5D`.
       * @param depth2 4th dimension of `tf.Tensor5D`.
       * @param depth3 5th dimension of 'tf.Tensor5D'
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      as5D(rows, columns, depth, depth2, depth3) {
          this.throwIfDisposed();
          return this.reshape([rows, columns, depth, depth2, depth3]);
      }
      /**
       * Casts a `tf.Tensor` to a specified dtype.
       *
       * @param dtype Data-type to cast the tensor to.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      asType(dtype) {
          this.throwIfDisposed();
          return opHandler.cast(this, dtype);
      }
      get rank() {
          return this.shape.length;
      }
      /**
       * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      async buffer() {
          const vals = await this.data();
          return opHandler.buffer(this.shape, this.dtype, vals);
      }
      /** Returns a `tf.TensorBuffer` that holds the underlying data. */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      bufferSync() {
          return opHandler.buffer(this.shape, this.dtype, this.dataSync());
      }
      /**
       * Returns the tensor data as a nested array. The transfer of data is done
       * asynchronously.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      async array() {
          const vals = await this.data();
          return toNestedArray(this.shape, vals);
      }
      /**
       * Returns the tensor data as a nested array. The transfer of data is done
       * synchronously.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      arraySync() {
          return toNestedArray(this.shape, this.dataSync());
      }
      /**
       * Asynchronously downloads the values from the `tf.Tensor`. Returns a
       * promise of `TypedArray` that resolves when the computation has finished.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      async data() {
          this.throwIfDisposed();
          const data = trackerFn().read(this.dataId);
          if (this.dtype === 'string') {
              const bytes = await data;
              try {
                  return bytes.map(b => decodeString(b));
              }
              catch (_a) {
                  throw new Error('Failed to decode the string bytes into utf-8. ' +
                      'To get the original bytes, call tensor.bytes().');
              }
          }
          return data;
      }
      /**
       * Synchronously downloads the values from the `tf.Tensor`. This blocks the
       * UI thread until the values are ready, which can cause performance issues.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      dataSync() {
          this.throwIfDisposed();
          const data = trackerFn().readSync(this.dataId);
          if (this.dtype === 'string') {
              try {
                  return data.map(b => decodeString(b));
              }
              catch (_a) {
                  throw new Error('Failed to decode the string bytes into utf-8. ' +
                      'To get the original bytes, call tensor.bytes().');
              }
          }
          return data;
      }
      /** Returns the underlying bytes of the tensor's data. */
      async bytes() {
          this.throwIfDisposed();
          const data = await trackerFn().read(this.dataId);
          if (this.dtype === 'string') {
              return data;
          }
          else {
              return new Uint8Array(data.buffer);
          }
      }
      /**
       * Disposes `tf.Tensor` from memory.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      dispose() {
          if (this.isDisposed) {
              return;
          }
          trackerFn().disposeTensor(this);
          this.isDisposedInternal = true;
      }
      get isDisposed() {
          return this.isDisposedInternal;
      }
      throwIfDisposed() {
          if (this.isDisposed) {
              throw new Error(`Tensor is disposed.`);
          }
      }
      /** Casts the array to type `float32` */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      toFloat() {
          return this.asType('float32');
      }
      /** Casts the array to type `int32` */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      toInt() {
          return this.asType('int32');
      }
      /** Casts the array to type `bool` */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      toBool() {
          return this.asType('bool');
      }
      /**
       * Prints the `tf.Tensor`. See `tf.print` for details.
       *
       * @param verbose Whether to print verbose information about the tensor,
       *    including dtype and size.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      print(verbose = false) {
          return opHandler.print(this, verbose);
      }
      /**
       * Reshapes the tensor into the provided shape.
       * See `tf.reshape` for more details.
       *
       * @param newShape An array of integers defining the output tensor shape.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      reshape(newShape) {
          this.throwIfDisposed();
          return opHandler.reshape(this, newShape);
      }
      /**
       * Reshapes the tensor into the shape of the provided tensor.
       *
       * @param x The tensor of required shape.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      reshapeAs(x) {
          this.throwIfDisposed();
          return this.reshape(x.shape);
      }
      /**
       * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
       * into the tensor's shape. See `tf.expandDims` for details.
       *
       * @param axis The dimension index at which to insert shape of 1. Defaults to
       *     0 (the first dimension).
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      expandDims(axis = 0) {
          return opHandler.expandDims(this, axis);
      }
      /**
       * Returns the cumulative sum of the `tf.Tensor` along `axis`.
       *
       * @param axis The axis along which to sum. Optional. Defaults to 0.
       * @param exclusive Whether to perform exclusive cumulative sum. Defaults to
       *    false. If set to true then the sum of each tensor entry does not
       * include its own value, but only the values previous to it along the
       * specified axis.
       * @param reverse Whether to sum in the opposite direction. Defaults to
       *    false.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      cumsum(axis = 0, exclusive = false, reverse = false) {
          return opHandler.cumsum(this, axis, exclusive, reverse);
      }
      /**
       * Returns a `tf.Tensor` with dimensions of size 1 removed from the shape.
       * See `tf.squeeze` for more details.
       *
       * @param axis A list of numbers. If specified, only squeezes the
       *    dimensions listed. The dimension index starts at 0. It is an error to
       *    squeeze a dimension that is not 1.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      squeeze(axis) {
          this.throwIfDisposed();
          return opHandler.squeeze(this, axis);
      }
      /** Returns a copy of the tensor. See `tf.clone` for details. */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      clone() {
          this.throwIfDisposed();
          return opHandler.clone(this);
      }
      /**
       * Returns a human-readable description of the tensor. Useful for logging.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      toString(verbose = false) {
          const vals = this.dataSync();
          return tensorToString(vals, this.shape, this.dtype, verbose);
      }
      // Below is chain API that is not exposed to docs to avoid repetition. To
      // expose a method, move it above this comment and add @doc and jsdoc.
      gather(indices, axis = 0) {
          this.throwIfDisposed();
          return opHandler.gather(this, indices, axis);
      }
      matMul(b, transposeA = false, transposeB = false) {
          this.throwIfDisposed();
          return opHandler.matMul(this, b, transposeA, transposeB);
      }
      dot(b) {
          this.throwIfDisposed();
          return opHandler.dot(this, b);
      }
      norm(ord = 'euclidean', axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.norm(this, ord, axis, keepDims);
      }
      slice(begin, size) {
          this.throwIfDisposed();
          return opHandler.slice(this, begin, size);
      }
      reverse(axis) {
          this.throwIfDisposed();
          return opHandler.reverse(this, axis);
      }
      concat(x, axis = 0) {
          this.throwIfDisposed();
          if (x instanceof Tensor) {
              x = [x];
          }
          return opHandler.concat([this, ...x], axis);
      }
      split(numOrSizeSplits, axis = 0) {
          this.throwIfDisposed();
          return opHandler.split(this, numOrSizeSplits, axis);
      }
      stack(x, axis = 0) {
          return opHandler.stack([this, x], axis);
      }
      unstack(axis = 0) {
          return opHandler.unstack(this, axis);
      }
      /**
       * @deprecated Use `tf.batchNorm` instead, and note the positional argument
       *     change of scale, offset, and varianceEpsilon.
       */
      batchNormalization(mean, variance, varianceEpsilon = .001, scale, offset) {
          deprecationWarningFn('tf.batchNormalization() is going away. ' +
              'Use tf.batchNorm() instead, and note the positional argument change ' +
              'of scale, offset, and varianceEpsilon');
          return this.batchNorm(mean, variance, offset, scale, varianceEpsilon);
      }
      // Reduction ops.
      all(axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.all(this, axis, keepDims);
      }
      any(axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.any(this, axis, keepDims);
      }
      logSumExp(axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.logSumExp(this, axis, keepDims);
      }
      sum(axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.sum(this, axis, keepDims);
      }
      prod(axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.prod(this, axis, keepDims);
      }
      mean(axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.mean(this, axis, keepDims);
      }
      min(axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.min(this, axis, keepDims);
      }
      max(axis = null, keepDims = false) {
          this.throwIfDisposed();
          return opHandler.max(this, axis, keepDims);
      }
      argMin(axis = null) {
          this.throwIfDisposed();
          return opHandler.argMin(this, axis);
      }
      argMax(axis = null) {
          this.throwIfDisposed();
          return opHandler.argMax(this, axis);
      }
      // Transformations
      cast(dtype) {
          this.throwIfDisposed();
          return opHandler.cast(this, dtype);
      }
      // Binary ops.
      addStrict(x) {
          this.throwIfDisposed();
          return opHandler.addStrict(this, x);
      }
      atan2(x) {
          this.throwIfDisposed();
          return opHandler.atan2(this, x);
      }
      subStrict(x) {
          this.throwIfDisposed();
          return opHandler.subStrict(this, x);
      }
      pow(exp) {
          this.throwIfDisposed();
          return opHandler.pow(this, exp);
      }
      powStrict(exp) {
          this.throwIfDisposed();
          return opHandler.powStrict(this, exp);
      }
      mul(x) {
          this.throwIfDisposed();
          return opHandler.mul(this, x);
      }
      mulStrict(x) {
          this.throwIfDisposed();
          return opHandler.mulStrict(this, x);
      }
      floorDiv(x) {
          this.throwIfDisposed();
          return opHandler.floorDiv(this, x);
      }
      divStrict(x) {
          this.throwIfDisposed();
          return opHandler.divStrict(this, x);
      }
      minimum(x) {
          this.throwIfDisposed();
          return opHandler.minimum(this, x);
      }
      minimumStrict(x) {
          this.throwIfDisposed();
          return opHandler.minimumStrict(this, x);
      }
      maximum(x) {
          this.throwIfDisposed();
          return opHandler.maximum(this, x);
      }
      maximumStrict(x) {
          this.throwIfDisposed();
          return opHandler.maximumStrict(this, x);
      }
      mod(x) {
          this.throwIfDisposed();
          return opHandler.mod(this, x);
      }
      modStrict(x) {
          this.throwIfDisposed();
          return opHandler.modStrict(this, x);
      }
      squaredDifferenceStrict(x) {
          this.throwIfDisposed();
          return opHandler.squaredDifferenceStrict(this, x);
      }
      // Compare ops.
      notEqualStrict(x) {
          this.throwIfDisposed();
          return opHandler.notEqualStrict(this, x);
      }
      less(x) {
          this.throwIfDisposed();
          return opHandler.less(this, x);
      }
      lessStrict(x) {
          this.throwIfDisposed();
          return opHandler.lessStrict(this, x);
      }
      equal(x) {
          this.throwIfDisposed();
          return opHandler.equal(this, x);
      }
      equalStrict(x) {
          this.throwIfDisposed();
          return opHandler.equalStrict(this, x);
      }
      lessEqual(x) {
          this.throwIfDisposed();
          return opHandler.lessEqual(this, x);
      }
      lessEqualStrict(x) {
          this.throwIfDisposed();
          return opHandler.lessEqualStrict(this, x);
      }
      greater(x) {
          this.throwIfDisposed();
          return opHandler.greater(this, x);
      }
      greaterStrict(x) {
          this.throwIfDisposed();
          return opHandler.greaterStrict(this, x);
      }
      greaterEqual(x) {
          this.throwIfDisposed();
          return opHandler.greaterEqual(this, x);
      }
      greaterEqualStrict(x) {
          this.throwIfDisposed();
          return opHandler.greaterEqualStrict(this, x);
      }
      // Compare ops.
      logicalAnd(x) {
          this.throwIfDisposed();
          return opHandler.logicalAnd(this, x);
      }
      logicalOr(x) {
          this.throwIfDisposed();
          return opHandler.logicalOr(this, x);
      }
      logicalNot() {
          this.throwIfDisposed();
          return opHandler.logicalNot(this);
      }
      logicalXor(x) {
          this.throwIfDisposed();
          return opHandler.logicalXor(this, x);
      }
      where(condition, x) {
          this.throwIfDisposed();
          return opHandler.where(condition, this, x);
      }
      // Unary ops.
      neg() {
          this.throwIfDisposed();
          return opHandler.neg(this);
      }
      ceil() {
          this.throwIfDisposed();
          return opHandler.ceil(this);
      }
      floor() {
          this.throwIfDisposed();
          return opHandler.floor(this);
      }
      sign() {
          this.throwIfDisposed();
          return opHandler.sign(this);
      }
      isNaN() {
          this.throwIfDisposed();
          return opHandler.isNaN(this);
      }
      isInf() {
          this.throwIfDisposed();
          return opHandler.isInf(this);
      }
      isFinite() {
          this.throwIfDisposed();
          return opHandler.isFinite(this);
      }
      exp() {
          this.throwIfDisposed();
          return opHandler.exp(this);
      }
      expm1() {
          this.throwIfDisposed();
          return opHandler.expm1(this);
      }
      log() {
          this.throwIfDisposed();
          return opHandler.log(this);
      }
      log1p() {
          this.throwIfDisposed();
          return opHandler.log1p(this);
      }
      sqrt() {
          this.throwIfDisposed();
          return opHandler.sqrt(this);
      }
      rsqrt() {
          this.throwIfDisposed();
          return opHandler.rsqrt(this);
      }
      square() {
          this.throwIfDisposed();
          return opHandler.square(this);
      }
      reciprocal() {
          this.throwIfDisposed();
          return opHandler.reciprocal(this);
      }
      abs() {
          this.throwIfDisposed();
          return opHandler.abs(this);
      }
      clipByValue(min, max) {
          this.throwIfDisposed();
          return opHandler.clipByValue(this, min, max);
      }
      relu() {
          this.throwIfDisposed();
          return opHandler.relu(this);
      }
      relu6() {
          this.throwIfDisposed();
          return opHandler.relu6(this);
      }
      elu() {
          this.throwIfDisposed();
          return opHandler.elu(this);
      }
      selu() {
          this.throwIfDisposed();
          return opHandler.selu(this);
      }
      leakyRelu(alpha = 0.2) {
          this.throwIfDisposed();
          return opHandler.leakyRelu(this, alpha);
      }
      prelu(alpha) {
          this.throwIfDisposed();
          return opHandler.prelu(this, alpha);
      }
      sigmoid() {
          this.throwIfDisposed();
          return opHandler.sigmoid(this);
      }
      logSigmoid() {
          this.throwIfDisposed();
          return opHandler.logSigmoid(this);
      }
      softplus() {
          this.throwIfDisposed();
          return opHandler.softplus(this);
      }
      zerosLike() {
          this.throwIfDisposed();
          return opHandler.zerosLike(this);
      }
      onesLike() {
          this.throwIfDisposed();
          return opHandler.onesLike(this);
      }
      sin() {
          this.throwIfDisposed();
          return opHandler.sin(this);
      }
      cos() {
          this.throwIfDisposed();
          return opHandler.cos(this);
      }
      tan() {
          this.throwIfDisposed();
          return opHandler.tan(this);
      }
      asin() {
          this.throwIfDisposed();
          return opHandler.asin(this);
      }
      acos() {
          this.throwIfDisposed();
          return opHandler.acos(this);
      }
      atan() {
          this.throwIfDisposed();
          return opHandler.atan(this);
      }
      sinh() {
          this.throwIfDisposed();
          return opHandler.sinh(this);
      }
      cosh() {
          this.throwIfDisposed();
          return opHandler.cosh(this);
      }
      tanh() {
          this.throwIfDisposed();
          return opHandler.tanh(this);
      }
      asinh() {
          this.throwIfDisposed();
          return opHandler.asinh(this);
      }
      acosh() {
          this.throwIfDisposed();
          return opHandler.acosh(this);
      }
      atanh() {
          this.throwIfDisposed();
          return opHandler.atanh(this);
      }
      erf() {
          this.throwIfDisposed();
          return opHandler.erf(this);
      }
      round() {
          this.throwIfDisposed();
          return opHandler.round(this);
      }
      step(alpha = 0.0) {
          this.throwIfDisposed();
          return opHandler.step(this, alpha);
      }
      softmax(dim = -1) {
          this.throwIfDisposed();
          return opHandler.softmax(this, dim);
      }
      logSoftmax(axis = -1) {
          this.throwIfDisposed();
          return opHandler.logSoftmax(this, axis);
      }
      // Image ops.
      resizeBilinear(newShape2D, alignCorners = false) {
          this.throwIfDisposed();
          return opHandler.image.resizeBilinear(this, newShape2D, alignCorners);
      }
      resizeNearestNeighbor(newShape2D, alignCorners = false) {
          this.throwIfDisposed();
          return opHandler.image.resizeNearestNeighbor(this, newShape2D, alignCorners);
      }
      // Convolutions.
      conv1d(filter, stride, pad, dataFormat = 'NWC', dilation = 1, dimRoundingMode) {
          this.throwIfDisposed();
          return opHandler.conv1d(this, filter, stride, pad, dataFormat, dilation, dimRoundingMode);
      }
      conv2d(filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode) {
          this.throwIfDisposed();
          return opHandler.conv2d(this, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
      }
      conv2dTranspose(filter, outputShape, strides, pad, dimRoundingMode) {
          this.throwIfDisposed();
          return opHandler.conv2dTranspose(this, filter, outputShape, strides, pad, dimRoundingMode);
      }
      depthwiseConv2D(filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode) {
          this.throwIfDisposed();
          return opHandler.depthwiseConv2d(this, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
      }
      separableConv2d(depthwiseFilter, pointwiseFilter, strides, pad, dilation = [1, 1], dataFormat = 'NHWC') {
          this.throwIfDisposed();
          return opHandler.separableConv2d(this, depthwiseFilter, pointwiseFilter, strides, pad, dilation, dataFormat);
      }
      // Pooling.
      avgPool(filterSize, strides, pad, dimRoundingMode) {
          this.throwIfDisposed();
          return opHandler.avgPool(this, filterSize, strides, pad, dimRoundingMode);
      }
      maxPool(filterSize, strides, pad, dimRoundingMode) {
          this.throwIfDisposed();
          return opHandler.maxPool(this, filterSize, strides, pad, dimRoundingMode);
      }
      localResponseNormalization(radius = 5, bias = 1, alpha = 1, beta = 0.5) {
          return opHandler.localResponseNormalization(this, radius, bias, alpha, beta);
      }
      pool(windowShape, poolingType, padding, dilationRate, strides) {
          this.throwIfDisposed();
          return opHandler.pool(this, windowShape, poolingType, padding, dilationRate, strides);
      }
      variable(trainable = true, name, dtype) {
          this.throwIfDisposed();
          return trackerFn().makeVariable(this, trainable, name, dtype);
      }
      unsortedSegmentSum(segmentIds, numSegments) {
          this.throwIfDisposed();
          return opHandler.unsortedSegmentSum(this, segmentIds, numSegments);
      }
      batchToSpaceND(blockShape, crops) {
          this.throwIfDisposed();
          return opHandler.batchToSpaceND(this, blockShape, crops);
      }
      spaceToBatchND(blockShape, paddings) {
          this.throwIfDisposed();
          return opHandler.spaceToBatchND(this, blockShape, paddings);
      }
      topk(k = 1, sorted = true) {
          this.throwIfDisposed();
          return opHandler.topk(this, k, sorted);
      }
      stridedSlice(begin, end, strides, beginMask = 0, endMask = 0, ellipsisMask = 0, newAxisMask = 0, shrinkAxisMask = 0) {
          this.throwIfDisposed();
          return opHandler.stridedSlice(this, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
      }
      depthToSpace(blockSize, dataFormat) {
          this.throwIfDisposed();
          return opHandler.depthToSpace(this, blockSize, dataFormat);
      }
      fft() {
          this.throwIfDisposed();
          return opHandler.spectral.fft(this);
      }
      ifft() {
          this.throwIfDisposed();
          return opHandler.spectral.ifft(this);
      }
      rfft() {
          this.throwIfDisposed();
          return opHandler.spectral.rfft(this);
      }
      irfft() {
          this.throwIfDisposed();
          return opHandler.spectral.irfft(this);
      }
  }
  Object.defineProperty(Tensor, Symbol.hasInstance, {
      value: (instance) => {
          return !!instance && instance.dataId != null && instance.shape != null &&
              instance.dtype != null;
      }
  });
  /**
   * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  class Variable extends Tensor {
      constructor(initialValue, trainable, name, tensorId) {
          super(initialValue.shape, initialValue.dtype, initialValue.dataId, tensorId);
          this.trainable = trainable;
          this.name = name;
      }
      /**
       * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
       * the same shape and dtype as the old `tf.Tensor`.
       *
       * @param newValue New tensor to be assigned to this variable.
       */
      /** @doc {heading: 'Tensors', subheading: 'Classes'} */
      assign(newValue) {
          if (newValue.dtype !== this.dtype) {
              throw new Error(`dtype of the new value (${newValue.dtype}) and ` +
                  `previous value (${this.dtype}) must match`);
          }
          if (!arraysEqual(newValue.shape, this.shape)) {
              throw new Error(`shape of the new value (${newValue.shape}) and ` +
                  `previous value (${this.shape}) must match`);
          }
          trackerFn().disposeTensor(this);
          this.dataId = newValue.dataId;
          trackerFn().incRef(this, null /* backend */);
      }
      dispose() {
          trackerFn().disposeVariable(this);
          this.isDisposedInternal = true;
      }
  }
  Object.defineProperty(Variable, Symbol.hasInstance, {
      value: (instance) => {
          return instance instanceof Tensor && instance.assign != null &&
              instance.assign instanceof Function;
      }
  });

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
  (function (Rank) {
      Rank["R0"] = "R0";
      Rank["R1"] = "R1";
      Rank["R2"] = "R2";
      Rank["R3"] = "R3";
      Rank["R4"] = "R4";
      Rank["R5"] = "R5";
      Rank["R6"] = "R6";
  })(exports.Rank || (exports.Rank = {}));
  // Looks for upcasting types. Used, for example, in operations with mixed dtype
  // inputs.
  var UpcastInt32AndMap;
  (function (UpcastInt32AndMap) {
      UpcastInt32AndMap["float32"] = "float32";
      UpcastInt32AndMap["int32"] = "int32";
      UpcastInt32AndMap["bool"] = "int32";
      UpcastInt32AndMap["complex64"] = "complex64";
  })(UpcastInt32AndMap || (UpcastInt32AndMap = {}));
  var UpcastBoolAndMap;
  (function (UpcastBoolAndMap) {
      UpcastBoolAndMap["float32"] = "float32";
      UpcastBoolAndMap["int32"] = "int32";
      UpcastBoolAndMap["bool"] = "bool";
      UpcastBoolAndMap["complex64"] = "complex64";
  })(UpcastBoolAndMap || (UpcastBoolAndMap = {}));
  var UpcastFloat32AndMap;
  (function (UpcastFloat32AndMap) {
      UpcastFloat32AndMap["float32"] = "float32";
      UpcastFloat32AndMap["int32"] = "float32";
      UpcastFloat32AndMap["bool"] = "float32";
      UpcastFloat32AndMap["complex64"] = "complex64";
  })(UpcastFloat32AndMap || (UpcastFloat32AndMap = {}));
  var UpcastComplex64AndMap;
  (function (UpcastComplex64AndMap) {
      UpcastComplex64AndMap["float32"] = "complex64";
      UpcastComplex64AndMap["int32"] = "complex64";
      UpcastComplex64AndMap["bool"] = "complex64";
      UpcastComplex64AndMap["complex64"] = "complex64";
  })(UpcastComplex64AndMap || (UpcastComplex64AndMap = {}));
  const upcastTypeMap = {
      'float32': UpcastFloat32AndMap,
      'int32': UpcastInt32AndMap,
      'bool': UpcastBoolAndMap,
      'complex64': UpcastComplex64AndMap
  };
  function upcastType(typeA, typeB) {
      if (typeA === 'string' || typeB === 'string') {
          if (typeA === 'string' && typeB === 'string') {
              return 'string';
          }
          throw new Error(`Can not upcast ${typeA} with ${typeB}`);
      }
      return upcastTypeMap[typeA][typeB];
  }
  /** Returns the output type after summation. */
  function sumOutType(type) {
      return upcastType(type, 'int32');
  }

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
  function makeTypesMatch(a, b) {
      if (a.dtype === b.dtype) {
          return [a, b];
      }
      const dtype = upcastType(a.dtype, b.dtype);
      return [a.cast(dtype), b.cast(dtype)];
  }
  function assertTypesMatch(a, b) {
      assert(a.dtype === b.dtype, () => `The dtypes of the first(${a.dtype}) and` +
          ` second(${b.dtype}) input must match`);
  }
  function isTensorInList(tensor, tensorList) {
      return tensorList.some(x => x.id === tensor.id);
  }
  /**
   * Extracts any `Tensor`s found within the provided object.
   *
   * @param container an object that may be a `Tensor` or may directly contain
   *   `Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. In general it
   *   is safe to pass any object here, except that `Promise`s are not
   *   supported.
   * @returns An array of `Tensors` found within the passed object. If the
   *   argument is simply a `Tensor', a list containing that `Tensor` is
   *   returned. If the object is not a `Tensor` or does not
   *   contain `Tensors`, an empty list is returned.
   */
  function getTensorsInContainer(result) {
      const list = [];
      const seen = new Set();
      walkTensorContainer(result, list, seen);
      return list;
  }
  function walkTensorContainer(container, list, seen) {
      if (container == null) {
          return;
      }
      if (container instanceof Tensor) {
          list.push(container);
          return;
      }
      if (!isIterable(container)) {
          return;
      }
      // Iteration over keys works also for arrays.
      const iterable = container;
      for (const k in iterable) {
          const val = iterable[k];
          if (!seen.has(val)) {
              seen.add(val);
              walkTensorContainer(val, list, seen);
          }
      }
  }
  // tslint:disable-next-line:no-any
  function isIterable(obj) {
      return Array.isArray(obj) || typeof obj === 'object';
  }

  var tensor_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    makeTypesMatch: makeTypesMatch,
    assertTypesMatch: assertTypesMatch,
    isTensorInList: isTensorInList,
    getTensorsInContainer: getTensorsInContainer
  });

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
  class EngineState {
      constructor() {
          // Public since optimizers will use it.
          this.registeredVariables = {};
          this.nextTapeNodeId = 0;
          this.numBytes = 0;
          this.numTensors = 0;
          this.numStringTensors = 0;
          this.numDataBuffers = 0;
          // Number of nested tf.grad() statements when computing higher-order
          // gradients. E.g. `1` for first-order gradients and `2` for second-order
          // gradients. Used to track if the tape should be removed after a backprop.
          this.gradientDepth = 0;
          // Number of nested kernel calls. When kernel depth is greater than 1, we turn
          // off the tape.
          this.kernelDepth = 0;
          this.scopeStack = [];
          /**
           * Keeps track of the number of data moves during a kernel execution. We
           * maintain a stack since kernels can call other kernels, recursively.
           */
          this.numDataMovesStack = [];
          this.nextScopeId = 0;
          this.tensorInfo = new WeakMap();
          this.profiling = false;
          this.activeProfile = { newBytes: 0, newTensors: 0, peakBytes: 0, kernels: [], result: null };
      }
      dispose() {
          for (const variableName in this.registeredVariables) {
              this.registeredVariables[variableName].dispose();
          }
      }
  }
  class Engine {
      constructor(ENV) {
          this.ENV = ENV;
          this.registry = {};
          this.registryFactory = {};
          this.pendingBackendInitId = 0;
          this.state = new EngineState();
      }
      async ready() {
          if (this.pendingBackendInit != null) {
              return this.pendingBackendInit.then(() => { });
          }
          if (this.backendInstance != null) {
              return;
          }
          const sortedBackends = this.getSortedBackends();
          for (let i = 0; i < sortedBackends.length; i++) {
              const backendName = sortedBackends[i];
              const success = await this.initializeBackend(backendName).success;
              if (success) {
                  await this.setBackend(backendName);
                  return;
              }
          }
          throw new Error(`Could not initialize any backends, all backend initializations ` +
              `failed.`);
      }
      get backend() {
          if (this.pendingBackendInit != null) {
              throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make ` +
                  `sure to await tf.ready() or await tf.setBackend() before calling ` +
                  `other methods`);
          }
          if (this.backendInstance == null) {
              const { name, asyncInit } = this.initializeBackendsAndReturnBest();
              if (asyncInit) {
                  throw new Error(`The highest priority backend '${name}' has not yet been ` +
                      `initialized. Make sure to await tf.ready() or ` +
                      `await tf.setBackend() before calling other methods`);
              }
              this.setBackend(name);
          }
          return this.backendInstance;
      }
      backendNames() {
          return Object.keys(this.registryFactory);
      }
      findBackend(backendName) {
          if (!(backendName in this.registry)) {
              // If the backend hasn't been initialized but we have a registry entry for
              // it, initialize it and return it.
              if (backendName in this.registryFactory) {
                  const { asyncInit } = this.initializeBackend(backendName);
                  if (asyncInit) {
                      // Backend is not ready yet.
                      return null;
                  }
              }
              else {
                  return null;
              }
          }
          return this.registry[backendName];
      }
      findBackendFactory(backendName) {
          if (!(backendName in this.registryFactory)) {
              return null;
          }
          return this.registryFactory[backendName].factory;
      }
      registerBackend(backendName, factory, priority = 1) {
          if (backendName in this.registryFactory) {
              console.warn(`${backendName} backend was already registered. ` +
                  `Reusing existing backend factory.`);
              return false;
          }
          this.registryFactory[backendName] = { factory, priority };
          return true;
      }
      async setBackend(backendName) {
          if (this.registryFactory[backendName] == null) {
              throw new Error(`Backend name '${backendName}' not found in registry`);
          }
          this.backendName = backendName;
          if (this.registry[backendName] == null) {
              this.backendInstance = null;
              const { success, asyncInit } = this.initializeBackend(backendName);
              const result = asyncInit ? await success : success;
              if (!result) {
                  return false;
              }
          }
          this.backendInstance = this.registry[backendName];
          this.setupRegisteredKernels();
          // Reset the profiler.
          this.profiler = new Profiler(this.backendInstance);
          return true;
      }
      setupRegisteredKernels() {
          const kernels = getKernelsForBackend(this.backendName);
          kernels.forEach(kernel => {
              if (kernel.setupFunc != null) {
                  kernel.setupFunc(this.backendInstance);
              }
          });
      }
      disposeRegisteredKernels(backendName) {
          const kernels = getKernelsForBackend(backendName);
          kernels.forEach(kernel => {
              if (kernel.disposeFunc != null) {
                  kernel.disposeFunc(this.registry[backendName]);
              }
          });
      }
      /**
       * Initializes a backend by looking up the backend name in the factory
       * registry and calling the factory method. Returns a boolean representing
       * whether the initialization of the backend suceeded. Throws an error if
       * there is no backend in the factory registry.
       */
      initializeBackend(backendName) {
          const registryFactoryEntry = this.registryFactory[backendName];
          if (registryFactoryEntry == null) {
              throw new Error(`Cannot initialize backend ${backendName}, no registration found.`);
          }
          try {
              const backend = registryFactoryEntry.factory();
              // Test if the factory returns a promise.
              if (Promise.resolve(backend) === backend) {
                  const promiseId = ++this.pendingBackendInitId;
                  const success = backend
                      .then(backendInstance => {
                      // Outdated promise. Another backend was set in the meantime.
                      if (promiseId < this.pendingBackendInitId) {
                          return false;
                      }
                      this.registry[backendName] = backendInstance;
                      this.pendingBackendInit = null;
                      return true;
                  })
                      .catch(err => {
                      // Outdated promise. Another backend was set in the meantime.
                      if (promiseId < this.pendingBackendInitId) {
                          return false;
                      }
                      this.pendingBackendInit = null;
                      console.warn(`Initialization of backend ${backendName} failed`);
                      console.warn(err.stack || err.message);
                      return false;
                  });
                  this.pendingBackendInit = success;
                  return { success, asyncInit: true };
              }
              else {
                  this.registry[backendName] = backend;
                  return { success: true, asyncInit: false };
              }
          }
          catch (err) {
              console.warn(`Initialization of backend ${backendName} failed`);
              console.warn(err.stack || err.message);
              return { success: false, asyncInit: false };
          }
      }
      removeBackend(backendName) {
          if (!(backendName in this.registryFactory)) {
              throw new Error(`${backendName} backend not found in registry`);
          }
          if (this.backendName === backendName && this.pendingBackendInit != null) {
              // There is a pending promise of the backend we want to remove. Make it
              // obsolete.
              this.pendingBackendInitId++;
          }
          if (backendName in this.registry) {
              this.disposeRegisteredKernels(backendName);
              this.registry[backendName].dispose();
              delete this.registry[backendName];
          }
          delete this.registryFactory[backendName];
          // Unset the backend if it is active.
          if (this.backendName === backendName) {
              this.pendingBackendInit = null;
              this.backendName = null;
              this.backendInstance = null;
          }
      }
      getSortedBackends() {
          if (Object.keys(this.registryFactory).length === 0) {
              throw new Error('No backend found in registry.');
          }
          return Object.keys(this.registryFactory).sort((a, b) => {
              // Highest priority comes first.
              return this.registryFactory[b].priority -
                  this.registryFactory[a].priority;
          });
      }
      initializeBackendsAndReturnBest() {
          const sortedBackends = this.getSortedBackends();
          for (let i = 0; i < sortedBackends.length; i++) {
              const backendName = sortedBackends[i];
              const { success, asyncInit } = this.initializeBackend(backendName);
              if (asyncInit || success) {
                  return { name: backendName, asyncInit };
              }
          }
          throw new Error(`Could not initialize any backends, all backend initializations ` +
              `failed.`);
      }
      moveData(backend, dataId) {
          const info = this.state.tensorInfo.get(dataId);
          const srcBackend = info.backend;
          const values = this.readSync(dataId);
          // Delete the tensor from the old backend and move it to the new
          // backend.
          srcBackend.disposeData(dataId);
          info.backend = backend;
          backend.move(dataId, values, info.shape, info.dtype);
          if (this.shouldCheckForMemLeaks()) {
              // Track the number of moves during a kernel execution to correctly
              // detect memory leaks.
              this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1]++;
          }
      }
      tidy(nameOrFn, fn) {
          let name = null;
          if (fn == null) {
              // Called with only 1 argument.
              if (typeof nameOrFn !== 'function') {
                  throw new Error('Please provide a function to tidy()');
              }
              fn = nameOrFn;
          }
          else {
              // Called with 2 arguments.
              if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
                  throw new Error('When calling with two arguments, the first argument ' +
                      'to tidy() must be a string');
              }
              if (typeof fn !== 'function') {
                  throw new Error('When calling with two arguments, the 2nd argument ' +
                      'to tidy() must be a function');
              }
              name = nameOrFn;
              // TODO(nsthorat,smilkov): Do operation logging and performance
              // profiling.
          }
          let result;
          return this.scopedRun(() => this.startScope(name), () => this.endScope(result), () => {
              result = fn();
              if (result instanceof Promise) {
                  console.error('Cannot return a Promise inside of tidy.');
              }
              return result;
          });
      }
      scopedRun(start, end, f) {
          start();
          try {
              const res = f();
              end();
              return res;
          }
          catch (ex) {
              end();
              throw ex;
          }
      }
      nextTensorId() {
          return Engine.nextTensorId++;
      }
      nextVariableId() {
          return Engine.nextVariableId++;
      }
      /**
       * This method is called instead of the public-facing tensor.clone() when
       * saving a tensor for backwards pass. It makes sure to add the clone
       * operation to the tape regardless of being called inside a kernel
       * execution.
       *
       * This method will go away once all kernels are modularized since we won't
       * need to turn off the tape inside runKernel().
       */
      clone(x) {
          const y = this.makeTensorFromDataId(x.dataId, x.shape, x.dtype);
          const inputs = { x };
          const grad = (dy) => ({ x: () => dy.toFloat() });
          const saved = [];
          this.addTapeNode(this.state.activeScope.name, inputs, [y], grad, saved, {});
          return y;
      }
      /**
       * Execute a kernel with the given name and return the output tensor.
       *
       * @param kernelName The name of the kernel to execute.
       * @param inputs A map of input names to tensors.
       * @param attrs A map of attribute names to their values. An attribute is a
       *     primitive (non-tensor) input to the kernel.
       * @param inputsToSave A list of tensors, inputs to save for the backprop
       *     computation.
       * @param outputsToSave A list of booleans, specifying which output to save
       *     for the backprop computation. These are booleans since the output
       * tensors are not visible to the user.
       */
      runKernel(kernelName, inputs, attrs, inputsToSave, outputsToSave) {
          const forwardFunc = null;
          const backwardsFunc = null;
          // Call runKernel as a stop-gap until we modularize all kernels.
          // Once we modularize all kernels, we will remove the existing
          // `runKernelFunc`.
          return this.runKernelFunc(forwardFunc, inputs, backwardsFunc, kernelName, attrs, inputsToSave, outputsToSave);
      }
      shouldCheckForMemLeaks() {
          return this.ENV.getBool('IS_TEST');
      }
      checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos) {
          const numDataIdsAfter = this.backend.numDataIds();
          // Count the number of data ids associated with the result of the kernel.
          let numOutputDataIds = 0;
          outInfos.forEach(info => {
              // Complex numbers allocate 3 data ids, one for 'real', one for
              // 'imaginary', and one for the container that holds the former two.
              numOutputDataIds += (info.dtype === 'complex64' ? 3 : 1);
          });
          // Account for the number of moves during kernel execution. A "data move"
          // can happen in the middle of a kernel execution, placing a new (key,value)
          // pair in the data storage. Since data moves have net zero effect (we
          // always remove the data from the old backend), we have to cancel them out
          // when detecting memory leaks.
          const numMoves = this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1];
          const dataIdsLeaked = numDataIdsAfter - numDataIdsBefore - numOutputDataIds - numMoves;
          if (dataIdsLeaked > 0) {
              throw new Error(`Backend '${this.backendName}' has an internal memory leak ` +
                  `(${dataIdsLeaked} data ids) after running '${kernelName}'`);
          }
      }
      /**
       * @deprecated Use `runKernel` for newly added kernels. Keep using this method
       *     only for kernels that are not yet fully modularized.
       */
      runKernelFunc(forwardFunc, inputs, backwardsFunc, kernelName, attrs, inputsToSave, outputsToSave) {
          let outputs;
          let saved = [];
          const isTapeOn = this.isTapeOn();
          if (kernelName == null) {
              kernelName =
                  this.state.activeScope != null ? this.state.activeScope.name : '';
          }
          const startingBytecount = this.state.numBytes;
          const startingNumTensors = this.state.numTensors;
          if (this.shouldCheckForMemLeaks()) {
              this.state.numDataMovesStack.push(0);
          }
          let kernelFunc;
          const kernel = getKernel(kernelName, this.backendName);
          let out;
          if (kernel != null) {
              kernelFunc = () => {
                  const numDataIdsBefore = this.backend.numDataIds();
                  out = kernel.kernelFunc({ inputs, attrs, backend: this.backend });
                  const outInfos = Array.isArray(out) ? out : [out];
                  if (this.shouldCheckForMemLeaks()) {
                      this.checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos);
                  }
                  const outTensors = outInfos.map(({ dataId, shape, dtype }) => this.makeTensorFromDataId(dataId, shape, dtype));
                  // Save the inputs and outputs.
                  // Do not save unless we are recording to the tape. Otherwise it would
                  // cause a mem leak since we would never run backprop, which disposes
                  // the kept tensors.
                  if (isTapeOn) {
                      let tensorsToSave = this.getTensorsForGradient(kernelName, inputs, outTensors);
                      if (tensorsToSave == null) {
                          // Fallback for ops that call runKernelFunc and pass in
                          // inputsToSave and outputsToSave. Currently this is the set of ops
                          // with kernel support in the WASM backend. Once those ops and
                          // respective gradients are modularised we can remove this path.
                          if (outputsToSave == null) {
                              outputsToSave = [];
                          }
                          const outsToSave = outTensors.filter((_, i) => outputsToSave[i]);
                          tensorsToSave = (inputsToSave || []).slice().concat(outsToSave);
                      }
                      saved = this.saveTensorsForBackwardMode(tensorsToSave);
                  }
                  return outTensors;
              };
          }
          else {
              const saveFunc = (tensors) => {
                  // Do not save unless we are recording to the tape. Otherwise it would
                  // cause a mem leak since we would never run backprop, which disposes
                  // the kept tensors.
                  if (!isTapeOn) {
                      return;
                  }
                  saved = tensors.map(tensor => this.keep(this.clone(tensor)));
              };
              kernelFunc = () => {
                  const numDataIdsBefore = this.backend.numDataIds();
                  out = this.tidy(() => forwardFunc(this.backend, saveFunc));
                  const outs = (Array.isArray(out) ? out : [out]);
                  if (this.shouldCheckForMemLeaks()) {
                      this.checkKernelForMemLeak(kernelName, numDataIdsBefore, outs);
                  }
                  return outs;
              };
          }
          // Stop recording to a tape when running a kernel.
          this.scopedRun(() => this.state.kernelDepth++, () => this.state.kernelDepth--, () => {
              if (!this.ENV.getBool('DEBUG')) {
                  outputs = kernelFunc();
              }
              else {
                  outputs = this.profiler.profileKernel(kernelName, inputs, () => kernelFunc());
              }
          });
          if (isTapeOn) {
              this.addTapeNode(kernelName, inputs, outputs, backwardsFunc, saved, attrs);
          }
          if (this.state.profiling) {
              this.state.activeProfile.kernels.push({
                  name: kernelName,
                  bytesAdded: this.state.numBytes - startingBytecount,
                  totalBytesSnapshot: this.state.numBytes,
                  tensorsAdded: this.state.numTensors - startingNumTensors,
                  totalTensorsSnapshot: this.state.numTensors,
                  inputShapes: Object.keys(inputs).map(key => inputs[key].shape),
                  outputShapes: outputs.map(item => item.shape)
              });
          }
          return (Array.isArray(out) ? outputs : outputs[0]);
      }
      /**
       * Saves tensors used in forward mode for use in backward mode.
       *
       * @param tensors the list of tensors to save.
       */
      saveTensorsForBackwardMode(tensors) {
          const saved = tensors.map(tensor => this.keep(this.clone(tensor)));
          return saved;
      }
      /**
       * Returns a list of tensors to save for a given gradient calculation.
       *
       * Returns undefined if their is no registered gradient for this kernel in the
       * gradient registry.
       *
       * @param kernelName name of kernel to look up gradient for.
       * @param inputs a map of input tensors.
       * @param outputs an array of output tensors from forward mode of kernel.
       */
      getTensorsForGradient(kernelName, inputs, outputs) {
          const gradConfig = getGradient(kernelName);
          if (gradConfig != null) {
              const inputsToSave = gradConfig.inputsToSave || [];
              const outputsToSave = gradConfig.outputsToSave || [];
              // If saveAllInputs is true, all inputs will be saved. Otherwise, inputs
              // specified in inputsToSave will be saved.
              let inputTensorsToSave;
              if (gradConfig.saveAllInputs) {
                  assert(Array.isArray(inputs), () => 'saveAllInputs is true, expected inputs to be an array.');
                  inputTensorsToSave = Object.keys(inputs).map((key) => inputs[key]);
              }
              else {
                  inputTensorsToSave = inputsToSave.map((inputName) => inputs[inputName]);
              }
              const outputTensorsToSave = outputs.filter((_, i) => outputsToSave[i]);
              return inputTensorsToSave.concat(outputTensorsToSave);
          }
          // TODO(yassogba) throw exception here once all runkernelFunc calls with
          // inputsToSave/outputsToSave are removed
          return null;
      }
      /**
       * Internal method used by public APIs for tensor creation. Makes a new
       * tensor with the provided shape, dtype and values. It always
       * creates a new data id and writes the values to the underlying backend.
       */
      makeTensor(values, shape, dtype, backend) {
          if (values == null) {
              throw new Error('Values passed to engine.makeTensor() are null');
          }
          dtype = dtype || 'float32';
          backend = backend || this.backend;
          let backendVals = values;
          if (dtype === 'string' && isString(values[0])) {
              backendVals = values.map(d => encodeString(d));
          }
          const dataId = backend.write(backendVals, shape, dtype);
          const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
          this.incRef(t, backend);
          // Count bytes for string tensors.
          if (dtype === 'string') {
              const info = this.state.tensorInfo.get(dataId);
              const newBytes = bytesFromStringArray(backendVals);
              this.state.numBytes += newBytes - info.bytes;
              info.bytes = newBytes;
          }
          return t;
      }
      /**
       * Internal method used by backends. Makes a new tensor
       * that is a wrapper around an existing data id. It doesn't create
       * a new data id, only increments the ref count used in memory tracking.
       */
      makeTensorFromDataId(dataId, shape, dtype, backend) {
          dtype = dtype || 'float32';
          const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
          this.incRef(t, backend);
          return t;
      }
      makeVariable(initialValue, trainable = true, name, dtype) {
          name = name || this.nextVariableId().toString();
          if (dtype != null && dtype !== initialValue.dtype) {
              initialValue = initialValue.asType(dtype);
          }
          const v = new Variable(initialValue, trainable, name, this.nextTensorId());
          if (this.state.registeredVariables[v.name] != null) {
              throw new Error(`Variable with name ${v.name} was already registered`);
          }
          this.state.registeredVariables[v.name] = v;
          this.incRef(v, this.backend);
          return v;
      }
      incRef(a, backend) {
          const refCount = this.state.tensorInfo.has(a.dataId) ?
              this.state.tensorInfo.get(a.dataId).refCount :
              0;
          this.state.numTensors++;
          if (a.dtype === 'string') {
              this.state.numStringTensors++;
          }
          if (refCount === 0) {
              this.state.numDataBuffers++;
              // Bytes for complex numbers are counted by their components. Bytes for
              // string tensors are counted when writing values.
              let bytes = 0;
              if (a.dtype !== 'complex64' && a.dtype !== 'string') {
                  bytes = a.size * bytesPerElement(a.dtype);
              }
              this.state.tensorInfo.set(a.dataId, {
                  backend: backend || this.backend,
                  dtype: a.dtype,
                  shape: a.shape,
                  bytes,
                  refCount: 0
              });
              this.state.numBytes += bytes;
          }
          this.state.tensorInfo.get(a.dataId).refCount++;
          if (!(a instanceof Variable)) {
              this.track(a);
          }
      }
      disposeTensor(a) {
          if (!this.state.tensorInfo.has(a.dataId)) {
              return;
          }
          this.state.numTensors--;
          if (a.dtype === 'string') {
              this.state.numStringTensors--;
          }
          const info = this.state.tensorInfo.get(a.dataId);
          const refCount = info.refCount;
          if (refCount <= 1) {
              // Don't count bytes for complex numbers as they are counted by their
              // components.
              if (a.dtype !== 'complex64') {
                  this.state.numBytes -= info.bytes;
              }
              this.state.numDataBuffers--;
              info.backend.disposeData(a.dataId);
              this.state.tensorInfo.delete(a.dataId);
          }
          else {
              this.state.tensorInfo.get(a.dataId).refCount--;
          }
          // TODO(nsthorat): Construct an error and save the stack trace for
          // debugging when in debug mode. Creating a stack trace is too expensive
          // to do unconditionally.
      }
      disposeVariables() {
          for (const varName in this.state.registeredVariables) {
              const v = this.state.registeredVariables[varName];
              this.disposeVariable(v);
          }
      }
      disposeVariable(v) {
          this.disposeTensor(v);
          if (this.state.registeredVariables[v.name] != null) {
              delete this.state.registeredVariables[v.name];
          }
      }
      memory() {
          const info = this.backend.memory();
          info.numTensors = this.state.numTensors;
          info.numDataBuffers = this.state.numDataBuffers;
          info.numBytes = this.state.numBytes;
          if (this.state.numStringTensors > 0) {
              info.unreliable = true;
              if (info.reasons == null) {
                  info.reasons = [];
              }
              info.reasons.push('Memory usage by string tensors is approximate ' +
                  '(2 bytes per character)');
          }
          return info;
      }
      async profile(query) {
          this.state.profiling = true;
          const startBytes = this.state.numBytes;
          const startNumTensors = this.state.numTensors;
          this.state.activeProfile.kernels = [];
          this.state.activeProfile.result = query();
          this.state.profiling = false;
          this.state.activeProfile.peakBytes = Math.max(...this.state.activeProfile.kernels.map(d => d.totalBytesSnapshot));
          this.state.activeProfile.newBytes = this.state.numBytes - startBytes;
          this.state.activeProfile.newTensors =
              this.state.numTensors - startNumTensors;
          return this.state.activeProfile;
      }
      isTapeOn() {
          return this.state.gradientDepth > 0 && this.state.kernelDepth === 0;
      }
      addTapeNode(kernelName, inputs, outputs, gradientsFunc, saved, attrs) {
          const tapeNode = { id: this.state.nextTapeNodeId++, kernelName, inputs, outputs, saved };
          const gradConfig = getGradient(kernelName);
          if (gradConfig != null) {
              gradientsFunc = gradConfig.gradFunc;
          }
          if (gradientsFunc != null) {
              tapeNode.gradient = (dys) => {
                  // TODO(smilkov): To optimize back-prop, pass dys that are not used in
                  // the backprop graph to the user as null instead of zeros
                  dys = dys.map((dy, i) => {
                      if (dy == null) {
                          const output = outputs[i];
                          const vals = makeZerosTypedArray(output.size, output.dtype);
                          return this.makeTensor(vals, output.shape, output.dtype);
                      }
                      return dy;
                  });
                  // Grad functions of ops with single outputs expect a dy, while ops
                  // with multiple outputs expect dys (array of dy).
                  return gradientsFunc(dys.length > 1 ? dys : dys[0], saved, attrs);
              };
          }
          this.state.activeTape.push(tapeNode);
      }
      keep(result) {
          result.kept = true;
          return result;
      }
      startTape() {
          if (this.state.gradientDepth === 0) {
              this.state.activeTape = [];
          }
          this.state.gradientDepth++;
      }
      endTape() {
          this.state.gradientDepth--;
      }
      /**
       * Start a scope. Use this with endScope() to achieve the same functionality
       * as scope() without the need for a function closure.
       */
      startScope(name) {
          const scopeInfo = {
              track: [],
              name: 'unnamed scope',
              id: this.state.nextScopeId++
          };
          if (name) {
              scopeInfo.name = name;
          }
          this.state.scopeStack.push(scopeInfo);
          this.state.activeScope = scopeInfo;
      }
      /**
       * End a scope. Use this with startScope() to achieve the same functionality
       * as scope() without the need for a function closure.
       */
      endScope(result) {
          const tensorsToTrackInParent = getTensorsInContainer(result);
          const tensorsToTrackInParentSet = new Set(tensorsToTrackInParent.map(t => t.id));
          // Dispose the arrays tracked in this scope.
          for (let i = 0; i < this.state.activeScope.track.length; i++) {
              const tensor = this.state.activeScope.track[i];
              if (!tensor.kept && !tensorsToTrackInParentSet.has(tensor.id)) {
                  tensor.dispose();
              }
          }
          const oldScope = this.state.scopeStack.pop();
          this.state.activeScope = this.state.scopeStack.length === 0 ?
              null :
              this.state.scopeStack[this.state.scopeStack.length - 1];
          // Track the current result in the parent scope.
          tensorsToTrackInParent.forEach(tensor => {
              // Only track the tensor if was allocated in the inner scope and is not
              // globally kept.
              if (!tensor.kept && tensor.scopeId === oldScope.id) {
                  this.track(tensor);
              }
          });
      }
      /**
       * Returns gradients of `f` with respect to each of the `xs`. The gradients
       * returned are of the same length as `xs`, but some might be null if `f`
       * was not a function of that `x`. It also takes optional dy to multiply the
       * gradient, which defaults to `1`.
       */
      gradients(f, xs, dy, allowNoGradients = false) {
          assert(xs.length > 0, () => 'gradients() received an empty list of xs.');
          if (dy != null && dy.dtype !== 'float32') {
              throw new Error(`dy must have 'float32' dtype, but has '${dy.dtype}'`);
          }
          const y = this.scopedRun(() => this.startTape(), () => this.endTape(), () => this.tidy('forward', f));
          assert(y instanceof Tensor, () => 'The result y returned by f() must be a tensor.');
          // Filter out the nodes that don't connect x => y.
          const filteredTape = getFilteredNodesXToY(this.state.activeTape, xs, y);
          if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
              throw new Error('Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
                  'that the f you passed encloses all operations that lead from x ' +
                  'to y.');
          }
          return this.tidy('backward', () => {
              const accumulatedGradientMap = {};
              accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;
              // Backprop gradients through the filtered nodes.
              backpropagateGradients(accumulatedGradientMap, filteredTape,
              // Pass the tidy function to avoid circular dep with `tape.ts`.
              f => this.tidy(f));
              const grads = xs.map(x => accumulatedGradientMap[x.id]);
              if (this.state.gradientDepth === 0) {
                  // This means that we are not computing higher-order gradients
                  // and can clean up the tape.
                  this.state.activeTape.forEach(node => {
                      for (const tensor of node.saved) {
                          tensor.dispose();
                      }
                  });
                  this.state.activeTape = null;
              }
              return { value: y, grads };
          });
      }
      customGrad(f) {
          assert(isFunction(f), () => 'The f passed in customGrad(f) must be a function.');
          return (...inputs) => {
              assert(inputs.every(t => t instanceof Tensor), () => 'The args passed in customGrad(f)(x1, x2,...) must all be ' +
                  'tensors');
              let res;
              const inputMap = {};
              inputs.forEach((input, i) => {
                  inputMap[i] = input;
              });
              return this.runKernelFunc((_, save) => {
                  res = f(...[...inputs, save]);
                  assert(res.value instanceof Tensor, () => 'The function f passed in customGrad(f) must return an ' +
                      'object where `obj.value` is a tensor');
                  assert(isFunction(res.gradFunc), () => 'The function f passed in customGrad(f) must return an ' +
                      'object where `obj.gradFunc` is a function.');
                  return res.value;
              }, inputMap, (dy, saved) => {
                  const gradRes = res.gradFunc(dy, saved);
                  const grads = Array.isArray(gradRes) ? gradRes : [gradRes];
                  assert(grads.length === inputs.length, () => 'The function f passed in customGrad(f) must return an ' +
                      'object where `obj.gradFunc` is a function that returns ' +
                      'the same number of tensors as inputs passed to f(...).');
                  assert(grads.every(t => t instanceof Tensor), () => 'The function f passed in customGrad(f) must return an ' +
                      'object where `obj.gradFunc` is a function that returns ' +
                      'a list of only tensors.');
                  const gradMap = {};
                  grads.forEach((grad, i) => {
                      gradMap[i] = () => grad;
                  });
                  return gradMap;
              });
          };
      }
      readSync(dataId) {
          // Route the read to the correct backend.
          const info = this.state.tensorInfo.get(dataId);
          return info.backend.readSync(dataId);
      }
      read(dataId) {
          // Route the read to the correct backend.
          const info = this.state.tensorInfo.get(dataId);
          return info.backend.read(dataId);
      }
      async time(query) {
          const start = now();
          const timingInfo = await this.backend.time(query);
          timingInfo.wallMs = now() - start;
          return timingInfo;
      }
      /**
       * Tracks a Tensor in the current scope to be automatically cleaned up
       * when the current scope ends, and returns the value.
       *
       * @param result The Tensor to track in the current scope.
       */
      track(result) {
          if (this.state.activeScope != null) {
              result.scopeId = this.state.activeScope.id;
              this.state.activeScope.track.push(result);
          }
          return result;
      }
      get registeredVariables() {
          return this.state.registeredVariables;
      }
      /**
       * Resets the engine state. Removes all backends but does not remove
       * registered backend factories.
       */
      reset() {
          // Make any pending promise obsolete.
          this.pendingBackendInitId++;
          this.state.dispose();
          this.ENV.reset();
          this.state = new EngineState();
          for (const backendName in this.registry) {
              this.disposeRegisteredKernels(backendName);
              this.registry[backendName].dispose();
              delete this.registry[backendName];
          }
          this.backendName = null;
          this.backendInstance = null;
          this.pendingBackendInit = null;
      }
  }
  Engine.nextTensorId = 0;
  Engine.nextVariableId = 0;
  function ones(shape) {
      const values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
      return ENGINE.makeTensor(values, shape, 'float32');
  }
  function getOrMakeEngine() {
      const ns = getGlobalNamespace();
      if (ns._tfengine == null) {
          const environment = new Environment(ns);
          ns._tfengine = new Engine(environment);
      }
      setEnvironmentGlobal(ns._tfengine.ENV);
      // Tell the current tensor interface that the global engine is responsible
      // for tracking.
      setTensorTracker(() => ns._tfengine);
      return ns._tfengine;
  }
  const ENGINE = getOrMakeEngine();

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
  // tslint:disable-next-line:no-any
  function _isNavigatorDefined() {
      return typeof navigator !== 'undefined' && navigator != null;
  }
  function isMobile() {
      if (_isNavigatorDefined()) {
          // tslint:disable-next-line:no-any
          const a = navigator.userAgent || navigator.vendor || window.opera;
          // tslint:disable-next-line:max-line-length
          return /(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i
              .test(a) ||
              // tslint:disable-next-line:max-line-length
              /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i
                  .test(a.substr(0, 4));
      }
      return false;
  }
  function isBrowser() {
      return (typeof window !== 'undefined' && window.document != null) ||
          //@ts-ignore
          (typeof WorkerGlobalScope !== 'undefined');
  }

  var device_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    isMobile: isMobile,
    isBrowser: isBrowser
  });

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
  const ENV = env();
  /**
   * This file contains environment-related flag registrations.
   */
  /** Whether to enable debug mode. */
  ENV.registerFlag('DEBUG', () => false, debugValue => {
      if (debugValue) {
          console.warn('Debugging mode is ON. The output of every math call will ' +
              'be downloaded to CPU and checked for NaNs. ' +
              'This significantly impacts performance.');
      }
  });
  /** Whether we are in a browser (as versus, say, node.js) environment. */
  ENV.registerFlag('IS_BROWSER', () => isBrowser());
  /** Whether we are in a browser (as versus, say, node.js) environment. */
  ENV.registerFlag('IS_NODE', () => (typeof process !== 'undefined') &&
      (typeof process.versions !== 'undefined') &&
      (typeof process.versions.node !== 'undefined'));
  /** Whether this browser is Chrome. */
  ENV.registerFlag('IS_CHROME', () => typeof navigator !== 'undefined' && navigator != null &&
      navigator.userAgent != null && /Chrome/.test(navigator.userAgent) &&
      /Google Inc/.test(navigator.vendor));
  /**
   * True when the environment is "production" where we disable safety checks
   * to gain performance.
   */
  ENV.registerFlag('PROD', () => false);
  /**
   * Whether to do sanity checks when inferring a shape from user-provided
   * values, used when creating a new tensor.
   */
  ENV.registerFlag('TENSORLIKE_CHECK_SHAPE_CONSISTENCY', () => ENV.getBool('DEBUG'));
  /** Whether deprecation warnings are enabled. */
  ENV.registerFlag('DEPRECATION_WARNINGS_ENABLED', () => true);
  /** True if running unit tests. */
  ENV.registerFlag('IS_TEST', () => false);

  const Add = 'Add';
  const AddN = 'AddN';
  const Div = 'Div';
  const FusedBatchNorm = 'FusedBatchNorm';
  const NotEqual = 'NotEqual';
  const SquaredDifference = 'SquaredDifference';
  const Square = 'Square';
  const Sub = 'Sub';
  const Transpose = 'Transpose';
  const NonMaxSuppressionV5 = 'NonMaxSuppressionV5';
  const BroadcastTo = 'BroadcastTo';
  const OneHot = 'OneHot';
  const Identity = 'Identity';
  const Tile = 'Tile';
  const PadV2 = 'PadV2';
  /**
   * TensorFlow.js-only kernels
   */
  const FromPixels = 'FromPixels';
  const MaxPoolWithArgmax = 'MaxPoolWithArgmax';

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
  /**
   * Returns the dimensions in the input shape that are broadcasted to
   * produce the provided output shape.
   *
   * The returned dimensions are 0-indexed and sorted. An example:
   * inShape = [4, 1, 3]
   * outShape = [5, 4, 3, 3]
   * result = [1]. Dimension 1 (2nd dimension of input) gets broadcasted 1 => 3.
   */
  function getBroadcastDims(inShape, outShape) {
      const inRank = inShape.length;
      const dims = [];
      for (let i = 0; i < inRank; i++) {
          const dim = inRank - 1 - i;
          const a = inShape[dim] || 1;
          const b = outShape[outShape.length - 1 - i] || 1;
          if (b > 1 && a === 1) {
              dims.unshift(dim);
          }
      }
      return dims;
  }
  /**
   * Returns the axes in the output space that should be reduced to produce
   * the input space.
   */
  function getReductionAxes(inShape, outShape) {
      const result = [];
      for (let i = 0; i < outShape.length; i++) {
          const inDim = inShape[inShape.length - i - 1];
          const outAxis = outShape.length - i - 1;
          const outDim = outShape[outAxis];
          if (inDim == null || (inDim === 1 && outDim > 1)) {
              result.unshift(outAxis);
          }
      }
      return result;
  }
  function assertAndGetBroadcastShape(shapeA, shapeB) {
      const result = [];
      const l = Math.max(shapeA.length, shapeB.length);
      for (let i = 0; i < l; i++) {
          let a = shapeA[shapeA.length - i - 1];
          if (a == null) {
              a = 1;
          }
          let b = shapeB[shapeB.length - i - 1];
          if (b == null) {
              b = 1;
          }
          if (a === 1) {
              result.unshift(b);
          }
          else if (b === 1) {
              result.unshift(a);
          }
          else if (a !== b) {
              const errMsg = `Operands could not be broadcast together with shapes ` +
                  `${shapeA} and ${shapeB}.`;
              throw Error(errMsg);
          }
          else {
              result.unshift(a);
          }
      }
      return result;
  }

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const addGradConfig = {
      kernelName: Add,
      inputsToSave: ['a', 'b'],
      gradFunc: (dy, saved) => {
          const [a, b] = saved;
          const outShape = assertAndGetBroadcastShape(a.shape, b.shape);
          const derA = () => {
              let res = dy;
              const reduceAxes = getReductionAxes(a.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = res.sum(reduceAxes);
              }
              return res.reshape(a.shape);
          };
          const derB = () => {
              let res = dy;
              const reduceAxes = getReductionAxes(b.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = res.sum(reduceAxes);
              }
              return res.reshape(b.shape);
          };
          return { a: derA, b: derB };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const addNGradConfig = {
      kernelName: AddN,
      saveAllInputs: true,
      gradFunc: (dy, saved) => {
          const ders = {};
          saved.forEach((_, i) => {
              ders[i] = () => dy.clone();
          });
          return ders;
      }
  };

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
  function inferShape(val, dtype) {
      let firstElem = val;
      if (isTypedArray(val)) {
          return dtype === 'string' ? [] : [val.length];
      }
      if (!Array.isArray(val)) {
          return []; // Scalar.
      }
      const shape = [];
      while (Array.isArray(firstElem) ||
          isTypedArray(firstElem) && dtype !== 'string') {
          shape.push(firstElem.length);
          firstElem = firstElem[0];
      }
      if (Array.isArray(val) &&
          env().getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')) {
          deepAssertShapeConsistency(val, shape, []);
      }
      return shape;
  }
  function deepAssertShapeConsistency(val, shape, indices) {
      indices = indices || [];
      if (!(Array.isArray(val)) && !isTypedArray(val)) {
          assert(shape.length === 0, () => `Element arr[${indices.join('][')}] is a primitive, ` +
              `but should be an array/TypedArray of ${shape[0]} elements`);
          return;
      }
      assert(shape.length > 0, () => `Element arr[${indices.join('][')}] should be a primitive, ` +
          `but is an array of ${val.length} elements`);
      assert(val.length === shape[0], () => `Element arr[${indices.join('][')}] should have ${shape[0]} ` +
          `elements, but has ${val.length} elements`);
      const subShape = shape.slice(1);
      for (let i = 0; i < val.length; ++i) {
          deepAssertShapeConsistency(val[i], subShape, indices.concat(i));
      }
  }
  function assertDtype(expectedDtype, actualDType, argName, functionName) {
      if (expectedDtype == null) {
          return;
      }
      if (expectedDtype !== 'numeric' && expectedDtype !== actualDType ||
          expectedDtype === 'numeric' && actualDType === 'string') {
          throw new Error(`Argument '${argName}' passed to '${functionName}' must ` +
              `be ${expectedDtype} tensor, but got ${actualDType} tensor`);
      }
  }
  function convertToTensor(x, argName, functionName, parseAsDtype = 'numeric') {
      if (x instanceof Tensor) {
          assertDtype(parseAsDtype, x.dtype, argName, functionName);
          return x;
      }
      let inferredDtype = inferDtype(x);
      // If the user expects a bool/int/float, use that info to update the
      // inferredDtype when it is not a string.
      if (inferredDtype !== 'string' &&
          ['bool', 'int32', 'float32'].indexOf(parseAsDtype) >= 0) {
          inferredDtype = parseAsDtype;
      }
      assertDtype(parseAsDtype, inferredDtype, argName, functionName);
      if ((x == null) ||
          (!isTypedArray(x) && !Array.isArray(x) && typeof x !== 'number' &&
              typeof x !== 'boolean' && typeof x !== 'string')) {
          const type = x == null ? 'null' : x.constructor.name;
          throw new Error(`Argument '${argName}' passed to '${functionName}' must be a ` +
              `Tensor or TensorLike, but got '${type}'`);
      }
      const inferredShape = inferShape(x, inferredDtype);
      if (!isTypedArray(x) && !Array.isArray(x)) {
          x = [x];
      }
      const skipTypedArray = true;
      const values = inferredDtype !== 'string' ?
          toTypedArray(x, inferredDtype, env().getBool('DEBUG')) :
          flatten(x, [], skipTypedArray);
      return ENGINE.makeTensor(values, inferredShape, inferredDtype);
  }
  function convertToTensorArray(arg, argName, functionName, parseAsDtype = 'numeric') {
      if (!Array.isArray(arg)) {
          throw new Error(`Argument ${argName} passed to ${functionName} must be a ` +
              '`Tensor[]` or `TensorLike[]`');
      }
      const tensors = arg;
      return tensors.map((t, i) => convertToTensor(t, `${argName}[${i}]`, functionName), parseAsDtype);
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Provided `f(x)`, returns another function `g(x, dy?)`, which gives the
   * gradient of `f(x)` with respect to `x`.
   *
   * If `dy` is provided, the gradient of `f(x).mul(dy).sum()` with respect to
   * `x` is computed instead. `f(x)` must take a single tensor `x` and return a
   * single tensor `y`. If `f()` takes multiple inputs, use `tf.grads` instead.
   *
   * ```js
   * // f(x) = x ^ 2
   * const f = x => x.square();
   * // f'(x) = 2x
   * const g = tf.grad(f);
   *
   * const x = tf.tensor1d([2, 3]);
   * g(x).print();
   * ```
   *
   * ```js
   * // f(x) = x ^ 3
   * const f = x => x.pow(tf.scalar(3, 'int32'));
   * // f'(x) = 3x ^ 2
   * const g = tf.grad(f);
   * // f''(x) = 6x
   * const gg = tf.grad(g);
   *
   * const x = tf.tensor1d([2, 3]);
   * gg(x).print();
   * ```
   *
   * @param f The function f(x), to compute gradient for.
   */
  /** @doc {heading: 'Training', subheading: 'Gradients'} */
  function grad(f) {
      assert(isFunction(f), () => 'The f passed in grad(f) must be a function');
      return (x, dy) => {
          // x can be of any dtype, thus null as the last argument.
          const $x = convertToTensor(x, 'x', 'tf.grad', null);
          const $dy = (dy != null) ? convertToTensor(dy, 'dy', 'tf.grad') : null;
          return ENGINE.tidy(() => {
              const { value, grads } = ENGINE.gradients(() => f($x), [$x], $dy);
              if ($dy != null) {
                  assertShapesMatch(value.shape, $dy.shape, 'The shape of dy passed in grad(f)(x, dy) must match the shape ' +
                      'returned by f(x)');
              }
              checkGrads(grads);
              return grads[0];
          });
      };
  }
  /**
   * Provided `f(x1, x2,...)`, returns another function `g([x1, x2,...], dy?)`,
   * which gives an array of gradients of `f()` with respect to each input
   * [`x1`,`x2`,...].
   *
   * If `dy` is passed when calling `g()`, the gradient of
   * `f(x1,...).mul(dy).sum()` with respect to each input is computed instead.
   * The provided `f` must take one or more tensors and return a single tensor
   * `y`. If `f()` takes a single input, we recommend using `tf.grad` instead.
   *
   * ```js
   * // f(a, b) = a * b
   * const f = (a, b) => a.mul(b);
   * // df / da = b, df / db = a
   * const g = tf.grads(f);
   *
   * const a = tf.tensor1d([2, 3]);
   * const b = tf.tensor1d([-2, -3]);
   * const [da, db] = g([a, b]);
   * console.log('da');
   * da.print();
   * console.log('db');
   * db.print();
   * ```
   *
   * @param f The function `f(x1, x2,...)` to compute gradients for.
   */
  /** @doc {heading: 'Training', subheading: 'Gradients'} */
  function grads(f) {
      assert(isFunction(f), () => 'The f passed in grads(f) must be a function');
      return (args, dy) => {
          assert(Array.isArray(args), () => 'The args passed in grads(f)(args) must be an array ' +
              'of `Tensor`s or `TensorLike`s');
          // args can be of any dtype, thus null as the last argument.
          const $args = convertToTensorArray(args, 'args', 'tf.grads', null);
          const $dy = (dy != null) ? convertToTensor(dy, 'dy', 'tf.grads') : null;
          return ENGINE.tidy(() => {
              const { value, grads } = ENGINE.gradients(() => f(...$args), $args, $dy);
              if ($dy != null) {
                  assertShapesMatch(value.shape, $dy.shape, 'The shape of dy passed in grads(f)([x1,...], dy) must ' +
                      'match the shape returned by f([x1,...])');
              }
              checkGrads(grads);
              return grads;
          });
      };
  }
  /**
   * Like `tf.grad`, but also returns the value of `f()`. Useful when `f()`
   * returns a metric you want to show.
   *
   * The result is a rich object with the following properties:
   * - grad: The gradient of `f(x)` w.r.t `x` (result of `tf.grad`).
   * - value: The value returned by `f(x)`.
   *
   * ```js
   * // f(x) = x ^ 2
   * const f = x => x.square();
   * // f'(x) = 2x
   * const g = tf.valueAndGrad(f);
   *
   * const x = tf.tensor1d([2, 3]);
   * const {value, grad} = g(x);
   *
   * console.log('value');
   * value.print();
   * console.log('grad');
   * grad.print();
   * ```
   */
  /** @doc {heading: 'Training', subheading: 'Gradients'} */
  function valueAndGrad(f) {
      assert(isFunction(f), () => 'The f passed in valueAndGrad(f) must be a function');
      return (x, dy) => {
          assert(x instanceof Tensor, () => 'The x passed in valueAndGrad(f)(x) must be a tensor');
          assert(dy == null || dy instanceof Tensor, () => 'The dy passed in valueAndGrad(f)(x, dy) must be a tensor');
          const { grads, value } = ENGINE.gradients(() => f(x), [x], dy);
          checkGrads(grads);
          return { grad: grads[0], value };
      };
  }
  /**
   * Like `tf.grads`, but returns also the value of `f()`. Useful when `f()`
   * returns a metric you want to show.
   *
   * The result is a rich object with the following properties:
   * - grads: The gradients of `f()` w.r.t each input (result of `tf.grads`).
   * - value: The value returned by `f(x)`.
   *
   * ```js
   * // f(a, b) = a * b
   * const f = (a, b) => a.mul(b);
   * // df/da = b, df/db = a
   * const g = tf.valueAndGrads(f);
   *
   * const a = tf.tensor1d([2, 3]);
   * const b = tf.tensor1d([-2, -3]);
   * const {value, grads} = g([a, b]);
   *
   * const [da, db] = grads;
   *
   * console.log('value');
   * value.print();
   *
   * console.log('da');
   * da.print();
   * console.log('db');
   * db.print();
   * ```
   */
  /** @doc {heading: 'Training', subheading: 'Gradients'} */
  function valueAndGrads(f) {
      assert(isFunction(f), () => 'The f passed in valueAndGrads(f) must be a function');
      return (args, dy) => {
          assert(Array.isArray(args) && args.every(arg => arg instanceof Tensor), () => 'The args passed in valueAndGrads(f)(args) must be array of ' +
              'tensors');
          assert(dy == null || dy instanceof Tensor, () => 'The dy passed in valueAndGrads(f)(args, dy) must be a tensor');
          const res = ENGINE.gradients(() => f(...args), args, dy);
          if (dy != null) {
              assertShapesMatch(res.value.shape, dy.shape, 'The shape of dy passed in valueAndGrads(f)([x1,...], dy) must ' +
                  'match the shape returned by f([x1,...])');
          }
          checkGrads(res.grads);
          return res;
      };
  }
  /**
   * Computes and returns the gradient of f(x) with respect to the list of
   * trainable variables provided by `varList`. If no list is provided, it
   * defaults to all trainable variables.
   *
   * ```js
   * const a = tf.variable(tf.tensor1d([3, 4]));
   * const b = tf.variable(tf.tensor1d([5, 6]));
   * const x = tf.tensor1d([1, 2]);
   *
   * // f(a, b) = a * x ^ 2 + b * x
   * const f = () => a.mul(x.square()).add(b.mul(x)).sum();
   * // df/da = x ^ 2, df/db = x
   * const {value, grads} = tf.variableGrads(f);
   *
   * Object.keys(grads).forEach(varName => grads[varName].print());
   * ```
   *
   * @param f The function to execute. f() should return a scalar.
   * @param varList The list of variables to compute the gradients with respect
   *     to. Defaults to all trainable variables.
   * @returns An object with the following keys and values:
   *   - `value`: The value of the function `f`.
   *   - `grads`: A map from the names of the variables to the gradients.
   *     If the `varList` argument is provided explicitly and contains a subset of
   *     non-trainable variables, this map in the return value will contain keys
   *     that map the names of the non-trainable variables to `null`.
   */
  /** @doc {heading: 'Training', subheading: 'Gradients'} */
  function variableGrads(f, varList) {
      assert(isFunction(f), () => 'The f passed in variableGrads(f) must be a function');
      assert(varList == null ||
          Array.isArray(varList) && varList.every(v => v instanceof Variable), () => 'The varList passed in variableGrads(f, varList) must be an array ' +
          'of variables');
      const specifiedVarList = varList != null;
      if (!specifiedVarList) {
          // Get all of the trainable variables.
          varList = [];
          for (const varName in ENGINE.registeredVariables) {
              varList.push(ENGINE.registeredVariables[varName]);
          }
      }
      const specifiedNonTrainable = specifiedVarList ? varList.filter(variable => !variable.trainable) : null;
      // Prune non-trainable variables.
      const originalVarCount = varList.length;
      varList = varList.filter(variable => variable.trainable);
      assert(varList.length > 0, () => `variableGrads() expects at least one of the input variables to ` +
          `be trainable, but none of the ${originalVarCount} variables is ` +
          `trainable.`);
      const allowNoGradients = true;
      const { value, grads } = ENGINE.gradients(f, varList, null, allowNoGradients);
      assert(grads.some(g => g != null), () => 'Cannot find a connection between any variable and the result of ' +
          'the loss function y=f(x). Please make sure the operations that ' +
          'use variables are inside the function f passed to minimize().');
      assert(value.rank === 0, () => `The f passed in variableGrads(f) must return a scalar, but it ` +
          `returned a rank-${value.rank} tensor`);
      const namedGrads = {};
      varList.forEach((v, i) => {
          if (grads[i] != null) {
              namedGrads[v.name] = grads[i];
          }
      });
      if (specifiedNonTrainable != null) {
          // If varList is explicitly provided and contains non-trainable values,
          // add them to the returned gradients with `null` values.
          specifiedNonTrainable.forEach(v => namedGrads[v.name] = null);
      }
      return { value, grads: namedGrads };
  }
  /**
   * Overrides the gradient computation of a function `f`.
   *
   * Takes a function
   * `f(...inputs, save) => {value: Tensor, gradFunc: (dy, saved) => Tensor[]}`
   * and returns another function `g(...inputs)` which takes the same inputs as
   * `f`. When called, `g` returns `f().value`. In backward mode, custom gradients
   * with respect to each input of `f` are computed using `f().gradFunc`.
   *
   * The `save` function passsed to `f` should be used for saving tensors needed
   * in the gradient. And the `saved` passed to the `gradFunc` is a
   * `NamedTensorMap`, which contains those saved tensor.
   *
   * ```js
   * const customOp = tf.customGrad((x, save) => {
   *   // Save x to make sure it's available later for the gradient.
   *   save([x]);
   *   // Override gradient of our custom x ^ 2 op to be dy * abs(x);
   *   return {
   *     value: x.square(),
   *     // Note `saved.x` which points to the `x` we saved earlier.
   *     gradFunc: (dy, saved) => [dy.mul(saved[0].abs())]
   *   };
   * });
   *
   * const x = tf.tensor1d([-1, -2, 3]);
   * const dx = tf.grad(x => customOp(x));
   *
   * console.log(`f(x):`);
   * customOp(x).print();
   * console.log(`f'(x):`);
   * dx(x).print();
   * ```
   *
   * @param f The function to evaluate in forward mode, which should return
   *     `{value: Tensor, gradFunc: (dy, saved) => Tensor[]}`, where `gradFunc`
   *     returns the custom gradients of `f` with respect to its inputs.
   */
  /** @doc {heading: 'Training', subheading: 'Gradients'} */
  function customGrad(f) {
      return ENGINE.customGrad(f);
  }
  function checkGrads(grads) {
      const numNullGradients = grads.filter(g => g == null).length;
      if (numNullGradients > 0) {
          throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`);
      }
  }

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
  /**
   * Returns true if the axis specifies the inner most dimensions of the
   * array.
   */
  function axesAreInnerMostDims(axes, rank) {
      for (let i = 0; i < axes.length; ++i) {
          if (axes[axes.length - i - 1] !== rank - 1 - i) {
              return false;
          }
      }
      return true;
  }
  function combineLocations(outputLoc, reduceLoc, axes) {
      const rank = outputLoc.length + reduceLoc.length;
      const loc = [];
      let outIdx = 0;
      let reduceIdx = 0;
      for (let dim = 0; dim < rank; dim++) {
          if (axes.indexOf(dim) === -1) {
              loc.push(outputLoc[outIdx++]);
          }
          else {
              loc.push(reduceLoc[reduceIdx++]);
          }
      }
      return loc;
  }
  function computeOutAndReduceShapes(aShape, axes) {
      const outShape = [];
      const rank = aShape.length;
      for (let dim = 0; dim < rank; dim++) {
          if (axes.indexOf(dim) === -1) {
              outShape.push(aShape[dim]);
          }
      }
      const reduceShape = axes.map(dim => aShape[dim]);
      return [outShape, reduceShape];
  }
  function expandShapeToKeepDim(shape, axes) {
      const reduceSubShape = axes.map(x => 1);
      return combineLocations(shape, reduceSubShape, axes);
  }
  function assertAxesAreInnerMostDims(msg, axes, rank) {
      assert(axesAreInnerMostDims(axes, rank), () => `${msg} supports only inner-most axes for now. ` +
          `Got axes ${axes} and rank-${rank} input.`);
  }
  /**
   * Returns the axes permutation to be used with `tf.transpose`, if such
   * permutation is necessary. Otherwise it returns null. This method is used by
   * operations that operate only on inner-most axes.
   */
  function getAxesPermutation(axes, rank) {
      if (axesAreInnerMostDims(axes, rank)) {
          return null;
      }
      const result = [];
      for (let i = 0; i < rank; ++i) {
          if (axes.indexOf(i) === -1) {
              result.push(i);
          }
      }
      axes.forEach(axis => result.push(axis));
      return result;
  }
  /** Returns the axes permutation that undoes the original permutation. */
  function getUndoAxesPermutation(axes) {
      return axes.map((axis, i) => [i, axis])
          .sort((a, b) => a[1] - b[1])
          .map(x => x[0]);
  }
  function getInnerMostAxes(numAxes, rank) {
      const res = [];
      for (let i = rank - numAxes; i < rank; ++i) {
          res.push(i);
      }
      return res;
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Used for wrapping functions that perform math operations on
   * Tensors. The function will be wrapped in a named scope that cleans all
   * memory usage after the function is done.
   */
  function op(f) {
      const keys = Object.keys(f);
      if (keys.length !== 1) {
          throw new Error(`Please provide an object with a single key ` +
              `(operation name) mapping to a function. Got an object with ` +
              `${keys.length} keys.`);
      }
      let opName = keys[0];
      const fn = f[opName];
      // Strip the underscore from the end of the function name.
      if (opName.endsWith('_')) {
          opName = opName.substring(0, opName.length - 1);
      }
      // tslint:disable-next-line:no-any
      const f2 = (...args) => {
          ENGINE.startScope(opName);
          try {
              const result = fn(...args);
              if (result instanceof Promise) {
                  console.error('Cannot return a Promise inside of tidy.');
              }
              ENGINE.endScope(result);
              return result;
          }
          catch (ex) {
              ENGINE.endScope(null);
              throw ex;
          }
      };
      Object.defineProperty(f2, 'name', { value: opName, configurable: true });
      // tslint:disable-next-line:no-any
      return f2;
  }

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
  /**
   * Converts two real numbers to a complex number.
   *
   * Given a tensor `real` representing the real part of a complex number, and a
   * tensor `imag` representing the imaginary part of a complex number, this
   * operation returns complex numbers elementwise of the form [r0, i0, r1, i1],
   * where r represents the real part and i represents the imag part.
   *
   * The input tensors real and imag must have the same shape.
   *
   * ```js
   * const real = tf.tensor1d([2.25, 3.25]);
   * const imag = tf.tensor1d([4.75, 5.75]);
   * const complex = tf.complex(real, imag);
   *
   * complex.print();
   * ```
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function complex_(real, imag) {
      const $real = convertToTensor(real, 'real', 'complex');
      const $imag = convertToTensor(imag, 'imag', 'complex');
      assertShapesMatch($real.shape, $imag.shape, `real and imag shapes, ${$real.shape} and ${$imag.shape}, ` +
          `must match in call to tf.complex().`);
      return ENGINE.runKernelFunc(backend => backend.complex($real, $imag), { $real, $imag });
  }
  /**
   * Returns the real part of a complex (or real) tensor.
   *
   * Given a tensor input, this operation returns a tensor of type float that is
   * the real part of each element in input considered as a complex number.
   *
   * If the input is real, it simply makes a clone.
   *
   * ```js
   * const x = tf.complex([-2.25, 3.25], [4.75, 5.75]);
   * tf.real(x).print();
   * ```
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function real_(input) {
      const $input = convertToTensor(input, 'input', 'real');
      return ENGINE.runKernelFunc(backend => backend.real($input), { $input });
  }
  /**
   * Returns the imaginary part of a complex (or real) tensor.
   *
   * Given a tensor input, this operation returns a tensor of type float that is
   * the imaginary part of each element in input considered as a complex number.
   * If input is real, a tensor of all zeros is returned.
   *
   * ```js
   * const x = tf.complex([-2.25, 3.25], [4.75, 5.75]);
   * tf.imag(x).print();
   * ```
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function imag_(input) {
      const $input = convertToTensor(input, 'input', 'imag');
      return ENGINE.runKernelFunc(backend => backend.imag($input), { $input });
  }
  const complex = op({ complex_ });
  const real = op({ real_ });
  const imag = op({ imag_ });

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
  /**
   * Creates a `tf.Tensor` with the provided values, shape and dtype.
   *
   * ```js
   * // Pass an array of values to create a vector.
   * tf.tensor([1, 2, 3, 4]).print();
   * ```
   *
   * ```js
   * // Pass a nested array of values to make a matrix or a higher
   * // dimensional tensor.
   * tf.tensor([[1, 2], [3, 4]]).print();
   * ```
   *
   * ```js
   * // Pass a flat array and specify a shape yourself.
   * tf.tensor([1, 2, 3, 4], [2, 2]).print();
   * ```
   *
   * @param values The values of the tensor. Can be nested array of numbers,
   *     or a flat array, or a `TypedArray`. If the values are strings,
   *     they will be encoded as utf-8 and kept as `Uint8Array[]`.
   * @param shape The shape of the tensor. Optional. If not provided,
   *   it is inferred from `values`.
   * @param dtype The data type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function tensor(values, shape, dtype) {
      const inferredShape = inferShape(values, dtype);
      return makeTensor(values, shape, inferredShape, dtype);
  }
  /** This is shared code across all tensor creation methods. */
  function makeTensor(values, shape, inferredShape, dtype) {
      if (dtype == null) {
          dtype = inferDtype(values);
      }
      if (dtype === 'complex64') {
          throw new Error(`Cannot construct a complex64 tensor directly. ` +
              `Please use tf.complex(real, imag).`);
      }
      if (!isTypedArray(values) && !Array.isArray(values) &&
          typeof values !== 'number' && typeof values !== 'boolean' &&
          typeof values !== 'string') {
          throw new Error('values passed to tensor(values) must be a number/boolean/string or ' +
              'an array of numbers/booleans/strings, or a TypedArray');
      }
      if (shape != null) {
          assertNonNegativeIntegerDimensions(shape);
          const providedSize = sizeFromShape(shape);
          const inferredSize = sizeFromShape(inferredShape);
          assert(providedSize === inferredSize, () => `Based on the provided shape, [${shape}], the tensor should have ` +
              `${providedSize} values but has ${inferredSize}`);
          for (let i = 0; i < inferredShape.length; ++i) {
              const inferred = inferredShape[i];
              const flatDimsDontMatch = i === inferredShape.length - 1 ?
                  inferred !== sizeFromShape(shape.slice(i)) :
                  true;
              assert(inferredShape[i] === shape[i] || !flatDimsDontMatch, () => `Error creating a new Tensor. Inferred shape ` +
                  `(${inferredShape}) does not match the provided ` +
                  `shape (${shape}). `);
          }
      }
      if (!isTypedArray(values) && !Array.isArray(values)) {
          values = [values];
      }
      shape = shape || inferredShape;
      values = dtype !== 'string' ?
          toTypedArray(values, dtype, env().getBool('DEBUG')) :
          flatten(values, [], true);
      return ENGINE.makeTensor(values, shape, dtype);
  }
  /**
   * Creates rank-0 `tf.Tensor` (scalar) with the provided value and dtype.
   *
   * The same functionality can be achieved with `tf.tensor`, but in general
   * we recommend using `tf.scalar` as it makes the code more readable.
   *
   * ```js
   * tf.scalar(3.14).print();
   * ```
   *
   * @param value The value of the scalar.
   * @param dtype The data type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function scalar(value, dtype) {
      if (((isTypedArray(value) && dtype !== 'string') || Array.isArray(value)) &&
          dtype !== 'complex64') {
          throw new Error('Error creating a new Scalar: value must be a primitive ' +
              '(number|boolean|string)');
      }
      if (dtype === 'string' && isTypedArray(value) &&
          !(value instanceof Uint8Array)) {
          throw new Error('When making a scalar from encoded string, ' +
              'the value must be `Uint8Array`.');
      }
      const shape = [];
      const inferredShape = [];
      return makeTensor(value, shape, inferredShape, dtype);
  }
  /**
   * Creates rank-1 `tf.Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tf.tensor`, but in general
   * we recommend using `tf.tensor1d` as it makes the code more readable.
   *
   * ```js
   * tf.tensor1d([1, 2, 3]).print();
   * ```
   *
   * @param values The values of the tensor. Can be array of numbers,
   *     or a `TypedArray`.
   * @param dtype The data type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function tensor1d(values, dtype) {
      assertNonNull(values);
      const inferredShape = inferShape(values, dtype);
      if (inferredShape.length !== 1) {
          throw new Error('tensor1d() requires values to be a flat/TypedArray');
      }
      const shape = null;
      return makeTensor(values, shape, inferredShape, dtype);
  }
  /**
   * Creates rank-2 `tf.Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tf.tensor`, but in general
   * we recommend using `tf.tensor2d` as it makes the code more readable.
   *
   *  ```js
   * // Pass a nested array.
   * tf.tensor2d([[1, 2], [3, 4]]).print();
   * ```
   * ```js
   * // Pass a flat array and specify a shape.
   * tf.tensor2d([1, 2, 3, 4], [2, 2]).print();
   * ```
   *
   * @param values The values of the tensor. Can be nested array of numbers,
   *     or a flat array, or a `TypedArray`.
   * @param shape The shape of the tensor. If not provided, it is inferred from
   *     `values`.
   * @param dtype The data type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function tensor2d(values, shape, dtype) {
      assertNonNull(values);
      if (shape != null && shape.length !== 2) {
          throw new Error('tensor2d() requires shape to have two numbers');
      }
      const inferredShape = inferShape(values, dtype);
      if (inferredShape.length !== 2 && inferredShape.length !== 1) {
          throw new Error('tensor2d() requires values to be number[][] or flat/TypedArray');
      }
      if (inferredShape.length === 1 && shape == null) {
          throw new Error('tensor2d() requires shape to be provided when `values` ' +
              'are a flat/TypedArray');
      }
      return makeTensor(values, shape, inferredShape, dtype);
  }
  /**
   * Creates rank-3 `tf.Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tf.tensor`, but in general
   * we recommend using `tf.tensor3d` as it makes the code more readable.
   *
   *  ```js
   * // Pass a nested array.
   * tf.tensor3d([[[1], [2]], [[3], [4]]]).print();
   * ```
   * ```js
   * // Pass a flat array and specify a shape.
   * tf.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
   * ```
   *
   * @param values The values of the tensor. Can be nested array of numbers,
   *     or a flat array, or a `TypedArray`.
   * @param shape The shape of the tensor. If not provided,  it is inferred from
   *     `values`.
   * @param dtype The data type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function tensor3d(values, shape, dtype) {
      assertNonNull(values);
      if (shape != null && shape.length !== 3) {
          throw new Error('tensor3d() requires shape to have three numbers');
      }
      const inferredShape = inferShape(values, dtype);
      if (inferredShape.length !== 3 && inferredShape.length !== 1) {
          throw new Error('tensor3d() requires values to be number[][][] or flat/TypedArray');
      }
      if (inferredShape.length === 1 && shape == null) {
          throw new Error('tensor3d() requires shape to be provided when `values` ' +
              'are a flat array');
      }
      return makeTensor(values, shape, inferredShape, dtype);
  }
  /**
   * Creates rank-4 `tf.Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tf.tensor`, but in general
   * we recommend using `tf.tensor4d` as it makes the code more readable.
   *
   *  ```js
   * // Pass a nested array.
   * tf.tensor4d([[[[1], [2]], [[3], [4]]]]).print();
   * ```
   * ```js
   * // Pass a flat array and specify a shape.
   * tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]).print();
   * ```
   *
   * @param values The values of the tensor. Can be nested array of numbers,
   *     or a flat array, or a `TypedArray`.
   * @param shape The shape of the tensor. Optional. If not provided,
   *   it is inferred from `values`.
   * @param dtype The data type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function tensor4d(values, shape, dtype) {
      assertNonNull(values);
      if (shape != null && shape.length !== 4) {
          throw new Error('tensor4d() requires shape to have four numbers');
      }
      const inferredShape = inferShape(values, dtype);
      if (inferredShape.length !== 4 && inferredShape.length !== 1) {
          throw new Error('tensor4d() requires values to be number[][][][] or flat/TypedArray');
      }
      if (inferredShape.length === 1 && shape == null) {
          throw new Error('tensor4d() requires shape to be provided when `values` ' +
              'are a flat array');
      }
      return makeTensor(values, shape, inferredShape, dtype);
  }
  /**
   * Creates rank-5 `tf.Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tf.tensor`, but in general
   * we recommend using `tf.tensor5d` as it makes the code more readable.
   *
   *  ```js
   * // Pass a nested array.
   * tf.tensor5d([[[[[1], [2]], [[3], [4]]]]]).print();
   * ```
   * ```js
   * // Pass a flat array and specify a shape.
   * tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]).print();
   * ```
   *
   * @param values The values of the tensor. Can be nested array of numbers,
   *     or a flat array, or a `TypedArray`.
   * @param shape The shape of the tensor. Optional. If not provided,
   *   it is inferred from `values`.
   * @param dtype The data type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function tensor5d(values, shape, dtype) {
      assertNonNull(values);
      if (shape != null && shape.length !== 5) {
          throw new Error('tensor5d() requires shape to have five numbers');
      }
      const inferredShape = inferShape(values, dtype);
      if (inferredShape.length !== 5 && inferredShape.length !== 1) {
          throw new Error('tensor5d() requires values to be ' +
              'number[][][][][] or flat/TypedArray');
      }
      if (inferredShape.length === 1 && shape == null) {
          throw new Error('tensor5d() requires shape to be provided when `values` ' +
              'are a flat array');
      }
      return makeTensor(values, shape, inferredShape, dtype);
  }
  /**
   * Creates rank-6 `tf.Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tf.tensor`, but in general
   * we recommend using `tf.tensor6d` as it makes the code more readable.
   *
   *  ```js
   * // Pass a nested array.
   * tf.tensor6d([[[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]]).print();
   * ```
   * ```js
   * // Pass a flat array and specify a shape.
   * tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2, 1]).print();
   * ```
   *
   * @param values The values of the tensor. Can be nested array of numbers,
   *     or a flat array, or a `TypedArray`.
   * @param shape The shape of the tensor. Optional. If not provided,
   *   it is inferred from `values`.
   * @param dtype The data type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function tensor6d(values, shape, dtype) {
      assertNonNull(values);
      if (shape != null && shape.length !== 6) {
          throw new Error('tensor6d() requires shape to have six numbers');
      }
      const inferredShape = inferShape(values, dtype);
      if (inferredShape.length !== 6 && inferredShape.length !== 1) {
          throw new Error('tensor6d() requires values to be number[][][][][][] or ' +
              'flat/TypedArray');
      }
      if (inferredShape.length === 1 && shape == null) {
          throw new Error('tensor6d() requires shape to be provided when `values` ' +
              'are a flat array');
      }
      shape = shape ||
          inferredShape;
      return makeTensor(values, shape, inferredShape, dtype);
  }
  /**
   * Creates a new variable with the provided initial value.
   * ```js
   * const x = tf.variable(tf.tensor([1, 2, 3]));
   * x.assign(tf.tensor([4, 5, 6]));
   *
   * x.print();
   * ```
   *
   * @param initialValue Initial value for the tensor.
   * @param trainable If true, optimizers are allowed to update it.
   * @param name Name of the variable. Defaults to a unique id.
   * @param dtype If set, initialValue will be converted to the given type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function variable(initialValue, trainable = true, name, dtype) {
      return ENGINE.makeVariable(initialValue, trainable, name, dtype);
  }
  /**
   * Creates a `tf.Tensor` with all elements set to 1.
   *
   * ```js
   * tf.ones([2, 2]).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param dtype The type of an element in the resulting tensor. Defaults to
   *     'float'.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function ones$1(shape, dtype = 'float32') {
      if (dtype === 'complex64') {
          const real = ones$1(shape, 'float32');
          const imag = zeros(shape, 'float32');
          return complex(real, imag);
      }
      const values = makeOnesTypedArray(sizeFromShape(shape), dtype);
      return ENGINE.makeTensor(values, shape, dtype);
  }
  /**
   * Creates a `tf.Tensor` with all elements set to 0.
   *
   * ```js
   * tf.zeros([2, 2]).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param dtype The type of an element in the resulting tensor. Can
   *     be 'float32', 'int32' or 'bool'. Defaults to 'float'.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function zeros(shape, dtype = 'float32') {
      if (dtype === 'complex64') {
          const real = zeros(shape, 'float32');
          const imag = zeros(shape, 'float32');
          return complex(real, imag);
      }
      const values = makeZerosTypedArray(sizeFromShape(shape), dtype);
      return ENGINE.makeTensor(values, shape, dtype);
  }
  /**
   * Creates a `tf.Tensor` filled with a scalar value.
   *
   * ```js
   * tf.fill([2, 2], 4).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param value The scalar value to fill the tensor with.
   * @param dtype The type of an element in the resulting tensor. Defaults to
   * 'float'.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function fill(shape, value, dtype) {
      return ENGINE.runKernelFunc(backend => backend.fill(shape, value, dtype), {});
  }
  /**
   * Creates a `tf.Tensor` with all elements set to 1 with the same shape as the
   * given tensor.
   *
   * ```js
   * const x = tf.tensor([1, 2]);
   * tf.onesLike(x).print();
   * ```
   * @param x A tensor.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function onesLike_(x) {
      const $x = convertToTensor(x, 'x', 'onesLike');
      if ($x.dtype === 'complex64') {
          const r = onesLike(real($x));
          const i = zerosLike(imag($x));
          return complex(r, i);
      }
      const der = (dy, saved) => ({ x: () => zerosLike(dy) });
      return ENGINE.runKernelFunc(backend => backend.onesLike($x), { x: $x }, der, 'OnesLike');
  }
  /**
   * Creates a `tf.Tensor` with all elements set to 0 with the same shape as the
   * given tensor.
   *
   * ```js
   * const x = tf.tensor([1, 2]);
   * tf.zerosLike(x).print();
   * ```
   *
   * @param x The tensor of required shape.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function zerosLike_(x) {
      const $x = convertToTensor(x, 'x', 'zerosLike');
      const der = (dy, saved) => ({ x: () => zerosLike(dy) });
      return ENGINE.runKernelFunc(backend => backend.zerosLike($x), { x: $x }, der, 'ZerosLike');
  }
  /**
   * Return an evenly spaced sequence of numbers over the given interval.
   *
   * ```js
   * tf.linspace(0, 9, 10).print();
   * ```
   * @param start The start value of the sequence.
   * @param stop The end value of the sequence.
   * @param num The number of values to generate.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function linspace(start, stop, num) {
      if (num <= 0) {
          throw new Error('The number of values should be positive.');
      }
      return ENGINE.runKernelFunc(backend => backend.linspace(start, stop, num), {});
  }
  /**
   * Creates a new `tf.Tensor1D` filled with the numbers in the range provided.
   *
   * The tensor is a is half-open interval meaning it includes start, but
   * excludes stop. Decrementing ranges and negative step values are also
   * supported.
   *
   * ```js
   * tf.range(0, 9, 2).print();
   * ```
   *
   * @param start An integer start value
   * @param stop An integer stop value
   * @param step An integer increment (will default to 1 or -1)
   * @param dtype The data type of the output tensor. Defaults to 'float32'.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function range(start, stop, step = 1, dtype = 'float32') {
      if (step === 0) {
          throw new Error('Cannot have a step of zero');
      }
      const sameStartStop = start === stop;
      const increasingRangeNegativeStep = start < stop && step < 0;
      const decreasingRangePositiveStep = stop < start && step > 1;
      if (sameStartStop || increasingRangeNegativeStep ||
          decreasingRangePositiveStep) {
          return zeros([0], dtype);
      }
      const numElements = Math.abs(Math.ceil((stop - start) / step));
      const values = makeZerosTypedArray(numElements, dtype);
      if (stop < start && step === 1) {
          // Auto adjust the step's sign if it hasn't been set
          // (or was set to 1)
          step = -1;
      }
      values[0] = start;
      for (let i = 1; i < values.length; i++) {
          values[i] = values[i - 1] + step;
      }
      return tensor1d(values, dtype);
  }
  const onesLike = op({ onesLike_ });
  const zerosLike = op({ zerosLike_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes the log(sum(exp(elements across the reduction dimensions)).
   *
   * Reduces the input along the dimensions given in `axis`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axis`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axis` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3]);
   *
   * x.logSumExp().print();  // or tf.logSumExp(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * const axis = 1;
   * x.logSumExp(axis).print();  // or tf.logSumExp(a, axis)
   * ```
   * @param x The input tensor.
   * @param axis The dimension(s) to reduce. If null (the default),
   *     reduces all dimensions.
   * @param keepDims If true, retains reduced dimensions with length
   *     of 1. Defaults to false.
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function logSumExp_(x, axis = null, keepDims = false) {
      const $x = convertToTensor(x, 'x', 'logSumExp');
      const axes = parseAxisParam(axis, $x.shape);
      const xMax = $x.max(axes, true /* keepDims */);
      const a = $x.sub(xMax);
      const b = a.exp();
      const c = b.sum(axes);
      const d = c.log();
      const res = xMax.reshape(d.shape).add(d);
      if (keepDims) {
          const newShape = expandShapeToKeepDim(res.shape, axes);
          return res.reshape(newShape);
      }
      return res;
  }
  /**
   * Computes the sum of elements across dimensions of a `tf.Tensor`.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
   * `axes`. If `keepDims` is true, the reduced dimensions are retained with
   * length 1. If axes has no entries, all dimensions are reduced, and a
   * `tf.Tensor` with a single element is returned.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3]);
   *
   * x.sum().print();  // or tf.sum(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * const axis = 1;
   * x.sum(axis).print();  // or tf.sum(x, axis)
   * ```
   *
   * @param x The input tensor to compute the sum over. If the dtype is `bool`
   *   it will be converted to `int32` and the output dtype will be `int32`.
   * @param axis The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims If true, retains reduced dimensions with size 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function sum_(x, axis = null, keepDims = false) {
      let $x = convertToTensor(x, 'x', 'sum');
      if ($x.dtype === 'bool') {
          $x = $x.toInt();
      }
      const axes = parseAxisParam(axis, $x.shape);
      // Use a custom gradient to bypass 2 gradient backprops since sum is used
      // extremely often.
      const customOp = customGrad((x) => {
          const permutation = getAxesPermutation(axes, x.rank);
          let reductionAxes = axes;
          let permutedX = x;
          if (permutation != null) {
              permutedX = x.transpose(permutation);
              reductionAxes = getInnerMostAxes(reductionAxes.length, x.rank);
          }
          const gradFunc = (dy) => {
              const expandedDyShape = x.shape.slice();
              axes.forEach(axis => {
                  expandedDyShape[axis] = 1;
              });
              const expandedDy = dy.reshape(expandedDyShape);
              const derX = expandedDy.mul(ones$1(x.shape, 'float32'));
              return derX;
          };
          const gradInputs = (dy) => {
              return { x: () => gradFunc(dy) };
          };
          const attrs = { axes: reductionAxes };
          let value = ENGINE.runKernelFunc(backend => backend.sum(permutedX, reductionAxes), { x: permutedX }, gradInputs, 'Sum', attrs);
          if (keepDims) {
              const newShape = expandShapeToKeepDim(value.shape, axes);
              value = value.reshape(newShape);
          }
          return { value, gradFunc };
      });
      return customOp($x);
  }
  /**
   * Computes the product of elements across dimensions of a `tf.Tensor`.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
   * `axes`. If `keepDims` is true, the reduced dimensions are retained with
   * length 1. If `axes` has no entries, all dimensions are reduced, and a
   * `tf.Tensor` with a single element is returned.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3]);
   *
   * x.prod().print();  // or tf.prod(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * const axis = 1;
   * x.prod(axis).print();  // or tf.prod(x, axis)
   * ```
   *
   * @param x The input tensor to compute the product over. If the dtype is `bool`
   *   it will be converted to `int32` and the output dtype will be `int32`.
   * @param axis The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims If true, retains reduced dimensions with size 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function prod_(x, axis = null, keepDims = false) {
      let $x = convertToTensor(x, 'x', 'prod');
      if ($x.dtype === 'bool') {
          $x = $x.toInt();
      }
      const axes = parseAxisParam(axis, $x.shape);
      const permutation = getAxesPermutation(axes, $x.rank);
      let reductionAxes = axes;
      let permutedX = $x;
      if (permutation != null) {
          permutedX = $x.transpose(permutation);
          reductionAxes = getInnerMostAxes(reductionAxes.length, $x.rank);
      }
      let value = ENGINE.runKernelFunc(backend => backend.prod(permutedX, reductionAxes), { permutedX });
      if (keepDims) {
          const newShape = expandShapeToKeepDim(value.shape, axes);
          value = value.reshape(newShape);
      }
      return value;
  }
  /**
   * Computes the mean of elements across dimensions of a `tf.Tensor`.
   *
   * Reduces `x` along the dimensions given in `axis`. Unless `keepDims` is
   * true, the rank of the `tf.Tensor` is reduced by 1 for each entry in `axis`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axis` has no entries, all dimensions are reduced, and a `tf.Tensor` with
   * a single element is returned.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3]);
   *
   * x.mean().print();  // or tf.mean(a)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * const axis = 1;
   * x.mean(axis).print();  // or tf.mean(x, axis)
   * ```
   *
   * @param x The input tensor.
   * @param axis The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims If true, retains reduced dimensions with size 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function mean_(x, axis = null, keepDims = false) {
      const $x = convertToTensor(x, 'x', 'mean');
      const axes = parseAxisParam(axis, $x.shape);
      const shapes = computeOutAndReduceShapes($x.shape, axes);
      const reduceShape = shapes[1];
      const reduceSize = sizeFromShape(reduceShape);
      // Use a custom gradient to bypass 2 gradient backprops since mean is used
      // extremely often.
      const customOp = customGrad((x) => {
          const reduceSizeScalar = scalar(reduceSize);
          // Cast if needed.
          const xReduce = reduceSizeScalar.dtype === x.dtype ? x : x.cast(reduceSizeScalar.dtype);
          const res = xReduce.div(reduceSizeScalar);
          const value = res.sum(axis, keepDims);
          const gradFunc = (dy) => {
              const expandedDyShape = x.shape.slice();
              axes.forEach(axis => {
                  expandedDyShape[axis] = 1;
              });
              const expandedDy = dy.reshape(expandedDyShape);
              const derX = expandedDy.mul(ones$1(x.shape, 'float32')).div(reduceSize);
              return derX;
          };
          return { value, gradFunc };
      });
      return customOp($x);
  }
  /**
   * Gradient helper function for the min and max operations.
   */
  function gradForMinAndMax(dy, y, xOrig, origAxes, permutedAxes) {
      if (y.rank < xOrig.rank) {
          y = y.reshape(expandShapeToKeepDim(y.shape, origAxes));
      }
      if (dy.rank < xOrig.rank) {
          dy = dy.reshape(expandShapeToKeepDim(dy.shape, origAxes));
      }
      return {
          x: () => {
              const dx = dy.mul(xOrig.equal(y).cast(dy.dtype));
              return permutedAxes == null ? dx : dx.transpose(permutedAxes);
          }
      };
  }
  /**
   * Computes the minimum value from the input.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axes` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3]);
   *
   * x.min().print();  // or tf.min(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * const axis = 1;
   * x.min(axis).print();  // or tf.min(x, axis)
   * ```
   *
   * @param x The input Tensor.
   * @param axis The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims If true, retains reduced dimensions with size 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function min_(x, axis = null, keepDims = false) {
      let $x = convertToTensor(x, 'x', 'min');
      const xOrig = $x;
      const origAxes = parseAxisParam(axis, $x.shape);
      let axes = origAxes;
      const permutedAxes = getAxesPermutation(axes, $x.rank);
      if (permutedAxes != null) {
          $x = $x.transpose(permutedAxes);
          axes = getInnerMostAxes(axes.length, $x.rank);
      }
      const grad = (dy, saved) => gradForMinAndMax(dy, saved[1], saved[0], origAxes, permutedAxes);
      const inputsToSave = [$x];
      const outputsToSave = [true];
      let res = ENGINE.runKernelFunc((backend, save) => {
          const y = backend.min($x, axes);
          save([xOrig, y]);
          return y;
      }, { x: $x }, grad, 'Min', { axes }, inputsToSave, outputsToSave);
      if (keepDims) {
          const newShape = expandShapeToKeepDim(res.shape, origAxes);
          res = res.reshape(newShape);
      }
      return res;
  }
  /**
   * Computes the maximum of elements across dimensions of a `tf.Tensor`.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
   * `axes`. If `keepDims` is true, the reduced dimensions are retained with
   * length 1. If `axes` has no entries, all dimensions are reduced, and an
   * `tf.Tensor` with a single element is returned.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3]);
   *
   * x.max().print();  // or tf.max(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * const axis = 1;
   * x.max(axis).print();  // or tf.max(x, axis)
   * ```
   *
   * @param x The input tensor.
   * @param axis The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims If true, retains reduced dimensions with size 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function max_(x, axis = null, keepDims = false) {
      let $x = convertToTensor(x, 'x', 'max');
      const xOrig = $x;
      const origAxes = parseAxisParam(axis, $x.shape);
      let axes = origAxes;
      const permutedAxes = getAxesPermutation(axes, $x.rank);
      if (permutedAxes != null) {
          $x = $x.transpose(permutedAxes);
          axes = getInnerMostAxes(axes.length, $x.rank);
      }
      const grad = (dy, saved) => gradForMinAndMax(dy, saved[1], saved[0], origAxes, permutedAxes);
      const inputsToSave = [$x];
      const outputsToSave = [true];
      let res = ENGINE.runKernelFunc((backend, save) => {
          const y = backend.max($x, axes);
          save([xOrig, y]);
          return y;
      }, { x: $x }, grad, 'Max', { axes }, inputsToSave, outputsToSave);
      if (keepDims) {
          const newShape = expandShapeToKeepDim(res.shape, origAxes);
          res = res.reshape(newShape);
      }
      return res;
  }
  /**
   * Returns the indices of the minimum values along an `axis`.
   *
   * The result has the same shape as `input` with the dimension along `axis`
   * removed.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3]);
   *
   * x.argMin().print();  // or tf.argMin(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 4, 3], [2, 2]);
   *
   * const axis = 1;
   * x.argMin(axis).print();  // or tf.argMin(x, axis)
   * ```
   *
   * @param x The input tensor.
   * @param axis The dimension to reduce. Defaults to 0 (outer-most dimension).
   *
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function argMin_(x, axis = 0) {
      let $x = convertToTensor(x, 'x', 'argMin');
      if (axis == null) {
          axis = 0;
      }
      let axes = parseAxisParam(axis, $x.shape);
      const permutedAxes = getAxesPermutation(axes, $x.rank);
      if (permutedAxes != null) {
          $x = $x.transpose(permutedAxes);
          axes = getInnerMostAxes(axes.length, $x.rank);
      }
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => zerosLike($x) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.argMin($x, axes[0]);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Returns the indices of the maximum values along an `axis`.
   *
   * The result has the same shape as `input` with the dimension along `axis`
   * removed.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3]);
   *
   * x.argMax().print();  // or tf.argMax(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 4, 3], [2, 2]);
   *
   * const axis = 1;
   * x.argMax(axis).print();  // or tf.argMax(x, axis)
   * ```
   *
   * @param x The input tensor.
   * @param axis The dimension to reduce. Defaults to 0 (outer-most dimension).
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function argMax_(x, axis = 0) {
      let $x = convertToTensor(x, 'x', 'argMax');
      if (axis == null) {
          axis = 0;
      }
      let axes = parseAxisParam(axis, $x.shape);
      const permutedAxes = getAxesPermutation(axes, $x.rank);
      if (permutedAxes != null) {
          $x = $x.transpose(permutedAxes);
          axes = getInnerMostAxes(axes.length, $x.rank);
      }
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { x: () => zerosLike($x) };
      };
      const attrs = { axis: axes[0] };
      const inputsToSave = [$x];
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.argMax($x, axes[0]);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'ArgMax', attrs, inputsToSave);
  }
  /**
   * Computes the logical and of elements across dimensions of a `tf.Tensor`.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
   * `axes`. If `keepDims` is true, the reduced dimensions are retained with
   * length 1. If `axes` has no entries, all dimensions are reduced, and an
   * `tf.Tensor` with a single element is returned.
   *
   * ```js
   * const x = tf.tensor1d([1, 1, 1], 'bool');
   *
   * x.all().print();  // or tf.all(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
   *
   * const axis = 1;
   * x.all(axis).print();  // or tf.all(x, axis)
   * ```
   *
   * @param x The input tensor. Must be of dtype bool.
   * @param axis The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims If true, retains reduced dimensions with size 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function all_(x, axis = null, keepDims = false) {
      let $x = convertToTensor(x, 'x', 'all', 'bool');
      const origAxes = parseAxisParam(axis, $x.shape);
      let axes = origAxes;
      const permutedAxes = getAxesPermutation(axes, $x.rank);
      if (permutedAxes != null) {
          $x = $x.transpose(permutedAxes);
          axes = getInnerMostAxes(axes.length, $x.rank);
      }
      const res = ENGINE.runKernelFunc(backend => backend.all($x, axes), { $x });
      if (keepDims) {
          const newShape = expandShapeToKeepDim(res.shape, origAxes);
          return res.reshape(newShape);
      }
      return res;
  }
  /**
   * Computes the logical or of elements across dimensions of a `tf.Tensor`.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
   * `axes`. If `keepDims` is true, the reduced dimensions are retained with
   * length 1. If `axes` has no entries, all dimensions are reduced, and an
   * `tf.Tensor` with a single element is returned.
   *
   * ```js
   * const x = tf.tensor1d([1, 1, 1], 'bool');
   *
   * x.any().print();  // or tf.any(x)
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
   *
   * const axis = 1;
   * x.any(axis).print();  // or tf.any(x, axis)
   * ```
   *
   * @param x The input tensor. Must be of dtype bool.
   * @param axis The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims If true, retains reduced dimensions with size 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Reduction'} */
  function any_(x, axis = null, keepDims = false) {
      let $x = convertToTensor(x, 'x', 'any', 'bool');
      const origAxes = parseAxisParam(axis, $x.shape);
      let axes = origAxes;
      const permutedAxes = getAxesPermutation(axes, $x.rank);
      if (permutedAxes != null) {
          $x = $x.transpose(permutedAxes);
          axes = getInnerMostAxes(axes.length, $x.rank);
      }
      const res = ENGINE.runKernelFunc(backend => backend.any($x, axes), { $x });
      if (keepDims) {
          const newShape = expandShapeToKeepDim(res.shape, origAxes);
          return res.reshape(newShape);
      }
      return res;
  }
  /**
   * Calculates the mean and variance of `x`. The mean and variance are
   * calculated by aggregating the contents of `x` across `axes`. If `x` is
   * 1-D and `axes = [0]` this is just the mean and variance of a vector.
   *
   * @param x The input tensor.
   * @param axis The dimension(s) along with to compute mean and
   *     variance. By default it reduces all dimensions.
   * @param keepDims If true, the moments have the same dimensionality as the
   *     input.
   * @return An object with two keys: `mean` and `variance`.
   */
  /** @doc {heading: 'Operations', subheading: 'Normalization'} */
  function moments_(x, axis = null, keepDims = false) {
      x = convertToTensor(x, 'x', 'moments');
      const axes = parseAxisParam(axis, x.shape);
      const mean = x.mean(axes, keepDims);
      let keepDimsShape = mean.shape;
      if (!keepDims) {
          keepDimsShape = expandShapeToKeepDim(mean.shape, axes);
      }
      const devSquared = x.toFloat().sub(mean.reshape(keepDimsShape)).square();
      const variance = devSquared.mean(axes, keepDims);
      return { mean, variance };
  }
  const all = op({ all_ });
  // tslint:disable-next-line:variable-name
  const any = op({ any_ });
  const argMax = op({ argMax_ });
  const argMin = op({ argMin_ });
  const logSumExp = op({ logSumExp_ });
  const max = op({ max_ });
  const mean = op({ mean_ });
  const min = op({ min_ });
  const moments = op({ moments_ });
  const sum$1 = op({ sum_ });
  const prod = op({ prod_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const broadcastToGradConfig = {
      kernelName: BroadcastTo,
      gradFunc: (dy, saved, attrs) => {
          const broadCastToAttrs = attrs;
          const inputShape = broadCastToAttrs.inputShape;
          const outputShape = broadCastToAttrs.shape;
          const reps = Array.from(outputShape);
          for (let i = inputShape.length - 1; i >= 0; i--) {
              if (inputShape[i] === outputShape[i]) {
                  reps[i] = 1;
              }
              else if (inputShape[i] !== 1) {
                  throw new Error(`broadcastTo(): [${inputShape}] cannot be broadcast to [${outputShape}].`);
              }
          }
          const axes = [];
          for (let i = 0; i < reps.length; i++) {
              if (reps[i] > 1) {
                  axes.push(i);
              }
          }
          return { x: () => sum$1(dy, axes, true /* keepDims */) };
      }
  };

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
  function assertParamsConsistent(shapes, axis) {
      const rank = shapes[0].length;
      shapes.forEach((shape, i) => {
          assert(shape.length === rank, () => `Error in concat${rank}D: rank of tensors[${i}] must be the same ` +
              `as the rank of the rest (${rank})`);
      });
      assert(axis >= 0 && axis < rank, () => `Error in concat${rank}D: axis must be between 0 and ${rank - 1}.`);
      const firstShape = shapes[0];
      shapes.forEach((shape, i) => {
          for (let r = 0; r < rank; r++) {
              assert((r === axis) || (shape[r] === firstShape[r]), () => `Error in concat${rank}D: Shape of tensors[${i}] (${shape}) ` +
                  `does not match the shape of the rest (${firstShape}) ` +
                  `along the non-concatenated axis ${i}.`);
          }
      });
  }
  function computeOutShape(shapes, axis) {
      const outputShape = shapes[0].slice();
      for (let i = 1; i < shapes.length; i++) {
          outputShape[axis] += shapes[i][axis];
      }
      return outputShape;
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Concatenates a list of`tf.Tensor1D`s along an axis. See `concat` for details.
   *
   * For example, if:
   * A: shape(3) = |r1, g1, b1|
   * B: shape(2) = |r2, g2|
   * C = tf.concat1d([A, B]) == |r1, g1, b1, r2, g2|
   *
   * @param tensors A list of`tf.Tensor`s to concatenate.
   * @return The concatenated array.
   */
  function concat1d_(tensors) {
      return concat(tensors, 0 /* axis */);
  }
  /**
   * Concatenates a list of`tf.Tensor2D`s along an axis. See `concat` for details.
   *
   * For example, if:
   * A: shape(2, 3) = | r1, g1, b1 |
   *                  | r2, g2, b2 |
   *
   * B: shape(2, 3) = | r3, g3, b3 |
   *                  | r4, g4, b4 |
   *
   * C = tf.concat2d([A, B], axis)
   *
   * if axis = 0:
   * C: shape(4, 3) = | r1, g1, b1 |
   *                  | r2, g2, b2 |
   *                  | r3, g3, b3 |
   *                  | r4, g4, b4 |
   *
   * if axis = 1:
   * C = shape(2, 6) = | r1, g1, b1, r3, g3, b3 |
   *                   | r2, g2, b2, r4, g4, b4 |
   *
   *
   * @param tensors A list of `tf.Tensor`s to concatenate.
   * @param axis The axis to concatenate along.
   * @return The concatenated array.
   */
  function concat2d_(tensors, axis) {
      return concat(tensors, axis);
  }
  /**
   * Concatenates a list of `tf.Tensor3D`s along an axis.
   * See `concat` for details.
   *
   * For example, if:
   * A: shape(2, 1, 3) = | r1, g1, b1 |
   *                     | r2, g2, b2 |
   *
   * B: shape(2, 1, 3) = | r3, g3, b3 |
   *                     | r4, g4, b4 |
   *
   * C = tf.concat3d([A, B], axis)
   *
   * if axis = 0:
   * C: shape(4, 1, 3) = | r1, g1, b1 |
   *                     | r2, g2, b2 |
   *                     | r3, g3, b3 |
   *                     | r4, g4, b4 |
   *
   * if axis = 1:
   * C: shape(2, 2, 3) = | r1, g1, b1, r3, g3, b3 |
   *                     | r2, g2, b2, r4, g4, b4 |
   *
   * if axis = 2:
   * C = shape(2, 1, 6) = | r1, g1, b1, r3, g3, b3 |
   *                      | r2, g2, b2, r4, g4, b4 |
   *
   * @param tensors A list of`tf.Tensor`s to concatenate.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  function concat3d_(tensors, axis) {
      return concat(tensors, axis);
  }
  /**
   * Concatenates a list of `tf.Tensor4D`s along an axis.
   * See `concat` for details.
   *
   * @param tensors A list of `tf.Tensor`s to concatenate.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  function concat4d_(tensors, axis) {
      return concat(tensors, axis);
  }
  /**
   * Concatenates a list of `tf.Tensor`s along a given axis.
   *
   * The tensors ranks and types must match, and their sizes must match in all
   * dimensions except `axis`.
   *
   * Also available are stricter rank-specific methods that assert that
   * `tensors` are of the given rank:
   *   - `tf.concat1d`
   *   - `tf.concat2d`
   *   - `tf.concat3d`
   *   - `tf.concat4d`
   *
   * Except `tf.concat1d` (which does not have axis param), all methods have
   * same signature as this method.
   *
   * ```js
   * const a = tf.tensor1d([1, 2]);
   * const b = tf.tensor1d([3, 4]);
   * a.concat(b).print();  // or a.concat(b)
   * ```
   *
   * ```js
   * const a = tf.tensor1d([1, 2]);
   * const b = tf.tensor1d([3, 4]);
   * const c = tf.tensor1d([5, 6]);
   * tf.concat([a, b, c]).print();
   * ```
   *
   * ```js
   * const a = tf.tensor2d([[1, 2], [10, 20]]);
   * const b = tf.tensor2d([[3, 4], [30, 40]]);
   * const axis = 1;
   * tf.concat([a, b], axis).print();
   * ```
   * @param tensors A list of tensors to concatenate.
   * @param axis The axis to concate along. Defaults to 0 (the first dim).
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  function concat_(tensors, axis = 0) {
      assert(tensors.length >= 1, () => 'Pass at least one tensor to concat');
      let $tensors = convertToTensorArray(tensors, 'tensors', 'concat');
      if ($tensors[0].dtype === 'complex64') {
          $tensors.forEach(tensor => {
              if (tensor.dtype !== 'complex64') {
                  throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${tensor.dtype}. `);
              }
          });
      }
      axis = parseAxisParam(axis, $tensors[0].shape)[0];
      const outShape = computeOutShape($tensors.map(t => t.shape), axis);
      if (sizeFromShape(outShape) === 0) {
          return tensor([], outShape);
      }
      // Keep only non-empty tensors (ignore tensors with 0 in their shape).
      $tensors = $tensors.filter(t => t.size > 0);
      if ($tensors.length === 1) {
          return $tensors[0];
      }
      const shapes = $tensors.map(t => t.shape);
      assertParamsConsistent(shapes, axis);
      const der = (dy) => {
          const sizeSplits = shapes.map(s => s[axis]);
          const derTensors = split(dy, sizeSplits, axis);
          return derTensors.map(t => () => t);
      };
      const inputs = $tensors;
      const attr = { axis };
      return ENGINE.runKernelFunc(backend => backend.concat($tensors, axis), inputs, der, 'Concat', attr);
  }
  /**
   * Splits a `tf.Tensor` into sub tensors.
   *
   * If `numOrSizeSplits` is a number, splits `x` along dimension `axis`
   * into `numOrSizeSplits` smaller tensors.
   * Requires that `numOrSizeSplits` evenly divides `x.shape[axis]`.
   *
   * If `numOrSizeSplits` is a number array, splits `x` into
   * `numOrSizeSplits.length` pieces. The shape of the `i`-th piece has the
   * same size as `x` except along dimension `axis` where the size is
   * `numOrSizeSplits[i]`.
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
   * const [a, b] = tf.split(x, 2, 1);
   * a.print();
   * b.print();
   *
   * const [c, d, e] = tf.split(x, [1, 2, 1], 1);
   * c.print();
   * d.print();
   * e.print();
   * ```
   *
   * @param x The input tensor to split.
   * @param numOrSizeSplits Either an integer indicating the number of
   * splits along the axis or an array of integers containing the sizes of
   * each output tensor along the axis. If a number then it must evenly divide
   * `x.shape[axis]`; otherwise the sum of sizes must match `x.shape[axis]`.
   * @param axis The dimension along which to split. Defaults to 0 (the first
   * dim).
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  function split_(x, numOrSizeSplits, axis = 0) {
      const $x = convertToTensor(x, 'x', 'split');
      axis = parseAxisParam(axis, $x.shape)[0];
      let splitSizes;
      if (typeof (numOrSizeSplits) === 'number') {
          assert($x.shape[axis] % numOrSizeSplits === 0, () => 'Number of splits must evenly divide the axis.');
          splitSizes =
              new Array(numOrSizeSplits).fill($x.shape[axis] / numOrSizeSplits);
      }
      else {
          assert($x.shape[axis] === numOrSizeSplits.reduce((a, b) => a + b), () => 'The sum of sizes must match the size of the axis dimension.');
          splitSizes = numOrSizeSplits;
      }
      const der = (dy) => ({ $x: () => concat(dy, axis) });
      return ENGINE.runKernelFunc(backend => backend.split($x, splitSizes, axis), { x: $x }, der, 'SplitV', {numOrSizeSplits, axis});
  }
  const concat = op({ concat_ });
  const concat1d = op({ concat1d_ });
  const concat2d = op({ concat2d_ });
  const concat3d = op({ concat3d_ });
  const concat4d = op({ concat4d_ });
  const split = op({ split_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Reshapes a `tf.Tensor` to a given shape.
   *
   * Given an input tensor, returns a new tensor with the same values as the
   * input tensor with shape `shape`.
   *
   * If one component of shape is the special value -1, the size of that
   * dimension is computed so that the total size remains constant. In
   * particular, a shape of [-1] flattens into 1-D. At most one component of
   * shape can be -1.
   *
   * If shape is 1-D or higher, then the operation returns a tensor with shape
   * shape filled with the values of tensor. In this case, the number of
   * elements implied by shape must be the same as the number of elements in
   * tensor.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * x.reshape([2, 2]).print();
   * ```
   *
   * @param x The input tensor to be reshaped.
   * @param shape An array of integers defining the output tensor shape.
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function reshape_(x, shape) {
      const $x = convertToTensor(x, 'x', 'reshape', null);
      shape = inferFromImplicitShape(shape, $x.size);
      assert($x.size === sizeFromShape(shape), () => 'new shape and old shape must have the same number of elements.');
      const grad = (dy) => {
          return { x: () => dy.reshape($x.shape) };
      };
      const attrs = { shape };
      return ENGINE.runKernelFunc(backend => backend.reshape($x, shape), { x: $x }, grad, 'Reshape', attrs);
  }
  /**
   * Removes dimensions of size 1 from the shape of a `tf.Tensor`.
   *
   * ```js
   * const x = tf.tensor([1, 2, 3, 4], [1, 1, 4]);
   * x.squeeze().print();
   * ```
   *
   * @param x The input tensor to be squeezed.
   * @param axis An optional list of numbers. If specified, only
   *     squeezes the dimensions listed. The dimension index starts at 0. It
   * is an error to squeeze a dimension that is not 1.
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function squeeze_(x, axis) {
      const $x = convertToTensor(x, 'x', 'squeeze');
      return reshape($x, squeezeShape($x.shape, axis).newShape);
  }
  /**
   * Casts a `tf.Tensor` to a new dtype.
   *
   * ```js
   * const x = tf.tensor1d([1.5, 2.5, 3]);
   * tf.cast(x, 'int32').print();
   * ```
   * @param x The input tensor to be casted.
   * @param dtype The dtype to cast the input tensor to.
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function cast_(x, dtype) {
      const $x = convertToTensor(x, 'x', 'cast');
      // Sanity checks.
      if (!isValidDtype(dtype)) {
          throw new Error(`Failed to cast to unknown dtype ${dtype}`);
      }
      if (dtype === 'string' && $x.dtype !== 'string' ||
          dtype !== 'string' && $x.dtype === 'string') {
          throw new Error('Only strings can be casted to strings');
      }
      const grad = (dy) => {
          return { x: () => dy.clone() };
      };
      const attrs = { dtype };
      return ENGINE.runKernelFunc(backend => backend.cast($x, dtype), { x: $x }, grad, 'Cast', attrs);
  }
  /**
   * Stacks a list of rank-`R` `tf.Tensor`s into one rank-`(R+1)` `tf.Tensor`.
   *
   * ```js
   * const a = tf.tensor1d([1, 2]);
   * const b = tf.tensor1d([3, 4]);
   * const c = tf.tensor1d([5, 6]);
   * tf.stack([a, b, c]).print();
   * ```
   *
   * @param tensors A list of tensor objects with the same shape and dtype.
   * @param axis The axis to stack along. Defaults to 0 (the first dim).
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  function stack_(tensors, axis = 0) {
      const $tensors = convertToTensorArray(tensors, 'tensors', 'stack');
      assert($tensors.length >= 1, () => 'Pass at least one tensor to tf.stack');
      if ($tensors.length === 1) {
          return $tensors[0].expandDims(axis);
      }
      const rank = $tensors[0].rank;
      const shape = $tensors[0].shape;
      const dtype = $tensors[0].dtype;
      assert(axis <= rank, () => 'Axis must be <= rank of the tensor');
      $tensors.forEach(t => {
          assertShapesMatch(shape, t.shape, 'All tensors passed to stack must have matching shapes');
      });
      $tensors.forEach(t => {
          assert(dtype === t.dtype, () => 'All tensors passed to stack must have matching dtypes');
      });
      const expandedTensors = $tensors.map(t => t.expandDims(axis));
      return concat(expandedTensors, axis);
  }
  /**
   * This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of
   * shape `blockShape + [batch]`, interleaves these blocks back into the grid
   * defined by the spatial dimensions `[1, ..., M]`, to obtain a result with
   * the same rank as the input. The spatial dimensions of this intermediate
   * result are then optionally cropped according to `crops` to produce the
   * output. This is the reverse of `tf.spaceToBatchND`. See below for a precise
   * description.
   *
   * ```js
   * const x = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
   * const blockShape = [2, 2];
   * const crops = [[0, 0], [0, 0]];
   *
   * x.batchToSpaceND(blockShape, crops).print();
   * ```
   *
   * @param x A `tf.Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
   * remainingShape`, where spatialShape has `M` dimensions.
   * @param blockShape A 1-D array. Must have shape `[M]`, all values must
   * be >= 1.
   * @param crops A 2-D array.  Must have shape `[M, 2]`, all values must be >= 0.
   * `crops[i] = [cropStart, cropEnd]` specifies the amount to crop from input
   * dimension `i + 1`, which corresponds to spatial dimension `i`. It is required
   * that `cropStart[i] + cropEnd[i] <= blockShape[i] * inputShape[i + 1]`
   *
   * This operation is equivalent to the following steps:
   *
   * 1. Reshape `x` to `reshaped` of shape: `[blockShape[0], ...,
   * blockShape[M-1], batch / prod(blockShape), x.shape[1], ...,
   * x.shape[N-1]]`
   *
   * 2. Permute dimensions of `reshaped`to produce `permuted` of shape `[batch /
   * prod(blockShape),x.shape[1], blockShape[0], ..., x.shape[M],
   * blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
   *
   * 3. Reshape `permuted` to produce `reshapedPermuted` of shape `[batch /
   * prod(blockShape),x.shape[1] * blockShape[0], ..., x.shape[M] *
   * blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
   *
   * 4. Crop the start and end of dimensions `[1, ..., M]` of `reshapedPermuted`
   * according to `crops` to produce the output of shape: `[batch /
   * prod(blockShape),x.shape[1] * blockShape[0] - crops[0,0] - crops[0,1],
   * ..., x.shape[M] * blockShape[M-1] - crops[M-1,0] -
   * crops[M-1,1],x.shape[M+1], ..., x.shape[N-1]]`
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function batchToSpaceND_(x, blockShape, crops) {
      const $x = convertToTensor(x, 'x', 'batchToSpaceND');
      const prod = blockShape.reduce((a, b) => a * b);
      assert($x.rank >= 1 + blockShape.length, () => `input rank is ${$x.rank} but should be > than blockShape.length ${blockShape.length}`);
      assert(crops.length === blockShape.length, () => `crops.length is ${crops.length} but should be equal to blockShape.length  ${blockShape.length}`);
      assert($x.shape[0] % prod === 0, () => `input tensor batch is ${$x.shape[0]} but is not divisible by the product of ` +
          `the elements of blockShape ${blockShape.join(' * ')} === ${prod}`);
      const grad = (dy) => {
          return { $x: () => dy.spaceToBatchND(blockShape, crops) };
      };
      return ENGINE.runKernelFunc(backend => backend.batchToSpaceND($x, blockShape, crops), { $x }, grad);
  }
  /**
   * This operation divides "spatial" dimensions `[1, ..., M]` of the input into
   * a grid of blocks of shape `blockShape`, and interleaves these blocks with
   * the "batch" dimension (0) such that in the output, the spatial
   * dimensions `[1, ..., M]` correspond to the position within the grid,
   * and the batch dimension combines both the position within a spatial block
   * and the original batch position. Prior to division into blocks,
   * the spatial dimensions of the input are optionally zero padded
   * according to `paddings`. See below for a precise description.
   *
   * ```js
   * const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
   * const blockShape = [2, 2];
   * const paddings = [[0, 0], [0, 0]];
   *
   * x.spaceToBatchND(blockShape, paddings).print();
   * ```
   *
   * @param x A `tf.Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
   * remainingShape`, where spatialShape has `M` dimensions.
   * @param blockShape A 1-D array. Must have shape `[M]`, all values must
   * be >= 1.
   * @param paddings A 2-D array. Must have shape `[M, 2]`, all values must be >=
   *     0. `paddings[i] = [padStart, padEnd]` specifies the amount to zero-pad
   * from input dimension `i + 1`, which corresponds to spatial dimension `i`. It
   * is required that
   * `(inputShape[i + 1] + padStart + padEnd) % blockShape[i] === 0`
   *
   * This operation is equivalent to the following steps:
   *
   * 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the input
   * according to `paddings` to produce `padded` of shape paddedShape.
   *
   * 2. Reshape `padded` to `reshapedPadded` of shape:
   * `[batch] + [paddedShape[1] / blockShape[0], blockShape[0], ...,
   * paddedShape[M] / blockShape[M-1], blockShape[M-1]] + remainingShape`
   *
   * 3. Permute dimensions of `reshapedPadded` to produce `permutedReshapedPadded`
   * of shape: `blockShape + [batch] + [paddedShape[1] / blockShape[0], ...,
   * paddedShape[M] / blockShape[M-1]] + remainingShape`
   *
   * 4. Reshape `permutedReshapedPadded` to flatten `blockShape` into the
   * batch dimension, producing an output tensor of shape:
   * `[batch * prod(blockShape)] + [paddedShape[1] / blockShape[0], ...,
   * paddedShape[M] / blockShape[M-1]] + remainingShape`
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function spaceToBatchND_(x, blockShape, paddings) {
      const $x = convertToTensor(x, 'x', 'spaceToBatchND');
      assert($x.rank >= 1 + blockShape.length, () => `input rank ${$x.rank} should be > than [blockShape] ${blockShape.length}`);
      assert(paddings.length === blockShape.length, () => `paddings.shape[0] ${paddings.length} must be equal to [blockShape] ${blockShape.length}`);
      assert($x.shape.reduce((a, b, i) => {
          if (i > 0 && i <= blockShape.length) {
              return a &&
                  ((b + paddings[i - 1][0] + paddings[i - 1][1]) %
                      blockShape[i - 1] ===
                      0);
          }
          return a;
      }, true), () => `input spatial dimensions ${$x.shape.slice(1)} with paddings ${paddings.toString()} must be divisible by blockShapes ${blockShape.toString()}`);
      const grad = (dy) => {
          return { $x: () => dy.batchToSpaceND(blockShape, paddings) };
      };
      return ENGINE.runKernelFunc(backend => backend.spaceToBatchND($x, blockShape, paddings), { $x }, grad);
  }
  /**
   * Unstacks a `tf.Tensor` of rank-`R` into a list of rank-`(R-1)` `tf.Tensor`s.
   *
   * ```js
   * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * tf.unstack(a).forEach(tensor => tensor.print());
   * ```
   *
   * @param x A tensor object.
   * @param axis The axis to unstack along. Defaults to 0 (the first dim).
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  function unstack_(x, axis = 0) {
      axis = axis || 0;
      const $x = convertToTensor(x, 'x', 'unstack');
      assert(axis >= -$x.shape.length && axis < $x.shape.length, () => `Axis = ${axis} is not in [-${$x.shape.length}, ${$x.shape.length})`);
      if (axis < 0) {
          axis += $x.shape.length;
      }
      const grad = (dy) => {
          return { x: () => stack(dy, axis) };
      };
      const attrs = { axis };
      return ENGINE.runKernelFunc(backend => backend.unstack($x, axis), { x: $x }, grad, 'Unpack', attrs);
  }
  /**
   * Computes the cumulative sum of a `tf.Tensor` along `axis`.
   *
   * ```js
   * const x = tf.tensor([1, 2, 3, 4]);
   * x.cumsum().print();
   * ```
   * ```js
   * const x = tf.tensor([[1, 2], [3, 4]]);
   * x.cumsum().print();
   * ```
   *
   * @param x The input tensor to be summed.
   * @param axis The axis along which to sum. Optional. Defaults to 0.
   * @param exclusive Whether to perform exclusive cumulative sum. Optional.
   *     Defaults to false. If set to true then the sum of each tensor entry
   *     does not include its own value, but only the values previous to it
   *     along the specified axis.
   * @param reverse Whether to sum in the opposite direction. Optional.
   *     Defaults to false.
   */
  /** @doc {heading: 'Operations', subheading: 'Scan'} */
  function cumsum_(x, axis = 0, exclusive = false, reverse = false) {
      const $x = convertToTensor(x, 'x', 'cumsum');
      axis = axis | 0;
      const permutation = getAxesPermutation([axis], $x.rank);
      let permutedX = $x;
      if (permutation != null) {
          permutedX = $x.transpose(permutation);
      }
      const permutedAxis = getInnerMostAxes(1, $x.rank)[0];
      const grad = (dy) => {
          return { permutedX: () => dy.cumsum(axis, exclusive, !reverse) };
      };
      let value = ENGINE.runKernelFunc(backend => backend.cumsum(permutedX, permutedAxis, exclusive, reverse), { permutedX }, grad);
      if (permutation != null) {
          value = value.transpose(permutation);
      }
      return value;
  }
  /**
   * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
   * into the tensor's shape.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * const axis = 1;
   * x.expandDims(axis).print();
   * ```
   *
   * @param x The input tensor whose dimensions to be expanded.
   * @param axis The dimension index at which to insert shape of `1`. Defaults
   *     to 0 (the first dimension).
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function expandDims_(x, axis = 0) {
      const parseAs = null;
      const $x = convertToTensor(x, 'x', 'expandDims', parseAs);
      assert(axis <= $x.rank, () => 'Axis must be <= rank of the tensor');
      const newShape = $x.shape.slice();
      if (axis < 0) {
          // Negative value is counted from the tail of rank.
          assert(-($x.rank + 1) <= axis, () => `Axis must be in the interval [${-($x.rank + 1)}, ${$x.rank}]`);
          axis = $x.rank + axis + 1;
      }
      newShape.splice(axis, 0, 1);
      return reshape($x, newShape);
  }
  /**
   * Rearranges data from depth into blocks of spatial data. More specifically,
   * this op outputs a copy of the input tensor where values from the `depth`
   * dimension are moved in spatial blocks to the `height` and `width` dimensions.
   * The attr `blockSize` indicates the input block size and how the data is
   * moved.
   *
   *  - Chunks of data of size `blockSize * blockSize` from depth are rearranged
   * into non-overlapping blocks of size `blockSize x blockSize`
   *
   *  - The width the output tensor is `inputWidth * blockSize`, whereas the
   * height is `inputHeight * blockSize`
   *
   *  - The Y, X coordinates within each block of the output image are determined
   * by the high order component of the input channel index
   *
   *  - The depth of the input tensor must be divisible by `blockSize *
   * blockSize`
   *
   * The `dataFormat` attr specifies the layout of the input and output tensors
   * with the following options: "NHWC": [ `batch, height, width, channels` ]
   * "NCHW": [ `batch, channels, height, width` ]
   *
   * ```js
   * const x = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
   * const blockSize = 2;
   * const dataFormat = "NHWC";
   *
   * tf.depthToSpace(x, blockSize, dataFormat).print();
   * ```
   *
   * @param x The input tensor of rank 4
   * @param blockSIze  An `int` that is `>= 2`. The size of the spatial block
   * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to "NHWC"
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function depthToSpace_(x, blockSize, dataFormat = 'NHWC') {
      const $x = convertToTensor(x, 'x', 'depthToSpace');
      const inputHeight = (dataFormat === 'NHWC') ? $x.shape[1] : $x.shape[2];
      const inputWidth = (dataFormat === 'NHWC') ? $x.shape[2] : $x.shape[3];
      const inputDepth = (dataFormat === 'NHWC') ? $x.shape[3] : $x.shape[1];
      assert(inputHeight * blockSize >= 0, () => `Negative dimension size caused by overflow when multiplying
      ${inputHeight} and ${blockSize}  for depthToSpace with input shape
      ${$x.shape}`);
      assert(inputWidth * blockSize >= 0, () => `Negative dimension size caused by overflow when multiplying
      ${inputWidth} and ${blockSize} for depthToSpace with input shape
          ${$x.shape}`);
      assert((inputDepth % (blockSize * blockSize) === 0), () => `Dimension size must be evenly divisible by ${blockSize * blockSize} but is ${inputDepth} for depthToSpace with input shape ${$x.shape}`);
      return ENGINE.runKernelFunc(backend => backend.depthToSpace($x, blockSize, dataFormat), { $x });
  }
  /**
   * Computes the difference between two lists of numbers.
   *
   * Given a Tensor `x` and a Tensor `y`, this operation returns a Tensor `out`
   * that represents all values that are in `x` but not in `y`. The returned
   * Tensor `out` is sorted in the same order that the numbers appear in `x`
   * (duplicates are preserved). This operation also returns a Tensor indices that
   * represents the position of each out element in `x`. In other words:
   *
   * `out[i] = x[idx[i]] for i in [0, 1, ..., out.length - 1]`
   *
   * ```js
   * const x = [1, 2, 3, 4, 5, 6];
   * const y = [1, 3, 5];
   *
   * const [out, indices] = await tf.setdiff1dAsync(x, y);
   * out.print(); // [2, 4, 6]
   * indices.print(); // [1, 3, 5]
   * ```
   *
   * @param x 1-D Tensor. Values to keep.
   * @param y 1-D Tensor. Must have the same type as x. Values to exclude in the
   *     output.
   * @returns Promise of Tensor tuple [out, indices].
   *  out: Tensor with the same type as x.
   *  indices: A Tensor of type int32.
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  async function setdiff1dAsync_(x, y) {
      const $x = convertToTensor(x, 'x', 'setdiff1d');
      const $y = convertToTensor(y, 'y', 'setdiff1d');
      assert($x.dtype === $y.dtype, () => `x and y should have the same dtype, but got x (${$x.dtype}) and y (${$y.dtype}).`);
      assert($x.rank === 1, () => `x should be 1D tensor, but got x (${$x.shape}).`);
      assert($y.rank === 1, () => `y should be 1D tensor, but got y (${$y.shape}).`);
      const xVals = await $x.data();
      const yVals = await $y.data();
      const ySet = new Set(yVals);
      let outputSize = 0;
      for (let i = 0; i < xVals.length; i++) {
          if (!ySet.has(xVals[i])) {
              outputSize++;
          }
      }
      const buffer = new TensorBuffer([outputSize], $x.dtype);
      const indices = new TensorBuffer([outputSize], 'int32');
      for (let i = 0, p = 0; i < xVals.length; i++) {
          if (!ySet.has(xVals[i])) {
              buffer.values[p] = xVals[i];
              indices.values[p] = i;
              p++;
          }
      }
      return [buffer.toTensor(), indices.toTensor()];
  }
  /**
   * Creates an empty `tf.TensorBuffer` with the specified `shape` and `dtype`.
   *
   * The values are stored in CPU as `TypedArray`. Fill the buffer using
   * `buffer.set()`, or by modifying directly `buffer.values`.
   *
   * When done, call `buffer.toTensor()` to get an immutable `tf.Tensor` with
   * those values.
   *
   * ```js
   * // Create a buffer and set values at particular indices.
   * const buffer = tf.buffer([2, 2]);
   * buffer.set(3, 0, 0);
   * buffer.set(5, 1, 0);
   *
   * // Convert the buffer back to a tensor.
   * buffer.toTensor().print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param dtype The dtype of the buffer. Defaults to 'float32'.
   * @param values The values of the buffer as `TypedArray`. Defaults to
   * zeros.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function buffer(shape, dtype = 'float32', values) {
      dtype = dtype || 'float32';
      assertNonNegativeIntegerDimensions(shape);
      return new TensorBuffer(shape, dtype, values);
  }
  /**
   * Prints information about the `tf.Tensor` including its data.
   *
   * ```js
   * const verbose = true;
   * tf.tensor2d([1, 2, 3, 4], [2, 2]).print(verbose);
   * ```
   * @param x The tensor to be printed.
   * @param verbose Whether to print verbose information about the ` Tensor`,
   * including dtype and size.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function print(x, verbose = false) {
      console.log(x.toString(verbose));
  }
  const batchToSpaceND = op({ batchToSpaceND_ });
  const cast = op({ cast_ });
  const cumsum = op({ cumsum_ });
  const depthToSpace = op({ depthToSpace_ });
  const expandDims = op({ expandDims_ });
  const reshape = op({ reshape_ });
  const spaceToBatchND = op({ spaceToBatchND_ });
  const squeeze = op({ squeeze_ });
  const stack = op({ stack_ });
  const unstack = op({ unstack_ });
  const setdiff1dAsync = setdiff1dAsync_;

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Adds two `tf.Tensor`s element-wise, A + B. Supports broadcasting.
   *
   * We also expose `tf.addStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3, 4]);
   * const b = tf.tensor1d([10, 20, 30, 40]);
   *
   * a.add(b).print();  // or tf.add(a, b)
   * ```
   *
   * ```js
   * // Broadcast add a with b.
   * const a = tf.scalar(5);
   * const b = tf.tensor1d([10, 20, 30, 40]);
   *
   * a.add(b).print();  // or tf.add(a, b)
   * ```
   * @param a The first `tf.Tensor` to add.
   * @param b The second `tf.Tensor` to add. Must have the same type as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function add_(a, b) {
      let $a = convertToTensor(a, 'a', 'add');
      let $b = convertToTensor(b, 'b', 'add');
      [$a, $b] = makeTypesMatch($a, $b);
      const forward = (backend, save) => {
          const res = backend.add($a, $b);
          save([$a, $b]);
          return res;
      };
      const inputs = { a: $a, b: $b };
      return ENGINE.runKernelFunc(forward, inputs, null /* gradient */, Add);
  }
  const add = op({ add_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes `-1 * x` element-wise.
   *
   * ```js
   * const x = tf.tensor2d([1, 2, -2, 0], [2, 2]);
   *
   * x.neg().print();  // or tf.neg(x)
   * ```
   *
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function neg_(x) {
      const $x = convertToTensor(x, 'x', 'neg');
      const grad = (dy) => {
          return { x: () => dy.neg() };
      };
      const attrs = {};
      const inputsToSave = [$x];
      return ENGINE.runKernelFunc(backend => backend.neg($x), { x: $x }, grad, 'Neg', attrs, inputsToSave);
  }
  /**
   * Computes ceiling of input `tf.Tensor` element-wise: `ceil(x)`
   *
   * ```js
   * const x = tf.tensor1d([.6, 1.1, -3.3]);
   *
   * x.ceil().print();  // or tf.ceil(x)
   * ```
   * @param x The input Tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function ceil_(x) {
      const $x = convertToTensor(x, 'x', 'ceil');
      // TODO(manrajgrover): Return null for gradients when backprop supports it.
      const grad = (dy) => {
          return { $x: () => zerosLike(dy) };
      };
      return ENGINE.runKernelFunc(backend => backend.ceil($x), { $x }, grad);
  }
  /**
   * Computes floor of input `tf.Tensor` element-wise: `floor(x)`.
   *
   * ```js
   * const x = tf.tensor1d([.6, 1.1, -3.3]);
   *
   * x.floor().print();  // or tf.floor(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function floor_(x) {
      const $x = convertToTensor(x, 'x', 'floor');
      // TODO(nsthorat): Let gradients be null for cases where we want to stop
      // backpropgation.
      const grad = (dy) => {
          return { $x: () => zerosLike(dy) };
      };
      return ENGINE.runKernelFunc(backend => backend.floor($x), { $x }, grad);
  }
  /**
   * Returns an element-wise indication of the sign of a number.
   *
   * ```js
   * const x = tf.tensor1d([.6, 1.1, -3.3, NaN, 0]);
   *
   * x.sign().print();  // or tf.sign(x)
   * ```
   * @param x The input Tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function sign_(x) {
      const $x = convertToTensor(x, 'x', 'sign');
      const grad = (dy) => {
          return { $x: () => zerosLike(dy) };
      };
      return ENGINE.runKernelFunc(backend => backend.sign($x), { $x }, grad);
  }
  /**
   * RReturns which elements of x are NaN.
   *
   * ```js
   * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
   *
   * x.isNaN().print();  // or tf.isNaN(x)
   * ```
   * @param x The input Tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function isNaN_(x) {
      const $x = convertToTensor(x, 'x', 'isNaN');
      // TODO(nsthorat): Let gradients be null for cases where we want to stop
      // backpropgation.
      const grad = (dy) => {
          return { $x: () => zerosLike(dy) };
      };
      return ENGINE.runKernelFunc(backend => backend.isNaN($x), { $x }, grad);
  }
  /**
   * Returns which elements of x are Infinity or -Infinity.
   *
   * ```js
   * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
   *
   * x.isInf().print();  // or tf.isNaN(x)
   * ```
   * @param x The input Tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function isInf_(x) {
      const $x = convertToTensor(x, 'x', 'isInf');
      // TODO(nsthorat): Let gradients be null for cases where we want to stop
      // backpropgation.
      const grad = (dy) => {
          return { $x: () => zerosLike(dy) };
      };
      return ENGINE.runKernelFunc(backend => backend.isInf($x), { $x }, grad);
  }
  /**
   * Returns which elements of x are finite.
   *
   * ```js
   * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
   *
   * x.isFinite().print();  // or tf.isNaN(x)
   * ```
   * @param x The input Tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function isFinite_(x) {
      const $x = convertToTensor(x, 'x', 'isFinite');
      // TODO(nsthorat): Let gradients be null for cases where we want to stop
      // backpropgation.
      const grad = (dy) => {
          return { $x: () => zerosLike(dy) };
      };
      return ENGINE.runKernelFunc(backend => backend.isFinite($x), { $x }, grad);
  }
  /**
   * Computes round of input `tf.Tensor` element-wise: `round(x)`.
   * It implements banker's rounding.
   *
   * ```js
   * const x = tf.tensor1d([.6, 1.1, -3.3]);
   *
   * x.round().print();  // or tf.round(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function round_(x) {
      const $x = convertToTensor(x, 'x', 'round');
      // TODO(nsthorat): Let gradients be null for cases where we want to stop
      // backpropgation.
      const grad = (dy) => {
          return { $x: () => zerosLike(dy) };
      };
      return ENGINE.runKernelFunc(backend => backend.round($x), { $x }, grad);
  }
  /**
   * Computes exponential of the input `tf.Tensor` element-wise. `e ^ x`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, -3]);
   *
   * x.exp().print();  // or tf.exp(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function exp_(x) {
      const $x = convertToTensor(x, 'x', 'exp');
      const bck = (dy, saved) => {
          return { x: () => dy.mulStrict(saved[0]) };
      };
      const attrs = {};
      const inputsToSave = [];
      const outputsToSave = [true];
      return ENGINE.runKernelFunc((backend, save) => {
          const y = backend.exp($x);
          save([y]);
          return y;
      }, { x: $x }, bck, 'Exp', attrs, inputsToSave, outputsToSave);
  }
  /**
   * Computes exponential of the input `tf.Tensor` minus one element-wise.
   * `e ^ x - 1`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, -3]);
   *
   * x.expm1().print();  // or tf.expm1(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function expm1_(x) {
      const $x = convertToTensor(x, 'x', 'expm1');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.mul($x.exp()) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.expm1($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes natural logarithm of the input `tf.Tensor` element-wise: `ln(x)`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, Math.E]);
   *
   * x.log().print();  // or tf.log(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function log_(x) {
      const $x = convertToTensor(x, 'x', 'log');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { x: () => dy.div($x.toFloat()) };
      };
      const attrs = {};
      const inputsToSave = [$x];
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.log($x);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'Log', attrs, inputsToSave);
  }
  /**
   * Computes natural logarithm of the input `tf.Tensor` plus one
   * element-wise: `ln(1 + x)`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, Math.E - 1]);
   *
   * x.log1p().print();  // or tf.log1p(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function log1p_(x) {
      const $x = convertToTensor(x, 'x', 'log1p');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.div($x.add(1)) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.log1p($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes square root of the input `tf.Tensor` element-wise: `y = sqrt(x)`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 4, -1]);
   *
   * x.sqrt().print();  // or tf.sqrt(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function sqrt_(x) {
      const $x = convertToTensor(x, 'x', 'sqrt');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.div($x.toFloat().sqrt().mul(2)) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.sqrt($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes reciprocal of square root of the input `tf.Tensor` element-wise:
   * `y = 1 / sqrt(x)`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 4, -1]);
   *
   * x.rsqrt().print();  // or tf.rsqrt(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function rsqrt_(x) {
      const $x = convertToTensor(x, 'x', 'rsqrt');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { x: () => dy.div($x.pow(1.5).mul(2)).neg() };
      };
      const inputsToSave = [$x];
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.rsqrt($x);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'Rsqrt', {} /* attrs */, inputsToSave);
  }
  /**
   * Computes reciprocal of x element-wise: `1 / x`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, 2]);
   *
   * x.reciprocal().print();  // or tf.reciprocal(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function reciprocal_(x) {
      const $x = convertToTensor(x, 'x', 'reciprocal');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.div($x.square().neg()) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.reciprocal($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes absolute value element-wise: `abs(x)`
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   *
   * x.abs().print();  // or tf.abs(x)
   * ```
   * @param x The input `tf.Tensor`.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function abs_(x) {
      const $x = convertToTensor(x, 'x', 'abs');
      if ($x.dtype === 'complex64') {
          return ENGINE.runKernelFunc(backend => backend.complexAbs($x), { $x });
      }
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { x: () => dy.mul($x.toFloat().step(-1)) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.abs($x);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'Abs');
  }
  /**
   * Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   *
   * x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
   * ```
   * @param x The input tensor.
   * @param clipValueMin Lower-bound of range to be clipped to.
   * @param clipValueMax Upper-bound of range to be clipped to.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function clipByValue_(x, clipValueMin, clipValueMax) {
      const $x = convertToTensor(x, 'x', 'clipByValue');
      assert((clipValueMin <= clipValueMax), () => `Error in clip: min (${clipValueMin}) must be ` +
          `less than or equal to max (${clipValueMax}).`);
      const grad = (dy, saved) => {
          const [$x] = saved;
          return {
              x: () => dy.where($x.greaterEqual(clipValueMin)
                  .logicalAnd($x.lessEqual(clipValueMax)), zerosLike(dy)),
          };
      };
      const inputsToSave = [$x];
      const attr = { min: clipValueMin, max: clipValueMax };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.clip($x, clipValueMin, clipValueMax);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'ClipByValue', attr, inputsToSave);
  }
  /**
   * Computes sigmoid element-wise, `1 / (1 + exp(-x))`
   *
   * ```js
   * const x = tf.tensor1d([0, -1, 2, -3]);
   *
   * x.sigmoid().print();  // or tf.sigmoid(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function sigmoid_(x) {
      const $x = convertToTensor(x, 'x', 'sigmoid');
      const grad = (dy, saved) => {
          const [y] = saved;
          return { x: () => dy.mul(y.mul(scalar(1).sub(y))) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const y = backend.sigmoid($x);
          save([y]);
          return y;
      }, { x: $x }, grad, 'Sigmoid');
  }
  /**
   * Computes log sigmoid of the input `tf.Tensor` element-wise:
   * `logSigmoid(x)`. For numerical stability, we use `-tf.softplus(-x)`.
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.logSigmoid().print();  // or tf.logSigmoid(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function logSigmoid_(x) {
      const $x = convertToTensor(x, 'x', 'logSigmoid');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.mul($x.neg().sigmoid()) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.softplus($x.neg()).neg();
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes softplus of the input `tf.Tensor` element-wise: `log(exp(x) + 1)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.softplus().print();  // or tf.softplus(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function softplus_(x) {
      const $x = convertToTensor(x, 'x', 'softplus');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.mul($x.sigmoid()) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.softplus($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes sin of the input Tensor element-wise: `sin(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
   *
   * x.sin().print();  // or tf.sin(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function sin_(x) {
      const $x = convertToTensor(x, 'x', 'sin');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { x: () => $x.toFloat().cos().mul(dy) };
      };
      const inputsToSave = [$x];
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.sin($x);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'Sin', {} /* attrs */, inputsToSave);
  }
  /**
   * Computes cos of the input `tf.Tensor` element-wise: `cos(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
   *
   * x.cos().print();  // or tf.cos(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function cos_(x) {
      const $x = convertToTensor(x, 'x', 'cos');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { x: () => $x.toFloat().sin().neg().mul(dy) };
      };
      const inputsToSave = [$x];
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.cos($x);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'Cos', {} /* attrs */, inputsToSave);
  }
  /**
   * Computes tan of the input `tf.Tensor` element-wise, `tan(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
   *
   * x.tan().print();  // or tf.tan(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function tan_(x) {
      const $x = convertToTensor(x, 'x', 'tan');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.div($x.cos().square()) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.tan($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes asin of the input `tf.Tensor` element-wise: `asin(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.asin().print();  // or tf.asin(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function asin_(x) {
      const $x = convertToTensor(x, 'x', 'asin');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return {
              $x: () => dy.divStrict(scalar(1).sub($x.toFloat().square()).sqrt())
          };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.asin($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes acos of the input `tf.Tensor` element-wise: `acos(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.acos().print();  // or tf.acos(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function acos_(x) {
      const $x = convertToTensor(x, 'x', 'acos');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return {
              $x: () => dy.divStrict(scalar(1).sub($x.toFloat().square()).sqrt()).neg()
          };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.acos($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes atan of the input `tf.Tensor` element-wise: `atan(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.atan().print();  // or tf.atan(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function atan_(x) {
      const $x = convertToTensor(x, 'x', 'atan');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.div($x.toFloat().square().add(1)) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.atan($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes hyperbolic sin of the input `tf.Tensor` element-wise: `sinh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.sinh().print();  // or tf.sinh(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function sinh_(x) {
      const $x = convertToTensor(x, 'x', 'sinh');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => $x.toFloat().cosh().mulStrict(dy) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.sinh($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes hyperbolic cos of the input `tf.Tensor` element-wise: `cosh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.cosh().print();  // or tf.cosh(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function cosh_(x) {
      const $x = convertToTensor(x, 'x', 'cosh');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => $x.toFloat().sinh().mulStrict(dy) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.cosh($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes hyperbolic tangent of the input `tf.Tensor` element-wise: `tanh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, 70]);
   *
   * x.tanh().print();  // or tf.tanh(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function tanh_(x) {
      const $x = convertToTensor(x, 'x', 'tanh');
      const grad = (dy, saved) => {
          const [y] = saved;
          return { x: () => scalar(1).sub(y.square()).mulStrict(dy) };
      };
      const outputsToSave = [true];
      return ENGINE.runKernelFunc((backend, save) => {
          const y = backend.tanh($x);
          save([y]);
          return y;
      }, { x: $x }, grad, 'Tanh', {} /* attrs */, null /* inputsToSave */, outputsToSave);
  }
  /**
   * Computes inverse hyperbolic sin of the input `tf.Tensor` element-wise:
   * `asinh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.asinh().print();  // or tf.asinh(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function asinh_(x) {
      const $x = convertToTensor(x, 'x', 'asinh');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return {
              $x: () => dy.divStrict(scalar(1).add($x.toFloat().square()).sqrt())
          };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.asinh($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes the inverse hyperbolic cos of the input `tf.Tensor` element-wise:
   * `acosh(x)`
   *
   * ```js
   * const x = tf.tensor1d([10, 1, 3, 5.7]);
   *
   * x.acosh().print();  // or tf.acosh(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function acosh_(x) {
      const $x = convertToTensor(x, 'x', 'acosh');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.divStrict($x.toFloat().square().sub(1).sqrt()) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.acosh($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes inverse hyperbolic tan of the input `tf.Tensor` element-wise:
   * `atanh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, .1, -.1, .7]);
   *
   * x.atanh().print();  // or tf.atanh(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function atanh_(x) {
      const $x = convertToTensor(x, 'x', 'atanh');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { $x: () => dy.div(scalar(1).sub($x.toFloat().square())) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.atanh($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes gause error function of the input `tf.Tensor` element-wise:
   * `erf(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, .1, -.1, .7]);
   *
   * x.erf().print(); // or tf.erf(x);
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function erf_(x) {
      let $x = convertToTensor(x, 'x', 'erf');
      assert($x.dtype === 'int32' || $x.dtype === 'float32', () => 'Input dtype must be `int32` or `float32`.');
      if ($x.dtype === 'int32') {
          $x = $x.toFloat();
      }
      const grad = (dy, saved) => {
          const [$x] = saved;
          return {
              $x: () => dy.mul($x.square().neg().exp().mul(2 / Math.sqrt(Math.PI)))
          };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.erf($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes step of the input `tf.Tensor` element-wise: `x > 0 ? 1 : alpha * x`
   *
   * ```js
   * const x = tf.tensor1d([0, 2, -1, -3]);
   *
   * x.step(.5).print();  // or tf.step(x, .5)
   * ```
   * @param x The input tensor.
   * @param alpha The gradient when input is negative.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function step_(x, alpha = 0.0) {
      const $x = convertToTensor(x, 'x', 'step');
      // TODO(manrajgrover): Return null for gradients when backprop supports
      // it.
      const grad = (dy) => {
          return { $x: () => zerosLike(dy) };
      };
      return ENGINE.runKernelFunc(backend => backend.step($x, alpha), { $x }, grad);
  }
  const abs = op({ abs_ });
  const acos = op({ acos_ });
  const acosh = op({ acosh_ });
  const asin = op({ asin_ });
  const asinh = op({ asinh_ });
  const atan = op({ atan_ });
  const atanh = op({ atanh_ });
  const ceil = op({ ceil_ });
  const clipByValue = op({ clipByValue_ });
  const cos = op({ cos_ });
  const cosh = op({ cosh_ });
  const erf = op({ erf_ });
  const exp = op({ exp_ });
  const expm1 = op({ expm1_ });
  const floor = op({ floor_ });
  const log = op({ log_ });
  const log1p = op({ log1p_ });
  const logSigmoid = op({ logSigmoid_ });
  const neg = op({ neg_ });
  const reciprocal = op({ reciprocal_ });
  const round = op({ round_ });
  const rsqrt = op({ rsqrt_ });
  const sigmoid = op({ sigmoid_ });
  const sign = op({ sign_ });
  const isNaN$1 = op({ isNaN_ });
  const isInf = op({ isInf_ });
  const isFinite$1 = op({ isFinite_ });
  const sin = op({ sin_ });
  const sinh = op({ sinh_ });
  const softplus = op({ softplus_ });
  const sqrt = op({ sqrt_ });
  const step = op({ step_ });
  const tan = op({ tan_ });
  const tanh$1 = op({ tanh_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Adds two `tf.Tensor`s element-wise, A + B.
   *
   * Inputs must be the same shape. For broadcasting support, use add() instead.
   *
   * @param a The first Tensor to add element-wise.
   * @param b The second Tensor to add element-wise.
   */
  function addStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'addStrict');
      const $b = convertToTensor(b, 'b', 'addStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in addStrict: ');
      return $a.add($b);
  }
  /**
   * Subtracts two `tf.Tensor`s element-wise, A - B. Inputs must
   * be the same shape.
   *
   * For broadcasting support, use `tf.sub` instead.
   *
   * @param a The first Tensor to subtract element-wise.
   * @param b The second Tensor to subtract element-wise.
   */
  function subStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'subStrict');
      const $b = convertToTensor(b, 'b', 'subStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in subStrict: ');
      return $a.sub($b);
  }
  /**
   * Computes the power of one `tf.Tensor` to another. Supports broadcasting.
   *
   * Given a `tf.Tensor` x and a `tf.Tensor` y, this operation computes x^y for
   * corresponding elements in x and y. The result's dtype will be the upcasted
   * type of the `base` and `exp` dtypes.
   *
   * ```js
   * const a = tf.tensor([[2, 3], [4, 5]])
   * const b = tf.tensor([[1, 2], [3, 0]]).toInt();
   *
   * a.pow(b).print();  // or tf.pow(a, b)
   * ```
   *
   * ```js
   * const a = tf.tensor([[1, 2], [3, 4]])
   * const b = tf.tensor(2).toInt();
   *
   * a.pow(b).print();  // or tf.pow(a, b)
   * ```
   * We also expose `powStrict` which has the same signature as this op and
   * asserts that `base` and `exp` are the same shape (does not broadcast).
   *
   * @param base The base `tf.Tensor` to pow element-wise.
   * @param exp The exponent `tf.Tensor` to pow element-wise.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function pow_(base, exp) {
      let $base = convertToTensor(base, 'base', 'pow');
      let $exp = convertToTensor(exp, 'exp', 'pow');
      [$base, $exp] = makeTypesMatch($base, $exp);
      const outShape = assertAndGetBroadcastShape($base.shape, $exp.shape);
      const grad = (dy, saved) => {
          const [$base, $exp, y] = saved;
          const derBase = () => {
              const expFloat = $exp.toFloat();
              let res = dy.mul(expFloat.mul($base.pow(expFloat.sub(scalar(1)))));
              const reduceAxes = getReductionAxes($base.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = res.sum(reduceAxes);
              }
              return res.reshape($base.shape);
          };
          const derExp = () => {
              const condition = $base.greater(0);
              const logBase = $base.log().where(condition, zerosLike($base));
              let res = dy.mul(y.mul(logBase));
              const reduceAxes = getReductionAxes($exp.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = res.sum(reduceAxes);
              }
              return res.reshape($exp.shape);
          };
          return { a: derBase, b: derExp };
      };
      const attrs = {};
      const inputsToSave = [$base, $exp];
      const outputsToSave = [true];
      return ENGINE.runKernelFunc((backend, save) => {
          const y = backend.pow($base, $exp);
          save([$base, $exp, y]);
          return y;
      }, { a: $base, b: $exp }, grad, 'Pow', attrs, inputsToSave, outputsToSave);
  }
  /**
   * Computes the power of one `tf.Tensor` to another. Inputs must
   * be the same shape.
   *
   * For broadcasting support, use `tf.pow` instead.
   *
   * @param base The base tensor to pow element-wise.
   * @param exp The exponent tensor to pow element-wise.
   */
  function powStrict_(base, exp) {
      assertShapesMatch(base.shape, exp.shape, 'Error in powStrict: ');
      return base.pow(exp);
  }
  /**
   * Multiplies two `tf.Tensor`s element-wise, A * B. Supports broadcasting.
   *
   * We also expose `tf.mulStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3, 4]);
   * const b = tf.tensor1d([2, 3, 4, 5]);
   *
   * a.mul(b).print();  // or tf.mul(a, b)
   * ```
   *
   * ```js
   * // Broadcast mul a with b.
   * const a = tf.tensor1d([1, 2, 3, 4]);
   * const b = tf.scalar(5);
   *
   * a.mul(b).print();  // or tf.mul(a, b)
   * ```
   * @param a The first tensor to multiply.
   * @param b The second tensor to multiply. Must have the same dtype as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function mul_(a, b) {
      let $a = convertToTensor(a, 'a', 'mul');
      let $b = convertToTensor(b, 'b', 'mul');
      [$a, $b] = makeTypesMatch($a, $b);
      const outShape = assertAndGetBroadcastShape($a.shape, $b.shape);
      const der = (dy, saved) => {
          const [$a, $b] = saved;
          const derA = () => {
              const res = dy.mul($b.toFloat());
              const reduceAxes = getReductionAxes($a.shape, outShape);
              if (reduceAxes.length > 0) {
                  return res.sum(reduceAxes).reshape($a.shape);
              }
              return res;
          };
          const derB = () => {
              const res = dy.mul($a.toFloat());
              const reduceAxes = getReductionAxes($b.shape, outShape);
              if (reduceAxes.length > 0) {
                  return res.sum(reduceAxes).reshape($b.shape);
              }
              return res;
          };
          return { a: derA, b: derB };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.multiply($a, $b);
          save([$a, $b]);
          return res;
      }, { a: $a, b: $b }, der, 'Mul');
  }
  /**
   * Multiplies two `tf.Tensor`s element-wise, A * B.
   *
   * Inputs must be the same shape. For broadcasting support, use `tf.mul`.
   *
   * @param a The first tensor to multiply.
   * @param b The first tensor to multiply. Must have the same
   *    dtype as `a`.
   */
  function mulStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'mul');
      const $b = convertToTensor(b, 'b', 'mul');
      assertShapesMatch($a.shape, $b.shape, 'Error in multiplyStrict: ');
      return $a.mul($b);
  }
  /**
   * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
   * The result is rounded with floor function.
   *
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 9, 16]);
   * const b = tf.tensor1d([1, 2, 3, 4]);
   *
   * a.floorDiv(b).print();  // or tf.div(a, b)
   * ```
   *
   * ```js
   * // Broadcast div a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(2);
   *
   * a.floorDiv(b).print();  // or tf.floorDiv(a, b)
   * ```
   *
   * @param a The first tensor as the numerator.
   * @param b The second tensor as the denominator. Must have the same dtype as
   * `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function floorDiv_(a, b) {
      let $a = convertToTensor(a, 'a', 'floorDiv');
      let $b = convertToTensor(b, 'b', 'floorDiv');
      [$a, $b] = makeTypesMatch($a, $b);
      const outShape = assertAndGetBroadcastShape($a.shape, $b.shape);
      const der = (dy, saved) => {
          const [$a, $b] = saved;
          const derA = () => {
              const res = dy.div($b.toFloat());
              const reduceAxes = getReductionAxes($a.shape, outShape);
              if (reduceAxes.length > 0) {
                  return res.sum(reduceAxes).reshape($a.shape);
              }
              return res;
          };
          const derB = () => {
              let res = dy.mul($a.toFloat());
              const reduceAxes = getReductionAxes($b.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = res.sum(reduceAxes).reshape($b.shape);
              }
              const tmp = $b.square();
              return res.div(tmp.toFloat()).neg();
          };
          return { a: derA, b: derB };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.floorDiv($a, $b);
          save([$a, $b]);
          return res;
      }, { a: $a, b: $b }, der, 'FloorDiv');
  }
  /**
   * Divides two `tf.Tensor`s element-wise, A / B. Inputs must
   * be the same shape.
   *
   * @param a The first tensor as the numerator for element-wise division.
   * @param b The second tensor as the denominator for element-wise division.
   */
  function divStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'div');
      const $b = convertToTensor(b, 'b', 'div');
      assertShapesMatch($a.shape, $b.shape, 'Error in divideStrict: ');
      return $a.div($b);
  }
  /**
   * Returns the mod of a and b element-wise.
   * `floor(x / y) * y + mod(x, y) = x`
   * Supports broadcasting.
   *
   * We also expose `tf.modStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 3, 16]);
   * const b = tf.tensor1d([1, 2, 9, 4]);
   *
   * a.mod(b).print();  // or tf.mod(a, b)
   * ```
   *
   * ```js
   * // Broadcast a mod b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(5);
   *
   * a.mod(b).print();  // or tf.mod(a, b)
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function mod_(a, b) {
      let $a = convertToTensor(a, 'a', 'mod');
      let $b = convertToTensor(b, 'b', 'mod');
      [$a, $b] = makeTypesMatch($a, $b);
      const outShape = assertAndGetBroadcastShape($a.shape, $b.shape);
      const der = (dy, saved) => {
          const [$a, $b] = saved;
          const derA = () => {
              const reduceAxes = getReductionAxes($a.shape, outShape);
              if (reduceAxes.length > 0) {
                  return dy.sum(reduceAxes).reshape($a.shape);
              }
              return dy;
          };
          const derB = () => {
              const res = dy.mul($a.div($b).floor().neg());
              const reduceAxes = getReductionAxes($b.shape, outShape);
              if (reduceAxes.length > 0) {
                  return res.sum(reduceAxes).reshape($b.shape);
              }
              return res;
          };
          return { $a: derA, $b: derB };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.mod($a, $b);
          save([$a, $b]);
          return res;
      }, { $a, $b }, der);
  }
  /**
   * Returns the mod of a and b (`a < b ? a : b`) element-wise. Inputs must
   * be the same shape. For broadcasting support, use mod().
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same dtype as `a`.
   */
  function modStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'modStrict');
      const $b = convertToTensor(b, 'b', 'modStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in modStrict: ');
      return $a.mod($b);
  }
  /**
   * Returns the min of a and b (`a < b ? a : b`) element-wise.
   * Supports broadcasting.
   *
   * We also expose `minimumStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 3, 16]);
   * const b = tf.tensor1d([1, 2, 9, 4]);
   *
   * a.minimum(b).print();  // or tf.minimum(a, b)
   * ```
   *
   * ```js
   * // Broadcast minimum a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(5);
   *
   * a.minimum(b).print();  // or tf.minimum(a, b)
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function minimum_(a, b) {
      let $a = convertToTensor(a, 'a', 'minimum');
      let $b = convertToTensor(b, 'b', 'minimum');
      [$a, $b] = makeTypesMatch($a, $b);
      if ($a.dtype === 'bool') {
          $a = $a.toInt();
          $b = $b.toInt();
      }
      assertAndGetBroadcastShape($a.shape, $b.shape);
      const der = (dy, saved) => {
          const [$a, $b] = saved;
          const derA = () => dy.mul($a.lessEqual($b).toFloat());
          const derB = () => dy.mul($a.greater($b).toFloat());
          return { a: derA, b: derB };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.minimum($a, $b);
          save([$a, $b]);
          return res;
      }, { a: $a, b: $b }, der, 'Minimum');
  }
  /**
   * Returns the min of a and b (`a < b ? a : b`) element-wise. Inputs must
   * be the same shape. For broadcasting support, use minimum().
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same dtype as `a`.
   */
  function minimumStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'minimumStrict');
      const $b = convertToTensor(b, 'b', 'minimumStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in minimumStrict: ');
      return $a.minimum($b);
  }
  /**
   * Returns the max of a and b (`a > b ? a : b`) element-wise.
   * Supports broadcasting.
   *
   * We also expose `tf.maximumStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 3, 16]);
   * const b = tf.tensor1d([1, 2, 9, 4]);
   *
   * a.maximum(b).print();  // or tf.maximum(a, b)
   * ```
   *
   * ```js
   * // Broadcast maximum a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(5);
   *
   * a.maximum(b).print();  // or tf.maximum(a, b)
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function maximum_(a, b) {
      let $a = convertToTensor(a, 'a', 'maximum');
      let $b = convertToTensor(b, 'b', 'maximum');
      [$a, $b] = makeTypesMatch($a, $b);
      if ($a.dtype === 'bool') {
          $a = $a.toInt();
          $b = $b.toInt();
      }
      assertAndGetBroadcastShape($a.shape, $b.shape);
      const der = (dy, saved) => {
          const [$a, $b] = saved;
          const derA = () => dy.mul($a.greaterEqual($b).toFloat());
          const derB = () => dy.mul($a.less($b).toFloat());
          return { a: derA, b: derB };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.maximum($a, $b);
          save([$a, $b]);
          return res;
      }, { a: $a, b: $b }, der, 'Maximum');
  }
  /**
   * Returns the max of a and b (`a > b ? a : b`) element-wise. Inputs must
   * be the same shape. For broadcasting support, use maximum().
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same dtype as `a`.
   */
  function maximumStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'maximumStrict');
      const $b = convertToTensor(b, 'b', 'maximumStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in maximumStrict: ');
      return $a.maximum($b);
  }
  /**
   * Returns (a - b) * (a - b) element-wise.
   *
   * Inputs must be the same shape. For broadcasting support, use
   * `tf.squaredDifference` instead.
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  function squaredDifferenceStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'squaredDifferenceStrict');
      const $b = convertToTensor(b, 'b', 'squaredDifferenceStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in squaredDifferenceStrict: ');
      return $a.squaredDifference($b);
  }
  /**
   * Computes arctangent of `tf.Tensor`s a / b element-wise: `atan2(a, b)`.
   * Supports broadcasting.
   *
   * ```js
   * const a = tf.tensor1d([1.0, 1.0, -1.0, .7]);
   * const b = tf.tensor1d([2.0, 13.0, 3.5, .21]);
   *
   * tf.atan2(a, b).print()
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same dtype as `a`.
   *
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function atan2_(a, b) {
      let $a = convertToTensor(a, 'a', 'atan2');
      let $b = convertToTensor(b, 'b', 'atan2');
      [$a, $b] = makeTypesMatch($a, $b);
      const outShape = assertAndGetBroadcastShape($a.shape, $b.shape);
      const der = (dy, saved) => {
          const [$a, $b] = saved;
          const derA = () => {
              const d = add($a.square(), $b.square());
              let res = dy.mul($b.div(d));
              const reduceAxes = getReductionAxes($a.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = res.sum(reduceAxes);
              }
              return res.reshape($a.shape);
          };
          const derB = () => {
              const d = add($a.square(), $b.square());
              let res = neg(dy.mul($a.div(d)));
              const reduceAxes = getReductionAxes($b.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = res.sum(reduceAxes);
              }
              return res.reshape($b.shape);
          };
          return { $a: derA, $b: derB };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.atan2($a, $b);
          save([$a, $b]);
          return res;
      }, { $a, $b }, der);
  }
  const addStrict = op({ addStrict_ });
  const atan2 = op({ atan2_ });
  const divStrict = op({ divStrict_ });
  const floorDiv = op({ floorDiv_ });
  const maximum = op({ maximum_ });
  const maximumStrict = op({ maximumStrict_ });
  const minimum = op({ minimum_ });
  const minimumStrict = op({ minimumStrict_ });
  const mod = op({ mod_ });
  const modStrict = op({ modStrict_ });
  const mul = op({ mul_ });
  const mulStrict = op({ mulStrict_ });
  const pow = op({ pow_ });
  const powStrict = op({ powStrict_ });
  const squaredDifferenceStrict = op({ squaredDifferenceStrict_ });
  const subStrict = op({ subStrict_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
   *
   * We also expose `tf.divStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 9, 16]);
   * const b = tf.tensor1d([1, 2, 3, 4]);
   *
   * a.div(b).print();  // or tf.div(a, b)
   * ```
   *
   * ```js
   * // Broadcast div a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(2);
   *
   * a.div(b).print();  // or tf.div(a, b)
   * ```
   *
   * @param a The first tensor as the numerator.
   * @param b The second tensor as the denominator. Must have the same dtype as
   * `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function div_(a, b) {
      let $a = convertToTensor(a, 'a', 'div');
      let $b = convertToTensor(b, 'b', 'div');
      [$a, $b] = makeTypesMatch($a, $b);
      if ($a.dtype === 'int32' && $b.dtype === 'int32') {
          return floorDiv($a, $b);
      }
      const forward = (backend, save) => {
          const res = backend.realDivide($a, $b);
          save([$a, $b]);
          return res;
      };
      const inputs = { a: $a, b: $b };
      const attrs = {};
      return ENGINE.runKernelFunc(forward, inputs, null /* gradient */, Div, attrs);
  }
  const div = op({ div_ });

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
  /**
   * Computes square of `x` element-wise: `x ^ 2`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, Math.sqrt(2), -1]);
   *
   * x.square().print();  // or tf.square(x)
   * ```
   * @param x The input Tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function square_(x) {
      const $x = convertToTensor(x, 'x', 'square');
      const attrs = {};
      const inputsToSave = [$x];
      const outputsToSave = [];
      return ENGINE.runKernelFunc((backend, save) => {
          save([$x]);
          return backend.square($x);
      }, { x: $x }, null /* grad */, 'Square', attrs, inputsToSave, outputsToSave);
  }
  const square = op({ square_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const divGradConfig = {
      kernelName: Div,
      inputsToSave: ['a', 'b'],
      gradFunc: (dy, saved) => {
          const [a, b] = saved;
          const outShape = assertAndGetBroadcastShape(a.shape, b.shape);
          const derA = () => {
              const res = div(dy, b.toFloat());
              const reduceAxes = getReductionAxes(a.shape, outShape);
              if (reduceAxes.length > 0) {
                  return sum$1(res, reduceAxes).reshape(a.shape);
              }
              return res;
          };
          const derB = () => {
              let res = mul(dy, a.toFloat());
              const reduceAxes = getReductionAxes(b.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = reshape(sum$1(res, reduceAxes), b.shape);
              }
              const tmp = square(b);
              return neg(div(res, tmp.toFloat()));
          };
          return { a: derA, b: derB };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Subtracts two `tf.Tensor`s element-wise, A - B. Supports broadcasting.
   *
   * We also expose `tf.subStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([10, 20, 30, 40]);
   * const b = tf.tensor1d([1, 2, 3, 4]);
   *
   * a.sub(b).print();  // or tf.sub(a, b)
   * ```
   *
   * ```js
   * // Broadcast subtract a with b.
   * const a = tf.tensor1d([10, 20, 30, 40]);
   * const b = tf.scalar(5);
   *
   * a.sub(b).print();  // or tf.sub(a, b)
   * ```
   * @param a The first `tf.Tensor` to subtract from.
   * @param b The second `tf.Tensor` to be subtracted. Must have the same dtype as
   * `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function sub_(a, b) {
      let $a = convertToTensor(a, 'a', 'sub');
      let $b = convertToTensor(b, 'b', 'sub');
      [$a, $b] = makeTypesMatch($a, $b);
      const forward = (backend, save) => {
          const res = backend.subtract($a, $b);
          save([$a, $b]);
          return res;
      };
      const inputs = { a: $a, b: $b };
      return ENGINE.runKernelFunc(forward, inputs, null /* grad */, Sub);
  }
  const sub = op({ sub_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Construct a tensor by repeating it the number of times given by reps.
   *
   * This operation creates a new tensor by replicating `input` `reps`
   * times. The output tensor's i'th dimension has `input.shape[i] *
   * reps[i]` elements, and the values of `input` are replicated
   * `reps[i]` times along the i'th dimension. For example, tiling
   * `[a, b, c, d]` by `[2]` produces `[a, b, c, d, a, b, c, d]`.
   *
   * ```js
   * const a = tf.tensor1d([1, 2]);
   *
   * a.tile([2]).print();    // or a.tile([2])
   * ```
   *
   * ```js
   * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * a.tile([1, 2]).print();  // or a.tile([1, 2])
   * ```
   * @param x The tensor to tile.
   * @param reps Determines the number of replications per dimension.
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  function tile_(x, reps) {
      const parseAs = null;
      const $x = convertToTensor(x, 'x', 'tile', parseAs);
      assert($x.rank === reps.length, () => `Error in transpose: rank of input ${$x.rank} ` +
          `must match length of reps ${reps}.`);
      const forward = (backend, save) => {
          const res = backend.tile($x, reps);
          save([$x]);
          return res;
      };
      const inputsToSave = [$x];
      const inputs = { x: $x };
      const attrs = { reps };
      return ENGINE.runKernelFunc(forward, inputs, null /* grad */, Tile, attrs, inputsToSave);
  }
  const tile = op({ tile_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const fusedBatchNormGradConfig = {
      kernelName: FusedBatchNorm,
      inputsToSave: ['x', 'mean', 'variance', 'scale'],
      gradFunc: (dy, saved, attrs) => {
          const { varianceEpsilon } = attrs;
          const [x, mean, variance, scale] = saved;
          const scaleValue = scale == null ? scalar(1) : scale;
          const reductionAxes = getReductionAxes(mean.shape, x.shape);
          const tileShape = [];
          if (mean.rank === 1) {
              for (let i = 0; i < x.shape.length - 1; ++i) {
                  tileShape.push(x.shape[i]);
              }
              tileShape.push(1);
          }
          const xMinusMean = sub(x, mean);
          const dyTimesScaleValue = mul(dy, scaleValue);
          const oneOverSqrtVariance = rsqrt(add(variance, scalar(varianceEpsilon)));
          const minusHalfRCube = mul(mul(mul(oneOverSqrtVariance, oneOverSqrtVariance), oneOverSqrtVariance), scalar(-0.5));
          const derX = () => {
              if (mean.rank === 1) {
                  return reshape(mul(mul(dy, tile(oneOverSqrtVariance.as4D(1, 1, 1, mean.shape[0]), tileShape)), scaleValue), x.shape);
              }
              else {
                  return reshape(mul(mul(dy, oneOverSqrtVariance), scaleValue), x.shape);
              }
          };
          const derMean = () => {
              let meanDer = mul(mul(oneOverSqrtVariance, scalar(-1)), dyTimesScaleValue);
              if (mean.rank === 1) {
                  meanDer = sum$1(meanDer, reductionAxes);
              }
              return reshape(meanDer, mean.shape);
          };
          const derVariance = () => {
              let varianceDer = mul(mul(minusHalfRCube, xMinusMean), dyTimesScaleValue);
              if (mean.rank === 1) {
                  varianceDer = sum$1(varianceDer, reductionAxes);
              }
              return reshape(varianceDer, mean.shape);
          };
          const derScale = () => {
              const xMinusMean2TimesRsqrt = mul(xMinusMean, oneOverSqrtVariance);
              let scaleDer = mul(dy, xMinusMean2TimesRsqrt);
              if (mean.rank === 1) {
                  scaleDer = sum$1(scaleDer, reductionAxes);
              }
              return reshape(scaleDer, mean.shape);
          };
          const derOffset = () => {
              let offsetDer = dy;
              if (mean.rank === 1) {
                  offsetDer = sum$1(offsetDer, reductionAxes);
              }
              return reshape(offsetDer, mean.shape);
          };
          return {
              x: derX,
              mean: derMean,
              variance: derVariance,
              scale: derScale,
              offset: derOffset
          };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const identityGradConfig = {
      kernelName: Identity,
      gradFunc: (dy) => {
          return { x: () => dy.toFloat() };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const oneHotGradConfig = {
      kernelName: OneHot,
      inputsToSave: ['indices'],
      gradFunc: (dy, saved) => {
          const indices = saved[0];
          return { indices: () => zeros(indices.shape, 'float32') };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const padV2GradConfig = {
      kernelName: PadV2,
      inputsToSave: ['x'],
      gradFunc: (dy, saved, attrs) => {
          // Pad introduces values around the original tensor, so the gradient
          // slices the original shape out of the gradient.
          const x = saved[0];
          const { paddings } = attrs;
          const begin = paddings.map(p => p[0]);
          return { x: () => dy.slice(begin, x.shape) };
      }
  };

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
  const squareGradConfig = {
      kernelName: Square,
      inputsToSave: ['x'],
      gradFunc: (dy, saved) => {
          const [x] = saved;
          return { x: () => mul(dy, mul(x.toFloat(), 2)) };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const squaredDifferenceGradConfig = {
      kernelName: SquaredDifference,
      inputsToSave: ['a', 'b'],
      gradFunc: (dy, saved) => {
          const [a, b] = saved;
          const two = scalar(2);
          const derA = () => mul(dy, mul(two, sub(a, b)));
          const derB = () => mul(dy, mul(two, sub(b, a)));
          return { a: derA, b: derB };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const subGradConfig = {
      kernelName: Sub,
      inputsToSave: ['a', 'b'],
      gradFunc: (dy, saved) => {
          const [a, b] = saved;
          const outShape = assertAndGetBroadcastShape(a.shape, b.shape);
          const derA = () => {
              let res = dy;
              const reduceAxes = getReductionAxes(a.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = sum$1(res, reduceAxes);
              }
              return reshape(res, a.shape);
          };
          const derB = () => {
              let res = dy;
              const reduceAxes = getReductionAxes(b.shape, outShape);
              if (reduceAxes.length > 0) {
                  res = sum$1(res, reduceAxes);
              }
              return reshape(neg(res), b.shape);
          };
          return { a: derA, b: derB };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Pads a `tf.Tensor` with a given value and paddings.
   *
   * This operation currently only implements the `CONSTANT` mode.
   *
   * Also available are stricter rank-specific methods with the same signature
   * as this method that assert that `paddings` is of given length.
   *   - `tf.pad1d`
   *   - `tf.pad2d`
   *   - `tf.pad3d`
   *   - `tf.pad4d`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * x.pad([[1, 2]]).print();
   * ```
   * @param x The tensor to pad.
   * @param paddings An array of length `R` (the rank of the tensor), where
   * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
   * specifying how much to pad along each dimension of the tensor.
   * @param constantValue The pad value to use. Defaults to 0.
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function pad_(x, paddings, constantValue = 0) {
      const $x = convertToTensor(x, 'x', 'pad');
      if ($x.rank === 0) {
          throw new Error('pad(scalar) is not defined. Pass non-scalar to pad');
      }
      const forward = (backend, save) => {
          save([$x]);
          return backend.pad($x, paddings, constantValue);
      };
      const attrs = { paddings, constantValue };
      const inputs = { x: $x };
      return ENGINE.runKernelFunc(forward, inputs, null /* grad */, PadV2, attrs);
  }
  const pad = op({ pad_ });

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
  function assertParamsValid(input, begin, size) {
      assert(input.rank === begin.length, () => `Error in slice${input.rank}D: Length of begin ${begin} must ` +
          `match the rank of the array (${input.rank}).`);
      assert(input.rank === size.length, () => `Error in slice${input.rank}D: Length of size ${size} must ` +
          `match the rank of the array (${input.rank}).`);
      for (let i = 0; i < input.rank; ++i) {
          assert(begin[i] + size[i] <= input.shape[i], () => `Error in slice${input.rank}D: begin[${i}] + size[${i}] ` +
              `(${begin[i] + size[i]}) would overflow input.shape[${i}] (${input.shape[i]})`);
      }
  }
  /** Converts a binary mask to an array of axes. Used in stridedSlice(). */
  function maskToAxes(mask) {
      const axes = [];
      let axis = 0;
      while (mask > 0) {
          if (mask & 1) {
              axes.push(axis);
          }
          mask /= 2;
          axis++;
      }
      return axes;
  }
  /** Computes the output shape given the strided slice params. */
  function computeOutShape$1(begin, end, strides) {
      const size = [];
      for (let axis = 0; axis < begin.length; axis++) {
          size[axis] = Math.ceil((end[axis] - begin[axis]) / strides[axis]);
      }
      return size;
  }
  function startForAxis(beginMask, startIndices, strides, inputShape, axis) {
      // Begin with the specified index
      let start = startIndices[axis];
      const stride = strides[axis] || 1;
      // Check the axis bit from right of beginMask or the begin index is not set
      // for the axis.
      if (beginMask & 1 << axis || start == null) {
          if (stride > 0) {
              // Forward iteration - use the first element. These values will get
              // clamped below (Note: We could have set them to 0 and axis_size-1, but
              // use lowest() and max() to maintain symmetry with StopForAxis())
              start = Number.MIN_SAFE_INTEGER;
          }
          else {
              // Backward iteration - use the last element.
              start = Number.MAX_SAFE_INTEGER;
          }
      }
      // Handle negative indices
      const axisSize = inputShape[axis];
      if (start < 0) {
          start += axisSize;
      }
      // Clamping
      start = clamp(0, start, axisSize - 1);
      return start;
  }
  function stopForAxis(endMask, stopIndices, strides, inputShape, axis) {
      // Begin with the specified index
      let stop = stopIndices[axis];
      const stride = strides[axis] || 1;
      // Check the axis bit from right of endMask or if the stop index is not set
      // for this axis.
      if (endMask & (1 << axis) || stop == null) {
          if (stride > 0) {
              // Forward iteration - use the last element. These values will get
              // clamped below
              stop = Number.MAX_SAFE_INTEGER;
          }
          else {
              // Backward iteration - use the first element.
              stop = Number.MIN_SAFE_INTEGER;
          }
      }
      // Handle negative indices
      const axisSize = inputShape[axis];
      if (stop < 0) {
          stop += axisSize;
      }
      // Clamping
      // Because the end index points one past the last element, we need slightly
      // different clamping ranges depending on the direction.
      if (stride > 0) {
          // Forward iteration
          stop = clamp(0, stop, axisSize);
      }
      else {
          // Backward iteration
          stop = clamp(-1, stop, axisSize - 1);
      }
      return stop;
  }
  /**
   * Returns true if the slice occupies a continous set of elements in the
   * 'flat' space.
   */
  function isSliceContinous(shape, begin, size) {
      // Index of the first axis that has size > 1.
      let firstNonOneAxis = size.length;
      for (let i = 0; i < size.length; i++) {
          if (size[i] > 1) {
              firstNonOneAxis = i;
              break;
          }
      }
      for (let i = firstNonOneAxis + 1; i < size.length; i++) {
          if (begin[i] > 0 || size[i] !== shape[i]) {
              return false;
          }
      }
      return true;
  }
  function computeFlatOffset(begin, strides) {
      let flatOffset = begin.length > 0 ? begin[begin.length - 1] : 1;
      for (let i = 0; i < begin.length - 1; i++) {
          flatOffset += begin[i] * strides[i];
      }
      return flatOffset;
  }

  var slice_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    assertParamsValid: assertParamsValid,
    maskToAxes: maskToAxes,
    computeOutShape: computeOutShape$1,
    startForAxis: startForAxis,
    stopForAxis: stopForAxis,
    isSliceContinous: isSliceContinous,
    computeFlatOffset: computeFlatOffset
  });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
   * of length `size`. See `slice` for details.
   */
  function slice1d_(x, begin, size) {
      const $x = convertToTensor(x, 'x', 'slice1d');
      assert($x.rank === 1, () => `slice1d expects a rank-1 tensor, but got a rank-${$x.rank} tensor`);
      return slice($x, [begin], [size]);
  }
  /**
   * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  function slice2d_(x, begin, size) {
      const $x = convertToTensor(x, 'x', 'slice2d');
      assert($x.rank === 2, () => `slice2d expects a rank-2 tensor, but got a rank-${$x.rank} tensor`);
      return slice($x, begin, size);
  }
  /**
   * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  function slice3d_(x, begin, size) {
      const $x = convertToTensor(x, 'x', 'slice3d');
      assert($x.rank === 3, () => `slice3d expects a rank-3 tensor, but got a rank-${$x.rank} tensor`);
      return slice($x, begin, size);
  }
  /**
   * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  function slice4d_(x, begin, size) {
      const $x = convertToTensor(x, 'x', 'slice4d');
      assert($x.rank === 4, () => `slice4d expects a rank-4 tensor, but got a rank-${$x.rank} tensor`);
      return slice($x, begin, size);
  }
  /**
   * Extracts a slice from a `tf.Tensor` starting at coordinates `begin`
   * and is of size `size`.
   *
   * Also available are stricter rank-specific methods with the same signature
   * as this method that assert that `x` is of the given rank:
   *   - `tf.slice1d`
   *   - `tf.slice2d`
   *   - `tf.slice3d`
   *   - `tf.slice4d`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   *
   * x.slice([1], [2]).print();
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * x.slice([1, 0], [1, 2]).print();
   * ```
   * @param x The input `tf.Tensor` to slice from.
   * @param begin The coordinates to start the slice from. The length can be
   *     less than the rank of x - the rest of the axes will have implicit 0 as
   *     start. Can also be a single number, in which case it specifies the
   *     first axis.
   * @param size The size of the slice. The length can be less than the rank of
   *     x - the rest of the axes will have implicit -1. A value of -1 requests
   *     the rest of the dimensions in the axis. Can also be a single number,
   *     in which case it specifies the size of the first axis.
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  function slice_(x, begin, size) {
      const $x = convertToTensor(x, 'x', 'slice');
      if ($x.rank === 0) {
          throw new Error('Slicing scalar is not possible');
      }
      // The following logic allows for more ergonomic calls.
      let begin_;
      if (typeof begin === 'number') {
          begin_ = [begin, ...new Array($x.rank - 1).fill(0)];
      }
      else if (begin.length < $x.rank) {
          begin_ = begin.concat(new Array($x.rank - begin.length).fill(0));
      }
      else {
          begin_ = begin.slice();
      }
      begin_.forEach(d => {
          assert(d !== -1, () => 'slice() does not support negative begin indexing.');
      });
      let size_;
      if (size == null) {
          size_ = new Array($x.rank).fill(-1);
      }
      else if (typeof size === 'number') {
          size_ = [size, ...new Array($x.rank - 1).fill(-1)];
      }
      else if (size.length < $x.rank) {
          size_ = size.concat(new Array($x.rank - size.length).fill(-1));
      }
      else {
          size_ = size;
      }
      size_ = size_.map((d, i) => {
          if (d >= 0) {
              return d;
          }
          else {
              assert(d === -1, () => `Negative size values should be exactly -1 but got ` +
                  `${d} for the slice() size at index ${i}.`);
              return $x.shape[i] - begin_[i];
          }
      });
      assertParamsValid($x, begin_, size_);
      const inputShape = $x.shape;
      const grad = (dy) => {
          // Create an Nx2 padding where the first column represents how many
          // zeros are prepended (at start) for each dimension, and the second
          // column indicates how many zeros are appended (at end).
          // The number of zeros to append is the shape of the input
          // elementwise-subtracted by both the begin vector and sizes vector.
          const paddings = [];
          for (let i = 0; i < dy.rank; i++) {
              paddings.push([begin_[i], inputShape[i] - begin_[i] - size_[i]]);
          }
          return { x: () => pad(dy, paddings) };
      };
      const attrs = { begin: begin_, size: size_ };
      return ENGINE.runKernelFunc(backend => backend.slice($x, begin_, size_), { x: $x }, grad, 'Slice', attrs);
  }
  const slice = op({ slice_ });
  const slice1d = op({ slice1d_ });
  const slice2d = op({ slice2d_ });
  const slice3d = op({ slice3d_ });
  const slice4d = op({ slice4d_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  const tileGradConfig = {
      kernelName: Tile,
      inputsToSave: ['x'],
      gradFunc: (dy, saved, attrs) => {
          const [x] = saved;
          const { reps } = attrs;
          const derX = () => {
              let xGrad = zerosLike(x);
              // TODO(cais): Maybe reduce memory footprint by avoiding repeated
              // slicing.
              if (x.rank === 1) {
                  for (let i = 0; i < reps[0]; ++i) {
                      xGrad = add(xGrad, slice(dy, [i * x.shape[0]], [x.shape[0]]));
                  }
              }
              else if (x.rank === 2) {
                  for (let i = 0; i < reps[0]; ++i) {
                      for (let j = 0; j < reps[1]; ++j) {
                          xGrad = add(xGrad, slice(dy, [i * x.shape[0], j * x.shape[1]], [
                              x.shape[0], x.shape[1]
                          ]));
                      }
                  }
              }
              else if (x.rank === 3) {
                  for (let i = 0; i < reps[0]; ++i) {
                      for (let j = 0; j < reps[1]; ++j) {
                          for (let k = 0; k < reps[2]; ++k) {
                              xGrad =
                                  add(xGrad, slice(dy, [i * x.shape[0], j * x.shape[1], k * x.shape[2]], [x.shape[0], x.shape[1], x.shape[2]]));
                          }
                      }
                  }
              }
              else if (x.rank === 4) {
                  for (let i = 0; i < reps[0]; ++i) {
                      for (let j = 0; j < reps[1]; ++j) {
                          for (let k = 0; k < reps[2]; ++k) {
                              for (let l = 0; l < reps[3]; ++l) {
                                  xGrad =
                                      add(xGrad, slice(dy, [
                                          i * x.shape[0], j * x.shape[1], k * x.shape[2],
                                          l * x.shape[3]
                                      ], [x.shape[0], x.shape[1], x.shape[2], x.shape[3]]));
                              }
                          }
                      }
                  }
              }
              else {
                  throw new Error(`Gradient for tile operation is not implemented for rank-` +
                      `${x.rank} tensors yet.`);
              }
              return xGrad;
          };
          return { x: derX };
      },
  };

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Transposes the `tf.Tensor`. Permutes the dimensions according to `perm`.
   *
   * The returned `tf.Tensor`'s dimension `i` will correspond to the input
   * dimension `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`,
   * where `n` is the rank of the input `tf.Tensor`. Hence by default, this
   * operation performs a regular matrix transpose on 2-D input `tf.Tensor`s.
   *
   * ```js
   * const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
   *
   * a.transpose().print();  // or tf.transpose(a)
   * ```
   *
   * @param x The tensor to transpose.
   * @param perm The permutation of the dimensions of a.
   */
  /** @doc {heading: 'Operations', subheading: 'Matrices'} */
  function transpose_(x, perm) {
      const $x = convertToTensor(x, 'x', 'transpose');
      if (perm == null) {
          perm = $x.shape.map((s, i) => i).reverse();
      }
      assert($x.rank === perm.length, () => `Error in transpose: rank of input ${$x.rank} ` +
          `must match length of perm ${perm}.`);
      perm.forEach(axis => {
          assert(axis >= 0 && axis < $x.rank, () => `All entries in 'perm' must be between 0 and ${$x.rank - 1}` +
              ` but got ${perm}`);
      });
      if ($x.rank <= 1) {
          return $x.clone();
      }
      const attrs = { perm };
      return ENGINE.runKernelFunc(backend => backend.transpose($x, perm), { x: $x }, null /* gradient */, 'Transpose', attrs);
  }
  const transpose = op({ transpose_ });

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  const transposeGradConfig = {
      kernelName: Transpose,
      gradFunc: (dy, saved, attrs) => {
          const transposeAttrs = attrs;
          const { perm } = transposeAttrs;
          const undoPerm = getUndoAxesPermutation(perm);
          return { x: () => transpose(dy, undoPerm) };
      }
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  // Export all kernel configs here so that the package can auto register them
  const gradConfigs = [
      addGradConfig, addNGradConfig, broadcastToGradConfig, divGradConfig,
      fusedBatchNormGradConfig, identityGradConfig, oneHotGradConfig,
      padV2GradConfig, squareGradConfig, squaredDifferenceGradConfig,
      tileGradConfig, transposeGradConfig, subGradConfig
  ];
  for (const gradientConfig of gradConfigs) {
      registerGradient(gradientConfig);
  }

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
  class PlatformBrowser {
      fetch(path, init) {
          return fetch(path, init);
      }
      now() {
          return performance.now();
      }
      encode(text, encoding) {
          if (encoding !== 'utf-8' && encoding !== 'utf8') {
              throw new Error(`Browser's encoder only supports utf-8, but got ${encoding}`);
          }
          if (this.textEncoder == null) {
              this.textEncoder = new TextEncoder();
          }
          return this.textEncoder.encode(text);
      }
      decode(bytes, encoding) {
          return new TextDecoder(encoding).decode(bytes);
      }
  }
  if (env().get('IS_BROWSER')) {
      env().setPlatform('browser', new PlatformBrowser());
  }

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
  // We are wrapping this within an object so it can be stubbed by Jasmine.
  const getNodeFetch = {
      // tslint:disable-next-line:no-require-imports
      importFetch: () => require('node-fetch')
  };
  let systemFetch;
  class PlatformNode {
      constructor() {
          // tslint:disable-next-line:no-require-imports
          this.util = require('util');
          // According to the spec, the built-in encoder can do only UTF-8 encoding.
          // https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/TextEncoder
          this.textEncoder = new this.util.TextEncoder();
      }
      fetch(path, requestInits) {
          if (env().global.fetch != null) {
              return env().global.fetch(path, requestInits);
          }
          if (systemFetch == null) {
              systemFetch = getNodeFetch.importFetch();
          }
          return systemFetch(path, requestInits);
      }
      now() {
          const time = process.hrtime();
          return time[0] * 1000 + time[1] / 1000000;
      }
      encode(text, encoding) {
          if (encoding !== 'utf-8' && encoding !== 'utf8') {
              throw new Error(`Node built-in encoder only supports utf-8, but got ${encoding}`);
          }
          return this.textEncoder.encode(text);
      }
      decode(bytes, encoding) {
          if (bytes.length === 0) {
              return '';
          }
          return new this.util.TextDecoder(encoding).decode(bytes);
      }
  }
  if (env().get('IS_NODE')) {
      env().setPlatform('node', new PlatformNode());
  }

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
  /* Type definitions for exporting and importing of models. */
  /**
   * A map from Tensor dtype to number of bytes per element of the Tensor.
   */
  const DTYPE_VALUE_SIZE_MAP = {
      'float32': 4,
      'int32': 4,
      'uint16': 2,
      'uint8': 1,
      'bool': 1,
  };

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
  /** Number of bytes reserved for the length of the string. (32bit integer). */
  const NUM_BYTES_STRING_LENGTH = 4;
  /**
   * Encode a map from names to weight values as an ArrayBuffer, along with an
   * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
   *
   * This function does not perform sharding.
   *
   * This function is the reverse of `decodeWeights`.
   *
   * @param tensors A map ("dict") from names to tensors.
   * @param group Group to which the weights belong (optional).
   * @returns A `Promise` of
   *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
   *     concatenated.
   *   - An `Array` of `WeightManifestEntry`s, carrying information including
   *     tensor names, `dtype`s and shapes.
   * @throws Error: on unsupported tensor `dtype`.
   */
  async function encodeWeights(tensors, group) {
      // TODO(adarob, cais): Support quantization.
      const specs = [];
      const dataPromises = [];
      const names = Array.isArray(tensors) ?
          tensors.map(tensor => tensor.name) :
          Object.keys(tensors);
      for (let i = 0; i < names.length; ++i) {
          const name = names[i];
          const t = Array.isArray(tensors) ? tensors[i].tensor : tensors[name];
          if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool' &&
              t.dtype !== 'string') {
              throw new Error(`Unsupported dtype in weight '${name}': ${t.dtype}`);
          }
          const spec = { name, shape: t.shape, dtype: t.dtype };
          if (t.dtype === 'string') {
              const utf8bytes = new Promise(async (resolve) => {
                  const vals = await t.bytes();
                  const totalNumBytes = vals.reduce((p, c) => p + c.length, 0) +
                      NUM_BYTES_STRING_LENGTH * vals.length;
                  const bytes = new Uint8Array(totalNumBytes);
                  let offset = 0;
                  for (let i = 0; i < vals.length; i++) {
                      const val = vals[i];
                      const bytesOfLength = new Uint8Array(new Uint32Array([val.length]).buffer);
                      bytes.set(bytesOfLength, offset);
                      offset += NUM_BYTES_STRING_LENGTH;
                      bytes.set(val, offset);
                      offset += val.length;
                  }
                  resolve(bytes);
              });
              dataPromises.push(utf8bytes);
          }
          else {
              dataPromises.push(t.data());
          }
          if (group != null) {
              spec.group = group;
          }
          specs.push(spec);
      }
      const tensorValues = await Promise.all(dataPromises);
      return { data: concatenateTypedArrays(tensorValues), specs };
  }
  /**
   * Decode flat ArrayBuffer as weights.
   *
   * This function does not handle sharding.
   *
   * This function is the reverse of `encodeWeights`.
   *
   * @param buffer A flat ArrayBuffer carrying the binary values of the tensors
   *   concatenated in the order specified in `specs`.
   * @param specs Specifications of the names, dtypes and shapes of the tensors
   *   whose value are encoded by `buffer`.
   * @return A map from tensor name to tensor value, with the names corresponding
   *   to names in `specs`.
   * @throws Error, if any of the tensors has unsupported dtype.
   */
  function decodeWeights(buffer, specs) {
      // TODO(adarob, cais): Support quantization.
      const out = {};
      let offset = 0;
      for (const spec of specs) {
          const name = spec.name;
          const dtype = spec.dtype;
          const shape = spec.shape;
          const size = sizeFromShape(shape);
          let values;
          if ('quantization' in spec) {
              const quantization = spec.quantization;
              if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
                  throw new Error(`Weight ${spec.name} has unknown ` +
                      `quantization dtype ${quantization.dtype}. ` +
                      `Supported quantization dtypes are: 'uint8' and 'uint16'.`);
              }
              const quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
              const byteBuffer = buffer.slice(offset, offset + size * quantizationSizeFactor);
              const quantizedArray = (quantization.dtype === 'uint8') ?
                  new Uint8Array(byteBuffer) :
                  new Uint16Array(byteBuffer);
              if (dtype === 'float32') {
                  values = Float32Array.from(quantizedArray, v => v * quantization.scale + quantization.min);
              }
              else if (dtype === 'int32') {
                  values = Int32Array.from(quantizedArray, v => Math.round(v * quantization.scale + quantization.min));
              }
              else {
                  throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
              }
              offset += size * quantizationSizeFactor;
          }
          else if (dtype === 'string') {
              const size = sizeFromShape(spec.shape);
              values = [];
              for (let i = 0; i < size; i++) {
                  const byteLength = new Uint32Array(buffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
                  offset += NUM_BYTES_STRING_LENGTH;
                  const bytes = new Uint8Array(buffer.slice(offset, offset + byteLength));
                  values.push(bytes);
                  offset += byteLength;
              }
          }
          else {
              const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
              const byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);
              if (dtype === 'float32') {
                  values = new Float32Array(byteBuffer);
              }
              else if (dtype === 'int32') {
                  values = new Int32Array(byteBuffer);
              }
              else if (dtype === 'bool') {
                  values = new Uint8Array(byteBuffer);
              }
              else {
                  throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
              }
              offset += size * dtypeFactor;
          }
          out[name] = tensor(values, shape, dtype);
      }
      return out;
  }
  /**
   * Concatenate TypedArrays into an ArrayBuffer.
   */
  function concatenateTypedArrays(xs) {
      // TODO(adarob, cais): Support quantization.
      if (xs === null) {
          throw new Error(`Invalid input value: ${JSON.stringify(xs)}`);
      }
      let totalByteLength = 0;
      // `normalizedXs` is here for this reason: a `TypedArray`'s `buffer'
      // can have a different byte length from that of the `TypedArray` itself,
      // for example, when the `TypedArray` is created from an offset in an
      // `ArrayBuffer`. `normliazedXs` holds `TypedArray`s whose `buffer`s match
      // the `TypedArray` in byte length. If an element of `xs` does not show
      // this property, a new `TypedArray` that satisfy this property will be
      // constructed and pushed into `normalizedXs`.
      const normalizedXs = [];
      xs.forEach((x) => {
          totalByteLength += x.byteLength;
          // tslint:disable:no-any
          normalizedXs.push(x.byteLength === x.buffer.byteLength ? x :
              new x.constructor(x));
          if (!(x instanceof Float32Array || x instanceof Int32Array ||
              x instanceof Uint8Array)) {
              throw new Error(`Unsupported TypedArray subtype: ${x.constructor.name}`);
          }
          // tslint:enable:no-any
      });
      const y = new Uint8Array(totalByteLength);
      let offset = 0;
      normalizedXs.forEach((x) => {
          y.set(new Uint8Array(x.buffer), offset);
          offset += x.byteLength;
      });
      return y.buffer;
  }
  // Use Buffer on Node.js instead of Blob/atob/btoa
  const useNodeBuffer = typeof Buffer !== 'undefined' &&
      (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
          typeof btoa === 'undefined');
  /**
   * Calculate the byte length of a JavaScript string.
   *
   * Note that a JavaScript string can contain wide characters, therefore the
   * length of the string is not necessarily equal to the byte length.
   *
   * @param str Input string.
   * @returns Byte length.
   */
  function stringByteLength(str) {
      if (useNodeBuffer) {
          return Buffer.byteLength(str);
      }
      return new Blob([str]).size;
  }
  /**
   * Encode an ArrayBuffer as a base64 encoded string.
   *
   * @param buffer `ArrayBuffer` to be converted.
   * @returns A string that base64-encodes `buffer`.
   */
  function arrayBufferToBase64String(buffer) {
      if (useNodeBuffer) {
          return Buffer.from(buffer).toString('base64');
      }
      const buf = new Uint8Array(buffer);
      let s = '';
      for (let i = 0, l = buf.length; i < l; i++) {
          s += String.fromCharCode(buf[i]);
      }
      return btoa(s);
  }
  /**
   * Decode a base64 string as an ArrayBuffer.
   *
   * @param str Base64 string.
   * @returns Decoded `ArrayBuffer`.
   */
  function base64StringToArrayBuffer(str) {
      if (useNodeBuffer) {
          const buf = Buffer.from(str, 'base64');
          return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      }
      const s = atob(str);
      const buffer = new Uint8Array(s.length);
      for (let i = 0; i < s.length; ++i) {
          buffer.set([s.charCodeAt(i)], i);
      }
      return buffer.buffer;
  }
  /**
   * Concatenate a number of ArrayBuffers into one.
   *
   * @param buffers A number of array buffers to concatenate.
   * @returns Result of concatenating `buffers` in order.
   */
  function concatenateArrayBuffers(buffers) {
      let totalByteLength = 0;
      buffers.forEach((buffer) => {
          totalByteLength += buffer.byteLength;
      });
      const temp = new Uint8Array(totalByteLength);
      let offset = 0;
      buffers.forEach((buffer) => {
          temp.set(new Uint8Array(buffer), offset);
          offset += buffer.byteLength;
      });
      return temp.buffer;
  }
  /**
   * Get the basename of a path.
   *
   * Behaves in a way analogous to Linux's basename command.
   *
   * @param path
   */
  function basename(path) {
      const SEPARATOR = '/';
      path = path.trim();
      while (path.endsWith(SEPARATOR)) {
          path = path.slice(0, path.length - 1);
      }
      const items = path.split(SEPARATOR);
      return items[items.length - 1];
  }
  /**
   * Populate ModelArtifactsInfo fields for a model with JSON topology.
   * @param modelArtifacts
   * @returns A ModelArtifactsInfo object.
   */
  function getModelArtifactsInfoForJSON(modelArtifacts) {
      if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
          throw new Error('Expected JSON model topology, received ArrayBuffer.');
      }
      return {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
          modelTopologyBytes: modelArtifacts.modelTopology == null ?
              0 :
              stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
          weightSpecsBytes: modelArtifacts.weightSpecs == null ?
              0 :
              stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
          weightDataBytes: modelArtifacts.weightData == null ?
              0 :
              modelArtifacts.weightData.byteLength,
      };
  }

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
  class IORouterRegistry {
      constructor() {
          this.saveRouters = [];
          this.loadRouters = [];
      }
      static getInstance() {
          if (IORouterRegistry.instance == null) {
              IORouterRegistry.instance = new IORouterRegistry();
          }
          return IORouterRegistry.instance;
      }
      /**
       * Register a save-handler router.
       *
       * @param saveRouter A function that maps a URL-like string onto an instance
       * of `IOHandler` with the `save` method defined or `null`.
       */
      static registerSaveRouter(saveRouter) {
          IORouterRegistry.getInstance().saveRouters.push(saveRouter);
      }
      /**
       * Register a load-handler router.
       *
       * @param loadRouter A function that maps a URL-like string onto an instance
       * of `IOHandler` with the `load` method defined or `null`.
       */
      static registerLoadRouter(loadRouter) {
          IORouterRegistry.getInstance().loadRouters.push(loadRouter);
      }
      /**
       * Look up IOHandler for saving, given a URL-like string.
       *
       * @param url
       * @returns If only one match is found, an instance of IOHandler with the
       * `save` method defined. If no match is found, `null`.
       * @throws Error, if more than one match is found.
       */
      static getSaveHandlers(url) {
          return IORouterRegistry.getHandlers(url, 'save');
      }
      /**
       * Look up IOHandler for loading, given a URL-like string.
       *
       * @param url
       * @param onProgress Optional, progress callback function, fired periodically
       *   before the load is completed.
       * @returns All valid handlers for `url`, given the currently registered
       *   handler routers.
       */
      static getLoadHandlers(url, onProgress) {
          return IORouterRegistry.getHandlers(url, 'load', onProgress);
      }
      static getHandlers(url, handlerType, onProgress) {
          const validHandlers = [];
          const routers = handlerType === 'load' ?
              IORouterRegistry.getInstance().loadRouters :
              IORouterRegistry.getInstance().saveRouters;
          routers.forEach(router => {
              const handler = router(url, onProgress);
              if (handler !== null) {
                  validHandlers.push(handler);
              }
          });
          return validHandlers;
      }
  }
  const registerSaveRouter = (loudRouter) => IORouterRegistry.registerSaveRouter(loudRouter);
  const registerLoadRouter = (loudRouter) => IORouterRegistry.registerLoadRouter(loudRouter);
  const getSaveHandlers = (url) => IORouterRegistry.getSaveHandlers(url);
  const getLoadHandlers = (url, onProgress) => IORouterRegistry.getLoadHandlers(url, onProgress);

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
  const URL_SCHEME_SUFFIX = '://';
  class ModelStoreManagerRegistry {
      constructor() {
          this.managers = {};
      }
      static getInstance() {
          if (ModelStoreManagerRegistry.instance == null) {
              ModelStoreManagerRegistry.instance = new ModelStoreManagerRegistry();
          }
          return ModelStoreManagerRegistry.instance;
      }
      /**
       * Register a save-handler router.
       *
       * @param saveRouter A function that maps a URL-like string onto an instance
       * of `IOHandler` with the `save` method defined or `null`.
       */
      static registerManager(scheme, manager) {
          assert(scheme != null, () => 'scheme must not be undefined or null.');
          if (scheme.endsWith(URL_SCHEME_SUFFIX)) {
              scheme = scheme.slice(0, scheme.indexOf(URL_SCHEME_SUFFIX));
          }
          assert(scheme.length > 0, () => 'scheme must not be an empty string.');
          const registry = ModelStoreManagerRegistry.getInstance();
          assert(registry.managers[scheme] == null, () => `A model store manager is already registered for scheme '${scheme}'.`);
          registry.managers[scheme] = manager;
      }
      static getManager(scheme) {
          const manager = this.getInstance().managers[scheme];
          if (manager == null) {
              throw new Error(`Cannot find model manager for scheme '${scheme}'`);
          }
          return manager;
      }
      static getSchemes() {
          return Object.keys(this.getInstance().managers);
      }
  }
  /**
   * Helper method for parsing a URL string into a scheme and a path.
   *
   * @param url E.g., 'localstorage://my-model'
   * @returns A dictionary with two fields: scheme and path.
   *   Scheme: e.g., 'localstorage' in the example above.
   *   Path: e.g., 'my-model' in the example above.
   */
  function parseURL(url) {
      if (url.indexOf(URL_SCHEME_SUFFIX) === -1) {
          throw new Error(`The url string provided does not contain a scheme. ` +
              `Supported schemes are: ` +
              `${ModelStoreManagerRegistry.getSchemes().join(',')}`);
      }
      return {
          scheme: url.split(URL_SCHEME_SUFFIX)[0],
          path: url.split(URL_SCHEME_SUFFIX)[1],
      };
  }
  async function cloneModelInternal(sourceURL, destURL, deleteSource = false) {
      assert(sourceURL !== destURL, () => `Old path and new path are the same: '${sourceURL}'`);
      const loadHandlers = IORouterRegistry.getLoadHandlers(sourceURL);
      assert(loadHandlers.length > 0, () => `Copying failed because no load handler is found for source URL ${sourceURL}.`);
      assert(loadHandlers.length < 2, () => `Copying failed because more than one (${loadHandlers.length}) ` +
          `load handlers for source URL ${sourceURL}.`);
      const loadHandler = loadHandlers[0];
      const saveHandlers = IORouterRegistry.getSaveHandlers(destURL);
      assert(saveHandlers.length > 0, () => `Copying failed because no save handler is found for destination ` +
          `URL ${destURL}.`);
      assert(saveHandlers.length < 2, () => `Copying failed because more than one (${loadHandlers.length}) ` +
          `save handlers for destination URL ${destURL}.`);
      const saveHandler = saveHandlers[0];
      const sourceScheme = parseURL(sourceURL).scheme;
      const sourcePath = parseURL(sourceURL).path;
      const sameMedium = sourceScheme === parseURL(sourceURL).scheme;
      const modelArtifacts = await loadHandler.load();
      // If moving within the same storage medium, remove the old model as soon as
      // the loading is done. Without doing this, it is possible that the combined
      // size of the two models will cause the cloning to fail.
      if (deleteSource && sameMedium) {
          await ModelStoreManagerRegistry.getManager(sourceScheme)
              .removeModel(sourcePath);
      }
      const saveResult = await saveHandler.save(modelArtifacts);
      // If moving between mediums, the deletion is done after the save succeeds.
      // This guards against the case in which saving to the destination medium
      // fails.
      if (deleteSource && !sameMedium) {
          await ModelStoreManagerRegistry.getManager(sourceScheme)
              .removeModel(sourcePath);
      }
      return saveResult.modelArtifactsInfo;
  }
  /**
   * List all models stored in registered storage mediums.
   *
   * For a web browser environment, the registered mediums are Local Storage and
   * IndexedDB.
   *
   * ```js
   * // First create and save a model.
   * const model = tf.sequential();
   * model.add(tf.layers.dense(
   *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
   * await model.save('localstorage://demo/management/model1');
   *
   * // Then list existing models.
   * console.log(JSON.stringify(await tf.io.listModels()));
   *
   * // Delete the model.
   * await tf.io.removeModel('localstorage://demo/management/model1');
   *
   * // List models again.
   * console.log(JSON.stringify(await tf.io.listModels()));
   * ```
   *
   * @returns A `Promise` of a dictionary mapping URLs of existing models to
   * their model artifacts info. URLs include medium-specific schemes, e.g.,
   *   'indexeddb://my/model/1'. Model artifacts info include type of the
   * model's topology, byte sizes of the topology, weights, etc.
   */
  /**
   * @doc {
   *   heading: 'Models',
   *   subheading: 'Management',
   *   namespace: 'io',
   *   ignoreCI: true
   * }
   */
  async function listModels() {
      const schemes = ModelStoreManagerRegistry.getSchemes();
      const out = {};
      for (const scheme of schemes) {
          const schemeOut = await ModelStoreManagerRegistry.getManager(scheme).listModels();
          for (const path in schemeOut) {
              const url = scheme + URL_SCHEME_SUFFIX + path;
              out[url] = schemeOut[path];
          }
      }
      return out;
  }
  /**
   * Remove a model specified by URL from a reigstered storage medium.
   *
   * ```js
   * // First create and save a model.
   * const model = tf.sequential();
   * model.add(tf.layers.dense(
   *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
   * await model.save('localstorage://demo/management/model1');
   *
   * // Then list existing models.
   * console.log(JSON.stringify(await tf.io.listModels()));
   *
   * // Delete the model.
   * await tf.io.removeModel('localstorage://demo/management/model1');
   *
   * // List models again.
   * console.log(JSON.stringify(await tf.io.listModels()));
   * ```
   *
   * @param url A URL to a stored model, with a scheme prefix, e.g.,
   *   'localstorage://my-model-1', 'indexeddb://my/model/2'.
   * @returns ModelArtifactsInfo of the deleted model (if and only if deletion
   *   is successful).
   * @throws Error if deletion fails, e.g., if no model exists at `path`.
   */
  /**
   * @doc {
   *   heading: 'Models',
   *   subheading: 'Management',
   *   namespace: 'io',
   *   ignoreCI: true
   * }
   */
  async function removeModel(url) {
      const schemeAndPath = parseURL(url);
      const manager = ModelStoreManagerRegistry.getManager(schemeAndPath.scheme);
      return manager.removeModel(schemeAndPath.path);
  }
  /**
   * Copy a model from one URL to another.
   *
   * This function supports:
   *
   * 1. Copying within a storage medium, e.g.,
   *    `tf.io.copyModel('localstorage://model-1', 'localstorage://model-2')`
   * 2. Copying between two storage mediums, e.g.,
   *    `tf.io.copyModel('localstorage://model-1', 'indexeddb://model-1')`
   *
   * ```js
   * // First create and save a model.
   * const model = tf.sequential();
   * model.add(tf.layers.dense(
   *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
   * await model.save('localstorage://demo/management/model1');
   *
   * // Then list existing models.
   * console.log(JSON.stringify(await tf.io.listModels()));
   *
   * // Copy the model, from Local Storage to IndexedDB.
   * await tf.io.copyModel(
   *     'localstorage://demo/management/model1',
   *     'indexeddb://demo/management/model1');
   *
   * // List models again.
   * console.log(JSON.stringify(await tf.io.listModels()));
   *
   * // Remove both models.
   * await tf.io.removeModel('localstorage://demo/management/model1');
   * await tf.io.removeModel('indexeddb://demo/management/model1');
   * ```
   *
   * @param sourceURL Source URL of copying.
   * @param destURL Destination URL of copying.
   * @returns ModelArtifactsInfo of the copied model (if and only if copying
   *   is successful).
   * @throws Error if copying fails, e.g., if no model exists at `sourceURL`, or
   *   if `oldPath` and `newPath` are identical.
   */
  /**
   * @doc {
   *   heading: 'Models',
   *   subheading: 'Management',
   *   namespace: 'io',
   *   ignoreCI: true
   * }
   */
  async function copyModel(sourceURL, destURL) {
      const deleteSource = false;
      return cloneModelInternal(sourceURL, destURL, deleteSource);
  }
  /**
   * Move a model from one URL to another.
   *
   * This function supports:
   *
   * 1. Moving within a storage medium, e.g.,
   *    `tf.io.moveModel('localstorage://model-1', 'localstorage://model-2')`
   * 2. Moving between two storage mediums, e.g.,
   *    `tf.io.moveModel('localstorage://model-1', 'indexeddb://model-1')`
   *
   * ```js
   * // First create and save a model.
   * const model = tf.sequential();
   * model.add(tf.layers.dense(
   *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
   * await model.save('localstorage://demo/management/model1');
   *
   * // Then list existing models.
   * console.log(JSON.stringify(await tf.io.listModels()));
   *
   * // Move the model, from Local Storage to IndexedDB.
   * await tf.io.moveModel(
   *     'localstorage://demo/management/model1',
   *     'indexeddb://demo/management/model1');
   *
   * // List models again.
   * console.log(JSON.stringify(await tf.io.listModels()));
   *
   * // Remove the moved model.
   * await tf.io.removeModel('indexeddb://demo/management/model1');
   * ```
   *
   * @param sourceURL Source URL of moving.
   * @param destURL Destination URL of moving.
   * @returns ModelArtifactsInfo of the copied model (if and only if copying
   *   is successful).
   * @throws Error if moving fails, e.g., if no model exists at `sourceURL`, or
   *   if `oldPath` and `newPath` are identical.
   */
  /**
   * @doc {
   *   heading: 'Models',
   *   subheading: 'Management',
   *   namespace: 'io',
   *   ignoreCI: true
   * }
   */
  async function moveModel(sourceURL, destURL) {
      const deleteSource = true;
      return cloneModelInternal(sourceURL, destURL, deleteSource);
  }

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
  const DATABASE_NAME = 'tensorflowjs';
  const DATABASE_VERSION = 1;
  // Model data and ModelArtifactsInfo (metadata) are stored in two separate
  // stores for efficient access of the list of stored models and their metadata.
  // 1. The object store for model data: topology, weights and weight manifests.
  const MODEL_STORE_NAME = 'models_store';
  // 2. The object store for ModelArtifactsInfo, including meta-information such
  //    as the type of topology (JSON vs binary), byte size of the topology, byte
  //    size of the weights, etc.
  const INFO_STORE_NAME = 'model_info_store';
  function getIndexedDBFactory() {
      if (!env().getBool('IS_BROWSER')) {
          // TODO(cais): Add more info about what IOHandler subtypes are available.
          //   Maybe point to a doc page on the web and/or automatically determine
          //   the available IOHandlers and print them in the error message.
          throw new Error('Failed to obtain IndexedDB factory because the current environment' +
              'is not a web browser.');
      }
      // tslint:disable-next-line:no-any
      const theWindow = window || self;
      const factory = theWindow.indexedDB || theWindow.mozIndexedDB ||
          theWindow.webkitIndexedDB || theWindow.msIndexedDB ||
          theWindow.shimIndexedDB;
      if (factory == null) {
          throw new Error('The current browser does not appear to support IndexedDB.');
      }
      return factory;
  }
  function setUpDatabase(openRequest) {
      const db = openRequest.result;
      db.createObjectStore(MODEL_STORE_NAME, { keyPath: 'modelPath' });
      db.createObjectStore(INFO_STORE_NAME, { keyPath: 'modelPath' });
  }
  /**
   * IOHandler subclass: Browser IndexedDB.
   *
   * See the doc string of `browserIndexedDB` for more details.
   */
  class BrowserIndexedDB {
      constructor(modelPath) {
          this.indexedDB = getIndexedDBFactory();
          if (modelPath == null || !modelPath) {
              throw new Error('For IndexedDB, modelPath must not be null, undefined or empty.');
          }
          this.modelPath = modelPath;
      }
      async save(modelArtifacts) {
          // TODO(cais): Support saving GraphDef models.
          if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
              throw new Error('BrowserLocalStorage.save() does not support saving model topology ' +
                  'in binary formats yet.');
          }
          return this.databaseAction(this.modelPath, modelArtifacts);
      }
      async load() {
          return this.databaseAction(this.modelPath);
      }
      /**
       * Perform database action to put model artifacts into or read model artifacts
       * from IndexedDB object store.
       *
       * Whether the action is put or get depends on whether `modelArtifacts` is
       * specified. If it is specified, the action will be put; otherwise the action
       * will be get.
       *
       * @param modelPath A unique string path for the model.
       * @param modelArtifacts If specified, it will be the model artifacts to be
       *   stored in IndexedDB.
       * @returns A `Promise` of `SaveResult`, if the action is put, or a `Promise`
       *   of `ModelArtifacts`, if the action is get.
       */
      databaseAction(modelPath, modelArtifacts) {
          return new Promise((resolve, reject) => {
              const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
              openRequest.onupgradeneeded = () => setUpDatabase(openRequest);
              openRequest.onsuccess = () => {
                  const db = openRequest.result;
                  if (modelArtifacts == null) {
                      // Read model out from object store.
                      const modelTx = db.transaction(MODEL_STORE_NAME, 'readonly');
                      const modelStore = modelTx.objectStore(MODEL_STORE_NAME);
                      const getRequest = modelStore.get(this.modelPath);
                      getRequest.onsuccess = () => {
                          if (getRequest.result == null) {
                              db.close();
                              return reject(new Error(`Cannot find model with path '${this.modelPath}' ` +
                                  `in IndexedDB.`));
                          }
                          else {
                              resolve(getRequest.result.modelArtifacts);
                          }
                      };
                      getRequest.onerror = error => {
                          db.close();
                          return reject(getRequest.error);
                      };
                      modelTx.oncomplete = () => db.close();
                  }
                  else {
                      // Put model into object store.
                      const modelArtifactsInfo = getModelArtifactsInfoForJSON(modelArtifacts);
                      // First, put ModelArtifactsInfo into info store.
                      const infoTx = db.transaction(INFO_STORE_NAME, 'readwrite');
                      let infoStore = infoTx.objectStore(INFO_STORE_NAME);
                      const putInfoRequest = infoStore.put({ modelPath: this.modelPath, modelArtifactsInfo });
                      let modelTx;
                      putInfoRequest.onsuccess = () => {
                          // Second, put model data into model store.
                          modelTx = db.transaction(MODEL_STORE_NAME, 'readwrite');
                          const modelStore = modelTx.objectStore(MODEL_STORE_NAME);
                          const putModelRequest = modelStore.put({
                              modelPath: this.modelPath,
                              modelArtifacts,
                              modelArtifactsInfo
                          });
                          putModelRequest.onsuccess = () => resolve({ modelArtifactsInfo });
                          putModelRequest.onerror = error => {
                              // If the put-model request fails, roll back the info entry as
                              // well.
                              infoStore = infoTx.objectStore(INFO_STORE_NAME);
                              const deleteInfoRequest = infoStore.delete(this.modelPath);
                              deleteInfoRequest.onsuccess = () => {
                                  db.close();
                                  return reject(putModelRequest.error);
                              };
                              deleteInfoRequest.onerror = error => {
                                  db.close();
                                  return reject(putModelRequest.error);
                              };
                          };
                      };
                      putInfoRequest.onerror = error => {
                          db.close();
                          return reject(putInfoRequest.error);
                      };
                      infoTx.oncomplete = () => {
                          if (modelTx == null) {
                              db.close();
                          }
                          else {
                              modelTx.oncomplete = () => db.close();
                          }
                      };
                  }
              };
              openRequest.onerror = error => reject(openRequest.error);
          });
      }
  }
  BrowserIndexedDB.URL_SCHEME = 'indexeddb://';
  const indexedDBRouter = (url) => {
      if (!env().getBool('IS_BROWSER')) {
          return null;
      }
      else {
          if (!Array.isArray(url) && url.startsWith(BrowserIndexedDB.URL_SCHEME)) {
              return browserIndexedDB(url.slice(BrowserIndexedDB.URL_SCHEME.length));
          }
          else {
              return null;
          }
      }
  };
  IORouterRegistry.registerSaveRouter(indexedDBRouter);
  IORouterRegistry.registerLoadRouter(indexedDBRouter);
  /**
   * Creates a browser IndexedDB IOHandler for saving and loading models.
   *
   * ```js
   * const model = tf.sequential();
   * model.add(
   *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
   *
   * const saveResult = await model.save('indexeddb://MyModel'));
   * console.log(saveResult);
   * ```
   *
   * @param modelPath A unique identifier for the model to be saved. Must be a
   *   non-empty string.
   * @returns An instance of `BrowserIndexedDB` (sublcass of `IOHandler`),
   *   which can be used with, e.g., `tf.Model.save`.
   */
  function browserIndexedDB(modelPath) {
      return new BrowserIndexedDB(modelPath);
  }
  function maybeStripScheme(key) {
      return key.startsWith(BrowserIndexedDB.URL_SCHEME) ?
          key.slice(BrowserIndexedDB.URL_SCHEME.length) :
          key;
  }
  class BrowserIndexedDBManager {
      constructor() {
          this.indexedDB = getIndexedDBFactory();
      }
      async listModels() {
          return new Promise((resolve, reject) => {
              const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
              openRequest.onupgradeneeded = () => setUpDatabase(openRequest);
              openRequest.onsuccess = () => {
                  const db = openRequest.result;
                  const tx = db.transaction(INFO_STORE_NAME, 'readonly');
                  const store = tx.objectStore(INFO_STORE_NAME);
                  // tslint:disable:max-line-length
                  // Need to cast `store` as `any` here because TypeScript's DOM
                  // library does not have the `getAll()` method even though the
                  // method is supported in the latest version of most mainstream
                  // browsers:
                  // https://developer.mozilla.org/en-US/docs/Web/API/IDBObjectStore/getAll
                  // tslint:enable:max-line-length
                  // tslint:disable-next-line:no-any
                  const getAllInfoRequest = store.getAll();
                  getAllInfoRequest.onsuccess = () => {
                      const out = {};
                      for (const item of getAllInfoRequest.result) {
                          out[item.modelPath] = item.modelArtifactsInfo;
                      }
                      resolve(out);
                  };
                  getAllInfoRequest.onerror = error => {
                      db.close();
                      return reject(getAllInfoRequest.error);
                  };
                  tx.oncomplete = () => db.close();
              };
              openRequest.onerror = error => reject(openRequest.error);
          });
      }
      async removeModel(path) {
          path = maybeStripScheme(path);
          return new Promise((resolve, reject) => {
              const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
              openRequest.onupgradeneeded = () => setUpDatabase(openRequest);
              openRequest.onsuccess = () => {
                  const db = openRequest.result;
                  const infoTx = db.transaction(INFO_STORE_NAME, 'readwrite');
                  const infoStore = infoTx.objectStore(INFO_STORE_NAME);
                  const getInfoRequest = infoStore.get(path);
                  let modelTx;
                  getInfoRequest.onsuccess = () => {
                      if (getInfoRequest.result == null) {
                          db.close();
                          return reject(new Error(`Cannot find model with path '${path}' ` +
                              `in IndexedDB.`));
                      }
                      else {
                          // First, delete the entry in the info store.
                          const deleteInfoRequest = infoStore.delete(path);
                          const deleteModelData = () => {
                              // Second, delete the entry in the model store.
                              modelTx = db.transaction(MODEL_STORE_NAME, 'readwrite');
                              const modelStore = modelTx.objectStore(MODEL_STORE_NAME);
                              const deleteModelRequest = modelStore.delete(path);
                              deleteModelRequest.onsuccess = () => resolve(getInfoRequest.result.modelArtifactsInfo);
                              deleteModelRequest.onerror = error => reject(getInfoRequest.error);
                          };
                          // Proceed with deleting model data regardless of whether deletion
                          // of info data succeeds or not.
                          deleteInfoRequest.onsuccess = deleteModelData;
                          deleteInfoRequest.onerror = error => {
                              deleteModelData();
                              db.close();
                              return reject(getInfoRequest.error);
                          };
                      }
                  };
                  getInfoRequest.onerror = error => {
                      db.close();
                      return reject(getInfoRequest.error);
                  };
                  infoTx.oncomplete = () => {
                      if (modelTx == null) {
                          db.close();
                      }
                      else {
                          modelTx.oncomplete = () => db.close();
                      }
                  };
              };
              openRequest.onerror = error => reject(openRequest.error);
          });
      }
  }
  if (env().getBool('IS_BROWSER')) {
      // Wrap the construction and registration, to guard against browsers that
      // don't support Local Storage.
      try {
          ModelStoreManagerRegistry.registerManager(BrowserIndexedDB.URL_SCHEME, new BrowserIndexedDBManager());
      }
      catch (err) {
      }
  }

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
  const PATH_SEPARATOR = '/';
  const PATH_PREFIX = 'tensorflowjs_models';
  const INFO_SUFFIX = 'info';
  const MODEL_TOPOLOGY_SUFFIX = 'model_topology';
  const WEIGHT_SPECS_SUFFIX = 'weight_specs';
  const WEIGHT_DATA_SUFFIX = 'weight_data';
  const MODEL_METADATA_SUFFIX = 'model_metadata';
  function getModelKeys(path) {
      return {
          info: [PATH_PREFIX, path, INFO_SUFFIX].join(PATH_SEPARATOR),
          topology: [PATH_PREFIX, path, MODEL_TOPOLOGY_SUFFIX].join(PATH_SEPARATOR),
          weightSpecs: [PATH_PREFIX, path, WEIGHT_SPECS_SUFFIX].join(PATH_SEPARATOR),
          weightData: [PATH_PREFIX, path, WEIGHT_DATA_SUFFIX].join(PATH_SEPARATOR),
          modelMetadata: [PATH_PREFIX, path, MODEL_METADATA_SUFFIX].join(PATH_SEPARATOR)
      };
  }
  /**
   * Get model path from a local-storage key.
   *
   * E.g., 'tensorflowjs_models/my/model/1/info' --> 'my/model/1'
   *
   * @param key
   */
  function getModelPathFromKey(key) {
      const items = key.split(PATH_SEPARATOR);
      if (items.length < 3) {
          throw new Error(`Invalid key format: ${key}`);
      }
      return items.slice(1, items.length - 1).join(PATH_SEPARATOR);
  }
  function maybeStripScheme$1(key) {
      return key.startsWith(BrowserLocalStorage.URL_SCHEME) ?
          key.slice(BrowserLocalStorage.URL_SCHEME.length) :
          key;
  }
  /**
   * IOHandler subclass: Browser Local Storage.
   *
   * See the doc string to `browserLocalStorage` for more details.
   */
  class BrowserLocalStorage {
      constructor(modelPath) {
          if (!env().getBool('IS_BROWSER') ||
              typeof window === 'undefined' ||
              typeof window.localStorage === 'undefined') {
              // TODO(cais): Add more info about what IOHandler subtypes are
              // available.
              //   Maybe point to a doc page on the web and/or automatically determine
              //   the available IOHandlers and print them in the error message.
              throw new Error('The current environment does not support local storage.');
          }
          this.LS = window.localStorage;
          if (modelPath == null || !modelPath) {
              throw new Error('For local storage, modelPath must not be null, undefined or empty.');
          }
          this.modelPath = modelPath;
          this.keys = getModelKeys(this.modelPath);
      }
      /**
       * Save model artifacts to browser local storage.
       *
       * See the documentation to `browserLocalStorage` for details on the saved
       * artifacts.
       *
       * @param modelArtifacts The model artifacts to be stored.
       * @returns An instance of SaveResult.
       */
      async save(modelArtifacts) {
          if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
              throw new Error('BrowserLocalStorage.save() does not support saving model topology ' +
                  'in binary formats yet.');
          }
          else {
              const topology = JSON.stringify(modelArtifacts.modelTopology);
              const weightSpecs = JSON.stringify(modelArtifacts.weightSpecs);
              const modelArtifactsInfo = getModelArtifactsInfoForJSON(modelArtifacts);
              try {
                  this.LS.setItem(this.keys.info, JSON.stringify(modelArtifactsInfo));
                  this.LS.setItem(this.keys.topology, topology);
                  this.LS.setItem(this.keys.weightSpecs, weightSpecs);
                  this.LS.setItem(this.keys.weightData, arrayBufferToBase64String(modelArtifacts.weightData));
                  this.LS.setItem(this.keys.modelMetadata, JSON.stringify({
                      format: modelArtifacts.format,
                      generatedBy: modelArtifacts.generatedBy,
                      convertedBy: modelArtifacts.convertedBy,
                      userDefinedMetadata: modelArtifacts.userDefinedMetadata
                  }));
                  return { modelArtifactsInfo };
              }
              catch (err) {
                  // If saving failed, clean up all items saved so far.
                  this.LS.removeItem(this.keys.info);
                  this.LS.removeItem(this.keys.topology);
                  this.LS.removeItem(this.keys.weightSpecs);
                  this.LS.removeItem(this.keys.weightData);
                  this.LS.removeItem(this.keys.modelMetadata);
                  throw new Error(`Failed to save model '${this.modelPath}' to local storage: ` +
                      `size quota being exceeded is a possible cause of this failure: ` +
                      `modelTopologyBytes=${modelArtifactsInfo.modelTopologyBytes}, ` +
                      `weightSpecsBytes=${modelArtifactsInfo.weightSpecsBytes}, ` +
                      `weightDataBytes=${modelArtifactsInfo.weightDataBytes}.`);
              }
          }
      }
      /**
       * Load a model from local storage.
       *
       * See the documentation to `browserLocalStorage` for details on the saved
       * artifacts.
       *
       * @returns The loaded model (if loading succeeds).
       */
      async load() {
          const info = JSON.parse(this.LS.getItem(this.keys.info));
          if (info == null) {
              throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);
          }
          if (info.modelTopologyType !== 'JSON') {
              throw new Error('BrowserLocalStorage does not support loading non-JSON model ' +
                  'topology yet.');
          }
          const out = {};
          // Load topology.
          const topology = JSON.parse(this.LS.getItem(this.keys.topology));
          if (topology == null) {
              throw new Error(`In local storage, the topology of model '${this.modelPath}' ` +
                  `is missing.`);
          }
          out.modelTopology = topology;
          // Load weight specs.
          const weightSpecs = JSON.parse(this.LS.getItem(this.keys.weightSpecs));
          if (weightSpecs == null) {
              throw new Error(`In local storage, the weight specs of model '${this.modelPath}' ` +
                  `are missing.`);
          }
          out.weightSpecs = weightSpecs;
          // Load meta-data fields.
          const metadataString = this.LS.getItem(this.keys.modelMetadata);
          if (metadataString != null) {
              const metadata = JSON.parse(metadataString);
              out.format = metadata['format'];
              out.generatedBy = metadata['generatedBy'];
              out.convertedBy = metadata['convertedBy'];
              out.userDefinedMetadata = metadata['userDefinedMetadata'];
          }
          // Load weight data.
          const weightDataBase64 = this.LS.getItem(this.keys.weightData);
          if (weightDataBase64 == null) {
              throw new Error(`In local storage, the binary weight values of model ` +
                  `'${this.modelPath}' are missing.`);
          }
          out.weightData = base64StringToArrayBuffer(weightDataBase64);
          return out;
      }
  }
  BrowserLocalStorage.URL_SCHEME = 'localstorage://';
  const localStorageRouter = (url) => {
      if (!env().getBool('IS_BROWSER')) {
          return null;
      }
      else {
          if (!Array.isArray(url) && url.startsWith(BrowserLocalStorage.URL_SCHEME)) {
              return browserLocalStorage(url.slice(BrowserLocalStorage.URL_SCHEME.length));
          }
          else {
              return null;
          }
      }
  };
  IORouterRegistry.registerSaveRouter(localStorageRouter);
  IORouterRegistry.registerLoadRouter(localStorageRouter);
  /**
   * Factory function for local storage IOHandler.
   *
   * This `IOHandler` supports both `save` and `load`.
   *
   * For each model's saved artifacts, four items are saved to local storage.
   *   - `${PATH_SEPARATOR}/${modelPath}/info`: Contains meta-info about the
   *     model, such as date saved, type of the topology, size in bytes, etc.
   *   - `${PATH_SEPARATOR}/${modelPath}/topology`: Model topology. For Keras-
   *     style models, this is a stringized JSON.
   *   - `${PATH_SEPARATOR}/${modelPath}/weight_specs`: Weight specs of the
   *     model, can be used to decode the saved binary weight values (see
   *     item below).
   *   - `${PATH_SEPARATOR}/${modelPath}/weight_data`: Concatenated binary
   *     weight values, stored as a base64-encoded string.
   *
   * Saving may throw an `Error` if the total size of the artifacts exceed the
   * browser-specific quota.
   *
   * @param modelPath A unique identifier for the model to be saved. Must be a
   *   non-empty string.
   * @returns An instance of `IOHandler`, which can be used with, e.g.,
   *   `tf.Model.save`.
   */
  function browserLocalStorage(modelPath) {
      return new BrowserLocalStorage(modelPath);
  }
  class BrowserLocalStorageManager {
      constructor() {
          assert(env().getBool('IS_BROWSER'), () => 'Current environment is not a web browser');
          assert(typeof window === 'undefined' ||
              typeof window.localStorage !== 'undefined', () => 'Current browser does not appear to support localStorage');
          this.LS = window.localStorage;
      }
      async listModels() {
          const out = {};
          const prefix = PATH_PREFIX + PATH_SEPARATOR;
          const suffix = PATH_SEPARATOR + INFO_SUFFIX;
          for (let i = 0; i < this.LS.length; ++i) {
              const key = this.LS.key(i);
              if (key.startsWith(prefix) && key.endsWith(suffix)) {
                  const modelPath = getModelPathFromKey(key);
                  out[modelPath] = JSON.parse(this.LS.getItem(key));
              }
          }
          return out;
      }
      async removeModel(path) {
          path = maybeStripScheme$1(path);
          const keys = getModelKeys(path);
          if (this.LS.getItem(keys.info) == null) {
              throw new Error(`Cannot find model at path '${path}'`);
          }
          const info = JSON.parse(this.LS.getItem(keys.info));
          this.LS.removeItem(keys.info);
          this.LS.removeItem(keys.topology);
          this.LS.removeItem(keys.weightSpecs);
          this.LS.removeItem(keys.weightData);
          return info;
      }
  }
  if (env().getBool('IS_BROWSER')) {
      // Wrap the construction and registration, to guard against browsers that
      // don't support Local Storage.
      try {
          ModelStoreManagerRegistry.registerManager(BrowserLocalStorage.URL_SCHEME, new BrowserLocalStorageManager());
      }
      catch (err) {
      }
  }

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
  const DEFAULT_FILE_NAME_PREFIX = 'model';
  const DEFAULT_JSON_EXTENSION_NAME = '.json';
  const DEFAULT_WEIGHT_DATA_EXTENSION_NAME = '.weights.bin';
  function defer(f) {
      return new Promise(resolve => setTimeout(resolve)).then(f);
  }
  class BrowserDownloads {
      constructor(fileNamePrefix) {
          if (!env().getBool('IS_BROWSER')) {
              // TODO(cais): Provide info on what IOHandlers are available under the
              //   current environment.
              throw new Error('browserDownloads() cannot proceed because the current environment ' +
                  'is not a browser.');
          }
          if (fileNamePrefix.startsWith(BrowserDownloads.URL_SCHEME)) {
              fileNamePrefix = fileNamePrefix.slice(BrowserDownloads.URL_SCHEME.length);
          }
          if (fileNamePrefix == null || fileNamePrefix.length === 0) {
              fileNamePrefix = DEFAULT_FILE_NAME_PREFIX;
          }
          this.modelTopologyFileName = fileNamePrefix + DEFAULT_JSON_EXTENSION_NAME;
          this.weightDataFileName =
              fileNamePrefix + DEFAULT_WEIGHT_DATA_EXTENSION_NAME;
      }
      async save(modelArtifacts) {
          if (typeof (document) === 'undefined') {
              throw new Error('Browser downloads are not supported in ' +
                  'this environment since `document` is not present');
          }
          const weightsURL = window.URL.createObjectURL(new Blob([modelArtifacts.weightData], { type: 'application/octet-stream' }));
          if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
              throw new Error('BrowserDownloads.save() does not support saving model topology ' +
                  'in binary formats yet.');
          }
          else {
              const weightsManifest = [{
                      paths: ['./' + this.weightDataFileName],
                      weights: modelArtifacts.weightSpecs
                  }];
              const modelTopologyAndWeightManifest = {
                  modelTopology: modelArtifacts.modelTopology,
                  format: modelArtifacts.format,
                  generatedBy: modelArtifacts.generatedBy,
                  convertedBy: modelArtifacts.convertedBy,
                  weightsManifest
              };
              const modelTopologyAndWeightManifestURL = window.URL.createObjectURL(new Blob([JSON.stringify(modelTopologyAndWeightManifest)], { type: 'application/json' }));
              // If anchor elements are not provided, create them without attaching them
              // to parents, so that the downloaded file names can be controlled.
              const jsonAnchor = this.jsonAnchor == null ? document.createElement('a') :
                  this.jsonAnchor;
              jsonAnchor.download = this.modelTopologyFileName;
              jsonAnchor.href = modelTopologyAndWeightManifestURL;
              // Trigger downloads by evoking a click event on the download anchors.
              // When multiple downloads are started synchronously, Firefox will only
              // save the last one.
              await defer(() => jsonAnchor.dispatchEvent(new MouseEvent('click')));
              if (modelArtifacts.weightData != null) {
                  const weightDataAnchor = this.weightDataAnchor == null ?
                      document.createElement('a') :
                      this.weightDataAnchor;
                  weightDataAnchor.download = this.weightDataFileName;
                  weightDataAnchor.href = weightsURL;
                  await defer(() => weightDataAnchor.dispatchEvent(new MouseEvent('click')));
              }
              return { modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts) };
          }
      }
  }
  BrowserDownloads.URL_SCHEME = 'downloads://';
  class BrowserFiles {
      constructor(files) {
          if (files == null || files.length < 1) {
              throw new Error(`When calling browserFiles, at least 1 file is required, ` +
                  `but received ${files}`);
          }
          this.files = files;
      }
      async load() {
          const jsonFile = this.files[0];
          const weightFiles = this.files.slice(1);
          return new Promise((resolve, reject) => {
              const jsonReader = new FileReader();
              jsonReader.onload = (event) => {
                  // tslint:disable-next-line:no-any
                  const modelJSON = JSON.parse(event.target.result);
                  const modelTopology = modelJSON.modelTopology;
                  if (modelTopology == null) {
                      reject(new Error(`modelTopology field is missing from file ${jsonFile.name}`));
                      return;
                  }
                  if (weightFiles.length === 0) {
                      resolve({ modelTopology });
                  }
                  const weightsManifest = modelJSON.weightsManifest;
                  if (weightsManifest == null) {
                      reject(new Error(`weightManifest field is missing from file ${jsonFile.name}`));
                      return;
                  }
                  let pathToFile;
                  try {
                      pathToFile =
                          this.checkManifestAndWeightFiles(weightsManifest, weightFiles);
                  }
                  catch (err) {
                      reject(err);
                      return;
                  }
                  const weightSpecs = [];
                  const paths = [];
                  const perFileBuffers = [];
                  weightsManifest.forEach(weightsGroup => {
                      weightsGroup.paths.forEach(path => {
                          paths.push(path);
                          perFileBuffers.push(null);
                      });
                      weightSpecs.push(...weightsGroup.weights);
                  });
                  weightsManifest.forEach(weightsGroup => {
                      weightsGroup.paths.forEach(path => {
                          const weightFileReader = new FileReader();
                          weightFileReader.onload = (event) => {
                              // tslint:disable-next-line:no-any
                              const weightData = event.target.result;
                              const index = paths.indexOf(path);
                              perFileBuffers[index] = weightData;
                              if (perFileBuffers.indexOf(null) === -1) {
                                  resolve({
                                      modelTopology,
                                      weightSpecs,
                                      weightData: concatenateArrayBuffers(perFileBuffers),
                                      format: modelJSON.format,
                                      generatedBy: modelJSON.generatedBy,
                                      convertedBy: modelJSON.convertedBy,
                                      userDefinedMetadata: modelJSON.userDefinedMetadata
                                  });
                              }
                          };
                          weightFileReader.onerror = error => reject(`Failed to weights data from file of path '${path}'.`);
                          weightFileReader.readAsArrayBuffer(pathToFile[path]);
                      });
                  });
              };
              jsonReader.onerror = error => reject(`Failed to read model topology and weights manifest JSON ` +
                  `from file '${jsonFile.name}'. BrowserFiles supports loading ` +
                  `Keras-style tf.Model artifacts only.`);
              jsonReader.readAsText(jsonFile);
          });
      }
      /**
       * Check the compatibility between weights manifest and weight files.
       */
      checkManifestAndWeightFiles(manifest, files) {
          const basenames = [];
          const fileNames = files.map(file => basename(file.name));
          const pathToFile = {};
          for (const group of manifest) {
              group.paths.forEach(path => {
                  const pathBasename = basename(path);
                  if (basenames.indexOf(pathBasename) !== -1) {
                      throw new Error(`Duplicate file basename found in weights manifest: ` +
                          `'${pathBasename}'`);
                  }
                  basenames.push(pathBasename);
                  if (fileNames.indexOf(pathBasename) === -1) {
                      throw new Error(`Weight file with basename '${pathBasename}' is not provided.`);
                  }
                  else {
                      pathToFile[path] = files[fileNames.indexOf(pathBasename)];
                  }
              });
          }
          if (basenames.length !== files.length) {
              throw new Error(`Mismatch in the number of files in weights manifest ` +
                  `(${basenames.length}) and the number of weight files provided ` +
                  `(${files.length}).`);
          }
          return pathToFile;
      }
  }
  const browserDownloadsRouter = (url) => {
      if (!env().getBool('IS_BROWSER')) {
          return null;
      }
      else {
          if (!Array.isArray(url) && url.startsWith(BrowserDownloads.URL_SCHEME)) {
              return browserDownloads(url.slice(BrowserDownloads.URL_SCHEME.length));
          }
          else {
              return null;
          }
      }
  };
  IORouterRegistry.registerSaveRouter(browserDownloadsRouter);
  /**
   * Creates an IOHandler that triggers file downloads from the browser.
   *
   * The returned `IOHandler` instance can be used as model exporting methods such
   * as `tf.Model.save` and supports only saving.
   *
   * ```js
   * const model = tf.sequential();
   * model.add(tf.layers.dense(
   *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
   * const saveResult = await model.save('downloads://mymodel');
   * // This will trigger downloading of two files:
   * //   'mymodel.json' and 'mymodel.weights.bin'.
   * console.log(saveResult);
   * ```
   *
   * @param fileNamePrefix Prefix name of the files to be downloaded. For use with
   *   `tf.Model`, `fileNamePrefix` should follow either of the following two
   *   formats:
   *   1. `null` or `undefined`, in which case the default file
   *      names will be used:
   *      - 'model.json' for the JSON file containing the model topology and
   *        weights manifest.
   *      - 'model.weights.bin' for the binary file containing the binary weight
   *        values.
   *   2. A single string or an Array of a single string, as the file name prefix.
   *      For example, if `'foo'` is provided, the downloaded JSON
   *      file and binary weights file will be named 'foo.json' and
   *      'foo.weights.bin', respectively.
   * @param config Additional configuration for triggering downloads.
   * @returns An instance of `BrowserDownloads` `IOHandler`.
   */
  /**
   * @doc {
   *   heading: 'Models',
   *   subheading: 'Loading',
   *   namespace: 'io',
   *   ignoreCI: true
   * }
   */
  function browserDownloads(fileNamePrefix = 'model') {
      return new BrowserDownloads(fileNamePrefix);
  }
  /**
   * Creates an IOHandler that loads model artifacts from user-selected files.
   *
   * This method can be used for loading from files such as user-selected files
   * in the browser.
   * When used in conjunction with `tf.loadLayersModel`, an instance of
   * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
   *
   * ```js
   * // Note: This code snippet won't run properly without the actual file input
   * //   elements in the HTML DOM.
   *
   * // Suppose there are two HTML file input (`<input type="file" ...>`)
   * // elements.
   * const uploadJSONInput = document.getElementById('upload-json');
   * const uploadWeightsInput = document.getElementById('upload-weights');
   * const model = await tf.loadLayersModel(tf.io.browserFiles(
   *     [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
   * ```
   *
   * @param files `File`s to load from. Currently, this function supports only
   *   loading from files that contain Keras-style models (i.e., `tf.Model`s), for
   *   which an `Array` of `File`s is expected (in that order):
   *   - A JSON file containing the model topology and weight manifest.
   *   - Optionally, One or more binary files containing the binary weights.
   *     These files must have names that match the paths in the `weightsManifest`
   *     contained by the aforementioned JSON file, or errors will be thrown
   *     during loading. These weights files have the same format as the ones
   *     generated by `tensorflowjs_converter` that comes with the `tensorflowjs`
   *     Python PIP package. If no weights files are provided, only the model
   *     topology will be loaded from the JSON file above.
   * @returns An instance of `Files` `IOHandler`.
   */
  /**
   * @doc {
   *   heading: 'Models',
   *   subheading: 'Loading',
   *   namespace: 'io',
   *   ignoreCI: true
   * }
   */
  function browserFiles(files) {
      return new BrowserFiles(files);
  }

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
  /**
   * Monitor Promise.all progress, fire onProgress callback function.
   *
   * @param promises Promise list going to be monitored
   * @param onProgress Callback function. Fired when a promise resolved.
   * @param startFraction Optional fraction start. Default to 0.
   * @param endFraction Optional fraction end. Default to 1.
   */
  function monitorPromisesProgress(promises, onProgress, startFraction, endFraction) {
      checkPromises(promises);
      startFraction = startFraction == null ? 0 : startFraction;
      endFraction = endFraction == null ? 1 : endFraction;
      checkFraction(startFraction, endFraction);
      let resolvedPromise = 0;
      const registerMonitor = (promise) => {
          promise.then(value => {
              const fraction = startFraction +
                  ++resolvedPromise / promises.length * (endFraction - startFraction);
              // pass fraction as parameter to callback function.
              onProgress(fraction);
              return value;
          });
          return promise;
      };
      function checkPromises(promises) {
          assert(promises != null && Array.isArray(promises) && promises.length > 0, () => 'promises must be a none empty array');
      }
      function checkFraction(startFraction, endFraction) {
          assert(startFraction >= 0 && startFraction <= 1, () => `Progress fraction must be in range [0, 1], but ` +
              `got startFraction ${startFraction}`);
          assert(endFraction >= 0 && endFraction <= 1, () => `Progress fraction must be in range [0, 1], but ` +
              `got endFraction ${endFraction}`);
          assert(endFraction >= startFraction, () => `startFraction must be no more than endFraction, but ` +
              `got startFraction ${startFraction} and endFraction ` +
              `${endFraction}`);
      }
      return Promise.all(promises.map(registerMonitor));
  }

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
  /**
   * Reads binary weights data from a number of URLs.
   *
   * @param fetchURLs URLs to send the HTTP requests at, using `fetch` calls.
   * @param requestOptions RequestInit (options) for the HTTP requests.
   * @param fetchFunc Optional overriding value for the `window.fetch` function.
   * @param onProgress Optional, progress callback function, fired periodically
   *   before the load is completed.
   * @returns A `Promise` of an Array of `ArrayBuffer`. The Array has the same
   *   length as `fetchURLs`.
   */
  async function loadWeightsAsArrayBuffer(fetchURLs, loadOptions) {
      if (loadOptions == null) {
          loadOptions = {};
      }
      const fetchFunc = loadOptions.fetchFunc == null ? env().platform.fetch :
          loadOptions.fetchFunc;
      // Create the requests for all of the weights in parallel.
      const requests = fetchURLs.map(fetchURL => fetchFunc(fetchURL, loadOptions.requestInit, { isBinary: true }));
      const fetchStartFraction = 0;
      const fetchEndFraction = 0.5;
      const responses = loadOptions.onProgress == null ?
          await Promise.all(requests) :
          await monitorPromisesProgress(requests, loadOptions.onProgress, fetchStartFraction, fetchEndFraction);
      const bufferPromises = responses.map(response => response.arrayBuffer());
      const bufferStartFraction = 0.5;
      const bufferEndFraction = 1;
      const buffers = loadOptions.onProgress == null ?
          await Promise.all(bufferPromises) :
          await monitorPromisesProgress(bufferPromises, loadOptions.onProgress, bufferStartFraction, bufferEndFraction);
      return buffers;
  }
  /**
   * Reads a weights manifest JSON configuration, fetches the weights and
   * returns them as `Tensor`s.
   *
   * @param manifest The weights manifest JSON.
   * @param filePathPrefix The path prefix for filenames given in the manifest.
   *     Defaults to the empty string.
   * @param weightNames The names of the weights to be fetched.
   */
  async function loadWeights(manifest, filePathPrefix = '', weightNames, requestInit) {
      // TODO(nsthorat): Groups are currently fetched atomically. If you need a
      // single weight from a group, the whole group will be fetched. At a future
      // date, we should support fetching only the individual shards within a
      // group that are needed to reconstruct the requested weight.
      // TODO(cais): Use `decodeWeights` for implementation.
      const fetchWeights = (fetchUrls) => loadWeightsAsArrayBuffer(fetchUrls, { requestInit });
      const loadWeights = weightsLoaderFactory(fetchWeights);
      return loadWeights(manifest, filePathPrefix, weightNames);
  }
  /**
   * Creates a function, which reads a weights manifest JSON configuration,
   * fetches the weight files using the specified function and returns them as
   * `Tensor`s.
   *
   * ```js
   * // example for creating a nodejs weight loader, which reads the weight files
   * // from disk using fs.readFileSync
   *
   * import * as fs from 'fs'
   *
   * const fetchWeightsFromDisk = (filePaths: string[]) =>
   *   filePaths.map(filePath => fs.readFileSync(filePath).buffer)
   *
   * const loadWeights = tf.io.weightsLoaderFactory(fetchWeightsFromDisk)
   *
   * const manifest = JSON.parse(
   *   fs.readFileSync('./my_model-weights_manifest').toString()
   * )
   * const weightMap = await loadWeights(manifest, './')
   * ```
   * @param fetchWeightsFunction The function used for fetching the weight files.
   * @returns Weight loading function.
   */
  function weightsLoaderFactory(fetchWeightsFunction) {
      return async (manifest, filePathPrefix = '', weightNames) => {
          // Collect all the groups, weights, and their relative offsets to be
          // fetched.
          const groupIndicesToFetchMap = manifest.map(() => false);
          const groupWeightsToFetch = {};
          const weightsFound = weightNames != null ? weightNames.map(() => false) : [];
          const allManifestWeightNames = [];
          manifest.forEach((manifestGroupConfig, groupIndex) => {
              let groupOffset = 0;
              manifestGroupConfig.weights.forEach(weightsEntry => {
                  const rawDtype = ('quantization' in weightsEntry) ?
                      weightsEntry.quantization.dtype :
                      weightsEntry.dtype;
                  const weightsBytes = DTYPE_VALUE_SIZE_MAP[rawDtype] *
                      sizeFromShape(weightsEntry.shape);
                  const enqueueWeightsForFetchingFn = () => {
                      groupIndicesToFetchMap[groupIndex] = true;
                      if (groupWeightsToFetch[groupIndex] == null) {
                          groupWeightsToFetch[groupIndex] = [];
                      }
                      groupWeightsToFetch[groupIndex].push({
                          manifestEntry: weightsEntry,
                          groupOffset,
                          sizeBytes: weightsBytes
                      });
                  };
                  if (weightNames != null) {
                      weightNames.forEach((weightName, weightIndex) => {
                          if (weightName === weightsEntry.name) {
                              enqueueWeightsForFetchingFn();
                              weightsFound[weightIndex] = true;
                          }
                      });
                  }
                  else {
                      enqueueWeightsForFetchingFn();
                  }
                  allManifestWeightNames.push(weightsEntry.name);
                  groupOffset += weightsBytes;
              });
          });
          if (!weightsFound.every(found => found)) {
              const weightsNotFound = weightNames.filter((_, i) => !weightsFound[i]);
              throw new Error(`Could not find weights in manifest with names: ` +
                  `${weightsNotFound.join(', ')}. \n` +
                  `Manifest JSON has weights with names: ` +
                  `${allManifestWeightNames.join(', ')}.`);
          }
          // Convert the one-hot boolean groupId => shouldFetch map to a list of group
          // IDs.
          const groupIndicesToFetch = groupIndicesToFetchMap.reduce((accumulator, shouldFetch, i) => {
              if (shouldFetch) {
                  accumulator.push(i);
              }
              return accumulator;
          }, []);
          const fetchUrls = [];
          groupIndicesToFetch.forEach(i => {
              manifest[i].paths.forEach(filepath => {
                  const fetchUrl = filePathPrefix +
                      (!filePathPrefix.endsWith('/') ? '/' : '') + filepath;
                  fetchUrls.push(fetchUrl);
              });
          });
          const buffers = await fetchWeightsFunction(fetchUrls);
          const weightsTensorMap = {};
          let bufferIndexOffset = 0;
          groupIndicesToFetch.forEach(i => {
              const numBuffers = manifest[i].paths.length;
              let groupBytes = 0;
              for (let i = 0; i < numBuffers; i++) {
                  groupBytes += buffers[bufferIndexOffset + i].byteLength;
              }
              // Create a buffer for the whole group.
              const groupBuffer = new ArrayBuffer(groupBytes);
              const groupByteBuffer = new Uint8Array(groupBuffer);
              let groupBufferOffset = 0;
              for (let i = 0; i < numBuffers; i++) {
                  const buffer = new Uint8Array(buffers[bufferIndexOffset + i]);
                  groupByteBuffer.set(buffer, groupBufferOffset);
                  groupBufferOffset += buffer.byteLength;
              }
              const weightsEntries = groupWeightsToFetch[i];
              weightsEntries.forEach(weightsEntry => {
                  const byteBuffer = groupBuffer.slice(weightsEntry.groupOffset, weightsEntry.groupOffset + weightsEntry.sizeBytes);
                  const nameToTensorMap = decodeWeights(byteBuffer, [weightsEntry.manifestEntry]);
                  for (const name in nameToTensorMap) {
                      weightsTensorMap[name] = nameToTensorMap[name];
                  }
              });
              bufferIndexOffset += numBuffers;
          });
          return weightsTensorMap;
      };
  }

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
  const OCTET_STREAM_MIME_TYPE = 'application/octet-stream';
  const JSON_TYPE = 'application/json';
  class HTTPRequest {
      constructor(path, loadOptions) {
          this.DEFAULT_METHOD = 'POST';
          if (loadOptions == null) {
              loadOptions = {};
          }
          this.weightPathPrefix = loadOptions.weightPathPrefix;
          this.onProgress = loadOptions.onProgress;
          if (loadOptions.fetchFunc != null) {
              assert(typeof loadOptions.fetchFunc === 'function', () => 'Must pass a function that matches the signature of ' +
                  '`fetch` (see ' +
                  'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)');
              this.fetch = loadOptions.fetchFunc;
          }
          else {
              this.fetch = env().platform.fetch;
          }
          assert(path != null && path.length > 0, () => 'URL path for http must not be null, undefined or ' +
              'empty.');
          if (Array.isArray(path)) {
              assert(path.length === 2, () => 'URL paths for http must have a length of 2, ' +
                  `(actual length is ${path.length}).`);
          }
          this.path = path;
          if (loadOptions.requestInit != null &&
              loadOptions.requestInit.body != null) {
              throw new Error('requestInit is expected to have no pre-existing body, but has one.');
          }
          this.requestInit = loadOptions.requestInit || {};
      }
      async save(modelArtifacts) {
          if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
              throw new Error('BrowserHTTPRequest.save() does not support saving model topology ' +
                  'in binary formats yet.');
          }
          const init = Object.assign({ method: this.DEFAULT_METHOD }, this.requestInit);
          init.body = new FormData();
          const weightsManifest = [{
                  paths: ['./model.weights.bin'],
                  weights: modelArtifacts.weightSpecs,
              }];
          const modelTopologyAndWeightManifest = {
              modelTopology: modelArtifacts.modelTopology,
              format: modelArtifacts.format,
              generatedBy: modelArtifacts.generatedBy,
              convertedBy: modelArtifacts.convertedBy,
              userDefinedMetadata: modelArtifacts.userDefinedMetadata,
              weightsManifest
          };
          init.body.append('model.json', new Blob([JSON.stringify(modelTopologyAndWeightManifest)], { type: JSON_TYPE }), 'model.json');
          if (modelArtifacts.weightData != null) {
              init.body.append('model.weights.bin', new Blob([modelArtifacts.weightData], { type: OCTET_STREAM_MIME_TYPE }), 'model.weights.bin');
          }
          const response = await this.fetch(this.path, init);
          if (response.ok) {
              return {
                  modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts),
                  responses: [response],
              };
          }
          else {
              throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ` +
                  `${response.status}.`);
          }
      }
      /**
       * Load model artifacts via HTTP request(s).
       *
       * See the documentation to `tf.io.http` for details on the saved
       * artifacts.
       *
       * @returns The loaded model artifacts (if loading succeeds).
       */
      async load() {
          const modelConfigRequest = await this.fetch(this.path, this.requestInit);
          if (!modelConfigRequest.ok) {
              throw new Error(`Request to ${this.path} failed with status code ` +
                  `${modelConfigRequest.status}. Please verify this URL points to ` +
                  `the model JSON of the model to load.`);
          }
          let modelConfig;
          try {
              modelConfig = await modelConfigRequest.json();
          }
          catch (e) {
              let message = `Failed to parse model JSON of response from ${this.path}.`;
              // TODO(nsthorat): Remove this after some time when we're comfortable that
              // .pb files are mostly gone.
              if (this.path.endsWith('.pb')) {
                  message += ' Your path contains a .pb file extension. ' +
                      'Support for .pb models have been removed in TensorFlow.js 1.0 ' +
                      'in favor of .json models. You can re-convert your Python ' +
                      'TensorFlow model using the TensorFlow.js 1.0 conversion scripts ' +
                      'or you can convert your.pb models with the \'pb2json\'' +
                      'NPM script in the tensorflow/tfjs-converter repository.';
              }
              else {
                  message += ' Please make sure the server is serving valid ' +
                      'JSON for this request.';
              }
              throw new Error(message);
          }
          const modelTopology = modelConfig.modelTopology;
          const weightsManifest = modelConfig.weightsManifest;
          const generatedBy = modelConfig.generatedBy;
          const convertedBy = modelConfig.convertedBy;
          const format = modelConfig.format;
          const userDefinedMetadata = modelConfig.userDefinedMetadata;
          // We do not allow both modelTopology and weightsManifest to be missing.
          if (modelTopology == null && weightsManifest == null) {
              throw new Error(`The JSON from HTTP path ${this.path} contains neither model ` +
                  `topology or manifest for weights.`);
          }
          let weightSpecs;
          let weightData;
          if (weightsManifest != null) {
              const results = await this.loadWeights(weightsManifest);
              [weightSpecs, weightData] = results;
          }
          return {
              modelTopology,
              weightSpecs,
              weightData,
              userDefinedMetadata,
              generatedBy,
              convertedBy,
              format
          };
      }
      async loadWeights(weightsManifest) {
          const weightPath = Array.isArray(this.path) ? this.path[1] : this.path;
          const [prefix, suffix] = parseUrl(weightPath);
          const pathPrefix = this.weightPathPrefix || prefix;
          const weightSpecs = [];
          for (const entry of weightsManifest) {
              weightSpecs.push(...entry.weights);
          }
          const fetchURLs = [];
          weightsManifest.forEach(weightsGroup => {
              weightsGroup.paths.forEach(path => {
                  fetchURLs.push(pathPrefix + path + suffix);
              });
          });
          const buffers = await loadWeightsAsArrayBuffer(fetchURLs, {
              requestInit: this.requestInit,
              fetchFunc: this.fetch,
              onProgress: this.onProgress
          });
          return [weightSpecs, concatenateArrayBuffers(buffers)];
      }
  }
  HTTPRequest.URL_SCHEME_REGEX = /^https?:\/\//;
  /**
   * Extract the prefix and suffix of the url, where the prefix is the path before
   * the last file, and suffix is the search params after the last file.
   * ```
   * const url = 'http://tfhub.dev/model/1/tensorflowjs_model.pb?tfjs-format=file'
   * [prefix, suffix] = parseUrl(url)
   * // prefix = 'http://tfhub.dev/model/1/'
   * // suffix = '?tfjs-format=file'
   * ```
   * @param url the model url to be parsed.
   */
  function parseUrl(url) {
      const lastSlash = url.lastIndexOf('/');
      const lastSearchParam = url.lastIndexOf('?');
      const prefix = url.substring(0, lastSlash);
      const suffix = lastSearchParam > lastSlash ? url.substring(lastSearchParam) : '';
      return [prefix + '/', suffix];
  }
  function isHTTPScheme(url) {
      return url.match(HTTPRequest.URL_SCHEME_REGEX) != null;
  }
  const httpRouter = (url, onProgress) => {
      if (typeof fetch === 'undefined') {
          // `http` uses `fetch` or `node-fetch`, if one wants to use it in
          // an environment that is not the browser or node they have to setup a
          // global fetch polyfill.
          return null;
      }
      else {
          let isHTTP = true;
          if (Array.isArray(url)) {
              isHTTP = url.every(urlItem => isHTTPScheme(urlItem));
          }
          else {
              isHTTP = isHTTPScheme(url);
          }
          if (isHTTP) {
              return http(url, { onProgress });
          }
      }
      return null;
  };
  IORouterRegistry.registerSaveRouter(httpRouter);
  IORouterRegistry.registerLoadRouter(httpRouter);
  /**
   * Creates an IOHandler subtype that sends model artifacts to HTTP server.
   *
   * An HTTP request of the `multipart/form-data` mime type will be sent to the
   * `path` URL. The form data includes artifacts that represent the topology
   * and/or weights of the model. In the case of Keras-style `tf.Model`, two
   * blobs (files) exist in form-data:
   *   - A JSON file consisting of `modelTopology` and `weightsManifest`.
   *   - A binary weights file consisting of the concatenated weight values.
   * These files are in the same format as the one generated by
   * [tfjs_converter](https://js.tensorflow.org/tutorials/import-keras.html).
   *
   * The following code snippet exemplifies the client-side code that uses this
   * function:
   *
   * ```js
   * const model = tf.sequential();
   * model.add(
   *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
   *
   * const saveResult = await model.save(tf.io.http(
   *     'http://model-server:5000/upload', {requestInit: {method: 'PUT'}}));
   * console.log(saveResult);
   * ```
   *
   * If the default `POST` method is to be used, without any custom parameters
   * such as headers, you can simply pass an HTTP or HTTPS URL to `model.save`:
   *
   * ```js
   * const saveResult = await model.save('http://model-server:5000/upload');
   * ```
   *
   * The following GitHub Gist
   * https://gist.github.com/dsmilkov/1b6046fd6132d7408d5257b0976f7864
   * implements a server based on [flask](https://github.com/pallets/flask) that
   * can receive the request. Upon receiving the model artifacts via the requst,
   * this particular server reconsistutes instances of [Keras
   * Models](https://keras.io/models/model/) in memory.
   *
   *
   * @param path A URL path to the model.
   *   Can be an absolute HTTP path (e.g.,
   *   'http://localhost:8000/model-upload)') or a relative path (e.g.,
   *   './model-upload').
   * @param requestInit Request configurations to be used when sending
   *    HTTP request to server using `fetch`. It can contain fields such as
   *    `method`, `credentials`, `headers`, `mode`, etc. See
   *    https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
   *    for more information. `requestInit` must not have a body, because the
   * body will be set by TensorFlow.js. File blobs representing the model
   * topology (filename: 'model.json') and the weights of the model (filename:
   * 'model.weights.bin') will be appended to the body. If `requestInit` has a
   * `body`, an Error will be thrown.
   * @param loadOptions Optional configuration for the loading. It includes the
   *   following fields:
   *   - weightPathPrefix Optional, this specifies the path prefix for weight
   *     files, by default this is calculated from the path param.
   *   - fetchFunc Optional, custom `fetch` function. E.g., in Node.js,
   *     the `fetch` from node-fetch can be used here.
   *   - onProgress Optional, progress callback function, fired periodically
   *     before the load is completed.
   * @returns An instance of `IOHandler`.
   */
  /**
   * @doc {
   *   heading: 'Models',
   *   subheading: 'Loading',
   *   namespace: 'io',
   *   ignoreCI: true
   * }
   */
  function http(path, loadOptions) {
      return new HTTPRequest(path, loadOptions);
  }
  /**
   * Deprecated. Use `tf.io.http`.
   * @param path
   * @param loadOptions
   */
  function browserHTTPRequest(path, loadOptions) {
      return http(path, loadOptions);
  }

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
  class PassthroughLoader {
      constructor(modelArtifacts) {
          this.modelArtifacts = modelArtifacts;
      }
      async load() {
          return this.modelArtifacts;
      }
  }
  class PassthroughSaver {
      constructor(saveHandler) {
          this.saveHandler = saveHandler;
      }
      async save(modelArtifacts) {
          return this.saveHandler(modelArtifacts);
      }
  }
  /**
   * Creates an IOHandler that loads model artifacts from memory.
   *
   * When used in conjunction with `tf.loadLayersModel`, an instance of
   * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
   *
   * ```js
   * const model = await tf.loadLayersModel(tf.io.fromMemory(
   *     modelTopology, weightSpecs, weightData));
   * ```
   *
   * @param modelArtifacts a object containing model topology (i.e., parsed from
   *   the JSON format).
   * @param weightSpecs An array of `WeightsManifestEntry` objects describing the
   *   names, shapes, types, and quantization of the weight data.
   * @param weightData A single `ArrayBuffer` containing the weight data,
   *   concatenated in the order described by the weightSpecs.
   * @param trainingConfig Model training configuration. Optional.
   *
   * @returns A passthrough `IOHandler` that simply loads the provided data.
   */
  function fromMemory(modelArtifacts, weightSpecs, weightData, trainingConfig) {
      if (arguments.length === 1) {
          const isModelArtifacts = modelArtifacts.modelTopology != null ||
              modelArtifacts.weightSpecs != null;
          if (isModelArtifacts) {
              return new PassthroughLoader(modelArtifacts);
          }
          else {
              // Legacy support: with only modelTopology.
              // TODO(cais): Remove this deprecated API.
              console.warn('Please call tf.io.fromMemory() with only one argument. ' +
                  'The argument should be of type ModelArtifacts. ' +
                  'The multi-argument signature of tf.io.fromMemory() has been ' +
                  'deprecated and will be removed in a future release.');
              return new PassthroughLoader({ modelTopology: modelArtifacts });
          }
      }
      else {
          // Legacy support.
          // TODO(cais): Remove this deprecated API.
          console.warn('Please call tf.io.fromMemory() with only one argument. ' +
              'The argument should be of type ModelArtifacts. ' +
              'The multi-argument signature of tf.io.fromMemory() has been ' +
              'deprecated and will be removed in a future release.');
          return new PassthroughLoader({
              modelTopology: modelArtifacts,
              weightSpecs,
              weightData,
              trainingConfig
          });
      }
  }
  /**
   * Creates an IOHandler that passes saved model artifacts to a callback.
   *
   * ```js
   * function handleSave(artifacts) {
   *   // ... do something with the artifacts ...
   *   return {modelArtifactsInfo: {...}, ...};
   * }
   *
   * const saveResult = model.save(tf.io.withSaveHandler(handleSave));
   * ```
   *
   * @param saveHandler A function that accepts a `ModelArtifacts` and returns a
   *     `SaveResult`.
   */
  function withSaveHandler(saveHandler) {
      return new PassthroughSaver(saveHandler);
  }

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

  var io = /*#__PURE__*/Object.freeze({
    __proto__: null,
    browserFiles: browserFiles,
    browserHTTPRequest: browserHTTPRequest,
    concatenateArrayBuffers: concatenateArrayBuffers,
    decodeWeights: decodeWeights,
    encodeWeights: encodeWeights,
    fromMemory: fromMemory,
    getLoadHandlers: getLoadHandlers,
    getModelArtifactsInfoForJSON: getModelArtifactsInfoForJSON,
    getSaveHandlers: getSaveHandlers,
    http: http,
    isHTTPScheme: isHTTPScheme,
    loadWeights: loadWeights,
    registerLoadRouter: registerLoadRouter,
    registerSaveRouter: registerSaveRouter,
    weightsLoaderFactory: weightsLoaderFactory,
    withSaveHandler: withSaveHandler,
    copyModel: copyModel,
    listModels: listModels,
    moveModel: moveModel,
    removeModel: removeModel
  });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Creates a one-hot `tf.Tensor`. The locations represented by `indices` take
   * value `onValue` (defaults to 1), while all other locations take value
   * `offValue` (defaults to 0). If `indices` is rank `R`, the output has rank
   * `R+1` with the last axis of size `depth`.
   *
   * ```js
   * tf.oneHot(tf.tensor1d([0, 1], 'int32'), 3).print();
   * ```
   *
   * @param indices `tf.Tensor` of indices with dtype `int32`.
   * @param depth The depth of the one hot dimension.
   * @param onValue A number used to fill in the output when the index matches
   * the location.
   * @param offValue A number used to fill in the output when the index does
   *     not match the location.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function oneHot_(indices, depth, onValue = 1, offValue = 0) {
      if (depth < 2) {
          throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
      }
      let $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');
      const outShape = [...$indices.shape, depth];
      $indices = $indices.flatten();
      const forward = (backend, save) => {
          save([$indices]);
          return reshape(backend.oneHot($indices, depth, onValue, offValue), outShape);
      };
      const inputs = { indices: $indices };
      const attrs = { depth, onValue, offValue };
      return ENGINE.runKernelFunc(forward, inputs, null /* grad */, OneHot, attrs);
  }
  const oneHot = op({ oneHot_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes the confusion matrix from true labels and predicted labels.
   *
   * ```js
   * const labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
   * const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
   * const numClasses = 3;
   * const out = tf.math.confusionMatrix(labels, predictions, numClasses);
   * out.print();
   * // Expected output matrix:
   * // [[2, 0, 0],
   * //  [0, 1, 1],
   * //  [0, 0, 1]]
   * ```
   *
   * @param labels The target labels, assumed to be 0-based integers
   *   for the classes. The shape is `[numExamples]`, where
   *   `numExamples` is the number of examples included.
   * @param predictions The predicted classes, assumed to be
   *   0-based integers for the classes. Must have the same shape as `labels`.
   * @param numClasses Number of all classes, as an integer.
   *   Its value must be larger than the largest element in `labels` and
   *   `predictions`.
   * @returns The confusion matrix as a int32-type 2D tensor. The value at
   *   row `r` and column `c` is the number of times examples of actual class
   *   `r` were predicted as class `c`.
   */
  /** @doc {heading: 'Operations', subheading: 'Evaluation'} */
  function confusionMatrix_(labels, predictions, numClasses) {
      const $labels = convertToTensor(labels, 'labels', 'confusionMatrix');
      const $predictions = convertToTensor(predictions, 'predictions', 'confusionMatrix');
      assert(numClasses == null || numClasses > 0 && Number.isInteger(numClasses), () => `If provided, numClasses must be a positive integer, ` +
          `but got ${numClasses}`);
      assert($labels.rank === 1, () => `Expected the rank of labels to be 1, but got ${$labels.rank}`);
      assert($predictions.rank === 1, () => `Expected the rank of predictions to be 1, ` +
          `but got ${$predictions.rank}`);
      assert($labels.shape[0] === $predictions.shape[0], () => `Mismatch in the number of examples: ` +
          `${$labels.shape[0]} vs. ${$predictions.shape[0]}. ` +
          `Labels and predictions should have the same number of elements.`);
      assert(numClasses > 0 && Number.isInteger(numClasses), () => `numClasses is required to be a positive integer, but got ` +
          `${numClasses}`);
      // TODO(cais): In the future, if oneHot supports tensors inputs for
      //   `numClasses`, `confusionMatrix` can make `numClasses` optional.
      const oneHotLabels = oneHot($labels.asType('int32'), numClasses);
      const oneHotPredictions = oneHot($predictions.asType('int32'), numClasses);
      const oneHotLabelsT = oneHotLabels.transpose();
      return oneHotLabelsT.matMul(oneHotPredictions).asType('int32');
  }
  const confusionMatrix = op({ confusionMatrix_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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

  var math = /*#__PURE__*/Object.freeze({
    __proto__: null,
    confusionMatrix: confusionMatrix
  });

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
  let fromPixels2DContext;
  /**
   * Creates a `tf.Tensor` from an image.
   *
   * ```js
   * const image = new ImageData(1, 1);
   * image.data[0] = 100;
   * image.data[1] = 150;
   * image.data[2] = 200;
   * image.data[3] = 255;
   *
   * tf.browser.fromPixels(image).print();
   * ```
   *
   * @param pixels The input image to construct the tensor from. The
   * supported image types are all 4-channel. You can also pass in an image
   * object with following attributes:
   * `{data: Uint8Array; width: number; height: number}`
   * @param numChannels The number of channels of the output tensor. A
   * numChannels value less than 4 allows you to ignore channels. Defaults to
   * 3 (ignores alpha channel of input image).
   */
  /** @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true} */
  function fromPixels_(pixels, numChannels = 3) {
      // Sanity checks.
      if (numChannels > 4) {
          throw new Error('Cannot construct Tensor with more than 4 channels from pixels.');
      }
      if (pixels == null) {
          throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
      }
      let isPixelData = false;
      let isImageData = false;
      let isVideo = false;
      let isImage = false;
      let isCanvasLike = false;
      if (pixels.data instanceof Uint8Array) {
          isPixelData = true;
      }
      else if (typeof (ImageData) !== 'undefined' && pixels instanceof ImageData) {
          isImageData = true;
      }
      else if (typeof (HTMLVideoElement) !== 'undefined' &&
          pixels instanceof HTMLVideoElement) {
          isVideo = true;
      }
      else if (typeof (HTMLImageElement) !== 'undefined' &&
          pixels instanceof HTMLImageElement) {
          isImage = true;
          // tslint:disable-next-line: no-any
      }
      else if (pixels.getContext != null) {
          isCanvasLike = true;
      }
      else {
          throw new Error('pixels passed to tf.browser.fromPixels() must be either an ' +
              `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
              `in browser, or OffscreenCanvas, ImageData in webworker` +
              ` or {data: Uint32Array, width: number, height: number}, ` +
              `but was ${pixels.constructor.name}`);
      }
      if (isVideo) {
          const HAVE_CURRENT_DATA_READY_STATE = 2;
          if (isVideo &&
              pixels.readyState <
                  HAVE_CURRENT_DATA_READY_STATE) {
              throw new Error('The video element has not loaded data yet. Please wait for ' +
                  '`loadeddata` event on the <video> element.');
          }
      }
      // If the current backend has 'FromPixels' registered, it has a more
      // efficient way of handling pixel uploads, so we call that.
      const kernel = getKernel('FromPixels', ENGINE.backendName);
      if (kernel != null) {
          return ENGINE.runKernel('FromPixels', { pixels }, { numChannels });
      }
      const [width, height] = isVideo ?
          [
              pixels.videoWidth,
              pixels.videoHeight
          ] :
          [pixels.width, pixels.height];
      let vals;
      if (isCanvasLike) {
          vals =
              // tslint:disable-next-line:no-any
              pixels.getContext('2d').getImageData(0, 0, width, height).data;
      }
      else if (isImageData || isPixelData) {
          vals = pixels.data;
      }
      else if (isImage || isVideo) {
          if (fromPixels2DContext == null) {
              fromPixels2DContext = document.createElement('canvas').getContext('2d');
          }
          fromPixels2DContext.canvas.width = width;
          fromPixels2DContext.canvas.height = height;
          fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
          vals = fromPixels2DContext.getImageData(0, 0, width, height).data;
      }
      let values;
      if (numChannels === 4) {
          values = new Int32Array(vals);
      }
      else {
          const numPixels = width * height;
          values = new Int32Array(numPixels * numChannels);
          for (let i = 0; i < numPixels; i++) {
              for (let channel = 0; channel < numChannels; ++channel) {
                  values[i * numChannels + channel] = vals[i * 4 + channel];
              }
          }
      }
      const outShape = [height, width, numChannels];
      return tensor3d(values, outShape, 'int32');
  }
  /**
   * Draws a `tf.Tensor` of pixel values to a byte array or optionally a
   * canvas.
   *
   * When the dtype of the input is 'float32', we assume values in the range
   * [0-1]. Otherwise, when input is 'int32', we assume values in the range
   * [0-255].
   *
   * Returns a promise that resolves when the canvas has been drawn to.
   *
   * @param img A rank-2 or rank-3 tensor. If rank-2, draws grayscale. If
   *     rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
   * grayscale. When depth of 3, we draw with the first three components of
   * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
   * 4, all four components of the depth dimension correspond to r, g, b, a.
   * @param canvas The canvas to draw to.
   */
  /** @doc {heading: 'Browser', namespace: 'browser'} */
  async function toPixels(img, canvas) {
      let $img = convertToTensor(img, 'img', 'toPixels');
      if (!(img instanceof Tensor)) {
          // Assume int32 if user passed a native array.
          const originalImgTensor = $img;
          $img = originalImgTensor.toInt();
          originalImgTensor.dispose();
      }
      if ($img.rank !== 2 && $img.rank !== 3) {
          throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${$img.rank}.`);
      }
      const [height, width] = $img.shape.slice(0, 2);
      const depth = $img.rank === 2 ? 1 : $img.shape[2];
      if (depth > 4 || depth === 2) {
          throw new Error(`toPixels only supports depth of size ` +
              `1, 3 or 4 but got ${depth}`);
      }
      const data = await $img.data();
      const minTensor = $img.min();
      const maxTensor = $img.max();
      const vals = await Promise.all([minTensor.data(), maxTensor.data()]);
      const minVals = vals[0];
      const maxVals = vals[1];
      const min = minVals[0];
      const max = maxVals[0];
      minTensor.dispose();
      maxTensor.dispose();
      if ($img.dtype === 'float32') {
          if (min < 0 || max > 1) {
              throw new Error(`Tensor values for a float32 Tensor must be in the ` +
                  `range [0 - 1] but got range [${min} - ${max}].`);
          }
      }
      else if ($img.dtype === 'int32') {
          if (min < 0 || max > 255) {
              throw new Error(`Tensor values for a int32 Tensor must be in the ` +
                  `range [0 - 255] but got range [${min} - ${max}].`);
          }
      }
      else {
          throw new Error(`Unsupported type for toPixels: ${$img.dtype}.` +
              ` Please use float32 or int32 tensors.`);
      }
      const multiplier = $img.dtype === 'float32' ? 255 : 1;
      const bytes = new Uint8ClampedArray(width * height * 4);
      for (let i = 0; i < height * width; ++i) {
          let r, g, b, a;
          if (depth === 1) {
              r = data[i] * multiplier;
              g = data[i] * multiplier;
              b = data[i] * multiplier;
              a = 255;
          }
          else if (depth === 3) {
              r = data[i * 3] * multiplier;
              g = data[i * 3 + 1] * multiplier;
              b = data[i * 3 + 2] * multiplier;
              a = 255;
          }
          else if (depth === 4) {
              r = data[i * 4] * multiplier;
              g = data[i * 4 + 1] * multiplier;
              b = data[i * 4 + 2] * multiplier;
              a = data[i * 4 + 3] * multiplier;
          }
          const j = i * 4;
          bytes[j + 0] = Math.round(r);
          bytes[j + 1] = Math.round(g);
          bytes[j + 2] = Math.round(b);
          bytes[j + 3] = Math.round(a);
      }
      if (canvas != null) {
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          const imageData = new ImageData(bytes, width, height);
          ctx.putImageData(imageData, 0, 0);
      }
      if ($img !== img) {
          $img.dispose();
      }
      return bytes;
  }
  const fromPixels = op({ fromPixels_ });

  var browser = /*#__PURE__*/Object.freeze({
    __proto__: null,
    toPixels: toPixels,
    fromPixels: fromPixels
  });

  /**
   * Validate gather nd inputs.
   *
   * @param tensor The tensor contains the source values.
   * @param indices The tensor contains the indices to slice the source.
   *
   * @returns [resultShape, numUpdates, sliceSize, strides]
   */
  function prepareAndValidate(tensor, indices) {
      if (tensor.rank < 1) {
          throw new Error('tf.gatherND() expects the input to be rank 1 or higher,' +
              ` but the rank was ${tensor.rank}.`);
      }
      if (indices.rank < 1) {
          throw new Error('tf.gatherND() expects the indices to be rank 1 or higher,' +
              ` but the rank was ${indices.rank}.`);
      }
      if (indices.dtype !== 'int32') {
          throw new Error('tf.gatherND() expects the indices to be int32 type,' +
              ` but the dtype was ${indices.dtype}.`);
      }
      if (indices.shape[indices.rank - 1] > tensor.rank) {
          throw new Error('index innermost dimension length must be <= tensor rank; saw: ' +
              `${indices.shape[indices.rank - 1]} vs. ${tensor.rank}`);
      }
      if (tensor.size === 0) {
          throw new Error('Requested more than 0 entries, but input is empty.' +
              ` Input shape: ${tensor.shape}.`);
      }
      const indicesShape = indices.shape;
      const sliceRank = indicesShape[indicesShape.length - 1];
      // The result shape is
      //   indices.shape[:-1] + params.shape[indices.shape[-1]:]
      let nResult = 1;
      for (let i = 0; i < indicesShape.length - 1; ++i) {
          nResult *= indicesShape[i];
      }
      const inputShape = tensor.shape;
      const resultShape = indicesShape.slice();
      resultShape.pop();
      let sliceSize = 1;
      for (let i = sliceRank; i < tensor.rank; ++i) {
          sliceSize *= inputShape[i];
          resultShape.push(inputShape[i]);
      }
      const strides = [...computeStrides(tensor.shape).map(stride => stride / sliceSize),
          1].slice(0, sliceRank);
      return [resultShape, nResult, sliceSize, strides];
  }

  var gather_nd_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    prepareAndValidate: prepareAndValidate
  });

  /**
   * Check whether updates.shape = indices.shape[:batchDim] +
   * shape[sliceDim:]
   *
   * @param x The input tensor.
   */
  function validateUpdateShape(shape, indices, updates) {
      const sliceDim = (indices.rank > 1) ? indices.shape[indices.rank - 1] : 1;
      const batchDim = (indices.rank > 1) ? indices.rank - 1 : 1;
      const shapeError = 'Must have updates.shape = indices.shape[:batchDim] + ' +
          `shape[sliceDim:], got updates.shape: ${updates.shape}` +
          `, indices.shape: ${indices.shape}, shape: ${shape}` +
          `, sliceDim: ${sliceDim}, and batchDim: ${batchDim}.`;
      if (updates.rank < batchDim) {
          throw new Error(shapeError + ` update.rank < ${batchDim}. `);
      }
      if (shape.length < sliceDim + (updates.rank - batchDim)) {
          throw new Error(shapeError +
              ` Output shape length < ${sliceDim + (updates.rank - batchDim)}`);
      }
      if (updates.rank !== batchDim + shape.length - sliceDim) {
          throw new Error(shapeError + ` update.rank != ${batchDim + shape.length - sliceDim}`);
      }
      for (let d = 0; d < batchDim; ++d) {
          if (updates.shape[d] !== indices.shape[d]) {
              throw new Error(shapeError +
                  ` updates.shape[${d}] (${updates.shape[d]}) != indices.shape[${d}] (${indices.shape[d]}).`);
          }
      }
      for (let d = 0; d < updates.rank - batchDim; ++d) {
          if (updates.shape[d + batchDim] !== shape[d + sliceDim]) {
              throw new Error(shapeError +
                  ` updates.shape[${d + batchDim}] (${updates.shape[d + batchDim]}) != shape[${d + batchDim}] (${shape[d + batchDim]})`);
          }
      }
  }
  /**
   * Validate scatter nd inputs.
   *
   * @param update The tensor contains the update values.
   * @param indices The tensor contains the indices for the update values.
   * @param shape The shape of the output tensor.
   */
  function validateInput(updates, indices, shape) {
      if (indices.rank < 1) {
          throw new Error('tf.scatterND() expects the indices to be rank 1 or higher,' +
              ` but the rank was ${indices.rank}.`);
      }
      if (updates.rank < 1) {
          throw new Error('tf.scatterND() expects the updates to be rank 1 or higher,' +
              ` but the rank was ${updates.rank}.`);
      }
      if (indices.dtype !== 'int32') {
          throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${indices.dtype}`);
      }
      if (shape.length < 1) {
          throw new Error(`Output rank must be greater or equal to 1, but got shape: ${shape}`);
      }
      if (shape.length === 0) {
          if (indices.size === 0) {
              throw new Error(`Indices specified for empty output. indices shape: ${indices.shape}`);
          }
          if (updates.size === 0) {
              throw new Error(`Updates specified for empty output. updates shape: ${updates.shape}`);
          }
      }
      validateUpdateShape(shape, indices, updates);
  }
  /**
   * Calculate the shape information for the output.
   *
   * @param update The tensor contains the update values.
   * @param indices The tensor contains the indices for the update values.
   * @param shape The shape of the output tensor.
   *
   * @returns ScatterShapeInfo
   */
  function calculateShapes(updates, indices, shape) {
      // Calculate the number of dimensions in indices
      const indicesRank = indices.shape.length;
      const sliceRank = (indicesRank > 1) ? indices.shape[indicesRank - 1] : 1;
      // Calculate the number of elements that make up each slice of our updated
      // tensor. This allows us to work with flattened tensors and copy over whole
      // slices at a time.
      const totalNd = shape.length;
      let sliceSize = 1;
      for (let i = sliceRank; i < totalNd; ++i) {
          sliceSize *= shape[i];
      }
      const safeSliceDim = (sliceRank < 1) ? 1 : sliceRank;
      const numUpdates = sizeFromShape(indices.shape) / safeSliceDim;
      const strides = [...computeStrides(shape.slice(0, sliceRank)), 1];
      const outputSize = sizeFromShape(shape);
      return { sliceRank, numUpdates, sliceSize, strides, outputSize };
  }

  var scatter_nd_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    validateUpdateShape: validateUpdateShape,
    validateInput: validateInput,
    calculateShapes: calculateShapes
  });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Serializable defines the serialization contract.
   *
   * TFJS requires serializable classes to return their className when asked
   * to avoid issues with minification.
   */
  class Serializable {
      /**
       * Return the class name for this class to use in serialization contexts.
       *
       * Generally speaking this will be the same thing that constructor.name
       * would have returned.  However, the class name needs to be robust
       * against minification for serialization/deserialization to work properly.
       *
       * There's also places such as initializers.VarianceScaling, where
       * implementation details between different languages led to different
       * class hierarchies and a non-leaf node is used for serialization purposes.
       */
      getClassName() {
          return this.constructor
              .className;
      }
      /**
       * Creates an instance of T from a ConfigDict.
       *
       * This works for most descendants of serializable.  A few need to
       * provide special handling.
       * @param cls A Constructor for the class to instantiate.
       * @param config The Configuration for the object.
       */
      /** @nocollapse */
      static fromConfig(cls, config) {
          return new cls(config);
      }
  }
  /**
   * Maps string keys to class constructors.
   *
   * Used during (de)serialization from the cross-language JSON format, which
   * requires the class name in the serialization format matches the class
   * names as used in Python, should it exist.
   */
  class SerializationMap {
      constructor() {
          this.classNameMap = {};
      }
      /**
       * Returns the singleton instance of the map.
       */
      static getMap() {
          if (SerializationMap.instance == null) {
              SerializationMap.instance = new SerializationMap();
          }
          return SerializationMap.instance;
      }
      /**
       * Registers the class as serializable.
       */
      static register(cls) {
          SerializationMap.getMap().classNameMap[cls.className] =
              [cls, cls.fromConfig];
      }
  }
  /**
   * Register a class with the serialization map of TensorFlow.js.
   *
   * This is often used for registering custom Layers, so they can be
   * serialized and deserialized.
   *
   * Example:
   *
   * ```js
   * class MyCustomLayer extends tf.layers.Layer {
   *   static className = 'MyCustomLayer';
   *
   *   constructor(config) {
   *     super(config);
   *   }
   * }
   * tf.serialization.registerClass(MyCustomLayer);
   * ```
   *
   * @param cls The class to be registered. It must have a public static member
   *   called `className` defined and the value must be a non-empty string.
   */
  /** @doc {heading: 'Models', subheading: 'Serialization', ignoreCI: true} */
  function registerClass(cls) {
      assert(cls.className != null, () => `Class being registered does not have the static className ` +
          `property defined.`);
      assert(typeof cls.className === 'string', () => `className is required to be a string, but got type ` +
          typeof cls.className);
      assert(cls.className.length > 0, () => `Class being registered has an empty-string as its className, ` +
          `which is disallowed.`);
      SerializationMap.register(cls);
  }

  var serialization = /*#__PURE__*/Object.freeze({
    __proto__: null,
    Serializable: Serializable,
    SerializationMap: SerializationMap,
    registerClass: registerClass
  });

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
  const TEST_EPSILON_FLOAT32 = 1e-3;
  const TEST_EPSILON_FLOAT16 = 1e-1;
  function expectArraysClose(actual, expected, epsilon) {
      if (epsilon == null) {
          epsilon = testEpsilon();
      }
      return expectArraysPredicate(actual, expected, (a, b) => areClose(a, b, epsilon));
  }
  function testEpsilon() {
      return ENGINE.backend.floatPrecision() === 32 ? TEST_EPSILON_FLOAT32 :
          TEST_EPSILON_FLOAT16;
  }
  function expectArraysPredicate(actual, expected, predicate) {
      let checkClassType = true;
      if (isTypedArray(actual) || isTypedArray(expected)) {
          checkClassType = false;
      }
      if (isTypedArray(actual) && isTypedArray(expected)) {
          checkClassType = true;
      }
      if (checkClassType) {
          const aType = actual.constructor.name;
          const bType = expected.constructor.name;
          if (aType !== bType) {
              throw new Error(`Arrays are of different type. Actual: ${aType}. ` +
                  `Expected: ${bType}`);
          }
      }
      if (Array.isArray(actual) && Array.isArray(expected)) {
          const actualShape = inferShape(actual);
          const expectedShape = inferShape(expected);
          if (!arraysEqual(actualShape, expectedShape)) {
              throw new Error(`Arrays have different shapes. ` +
                  `Actual: [${actualShape}]. Expected: [${expectedShape}]`);
          }
      }
      const actualFlat = isTypedArray(actual) ? actual : flatten(actual);
      const expectedFlat = isTypedArray(expected) ?
          expected :
          flatten(expected);
      if (actualFlat.length !== expectedFlat.length) {
          throw new Error(`Arrays have different lengths actual: ${actualFlat.length} vs ` +
              `expected: ${expectedFlat.length}.\n` +
              `Actual:   ${actualFlat}.\n` +
              `Expected: ${expectedFlat}.`);
      }
      for (let i = 0; i < expectedFlat.length; ++i) {
          const a = actualFlat[i];
          const e = expectedFlat[i];
          if (!predicate(a, e)) {
              throw new Error(`Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
                  `Actual:   ${actualFlat}.\n` +
                  `Expected: ${expectedFlat}.`);
          }
      }
  }
  function expectPromiseToFail(fn, done) {
      fn().then(() => done.fail(), () => done());
  }
  function expectArraysEqual(actual, expected) {
      const exp = typeof expected === 'string' || typeof expected === 'number' ||
          typeof expected === 'boolean' ?
          [expected] :
          expected;
      if (isString(actual) || isString(actual[0]) ||
          isString(expected) || isString(expected[0])) {
          // tslint:disable-next-line: triple-equals
          return expectArraysPredicate(actual, exp, (a, b) => a == b);
      }
      return expectArraysPredicate(actual, expected, (a, b) => areClose(a, b, 0));
  }
  function expectNumbersClose(a, e, epsilon) {
      if (epsilon == null) {
          epsilon = testEpsilon();
      }
      if (!areClose(a, e, epsilon)) {
          throw new Error(`Numbers differ: actual === ${a}, expected === ${e}`);
      }
  }
  function areClose(a, e, epsilon) {
      if (!isFinite(a) && !isFinite(e)) {
          return true;
      }
      if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
          return false;
      }
      return true;
  }
  function expectValuesInRange(actual, low, high) {
      for (let i = 0; i < actual.length; i++) {
          if (actual[i] < low || actual[i] > high) {
              throw new Error(`Value out of range:${actual[i]} low: ${low}, high: ${high}`);
          }
      }
  }
  function expectArrayBuffersEqual(actual, expected) {
      // Safari & Jasmine don't like comparing ArrayBuffers directly. Wrapping in
      // a Float32Array solves this issue.
      expect(new Float32Array(actual)).toEqual(new Float32Array(expected));
  }

  var test_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    TEST_EPSILON_FLOAT16: TEST_EPSILON_FLOAT16,
    expectArraysClose: expectArraysClose,
    testEpsilon: testEpsilon,
    expectPromiseToFail: expectPromiseToFail,
    expectArraysEqual: expectArraysEqual,
    expectNumbersClose: expectNumbersClose,
    expectValuesInRange: expectValuesInRange,
    expectArrayBuffersEqual: expectArrayBuffersEqual
  });

  /** @license See the LICENSE file. */
  // This code is auto-generated, do not modify this file!
  const version = '0.0.0';

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Enables production mode which disables correctness checks in favor of
   * performance.
   */
  /** @doc {heading: 'Environment'} */
  function enableProdMode() {
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
  function enableDebugMode() {
      env().set('DEBUG', true);
  }
  /** Globally disables deprecation warnings */
  function disableDeprecationWarnings() {
      env().set('DEPRECATION_WARNINGS_ENABLED', false);
      console.warn(`TensorFlow.js deprecation warnings have been disabled.`);
  }
  /** Warn users about deprecated functionality. */
  function deprecationWarn(msg) {
      if (env().getBool('DEPRECATION_WARNINGS_ENABLED')) {
          console.warn(msg + ' You can disable deprecation warnings with ' +
              'tf.disableDeprecationWarnings().');
      }
  }
  setDeprecationWarningFn(deprecationWarn);
  /**
   * Dispose all variables kept in backend engine.
   */
  /** @doc {heading: 'Environment'} */
  function disposeVariables() {
      ENGINE.disposeVariables();
  }
  /**
   * It returns the global engine that keeps track of all tensors and backends.
   */
  /** @doc {heading: 'Environment'} */
  function engine() {
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
  function memory() {
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
  function profile(f) {
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
  function tidy(nameOrFn, fn) {
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
  function dispose(container) {
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
  function keep(result) {
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
  function time(f) {
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
  function setBackend(backendName) {
      return ENGINE.setBackend(backendName);
  }
  /**
   * Returns a promise that resolves when the currently selected backend (or the
   * highest priority one) has initialized. Await this promise when you are using
   * a backend that has async initialization.
   */
  /** @doc {heading: 'Backends'} */
  function ready() {
      return ENGINE.ready();
  }
  /**
   * Returns the current backend name (cpu, webgl, etc). The backend is
   * responsible for creating tensors and executing operations on those tensors.
   */
  /** @doc {heading: 'Backends'} */
  function getBackend() {
      return ENGINE.backendName;
  }
  /**
   * Removes a backend and the registered factory.
   */
  /** @doc {heading: 'Backends'} */
  function removeBackend(name) {
      ENGINE.removeBackend(name);
  }
  /**
   * Finds the backend registered under the provided name. Returns null if the
   * name is not in the registry, or the registration hasn't finished yet.
   */
  function findBackend(name) {
      return ENGINE.findBackend(name);
  }
  /**
   * Finds the backend factory registered under the provided name. Returns a
   * function that produces a new backend when called. Returns null if the name
   * is not in the registry.
   */
  function findBackendFactory(name) {
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
  function registerBackend(name, factory, priority = 1) {
      return ENGINE.registerBackend(name, factory, priority);
  }
  /**
   * Gets the current backend. If no backends have been initialized, this will
   * attempt to initialize the best backend. Will throw an error if the highest
   * priority backend has async initialization, in which case, you should call
   * 'await tf.ready()' before running other code.
   */
  /** @doc {heading: 'Backends'} */
  function backend() {
      return ENGINE.backend;
  }
  /**
   * Sets the global platform.
   *
   * @param platformName The name of this platform.
   * @param platform A platform implementation.
   */
  function setPlatform(platformName, platform) {
      env().setPlatform(platformName, platform);
  }

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Adds a list of `tf.Tensor`s element-wise, each with the same shape and dtype.
   *
   * ```js
   * const a = tf.tensor1d([1, 2]);
   * const b = tf.tensor1d([3, 4]);
   * const c = tf.tensor1d([5, 6]);
   *
   * tf.addN([a, b, c]).print();
   * ```
   * @param tensors A list of tensors with the same shape and dtype.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function addN_(tensors) {
      assert(Array.isArray(tensors), () => 'The argument passed to tf.addN() must be a list of tensors');
      assert(tensors.length >= 1, () => `Must pass at least one tensor to tf.addN(), but got ` +
          `${tensors.length}`);
      const $tensors = tensors.map((t, i) => convertToTensor(t, `tensors${i}`, 'addN'));
      const firstTensor = $tensors[0];
      $tensors.forEach(t => {
          if (t.dtype !== firstTensor.dtype) {
              throw new Error('All tensors passed to tf.addN() must have the same dtype');
          }
      });
      $tensors.forEach(t => {
          if (!arraysEqual(t.shape, firstTensor.shape)) {
              throw new Error('All tensors passed to tf.addN() must have the same shape');
          }
      });
      const forward = (backend, save) => backend.addN($tensors);
      const inputs = $tensors;
      return ENGINE.runKernelFunc(forward, inputs, null /* grad */, AddN);
  }
  const addN = op({ addN_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  function warnDeprecation() {
      deprecationWarn('tf.batchNormalization() is going away. ' +
          'Use tf.batchNorm() instead, and note the positional argument change ' +
          'of scale, offset, and varianceEpsilon');
  }
  function xAs4D(x) {
      let x4D;
      if (x.rank === 0 || x.rank === 1) {
          x4D = x.as4D(1, 1, 1, x.size);
      }
      else if (x.rank === 2) {
          x4D = x.as4D(1, 1, x.shape[0], x.shape[1]);
      }
      else if (x.rank === 3) {
          x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
      }
      else {
          x4D = x;
      }
      return x4D;
  }

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * @deprecated Please use `tf.batchNorm` instead and note the positional
   *     argument change of scale, offset, and varianceEpsilon.
   */
  function batchNormalization_(x, mean, variance, varianceEpsilon = .001, scale, offset) {
      warnDeprecation();
      return batchNorm_(x, mean, variance, offset, scale, varianceEpsilon);
  }
  /**
   * Batch normalization.
   *
   * As described in
   * [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167).
   *
   * Mean, variance, scale, and offset can be of two shapes:
   *   - The same shape as the input.
   *   - In the common case, the depth dimension is the last dimension of x, so
   *     the values would be an `tf.Tensor1D` of shape [depth].
   *
   * Also available are stricter rank-specific methods with the same signature
   * as this method that assert that parameters passed are of given rank
   *   - `tf.batchNorm2d`
   *   - `tf.batchNorm3d`
   *   - `tf.batchNorm4d`
   *
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param offset An offset Tensor.
   * @param scale A scale Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   */
  /** @doc {heading: 'Operations', subheading: 'Normalization'} */
  function batchNorm_(x, mean, variance, offset, scale, varianceEpsilon) {
      if (varianceEpsilon == null) {
          varianceEpsilon = 0.001;
      }
      const $x = convertToTensor(x, 'x', 'batchNorm');
      const $mean = convertToTensor(mean, 'mean', 'batchNorm');
      const $variance = convertToTensor(variance, 'variance', 'batchNorm');
      let $scale;
      if (scale != null) {
          $scale = convertToTensor(scale, 'scale', 'batchNorm');
      }
      let $offset;
      if (offset != null) {
          $offset = convertToTensor(offset, 'offset', 'batchNorm');
      }
      assert($mean.rank === $variance.rank, () => 'Batch normalization gradient requires mean and variance to have ' +
          'equal ranks.');
      assert($offset == null || $mean.rank === $offset.rank, () => 'Batch normalization gradient requires mean and offset to have ' +
          'equal ranks.');
      assert($scale == null || $mean.rank === $scale.rank, () => 'Batch normalization gradient requires mean and scale to have ' +
          'equal ranks.');
      const x4D = xAs4D($x);
      const forward = (backend, save) => {
          save([x4D, $mean, $variance, $scale]);
          return backend.batchNormalization(x4D, as1DOr4D($mean), as1DOr4D($variance), varianceEpsilon, as1DOr4D($scale), as1DOr4D($offset));
      };
      const inputs = {
          x: x4D,
          scale: $scale,
          offset: $offset,
          mean: $mean,
          variance: $variance
      };
      const attrs = { varianceEpsilon };
      const res = ENGINE.runKernelFunc(forward, inputs, null /* gradient */, FusedBatchNorm, attrs);
      return reshape(res, $x.shape);
  }
  function as1DOr4D(x) {
      if (x == null) {
          return null;
      }
      if (x.rank === 0) {
          return x.as1D();
      }
      else if (x.rank === 1) {
          return x;
      }
      else if (x.rank === 2) {
          return x.as4D(1, 1, x.shape[0], x.shape[1]);
      }
      else if (x.rank === 3) {
          return x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
      }
      return x;
  }
  // todo(yassogba): Remove batchNormalization since it is deprecated.
  const batchNormalization = op({ batchNormalization_ });
  const batchNorm = op({ batchNorm_ });

  /**
   * Batch normalization, strictly for 2D. For the more relaxed version, see
   * `tf.batchNorm`.
   *
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param offset An offset Tensor.
   * @param scale A scale Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   */
  function batchNorm2d_(x, mean, variance, offset, scale, varianceEpsilon) {
      const $x = convertToTensor(x, 'x', 'batchNorm');
      const $mean = convertToTensor(mean, 'mean', 'batchNorm');
      const $variance = convertToTensor(variance, 'variance', 'batchNorm');
      let $scale;
      if (scale != null) {
          $scale = convertToTensor(scale, 'scale', 'batchNorm');
      }
      let $offset;
      if (offset != null) {
          $offset = convertToTensor(offset, 'offset', 'batchNorm');
      }
      assert($x.rank === 2, () => `Error in batchNorm3D: x must be rank 3 but got rank ` +
          `${$x.rank}.`);
      assert($mean.rank === 2 || $mean.rank === 1, () => `Error in batchNorm2D: mean must be rank 2 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
      assert($variance.rank === 2 || $variance.rank === 1, () => `Error in batchNorm2D: variance must be rank 2 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
      if ($scale != null) {
          assert($scale.rank === 2 || $scale.rank === 1, () => `Error in batchNorm2D: scale must be rank 2 or rank 1 ` +
              `but got rank ${$scale.rank}.`);
      }
      if ($offset != null) {
          assert($offset.rank === 2 || $offset.rank === 1, () => `Error in batchNorm2D: offset must be rank 2 or rank 1 ` +
              `but got rank ${$offset.rank}.`);
      }
      return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
  }
  /**
   * @deprecated Please use `tf.batchNorm2d` instead and note the positional
   *     argument change of scale, offset, and varianceEpsilon.
   */
  function batchNormalization2d_(x, mean, variance, varianceEpsilon = .001, scale, offset) {
      warnDeprecation();
      return batchNorm2d_(x, mean, variance, offset, scale, varianceEpsilon);
  }
  // todo(yassogba): Remove batchNormalization2d since it is deprecated.
  const batchNormalization2d = op({ batchNormalization2d_ });
  const batchNorm2d = op({ batchNorm2d_ });

  /**
   * Batch normalization, strictly for 3D. For the more relaxed version, see
   * `tf.batchNorm`.
   *
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param offset An offset Tensor.
   * @param scale A scale Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   */
  function batchNorm3d_(x, mean, variance, offset, scale, varianceEpsilon) {
      const $x = convertToTensor(x, 'x', 'batchNorm');
      const $mean = convertToTensor(mean, 'mean', 'batchNorm');
      const $variance = convertToTensor(variance, 'variance', 'batchNorm');
      let $scale;
      if (scale != null) {
          $scale = convertToTensor(scale, 'scale', 'batchNorm');
      }
      let $offset;
      if (offset != null) {
          $offset = convertToTensor(offset, 'offset', 'batchNorm');
      }
      assert($x.rank === 3, () => `Error in batchNorm3D: x must be rank 3 but got rank ` +
          `${$x.rank}.`);
      assert($mean.rank === 3 || $mean.rank === 1, () => `Error in batchNorm3D: mean must be rank 3 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
      assert($variance.rank === 3 || $variance.rank === 1, () => `Error in batchNorm3D: variance must be rank 3 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
      if ($scale != null) {
          assert($scale.rank === 3 || $scale.rank === 1, () => `Error in batchNorm3D: scale must be rank 3 or rank 1 ` +
              `but got rank ${$scale.rank}.`);
      }
      if ($offset != null) {
          assert($offset.rank === 3 || $offset.rank === 1, () => `Error in batchNorm3D: offset must be rank 3 or rank 1 ` +
              `but got rank ${$offset.rank}.`);
      }
      return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
  }
  /**
   * @deprecated Please use `tf.batchNorm3d` instead and note the positional
   *     argument change of scale, offset, and varianceEpsilon.
   */
  function batchNormalization3d_(x, mean, variance, varianceEpsilon = .001, scale, offset) {
      warnDeprecation();
      return batchNorm3d_(x, mean, variance, offset, scale, varianceEpsilon);
  }
  // todo(yassogba): Remove batchNormalization3d since it is deprecated.
  const batchNormalization3d = op({ batchNormalization3d_ });
  const batchNorm3d = op({ batchNorm3d_ });

  /**
   * Batch normalization, strictly for 4D. For the more relaxed version, see
   * `tf.batchNorm`.
   *
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param offset An offset Tensor.
   * @param scale A scale Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   */
  function batchNorm4d_(x, mean, variance, offset, scale, varianceEpsilon) {
      const $x = convertToTensor(x, 'x', 'batchNorm');
      const $mean = convertToTensor(mean, 'mean', 'batchNorm');
      const $variance = convertToTensor(variance, 'variance', 'batchNorm');
      let $scale;
      if (scale != null) {
          $scale = convertToTensor(scale, 'scale', 'batchNorm');
      }
      let $offset;
      if (offset != null) {
          $offset = convertToTensor(offset, 'offset', 'batchNorm');
      }
      assert($x.rank === 4, () => `Error in batchNorm4D: x must be rank 4 but got rank ` +
          `${$x.rank}.`);
      assert($mean.rank === 4 || $mean.rank === 1, () => `Error in batchNorm4D: mean must be rank 4 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
      assert($variance.rank === 4 || $variance.rank === 1, () => `Error in batchNorm4D: variance must be rank 4 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
      if ($scale != null) {
          assert($scale.rank === 4 || $scale.rank === 1, () => `Error in batchNorm4D: scale must be rank 4 or rank 1 ` +
              `but got rank ${$scale.rank}.`);
      }
      if ($offset != null) {
          assert($offset.rank === 4 || $offset.rank === 1, () => `Error in batchNorm4D: offset must be rank 4 or rank 1 ` +
              `but got rank ${$offset.rank}.`);
      }
      return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
  }
  /**
   * @deprecated Please use `tf.batchNorm4d` instead and note the positional
   *     argument change of scale, offset, and varianceEpsilon.
   */
  function batchNormalization4d_(x, mean, variance, varianceEpsilon = .001, scale, offset) {
      warnDeprecation();
      return batchNorm4d_(x, mean, variance, offset, scale, varianceEpsilon);
  }
  // todo(yassogba): Remove batchNormalization4d since it is deprecated.
  const batchNormalization4d = op({ batchNormalization4d_ });
  const batchNorm4d = op({ batchNorm4d_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Creates a new tensor with the same values and shape as the specified
   * tensor.
   *
   * ```js
   * const x = tf.tensor([1, 2]);
   *
   * x.clone().print();
   * ```
   *
   * @param x The tensor to clone.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function clone_(x) {
      const $x = convertToTensor(x, 'x', 'clone', null);
      const forward = () => ENGINE.makeTensorFromDataId($x.dataId, $x.shape, $x.dtype);
      // Note this op is called tf.identity in python. Hence the kernel name used
      // here.
      return ENGINE.runKernelFunc(forward, { x: $x }, null /* grad */, Identity);
  }
  const clone = op({ clone_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Broadcast an array to a compatible shape NumPy-style.
   *
   * The tensor's shape is compared to the broadcast shape from end to beginning.
   * Ones are prepended to the tensor's shape until is has the same length as
   * the broadcast shape. If input.shape[i]==shape[i], the (i+1)-th axis is
   * already broadcast-compatible. If input.shape[i]==1 and shape[i]==N, then
   * the input tensor is tiled N times along that axis (using tf.tile).
   *
   * @param input The tensor that is to be broadcasted.
   * @param shape The input is to be broadcast to this shape.
   */
  /** @doc {heading: 'Tensors', subheading: 'Transformations'} */
  function broadcastTo_(x, shape) {
      let input = convertToTensor(x, 'broadcastTo', 'x');
      const xShape = input.shape;
      if (shape.some(d => !(d > 0) || d % 1 !== 0)) {
          throw new Error(`broadcastTo(): Invalid broadcast shape [${shape}].`);
      }
      if (shape.length < input.rank) {
          throw new Error(`broadcastTo(): shape.length=${shape.length} < input.rank=${input.rank}.`);
      }
      if (shape.length > input.rank) {
          const newShape = input.shape.slice();
          while (newShape.length < shape.length) {
              newShape.unshift(1);
          }
          input = reshape(input, newShape);
      }
      const inputShape = input.shape;
      const reps = Array.from(shape);
      for (let i = shape.length - 1; i >= 0; i--) {
          if (inputShape[i] === shape[i]) {
              reps[i] = 1;
          }
          else if (input.shape[i] !== 1) {
              throw new Error(`broadcastTo(): [${xShape}] cannot be broadcast to [${shape}].`);
          }
      }
      const axes = reps.map((n, i) => n > 1 ? i : -1).filter(i => i >= 0);
      if (axes.length === 0) {
          return clone(input);
      }
      const forward = (backend) => backend.tile(input, reps);
      const inputs = { x: input };
      const attrs = { shape, inputShape };
      return ENGINE.runKernelFunc(forward, inputs, null /* grad */, BroadcastTo, attrs);
  }
  const broadcastTo = op({ broadcastTo_ });

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
  function whereImpl(condShape, condVals) {
      const indices = [];
      for (let i = 0; i < condVals.length; i++) {
          if (condVals[i]) {
              indices.push(i);
          }
      }
      const inBuffer = buffer(condShape, 'int32');
      const out = buffer([indices.length, condShape.length], 'int32');
      for (let i = 0; i < indices.length; i++) {
          const loc = inBuffer.indexToLoc(indices[i]);
          const offset = i * condShape.length;
          out.values.set(loc, offset);
      }
      return out.toTensor();
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Returns the truth value of `NOT x` element-wise.
   *
   * ```js
   * const a = tf.tensor1d([false, true], 'bool');
   *
   * a.logicalNot().print();
   * ```
   *
   * @param x The input tensor. Must be of dtype 'bool'.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function logicalNot_(x) {
      const $x = convertToTensor(x, 'x', 'logicalNot', 'bool');
      return ENGINE.runKernelFunc(backend => backend.logicalNot($x), { $x });
  }
  /**
   * Returns the truth value of `a AND b` element-wise. Supports broadcasting.
   *
   * ```js
   * const a = tf.tensor1d([false, false, true, true], 'bool');
   * const b = tf.tensor1d([false, true, false, true], 'bool');
   *
   * a.logicalAnd(b).print();
   * ```
   *
   * @param a The first input tensor. Must be of dtype bool.
   * @param b The second input tensor. Must be of dtype bool.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function logicalAnd_(a, b) {
      const $a = convertToTensor(a, 'a', 'logicalAnd', 'bool');
      const $b = convertToTensor(b, 'b', 'logicalAnd', 'bool');
      assertAndGetBroadcastShape($a.shape, $b.shape);
      return ENGINE.runKernelFunc(backend => backend.logicalAnd($a, $b), { a: $a, b: $b }, null /* grad */, 'LogicalAnd');
  }
  /**
   * Returns the truth value of `a OR b` element-wise. Supports broadcasting.
   *
   * ```js
   * const a = tf.tensor1d([false, false, true, true], 'bool');
   * const b = tf.tensor1d([false, true, false, true], 'bool');
   *
   * a.logicalOr(b).print();
   * ```
   * @param a The first input tensor. Must be of dtype bool.
   * @param b The second input tensor. Must be of dtype bool.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function logicalOr_(a, b) {
      const $a = convertToTensor(a, 'a', 'logicalOr', 'bool');
      const $b = convertToTensor(b, 'b', 'logicalOr', 'bool');
      assertAndGetBroadcastShape($a.shape, $b.shape);
      return ENGINE.runKernelFunc(backend => backend.logicalOr($a, $b), { $a, $b });
  }
  /**
   * Returns the truth value of `a XOR b` element-wise. Supports broadcasting.
   *
   * ```js
   * const a = tf.tensor1d([false, false, true, true], 'bool');
   * const b = tf.tensor1d([false, true, false, true], 'bool');
   *
   * a.logicalXor(b).print();
   * ```
   *
   * @param a The first input tensor. Must be of dtype bool.
   * @param b The second input tensor. Must be of dtype bool.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function logicalXor_(a, b) {
      const $a = convertToTensor(a, 'a', 'logicalXor', 'bool');
      const $b = convertToTensor(b, 'b', 'logicalXor', 'bool');
      assertAndGetBroadcastShape($a.shape, $b.shape);
      // x ^ y = (x | y) & ~(x & y)
      return logicalOr(a, b).logicalAnd(logicalAnd(a, b).logicalNot());
  }
  /**
   * Returns the elements, either `a` or `b` depending on the `condition`.
   *
   * If the condition is true, select from `a`, otherwise select from `b`.
   *
   * ```js
   * const cond = tf.tensor1d([false, false, true], 'bool');
   * const a = tf.tensor1d([1 , 2, 3]);
   * const b = tf.tensor1d([-1, -2, -3]);
   *
   * a.where(cond, b).print();
   * ```
   *
   * @param condition The input condition. Must be of dtype bool.
   * @param a If `condition` is rank 1, `a` may have a higher rank but
   *     its first dimension must match the size of `condition`.
   * @param b A tensor with the same shape and type as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function where_(condition, a, b) {
      const $a = convertToTensor(a, 'a', 'where');
      const $b = convertToTensor(b, 'b', 'where');
      const $condition = convertToTensor(condition, 'condition', 'where', 'bool');
      assertShapesMatch($a.shape, $b.shape, 'Error in where: ');
      if ($condition.rank === 1) {
          // If condition rank is 1, then the first dimension must match the size of
          // condition.
          assert($condition.shape[0] === $a.shape[0], () => 'The first dimension of `a` must match the size of `condition`.');
      }
      else {
          // A must have the same shape as condition.
          assertShapesMatch($condition.shape, $b.shape, 'Error in where: ');
      }
      // TODO(julianoks): Return null for condition gradient
      // when backprop supports it.
      const grad = (dy, saved) => {
          const [$condition] = saved;
          return {
              $condition: () => zerosLike($condition).toFloat(),
              $a: () => dy.mul($condition.cast(dy.dtype)),
              $b: () => dy.mul($condition.logicalNot().cast(dy.dtype))
          };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.select($condition, $a, $b);
          save([$condition]);
          return res;
      }, { $condition, $a, $b }, grad);
  }
  /**
   * Returns the coordinates of true elements of condition.
   *
   * The coordinates are returned in a 2-D tensor where the first dimension (rows)
   * represents the number of true elements, and the second dimension (columns)
   * represents the coordinates of the true elements. Keep in mind, the shape of
   * the output tensor can vary depending on how many true values there are in
   * input. Indices are output in row-major order. The resulting tensor has the
   * shape `[numTrueElems, condition.rank]`.
   *
   * This is analogous to calling the python `tf.where(cond)` without an x or y.
   *
   * ```js
   * const cond = tf.tensor1d([false, false, true], 'bool');
   * const result = await tf.whereAsync(cond);
   * result.print();
   * ```
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  async function whereAsync_(condition) {
      const $condition = convertToTensor(condition, 'condition', 'whereAsync', 'bool');
      const vals = await $condition.data();
      const res = whereImpl($condition.shape, vals);
      if (condition !== $condition) {
          $condition.dispose();
      }
      return res;
  }
  const logicalAnd = op({ logicalAnd_ });
  const logicalNot = op({ logicalNot_ });
  const logicalOr = op({ logicalOr_ });
  const logicalXor = op({ logicalXor_ });
  const where = op({ where_ });
  const whereAsync = whereAsync_;

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting. Return 0
   * if denominator is 0.
   *
   * We also expose `tf.divStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 9, 16]);
   * const b = tf.tensor1d([1, 2, 3, 4]);
   * const c = tf.tensor1d([0, 0, 0, 0]);
   *
   * a.divNoNan(b).print();  // or tf.divNoNan(a, b)
   * a.divNoNan(c).print();  // or tf.divNoNan(a, c)
   * ```
   *
   * ```js
   * // Broadcast div a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(2);
   * const c = tf.scalar(0);
   *
   * a.divNoNan(b).print();  // or tf.divNoNan(a, b)
   * a.divNoNan(c).print();  // or tf.divNoNan(a, c)
   * ```
   *
   * @param a The first tensor as the numerator.
   * @param b The second tensor as the denominator. Must have the same dtype as
   * `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function divNoNan_(a, b) {
      // TODO: Make this into its own kernel.
      let $a = convertToTensor(a, 'a', 'div');
      let $b = convertToTensor(b, 'b', 'div');
      [$a, $b] = makeTypesMatch($a, $b);
      const divResult = div($a, $b);
      const zeros = zerosLike(divResult);
      const bEqualsZero = $b.equal(zeros);
      return where(bEqualsZero, zeros, divResult);
  }
  const divNoNan = op({ divNoNan_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Create an identity matrix.
   *
   * @param numRows Number of rows.
   * @param numColumns Number of columns. Defaults to `numRows`.
   * @param batchShape If provided, will add the batch shape to the beginning
   *   of the shape of the returned `tf.Tensor` by repeating the identity
   *   matrix.
   * @param dtype Data type.
   * @returns Identity matrix of the specified size and data type, possibly
   *   with batch repetition if `batchShape` is specified.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function eye_(numRows, numColumns, batchShape, dtype = 'float32') {
      if (numColumns == null) {
          numColumns = numRows;
      }
      const buff = buffer([numRows, numColumns], dtype);
      const n = numRows <= numColumns ? numRows : numColumns;
      for (let i = 0; i < n; ++i) {
          buff.set(1, i, i);
      }
      const out = buff.toTensor().as2D(numRows, numColumns);
      if (batchShape == null) {
          return out;
      }
      else {
          if (batchShape.length === 1) {
              return tile(expandDims(out, 0), [batchShape[0], 1, 1]);
          }
          else if (batchShape.length === 2) {
              return tile(expandDims(expandDims(out, 0), 0), [batchShape[0], batchShape[1], 1, 1]);
          }
          else if (batchShape.length === 3) {
              return tile(expandDims(expandDims(expandDims(out, 0), 0), 0), [batchShape[0], batchShape[1], batchShape[2], 1, 1]);
          }
          else {
              throw new Error(`eye() currently supports only 1D and 2D ` +
                  // tslint:disable-next-line:no-any
                  `batchShapes, but received ${batchShape.length}D.`);
          }
      }
  }
  const eye = op({ eye_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Creates a `tf.Tensor` with values drawn from a multinomial distribution.
   *
   * ```js
   * const probs = tf.tensor([.75, .25]);
   * tf.multinomial(probs, 3).print();
   * ```
   *
   * @param logits 1D array with unnormalized log-probabilities, or
   *     2D array of shape `[batchSize, numOutcomes]`. See the `normalized`
   *     parameter.
   * @param numSamples Number of samples to draw for each row slice.
   * @param seed The seed number.
   * @param normalized Whether the provided `logits` are normalized true
   *     probabilities (sum to 1). Defaults to false.
   * @return 1D array of shape `[numSamples]`, or 2D array of shape
   *     `[batchSize, numSamples]`, depending on the rank of the input.
   */
  /** @doc {heading: 'Tensors', subheading: 'Random'} */
  function multinomial_(logits, numSamples, seed, normalized = false) {
      const $logits = convertToTensor(logits, 'logits', 'multinomial');
      const numOutcomes = $logits.size;
      const origRank = $logits.rank;
      if (numOutcomes < 2) {
          throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ` +
              `${numOutcomes}.`);
      }
      if (origRank > 2) {
          throw new Error(`Rank of probabilities must be 1 or 2, but is ${origRank}`);
      }
      seed = seed || Math.random();
      const logits2D = origRank === 1 ? $logits.as2D(1, -1) : $logits;
      const res = ENGINE.runKernelFunc(backend => backend.multinomial(logits2D, normalized, numSamples, seed), { logits2D });
      return origRank === 1 ? res.as1D() : res;
  }
  const multinomial = op({ multinomial_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Returns the truth value of (a != b) element-wise. Supports broadcasting.
   *
   * We also expose `tf.notEqualStrict` which has the same signature as this op
   * and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([0, 2, 3]);
   *
   * a.notEqual(b).print();
   * ```
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function notEqual_(a, b) {
      let $a = convertToTensor(a, 'a', 'notEqual');
      let $b = convertToTensor(b, 'b', 'notEqual');
      [$a, $b] = makeTypesMatch($a, $b);
      assertAndGetBroadcastShape($a.shape, $b.shape);
      const forward = (backend) => backend.notEqual($a, $b);
      const inputs = { a: $a, b: $b };
      return ENGINE.runKernelFunc(forward, inputs, null /* grad */, NotEqual);
  }
  const notEqual = op({ notEqual_ });

  /**
   * Pads a `tf.Tensor1D` with a given value and paddings. See `pad` for details.
   */
  function pad1d_(x, paddings, constantValue = 0) {
      assert(paddings.length === 2, () => 'Invalid number of paddings. Must be length of 2.');
      return pad(x, [paddings], constantValue);
  }
  const pad1d = op({ pad1d_ });

  /**
   * Pads a `tf.Tensor2D` with a given value and paddings. See `pad` for details.
   */
  function pad2d_(x, paddings, constantValue = 0) {
      assert(paddings.length === 2 && paddings[0].length === 2 &&
          paddings[1].length === 2, () => 'Invalid number of paddings. Must be length of 2 each.');
      return pad(x, paddings, constantValue);
  }
  const pad2d = op({ pad2d_ });

  /**
   * Pads a `tf.Tensor3D` with a given value and paddings. See `pad` for details.
   */
  function pad3d_(x, paddings, constantValue = 0) {
      assert(paddings.length === 3 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2, () => 'Invalid number of paddings. Must be length of 2 each.');
      return pad(x, paddings, constantValue);
  }
  const pad3d = op({ pad3d_ });

  /**
   * Pads a `tf.Tensor4D` with a given value and paddings. See `pad` for details.
   */
  function pad4d_(x, paddings, constantValue = 0) {
      assert(paddings.length === 4 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2 &&
          paddings[3].length === 2, () => 'Invalid number of paddings. Must be length of 2 each.');
      return pad(x, paddings, constantValue);
  }
  const pad4d = op({ pad4d_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Creates a `tf.Tensor` with values sampled from a random number generator
   * function defined by the user.
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param randFunction A random number generator function which is called
   * for each element in the output tensor.
   * @param dtype The data type of the output tensor. Defaults to 'float32'.
   */
  function rand_(shape, randFunction, dtype) {
      const size = sizeFromShape(shape);
      let values = null;
      if (dtype == null || dtype === 'float32') {
          values = new Float32Array(size);
      }
      else if (dtype === 'int32') {
          values = new Int32Array(size);
      }
      else if (dtype === 'bool') {
          values = new Uint8Array(size);
      }
      else {
          throw new Error(`Unknown data type ${dtype}`);
      }
      for (let i = 0; i < size; i++) {
          values[i] = randFunction();
      }
      return ENGINE.makeTensor(values, shape, dtype);
  }
  const rand = op({ rand_ });

  var commonjsGlobal = typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};

  function createCommonjsModule(fn, module) {
  	return module = { exports: {} }, fn(module, module.exports), module.exports;
  }

  var alea = createCommonjsModule(function (module) {
  // A port of an algorithm by Johannes Baagøe <baagoe@baagoe.com>, 2010
  // http://baagoe.com/en/RandomMusings/javascript/
  // https://github.com/nquinlan/better-random-numbers-for-javascript-mirror
  // Original work is under MIT license -

  // Copyright (C) 2010 by Johannes Baagøe <baagoe@baagoe.org>
  //
  // Permission is hereby granted, free of charge, to any person obtaining a copy
  // of this software and associated documentation files (the "Software"), to deal
  // in the Software without restriction, including without limitation the rights
  // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  // copies of the Software, and to permit persons to whom the Software is
  // furnished to do so, subject to the following conditions:
  //
  // The above copyright notice and this permission notice shall be included in
  // all copies or substantial portions of the Software.
  //
  // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  // THE SOFTWARE.



  (function(global, module, define) {

  function Alea(seed) {
    var me = this, mash = Mash();

    me.next = function() {
      var t = 2091639 * me.s0 + me.c * 2.3283064365386963e-10; // 2^-32
      me.s0 = me.s1;
      me.s1 = me.s2;
      return me.s2 = t - (me.c = t | 0);
    };

    // Apply the seeding algorithm from Baagoe.
    me.c = 1;
    me.s0 = mash(' ');
    me.s1 = mash(' ');
    me.s2 = mash(' ');
    me.s0 -= mash(seed);
    if (me.s0 < 0) { me.s0 += 1; }
    me.s1 -= mash(seed);
    if (me.s1 < 0) { me.s1 += 1; }
    me.s2 -= mash(seed);
    if (me.s2 < 0) { me.s2 += 1; }
    mash = null;
  }

  function copy(f, t) {
    t.c = f.c;
    t.s0 = f.s0;
    t.s1 = f.s1;
    t.s2 = f.s2;
    return t;
  }

  function impl(seed, opts) {
    var xg = new Alea(seed),
        state = opts && opts.state,
        prng = xg.next;
    prng.int32 = function() { return (xg.next() * 0x100000000) | 0; };
    prng.double = function() {
      return prng() + (prng() * 0x200000 | 0) * 1.1102230246251565e-16; // 2^-53
    };
    prng.quick = prng;
    if (state) {
      if (typeof(state) == 'object') copy(state, xg);
      prng.state = function() { return copy(xg, {}); };
    }
    return prng;
  }

  function Mash() {
    var n = 0xefc8249d;

    var mash = function(data) {
      data = data.toString();
      for (var i = 0; i < data.length; i++) {
        n += data.charCodeAt(i);
        var h = 0.02519603282416938 * n;
        n = h >>> 0;
        h -= n;
        h *= n;
        n = h >>> 0;
        h -= n;
        n += h * 0x100000000; // 2^32
      }
      return (n >>> 0) * 2.3283064365386963e-10; // 2^-32
    };

    return mash;
  }


  if (module && module.exports) {
    module.exports = impl;
  } else if (define && define.amd) {
    define(function() { return impl; });
  } else {
    this.alea = impl;
  }

  })(
    commonjsGlobal,
     module,    // present in node.js
    (typeof undefined) == 'function'    // present with an AMD loader
  );
  });

  var xor128 = createCommonjsModule(function (module) {
  // A Javascript implementaion of the "xor128" prng algorithm by
  // George Marsaglia.  See http://www.jstatsoft.org/v08/i14/paper

  (function(global, module, define) {

  function XorGen(seed) {
    var me = this, strseed = '';

    me.x = 0;
    me.y = 0;
    me.z = 0;
    me.w = 0;

    // Set up generator function.
    me.next = function() {
      var t = me.x ^ (me.x << 11);
      me.x = me.y;
      me.y = me.z;
      me.z = me.w;
      return me.w ^= (me.w >>> 19) ^ t ^ (t >>> 8);
    };

    if (seed === (seed | 0)) {
      // Integer seed.
      me.x = seed;
    } else {
      // String seed.
      strseed += seed;
    }

    // Mix in string seed, then discard an initial batch of 64 values.
    for (var k = 0; k < strseed.length + 64; k++) {
      me.x ^= strseed.charCodeAt(k) | 0;
      me.next();
    }
  }

  function copy(f, t) {
    t.x = f.x;
    t.y = f.y;
    t.z = f.z;
    t.w = f.w;
    return t;
  }

  function impl(seed, opts) {
    var xg = new XorGen(seed),
        state = opts && opts.state,
        prng = function() { return (xg.next() >>> 0) / 0x100000000; };
    prng.double = function() {
      do {
        var top = xg.next() >>> 11,
            bot = (xg.next() >>> 0) / 0x100000000,
            result = (top + bot) / (1 << 21);
      } while (result === 0);
      return result;
    };
    prng.int32 = xg.next;
    prng.quick = prng;
    if (state) {
      if (typeof(state) == 'object') copy(state, xg);
      prng.state = function() { return copy(xg, {}); };
    }
    return prng;
  }

  if (module && module.exports) {
    module.exports = impl;
  } else if (define && define.amd) {
    define(function() { return impl; });
  } else {
    this.xor128 = impl;
  }

  })(
    commonjsGlobal,
     module,    // present in node.js
    (typeof undefined) == 'function'    // present with an AMD loader
  );
  });

  var xorwow = createCommonjsModule(function (module) {
  // A Javascript implementaion of the "xorwow" prng algorithm by
  // George Marsaglia.  See http://www.jstatsoft.org/v08/i14/paper

  (function(global, module, define) {

  function XorGen(seed) {
    var me = this, strseed = '';

    // Set up generator function.
    me.next = function() {
      var t = (me.x ^ (me.x >>> 2));
      me.x = me.y; me.y = me.z; me.z = me.w; me.w = me.v;
      return (me.d = (me.d + 362437 | 0)) +
         (me.v = (me.v ^ (me.v << 4)) ^ (t ^ (t << 1))) | 0;
    };

    me.x = 0;
    me.y = 0;
    me.z = 0;
    me.w = 0;
    me.v = 0;

    if (seed === (seed | 0)) {
      // Integer seed.
      me.x = seed;
    } else {
      // String seed.
      strseed += seed;
    }

    // Mix in string seed, then discard an initial batch of 64 values.
    for (var k = 0; k < strseed.length + 64; k++) {
      me.x ^= strseed.charCodeAt(k) | 0;
      if (k == strseed.length) {
        me.d = me.x << 10 ^ me.x >>> 4;
      }
      me.next();
    }
  }

  function copy(f, t) {
    t.x = f.x;
    t.y = f.y;
    t.z = f.z;
    t.w = f.w;
    t.v = f.v;
    t.d = f.d;
    return t;
  }

  function impl(seed, opts) {
    var xg = new XorGen(seed),
        state = opts && opts.state,
        prng = function() { return (xg.next() >>> 0) / 0x100000000; };
    prng.double = function() {
      do {
        var top = xg.next() >>> 11,
            bot = (xg.next() >>> 0) / 0x100000000,
            result = (top + bot) / (1 << 21);
      } while (result === 0);
      return result;
    };
    prng.int32 = xg.next;
    prng.quick = prng;
    if (state) {
      if (typeof(state) == 'object') copy(state, xg);
      prng.state = function() { return copy(xg, {}); };
    }
    return prng;
  }

  if (module && module.exports) {
    module.exports = impl;
  } else if (define && define.amd) {
    define(function() { return impl; });
  } else {
    this.xorwow = impl;
  }

  })(
    commonjsGlobal,
     module,    // present in node.js
    (typeof undefined) == 'function'    // present with an AMD loader
  );
  });

  var xorshift7 = createCommonjsModule(function (module) {
  // A Javascript implementaion of the "xorshift7" algorithm by
  // François Panneton and Pierre L'ecuyer:
  // "On the Xorgshift Random Number Generators"
  // http://saluc.engr.uconn.edu/refs/crypto/rng/panneton05onthexorshift.pdf

  (function(global, module, define) {

  function XorGen(seed) {
    var me = this;

    // Set up generator function.
    me.next = function() {
      // Update xor generator.
      var X = me.x, i = me.i, t, v;
      t = X[i]; t ^= (t >>> 7); v = t ^ (t << 24);
      t = X[(i + 1) & 7]; v ^= t ^ (t >>> 10);
      t = X[(i + 3) & 7]; v ^= t ^ (t >>> 3);
      t = X[(i + 4) & 7]; v ^= t ^ (t << 7);
      t = X[(i + 7) & 7]; t = t ^ (t << 13); v ^= t ^ (t << 9);
      X[i] = v;
      me.i = (i + 1) & 7;
      return v;
    };

    function init(me, seed) {
      var j, w, X = [];

      if (seed === (seed | 0)) {
        // Seed state array using a 32-bit integer.
        w = X[0] = seed;
      } else {
        // Seed state using a string.
        seed = '' + seed;
        for (j = 0; j < seed.length; ++j) {
          X[j & 7] = (X[j & 7] << 15) ^
              (seed.charCodeAt(j) + X[(j + 1) & 7] << 13);
        }
      }
      // Enforce an array length of 8, not all zeroes.
      while (X.length < 8) X.push(0);
      for (j = 0; j < 8 && X[j] === 0; ++j);
      if (j == 8) w = X[7] = -1; else w = X[j];

      me.x = X;
      me.i = 0;

      // Discard an initial 256 values.
      for (j = 256; j > 0; --j) {
        me.next();
      }
    }

    init(me, seed);
  }

  function copy(f, t) {
    t.x = f.x.slice();
    t.i = f.i;
    return t;
  }

  function impl(seed, opts) {
    if (seed == null) seed = +(new Date);
    var xg = new XorGen(seed),
        state = opts && opts.state,
        prng = function() { return (xg.next() >>> 0) / 0x100000000; };
    prng.double = function() {
      do {
        var top = xg.next() >>> 11,
            bot = (xg.next() >>> 0) / 0x100000000,
            result = (top + bot) / (1 << 21);
      } while (result === 0);
      return result;
    };
    prng.int32 = xg.next;
    prng.quick = prng;
    if (state) {
      if (state.x) copy(state, xg);
      prng.state = function() { return copy(xg, {}); };
    }
    return prng;
  }

  if (module && module.exports) {
    module.exports = impl;
  } else if (define && define.amd) {
    define(function() { return impl; });
  } else {
    this.xorshift7 = impl;
  }

  })(
    commonjsGlobal,
     module,    // present in node.js
    (typeof undefined) == 'function'    // present with an AMD loader
  );
  });

  var xor4096 = createCommonjsModule(function (module) {
  // A Javascript implementaion of Richard Brent's Xorgens xor4096 algorithm.
  //
  // This fast non-cryptographic random number generator is designed for
  // use in Monte-Carlo algorithms. It combines a long-period xorshift
  // generator with a Weyl generator, and it passes all common batteries
  // of stasticial tests for randomness while consuming only a few nanoseconds
  // for each prng generated.  For background on the generator, see Brent's
  // paper: "Some long-period random number generators using shifts and xors."
  // http://arxiv.org/pdf/1004.3115v1.pdf
  //
  // Usage:
  //
  // var xor4096 = require('xor4096');
  // random = xor4096(1);                        // Seed with int32 or string.
  // assert.equal(random(), 0.1520436450538547); // (0, 1) range, 53 bits.
  // assert.equal(random.int32(), 1806534897);   // signed int32, 32 bits.
  //
  // For nonzero numeric keys, this impelementation provides a sequence
  // identical to that by Brent's xorgens 3 implementaion in C.  This
  // implementation also provides for initalizing the generator with
  // string seeds, or for saving and restoring the state of the generator.
  //
  // On Chrome, this prng benchmarks about 2.1 times slower than
  // Javascript's built-in Math.random().

  (function(global, module, define) {

  function XorGen(seed) {
    var me = this;

    // Set up generator function.
    me.next = function() {
      var w = me.w,
          X = me.X, i = me.i, t, v;
      // Update Weyl generator.
      me.w = w = (w + 0x61c88647) | 0;
      // Update xor generator.
      v = X[(i + 34) & 127];
      t = X[i = ((i + 1) & 127)];
      v ^= v << 13;
      t ^= t << 17;
      v ^= v >>> 15;
      t ^= t >>> 12;
      // Update Xor generator array state.
      v = X[i] = v ^ t;
      me.i = i;
      // Result is the combination.
      return (v + (w ^ (w >>> 16))) | 0;
    };

    function init(me, seed) {
      var t, v, i, j, w, X = [], limit = 128;
      if (seed === (seed | 0)) {
        // Numeric seeds initialize v, which is used to generates X.
        v = seed;
        seed = null;
      } else {
        // String seeds are mixed into v and X one character at a time.
        seed = seed + '\0';
        v = 0;
        limit = Math.max(limit, seed.length);
      }
      // Initialize circular array and weyl value.
      for (i = 0, j = -32; j < limit; ++j) {
        // Put the unicode characters into the array, and shuffle them.
        if (seed) v ^= seed.charCodeAt((j + 32) % seed.length);
        // After 32 shuffles, take v as the starting w value.
        if (j === 0) w = v;
        v ^= v << 10;
        v ^= v >>> 15;
        v ^= v << 4;
        v ^= v >>> 13;
        if (j >= 0) {
          w = (w + 0x61c88647) | 0;     // Weyl.
          t = (X[j & 127] ^= (v + w));  // Combine xor and weyl to init array.
          i = (0 == t) ? i + 1 : 0;     // Count zeroes.
        }
      }
      // We have detected all zeroes; make the key nonzero.
      if (i >= 128) {
        X[(seed && seed.length || 0) & 127] = -1;
      }
      // Run the generator 512 times to further mix the state before using it.
      // Factoring this as a function slows the main generator, so it is just
      // unrolled here.  The weyl generator is not advanced while warming up.
      i = 127;
      for (j = 4 * 128; j > 0; --j) {
        v = X[(i + 34) & 127];
        t = X[i = ((i + 1) & 127)];
        v ^= v << 13;
        t ^= t << 17;
        v ^= v >>> 15;
        t ^= t >>> 12;
        X[i] = v ^ t;
      }
      // Storing state as object members is faster than using closure variables.
      me.w = w;
      me.X = X;
      me.i = i;
    }

    init(me, seed);
  }

  function copy(f, t) {
    t.i = f.i;
    t.w = f.w;
    t.X = f.X.slice();
    return t;
  }
  function impl(seed, opts) {
    if (seed == null) seed = +(new Date);
    var xg = new XorGen(seed),
        state = opts && opts.state,
        prng = function() { return (xg.next() >>> 0) / 0x100000000; };
    prng.double = function() {
      do {
        var top = xg.next() >>> 11,
            bot = (xg.next() >>> 0) / 0x100000000,
            result = (top + bot) / (1 << 21);
      } while (result === 0);
      return result;
    };
    prng.int32 = xg.next;
    prng.quick = prng;
    if (state) {
      if (state.X) copy(state, xg);
      prng.state = function() { return copy(xg, {}); };
    }
    return prng;
  }

  if (module && module.exports) {
    module.exports = impl;
  } else if (define && define.amd) {
    define(function() { return impl; });
  } else {
    this.xor4096 = impl;
  }

  })(
    commonjsGlobal,                                     // window object or global
     module,    // present in node.js
    (typeof undefined) == 'function'    // present with an AMD loader
  );
  });

  var tychei = createCommonjsModule(function (module) {
  // A Javascript implementaion of the "Tyche-i" prng algorithm by
  // Samuel Neves and Filipe Araujo.
  // See https://eden.dei.uc.pt/~sneves/pubs/2011-snfa2.pdf

  (function(global, module, define) {

  function XorGen(seed) {
    var me = this, strseed = '';

    // Set up generator function.
    me.next = function() {
      var b = me.b, c = me.c, d = me.d, a = me.a;
      b = (b << 25) ^ (b >>> 7) ^ c;
      c = (c - d) | 0;
      d = (d << 24) ^ (d >>> 8) ^ a;
      a = (a - b) | 0;
      me.b = b = (b << 20) ^ (b >>> 12) ^ c;
      me.c = c = (c - d) | 0;
      me.d = (d << 16) ^ (c >>> 16) ^ a;
      return me.a = (a - b) | 0;
    };

    /* The following is non-inverted tyche, which has better internal
     * bit diffusion, but which is about 25% slower than tyche-i in JS.
    me.next = function() {
      var a = me.a, b = me.b, c = me.c, d = me.d;
      a = (me.a + me.b | 0) >>> 0;
      d = me.d ^ a; d = d << 16 ^ d >>> 16;
      c = me.c + d | 0;
      b = me.b ^ c; b = b << 12 ^ d >>> 20;
      me.a = a = a + b | 0;
      d = d ^ a; me.d = d = d << 8 ^ d >>> 24;
      me.c = c = c + d | 0;
      b = b ^ c;
      return me.b = (b << 7 ^ b >>> 25);
    }
    */

    me.a = 0;
    me.b = 0;
    me.c = 2654435769 | 0;
    me.d = 1367130551;

    if (seed === Math.floor(seed)) {
      // Integer seed.
      me.a = (seed / 0x100000000) | 0;
      me.b = seed | 0;
    } else {
      // String seed.
      strseed += seed;
    }

    // Mix in string seed, then discard an initial batch of 64 values.
    for (var k = 0; k < strseed.length + 20; k++) {
      me.b ^= strseed.charCodeAt(k) | 0;
      me.next();
    }
  }

  function copy(f, t) {
    t.a = f.a;
    t.b = f.b;
    t.c = f.c;
    t.d = f.d;
    return t;
  }
  function impl(seed, opts) {
    var xg = new XorGen(seed),
        state = opts && opts.state,
        prng = function() { return (xg.next() >>> 0) / 0x100000000; };
    prng.double = function() {
      do {
        var top = xg.next() >>> 11,
            bot = (xg.next() >>> 0) / 0x100000000,
            result = (top + bot) / (1 << 21);
      } while (result === 0);
      return result;
    };
    prng.int32 = xg.next;
    prng.quick = prng;
    if (state) {
      if (typeof(state) == 'object') copy(state, xg);
      prng.state = function() { return copy(xg, {}); };
    }
    return prng;
  }

  if (module && module.exports) {
    module.exports = impl;
  } else if (define && define.amd) {
    define(function() { return impl; });
  } else {
    this.tychei = impl;
  }

  })(
    commonjsGlobal,
     module,    // present in node.js
    (typeof undefined) == 'function'    // present with an AMD loader
  );
  });

  var seedrandom = createCommonjsModule(function (module) {
  /*
  Copyright 2014 David Bau.

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

  */

  (function (pool, math) {
  //
  // The following constants are related to IEEE 754 limits.
  //
  var global = this,
      width = 256,        // each RC4 output is 0 <= x < 256
      chunks = 6,         // at least six RC4 outputs for each double
      digits = 52,        // there are 52 significant digits in a double
      rngname = 'random', // rngname: name for Math.random and Math.seedrandom
      startdenom = math.pow(width, chunks),
      significance = math.pow(2, digits),
      overflow = significance * 2,
      mask = width - 1,
      nodecrypto;         // node.js crypto module, initialized at the bottom.

  //
  // seedrandom()
  // This is the seedrandom function described above.
  //
  function seedrandom(seed, options, callback) {
    var key = [];
    options = (options == true) ? { entropy: true } : (options || {});

    // Flatten the seed string or build one from local entropy if needed.
    var shortseed = mixkey(flatten(
      options.entropy ? [seed, tostring(pool)] :
      (seed == null) ? autoseed() : seed, 3), key);

    // Use the seed to initialize an ARC4 generator.
    var arc4 = new ARC4(key);

    // This function returns a random double in [0, 1) that contains
    // randomness in every bit of the mantissa of the IEEE 754 value.
    var prng = function() {
      var n = arc4.g(chunks),             // Start with a numerator n < 2 ^ 48
          d = startdenom,                 //   and denominator d = 2 ^ 48.
          x = 0;                          //   and no 'extra last byte'.
      while (n < significance) {          // Fill up all significant digits by
        n = (n + x) * width;              //   shifting numerator and
        d *= width;                       //   denominator and generating a
        x = arc4.g(1);                    //   new least-significant-byte.
      }
      while (n >= overflow) {             // To avoid rounding up, before adding
        n /= 2;                           //   last byte, shift everything
        d /= 2;                           //   right using integer math until
        x >>>= 1;                         //   we have exactly the desired bits.
      }
      return (n + x) / d;                 // Form the number within [0, 1).
    };

    prng.int32 = function() { return arc4.g(4) | 0; };
    prng.quick = function() { return arc4.g(4) / 0x100000000; };
    prng.double = prng;

    // Mix the randomness into accumulated entropy.
    mixkey(tostring(arc4.S), pool);

    // Calling convention: what to return as a function of prng, seed, is_math.
    return (options.pass || callback ||
        function(prng, seed, is_math_call, state) {
          if (state) {
            // Load the arc4 state from the given state if it has an S array.
            if (state.S) { copy(state, arc4); }
            // Only provide the .state method if requested via options.state.
            prng.state = function() { return copy(arc4, {}); };
          }

          // If called as a method of Math (Math.seedrandom()), mutate
          // Math.random because that is how seedrandom.js has worked since v1.0.
          if (is_math_call) { math[rngname] = prng; return seed; }

          // Otherwise, it is a newer calling convention, so return the
          // prng directly.
          else return prng;
        })(
    prng,
    shortseed,
    'global' in options ? options.global : (this == math),
    options.state);
  }
  math['seed' + rngname] = seedrandom;

  //
  // ARC4
  //
  // An ARC4 implementation.  The constructor takes a key in the form of
  // an array of at most (width) integers that should be 0 <= x < (width).
  //
  // The g(count) method returns a pseudorandom integer that concatenates
  // the next (count) outputs from ARC4.  Its return value is a number x
  // that is in the range 0 <= x < (width ^ count).
  //
  function ARC4(key) {
    var t, keylen = key.length,
        me = this, i = 0, j = me.i = me.j = 0, s = me.S = [];

    // The empty key [] is treated as [0].
    if (!keylen) { key = [keylen++]; }

    // Set up S using the standard key scheduling algorithm.
    while (i < width) {
      s[i] = i++;
    }
    for (i = 0; i < width; i++) {
      s[i] = s[j = mask & (j + key[i % keylen] + (t = s[i]))];
      s[j] = t;
    }

    // The "g" method returns the next (count) outputs as one number.
    (me.g = function(count) {
      // Using instance members instead of closure state nearly doubles speed.
      var t, r = 0,
          i = me.i, j = me.j, s = me.S;
      while (count--) {
        t = s[i = mask & (i + 1)];
        r = r * width + s[mask & ((s[i] = s[j = mask & (j + t)]) + (s[j] = t))];
      }
      me.i = i; me.j = j;
      return r;
      // For robust unpredictability, the function call below automatically
      // discards an initial batch of values.  This is called RC4-drop[256].
      // See http://google.com/search?q=rsa+fluhrer+response&btnI
    })(width);
  }

  //
  // copy()
  // Copies internal state of ARC4 to or from a plain object.
  //
  function copy(f, t) {
    t.i = f.i;
    t.j = f.j;
    t.S = f.S.slice();
    return t;
  }
  //
  // flatten()
  // Converts an object tree to nested arrays of strings.
  //
  function flatten(obj, depth) {
    var result = [], typ = (typeof obj), prop;
    if (depth && typ == 'object') {
      for (prop in obj) {
        try { result.push(flatten(obj[prop], depth - 1)); } catch (e) {}
      }
    }
    return (result.length ? result : typ == 'string' ? obj : obj + '\0');
  }

  //
  // mixkey()
  // Mixes a string seed into a key that is an array of integers, and
  // returns a shortened string seed that is equivalent to the result key.
  //
  function mixkey(seed, key) {
    var stringseed = seed + '', smear, j = 0;
    while (j < stringseed.length) {
      key[mask & j] =
        mask & ((smear ^= key[mask & j] * 19) + stringseed.charCodeAt(j++));
    }
    return tostring(key);
  }

  //
  // autoseed()
  // Returns an object for autoseeding, using window.crypto and Node crypto
  // module if available.
  //
  function autoseed() {
    try {
      var out;
      if (nodecrypto && (out = nodecrypto.randomBytes)) {
        // The use of 'out' to remember randomBytes makes tight minified code.
        out = out(width);
      } else {
        out = new Uint8Array(width);
        (global.crypto || global.msCrypto).getRandomValues(out);
      }
      return tostring(out);
    } catch (e) {
      var browser = global.navigator,
          plugins = browser && browser.plugins;
      return [+new Date, global, plugins, global.screen, tostring(pool)];
    }
  }

  //
  // tostring()
  // Converts an array of charcodes to a string
  //
  function tostring(a) {
    return String.fromCharCode.apply(0, a);
  }

  //
  // When seedrandom.js is loaded, we immediately mix a few bits
  // from the built-in RNG into the entropy pool.  Because we do
  // not want to interfere with deterministic PRNG state later,
  // seedrandom will not call math.random on its own again after
  // initialization.
  //
  mixkey(math.random(), pool);

  //
  // Nodejs and AMD support: export the implementation as a module using
  // either convention.
  //
  if ( module.exports) {
    module.exports = seedrandom;
    // When in node.js, try using crypto package for autoseeding.
    try {
      nodecrypto = require('crypto');
    } catch (ex) {}
  }

  // End anonymous scope, and pass initial values.
  })(
    [],     // pool: entropy pool starts empty
    Math    // math: package containing random, pow, and seedrandom
  );
  });

  // A library of seedable RNGs implemented in Javascript.
  //
  // Usage:
  //
  // var seedrandom = require('seedrandom');
  // var random = seedrandom(1); // or any seed.
  // var x = random();       // 0 <= x < 1.  Every bit is random.
  // var x = random.quick(); // 0 <= x < 1.  32 bits of randomness.

  // alea, a 53-bit multiply-with-carry generator by Johannes Baagøe.
  // Period: ~2^116
  // Reported to pass all BigCrush tests.


  // xor128, a pure xor-shift generator by George Marsaglia.
  // Period: 2^128-1.
  // Reported to fail: MatrixRank and LinearComp.


  // xorwow, George Marsaglia's 160-bit xor-shift combined plus weyl.
  // Period: 2^192-2^32
  // Reported to fail: CollisionOver, SimpPoker, and LinearComp.


  // xorshift7, by François Panneton and Pierre L'ecuyer, takes
  // a different approach: it adds robustness by allowing more shifts
  // than Marsaglia's original three.  It is a 7-shift generator
  // with 256 bits, that passes BigCrush with no systmatic failures.
  // Period 2^256-1.
  // No systematic BigCrush failures reported.


  // xor4096, by Richard Brent, is a 4096-bit xor-shift with a
  // very long period that also adds a Weyl generator. It also passes
  // BigCrush with no systematic failures.  Its long period may
  // be useful if you have many generators and need to avoid
  // collisions.
  // Period: 2^4128-2^32.
  // No systematic BigCrush failures reported.


  // Tyche-i, by Samuel Neves and Filipe Araujo, is a bit-shifting random
  // number generator derived from ChaCha, a modern stream cipher.
  // https://eden.dei.uc.pt/~sneves/pubs/2011-snfa2.pdf
  // Period: ~2^127
  // No systematic BigCrush failures reported.


  // The original ARC4-based prng included in this library.
  // Period: ~2^1600


  seedrandom.alea = alea;
  seedrandom.xor128 = xor128;
  seedrandom.xorwow = xorwow;
  seedrandom.xorshift7 = xorshift7;
  seedrandom.xor4096 = xor4096;
  seedrandom.tychei = tychei;

  var seedrandom$1 = seedrandom;
  var seedrandom_1 = seedrandom$1.alea;

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  // https://en.wikipedia.org/wiki/Marsaglia_polar_method
  class MPRandGauss {
      constructor(mean, stdDeviation, dtype, truncated, seed) {
          this.mean = mean;
          this.stdDev = stdDeviation;
          this.dtype = dtype;
          this.nextVal = NaN;
          this.truncated = truncated;
          if (this.truncated) {
              this.upper = this.mean + this.stdDev * 2;
              this.lower = this.mean - this.stdDev * 2;
          }
          const seedValue = seed ? seed : Math.random();
          this.random = seedrandom_1(seedValue.toString());
      }
      /** Returns next sample from a Gaussian distribution. */
      nextValue() {
          if (!isNaN(this.nextVal)) {
              const value = this.nextVal;
              this.nextVal = NaN;
              return value;
          }
          let resultX, resultY;
          let isValid = false;
          while (!isValid) {
              let v1, v2, s;
              do {
                  v1 = 2 * this.random() - 1;
                  v2 = 2 * this.random() - 1;
                  s = v1 * v1 + v2 * v2;
              } while (s >= 1 || s === 0);
              const mul = Math.sqrt(-2.0 * Math.log(s) / s);
              resultX = this.mean + this.stdDev * v1 * mul;
              resultY = this.mean + this.stdDev * v2 * mul;
              if (!this.truncated || this.isValidTruncated(resultX)) {
                  isValid = true;
              }
          }
          if (!this.truncated || this.isValidTruncated(resultY)) {
              this.nextVal = this.convertValue(resultY);
          }
          return this.convertValue(resultX);
      }
      /** Handles proper rounding for non-floating-point numbers. */
      convertValue(value) {
          if (this.dtype == null || this.dtype === 'float32') {
              return value;
          }
          return Math.round(value);
      }
      /** Returns true if less than 2-standard-deviations from the mean. */
      isValidTruncated(value) {
          return value <= this.upper && value >= this.lower;
      }
  }
  // Marsaglia, George, and Wai Wan Tsang. 2000. "A Simple Method for Generating
  // Gamma Variables."
  class RandGamma {
      constructor(alpha, beta, dtype, seed) {
          this.alpha = alpha;
          this.beta = 1 / beta; // convert rate to scale parameter
          this.dtype = dtype;
          const seedValue = seed ? seed : Math.random();
          this.randu = seedrandom_1(seedValue.toString());
          this.randn = new MPRandGauss(0, 1, dtype, false, this.randu());
          if (alpha < 1) {
              this.d = alpha + (2 / 3);
          }
          else {
              this.d = alpha - (1 / 3);
          }
          this.c = 1 / Math.sqrt(9 * this.d);
      }
      /** Returns next sample from a gamma distribution. */
      nextValue() {
          let x2, v0, v1, x, u, v;
          while (true) {
              do {
                  x = this.randn.nextValue();
                  v = 1 + (this.c * x);
              } while (v <= 0);
              v *= v * v;
              x2 = x * x;
              v0 = 1 - (0.331 * x2 * x2);
              v1 = (0.5 * x2) + (this.d * (1 - v + Math.log(v)));
              u = this.randu();
              if (u < v0 || Math.log(u) < v1) {
                  break;
              }
          }
          v = (1 / this.beta) * this.d * v;
          if (this.alpha < 1) {
              v *= Math.pow(this.randu(), 1 / this.alpha);
          }
          return this.convertValue(v);
      }
      /** Handles proper rounding for non-floating-point numbers. */
      convertValue(value) {
          if (this.dtype === 'float32') {
              return value;
          }
          return Math.round(value);
      }
  }
  class UniformRandom {
      constructor(min = 0, max = 1, dtype, seed) {
          /** Handles proper rounding for non floating point numbers. */
          this.canReturnFloat = () => (this.dtype == null || this.dtype === 'float32');
          this.min = min;
          this.range = max - min;
          this.dtype = dtype;
          if (seed == null) {
              seed = Math.random();
          }
          if (typeof seed === 'number') {
              seed = seed.toString();
          }
          if (!this.canReturnFloat() && this.range <= 1) {
              throw new Error(`The difference between ${min} - ${max} <= 1 and dtype is not float`);
          }
          this.random = seedrandom_1(seed);
      }
      convertValue(value) {
          if (this.canReturnFloat()) {
              return value;
          }
          return Math.round(value);
      }
      nextValue() {
          return this.convertValue(this.min + this.range * this.random());
      }
  }

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Creates a `tf.Tensor` with values sampled from a gamma distribution.
   *
   * ```js
   * tf.randomGamma([2, 2], 1).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param alpha The shape parameter of the gamma distribution.
   * @param beta The inverse scale parameter of the gamma distribution. Defaults
   *     to 1.
   * @param dtype The data type of the output. Defaults to float32.
   * @param seed The seed for the random number generator.
   */
  /** @doc {heading: 'Tensors', subheading: 'Random'} */
  function randomGamma_(shape, alpha, beta = 1, dtype = 'float32', seed) {
      if (beta == null) {
          beta = 1;
      }
      if (dtype == null) {
          dtype = 'float32';
      }
      if (dtype !== 'float32' && dtype !== 'int32') {
          throw new Error(`Unsupported data type ${dtype}`);
      }
      const rgamma = new RandGamma(alpha, beta, dtype, seed);
      const res = buffer(shape, dtype);
      for (let i = 0; i < res.values.length; i++) {
          res.values[i] = rgamma.nextValue();
      }
      return res.toTensor();
  }
  const randomGamma = op({ randomGamma_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Creates a `tf.Tensor` with values sampled from a normal distribution.
   *
   * ```js
   * tf.randomNormal([2, 2]).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param mean The mean of the normal distribution.
   * @param stdDev The standard deviation of the normal distribution.
   * @param dtype The data type of the output.
   * @param seed The seed for the random number generator.
   */
  /** @doc {heading: 'Tensors', subheading: 'Random'} */
  function randomNormal_(shape, mean = 0, stdDev = 1, dtype, seed) {
      if (dtype != null && dtype === 'bool') {
          throw new Error(`Unsupported data type ${dtype}`);
      }
      const randGauss = new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
      const res = buffer(shape, dtype);
      for (let i = 0; i < res.values.length; i++) {
          res.values[i] = randGauss.nextValue();
      }
      return res.toTensor();
  }
  const randomNormal = op({ randomNormal_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Creates a `tf.Tensor` with values sampled from a uniform distribution.
   *
   * The generated values follow a uniform distribution in the range [minval,
   * maxval). The lower bound minval is included in the range, while the upper
   * bound maxval is excluded.
   *
   * ```js
   * tf.randomUniform([2, 2]).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param minval The lower bound on the range of random values to generate.
   *   Defaults to 0.
   * @param maxval The upper bound on the range of random values to generate.
   *   Defaults to 1.
   * @param dtype The data type of the output tensor. Defaults to 'float32'.
   */
  /** @doc {heading: 'Tensors', subheading: 'Random'} */
  function randomUniform_(shape, minval = 0, maxval = 1, dtype = 'float32', seed) {
      const res = buffer(shape, dtype);
      const random = new UniformRandom(minval, maxval, null, seed);
      for (let i = 0; i < res.values.length; i++) {
          res.values[i] = random.nextValue();
      }
      return res.toTensor();
  }
  const randomUniform = op({ randomUniform_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Returns (a - b) * (a - b) element-wise.
   * Supports broadcasting.
   *
   * We also expose `tf.squaredDifferenceStrict` which has the same signature as
   * this op and asserts that `a` and `b` are the same shape (does not
   * broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 3, 16]);
   * const b = tf.tensor1d([1, 2, 9, 4]);
   *
   * a.squaredDifference(b).print();  // or tf.squaredDifference(a, b)
   * ```
   *
   * ```js
   * // Broadcast squared difference  a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(5);
   *
   * a.squaredDifference(b).print();  // or tf.squaredDifference(a, b)
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
  function squaredDifference_(a, b) {
      let $a = convertToTensor(a, 'a', 'squaredDifference');
      let $b = convertToTensor(b, 'b', 'squaredDifference');
      [$a, $b] = makeTypesMatch($a, $b);
      assertAndGetBroadcastShape($a.shape, $b.shape);
      const forward = (backend, save) => {
          const res = backend.squaredDifference($a, $b);
          save([$a, $b]);
          return res;
      };
      const inputs = { a: $a, b: $b };
      const attrs = {};
      return ENGINE.runKernelFunc(forward, inputs, null /* grad */, SquaredDifference, attrs);
  }
  const squaredDifference = op({ squaredDifference_ });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  /**
   * Creates a `tf.Tensor` with values sampled from a truncated normal
   * distribution.
   *
   * ```js
   * tf.truncatedNormal([2, 2]).print();
   * ```
   *
   * The generated values follow a normal distribution with specified mean and
   * standard deviation, except that values whose magnitude is more than 2
   * standard deviations from the mean are dropped and re-picked.
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param mean The mean of the normal distribution.
   * @param stdDev The standard deviation of the normal distribution.
   * @param dtype The data type of the output tensor.
   * @param seed The seed for the random number generator.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  function truncatedNormal_(shape, mean = 0, stdDev = 1, dtype, seed) {
      if (dtype != null && dtype === 'bool') {
          throw new Error(`Unsupported data type $ { dtype }`);
      }
      const randGauss = new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
      const res = buffer(shape, dtype);
      for (let i = 0; i < res.values.length; i++) {
          res.values[i] = randGauss.nextValue();
      }
      return res.toTensor();
  }
  const truncatedNormal = op({ truncatedNormal_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Strict version of `tf.notEqual` that forces `a` and `b` to be of the same
   * shape.
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same shape and dtype as
   *     `a`.
   */
  function notEqualStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'notEqualStrict');
      const $b = convertToTensor(b, 'b', 'notEqualStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in notEqualStrict: ');
      return $a.notEqual($b);
  }
  /**
   * Returns the truth value of (a < b) element-wise. Supports broadcasting.
   *
   * We also expose `tf.lessStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.less(b).print();
   * ```
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function less_(a, b) {
      let $a = convertToTensor(a, 'a', 'less');
      let $b = convertToTensor(b, 'b', 'less');
      [$a, $b] = makeTypesMatch($a, $b);
      assertAndGetBroadcastShape($a.shape, $b.shape);
      return ENGINE.runKernelFunc(backend => backend.less($a, $b), { a: $a, b: $b }, null /* grad */, 'Less');
  }
  /**
   * Strict version of `tf.less` that forces `a` and `b` to be of the same
   * shape.
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same shape and dtype as
   *     `a`.
   */
  function lessStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'lessStrict');
      const $b = convertToTensor(b, 'b', 'lessStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in lessStrict: ');
      return $a.less($b);
  }
  /**
   * Returns the truth value of (a == b) element-wise. Supports broadcasting.
   *
   * We also expose `tf.equalStrict` which has the same signature as this op
   * and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.equal(b).print();
   * ```
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function equal_(a, b) {
      let $a = convertToTensor(a, 'a', 'equal');
      let $b = convertToTensor(b, 'b', 'equal');
      [$a, $b] = makeTypesMatch($a, $b);
      assertAndGetBroadcastShape($a.shape, $b.shape);
      return ENGINE.runKernelFunc(backend => backend.equal($a, $b), { $a, $b });
  }
  function equalStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'equalStrict');
      const $b = convertToTensor(b, 'b', 'equalStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in equalStrict: ');
      return $a.equal($b);
  }
  /**
   * Returns the truth value of (a <= b) element-wise. Supports broadcasting.
   *
   * We also expose `tf.lessEqualStrict` which has the same signature as this op
   * and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.lessEqual(b).print();
   * ```
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function lessEqual_(a, b) {
      let $a = convertToTensor(a, 'a', 'lessEqual');
      let $b = convertToTensor(b, 'b', 'lessEqual');
      [$a, $b] = makeTypesMatch($a, $b);
      assertAndGetBroadcastShape($a.shape, $b.shape);
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.lessEqual($a, $b);
          save([$a, $b]);
          return res;
      }, { a: $a, b: $b }, null /* grad */, 'LessEqual');
  }
  function lessEqualStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'lessEqualStrict');
      const $b = convertToTensor(b, 'b', 'lessEqualStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in lessEqualStrict: ');
      return $a.lessEqual($b);
  }
  /**
   * Returns the truth value of (a > b) element-wise. Supports broadcasting.
   *
   * We also expose `tf.greaterStrict` which has the same signature as this
   * op and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.greater(b).print();
   * ```
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function greater_(a, b) {
      let $a = convertToTensor(a, 'a', 'greater');
      let $b = convertToTensor(b, 'b', 'greater');
      [$a, $b] = makeTypesMatch($a, $b);
      assertAndGetBroadcastShape($a.shape, $b.shape);
      return ENGINE.runKernelFunc(backend => backend.greater($a, $b), { a: $a, b: $b }, null /* grad */, 'Greater');
  }
  function greaterStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'greaterStrict');
      const $b = convertToTensor(b, 'b', 'greaterStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in greaterStrict: ');
      return $a.greater($b);
  }
  /**
   * Returns the truth value of (a >= b) element-wise. Supports broadcasting.
   *
   * We also expose `tf.greaterEqualStrict` which has the same signature as this
   * op and asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([2, 2, 2]);
   *
   * a.greaterEqual(b).print();
   * ```
   *
   * @param a The first input tensor.
   * @param b The second input tensor. Must have the same dtype as `a`.
   */
  /** @doc {heading: 'Operations', subheading: 'Logical'} */
  function greaterEqual_(a, b) {
      let $a = convertToTensor(a, 'a', 'greaterEqual');
      let $b = convertToTensor(b, 'b', 'greaterEqual');
      [$a, $b] = makeTypesMatch($a, $b);
      assertAndGetBroadcastShape($a.shape, $b.shape);
      const grad = (dy, saved) => {
          const [$a, $b] = saved;
          return { a: () => zerosLike($a), b: () => zerosLike($b) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.greaterEqual($a, $b);
          save([$a, $b]);
          return res;
      }, { a: $a, b: $b }, grad, 'GreaterEqual');
  }
  function greaterEqualStrict_(a, b) {
      const $a = convertToTensor(a, 'a', 'greaterEqualStrict');
      const $b = convertToTensor(b, 'b', 'greaterEqualStrict');
      assertShapesMatch($a.shape, $b.shape, 'Error in greaterEqualStrict: ');
      return $a.greaterEqual($b);
  }
  const equal = op({ equal_ });
  const equalStrict = op({ equalStrict_ });
  const greater = op({ greater_ });
  const greaterEqual = op({ greaterEqual_ });
  const greaterEqualStrict = op({ greaterEqualStrict_ });
  const greaterStrict = op({ greaterStrict_ });
  const less = op({ less_ });
  const lessEqual = op({ lessEqual_ });
  const lessEqualStrict = op({ lessEqualStrict_ });
  const lessStrict = op({ lessStrict_ });
  const notEqualStrict = op({ notEqualStrict_ });

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
  const PARALLELIZE_THRESHOLD = 30;
  function computeOptimalWindowSize(inSize) {
      if (inSize <= PARALLELIZE_THRESHOLD) {
          return inSize;
      }
      return nearestDivisor(inSize, Math.floor(Math.sqrt(inSize)));
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  function segOpComputeOptimalWindowSize(inSize, numSegments) {
      let done = false;
      let res;
      if (inSize <= PARALLELIZE_THRESHOLD) {
          res = inSize;
          done = true;
      }
      else {
          res = nearestDivisor(inSize, Math.floor(Math.sqrt(inSize)));
      }
      while (!done) {
          if (res > numSegments || res === inSize) {
              done = true;
          }
          else {
              res = nearestDivisor(inSize, res + 1);
          }
      }
      return res;
  }
  function computeOutShape$2(aShape, axis, numSegments) {
      const outShape = [];
      const rank = aShape.length;
      for (let dim = 0; dim < rank; dim++) {
          if (dim !== axis) {
              outShape.push(aShape[dim]);
          }
          else {
              outShape.push(numSegments);
          }
      }
      return outShape;
  }
  function collectGatherOpShapeInfo(x, indices, axis) {
      const dimSize = x.shape[axis];
      const outputShape = [];
      let batchSize = 1;
      let sliceSize = 1;
      for (let i = 0; i < axis; i++) {
          outputShape.push(x.shape[i]);
          batchSize *= x.shape[i];
      }
      for (let i = 0; i < indices.rank; i++) {
          outputShape.push(indices.shape[i]);
      }
      for (let i = axis + 1; i < x.rank; i++) {
          outputShape.push(x.shape[i]);
          sliceSize *= x.shape[i];
      }
      return { batchSize, sliceSize, dimSize, outputShape };
  }

  var segment_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    segOpComputeOptimalWindowSize: segOpComputeOptimalWindowSize,
    computeOutShape: computeOutShape$2,
    collectGatherOpShapeInfo: collectGatherOpShapeInfo
  });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes the sum along segments of a `tf.Tensor`.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * const segmentIds = tf.tensor1d([1, 2, 0, 1], 'int32');
   * const numSegments = 3;
   *
   * x.unsortedSegmentSum(segmentIds, numSegments).print()
   * //or tf.unsortedSegmentSum(x, segmentIds, numSegments)
   * ```
   * @param x The `tf.Tensor` that will be summed along its segments.
   * @param segmentIds A `tf.Tensor1D` whose rank is equal to the rank of `x`'s
   * dimension along the `axis`.  Maps each element of `x` to a segment.
   * @param numSegments The number of distinct `segmentIds`.
   */
  /** @doc {heading: 'Operations', subheading: 'Segment'} */
  function unsortedSegmentSum_(x, segmentIds, numSegments) {
      const $x = convertToTensor(x, 'x', 'unsortedSegmentSum');
      const $segmentIds = convertToTensor(segmentIds, 'segmentIds', 'unsortedSegmentSum', 'int32');
      assert(isInt(numSegments), () => 'numSegments must be of dtype int');
      const gradFunc = (dy, saved) => {
          const [$segmentIds] = saved;
          const derX = () => {
              return gatherDropNegatives(dy, $segmentIds);
          };
          return { $x: derX };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.unsortedSegmentSum($x, $segmentIds, numSegments);
          save([$segmentIds]);
          return res;
      }, { $x }, gradFunc);
  }
  /**
   * Gather slices from tensor `x`'s axis `axis` according to `indices`.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * const indices = tf.tensor1d([1, 3, 3], 'int32');
   *
   * x.gather(indices).print();
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   * const indices = tf.tensor1d([1, 1, 0], 'int32');
   *
   * x.gather(indices).print();
   * ```
   * @param x The input tensor whose slices to be gathered.
   * @param indices The indices of the values to extract.
   * @param axis The axis over which to select values. Defaults to 0.
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  function gather_(x, indices, axis = 0) {
      const $x = convertToTensor(x, 'x', 'gather');
      const $indices = convertToTensor(indices, 'indices', 'gather', 'int32');
      axis = parseAxisParam(axis, $x.shape)[0];
      const shapeInfo = collectGatherOpShapeInfo($x, $indices, axis);
      const grad = (dy, saved) => {
          const [$indices] = saved;
          const derX = () => {
              const paramsShape = $x.shape;
              const indicesSize = $indices.size;
              const outerShape = paramsShape.slice(0, axis);
              const outerDims = outerShape.length;
              const innerShape = paramsShape.slice(axis, paramsShape.length).slice(1);
              const innerDims = innerShape.length;
              const outerAxesIndices = arrayRange(0, outerDims);
              const innerAxesIndices = arrayRange(outerDims + 1, outerDims + 1 + innerDims);
              const valuesShape = arrayConcat([outerShape, [indicesSize], innerShape]);
              const values = dy.reshape(valuesShape);
              const reshapedIndices = $indices.reshape([indicesSize]);
              const transposeDims = arrayConcat([[outerDims], outerAxesIndices, innerAxesIndices]);
              const valuesTranspose = values.transpose(transposeDims);
              let paramsGrad = unsortedSegmentSum(valuesTranspose, reshapedIndices, $x.shape[axis]);
              const invertTransposeDims = getUndoAxesPermutation(transposeDims);
              paramsGrad = paramsGrad.transpose(invertTransposeDims);
              return paramsGrad;
          };
          return { x: derX, indices: () => $indices };
      };
      return (ENGINE.runKernelFunc((backend, save) => {
          const res = backend.gather($x, $indices.flatten(), axis);
          save([$indices]);
          return res;
      }, { x: $x, indices: $indices }, grad, 'Gather', { axis }))
          .reshape(shapeInfo.outputShape);
  }
  function arrayRange(start, stop) {
      const result = [];
      for (let i = start; i < stop; ++i) {
          result.push(i);
      }
      return result;
  }
  function arrayConcat(arrays) {
      const result = [];
      for (let i = 0; i < arrays.length; ++i) {
          for (let j = 0; j < arrays[i].length; ++j) {
              result.push(arrays[i][j]);
          }
      }
      return result;
  }
  function gatherDropNegatives(x, indices) {
      // Helper function for unsorted segment ops. Gathers params for
      // positive segment ids and gathers 0 for inputs with negative segment id.
      // Mirrors _GatherDropNegatives from tensorflow/python/ops/math_grad.py
      const zeroClippedIndices = maximum(indices, zerosLike(indices));
      const gathered = gather(x, zeroClippedIndices);
      let isPositive = greaterEqual(indices, scalar(0, 'int32'));
      const numIters = gathered.rank - isPositive.rank;
      for (let i = 0; i < numIters; ++i) {
          isPositive = expandDims(isPositive, i + 1);
      }
      isPositive = logicalAnd(isPositive, ones$1(gathered.shape, 'bool'));
      const zeroSlice = zerosLike(gathered);
      return where(isPositive, gathered, zeroSlice);
  }
  const gather = op({ gather_ });
  const unsortedSegmentSum = op({ unsortedSegmentSum_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Apply boolean mask to tensor.
   *
   * ```js
   * const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
   * const mask = tf.tensor1d([1, 0, 1], 'bool');
   * const result = await tf.booleanMaskAsync(tensor, mask);
   * result.print();
   * ```
   *
   * @param tensor N-D tensor.
   * @param mask K-D boolean tensor, K <= N and K must be known statically.
   * @param axis A 0-D int Tensor representing the axis in tensor to mask from.
   *     By default, axis is 0 which will mask from the first dimension.
   *     Otherwise K + axis <= N.
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  async function booleanMaskAsync_(tensor, mask, axis) {
      const $tensor = convertToTensor(tensor, 'tensor', 'boolMask');
      const $mask = convertToTensor(mask, 'mask', 'boolMask', 'bool');
      const axisFrom = axis == null ? 0 : axis;
      const maskDim = $mask.rank;
      const tensorShape = $tensor.shape;
      assert(maskDim > 0, () => 'mask cannot be scalar');
      assertShapesMatch(tensorShape.slice(axisFrom, axisFrom + maskDim), $mask.shape, `mask's shape must match the first K dimensions of tensor's shape,`);
      let leadingSize = 1;
      for (let i = axisFrom; i < axisFrom + maskDim; i++) {
          leadingSize *= tensorShape[i];
      }
      const targetTensorShape = tensorShape.slice(0, axisFrom)
          .concat([leadingSize], tensorShape.slice(axisFrom + maskDim));
      const reshapedTensor = $tensor.reshape(targetTensorShape);
      const reshapedMask = $mask.reshape([-1]);
      const positivePositions = await whereAsync(reshapedMask);
      const indices = positivePositions.squeeze([1]);
      const res = gather(reshapedTensor, indices, axisFrom);
      // Ensure no memory leak.
      if (tensor !== $tensor) {
          $tensor.dispose();
      }
      if (mask !== $mask) {
          $mask.dispose();
      }
      indices.dispose();
      reshapedTensor.dispose();
      reshapedMask.dispose();
      positivePositions.dispose();
      return res;
  }
  const booleanMaskAsync = booleanMaskAsync_;

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
  function computePool2DInfo(inShape, filterSize, strides, dilations, pad, roundingMode, dataFormat = 'channelsLast') {
      const [filterHeight, filterWidth] = parseTupleParam(filterSize);
      let filterShape;
      if (dataFormat === 'channelsLast') {
          filterShape = [filterHeight, filterWidth, inShape[3], inShape[3]];
      }
      else if (dataFormat === 'channelsFirst') {
          filterShape = [filterHeight, filterWidth, inShape[1], inShape[1]];
      }
      else {
          throw new Error(`Unknown dataFormat ${dataFormat}`);
      }
      return computeConv2DInfo(inShape, filterShape, strides, dilations, pad, roundingMode, false, dataFormat);
  }
  /**
   * Computes the information for a forward pass of a pooling3D operation.
   */
  function computePool3DInfo(inShape, filterSize, strides, dilations, pad, roundingMode, dataFormat = 'NDHWC') {
      const [filterDepth, filterHeight, filterWidth] = parse3TupleParam(filterSize);
      let filterShape;
      let $dataFormat;
      if (dataFormat === 'NDHWC') {
          $dataFormat = 'channelsLast';
          filterShape =
              [filterDepth, filterHeight, filterWidth, inShape[4], inShape[4]];
      }
      else if (dataFormat === 'NCDHW') {
          $dataFormat = 'channelsFirst';
          filterShape =
              [filterDepth, filterHeight, filterWidth, inShape[1], inShape[1]];
      }
      else {
          throw new Error(`Unknown dataFormat ${dataFormat}`);
      }
      return computeConv3DInfo(inShape, filterShape, strides, dilations, pad, false, $dataFormat, roundingMode);
  }
  /**
   * Computes the information for a forward pass of a convolution/pooling
   * operation.
   */
  function computeConv2DInfo(inShape, filterShape, strides, dilations, pad, roundingMode, depthwise = false, dataFormat = 'channelsLast') {
      let [batchSize, inHeight, inWidth, inChannels] = [-1, -1, -1, -1];
      if (dataFormat === 'channelsLast') {
          [batchSize, inHeight, inWidth, inChannels] = inShape;
      }
      else if (dataFormat === 'channelsFirst') {
          [batchSize, inChannels, inHeight, inWidth] = inShape;
      }
      else {
          throw new Error(`Unknown dataFormat ${dataFormat}`);
      }
      const [filterHeight, filterWidth, , filterChannels] = filterShape;
      const [strideHeight, strideWidth] = parseTupleParam(strides);
      const [dilationHeight, dilationWidth] = parseTupleParam(dilations);
      const effectiveFilterHeight = getEffectiveFilterSize(filterHeight, dilationHeight);
      const effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
      const { padInfo, outHeight, outWidth } = getPadAndOutInfo(pad, inHeight, inWidth, strideHeight, strideWidth, effectiveFilterHeight, effectiveFilterWidth, roundingMode);
      const outChannels = depthwise ? filterChannels * inChannels : filterChannels;
      let outShape;
      if (dataFormat === 'channelsFirst') {
          outShape = [batchSize, outChannels, outHeight, outWidth];
      }
      else if (dataFormat === 'channelsLast') {
          outShape = [batchSize, outHeight, outWidth, outChannels];
      }
      return {
          batchSize,
          dataFormat,
          inHeight,
          inWidth,
          inChannels,
          outHeight,
          outWidth,
          outChannels,
          padInfo,
          strideHeight,
          strideWidth,
          filterHeight,
          filterWidth,
          effectiveFilterHeight,
          effectiveFilterWidth,
          dilationHeight,
          dilationWidth,
          inShape,
          outShape,
          filterShape
      };
  }
  /**
   * Computes the information for a forward pass of a 3D convolution/pooling
   * operation.
   */
  function computeConv3DInfo(inShape, filterShape, strides, dilations, pad, depthwise = false, dataFormat = 'channelsLast', roundingMode) {
      let [batchSize, inDepth, inHeight, inWidth, inChannels] = [-1, -1, -1, -1, -1];
      if (dataFormat === 'channelsLast') {
          [batchSize, inDepth, inHeight, inWidth, inChannels] = inShape;
      }
      else if (dataFormat === 'channelsFirst') {
          [batchSize, inChannels, inDepth, inHeight, inWidth] = inShape;
      }
      else {
          throw new Error(`Unknown dataFormat ${dataFormat}`);
      }
      const [filterDepth, filterHeight, filterWidth, , filterChannels] = filterShape;
      const [strideDepth, strideHeight, strideWidth] = parse3TupleParam(strides);
      const [dilationDepth, dilationHeight, dilationWidth] = parse3TupleParam(dilations);
      const effectiveFilterDepth = getEffectiveFilterSize(filterDepth, dilationDepth);
      const effectiveFilterHeight = getEffectiveFilterSize(filterHeight, dilationHeight);
      const effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
      const { padInfo, outDepth, outHeight, outWidth } = get3DPadAndOutInfo(pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth, effectiveFilterDepth, effectiveFilterHeight, effectiveFilterWidth, roundingMode);
      const outChannels = depthwise ? filterChannels * inChannels : filterChannels;
      let outShape;
      if (dataFormat === 'channelsFirst') {
          outShape = [batchSize, outChannels, outDepth, outHeight, outWidth];
      }
      else if (dataFormat === 'channelsLast') {
          outShape = [batchSize, outDepth, outHeight, outWidth, outChannels];
      }
      return {
          batchSize,
          dataFormat,
          inDepth,
          inHeight,
          inWidth,
          inChannels,
          outDepth,
          outHeight,
          outWidth,
          outChannels,
          padInfo,
          strideDepth,
          strideHeight,
          strideWidth,
          filterDepth,
          filterHeight,
          filterWidth,
          effectiveFilterDepth,
          effectiveFilterHeight,
          effectiveFilterWidth,
          dilationDepth,
          dilationHeight,
          dilationWidth,
          inShape,
          outShape,
          filterShape
      };
  }
  function computeOutputShape2D(inShape, fieldSize, stride, zeroPad, roundingMode) {
      if (zeroPad == null) {
          zeroPad = computeDefaultPad(inShape, fieldSize, stride);
      }
      const inputRows = inShape[0];
      const inputCols = inShape[1];
      const outputRows = conditionalRound((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
      assert(isInt(outputRows), () => `The output # of rows (${outputRows}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);
      const outputCols = conditionalRound((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
      assert(isInt(outputCols), () => `The output # of columns (${outputCols}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);
      return [outputRows, outputCols];
  }
  function computeOutputShape4D(inShape, fieldSize, outChannels, stride, zeroPad, roundingMode) {
      if (zeroPad == null) {
          zeroPad = computeDefaultPad(inShape, fieldSize, stride);
      }
      const inputDepth = inShape[0];
      const inputRows = inShape[1];
      const inputCols = inShape[2];
      const outputDepths = conditionalRound((inputDepth - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
      assert(isInt(outputDepths), () => `The output # of depths (${outputDepths}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);
      const outputRows = conditionalRound((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
      assert(isInt(outputRows), () => `The output # of rows (${outputRows}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);
      const outputCols = conditionalRound((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
      assert(isInt(outputCols), () => `The output # of columns (${outputCols}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);
      return [outputDepths, outputRows, outputCols, outChannels];
  }
  function computeDefaultPad(inputShape, fieldSize, stride, dilation = 1) {
      const effectiveFieldSize = getEffectiveFilterSize(fieldSize, dilation);
      return Math.floor((inputShape[0] * (stride - 1) - stride + effectiveFieldSize) / 2);
  }
  function parseTupleParam(param) {
      if (typeof param === 'number') {
          return [param, param, param];
      }
      if (param.length === 2) {
          return [param[0], param[1], 1];
      }
      return param;
  }
  function parse3TupleParam(param) {
      return typeof param === 'number' ? [param, param, param] : param;
  }
  /* See https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
   * Atrous convolution is equivalent to standard convolution with upsampled
   * filters with effective_filter_height =
   * filter_height + (filter_height - 1) * (dilation - 1)
   * and effective_filter_width =
   * filter_width + (filter_width - 1) * (dilation - 1),
   * produced by inserting dilation - 1 zeros along consecutive elements across
   * the filters' spatial dimensions.
   * When there is a dilation, this converts a filter dimension to the
   * effective filter dimension, so it can be used in a standard convolution.
   */
  function getEffectiveFilterSize(filterSize, dilation) {
      if (dilation <= 1) {
          return filterSize;
      }
      return filterSize + (filterSize - 1) * (dilation - 1);
  }
  function getPadAndOutInfo(pad, inHeight, inWidth, strideHeight, strideWidth, filterHeight, filterWidth, roundingMode) {
      let padInfo;
      let outHeight;
      let outWidth;
      if (typeof pad === 'number') {
          const padType = (pad === 0) ? 'VALID' : 'NUMBER';
          padInfo = { top: pad, bottom: pad, left: pad, right: pad, type: padType };
          const outShape = computeOutputShape2D([inHeight, inWidth], filterHeight, strideHeight, pad, roundingMode);
          outHeight = outShape[0];
          outWidth = outShape[1];
      }
      else if (pad === 'same') {
          outHeight = Math.ceil(inHeight / strideHeight);
          outWidth = Math.ceil(inWidth / strideWidth);
          const padAlongHeight = Math.max(0, (outHeight - 1) * strideHeight + filterHeight - inHeight);
          const padAlongWidth = Math.max(0, (outWidth - 1) * strideWidth + filterWidth - inWidth);
          const top = Math.floor(padAlongHeight / 2);
          const bottom = padAlongHeight - top;
          const left = Math.floor(padAlongWidth / 2);
          const right = padAlongWidth - left;
          padInfo = { top, bottom, left, right, type: 'SAME' };
      }
      else if (pad === 'valid') {
          padInfo = { top: 0, bottom: 0, left: 0, right: 0, type: 'VALID' };
          outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
          outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
      }
      else {
          throw Error(`Unknown padding parameter: ${pad}`);
      }
      return { padInfo, outHeight, outWidth };
  }
  function get3DPadAndOutInfo(pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth, filterDepth, filterHeight, filterWidth, roundingMode) {
      let padInfo;
      let outDepth;
      let outHeight;
      let outWidth;
      if (typeof pad === 'number') {
          const padType = (pad === 0) ? 'VALID' : 'NUMBER';
          padInfo = {
              top: pad,
              bottom: pad,
              left: pad,
              right: pad,
              front: pad,
              back: pad,
              type: padType
          };
          const outShape = computeOutputShape4D([inDepth, inHeight, inWidth, 1], filterDepth, 1, strideDepth, pad, roundingMode);
          outDepth = outShape[0];
          outHeight = outShape[1];
          outWidth = outShape[2];
      }
      else if (pad === 'same') {
          outDepth = Math.ceil(inDepth / strideDepth);
          outHeight = Math.ceil(inHeight / strideHeight);
          outWidth = Math.ceil(inWidth / strideWidth);
          const padAlongDepth = (outDepth - 1) * strideDepth + filterDepth - inDepth;
          const padAlongHeight = (outHeight - 1) * strideHeight + filterHeight - inHeight;
          const padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
          const front = Math.floor(padAlongDepth / 2);
          const back = padAlongDepth - front;
          const top = Math.floor(padAlongHeight / 2);
          const bottom = padAlongHeight - top;
          const left = Math.floor(padAlongWidth / 2);
          const right = padAlongWidth - left;
          padInfo = { top, bottom, left, right, front, back, type: 'SAME' };
      }
      else if (pad === 'valid') {
          padInfo = {
              top: 0,
              bottom: 0,
              left: 0,
              right: 0,
              front: 0,
              back: 0,
              type: 'VALID'
          };
          outDepth = Math.ceil((inDepth - filterDepth + 1) / strideDepth);
          outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
          outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
      }
      else {
          throw Error(`Unknown padding parameter: ${pad}`);
      }
      return { padInfo, outDepth, outHeight, outWidth };
  }
  /**
   * Rounds a value depending on the rounding mode
   * @param value
   * @param roundingMode
   */
  function conditionalRound(value, roundingMode) {
      if (!roundingMode) {
          return value;
      }
      switch (roundingMode) {
          case 'round':
              // used for Caffe Conv
              return Math.round(value);
          case 'ceil':
              // used for Caffe Pool
              return Math.ceil(value);
          case 'floor':
              return Math.floor(value);
          default:
              throw new Error(`Unknown roundingMode ${roundingMode}`);
      }
  }
  function tupleValuesAreOne(param) {
      const [dimA, dimB, dimC] = parseTupleParam(param);
      return dimA === 1 && dimB === 1 && dimC === 1;
  }
  function eitherStridesOrDilationsAreOne(strides, dilations) {
      return tupleValuesAreOne(strides) || tupleValuesAreOne(dilations);
  }
  /**
   * Convert Conv2D dataFormat from 'NHWC'|'NCHW' to
   *    'channelsLast'|'channelsFirst'
   * @param dataFormat in 'NHWC'|'NCHW' mode
   * @return dataFormat in 'channelsLast'|'channelsFirst' mode
   * @throws unknown dataFormat
   */
  function convertConv2DDataFormat(dataFormat) {
      if (dataFormat === 'NHWC') {
          return 'channelsLast';
      }
      else if (dataFormat === 'NCHW') {
          return 'channelsFirst';
      }
      else {
          throw new Error(`Unknown dataFormat ${dataFormat}`);
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes a 1D convolution over the input x.
   *
   * @param x The input tensor, of rank 3 or rank 2, of shape
   *     `[batch, width, inChannels]`. If rank 2, batch of 1 is assumed.
   * @param filter The filter, rank 3, of shape
   *     `[filterWidth, inDepth, outDepth]`.
   * @param stride The number of entries by which the filter is moved right at
   *     each step.
   * @param pad The type of padding algorithm.
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dataFormat An optional string from "NWC", "NCW". Defaults to "NWC",
   *     the data is stored in the order of [batch, in_width, in_channels]. Only
   *     "NWC" is currently supported.
   * @param dilation The dilation rate in which we sample input values in
   *     atrous convolution. Defaults to `1`. If it is greater than 1, then
   *     stride must be `1`.
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function conv1d_(x, filter, stride, pad, dataFormat = 'NWC', dilation = 1, dimRoundingMode) {
      const $x = convertToTensor(x, 'x', 'conv1d');
      const $filter = convertToTensor(filter, 'filter', 'conv1d');
      let x3D = $x;
      let reshapedTo3D = false;
      if ($x.rank === 2) {
          reshapedTo3D = true;
          x3D = $x.as3D(1, $x.shape[0], $x.shape[1]);
      }
      assert(x3D.rank === 3, () => `Error in conv1d: input must be rank 3, but got rank ${x3D.rank}.`);
      assert($filter.rank === 3, () => `Error in conv1d: filter must be rank 3, but got rank ` +
          `${$filter.rank}.`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in conv1d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      assert(x3D.shape[2] === $filter.shape[1], () => `Error in conv1d: depth of input (${x3D.shape[2]}) must match ` +
          `input depth for filter ${$filter.shape[1]}.`);
      assert(eitherStridesOrDilationsAreOne(stride, dilation), () => 'Error in conv1D: Either stride or dilation must be 1. ' +
          `Got stride ${stride} and dilation '${dilation}'`);
      assert(dataFormat === 'NWC', () => `Error in conv1d: got dataFormat of ${dataFormat} but only NWC is currently supported.`);
      const filter4D = $filter.as4D(1, $filter.shape[0], $filter.shape[1], $filter.shape[2]);
      const input4D = x3D.as4D(x3D.shape[0], 1, x3D.shape[1], x3D.shape[2]);
      const strides = [1, stride];
      const dilations = [1, dilation];
      const conv2dDataFormat = 'NHWC';
      const res = conv2d(input4D, filter4D, strides, pad, conv2dDataFormat, dilations, dimRoundingMode);
      if (reshapedTo3D) {
          return res.as2D(res.shape[2], res.shape[3]);
      }
      return res.as3D(res.shape[0], res.shape[2], res.shape[3]);
  }
  /**
   * Computes a 2D convolution over the input x.
   *
   * @param x The input tensor, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
   * assumed.
   * @param filter The filter, rank 4, of shape
   *     `[filterHeight, filterWidth, inDepth, outDepth]`.
   * @param strides The strides of the convolution: `[strideHeight,
   * strideWidth]`.
   * @param pad The type of padding algorithm.
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
   *     "NHWC". Specify the data format of the input and output data. With the
   *     default format "NHWC", the data is stored in the order of: [batch,
   *     height, width, channels].
   * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
   *     in which we sample input values across the height and width dimensions
   *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
   *     number, then `dilationHeight == dilationWidth`. If it is greater than
   *     1, then all values of `strides` must be 1.
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function conv2d_(x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode) {
      const $x = convertToTensor(x, 'x', 'conv2d');
      const $filter = convertToTensor(filter, 'filter', 'conv2d');
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      assert(x4D.rank === 4, () => `Error in conv2d: input must be rank 4, but got rank ${x4D.rank}.`);
      assert($filter.rank === 4, () => `Error in conv2d: filter must be rank 4, but got rank ` +
          `${$filter.rank}.`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in conv2d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const inDepth = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
      assert(inDepth === $filter.shape[2], () => `Error in conv2d: depth of input (${inDepth}) must match ` +
          `input depth for filter ${$filter.shape[2]}.`);
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in conv2D: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      const $dataFormat = convertConv2DDataFormat(dataFormat);
      const convInfo = computeConv2DInfo(x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode, false, $dataFormat);
      const grad = (dy, saved) => {
          const [$filter, x4D] = saved;
          assert(tupleValuesAreOne(dilations), () => 'Error in gradient of conv2D: dilation rates greater than 1 ' +
              `are not yet supported in gradients. Got dilations '${dilations}'`);
          return {
              x: () => conv2dDerInput(x4D.shape, dy, $filter, strides, pad, dataFormat),
              filter: () => conv2dDerFilter(x4D, dy, $filter.shape, strides, pad, dataFormat)
          };
      };
      const inputsToSave = [$filter, x4D];
      const res = ENGINE.runKernelFunc((backend, save) => {
          const res = backend.conv2d(x4D, $filter, convInfo);
          save([$filter, x4D]);
          return res;
      }, { x: x4D, filter: $filter }, grad, 'Conv2D', convInfo, inputsToSave);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * Computes the derivative of the input of a 2D convolution.
   *
   * @param xShape The shape of the input: [batch, height, width, inDepth].
   * If length of 3, batch of 1 is assumed.
   * @param dy The derivative of the output, of rank 4 or rank 3 of shape
   *   `[batch, outHeight, outWidth, outDepth]`. If rank 3, batch of 1 is
   * assumed.
   * @param filter The filter, rank 4, of shape
   *     `[filterHeight, filterWidth, inDepth, outDepth]`.
   * @param strides The strides of the convolution: `[strideHeight,
   * strideWidth]`.
   * @param pad The type of padding algorithm used:
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
   *     "NHWC". Specify the data format of the input and output data. With the
   *     default format "NHWC", the data is stored in the order of: [batch,
   *     height, width, channels].
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   */
  function conv2dDerInput_(xShape, dy, filter, strides, pad, dataFormat = 'NHWC', dimRoundingMode) {
      assert(xShape.length === dy.rank, () => `Length of inShape ` +
          `(${xShape.length}) and rank of dy (${dy.rank}) must match`);
      let xShape4D = xShape;
      let dy4D = dy;
      let reshapedTo4D = false;
      if (dy.rank === 3) {
          reshapedTo4D = true;
          dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
          xShape4D = [1, xShape[0], xShape[1], xShape[2]];
      }
      assert(xShape4D.length === 4, () => `Error in conv2dDerInput: inShape must be length 4, but got length ` +
          `${xShape4D.length}.`);
      assert(dy4D.rank === 4, () => `Error in conv2dDerInput: dy must be rank 4, but got ` +
          `rank ${dy4D.rank}`);
      assert(filter.rank === 4, () => `Error in conv2dDerInput: filter must be rank 4, but got ` +
          `rank ${filter.rank}`);
      const inDepth = dataFormat === 'NHWC' ? xShape4D[3] : xShape4D[1];
      const outDepth = dataFormat === 'NHWC' ? dy4D.shape[3] : dy4D.shape[1];
      assert(inDepth === filter.shape[2], () => `Error in conv2dDerInput: depth of input (${inDepth}) must ` +
          `match input depth for filter ${filter.shape[2]}.`);
      assert(outDepth === filter.shape[3], () => `Error in conv2dDerInput: depth of output (${outDepth}) must ` +
          `match output depth for filter ${filter.shape[3]}.`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in conv2dDerInput: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const dilations = 1;
      const grad = (ddx, saved) => {
          const [filter, dy4D] = saved;
          return {
              dy4D: () => conv2d(ddx, filter, strides, pad, dataFormat, dilations, dimRoundingMode),
              filter: () => conv2dDerFilter(ddx, dy4D, filter.shape, strides, pad, dataFormat, dimRoundingMode)
          };
      };
      const $dataFormat = convertConv2DDataFormat(dataFormat);
      const convInfo = computeConv2DInfo(xShape4D, filter.shape, strides, dilations, pad, dimRoundingMode, false, $dataFormat);
      const res = ENGINE.runKernelFunc((backend, save) => {
          const res = backend.conv2dDerInput(dy4D, filter, convInfo);
          save([filter, dy4D]);
          return res;
      }, { dy4D, filter }, grad);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * Computes the derivative of the filter of a 2D convolution.
   *
   * @param x The input tensor, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param dy The dy image, of rank 4 or rank 3, of shape
   *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
   * @param filterShape The shape of the filter, length 4,
   *     [filterHeight, filterWidth, inDepth, outDepth].
   * @param strides The strides of the convolution: [strideHeight,
   * strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
   *     "NHWC". Specify the data format of the input and output data. With the
   *     default format "NHWC", the data is stored in the order of: [batch,
   *     height, width, channels].
   * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
   *     rounding mode used when computing output dimensions if pad is a
   *     number. If none is provided, it will not round and error if the output
   *     is of fractional size.
   */
  function conv2dDerFilter_(x, dy, filterShape, strides, pad, dataFormat = 'NHWC', dimRoundingMode) {
      let x4D = x;
      if (x.rank === 3) {
          x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
      }
      let dy4D = dy;
      if (dy4D.rank === 3) {
          dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
      }
      assert(x4D.rank === 4, () => `Error in conv2dDerFilter: input must be rank 4, but got shape ` +
          `${x4D.shape}.`);
      assert(dy4D.rank === 4, () => `Error in conv2dDerFilter: dy must be rank 4, but got shape ` +
          `${dy4D.shape}.`);
      assert(filterShape.length === 4, () => `Error in conv2dDerFilter: filterShape must be length 4, but got ` +
          `${filterShape}.`);
      const inDepth = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
      const outDepth = dataFormat === 'NHWC' ? dy4D.shape[3] : dy4D.shape[1];
      assert(inDepth === filterShape[2], () => `Error in conv2dDerFilter: depth of input ${inDepth}) must ` +
          `match input depth in filter (${filterShape[2]}.`);
      assert(outDepth === filterShape[3], () => `Error in conv2dDerFilter: depth of dy (${outDepth}) must ` +
          `match output depth for filter (${filterShape[3]}).`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in conv2dDerFilter: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const dilations = 1;
      const $dataFormat = convertConv2DDataFormat(dataFormat);
      const convInfo = computeConv2DInfo(x4D.shape, filterShape, strides, dilations, pad, dimRoundingMode, false, $dataFormat);
      return ENGINE.runKernelFunc(backend => backend.conv2dDerFilter(x4D, dy4D, convInfo), { x4D, dy4D });
  }
  /**
   * Computes the transposed 2D convolution of an image, also known as a
   * deconvolution.
   *
   * @param x The input image, of rank 4 or rank 3, of shape
   *   `[batch, height, width, inDepth]`. If rank 3, batch of 1 is assumed.
   * @param filter The filter, rank 4, of shape
   *     `[filterHeight, filterWidth, outDepth, inDepth]`.
   *     `inDepth` must match `inDepth` in `x`.
   * @param outputShape Output shape, of rank 4 or rank 3:
   *     `[batch, height, width, outDepth]`. If rank 3, batch of 1 is assumed.
   * @param strides The strides of the original convolution:
   *     `[strideHeight, strideWidth]`.
   * @param pad  The type of padding algorithm used in the non-transpose version
   *    of the op.
   * @param dimRoundingMode The rounding mode used when computing output
   *    dimensions if pad is a number. If none is provided, it will not round
   *    and error if the output is of fractional size.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function conv2dTranspose_(x, filter, outputShape, strides, pad, dimRoundingMode) {
      const $x = convertToTensor(x, 'x', 'conv2dTranspose');
      const $filter = convertToTensor(filter, 'filter', 'conv2dTranspose');
      return conv2dDerInput_(outputShape, $x, $filter, strides, pad, 'NHWC', dimRoundingMode);
  }
  /**
   * Depthwise 2D convolution.
   *
   * Given a 4D `input` array and a `filter` array of shape
   * `[filterHeight, filterWidth, inChannels, channelMultiplier]` containing
   * `inChannels` convolutional filters of depth 1, this op applies a
   * different filter to each input channel (expanding from 1 channel to
   * `channelMultiplier` channels for each), then concatenates the results
   * together. The output has `inChannels * channelMultiplier` channels.
   *
   * See
   * [https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d](
   *     https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)
   * for more details.
   *
   * @param x The input tensor, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
   * assumed.
   * @param filter The filter tensor, rank 4, of shape
   *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`.
   * @param strides The strides of the convolution: `[strideHeight,
   * strideWidth]`. If strides is a single number, then `strideHeight ==
   * strideWidth`.
   * @param pad The type of padding algorithm.
   *   - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *   - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
   *     in which we sample input values across the height and width dimensions
   *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
   *     number, then `dilationHeight == dilationWidth`. If it is greater than
   *     1, then all values of `strides` must be 1.
   * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
   *     "NHWC". Specify the data format of the input and output data. With the
   *     default format "NHWC", the data is stored in the order of: [batch,
   *     height, width, channels]. Only "NHWC" is currently supported.
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function depthwiseConv2d_(x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode) {
      const $x = convertToTensor(x, 'x', 'depthwiseConv2d');
      const $filter = convertToTensor(filter, 'filter', 'depthwiseConv2d');
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      assert(x4D.rank === 4, () => `Error in depthwiseConv2d: input must be rank 4, but got ` +
          `rank ${x4D.rank}.`);
      assert($filter.rank === 4, () => `Error in depthwiseConv2d: filter must be rank 4, but got rank ` +
          `${$filter.rank}.`);
      assert(x4D.shape[3] === $filter.shape[2], () => `Error in depthwiseConv2d: number of input channels ` +
          `(${x4D.shape[3]}) must match the inChannels dimension in ` +
          `filter ${$filter.shape[2]}.`);
      if (dilations == null) {
          dilations = [1, 1];
      }
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in depthwiseConv2d: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in depthwiseConv2d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computeConv2DInfo(x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode, true /* depthwise */);
      const grad = (dy, saved) => {
          assert(tupleValuesAreOne(dilations), () => 'Error in gradient of depthwiseConv2d: dilation rates ' +
              `greater than 1 are not yet supported. Got dilations ` +
              `'${dilations}'`);
          const [x4D, $filter] = saved;
          return {
              x: () => depthwiseConv2dDerInput(x4D.shape, dy, $filter, convInfo),
              filter: () => depthwiseConv2dDerFilter(x4D, dy, $filter.shape, convInfo),
          };
      };
      const inputsToSave = [x4D, $filter];
      const res = ENGINE.runKernelFunc((backend, save) => {
          const res = backend.depthwiseConv2D(x4D, $filter, convInfo);
          save([x4D, $filter]);
          return res;
      }, { x: x4D, filter: $filter }, grad, 'DepthwiseConv2dNative', convInfo, inputsToSave);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * 2-D convolution with separable filters.
   *
   * Performs a depthwise convolution that acts separately on channels followed
   * by a pointwise convolution that mixes channels. Note that this is
   * separability between dimensions [1, 2] and 3, not spatial separability
   * between dimensions 1 and 2.
   *
   * See
   * [https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d](
   *     https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)
   * for more details.
   *
   * @param x The input tensor, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
   * assumed.
   * @param depthwiseFilter The depthwise filter tensor, rank 4, of shape
   *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`. This is
   *     the filter used in the first step.
   * @param pointwiseFilter The pointwise filter tensor, rank 4, of shape
   *     `[1, 1, inChannels * channelMultiplier, outChannels]`. This is
   *     the filter used in the second step.
   * @param strides The strides of the convolution: `[strideHeight,
   * strideWidth]`. If strides is a single number, then `strideHeight ==
   * strideWidth`.
   * @param pad The type of padding algorithm.
   *   - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *   - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
   *     in which we sample input values across the height and width dimensions
   *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
   *     number, then `dilationHeight == dilationWidth`. If it is greater than
   *     1, then all values of `strides` must be 1.
   * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
   *     "NHWC". Specify the data format of the input and output data. With the
   *     default format "NHWC", the data is stored in the order of: [batch,
   *     height, width, channels]. Only "NHWC" is currently supported.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function separableConv2d_(x, depthwiseFilter, pointwiseFilter, strides, pad, dilation = [1, 1], dataFormat = 'NHWC') {
      const $x = convertToTensor(x, 'x', 'separableConv2d');
      const $depthwiseFilter = convertToTensor(depthwiseFilter, 'depthwiseFilter', 'separableConv2d');
      const $pointwiseFilter = convertToTensor(pointwiseFilter, 'pointwiseFilter', 'separableConv2d');
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      if (dataFormat === 'NCHW') {
          throw new Error('separableConv2d currently does not support dataFormat NCHW; only ' +
              'NHWC is supported');
      }
      assert(x4D.rank === 4, () => `Error in separableConv2d: input must be rank 4, but got ` +
          `rank ${x4D.rank}.`);
      assert($depthwiseFilter.rank === 4, () => `Error in separableConv2d: depthwise filter must be rank 4, but ` +
          `got rank ${$depthwiseFilter.rank}.`);
      assert($pointwiseFilter.rank === 4, () => `Error in separableConv2d: pointwise filter must be rank 4, but ` +
          `got rank ${$depthwiseFilter.rank}.`);
      assert($pointwiseFilter.shape[0] === 1, () => `Error in separableConv2d: the first dimension of pointwise filter ` +
          ` must be 1, but got ${$pointwiseFilter.shape[0]}.`);
      assert($pointwiseFilter.shape[1] === 1, () => `Error in separableConv2d: the second dimension of pointwise ` +
          `filter must be 1, but got ${$pointwiseFilter.shape[1]}.`);
      const inChannels = $depthwiseFilter.shape[2];
      const channelMultiplier = $depthwiseFilter.shape[3];
      assert($pointwiseFilter.shape[2] === inChannels * channelMultiplier, () => `Error in separableConv2d: the third dimension of pointwise filter ` +
          `must be ${inChannels * channelMultiplier}, ` +
          `but got ${$pointwiseFilter.shape[2]}.`);
      const depthwise = depthwiseConv2d(x4D, $depthwiseFilter, strides, pad, dataFormat, dilation);
      const pointwiseStride = 1;
      const res = conv2d(depthwise, $pointwiseFilter, pointwiseStride, 'valid', dataFormat);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  function parseTupleParam$1(param) {
      if (typeof param === 'number') {
          return [param, param, param];
      }
      if (param.length === 2) {
          return [param[0], param[1], 1];
      }
      return param;
  }
  function tupleValuesAreOne$1(param) {
      const [dimA, dimB, dimC] = parseTupleParam$1(param);
      return dimA === 1 && dimB === 1 && dimC === 1;
  }
  function eitherStridesOrDilationsAreOne$1(strides, dilations) {
      return tupleValuesAreOne$1(strides) || tupleValuesAreOne$1(dilations);
  }
  function depthwiseConv2dDerInput_(xShape, dy, filter, convInfo) {
      let dy4D = dy;
      let reshapedTo4D = false;
      if (dy.rank === 3) {
          reshapedTo4D = true;
          dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
      }
      const res = ENGINE.runKernelFunc(backend => backend.depthwiseConv2DDerInput(dy4D, filter, convInfo), { dy4D });
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  function depthwiseConv2dDerFilter_(x, dy, filterShape, convInfo) {
      let x4D = x;
      if (x.rank === 3) {
          x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
      }
      let dy4D = dy;
      if (dy4D.rank === 3) {
          dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
      }
      return ENGINE.runKernelFunc(backend => backend.depthwiseConv2DDerFilter(x4D, dy4D, convInfo), { x4D, dy4D });
  }
  /**
   * Computes a 3D convolution over the input x.
   *
   * @param x The input tensor, of rank 5 or rank 4, of shape
   *     `[batch, depth, height, width, channels]`. If rank 4,
   * batch of 1 is assumed.
   * @param filter The filter, rank 5, of shape
   *     `[filterDepth, filterHeight, filterWidth, inChannels, outChannels]`.
   *      inChannels must match between input and filter.
   * @param strides The strides of the convolution: `[strideDepth, strideHeight,
   * strideWidth]`.
   * @param pad The type of padding algorithm.
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dataFormat: An optional string from: "NDHWC", "NCDHW". Defaults to
   *     "NDHWC". Specify the data format of the input and output data. With the
   *     default format "NDHWC", the data is stored in the order of: [batch,
   *     depth, height, width, channels]. Only "NDHWC" is currently supported.
   * @param dilations The dilation rates: `[dilationDepth, dilationHeight,
   *     dilationWidth]` in which we sample input values across the height
   *     and width dimensions in atrous convolution. Defaults to `[1, 1, 1]`.
   *     If `dilations` is a single number, then
   *     `dilationDepth == dilationHeight == dilationWidth`. If it is greater
   *     than 1, then all values of `strides` must be 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function conv3d_(x, filter, strides, pad, dataFormat = 'NDHWC', dilations = [1, 1, 1]) {
      const $x = convertToTensor(x, 'x', 'conv3d');
      const $filter = convertToTensor(filter, 'filter', 'conv3d');
      let x5D = $x;
      let reshapedTo5D = false;
      if ($x.rank === 4) {
          reshapedTo5D = true;
          x5D = $x.as5D(1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]);
      }
      assert(x5D.rank === 5, () => `Error in conv3d: input must be rank 5, but got rank ${x5D.rank}.`);
      assert($filter.rank === 5, () => `Error in conv3d: filter must be rank 5, but got rank ` +
          `${$filter.rank}.`);
      assert(x5D.shape[4] === $filter.shape[3], () => `Error in conv3d: depth of input (${x5D.shape[4]}) must match ` +
          `input depth for filter ${$filter.shape[3]}.`);
      assert(eitherStridesOrDilationsAreOne$1(strides, dilations), () => 'Error in conv3D: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      assert(dataFormat === 'NDHWC', () => `Error in conv3d: got dataFormat of ${dataFormat} but only NDHWC is currently supported.`);
      const convInfo = computeConv3DInfo(x5D.shape, $filter.shape, strides, dilations, pad);
      const grad = (dy, saved) => {
          assert(tupleValuesAreOne$1(dilations), () => 'Error in gradient of conv3D: dilation rates greater than 1 are ' +
              `not yet supported in gradients. Got dilations '${dilations}'`);
          const [x5D, $filter] = saved;
          return {
              x: () => conv3dDerInput_(x5D.shape, dy, $filter, strides, pad),
              $filter: () => conv3dDerFilter_(x5D, dy, $filter.shape, strides, pad)
          };
      };
      const res = ENGINE.runKernelFunc((backend, save) => {
          const res = backend.conv3d(x5D, $filter, convInfo);
          save([x5D, $filter]);
          return res;
      }, { x: x5D, $filter }, grad);
      if (reshapedTo5D) {
          return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]);
      }
      return res;
  }
  /**
   * Computes the derivative of the input of a 3D convolution.
   *
   * @param xShape The shape of the input: [batch, depth, height, width,
   * in_channels]. If length of 4, batch of 1 is assumed.
   * @param dy The derivative of the output, of rank 5 or rank 4 of shape
   *   `[batch, outDepth, outHeight, outWidth, in_channels]`.
   * If rank 4, batch of 1 is assumed.
   * @param filter The filter, rank 5, of shape
   *     `[filterDepth, filterHeight, filterWidth, inDepth, outDepth]`.
   * @param strides The strides of the convolution: `[strideDepth, strideHeight,
   * strideWidth]`.
   * @param pad The type of padding algorithm used:
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   */
  function conv3dDerInput_(xShape, dy, filter, strides, pad) {
      assert(xShape.length === dy.rank, () => `Length of inShape ` +
          `(${xShape.length}) and rank of dy (${dy.rank}) must match`);
      let xShape5D = xShape;
      let dy5D = dy;
      let reshapedTo5D = false;
      if (dy.rank === 4) {
          reshapedTo5D = true;
          dy5D = dy.as5D(1, dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]);
          xShape5D = [1, xShape[0], xShape[1], xShape[2], xShape[3]];
      }
      const inDepth = xShape5D[4];
      const outDepth = dy5D.shape[4];
      assert(xShape5D.length === 5, () => `Error in conv3dDerInput: inShape must be length 5, but got length ` +
          `${xShape5D.length}.`);
      assert(dy5D.rank === 5, () => `Error in conv3dDerInput: dy must be rank 5, but got ` +
          `rank ${dy5D.rank}`);
      assert(filter.rank === 5, () => `Error in conv3dDerInput: filter must be rank 5, but got ` +
          `rank ${filter.rank}`);
      assert(inDepth === filter.shape[3], () => `Error in conv3dDerInput: depth of input (${inDepth}) must ` +
          `match input depth for filter ${filter.shape[3]}.`);
      assert(outDepth === filter.shape[4], () => `Error in conv3dDerInput: depth of output (${outDepth}) must ` +
          `match output depth for filter ${filter.shape[4]}.`);
      const dilations = 1;
      const convInfo = computeConv3DInfo(xShape5D, filter.shape, strides, dilations, pad);
      const res = ENGINE.runKernelFunc(backend => backend.conv3dDerInput(dy5D, filter, convInfo), { dy5D });
      if (reshapedTo5D) {
          return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]);
      }
      return res;
  }
  /**
   * Computes the derivative of the filter of a 3D convolution.
   *
   * @param x The input tensor, of rank 5 or rank 4 of shape
   *     [batch, depth, height, width, inChannels]. If rank 4, batch of 1 is
   *     assumed.
   * @param dy The dy image, of rank 5 or rank 4, of shape
   *     [batch, depth, height, width, outDepth]. If rank 4, batch of 1 is
   *     assumed.
   * @param filterShape The shape of the filter, length 5,
   *     [filterDepth, filterHeight, filterWidth, inDepth, outDepth].
   * @param strides The strides of the convolution: [strideDepth, strideHeight,
   * strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  function conv3dDerFilter_(x, dy, filterShape, strides, pad) {
      let x5D = x;
      if (x.rank === 4) {
          x5D = x.as5D(1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
      }
      let dy5D = dy;
      if (dy5D.rank === 4) {
          dy5D = dy.as5D(1, dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]);
      }
      assert(x5D.rank === 5, () => `Error in conv3dDerFilter: input must be rank 5, but got shape ` +
          `${x5D.shape}.`);
      assert(dy5D.rank === 5, () => `Error in conv3dDerFilter: dy must be rank 5, but got shape ` +
          `${dy5D.shape}.`);
      assert(filterShape.length === 5, () => `Error in conv3dDerFilter: filterShape must be length 5, but got ` +
          `${filterShape}.`);
      assert(x5D.shape[4] === filterShape[3], () => `Error in conv3dDerFilter: depth of input ${x5D.shape[4]}) must ` +
          `match input depth in filter (${filterShape[3]}.`);
      assert(dy5D.shape[4] === filterShape[4], () => `Error in conv3dDerFilter: depth of dy (${dy5D.shape[4]}) must ` +
          `match output depth for filter (${filterShape[4]}).`);
      const dilations = 1;
      const convInfo = computeConv3DInfo(x5D.shape, filterShape, strides, dilations, pad);
      return ENGINE.runKernelFunc(backend => backend.conv3dDerFilter(x5D, dy5D, convInfo), { x5D, dy5D });
  }
  /**
   * Computes the transposed 3D convolution of a volume, also known as a
   * deconvolution.
   *
   * @param x The input image, of rank 5 or rank 4, of shape
   *   `[batch, depth, height, width, inDepth]`. If rank 4, batch of 1 is assumed.
   * @param filter The filter, rank 4, of shape
   *     `[depth, filterHeight, filterWidth, outDepth, inDepth]`.
   *     `inDepth` must match `inDepth` in `x`.
   * @param outputShape Output shape, of rank 5 or rank 4:
   *     `[batch, depth, height, width, outDepth]`. If rank 3, batch of 1 is
   *    assumed.
   * @param strides The strides of the original convolution:
   *     `[strideDepth, strideHeight, strideWidth]`.
   * @param pad  The type of padding algorithm used in the non-transpose version
   *    of the op.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function conv3dTranspose_(x, filter, outputShape, strides, pad) {
      const $x = convertToTensor(x, 'x', 'conv3dTranspose');
      const $filter = convertToTensor(filter, 'filter', 'conv3dTranspose');
      return conv3dDerInput_(outputShape, $x, $filter, strides, pad);
  }
  const conv1d = op({ conv1d_ });
  const conv2d = op({ conv2d_ });
  const conv3d = op({ conv3d_ });
  const conv2dDerFilter = op({ conv2dDerFilter_ });
  const conv2dDerInput = op({ conv2dDerInput_ });
  const depthwiseConv2d = op({ depthwiseConv2d_ });
  const depthwiseConv2dDerInput = op({ depthwiseConv2dDerInput_ });
  const depthwiseConv2dDerFilter = op({ depthwiseConv2dDerFilter_ });
  const separableConv2d = op({ separableConv2d_ });
  const conv2dTranspose = op({ conv2dTranspose_ });
  const conv3dTranspose = op({ conv3dTranspose_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes the dot product of two matrices, A * B. These must be matrices.
   *
   * ```js
   * const a = tf.tensor2d([1, 2], [1, 2]);
   * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * a.matMul(b).print();  // or tf.matMul(a, b)
   * ```
   * @param a First matrix in dot product operation.
   * @param b Second matrix in dot product operation.
   * @param transposeA If true, `a` is transposed before multiplication.
   * @param transposeB If true, `b` is transposed before multiplication.
   */
  /** @doc {heading: 'Operations', subheading: 'Matrices'} */
  function matMul_(a, b, transposeA = false, transposeB = false) {
      let $a = convertToTensor(a, 'a', 'matMul');
      let $b = convertToTensor(b, 'b', 'matMul');
      [$a, $b] = makeTypesMatch($a, $b);
      const innerShapeA = transposeA ? $a.shape[$a.rank - 2] : $a.shape[$a.rank - 1];
      const innerShapeB = transposeB ? $b.shape[$b.rank - 1] : $b.shape[$b.rank - 2];
      const outerShapeA = transposeA ? $a.shape[$a.rank - 1] : $a.shape[$a.rank - 2];
      const outerShapeB = transposeB ? $b.shape[$b.rank - 2] : $b.shape[$b.rank - 1];
      const outerDimsA = $a.shape.slice(0, -2);
      const outerDimsB = $b.shape.slice(0, -2);
      const batchDimA = sizeFromShape(outerDimsA);
      const batchDimB = sizeFromShape(outerDimsB);
      assert($a.rank >= 2 && $b.rank >= 2 && $a.rank === $b.rank, () => `Error in matMul: inputs must have the same rank of at least 2, ` +
          `got ranks ${$a.rank} and ${$b.rank}.`);
      assert(arraysEqual(outerDimsA, outerDimsB), () => `Error in matMul: outer dimensions (${outerDimsA}) and (` +
          `${outerDimsB}) of Tensors with shapes ${$a.shape} and ` +
          `${$b.shape} must match.`);
      assert(innerShapeA === innerShapeB, () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${$a.shape} and ` +
          `${$b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);
      const outShape = $a.shape.slice(0, -2).concat([outerShapeA, outerShapeB]);
      const a3D = transposeA ? $a.as3D(batchDimA, innerShapeA, outerShapeA) :
          $a.as3D(batchDimA, outerShapeA, innerShapeA);
      const b3D = transposeB ? $b.as3D(batchDimB, outerShapeB, innerShapeB) :
          $b.as3D(batchDimB, innerShapeB, outerShapeB);
      const grad = (dy, saved) => {
          const [a3D, b3D] = saved;
          if (!transposeA && !transposeB) {
              return {
                  a: () => dy.matMul(b3D, false, true),
                  b: () => a3D.matMul(dy, true, false)
              };
          }
          else if (!transposeA && transposeB) {
              return {
                  a: () => dy.matMul(b3D, false, false),
                  b: () => dy.matMul(a3D, true, false)
              };
          }
          else if (transposeA && !transposeB) {
              return {
                  a: () => b3D.matMul(dy, false, true),
                  b: () => a3D.matMul(dy, false, false)
              };
          }
          else {
              return {
                  a: () => b3D.matMul(dy, true, true),
                  b: () => dy.matMul(a3D, true, true)
              };
          }
      };
      const attrs = { transposeA, transposeB };
      const res = ENGINE.runKernelFunc((backend, save) => {
          const res = backend.batchMatMul(a3D, b3D, transposeA, transposeB);
          save([a3D, b3D]);
          return res;
      }, { a: a3D, b: b3D }, grad, 'BatchMatMul', attrs);
      return res.reshape(outShape);
  }
  /**
   * Computes the outer product of two vectors, `v1` and `v2`.
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([3, 4, 5]);
   *
   * tf.outerProduct(a, b).print();
   * ```
   * @param v1 The first vector in the outer product operation.
   * @param v2 The second vector in the outer product operation.
   */
  /** @doc {heading: 'Operations', subheading: 'Matrices'} */
  function outerProduct_(v1, v2) {
      const $v1 = convertToTensor(v1, 'v1', 'outerProduct');
      const $v2 = convertToTensor(v2, 'v2', 'outerProduct');
      assert($v1.rank === 1 && $v2.rank === 1, () => `Error in outerProduct: inputs must be rank 1, but got ranks ` +
          `${$v1.rank} and ${$v2.rank}.`);
      return $v1.as2D(-1, 1).matMul($v2.as2D(1, -1));
  }
  /**
   * Computes the dot product of two matrices and/or vectors, `t1` and `t2`.
   *
   * ```js
   * const a = tf.tensor1d([1, 2]);
   * const b = tf.tensor2d([[1, 2], [3, 4]]);
   * const c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
   *
   * a.dot(b).print();  // or tf.dot(a, b)
   * b.dot(a).print();
   * b.dot(c).print();
   * ```
   * @param t1 The first tensor in the dot operation.
   * @param t2 The second tensor in the dot operation.
   */
  /** @doc {heading: 'Operations', subheading: 'Matrices'} */
  function dot_(t1, t2) {
      const $t1 = convertToTensor(t1, 't1', 'dot');
      const $t2 = convertToTensor(t2, 't2', 'dot');
      assert(($t1.rank === 1 || $t1.rank === 2) && ($t2.rank === 1 || $t2.rank === 2), () => `Error in dot: inputs must all be rank 1 or 2, but got ranks ` +
          `${$t1.rank} and ${$t2.rank}.`);
      const t1Inner = ($t1.rank === 1 ? $t1.size : $t1.shape[1]);
      const t2Inner = ($t2.rank === 1 ? $t2.size : $t2.shape[0]);
      assert(t1Inner === t2Inner, () => `Error in dot: inner dimensions of inputs must match, but got ` +
          `${t1Inner} and ${t2Inner}.`);
      if ($t1.rank === 1 && $t2.rank === 1) {
          return $t1.as2D(1, -1).matMul($t2.as2D(-1, 1)).asScalar();
      }
      else if ($t1.rank === 1 && $t2.rank === 2) {
          return $t1.as2D(1, -1).matMul($t2.as2D($t2.shape[0], $t2.shape[1])).as1D();
      }
      else if ($t1.rank === 2 && $t2.rank === 1) {
          return $t1.matMul($t2.as2D(-1, 1)).as1D();
      }
      else {
          return $t1.matMul($t2.as2D($t2.shape[0], $t2.shape[1]));
      }
  }
  const matMul = op({ matMul_ });
  const dot = op({ dot_ });
  const outerProduct = op({ outerProduct_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Reverses a `tf.Tensor1D`.
   *
   * @param x The input tensor.
   */
  function reverse1d_(x) {
      const $x = convertToTensor(x, 'x', 'reverse');
      assert($x.rank === 1, () => `Error in reverse1D: x must be rank 1 but got rank ${$x.rank}.`);
      return reverse($x, 0);
  }
  /**
   * Reverses a `tf.Tensor2D` along a specified axis.
   *
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)). Defaults to all axes.
   */
  function reverse2d_(x, axis) {
      const $x = convertToTensor(x, 'x', 'reverse');
      assert($x.rank === 2, () => `Error in reverse2D: x must be rank 2 but got rank ${$x.rank}.`);
      return reverse($x, axis);
  }
  /**
   * Reverses a `tf.Tensor3D` along a specified axis.
   *
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)). Defaults to all axes.
   */
  function reverse3d_(x, axis) {
      const $x = convertToTensor(x, 'x', 'reverse');
      assert($x.rank === 3, () => `Error in reverse3D: x must be rank 3 but got rank ${$x.rank}.`);
      return reverse($x, axis);
  }
  /**
   * Reverses a `tf.Tensor4D` along a specified axis.
   *
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)). Defaults to all axes.
   */
  function reverse4d_(x, axis) {
      const $x = convertToTensor(x, 'x', 'reverse');
      assert($x.rank === 4, () => `Error in reverse4D: x must be rank 4 but got rank ${$x.rank}.`);
      return reverse($x, axis);
  }
  /**
   * Reverses a `tf.Tensor` along a specified axis.
   *
   * Also available are stricter rank-specific methods that assert that `x` is
   * of the given rank:
   *   - `tf.reverse1d`
   *   - `tf.reverse2d`
   *   - `tf.reverse3d`
   *   - `tf.reverse4d`
   *
   * Except `tf.reverse1d` (which does not have axis param), all methods have
   * same signature as this method.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   *
   * x.reverse().print();
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * const axis = 1;
   * x.reverse(axis).print();
   * ```
   * @param x The input tensor to be reversed.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)). Defaults to all axes.
   */
  /** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
  function reverse_(x, axis) {
      const $x = convertToTensor(x, 'x', 'reverse');
      if ($x.rank === 0) {
          return $x.clone();
      }
      const axes = parseAxisParam(axis, $x.shape);
      const grad = (dy) => {
          return { $x: () => dy.reverse(axes) };
      };
      const res = ENGINE.runKernelFunc(backend => backend.reverse($x, axes), { $x }, grad);
      return res.reshapeAs($x);
  }
  const reverse = op({ reverse_ });
  const reverse1d = op({ reverse1d_ });
  const reverse2d = op({ reverse2d_ });
  const reverse3d = op({ reverse3d_ });
  const reverse4d = op({ reverse4d_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes the 2D max pooling of an image.
   *
   * @param x The input tensor, of rank 4 or rank 3 of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
   *     `filterSize` is a single number, then `filterHeight == filterWidth`.
   * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
   *     in which we sample input values across the height and width dimensions
   *     in dilated pooling. Defaults to `[1, 1]`. If `dilations` is a single
   *     number, then `dilationHeight == dilationWidth`. If it is greater than
   *     1, then all values of `strides` must be 1.
   * @param pad The type of padding algorithm.
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *    - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   */
  function maxPoolImpl_(x, filterSize, strides, dilations, pad, dimRoundingMode) {
      const $x = convertToTensor(x, 'x', 'maxPool');
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      if (dilations == null) {
          dilations = [1, 1];
      }
      assert(x4D.rank === 4, () => `Error in maxPool: input must be rank 4 but got rank ${x4D.rank}.`);
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in maxPool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in maxPool: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computePool2DInfo(x4D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
      if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
          arraysEqual(convInfo.inShape, convInfo.outShape)) {
          return $x.clone();
      }
      const grad = (dy, saved) => {
          const [x4D, y] = saved;
          return {
              x: () => maxPoolBackprop(dy, x4D, y, filterSize, strides, dilations, pad)
          };
      };
      const inputsToSave = [x4D];
      const res = ENGINE.runKernelFunc((backend, save) => {
          const y = backend.maxPool(x4D, convInfo);
          save([x4D, y]);
          return y;
      }, { x: x4D }, grad, 'MaxPool', convInfo, inputsToSave);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * Computes the 2D max pooling of an image.
   *
   * @param x The input tensor, of rank 4 or rank 3 of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
   *     `filterSize` is a single number, then `filterHeight == filterWidth`.
   * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param pad The type of padding algorithm.
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *    - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function maxPool_(x, filterSize, strides, pad, dimRoundingMode) {
      return maxPoolImpl_(x, filterSize, strides, 1, pad, dimRoundingMode);
  }
  /**
   * Computes the 2D average pooling of an image.
   *
   * @param x The input tensor, of rank 4 or rank 3 of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
   *     `filterSize` is a single number, then `filterHeight == filterWidth`.
   * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
   *     in which we sample input values across the height and width dimensions
   *     in dilated pooling. Defaults to `[1, 1]`. If `dilations` is a single
   *     number, then `dilationHeight == dilationWidth`. If it is greater than
   *     1, then all values of `strides` must be 1.
   * @param pad The type of padding algorithm:
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *    - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   */
  function avgPoolImpl_(x, filterSize, strides, dilations, pad, dimRoundingMode) {
      const $x = convertToTensor(x, 'x', 'avgPool', 'float32');
      if (dilations == null) {
          dilations = [1, 1];
      }
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in avgPool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      assert(x4D.rank === 4, () => `Error in avgPool: x must be rank 4 but got rank ${x4D.rank}.`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in avgPool: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computePool2DInfo(x4D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
      if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
          arraysEqual(convInfo.inShape, convInfo.outShape)) {
          return $x.clone();
      }
      const grad = (dy) => {
          return {
              x: () => avgPoolBackprop(dy, x4D, filterSize, strides, dilations, pad)
          };
      };
      let res = ENGINE.runKernelFunc(backend => backend.avgPool(x4D, convInfo), { x: x4D }, grad, 'AvgPool', convInfo);
      res = res.cast($x.dtype);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * Computes the 2D average pooling of an image.
   *
   * @param x The input tensor, of rank 4 or rank 3 of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
   *     `filterSize` is a single number, then `filterHeight == filterWidth`.
   * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param pad The type of padding algorithm:
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *    - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function avgPool_(x, filterSize, strides, pad, dimRoundingMode) {
      return avgPoolImpl_(x, filterSize, strides, 1, pad, dimRoundingMode);
  }
  /**
   * Performs an N-D pooling operation
   *
   * @param input The input tensor, of rank 4 or rank 3 of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param windowShape The filter size: `[filterHeight, filterWidth]`. If
   *     `filterSize` is a single number, then `filterHeight == filterWidth`.
   * @param poolingType The type of pooling, either 'max' or 'avg'.
   * @param pad The type of padding algorithm:
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *    - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
   *     in which we sample input values across the height and width dimensions
   *     in dilated pooling. Defaults to `[1, 1]`. If `dilationRate` is a single
   *     number, then `dilationHeight == dilationWidth`. If it is greater than
   *     1, then all values of `strides` must be 1.
   * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function pool_(input, windowShape, poolingType, pad, dilations, strides) {
      if (dilations == null) {
          dilations = [1, 1];
      }
      if (strides == null) {
          strides = 1;
      }
      if (pad === 0) {
          pad = 'valid';
      }
      const $x = convertToTensor(input, 'x', 'maxPool');
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in pool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      const convInfo = computePool2DInfo(x4D.shape, windowShape, strides, dilations, pad);
      const dilation = [convInfo.dilationHeight, convInfo.dilationWidth];
      // The following implementation does batchToSpace(pool(spaceToBatch(x)))
      // whenever dilation > 1 since the TF kernels do not support dilation > 1.
      // tslint:disable-next-line:max-line-length
      // https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L1037
      let basePadding;
      if (pad === 'same') {
          basePadding = withSpaceToBatchBasePaddings([convInfo.filterHeight, convInfo.filterWidth], dilation);
      }
      else {
          basePadding = [[0, 0], [0, 0]];
      }
      const isDilationOne = dilation[0] === 1 && dilation[1] === 1;
      const [adjustedPadding, adjustedCrops] = requiredSpaceToBatchPaddings([convInfo.inHeight, convInfo.inWidth], dilation, basePadding);
      const convertedPad = isDilationOne ? pad : 'valid';
      const convertedX = isDilationOne ? x4D : spaceToBatchND(x4D, dilation, adjustedPadding);
      const forwardOp = poolingType === 'avg' ?
          () => avgPoolImpl_(convertedX, windowShape, strides, 1 /* dilation */, convertedPad) :
          () => maxPoolImpl_(convertedX, windowShape, strides, 1 /* dilation */, convertedPad);
      const y = forwardOp();
      const res = isDilationOne ? y : batchToSpaceND(y, dilation, adjustedCrops);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * Computes the backprop of a 2D max pool.
   *
   * @param dy The dy error, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param input The original input image, of rank 4, of shape
   *     [batchSize, height, width, channels].
   * @param output The original output image, of rank 4, of shape
   *     [batchSize, outHeight, outWidth, channels].
   * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
   *     `filterSize` is a single number, then `filterHeight == filterWidth`.
   * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
   *     rounding mode used when computing output dimensions if pad is a
   *     number. If none is provided, it will not round and error if the output
   *     is of fractional size.
   */
  function maxPoolBackprop(dy, input, output, filterSize, strides, dilations, pad, dimRoundingMode) {
      const $dy = convertToTensor(dy, 'dy', 'maxPoolBackprop');
      const $input = convertToTensor(input, 'input', 'maxPoolBackprop');
      const $output = convertToTensor(output, 'output', 'maxPoolBackprop');
      assert($input.rank === $dy.rank, () => `Rank of input (${$input.rank}) does not match rank of dy ` +
          `(${$dy.rank})`);
      if (dilations == null) {
          dilations = [1, 1];
      }
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in maxPoolBackProp: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      assert($dy.rank === 4, () => `Error in maxPoolBackprop: dy must be rank 4 but got rank ` +
          `${$dy.rank}.`);
      assert($input.rank === 4, () => `Error in maxPoolBackprop: input must be rank 4 but got rank ` +
          `${$input.rank}.`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in maxPoolBackprop: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computePool2DInfo($input.shape, filterSize, strides, dilations, pad, dimRoundingMode);
      const res = ENGINE.runKernelFunc(backend => backend.maxPoolBackprop($dy, $input, $output, convInfo), { $dy, $input });
      return res;
  }
  /**
   * Computes the backprop of an 2D avg pool.
   *
   * @param dy The dy error, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param input The input image, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
   *     `filterSize` is a single number, then `filterHeight == filterWidth`.
   * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  function avgPoolBackprop(dy, input, filterSize, strides, dilations, pad) {
      const $dy = convertToTensor(dy, 'dy', 'avgPoolBackprop');
      const $input = convertToTensor(input, 'input', 'avgPoolBackprop');
      assert($input.rank === $dy.rank, () => `Rank of input (${$input.rank}) does not match rank of dy (${$dy.rank})`);
      if (dilations == null) {
          dilations = [1, 1];
      }
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in avgPoolBackprop: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      let input4D = $input;
      let dy4D = $dy;
      let reshapedTo4D = false;
      if ($input.rank === 3) {
          reshapedTo4D = true;
          input4D = $input.as4D(1, $input.shape[0], $input.shape[1], $input.shape[2]);
          dy4D = $dy.as4D(1, $dy.shape[0], $dy.shape[1], $dy.shape[2]);
      }
      assert(dy4D.rank === 4, () => `Error in avgPoolBackprop: dy must be rank 4 but got rank ` +
          `${dy4D.rank}.`);
      assert(input4D.rank === 4, () => `Error in avgPoolBackprop: input must be rank 4 but got rank ` +
          `${input4D.rank}.`);
      const convInfo = computePool2DInfo(input4D.shape, filterSize, strides, dilations, pad);
      const res = ENGINE.runKernelFunc(backend => backend.avgPoolBackprop(dy4D, input4D, convInfo), { dy4D, input4D });
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  // Helper function to compute crops and paddings for pool with dilation > 1.
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/array_ops.py#L2184
  function requiredSpaceToBatchPaddings(inputShape, blockShape, basePadding) {
      const padStart = basePadding.map(b => b[0]);
      const origPadEnd = basePadding.map(b => b[1]);
      const fullInputShape = inputShape.concat(padStart, origPadEnd);
      const padEndExtra = blockShape.map((b, i) => (b - fullInputShape[i] % b) % b);
      const padEnd = origPadEnd.map((s, i) => s + padEndExtra[i]);
      const paddings = blockShape.map((_, i) => [padStart[i], padEnd[i]]);
      const crops = blockShape.map((_, i) => [0, padEndExtra[i]]);
      return [paddings, crops];
  }
  // Helper function to compute base paddings for pool with dilation > 1.
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L524
  function withSpaceToBatchBasePaddings(filterShape, dilation) {
      // Spatial dimensions of the filters and the upsampled filters in which we
      // introduce (rate - 1) zeros between consecutive filter values.
      const dilatedFilterShape = filterShape.map((s, i) => {
          return s + (s - 1) * (dilation[i] - 1);
      });
      const padExtraShape = dilatedFilterShape.map(s => s - 1);
      // When padding is odd, we pad more at end, following the same
      // convention as conv2d.
      const padExtraStart = padExtraShape.map(s => Math.floor(s / 2));
      const padExtraEnd = padExtraShape.map((s, i) => s - padExtraStart[i]);
      return padExtraShape.map((_, i) => {
          return [padExtraStart[i], padExtraEnd[i]];
      });
  }
  /**
   * Computes the 3D average pooling.
   *
   * ```js
   * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
   * const result = tf.avgPool3d(x, 2, 1, 'valid');
   * result.print();
   * ```
   *
   * @param x The input tensor, of rank 5 or rank 4 of shape
   *     `[batch, depth, height, width, inChannels]`.
   * @param filterSize The filter size:
   *     `[filterDepth, filterHeight, filterWidth]`.
   *     If `filterSize` is a single number,
   *     then `filterDepth == filterHeight == filterWidth`.
   * @param strides The strides of the pooling:
   *     `[strideDepth, strideHeight, strideWidth]`.
   *     If `strides` is a single number,
   *     then `strideDepth == strideHeight == strideWidth`.
   * @param pad The type of padding algorithm.
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1*1x1.
   *    - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
   *     "NDHWC". Specify the data format of the input and output data. With the
   *     default format "NDHWC", the data is stored in the order of: [batch,
   *     depth, height, width, channels]. Only "NDHWC" is currently supported.
   * @param dilations The dilation rates:
   *     `[dilationDepth, dilationHeight, dilationWidth]`
   *     in which we sample input values across the depth, height and width
   *     dimensions in dilated pooling.
   *     Defaults to `[1, 1, 1]`. If `dilations` is a single number,
   *     then `dilationDepth == dilationHeight == dilationWidth`.
   *     If it is greater than 1, then all values of `strides` must be 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function avgPool3d_(x, filterSize, strides, pad, dimRoundingMode, dataFormat = 'NDHWC', dilations) {
      const $x = convertToTensor(x, 'x', 'avgPool3d', 'float32');
      let x5D = $x;
      let reshapedTo5D = false;
      if ($x.rank === 4) {
          reshapedTo5D = true;
          x5D = $x.as5D(1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]);
      }
      if (dilations == null) {
          dilations = [1, 1, 1];
      }
      assert(x5D.rank === 5, () => `Error in avgPool3d: x must be rank 5 but got rank ${x5D.rank}.`);
      assert(dataFormat === 'NDHWC', () => `Error in avgPool3d: Only NDHWC is currently supported, ` +
          `but got dataFormat of ${dataFormat}`);
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in avgPool3d: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in avgPool3d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computePool3DInfo(x5D.shape, filterSize, strides, dilations, pad, dimRoundingMode, dataFormat);
      const grad = (dy) => {
          return {
              x: () => avgPool3dBackprop(dy, x5D, filterSize, strides, dilations, pad, dimRoundingMode)
          };
      };
      let res = ENGINE.runKernelFunc(backend => backend.avgPool3d(x5D, convInfo), { x: x5D }, grad);
      res = res.cast(x5D.dtype);
      if (reshapedTo5D) {
          return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]);
      }
      return res;
  }
  /**
   * Computes the backprop of a 3d avg pool.
   *
   * @param dy The dy error, of rank 5 of shape
   *     [batchSize, depth, height, width, channels].
   * assumed.
   * @param input The original input image, of rank 5 or rank4 of shape
   *     [batchSize, depth, height, width, channels].
   * @param filterSize The filter size:
   *     `[filterDepth, filterHeight, filterWidth]`.
   *     `filterSize` is a single number,
   *     then `filterDepth == filterHeight == filterWidth`.
   * @param strides The strides of the pooling:
   *     `[strideDepth, strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param dilations The dilation rates:
   *     `[dilationDepth, dilationHeight, dilationWidth]`
   *     in which we sample input values across the depth, height and width
   *     dimensions in dilated pooling.
   *     Defaults to `[1, 1, 1]`. If `dilations` is a single number,
   *     then `dilationDepth == dilationHeight == dilationWidth`.
   *     If it is greater than 1, then all values of `strides` must be 1.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
   *     rounding mode used when computing output dimensions if pad is a
   *     number. If none is provided, it will not round and error if the output
   *     is of fractional size.
   */
  function avgPool3dBackprop(dy, input, filterSize, strides, dilations, pad, dimRoundingMode) {
      const $dy = convertToTensor(dy, 'dy', 'avgPool3dBackprop');
      const $input = convertToTensor(input, 'input', 'avgPool3dBackprop');
      let dy5D = $dy;
      let input5D = $input;
      let reshapedTo5D = false;
      if ($input.rank === 4) {
          reshapedTo5D = true;
          dy5D = $dy.as5D(1, $dy.shape[0], $dy.shape[1], $dy.shape[2], $dy.shape[3]);
          input5D = $input.as5D(1, $input.shape[0], $input.shape[1], $input.shape[2], $input.shape[3]);
      }
      assert(dy5D.rank === 5, () => `Error in avgPool3dBackprop: dy must be rank 5 but got rank ` +
          `${dy5D.rank}.`);
      assert(input5D.rank === 5, () => `Error in avgPool3dBackprop: input must be rank 5 but got rank ` +
          `${input5D.rank}.`);
      if (dilations == null) {
          dilations = [1, 1, 1];
      }
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in avgPool3dBackprop: Either strides or dilations ' +
          `must be 1. Got strides ${strides} and dilations '${dilations}'`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in maxPool3dBackprop: pad must be an integer when ` +
              `using, dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computePool3DInfo(input5D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
      const res = ENGINE.runKernelFunc(backend => backend.avgPool3dBackprop(dy5D, input5D, convInfo), { dy5D, input5D });
      if (reshapedTo5D) {
          return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]);
      }
      return res;
  }
  /**
   * Computes the 3D max pooling.
   *
   * ```js
   * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
   * const result = tf.maxPool3d(x, 2, 1, 'valid');
   * result.print();
   * ```
   *
   * @param x The input tensor, of rank 5 or rank 4 of shape
   *     `[batch, depth, height, width, inChannels]`.
   * @param filterSize The filter size:
   *     `[filterDepth, filterHeight, filterWidth]`.
   *     If `filterSize` is a single number,
   *     then `filterDepth == filterHeight == filterWidth`.
   * @param strides The strides of the pooling:
   *     `[strideDepth, strideHeight, strideWidth]`.
   *     If `strides` is a single number,
   *     then `strideDepth == strideHeight == strideWidth`.
   * @param pad The type of padding algorithm.
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1*1x1.
   *    - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
   *     "NDHWC". Specify the data format of the input and output data. With the
   *     default format "NDHWC", the data is stored in the order of: [batch,
   *     depth, height, width, channels]. Only "NDHWC" is currently supported.
   * @param dilations The dilation rates:
   *     `[dilationDepth, dilationHeight, dilationWidth]`
   *     in which we sample input values across the depth, height and width
   *     dimensions in dilated pooling.
   *     Defaults to `[1, 1, 1]`. If `dilations` is a single number,
   *     then `dilationDepth == dilationHeight == dilationWidth`.
   *     If it is greater than 1, then all values of `strides` must be 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function maxPool3d_(x, filterSize, strides, pad, dimRoundingMode, dataFormat = 'NDHWC', dilations) {
      const $x = convertToTensor(x, 'x', 'maxPool3d');
      let x5D = $x;
      let reshapedTo5D = false;
      if ($x.rank === 4) {
          reshapedTo5D = true;
          x5D = $x.as5D(1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]);
      }
      if (dilations == null) {
          dilations = [1, 1, 1];
      }
      assert(x5D.rank === 5, () => `Error in maxPool3d: x must be rank 5 but got rank ${x5D.rank}.`);
      assert(dataFormat === 'NDHWC', () => `Error in maxPool3d: Only NDHWC is currently supported, ` +
          `but got dataFormat of ${dataFormat}`);
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in maxPool3d: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in maxPool3d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computePool3DInfo(x5D.shape, filterSize, strides, dilations, pad, dimRoundingMode, dataFormat);
      const grad = (dy, saved) => {
          const [x5D, y] = saved;
          return {
              x: () => maxPool3dBackprop(dy, x5D, y, filterSize, strides, dilations, pad, dimRoundingMode)
          };
      };
      const res = ENGINE.runKernelFunc((backend, save) => {
          const y = backend.maxPool3d(x5D, convInfo);
          save([x5D, y]);
          return y;
      }, { x: x5D }, grad);
      if (reshapedTo5D) {
          return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]);
      }
      return res;
  }
  /**
   * Computes the backprop of a 3d max pool.
   *
   * @param dy The dy error, of rank 5 of shape
   *     [batchSize, depth, height, width, channels].
   * assumed.
   * @param input The original input image, of rank 5 or rank 4 of shape
   *     [batchSize, depth, height, width, channels].
   * @param output The original output image, of rank 5 of shape
   *     [batchSize, outDepth, outHeight, outWidth, channels].
   * @param filterSize The filter size:
   *     `[filterDepth, filterHeight, filterWidth]`.
   *     `filterSize` is a single number,
   *     then `filterDepth == filterHeight == filterWidth`.
   * @param strides The strides of the pooling:
   *     `[strideDepth, strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param dilations The dilation rates:
   *     `[dilationDepth, dilationHeight, dilationWidth]`
   *     in which we sample input values across the depth, height and width
   *     dimensions in dilated pooling.
   *     Defaults to `[1, 1, 1]`. If `dilations` is a single number,
   *     then `dilationDepth == dilationHeight == dilationWidth`.
   *     If it is greater than 1, then all values of `strides` must be 1.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
   *     rounding mode used when computing output dimensions if pad is a
   *     number. If none is provided, it will not round and error if the output
   *     is of fractional size.
   */
  function maxPool3dBackprop(dy, input, output, filterSize, strides, dilations, pad, dimRoundingMode) {
      const $dy = convertToTensor(dy, 'dy', 'maxPool3dBackprop');
      const $input = convertToTensor(input, 'input', 'maxPool3dBackprop');
      const $output = convertToTensor(output, 'output', 'maxPool3dBackprop');
      let dy5D = $dy;
      let input5D = $input;
      let output5D = $output;
      let reshapedTo5D = false;
      if ($input.rank === 4) {
          reshapedTo5D = true;
          dy5D = $dy.as5D(1, $dy.shape[0], $dy.shape[1], $dy.shape[2], $dy.shape[3]);
          input5D = $input.as5D(1, $input.shape[0], $input.shape[1], $input.shape[2], $input.shape[3]);
          output5D = $output.as5D(1, $output.shape[0], $output.shape[1], $output.shape[2], $output.shape[3]);
      }
      assert(dy5D.rank === 5, () => `Error in maxPool3dBackprop: dy must be rank 5 but got rank ` +
          `${dy5D.rank}.`);
      assert(input5D.rank === 5, () => `Error in maxPool3dBackprop: input must be rank 5 but got rank ` +
          `${input5D.rank}.`);
      assert(output5D.rank === 5, () => `Error in maxPool3dBackprop: output must be rank 5 but got rank ` +
          `${output5D.rank}.`);
      if (dilations == null) {
          dilations = [1, 1, 1];
      }
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in maxPool3dBackprop: Either strides or dilations ' +
          `must be 1. Got strides ${strides} and dilations '${dilations}'`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in maxPool3dBackprop: pad must be an integer when ` +
              `using, dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computePool3DInfo(input5D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
      const res = ENGINE.runKernelFunc(backend => backend.maxPool3dBackprop(dy5D, input5D, output5D, convInfo), { dy5D, input5D });
      if (reshapedTo5D) {
          return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]);
      }
      return res;
  }
  /**
   * Computes the 2D max pooling of an image with Argmax index.
   * The indices in argmax are flattened, so that a maximum value at position `[b,
   * y, x, c]` becomes flattened index: `(y * width + x) * channels + c` if
   * include_batch_in_index is False; `((b * height + y) * width + x) * channels
   * +c` if include_batch_in_index is True.
   *
   * The indices returned are always in `[0, height) x [0, width)` before
   * flattening.
   *
   * @param x The input tensor, of rank 4 or rank 3 of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
   *     `filterSize` is a single number, then `filterHeight == filterWidth`.
   * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
   *     `strides` is a single number, then `strideHeight == strideWidth`.
   * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
   *     "NDHWC". Specify the data format of the input and output data. With the
   *     default format "NDHWC", the data is stored in the order of: [batch,
   *     depth, height, width, channels]. Only "NDHWC" is currently supported.
   * @param pad The type of padding algorithm.
   *    - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *    - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param includeBatchIndex Defaults to False. Whether to include batch
   *    dimension in flattened index of argmax.
   */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  /** @doc {heading: 'Operations', subheading: 'Convolution'} */
  function maxPoolWithArgmax_(x, filterSize, strides, pad, includeBatchInIndex = false) {
      const $x = convertToTensor(x, 'x', 'maxPoolWithArgmax');
      const attrs = { filterSize, strides, pad, includeBatchInIndex };
      const result = ENGINE.runKernel('MaxPoolWithArgmax', { x: $x }, attrs);
      return { result: result[0], indexes: result[1] };
  }
  const maxPool = op({ maxPool_ });
  const avgPool = op({ avgPool_ });
  const pool = op({ pool_ });
  const maxPool3d = op({ maxPool3d_ });
  const avgPool3d = op({ avgPool3d_ });
  const maxPoolWithArgmax = op({ maxPoolWithArgmax_ });

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
  const SELU_SCALEALPHA = 1.7580993408473768599402175208123;
  const SELU_SCALE = 1.0507009873554804934193349852946;

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
  /**
   * Computes rectified linear element-wise: `max(x, 0)`.
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   *
   * x.relu().print();  // or tf.relu(x)
   * ```
   * @param x The input tensor. If the dtype is `bool`, the output dtype will be
   *     `int32'.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function relu_(x) {
      const $x = convertToTensor(x, 'x', 'relu');
      if ($x.dtype === 'bool') {
          return $x.toInt();
      }
      const grad = (dy, saved) => {
          const [$x] = saved;
          return { x: () => dy.mulStrict($x.step().toFloat()) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.relu($x);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'Relu');
  }
  /**
   * Computes rectified linear 6 element-wise: `min(max(x, 0), 6)`.
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 8]);
   *
   * x.relu6().print();  // or tf.relu6(x)
   * ```
   * @param x The input tensor. If the dtype is `bool`, the output dtype will be
   *     `int32'.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function relu6_(x) {
      const $x = convertToTensor(x, 'x', 'relu6');
      if ($x.dtype === 'bool') {
          return $x.toInt();
      }
      const grad = (dy, saved) => {
          const [$x] = saved;
          const mask = $x.lessEqual(6).mul($x.step());
          return { x: () => dy.mulStrict(mask.toFloat()) };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.relu6($x);
          save([$x]);
          return res;
      }, { x: $x }, grad, 'Relu6');
  }
  /**
   * Computes exponential linear element-wise: `x > 0 ? e ^ x - 1 : 0`.
   *
   * ```js
   * const x = tf.tensor1d([-1, 1, -3, 2]);
   *
   * x.elu().print();  // or tf.elu(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function elu_(x) {
      const $x = convertToTensor(x, 'x', 'elu');
      const grad = (dy, saved) => {
          const [y] = saved;
          return {
              $x: () => ENGINE.runKernelFunc(backend => backend.eluDer(dy, y), { dy, y })
          };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const y = backend.elu($x);
          save([y]);
          return y;
      }, { $x }, grad);
  }
  /**
   * Computes scaled exponential linear element-wise.
   *
   * `x < 0 ? scale * alpha * (exp(x) - 1) : x`
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   *
   * x.selu().print();  // or tf.selu(x)
   * ```
   * @param x The input tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function selu_(x) {
      const $x = convertToTensor(x, 'x', 'selu');
      const grad = (dy, saved) => {
          const [$x] = saved;
          return {
              $x: () => {
                  const mask = $x.greater(scalar(0));
                  const scaleAlpha = scalar(SELU_SCALEALPHA);
                  const scale = scalar(SELU_SCALE);
                  const greaterThanZeroDer = dy.mul(scale);
                  const lessEqualZeroDer = dy.mul(scaleAlpha).mul($x.toFloat().exp());
                  return where(mask, greaterThanZeroDer, lessEqualZeroDer);
              }
          };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.selu($x);
          save([$x]);
          return res;
      }, { $x }, grad);
  }
  /**
   * Computes leaky rectified linear element-wise.
   *
   * See
   * [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf](
   *     http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   *
   * x.leakyRelu(0.1).print();  // or tf.leakyRelu(x, 0.1)
   * ```
   * @param x The input tensor.
   * @param alpha The scaling factor for negative values, defaults to 0.2.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function leakyRelu_(x, alpha = 0.2) {
      const $x = convertToTensor(x, 'x', 'leakyRelu');
      return maximum(scalar(alpha).mul($x), $x);
  }
  /**
   * Computes leaky rectified linear element-wise with parametric alphas.
   *
   * `x < 0 ? alpha * x : f(x) = x`
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   * const alpha = tf.scalar(0.1);
   *
   * x.prelu(alpha).print();  // or tf.prelu(x, alpha)
   * ```
   * @param x The input tensor.
   * @param alpha Scaling factor for negative values.
   */
  /** @doc {heading: 'Operations', subheading: 'Basic math'} */
  function prelu_(x, alpha) {
      const $x = convertToTensor(x, 'x', 'prelu');
      const $alpha = convertToTensor(alpha, 'alpha', 'prelu');
      const grad = (dy, saved) => {
          const [$x, $alpha] = saved;
          const mask = $x.greater(0);
          return {
              x: () => where(mask, dy, dy.mul($alpha)),
              alpha: () => {
                  let res = where(mask, zerosLike(dy), dy.mul($x));
                  const reduceAxes = getReductionAxes($alpha.shape, dy.shape);
                  if (reduceAxes.length > 0) {
                      res = res.sum(reduceAxes);
                  }
                  return res.reshape($alpha.shape);
              }
          };
      };
      return ENGINE.runKernelFunc((backend, save) => {
          const res = backend.prelu($x, $alpha);
          save([$x, $alpha]);
          return res;
      }, { x: $x, alpha: $alpha }, grad, 'Prelu');
  }
  const elu = op({ elu_ });
  const leakyRelu = op({ leakyRelu_ });
  const prelu = op({ prelu_ });
  const relu = op({ relu_ });
  const relu6 = op({ relu6_ });
  const selu = op({ selu_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes the softmax normalized vector given the logits.
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   *
   * a.softmax().print();  // or tf.softmax(a)
   * ```
   *
   * ```js
   * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
   *
   * a.softmax().print();  // or tf.softmax(a)
   * ```
   *
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to `-1`
   *     which indicates the last dimension.
   */
  /** @doc {heading: 'Operations', subheading: 'Normalization'} */
  function softmax_(logits, dim = -1) {
      const $logits = convertToTensor(logits, 'logits', 'softmax', 'float32');
      if (dim === -1) {
          dim = $logits.rank - 1;
      }
      if (dim !== $logits.rank - 1) {
          throw Error('Softmax along a non-last dimension is not yet supported. ' +
              `Logits was rank ${$logits.rank} and dim was ${dim}`);
      }
      const inputsToSave = [];
      const outputsToSave = [true];
      return ENGINE.runKernelFunc((backend, save) => {
          const y = backend.softmax($logits, dim);
          save([y]);
          return y;
      }, { logits: $logits }, (dy, saved) => {
          const [y] = saved;
          const dyTimesY = dy.mul(y);
          const keepDims = true;
          return {
              logits: () => dyTimesY.sub(dyTimesY.sum([dim], keepDims).mul(y))
          };
      }, 'Softmax', { dim }, inputsToSave, outputsToSave);
  }
  /**
   * Computes the log softmax.
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   *
   * a.logSoftmax().print();  // or tf.logSoftmax(a)
   * ```
   *
   * ```js
   * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
   *
   * a.logSoftmax().print();  // or tf.logSoftmax(a)
   * ```
   *
   * @param logits The logits array.
   * @param axis The dimension softmax would be performed on. Defaults to `-1`
   *     which indicates the last dimension.
   */
  /** @doc {heading: 'Operations', subheading: 'Normalization'} */
  function logSoftmax_(logits, axis = -1) {
      const $logits = convertToTensor(logits, 'logits', 'logSoftmax');
      if (axis === -1) {
          axis = $logits.rank - 1;
      }
      if (axis !== $logits.rank - 1) {
          throw Error('Log Softmax along a non-last dimension is not yet supported. ' +
              `Logits was rank ${$logits.rank} and axis was ${axis}`);
      }
      const customOp = customGrad((logits, save) => {
          const keepDims = true;
          const xMax = logits.max(axis, true);
          const shifted = logits.sub(xMax);
          const value = shifted.toFloat().sub(shifted.exp().sum(axis, keepDims).log());
          save([value]);
          const gradFunc = (dy, saved) => {
              const [value] = saved;
              const softmax = value.exp();
              return dy.sub(dy.sum(axis, keepDims).mul(softmax));
          };
          return { value, gradFunc };
      });
      return customOp($logits);
  }
  const softmax = op({ softmax_ });
  const logSoftmax = op({ logSoftmax_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Normalizes the activation of a local neighborhood across or within
   * channels.
   *
   * @param x The input tensor. The 4-D input tensor is treated as a 3-D array
   *     of 1D vectors (along the last dimension), and each vector is
   *     normalized independently.
   * @param depthRadius The number of adjacent channels in the 1D normalization
   *     window.
   * @param bias A constant bias term for the basis.
   * @param alpha A scale factor, usually positive.
   * @param beta An exponent.
   */
  /** @doc {heading: 'Operations', subheading: 'Normalization'} */
  function localResponseNormalization_(x, depthRadius = 5, bias = 1, alpha = 1, beta = 0.5) {
      const $x = convertToTensor(x, 'x', 'localResponseNormalization');
      assert($x.rank === 4 || $x.rank === 3, () => `Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${$x.rank}.`);
      assert(isInt(depthRadius), () => `Error in localResponseNormalization: depthRadius must be an ` +
          `integer but got depthRadius ${depthRadius}.`);
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      const backward = (dy, saved) => {
          const [x4D, y] = saved;
          return {
              x4D: () => ENGINE.runKernelFunc(backend => backend.LRNGrad(dy, x4D, y, depthRadius, bias, alpha, beta), {})
          };
      };
      const res = ENGINE.runKernelFunc((backend, save) => {
          const y = backend.localResponseNormalization4D(x4D, depthRadius, bias, alpha, beta);
          save([x4D, y]);
          return y;
      }, { x4D }, backward);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      else {
          return res;
      }
  }
  const localResponseNormalization = op({ localResponseNormalization_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes the norm of scalar, vectors, and matrices.
   * This function can compute several different vector norms (the 1-norm, the
   * Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0)
   * and matrix norms (Frobenius, 1-norm, and inf-norm).
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   *
   * x.norm().print();  // or tf.norm(x)
   * ```
   *
   * @param x The input array.
   * @param ord Optional. Order of the norm. Supported norm types are
   * following:
   *
   *  | ord        | norm for matrices         | norm for vectors
   *  |------------|---------------------------|---------------------
   *  |'euclidean' |Frobenius norm             |2-norm
   *  |'fro'       |Frobenius norm	           |
   *  |Infinity    |max(sum(abs(x), axis=1))   |max(abs(x))
   *  |-Infinity   |min(sum(abs(x), axis=1))   |min(abs(x))
   *  |1           |max(sum(abs(x), axis=0))   |sum(abs(x))
   *  |2           |                           |sum(abs(x)^2)^1/2*
   *
   * @param axis Optional. If axis is null (the default), the input is
   * considered a vector and a single vector norm is computed over the entire
   * set of values in the Tensor, i.e. norm(x, ord) is equivalent
   * to norm(x.reshape([-1]), ord). If axis is a integer, the input
   * is considered a batch of vectors, and axis determines the axis in x
   * over which to compute vector norms. If axis is a 2-tuple of integer it is
   * considered a batch of matrices and axis determines the axes in NDArray
   * over which to compute a matrix norm.
   * @param keepDims Optional. If true, the norm have the same dimensionality
   * as the input.
   */
  /** @doc {heading: 'Operations', subheading: 'Matrices'} */
  function norm_(x, ord = 'euclidean', axis = null, keepDims = false) {
      x = convertToTensor(x, 'x', 'norm');
      const norm = normImpl(x, ord, axis);
      let keepDimsShape = norm.shape;
      if (keepDims) {
          const axes = parseAxisParam(axis, x.shape);
          keepDimsShape = expandShapeToKeepDim(norm.shape, axes);
      }
      return norm.reshape(keepDimsShape);
  }
  function normImpl(x, p, axis = null) {
      if (x.rank === 0) {
          return x.abs();
      }
      // consider vector when no axis is specified
      if (x.rank !== 1 && axis === null) {
          return normImpl(x.reshape([-1]), p, axis);
      }
      // vector
      if (x.rank === 1 || typeof axis === 'number' ||
          Array.isArray(axis) && axis.length === 1) {
          if (p === 1) {
              return x.abs().sum(axis);
          }
          if (p === Infinity) {
              return x.abs().max(axis);
          }
          if (p === -Infinity) {
              return x.abs().min(axis);
          }
          if (p === 'euclidean' || p === 2) {
              // norm(x, 2) = sum(abs(xi) ^ 2) ^ 1/2
              return x.abs().pow(scalar(2, 'int32')).sum(axis).sqrt();
          }
          throw new Error(`Error in norm: invalid ord value: ${p}`);
      }
      // matrix (assumption axis[0] < axis[1])
      if (Array.isArray(axis) && axis.length === 2) {
          if (p === 1) {
              return x.abs().sum(axis[0]).max(axis[1] - 1);
          }
          if (p === Infinity) {
              return x.abs().sum(axis[1]).max(axis[0]);
          }
          if (p === -Infinity) {
              return x.abs().sum(axis[1]).min(axis[0]);
          }
          if (p === 'fro' || p === 'euclidean') {
              // norm(x) = sqrt(sum(pow(x, 2)))
              return x.square().sum(axis).sqrt();
          }
          throw new Error(`Error in norm: invalid ord value: ${p}`);
      }
      throw new Error(`Error in norm: invalid axis: ${axis}`);
  }
  const norm = op({ norm_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes the next states and outputs of a stack of LSTMCells.
   *
   * Each cell output is used as input to the next cell.
   *
   * Returns `[cellState, cellOutput]`.
   *
   * Derived from tf.contrib.rn.MultiRNNCell.
   *
   * @param lstmCells Array of LSTMCell functions.
   * @param data The input to the cell.
   * @param c Array of previous cell states.
   * @param h Array of previous cell outputs.
   */
  /** @doc {heading: 'Operations', subheading: 'RNN'} */
  function multiRNNCell_(lstmCells, data, c, h) {
      const $data = convertToTensor(data, 'data', 'multiRNNCell');
      const $c = convertToTensorArray(c, 'c', 'multiRNNCell');
      const $h = convertToTensorArray(h, 'h', 'multiRNNCell');
      let input = $data;
      const newStates = [];
      for (let i = 0; i < lstmCells.length; i++) {
          const output = lstmCells[i](input, $c[i], $h[i]);
          newStates.push(output[0]);
          newStates.push(output[1]);
          input = output[1];
      }
      const newC = [];
      const newH = [];
      for (let i = 0; i < newStates.length; i += 2) {
          newC.push(newStates[i]);
          newH.push(newStates[i + 1]);
      }
      return [newC, newH];
  }
  /**
   * Computes the next state and output of a BasicLSTMCell.
   *
   * Returns `[newC, newH]`.
   *
   * Derived from tf.contrib.rnn.BasicLSTMCell.
   *
   * @param forgetBias Forget bias for the cell.
   * @param lstmKernel The weights for the cell.
   * @param lstmBias The bias for the cell.
   * @param data The input to the cell.
   * @param c Previous cell state.
   * @param h Previous cell output.
   */
  /** @doc {heading: 'Operations', subheading: 'RNN'} */
  function basicLSTMCell_(forgetBias, lstmKernel, lstmBias, data, c, h) {
      const $forgetBias = convertToTensor(forgetBias, 'forgetBias', 'basicLSTMCell');
      const $lstmKernel = convertToTensor(lstmKernel, 'lstmKernel', 'basicLSTMCell');
      const $lstmBias = convertToTensor(lstmBias, 'lstmBias', 'basicLSTMCell');
      const $data = convertToTensor(data, 'data', 'basicLSTMCell');
      const $c = convertToTensor(c, 'c', 'basicLSTMCell');
      const $h = convertToTensor(h, 'h', 'basicLSTMCell');
      const combined = $data.concat($h, 1);
      const weighted = combined.matMul($lstmKernel);
      const res = weighted.add($lstmBias);
      // i = input_gate, j = new_input, f = forget_gate, o = output_gate
      const batchSize = res.shape[0];
      const sliceCols = res.shape[1] / 4;
      const sliceSize = [batchSize, sliceCols];
      const i = res.slice([0, 0], sliceSize);
      const j = res.slice([0, sliceCols], sliceSize);
      const f = res.slice([0, sliceCols * 2], sliceSize);
      const o = res.slice([0, sliceCols * 3], sliceSize);
      const newC = i.sigmoid().mulStrict(j.tanh()).addStrict($c.mulStrict($forgetBias.add(f).sigmoid()));
      const newH = newC.tanh().mulStrict(o.sigmoid());
      return [newC, newH];
  }
  const basicLSTMCell = op({ basicLSTMCell_ });
  const multiRNNCell = op({ multiRNNCell_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Compute the moving average of a variable.
   *
   * Without zeroDebias, the moving average operation is defined by:
   *   `v += delta`
   * where
   *   `delta = (1 - decay) * (x - v)`
   *
   * With zeroDebias (default), the `delta` term is scaled to debias the
   * effect of the (assumed) zero-initialization of `v`.
   *   `delta /= (1 - decay ^ step)`
   *
   * For more details on the zero-debiasing algorithm, see:
   *   https://arxiv.org/abs/1412.6980
   *
   * Note that this function is completely stateless and does not keep track of
   * step count. The step count needs to be maintained by the caller and passed
   * in as `step`.
   *
   * @param v The current moving average value.
   * @param x New input value, must have the same shape and dtype as `v`.
   * @param decay The decay factor. Typical values are 0.95 and 0.99.
   * @param step Step count.
   * @param zeroDebias: Whether zeroDebias is to be performed (default: `true`).
   * @returns The new moving average value.
   */
  /** @doc {heading: 'Operations', subheading: 'Moving Average'} */
  function movingAverage_(v, x, decay, step, zeroDebias = true) {
      const $v = convertToTensor(v, 'v', 'movingAverage');
      const $x = convertToTensor(x, 'x', 'movingAverage');
      const $decay = convertToTensor(decay, 'decay', 'movingAverage');
      assertTypesMatch($v, $x);
      assert(arraysEqual($v.shape, $x.shape), () => 'Shape mismatch in v and x');
      const one = scalar(1);
      const oneMinusDecay = one.sub($decay);
      let update = $x.sub($v).mul(oneMinusDecay);
      if (zeroDebias) {
          assert(step != null, () => 'When using zeroDebias: true, step is required.');
          const $step = convertToTensor(step, 'step', 'movingAverage');
          update = update.div(one.sub(pow($decay, $step)));
      }
      return $v.add(update);
  }
  const movingAverage = op({ movingAverage_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Extracts a strided slice of a tensor.
   *
   * Roughly speaking, this op extracts a slice of size (end-begin)/stride from
   * the given input tensor (x). Starting at the location specified by begin the
   * slice continues by adding stride to the index until all dimensions are not
   * less than end. Note that a stride can be negative, which causes a reverse
   * slice.
   *
   * ```js
   * const t = tf.tensor3d([1, 1, 1 ,2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
   *    [3, 2, 3]);
   * t.stridedSlice([1, 0, 0], [2, 1, 3], [1, 1, 1]).print()  // [[[3, 3, 3]]]
   * t.stridedSlice([1, 0, 0], [2, 2, 3], [1, 1, 1]).print()  // [[[3, 3, 3],
   *                                                     // [4, 4, 4]]]
   * t.stridedSlice([1, -1, 0], [2, -3, 3], [1, -1, 1]).print() // [[[4, 4, 4],
   *                                                     // [3, 3, 3]]]
   * ```
   *
   * @param x The tensor to stride slice.
   * @param begin The coordinates to start the slice from.
   * @param end: The coordinates to end the slice at.
   * @param strides: The size of the slice.
   * @param beginMask: If the ith bit of beginMask is set, begin[i] is ignored
   *      and the fullest possible range in that dimension is used instead.
   * @param endMask: If the ith bit of endMask is set, end[i] is ignored
   *      and the fullest possible range in that dimension is used instead.
   * @param shrinkAxisMask: a bitmask where bit i implies that
   * the ith specification should shrink the dimensionality. begin and end must
   * imply a slice of size 1 in the dimension.
   */
  /** @doc {heading: 'Operations', subheading: 'Slicing and Joining'} */
  function stridedSlice_(x, begin, end, strides, beginMask = 0, endMask = 0, ellipsisMask = 0, newAxisMask = 0, shrinkAxisMask = 0) {
      if (strides == null) {
          strides = new Array(begin.length);
      }
      if (ellipsisMask !== 0) {
          throw new Error('ellipsis mask is not yet supported');
      }
      let $x = convertToTensor(x, 'x', 'stridedSlice');
      // Expand the dims of x based on the newAxisMask.
      const expandAxes = maskToAxes(newAxisMask);
      const newShape = $x.shape.slice();
      expandAxes.forEach(axis => {
          begin[axis] = 0;
          end[axis] = 1;
          newShape.splice(axis, 0, 1);
      });
      $x = $x.reshape(newShape);
      // Normalize the start, end and strides.
      for (let axis = 0; axis < $x.rank; axis++) {
          begin[axis] = startForAxis(beginMask, begin, strides, $x.shape, axis);
          end[axis] = stopForAxis(endMask, end, strides, $x.shape, axis);
          strides[axis] = strides[axis] || 1;
      }
      const shrinkAxes = maskToAxes(shrinkAxisMask);
      // Adjust the ends based on the shrink mask.
      shrinkAxes.forEach(axis => {
          end[axis] = begin[axis] + 1;
          strides[axis] = 1;
      });
      // Figure out the output shape.
      const size = computeOutShape$1(begin, end, strides);
      // Remove the axes based on shrinkMask.
      const outShape = size.filter((_, axis) => shrinkAxes.indexOf(axis) === -1);
      const nonStrided = strides.every(v => v === 1);
      if (nonStrided) {
          return slice($x, begin, size).reshape(outShape);
      }
      const res = ENGINE.runKernelFunc(backend => backend.stridedSlice($x, begin, end, strides), { $x });
      return res.reshape(outShape);
  }
  const stridedSlice = op({ stridedSlice_ });

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
  /**
   * Finds the values and indices of the `k` largest entries along the last
   * dimension.
   *
   * If the input is a vector (rank=1), finds the k largest entries in the vector
   * and outputs their values and indices as vectors. Thus values[j] is the j-th
   * largest entry in input, and its index is indices[j].
   * For higher rank inputs, computes the top k entries along the last dimension.
   *
   * If two elements are equal, the lower-index element appears first.
   *
   * ```js
   * const a = tf.tensor2d([[1, 5], [4, 3]]);
   * const {values, indices} = tf.topk(a);
   * values.print();
   * indices.print();
   * ```
   * @param x 1-D or higher `tf.Tensor` with last dimension being at least `k`.
   * @param k Number of top elements to look for along the last dimension.
   * @param sorted If true, the resulting `k` elements will be sorted by the
   *     values in descending order.
   */
  /** @doc {heading: 'Operations', subheading: 'Evaluation'} */
  function topk_(x, k = 1, sorted = true) {
      const $x = convertToTensor(x, 'x', 'topk');
      if ($x.rank === 0) {
          throw new Error('topk() expects the input to be of rank 1 or higher');
      }
      const lastDim = $x.shape[$x.shape.length - 1];
      if (k > lastDim) {
          throw new Error(`'k' passed to topk() must be <= the last dimension (${lastDim}) ` +
              `but got ${k}`);
      }
      const [values, indices] = ENGINE.runKernelFunc(b => b.topk($x, k, sorted), { $x });
      return { values, indices };
  }
  const topk = op({ topk_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Creates a new tensor by applying sparse updates to individual
   * values or slices within a zero tensor of the given shape tensor according to
   * indices. This operator is the inverse of the `tf.gatherND` operator which
   * extracts values or slices from a given tensor.
   *
   * ```js
   * const indices = tf.tensor2d([4, 3, 1, 7], [4, 1], 'int32');
   * const updates = tf.tensor1d([9, 10, 11, 12]);
   * const shape = [8];
   * tf.scatterND(indices, updates, shape).print() //[0, 11, 0, 10, 9, 0, 0, 12]
   * ```
   *
   * @param indices The tensor contains the indices into the output tensor.
   * @param updates The tensor contains the value for the indices.
   * @param shape: The shape of the output tensor.
   */
  /** @doc {heading: 'Operations', subheading: 'Slicing and Joining'} */
  function scatterND_(indices, updates, shape) {
      const $indices = convertToTensor(indices, 'indices', 'scatterND', 'int32');
      const $updates = convertToTensor(updates, 'updates', 'scatterND');
      validateInput($updates, $indices, shape);
      return ENGINE.runKernelFunc(backend => backend.scatterND($indices, $updates, shape), { indices: $indices, updates: $updates }, null /* backward */, 'ScatterNd', { shape });
  }
  const scatterND = op({ scatterND_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Fast Fourier transform.
   *
   * Computes the 1-dimensional discrete Fourier transform over the inner-most
   * dimension of input.
   *
   * ```js
   * const real = tf.tensor1d([1, 2, 3]);
   * const imag = tf.tensor1d([1, 2, 3]);
   * const x = tf.complex(real, imag);
   *
   * x.fft().print();  // tf.spectral.fft(x).print();
   * ```
   * @param input The complex input to compute an fft over.
   */
  /**
   * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
   */
  function fft_(input) {
      assert(input.dtype === 'complex64', () => `The dtype for tf.spectral.fft() must be complex64 ` +
          `but got ${input.dtype}.`);
      // Collapse all outer dimensions to a single batch dimension.
      const innerDimensionSize = input.shape[input.shape.length - 1];
      const batch = input.size / innerDimensionSize;
      const input2D = input.as2D(batch, innerDimensionSize);
      const ret = ENGINE.runKernelFunc(backend => backend.fft(input2D), { input });
      return ret.reshape(input.shape);
  }
  /**
   * Inverse fast Fourier transform.
   *
   * Computes the inverse 1-dimensional discrete Fourier transform over the
   * inner-most dimension of input.
   *
   * ```js
   * const real = tf.tensor1d([1, 2, 3]);
   * const imag = tf.tensor1d([1, 2, 3]);
   * const x = tf.complex(real, imag);
   *
   * x.ifft().print();  // tf.spectral.ifft(x).print();
   * ```
   * @param input The complex input to compute an ifft over.
   */
  /**
   * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
   */
  function ifft_(input) {
      assert(input.dtype === 'complex64', () => `The dtype for tf.spectral.ifft() must be complex64 ` +
          `but got ${input.dtype}.`);
      // Collapse all outer dimensions to a single batch dimension.
      const innerDimensionSize = input.shape[input.shape.length - 1];
      const batch = input.size / innerDimensionSize;
      const input2D = input.as2D(batch, innerDimensionSize);
      const ret = ENGINE.runKernelFunc(backend => backend.ifft(input2D), { input });
      return ret.reshape(input.shape);
  }
  /**
   * Real value input fast Fourier transform.
   *
   * Computes the 1-dimensional discrete Fourier transform over the
   * inner-most dimension of the real input.
   *
   * ```js
   * const real = tf.tensor1d([1, 2, 3]);
   *
   * real.rfft().print();
   * ```
   * @param input The real value input to compute an rfft over.
   */
  /**
   * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
   */
  function rfft_(input, fftLength) {
      assert(input.dtype === 'float32', () => `The dtype for rfft() must be real value but got ${input.dtype}`);
      let innerDimensionSize = input.shape[input.shape.length - 1];
      const batch = input.size / innerDimensionSize;
      let adjustedInput;
      if (fftLength != null && fftLength < innerDimensionSize) {
          // Need to crop
          const begin = input.shape.map(v => 0);
          const size = input.shape.map(v => v);
          size[input.shape.length - 1] = fftLength;
          adjustedInput = input.slice(begin, size);
          innerDimensionSize = fftLength;
      }
      else if (fftLength != null && fftLength > innerDimensionSize) {
          // Need to pad with zeros
          const zerosShape = input.shape.map(v => v);
          zerosShape[input.shape.length - 1] = fftLength - innerDimensionSize;
          adjustedInput = input.concat(zeros(zerosShape), input.shape.length - 1);
          innerDimensionSize = fftLength;
      }
      else {
          adjustedInput = input;
      }
      // Complement the input with zero imaginary numbers.
      const zerosInput = adjustedInput.zerosLike();
      const complexInput = complex(adjustedInput, zerosInput).as2D(batch, innerDimensionSize);
      const ret = fft(complexInput);
      // Exclude complex conjugations. These conjugations are put symmetrically.
      const half = Math.floor(innerDimensionSize / 2) + 1;
      const realValues = real(ret);
      const imagValues = imag(ret);
      const realComplexConjugate = realValues.split([half, innerDimensionSize - half], realValues.shape.length - 1);
      const imagComplexConjugate = imagValues.split([half, innerDimensionSize - half], imagValues.shape.length - 1);
      const outputShape = adjustedInput.shape.slice();
      outputShape[adjustedInput.shape.length - 1] = half;
      return complex(realComplexConjugate[0], imagComplexConjugate[0])
          .reshape(outputShape);
  }
  /**
   * Inversed real value input fast Fourier transform.
   *
   * Computes the 1-dimensional inversed discrete Fourier transform over the
   * inner-most dimension of the real input.
   *
   * ```js
   * const real = tf.tensor1d([1, 2, 3]);
   * const imag = tf.tensor1d([0, 0, 0]);
   * const x = tf.complex(real, imag);
   *
   * x.irfft().print();
   * ```
   * @param input The real value input to compute an irfft over.
   */
  /**
   * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
   */
  function irfft_(input) {
      const innerDimensionSize = input.shape[input.shape.length - 1];
      const batch = input.size / innerDimensionSize;
      if (innerDimensionSize <= 2) {
          const complexInput = input.as2D(batch, innerDimensionSize);
          const ret = ifft(complexInput);
          return real(ret);
      }
      else {
          // The length of unique components of the DFT of a real-valued signal
          // is 2 * (input_len - 1)
          const outputShape = [batch, 2 * (innerDimensionSize - 1)];
          const realInput = real(input).as2D(batch, innerDimensionSize);
          const imagInput = imag(input).as2D(batch, innerDimensionSize);
          const realConjugate = realInput.slice([0, 1], [batch, innerDimensionSize - 2]).reverse(1);
          const imagConjugate = imagInput.slice([0, 1], [batch, innerDimensionSize - 2])
              .reverse(1)
              .mul(scalar(-1));
          const r = realInput.concat(realConjugate, 1);
          const i = imagInput.concat(imagConjugate, 1);
          const complexInput = complex(r, i).as2D(outputShape[0], outputShape[1]);
          const ret = ifft(complexInput);
          return real(ret);
      }
  }
  const fft = op({ fft_ });
  const ifft = op({ ifft_ });
  const rfft = op({ rfft_ });
  const irfft = op({ irfft_ });

  var spectral_ops = /*#__PURE__*/Object.freeze({
    __proto__: null,
    fft: fft,
    ifft: ifft,
    rfft: rfft,
    irfft: irfft
  });

  /**
   * Validate sparseToDense inputs.
   *
   * @param sparseIndices A 0-D, 1-D, or 2-D Tensor of type int32.
   * sparseIndices[i] contains the complete index where sparseValues[i] will be
   * placed.
   * @param sparseValues A 0-D or 1-D Tensor. Values
   * corresponding to each row of sparseIndices, or a scalar value to be used for
   * all sparse indices.
   * @param outputShape number[]. Shape of the dense output tensor.
   * @param validateIndices boolean. indice validation is not supported, error
   * will be thrown if it is set.
   */
  function validateInput$1(sparseIndices, sparseValues, outputShape, defaultValues) {
      if (sparseIndices.dtype !== 'int32') {
          throw new Error('tf.sparseToDense() expects the indices to be int32 type,' +
              ` but the dtype was ${sparseIndices.dtype}.`);
      }
      if (sparseIndices.rank > 2) {
          throw new Error('sparseIndices should be a scalar, vector, or matrix,' +
              ` but got shape ${sparseIndices.shape}.`);
      }
      const numElems = sparseIndices.rank > 0 ? sparseIndices.shape[0] : 1;
      const numDims = sparseIndices.rank > 1 ? sparseIndices.shape[1] : 1;
      if (outputShape.length !== numDims) {
          throw new Error('outputShape has incorrect number of elements:,' +
              ` ${outputShape.length}, should be: ${numDims}.`);
      }
      const numValues = sparseValues.size;
      if (!(sparseValues.rank === 0 ||
          sparseValues.rank === 1 && numValues === numElems)) {
          throw new Error('sparseValues has incorrect shape ' +
              `${sparseValues.shape}, should be [] or [${numElems}]`);
      }
      if (sparseValues.dtype !== defaultValues.dtype) {
          throw new Error('sparseValues.dtype must match defaultValues.dtype');
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Converts a sparse representation into a dense tensor.
   *
   * Builds an array dense with shape outputShape such that:
   *
   * // If sparseIndices is scalar
   * dense[i] = (i == sparseIndices ? sparseValues : defaultValue)
   *
   * // If sparseIndices is a vector, then for each i
   * dense[sparseIndices[i]] = sparseValues[i]
   *
   * // If sparseIndices is an n by d matrix, then for each i in [0, n)
   * dense[sparseIndices[i][0], ..., sparseIndices[i][d-1]] = sparseValues[i]
   * All other values in dense are set to defaultValue. If sparseValues is a
   * scalar, all sparse indices are set to this single value.
   *
   * If indices are repeated the final value is summed over all values for those
   * indices.
   *
   * ```js
   * const indices = tf.tensor1d([4, 5, 6, 1, 2, 3], 'int32');
   * const values = tf.tensor1d([10, 11, 12, 13, 14, 15], 'float32');
   * const shape = [8];
   * tf.sparseToDense(indices, values, shape).print();
   * ```
   *
   * @param sparseIndices A 0-D, 1-D, or 2-D Tensor of type int32.
   * sparseIndices[i] contains the complete index where sparseValues[i] will be
   * placed.
   * @param sparseValues A 0-D or 1-D Tensor. Values
   * corresponding to each row of sparseIndices, or a scalar value to be used for
   * all sparse indices.
   * @param outputShape Shape of the dense output tensor. the type is inferred.
   * @param defaultValue Scalar. Value to set for indices not specified in
   * sparseIndices. Defaults to zero.
   */
  /** @doc {heading: 'Operations', subheading: 'Normalization'} */
  function sparseToDense_(sparseIndices, sparseValues, outputShape, defaultValue = 0) {
      const $sparseIndices = convertToTensor(sparseIndices, 'sparseIndices', 'sparseToDense', 'int32');
      const $sparseValues = convertToTensor(sparseValues, 'sparseValues', 'sparseToDense');
      const $defaultValue = convertToTensor(defaultValue, 'defaultValue', 'sparseToDense', $sparseValues.dtype);
      validateInput$1($sparseIndices, $sparseValues, outputShape, $defaultValue);
      return ENGINE.runKernelFunc(backend => backend.sparseToDense($sparseIndices, $sparseValues, outputShape, $defaultValue), { $sparseIndices, $sparseValues, $defaultValue });
  }
  const sparseToDense = op({ sparseToDense_ });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Gather slices from input tensor into a Tensor with shape specified by
   * `indices`.
   *
   * `indices` is an K-dimensional integer tensor, best thought of as a
   * (K-1)-dimensional tensor of indices into input, where each element defines a
   * slice of input:
   * output[\\(i_0, ..., i_{K-2}\\)] = input[indices[\\(i_0, ..., i_{K-2}\\)]]
   *
   * Whereas in `tf.gather`, `indices` defines slices into the first dimension of
   * input, in `tf.gatherND`, `indices` defines slices into the first N dimensions
   * of input, where N = indices.shape[-1].
   *
   * The last dimension of indices can be at most the rank of input:
   * indices.shape[-1] <= input.rank
   *
   * The last dimension of `indices` corresponds to elements
   * (if indices.shape[-1] == input.rank) or slices
   * (if indices.shape[-1] < input.rank) along dimension indices.shape[-1] of
   * input.
   * The output tensor has shape
   * indices.shape[:-1] + input.shape[indices.shape[-1]:]
   *
   * Note that on CPU, if an out of bound index is found, an error is returned. On
   * GPU, if an out of bound index is found, a 0 is stored in the corresponding
   * output value.
   *
   * ```js
   * const indices = tf.tensor2d([0, 1, 1, 0], [2,2], 'int32');
   * const input = tf.tensor2d([9, 10, 11, 12], [2, 2]);
   * tf.gatherND(input, indices).print() // [10, 11]
   * ```
   *
   * @param x The tensor from which to gather values.
   * @param indices Index tensor, must be of type int32.
   */
  /** @doc {heading: 'Operations', subheading: 'Slicing and Joining'} */
  function gatherND_(x, indices) {
      const $indices = convertToTensor(indices, 'indices', 'gatherND', 'int32');
      const $x = convertToTensor(x, 'x', 'gatherND');
      return ENGINE.runKernelFunc(backend => backend.gatherND($x, $indices), { x: $x, indices: $indices }, null /* backward */, 'GatherNd');
  }
  const gatherND = op({ gatherND_ });

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
  /**
   * Returns a diagonal tensor with a given diagonal values.
   *
   * Given a diagonal, this operation returns a tensor with the diagonal and
   * everything else padded with zeros.
   *
   * Assume the input has dimensions `[D1,..., Dk]`, then the output is a tensor
   * of rank 2k with dimensions `[D1,..., Dk, D1,..., Dk]`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   *
   * tf.diag(x).print()
   * ```
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4, 5, 6, 6, 8], [4, 2])
   *
   * tf.diag(x).print()
   * ```
   * @param x The input tensor.
   */
  function diag_(x) {
      const $x = convertToTensor(x, 'x', 'diag').flatten();
      const outShape = [...x.shape, ...x.shape];
      return ENGINE.runKernelFunc(backend => backend.diag($x), { $x })
          .reshape(outShape);
  }
  const diag = op({ diag_ });

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
  /**
   * Normalize noise shape based on provided tensor and noise shape.
   *
   * @param x Tensor.
   * @param noiseShape The shape for the randomly generated keep/drop flags, as
   *   an array of numbers. Optional.
   * @returns Normalized noise shape.
   */
  function getNoiseShape(x, noiseShape) {
      if (noiseShape == null) {
          return x.shape.slice();
      }
      if (arraysEqual(x.shape, noiseShape)) {
          return noiseShape;
      }
      if (x.shape.length === noiseShape.length) {
          const newDimension = [];
          for (let i = 0; i < x.shape.length; i++) {
              if (noiseShape[i] == null && x.shape[i] != null) {
                  newDimension.push(x.shape[i]);
              }
              else {
                  newDimension.push(noiseShape[i]);
              }
          }
          return newDimension;
      }
      return noiseShape;
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Computes dropout.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 2, 1]);
   * const rate = 0.75;
   * const output = tf.dropout(x, rate);
   * output.print();
   * ```
   *
   * @param x A floating point Tensor or TensorLike.
   * @param rate A float in the range [0, 1). The probability that each element
   *   of x is discarded.
   * @param noiseShape An array of numbers of type int32, representing the
   * shape for randomly generated keep/drop flags. If the noiseShape has null
   * value, it will be automatically replaced with the x's relative dimension
   * size. Optional.
   * @param seed Used to create random seeds. Optional.
   * @returns A Tensor of the same shape of x.
   */
  /** @doc {heading: 'Operations', subheading: 'Dropout'} */
  function dropout_(x, rate, noiseShape, seed) {
      const $x = convertToTensor(x, 'x', 'dropout');
      assert($x.dtype === 'float32', () => `x has to be a floating point tensor since it's going to be ` +
          `scaled, but got a ${$x.dtype} tensor instead.`);
      assert(rate >= 0 && rate < 1, () => `rate must be a float in the range [0, 1), but got ${rate}.`);
      if (rate === 0) {
          return x instanceof Tensor ? $x.clone() : $x;
      }
      const $noiseShape = getNoiseShape($x, noiseShape);
      const keepProb = 1 - rate;
      const multiplier = randomUniform($noiseShape, 0, 1, 'float32', seed)
          .add(keepProb)
          .floor()
          .div(keepProb);
      return $x.mul(multiplier);
  }
  const dropout = op({ dropout_ });

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
  /**
   * Generate a Hann window.
   *
   * See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
   *
   * ```js
   * tf.signal.hannWindow(10).print();
   * ```
   * @param The length of window
   */
  /**
   * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
   */
  function hannWindow_(windowLength) {
      return cosineWindow(windowLength, 0.5, 0.5);
  }
  /**
   * Generate a hamming window.
   *
   * See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
   *
   * ```js
   * tf.signal.hammingWindow(10).print();
   * ```
   * @param The length of window
   */
  /**
   * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
   */
  function hammingWindow_(windowLength) {
      return cosineWindow(windowLength, 0.54, 0.46);
  }
  /**
   * Expands input into frames of frameLength.
   * Slides a window size with frameStep.
   *
   * ```js
   * tf.signal.frame([1, 2, 3], 2, 1).print();
   * ```
   * @param signal The input tensor to be expanded
   * @param frameLength Length of each frame
   * @param frameStep The frame hop size in samples.
   * @param padEnd Whether to pad the end of signal with padValue.
   * @param padValue An number to use where the input signal does
   *     not exist when padEnd is True.
   */
  /**
   * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
   */
  function frame_(signal, frameLength, frameStep, padEnd = false, padValue = 0) {
      let start = 0;
      const output = [];
      while (start + frameLength <= signal.size) {
          output.push(slice(signal, start, frameLength));
          start += frameStep;
      }
      if (padEnd) {
          while (start < signal.size) {
              const padLen = (start + frameLength) - signal.size;
              const pad = concat([slice(signal, start, frameLength - padLen),
                  fill([padLen], padValue)]);
              output.push(pad);
              start += frameStep;
          }
      }
      if (output.length === 0) {
          return tensor2d([], [0, frameLength]);
      }
      return concat(output).as2D(output.length, frameLength);
  }
  /**
   * Computes the Short-time Fourier Transform of signals
   * See: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
   *
   * ```js
   * const input = tf.tensor1d([1, 1, 1, 1, 1])
   * tf.signal.stft(input, 3, 1).print();
   * ```
   * @param signal 1-dimensional real value tensor.
   * @param frameLength The window length of samples.
   * @param frameStep The number of samples to step.
   * @param fftLength The size of the FFT to apply.
   * @param windowFn A callable that takes a window length and returns 1-d tensor.
   */
  /**
   * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
   */
  function stft_(signal, frameLength, frameStep, fftLength, windowFn = hannWindow) {
      if (fftLength == null) {
          fftLength = enclosingPowerOfTwo(frameLength);
      }
      const framedSignal = frame(signal, frameLength, frameStep);
      const windowedSignal = mul(framedSignal, windowFn(frameLength));
      const output = [];
      for (let i = 0; i < framedSignal.shape[0]; i++) {
          output.push(rfft(windowedSignal.slice([i, 0], [1, frameLength]), fftLength));
      }
      return concat(output);
  }
  function enclosingPowerOfTwo(value) {
      // Return 2**N for integer N such that 2**N >= value.
      return Math.floor(Math.pow(2, Math.ceil(Math.log(value) / Math.log(2.0))));
  }
  function cosineWindow(windowLength, a, b) {
      const even = 1 - windowLength % 2;
      const newValues = new Float32Array(windowLength);
      for (let i = 0; i < windowLength; ++i) {
          const cosArg = (2.0 * Math.PI * i) / (windowLength + even - 1);
          newValues[i] = a - b * Math.cos(cosArg);
      }
      return tensor1d(newValues, 'float32');
  }
  const hannWindow = op({ hannWindow_ });
  const hammingWindow = op({ hammingWindow_ });
  const frame = op({ frame_ });
  const stft = op({ stft_ });

  var signal_ops = /*#__PURE__*/Object.freeze({
    __proto__: null,
    hannWindow: hannWindow,
    hammingWindow: hammingWindow,
    frame: frame,
    stft: stft
  });

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
  /**
   * Returns whether the targets are in the top K predictions.
   *
   * ```js
   * const predictions = tf.tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
   * const targets = tf.tensor1d([2, 0]);
   * const precision = await tf.inTopKAsync(predictions, targets);
   * precision.print();
   * ```
   * @param predictions 2-D or higher `tf.Tensor` with last dimension being
   *     at least `k`.
   * @param targets 1-D or higher `tf.Tensor`.
   * @param k Optional Number of top elements to look at for computing precision,
   *     default to 1.
   */
  /** @doc {heading: 'Operations', subheading: 'Evaluation'} */
  async function inTopKAsync_(predictions, targets, k = 1) {
      const $predictions = convertToTensor(predictions, 'predictions', 'inTopK');
      const $targets = convertToTensor(targets, 'targets', 'inTopK');
      assert($predictions.rank > 1, () => 'inTopK() expects the predictions to be of rank 2 or higher, ' +
          `but got ${$predictions.rank}`);
      assert($predictions.rank - 1 === $targets.rank, () => `predictions rank should be 1 larger than ` +
          `targets rank, but got predictions rank ` +
          `${$predictions.rank} and targets rank ${$targets.rank}`);
      assertShapesMatch($predictions.shape.slice(0, $predictions.shape.length - 1), $targets.shape, `predictions's shape should be align with the targets' shape, ` +
          'except the last dimension.');
      const lastDim = $predictions.shape[$predictions.shape.length - 1];
      assert(k > 0 && k <= lastDim, () => `'k' passed to inTopK() must be > 0 && <= the predictions last ` +
          `dimension (${lastDim}), but got ${k}`);
      const predictionsVals = await $predictions.data();
      const targetsVals = await $targets.data();
      // Reshape predictionsVals into a 2d tensor [batch, lastDim]
      // and look up topK along lastDim.
      const [batch, size] = [predictionsVals.length / lastDim, lastDim];
      const precision = getTypedArrayFromDType('bool', batch);
      for (let b = 0; b < batch; b++) {
          const offset = b * size;
          const vals = predictionsVals.subarray(offset, offset + size);
          const valAndInd = [];
          for (let i = 0; i < vals.length; i++) {
              valAndInd.push({ value: vals[i], index: i });
          }
          valAndInd.sort((a, b) => b.value - a.value);
          precision[b] = 0;
          for (let i = 0; i < k; i++) {
              if (valAndInd[i].index === targetsVals[b]) {
                  precision[b] = 1;
                  break;
              }
          }
      }
      if (predictions !== $predictions) {
          $predictions.dispose();
      }
      if (targets !== $targets) {
          $targets.dispose();
      }
      // Output precision has the same shape as targets.
      return tensor(precision, $targets.shape, 'bool');
  }
  const inTopKAsync = inTopKAsync_;

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  (function (Reduction) {
      Reduction[Reduction["NONE"] = 0] = "NONE";
      Reduction[Reduction["MEAN"] = 1] = "MEAN";
      Reduction[Reduction["SUM"] = 2] = "SUM";
      Reduction[Reduction["SUM_BY_NONZERO_WEIGHTS"] = 3] = "SUM_BY_NONZERO_WEIGHTS";
  })(exports.Reduction || (exports.Reduction = {}));
  /**
   * Computes the weighted loss between two tensors.
   *
   * @param losses Tensor of shape `[batch_size, d1, ... dN]`.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `losses`, and must be broadcastable to `losses` (i.e., all
   *    dimensions must be either `1`, or the same as the corresponding
   *    `losses` dimension).
   */
  /** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
  function computeWeightedLoss_(losses, weights, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      const $losses = convertToTensor(losses, 'losses', 'computeWeightedLoss');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'computeWeightedLoss');
      }
      const weightedLoss = ($weights == null) ? $losses : $losses.mul($weights);
      if (reduction === exports.Reduction.NONE) {
          return weightedLoss;
      }
      if (reduction === exports.Reduction.SUM) {
          return weightedLoss.sum();
      }
      if (reduction === exports.Reduction.MEAN) {
          if ($weights == null) {
              return weightedLoss.mean();
          }
          else {
              const broadcastFactor = $losses.size / $weights.size;
              const result = weightedLoss.sum().div($weights.sum());
              return broadcastFactor > 1 ? result.div(scalar(broadcastFactor)) :
                  result;
          }
      }
      if (reduction === exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
          if ($weights == null) {
              return weightedLoss.sum().div(scalar($losses.size));
          }
          else {
              const broadcastedWeights = $weights.mul(ones$1($losses.shape));
              const numNonZeros = broadcastedWeights.notEqual(scalar(0)).sum().toFloat();
              return weightedLoss.sum().div(numNonZeros);
          }
      }
      throw Error(`Unknown reduction: ${reduction}`);
  }
  /**
   * Computes the absolute difference loss between two tensors.
   *
   * @param labels The ground truth output tensor, same dimensions as
   *    'predictions'.
   * @param predictions The predicted outputs.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
   *    must be either `1`, or the same as the corresponding `losses`
   *    dimension).
   * @param reduction Type of reduction to apply to loss. Should be of type
   *    `Reduction`
   */
  /** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
  function absoluteDifference_(labels, predictions, weights, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      const $labels = convertToTensor(labels, 'labels', 'absoluteDifference');
      const $predictions = convertToTensor(predictions, 'predictions', 'absoluteDifference');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'absoluteDifference');
      }
      assertShapesMatch($labels.shape, $predictions.shape, 'Error in absoluteDifference: ');
      const losses = $labels.sub($predictions).abs();
      return computeWeightedLoss(losses, $weights, reduction);
  }
  /**
   * Computes the mean squared error between two tensors.
   *
   * @param labels The ground truth output tensor, same dimensions as
   *    'predictions'.
   * @param predictions The predicted outputs.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
   *    must be either `1`, or the same as the corresponding `losses`
   *    dimension).
   * @param reduction Type of reduction to apply to loss. Should be of type
   *    `Reduction`
   */
  /** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
  function meanSquaredError_(labels, predictions, weights, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      const $labels = convertToTensor(labels, 'labels', 'meanSquaredError');
      const $predictions = convertToTensor(predictions, 'predictions', 'meanSquaredError');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'meanSquaredError');
      }
      assertShapesMatch($labels.shape, $predictions.shape, 'Error in meanSquaredError: ');
      const losses = $labels.squaredDifference($predictions);
      return computeWeightedLoss(losses, $weights, reduction);
  }
  /**
   * Computes the cosine distance loss between two tensors.
   *
   * @param labels The ground truth output tensor, same dimensions as
   *    'predictions'.
   * @param predictions The predicted outputs.
   * @param axis The dimension along which the cosine distance is computed.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
   *    must be either `1`, or the same as the corresponding `losses`
   *    dimension).
   * @param reduction Type of reduction to apply to loss. Should be of type
   *    `Reduction`
   */
  /** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
  function cosineDistance_(labels, predictions, axis, weights, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      const $labels = convertToTensor(labels, 'labels', 'cosineDistance');
      const $predictions = convertToTensor(predictions, 'predictions', 'cosineDistance');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'cosineDistance');
      }
      assertShapesMatch($labels.shape, $predictions.shape, 'Error in cosineDistance: ');
      const one = scalar(1);
      const losses = one.sub($labels.mul($predictions).sum(axis, true));
      return computeWeightedLoss(losses, $weights, reduction);
  }
  /**
   * Computes the Hinge loss between two tensors.
   *
   * @param labels The ground truth output tensor, same dimensions as
   *    'predictions'.
   * @param predictions The predicted outputs.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
   *    must be either `1`, or the same as the corresponding `losses`
   *    dimension).
   * @param reduction Type of reduction to apply to loss. Should be of type
   *    `Reduction`
   */
  /** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
  function hingeLoss_(labels, predictions, weights, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      let $labels = convertToTensor(labels, 'labels', 'hingeLoss');
      const $predictions = convertToTensor(predictions, 'predictions', 'hingeLoss');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'hingeLoss');
      }
      assertShapesMatch($labels.shape, $predictions.shape, 'Error in hingeLoss: ');
      const one = scalar(1);
      // Convert binary labels to (-1, 1)
      $labels = scalar(2).mul($labels).sub(one);
      const losses = one.sub($labels.mul($predictions)).relu();
      return computeWeightedLoss(losses, $weights, reduction);
  }
  /**
   * Computes the log loss between two tensors.
   *
   * @param labels The ground truth output tensor, same dimensions as
   *    'predictions'.
   * @param predictions The predicted outputs.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
   *    must be either `1`, or the same as the corresponding `losses`
   *    dimension).
   * @param epsilon A small increment to avoid taking log of zero
   * @param reduction Type of reduction to apply to loss. Should be of type
   *    `Reduction`
   */
  /** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
  function logLoss_(labels, predictions, weights, epsilon = 1e-7, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      const $labels = convertToTensor(labels, 'labels', 'logLoss');
      const $predictions = convertToTensor(predictions, 'predictions', 'logLoss');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'logLoss');
      }
      assertShapesMatch($labels.shape, $predictions.shape, 'Error in logLoss: ');
      const one = scalar(1);
      const epsilonScalar = scalar(epsilon);
      const losses = $labels.mul($predictions.add(epsilonScalar).log())
          .neg()
          .sub(one.sub($labels).mul(one.sub($predictions).add(epsilonScalar).log()));
      return computeWeightedLoss(losses, $weights, reduction);
  }
  function sigmoidCrossEntropyWithLogits_(labels, logits) {
      const $labels = convertToTensor(labels, 'labels', 'sigmoidCrossEntropyWithLogits');
      const $logits = convertToTensor(logits, 'logits', 'sigmoidCrossEntropyWithLogits');
      assertShapesMatch($labels.shape, $logits.shape, 'Error in sigmoidCrossEntropyWithLogits: ');
      /**
       * Implementation Details:
       *
       * For brevity, let `x = logits`, `z = labels`.  The logistic loss is
       *     z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
       *   = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
       *   = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
       *   = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
       *   = (1 - z) * x + log(1 + exp(-x))
       *   = x - x * z + log(1 + exp(-x))
       *
       *   For x < 0, to avoid overflow in exp(-x), we reformulate the above
       *     x - x * z + log(1 + exp(-x))
       *   = log(exp(x)) - x * z + log(1 + exp(-x))
       *   = - x * z + log(1 + exp(x))
       *
       * Hence, to ensure stability and avoid overflow, the implementation uses
       * this equivalent formulation:
       *     max(x, 0) - x * z + log(1 + exp(-abs(x)))
       */
      const maxOutput = $logits.relu();
      const outputXTarget = $logits.mul($labels);
      const sigmoidOutput = $logits.abs().neg().exp().log1p();
      return maxOutput.sub(outputXTarget).add(sigmoidOutput);
  }
  /**
   * Computes the sigmoid cross entropy loss between two tensors.
   *
   * If labelSmoothing is nonzero, smooth the labels towards 1/2:
   *
   *   newMulticlassLabels = multiclassLabels * (1 - labelSmoothing)
   *                         + 0.5 * labelSmoothing
   *
   * @param multiClassLabels The ground truth output tensor of shape
   * [batch_size, num_classes], same dimensions as 'predictions'.
   * @param logits The predicted outputs.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
   *    must be either `1`, or the same as the corresponding `losses`
   *    dimension).
   * @param labelSmoothing If greater than 0, then smooth the labels.
   * @param reduction Type of reduction to apply to loss. Should be of type
   *    `Reduction`
   */
  /** @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' } */
  function sigmoidCrossEntropy_(multiClassLabels, logits, weights, labelSmoothing = 0, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      let $multiClassLabels = convertToTensor(multiClassLabels, 'multiClassLabels', 'sigmoidCrossEntropy');
      const $logits = convertToTensor(logits, 'logits', 'sigmoidCrossEntropy');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'sigmoidCrossEntropy');
      }
      assertShapesMatch($multiClassLabels.shape, $logits.shape, 'Error in sigmoidCrossEntropy: ');
      if (labelSmoothing > 0) {
          const labelSmoothingScalar = scalar(labelSmoothing);
          const one = scalar(1);
          const half = scalar(0.5);
          $multiClassLabels = $multiClassLabels.mul(one.sub(labelSmoothingScalar))
              .add(half.mul(labelSmoothingScalar));
      }
      const losses = sigmoidCrossEntropyWithLogits_($multiClassLabels, $logits);
      return computeWeightedLoss(losses, $weights, reduction);
  }
  /**
   * Computes the huber loss between two tensors.
   *
   * @param labels The ground truth output tensor, same dimensions as
   *    'predictions'.
   * @param predictions The predicted outputs.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
   *    must be either `1`, or the same as the corresponding `losses`
   *    dimension).
   * @param delta Point where huber loss changes from quadratic to linear.
   * @param reduction Type of reduction to apply to loss. Should be of type
   *    `Reduction`.
   */
  /** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
  function huberLoss_(labels, predictions, weights, delta = 1.0, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      const $labels = convertToTensor(labels, 'labels', 'huberLoss');
      const $predictions = convertToTensor(predictions, 'predictions', 'huberLoss');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'huberLoss');
      }
      assertShapesMatch($labels.shape, $predictions.shape, 'Error in huberLoss: ');
      const deltaScalar = scalar(delta);
      const error = $predictions.sub($labels).abs();
      const quadratic = minimum(error, deltaScalar);
      const linear = error.sub(quadratic);
      const losses = scalar(0.5).mul(quadratic.square()).add(deltaScalar.mul(linear));
      return computeWeightedLoss(losses, $weights, reduction);
  }
  /**
   * Computes softmax cross entropy between logits and labels.
   *
   * Measures the probability error in discrete classification tasks in which
   * the classes are mutually exclusive (each entry is in exactly one class).
   * For example, each CIFAR-10 image is labeled with one and only one label: an
   * image can be a dog or a truck, but not both.
   *
   * `NOTE`: While the classes are mutually exclusive, their probabilities need
   * not be. All that is required is that each row of labels is a valid
   * probability distribution. If they are not, the computation of the gradient
   * will be incorrect.
   *
   * `WARNING`: This op expects unscaled logits, since it performs a softmax on
   * logits internally for efficiency. Do not call this op with the output of
   * softmax, as it will produce incorrect results.
   *
   * logits and labels must have the same shape, e.g. [batch_size, num_classes]
   * and the same dtype.
   * @param labels The labels array.
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to `-1`
   *     which indicates the last dimension.
   */
  function softmaxCrossEntropyWithLogits_(labels, logits, dim = -1) {
      if (dim === -1) {
          dim = logits.rank - 1;
      }
      if (dim !== logits.rank - 1) {
          throw Error(`Softmax cross entropy along a non-last dimension is not yet ` +
              `supported. Labels / logits was rank ${logits.rank} ` +
              `and dim was ${dim}`);
      }
      // Use a custom gradient for numerical stability.
      const customOp = customGrad((labels, logits, save) => {
          // Reference:
          //   1. http://cs231n.github.io/linear-classify/#softmax
          //   2. https://blog.feedly.com/tricks-of-the-trade-logsumexp/
          const keepDims = true;
          const lse = logits.logSumExp([dim], keepDims);
          const logResult = logits.toFloat().sub(lse);
          save([labels, logResult]);
          const costVector = logResult.mul(labels).neg();
          const value = costVector.sum([dim]);
          const gradFunc = (dy, saved) => {
              const [labels, logResult] = saved;
              const dyShape = expandShapeToKeepDim(dy.shape, [dim]);
              return [
                  dy.reshape(dyShape).mul(labels.toFloat().sub(logResult.exp())),
                  dy.reshape(dyShape).mul(logResult.exp().sub(labels.toFloat())),
              ];
          };
          return { value, gradFunc };
      });
      return customOp(labels, logits);
  }
  /**
   * Computes the softmax cross entropy loss between two tensors.
   *
   * If labelSmoothing is nonzero, smooth the labels towards 1/2:
   *
   *   newOnehotLabels = onehotLabels * (1 - labelSmoothing)
   *                         + labelSmoothing / numClasses
   *
   * @param onehotLabels One hot encoded labels
   *    [batch_size, num_classes], same dimensions as 'predictions'.
   * @param logits The predicted outputs.
   * @param weights Tensor whose rank is either 0, or 1, and must be
   *    broadcastable to `loss`  of shape [batch_size]
   * @param labelSmoothing If greater than 0, then smooth the labels.
   * @param reduction Type of reduction to apply to loss. Should be of type
   *    `Reduction`
   */
  /** @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' } */
  function softmaxCrossEntropy_(onehotLabels, logits, weights, labelSmoothing = 0, reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
      let $onehotLabels = convertToTensor(onehotLabels, 'onehotLabels', 'softmaxCrossEntropy');
      const $logits = convertToTensor(logits, 'logits', 'softmaxCrossEntropy');
      let $weights = null;
      if (weights != null) {
          $weights = convertToTensor(weights, 'weights', 'softmaxCrossEntropy');
      }
      assertShapesMatch($onehotLabels.shape, $logits.shape, 'Error in softmaxCrossEntropy: ');
      if (labelSmoothing > 0) {
          const labelSmoothingScalar = scalar(labelSmoothing);
          const one = scalar(1);
          const numClasses = scalar($onehotLabels.shape[1]);
          $onehotLabels = $onehotLabels.mul(one.sub(labelSmoothingScalar))
              .add(labelSmoothingScalar.div(numClasses));
      }
      const losses = softmaxCrossEntropyWithLogits_($onehotLabels, $logits);
      return computeWeightedLoss(losses, $weights, reduction);
  }
  const absoluteDifference = op({ absoluteDifference_ });
  const computeWeightedLoss = op({ computeWeightedLoss_ });
  const cosineDistance = op({ cosineDistance_ });
  const hingeLoss = op({ hingeLoss_ });
  const huberLoss = op({ huberLoss_ });
  const logLoss = op({ logLoss_ });
  const meanSquaredError = op({ meanSquaredError_ });
  const sigmoidCrossEntropy = op({ sigmoidCrossEntropy_ });
  const softmaxCrossEntropy = op({ softmaxCrossEntropy_ });

  var loss_ops = /*#__PURE__*/Object.freeze({
    __proto__: null,
    get Reduction () { return exports.Reduction; },
    absoluteDifference: absoluteDifference,
    computeWeightedLoss: computeWeightedLoss,
    cosineDistance: cosineDistance,
    hingeLoss: hingeLoss,
    huberLoss: huberLoss,
    logLoss: logLoss,
    meanSquaredError: meanSquaredError,
    sigmoidCrossEntropy: sigmoidCrossEntropy,
    softmaxCrossEntropy: softmaxCrossEntropy
  });

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
  /**
   * Copy a tensor setting everything outside a central band in each innermost
   * matrix to zero.
   *
   * The band part is computed as follows: Assume input has `k` dimensions
   * `[I, J, K, ..., M, N]`, then the output is a tensor with the same shape where
   * `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
   * The indicator function
   * `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower))`
   * `&& (num_upper < 0 || (n-m) <= num_upper)`
   *
   * ```js
   * const x = tf.tensor2d([[ 0,  1,  2, 3],
   *                        [-1,  0,  1, 2],
   *                        [-2, -1,  0, 1],
   *                        [-3, -2, -1, 0]]);
   * let y = tf.linalg.bandPart(x, 1, -1);
   * y.print(); // [[ 0,  1,  2, 3],
   *            //  [-1,  0,  1, 2],
   *            //  [ 0, -1,  0, 1],
   *            //  [ 0, 0 , -1, 0]]
   * let z = tf.linalg.bandPart(x, 2, 1);
   * z.print(); // [[ 0,  1,  0, 0],
   *            //  [-1,  0,  1, 0],
   *            //  [-2, -1,  0, 1],
   *            //  [ 0, -2, -1, 0]]
   * ```
   *
   * @param x Rank `k` tensor
   * @param numLower Number of subdiagonals to keep.
   *   If negative, keep entire lower triangle.
   * @param numUpper Number of subdiagonals to keep.
   *   If negative, keep entire upper triangle.
   * @returns Rank `k` tensor of the same shape as input.
   *   The extracted banded tensor.
   */
  /**
   * @doc {heading:'Operations',
   *       subheading:'Linear Algebra',
   *       namespace:'linalg'}
   */
  function bandPart_(a, numLower, numUpper) {
      if (numLower % 1 !== 0) {
          throw new Error(`bandPart(): numLower must be an integer, got ${numLower}.`);
      }
      if (numUpper % 1 !== 0) {
          throw new Error(`bandPart(): numUpper must be an integer, got ${numUpper}.`);
      }
      const $a = convertToTensor(a, 'a', 'bandPart');
      if ($a.rank < 2) {
          throw new Error(`bandPart(): Rank must be at least 2, got ${$a.rank}.`);
      }
      const shape = $a.shape, [M, N] = $a.shape.slice(-2);
      if (!(numLower <= M)) {
          throw new Error(`bandPart(): numLower (${numLower})` +
              ` must not be greater than the number of rows (${M}).`);
      }
      if (!(numUpper <= N)) {
          throw new Error(`bandPart(): numUpper (${numUpper})` +
              ` must not be greater than the number of columns (${N}).`);
      }
      if (numLower < 0) {
          numLower = M;
      }
      if (numUpper < 0) {
          numUpper = N;
      }
      const i = range(0, M, 1, 'int32').reshape([-1, 1]), j = range(0, N, 1, 'int32'), ij = sub(i, j);
      const inBand = logicalAnd(ij.lessEqual(scalar(+numLower, 'int32')), ij.greaterEqual(scalar(-numUpper, 'int32')));
      const zero = zeros([M, N], $a.dtype);
      return stack(unstack($a.reshape([-1, M, N]))
          .map(mat => where(inBand, mat, zero)))
          .reshape(shape);
  }
  /**
   * Gram-Schmidt orthogonalization.
   *
   * ```js
   * const x = tf.tensor2d([[1, 2], [3, 4]]);
   * let y = tf.linalg.gramSchmidt(x);
   * y.print();
   * console.log('Othogonalized:');
   * y.dot(y.transpose()).print();  // should be nearly the identity matrix.
   * console.log('First row direction maintained:');
   * const data = await y.array();
   * console.log(data[0][1] / data[0][0]);  // should be nearly 2.
   * ```
   *
   * @param xs The vectors to be orthogonalized, in one of the two following
   *   formats:
   *   - An Array of `tf.Tensor1D`.
   *   - A `tf.Tensor2D`, i.e., a matrix, in which case the vectors are the rows
   *     of `xs`.
   *   In each case, all the vectors must have the same length and the length
   *   must be greater than or equal to the number of vectors.
   * @returns The orthogonalized and normalized vectors or matrix.
   *   Orthogonalization means that the vectors or the rows of the matrix
   *   are orthogonal (zero inner products). Normalization means that each
   *   vector or each row of the matrix has an L2 norm that equals `1`.
   */
  /**
   * @doc {heading:'Operations',
   *       subheading:'Linear Algebra',
   *       namespace:'linalg'}
   */
  function gramSchmidt_(xs) {
      let inputIsTensor2D;
      if (Array.isArray(xs)) {
          inputIsTensor2D = false;
          assert(xs != null && xs.length > 0, () => 'Gram-Schmidt process: input must not be null, undefined, or ' +
              'empty');
          const dim = xs[0].shape[0];
          for (let i = 1; i < xs.length; ++i) {
              assert(xs[i].shape[0] === dim, () => 'Gram-Schmidt: Non-unique lengths found in the input vectors: ' +
                  `(${xs[i].shape[0]} vs. ${dim})`);
          }
      }
      else {
          inputIsTensor2D = true;
          xs = split(xs, xs.shape[0], 0).map(x => squeeze(x, [0]));
      }
      assert(xs.length <= xs[0].shape[0], () => `Gram-Schmidt: Number of vectors (${xs.length}) exceeds ` +
          `number of dimensions (${xs[0].shape[0]}).`);
      const ys = [];
      const xs1d = xs;
      for (let i = 0; i < xs.length; ++i) {
          ys.push(ENGINE.tidy(() => {
              let x = xs1d[i];
              if (i > 0) {
                  for (let j = 0; j < i; ++j) {
                      const proj = sum$1(ys[j].mulStrict(x)).mul(ys[j]);
                      x = x.sub(proj);
                  }
              }
              return x.div(norm(x, 'euclidean'));
          }));
      }
      if (inputIsTensor2D) {
          return stack(ys, 0);
      }
      else {
          return ys;
      }
  }
  /**
   * Compute QR decomposition of m-by-n matrix using Householder transformation.
   *
   * Implementation based on
   *   [http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf]
   * (http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf)
   *
   * ```js
   * const a = tf.tensor2d([[1, 2], [3, 4]]);
   * let [q, r] = tf.linalg.qr(a);
   * console.log('Q');
   * q.print();
   * console.log('R');
   * r.print();
   * console.log('Orthogonalized');
   * q.dot(q.transpose()).print()  // should be nearly the identity matrix.
   * console.log('Reconstructed');
   * q.dot(r).print(); // should be nearly [[1, 2], [3, 4]];
   * ```
   *
   * @param x The `tf.Tensor` to be QR-decomposed. Must have rank >= 2. Suppose
   *   it has the shape `[..., M, N]`.
   * @param fullMatrices An optional boolean parameter. Defaults to `false`.
   *   If `true`, compute full-sized `Q`. If `false` (the default),
   *   compute only the leading N columns of `Q` and `R`.
   * @returns An `Array` of two `tf.Tensor`s: `[Q, R]`. `Q` is a unitary matrix,
   *   i.e., its columns all have unit norm and are mutually orthogonal.
   *   If `M >= N`,
   *     If `fullMatrices` is `false` (default),
   *       - `Q` has a shape of `[..., M, N]`,
   *       - `R` has a shape of `[..., N, N]`.
   *     If `fullMatrices` is `true` (default),
   *       - `Q` has a shape of `[..., M, M]`,
   *       - `R` has a shape of `[..., M, N]`.
   *   If `M < N`,
   *     - `Q` has a shape of `[..., M, M]`,
   *     - `R` has a shape of `[..., M, N]`.
   * @throws If the rank of `x` is less than 2.
   */
  /**
   * @doc {heading:'Operations',
   *       subheading:'Linear Algebra',
   *       namespace:'linalg'}
   */
  function qr_(x, fullMatrices = false) {
      if (x.rank < 2) {
          throw new Error(`qr() requires input tensor to have a rank >= 2, but got rank ${x.rank}`);
      }
      else if (x.rank === 2) {
          return qr2d(x, fullMatrices);
      }
      else {
          // Rank > 2.
          // TODO(cais): Below we split the input into individual 2D tensors,
          //   perform QR decomposition on them and then stack the results back
          //   together. We should explore whether this can be parallelized.
          const outerDimsProd = x.shape.slice(0, x.shape.length - 2)
              .reduce((value, prev) => value * prev);
          const x2ds = unstack(x.reshape([
              outerDimsProd, x.shape[x.shape.length - 2],
              x.shape[x.shape.length - 1]
          ]), 0);
          const q2ds = [];
          const r2ds = [];
          x2ds.forEach(x2d => {
              const [q2d, r2d] = qr2d(x2d, fullMatrices);
              q2ds.push(q2d);
              r2ds.push(r2d);
          });
          const q = stack(q2ds, 0).reshape(x.shape);
          const r = stack(r2ds, 0).reshape(x.shape);
          return [q, r];
      }
  }
  function qr2d(x, fullMatrices = false) {
      return ENGINE.tidy(() => {
          if (x.shape.length !== 2) {
              throw new Error(`qr2d() requires a 2D Tensor, but got a ${x.shape.length}D Tensor.`);
          }
          const m = x.shape[0];
          const n = x.shape[1];
          let q = eye(m); // Orthogonal transform so far.
          let r = x.clone(); // Transformed matrix so far.
          const one2D = tensor2d([[1]], [1, 1]);
          let w = one2D.clone();
          const iters = m >= n ? n : m;
          for (let j = 0; j < iters; ++j) {
              // This tidy within the for-loop ensures we clean up temporary
              // tensors as soon as they are no longer needed.
              const rTemp = r;
              const wTemp = w;
              const qTemp = q;
              [w, r, q] = ENGINE.tidy(() => {
                  // Find H = I - tau * w * w', to put zeros below R(j, j).
                  const rjEnd1 = r.slice([j, j], [m - j, 1]);
                  const normX = rjEnd1.norm();
                  const rjj = r.slice([j, j], [1, 1]);
                  // The sign() function returns 0 on 0, which causes division by zero.
                  const s = tensor2d([[-1]]).where(rjj.greater(0), tensor2d([[1]]));
                  const u1 = rjj.sub(s.mul(normX));
                  const wPre = rjEnd1.div(u1);
                  if (wPre.shape[0] === 1) {
                      w = one2D.clone();
                  }
                  else {
                      w = one2D.concat(wPre.slice([1, 0], [wPre.shape[0] - 1, wPre.shape[1]]), 0);
                  }
                  const tau = s.matMul(u1).div(normX).neg();
                  // -- R := HR, Q := QH.
                  const rjEndAll = r.slice([j, 0], [m - j, n]);
                  const tauTimesW = tau.mul(w);
                  const wT = w.transpose();
                  if (j === 0) {
                      r = rjEndAll.sub(tauTimesW.matMul(wT.matMul(rjEndAll)));
                  }
                  else {
                      const rTimesTau = rjEndAll.sub(tauTimesW.matMul(wT.matMul(rjEndAll)));
                      r = r.slice([0, 0], [j, n]).concat(rTimesTau, 0);
                  }
                  const tawTimesWT = tauTimesW.transpose();
                  const qAllJEnd = q.slice([0, j], [m, q.shape[1] - j]);
                  if (j === 0) {
                      q = qAllJEnd.sub(qAllJEnd.matMul(w).matMul(tawTimesWT));
                  }
                  else {
                      const qTimesTau = qAllJEnd.sub(qAllJEnd.matMul(w).matMul(tawTimesWT));
                      q = q.slice([0, 0], [m, j]).concat(qTimesTau, 1);
                  }
                  return [w, r, q];
              });
              dispose([rTemp, wTemp, qTemp]);
          }
          if (!fullMatrices && m > n) {
              q = q.slice([0, 0], [m, n]);
              r = r.slice([0, 0], [n, n]);
          }
          return [q, r];
      });
  }
  const bandPart = op({ bandPart_ });
  const gramSchmidt = op({ gramSchmidt_ });
  const qr = op({ qr_ });

  var linalg_ops = /*#__PURE__*/Object.freeze({
    __proto__: null,
    bandPart: bandPart,
    gramSchmidt: gramSchmidt,
    qr: qr
  });

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
  /**
   * Inserts a value into a sorted array. This method allows duplicate, meaning it
   * allows inserting duplicate value, in which case, the element will be inserted
   * at the lowest index of the value.
   * @param arr The array to modify.
   * @param element The element to insert.
   * @param comparator Optional. If no comparator is specified, elements are
   * compared using array_util.defaultComparator, which is suitable for Strings
   * and Numbers in ascending arrays. If the array contains multiple instances of
   * the target value, the left-most instance will be returned. To provide a
   * comparator, it should take 2 arguments to compare and return a negative,
   * zero, or a positive number.
   */
  function binaryInsert(arr, element, comparator) {
      const index = binarySearch(arr, element, comparator);
      const insertionPoint = index < 0 ? -(index + 1) : index;
      arr.splice(insertionPoint, 0, element);
  }
  /**
   * Searches the array for the target using binary search, returns the index
   * of the found element, or position to insert if element not found. If no
   * comparator is specified, elements are compared using array_
   * util.defaultComparator, which is suitable for Strings and Numbers in
   * ascending arrays. If the array contains multiple instances of the target
   * value, the left-most instance will be returned.
   * @param arr The array to be searched in.
   * @param target The target to be searched for.
   * @param comparator Should take 2 arguments to compare and return a negative,
   *    zero, or a positive number.
   * @return Lowest index of the target value if found, otherwise the insertion
   *    point where the target should be inserted, in the form of
   *    (-insertionPoint - 1).
   */
  function binarySearch(arr, target, comparator) {
      return binarySearch_(arr, target, comparator || defaultComparator);
  }
  /**
   * Compares its two arguments for order.
   * @param a The first element to be compared.
   * @param b The second element to be compared.
   * @return A negative number, zero, or a positive number as the first
   *     argument is less than, equal to, or greater than the second.
   */
  function defaultComparator(a, b) {
      return a > b ? 1 : a < b ? -1 : 0;
  }
  function binarySearch_(arr, target, comparator) {
      let left = 0;
      let right = arr.length;
      let middle = 0;
      let found = false;
      while (left < right) {
          middle = left + ((right - left) >>> 1);
          const compareResult = comparator(target, arr[middle]);
          if (compareResult > 0) {
              left = middle + 1;
          }
          else {
              right = middle;
              // If compareResult is 0, the value is found. We record it is found,
              // and then keep looking because there may be duplicate.
              found = !compareResult;
          }
      }
      return found ? left : -left - 1;
  }

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
  function nonMaxSuppressionV3(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
      const dummySoftNmsSigma = 0.0;
      return nonMaxSuppressionImpl_(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, dummySoftNmsSigma)
          .selectedIndices;
  }
  function nonMaxSuppressionV5(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma) {
      // For NonMaxSuppressionV5Op, we always return a second output holding
      // corresponding scores.
      const returnScoresTensor = true;
      const result = nonMaxSuppressionImpl_(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma, returnScoresTensor);
      result.numValidOutputs.dispose();
      return {
          selectedIndices: result.selectedIndices,
          selectedScores: result.selectedScores
      };
  }
  function nonMaxSuppressionImpl_(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma, returnScoresTensor = false, padToMaxOutputSize = false) {
      // The list is sorted in ascending order, so that we can always pop the
      // candidate with the largest score in O(1) time.
      const candidates = Array.from(scores)
          .map((score, boxIndex) => ({ score, boxIndex, suppressBeginIndex: 0 }))
          .filter(c => c.score > scoreThreshold)
          .sort(ascendingComparator);
      // If softNmsSigma is 0, the outcome of this algorithm is exactly same as
      // before.
      const scale = softNmsSigma > 0 ? (-0.5 / softNmsSigma) : 0.0;
      const selectedIndices = [];
      const selectedScores = [];
      while (selectedIndices.length < maxOutputSize && candidates.length > 0) {
          const candidate = candidates.pop();
          const { score: originalScore, boxIndex, suppressBeginIndex } = candidate;
          if (originalScore < scoreThreshold) {
              break;
          }
          // Overlapping boxes are likely to have similar scores, therefore we
          // iterate through the previously selected boxes backwards in order to
          // see if candidate's score should be suppressed. We use
          // suppressBeginIndex to track and ensure a candidate can be suppressed
          // by a selected box no more than once. Also, if the overlap exceeds
          // iouThreshold, we simply ignore the candidate.
          let ignoreCandidate = false;
          for (let j = selectedIndices.length - 1; j >= suppressBeginIndex; --j) {
              const iou = intersectionOverUnion(boxes, boxIndex, selectedIndices[j]);
              if (iou >= iouThreshold) {
                  ignoreCandidate = true;
                  break;
              }
              candidate.score =
                  candidate.score * suppressWeight(iouThreshold, scale, iou);
              if (candidate.score <= scoreThreshold) {
                  break;
              }
          }
          // At this point, if `candidate.score` has not dropped below
          // `scoreThreshold`, then we know that we went through all of the
          // previous selections and can safely update `suppressBeginIndex` to the
          // end of the selected array. Then we can re-insert the candidate with
          // the updated score and suppressBeginIndex back in the candidate list.
          // If on the other hand, `candidate.score` has dropped below the score
          // threshold, we will not add it back to the candidates list.
          candidate.suppressBeginIndex = selectedIndices.length;
          if (!ignoreCandidate) {
              // Candidate has passed all the tests, and is not suppressed, so
              // select the candidate.
              if (candidate.score === originalScore) {
                  selectedIndices.push(boxIndex);
                  selectedScores.push(candidate.score);
              }
              else if (candidate.score > scoreThreshold) {
                  // Candidate's score is suppressed but is still high enough to be
                  // considered, so add back to the candidates list.
                  binaryInsert(candidates, candidate, ascendingComparator);
              }
          }
      }
      // NonMaxSuppressionV4 feature: padding output to maxOutputSize.
      const numValidOutputs = selectedIndices.length;
      if (padToMaxOutputSize) {
          selectedIndices.fill(0, numValidOutputs);
          selectedScores.fill(0.0, numValidOutputs);
      }
      return {
          selectedIndices: tensor1d(selectedIndices, 'int32'),
          selectedScores: tensor1d(selectedScores, 'float32'),
          numValidOutputs: scalar(numValidOutputs, 'int32')
      };
  }
  function intersectionOverUnion(boxes, i, j) {
      const iCoord = boxes.subarray(i * 4, i * 4 + 4);
      const jCoord = boxes.subarray(j * 4, j * 4 + 4);
      const yminI = Math.min(iCoord[0], iCoord[2]);
      const xminI = Math.min(iCoord[1], iCoord[3]);
      const ymaxI = Math.max(iCoord[0], iCoord[2]);
      const xmaxI = Math.max(iCoord[1], iCoord[3]);
      const yminJ = Math.min(jCoord[0], jCoord[2]);
      const xminJ = Math.min(jCoord[1], jCoord[3]);
      const ymaxJ = Math.max(jCoord[0], jCoord[2]);
      const xmaxJ = Math.max(jCoord[1], jCoord[3]);
      const areaI = (ymaxI - yminI) * (xmaxI - xminI);
      const areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
      if (areaI <= 0 || areaJ <= 0) {
          return 0.0;
      }
      const intersectionYmin = Math.max(yminI, yminJ);
      const intersectionXmin = Math.max(xminI, xminJ);
      const intersectionYmax = Math.min(ymaxI, ymaxJ);
      const intersectionXmax = Math.min(xmaxI, xmaxJ);
      const intersectionArea = Math.max(intersectionYmax - intersectionYmin, 0.0) *
          Math.max(intersectionXmax - intersectionXmin, 0.0);
      return intersectionArea / (areaI + areaJ - intersectionArea);
  }
  // A Gaussian penalty function, this method always returns values in [0, 1].
  // The weight is a function of similarity, the more overlap two boxes are, the
  // smaller the weight is, meaning highly overlapping boxe will be significantly
  // penalized. On the other hand, a non-overlapping box will not be penalized.
  function suppressWeight(iouThreshold, scale, iou) {
      const weight = Math.exp(scale * iou * iou);
      return iou <= iouThreshold ? weight : 0.0;
  }
  function ascendingComparator(c1, c2) {
      // For objects with same scores, we make the object with the larger index go
      // first. In an array that pops from the end, this means that the object with
      // the smaller index will be popped first. This ensures the same output as
      // the TensorFlow python version.
      return (c1.score - c2.score) ||
          ((c1.score === c2.score) && (c2.boxIndex - c1.boxIndex));
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /**
   * Bilinear resize a batch of 3D images to a new shape.
   *
   * @param images The images, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param size The new shape `[newHeight, newWidth]` to resize the
   *     images to. Each channel is resized individually.
   * @param alignCorners Defaults to False. If true, rescale
   *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
   *     corners of images and resized images. If false, rescale by
   *     `new_height / height`. Treat similarly the width dimension.
   */
  /** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
  function resizeBilinear_(images, size, alignCorners = false) {
      const $images = convertToTensor(images, 'images', 'resizeBilinear');
      assert($images.rank === 3 || $images.rank === 4, () => `Error in resizeBilinear: x must be rank 3 or 4, but got ` +
          `rank ${$images.rank}.`);
      assert(size.length === 2, () => `Error in resizeBilinear: new shape must 2D, but got shape ` +
          `${size}.`);
      let batchImages = $images;
      let reshapedTo4D = false;
      if ($images.rank === 3) {
          reshapedTo4D = true;
          batchImages =
              $images.as4D(1, $images.shape[0], $images.shape[1], $images.shape[2]);
      }
      const [newHeight, newWidth] = size;
      const forward = (backend, save) => {
          save([batchImages]);
          return backend.resizeBilinear(batchImages, newHeight, newWidth, alignCorners);
      };
      const backward = (dy, saved) => {
          return {
              x: () => ENGINE.runKernelFunc(backend => backend.resizeBilinearBackprop(dy, saved[0], alignCorners), {})
          };
      };
      const res = ENGINE.runKernelFunc(forward, { x: batchImages }, backward, 'ResizeBilinear', { alignCorners, newHeight, newWidth });
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * NearestNeighbor resize a batch of 3D images to a new shape.
   *
   * @param images The images, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param size The new shape `[newHeight, newWidth]` to resize the
   *     images to. Each channel is resized individually.
   * @param alignCorners Defaults to False. If true, rescale
   *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
   *     corners of images and resized images. If false, rescale by
   *     `new_height / height`. Treat similarly the width dimension.
   */
  /** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
  function resizeNearestNeighbor_(images, size, alignCorners = false) {
      const $images = convertToTensor(images, 'images', 'resizeNearestNeighbor');
      assert($images.rank === 3 || $images.rank === 4, () => `Error in resizeNearestNeighbor: x must be rank 3 or 4, but got ` +
          `rank ${$images.rank}.`);
      assert(size.length === 2, () => `Error in resizeNearestNeighbor: new shape must 2D, but got shape ` +
          `${size}.`);
      assert($images.dtype === 'float32' || $images.dtype === 'int32', () => '`images` must have `int32` or `float32` as dtype');
      let batchImages = $images;
      let reshapedTo4D = false;
      if ($images.rank === 3) {
          reshapedTo4D = true;
          batchImages =
              $images.as4D(1, $images.shape[0], $images.shape[1], $images.shape[2]);
      }
      const [newHeight, newWidth] = size;
      const forward = (backend, save) => {
          save([batchImages]);
          return backend.resizeNearestNeighbor(batchImages, newHeight, newWidth, alignCorners);
      };
      const backward = (dy, saved) => {
          return {
              batchImages: () => ENGINE.runKernelFunc(backend => backend.resizeNearestNeighborBackprop(dy, saved[0], alignCorners), {})
          };
      };
      const res = ENGINE.runKernelFunc(forward, { batchImages }, backward);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * Performs non maximum suppression of bounding boxes based on
   * iou (intersection over union).
   *
   * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
   *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
   *     the bounding box.
   * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
   * @param maxOutputSize The maximum number of boxes to be selected.
   * @param iouThreshold A float representing the threshold for deciding whether
   *     boxes overlap too much with respect to IOU. Must be between [0, 1].
   *     Defaults to 0.5 (50% box overlap).
   * @param scoreThreshold A threshold for deciding when to remove boxes based
   *     on score. Defaults to -inf, which means any score is accepted.
   * @return A 1D tensor with the selected box indices.
   */
  /** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
  function nonMaxSuppression_(boxes, scores, maxOutputSize, iouThreshold = 0.5, scoreThreshold = Number.NEGATIVE_INFINITY) {
      const $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppression');
      const $scores = convertToTensor(scores, 'scores', 'nonMaxSuppression');
      const inputs = nonMaxSuppSanityCheck($boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold);
      maxOutputSize = inputs.maxOutputSize;
      iouThreshold = inputs.iouThreshold;
      scoreThreshold = inputs.scoreThreshold;
      const attrs = { maxOutputSize, iouThreshold, scoreThreshold };
      return ENGINE.runKernelFunc(b => b.nonMaxSuppression($boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold), { boxes: $boxes, scores: $scores }, null /* grad */, 'NonMaxSuppressionV3', attrs);
  }
  /** This is the async version of `nonMaxSuppression` */
  async function nonMaxSuppressionAsync_(boxes, scores, maxOutputSize, iouThreshold = 0.5, scoreThreshold = Number.NEGATIVE_INFINITY) {
      const $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
      const $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');
      const inputs = nonMaxSuppSanityCheck($boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold);
      maxOutputSize = inputs.maxOutputSize;
      iouThreshold = inputs.iouThreshold;
      scoreThreshold = inputs.scoreThreshold;
      const boxesAndScores = await Promise.all([$boxes.data(), $scores.data()]);
      const boxesVals = boxesAndScores[0];
      const scoresVals = boxesAndScores[1];
      const res = nonMaxSuppressionV3(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
      if ($boxes !== boxes) {
          $boxes.dispose();
      }
      if ($scores !== scores) {
          $scores.dispose();
      }
      return res;
  }
  /**
   * Performs non maximum suppression of bounding boxes based on
   * iou (intersection over union).
   *
   * This op also supports a Soft-NMS mode (c.f.
   * Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
   * of other overlapping boxes, therefore favoring different regions of the image
   * with high scores. To enable this Soft-NMS mode, set the `softNmsSigma`
   * parameter to be larger than 0.
   *
   * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
   *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
   *     the bounding box.
   * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
   * @param maxOutputSize The maximum number of boxes to be selected.
   * @param iouThreshold A float representing the threshold for deciding whether
   *     boxes overlap too much with respect to IOU. Must be between [0, 1].
   *     Defaults to 0.5 (50% box overlap).
   * @param scoreThreshold A threshold for deciding when to remove boxes based
   *     on score. Defaults to -inf, which means any score is accepted.
   * @param softNmsSigma A float representing the sigma parameter for Soft NMS.
   *     When sigma is 0, it falls back to nonMaxSuppression.
   * @return A map with the following properties:
   *     - selectedIndices: A 1D tensor with the selected box indices.
   *     - selectedScores: A 1D tensor with the corresponding scores for each
   *       selected box.
   */
  /** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
  function nonMaxSuppressionWithScore_(boxes, scores, maxOutputSize, iouThreshold = 0.5, scoreThreshold = Number.NEGATIVE_INFINITY, softNmsSigma = 0.0) {
      const $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppression');
      const $scores = convertToTensor(scores, 'scores', 'nonMaxSuppression');
      const inputs = nonMaxSuppSanityCheck($boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma);
      maxOutputSize = inputs.maxOutputSize;
      iouThreshold = inputs.iouThreshold;
      scoreThreshold = inputs.scoreThreshold;
      softNmsSigma = inputs.softNmsSigma;
      const attrs = { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma };
      const result = ENGINE.runKernel('NonMaxSuppressionV5', { boxes: $boxes, scores: $scores }, attrs);
      return { selectedIndices: result[0], selectedScores: result[1] };
  }
  /** This is the async version of `nonMaxSuppressionWithScore` */
  async function nonMaxSuppressionWithScoreAsync_(boxes, scores, maxOutputSize, iouThreshold = 0.5, scoreThreshold = Number.NEGATIVE_INFINITY, softNmsSigma = 0.0) {
      const $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
      const $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');
      const inputs = nonMaxSuppSanityCheck($boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma);
      maxOutputSize = inputs.maxOutputSize;
      iouThreshold = inputs.iouThreshold;
      scoreThreshold = inputs.scoreThreshold;
      softNmsSigma = inputs.softNmsSigma;
      const boxesAndScores = await Promise.all([$boxes.data(), $scores.data()]);
      const boxesVals = boxesAndScores[0];
      const scoresVals = boxesAndScores[1];
      const res = nonMaxSuppressionV5(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma);
      if ($boxes !== boxes) {
          $boxes.dispose();
      }
      if ($scores !== scores) {
          $scores.dispose();
      }
      return res;
  }
  function nonMaxSuppSanityCheck(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma) {
      if (iouThreshold == null) {
          iouThreshold = 0.5;
      }
      if (scoreThreshold == null) {
          scoreThreshold = Number.NEGATIVE_INFINITY;
      }
      if (softNmsSigma == null) {
          softNmsSigma = 0.0;
      }
      const numBoxes = boxes.shape[0];
      maxOutputSize = Math.min(maxOutputSize, numBoxes);
      assert(0 <= iouThreshold && iouThreshold <= 1, () => `iouThreshold must be in [0, 1], but was '${iouThreshold}'`);
      assert(boxes.rank === 2, () => `boxes must be a 2D tensor, but was of rank '${boxes.rank}'`);
      assert(boxes.shape[1] === 4, () => `boxes must have 4 columns, but 2nd dimension was ${boxes.shape[1]}`);
      assert(scores.rank === 1, () => 'scores must be a 1D tensor');
      assert(scores.shape[0] === numBoxes, () => `scores has incompatible shape with boxes. Expected ${numBoxes}, ` +
          `but was ${scores.shape[0]}`);
      assert(0 <= softNmsSigma && softNmsSigma <= 1, () => `softNmsSigma must be in [0, 1], but was '${softNmsSigma}'`);
      return { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma };
  }
  /**
   * Extracts crops from the input image tensor and resizes them using bilinear
   * sampling or nearest neighbor sampling (possibly with aspect ratio change)
   * to a common output size specified by crop_size.
   *
   * @param image 4d tensor of shape `[batch,imageHeight,imageWidth, depth]`,
   *     where imageHeight and imageWidth must be positive, specifying the
   *     batch of images from which to take crops
   * @param boxes 2d float32 tensor of shape `[numBoxes, 4]`. Each entry is
   *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the normalized
   *     coordinates of the box in the boxInd[i]'th image in the batch
   * @param boxInd 1d int32 tensor of shape `[numBoxes]` with values in range
   *     `[0, batch)` that specifies the image that the `i`-th box refers to.
   * @param cropSize 1d int32 tensor of 2 elements `[cropHeigh, cropWidth]`
   *     specifying the size to which all crops are resized to.
   * @param method Optional string from `'bilinear' | 'nearest'`,
   *     defaults to bilinear, which specifies the sampling method for resizing
   * @param extrapolationValue A threshold for deciding when to remove boxes based
   *     on score. Defaults to 0.
   * @return A 4D tensor of the shape `[numBoxes,cropHeight,cropWidth,depth]`
   */
  /** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
  function cropAndResize_(image, boxes, boxInd, cropSize, method, extrapolationValue) {
      const $image = convertToTensor(image, 'image', 'cropAndResize');
      const $boxes = convertToTensor(boxes, 'boxes', 'cropAndResize', 'float32');
      const $boxInd = convertToTensor(boxInd, 'boxInd', 'cropAndResize', 'int32');
      method = method || 'bilinear';
      extrapolationValue = extrapolationValue || 0;
      const numBoxes = $boxes.shape[0];
      assert($image.rank === 4, () => 'Error in cropAndResize: image must be rank 4,' +
          `but got rank ${$image.rank}.`);
      assert($boxes.rank === 2 && $boxes.shape[1] === 4, () => `Error in cropAndResize: boxes must be have size [${numBoxes},4] ` +
          `but had shape ${$boxes.shape}.`);
      assert($boxInd.rank === 1 && $boxInd.shape[0] === numBoxes, () => `Error in cropAndResize: boxInd must be have size [${numBoxes}] ` +
          `but had shape ${$boxes.shape}.`);
      assert(cropSize.length === 2, () => `Error in cropAndResize: cropSize must be of length 2, but got ` +
          `length ${cropSize.length}.`);
      assert(cropSize[0] >= 1 && cropSize[1] >= 1, () => `cropSize must be atleast [1,1], but was ${cropSize}`);
      assert(method === 'bilinear' || method === 'nearest', () => `method must be bilinear or nearest, but was ${method}`);
      const forward = (backend, save) => backend.cropAndResize($image, $boxes, $boxInd, cropSize, method, extrapolationValue);
      const res = ENGINE.runKernelFunc(forward, { images: $image, boxes: $boxes, boxInd: $boxInd }, null /* der */, 'CropAndResize', { method, extrapolationValue, cropSize });
      return res;
  }
  const resizeBilinear = op({ resizeBilinear_ });
  const resizeNearestNeighbor = op({ resizeNearestNeighbor_ });
  const nonMaxSuppression = op({ nonMaxSuppression_ });
  const nonMaxSuppressionAsync = nonMaxSuppressionAsync_;
  const nonMaxSuppressionWithScore = op({ nonMaxSuppressionWithScore_ });
  const nonMaxSuppressionWithScoreAsync = nonMaxSuppressionWithScoreAsync_;
  const cropAndResize = op({ cropAndResize_ });

  var image_ops = /*#__PURE__*/Object.freeze({
    __proto__: null,
    resizeBilinear: resizeBilinear,
    resizeNearestNeighbor: resizeNearestNeighbor,
    nonMaxSuppression: nonMaxSuppression,
    nonMaxSuppressionAsync: nonMaxSuppressionAsync,
    nonMaxSuppressionWithScore: nonMaxSuppressionWithScore,
    nonMaxSuppressionWithScoreAsync: nonMaxSuppressionWithScoreAsync,
    cropAndResize: cropAndResize
  });

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
  // Whether we should call fused ops.
  const shouldFuse = (gradientDepth, activation) => {
      const gradientMode = gradientDepth > 0;
      return !gradientMode || activation === 'linear';
  };

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
  // Returns gradient for fused activation.
  const getFusedDyActivation = (dy, y, activation) => {
      if (activation == null || activation === 'linear') {
          return dy;
      }
      if (activation === 'relu') {
          return dy.mul(y.step());
      }
      throw new Error(`Gradient for activation ${activation} has not been ` +
          `implemented yet.`);
  };
  // Returns gradient for fused bias.
  const getFusedBiasGradient = (bias, dyActivation) => {
      let res = dyActivation;
      const reduceAxes = getReductionAxes(bias.shape, dyActivation.shape);
      if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
      }
      return res.reshape(bias.shape);
  };
  const applyActivation = (x, activation, preluActivationWeights) => {
      if (activation === 'linear') {
          return x;
      }
      else if (activation === 'relu') {
          return relu(x);
      }
      else if (activation === 'elu') {
          return elu(x);
      }
      else if (activation === 'relu6') {
          return relu6(x);
      }
      else if (activation === 'prelu') {
          return prelu(x, preluActivationWeights);
      }
      throw new Error(`Unknown fused activation ${activation}.`);
  };
  /**
   * Computes the dot product of two matrices with optional activation and bias.
   *
   * ```js
   * const a = tf.tensor2d([-1, -2], [1, 2]);
   * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   * const bias = tf.tensor2d([1, 2], [1, 2]);
   *
   * tf.fused.matMul({a, b, bias, activation: 'relu'}).print();
   * ```
   *
   * @param obj An object with the following properties:
   * - `a` First matrix in dot product operation.
   * - `b` Second matrix in dot product operation.
   * - `transposeA` If true, `a` is transposed before multiplication.
   * - `transposeB` If true, `b` is transposed before multiplication.
   * - `bias` Matrix to be added to the result.
   * - `activation` Name of activation kernel (defaults to `linear`).
   * - `preluActivationWeights` Tensor of prelu weights.
   */
  function fusedMatMul_({ a, b, transposeA = false, transposeB = false, bias, activation = 'linear', preluActivationWeights }) {
      if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
          let result = matMul(a, b, transposeA, transposeB);
          if (bias != null) {
              result = add(result, bias);
          }
          return applyActivation(result, activation, preluActivationWeights);
      }
      let $a = convertToTensor(a, 'a', 'fused matMul');
      let $b = convertToTensor(b, 'b', 'fused matMul');
      [$a, $b] = makeTypesMatch($a, $b);
      const innerShapeA = transposeA ? $a.shape[$a.rank - 2] : $a.shape[$a.rank - 1];
      const innerShapeB = transposeB ? $b.shape[$b.rank - 1] : $b.shape[$b.rank - 2];
      const outerShapeA = transposeA ? $a.shape[$a.rank - 1] : $a.shape[$a.rank - 2];
      const outerShapeB = transposeB ? $b.shape[$b.rank - 2] : $b.shape[$b.rank - 1];
      const outerDimsA = $a.shape.slice(0, -2);
      const outerDimsB = $b.shape.slice(0, -2);
      const batchDimA = sizeFromShape(outerDimsA);
      const batchDimB = sizeFromShape(outerDimsB);
      assert($a.rank >= 2 && $b.rank >= 2 && $a.rank === $b.rank, () => `Error in fused matMul: inputs must have the same rank of at least ` +
          `2, got ranks ${$a.rank} and ${$b.rank}.`);
      assert(arraysEqual(outerDimsA, outerDimsB), () => `Error in fused matMul: outer dimensions (${outerDimsA}) and (` +
          `${outerDimsB}) of Tensors with shapes ${$a.shape} and ` +
          `${$b.shape} must match.`);
      assert(innerShapeA === innerShapeB, () => `Error in fused matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${$a.shape} and ` +
          `${$b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);
      const outShape = $a.shape.slice(0, -2).concat([outerShapeA, outerShapeB]);
      const a3D = transposeA ? $a.as3D(batchDimA, innerShapeA, outerShapeA) :
          $a.as3D(batchDimA, outerShapeA, innerShapeA);
      const b3D = transposeB ? $b.as3D(batchDimB, outerShapeB, innerShapeB) :
          $b.as3D(batchDimB, innerShapeB, outerShapeB);
      let $bias;
      if (bias != null) {
          $bias = convertToTensor(bias, 'bias', 'fused matMul');
          [$bias] = makeTypesMatch($bias, $a);
          assertAndGetBroadcastShape(outShape, $bias.shape);
      }
      let $preluActivationWeights;
      if (preluActivationWeights != null) {
          $preluActivationWeights = convertToTensor(preluActivationWeights, 'prelu weights', 'fused matMul');
      }
      const grad = (dy, saved) => {
          const [a3D, b3D, y] = saved;
          const dyActivation = getFusedDyActivation(dy, y, activation);
          let biasGradient = {};
          if (bias != null) {
              biasGradient = { bias: () => getFusedBiasGradient($bias, dyActivation) };
          }
          if (!transposeA && !transposeB) {
              return Object.assign({
                  a: () => dyActivation.matMul(b3D, false, true),
                  b: () => a3D.matMul(dyActivation, true, false)
              }, biasGradient);
          }
          else if (!transposeA && transposeB) {
              return Object.assign({
                  a: () => dyActivation.matMul(b3D, false, false),
                  b: () => dyActivation.matMul(a3D, true, false)
              }, biasGradient);
          }
          else if (transposeA && !transposeB) {
              return Object.assign({
                  a: () => b3D.matMul(dyActivation, false, true),
                  b: () => a3D.matMul(dyActivation, false, false)
              }, biasGradient);
          }
          else {
              return Object.assign({
                  a: () => b3D.matMul(dyActivation, true, true),
                  b: () => dyActivation.matMul(a3D, true, true)
              }, biasGradient);
          }
      };
      const inputs = { a: a3D, b: b3D };
      if (bias != null) {
          inputs.bias = $bias;
      }
      if (preluActivationWeights != null) {
          inputs.preluActivationWeights = $preluActivationWeights;
      }
      const inputsToSave = [a3D, b3D];
      const outputsToSave = [true];
      const res = ENGINE.runKernelFunc((backend, save) => {
          const y = backend.fusedBatchMatMul({
              a: a3D,
              b: b3D,
              transposeA,
              transposeB,
              bias: $bias,
              activation,
              preluActivationWeights: $preluActivationWeights
          });
          save([a3D, b3D, y]);
          return y;
      }, inputs, grad, '_FusedMatMul', { transposeA, transposeB, activation }, inputsToSave, outputsToSave);
      return res.reshape(outShape);
  }
  /**
   * Computes a 2D convolution over the input x, optionally fused with adding a
   * bias and applying an activation.
   *
   * ```js
   * const inputDepth = 2;
   * const inShape = [2, 2, 2, inputDepth];
   * const outputDepth = 2;
   * const fSize = 1;
   * const pad = 0;
   * const strides = 1;
   *
   * const x = tf.tensor4d( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
   * 16], inShape);
   * const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth,
   * outputDepth]);
   *
   * tf.fused.conv2d({ x, filter: w, strides, pad, dataFormat: 'NHWC',
   * dilations: [1, 1], bias: tf.scalar(5), activation: 'relu' }).print();
   * ```
   *
   * @param obj An object with the following properties:
   * @param x The input tensor, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
   * assumed.
   * @param filter The filter, rank 4, of shape
   *     `[filterHeight, filterWidth, inDepth, outDepth]`.
   * @param strides The strides of the convolution: `[strideHeight,
   * strideWidth]`.
   * @param pad The type of padding algorithm.
   *   - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *   - `valid` output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to
   *     "NHWC". Specify the data format of the input and output data. With the
   *     default format "NHWC", the data is stored in the order of: [batch,
   *     height, width, channels]. Only "NHWC" is currently supported.
   * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
   *     in which we sample input values across the height and width dimensions
   *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
   *     number, then `dilationHeight == dilationWidth`. If it is greater than
   *     1, then all values of `strides` must be 1.
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   * @param bias Tensor to be added to the result.
   * @param activation Name of activation kernel (defaults to `linear`) to be
   *     applied
   *      after biasAdd.
   * @param preluActivationWeights Tensor of prelu weights to be applied as part
   *     of a `prelu` activation, typically the same shape as `x`.
   */
  function fusedConv2d_({ x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode, bias, activation = 'linear', preluActivationWeights }) {
      activation = activation || 'linear';
      if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
          let result = conv2d(x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
          if (bias != null) {
              result = add(result, bias);
          }
          return applyActivation(result, activation, preluActivationWeights);
      }
      const $x = convertToTensor(x, 'x', 'conv2d');
      const $filter = convertToTensor(filter, 'filter', 'conv2d');
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      assert(x4D.rank === 4, () => `Error in fused conv2d: input must be rank 4, but got rank ` +
          `${x4D.rank}.`);
      assert($filter.rank === 4, () => `Error in fused conv2d: filter must be rank 4, but got rank ` +
          `${$filter.rank}.`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in fused conv2d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      assert(x4D.shape[3] === $filter.shape[2], () => `Error in conv2d: depth of input (${x4D.shape[3]}) must match ` +
          `input depth for filter ${$filter.shape[2]}.`);
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in conv2D: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
      assert(dataFormat === 'NHWC', () => `Error in conv2d: got dataFormat of ${dataFormat} but only NHWC is currently supported.`);
      const convInfo = computeConv2DInfo(x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode);
      let $bias;
      if (bias != null) {
          $bias = convertToTensor(bias, 'bias', 'fused conv2d');
          [$bias] = makeTypesMatch($bias, $x);
          assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
      }
      let $preluActivationWeights;
      if (preluActivationWeights != null) {
          $preluActivationWeights = convertToTensor(preluActivationWeights, 'prelu weights', 'fused conv2d');
      }
      const grad = (dy, saved) => {
          const [$filter, x4D, y] = saved;
          const dyActivation = getFusedDyActivation(dy, y, activation);
          assert(tupleValuesAreOne(dilations), () => 'Error in gradient of fused conv2D: ' +
              `dilation rates greater than 1 ` +
              `are not yet supported in gradients. Got dilations '${dilations}'`);
          let biasGradient = {};
          if (bias != null) {
              biasGradient = { bias: () => getFusedBiasGradient($bias, dyActivation) };
          }
          return Object.assign({
              x: () => conv2dDerInput(x4D.shape, dyActivation, $filter, strides, pad),
              filter: () => conv2dDerFilter(x4D, dyActivation, $filter.shape, strides, pad)
          }, biasGradient);
      };
      const inputs = { x: x4D, filter: $filter };
      if (bias != null) {
          inputs.bias = $bias;
      }
      if (preluActivationWeights != null) {
          inputs.preluActivationWeights = $preluActivationWeights;
      }
      const inputsToSave = [$filter, x4D];
      const outputsToSave = [true]; // Save the only output.
      const res = ENGINE.runKernelFunc((backend, save) => {
          const res = backend.fusedConv2d({
              input: x4D,
              filter: $filter,
              convInfo,
              bias: $bias,
              activation,
              preluActivationWeights: $preluActivationWeights
          });
          save([$filter, x4D, res]);
          return res;
      }, inputs, grad, 'FusedConv2D', { convInfo, activation }, inputsToSave, outputsToSave);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  /**
   * Computes depthwise 2D convolution, optionally fused with adding a
   * bias and applying an activation.
   *
   * Given a 4D `input` array and a `filter` array of shape
   * `[filterHeight, filterWidth, inChannels, channelMultiplier]` containing
   * `inChannels` convolutional filters of depth 1, this op applies a
   * different filter to each input channel (expanding from 1 channel to
   * `channelMultiplier` channels for each), then concatenates the results
   * together. The output has `inChannels * channelMultiplier` channels.
   *
   * See
   * [https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d](
   *     https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)
   * for more details.
   *
   * @param obj An object with the following properties:
   * @param x The input tensor, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
   * assumed.
   * @param filter The filter tensor, rank 4, of shape
   *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`.
   * @param strides The strides of the convolution: `[strideHeight,
   * strideWidth]`. If strides is a single number, then `strideHeight ==
   * strideWidth`.
   * @param pad The type of padding algorithm.
   *   - `same` and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *   - `valid`: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
   *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
   * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
   *     in which we sample input values across the height and width dimensions
   *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
   *     number, then `dilationHeight == dilationWidth`. If it is greater than
   *     1, then all values of `strides` must be 1.
   * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
   *     "NHWC". Specify the data format of the input and output data. With the
   *     default format "NHWC", the data is stored in the order of: [batch,
   *     height, width, channels]. Only "NHWC" is currently supported.
   * @param dimRoundingMode The rounding mode used when computing output
   *     dimensions if pad is a number. If none is provided, it will not round
   *     and error if the output is of fractional size.
   * @param bias Tensor to be added to the result.
   * @param activation Name of activation kernel (defaults to `linear`).
   * @param preluActivationWeights Tensor of prelu weights to be applied as part
   *     of a `prelu` activation, typically the same shape as `x`.
   */
  function fusedDepthwiseConv2d_({ x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode, bias, activation = 'linear', preluActivationWeights }) {
      if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
          let result = depthwiseConv2d(x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
          if (bias != null) {
              result = add(result, bias);
          }
          return applyActivation(result, activation, preluActivationWeights);
      }
      const $x = convertToTensor(x, 'x', 'depthwiseConv2d');
      const $filter = convertToTensor(filter, 'filter', 'depthwiseConv2d');
      let x4D = $x;
      let reshapedTo4D = false;
      if ($x.rank === 3) {
          reshapedTo4D = true;
          x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
      }
      assert(x4D.rank === 4, () => `Error in fused depthwiseConv2d: input must be rank 4, but got ` +
          `rank ${x4D.rank}.`);
      assert($filter.rank === 4, () => `Error in fused depthwiseConv2d: filter must be rank 4, ` +
          `but got rank ${$filter.rank}.`);
      assert(x4D.shape[3] === $filter.shape[2], () => `Error in fused depthwiseConv2d: number of input channels ` +
          `(${x4D.shape[3]}) must match the inChannels dimension in ` +
          `filter ${$filter.shape[2]}.`);
      if (dilations == null) {
          dilations = [1, 1];
      }
      assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in fused depthwiseConv2d: Either strides or dilations must ' +
          `be 1. Got strides ${strides} and dilations '${dilations}'`);
      if (dimRoundingMode != null) {
          assert(isInt(pad), () => `Error in fused depthwiseConv2d: pad must be an integer when ` +
              `using dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
      }
      const convInfo = computeConv2DInfo(x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode, true /* depthwise */);
      let $bias;
      if (bias != null) {
          $bias = convertToTensor(bias, 'bias', 'fused conv2d');
          [$bias] = makeTypesMatch($bias, $x);
          assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
      }
      let $preluActivationWeights;
      if (preluActivationWeights != null) {
          $preluActivationWeights = convertToTensor(preluActivationWeights, 'prelu weights', 'fused depthwiseConv2d');
      }
      const grad = (dy, saved) => {
          assert(tupleValuesAreOne(dilations), () => 'Error in gradient of fused depthwiseConv2d: dilation rates ' +
              `greater than 1 are not yet supported. Got dilations ` +
              `'${dilations}'`);
          const [$filter, x4D, y] = saved;
          const dyActivation = getFusedDyActivation(dy, y, activation);
          let biasGradient = {};
          if (bias != null) {
              biasGradient = { bias: () => getFusedBiasGradient($bias, dyActivation) };
          }
          return Object.assign({
              x: () => depthwiseConv2dDerInput(x4D.shape, dyActivation, $filter, convInfo),
              filter: () => depthwiseConv2dDerFilter(x4D, dyActivation, $filter.shape, convInfo),
          }, biasGradient);
      };
      const inputs = { x: x4D, filter: $filter };
      if (bias != null) {
          inputs.bias = $bias;
      }
      if (preluActivationWeights != null) {
          inputs.preluActivationWeights = $preluActivationWeights;
      }
      const inputsToSave = [$filter, x4D];
      const outputsToSave = [true];
      const res = ENGINE.runKernelFunc((backend, save) => {
          const res = backend.fusedDepthwiseConv2D({
              input: x4D,
              filter: $filter,
              convInfo,
              bias: $bias,
              activation,
              preluActivationWeights: $preluActivationWeights
          });
          save([$filter, x4D, res]);
          return res;
      }, inputs, grad, 'FusedDepthwiseConv2D', { convInfo, activation }, inputsToSave, outputsToSave);
      if (reshapedTo4D) {
          return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
      }
      return res;
  }
  const matMul$1 = op({ fusedMatMul_ });
  const conv2d$1 = op({ fusedConv2d_ });
  const depthwiseConv2d$1 = op({ fusedDepthwiseConv2d_ });

  var fused_ops = /*#__PURE__*/Object.freeze({
    __proto__: null,
    matMul: matMul$1,
    conv2d: conv2d$1,
    depthwiseConv2d: depthwiseConv2d$1
  });

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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

  var ops = /*#__PURE__*/Object.freeze({
    __proto__: null,
    image: image_ops,
    linalg: linalg_ops,
    losses: loss_ops,
    spectral: spectral_ops,
    fused: fused_ops,
    signal: signal_ops,
    add: add,
    addN: addN,
    batchNorm: batchNorm,
    batchNormalization: batchNormalization,
    batchNorm2d: batchNorm2d,
    batchNormalization2d: batchNormalization2d,
    batchNorm3d: batchNorm3d,
    batchNormalization3d: batchNormalization3d,
    batchNorm4d: batchNorm4d,
    batchNormalization4d: batchNormalization4d,
    broadcastTo: broadcastTo,
    clone: clone,
    div: div,
    divNoNan: divNoNan,
    eye: eye,
    multinomial: multinomial,
    notEqual: notEqual,
    oneHot: oneHot,
    pad: pad,
    pad1d: pad1d,
    pad2d: pad2d,
    pad3d: pad3d,
    pad4d: pad4d,
    rand: rand,
    randomGamma: randomGamma,
    randomNormal: randomNormal,
    randomUniform: randomUniform,
    square: square,
    squaredDifference: squaredDifference,
    sub: sub,
    tile: tile,
    truncatedNormal: truncatedNormal,
    conv1d: conv1d,
    conv2d: conv2d,
    conv3d: conv3d,
    depthwiseConv2d: depthwiseConv2d,
    separableConv2d: separableConv2d,
    conv2dTranspose: conv2dTranspose,
    conv3dTranspose: conv3dTranspose,
    op: op,
    booleanMaskAsync: booleanMaskAsync,
    complex: complex,
    real: real,
    imag: imag,
    concat: concat,
    concat1d: concat1d,
    concat2d: concat2d,
    concat3d: concat3d,
    concat4d: concat4d,
    split: split,
    matMul: matMul,
    dot: dot,
    outerProduct: outerProduct,
    reverse: reverse,
    reverse1d: reverse1d,
    reverse2d: reverse2d,
    reverse3d: reverse3d,
    reverse4d: reverse4d,
    maxPool: maxPool,
    avgPool: avgPool,
    pool: pool,
    maxPool3d: maxPool3d,
    avgPool3d: avgPool3d,
    maxPoolWithArgmax: maxPoolWithArgmax,
    slice: slice,
    slice1d: slice1d,
    slice2d: slice2d,
    slice3d: slice3d,
    slice4d: slice4d,
    abs: abs,
    acos: acos,
    acosh: acosh,
    asin: asin,
    asinh: asinh,
    atan: atan,
    atanh: atanh,
    ceil: ceil,
    clipByValue: clipByValue,
    cos: cos,
    cosh: cosh,
    erf: erf,
    exp: exp,
    expm1: expm1,
    floor: floor,
    log: log,
    log1p: log1p,
    logSigmoid: logSigmoid,
    neg: neg,
    reciprocal: reciprocal,
    round: round,
    rsqrt: rsqrt,
    sigmoid: sigmoid,
    sign: sign,
    isNaN: isNaN$1,
    isInf: isInf,
    isFinite: isFinite$1,
    sin: sin,
    sinh: sinh,
    softplus: softplus,
    sqrt: sqrt,
    step: step,
    tan: tan,
    tanh: tanh$1,
    all: all,
    any: any,
    argMax: argMax,
    argMin: argMin,
    logSumExp: logSumExp,
    max: max,
    mean: mean,
    min: min,
    moments: moments,
    sum: sum$1,
    prod: prod,
    equal: equal,
    equalStrict: equalStrict,
    greater: greater,
    greaterEqual: greaterEqual,
    greaterEqualStrict: greaterEqualStrict,
    greaterStrict: greaterStrict,
    less: less,
    lessEqual: lessEqual,
    lessEqualStrict: lessEqualStrict,
    lessStrict: lessStrict,
    notEqualStrict: notEqualStrict,
    addStrict: addStrict,
    atan2: atan2,
    divStrict: divStrict,
    floorDiv: floorDiv,
    maximum: maximum,
    maximumStrict: maximumStrict,
    minimum: minimum,
    minimumStrict: minimumStrict,
    mod: mod,
    modStrict: modStrict,
    mul: mul,
    mulStrict: mulStrict,
    pow: pow,
    powStrict: powStrict,
    squaredDifferenceStrict: squaredDifferenceStrict,
    subStrict: subStrict,
    elu: elu,
    leakyRelu: leakyRelu,
    prelu: prelu,
    relu: relu,
    relu6: relu6,
    selu: selu,
    logicalAnd: logicalAnd,
    logicalNot: logicalNot,
    logicalOr: logicalOr,
    logicalXor: logicalXor,
    where: where,
    whereAsync: whereAsync,
    buffer: buffer,
    print: print,
    batchToSpaceND: batchToSpaceND,
    cast: cast,
    cumsum: cumsum,
    depthToSpace: depthToSpace,
    expandDims: expandDims,
    reshape: reshape,
    spaceToBatchND: spaceToBatchND,
    squeeze: squeeze,
    stack: stack,
    unstack: unstack,
    setdiff1dAsync: setdiff1dAsync,
    fill: fill,
    linspace: linspace,
    ones: ones$1,
    range: range,
    scalar: scalar,
    tensor: tensor,
    tensor1d: tensor1d,
    tensor2d: tensor2d,
    tensor3d: tensor3d,
    tensor4d: tensor4d,
    tensor5d: tensor5d,
    tensor6d: tensor6d,
    variable: variable,
    zeros: zeros,
    onesLike: onesLike,
    zerosLike: zerosLike,
    transpose: transpose,
    softmax: softmax,
    logSoftmax: logSoftmax,
    localResponseNormalization: localResponseNormalization,
    norm: norm,
    gather: gather,
    unsortedSegmentSum: unsortedSegmentSum,
    basicLSTMCell: basicLSTMCell,
    multiRNNCell: multiRNNCell,
    movingAverage: movingAverage,
    stridedSlice: stridedSlice,
    topk: topk,
    scatterND: scatterND,
    fft: fft,
    ifft: ifft,
    rfft: rfft,
    irfft: irfft,
    sparseToDense: sparseToDense,
    gatherND: gatherND,
    diag: diag,
    dropout: dropout,
    hannWindow: hannWindow,
    hammingWindow: hammingWindow,
    frame: frame,
    stft: stft,
    inTopKAsync: inTopKAsync
  });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /** @doc {heading: 'Training', subheading: 'Classes', namespace: 'train'} */
  class Optimizer extends Serializable {
      /**
       * Executes `f()` and minimizes the scalar output of `f()` by computing
       * gradients of y with respect to the list of trainable variables provided by
       * `varList`. If no list is provided, it defaults to all trainable variables.
       *
       * @param f The function to execute and whose output to minimize.
       * @param returnCost Whether to return the scalar cost value produced by
       * executing `f()`.
       * @param varList An optional list of variables to update. If specified, only
       * the trainable variables in varList will be updated by minimize. Defaults to
       * all trainable variables.
       */
      /** @doc {heading: 'Training', subheading: 'Optimizers'} */
      minimize(f, returnCost = false, varList) {
          const { value, grads } = this.computeGradients(f, varList);
          if (varList != null) {
              const gradArray = varList.map(v => ({ name: v.name, tensor: grads[v.name] }));
              this.applyGradients(gradArray);
          }
          else {
              this.applyGradients(grads);
          }
          // Dispose gradients.
          dispose(grads);
          if (returnCost) {
              return value;
          }
          else {
              value.dispose();
              return null;
          }
      }
      /**
       * The number of iterations that this optimizer instance has been invoked for.
       */
      get iterations() {
          if (this.iterations_ == null) {
              this.iterations_ = 0;
          }
          return this.iterations_;
      }
      incrementIterations() {
          this.iterations_ = this.iterations + 1;
      }
      /**
       * Executes f() and computes the gradient of the scalar output of f() with
       * respect to the list of trainable variables provided by `varList`. If no
       * list is provided, it defaults to all trainable variables.
       *
       * @param f The function to execute and whose output to use for computing
       * gradients with respect to variables.
       * @param varList An optional list of variables to compute gradients with
       * respect to. If specified, only the trainable variables in varList will have
       * gradients computed with respect to. Defaults to all trainable variables.
       */
      computeGradients(f, varList) {
          return variableGrads(f, varList);
      }
      /**
       * Dispose the variables (if any) owned by this optimizer instance.
       */
      dispose() {
          if (this.iterations_ != null) {
              dispose(this.iterations_);
          }
      }
      async saveIterations() {
          if (this.iterations_ == null) {
              this.iterations_ = 0;
          }
          return {
              name: 'iter',
              // TODO(cais): Use 'int64' type when available.
              tensor: scalar(this.iterations_, 'int32')
          };
      }
      async getWeights() {
          throw new Error('getWeights() is not implemented for this optimizer yet.');
      }
      async setWeights(weightValues) {
          throw new Error(`setWeights() is not implemented for this optimizer class ` +
              `${this.getClassName()}`);
      }
      /**
       * Extract the first element of the weight values and set it
       * as the iterations counter variable of this instance of optimizer.
       *
       * @param weightValues
       * @returns Weight values with the first element consumed and excluded.
       */
      async extractIterations(weightValues) {
          this.iterations_ = (await weightValues[0].tensor.data())[0];
          return weightValues.slice(1);
      }
  }
  Object.defineProperty(Optimizer, Symbol.hasInstance, {
      value: (instance) => {
          return instance.minimize != null && instance.computeGradients != null &&
              instance.applyGradients != null;
      }
  });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /** @doclink Optimizer */
  class AdadeltaOptimizer extends Optimizer {
      constructor(learningRate, rho, epsilon = null) {
          super();
          this.learningRate = learningRate;
          this.rho = rho;
          this.epsilon = epsilon;
          this.accumulatedGrads = [];
          this.accumulatedUpdates = [];
          if (epsilon == null) {
              this.epsilon = ENGINE.backend.epsilon();
          }
      }
      applyGradients(variableGradients) {
          const variableNames = Array.isArray(variableGradients) ?
              variableGradients.map(item => item.name) :
              Object.keys(variableGradients);
          variableNames.forEach((name, i) => {
              const value = ENGINE.registeredVariables[name];
              const trainable = false;
              if (this.accumulatedGrads[i] == null) {
                  this.accumulatedGrads[i] = {
                      originalName: `${name}/accum_grad`,
                      variable: tidy(() => zerosLike(value).variable(trainable))
                  };
              }
              if (this.accumulatedUpdates[i] == null) {
                  this.accumulatedUpdates[i] = {
                      originalName: `${name}/accum_var`,
                      variable: tidy(() => zerosLike(value).variable(trainable))
                  };
              }
              const gradient = Array.isArray(variableGradients) ?
                  variableGradients[i].tensor :
                  variableGradients[name];
              if (gradient == null) {
                  return;
              }
              const accumulatedGrad = this.accumulatedGrads[i].variable;
              const accumulatedUpdate = this.accumulatedUpdates[i].variable;
              tidy(() => {
                  const newAccumulatedGrad = accumulatedGrad.mul(this.rho).add(gradient.square().mul(1 - this.rho));
                  const updates = accumulatedUpdate.add(this.epsilon)
                      .sqrt()
                      .div(accumulatedGrad.add(this.epsilon).sqrt())
                      .mul(gradient);
                  const newAccumulatedUpdate = accumulatedUpdate.mul(this.rho).add(updates.square().mul(1 - this.rho));
                  accumulatedGrad.assign(newAccumulatedGrad);
                  accumulatedUpdate.assign(newAccumulatedUpdate);
                  const newValue = updates.mul(-this.learningRate).add(value);
                  value.assign(newValue);
              });
          });
          this.incrementIterations();
      }
      dispose() {
          if (this.accumulatedUpdates != null) {
              dispose(this.accumulatedGrads.map(v => v.variable));
              dispose(this.accumulatedUpdates.map(v => v.variable));
          }
      }
      async getWeights() {
          // Order matters for Python compatibility.
          const variables = [...this.accumulatedGrads, ...this.accumulatedUpdates];
          return [await this.saveIterations()].concat(variables.map(v => ({ name: v.originalName, tensor: v.variable })));
      }
      async setWeights(weightValues) {
          weightValues = await this.extractIterations(weightValues);
          const variableCount = weightValues.length / 2;
          const trainable = false;
          this.accumulatedGrads =
              weightValues.slice(0, variableCount).map(v => ({
                  originalName: v.name,
                  variable: v.tensor.variable(trainable)
              }));
          this.accumulatedUpdates =
              weightValues.slice(variableCount, variableCount * 2)
                  .map(v => ({
                  originalName: v.name,
                  variable: v.tensor.variable(trainable)
              }));
      }
      getConfig() {
          return {
              'learningRate': this.learningRate,
              'rho': this.rho,
              'epsilon': this.epsilon
          };
      }
      /** @nocollapse */
      static fromConfig(cls, config) {
          return new cls(config['learningRate'], config['rho'], config['epsilon']);
      }
  }
  /** @nocollapse */
  AdadeltaOptimizer.className = 'Adadelta'; // Name matters for Python compatibility.
  registerClass(AdadeltaOptimizer);

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /** @doclink Optimizer */
  class AdagradOptimizer extends Optimizer {
      constructor(learningRate, initialAccumulatorValue = 0.1) {
          super();
          this.learningRate = learningRate;
          this.initialAccumulatorValue = initialAccumulatorValue;
          this.accumulatedGrads = [];
      }
      applyGradients(variableGradients) {
          const variableNames = Array.isArray(variableGradients) ?
              variableGradients.map(item => item.name) :
              Object.keys(variableGradients);
          variableNames.forEach((name, i) => {
              const value = ENGINE.registeredVariables[name];
              if (this.accumulatedGrads[i] == null) {
                  const trainable = false;
                  this.accumulatedGrads[i] = {
                      originalName: `${name}/accumulator`,
                      variable: tidy(() => fill(value.shape, this.initialAccumulatorValue)
                          .variable(trainable))
                  };
              }
              const gradient = Array.isArray(variableGradients) ?
                  variableGradients[i].tensor :
                  variableGradients[name];
              if (gradient == null) {
                  return;
              }
              const accumulatedGrad = this.accumulatedGrads[i].variable;
              tidy(() => {
                  const newAccumulatedGrad = accumulatedGrad.add(gradient.square());
                  accumulatedGrad.assign(newAccumulatedGrad);
                  const newValue = gradient
                      .div(newAccumulatedGrad.add(ENGINE.backend.epsilon()).sqrt())
                      .mul(-this.learningRate)
                      .add(value);
                  value.assign(newValue);
              });
          });
          this.incrementIterations();
      }
      dispose() {
          if (this.accumulatedGrads != null) {
              dispose(this.accumulatedGrads.map(v => v.variable));
          }
      }
      async getWeights() {
          // Order matters for Python compatibility.
          return [await this.saveIterations()].concat(this.accumulatedGrads.map(v => ({ name: v.originalName, tensor: v.variable })));
      }
      async setWeights(weightValues) {
          weightValues = await this.extractIterations(weightValues);
          const trainable = false;
          this.accumulatedGrads = weightValues.map(v => ({ originalName: v.name, variable: v.tensor.variable(trainable) }));
      }
      getConfig() {
          return {
              'learningRate': this.learningRate,
              'initialAccumulatorValue': this.initialAccumulatorValue,
          };
      }
      /** @nocollapse */
      static fromConfig(cls, config) {
          return new cls(config['learningRate'], config['initialAccumulatorValue']);
      }
  }
  /** @nocollapse */
  AdagradOptimizer.className = 'Adagrad'; // Note: Name matters for Python compatibility.
  registerClass(AdagradOptimizer);

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class AdamOptimizer extends Optimizer {
      constructor(learningRate, beta1, beta2, epsilon = null) {
          super();
          this.learningRate = learningRate;
          this.beta1 = beta1;
          this.beta2 = beta2;
          this.epsilon = epsilon;
          this.accumulatedFirstMoment = [];
          this.accumulatedSecondMoment = [];
          tidy(() => {
              // accB* will be updated by batch.
              this.accBeta1 = scalar(beta1).variable();
              this.accBeta2 = scalar(beta2).variable();
          });
          if (epsilon == null) {
              this.epsilon = ENGINE.backend.epsilon();
          }
      }
      applyGradients(variableGradients) {
          const varNames = Array.isArray(variableGradients) ?
              variableGradients.map(v => v.name) :
              Object.keys(variableGradients);
          tidy(() => {
              const oneMinusAccBeta1 = sub(1, this.accBeta1);
              const oneMinusAccBeta2 = sub(1, this.accBeta2);
              varNames.forEach((name, i) => {
                  const value = ENGINE.registeredVariables[name];
                  const trainable = false;
                  if (this.accumulatedFirstMoment[i] == null) {
                      this.accumulatedFirstMoment[i] = {
                          originalName: `${name}/m`,
                          variable: tidy(() => zerosLike(value).variable(trainable))
                      };
                  }
                  if (this.accumulatedSecondMoment[i] == null) {
                      this.accumulatedSecondMoment[i] = {
                          originalName: `${name}/v`,
                          variable: tidy(() => zerosLike(value).variable(trainable))
                      };
                  }
                  const gradient = Array.isArray(variableGradients) ?
                      variableGradients[i].tensor :
                      variableGradients[name];
                  if (gradient == null) {
                      return;
                  }
                  const firstMoment = this.accumulatedFirstMoment[i].variable;
                  const secondMoment = this.accumulatedSecondMoment[i].variable;
                  const newFirstMoment = firstMoment.mul(this.beta1).add(gradient.mul(1 - this.beta1));
                  const newSecondMoment = secondMoment.mul(this.beta2)
                      .add(gradient.square().mul(1 - this.beta2));
                  const biasCorrectedFirstMoment = newFirstMoment.div(oneMinusAccBeta1);
                  const biasCorrectedSecondMoment = newSecondMoment.div(oneMinusAccBeta2);
                  firstMoment.assign(newFirstMoment);
                  secondMoment.assign(newSecondMoment);
                  const newValue = biasCorrectedFirstMoment
                      .div(biasCorrectedSecondMoment.sqrt().add(this.epsilon))
                      .mul(-this.learningRate)
                      .add(value);
                  value.assign(newValue);
              });
              this.accBeta1.assign(this.accBeta1.mul(this.beta1));
              this.accBeta2.assign(this.accBeta2.mul(this.beta2));
          });
          this.incrementIterations();
      }
      dispose() {
          this.accBeta1.dispose();
          this.accBeta2.dispose();
          if (this.accumulatedFirstMoment != null) {
              dispose(this.accumulatedFirstMoment.map(v => v.variable));
          }
          if (this.accumulatedSecondMoment != null) {
              dispose(this.accumulatedSecondMoment.map(v => v.variable));
          }
      }
      async getWeights() {
          // Order matters for Python compatibility.
          const variables = [...this.accumulatedFirstMoment, ...this.accumulatedSecondMoment];
          return [await this.saveIterations()].concat(variables.map(v => ({ name: v.originalName, tensor: v.variable })));
      }
      async setWeights(weightValues) {
          weightValues = await this.extractIterations(weightValues);
          tidy(() => {
              this.accBeta1.assign(pow(this.beta1, this.iterations_ + 1));
              this.accBeta2.assign(pow(this.beta2, this.iterations_ + 1));
          });
          const variableCount = weightValues.length / 2;
          const trainable = false;
          this.accumulatedFirstMoment =
              weightValues.slice(0, variableCount).map(v => ({
                  originalName: v.name,
                  variable: v.tensor.variable(trainable)
              }));
          this.accumulatedSecondMoment =
              weightValues.slice(variableCount, variableCount * 2)
                  .map(v => ({
                  originalName: v.name,
                  variable: v.tensor.variable(trainable)
              }));
      }
      getConfig() {
          return {
              'learningRate': this.learningRate,
              'beta1': this.beta1,
              'beta2': this.beta2,
              'epsilon': this.epsilon,
          };
      }
      /** @nocollapse */
      static fromConfig(cls, config) {
          return new cls(config['learningRate'], config['beta1'], config['beta2'], config['epsilon']);
      }
  }
  /** @nocollapse */
  AdamOptimizer.className = 'Adam'; // Note: Name matters for Python compatibility.
  registerClass(AdamOptimizer);

  /**
  * @license
  * Copyright 2018 Google Inc. All Rights Reserved.
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
  class AdamaxOptimizer extends Optimizer {
      constructor(learningRate, beta1, beta2, epsilon = null, decay = 0.0) {
          super();
          this.learningRate = learningRate;
          this.beta1 = beta1;
          this.beta2 = beta2;
          this.epsilon = epsilon;
          this.decay = decay;
          this.accumulatedFirstMoment = [];
          this.accumulatedWeightedInfNorm = [];
          tidy(() => {
              this.iteration = scalar(0).variable();
              this.accBeta1 = scalar(beta1).variable();
          });
          if (epsilon == null) {
              this.epsilon = ENGINE.backend.epsilon();
          }
      }
      applyGradients(variableGradients) {
          const variableNames = Array.isArray(variableGradients) ?
              variableGradients.map(item => item.name) :
              Object.keys(variableGradients);
          tidy(() => {
              const oneMinusAccBeta1 = sub(1, this.accBeta1);
              const lr = div(-this.learningRate, this.iteration.mul(this.decay).add(1));
              variableNames.forEach((name, i) => {
                  const value = ENGINE.registeredVariables[name];
                  const trainable = false;
                  if (this.accumulatedFirstMoment[i] == null) {
                      this.accumulatedFirstMoment[i] = {
                          originalName: `${name}/m`,
                          variable: zerosLike(value).variable(trainable)
                      };
                  }
                  if (this.accumulatedWeightedInfNorm[i] == null) {
                      this.accumulatedWeightedInfNorm[i] = {
                          originalName: `${name}/v`,
                          variable: zerosLike(value).variable(trainable)
                      };
                  }
                  const gradient = Array.isArray(variableGradients) ?
                      variableGradients[i].tensor :
                      variableGradients[name];
                  if (gradient == null) {
                      return;
                  }
                  const firstMoment = this.accumulatedFirstMoment[i].variable;
                  const weightedInfNorm = this.accumulatedWeightedInfNorm[i].variable;
                  const newFirstMoment = firstMoment.mul(this.beta1).add(gradient.mul(1 - this.beta1));
                  const ut0 = weightedInfNorm.mul(this.beta2);
                  const ut1 = gradient.abs();
                  const newWeightedInfNorm = ut0.maximum(ut1);
                  firstMoment.assign(newFirstMoment);
                  weightedInfNorm.assign(newWeightedInfNorm);
                  const newValue = lr.div(oneMinusAccBeta1)
                      .mul(newFirstMoment.div(newWeightedInfNorm.add(this.epsilon)))
                      .add(value);
                  value.assign(newValue);
              });
              this.iteration.assign(this.iteration.add(1));
              this.accBeta1.assign(this.accBeta1.mul(this.beta1));
          });
          this.incrementIterations();
      }
      dispose() {
          this.accBeta1.dispose();
          this.iteration.dispose();
          if (this.accumulatedFirstMoment != null) {
              dispose(this.accumulatedFirstMoment.map(v => v.variable));
          }
          if (this.accumulatedWeightedInfNorm != null) {
              dispose(this.accumulatedWeightedInfNorm.map(v => v.variable));
          }
      }
      async getWeights() {
          throw new Error('getWeights() is not implemented for Adamax yet.');
      }
      async setWeights(weightValues) {
          throw new Error('setWeights() is not implemented for Adamax yet.');
      }
      getConfig() {
          return {
              'learningRate': this.learningRate,
              'beta1': this.beta1,
              'beta2': this.beta2,
              'epsilon': this.epsilon,
              'decay': this.decay
          };
      }
      /** @nocollapse */
      static fromConfig(cls, config) {
          return new cls(config['learningRate'], config['beta1'], config['beta2'], config['epsilon'], config['decay']);
      }
  }
  /** @nocollapse */
  AdamaxOptimizer.className = 'Adamax'; // Note: Name matters for Python compatbility.
  registerClass(AdamaxOptimizer);

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /** @doclink Optimizer */
  class SGDOptimizer extends Optimizer {
      constructor(learningRate) {
          super();
          this.learningRate = learningRate;
          this.setLearningRate(learningRate);
      }
      applyGradients(variableGradients) {
          const varNames = Array.isArray(variableGradients) ?
              variableGradients.map(v => v.name) :
              Object.keys(variableGradients);
          varNames.forEach((name, i) => {
              const gradient = Array.isArray(variableGradients) ?
                  variableGradients[i].tensor :
                  variableGradients[name];
              if (gradient == null) {
                  return;
              }
              const value = ENGINE.registeredVariables[name];
              tidy(() => {
                  const newValue = this.c.mul(gradient).add(value);
                  value.assign(newValue);
              });
          });
          this.incrementIterations();
      }
      /**
       * Sets the learning rate of the optimizer.
       */
      setLearningRate(learningRate) {
          this.learningRate = learningRate;
          if (this.c != null) {
              this.c.dispose();
          }
          this.c = keep(scalar(-learningRate));
      }
      dispose() {
          this.c.dispose();
      }
      async getWeights() {
          return [await this.saveIterations()];
      }
      async setWeights(weightValues) {
          weightValues = await this.extractIterations(weightValues);
          if (weightValues.length !== 0) {
              throw new Error('SGD optimizer does not have settable weights.');
          }
      }
      getConfig() {
          return { 'learningRate': this.learningRate };
      }
      /** @nocollapse */
      static fromConfig(cls, config) {
          return new cls(config['learningRate']);
      }
  }
  /** @nocollapse */
  SGDOptimizer.className = 'SGD'; // Note: Name matters for Python compatibility.
  registerClass(SGDOptimizer);

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /** @doclink Optimizer */
  class MomentumOptimizer extends SGDOptimizer {
      constructor(learningRate, momentum, useNesterov = false) {
          super(learningRate);
          this.learningRate = learningRate;
          this.momentum = momentum;
          this.useNesterov = useNesterov;
          this.accumulations = [];
          this.m = scalar(this.momentum);
      }
      applyGradients(variableGradients) {
          const variableNames = Array.isArray(variableGradients) ?
              variableGradients.map(item => item.name) :
              Object.keys(variableGradients);
          variableNames.forEach((name, i) => {
              const value = ENGINE.registeredVariables[name];
              if (this.accumulations[i] == null) {
                  const trainable = false;
                  this.accumulations[i] = {
                      originalName: `${name}/momentum`,
                      variable: tidy(() => zerosLike(value).variable(trainable))
                  };
              }
              const accumulation = this.accumulations[i].variable;
              const gradient = Array.isArray(variableGradients) ?
                  variableGradients[i].tensor :
                  variableGradients[name];
              if (gradient == null) {
                  return;
              }
              tidy(() => {
                  let newValue;
                  const newAccumulation = this.m.mul(accumulation).add(gradient);
                  if (this.useNesterov) {
                      newValue =
                          this.c.mul(gradient.add(newAccumulation.mul(this.m))).add(value);
                  }
                  else {
                      newValue = this.c.mul(newAccumulation).add(value);
                  }
                  accumulation.assign(newAccumulation);
                  value.assign(newValue);
              });
          });
          this.incrementIterations();
      }
      dispose() {
          this.m.dispose();
          if (this.accumulations != null) {
              dispose(this.accumulations.map(v => v.variable));
          }
      }
      /**
       * Sets the momentum of the optimizer.
       *
       * @param momentum
       */
      setMomentum(momentum) {
          this.momentum = momentum;
      }
      async getWeights() {
          // Order matters for Python compatibility.
          return [await this.saveIterations()].concat(this.accumulations.map(v => ({ name: v.originalName, tensor: v.variable })));
      }
      async setWeights(weightValues) {
          weightValues = await this.extractIterations(weightValues);
          const trainable = false;
          this.accumulations = weightValues.map(v => ({ originalName: v.name, variable: v.tensor.variable(trainable) }));
      }
      getConfig() {
          return {
              'learningRate': this.learningRate,
              'momentum': this.momentum,
              'useNesterov': this.useNesterov
          };
      }
      /** @nocollapse */
      static fromConfig(cls, config) {
          return new cls(config['learningRate'], config['momentum'], config['useNesterov']);
      }
  }
  /** @nocollapse */
  MomentumOptimizer.className = 'Momentum'; // Name matters for Python compatibility.
  registerClass(MomentumOptimizer);

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /** @doclink Optimizer */
  class RMSPropOptimizer extends Optimizer {
      constructor(learningRate, decay = 0.9, momentum = 0.0, epsilon = null, centered = false) {
          super();
          this.learningRate = learningRate;
          this.decay = decay;
          this.momentum = momentum;
          this.epsilon = epsilon;
          this.accumulatedMeanSquares = [];
          this.accumulatedMoments = [];
          this.accumulatedMeanGrads = [];
          this.centered = centered;
          if (epsilon == null) {
              this.epsilon = ENGINE.backend.epsilon();
          }
          if (learningRate == null) {
              throw new Error(`learningRate for RMSPropOptimizer must be defined.`);
          }
      }
      applyGradients(variableGradients) {
          const variableNames = Array.isArray(variableGradients) ?
              variableGradients.map(item => item.name) :
              Object.keys(variableGradients);
          variableNames.forEach((name, i) => {
              const value = ENGINE.registeredVariables[name];
              const trainable = false;
              if (this.accumulatedMeanSquares[i] == null) {
                  this.accumulatedMeanSquares[i] = {
                      originalName: `${name}/rms`,
                      variable: tidy(() => zerosLike(value).variable(trainable))
                  };
              }
              if (this.accumulatedMoments[i] == null) {
                  this.accumulatedMoments[i] = {
                      originalName: `${name}/momentum`,
                      variable: tidy(() => zerosLike(value).variable(trainable))
                  };
              }
              if (this.accumulatedMeanGrads[i] == null && this.centered) {
                  this.accumulatedMeanGrads[i] = {
                      originalName: `${name}/mg`,
                      variable: tidy(() => zerosLike(value).variable(trainable))
                  };
              }
              const gradient = Array.isArray(variableGradients) ?
                  variableGradients[i].tensor :
                  variableGradients[name];
              if (gradient == null) {
                  return;
              }
              const accumulatedMeanSquare = this.accumulatedMeanSquares[i].variable;
              const accumulatedMoments = this.accumulatedMoments[i].variable;
              tidy(() => {
                  const newAccumulatedMeanSquare = accumulatedMeanSquare.mul(this.decay)
                      .add(gradient.square().mul(1 - this.decay));
                  if (this.centered) {
                      const accumulatedMeanGrad = this.accumulatedMeanGrads[i].variable;
                      // Centered gradient
                      const newAccumulatedMeanGrad = accumulatedMeanGrad.mul(this.decay)
                          .add(gradient.mul(1 - this.decay));
                      const newAccumulatedMoments = accumulatedMoments.mul(this.momentum)
                          .add(gradient.mul(this.learningRate)
                          .div(newAccumulatedMeanSquare
                          .sub(newAccumulatedMeanGrad.square().add(this.epsilon))
                          .sqrt()));
                      accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
                      accumulatedMeanGrad.assign(newAccumulatedMeanGrad);
                      accumulatedMoments.assign(newAccumulatedMoments);
                      const newValue = value.sub(newAccumulatedMoments);
                      value.assign(newValue);
                  }
                  else {
                      // Plain gradient
                      const newAccumulatedMeanSquare = accumulatedMeanSquare.mul(this.decay)
                          .add(gradient.square().mul(1 - this.decay));
                      const newAccumulatedMoments = accumulatedMoments.mul(this.momentum)
                          .add(gradient.mul(this.learningRate)
                          .div(newAccumulatedMeanSquare.add(this.epsilon)
                          .sqrt()));
                      accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
                      accumulatedMoments.assign(newAccumulatedMoments);
                      const newValue = value.sub(newAccumulatedMoments);
                      value.assign(newValue);
                  }
              });
          });
          this.incrementIterations();
      }
      dispose() {
          if (this.accumulatedMeanSquares != null) {
              dispose(this.accumulatedMeanSquares.map(v => v.variable));
          }
          if (this.accumulatedMeanGrads != null && this.centered) {
              dispose(this.accumulatedMeanGrads.map(v => v.variable));
          }
          if (this.accumulatedMoments != null) {
              dispose(this.accumulatedMoments.map(v => v.variable));
          }
      }
      async getWeights() {
          // Order matters for Python compatibility.
          const variables = [...this.accumulatedMeanSquares, ...this.accumulatedMoments];
          if (this.centered) {
              variables.push(...this.accumulatedMeanGrads);
          }
          return [await this.saveIterations()].concat(variables.map(v => ({ name: v.originalName, tensor: v.variable })));
      }
      async setWeights(weightValues) {
          weightValues = await this.extractIterations(weightValues);
          const variableCount = this.centered ? weightValues.length / 3 : weightValues.length / 2;
          const trainable = false;
          this.accumulatedMeanSquares =
              weightValues.slice(0, variableCount).map(v => ({
                  originalName: v.name,
                  variable: v.tensor.variable(trainable)
              }));
          this.accumulatedMoments =
              weightValues.slice(variableCount, variableCount * 2)
                  .map(v => ({
                  originalName: v.name,
                  variable: v.tensor.variable(trainable)
              }));
          if (this.centered) {
              this.accumulatedMeanGrads =
                  weightValues.slice(variableCount * 2, variableCount * 3)
                      .map(v => ({
                      originalName: v.name,
                      variable: v.tensor.variable(trainable)
                  }));
          }
      }
      getConfig() {
          return {
              'learningRate': this.learningRate,
              'decay': this.decay,
              'momentum': this.momentum,
              'epsilon': this.epsilon,
              'centered': this.centered
          };
      }
      /** @nocollapse */
      static fromConfig(cls, config) {
          return new cls(config['learningRate'], config['decay'], config['momentum'], config['epsilon'], config['centered']);
      }
  }
  /** @nocollapse */
  RMSPropOptimizer.className = 'RMSProp'; // Note: Name matters for Python compatibility.
  registerClass(RMSPropOptimizer);

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class OptimizerConstructors {
      /**
       * Constructs a `tf.SGDOptimizer` that uses stochastic gradient descent.
       *
       * ```js
       * // Fit a quadratic function by learning the coefficients a, b, c.
       * const xs = tf.tensor1d([0, 1, 2, 3]);
       * const ys = tf.tensor1d([1.1, 5.9, 16.8, 33.9]);
       *
       * const a = tf.scalar(Math.random()).variable();
       * const b = tf.scalar(Math.random()).variable();
       * const c = tf.scalar(Math.random()).variable();
       *
       * // y = a * x^2 + b * x + c.
       * const f = x => a.mul(x.square()).add(b.mul(x)).add(c);
       * const loss = (pred, label) => pred.sub(label).square().mean();
       *
       * const learningRate = 0.01;
       * const optimizer = tf.train.sgd(learningRate);
       *
       * // Train the model.
       * for (let i = 0; i < 10; i++) {
       *   optimizer.minimize(() => loss(f(xs), ys));
       * }
       *
       * // Make predictions.
       * console.log(
       *     `a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);
       * const preds = f(xs).dataSync();
       * preds.forEach((pred, i) => {
       *   console.log(`x: ${i}, pred: ${pred}`);
       * });
       * ```
       *
       * @param learningRate The learning rate to use for the SGD algorithm.
       */
      /**
       * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
       */
      static sgd(learningRate) {
          return new SGDOptimizer(learningRate);
      }
      /**
       * Constructs a `tf.MomentumOptimizer` that uses momentum gradient
       * descent.
       *
       * See
       * [http://proceedings.mlr.press/v28/sutskever13.pdf](
       * http://proceedings.mlr.press/v28/sutskever13.pdf)
       *
       * @param learningRate The learning rate to use for the Momentum gradient
       * descent algorithm.
       * @param momentum The momentum to use for the momentum gradient descent
       * algorithm.
       */
      /**
       * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
       */
      static momentum(learningRate, momentum, useNesterov = false) {
          return new MomentumOptimizer(learningRate, momentum, useNesterov);
      }
      /**
       * Constructs a `tf.RMSPropOptimizer` that uses RMSProp gradient
       * descent. This implementation uses plain momentum and is not centered
       * version of RMSProp.
       *
       * See
       * [http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](
       * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
       *
       * @param learningRate The learning rate to use for the RMSProp gradient
       * descent algorithm.
       * @param decay The discounting factor for the history/coming gradient.
       * @param momentum The momentum to use for the RMSProp gradient descent
       * algorithm.
       * @param epsilon Small value to avoid zero denominator.
       * @param centered If true, gradients are normalized by the estimated
       * variance of the gradient.
       */
      /**
       * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
       */
      static rmsprop(learningRate, decay = .9, momentum = 0.0, epsilon = null, centered = false) {
          return new RMSPropOptimizer(learningRate, decay, momentum, epsilon, centered);
      }
      /**
       * Constructs a `tf.AdamOptimizer` that uses the Adam algorithm.
       * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
       *
       * @param learningRate The learning rate to use for the Adam gradient
       * descent algorithm.
       * @param beta1 The exponential decay rate for the 1st moment estimates.
       * @param beta2 The exponential decay rate for the 2nd moment estimates.
       * @param epsilon A small constant for numerical stability.
       */
      /**
       * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
       */
      static adam(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = null) {
          return new AdamOptimizer(learningRate, beta1, beta2, epsilon);
      }
      /**
       * Constructs a `tf.AdadeltaOptimizer` that uses the Adadelta algorithm.
       * See [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
       *
       * @param learningRate The learning rate to use for the Adadelta gradient
       * descent algorithm.
       * @param rho The learning rate decay over each update.
       * @param epsilon A constant epsilon used to better condition the grad
       * update.
       */
      /**
       * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
       */
      static adadelta(learningRate = .001, rho = .95, epsilon = null) {
          return new AdadeltaOptimizer(learningRate, rho, epsilon);
      }
      /**
       * Constructs a `tf.AdamaxOptimizer` that uses the Adamax algorithm.
       * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
       *
       * @param learningRate The learning rate to use for the Adamax gradient
       * descent algorithm.
       * @param beta1 The exponential decay rate for the 1st moment estimates.
       * @param beta2 The exponential decay rate for the 2nd moment estimates.
       * @param epsilon A small constant for numerical stability.
       * @param decay The learning rate decay over each update.
       */
      /**
       * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
       */
      static adamax(learningRate = 0.002, beta1 = 0.9, beta2 = 0.999, epsilon = null, decay = 0.0) {
          return new AdamaxOptimizer(learningRate, beta1, beta2, epsilon, decay);
      }
      /**
       * Constructs a `tf.AdagradOptimizer` that uses the Adagrad algorithm.
       * See
       * [http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](
       * http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
       * or
       * [http://ruder.io/optimizing-gradient-descent/index.html#adagrad](
       * http://ruder.io/optimizing-gradient-descent/index.html#adagrad)
       *
       * @param learningRate The learning rate to use for the Adagrad gradient
       * descent algorithm.
       * @param initialAccumulatorValue Starting value for the accumulators, must be
       * positive.
       */
      /**
       * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
       */
      static adagrad(learningRate, initialAccumulatorValue = 0.1) {
          return new AdagradOptimizer(learningRate, initialAccumulatorValue);
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  const train = {
      sgd: OptimizerConstructors.sgd,
      momentum: OptimizerConstructors.momentum,
      adadelta: OptimizerConstructors.adadelta,
      adagrad: OptimizerConstructors.adagrad,
      rmsprop: OptimizerConstructors.rmsprop,
      adamax: OptimizerConstructors.adamax,
      adam: OptimizerConstructors.adam
  };

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
  const delayCallback = (() => {
      if (typeof requestAnimationFrame !== 'undefined') {
          return requestAnimationFrame;
      }
      else if (typeof setImmediate !== 'undefined') {
          return setImmediate;
      }
      return (f) => f(); // no delays
  })();
  /**
   * Returns a promise that resolve when a requestAnimationFrame has completed.
   *
   * On Node.js this uses setImmediate instead of requestAnimationFrame.
   *
   * This is simply a sugar method so that users can do the following:
   * `await tf.nextFrame();`
   */
  /** @doc {heading: 'Performance', subheading: 'Timing'} */
  function nextFrame() {
      return new Promise(resolve => delayCallback(() => resolve()));
  }

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
  /**
   * Gets the new shape of the input Tensor after it's been reshaped
   * to:
   * [blockShape[0], ..., blockShape[M-1], batch / prod(blockShape),
   * inputShape[1], ..., inputShape[N-1]]
   *
   * See step 1: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
   */
  function getReshaped(inputShape, blockShape, prod, batchToSpace = true) {
      let reshaped = [];
      if (batchToSpace) {
          reshaped = reshaped.concat(blockShape.slice(0));
          reshaped.push(inputShape[0] / prod);
          reshaped = reshaped.concat(inputShape.slice(1));
      }
      else {
          reshaped = reshaped.concat(inputShape[0]);
          const spatialLength = blockShape.length;
          for (let i = 0; i < spatialLength; ++i) {
              reshaped =
                  reshaped.concat([inputShape[i + 1] / blockShape[i], blockShape[i]]);
          }
          reshaped = reshaped.concat(inputShape.slice(spatialLength + 1));
      }
      return reshaped;
  }
  /**
   * Gets the permutation that will transpose the dimensions of the
   * reshaped tensor to shape:
   *
   * [batch / prod(block_shape),inputShape[1], blockShape[0], ...,
   * inputShape[M], blockShape[M-1],inputShape[M+1], ..., inputShape[N-1]]
   *
   * see step 2: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
   */
  function getPermuted(reshapedRank, blockShapeRank, batchToSpace = true) {
      const permuted = [];
      if (batchToSpace) {
          permuted.push(blockShapeRank);
          for (let i = blockShapeRank + 1; i < reshapedRank; ++i) {
              if (i <= 2 * blockShapeRank) {
                  permuted.push(i);
                  permuted.push(i - (blockShapeRank + 1));
              }
              else {
                  permuted.push(i);
              }
          }
      }
      else {
          const permutedBeforeBatch = [];
          const permutedAfterBatch = [];
          for (let i = 1; i < reshapedRank; ++i) {
              if (i >= blockShapeRank * 2 + 1 || i % 2 === 1) {
                  permutedAfterBatch.push(i);
              }
              else {
                  permutedBeforeBatch.push(i);
              }
          }
          permuted.push(...permutedBeforeBatch);
          permuted.push(0);
          permuted.push(...permutedAfterBatch);
      }
      return permuted;
  }
  /**
   * Gets the shape of the reshaped and permuted input Tensor before any cropping
   * is applied.  The new shape will be:
   *
   * [batch / prod(blockShape),inputShape[1] * blockShape[0], ...,
   * inputShape[M] * blockShape[M-1],inputShape[M+1], ..., inputShape[N-1]]
   *
   * See step 3: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
   */
  function getReshapedPermuted(inputShape, blockShape, prod, batchToSpace = true) {
      const reshapedPermuted = [];
      if (batchToSpace) {
          reshapedPermuted.push(inputShape[0] / prod);
      }
      else {
          reshapedPermuted.push(inputShape[0] * prod);
      }
      for (let i = 1; i < inputShape.length; ++i) {
          if (i <= blockShape.length) {
              if (batchToSpace) {
                  reshapedPermuted.push(blockShape[i - 1] * inputShape[i]);
              }
              else {
                  reshapedPermuted.push(inputShape[i] / blockShape[i - 1]);
              }
          }
          else {
              reshapedPermuted.push(inputShape[i]);
          }
      }
      return reshapedPermuted;
  }
  /**
   * Converts the crops argument into the beginning coordinates of a slice
   * operation.
   */
  function getSliceBeginCoords(crops, blockShape) {
      const sliceBeginCoords = [0];
      for (let i = 0; i < blockShape; ++i) {
          sliceBeginCoords.push(crops[i][0]);
      }
      return sliceBeginCoords;
  }
  /**
   * Converts the crops argument into the size of a slice operation.  When
   * combined with getSliceBeginCoords this function allows the reshaped and
   * permuted Tensor to be cropped to its final output shape of:
   *
   * inputShape[1] * blockShape[0] - crops[0,0] - crops[0,1], ...,
   * inputShape[M] * blockShape[M-1] -crops[M-1,0] -
   * crops[M-1,1],inputShape[M+1], ..., inputShape[N-1]]
   *
   * See step 4: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
   */
  function getSliceSize(uncroppedShape, crops, blockShape) {
      const sliceSize = uncroppedShape.slice(0, 1);
      for (let i = 0; i < blockShape; ++i) {
          sliceSize.push(uncroppedShape[i + 1] - crops[i][0] - crops[i][1]);
      }
      return sliceSize;
  }

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
  const ERF_P = 0.3275911;
  const ERF_A1 = 0.254829592;
  const ERF_A2 = -0.284496736;
  const ERF_A3 = 1.421413741;
  const ERF_A4 = -1.453152027;
  const ERF_A5 = 1.061405429;

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
  function warn(...msg) {
      if (!env().getBool('IS_TEST')) {
          console.warn(...msg);
      }
  }
  function log$1(...msg) {
      if (!env().getBool('IS_TEST')) {
          console.log(...msg);
      }
  }

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
  /**
   * Merges real and imaginary Float32Arrays into a single complex Float32Array.
   *
   * The memory layout is interleaved as follows:
   * real: [r0, r1, r2]
   * imag: [i0, i1, i2]
   * complex: [r0, i0, r1, i1, r2, i2]
   *
   * This is the inverse of splitRealAndImagArrays.
   *
   * @param real The real values of the complex tensor values.
   * @param imag The imag values of the complex tensor values.
   * @returns A complex tensor as a Float32Array with merged values.
   */
  function mergeRealAndImagArrays(real, imag) {
      if (real.length !== imag.length) {
          throw new Error(`Cannot merge real and imag arrays of different lengths. real:` +
              `${real.length}, imag: ${imag.length}.`);
      }
      const result = new Float32Array(real.length * 2);
      for (let i = 0; i < result.length; i += 2) {
          result[i] = real[i / 2];
          result[i + 1] = imag[i / 2];
      }
      return result;
  }
  /**
   * Splits a complex Float32Array into real and imag parts.
   *
   * The memory layout is interleaved as follows:
   * complex: [r0, i0, r1, i1, r2, i2]
   * real: [r0, r1, r2]
   * imag: [i0, i1, i2]
   *
   * This is the inverse of mergeRealAndImagArrays.
   *
   * @param complex The complex tensor values.
   * @returns An object with real and imag Float32Array components of the complex
   *     tensor.
   */
  function splitRealAndImagArrays(complex) {
      const real = new Float32Array(complex.length / 2);
      const imag = new Float32Array(complex.length / 2);
      for (let i = 0; i < complex.length; i += 2) {
          real[i / 2] = complex[i];
          imag[i / 2] = complex[i + 1];
      }
      return { real, imag };
  }
  /**
   * Extracts even indexed complex values in the given array.
   * @param complex The complex tensor values
   */
  function complexWithEvenIndex(complex) {
      const len = Math.ceil(complex.length / 4);
      const real = new Float32Array(len);
      const imag = new Float32Array(len);
      for (let i = 0; i < complex.length; i += 4) {
          real[Math.floor(i / 4)] = complex[i];
          imag[Math.floor(i / 4)] = complex[i + 1];
      }
      return { real, imag };
  }
  /**
   * Extracts odd indexed comple values in the given array.
   * @param complex The complex tensor values
   */
  function complexWithOddIndex(complex) {
      const len = Math.floor(complex.length / 4);
      const real = new Float32Array(len);
      const imag = new Float32Array(len);
      for (let i = 2; i < complex.length; i += 4) {
          real[Math.floor(i / 4)] = complex[i];
          imag[Math.floor(i / 4)] = complex[i + 1];
      }
      return { real, imag };
  }
  /**
   * Get the map representing a complex value in the given array.
   * @param complex The complex tensor values.
   * @param index An index of the target complex value.
   */
  function getComplexWithIndex(complex, index) {
      const real = complex[index * 2];
      const imag = complex[index * 2 + 1];
      return { real, imag };
  }
  /**
   * Insert a given complex value into the TypedArray.
   * @param data The array in which the complex value is inserted.
   * @param c The complex value to be inserted.
   * @param index An index of the target complex value.
   */
  function assignToTypedArray(data, real, imag, index) {
      data[index * 2] = real;
      data[index * 2 + 1] = imag;
  }
  /**
   * Make the list of exponent terms used by FFT.
   */
  function exponents(n, inverse) {
      const real = new Float32Array(n / 2);
      const imag = new Float32Array(n / 2);
      for (let i = 0; i < Math.ceil(n / 2); i++) {
          const x = (inverse ? 2 : -2) * Math.PI * (i / n);
          real[i] = Math.cos(x);
          imag[i] = Math.sin(x);
      }
      return { real, imag };
  }
  /**
   * Make the exponent term used by FFT.
   */
  function exponent(k, n, inverse) {
      const x = (inverse ? 2 : -2) * Math.PI * (k / n);
      const real = Math.cos(x);
      const imag = Math.sin(x);
      return { real, imag };
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  function castTensor(x, dtype, backend) {
      if (dtype === 'complex64') {
          if (x.dtype === 'complex64') {
              return x.clone();
          }
          const zerosTensor = zeros(x.shape);
          const floatX = x.toFloat();
          const result = backend.complex(floatX, zerosTensor);
          zerosTensor.dispose();
          floatX.dispose();
          return result;
      }
      if (!hasEncodingLoss(x.dtype, dtype)) {
          // We don't change the underlying data, since we cast to higher
          // precision.
          return ENGINE.makeTensorFromDataId(x.dataId, x.shape, dtype);
      }
      if (x.dtype === 'complex64') {
          const real = backend.real(x);
          const result = real.cast(dtype);
          real.dispose();
          return result;
      }
      if (dtype === 'int32') {
          return backend.int(x);
      }
      else if (dtype === 'bool') {
          const zero = scalar(0, x.dtype);
          const result = backend.notEqual(x, zero);
          zero.dispose();
          return result;
      }
      else {
          throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
      }
  }
  function reshapeTensor(x, shape) {
      return ENGINE.makeTensorFromDataId(x.dataId, shape, x.dtype);
  }
  function linspaceImpl(start, stop, num) {
      const step = (stop - start) / (num - 1);
      const values = makeZerosTypedArray(num, 'float32');
      values[0] = start;
      for (let i = 1; i < values.length; i++) {
          values[i] = values[i - 1] + step;
      }
      return tensor1d(values, 'float32');
  }

  var backend_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    segment_util: segment_util,
    castTensor: castTensor,
    reshapeTensor: reshapeTensor,
    linspaceImpl: linspaceImpl,
    upcastType: upcastType,
    axesAreInnerMostDims: axesAreInnerMostDims,
    combineLocations: combineLocations,
    computeOutAndReduceShapes: computeOutAndReduceShapes,
    expandShapeToKeepDim: expandShapeToKeepDim,
    assertAxesAreInnerMostDims: assertAxesAreInnerMostDims,
    getAxesPermutation: getAxesPermutation,
    getUndoAxesPermutation: getUndoAxesPermutation,
    getInnerMostAxes: getInnerMostAxes,
    getBroadcastDims: getBroadcastDims,
    getReductionAxes: getReductionAxes,
    assertAndGetBroadcastShape: assertAndGetBroadcastShape,
    assertParamsConsistent: assertParamsConsistent,
    computeOutShape: computeOutShape,
    computePool2DInfo: computePool2DInfo,
    computePool3DInfo: computePool3DInfo,
    computeConv2DInfo: computeConv2DInfo,
    computeConv3DInfo: computeConv3DInfo,
    computeDefaultPad: computeDefaultPad,
    tupleValuesAreOne: tupleValuesAreOne,
    eitherStridesOrDilationsAreOne: eitherStridesOrDilationsAreOne,
    convertConv2DDataFormat: convertConv2DDataFormat,
    PARALLELIZE_THRESHOLD: PARALLELIZE_THRESHOLD,
    computeOptimalWindowSize: computeOptimalWindowSize,
    getReshaped: getReshaped,
    getPermuted: getPermuted,
    getReshapedPermuted: getReshapedPermuted,
    getSliceBeginCoords: getSliceBeginCoords,
    getSliceSize: getSliceSize,
    prepareAndValidate: prepareAndValidate,
    validateUpdateShape: validateUpdateShape,
    validateInput: validateInput,
    calculateShapes: calculateShapes,
    SELU_SCALEALPHA: SELU_SCALEALPHA,
    SELU_SCALE: SELU_SCALE,
    shouldFuse: shouldFuse,
    ERF_P: ERF_P,
    ERF_A1: ERF_A1,
    ERF_A2: ERF_A2,
    ERF_A3: ERF_A3,
    ERF_A4: ERF_A4,
    ERF_A5: ERF_A5,
    warn: warn,
    log: log$1,
    mergeRealAndImagArrays: mergeRealAndImagArrays,
    splitRealAndImagArrays: splitRealAndImagArrays,
    complexWithEvenIndex: complexWithEvenIndex,
    complexWithOddIndex: complexWithOddIndex,
    getComplexWithIndex: getComplexWithIndex,
    assignToTypedArray: assignToTypedArray,
    exponents: exponents,
    exponent: exponent
  });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /** Shared implementation of the split kernel across WebGL and CPU. */
  function split$1(x, sizeSplits, axis) {
      const begin = new Array(x.rank).fill(0);
      const size = x.shape.slice();
      return sizeSplits.map(s => {
          size[axis] = s;
          const slice = x.slice(begin, size);
          begin[axis] += s;
          return slice;
      });
  }

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
  function tile$1(xBuf, reps) {
      const newShape = new Array(xBuf.rank);
      for (let i = 0; i < newShape.length; i++) {
          newShape[i] = xBuf.shape[i] * reps[i];
      }
      const result = buffer(newShape, xBuf.dtype);
      for (let i = 0; i < result.values.length; ++i) {
          const newLoc = result.indexToLoc(i);
          const originalLoc = new Array(xBuf.rank);
          for (let j = 0; j < originalLoc.length; j++) {
              originalLoc[j] = newLoc[j] % xBuf.shape[j];
          }
          const originalIndex = xBuf.locToIndex(originalLoc);
          result.values[i] = xBuf.values[originalIndex];
      }
      return result.toTensor();
  }

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
  function topkImpl(x, xShape, xDtype, k, sorted) {
      // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
      const lastDim = xShape[xShape.length - 1];
      const [batch, size] = [x.length / lastDim, lastDim];
      const allTopKVals = getTypedArrayFromDType(xDtype, batch * k);
      const allTopKIndices = getTypedArrayFromDType('int32', batch * k);
      for (let b = 0; b < batch; b++) {
          const offset = b * size;
          const vals = x.subarray(offset, offset + size);
          const valAndInd = [];
          for (let i = 0; i < vals.length; i++) {
              valAndInd.push({ value: vals[i], index: i });
          }
          valAndInd.sort((a, b) => b.value - a.value);
          const outOffset = b * k;
          const topKVals = allTopKVals.subarray(outOffset, outOffset + k);
          const topKIndices = allTopKIndices.subarray(outOffset, outOffset + k);
          for (let i = 0; i < k; i++) {
              topKVals[i] = valAndInd[i].value;
              topKIndices[i] = valAndInd[i].index;
          }
      }
      // Reshape back to the original input shape, except that the last
      // dimension is k.
      const outputShape = xShape.slice();
      outputShape[outputShape.length - 1] = k;
      return [
          tensor(allTopKVals, outputShape, xDtype),
          tensor(allTopKIndices, outputShape, 'int32')
      ];
  }

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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

  var kernel_impls = /*#__PURE__*/Object.freeze({
    __proto__: null,
    nonMaxSuppressionV3: nonMaxSuppressionV3,
    nonMaxSuppressionV5: nonMaxSuppressionV5,
    split: split$1,
    tile: tile$1,
    topkImpl: topkImpl,
    whereImpl: whereImpl
  });

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  const EPSILON_FLOAT32 = 1e-7;
  const EPSILON_FLOAT16 = 1e-4;
  /** Convenient class for storing tensor-related data. */
  class DataStorage {
      constructor(backend, dataMover) {
          this.backend = backend;
          this.dataMover = dataMover;
          this.data = new WeakMap();
          this.dataIdsCount = 0;
      }
      get(dataId) {
          if (!this.data.has(dataId)) {
              this.dataMover.moveData(this.backend, dataId);
          }
          return this.data.get(dataId);
      }
      set(dataId, value) {
          this.dataIdsCount++;
          this.data.set(dataId, value);
      }
      has(dataId) {
          return this.data.has(dataId);
      }
      delete(dataId) {
          this.dataIdsCount--;
          return this.data.delete(dataId);
      }
      numDataIds() {
          return this.dataIdsCount;
      }
  }
  /**
   * The interface that defines the kernels that should be implemented when
   * adding a new backend. New backends don't need to implement every one of the
   * methods, this can be done gradually (throw an error for unimplemented
   * methods).
   */
  class KernelBackend {
      time(f) {
          return notYetImplemented('time');
      }
      read(dataId) {
          return notYetImplemented('read');
      }
      readSync(dataId) {
          return notYetImplemented('readSync');
      }
      numDataIds() {
          return notYetImplemented('numDataIds');
      }
      disposeData(dataId) {
          return notYetImplemented('disposeData');
      }
      write(values, shape, dtype) {
          return notYetImplemented('write');
      }
      move(dataId, values, shape, dtype) {
          return notYetImplemented('move');
      }
      memory() {
          return notYetImplemented('memory');
      }
      /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
      floatPrecision() {
          return notYetImplemented('floatPrecision');
      }
      /** Returns the smallest representable number.  */
      epsilon() {
          return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
      }
      batchMatMul(a, b, transposeA, transposeB) {
          return notYetImplemented('batchMatMul');
      }
      fusedBatchMatMul({ a, b, transposeA, transposeB, bias, activation, preluActivationWeights }) {
          return notYetImplemented('fusedBatchMatMul');
      }
      slice(x, begin, size) {
          return notYetImplemented('slice');
      }
      stridedSlice(x, begin, end, strides) {
          return notYetImplemented('stridedSlice');
      }
      unstack(x, axis) {
          return notYetImplemented('unstack');
      }
      reverse(a, axis) {
          return notYetImplemented('reverse');
      }
      concat(tensors, axis) {
          return notYetImplemented('concat');
      }
      neg(a) {
          return notYetImplemented('neg');
      }
      add(a, b) {
          return notYetImplemented('add');
      }
      addN(tensors) {
          return notYetImplemented('addN');
      }
      subtract(a, b) {
          return notYetImplemented('subtract');
      }
      multiply(a, b) {
          return notYetImplemented('multiply');
      }
      realDivide(a, b) {
          return notYetImplemented('realDivide');
      }
      floorDiv(a, b) {
          return notYetImplemented('floorDiv');
      }
      sum(x, axes) {
          return notYetImplemented('sum');
      }
      prod(x, axes) {
          return notYetImplemented('prod');
      }
      unsortedSegmentSum(x, segmentIds, numSegments) {
          return notYetImplemented('unsortedSegmentSum');
      }
      argMin(x, axis) {
          return notYetImplemented('argMin');
      }
      argMax(x, axis) {
          return notYetImplemented('argMax');
      }
      equal(a, b) {
          return notYetImplemented('equal');
      }
      notEqual(a, b) {
          return notYetImplemented('notEqual');
      }
      less(a, b) {
          return notYetImplemented('less');
      }
      lessEqual(a, b) {
          return notYetImplemented('lessEqual');
      }
      greater(a, b) {
          return notYetImplemented('greater');
      }
      greaterEqual(a, b) {
          return notYetImplemented('greaterEqual');
      }
      logicalNot(a) {
          return notYetImplemented('logicalNot');
      }
      logicalAnd(a, b) {
          return notYetImplemented('logicalAnd');
      }
      logicalOr(a, b) {
          return notYetImplemented('logicalOr');
      }
      where(condition) {
          return notYetImplemented('where');
      }
      select(condition, a, b) {
          return notYetImplemented('select');
      }
      topk(x, k, sorted) {
          return notYetImplemented('topk');
      }
      min(x, axes) {
          return notYetImplemented('min');
      }
      minimum(a, b) {
          return notYetImplemented('minimum');
      }
      mod(a, b) {
          return notYetImplemented('mod');
      }
      max(x, axes) {
          return notYetImplemented('max');
      }
      maximum(a, b) {
          return notYetImplemented('maximum');
      }
      all(x, axes) {
          return notYetImplemented('all');
      }
      any(x, axes) {
          return notYetImplemented('any');
      }
      squaredDifference(a, b) {
          return notYetImplemented('squaredDifference');
      }
      ceil(x) {
          return notYetImplemented('ceil');
      }
      floor(x) {
          return notYetImplemented('floor');
      }
      round(x) {
          return notYetImplemented('round');
      }
      sign(x) {
          return notYetImplemented('sign');
      }
      isNaN(x) {
          return notYetImplemented('isNaN');
      }
      isInf(x) {
          return notYetImplemented('isInf');
      }
      isFinite(x) {
          return notYetImplemented('isFinite');
      }
      pow(a, b) {
          return notYetImplemented('pow');
      }
      exp(x) {
          return notYetImplemented('exp');
      }
      expm1(x) {
          return notYetImplemented('expm1');
      }
      softmax(x, dim) {
          return notYetImplemented('softmax');
      }
      log(x) {
          return notYetImplemented('log');
      }
      log1p(x) {
          return notYetImplemented('log1p');
      }
      sqrt(x) {
          return notYetImplemented('sqrt');
      }
      rsqrt(x) {
          return notYetImplemented('rsqrt');
      }
      square(x) {
          return notYetImplemented('square');
      }
      reciprocal(x) {
          return notYetImplemented('reciprocal');
      }
      relu(x) {
          return notYetImplemented('relu');
      }
      relu6(x) {
          return notYetImplemented('relu6');
      }
      prelu(x, a) {
          return notYetImplemented('prelu');
      }
      elu(x) {
          return notYetImplemented('elu');
      }
      eluDer(dy, y) {
          return notYetImplemented('eluDer');
      }
      selu(x) {
          return notYetImplemented('selu');
      }
      int(x) {
          return notYetImplemented('int');
      }
      clip(x, min, max) {
          return notYetImplemented('clip');
      }
      abs(x) {
          return notYetImplemented('abs');
      }
      complexAbs(x) {
          return notYetImplemented('complexAbs');
      }
      sigmoid(x) {
          return notYetImplemented('sigmoid');
      }
      softplus(x) {
          return notYetImplemented('softplus');
      }
      sin(x) {
          return notYetImplemented('sin');
      }
      cos(x) {
          return notYetImplemented('cos');
      }
      tan(x) {
          return notYetImplemented('tan');
      }
      asin(x) {
          return notYetImplemented('asin');
      }
      acos(x) {
          return notYetImplemented('acos');
      }
      atan(x) {
          return notYetImplemented('atan');
      }
      atan2(a, b) {
          return notYetImplemented('atan2');
      }
      sinh(x) {
          return notYetImplemented('sinh');
      }
      cosh(x) {
          return notYetImplemented('cosh');
      }
      tanh(x) {
          return notYetImplemented('tanh');
      }
      asinh(x) {
          return notYetImplemented('asinh');
      }
      acosh(x) {
          return notYetImplemented('acosh');
      }
      atanh(x) {
          return notYetImplemented('atanh');
      }
      erf(x) {
          return notYetImplemented('erf');
      }
      step(x, alpha) {
          return notYetImplemented('step');
      }
      fusedConv2d({ input, filter, convInfo, bias, activation, preluActivationWeights }) {
          return notYetImplemented('fusedConv2d');
      }
      conv2d(x, filter, convInfo) {
          return notYetImplemented('conv2d');
      }
      conv2dDerInput(dy, filter, convInfo) {
          return notYetImplemented('conv2dDerInput');
      }
      conv2dDerFilter(x, dY, convInfo) {
          return notYetImplemented('conv2dDerFilter');
      }
      fusedDepthwiseConv2D({ input, filter, convInfo, bias, activation, preluActivationWeights }) {
          return notYetImplemented('fusedDepthwiseConv2D');
      }
      depthwiseConv2D(input, filter, convInfo) {
          return notYetImplemented('depthwiseConv2D');
      }
      depthwiseConv2DDerInput(dy, filter, convInfo) {
          return notYetImplemented('depthwiseConv2DDerInput');
      }
      depthwiseConv2DDerFilter(x, dY, convInfo) {
          return notYetImplemented('depthwiseConv2DDerFilter');
      }
      conv3d(x, filter, convInfo) {
          return notYetImplemented('conv3d');
      }
      conv3dDerInput(dy, filter, convInfo) {
          return notYetImplemented('conv3dDerInput');
      }
      conv3dDerFilter(x, dY, convInfo) {
          return notYetImplemented('conv3dDerFilter');
      }
      maxPool(x, convInfo) {
          return notYetImplemented('maxPool');
      }
      maxPoolBackprop(dy, x, y, convInfo) {
          return notYetImplemented('maxPoolBackprop');
      }
      avgPool(x, convInfo) {
          return notYetImplemented('avgPool');
      }
      avgPoolBackprop(dy, x, convInfo) {
          return notYetImplemented('avgPoolBackprop');
      }
      avgPool3d(x, convInfo) {
          return notYetImplemented('avgPool3d');
      }
      avgPool3dBackprop(dy, x, convInfo) {
          return notYetImplemented('avgPool3dBackprop');
      }
      maxPool3d(x, convInfo) {
          return notYetImplemented('maxPool3d');
      }
      maxPool3dBackprop(dy, x, y, convInfo) {
          return notYetImplemented('maxPool3dBackprop');
      }
      reshape(x, shape) {
          return notYetImplemented('reshape');
      }
      cast(x, dtype) {
          return notYetImplemented('cast');
      }
      tile(x, reps) {
          return notYetImplemented('tile');
      }
      pad(x, paddings, constantValue) {
          return notYetImplemented('pad');
      }
      transpose(x, perm) {
          return notYetImplemented('transpose');
      }
      gather(x, indices, axis) {
          return notYetImplemented('gather');
      }
      gatherND(x, indices) {
          return notYetImplemented('gatherND');
      }
      scatterND(indices, updates, shape) {
          return notYetImplemented('scatterND');
      }
      batchToSpaceND(x, blockShape, crops) {
          return notYetImplemented('batchToSpaceND');
      }
      spaceToBatchND(x, blockShape, paddings) {
          return notYetImplemented('spaceToBatchND');
      }
      resizeBilinear(x, newHeight, newWidth, alignCorners) {
          return notYetImplemented('resizeBilinear');
      }
      resizeBilinearBackprop(dy, x, alignCorners) {
          return notYetImplemented('resizeBilinearBackprop');
      }
      resizeNearestNeighbor(x, newHEight, newWidth, alignCorners) {
          return notYetImplemented('resizeNearestNeighbor');
      }
      resizeNearestNeighborBackprop(dy, x, alignCorners) {
          return notYetImplemented('resizeNearestNeighborBackprop');
      }
      batchNormalization(x, mean, variance, varianceEpsilon, scale, offset) {
          return notYetImplemented('batchNormalization');
      }
      localResponseNormalization4D(x, radius, bias, alpha, beta) {
          return notYetImplemented('localResponseNormalization4D');
      }
      LRNGrad(dy, inputImage, outputImage, radius, bias, alpha, beta) {
          return notYetImplemented('LRNGrad');
      }
      multinomial(logits, normalized, numSamples, seed) {
          return notYetImplemented('multinomial');
      }
      oneHot(indices, depth, onValue, offValue) {
          return notYetImplemented('oneHot');
      }
      cumsum(x, axis, exclusive, reverse) {
          return notYetImplemented('cumsum');
      }
      nonMaxSuppression(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
          return notYetImplemented('nonMaxSuppression');
      }
      fft(x) {
          return notYetImplemented('fft');
      }
      ifft(x) {
          return notYetImplemented('ifft');
      }
      complex(real, imag) {
          return notYetImplemented('complex');
      }
      real(input) {
          return notYetImplemented('real');
      }
      imag(input) {
          return notYetImplemented('imag');
      }
      cropAndResize(image, boxes, boxIndex, cropSize, method, extrapolationValue) {
          return notYetImplemented('cropAndResize');
      }
      depthToSpace(x, blockSize, dataFormat) {
          return notYetImplemented('depthToSpace');
      }
      // Aligns with the "SplitV" kernel in TensorFlow.
      split(value, sizeSplits, axis) {
          return notYetImplemented('split');
      }
      sparseToDense(sparseIndices, sparseValues, outputShape, defaultValue) {
          return notYetImplemented('sparseToDense');
      }
      diag(x) {
          return notYetImplemented('diag');
      }
      fill(shape, value, dtype) {
          return notYetImplemented('fill');
      }
      onesLike(x) {
          return notYetImplemented('onesLike');
      }
      zerosLike(x) {
          return notYetImplemented('zerosLike');
      }
      linspace(start, stop, num) {
          return notYetImplemented('linspace');
      }
      dispose() {
          return notYetImplemented('dispose');
      }
  }
  function notYetImplemented(kernelName) {
      throw new Error(`'${kernelName}' not yet implemented or not found in the registry. ` +
          `Did you forget to import the kernel?`);
  }

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.add = function (b) {
      this.throwIfDisposed();
      return add(this, b);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.batchNorm = function (mean, variance, offset, scale, varianceEpsilon) {
      this.throwIfDisposed();
      return batchNorm(this, mean, variance, offset, scale, varianceEpsilon);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.broadcastTo = function (shape) {
      this.throwIfDisposed();
      return broadcastTo(this, shape);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.div = function (b) {
      this.throwIfDisposed();
      return div(this, b);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.divNoNan = function (b) {
      this.throwIfDisposed();
      return divNoNan(this, b);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.oneHot = function (depth, onValue = 1, offValue = 0) {
      this.throwIfDisposed();
      return oneHot(this, depth, onValue, offValue);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.notEqual = function (b) {
      this.throwIfDisposed();
      return notEqual(this, b);
  };

  /**
   * @license
   * Copyright 2020 Google Inc. All Rights Reserved.
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
  Tensor.prototype.pad = function (paddings, constantValue) {
      this.throwIfDisposed();
      return pad(this, paddings, constantValue);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.squaredDifference = function (b) {
      this.throwIfDisposed();
      return squaredDifference(this, b);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.sub = function (b) {
      this.throwIfDisposed();
      return sub(this, b);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.tile = function (reps) {
      this.throwIfDisposed();
      return tile(this, reps);
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  Tensor.prototype.transpose = function (perm) {
      this.throwIfDisposed();
      return transpose(this, perm);
  };

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
  setOpHandler(ops);

  exports.AdadeltaOptimizer = AdadeltaOptimizer;
  exports.AdagradOptimizer = AdagradOptimizer;
  exports.AdamOptimizer = AdamOptimizer;
  exports.AdamaxOptimizer = AdamaxOptimizer;
  exports.Add = Add;
  exports.AddN = AddN;
  exports.BroadcastTo = BroadcastTo;
  exports.DataStorage = DataStorage;
  exports.Div = Div;
  exports.Environment = Environment;
  exports.FromPixels = FromPixels;
  exports.FusedBatchNorm = FusedBatchNorm;
  exports.Identity = Identity;
  exports.KernelBackend = KernelBackend;
  exports.MaxPoolWithArgmax = MaxPoolWithArgmax;
  exports.MomentumOptimizer = MomentumOptimizer;
  exports.NonMaxSuppressionV5 = NonMaxSuppressionV5;
  exports.NotEqual = NotEqual;
  exports.OneHot = OneHot;
  exports.Optimizer = Optimizer;
  exports.PadV2 = PadV2;
  exports.RMSPropOptimizer = RMSPropOptimizer;
  exports.SGDOptimizer = SGDOptimizer;
  exports.Square = Square;
  exports.SquaredDifference = SquaredDifference;
  exports.Sub = Sub;
  exports.Tensor = Tensor;
  exports.TensorBuffer = TensorBuffer;
  exports.Tile = Tile;
  exports.Transpose = Transpose;
  exports.Variable = Variable;
  exports.abs = abs;
  exports.acos = acos;
  exports.acosh = acosh;
  exports.add = add;
  exports.addN = addN;
  exports.addStrict = addStrict;
  exports.all = all;
  exports.any = any;
  exports.argMax = argMax;
  exports.argMin = argMin;
  exports.asin = asin;
  exports.asinh = asinh;
  exports.atan = atan;
  exports.atan2 = atan2;
  exports.atanh = atanh;
  exports.avgPool = avgPool;
  exports.avgPool3d = avgPool3d;
  exports.backend = backend;
  exports.backend_util = backend_util;
  exports.basicLSTMCell = basicLSTMCell;
  exports.batchNorm = batchNorm;
  exports.batchNorm2d = batchNorm2d;
  exports.batchNorm3d = batchNorm3d;
  exports.batchNorm4d = batchNorm4d;
  exports.batchNormalization = batchNormalization;
  exports.batchNormalization2d = batchNormalization2d;
  exports.batchNormalization3d = batchNormalization3d;
  exports.batchNormalization4d = batchNormalization4d;
  exports.batchToSpaceND = batchToSpaceND;
  exports.booleanMaskAsync = booleanMaskAsync;
  exports.broadcastTo = broadcastTo;
  exports.browser = browser;
  exports.buffer = buffer;
  exports.cast = cast;
  exports.ceil = ceil;
  exports.clipByValue = clipByValue;
  exports.clone = clone;
  exports.complex = complex;
  exports.concat = concat;
  exports.concat1d = concat1d;
  exports.concat2d = concat2d;
  exports.concat3d = concat3d;
  exports.concat4d = concat4d;
  exports.conv1d = conv1d;
  exports.conv2d = conv2d;
  exports.conv2dTranspose = conv2dTranspose;
  exports.conv3d = conv3d;
  exports.conv3dTranspose = conv3dTranspose;
  exports.cos = cos;
  exports.cosh = cosh;
  exports.cumsum = cumsum;
  exports.customGrad = customGrad;
  exports.deprecationWarn = deprecationWarn;
  exports.depthToSpace = depthToSpace;
  exports.depthwiseConv2d = depthwiseConv2d;
  exports.device_util = device_util;
  exports.diag = diag;
  exports.disableDeprecationWarnings = disableDeprecationWarnings;
  exports.dispose = dispose;
  exports.disposeVariables = disposeVariables;
  exports.div = div;
  exports.divNoNan = divNoNan;
  exports.divStrict = divStrict;
  exports.dot = dot;
  exports.dropout = dropout;
  exports.elu = elu;
  exports.enableDebugMode = enableDebugMode;
  exports.enableProdMode = enableProdMode;
  exports.engine = engine;
  exports.env = env;
  exports.equal = equal;
  exports.equalStrict = equalStrict;
  exports.erf = erf;
  exports.exp = exp;
  exports.expandDims = expandDims;
  exports.expm1 = expm1;
  exports.eye = eye;
  exports.fft = fft;
  exports.fill = fill;
  exports.findBackend = findBackend;
  exports.findBackendFactory = findBackendFactory;
  exports.floor = floor;
  exports.floorDiv = floorDiv;
  exports.frame = frame;
  exports.fused = fused_ops;
  exports.gather = gather;
  exports.gatherND = gatherND;
  exports.gather_util = gather_nd_util;
  exports.getBackend = getBackend;
  exports.getGradient = getGradient;
  exports.getKernel = getKernel;
  exports.getKernelsForBackend = getKernelsForBackend;
  exports.grad = grad;
  exports.grads = grads;
  exports.greater = greater;
  exports.greaterEqual = greaterEqual;
  exports.greaterEqualStrict = greaterEqualStrict;
  exports.greaterStrict = greaterStrict;
  exports.hammingWindow = hammingWindow;
  exports.hannWindow = hannWindow;
  exports.ifft = ifft;
  exports.imag = imag;
  exports.image = image_ops;
  exports.inTopKAsync = inTopKAsync;
  exports.io = io;
  exports.irfft = irfft;
  exports.isFinite = isFinite$1;
  exports.isInf = isInf;
  exports.isNaN = isNaN$1;
  exports.keep = keep;
  exports.kernel_impls = kernel_impls;
  exports.leakyRelu = leakyRelu;
  exports.less = less;
  exports.lessEqual = lessEqual;
  exports.lessEqualStrict = lessEqualStrict;
  exports.lessStrict = lessStrict;
  exports.linalg = linalg_ops;
  exports.linspace = linspace;
  exports.localResponseNormalization = localResponseNormalization;
  exports.log = log;
  exports.log1p = log1p;
  exports.logSigmoid = logSigmoid;
  exports.logSoftmax = logSoftmax;
  exports.logSumExp = logSumExp;
  exports.logicalAnd = logicalAnd;
  exports.logicalNot = logicalNot;
  exports.logicalOr = logicalOr;
  exports.logicalXor = logicalXor;
  exports.losses = loss_ops;
  exports.matMul = matMul;
  exports.math = math;
  exports.max = max;
  exports.maxPool = maxPool;
  exports.maxPool3d = maxPool3d;
  exports.maxPoolWithArgmax = maxPoolWithArgmax;
  exports.maximum = maximum;
  exports.maximumStrict = maximumStrict;
  exports.mean = mean;
  exports.memory = memory;
  exports.min = min;
  exports.minimum = minimum;
  exports.minimumStrict = minimumStrict;
  exports.mod = mod;
  exports.modStrict = modStrict;
  exports.moments = moments;
  exports.movingAverage = movingAverage;
  exports.mul = mul;
  exports.mulStrict = mulStrict;
  exports.multiRNNCell = multiRNNCell;
  exports.multinomial = multinomial;
  exports.neg = neg;
  exports.nextFrame = nextFrame;
  exports.norm = norm;
  exports.notEqual = notEqual;
  exports.notEqualStrict = notEqualStrict;
  exports.oneHot = oneHot;
  exports.ones = ones$1;
  exports.onesLike = onesLike;
  exports.op = op;
  exports.outerProduct = outerProduct;
  exports.pad = pad;
  exports.pad1d = pad1d;
  exports.pad2d = pad2d;
  exports.pad3d = pad3d;
  exports.pad4d = pad4d;
  exports.pool = pool;
  exports.pow = pow;
  exports.powStrict = powStrict;
  exports.prelu = prelu;
  exports.print = print;
  exports.prod = prod;
  exports.profile = profile;
  exports.rand = rand;
  exports.randomGamma = randomGamma;
  exports.randomNormal = randomNormal;
  exports.randomUniform = randomUniform;
  exports.range = range;
  exports.ready = ready;
  exports.real = real;
  exports.reciprocal = reciprocal;
  exports.registerBackend = registerBackend;
  exports.registerGradient = registerGradient;
  exports.registerKernel = registerKernel;
  exports.relu = relu;
  exports.relu6 = relu6;
  exports.removeBackend = removeBackend;
  exports.reshape = reshape;
  exports.reverse = reverse;
  exports.reverse1d = reverse1d;
  exports.reverse2d = reverse2d;
  exports.reverse3d = reverse3d;
  exports.reverse4d = reverse4d;
  exports.rfft = rfft;
  exports.round = round;
  exports.rsqrt = rsqrt;
  exports.scalar = scalar;
  exports.scatterND = scatterND;
  exports.scatter_util = scatter_nd_util;
  exports.selu = selu;
  exports.separableConv2d = separableConv2d;
  exports.serialization = serialization;
  exports.setBackend = setBackend;
  exports.setPlatform = setPlatform;
  exports.setdiff1dAsync = setdiff1dAsync;
  exports.sigmoid = sigmoid;
  exports.sign = sign;
  exports.signal = signal_ops;
  exports.sin = sin;
  exports.sinh = sinh;
  exports.slice = slice;
  exports.slice1d = slice1d;
  exports.slice2d = slice2d;
  exports.slice3d = slice3d;
  exports.slice4d = slice4d;
  exports.slice_util = slice_util;
  exports.softmax = softmax;
  exports.softplus = softplus;
  exports.spaceToBatchND = spaceToBatchND;
  exports.sparseToDense = sparseToDense;
  exports.spectral = spectral_ops;
  exports.split = split;
  exports.sqrt = sqrt;
  exports.square = square;
  exports.squaredDifference = squaredDifference;
  exports.squaredDifferenceStrict = squaredDifferenceStrict;
  exports.squeeze = squeeze;
  exports.stack = stack;
  exports.step = step;
  exports.stft = stft;
  exports.stridedSlice = stridedSlice;
  exports.sub = sub;
  exports.subStrict = subStrict;
  exports.sum = sum$1;
  exports.sumOutType = sumOutType;
  exports.tan = tan;
  exports.tanh = tanh$1;
  exports.tensor = tensor;
  exports.tensor1d = tensor1d;
  exports.tensor2d = tensor2d;
  exports.tensor3d = tensor3d;
  exports.tensor4d = tensor4d;
  exports.tensor5d = tensor5d;
  exports.tensor6d = tensor6d;
  exports.tensor_util = tensor_util;
  exports.test_util = test_util;
  exports.tidy = tidy;
  exports.tile = tile;
  exports.time = time;
  exports.topk = topk;
  exports.train = train;
  exports.transpose = transpose;
  exports.truncatedNormal = truncatedNormal;
  exports.unregisterGradient = unregisterGradient;
  exports.unregisterKernel = unregisterKernel;
  exports.unsortedSegmentSum = unsortedSegmentSum;
  exports.unstack = unstack;
  exports.upcastType = upcastType;
  exports.util = util;
  exports.valueAndGrad = valueAndGrad;
  exports.valueAndGrads = valueAndGrads;
  exports.variable = variable;
  exports.variableGrads = variableGrads;
  exports.version_core = version;
  exports.where = where;
  exports.whereAsync = whereAsync;
  exports.zeros = zeros;
  exports.zerosLike = zerosLike;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-core.es2017.js.map
