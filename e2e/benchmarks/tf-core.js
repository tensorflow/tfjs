/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
(function(global, factory) {
typeof exports === 'object' && typeof module !== 'undefined' ?
    factory(exports) :
    typeof define === 'function' && define.amd ?
    define(['exports'], factory) :
    (global = global || self, factory(global.tf = global.tf || {}));
}(this, (function(exports) {
'use strict';

/*!
*****************************************************************************
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
MERCHANTABLITY OR NON-INFRINGEMENT.

See the Apache Version 2.0 License for specific language governing permissions
and limitations under the License.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
  extendStatics = Object.setPrototypeOf ||
      ({__proto__: []} instanceof Array && function(d, b) {
                    d.__proto__ = b;
                  }) || function(d, b) {
        for (var p in b)
          if (b.hasOwnProperty(p)) d[p] = b[p];
      };
  return extendStatics(d, b);
};

function __extends(d, b) {
  extendStatics(d, b);
  function __() {
    this.constructor = d;
  }
  d.prototype =
      b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

function __awaiter(thisArg, _arguments, P, generator) {
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator['throw'](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : new P(function(resolve) {
                                              resolve(result.value);
                                            }).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
}

function __generator(thisArg, body) {
  var _ = {
    label: 0,
    sent: function() {
      if (t[0] & 1) throw t[1];
      return t[1];
    },
    trys: [],
    ops: []
  },
      f, y, t, g;
  return g = {next: verb(0), 'throw': verb(1), 'return': verb(2)},
         typeof Symbol === 'function' && (g[Symbol.iterator] = function() {
           return this;
         }), g;
  function verb(n) {
    return function(v) {
      return step([n, v]);
    };
  }
  function step(op) {
    if (f) throw new TypeError('Generator is already executing.');
    while (_) try {
        if (f = 1,
            y &&
                (t = op[0] & 2 ?
                     y['return'] :
                     op[0] ? y['throw'] || ((t = y['return']) && t.call(y), 0) :
                             y.next) &&
                !(t = t.call(y, op[1])).done)
          return t;
        if (y = 0, t) op = [op[0] & 2, t.value];
        switch (op[0]) {
          case 0:
          case 1:
            t = op;
            break;
          case 4:
            _.label++;
            return {value: op[1], done: false};
          case 5:
            _.label++;
            y = op[1];
            op = [0];
            continue;
          case 7:
            op = _.ops.pop();
            _.trys.pop();
            continue;
          default:
            if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) &&
                (op[0] === 6 || op[0] === 2)) {
              _ = 0;
              continue;
            }
            if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) {
              _.label = op[1];
              break;
            }
            if (op[0] === 6 && _.label < t[1]) {
              _.label = t[1];
              t = op;
              break;
            }
            if (t && _.label < t[2]) {
              _.label = t[2];
              _.ops.push(op);
              break;
            }
            if (t[2]) _.ops.pop();
            _.trys.pop();
            continue;
        }
        op = body.call(thisArg, _);
      } catch (e) {
        op = [6, e];
        y = 0;
      } finally {
        f = t = 0;
      }
    if (op[0] & 5) throw op[1];
    return {value: op[0] ? op[1] : void 0, done: true};
  }
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
var EPSILON_FLOAT32 = 1e-7;
var EPSILON_FLOAT16 = 1e-4;
/** Convenient class for storing tensor-related data. */
var DataStorage = /** @class */ (function() {
  function DataStorage(backend, dataMover) {
    this.backend = backend;
    this.dataMover = dataMover;
    this.data = new WeakMap();
    this.dataIdsCount = 0;
  }
  DataStorage.prototype.get = function(dataId) {
    if (!this.data.has(dataId)) {
      this.dataMover.moveData(this.backend, dataId);
    }
    return this.data.get(dataId);
  };
  DataStorage.prototype.set = function(dataId, value) {
    this.dataIdsCount++;
    this.data.set(dataId, value);
  };
  DataStorage.prototype.has = function(dataId) {
    return this.data.has(dataId);
  };
  DataStorage.prototype.delete = function(dataId) {
    this.dataIdsCount--;
    return this.data.delete(dataId);
  };
  DataStorage.prototype.numDataIds = function() {
    return this.dataIdsCount;
  };
  return DataStorage;
}());
/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
var KernelBackend = /** @class */ (function() {
  function KernelBackend() {}
  KernelBackend.prototype.refCount = function(dataId) {
    return notYetImplemented('refCount');
  };
  KernelBackend.prototype.incRef = function(dataId) {
    return notYetImplemented('incRef');
  };
  KernelBackend.prototype.timerAvailable = function() {
    return true;
  };
  KernelBackend.prototype.time = function(f) {
    return notYetImplemented('time');
  };
  KernelBackend.prototype.read = function(dataId) {
    return notYetImplemented('read');
  };
  KernelBackend.prototype.readSync = function(dataId) {
    return notYetImplemented('readSync');
  };
  KernelBackend.prototype.numDataIds = function() {
    return notYetImplemented('numDataIds');
  };
  KernelBackend.prototype.disposeData = function(dataId, force) {
    return notYetImplemented('disposeData');
  };
  KernelBackend.prototype.write = function(values, shape, dtype) {
    return notYetImplemented('write');
  };
  KernelBackend.prototype.move = function(
      dataId, values, shape, dtype, refCount) {
    return notYetImplemented('move');
  };
  KernelBackend.prototype.memory = function() {
    return notYetImplemented('memory');
  };
  /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
  KernelBackend.prototype.floatPrecision = function() {
    return notYetImplemented('floatPrecision');
  };
  /** Returns the smallest representable number.  */
  KernelBackend.prototype.epsilon = function() {
    return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
  };
  KernelBackend.prototype.dispose = function() {
    return notYetImplemented('dispose');
  };
  return KernelBackend;
}());
function notYetImplemented(kernelName) {
  throw new Error(
      '\'' + kernelName +
      '\' not yet implemented or not found in the registry. ' +
      'This kernel may not be supported by the tfjs backend you have chosen');
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
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
// tslint:disable-next-line:no-any
function shuffle(array) {
  var counter = array.length;
  var temp = 0;
  var index = 0;
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
/**
 * Shuffles two arrays in-place the same way using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1,2,3,4,5];
 * const b = [11,22,33,44,55];
 * tf.util.shuffleCombo(a, b);
 * console.log(a, b);
 * ```
 *
 * @param array The first array to shuffle in-place.
 * @param array2 The second array to shuffle in-place with the same permutation
 *     as the first array.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
// tslint:disable-next-line:no-any
function shuffleCombo(
    array,
    // tslint:disable-next-line:no-any
    array2) {
  if (array.length !== array2.length) {
    throw Error(
        'Array sizes must match to be shuffled together ' +
        ('First array length was ' + array.length) +
        ('Second array length was ' + array2.length));
  }
  var counter = array.length;
  var temp, temp2;
  var index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = (Math.random() * counter) | 0;
    // Decrease counter by 1
    counter--;
    // And swap the last element of each array with it
    temp = array[counter];
    temp2 = array2[counter];
    array[counter] = array[index];
    array2[counter] = array2[index];
    array[index] = temp;
    array2[index] = temp2;
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
  var sum = 0;
  for (var i = 0; i < arr.length; i++) {
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
  var r = Math.random();
  return (b * r) + (1 - r) * a;
}
/** Returns the squared Euclidean distance between two vectors. */
function distSquared(a, b) {
  var result = 0;
  for (var i = 0; i < a.length; i++) {
    var diff = Number(a[i]) - Number(b[i]);
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
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function assert(expr, msg) {
  if (!expr) {
    throw new Error(typeof msg === 'string' ? msg : msg());
  }
}
function assertShapesMatch(shapeA, shapeB, errorMessagePrefix) {
  if (errorMessagePrefix === void 0) {
    errorMessagePrefix = '';
  }
  assert(arraysEqual(shapeA, shapeB), function() {
    return errorMessagePrefix +
        (' Shapes ' + shapeA + ' and ' + shapeB + ' must match');
  });
}
function assertNonNull(a) {
  assert(a != null, function() {
    return 'The input to the tensor constructor must be a non-null value.';
  });
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
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function flatten(arr, result, skipTypedArray) {
  if (result === void 0) {
    result = [];
  }
  if (skipTypedArray === void 0) {
    skipTypedArray = false;
  }
  if (result == null) {
    result = [];
  }
  if (Array.isArray(arr) || isTypedArray(arr) && !skipTypedArray) {
    for (var i = 0; i < arr.length; ++i) {
      flatten(arr[i], result, skipTypedArray);
    }
  } else {
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
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function sizeFromShape(shape) {
  if (shape.length === 0) {
    // Scalar.
    return 1;
  }
  var size = shape[0];
  for (var i = 1; i < shape.length; i++) {
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
  for (var i = 0; i < n1.length; i++) {
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
  } else if (x === -Infinity) {
    return -1;
  } else {
    var e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
}
function sizeToSquarishShape(size) {
  var width = Math.ceil(Math.sqrt(size));
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
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function createShuffledIndices(n) {
  var shuffledIndices = new Uint32Array(n);
  for (var i = 0; i < n; ++i) {
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
function repeatedTry(checkFn, delayFn, maxCounter) {
  if (delayFn === void 0) {
    delayFn = function(counter) {
      return 0;
    };
  }
  return new Promise(function(resolve, reject) {
    var tryCount = 0;
    var tryFn = function() {
      if (checkFn()) {
        resolve();
        return;
      }
      tryCount++;
      var nextBackoff = delayFn(tryCount);
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
  var shapeProd = 1;
  var implicitIdx = -1;
  for (var i = 0; i < shape.length; ++i) {
    if (shape[i] >= 0) {
      shapeProd *= shape[i];
    } else if (shape[i] === -1) {
      if (implicitIdx !== -1) {
        throw Error(
            'Shapes can only have 1 implicit size. ' +
            ('Found -1 at dim ' + implicitIdx + ' and dim ' + i));
      }
      implicitIdx = i;
    } else if (shape[i] < 0) {
      throw Error('Shapes can not be < 0. Found ' + shape[i] + ' at dim ' + i);
    }
  }
  if (implicitIdx === -1) {
    if (size > 0 && size !== shapeProd) {
      throw Error(
          'Size(' + size + ') must match the product of shape ' + shape);
    }
    return shape;
  }
  if (shapeProd === 0) {
    throw Error(
        'Cannot infer the missing size in [' + shape + '] when ' +
        'there are 0 elements');
  }
  if (size % shapeProd !== 0) {
    throw Error(
        'The implicit shape can\'t be a fractional number. ' +
        ('Got ' + size + ' / ' + shapeProd));
  }
  var newShape = shape.slice();
  newShape[implicitIdx] = size / shapeProd;
  return newShape;
}
function parseAxisParam(axis, shape) {
  var rank = shape.length;
  // Normalize input
  axis = axis == null ? shape.map(function(s, i) {
    return i;
  }) :
                        [].concat(axis);
  // Check for valid range
  assert(
      axis.every(function(ax) {
        return ax >= -rank && ax < rank;
      }),
      function() {
        return 'All values in axis param must be in range [-' + rank + ', ' +
            rank + ') but ' + ('got axis ' + axis);
      });
  // Check for only integers
  assert(
      axis.every(function(ax) {
        return isInt(ax);
      }),
      function() {
        return 'All values in axis param must be integers but ' +
            ('got axis ' + axis);
      });
  // Handle negative axis.
  return axis.map(function(a) {
    return a < 0 ? rank + a : a;
  });
}
/** Reduces the shape by removing all dimensions of shape 1. */
function squeezeShape(shape, axis) {
  var newShape = [];
  var keptDims = [];
  var isEmptyArray = axis != null && Array.isArray(axis) && axis.length === 0;
  var axes = (axis == null || isEmptyArray) ?
      null :
      parseAxisParam(axis, shape).sort();
  var j = 0;
  for (var i = 0; i < shape.length; ++i) {
    if (axes != null) {
      if (axes[j] === i && shape[i] !== 1) {
        throw new Error(
            'Can\'t squeeze axis ' + i + ' since its dim \'' + shape[i] +
            '\' is not 1');
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
  return {newShape: newShape, keptDims: keptDims};
}
function getTypedArrayFromDType(dtype, size) {
  var values = null;
  if (dtype == null || dtype === 'float32') {
    values = new Float32Array(size);
  } else if (dtype === 'int32') {
    values = new Int32Array(size);
  } else if (dtype === 'bool') {
    values = new Uint8Array(size);
  } else {
    throw new Error('Unknown data type ' + dtype);
  }
  return values;
}
function getArrayFromDType(dtype, size) {
  var values = null;
  if (dtype == null || dtype === 'float32') {
    values = new Float32Array(size);
  } else if (dtype === 'int32') {
    values = new Int32Array(size);
  } else if (dtype === 'bool') {
    values = new Uint8Array(size);
  } else if (dtype === 'string') {
    values = new Array(size);
  } else {
    throw new Error('Unknown data type ' + dtype);
  }
  return values;
}
function checkConversionForErrors(vals, dtype) {
  for (var i = 0; i < vals.length; i++) {
    var num = vals[i];
    if (isNaN(num) || !isFinite(num)) {
      throw Error(
          'A tensor of type ' + dtype + ' being uploaded contains ' + num +
          '.');
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
  } else if (dtype === 'complex64') {
    return 8;
  } else if (dtype === 'bool') {
    return 1;
  } else {
    throw new Error('Unknown dtype ' + dtype);
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
  var bytes = 0;
  arr.forEach(function(x) {
    return bytes += x.length;
  });
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
  } else if (values instanceof Int32Array || values instanceof Uint8Array) {
    return 'int32';
  } else if (isNumber(values)) {
    return 'float32';
  } else if (isString(values)) {
    return 'string';
  } else if (isBoolean(values)) {
    return 'bool';
  }
  return 'float32';
}
function isFunction(f) {
  return !!(f && f.constructor && f.call && f.apply);
}
function nearestDivisor(size, start) {
  for (var i = start; i < size; ++i) {
    if (size % i === 0) {
      return i;
    }
  }
  return size;
}
function computeStrides(shape) {
  var rank = shape.length;
  if (rank < 2) {
    return [];
  }
  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  var strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (var i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}
function createNestedArray(offset, shape, a) {
  var ret = new Array();
  if (shape.length === 1) {
    var d = shape[0];
    for (var i = 0; i < d; i++) {
      ret[i] = a[offset + i];
    }
  } else {
    var d = shape[0];
    var rest = shape.slice(1);
    var len = rest.reduce(function(acc, c) {
      return acc * c;
    });
    for (var i = 0; i < d; i++) {
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
  var size = shape.reduce(function(acc, c) {
    return acc * c;
  });
  if (size === 0) {
    // A tensor with shape zero should be turned into empty list.
    return [];
  }
  if (size !== a.length) {
    throw new Error(
        '[' + shape + '] does not match the input size ' + a.length + '.');
  }
  return createNestedArray(0, shape, a);
}
function makeOnesTypedArray(size, dtype) {
  var array = makeZerosTypedArray(size, dtype);
  for (var i = 0; i < array.length; i++) {
    array[i] = 1;
  }
  return array;
}
function makeZerosTypedArray(size, dtype) {
  if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
    return new Float32Array(size);
  } else if (dtype === 'int32') {
    return new Int32Array(size);
  } else if (dtype === 'bool') {
    return new Uint8Array(size);
  } else {
    throw new Error('Unknown data type ' + dtype);
  }
}
/**
 * Make nested `TypedArray` filled with zeros.
 * @param shape The shape information for the nested array.
 * @param dtype dtype of the array element.
 */
function makeZerosNestedTypedArray(shape, dtype) {
  var size = shape.reduce(function(prev, curr) {
    return prev * curr;
  }, 1);
  if (dtype == null || dtype === 'float32') {
    return toNestedArray(shape, new Float32Array(size));
  } else if (dtype === 'int32') {
    return toNestedArray(shape, new Int32Array(size));
  } else if (dtype === 'bool') {
    return toNestedArray(shape, new Uint8Array(size));
  } else {
    throw new Error('Unknown data type ' + dtype);
  }
}
function assertNonNegativeIntegerDimensions(shape) {
  shape.forEach(function(dimSize) {
    assert(Number.isInteger(dimSize) && dimSize >= 0, function() {
      return 'Tensor must have a shape comprised of positive integers but got ' +
          ('shape [' + shape + '].');
    });
  });
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
  } else if (rank === 1) {
    return locs[0];
  }
  var index = locs[locs.length - 1];
  for (var i = 0; i < locs.length - 1; ++i) {
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
  } else if (rank === 1) {
    return [index];
  }
  var locs = new Array(rank);
  for (var i = 0; i < locs.length - 1; ++i) {
    locs[i] = Math.floor(index / strides[i]);
    index -= locs[i] * strides[i];
  }
  locs[locs.length - 1] = index;
  return locs;
}
/**
 * This method asserts whether an object is a Promise instance.
 * @param object
 */
// tslint:disable-next-line: no-any
function isPromise(object) {
  //  We chose to not use 'obj instanceOf Promise' for two reasons:
  //  1. It only reliably works for es6 Promise, not other Promise
  //  implementations.
  //  2. It doesn't work with framework that uses zone.js. zone.js monkey patch
  //  the async calls, so it is possible the obj (patched) is comparing to a
  //  pre-patched Promise.
  return object && object.then && typeof object.then === 'function';
}

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
var TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
/**
 * The environment contains evaluated flags as well as the registered platform.
 * This is always used as a global singleton and can be retrieved with
 * `tf.env()`.
 *
 * @doc {heading: 'Environment'}
 */
var Environment = /** @class */ (function() {
  // tslint:disable-next-line: no-any
  function Environment(global) {
    this.global = global;
    this.flags = {};
    this.flagRegistry = {};
    this.urlFlags = {};
    this.populateURLFlags();
  }
  Environment.prototype.setPlatform = function(platformName, platform) {
    if (this.platform != null) {
      console.warn(
          'Platform ' + this.platformName + ' has already been set. ' +
          ('Overwriting the platform with ' + platform + '.'));
    }
    this.platformName = platformName;
    this.platform = platform;
  };
  Environment.prototype.registerFlag = function(
      flagName, evaluationFn, setHook) {
    this.flagRegistry[flagName] = {
      evaluationFn: evaluationFn,
      setHook: setHook
    };
    // Override the flag value from the URL. This has to happen here because the
    // environment is initialized before flags get registered.
    if (this.urlFlags[flagName] != null) {
      var flagValue = this.urlFlags[flagName];
      console.warn(
          'Setting feature override from URL ' + flagName + ': ' + flagValue +
          '.');
      this.set(flagName, flagValue);
    }
  };
  Environment.prototype.getAsync = function(flagName) {
    return __awaiter(this, void 0, void 0, function() {
      var _a, _b;
      return __generator(this, function(_c) {
        switch (_c.label) {
          case 0:
            if (flagName in this.flags) {
              return [2 /*return*/, this.flags[flagName]];
            }
            _a = this.flags;
            _b = flagName;
            return [4 /*yield*/, this.evaluateFlag(flagName)];
          case 1:
            _a[_b] = _c.sent();
            return [2 /*return*/, this.flags[flagName]];
        }
      });
    });
  };
  Environment.prototype.get = function(flagName) {
    if (flagName in this.flags) {
      return this.flags[flagName];
    }
    var flagValue = this.evaluateFlag(flagName);
    if (isPromise(flagValue)) {
      throw new Error(
          'Flag ' + flagName + ' cannot be synchronously evaluated. ' +
          'Please use getAsync() instead.');
    }
    this.flags[flagName] = flagValue;
    return this.flags[flagName];
  };
  Environment.prototype.getNumber = function(flagName) {
    return this.get(flagName);
  };
  Environment.prototype.getBool = function(flagName) {
    return this.get(flagName);
  };
  Environment.prototype.getFlags = function() {
    return this.flags;
  };
  Object.defineProperty(Environment.prototype, 'features', {
    // For backwards compatibility.
    get: function() {
      return this.flags;
    },
    enumerable: true,
    configurable: true
  });
  Environment.prototype.set = function(flagName, value) {
    if (this.flagRegistry[flagName] == null) {
      throw new Error(
          'Cannot set flag ' + flagName + ' as it has not been registered.');
    }
    this.flags[flagName] = value;
    if (this.flagRegistry[flagName].setHook != null) {
      this.flagRegistry[flagName].setHook(value);
    }
  };
  Environment.prototype.evaluateFlag = function(flagName) {
    if (this.flagRegistry[flagName] == null) {
      throw new Error(
          'Cannot evaluate flag \'' + flagName +
          '\': no evaluation function found.');
    }
    return this.flagRegistry[flagName].evaluationFn();
  };
  Environment.prototype.setFlags = function(flags) {
    this.flags = Object.assign({}, flags);
  };
  Environment.prototype.reset = function() {
    this.flags = {};
    this.urlFlags = {};
    this.populateURLFlags();
  };
  Environment.prototype.populateURLFlags = function() {
    var _this = this;
    if (typeof this.global === 'undefined' ||
        typeof this.global.location === 'undefined' ||
        typeof this.global.location.search === 'undefined') {
      return;
    }
    var urlParams = getQueryParams(this.global.location.search);
    if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
      var keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
      keyValues.forEach(function(keyValue) {
        var _a = keyValue.split(':'), key = _a[0], value = _a[1];
        _this.urlFlags[key] = parseValue(key, value);
      });
    }
  };
  return Environment;
}());
function getQueryParams(queryString) {
  var params = {};
  queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, function(s) {
    var t = [];
    for (var _i = 1; _i < arguments.length; _i++) {
      t[_i - 1] = arguments[_i];
    }
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
  } else if ('' + +value === value) {
    return +value;
  }
  throw new Error(
      'Could not parse value flag value ' + value + ' for flag ' + flagName +
      '.');
}
/**
 * Returns the current environment (a global singleton).
 *
 * The environment object contains the evaluated feature values as well as the
 * active platform.
 *
 * @doc {heading: 'Environment'}
 */
function env() {
  return exports.ENV;
}
exports.ENV = null;
function setEnvironmentGlobal(environment) {
  exports.ENV = environment;
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
// Note that the identifier globalNameSpace is scoped to this module, but will
// always resolve to the same global object regardless of how the module is
// resolved.
// tslint:disable-next-line:no-any
var globalNameSpace;
// tslint:disable-next-line:no-any
function getGlobalNamespace() {
  if (globalNameSpace == null) {
    // tslint:disable-next-line:no-any
    var ns = void 0;
    if (typeof (window) !== 'undefined') {
      ns = window;
    } else if (typeof (global) !== 'undefined') {
      ns = global;
    } else if (typeof (process) !== 'undefined') {
      ns = process;
    } else if (typeof (self) !== 'undefined') {
      ns = self;
    } else {
      throw new Error('Could not find a global object');
    }
    globalNameSpace = ns;
  }
  return globalNameSpace;
}
// tslint:disable-next-line:no-any
function getGlobalMap() {
  var ns = getGlobalNamespace();
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
  var globalMap = getGlobalMap();
  if (globalMap.has(key)) {
    return globalMap.get(key);
  } else {
    var singleton = init();
    globalMap.set(key, singleton);
    return globalMap.get(key);
  }
}

var Abs = 'Abs';
var Acos = 'Acos';
var Acosh = 'Acosh';
var Add = 'Add';
var AddN = 'AddN';
var All = 'All';
var Any = 'Any';
var ArgMax = 'ArgMax';
var ArgMin = 'ArgMin';
var Asin = 'Asin';
var Asinh = 'Asinh';
var Atan = 'Atan';
var Atanh = 'Atanh';
var Atan2 = 'Atan2';
var AvgPool = 'AvgPool';
var AvgPoolGrad = 'AvgPoolGrad';
var AvgPool3D = 'AvgPool3D';
var AvgPool3DGrad = 'AvgPool3DGrad';
var BatchMatMul = 'BatchMatMul';
var BatchToSpaceND = 'BatchToSpaceND';
var Bincount = 'Bincount';
var BroadcastTo = 'BroadcastTo';
var Cast = 'Cast';
var Ceil = 'Ceil';
var ClipByValue = 'ClipByValue';
var Complex = 'Complex';
var ComplexAbs = 'ComplexAbs';
var Concat = 'Concat';
var Conv2D = 'Conv2D';
var Conv2DBackpropFilter = 'Conv2DBackpropFilter';
var Conv2DBackpropInput = 'Conv2DBackpropInput';
var Conv3D = 'Conv3D';
var Conv3DBackpropFilterV2 = 'Conv3DBackpropFilterV2';
var Conv3DBackpropInputV2 = 'Conv3DBackpropInputV2';
var Cos = 'Cos';
var Cosh = 'Cosh';
var Cumsum = 'Cumsum';
var CropAndResize = 'CropAndResize';
var DenseBincount = 'DenseBincount';
var DepthToSpace = 'DepthToSpace';
var DepthwiseConv2dNative = 'DepthwiseConv2dNative';
var DepthwiseConv2dNativeBackpropFilter = 'DepthwiseConv2dNativeBackpropFilter';
var DepthwiseConv2dNativeBackpropInput = 'DepthwiseConv2dNativeBackpropInput';
var Diag = 'Diag';
var Dilation2D = 'Dilation2D';
var Dilation2DBackpropInput = 'Dilation2DBackpropInput';
var Dilation2DBackpropFilter = 'Dilation2DBackpropFilter';
var RealDiv = 'RealDiv';
var Elu = 'Elu';
var EluGrad = 'EluGrad';
var Erf = 'Erf';
var Equal = 'Equal';
var Exp = 'Exp';
var ExpandDims = 'ExpandDims';
var Expm1 = 'Expm1';
var FFT = 'FFT';
var Fill = 'Fill';
var FlipLeftRight = 'FlipLeftRight';
var Floor = 'Floor';
var FloorDiv = 'FloorDiv';
var FusedBatchNorm = 'FusedBatchNorm';
var GatherV2 = 'GatherV2';
var GatherNd = 'GatherNd';
var Greater = 'Greater';
var GreaterEqual = 'GreaterEqual';
var Identity = 'Identity';
var IFFT = 'IFFT';
var Imag = 'Imag';
var IsFinite = 'IsFinite';
var IsInf = 'IsInf';
var IsNan = 'IsNan';
var LeakyRelu = 'LeakyRelu';
var Less = 'Less';
var LessEqual = 'LessEqual';
var LinSpace = 'LinSpace';
var Log = 'Log';
var Log1p = 'Log1p';
var LogicalAnd = 'LogicalAnd';
var LogicalNot = 'LogicalNot';
var LogicalOr = 'LogicalOr';
var LogSoftmax = 'LogSoftmax';
var LRN = 'LRN';
var LRNGrad = 'LRNGrad';
var Max = 'Max';
var Maximum = 'Maximum';
var MaxPool = 'MaxPool';
var MaxPoolGrad = 'MaxPoolGrad';
var MaxPool3D = 'MaxPool3D';
var MaxPool3DGrad = 'MaxPool3DGrad';
var MaxPoolWithArgmax = 'MaxPoolWithArgmax';
var Mean = 'Mean';
var Min = 'Min';
var Minimum = 'Minimum';
var MirrorPad = 'MirrorPad';
var Mod = 'Mod';
var Multinomial = 'Multinomial';
var Multiply = 'Multiply';
var Neg = 'Neg';
var NotEqual = 'NotEqual';
var NonMaxSuppressionV3 = 'NonMaxSuppressionV3';
var NonMaxSuppressionV4 = 'NonMaxSuppressionV4';
var NonMaxSuppressionV5 = 'NonMaxSuppressionV5';
var OnesLike = 'OnesLike';
var OneHot = 'OneHot';
var Pack = 'Pack';
var PadV2 = 'PadV2';
var Pool = 'Pool';
var Pow = 'Pow';
var Prelu = 'Prelu';
var Prod = 'Prod';
var Range = 'Range';
var Real = 'Real';
var Reciprocal = 'Reciprocal';
var Relu = 'Relu';
var Reshape = 'Reshape';
var ResizeNearestNeighbor = 'ResizeNearestNeighbor';
var ResizeNearestNeighborGrad = 'ResizeNearestNeighborGrad';
var ResizeBilinear = 'ResizeBilinear';
var ResizeBilinearGrad = 'ResizeBilinearGrad';
var Relu6 = 'Relu6';
var Reverse = 'Reverse';
var Round = 'Round';
var Rsqrt = 'Rsqrt';
var ScatterNd = 'ScatterNd';
var Select = 'Select';
var Selu = 'Selu';
var Slice = 'Slice';
var Sin = 'Sin';
var Sinh = 'Sinh';
var Sign = 'Sign';
var Sigmoid = 'Sigmoid';
var Softplus = 'Softplus';
var Sqrt = 'Sqrt';
var Sum = 'Sum';
var SpaceToBatchND = 'SpaceToBatchND';
var SplitV = 'SplitV';
var Softmax = 'Softmax';
var SquaredDifference = 'SquaredDifference';
var Square = 'Square';
var Sub = 'Sub';
var SparseToDense = 'SparseToDense';
var StridedSlice = 'StridedSlice';
var Tan = 'Tan';
var Tanh = 'Tanh';
var Tile = 'Tile';
var TopK = 'TopK';
var Transpose = 'Transpose';
var Unique = 'Unique';
var Unpack = 'Unpack';
var UnsortedSegmentSum = 'UnsortedSegmentSum';
var ZerosLike = 'ZerosLike';
/**
 * TensorFlow.js-only kernels
 */
var Step = 'Step';
var FromPixels = 'FromPixels';
var RotateWithOffset = 'RotateWithOffset';
var _FusedMatMul = '_FusedMatMul';
var FusedConv2D = 'FusedConv2D';
var FusedDepthwiseConv2D = 'FusedDepthwiseConv2D';

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
var kernelRegistry = getGlobal('kernelRegistry', function() {
  return new Map();
});
var gradRegistry = getGlobal('gradRegistry', function() {
  return new Map();
});
/**
 * Returns the kernel function (code) associated with the provided names.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 */
function getKernel(kernelName, backendName) {
  var key = makeKey(kernelName, backendName);
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
  var it = kernelRegistry.entries();
  var result = [];
  while (true) {
    var _a = it.next(), done = _a.done, value = _a.value;
    if (done) {
      break;
    }
    var key = value[0], config = value[1];
    var backend = key.split('_')[0];
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
  var kernelName = config.kernelName, backendName = config.backendName;
  var key = makeKey(kernelName, backendName);
  if (kernelRegistry.has(key)) {
    console.warn(
        'The kernel \'' + kernelName + '\' for backend ' +
        ('\'' + backendName + '\' is already registered'));
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
  var kernelName = config.kernelName;
  if (gradRegistry.has(kernelName)) {
    // TODO (yassogba) after 3.0 assess whether we need to keep this gated
    // to debug mode.
    if (env().getBool('DEBUG')) {
      console.warn('Overriding the gradient for \'' + kernelName + '\'');
    }
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
  var key = makeKey(kernelName, backendName);
  if (!kernelRegistry.has(key)) {
    throw new Error(
        'The kernel \'' + kernelName + '\' for backend ' +
        ('\'' + backendName + '\' is not registered'));
  }
  kernelRegistry.delete(key);
}
/** Removes the registered gradient from the global registry. */
function unregisterGradient(kernelName) {
  if (!gradRegistry.has(kernelName)) {
    throw new Error(
        'The gradient \'' + kernelName + '\' for backend is not registered');
  }
  gradRegistry.delete(kernelName);
}
/**
 * Finds kernels that have already been registered to a backend and re-registers
 * them for a new backend. Useful for registering custom backends.
 * @param registeredBackendName Already registered backend.
 * @param newBackendName New backend.
 */
function copyRegisteredKernels(registeredBackendName, newBackendName) {
  var kernels = getKernelsForBackend(registeredBackendName);
  kernels.forEach(function(kernelConfig) {
    var newKernelConfig =
        Object.assign({}, kernelConfig, {backendName: newBackendName});
    registerKernel(newKernelConfig);
  });
}
function makeKey(kernelName, backendName) {
  return backendName + '_' + kernelName;
}

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
 * Create typed array for scalar value. Used for storing in `DataStorage`.
 */
function createScalarValue(value, dtype) {
  if (dtype === 'string') {
    return encodeString(value);
  }
  return toTypedArray([value], dtype);
}
function noConversionNeeded(a, dtype) {
  return (a instanceof Float32Array && dtype === 'float32') ||
      (a instanceof Int32Array && dtype === 'int32') ||
      (a instanceof Uint8Array && dtype === 'bool');
}
function toTypedArray(a, dtype) {
  if (dtype === 'string') {
    throw new Error('Cannot convert a string[] to a TypedArray');
  }
  if (Array.isArray(a)) {
    a = flatten(a);
  }
  if (env().getBool('DEBUG')) {
    checkConversionForErrors(a, dtype);
  }
  if (noConversionNeeded(a, dtype)) {
    return a;
  }
  if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
    return new Float32Array(a);
  } else if (dtype === 'int32') {
    return new Int32Array(a);
  } else if (dtype === 'bool') {
    var bool = new Uint8Array(a.length);
    for (var i = 0; i < bool.length; ++i) {
      if (Math.round(a[i]) !== 0) {
        bool[i] = 1;
      }
    }
    return bool;
  } else {
    throw new Error('Unknown data type ' + dtype);
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
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function now() {
  return env().platform.now();
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
 *
 * @doc {heading: 'Util'}
 */
function fetch$1(path, requestInits) {
  return env().platform.fetch(path, requestInits);
}
/**
 * Encodes the provided string into bytes using the provided encoding scheme.
 *
 * @param s The string to encode.
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
function encodeString(s, encoding) {
  if (encoding === void 0) {
    encoding = 'utf-8';
  }
  encoding = encoding || 'utf-8';
  return env().platform.encode(s, encoding);
}
/**
 * Decodes the provided bytes into a string using the provided encoding scheme.
 * @param bytes The bytes to decode.
 *
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
function decodeString(bytes, encoding) {
  if (encoding === void 0) {
    encoding = 'utf-8';
  }
  encoding = encoding || 'utf-8';
  return env().platform.decode(bytes, encoding);
}

var util = {
  __proto__: null,
  createScalarValue: createScalarValue,
  toTypedArray: toTypedArray,
  now: now,
  fetch: fetch$1,
  encodeString: encodeString,
  decodeString: decodeString,
  shuffle: shuffle,
  shuffleCombo: shuffleCombo,
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
  toNestedArray: toNestedArray,
  makeOnesTypedArray: makeOnesTypedArray,
  makeZerosTypedArray: makeZerosTypedArray,
  makeZerosNestedTypedArray: makeZerosNestedTypedArray,
  assertNonNegativeIntegerDimensions: assertNonNegativeIntegerDimensions,
  locToIndex: locToIndex,
  indexToLoc: indexToLoc,
  isPromise: isPromise
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
var Profiler = /** @class */ (function() {
  function Profiler(backendTimer, logger) {
    this.backendTimer = backendTimer;
    this.logger = logger;
    if (logger == null) {
      this.logger = new Logger();
    }
  }
  Profiler.prototype.profileKernel = function(kernelName, inputs, f) {
    var outputs;
    var holdResultWrapperFn = function() {
      outputs = f();
    };
    var timer;
    if (this.backendTimer.timerAvailable()) {
      timer = this.backendTimer.time(holdResultWrapperFn);
    } else {
      // warming up to remove the upload delay
      holdResultWrapperFn();
      for (let i = 0; i < outputs.length; i++) {
        outputs[i].dataSync();
      };
      // first execution
      var start = now();
      for (let i = 0; i < 2; i++) {
        holdResultWrapperFn();
      }
      for (let i = 0; i < outputs.length; i++) {
        outputs[i].dataSync();
      };
      // first execution time = 1 * kernelExeuctionTime + downloadTime
      var firstExecutionTime = now() - start;
      // second execution that contains two kernel runs
      var secondStart = now();
      for (let i = 0; i < 1; i++) {
        holdResultWrapperFn();
      }
      for (let i = 0; i < outputs.length; i++) {
        outputs[i].dataSync();
      };
      // second execution time = 2 * kernelExeuctionTime + downloadTime
      var secondExecutionTime = now() - secondStart;
      // the kernel exeuction time = secondExecutionTime - firstExecutionTime
      //                           = 1 * kernelExecutionTime
      // Assume variance of downloadTime is small.
      timer = Promise.resolve(
          {kernelMs: (firstExecutionTime - secondExecutionTime)});
    }
    if (env().getBool('CHECK_COMPUTATION_FOR_ERRORS')) {
      var _loop_1 = function(i) {
        var output = outputs[i];
        // Dangling promise here because we don't want to propagate up
        // asynchronicity.
        output.data().then(function(tensorVals) {
          checkComputationForErrors(tensorVals, output.dtype, kernelName);
        });
      };
      for (var i = 0; i < outputs.length; i++) {
        _loop_1(i);
      }
    }
    var kernelProfile = {
      kernelName: kernelName,
      outputs: outputs,
      inputs: inputs,
      timeMs: timer.then(function(timing) {
        return timing.kernelMs;
      }),
      extraInfo: timer.then(function(timing) {
        return timing.getExtraProfileInfo != null ?
            timing.getExtraProfileInfo() :
            '';
      })
    };
    return kernelProfile;
  };
  Profiler.prototype.logKernelProfile = function(kernelProfile) {
    var _this = this;
    var kernelName = kernelProfile.kernelName, outputs = kernelProfile.outputs,
        timeMs = kernelProfile.timeMs, inputs = kernelProfile.inputs,
        extraInfo = kernelProfile.extraInfo;
    outputs.forEach(function(result) {
      Promise.all([result.data(), timeMs, extraInfo])
          .then(function(valueContainer) {
            _this.logger.logKernelProfile(
                kernelName, result, valueContainer[0], valueContainer[1],
                inputs, valueContainer[2]);
          });
    });
  };
  return Profiler;
}());
function checkComputationForErrors(vals, dtype, kernelName) {
  if (dtype !== 'float32') {
    // Only floating point computations will generate NaN values
    return false;
  }
  for (var i = 0; i < vals.length; i++) {
    var num = vals[i];
    if (isNaN(num) || !isFinite(num)) {
      // Throwing custom exception so behavior is testable.
      console.warn('Found ' + num + ' in the result of \'' + kernelName + '\'');
      return true;
    }
  }
  return false;
}
var Logger = /** @class */ (function() {
  function Logger() {}
  Logger.prototype.logKernelProfile = function(
      name, result, vals, timeMs, inputs, extraInfo) {
    var time = typeof timeMs === 'number' ? rightPad(timeMs + 'ms', 9) :
                                            timeMs['error'];
    var paddedName = rightPad(name, 25);
    var rank = result.rank;
    var size = result.size;
    var shape = rightPad(result.shape.toString(), 14);
    var inputShapesDescription = '';
    for (var name_1 in inputs) {
      var input = inputs[name_1];
      if (input != null) {
        // The input might be a non-tensor (e.g HTMLImageElement), in which case
        // we claim the output shape as input shape.
        var inputShape = input.shape || result.shape;
        var inputRank = inputShape.length;
        inputShapesDescription += name_1 + ': ' + inputRank + 'D ' +
            (inputRank > 0 ? inputShape : '') + ' ';
      }
    }
    console.log(
        '%c' + paddedName + '\t%c' + time + '\t%c' + rank + 'D ' + shape +
            '\t%c' + size + '\t%c' + inputShapesDescription + '\t%c' +
            extraInfo,
        'font-weight:bold', 'color:red', 'color:blue', 'color: orange',
        'color: green', 'color: steelblue');
  };
  return Logger;
}());

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
  var tensorsFromX = {};
  var nodesFromX = {};
  for (var i = 0; i < xs.length; i++) {
    tensorsFromX[xs[i].id] = true;
  }
  for (var i = 0; i < tape.length; i++) {
    var node = tape[i];
    var nodeInputs = node.inputs;
    for (var inputName in nodeInputs) {
      var input = nodeInputs[inputName];
      var anyInputFromX = false;
      for (var j = 0; j < xs.length; j++) {
        if (tensorsFromX[input.id]) {
          node.outputs.forEach(function(output) {
            return tensorsFromX[output.id] = true;
          });
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
  var tensorsLeadToY = {};
  tensorsLeadToY[y.id] = true;
  var nodesToY = {};
  for (var i = tape.length - 1; i >= 0; i--) {
    var node = tape[i];
    var nodeInputs = node.inputs;
    // If any of the outputs lead to y, mark all of the inputs as leading to y.
    for (var j = 0; j < node.outputs.length; j++) {
      if (tensorsLeadToY[node.outputs[j].id]) {
        for (var inputName in nodeInputs) {
          tensorsLeadToY[nodeInputs[inputName].id] = true;
          nodesToY[node.id] = true;
        }
        break;
      }
    }
  }
  // Return the paths that come from x and lead to y.
  var filteredTape = [];
  for (var i = 0; i < tape.length; i++) {
    var node = tape[i];
    if (nodesFromX[node.id] && nodesToY[node.id]) {
      // Prune the inputs from the node that aren't a function of x.
      var prunedInputs = {};
      for (var inputName in node.inputs) {
        var nodeInput = node.inputs[inputName];
        if (tensorsFromX[nodeInput.id]) {
          prunedInputs[inputName] = nodeInput;
        }
      }
      // Copy the node and overwrite inputsAndArgs to the pruned version.
      var prunedNode = Object.assign({}, node);
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
function backpropagateGradients(
    tensorAccumulatedGradientMap, filteredTape, tidy, add) {
  var _loop_1 = function(i) {
    var node = filteredTape[i];
    var dys = [];
    node.outputs.forEach(function(o) {
      var gradTensor = tensorAccumulatedGradientMap[o.id];
      if (gradTensor != null) {
        dys.push(gradTensor);
      } else {
        // This particular output is not in the back-propagation subgraph, so it
        // does not affect the final output, thus we put null for its dy.
        dys.push(null);
      }
    });
    if (node.gradient == null) {
      throw new Error(
          'Cannot compute gradient: gradient function not found ' +
          ('for ' + node.kernelName + '.'));
    }
    // Backprop dy through this node and accumulate gradients over the inputs.
    var inputGradients = node.gradient(dys);
    var _loop_2 = function(inputName) {
      if (!(inputName in inputGradients)) {
        throw new Error(
            'Cannot backprop through input ' + inputName + '. ' +
            ('Available gradients found: ' + Object.keys(inputGradients) +
             '.'));
      }
      // Call the gradient function.
      var dx = tidy(function() {
        return inputGradients[inputName]();
      });
      if (dx.dtype !== 'float32') {
        throw new Error(
            'Error in gradient for op ' + node.kernelName +
            '. The gradient of input ' +
            (inputName + ' must have \'float32\' dtype, but has \'' + dx.dtype +
             '\''));
      }
      var x = node.inputs[inputName];
      if (!arraysEqual(dx.shape, x.shape)) {
        throw new Error(
            'Error in gradient for op ' + node.kernelName +
            '. The gradient of input ' +
            ('\'' + inputName + '\' has shape \'' + dx.shape +
             '\', which does not match ') +
            ('the shape of the input \'' + x.shape + '\''));
      }
      if (tensorAccumulatedGradientMap[x.id] == null) {
        tensorAccumulatedGradientMap[x.id] = dx;
      } else {
        var curGradient = tensorAccumulatedGradientMap[x.id];
        tensorAccumulatedGradientMap[x.id] = add(curGradient, dx);
        curGradient.dispose();
      }
    };
    for (var inputName in node.inputs) {
      _loop_2(inputName);
    }
  };
  // Walk the tape backward and keep a map of Tensor to its gradient.
  for (var i = filteredTape.length - 1; i >= 0; i--) {
    _loop_1(i);
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
var FORMAT_LIMIT_NUM_VALS = 20;
// Number of first and last values to show when displaying a, b,...,y, z.
var FORMAT_NUM_FIRST_LAST_VALS = 3;
// Number of significant digits to show.
var FORMAT_NUM_SIG_DIGITS = 7;
function tensorToString(vals, shape, dtype, verbose) {
  var strides = computeStrides(shape);
  var padPerCol = computeMaxSizePerColumn(vals, shape, dtype, strides);
  var rank = shape.length;
  var valsLines = subTensorToString(vals, shape, dtype, strides, padPerCol);
  var lines = ['Tensor'];
  if (verbose) {
    lines.push('  dtype: ' + dtype);
    lines.push('  rank: ' + rank);
    lines.push('  shape: [' + shape + ']');
    lines.push('  values:');
  }
  lines.push(valsLines
                 .map(function(l) {
                   return '    ' + l;
                 })
                 .join('\n'));
  return lines.join('\n');
}
function computeMaxSizePerColumn(vals, shape, dtype, strides) {
  var n = sizeFromShape(shape);
  var numCols = strides[strides.length - 1];
  var padPerCol = new Array(numCols).fill(0);
  var rank = shape.length;
  var valuesOrTuples = dtype === 'complex64' ? createComplexTuples(vals) : vals;
  if (rank > 1) {
    for (var row = 0; row < n / numCols; row++) {
      var offset = row * numCols;
      for (var j = 0; j < numCols; j++) {
        padPerCol[j] = Math.max(
            padPerCol[j],
            valToString(valuesOrTuples[offset + j], 0, dtype).length);
      }
    }
  }
  return padPerCol;
}
function valToString(val, pad, dtype) {
  var valStr;
  if (Array.isArray(val)) {
    valStr = parseFloat(val[0].toFixed(FORMAT_NUM_SIG_DIGITS)) + ' + ' +
        (parseFloat(val[1].toFixed(FORMAT_NUM_SIG_DIGITS)) + 'j');
  } else if (isString(val)) {
    valStr = '\'' + val + '\'';
  } else if (dtype === 'bool') {
    valStr = boolNumToString(val);
  } else {
    valStr = parseFloat(val.toFixed(FORMAT_NUM_SIG_DIGITS)).toString();
  }
  return rightPad(valStr, pad);
}
function boolNumToString(v) {
  return v === 0 ? 'false' : 'true';
}
function subTensorToString(vals, shape, dtype, strides, padPerCol, isLast) {
  if (isLast === void 0) {
    isLast = true;
  }
  var storagePerElement = dtype === 'complex64' ? 2 : 1;
  var size = shape[0];
  var rank = shape.length;
  if (rank === 0) {
    if (dtype === 'complex64') {
      var complexTuple = createComplexTuples(vals);
      return [valToString(complexTuple[0], 0, dtype)];
    }
    if (dtype === 'bool') {
      return [boolNumToString(vals[0])];
    }
    return [vals[0].toString()];
  }
  if (rank === 1) {
    if (size > FORMAT_LIMIT_NUM_VALS) {
      var firstValsSize = FORMAT_NUM_FIRST_LAST_VALS * storagePerElement;
      var firstVals = Array.from(vals.slice(0, firstValsSize));
      var lastVals = Array.from(vals.slice(
          (size - FORMAT_NUM_FIRST_LAST_VALS) * storagePerElement,
          size * storagePerElement));
      if (dtype === 'complex64') {
        firstVals = createComplexTuples(firstVals);
        lastVals = createComplexTuples(lastVals);
      }
      return [
        '[' +
        firstVals
            .map(function(x, i) {
              return valToString(x, padPerCol[i], dtype);
            })
            .join(', ') +
        ', ..., ' +
        lastVals
            .map(function(x, i) {
              return valToString(
                  x, padPerCol[size - FORMAT_NUM_FIRST_LAST_VALS + i], dtype);
            })
            .join(', ') +
        ']'
      ];
    }
    var displayVals =
        dtype === 'complex64' ? createComplexTuples(vals) : Array.from(vals);
    return [
      '[' +
      displayVals
          .map(function(x, i) {
            return valToString(x, padPerCol[i], dtype);
          })
          .join(', ') +
      ']'
    ];
  }
  // The array is rank 2 or more.
  var subshape = shape.slice(1);
  var substrides = strides.slice(1);
  var stride = strides[0] * storagePerElement;
  var lines = [];
  if (size > FORMAT_LIMIT_NUM_VALS) {
    for (var i = 0; i < FORMAT_NUM_FIRST_LAST_VALS; i++) {
      var start = i * stride;
      var end = start + stride;
      lines.push.apply(
          lines,
          subTensorToString(
              vals.slice(start, end), subshape, dtype, substrides, padPerCol,
              false /* isLast */));
    }
    lines.push('...');
    for (var i = size - FORMAT_NUM_FIRST_LAST_VALS; i < size; i++) {
      var start = i * stride;
      var end = start + stride;
      lines.push.apply(
          lines,
          subTensorToString(
              vals.slice(start, end), subshape, dtype, substrides, padPerCol,
              i === size - 1 /* isLast */));
    }
  } else {
    for (var i = 0; i < size; i++) {
      var start = i * stride;
      var end = start + stride;
      lines.push.apply(
          lines,
          subTensorToString(
              vals.slice(start, end), subshape, dtype, substrides, padPerCol,
              i === size - 1 /* isLast */));
    }
  }
  var sep = rank === 2 ? ',' : '';
  lines[0] = '[' + lines[0] + sep;
  for (var i = 1; i < lines.length - 1; i++) {
    lines[i] = ' ' + lines[i] + sep;
  }
  var newLineSep = ',\n';
  for (var i = 2; i < rank; i++) {
    newLineSep += '\n';
  }
  lines[lines.length - 1] =
      ' ' + lines[lines.length - 1] + ']' + (isLast ? '' : newLineSep);
  return lines;
}
function createComplexTuples(vals) {
  var complexTuples = [];
  for (var i = 0; i < vals.length; i += 2) {
    complexTuples.push([vals[i], vals[i + 1]]);
  }
  return complexTuples;
}

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
var TensorBuffer = /** @class */ (function() {
  function TensorBuffer(shape, dtype, values) {
    var _this = this;
    this.dtype = dtype;
    this.shape = shape.slice();
    this.size = sizeFromShape(shape);
    if (values != null) {
      var n_1 = values.length;
      assert(n_1 === this.size, function() {
        return 'Length of values \'' + n_1 + '\' does not match the size ' +
            ('inferred by the shape \'' + _this.size + '\'.');
      });
    }
    if (dtype === 'complex64') {
      throw new Error(
          'complex64 dtype TensorBuffers are not supported. Please create ' +
          'a TensorBuffer for the real and imaginary parts separately and ' +
          'call tf.complex(real, imag).');
    }
    this.values = values || getArrayFromDType(dtype, this.size);
    this.strides = computeStrides(shape);
  }
  /**
   * Sets a value in the buffer at a given location.
   *
   * @param value The value to set.
   * @param locs  The location indices.
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
  TensorBuffer.prototype.set = function(value) {
    var _this = this;
    var locs = [];
    for (var _i = 1; _i < arguments.length; _i++) {
      locs[_i - 1] = arguments[_i];
    }
    if (locs.length === 0) {
      locs = [0];
    }
    assert(locs.length === this.rank, function() {
      return 'The number of provided coordinates (' + locs.length + ') must ' +
          ('match the rank (' + _this.rank + ')');
    });
    var index = this.locToIndex(locs);
    this.values[index] = value;
  };
  /**
   * Returns the value in the buffer at the provided location.
   *
   * @param locs The location indices.
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
  TensorBuffer.prototype.get = function() {
    var locs = [];
    for (var _i = 0; _i < arguments.length; _i++) {
      locs[_i] = arguments[_i];
    }
    if (locs.length === 0) {
      locs = [0];
    }
    var i = 0;
    for (var _a = 0, locs_1 = locs; _a < locs_1.length; _a++) {
      var loc = locs_1[_a];
      if (loc < 0 || loc >= this.shape[i]) {
        var msg = 'Requested out of range element at ' + locs + '. ' +
            ('  Buffer shape=' + this.shape);
        throw new Error(msg);
      }
      i++;
    }
    var index = locs[locs.length - 1];
    for (var i_1 = 0; i_1 < locs.length - 1; ++i_1) {
      index += this.strides[i_1] * locs[i_1];
    }
    return this.values[index];
  };
  TensorBuffer.prototype.locToIndex = function(locs) {
    if (this.rank === 0) {
      return 0;
    } else if (this.rank === 1) {
      return locs[0];
    }
    var index = locs[locs.length - 1];
    for (var i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  };
  TensorBuffer.prototype.indexToLoc = function(index) {
    if (this.rank === 0) {
      return [];
    } else if (this.rank === 1) {
      return [index];
    }
    var locs = new Array(this.shape.length);
    for (var i = 0; i < locs.length - 1; ++i) {
      locs[i] = Math.floor(index / this.strides[i]);
      index -= locs[i] * this.strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
  };
  Object.defineProperty(TensorBuffer.prototype, 'rank', {
    get: function() {
      return this.shape.length;
    },
    enumerable: true,
    configurable: true
  });
  /**
   * Creates an immutable `tf.Tensor` object from the buffer.
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
  TensorBuffer.prototype.toTensor = function() {
    return trackerFn().makeTensor(this.values, this.shape, this.dtype);
  };
  return TensorBuffer;
}());
// For tracking tensor creation and disposal.
var trackerFn = null;
// Used by chaining methods to call into ops.
var opHandler = null;
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
 * A `tf.Tensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * See `tf.tensor` for details on how to create a `tf.Tensor`.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
var Tensor = /** @class */ (function() {
  function Tensor(shape, dtype, dataId, id) {
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
  Object.defineProperty(Tensor.prototype, 'rank', {
    get: function() {
      return this.shape.length;
    },
    enumerable: true,
    configurable: true
  });
  /**
   * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.buffer = function() {
    return __awaiter(this, void 0, void 0, function() {
      var vals;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.data()];
          case 1:
            vals = _a.sent();
            return [
              2 /*return*/, opHandler.buffer(this.shape, this.dtype, vals)
            ];
        }
      });
    });
  };
  /**
   * Returns a `tf.TensorBuffer` that holds the underlying data.
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.bufferSync = function() {
    return opHandler.buffer(this.shape, this.dtype, this.dataSync());
  };
  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * asynchronously.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.array = function() {
    return __awaiter(this, void 0, void 0, function() {
      var vals;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.data()];
          case 1:
            vals = _a.sent();
            return [2 /*return*/, toNestedArray(this.shape, vals)];
        }
      });
    });
  };
  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * synchronously.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.arraySync = function() {
    return toNestedArray(this.shape, this.dataSync());
  };
  /**
   * Asynchronously downloads the values from the `tf.Tensor`. Returns a
   * promise of `TypedArray` that resolves when the computation has finished.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.data = function() {
    return __awaiter(this, void 0, void 0, function() {
      var data, bytes;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            this.throwIfDisposed();
            data = trackerFn().read(this.dataId);
            if (!(this.dtype === 'string')) return [3 /*break*/, 2];
            return [4 /*yield*/, data];
          case 1:
            bytes = _a.sent();
            try {
              return [
                2 /*return*/, bytes.map(function(b) {
                  return decodeString(b);
                })
              ];
            } catch (_b) {
              throw new Error(
                  'Failed to decode the string bytes into utf-8. ' +
                  'To get the original bytes, call tensor.bytes().');
            }
            _a.label = 2;
          case 2:
            return [2 /*return*/, data];
        }
      });
    });
  };
  /**
   * Synchronously downloads the values from the `tf.Tensor`. This blocks the
   * UI thread until the values are ready, which can cause performance issues.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.dataSync = function() {
    this.throwIfDisposed();
    var data = trackerFn().readSync(this.dataId);
    if (this.dtype === 'string') {
      try {
        return data.map(function(b) {
          return decodeString(b);
        });
      } catch (_a) {
        throw new Error(
            'Failed to decode the string bytes into utf-8. ' +
            'To get the original bytes, call tensor.bytes().');
      }
    }
    return data;
  };
  /** Returns the underlying bytes of the tensor's data. */
  Tensor.prototype.bytes = function() {
    return __awaiter(this, void 0, void 0, function() {
      var data;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            this.throwIfDisposed();
            return [4 /*yield*/, trackerFn().read(this.dataId)];
          case 1:
            data = _a.sent();
            if (this.dtype === 'string') {
              return [2 /*return*/, data];
            } else {
              return [2 /*return*/, new Uint8Array(data.buffer)];
            }
        }
      });
    });
  };
  /**
   * Disposes `tf.Tensor` from memory.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.dispose = function() {
    if (this.isDisposed) {
      return;
    }
    trackerFn().disposeTensor(this);
    this.isDisposedInternal = true;
  };
  Object.defineProperty(Tensor.prototype, 'isDisposed', {
    get: function() {
      return this.isDisposedInternal;
    },
    enumerable: true,
    configurable: true
  });
  Tensor.prototype.throwIfDisposed = function() {
    if (this.isDisposed) {
      throw new Error('Tensor is disposed.');
    }
  };
  /**
   * Prints the `tf.Tensor`. See `tf.print` for details.
   *
   * @param verbose Whether to print verbose information about the tensor,
   *    including dtype and size.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.print = function(verbose) {
    if (verbose === void 0) {
      verbose = false;
    }
    return opHandler.print(this, verbose);
  };
  /**
   * Returns a copy of the tensor. See `tf.clone` for details.
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.clone = function() {
    this.throwIfDisposed();
    return opHandler.clone(this);
  };
  /**
   * Returns a human-readable description of the tensor. Useful for logging.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Tensor.prototype.toString = function(verbose) {
    if (verbose === void 0) {
      verbose = false;
    }
    var vals = this.dataSync();
    return tensorToString(vals, this.shape, this.dtype, verbose);
  };
  Tensor.prototype.cast = function(dtype) {
    this.throwIfDisposed();
    return opHandler.cast(this, dtype);
  };
  Tensor.prototype.variable = function(trainable, name, dtype) {
    if (trainable === void 0) {
      trainable = true;
    }
    this.throwIfDisposed();
    return trackerFn().makeVariable(this, trainable, name, dtype);
  };
  return Tensor;
}());
Object.defineProperty(Tensor, Symbol.hasInstance, {
  value: function(instance) {
    // Implementation note: we should use properties of the object that will be
    // defined before the constructor body has finished executing (methods).
    // This is because when this code is transpiled by babel, babel will call
    // classCallCheck before the constructor body is run.
    // See https://github.com/tensorflow/tfjs/issues/3384 for backstory.
    return !!instance && instance.data != null && instance.dataSync != null &&
        instance.throwIfDisposed != null;
  }
});
function getGlobalTensorClass() {
  // Use getGlobal so that we can augment the Tensor class across package
  // boundaries becase the node resolution alg may result in different modules
  // being returned for this file depending on the path they are loaded from.
  return getGlobal('Tensor', function() {
    return Tensor;
  });
}
// Global side effect. Cache global reference to Tensor class
getGlobalTensorClass();
/**
 * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
var Variable = /** @class */ (function(_super) {
  __extends(Variable, _super);
  function Variable(initialValue, trainable, name, tensorId) {
    var _this = _super.call(
                    this, initialValue.shape, initialValue.dtype,
                    initialValue.dataId, tensorId) ||
        this;
    _this.trainable = trainable;
    _this.name = name;
    return _this;
  }
  /**
   * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
   * the same shape and dtype as the old `tf.Tensor`.
   *
   * @param newValue New tensor to be assigned to this variable.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  Variable.prototype.assign = function(newValue) {
    if (newValue.dtype !== this.dtype) {
      throw new Error(
          'dtype of the new value (' + newValue.dtype + ') and ' +
          ('previous value (' + this.dtype + ') must match'));
    }
    if (!arraysEqual(newValue.shape, this.shape)) {
      throw new Error(
          'shape of the new value (' + newValue.shape + ') and ' +
          ('previous value (' + this.shape + ') must match'));
    }
    trackerFn().disposeTensor(this);
    this.dataId = newValue.dataId;
    trackerFn().incRef(this, null /* backend */);
  };
  Variable.prototype.dispose = function() {
    trackerFn().disposeVariable(this);
    this.isDisposedInternal = true;
  };
  return Variable;
}(Tensor));
Object.defineProperty(Variable, Symbol.hasInstance, {
  value: function(instance) {
    return instance instanceof Tensor && instance.assign != null &&
        instance.assign instanceof Function;
  }
});

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
(function(Rank) {
Rank['R0'] = 'R0';
Rank['R1'] = 'R1';
Rank['R2'] = 'R2';
Rank['R3'] = 'R3';
Rank['R4'] = 'R4';
Rank['R5'] = 'R5';
Rank['R6'] = 'R6';
})(exports.Rank || (exports.Rank = {}));
// Looks for upcasting types. Used, for example, in operations with mixed dtype
// inputs.
var UpcastInt32AndMap;
(function(UpcastInt32AndMap) {
UpcastInt32AndMap['float32'] = 'float32';
UpcastInt32AndMap['int32'] = 'int32';
UpcastInt32AndMap['bool'] = 'int32';
UpcastInt32AndMap['complex64'] = 'complex64';
})(UpcastInt32AndMap || (UpcastInt32AndMap = {}));
var UpcastBoolAndMap;
(function(UpcastBoolAndMap) {
UpcastBoolAndMap['float32'] = 'float32';
UpcastBoolAndMap['int32'] = 'int32';
UpcastBoolAndMap['bool'] = 'bool';
UpcastBoolAndMap['complex64'] = 'complex64';
})(UpcastBoolAndMap || (UpcastBoolAndMap = {}));
var UpcastFloat32AndMap;
(function(UpcastFloat32AndMap) {
UpcastFloat32AndMap['float32'] = 'float32';
UpcastFloat32AndMap['int32'] = 'float32';
UpcastFloat32AndMap['bool'] = 'float32';
UpcastFloat32AndMap['complex64'] = 'complex64';
})(UpcastFloat32AndMap || (UpcastFloat32AndMap = {}));
var UpcastComplex64AndMap;
(function(UpcastComplex64AndMap) {
UpcastComplex64AndMap['float32'] = 'complex64';
UpcastComplex64AndMap['int32'] = 'complex64';
UpcastComplex64AndMap['bool'] = 'complex64';
UpcastComplex64AndMap['complex64'] = 'complex64';
})(UpcastComplex64AndMap || (UpcastComplex64AndMap = {}));
var upcastTypeMap = {
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
    throw new Error('Can not upcast ' + typeA + ' with ' + typeB);
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
  var dtype = upcastType(a.dtype, b.dtype);
  return [a.cast(dtype), b.cast(dtype)];
}
function assertTypesMatch(a, b) {
  assert(a.dtype === b.dtype, function() {
    return 'The dtypes of the first(' + a.dtype + ') and' +
        (' second(' + b.dtype + ') input must match');
  });
}
function isTensorInList(tensor, tensorList) {
  return tensorList.some(function(x) {
    return x.id === tensor.id;
  });
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
  var list = [];
  var seen = new Set();
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
  var iterable = container;
  for (var k in iterable) {
    var val = iterable[k];
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

var tensor_util = {
  __proto__: null,
  makeTypesMatch: makeTypesMatch,
  assertTypesMatch: assertTypesMatch,
  isTensorInList: isTensorInList,
  getTensorsInContainer: getTensorsInContainer
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
function isRegisteredKernelInvocation(kernelInvocation) {
  return kernelInvocation.kernelName != null;
}
var EngineState = /** @class */ (function() {
  function EngineState() {
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
    // Number of nested kernel calls. When kernel depth is greater than 1, we
    // turn off the tape.
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
    this.activeProfile = {
      newBytes: 0,
      newTensors: 0,
      peakBytes: 0,
      kernels: [],
      result: null,
      get kernelNames() {
        return Array.from(new Set(this.kernels.map(function(k) {
          return k.name;
        })));
      }
    };
  }
  EngineState.prototype.dispose = function() {
    for (var variableName in this.registeredVariables) {
      this.registeredVariables[variableName].dispose();
    }
  };
  return EngineState;
}());
var Engine = /** @class */ (function() {
  function Engine(ENV) {
    this.ENV = ENV;
    this.registry = {};
    this.registryFactory = {};
    this.pendingBackendInitId = 0;
    this.state = new EngineState();
  }
  Engine.prototype.ready = function() {
    return __awaiter(this, void 0, void 0, function() {
      var sortedBackends, i, backendName, success;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            if (this.pendingBackendInit != null) {
              return [
                2 /*return*/, this.pendingBackendInit.then(function() {})
              ];
            }
            if (this.backendInstance != null) {
              return [2 /*return*/];
            }
            sortedBackends = this.getSortedBackends();
            i = 0;
            _a.label = 1;
          case 1:
            if (!(i < sortedBackends.length)) return [3 /*break*/, 5];
            backendName = sortedBackends[i];
            return [4 /*yield*/, this.initializeBackend(backendName).success];
          case 2:
            success = _a.sent();
            if (!success) return [3 /*break*/, 4];
            return [4 /*yield*/, this.setBackend(backendName)];
          case 3:
            _a.sent();
            return [2 /*return*/];
          case 4:
            i++;
            return [3 /*break*/, 1];
          case 5:
            throw new Error(
                'Could not initialize any backends, all backend initializations ' +
                'failed.');
        }
      });
    });
  };
  Object.defineProperty(Engine.prototype, 'backend', {
    get: function() {
      if (this.pendingBackendInit != null) {
        throw new Error(
            'Backend \'' + this.backendName +
            '\' has not yet been initialized. Make ' +
            'sure to await tf.ready() or await tf.setBackend() before calling ' +
            'other methods');
      }
      if (this.backendInstance == null) {
        var _a = this.initializeBackendsAndReturnBest(), name_1 = _a.name,
            asyncInit = _a.asyncInit;
        if (asyncInit) {
          throw new Error(
              'The highest priority backend \'' + name_1 +
              '\' has not yet been ' +
              'initialized. Make sure to await tf.ready() or ' +
              'await tf.setBackend() before calling other methods');
        }
        this.setBackend(name_1);
      }
      return this.backendInstance;
    },
    enumerable: true,
    configurable: true
  });
  Engine.prototype.backendNames = function() {
    return Object.keys(this.registryFactory);
  };
  Engine.prototype.findBackend = function(backendName) {
    if (!(backendName in this.registry)) {
      // If the backend hasn't been initialized but we have a registry entry for
      // it, initialize it and return it.
      if (backendName in this.registryFactory) {
        var asyncInit = this.initializeBackend(backendName).asyncInit;
        if (asyncInit) {
          // Backend is not ready yet.
          return null;
        }
      } else {
        return null;
      }
    }
    return this.registry[backendName];
  };
  Engine.prototype.findBackendFactory = function(backendName) {
    if (!(backendName in this.registryFactory)) {
      return null;
    }
    return this.registryFactory[backendName].factory;
  };
  Engine.prototype.registerBackend = function(backendName, factory, priority) {
    if (priority === void 0) {
      priority = 1;
    }
    if (backendName in this.registryFactory) {
      console.warn(
          backendName + ' backend was already registered. ' +
          'Reusing existing backend factory.');
      return false;
    }
    this.registryFactory[backendName] = {factory: factory, priority: priority};
    return true;
  };
  Engine.prototype.setBackend = function(backendName) {
    return __awaiter(this, void 0, void 0, function() {
      var _a, success, asyncInit, result, _b;
      return __generator(this, function(_c) {
        switch (_c.label) {
          case 0:
            if (this.registryFactory[backendName] == null) {
              throw new Error(
                  'Backend name \'' + backendName + '\' not found in registry');
            }
            this.backendName = backendName;
            if (!(this.registry[backendName] == null)) return [3 /*break*/, 4];
            this.backendInstance = null;
            _a = this.initializeBackend(backendName), success = _a.success,
            asyncInit = _a.asyncInit;
            if (!asyncInit) return [3 /*break*/, 2];
            return [4 /*yield*/, success];
          case 1:
            _b = _c.sent();
            return [3 /*break*/, 3];
          case 2:
            _b = success;
            _c.label = 3;
          case 3:
            result = _b;
            if (!result) {
              return [2 /*return*/, false];
            }
            _c.label = 4;
          case 4:
            this.backendInstance = this.registry[backendName];
            this.setupRegisteredKernels();
            // Reset the profiler.
            this.profiler = new Profiler(this.backendInstance);
            return [2 /*return*/, true];
        }
      });
    });
  };
  Engine.prototype.setupRegisteredKernels = function() {
    var _this = this;
    var kernels = getKernelsForBackend(this.backendName);
    kernels.forEach(function(kernel) {
      if (kernel.setupFunc != null) {
        kernel.setupFunc(_this.backendInstance);
      }
    });
  };
  Engine.prototype.disposeRegisteredKernels = function(backendName) {
    var _this = this;
    var kernels = getKernelsForBackend(backendName);
    kernels.forEach(function(kernel) {
      if (kernel.disposeFunc != null) {
        kernel.disposeFunc(_this.registry[backendName]);
      }
    });
  };
  /**
   * Initializes a backend by looking up the backend name in the factory
   * registry and calling the factory method. Returns a boolean representing
   * whether the initialization of the backend suceeded. Throws an error if
   * there is no backend in the factory registry.
   */
  Engine.prototype.initializeBackend = function(backendName) {
    var _this = this;
    var registryFactoryEntry = this.registryFactory[backendName];
    if (registryFactoryEntry == null) {
      throw new Error(
          'Cannot initialize backend ' + backendName +
          ', no registration found.');
    }
    try {
      var backend = registryFactoryEntry.factory();
      /* Test if the factory returns a promise.
      Done in a more liberal way than
      previous 'Promise.resolve(backend)===backend'
      as we needed to account for custom Promise
      implementations (e.g. Angular) */
      if (backend && !(backend instanceof KernelBackend) &&
          typeof backend.then === 'function') {
        var promiseId_1 = ++this.pendingBackendInitId;
        var success =
            backend
                .then(function(backendInstance) {
                  // Outdated promise. Another backend was set in the meantime.
                  if (promiseId_1 < _this.pendingBackendInitId) {
                    return false;
                  }
                  _this.registry[backendName] = backendInstance;
                  _this.pendingBackendInit = null;
                  return true;
                })
                .catch(function(err) {
                  // Outdated promise. Another backend was set in the meantime.
                  if (promiseId_1 < _this.pendingBackendInitId) {
                    return false;
                  }
                  _this.pendingBackendInit = null;
                  console.warn(
                      'Initialization of backend ' + backendName + ' failed');
                  console.warn(err.stack || err.message);
                  return false;
                });
        this.pendingBackendInit = success;
        return {success: success, asyncInit: true};
      } else {
        this.registry[backendName] = backend;
        return {success: true, asyncInit: false};
      }
    } catch (err) {
      console.warn('Initialization of backend ' + backendName + ' failed');
      console.warn(err.stack || err.message);
      return {success: false, asyncInit: false};
    }
  };
  Engine.prototype.removeBackend = function(backendName) {
    if (!(backendName in this.registryFactory)) {
      throw new Error(backendName + ' backend not found in registry');
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
  };
  Engine.prototype.getSortedBackends = function() {
    var _this = this;
    if (Object.keys(this.registryFactory).length === 0) {
      throw new Error('No backend found in registry.');
    }
    return Object.keys(this.registryFactory).sort(function(a, b) {
      // Highest priority comes first.
      return _this.registryFactory[b].priority -
          _this.registryFactory[a].priority;
    });
  };
  Engine.prototype.initializeBackendsAndReturnBest = function() {
    var sortedBackends = this.getSortedBackends();
    for (var i = 0; i < sortedBackends.length; i++) {
      var backendName = sortedBackends[i];
      var _a = this.initializeBackend(backendName), success = _a.success,
          asyncInit = _a.asyncInit;
      if (asyncInit || success) {
        return {name: backendName, asyncInit: asyncInit};
      }
    }
    throw new Error(
        'Could not initialize any backends, all backend initializations ' +
        'failed.');
  };
  Engine.prototype.moveData = function(backend, dataId) {
    var info = this.state.tensorInfo.get(dataId);
    var srcBackend = info.backend;
    var values = this.readSync(dataId);
    var refCount = srcBackend.refCount(dataId);
    // Delete the tensor from the old backend and move it to the new
    // backend.
    srcBackend.disposeData(dataId, true);
    info.backend = backend;
    backend.move(dataId, values, info.shape, info.dtype, refCount);
    if (this.shouldCheckForMemLeaks()) {
      // Track the number of moves during a kernel execution to correctly
      // detect memory leaks.
      this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1]++;
    }
  };
  Engine.prototype.tidy = function(nameOrFn, fn) {
    var _this = this;
    var name = null;
    if (fn == null) {
      // Called with only 1 argument.
      if (typeof nameOrFn !== 'function') {
        throw new Error('Please provide a function to tidy()');
      }
      fn = nameOrFn;
    } else {
      // Called with 2 arguments.
      if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
        throw new Error(
            'When calling with two arguments, the first argument ' +
            'to tidy() must be a string');
      }
      if (typeof fn !== 'function') {
        throw new Error(
            'When calling with two arguments, the 2nd argument ' +
            'to tidy() must be a function');
      }
      name = nameOrFn;
      // TODO(nsthorat,smilkov): Do operation logging and performance
      // profiling.
    }
    var result;
    return this.scopedRun(
        function() {
          return _this.startScope(name);
        },
        function() {
          return _this.endScope(result);
        },
        function() {
          result = fn();
          if (result instanceof Promise) {
            console.error('Cannot return a Promise inside of tidy.');
          }
          return result;
        });
  };
  Engine.prototype.scopedRun = function(start, end, f) {
    start();
    try {
      var res = f();
      end();
      return res;
    } catch (ex) {
      end();
      throw ex;
    }
  };
  Engine.prototype.nextTensorId = function() {
    return Engine.nextTensorId++;
  };
  Engine.prototype.nextVariableId = function() {
    return Engine.nextVariableId++;
  };
  /**
   * This method is called instead of the public-facing tensor.clone() when
   * saving a tensor for backwards pass. It makes sure to add the clone
   * operation to the tape regardless of being called inside a kernel
   * execution.
   */
  Engine.prototype.clone = function(x) {
    var y = ENGINE.runKernel(Identity, {x: x});
    var inputs = {x: x};
    var grad = function(dy) {
      return ({
        x: function() {
          var dtype = 'float32';
          var gradInputs = {x: dy};
          var attrs = {dtype: dtype};
          return ENGINE.runKernel(
              Cast, gradInputs,
              // tslint:disable-next-line: no-unnecessary-type-assertion
              attrs);
        }
      });
    };
    var saved = [];
    this.addTapeNode(this.state.activeScope.name, inputs, [y], grad, saved, {});
    return y;
  };
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
  Engine.prototype.runKernel = function(kernelName, inputs, attrs) {
    var hasKernel = getKernel(kernelName, this.backendName) != null;
    if (!hasKernel) {
      throw new Error(
          'Kernel \'' + kernelName + '\' not registered for backend \'' +
          this.backendName + '\'');
    }
    return this.runKernelFunc(
        {kernelName: kernelName, inputs: inputs, attrs: attrs});
  };
  Engine.prototype.shouldCheckForMemLeaks = function() {
    return this.ENV.getBool('IS_TEST');
  };
  Engine.prototype.checkKernelForMemLeak = function(
      kernelName, numDataIdsBefore, outInfos) {
    var numDataIdsAfter = this.backend.numDataIds();
    // Count the number of data ids associated with the result of the kernel.
    var numOutputDataIds = 0;
    outInfos.forEach(function(info) {
      // Complex numbers allocate 3 data ids, one for 'real', one for
      // 'imaginary', and one for the container that holds the former two.
      numOutputDataIds += (info.dtype === 'complex64' ? 3 : 1);
    });
    // Account for the number of moves during kernel execution. A "data move"
    // can happen in the middle of a kernel execution, placing a new (key,value)
    // pair in the data storage. Since data moves have net zero effect (we
    // always remove the data from the old backend), we have to cancel them out
    // when detecting memory leaks.
    var numMoves =
        this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1];
    var dataIdsLeaked =
        numDataIdsAfter - numDataIdsBefore - numOutputDataIds - numMoves;
    if (dataIdsLeaked > 0) {
      throw new Error(
          'Backend \'' + this.backendName + '\' has an internal memory leak ' +
          ('(' + dataIdsLeaked + ' data ids) after running \'' + kernelName +
           '\''));
    }
  };
  /**
   * Internal helper method to execute a kernel Func
   *
   * Use `runKernel` to execute kernels from outside of engine.
   */
  Engine.prototype.runKernelFunc = function(kernelParams) {
    var _this = this;
    var outputs;
    var saved = [];
    var isTapeOn = this.isTapeOn();
    var startingBytecount = this.state.numBytes;
    var startingNumTensors = this.state.numTensors;
    if (this.shouldCheckForMemLeaks()) {
      this.state.numDataMovesStack.push(0);
    }
    var kernelFunc;
    if (this.backendName == null) {
      // backend has not been initialized yet (backend initialization is lazy
      // can be deferred until an op/ kernel is run).
      // The below getter has side effects that will try to initialize the
      // backend and set properties like this.backendName
      // tslint:disable-next-line: no-unused-expression
      this.backend;
    }
    var out;
    var kernelOrScopeName = isRegisteredKernelInvocation(kernelParams) ?
        kernelParams.kernelName :
        this.state.activeScope != null ? this.state.activeScope.name : '';
    // Create the kernelFunc from either a registered kernel OR passed in
    // forward/backward functions (used by custom grad). In this context a
    // kernelFunc wraps a kernel implementation with some bookkeeping.
    if (isRegisteredKernelInvocation(kernelParams)) {
      var kernelName_1 = kernelParams.kernelName,
          inputs_1 = kernelParams.inputs, attrs_1 = kernelParams.attrs;
      if (this.backendName == null) {
        // backend has not been initialized yet (backend initialization is lazy
        // can be deferred until an op/ kernel is run).
        // The below getter has side effects that will try to initialize the
        // backend and set properties like this.backendName
        // tslint:disable-next-line: no-unused-expression
        this.backend;
      }
      var kernel_1 = getKernel(kernelName_1, this.backendName);
      assert(kernel_1 != null, function() {
        return 'Cannot find registered kernel \'' + kernelName_1 +
            '\' for backend \'' + _this.backendName + '\'';
      });
      kernelFunc = function() {
        var numDataIdsBefore = _this.backend.numDataIds();
        out = kernel_1.kernelFunc(
            {inputs: inputs_1, attrs: attrs_1, backend: _this.backend});
        var outInfos = Array.isArray(out) ? out : [out];
        if (_this.shouldCheckForMemLeaks()) {
          _this.checkKernelForMemLeak(kernelName_1, numDataIdsBefore, outInfos);
        }
        var outTensors = outInfos.map(function(outInfo) {
          // todo (yassogba) remove this option (Tensor) when node backend
          // methods have been modularized and they all return tensorInfo.
          // TensorInfos do not have a rank attribute.
          if (outInfo.rank != null) {
            return outInfo;
          }
          var _a = outInfo, dataId = _a.dataId, shape = _a.shape,
              dtype = _a.dtype;
          return _this.makeTensorFromDataId(dataId, shape, dtype);
        });
        // Save any required inputs and outputs.
        // Do not save unless we are recording to the tape. Otherwise it would
        // cause a mem leak since there would be no backprop for these tensors
        // (which would otherwise dispose them).
        if (isTapeOn) {
          var tensorsToSave =
              _this.getTensorsForGradient(kernelName_1, inputs_1, outTensors);
          saved = _this.saveTensorsForBackwardMode(tensorsToSave);
        }
        return outTensors;
      };
    } else {
      var forwardFunc_1 = kernelParams.forwardFunc;
      // Running a customGrad op.
      var saveFunc_1 = function(tensors) {
        // Do not save unless we are recording to the tape. Otherwise it would
        // cause a mem leak since we would never run backprop, which disposes
        // the kept tensors.
        if (!isTapeOn) {
          return;
        }
        saved = tensors.map(function(tensor) {
          return _this.keep(_this.clone(tensor));
        });
      };
      kernelFunc = function() {
        var numDataIdsBefore = _this.backend.numDataIds();
        out = _this.tidy(function() {
          return forwardFunc_1(_this.backend, saveFunc_1);
        });
        var outs = (Array.isArray(out) ? out : [out]);
        if (_this.shouldCheckForMemLeaks()) {
          // Scope name is used to print a more helpful error message if needed.
          _this.checkKernelForMemLeak(
              kernelOrScopeName, numDataIdsBefore, outs);
        }
        return outs;
      };
    }
    //
    // Run the kernelFunc. Optionally profiling it.
    //
    var inputs = kernelParams.inputs, attrs = kernelParams.attrs;
    var backwardsFunc = isRegisteredKernelInvocation(kernelParams) ?
        null :
        kernelParams.backwardsFunc;
    var kernelProfile;
    this.scopedRun(
        // Stop recording to a tape when running a kernel.
        function() {
          return _this.state.kernelDepth++;
        },
        function() {
          return _this.state.kernelDepth--;
        },
        function() {
          if (!_this.ENV.getBool('DEBUG') && !_this.state.profiling) {
            outputs = kernelFunc();
          } else {
            kernelProfile = _this.profiler.profileKernel(
                kernelOrScopeName, inputs, function() {
                  return kernelFunc();
                });
            if (_this.ENV.getBool('DEBUG')) {
              _this.profiler.logKernelProfile(kernelProfile);
            }
            outputs = kernelProfile.outputs;
          }
        });
    if (isTapeOn) {
      this.addTapeNode(
          kernelOrScopeName, inputs, outputs, backwardsFunc, saved, attrs);
    }
    if (this.state.profiling) {
      this.state.activeProfile.kernels.push({
        name: kernelOrScopeName,
        bytesAdded: this.state.numBytes - startingBytecount,
        totalBytesSnapshot: this.state.numBytes,
        tensorsAdded: this.state.numTensors - startingNumTensors,
        totalTensorsSnapshot: this.state.numTensors,
        inputShapes: Object.keys(inputs).map(function(key) {
          return inputs[key] != null ? inputs[key].shape : null;
        }),
        outputShapes: outputs.map(function(item) {
          return item.shape;
        }),
        kernelTimeMs: kernelProfile.timeMs,
        extraInfo: kernelProfile.extraInfo
      });
    }
    return (Array.isArray(out) ? outputs : outputs[0]);
  };
  /**
   * Saves tensors used in forward mode for use in backward mode.
   *
   * @param tensors the list of tensors to save.
   */
  Engine.prototype.saveTensorsForBackwardMode = function(tensors) {
    var _this = this;
    var saved = tensors.map(function(tensor) {
      return _this.keep(_this.clone(tensor));
    });
    return saved;
  };
  /**
   * Returns a list of tensors to save for a given gradient calculation.
   *
   * @param kernelName name of kernel to look up gradient for.
   * @param inputs a map of input tensors.
   * @param outputs an array of output tensors from forward mode of kernel.
   */
  Engine.prototype.getTensorsForGradient = function(
      kernelName, inputs, outputs) {
    var gradConfig = getGradient(kernelName);
    if (gradConfig != null) {
      var inputsToSave = gradConfig.inputsToSave || [];
      var outputsToSave_1 = gradConfig.outputsToSave || [];
      // If saveAllInputs is true, all inputs will be saved. Otherwise, inputs
      // specified in inputsToSave will be saved.
      var inputTensorsToSave = void 0;
      if (gradConfig.saveAllInputs) {
        assert(Array.isArray(inputs), function() {
          return 'saveAllInputs is true, expected inputs to be an array.';
        });
        inputTensorsToSave = Object.keys(inputs).map(function(key) {
          return inputs[key];
        });
      } else {
        inputTensorsToSave = inputsToSave.map(function(inputName) {
          return inputs[inputName];
        });
      }
      var outputTensorsToSave = outputs.filter(function(_, i) {
        return outputsToSave_1[i];
      });
      return inputTensorsToSave.concat(outputTensorsToSave);
    }
    // We return an empty list rather than throw an error because the kernel we
    // are looking up may not actually be relevant to backproping through the
    // overall function
    //
    // See 'does not error if irrelevant (pruned) ops are missing grads' test
    // in gradients_test.ts for an example.
    return [];
  };
  /**
   * Internal method used by public APIs for tensor creation. Makes a new
   * tensor with the provided shape, dtype and values. It always
   * creates a new data id and writes the values to the underlying backend.
   */
  Engine.prototype.makeTensor = function(values, shape, dtype, backend) {
    if (values == null) {
      throw new Error('Values passed to engine.makeTensor() are null');
    }
    dtype = dtype || 'float32';
    backend = backend || this.backend;
    var backendVals = values;
    if (dtype === 'string' && isString(values[0])) {
      backendVals = values.map(function(d) {
        return encodeString(d);
      });
    }
    var dataId = backend.write(backendVals, shape, dtype);
    var t = new Tensor(shape, dtype, dataId, this.nextTensorId());
    this.trackTensor(t, backend);
    // Count bytes for string tensors.
    if (dtype === 'string') {
      var info = this.state.tensorInfo.get(dataId);
      var newBytes = bytesFromStringArray(backendVals);
      this.state.numBytes += newBytes - info.bytes;
      info.bytes = newBytes;
    }
    return t;
  };
  /**
   * Internal method used by backends. Makes a new tensor
   * that is a wrapper around an existing data id. It doesn't create
   * a new data id, only increments the ref count used in memory tracking.
   */
  Engine.prototype.makeTensorFromDataId = function(
      dataId, shape, dtype, backend) {
    dtype = dtype || 'float32';
    var t = new Tensor(shape, dtype, dataId, this.nextTensorId());
    this.trackTensor(t, backend);
    return t;
  };
  Engine.prototype.makeVariable = function(
      initialValue, trainable, name, dtype) {
    if (trainable === void 0) {
      trainable = true;
    }
    name = name || this.nextVariableId().toString();
    if (dtype != null && dtype !== initialValue.dtype) {
      initialValue = initialValue.cast(dtype);
    }
    var v = new Variable(initialValue, trainable, name, this.nextTensorId());
    if (this.state.registeredVariables[v.name] != null) {
      throw new Error(
          'Variable with name ' + v.name + ' was already registered');
    }
    this.state.registeredVariables[v.name] = v;
    this.incRef(v, this.backend);
    return v;
  };
  Engine.prototype.trackTensor = function(a, backend) {
    this.state.numTensors++;
    if (a.dtype === 'string') {
      this.state.numStringTensors++;
    }
    // Bytes for complex numbers are counted by their components. Bytes for
    // string tensors are counted when writing values.
    var bytes = 0;
    if (a.dtype !== 'complex64' && a.dtype !== 'string') {
      bytes = a.size * bytesPerElement(a.dtype);
    }
    this.state.numBytes += bytes;
    if (!this.state.tensorInfo.has(a.dataId)) {
      this.state.numDataBuffers++;
      this.state.tensorInfo.set(a.dataId, {
        backend: backend || this.backend,
        dtype: a.dtype,
        shape: a.shape,
        bytes: bytes
      });
    }
    if (!(a instanceof Variable)) {
      this.track(a);
    }
  };
  // Track the tensor by dataId and increase the refCount for the dataId in the
  // backend.
  // TODO(pyu10055): This is currently used by makeVariable method, to increase
  // refCount on the backend for the dataId. It can potentially be replaced with
  // Identity op indead of calling backend directly.
  Engine.prototype.incRef = function(a, backend) {
    this.trackTensor(a, backend);
    this.backend.incRef(a.dataId);
  };
  Engine.prototype.removeDataId = function(dataId, backend) {
    if (this.state.tensorInfo.has(dataId) &&
        this.state.tensorInfo.get(dataId).backend === backend) {
      this.state.tensorInfo.delete(dataId);
      this.state.numDataBuffers--;
    }
  };
  Engine.prototype.disposeTensor = function(a) {
    if (!this.state.tensorInfo.has(a.dataId)) {
      return;
    }
    var info = this.state.tensorInfo.get(a.dataId);
    this.state.numTensors--;
    if (a.dtype === 'string') {
      this.state.numStringTensors--;
      this.state.numBytes -= info.bytes;
    }
    // Don't count bytes for complex numbers as they are counted by their
    // components.
    if (a.dtype !== 'complex64' && a.dtype !== 'string') {
      var bytes = a.size * bytesPerElement(a.dtype);
      this.state.numBytes -= bytes;
    }
    // Remove the reference to dataId if backend dispose the data successfully
    if (info.backend.disposeData(a.dataId)) {
      this.removeDataId(a.dataId, info.backend);
    }
    // TODO(nsthorat): Construct an error and save the stack trace for
    // debugging when in debug mode. Creating a stack trace is too expensive
    // to do unconditionally.
  };
  Engine.prototype.disposeVariables = function() {
    for (var varName in this.state.registeredVariables) {
      var v = this.state.registeredVariables[varName];
      this.disposeVariable(v);
    }
  };
  Engine.prototype.disposeVariable = function(v) {
    this.disposeTensor(v);
    if (this.state.registeredVariables[v.name] != null) {
      delete this.state.registeredVariables[v.name];
    }
  };
  Engine.prototype.memory = function() {
    var info = this.backend.memory();
    info.numTensors = this.state.numTensors;
    info.numDataBuffers = this.state.numDataBuffers;
    info.numBytes = this.state.numBytes;
    if (this.state.numStringTensors > 0) {
      info.unreliable = true;
      if (info.reasons == null) {
        info.reasons = [];
      }
      info.reasons.push(
          'Memory usage by string tensors is approximate ' +
          '(2 bytes per character)');
    }
    return info;
  };
  Engine.prototype.profile = function(query) {
    return __awaiter(this, void 0, void 0, function() {
      var startBytes, startNumTensors, _a, _i, _b, kernel, _c, _d;
      return __generator(this, function(_e) {
        switch (_e.label) {
          case 0:
            this.state.profiling = true;
            startBytes = this.state.numBytes;
            startNumTensors = this.state.numTensors;
            this.state.activeProfile.kernels = [];
            _a = this.state.activeProfile;
            return [4 /*yield*/, query()];
          case 1:
            _a.result = _e.sent();
            this.state.profiling = false;
            this.state.activeProfile.peakBytes = Math.max.apply(
                Math, this.state.activeProfile.kernels.map(function(d) {
                  return d.totalBytesSnapshot;
                }));
            this.state.activeProfile.newBytes =
                this.state.numBytes - startBytes;
            this.state.activeProfile.newTensors =
                this.state.numTensors - startNumTensors;
            _i = 0, _b = this.state.activeProfile.kernels;
            _e.label = 2;
          case 2:
            if (!(_i < _b.length)) return [3 /*break*/, 6];
            kernel = _b[_i];
            _c = kernel;
            return [4 /*yield*/, kernel.kernelTimeMs];
          case 3:
            _c.kernelTimeMs = _e.sent();
            _d = kernel;
            return [4 /*yield*/, kernel.extraInfo];
          case 4:
            _d.extraInfo = _e.sent();
            _e.label = 5;
          case 5:
            _i++;
            return [3 /*break*/, 2];
          case 6:
            return [2 /*return*/, this.state.activeProfile];
        }
      });
    });
  };
  Engine.prototype.isTapeOn = function() {
    return this.state.gradientDepth > 0 && this.state.kernelDepth === 0;
  };
  Engine.prototype.addTapeNode = function(
      kernelName, inputs, outputs, gradientsFunc, saved, attrs) {
    var _this = this;
    var tapeNode = {
      id: this.state.nextTapeNodeId++,
      kernelName: kernelName,
      inputs: inputs,
      outputs: outputs,
      saved: saved
    };
    var gradConfig = getGradient(kernelName);
    if (gradConfig != null) {
      gradientsFunc = gradConfig.gradFunc;
    }
    if (gradientsFunc != null) {
      tapeNode.gradient = function(dys) {
        // TODO(smilkov): To optimize back-prop, pass dys that are not used in
        // the backprop graph to the user as null instead of zeros
        dys = dys.map(function(dy, i) {
          if (dy == null) {
            var output = outputs[i];
            var vals = makeZerosTypedArray(output.size, output.dtype);
            return _this.makeTensor(vals, output.shape, output.dtype);
          }
          return dy;
        });
        // Grad functions of ops with single outputs expect a dy, while ops
        // with multiple outputs expect dys (array of dy).
        return gradientsFunc(dys.length > 1 ? dys : dys[0], saved, attrs);
      };
    }
    this.state.activeTape.push(tapeNode);
  };
  Engine.prototype.keep = function(result) {
    result.kept = true;
    return result;
  };
  Engine.prototype.startTape = function() {
    if (this.state.gradientDepth === 0) {
      this.state.activeTape = [];
    }
    this.state.gradientDepth++;
  };
  Engine.prototype.endTape = function() {
    this.state.gradientDepth--;
  };
  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  Engine.prototype.startScope = function(name) {
    var scopeInfo = {
      track: [],
      name: 'unnamed scope',
      id: this.state.nextScopeId++
    };
    if (name) {
      scopeInfo.name = name;
    }
    this.state.scopeStack.push(scopeInfo);
    this.state.activeScope = scopeInfo;
  };
  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  Engine.prototype.endScope = function(result) {
    var _this = this;
    var tensorsToTrackInParent = getTensorsInContainer(result);
    var tensorsToTrackInParentSet =
        new Set(tensorsToTrackInParent.map(function(t) {
          return t.id;
        }));
    // Dispose the arrays tracked in this scope.
    for (var i = 0; i < this.state.activeScope.track.length; i++) {
      var tensor = this.state.activeScope.track[i];
      if (!tensor.kept && !tensorsToTrackInParentSet.has(tensor.id)) {
        tensor.dispose();
      }
    }
    var oldScope = this.state.scopeStack.pop();
    this.state.activeScope = this.state.scopeStack.length === 0 ?
        null :
        this.state.scopeStack[this.state.scopeStack.length - 1];
    // Track the current result in the parent scope.
    tensorsToTrackInParent.forEach(function(tensor) {
      // Only track the tensor if was allocated in the inner scope and is not
      // globally kept.
      if (!tensor.kept && tensor.scopeId === oldScope.id) {
        _this.track(tensor);
      }
    });
  };
  /**
   * Returns gradients of `f` with respect to each of the `xs`. The gradients
   * returned are of the same length as `xs`, but some might be null if `f`
   * was not a function of that `x`. It also takes optional dy to multiply the
   * gradient, which defaults to `1`.
   */
  Engine.prototype.gradients = function(f, xs, dy, allowNoGradients) {
    var _this = this;
    if (allowNoGradients === void 0) {
      allowNoGradients = false;
    }
    assert(xs.length > 0, function() {
      return 'gradients() received an empty list of xs.';
    });
    if (dy != null && dy.dtype !== 'float32') {
      throw new Error(
          'dy must have \'float32\' dtype, but has \'' + dy.dtype + '\'');
    }
    var y = this.scopedRun(
        function() {
          return _this.startTape();
        },
        function() {
          return _this.endTape();
        },
        function() {
          return _this.tidy('forward', f);
        });
    assert(y instanceof Tensor, function() {
      return 'The result y returned by f() must be a tensor.';
    });
    // Filter out the nodes that don't connect x => y.
    var filteredTape = getFilteredNodesXToY(this.state.activeTape, xs, y);
    if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
      throw new Error(
          'Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
          'that the f you passed encloses all operations that lead from x ' +
          'to y.');
    }
    return this.tidy('backward', function() {
      var accumulatedGradientMap = {};
      accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;
      // Backprop gradients through the filtered nodes.
      backpropagateGradients(
          accumulatedGradientMap, filteredTape,
          // Pass the tidy function to avoid circular dep with `tape.ts`.
          function(f) {
            return _this.tidy(f);
          },
          // Pass an add function to avoide a circular dep with `tape.ts`.
          add);
      var grads = xs.map(function(x) {
        return accumulatedGradientMap[x.id];
      });
      if (_this.state.gradientDepth === 0) {
        // This means that we are not computing higher-order gradients
        // and can clean up the tape.
        _this.state.activeTape.forEach(function(node) {
          for (var _i = 0, _a = node.saved; _i < _a.length; _i++) {
            var tensor = _a[_i];
            tensor.dispose();
          }
        });
        _this.state.activeTape = null;
      }
      return {value: y, grads: grads};
    });
  };
  Engine.prototype.customGrad = function(f) {
    var _this = this;
    assert(isFunction(f), function() {
      return 'The f passed in customGrad(f) must be a function.';
    });
    return function() {
      var inputs = [];
      for (var _i = 0; _i < arguments.length; _i++) {
        inputs[_i] = arguments[_i];
      }
      assert(
          inputs.every(function(t) {
            return t instanceof Tensor;
          }),
          function() {
            return 'The args passed in customGrad(f)(x1, x2,...) must all be ' +
                'tensors';
          });
      var res;
      var inputMap = {};
      inputs.forEach(function(input, i) {
        inputMap[i] = input;
      });
      var forwardFunc = function(_, save) {
        res = f.apply(void 0, inputs.concat([save]));
        assert(res.value instanceof Tensor, function() {
          return 'The function f passed in customGrad(f) must return an ' +
              'object where `obj.value` is a tensor';
        });
        assert(isFunction(res.gradFunc), function() {
          return 'The function f passed in customGrad(f) must return an ' +
              'object where `obj.gradFunc` is a function.';
        });
        return res.value;
      };
      var backwardsFunc = function(dy, saved) {
        var gradRes = res.gradFunc(dy, saved);
        var grads = Array.isArray(gradRes) ? gradRes : [gradRes];
        assert(grads.length === inputs.length, function() {
          return 'The function f passed in customGrad(f) must return an ' +
              'object where `obj.gradFunc` is a function that returns ' +
              'the same number of tensors as inputs passed to f(...).';
        });
        assert(
            grads.every(function(t) {
              return t instanceof Tensor;
            }),
            function() {
              return 'The function f passed in customGrad(f) must return an ' +
                  'object where `obj.gradFunc` is a function that returns ' +
                  'a list of only tensors.';
            });
        var gradMap = {};
        grads.forEach(function(grad, i) {
          gradMap[i] = function() {
            return grad;
          };
        });
        return gradMap;
      };
      return _this.runKernelFunc({
        forwardFunc: forwardFunc,
        backwardsFunc: backwardsFunc,
        inputs: inputMap,
      });
    };
  };
  Engine.prototype.readSync = function(dataId) {
    // Route the read to the correct backend.
    var info = this.state.tensorInfo.get(dataId);
    return info.backend.readSync(dataId);
  };
  Engine.prototype.read = function(dataId) {
    // Route the read to the correct backend.
    var info = this.state.tensorInfo.get(dataId);
    return info.backend.read(dataId);
  };
  Engine.prototype.time = function(query) {
    return __awaiter(this, void 0, void 0, function() {
      var start, timingInfo;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            start = now();
            return [4 /*yield*/, this.backend.time(query)];
          case 1:
            timingInfo = _a.sent();
            timingInfo.wallMs = now() - start;
            return [2 /*return*/, timingInfo];
        }
      });
    });
  };
  /**
   * Tracks a Tensor in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The Tensor to track in the current scope.
   */
  Engine.prototype.track = function(result) {
    if (this.state.activeScope != null) {
      result.scopeId = this.state.activeScope.id;
      this.state.activeScope.track.push(result);
    }
    return result;
  };
  Object.defineProperty(Engine.prototype, 'registeredVariables', {
    get: function() {
      return this.state.registeredVariables;
    },
    enumerable: true,
    configurable: true
  });
  /**
   * Resets the engine state. Removes all backends but does not remove
   * registered backend factories.
   */
  Engine.prototype.reset = function() {
    // Make any pending promise obsolete.
    this.pendingBackendInitId++;
    this.state.dispose();
    this.ENV.reset();
    this.state = new EngineState();
    for (var backendName in this.registry) {
      this.disposeRegisteredKernels(backendName);
      this.registry[backendName].dispose();
      delete this.registry[backendName];
    }
    this.backendName = null;
    this.backendInstance = null;
    this.pendingBackendInit = null;
  };
  Engine.nextTensorId = 0;
  Engine.nextVariableId = 0;
  return Engine;
}());
function ones(shape) {
  var values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
  return ENGINE.makeTensor(values, shape, 'float32');
}
function getOrMakeEngine() {
  var ns = getGlobalNamespace();
  if (ns._tfengine == null) {
    var environment = new Environment(ns);
    ns._tfengine = new Engine(environment);
  }
  setEnvironmentGlobal(ns._tfengine.ENV);
  // Tell the current tensor interface that the global engine is responsible
  // for tracking.
  setTensorTracker(function() {
    return ns._tfengine;
  });
  return ns._tfengine;
}
var ENGINE = getOrMakeEngine();
/**
 * A implementation of the add op for use within engine and tape.
 *
 * This allows us to avoid a circular dependency between add.ts and engine.
 * It is exported to be available in tape tests.
 */
function add(a, b) {
  // We duplicate Add here to avoid a circular dependency with add.ts.
  var inputs = {a: a, b: b};
  return ENGINE.runKernel(Add, inputs);
}

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
    var a = navigator.userAgent || navigator.vendor || window.opera;
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

var device_util = {__proto__: null, isMobile: isMobile, isBrowser: isBrowser};

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
var ENV = env();
/**
 * This file contains environment-related flag registrations.
 */
/** Whether to enable debug mode. */
ENV.registerFlag(
    'DEBUG',
    function() {
      return false;
    },
    function(debugValue) {
      if (debugValue) {
        console.warn(
            'Debugging mode is ON. The output of every math call will ' +
            'be downloaded to CPU and checked for NaNs. ' +
            'This significantly impacts performance.');
      }
    });
/** Whether we are in a browser (as versus, say, node.js) environment. */
ENV.registerFlag('IS_BROWSER', function() {
  return isBrowser();
});
/** Whether we are in a browser (as versus, say, node.js) environment. */
ENV.registerFlag('IS_NODE', function() {
  return (typeof process !== 'undefined') &&
      (typeof process.versions !== 'undefined') &&
      (typeof process.versions.node !== 'undefined');
});
/** Whether this browser is Chrome. */
ENV.registerFlag('IS_CHROME', function() {
  return typeof navigator !== 'undefined' && navigator != null &&
      navigator.userAgent != null && /Chrome/.test(navigator.userAgent) &&
      /Google Inc/.test(navigator.vendor);
});
/**
 * True when the environment is "production" where we disable safety checks
 * to gain performance.
 */
ENV.registerFlag('PROD', function() {
  return false;
});
/**
 * Whether to do sanity checks when inferring a shape from user-provided
 * values, used when creating a new tensor.
 */
ENV.registerFlag('TENSORLIKE_CHECK_SHAPE_CONSISTENCY', function() {
  return ENV.getBool('DEBUG');
});
/** Whether deprecation warnings are enabled. */
ENV.registerFlag('DEPRECATION_WARNINGS_ENABLED', function() {
  return true;
});
/** True if running unit tests. */
ENV.registerFlag('IS_TEST', function() {
  return false;
});
/** Whether to check computation result for errors. */
ENV.registerFlag('CHECK_COMPUTATION_FOR_ERRORS', function() {
  return true;
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
function inferShape(val, dtype) {
  var firstElem = val;
  if (isTypedArray(val)) {
    return dtype === 'string' ? [] : [val.length];
  }
  if (!Array.isArray(val)) {
    return [];  // Scalar.
  }
  var shape = [];
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
    assert(shape.length === 0, function() {
      return 'Element arr[' + indices.join('][') + '] is a primitive, ' +
          ('but should be an array/TypedArray of ' + shape[0] + ' elements');
    });
    return;
  }
  assert(shape.length > 0, function() {
    return 'Element arr[' + indices.join('][') + '] should be a primitive, ' +
        ('but is an array of ' + val.length + ' elements');
  });
  assert(val.length === shape[0], function() {
    return 'Element arr[' + indices.join('][') + '] should have ' + shape[0] +
        ' ' + ('elements, but has ' + val.length + ' elements');
  });
  var subShape = shape.slice(1);
  for (var i = 0; i < val.length; ++i) {
    deepAssertShapeConsistency(val[i], subShape, indices.concat(i));
  }
}
function assertDtype(expectedDtype, actualDType, argName, functionName) {
  if (expectedDtype === 'string_or_numeric') {
    return;
  }
  if (expectedDtype == null) {
    throw new Error('Expected dtype cannot be null.');
  }
  if (expectedDtype !== 'numeric' && expectedDtype !== actualDType ||
      expectedDtype === 'numeric' && actualDType === 'string') {
    throw new Error(
        'Argument \'' + argName + '\' passed to \'' + functionName +
        '\' must ' +
        ('be ' + expectedDtype + ' tensor, but got ' + actualDType +
         ' tensor'));
  }
}
function convertToTensor(x, argName, functionName, parseAsDtype) {
  if (parseAsDtype === void 0) {
    parseAsDtype = 'numeric';
  }
  if (x instanceof Tensor) {
    assertDtype(parseAsDtype, x.dtype, argName, functionName);
    return x;
  }
  var inferredDtype = inferDtype(x);
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
    var type = x == null ? 'null' : x.constructor.name;
    throw new Error(
        'Argument \'' + argName + '\' passed to \'' + functionName +
        '\' must be a ' + ('Tensor or TensorLike, but got \'' + type + '\''));
  }
  var inferredShape = inferShape(x, inferredDtype);
  if (!isTypedArray(x) && !Array.isArray(x)) {
    x = [x];
  }
  var skipTypedArray = true;
  var values = inferredDtype !== 'string' ? toTypedArray(x, inferredDtype) :
                                            flatten(x, [], skipTypedArray);
  return ENGINE.makeTensor(values, inferredShape, inferredDtype);
}
function convertToTensorArray(arg, argName, functionName, parseAsDtype) {
  if (parseAsDtype === void 0) {
    parseAsDtype = 'numeric';
  }
  if (!Array.isArray(arg)) {
    throw new Error(
        'Argument ' + argName + ' passed to ' + functionName + ' must be a ' +
        '`Tensor[]` or `TensorLike[]`');
  }
  var tensors = arg;
  return tensors.map(function(t, i) {
    return convertToTensor(
        t, argName + '[' + i + ']', functionName, parseAsDtype);
  });
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
var OP_SCOPE_SUFFIX = '__op';
/**
 * Used for wrapping functions that perform math operations on
 * Tensors. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
function op(f) {
  var keys = Object.keys(f);
  if (keys.length !== 1) {
    throw new Error(
        'Please provide an object with a single key ' +
        '(operation name) mapping to a function. Got an object with ' +
        (keys.length + ' keys.'));
  }
  var opName = keys[0];
  var fn = f[opName];
  // Strip the underscore from the end of the function name.
  if (opName.endsWith('_')) {
    opName = opName.substring(0, opName.length - 1);
  }
  // add an __op suffix to distinguish ops from kernels in tf.profile
  opName = opName + OP_SCOPE_SUFFIX;
  // tslint:disable-next-line:no-any
  var f2 = function() {
    var args = [];
    for (var _i = 0; _i < arguments.length; _i++) {
      args[_i] = arguments[_i];
    }
    ENGINE.startScope(opName);
    try {
      var result = fn.apply(void 0, args);
      if (isPromise(result)) {
        console.error('Cannot return a Promise inside of tidy.');
      }
      ENGINE.endScope(result);
      return result;
    } catch (ex) {
      ENGINE.endScope(null);
      throw ex;
    }
  };
  Object.defineProperty(f2, 'name', {value: opName, configurable: true});
  // tslint:disable-next-line:no-any
  return f2;
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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function complex_(real, imag) {
  var $real = convertToTensor(real, 'real', 'complex');
  var $imag = convertToTensor(imag, 'imag', 'complex');
  assertShapesMatch(
      $real.shape, $imag.shape,
      'real and imag shapes, ' + $real.shape + ' and ' + $imag.shape + ', ' +
          'must match in call to tf.complex().');
  var inputs = {real: $real, imag: $imag};
  return ENGINE.runKernel(Complex, inputs);
}
var complex = op({complex_: complex_});

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
/** This is shared code across all tensor creation methods. */
function makeTensor(values, shape, inferredShape, dtype) {
  if (dtype == null) {
    dtype = inferDtype(values);
  }
  if (dtype === 'complex64') {
    throw new Error(
        'Cannot construct a complex64 tensor directly. ' +
        'Please use tf.complex(real, imag).');
  }
  if (!isTypedArray(values) && !Array.isArray(values) &&
      typeof values !== 'number' && typeof values !== 'boolean' &&
      typeof values !== 'string') {
    throw new Error(
        'values passed to tensor(values) must be a number/boolean/string or ' +
        'an array of numbers/booleans/strings, or a TypedArray');
  }
  if (shape != null) {
    assertNonNegativeIntegerDimensions(shape);
    var providedSize_1 = sizeFromShape(shape);
    var inferredSize_1 = sizeFromShape(inferredShape);
    assert(providedSize_1 === inferredSize_1, function() {
      return 'Based on the provided shape, [' + shape +
          '], the tensor should have ' +
          (providedSize_1 + ' values but has ' + inferredSize_1);
    });
    for (var i = 0; i < inferredShape.length; ++i) {
      var inferred = inferredShape[i];
      var flatDimsDontMatch = i === inferredShape.length - 1 ?
          inferred !== sizeFromShape(shape.slice(i)) :
          true;
      assert(inferredShape[i] === shape[i] || !flatDimsDontMatch, function() {
        return 'Error creating a new Tensor. Inferred shape ' +
            ('(' + inferredShape + ') does not match the provided ') +
            ('shape (' + shape + '). ');
      });
    }
  }
  if (!isTypedArray(values) && !Array.isArray(values)) {
    values = [values];
  }
  shape = shape || inferredShape;
  values = dtype !== 'string' ? toTypedArray(values, dtype) :
                                flatten(values, [], true);
  return ENGINE.makeTensor(values, shape, dtype);
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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor(values, shape, dtype) {
  var inferredShape = inferShape(values, dtype);
  return makeTensor(values, shape, inferredShape, dtype);
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
var DTYPE_VALUE_SIZE_MAP = {
  'float32': 4,
  'float16': 2,
  'int32': 4,
  'uint16': 2,
  'uint8': 1,
  'bool': 1,
  'complex64': 8
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
var NUM_BYTES_STRING_LENGTH = 4;
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
function encodeWeights(tensors, group) {
  return __awaiter(this, void 0, void 0, function() {
    var specs, dataPromises, names, _loop_1, i, tensorValues;
    var _this = this;
    return __generator(this, function(_a) {
      switch (_a.label) {
        case 0:
          specs = [];
          dataPromises = [];
          names = Array.isArray(tensors) ? tensors.map(function(tensor) {
            return tensor.name;
          }) :
                                           Object.keys(tensors);
          _loop_1 = function(i) {
            var name_1 = names[i];
            var t =
                Array.isArray(tensors) ? tensors[i].tensor : tensors[name_1];
            if (t.dtype !== 'float32' && t.dtype !== 'int32' &&
                t.dtype !== 'bool' && t.dtype !== 'string' &&
                t.dtype !== 'complex64') {
              throw new Error(
                  'Unsupported dtype in weight \'' + name_1 + '\': ' + t.dtype);
            }
            var spec = {name: name_1, shape: t.shape, dtype: t.dtype};
            if (t.dtype === 'string') {
              var utf8bytes = new Promise(function(resolve) {
                return __awaiter(_this, void 0, void 0, function() {
                  var vals, totalNumBytes, bytes, offset, i_1, val,
                      bytesOfLength;
                  return __generator(this, function(_a) {
                    switch (_a.label) {
                      case 0:
                        return [4 /*yield*/, t.bytes()];
                      case 1:
                        vals = _a.sent();
                        totalNumBytes = vals.reduce(function(p, c) {
                          return p + c.length;
                        }, 0) + NUM_BYTES_STRING_LENGTH * vals.length;
                        bytes = new Uint8Array(totalNumBytes);
                        offset = 0;
                        for (i_1 = 0; i_1 < vals.length; i_1++) {
                          val = vals[i_1];
                          bytesOfLength = new Uint8Array(
                              new Uint32Array([val.length]).buffer);
                          bytes.set(bytesOfLength, offset);
                          offset += NUM_BYTES_STRING_LENGTH;
                          bytes.set(val, offset);
                          offset += val.length;
                        }
                        resolve(bytes);
                        return [2 /*return*/];
                    }
                  });
                });
              });
              dataPromises.push(utf8bytes);
            } else {
              dataPromises.push(t.data());
            }
            if (group != null) {
              spec.group = group;
            }
            specs.push(spec);
          };
          for (i = 0; i < names.length; ++i) {
            _loop_1(i);
          }
          return [4 /*yield*/, Promise.all(dataPromises)];
        case 1:
          tensorValues = _a.sent();
          return [
            2 /*return*/,
            {data: concatenateTypedArrays(tensorValues), specs: specs}
          ];
      }
    });
  });
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
  var out = {};
  var float16Decode;
  var offset = 0;
  for (var _i = 0, specs_1 = specs; _i < specs_1.length; _i++) {
    var spec = specs_1[_i];
    var name_2 = spec.name;
    var dtype = spec.dtype;
    var shape = spec.shape;
    var size = sizeFromShape(shape);
    var values = void 0;
    if ('quantization' in spec) {
      var quantization = spec.quantization;
      if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
        if (!('min' in quantization && 'scale' in quantization)) {
          throw new Error(
              'Weight ' + spec.name + ' with quantization ' +
              quantization.dtype + ' ' +
              'doesn\'t have corresponding metadata min and scale.');
        }
      } else if (quantization.dtype === 'float16') {
        if (dtype !== 'float32') {
          throw new Error(
              'Weight ' + spec.name + ' is quantized with ' +
              quantization.dtype + ' ' +
              ('which only supports weights of type float32 not ' + dtype +
               '.'));
        }
      } else {
        throw new Error(
            'Weight ' + spec.name + ' has unknown ' +
            ('quantization dtype ' + quantization.dtype + '. ') +
            'Supported quantization dtypes are: ' +
            '\'uint8\', \'uint16\', and \'float16\'.');
      }
      var quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
      var byteBuffer =
          buffer.slice(offset, offset + size * quantizationSizeFactor);
      var quantizedArray = (quantization.dtype === 'uint8') ?
          new Uint8Array(byteBuffer) :
          new Uint16Array(byteBuffer);
      if (dtype === 'float32') {
        if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
          values = new Float32Array(quantizedArray.length);
          for (var i = 0; i < quantizedArray.length; i++) {
            var v = quantizedArray[i];
            values[i] = v * quantization.scale + quantization.min;
          }
        } else if (quantization.dtype === 'float16') {
          if (float16Decode === undefined) {
            float16Decode = getFloat16Decoder();
          }
          values = float16Decode(quantizedArray);
        } else {
          throw new Error(
              'Unsupported quantization type ' + quantization.dtype + ' ' +
              'for weight type float32.');
        }
      } else if (dtype === 'int32') {
        if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
          throw new Error(
              'Unsupported quantization type ' + quantization.dtype + ' ' +
              'for weight type int32.');
        }
        values = new Int32Array(quantizedArray.length);
        for (var i = 0; i < quantizedArray.length; i++) {
          var v = quantizedArray[i];
          values[i] = Math.round(v * quantization.scale + quantization.min);
        }
      } else {
        throw new Error(
            'Unsupported dtype in weight \'' + name_2 + '\': ' + dtype);
      }
      offset += size * quantizationSizeFactor;
    } else if (dtype === 'string') {
      var size_1 = sizeFromShape(spec.shape);
      values = [];
      for (var i = 0; i < size_1; i++) {
        var byteLength = new Uint32Array(
            buffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
        offset += NUM_BYTES_STRING_LENGTH;
        var bytes = new Uint8Array(buffer.slice(offset, offset + byteLength));
        values.push(bytes);
        offset += byteLength;
      }
    } else {
      var dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
      var byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);
      if (dtype === 'float32') {
        values = new Float32Array(byteBuffer);
      } else if (dtype === 'int32') {
        values = new Int32Array(byteBuffer);
      } else if (dtype === 'bool') {
        values = new Uint8Array(byteBuffer);
      } else if (dtype === 'complex64') {
        values = new Float32Array(byteBuffer);
        var real = new Float32Array(values.length / 2);
        var image = new Float32Array(values.length / 2);
        for (var i = 0; i < real.length; i++) {
          real[i] = values[i * 2];
          image[i] = values[i * 2 + 1];
        }
        var realTensor = tensor(real, shape, 'float32');
        var imageTensor = tensor(image, shape, 'float32');
        out[name_2] = complex(realTensor, imageTensor);
        realTensor.dispose();
        imageTensor.dispose();
      } else {
        throw new Error(
            'Unsupported dtype in weight \'' + name_2 + '\': ' + dtype);
      }
      offset += size * dtypeFactor;
    }
    if (dtype !== 'complex64') {
      out[name_2] = tensor(values, shape, dtype);
    }
  }
  return out;
}
/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
function concatenateTypedArrays(xs) {
  // TODO(adarob, cais): Support quantization.
  if (xs === null) {
    throw new Error('Invalid input value: ' + JSON.stringify(xs));
  }
  var totalByteLength = 0;
  // `normalizedXs` is here for this reason: a `TypedArray`'s `buffer'
  // can have a different byte length from that of the `TypedArray` itself,
  // for example, when the `TypedArray` is created from an offset in an
  // `ArrayBuffer`. `normliazedXs` holds `TypedArray`s whose `buffer`s match
  // the `TypedArray` in byte length. If an element of `xs` does not show
  // this property, a new `TypedArray` that satisfy this property will be
  // constructed and pushed into `normalizedXs`.
  var normalizedXs = [];
  xs.forEach(function(x) {
    totalByteLength += x.byteLength;
    // tslint:disable:no-any
    normalizedXs.push(
        x.byteLength === x.buffer.byteLength ? x : new x.constructor(x));
    if (!(x instanceof Float32Array || x instanceof Int32Array ||
          x instanceof Uint8Array)) {
      throw new Error('Unsupported TypedArray subtype: ' + x.constructor.name);
    }
    // tslint:enable:no-any
  });
  var y = new Uint8Array(totalByteLength);
  var offset = 0;
  normalizedXs.forEach(function(x) {
    y.set(new Uint8Array(x.buffer), offset);
    offset += x.byteLength;
  });
  return y.buffer;
}
// Use Buffer on Node.js instead of Blob/atob/btoa
var useNodeBuffer = typeof Buffer !== 'undefined' &&
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
  var buf = new Uint8Array(buffer);
  var s = '';
  for (var i = 0, l = buf.length; i < l; i++) {
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
    var buf = Buffer.from(str, 'base64');
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
  var s = atob(str);
  var buffer = new Uint8Array(s.length);
  for (var i = 0; i < s.length; ++i) {
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
  if (buffers.length === 1) {
    return buffers[0];
  }
  var totalByteLength = 0;
  buffers.forEach(function(buffer) {
    totalByteLength += buffer.byteLength;
  });
  var temp = new Uint8Array(totalByteLength);
  var offset = 0;
  buffers.forEach(function(buffer) {
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
  var SEPARATOR = '/';
  path = path.trim();
  while (path.endsWith(SEPARATOR)) {
    path = path.slice(0, path.length - 1);
  }
  var items = path.split(SEPARATOR);
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
 * Computes mantisa table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 2048 mantissa lookup values.
 */
function computeFloat16MantisaTable() {
  var convertMantissa = function(i) {
    var m = i << 13;
    var e = 0;
    while ((m & 0x00800000) === 0) {
      e -= 0x00800000;
      m <<= 1;
    }
    m &= ~0x00800000;
    e += 0x38800000;
    return m | e;
  };
  var mantisaTable = new Uint32Array(2048);
  mantisaTable[0] = 0;
  for (var i = 1; i < 1024; i++) {
    mantisaTable[i] = convertMantissa(i);
  }
  for (var i = 1024; i < 2048; i++) {
    mantisaTable[i] = 0x38000000 + ((i - 1024) << 13);
  }
  return mantisaTable;
}
/**
 * Computes exponent table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 64 exponent lookup values.
 */
function computeFloat16ExponentTable() {
  var exponentTable = new Uint32Array(64);
  exponentTable[0] = 0;
  exponentTable[31] = 0x47800000;
  exponentTable[32] = 0x80000000;
  exponentTable[63] = 0xc7800000;
  for (var i = 1; i < 31; i++) {
    exponentTable[i] = i << 23;
  }
  for (var i = 33; i < 63; i++) {
    exponentTable[i] = 0x80000000 + ((i - 32) << 23);
  }
  return exponentTable;
}
/**
 * Computes offset table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 6d offset values.
 */
function computeFloat16OffsetTable() {
  var offsetTable = new Uint32Array(64);
  for (var i = 0; i < 64; i++) {
    offsetTable[i] = 1024;
  }
  offsetTable[0] = offsetTable[32] = 0;
  return offsetTable;
}
/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
function getFloat16Decoder() {
  // Algorithm is based off of
  // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
  // Cache lookup tables
  var mantisaTable = computeFloat16MantisaTable();
  var exponentTable = computeFloat16ExponentTable();
  var offsetTable = computeFloat16OffsetTable();
  return function(quantizedArray) {
    var buffer = new ArrayBuffer(4 * quantizedArray.length);
    var bufferUint32View = new Uint32Array(buffer);
    for (var index = 0; index < quantizedArray.length; index++) {
      var float16Bits = quantizedArray[index];
      var float32Bits =
          mantisaTable[offsetTable[float16Bits >> 10] + (float16Bits & 0x3ff)] +
          exponentTable[float16Bits >> 10];
      bufferUint32View[index] = float32Bits;
    }
    return new Float32Array(buffer);
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
var IORouterRegistry = /** @class */ (function() {
  function IORouterRegistry() {
    this.saveRouters = [];
    this.loadRouters = [];
  }
  IORouterRegistry.getInstance = function() {
    if (IORouterRegistry.instance == null) {
      IORouterRegistry.instance = new IORouterRegistry();
    }
    return IORouterRegistry.instance;
  };
  /**
   * Register a save-handler router.
   *
   * @param saveRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `save` method defined or `null`.
   */
  IORouterRegistry.registerSaveRouter = function(saveRouter) {
    IORouterRegistry.getInstance().saveRouters.push(saveRouter);
  };
  /**
   * Register a load-handler router.
   *
   * @param loadRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `load` method defined or `null`.
   */
  IORouterRegistry.registerLoadRouter = function(loadRouter) {
    IORouterRegistry.getInstance().loadRouters.push(loadRouter);
  };
  /**
   * Look up IOHandler for saving, given a URL-like string.
   *
   * @param url
   * @returns If only one match is found, an instance of IOHandler with the
   * `save` method defined. If no match is found, `null`.
   * @throws Error, if more than one match is found.
   */
  IORouterRegistry.getSaveHandlers = function(url) {
    return IORouterRegistry.getHandlers(url, 'save');
  };
  /**
   * Look up IOHandler for loading, given a URL-like string.
   *
   * @param url
   * @param loadOptions Optional, custom load options.
   * @returns All valid handlers for `url`, given the currently registered
   *   handler routers.
   */
  IORouterRegistry.getLoadHandlers = function(url, loadOptions) {
    return IORouterRegistry.getHandlers(url, 'load', loadOptions);
  };
  IORouterRegistry.getHandlers = function(url, handlerType, loadOptions) {
    var validHandlers = [];
    var routers = handlerType === 'load' ?
        IORouterRegistry.getInstance().loadRouters :
        IORouterRegistry.getInstance().saveRouters;
    routers.forEach(function(router) {
      var handler = router(url, loadOptions);
      if (handler !== null) {
        validHandlers.push(handler);
      }
    });
    return validHandlers;
  };
  return IORouterRegistry;
}());
var registerSaveRouter = function(loudRouter) {
  return IORouterRegistry.registerSaveRouter(loudRouter);
};
var registerLoadRouter = function(loudRouter) {
  return IORouterRegistry.registerLoadRouter(loudRouter);
};
var getSaveHandlers = function(url) {
  return IORouterRegistry.getSaveHandlers(url);
};
var getLoadHandlers = function(url, loadOptions) {
  return IORouterRegistry.getLoadHandlers(url, loadOptions);
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
var DATABASE_NAME = 'tensorflowjs';
var DATABASE_VERSION = 1;
// Model data and ModelArtifactsInfo (metadata) are stored in two separate
// stores for efficient access of the list of stored models and their metadata.
// 1. The object store for model data: topology, weights and weight manifests.
var MODEL_STORE_NAME = 'models_store';
// 2. The object store for ModelArtifactsInfo, including meta-information such
//    as the type of topology (JSON vs binary), byte size of the topology, byte
//    size of the weights, etc.
var INFO_STORE_NAME = 'model_info_store';
function getIndexedDBFactory() {
  if (!env().getBool('IS_BROWSER')) {
    // TODO(cais): Add more info about what IOHandler subtypes are available.
    //   Maybe point to a doc page on the web and/or automatically determine
    //   the available IOHandlers and print them in the error message.
    throw new Error(
        'Failed to obtain IndexedDB factory because the current environment' +
        'is not a web browser.');
  }
  // tslint:disable-next-line:no-any
  var theWindow = typeof window === 'undefined' ? self : window;
  var factory = theWindow.indexedDB || theWindow.mozIndexedDB ||
      theWindow.webkitIndexedDB || theWindow.msIndexedDB ||
      theWindow.shimIndexedDB;
  if (factory == null) {
    throw new Error(
        'The current browser does not appear to support IndexedDB.');
  }
  return factory;
}
function setUpDatabase(openRequest) {
  var db = openRequest.result;
  db.createObjectStore(MODEL_STORE_NAME, {keyPath: 'modelPath'});
  db.createObjectStore(INFO_STORE_NAME, {keyPath: 'modelPath'});
}
/**
 * IOHandler subclass: Browser IndexedDB.
 *
 * See the doc string of `browserIndexedDB` for more details.
 */
var BrowserIndexedDB = /** @class */ (function() {
  function BrowserIndexedDB(modelPath) {
    this.indexedDB = getIndexedDBFactory();
    if (modelPath == null || !modelPath) {
      throw new Error(
          'For IndexedDB, modelPath must not be null, undefined or empty.');
    }
    this.modelPath = modelPath;
  }
  BrowserIndexedDB.prototype.save = function(modelArtifacts) {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        // TODO(cais): Support saving GraphDef models.
        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
          throw new Error(
              'BrowserLocalStorage.save() does not support saving model topology ' +
              'in binary formats yet.');
        }
        return [
          2 /*return*/, this.databaseAction(this.modelPath, modelArtifacts)
        ];
      });
    });
  };
  BrowserIndexedDB.prototype.load = function() {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        return [2 /*return*/, this.databaseAction(this.modelPath)];
      });
    });
  };
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
  BrowserIndexedDB.prototype.databaseAction = function(
      modelPath, modelArtifacts) {
    var _this = this;
    return new Promise(function(resolve, reject) {
      var openRequest = _this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
      openRequest.onupgradeneeded = function() {
        return setUpDatabase(openRequest);
      };
      openRequest.onsuccess = function() {
        var db = openRequest.result;
        if (modelArtifacts == null) {
          // Read model out from object store.
          var modelTx = db.transaction(MODEL_STORE_NAME, 'readonly');
          var modelStore = modelTx.objectStore(MODEL_STORE_NAME);
          var getRequest_1 = modelStore.get(_this.modelPath);
          getRequest_1.onsuccess = function() {
            if (getRequest_1.result == null) {
              db.close();
              return reject(new Error(
                  'Cannot find model with path \'' + _this.modelPath + '\' ' +
                  'in IndexedDB.'));
            } else {
              resolve(getRequest_1.result.modelArtifacts);
            }
          };
          getRequest_1.onerror = function(error) {
            db.close();
            return reject(getRequest_1.error);
          };
          modelTx.oncomplete = function() {
            return db.close();
          };
        } else {
          // Put model into object store.
          var modelArtifactsInfo_1 =
              getModelArtifactsInfoForJSON(modelArtifacts);
          // First, put ModelArtifactsInfo into info store.
          var infoTx_1 = db.transaction(INFO_STORE_NAME, 'readwrite');
          var infoStore_1 = infoTx_1.objectStore(INFO_STORE_NAME);
          var putInfoRequest_1 = infoStore_1.put({
            modelPath: _this.modelPath,
            modelArtifactsInfo: modelArtifactsInfo_1
          });
          var modelTx_1;
          putInfoRequest_1.onsuccess = function() {
            // Second, put model data into model store.
            modelTx_1 = db.transaction(MODEL_STORE_NAME, 'readwrite');
            var modelStore = modelTx_1.objectStore(MODEL_STORE_NAME);
            var putModelRequest = modelStore.put({
              modelPath: _this.modelPath,
              modelArtifacts: modelArtifacts,
              modelArtifactsInfo: modelArtifactsInfo_1
            });
            putModelRequest.onsuccess = function() {
              return resolve({modelArtifactsInfo: modelArtifactsInfo_1});
            };
            putModelRequest.onerror = function(error) {
              // If the put-model request fails, roll back the info entry as
              // well.
              infoStore_1 = infoTx_1.objectStore(INFO_STORE_NAME);
              var deleteInfoRequest = infoStore_1.delete(_this.modelPath);
              deleteInfoRequest.onsuccess = function() {
                db.close();
                return reject(putModelRequest.error);
              };
              deleteInfoRequest.onerror = function(error) {
                db.close();
                return reject(putModelRequest.error);
              };
            };
          };
          putInfoRequest_1.onerror = function(error) {
            db.close();
            return reject(putInfoRequest_1.error);
          };
          infoTx_1.oncomplete = function() {
            if (modelTx_1 == null) {
              db.close();
            } else {
              modelTx_1.oncomplete = function() {
                return db.close();
              };
            }
          };
        }
      };
      openRequest.onerror = function(error) {
        return reject(openRequest.error);
      };
    });
  };
  BrowserIndexedDB.URL_SCHEME = 'indexeddb://';
  return BrowserIndexedDB;
}());
var indexedDBRouter = function(url) {
  if (!env().getBool('IS_BROWSER')) {
    return null;
  } else {
    if (!Array.isArray(url) && url.startsWith(BrowserIndexedDB.URL_SCHEME)) {
      return browserIndexedDB(url.slice(BrowserIndexedDB.URL_SCHEME.length));
    } else {
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
var BrowserIndexedDBManager = /** @class */ (function() {
  function BrowserIndexedDBManager() {
    this.indexedDB = getIndexedDBFactory();
  }
  BrowserIndexedDBManager.prototype.listModels = function() {
    return __awaiter(this, void 0, void 0, function() {
      var _this = this;
      return __generator(this, function(_a) {
        return [
          2 /*return*/, new Promise(function(resolve, reject) {
            var openRequest =
                _this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
            openRequest.onupgradeneeded = function() {
              return setUpDatabase(openRequest);
            };
            openRequest.onsuccess = function() {
              var db = openRequest.result;
              var tx = db.transaction(INFO_STORE_NAME, 'readonly');
              var store = tx.objectStore(INFO_STORE_NAME);
              // tslint:disable:max-line-length
              // Need to cast `store` as `any` here because TypeScript's DOM
              // library does not have the `getAll()` method even though the
              // method is supported in the latest version of most mainstream
              // browsers:
              // https://developer.mozilla.org/en-US/docs/Web/API/IDBObjectStore/getAll
              // tslint:enable:max-line-length
              // tslint:disable-next-line:no-any
              var getAllInfoRequest = store.getAll();
              getAllInfoRequest.onsuccess = function() {
                var out = {};
                for (var _i = 0, _a = getAllInfoRequest.result; _i < _a.length;
                     _i++) {
                  var item = _a[_i];
                  out[item.modelPath] = item.modelArtifactsInfo;
                }
                resolve(out);
              };
              getAllInfoRequest.onerror = function(error) {
                db.close();
                return reject(getAllInfoRequest.error);
              };
              tx.oncomplete = function() {
                return db.close();
              };
            };
            openRequest.onerror = function(error) {
              return reject(openRequest.error);
            };
          })
        ];
      });
    });
  };
  BrowserIndexedDBManager.prototype.removeModel = function(path) {
    return __awaiter(this, void 0, void 0, function() {
      var _this = this;
      return __generator(this, function(_a) {
        path = maybeStripScheme(path);
        return [
          2 /*return*/, new Promise(function(resolve, reject) {
            var openRequest =
                _this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
            openRequest.onupgradeneeded = function() {
              return setUpDatabase(openRequest);
            };
            openRequest.onsuccess = function() {
              var db = openRequest.result;
              var infoTx = db.transaction(INFO_STORE_NAME, 'readwrite');
              var infoStore = infoTx.objectStore(INFO_STORE_NAME);
              var getInfoRequest = infoStore.get(path);
              var modelTx;
              getInfoRequest.onsuccess = function() {
                if (getInfoRequest.result == null) {
                  db.close();
                  return reject(new Error(
                      'Cannot find model with path \'' + path + '\' ' +
                      'in IndexedDB.'));
                } else {
                  // First, delete the entry in the info store.
                  var deleteInfoRequest = infoStore.delete(path);
                  var deleteModelData_1 = function() {
                    // Second, delete the entry in the model store.
                    modelTx = db.transaction(MODEL_STORE_NAME, 'readwrite');
                    var modelStore = modelTx.objectStore(MODEL_STORE_NAME);
                    var deleteModelRequest = modelStore.delete(path);
                    deleteModelRequest.onsuccess = function() {
                      return resolve(getInfoRequest.result.modelArtifactsInfo);
                    };
                    deleteModelRequest.onerror = function(error) {
                      return reject(getInfoRequest.error);
                    };
                  };
                  // Proceed with deleting model data regardless of whether
                  // deletion of info data succeeds or not.
                  deleteInfoRequest.onsuccess = deleteModelData_1;
                  deleteInfoRequest.onerror = function(error) {
                    deleteModelData_1();
                    db.close();
                    return reject(getInfoRequest.error);
                  };
                }
              };
              getInfoRequest.onerror = function(error) {
                db.close();
                return reject(getInfoRequest.error);
              };
              infoTx.oncomplete = function() {
                if (modelTx == null) {
                  db.close();
                } else {
                  modelTx.oncomplete = function() {
                    return db.close();
                  };
                }
              };
            };
            openRequest.onerror = function(error) {
              return reject(openRequest.error);
            };
          })
        ];
      });
    });
  };
  return BrowserIndexedDBManager;
}());

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
var PATH_SEPARATOR = '/';
var PATH_PREFIX = 'tensorflowjs_models';
var INFO_SUFFIX = 'info';
var MODEL_TOPOLOGY_SUFFIX = 'model_topology';
var WEIGHT_SPECS_SUFFIX = 'weight_specs';
var WEIGHT_DATA_SUFFIX = 'weight_data';
var MODEL_METADATA_SUFFIX = 'model_metadata';
function getModelKeys(path) {
  return {
    info: [PATH_PREFIX, path, INFO_SUFFIX].join(PATH_SEPARATOR),
    topology: [PATH_PREFIX, path, MODEL_TOPOLOGY_SUFFIX].join(PATH_SEPARATOR),
    weightSpecs: [PATH_PREFIX, path, WEIGHT_SPECS_SUFFIX].join(PATH_SEPARATOR),
    weightData: [PATH_PREFIX, path, WEIGHT_DATA_SUFFIX].join(PATH_SEPARATOR),
    modelMetadata:
        [PATH_PREFIX, path, MODEL_METADATA_SUFFIX].join(PATH_SEPARATOR)
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
  var items = key.split(PATH_SEPARATOR);
  if (items.length < 3) {
    throw new Error('Invalid key format: ' + key);
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
var BrowserLocalStorage = /** @class */ (function() {
  function BrowserLocalStorage(modelPath) {
    if (!env().getBool('IS_BROWSER') || typeof window === 'undefined' ||
        typeof window.localStorage === 'undefined') {
      // TODO(cais): Add more info about what IOHandler subtypes are
      // available.
      //   Maybe point to a doc page on the web and/or automatically determine
      //   the available IOHandlers and print them in the error message.
      throw new Error(
          'The current environment does not support local storage.');
    }
    this.LS = window.localStorage;
    if (modelPath == null || !modelPath) {
      throw new Error(
          'For local storage, modelPath must not be null, undefined or empty.');
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
  BrowserLocalStorage.prototype.save = function(modelArtifacts) {
    return __awaiter(this, void 0, void 0, function() {
      var topology, weightSpecs, modelArtifactsInfo, result;
      return __generator(this, function(_a) {
        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
          throw new Error(
              'BrowserLocalStorage.save() does not support saving model topology ' +
              'in binary formats yet.');
        } else {
          topology = JSON.stringify(modelArtifacts.modelTopology);
          weightSpecs = JSON.stringify(modelArtifacts.weightSpecs);
          modelArtifactsInfo = getModelArtifactsInfoForJSON(modelArtifacts);
          try {
            this.LS.setItem(this.keys.info, JSON.stringify(modelArtifactsInfo));
            this.LS.setItem(this.keys.topology, topology);
            this.LS.setItem(this.keys.weightSpecs, weightSpecs);
            this.LS.setItem(
                this.keys.weightData,
                arrayBufferToBase64String(modelArtifacts.weightData));
            result = {
              format: modelArtifacts.format,
              generatedBy: modelArtifacts.generatedBy,
              convertedBy: modelArtifacts.convertedBy
            };
            if (modelArtifacts.signature != null) {
              result.signature = modelArtifacts.signature;
            }
            if (modelArtifacts.userDefinedMetadata != null) {
              result.userDefinedMetadata = modelArtifacts.userDefinedMetadata;
            }
            if (modelArtifacts.modelInitializer != null) {
              result.modelInitializer = modelArtifacts.modelInitializer;
            }
            this.LS.setItem(this.keys.modelMetadata, JSON.stringify(result));
            return [2 /*return*/, {modelArtifactsInfo: modelArtifactsInfo}];
          } catch (err) {
            // If saving failed, clean up all items saved so far.
            this.LS.removeItem(this.keys.info);
            this.LS.removeItem(this.keys.topology);
            this.LS.removeItem(this.keys.weightSpecs);
            this.LS.removeItem(this.keys.weightData);
            this.LS.removeItem(this.keys.modelMetadata);
            throw new Error(
                'Failed to save model \'' + this.modelPath +
                '\' to local storage: ' +
                'size quota being exceeded is a possible cause of this failure: ' +
                ('modelTopologyBytes=' + modelArtifactsInfo.modelTopologyBytes +
                 ', ') +
                ('weightSpecsBytes=' + modelArtifactsInfo.weightSpecsBytes +
                 ', ') +
                ('weightDataBytes=' + modelArtifactsInfo.weightDataBytes +
                 '.'));
          }
        }
        return [2 /*return*/];
      });
    });
  };
  /**
   * Load a model from local storage.
   *
   * See the documentation to `browserLocalStorage` for details on the saved
   * artifacts.
   *
   * @returns The loaded model (if loading succeeds).
   */
  BrowserLocalStorage.prototype.load = function() {
    return __awaiter(this, void 0, void 0, function() {
      var info, out, topology, weightSpecs, metadataString, metadata,
          weightDataBase64;
      return __generator(this, function(_a) {
        info = JSON.parse(this.LS.getItem(this.keys.info));
        if (info == null) {
          throw new Error(
              'In local storage, there is no model with name \'' +
              this.modelPath + '\'');
        }
        if (info.modelTopologyType !== 'JSON') {
          throw new Error(
              'BrowserLocalStorage does not support loading non-JSON model ' +
              'topology yet.');
        }
        out = {};
        topology = JSON.parse(this.LS.getItem(this.keys.topology));
        if (topology == null) {
          throw new Error(
              'In local storage, the topology of model \'' + this.modelPath +
              '\' ' +
              'is missing.');
        }
        out.modelTopology = topology;
        weightSpecs = JSON.parse(this.LS.getItem(this.keys.weightSpecs));
        if (weightSpecs == null) {
          throw new Error(
              'In local storage, the weight specs of model \'' +
              this.modelPath + '\' ' +
              'are missing.');
        }
        out.weightSpecs = weightSpecs;
        metadataString = this.LS.getItem(this.keys.modelMetadata);
        if (metadataString != null) {
          metadata = JSON.parse(metadataString);
          out.format = metadata['format'];
          out.generatedBy = metadata['generatedBy'];
          out.convertedBy = metadata['convertedBy'];
          if (metadata['signature'] != null) {
            out.signature = metadata['signature'];
          }
          if (metadata['userDefinedMetadata'] != null) {
            out.userDefinedMetadata = metadata['userDefinedMetadata'];
          }
          if (metadata['modelInitializer'] != null) {
            out.modelInitializer = metadata['modelInitializer'];
          }
        }
        weightDataBase64 = this.LS.getItem(this.keys.weightData);
        if (weightDataBase64 == null) {
          throw new Error(
              'In local storage, the binary weight values of model ' +
              ('\'' + this.modelPath + '\' are missing.'));
        }
        out.weightData = base64StringToArrayBuffer(weightDataBase64);
        return [2 /*return*/, out];
      });
    });
  };
  BrowserLocalStorage.URL_SCHEME = 'localstorage://';
  return BrowserLocalStorage;
}());
var localStorageRouter = function(url) {
  if (!env().getBool('IS_BROWSER')) {
    return null;
  } else {
    if (!Array.isArray(url) && url.startsWith(BrowserLocalStorage.URL_SCHEME)) {
      return browserLocalStorage(
          url.slice(BrowserLocalStorage.URL_SCHEME.length));
    } else {
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
var BrowserLocalStorageManager = /** @class */ (function() {
  function BrowserLocalStorageManager() {
    assert(env().getBool('IS_BROWSER'), function() {
      return 'Current environment is not a web browser';
    });
    assert(
        typeof window === 'undefined' ||
            typeof window.localStorage !== 'undefined',
        function() {
          return 'Current browser does not appear to support localStorage';
        });
    this.LS = window.localStorage;
  }
  BrowserLocalStorageManager.prototype.listModels = function() {
    return __awaiter(this, void 0, void 0, function() {
      var out, prefix, suffix, i, key, modelPath;
      return __generator(this, function(_a) {
        out = {};
        prefix = PATH_PREFIX + PATH_SEPARATOR;
        suffix = PATH_SEPARATOR + INFO_SUFFIX;
        for (i = 0; i < this.LS.length; ++i) {
          key = this.LS.key(i);
          if (key.startsWith(prefix) && key.endsWith(suffix)) {
            modelPath = getModelPathFromKey(key);
            out[modelPath] = JSON.parse(this.LS.getItem(key));
          }
        }
        return [2 /*return*/, out];
      });
    });
  };
  BrowserLocalStorageManager.prototype.removeModel = function(path) {
    return __awaiter(this, void 0, void 0, function() {
      var keys, info;
      return __generator(this, function(_a) {
        path = maybeStripScheme$1(path);
        keys = getModelKeys(path);
        if (this.LS.getItem(keys.info) == null) {
          throw new Error('Cannot find model at path \'' + path + '\'');
        }
        info = JSON.parse(this.LS.getItem(keys.info));
        this.LS.removeItem(keys.info);
        this.LS.removeItem(keys.topology);
        this.LS.removeItem(keys.weightSpecs);
        this.LS.removeItem(keys.weightData);
        return [2 /*return*/, info];
      });
    });
  };
  return BrowserLocalStorageManager;
}());

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
var URL_SCHEME_SUFFIX = '://';
var ModelStoreManagerRegistry = /** @class */ (function() {
  function ModelStoreManagerRegistry() {
    this.managers = {};
  }
  ModelStoreManagerRegistry.getInstance = function() {
    if (ModelStoreManagerRegistry.instance == null) {
      ModelStoreManagerRegistry.instance = new ModelStoreManagerRegistry();
    }
    return ModelStoreManagerRegistry.instance;
  };
  /**
   * Register a save-handler router.
   *
   * @param saveRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `save` method defined or `null`.
   */
  ModelStoreManagerRegistry.registerManager = function(scheme, manager) {
    assert(scheme != null, function() {
      return 'scheme must not be undefined or null.';
    });
    if (scheme.endsWith(URL_SCHEME_SUFFIX)) {
      scheme = scheme.slice(0, scheme.indexOf(URL_SCHEME_SUFFIX));
    }
    assert(scheme.length > 0, function() {
      return 'scheme must not be an empty string.';
    });
    var registry = ModelStoreManagerRegistry.getInstance();
    assert(registry.managers[scheme] == null, function() {
      return 'A model store manager is already registered for scheme \'' +
          scheme + '\'.';
    });
    registry.managers[scheme] = manager;
  };
  ModelStoreManagerRegistry.getManager = function(scheme) {
    var manager = this.getInstance().managers[scheme];
    if (manager == null) {
      throw new Error(
          'Cannot find model manager for scheme \'' + scheme + '\'');
    }
    return manager;
  };
  ModelStoreManagerRegistry.getSchemes = function() {
    return Object.keys(this.getInstance().managers);
  };
  return ModelStoreManagerRegistry;
}());
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
    throw new Error(
        'The url string provided does not contain a scheme. ' +
        'Supported schemes are: ' +
        ('' + ModelStoreManagerRegistry.getSchemes().join(',')));
  }
  return {
    scheme: url.split(URL_SCHEME_SUFFIX)[0],
    path: url.split(URL_SCHEME_SUFFIX)[1],
  };
}
function cloneModelInternal(sourceURL, destURL, deleteSource) {
  if (deleteSource === void 0) {
    deleteSource = false;
  }
  return __awaiter(this, void 0, void 0, function() {
    var loadHandlers, loadHandler, saveHandlers, saveHandler, sourceScheme,
        sourcePath, sameMedium, modelArtifacts, saveResult;
    return __generator(this, function(_a) {
      switch (_a.label) {
        case 0:
          assert(sourceURL !== destURL, function() {
            return 'Old path and new path are the same: \'' + sourceURL + '\'';
          });
          loadHandlers = IORouterRegistry.getLoadHandlers(sourceURL);
          assert(loadHandlers.length > 0, function() {
            return 'Copying failed because no load handler is found for source URL ' +
                sourceURL + '.';
          });
          assert(loadHandlers.length < 2, function() {
            return 'Copying failed because more than one (' +
                loadHandlers.length + ') ' +
                ('load handlers for source URL ' + sourceURL + '.');
          });
          loadHandler = loadHandlers[0];
          saveHandlers = IORouterRegistry.getSaveHandlers(destURL);
          assert(saveHandlers.length > 0, function() {
            return 'Copying failed because no save handler is found for destination ' +
                ('URL ' + destURL + '.');
          });
          assert(saveHandlers.length < 2, function() {
            return 'Copying failed because more than one (' +
                loadHandlers.length + ') ' +
                ('save handlers for destination URL ' + destURL + '.');
          });
          saveHandler = saveHandlers[0];
          sourceScheme = parseURL(sourceURL).scheme;
          sourcePath = parseURL(sourceURL).path;
          sameMedium = sourceScheme === parseURL(sourceURL).scheme;
          return [4 /*yield*/, loadHandler.load()];
        case 1:
          modelArtifacts = _a.sent();
          if (!(deleteSource && sameMedium)) return [3 /*break*/, 3];
          return [
            4 /*yield*/,
            ModelStoreManagerRegistry.getManager(sourceScheme)
                .removeModel(sourcePath)
          ];
        case 2:
          _a.sent();
          _a.label = 3;
        case 3:
          return [4 /*yield*/, saveHandler.save(modelArtifacts)];
        case 4:
          saveResult = _a.sent();
          if (!(deleteSource && !sameMedium)) return [3 /*break*/, 6];
          return [
            4 /*yield*/,
            ModelStoreManagerRegistry.getManager(sourceScheme)
                .removeModel(sourcePath)
          ];
        case 5:
          _a.sent();
          _a.label = 6;
        case 6:
          return [2 /*return*/, saveResult.modelArtifactsInfo];
      }
    });
  });
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
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
function listModels() {
  return __awaiter(this, void 0, void 0, function() {
    var schemes, out, _i, schemes_1, scheme, schemeOut, path, url;
    return __generator(this, function(_a) {
      switch (_a.label) {
        case 0:
          schemes = ModelStoreManagerRegistry.getSchemes();
          out = {};
          _i = 0, schemes_1 = schemes;
          _a.label = 1;
        case 1:
          if (!(_i < schemes_1.length)) return [3 /*break*/, 4];
          scheme = schemes_1[_i];
          return [
            4 /*yield*/,
            ModelStoreManagerRegistry.getManager(scheme).listModels()
          ];
        case 2:
          schemeOut = _a.sent();
          for (path in schemeOut) {
            url = scheme + URL_SCHEME_SUFFIX + path;
            out[url] = schemeOut[path];
          }
          _a.label = 3;
        case 3:
          _i++;
          return [3 /*break*/, 1];
        case 4:
          return [2 /*return*/, out];
      }
    });
  });
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
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
function removeModel(url) {
  return __awaiter(this, void 0, void 0, function() {
    var schemeAndPath, manager;
    return __generator(this, function(_a) {
      schemeAndPath = parseURL(url);
      manager = ModelStoreManagerRegistry.getManager(schemeAndPath.scheme);
      return [2 /*return*/, manager.removeModel(schemeAndPath.path)];
    });
  });
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
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
function copyModel(sourceURL, destURL) {
  return __awaiter(this, void 0, void 0, function() {
    var deleteSource;
    return __generator(this, function(_a) {
      deleteSource = false;
      return [
        2 /*return*/, cloneModelInternal(sourceURL, destURL, deleteSource)
      ];
    });
  });
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
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
function moveModel(sourceURL, destURL) {
  return __awaiter(this, void 0, void 0, function() {
    var deleteSource;
    return __generator(this, function(_a) {
      deleteSource = true;
      return [
        2 /*return*/, cloneModelInternal(sourceURL, destURL, deleteSource)
      ];
    });
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
var PlatformBrowser = /** @class */ (function() {
  function PlatformBrowser() {}
  PlatformBrowser.prototype.fetch = function(path, init) {
    return fetch(path, init);
  };
  PlatformBrowser.prototype.now = function() {
    return performance.now();
  };
  PlatformBrowser.prototype.encode = function(text, encoding) {
    if (encoding !== 'utf-8' && encoding !== 'utf8') {
      throw new Error(
          'Browser\'s encoder only supports utf-8, but got ' + encoding);
    }
    if (this.textEncoder == null) {
      this.textEncoder = new TextEncoder();
    }
    return this.textEncoder.encode(text);
  };
  PlatformBrowser.prototype.decode = function(bytes, encoding) {
    return new TextDecoder(encoding).decode(bytes);
  };
  return PlatformBrowser;
}());
if (env().get('IS_BROWSER')) {
  env().setPlatform('browser', new PlatformBrowser());
  // Register LocalStorage IOHandler
  try {
    ModelStoreManagerRegistry.registerManager(
        BrowserLocalStorage.URL_SCHEME, new BrowserLocalStorageManager());
  } catch (err) {
  }
  // Register IndexedDB IOHandler
  try {
    ModelStoreManagerRegistry.registerManager(
        BrowserIndexedDB.URL_SCHEME, new BrowserIndexedDBManager());
  } catch (err) {
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
// We are wrapping this within an object so it can be stubbed by Jasmine.
var getNodeFetch = {
  // tslint:disable-next-line:no-require-imports
  importFetch: function() {
    return require('node-fetch');
  }
};
var systemFetch;
var PlatformNode = /** @class */ (function() {
  function PlatformNode() {
    // tslint:disable-next-line:no-require-imports
    this.util = require('util');
    // According to the spec, the built-in encoder can do only UTF-8 encoding.
    // https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/TextEncoder
    this.textEncoder = new this.util.TextEncoder();
  }
  PlatformNode.prototype.fetch = function(path, requestInits) {
    if (env().global.fetch != null) {
      return env().global.fetch(path, requestInits);
    }
    if (systemFetch == null) {
      systemFetch = getNodeFetch.importFetch();
    }
    return systemFetch(path, requestInits);
  };
  PlatformNode.prototype.now = function() {
    var time = process.hrtime();
    return time[0] * 1000 + time[1] / 1000000;
  };
  PlatformNode.prototype.encode = function(text, encoding) {
    if (encoding !== 'utf-8' && encoding !== 'utf8') {
      throw new Error(
          'Node built-in encoder only supports utf-8, but got ' + encoding);
    }
    return this.textEncoder.encode(text);
  };
  PlatformNode.prototype.decode = function(bytes, encoding) {
    if (bytes.length === 0) {
      return '';
    }
    return new this.util.TextDecoder(encoding).decode(bytes);
  };
  return PlatformNode;
}());
if (env().get('IS_NODE')) {
  env().setPlatform('node', new PlatformNode());
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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function buffer(shape, dtype, values) {
  if (dtype === void 0) {
    dtype = 'float32';
  }
  dtype = dtype || 'float32';
  assertNonNegativeIntegerDimensions(shape);
  return new TensorBuffer(shape, dtype, values);
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
 * Casts a `tf.Tensor` to a new dtype.
 *
 * ```js
 * const x = tf.tensor1d([1.5, 2.5, 3]);
 * tf.cast(x, 'int32').print();
 * ```
 * @param x The input tensor to be casted.
 * @param dtype The dtype to cast the input tensor to.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function cast_(x, dtype) {
  var $x = convertToTensor(x, 'x', 'cast');
  // Sanity checks.
  if (!isValidDtype(dtype)) {
    throw new Error('Failed to cast to unknown dtype ' + dtype);
  }
  if (dtype === 'string' && $x.dtype !== 'string' ||
      dtype !== 'string' && $x.dtype === 'string') {
    throw new Error('Only strings can be casted to strings');
  }
  var inputs = {x: $x};
  var attrs = {dtype: dtype};
  return ENGINE.runKernel(Cast, inputs, attrs);
}
var cast = op({cast_: cast_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function clone_(x) {
  var $x = convertToTensor(x, 'x', 'clone', 'string_or_numeric');
  var inputs = {x: $x};
  // Note this op is called tf.identity in python. Hence the kernel name used
  // here.
  return ENGINE.runKernel(Identity, inputs);
}
var clone = op({clone_: clone_});

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
 * Prints information about the `tf.Tensor` including its data.
 *
 * ```js
 * const verbose = true;
 * tf.tensor2d([1, 2, 3, 4], [2, 2]).print(verbose);
 * ```
 * @param x The tensor to be printed.
 * @param verbose Whether to print verbose information about the ` Tensor`,
 * including dtype and size.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function print(x, verbose) {
  if (verbose === void 0) {
    verbose = false;
  }
  console.log(x.toString(verbose));
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
getOrMakeEngine();
var opHandler$1 = {buffer: buffer, cast: cast, clone: clone, print: print};
setOpHandler(opHandler$1);

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
var DEFAULT_FILE_NAME_PREFIX = 'model';
var DEFAULT_JSON_EXTENSION_NAME = '.json';
var DEFAULT_WEIGHT_DATA_EXTENSION_NAME = '.weights.bin';
function defer(f) {
  return new Promise(function(resolve) {
           return setTimeout(resolve);
         })
      .then(f);
}
var BrowserDownloads = /** @class */ (function() {
  function BrowserDownloads(fileNamePrefix) {
    if (!env().getBool('IS_BROWSER')) {
      // TODO(cais): Provide info on what IOHandlers are available under the
      //   current environment.
      throw new Error(
          'browserDownloads() cannot proceed because the current environment ' +
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
  BrowserDownloads.prototype.save = function(modelArtifacts) {
    return __awaiter(this, void 0, void 0, function() {
      var weightsURL, weightsManifest, modelTopologyAndWeightManifest,
          modelTopologyAndWeightManifestURL, jsonAnchor_1, weightDataAnchor_1;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            if (typeof (document) === 'undefined') {
              throw new Error(
                  'Browser downloads are not supported in ' +
                  'this environment since `document` is not present');
            }
            weightsURL = window.URL.createObjectURL(new Blob(
                [modelArtifacts.weightData],
                {type: 'application/octet-stream'}));
            if (!(modelArtifacts.modelTopology instanceof ArrayBuffer))
              return [3 /*break*/, 1];
            throw new Error(
                'BrowserDownloads.save() does not support saving model topology ' +
                'in binary formats yet.');
          case 1:
            weightsManifest = [{
              paths: ['./' + this.weightDataFileName],
              weights: modelArtifacts.weightSpecs
            }];
            modelTopologyAndWeightManifest = {
              modelTopology: modelArtifacts.modelTopology,
              format: modelArtifacts.format,
              generatedBy: modelArtifacts.generatedBy,
              convertedBy: modelArtifacts.convertedBy,
              weightsManifest: weightsManifest
            };
            if (modelArtifacts.signature != null) {
              modelTopologyAndWeightManifest.signature =
                  modelArtifacts.signature;
            }
            if (modelArtifacts.userDefinedMetadata != null) {
              modelTopologyAndWeightManifest.userDefinedMetadata =
                  modelArtifacts.userDefinedMetadata;
            }
            if (modelArtifacts.modelInitializer != null) {
              modelTopologyAndWeightManifest.modelInitializer =
                  modelArtifacts.modelInitializer;
            }
            modelTopologyAndWeightManifestURL =
                window.URL.createObjectURL(new Blob(
                    [JSON.stringify(modelTopologyAndWeightManifest)],
                    {type: 'application/json'}));
            jsonAnchor_1 = this.jsonAnchor == null ?
                document.createElement('a') :
                this.jsonAnchor;
            jsonAnchor_1.download = this.modelTopologyFileName;
            jsonAnchor_1.href = modelTopologyAndWeightManifestURL;
            // Trigger downloads by evoking a click event on the download
            // anchors. When multiple downloads are started synchronously,
            // Firefox will only save the last one.
            return [
              4 /*yield*/, defer(function() {
                return jsonAnchor_1.dispatchEvent(new MouseEvent('click'));
              })
            ];
          case 2:
            // Trigger downloads by evoking a click event on the download
            // anchors. When multiple downloads are started synchronously,
            // Firefox will only save the last one.
            _a.sent();
            if (!(modelArtifacts.weightData != null)) return [3 /*break*/, 4];
            weightDataAnchor_1 = this.weightDataAnchor == null ?
                document.createElement('a') :
                this.weightDataAnchor;
            weightDataAnchor_1.download = this.weightDataFileName;
            weightDataAnchor_1.href = weightsURL;
            return [
              4 /*yield*/, defer(function() {
                return weightDataAnchor_1.dispatchEvent(
                    new MouseEvent('click'));
              })
            ];
          case 3:
            _a.sent();
            _a.label = 4;
          case 4:
            return [
              2 /*return*/,
              {modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts)}
            ];
        }
      });
    });
  };
  BrowserDownloads.URL_SCHEME = 'downloads://';
  return BrowserDownloads;
}());
var BrowserFiles = /** @class */ (function() {
  function BrowserFiles(files) {
    if (files == null || files.length < 1) {
      throw new Error(
          'When calling browserFiles, at least 1 file is required, ' +
          ('but received ' + files));
    }
    this.files = files;
  }
  BrowserFiles.prototype.load = function() {
    return __awaiter(this, void 0, void 0, function() {
      var jsonFile, weightFiles;
      var _this = this;
      return __generator(this, function(_a) {
        jsonFile = this.files[0];
        weightFiles = this.files.slice(1);
        return [
          2 /*return*/, new Promise(function(resolve, reject) {
            var jsonReader = new FileReader();
            jsonReader.onload = function(event) {
              // tslint:disable-next-line:no-any
              var modelJSON = JSON.parse(event.target.result);
              var modelTopology = modelJSON.modelTopology;
              if (modelTopology == null) {
                reject(new Error(
                    'modelTopology field is missing from file ' +
                    jsonFile.name));
                return;
              }
              if (weightFiles.length === 0) {
                resolve({modelTopology: modelTopology});
              }
              var weightsManifest = modelJSON.weightsManifest;
              if (weightsManifest == null) {
                reject(new Error(
                    'weightManifest field is missing from file ' +
                    jsonFile.name));
                return;
              }
              var pathToFile;
              try {
                pathToFile = _this.checkManifestAndWeightFiles(
                    weightsManifest, weightFiles);
              } catch (err) {
                reject(err);
                return;
              }
              var weightSpecs = [];
              var paths = [];
              var perFileBuffers = [];
              weightsManifest.forEach(function(weightsGroup) {
                weightsGroup.paths.forEach(function(path) {
                  paths.push(path);
                  perFileBuffers.push(null);
                });
                weightSpecs.push.apply(weightSpecs, weightsGroup.weights);
              });
              weightsManifest.forEach(function(weightsGroup) {
                weightsGroup.paths.forEach(function(path) {
                  var weightFileReader = new FileReader();
                  weightFileReader.onload = function(event) {
                    // tslint:disable-next-line:no-any
                    var weightData = event.target.result;
                    var index = paths.indexOf(path);
                    perFileBuffers[index] = weightData;
                    if (perFileBuffers.indexOf(null) === -1) {
                      var result = {
                        modelTopology: modelTopology,
                        weightSpecs: weightSpecs,
                        weightData: concatenateArrayBuffers(perFileBuffers),
                        format: modelJSON.format,
                        generatedBy: modelJSON.generatedBy,
                        convertedBy: modelJSON.convertedBy
                      };
                      if (modelJSON.signature != null) {
                        result.signature = modelJSON.signature;
                      }
                      if (modelJSON.userDefinedMetadata != null) {
                        result.userDefinedMetadata =
                            modelJSON.userDefinedMetadata;
                      }
                      if (modelJSON.modelInitializer != null) {
                        result.modelInitializer = modelJSON.modelInitializer;
                      }
                      resolve(result);
                    }
                  };
                  weightFileReader.onerror = function(error) {
                    return reject(
                        'Failed to weights data from file of path \'' + path +
                        '\'.');
                  };
                  weightFileReader.readAsArrayBuffer(pathToFile[path]);
                });
              });
            };
            jsonReader.onerror = function(error) {
              return reject(
                  'Failed to read model topology and weights manifest JSON ' +
                  ('from file \'' + jsonFile.name +
                   '\'. BrowserFiles supports loading ') +
                  'Keras-style tf.Model artifacts only.');
            };
            jsonReader.readAsText(jsonFile);
          })
        ];
      });
    });
  };
  /**
   * Check the compatibility between weights manifest and weight files.
   */
  BrowserFiles.prototype.checkManifestAndWeightFiles = function(
      manifest, files) {
    var basenames = [];
    var fileNames = files.map(function(file) {
      return basename(file.name);
    });
    var pathToFile = {};
    for (var _i = 0, manifest_1 = manifest; _i < manifest_1.length; _i++) {
      var group = manifest_1[_i];
      group.paths.forEach(function(path) {
        var pathBasename = basename(path);
        if (basenames.indexOf(pathBasename) !== -1) {
          throw new Error(
              'Duplicate file basename found in weights manifest: ' +
              ('\'' + pathBasename + '\''));
        }
        basenames.push(pathBasename);
        if (fileNames.indexOf(pathBasename) === -1) {
          throw new Error(
              'Weight file with basename \'' + pathBasename +
              '\' is not provided.');
        } else {
          pathToFile[path] = files[fileNames.indexOf(pathBasename)];
        }
      });
    }
    if (basenames.length !== files.length) {
      throw new Error(
          'Mismatch in the number of files in weights manifest ' +
          ('(' + basenames.length +
           ') and the number of weight files provided ') +
          ('(' + files.length + ').'));
    }
    return pathToFile;
  };
  return BrowserFiles;
}());
var browserDownloadsRouter = function(url) {
  if (!env().getBool('IS_BROWSER')) {
    return null;
  } else {
    if (!Array.isArray(url) && url.startsWith(BrowserDownloads.URL_SCHEME)) {
      return browserDownloads(url.slice(BrowserDownloads.URL_SCHEME.length));
    } else {
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
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
function browserDownloads(fileNamePrefix) {
  if (fileNamePrefix === void 0) {
    fileNamePrefix = 'model';
  }
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
 *
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
 * Monitor Promise.all progress, fire onProgress callback function.
 *
 * @param promises Promise list going to be monitored
 * @param onProgress Callback function. Fired when a promise resolved.
 * @param startFraction Optional fraction start. Default to 0.
 * @param endFraction Optional fraction end. Default to 1.
 */
function monitorPromisesProgress(
    promises, onProgress, startFraction, endFraction) {
  checkPromises(promises);
  startFraction = startFraction == null ? 0 : startFraction;
  endFraction = endFraction == null ? 1 : endFraction;
  checkFraction(startFraction, endFraction);
  var resolvedPromise = 0;
  var registerMonitor = function(promise) {
    promise.then(function(value) {
      var fraction = startFraction +
          ++resolvedPromise / promises.length * (endFraction - startFraction);
      // pass fraction as parameter to callback function.
      onProgress(fraction);
      return value;
    });
    return promise;
  };
  function checkPromises(promises) {
    assert(
        promises != null && Array.isArray(promises) && promises.length > 0,
        function() {
          return 'promises must be a none empty array';
        });
  }
  function checkFraction(startFraction, endFraction) {
    assert(startFraction >= 0 && startFraction <= 1, function() {
      return 'Progress fraction must be in range [0, 1], but ' +
          ('got startFraction ' + startFraction);
    });
    assert(endFraction >= 0 && endFraction <= 1, function() {
      return 'Progress fraction must be in range [0, 1], but ' +
          ('got endFraction ' + endFraction);
    });
    assert(endFraction >= startFraction, function() {
      return 'startFraction must be no more than endFraction, but ' +
          ('got startFraction ' + startFraction + ' and endFraction ') +
          ('' + endFraction);
    });
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
function loadWeightsAsArrayBuffer(fetchURLs, loadOptions) {
  return __awaiter(this, void 0, void 0, function() {
    var fetchFunc, requests, fetchStartFraction, fetchEndFraction, responses,
        _a, bufferPromises, bufferStartFraction, bufferEndFraction, buffers, _b;
    return __generator(this, function(_c) {
      switch (_c.label) {
        case 0:
          if (loadOptions == null) {
            loadOptions = {};
          }
          fetchFunc = loadOptions.fetchFunc == null ? env().platform.fetch :
                                                      loadOptions.fetchFunc;
          requests = fetchURLs.map(function(fetchURL) {
            return fetchFunc(
                fetchURL, loadOptions.requestInit, {isBinary: true});
          });
          fetchStartFraction = 0;
          fetchEndFraction = 0.5;
          if (!(loadOptions.onProgress == null)) return [3 /*break*/, 2];
          return [4 /*yield*/, Promise.all(requests)];
        case 1:
          _a = _c.sent();
          return [3 /*break*/, 4];
        case 2:
          return [
            4 /*yield*/,
            monitorPromisesProgress(
                requests, loadOptions.onProgress, fetchStartFraction,
                fetchEndFraction)
          ];
        case 3:
          _a = _c.sent();
          _c.label = 4;
        case 4:
          responses = _a;
          bufferPromises = responses.map(function(response) {
            return response.arrayBuffer();
          });
          bufferStartFraction = 0.5;
          bufferEndFraction = 1;
          if (!(loadOptions.onProgress == null)) return [3 /*break*/, 6];
          return [4 /*yield*/, Promise.all(bufferPromises)];
        case 5:
          _b = _c.sent();
          return [3 /*break*/, 8];
        case 6:
          return [
            4 /*yield*/,
            monitorPromisesProgress(
                bufferPromises, loadOptions.onProgress, bufferStartFraction,
                bufferEndFraction)
          ];
        case 7:
          _b = _c.sent();
          _c.label = 8;
        case 8:
          buffers = _b;
          return [2 /*return*/, buffers];
      }
    });
  });
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
function loadWeights(manifest, filePathPrefix, weightNames, requestInit) {
  if (filePathPrefix === void 0) {
    filePathPrefix = '';
  }
  return __awaiter(this, void 0, void 0, function() {
    var fetchWeights, loadWeights;
    return __generator(this, function(_a) {
      fetchWeights = function(fetchUrls) {
        return loadWeightsAsArrayBuffer(fetchUrls, {requestInit: requestInit});
      };
      loadWeights = weightsLoaderFactory(fetchWeights);
      return [2 /*return*/, loadWeights(manifest, filePathPrefix, weightNames)];
    });
  });
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
  var _this = this;
  return function(manifest, filePathPrefix, weightNames) {
    if (filePathPrefix === void 0) {
      filePathPrefix = '';
    }
    return __awaiter(_this, void 0, void 0, function() {
      var groupIndicesToFetchMap, groupWeightsToFetch, weightsFound,
          allManifestWeightNames, weightsNotFound, groupIndicesToFetch,
          fetchUrls, buffers, weightsTensorMap, bufferIndexOffset;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            groupIndicesToFetchMap = manifest.map(function() {
              return false;
            });
            groupWeightsToFetch = {};
            weightsFound = weightNames != null ? weightNames.map(function() {
              return false;
            }) :
                                                 [];
            allManifestWeightNames = [];
            manifest.forEach(function(manifestGroupConfig, groupIndex) {
              var groupOffset = 0;
              manifestGroupConfig.weights.forEach(function(weightsEntry) {
                var rawDtype = ('quantization' in weightsEntry) ?
                    weightsEntry.quantization.dtype :
                    weightsEntry.dtype;
                var weightsBytes = DTYPE_VALUE_SIZE_MAP[rawDtype] *
                    sizeFromShape(weightsEntry.shape);
                var enqueueWeightsForFetchingFn = function() {
                  groupIndicesToFetchMap[groupIndex] = true;
                  if (groupWeightsToFetch[groupIndex] == null) {
                    groupWeightsToFetch[groupIndex] = [];
                  }
                  groupWeightsToFetch[groupIndex].push({
                    manifestEntry: weightsEntry,
                    groupOffset: groupOffset,
                    sizeBytes: weightsBytes
                  });
                };
                if (weightNames != null) {
                  weightNames.forEach(function(weightName, weightIndex) {
                    if (weightName === weightsEntry.name) {
                      enqueueWeightsForFetchingFn();
                      weightsFound[weightIndex] = true;
                    }
                  });
                } else {
                  enqueueWeightsForFetchingFn();
                }
                allManifestWeightNames.push(weightsEntry.name);
                groupOffset += weightsBytes;
              });
            });
            if (!weightsFound.every(function(found) {
                  return found;
                })) {
              weightsNotFound = weightNames.filter(function(_, i) {
                return !weightsFound[i];
              });
              throw new Error(
                  'Could not find weights in manifest with names: ' +
                  (weightsNotFound.join(', ') + '. \n') +
                  'Manifest JSON has weights with names: ' +
                  (allManifestWeightNames.join(', ') + '.'));
            }
            groupIndicesToFetch = groupIndicesToFetchMap.reduce(
                function(accumulator, shouldFetch, i) {
                  if (shouldFetch) {
                    accumulator.push(i);
                  }
                  return accumulator;
                },
                []);
            fetchUrls = [];
            groupIndicesToFetch.forEach(function(i) {
              manifest[i].paths.forEach(function(filepath) {
                var fetchUrl = filePathPrefix +
                    (!filePathPrefix.endsWith('/') ? '/' : '') + filepath;
                fetchUrls.push(fetchUrl);
              });
            });
            return [4 /*yield*/, fetchWeightsFunction(fetchUrls)];
          case 1:
            buffers = _a.sent();
            weightsTensorMap = {};
            bufferIndexOffset = 0;
            groupIndicesToFetch.forEach(function(i) {
              var numBuffers = manifest[i].paths.length;
              var groupBytes = 0;
              for (var i_1 = 0; i_1 < numBuffers; i_1++) {
                groupBytes += buffers[bufferIndexOffset + i_1].byteLength;
              }
              // Create a buffer for the whole group.
              var groupBuffer = new ArrayBuffer(groupBytes);
              var groupByteBuffer = new Uint8Array(groupBuffer);
              var groupBufferOffset = 0;
              for (var i_2 = 0; i_2 < numBuffers; i_2++) {
                var buffer = new Uint8Array(buffers[bufferIndexOffset + i_2]);
                groupByteBuffer.set(buffer, groupBufferOffset);
                groupBufferOffset += buffer.byteLength;
              }
              var weightsEntries = groupWeightsToFetch[i];
              weightsEntries.forEach(function(weightsEntry) {
                var byteBuffer = groupBuffer.slice(
                    weightsEntry.groupOffset,
                    weightsEntry.groupOffset + weightsEntry.sizeBytes);
                var nameToTensorMap =
                    decodeWeights(byteBuffer, [weightsEntry.manifestEntry]);
                for (var name_1 in nameToTensorMap) {
                  weightsTensorMap[name_1] = nameToTensorMap[name_1];
                }
              });
              bufferIndexOffset += numBuffers;
            });
            return [2 /*return*/, weightsTensorMap];
        }
      });
    });
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
var OCTET_STREAM_MIME_TYPE = 'application/octet-stream';
var JSON_TYPE = 'application/json';
var HTTPRequest = /** @class */ (function() {
  function HTTPRequest(path, loadOptions) {
    this.DEFAULT_METHOD = 'POST';
    if (loadOptions == null) {
      loadOptions = {};
    }
    this.weightPathPrefix = loadOptions.weightPathPrefix;
    this.onProgress = loadOptions.onProgress;
    this.weightUrlConverter = loadOptions.weightUrlConverter;
    if (loadOptions.fetchFunc != null) {
      assert(typeof loadOptions.fetchFunc === 'function', function() {
        return 'Must pass a function that matches the signature of ' +
            '`fetch` (see ' +
            'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)';
      });
      this.fetch = loadOptions.fetchFunc;
    } else {
      this.fetch = env().platform.fetch;
    }
    assert(path != null && path.length > 0, function() {
      return 'URL path for http must not be null, undefined or ' +
          'empty.';
    });
    if (Array.isArray(path)) {
      assert(path.length === 2, function() {
        return 'URL paths for http must have a length of 2, ' +
            ('(actual length is ' + path.length + ').');
      });
    }
    this.path = path;
    if (loadOptions.requestInit != null &&
        loadOptions.requestInit.body != null) {
      throw new Error(
          'requestInit is expected to have no pre-existing body, but has one.');
    }
    this.requestInit = loadOptions.requestInit || {};
  }
  HTTPRequest.prototype.save = function(modelArtifacts) {
    return __awaiter(this, void 0, void 0, function() {
      var init, weightsManifest, modelTopologyAndWeightManifest, response;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
              throw new Error(
                  'BrowserHTTPRequest.save() does not support saving model topology ' +
                  'in binary formats yet.');
            }
            init =
                Object.assign({method: this.DEFAULT_METHOD}, this.requestInit);
            init.body = new FormData();
            weightsManifest = [{
              paths: ['./model.weights.bin'],
              weights: modelArtifacts.weightSpecs,
            }];
            modelTopologyAndWeightManifest = {
              modelTopology: modelArtifacts.modelTopology,
              format: modelArtifacts.format,
              generatedBy: modelArtifacts.generatedBy,
              convertedBy: modelArtifacts.convertedBy,
              weightsManifest: weightsManifest
            };
            if (modelArtifacts.signature != null) {
              modelTopologyAndWeightManifest.signature =
                  modelArtifacts.signature;
            }
            if (modelArtifacts.userDefinedMetadata != null) {
              modelTopologyAndWeightManifest.userDefinedMetadata =
                  modelArtifacts.userDefinedMetadata;
            }
            if (modelArtifacts.modelInitializer != null) {
              modelTopologyAndWeightManifest.modelInitializer =
                  modelArtifacts.modelInitializer;
            }
            init.body.append(
                'model.json',
                new Blob(
                    [JSON.stringify(modelTopologyAndWeightManifest)],
                    {type: JSON_TYPE}),
                'model.json');
            if (modelArtifacts.weightData != null) {
              init.body.append(
                  'model.weights.bin', new Blob([modelArtifacts.weightData], {
                    type: OCTET_STREAM_MIME_TYPE
                  }),
                  'model.weights.bin');
            }
            return [4 /*yield*/, this.fetch(this.path, init)];
          case 1:
            response = _a.sent();
            if (response.ok) {
              return [
                2 /*return*/, {
                  modelArtifactsInfo:
                      getModelArtifactsInfoForJSON(modelArtifacts),
                  responses: [response],
                }
              ];
            } else {
              throw new Error(
                  'BrowserHTTPRequest.save() failed due to HTTP response status ' +
                  (response.status + '.'));
            }
        }
      });
    });
  };
  /**
   * Load model artifacts via HTTP request(s).
   *
   * See the documentation to `tf.io.http` for details on the saved
   * artifacts.
   *
   * @returns The loaded model artifacts (if loading succeeds).
   */
  HTTPRequest.prototype.load = function() {
    return __awaiter(this, void 0, void 0, function() {
      var modelConfigRequest, modelConfig, e_1, message, modelTopology,
          weightsManifest, generatedBy, convertedBy, format, signature,
          userDefinedMetadata, weightSpecs, weightData, results, artifacts,
          initializer;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.fetch(this.path, this.requestInit)];
          case 1:
            modelConfigRequest = _a.sent();
            if (!modelConfigRequest.ok) {
              throw new Error(
                  'Request to ' + this.path + ' failed with status code ' +
                  (modelConfigRequest.status +
                   '. Please verify this URL points to ') +
                  'the model JSON of the model to load.');
            }
            _a.label = 2;
          case 2:
            _a.trys.push([2, 4, , 5]);
            return [4 /*yield*/, modelConfigRequest.json()];
          case 3:
            modelConfig = _a.sent();
            return [3 /*break*/, 5];
          case 4:
            e_1 = _a.sent();
            message = 'Failed to parse model JSON of response from ' +
                this.path + '.';
            // TODO(nsthorat): Remove this after some time when we're
            // comfortable that .pb files are mostly gone.
            if (this.path.endsWith('.pb')) {
              message += ' Your path contains a .pb file extension. ' +
                  'Support for .pb models have been removed in TensorFlow.js 1.0 ' +
                  'in favor of .json models. You can re-convert your Python ' +
                  'TensorFlow model using the TensorFlow.js 1.0 conversion scripts ' +
                  'or you can convert your.pb models with the \'pb2json\'' +
                  'NPM script in the tensorflow/tfjs-converter repository.';
            } else {
              message += ' Please make sure the server is serving valid ' +
                  'JSON for this request.';
            }
            throw new Error(message);
          case 5:
            modelTopology = modelConfig.modelTopology;
            weightsManifest = modelConfig.weightsManifest;
            generatedBy = modelConfig.generatedBy;
            convertedBy = modelConfig.convertedBy;
            format = modelConfig.format;
            signature = modelConfig.signature;
            userDefinedMetadata = modelConfig.userDefinedMetadata;
            // We do not allow both modelTopology and weightsManifest to be
            // missing.
            if (modelTopology == null && weightsManifest == null) {
              throw new Error(
                  'The JSON from HTTP path ' + this.path +
                  ' contains neither model ' +
                  'topology or manifest for weights.');
            }
            if (!(weightsManifest != null)) return [3 /*break*/, 7];
            return [4 /*yield*/, this.loadWeights(weightsManifest)];
          case 6:
            results = _a.sent();
            weightSpecs = results[0], weightData = results[1];
            _a.label = 7;
          case 7:
            artifacts = {
              modelTopology: modelTopology,
              weightSpecs: weightSpecs,
              weightData: weightData,
              generatedBy: generatedBy,
              convertedBy: convertedBy,
              format: format
            };
            if (signature != null) {
              artifacts.signature = signature;
            }
            if (userDefinedMetadata != null) {
              artifacts.userDefinedMetadata = userDefinedMetadata;
            }
            initializer = modelConfig.modelInitializer;
            if (initializer) {
              artifacts.modelInitializer = initializer;
            }
            return [2 /*return*/, artifacts];
        }
      });
    });
  };
  HTTPRequest.prototype.loadWeights = function(weightsManifest) {
    return __awaiter(this, void 0, void 0, function() {
      var weightPath, _a, prefix, suffix, pathPrefix, weightSpecs, _i,
          weightsManifest_1, entry, fetchURLs, urlPromises, _b,
          weightsManifest_2, weightsGroup, _c, _d, path, _e, _f, _g, buffers;
      return __generator(this, function(_h) {
        switch (_h.label) {
          case 0:
            weightPath = Array.isArray(this.path) ? this.path[1] : this.path;
            _a = parseUrl(weightPath), prefix = _a[0], suffix = _a[1];
            pathPrefix = this.weightPathPrefix || prefix;
            weightSpecs = [];
            for (_i = 0, weightsManifest_1 = weightsManifest;
                 _i < weightsManifest_1.length; _i++) {
              entry = weightsManifest_1[_i];
              weightSpecs.push.apply(weightSpecs, entry.weights);
            }
            fetchURLs = [];
            urlPromises = [];
            for (_b = 0, weightsManifest_2 = weightsManifest;
                 _b < weightsManifest_2.length; _b++) {
              weightsGroup = weightsManifest_2[_b];
              for (_c = 0, _d = weightsGroup.paths; _c < _d.length; _c++) {
                path = _d[_c];
                if (this.weightUrlConverter != null) {
                  urlPromises.push(this.weightUrlConverter(path));
                } else {
                  fetchURLs.push(pathPrefix + path + suffix);
                }
              }
            }
            if (!this.weightUrlConverter) return [3 /*break*/, 2];
            _f = (_e = fetchURLs.push).apply;
            _g = [fetchURLs];
            return [4 /*yield*/, Promise.all(urlPromises)];
          case 1:
            _f.apply(_e, _g.concat([_h.sent()]));
            _h.label = 2;
          case 2:
            return [
              4 /*yield*/, loadWeightsAsArrayBuffer(fetchURLs, {
                requestInit: this.requestInit,
                fetchFunc: this.fetch,
                onProgress: this.onProgress
              })
            ];
          case 3:
            buffers = _h.sent();
            return [
              2 /*return*/, [weightSpecs, concatenateArrayBuffers(buffers)]
            ];
        }
      });
    });
  };
  HTTPRequest.URL_SCHEME_REGEX = /^https?:\/\//;
  return HTTPRequest;
}());
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
  var lastSlash = url.lastIndexOf('/');
  var lastSearchParam = url.lastIndexOf('?');
  var prefix = url.substring(0, lastSlash);
  var suffix =
      lastSearchParam > lastSlash ? url.substring(lastSearchParam) : '';
  return [prefix + '/', suffix];
}
function isHTTPScheme(url) {
  return url.match(HTTPRequest.URL_SCHEME_REGEX) != null;
}
var httpRouter = function(url, loadOptions) {
  if (typeof fetch === 'undefined' &&
      (loadOptions == null || loadOptions.fetchFunc == null)) {
    // `http` uses `fetch` or `node-fetch`, if one wants to use it in
    // an environment that is not the browser or node they have to setup a
    // global fetch polyfill.
    return null;
  } else {
    var isHTTP = true;
    if (Array.isArray(url)) {
      isHTTP = url.every(function(urlItem) {
        return isHTTPScheme(urlItem);
      });
    } else {
      isHTTP = isHTTPScheme(url);
    }
    if (isHTTP) {
      return http(url, loadOptions);
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
 *
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
var PassthroughLoader = /** @class */ (function() {
  function PassthroughLoader(modelArtifacts) {
    this.modelArtifacts = modelArtifacts;
  }
  PassthroughLoader.prototype.load = function() {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        return [2 /*return*/, this.modelArtifacts];
      });
    });
  };
  return PassthroughLoader;
}());
var PassthroughSaver = /** @class */ (function() {
  function PassthroughSaver(saveHandler) {
    this.saveHandler = saveHandler;
  }
  PassthroughSaver.prototype.save = function(modelArtifacts) {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        return [2 /*return*/, this.saveHandler(modelArtifacts)];
      });
    });
  };
  return PassthroughSaver;
}());
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
    var isModelArtifacts = modelArtifacts.modelTopology != null ||
        modelArtifacts.weightSpecs != null;
    if (isModelArtifacts) {
      return new PassthroughLoader(modelArtifacts);
    } else {
      // Legacy support: with only modelTopology.
      // TODO(cais): Remove this deprecated API.
      console.warn(
          'Please call tf.io.fromMemory() with only one argument. ' +
          'The argument should be of type ModelArtifacts. ' +
          'The multi-argument signature of tf.io.fromMemory() has been ' +
          'deprecated and will be removed in a future release.');
      return new PassthroughLoader({modelTopology: modelArtifacts});
    }
  } else {
    // Legacy support.
    // TODO(cais): Remove this deprecated API.
    console.warn(
        'Please call tf.io.fromMemory() with only one argument. ' +
        'The argument should be of type ModelArtifacts. ' +
        'The multi-argument signature of tf.io.fromMemory() has been ' +
        'deprecated and will be removed in a future release.');
    return new PassthroughLoader({
      modelTopology: modelArtifacts,
      weightSpecs: weightSpecs,
      weightData: weightData,
      trainingConfig: trainingConfig
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

var io = {
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
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function matMul_(a, b, transposeA, transposeB) {
  var _a;
  if (transposeA === void 0) {
    transposeA = false;
  }
  if (transposeB === void 0) {
    transposeB = false;
  }
  var $a = convertToTensor(a, 'a', 'matMul');
  var $b = convertToTensor(b, 'b', 'matMul');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  var inputs = {a: $a, b: $b};
  var attrs = {transposeA: transposeA, transposeB: transposeB};
  return ENGINE.runKernel(BatchMatMul, inputs, attrs);
}
var matMul = op({matMul_: matMul_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function oneHot_(indices, depth, onValue, offValue) {
  if (onValue === void 0) {
    onValue = 1;
  }
  if (offValue === void 0) {
    offValue = 0;
  }
  if (depth < 2) {
    throw new Error('Error in oneHot: depth must be >=2, but it is ' + depth);
  }
  var $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');
  var inputs = {indices: $indices};
  var attrs = {depth: depth, onValue: onValue, offValue: offValue};
  return ENGINE.runKernel(OneHot, inputs, attrs);
}
var oneHot = op({oneHot_: oneHot_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function transpose_(x, perm) {
  var $x = convertToTensor(x, 'x', 'transpose');
  if (perm == null) {
    perm = $x.shape
               .map(function(s, i) {
                 return i;
               })
               .reverse();
  }
  assert($x.rank === perm.length, function() {
    return 'Error in transpose: rank of input ' + $x.rank + ' ' +
        ('must match length of perm ' + perm + '.');
  });
  perm.forEach(function(axis) {
    assert(axis >= 0 && axis < $x.rank, function() {
      return 'All entries in \'perm\' must be between 0 and ' + ($x.rank - 1) +
          (' but got ' + perm);
    });
  });
  if ($x.rank <= 1) {
    return $x.clone();
  }
  var inputs = {x: $x};
  var attrs = {perm: perm};
  return ENGINE.runKernel(Transpose, inputs, attrs);
}
var transpose = op({transpose_: transpose_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function confusionMatrix_(labels, predictions, numClasses) {
  var $labels = convertToTensor(labels, 'labels', 'confusionMatrix');
  var $predictions =
      convertToTensor(predictions, 'predictions', 'confusionMatrix');
  assert(
      numClasses == null || numClasses > 0 && Number.isInteger(numClasses),
      function() {
        return 'If provided, numClasses must be a positive integer, ' +
            ('but got ' + numClasses);
      });
  assert($labels.rank === 1, function() {
    return 'Expected the rank of labels to be 1, but got ' + $labels.rank;
  });
  assert($predictions.rank === 1, function() {
    return 'Expected the rank of predictions to be 1, ' +
        ('but got ' + $predictions.rank);
  });
  assert($labels.shape[0] === $predictions.shape[0], function() {
    return 'Mismatch in the number of examples: ' +
        ($labels.shape[0] + ' vs. ' + $predictions.shape[0] + '. ') +
        'Labels and predictions should have the same number of elements.';
  });
  assert(numClasses > 0 && Number.isInteger(numClasses), function() {
    return 'numClasses is required to be a positive integer, but got ' +
        ('' + numClasses);
  });
  // TODO(cais): In the future, if oneHot supports tensors inputs for
  //   `numClasses`, `confusionMatrix` can make `numClasses` optional.
  var oneHotLabels = oneHot(cast($labels, 'int32'), numClasses);
  var oneHotPredictions = oneHot(cast($predictions, 'int32'), numClasses);
  var oneHotLabelsT = transpose(oneHotLabels);
  var product = matMul(oneHotLabelsT, oneHotPredictions);
  return cast(product, 'int32');
}
var confusionMatrix = op({confusionMatrix_: confusionMatrix_});

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

var math = {__proto__: null, confusionMatrix: confusionMatrix};

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor3d(values, shape, dtype) {
  assertNonNull(values);
  if (shape != null && shape.length !== 3) {
    throw new Error('tensor3d() requires shape to have three numbers');
  }
  var inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 3 && inferredShape.length !== 1) {
    throw new Error(
        'tensor3d() requires values to be number[][][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor3d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  return makeTensor(values, shape, inferredShape, dtype);
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
var fromPixels2DContext;
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
 *
 * @returns A Tensor3D with the shape `[height, width, numChannels]`.
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
function fromPixels_(pixels, numChannels) {
  if (numChannels === void 0) {
    numChannels = 3;
  }
  // Sanity checks.
  if (numChannels > 4) {
    throw new Error(
        'Cannot construct Tensor with more than 4 channels from pixels.');
  }
  if (pixels == null) {
    throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
  }
  var isPixelData = false;
  var isImageData = false;
  var isVideo = false;
  var isImage = false;
  var isCanvasLike = false;
  var isImageBitmap = false;
  if (pixels.data instanceof Uint8Array) {
    isPixelData = true;
  } else if (
      typeof (ImageData) !== 'undefined' && pixels instanceof ImageData) {
    isImageData = true;
  } else if (
      typeof (HTMLVideoElement) !== 'undefined' &&
      pixels instanceof HTMLVideoElement) {
    isVideo = true;
  } else if (
      typeof (HTMLImageElement) !== 'undefined' &&
      pixels instanceof HTMLImageElement) {
    isImage = true;
    // tslint:disable-next-line: no-any
  } else if (pixels.getContext != null) {
    isCanvasLike = true;
  } else if (
      typeof (ImageBitmap) !== 'undefined' && pixels instanceof ImageBitmap) {
    isImageBitmap = true;
  } else {
    throw new Error(
        'pixels passed to tf.browser.fromPixels() must be either an ' +
        'HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ' +
        'in browser, or OffscreenCanvas, ImageData in webworker' +
        ' or {data: Uint32Array, width: number, height: number}, ' +
        ('but was ' + pixels.constructor.name));
  }
  if (isVideo) {
    var HAVE_CURRENT_DATA_READY_STATE = 2;
    if (isVideo && pixels.readyState < HAVE_CURRENT_DATA_READY_STATE) {
      throw new Error(
          'The video element has not loaded data yet. Please wait for ' +
          '`loadeddata` event on the <video> element.');
    }
  }
  // If the current backend has 'FromPixels' registered, it has a more
  // efficient way of handling pixel uploads, so we call that.
  var kernel = getKernel(FromPixels, ENGINE.backendName);
  if (kernel != null) {
    var inputs = {pixels: pixels};
    var attrs = {numChannels: numChannels};
    return ENGINE.runKernel(FromPixels, inputs, attrs);
  }
  var _a = isVideo ? [pixels.videoWidth, pixels.videoHeight] :
                     [pixels.width, pixels.height],
      width = _a[0], height = _a[1];
  var vals;
  if (isCanvasLike) {
    vals =
        // tslint:disable-next-line:no-any
        pixels.getContext('2d').getImageData(0, 0, width, height).data;
  } else if (isImageData || isPixelData) {
    vals = pixels.data;
  } else if (isImage || isVideo || isImageBitmap) {
    if (fromPixels2DContext == null) {
      fromPixels2DContext = document.createElement('canvas').getContext('2d');
    }
    fromPixels2DContext.canvas.width = width;
    fromPixels2DContext.canvas.height = height;
    fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
    vals = fromPixels2DContext.getImageData(0, 0, width, height).data;
  }
  var values;
  if (numChannels === 4) {
    values = new Int32Array(vals);
  } else {
    var numPixels = width * height;
    values = new Int32Array(numPixels * numChannels);
    for (var i = 0; i < numPixels; i++) {
      for (var channel = 0; channel < numChannels; ++channel) {
        values[i * numChannels + channel] = vals[i * 4 + channel];
      }
    }
  }
  var outShape = [height, width, numChannels];
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
 * @param img A rank-2 tensor with shape `[height, width]`, or a rank-3 tensor
 * of shape `[height, width, numChannels]`. If rank-2, draws grayscale. If
 * rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
 * grayscale. When depth of 3, we draw with the first three components of
 * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
 * 4, all four components of the depth dimension correspond to r, g, b, a.
 * @param canvas The canvas to draw to.
 *
 * @doc {heading: 'Browser', namespace: 'browser'}
 */
function toPixels(img, canvas) {
  return __awaiter(this, void 0, void 0, function() {
    var $img, originalImgTensor, _a, height, width, depth, data, multiplier,
        bytes, i, rgba, d, value, j, ctx, imageData;
    return __generator(this, function(_b) {
      switch (_b.label) {
        case 0:
          $img = convertToTensor(img, 'img', 'toPixels');
          if (!(img instanceof Tensor)) {
            originalImgTensor = $img;
            $img = cast(originalImgTensor, 'int32');
            originalImgTensor.dispose();
          }
          if ($img.rank !== 2 && $img.rank !== 3) {
            throw new Error(
                'toPixels only supports rank 2 or 3 tensors, got rank ' +
                $img.rank + '.');
          }
          _a = $img.shape.slice(0, 2), height = _a[0], width = _a[1];
          depth = $img.rank === 2 ? 1 : $img.shape[2];
          if (depth > 4 || depth === 2) {
            throw new Error(
                'toPixels only supports depth of size ' +
                ('1, 3 or 4 but got ' + depth));
          }
          if ($img.dtype !== 'float32' && $img.dtype !== 'int32') {
            throw new Error(
                'Unsupported type for toPixels: ' + $img.dtype + '.' +
                ' Please use float32 or int32 tensors.');
          }
          return [4 /*yield*/, $img.data()];
        case 1:
          data = _b.sent();
          multiplier = $img.dtype === 'float32' ? 255 : 1;
          bytes = new Uint8ClampedArray(width * height * 4);
          for (i = 0; i < height * width; ++i) {
            rgba = [0, 0, 0, 255];
            for (d = 0; d < depth; d++) {
              value = data[i * depth + d];
              if ($img.dtype === 'float32') {
                if (value < 0 || value > 1) {
                  throw new Error(
                      'Tensor values for a float32 Tensor must be in the ' +
                      ('range [0 - 1] but encountered ' + value + '.'));
                }
              } else if ($img.dtype === 'int32') {
                if (value < 0 || value > 255) {
                  throw new Error(
                      'Tensor values for a int32 Tensor must be in the ' +
                      ('range [0 - 255] but encountered ' + value + '.'));
                }
              }
              if (depth === 1) {
                rgba[0] = value * multiplier;
                rgba[1] = value * multiplier;
                rgba[2] = value * multiplier;
              } else {
                rgba[d] = value * multiplier;
              }
            }
            j = i * 4;
            bytes[j + 0] = Math.round(rgba[0]);
            bytes[j + 1] = Math.round(rgba[1]);
            bytes[j + 2] = Math.round(rgba[2]);
            bytes[j + 3] = Math.round(rgba[3]);
          }
          if (canvas != null) {
            canvas.width = width;
            canvas.height = height;
            ctx = canvas.getContext('2d');
            imageData = new ImageData(bytes, width, height);
            ctx.putImageData(imageData, 0, 0);
          }
          if ($img !== img) {
            $img.dispose();
          }
          return [2 /*return*/, bytes];
      }
    });
  });
}
var fromPixels = op({fromPixels_: fromPixels_});

var browser = {__proto__: null, toPixels: toPixels, fromPixels: fromPixels};

/**
 * Validate gather nd inputs.
 *
 * @param tensor The tensor contains the source values.
 * @param indices The tensor contains the indices to slice the source.
 *
 * @returns [resultShape, numUpdates, sliceSize, strides]
 */
function prepareAndValidate(tensor, indices) {
  var tensorRank = tensor.shape.length;
  var indicesRank = indices.shape.length;
  if (tensorRank < 1) {
    throw new Error(
        'tf.gatherND() expects the input to be rank 1 or higher,' +
        (' but the rank was ' + tensorRank + '.'));
  }
  if (indicesRank < 1) {
    throw new Error(
        'tf.gatherND() expects the indices to be rank 1 or higher,' +
        (' but the rank was ' + indicesRank + '.'));
  }
  if (indices.dtype !== 'int32') {
    throw new Error(
        'tf.gatherND() expects the indices to be int32 type,' +
        (' but the dtype was ' + indices.dtype + '.'));
  }
  if (indices.shape[indicesRank - 1] > tensorRank) {
    throw new Error(
        'index innermost dimension length must be <= tensor rank; saw: ' +
        (indices.shape[indicesRank - 1] + ' vs. ' + tensorRank));
  }
  if (sizeFromShape(tensor.shape) === 0) {
    throw new Error(
        'Requested more than 0 entries, but input is empty.' +
        (' Input shape: ' + tensor.shape + '.'));
  }
  var indicesShape = indices.shape;
  var sliceRank = indicesShape[indicesShape.length - 1];
  // The result shape is
  //   indices.shape[:-1] + params.shape[indices.shape[-1]:]
  var nResult = 1;
  for (var i = 0; i < indicesShape.length - 1; ++i) {
    nResult *= indicesShape[i];
  }
  var inputShape = tensor.shape;
  var resultShape = indicesShape.slice();
  resultShape.pop();
  var sliceSize = 1;
  for (var i = sliceRank; i < tensorRank; ++i) {
    sliceSize *= inputShape[i];
    resultShape.push(inputShape[i]);
  }
  var strides = computeStrides(tensor.shape)
                    .map(function(stride) {
                      return stride / sliceSize;
                    })
                    .concat([1])
                    .slice(0, sliceRank);
  return [resultShape, nResult, sliceSize, strides];
}

var gather_nd_util = {__proto__: null, prepareAndValidate: prepareAndValidate};

/**
 * Check whether updates.shape = indices.shape[:batchDim] +
 * shape[sliceDim:]
 *
 * @param x The input tensor.
 */
function validateUpdateShape(shape, indices, updates) {
  var sliceDim = (indices.rank > 1) ? indices.shape[indices.rank - 1] : 1;
  var batchDim = (indices.rank > 1) ? indices.rank - 1 : 1;
  var shapeError = 'Must have updates.shape = indices.shape[:batchDim] + ' +
      ('shape[sliceDim:], got updates.shape: ' + updates.shape) +
      (', indices.shape: ' + indices.shape + ', shape: ' + shape) +
      (', sliceDim: ' + sliceDim + ', and batchDim: ' + batchDim + '.');
  if (updates.rank < batchDim) {
    throw new Error(shapeError + (' update.rank < ' + batchDim + '. '));
  }
  if (shape.length < sliceDim + (updates.rank - batchDim)) {
    throw new Error(
        shapeError +
        (' Output shape length < ' + (sliceDim + (updates.rank - batchDim))));
  }
  if (updates.rank !== batchDim + shape.length - sliceDim) {
    throw new Error(
        shapeError +
        (' update.rank != ' + (batchDim + shape.length - sliceDim)));
  }
  for (var d = 0; d < batchDim; ++d) {
    if (updates.shape[d] !== indices.shape[d]) {
      throw new Error(
          shapeError +
          (' updates.shape[' + d + '] (' + updates.shape[d] +
           ') != indices.shape[' + d + '] (' + indices.shape[d] + ').'));
    }
  }
  for (var d = 0; d < updates.rank - batchDim; ++d) {
    if (updates.shape[d + batchDim] !== shape[d + sliceDim]) {
      throw new Error(
          shapeError +
          (' updates.shape[' + (d + batchDim) + '] (' +
           updates.shape[d + batchDim] + ') != shape[' + (d + batchDim) +
           '] (' + shape[d + batchDim] + ')'));
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
    throw new Error(
        'tf.scatterND() expects the indices to be rank 1 or higher,' +
        (' but the rank was ' + indices.rank + '.'));
  }
  if (updates.rank < 1) {
    throw new Error(
        'tf.scatterND() expects the updates to be rank 1 or higher,' +
        (' but the rank was ' + updates.rank + '.'));
  }
  if (indices.dtype !== 'int32') {
    throw new Error(
        'The dtype of \'indices\' should be int32, but got dtype: ' +
        indices.dtype);
  }
  if (shape.length < 1) {
    throw new Error(
        'Output rank must be greater or equal to 1, but got shape: ' + shape);
  }
  if (shape.length === 0) {
    if (indices.size === 0) {
      throw new Error(
          'Indices specified for empty output. indices shape: ' +
          indices.shape);
    }
    if (updates.size === 0) {
      throw new Error(
          'Updates specified for empty output. updates shape: ' +
          updates.shape);
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
  var indicesRank = indices.shape.length;
  var sliceRank = (indicesRank > 1) ? indices.shape[indicesRank - 1] : 1;
  // Calculate the number of elements that make up each slice of our updated
  // tensor. This allows us to work with flattened tensors and copy over whole
  // slices at a time.
  var totalNd = shape.length;
  var sliceSize = 1;
  for (var i = sliceRank; i < totalNd; ++i) {
    sliceSize *= shape[i];
  }
  var safeSliceDim = (sliceRank < 1) ? 1 : sliceRank;
  var numUpdates = sizeFromShape(indices.shape) / safeSliceDim;
  var strides = computeStrides(shape.slice(0, sliceRank)).concat([1]);
  var outputSize = sizeFromShape(shape);
  return {
    sliceRank: sliceRank,
    numUpdates: numUpdates,
    sliceSize: sliceSize,
    strides: strides,
    outputSize: outputSize
  };
}

var scatter_nd_util = {
  __proto__: null,
  validateUpdateShape: validateUpdateShape,
  validateInput: validateInput,
  calculateShapes: calculateShapes
};

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
  var inputRank = input.shape.length;
  assert(inputRank === begin.length, function() {
    return 'Error in slice' + inputRank + 'D: Length of begin ' + begin +
        ' must ' + ('match the rank of the array (' + inputRank + ').');
  });
  assert(inputRank === size.length, function() {
    return 'Error in slice' + inputRank + 'D: Length of size ' + size +
        ' must ' + ('match the rank of the array (' + inputRank + ').');
  });
  var _loop_1 = function(i) {
    assert(begin[i] + size[i] <= input.shape[i], function() {
      return 'Error in slice' + inputRank + 'D: begin[' + i + '] + size[' + i +
          '] ' +
          ('(' + (begin[i] + size[i]) + ') would overflow input.shape[' + i +
           '] (' + input.shape[i] + ')');
    });
  };
  for (var i = 0; i < inputRank; ++i) {
    _loop_1(i);
  }
}
/** Converts a binary mask to an array of axes. Used in stridedSlice(). */
function maskToAxes(mask) {
  var axes = [];
  var axis = 0;
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
function computeOutShape(begin, end, strides) {
  var size = [];
  for (var axis = 0; axis < begin.length; axis++) {
    size[axis] = Math.ceil((end[axis] - begin[axis]) / strides[axis]);
  }
  return size;
}
// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stride value. Otherwise, insert.
function stridesWithElidedDims(
    strides, ellipsisInsertionIndex, numElidedAxes, inputShape) {
  var newStrides = strides.slice();
  for (var i = newStrides.length; i < inputShape.length; i++) {
    newStrides.push(1);
  }
  for (var i = 0; i < numElidedAxes; i++) {
    if (i === 0) {
      newStrides[ellipsisInsertionIndex] = 1;
    } else {
      newStrides.splice(
          ellipsisInsertionIndex, 0 /* num elements to delete */,
          1 /* element to add */);
      newStrides.pop();
    }
  }
  return newStrides;
}
function unnormalizeAxis(
    ellipsisInsertionIndex, numElidedAxes, normalizedAxis) {
  if (normalizedAxis <= ellipsisInsertionIndex) {
    return normalizedAxis;
  }
  return normalizedAxis - (numElidedAxes - 1);
}
function getElidedAxes(numElidedAxes, ellipsisInsertionIndex) {
  var elidedAxes = [];
  for (var i = 0; i < numElidedAxes; i++) {
    elidedAxes.push(ellipsisInsertionIndex + i);
  }
  return elidedAxes;
}
// Normalize the start, end and strides.
function getNormalizedAxes(
    inputShape, ellipsisAxes, numInterpolatedAxes, begin, end, strides,
    beginMask, endMask, ellipsisMask) {
  var inputRank = inputShape.length;
  var normalizedBegin = new Array(inputRank),
      normalizedEnd = new Array(inputRank),
      normalizedStrides = new Array(inputRank);
  if (ellipsisAxes.length && numInterpolatedAxes > 0) {
    var fullIndex = ellipsisAxes[0];
    // The ellipsis applies to the masked index as well as any dimensions
    // that are interpolated.
    var numElidedAxes = numInterpolatedAxes + 1;
    normalizedBegin = startIndicesWithElidedDims(
        beginMask, fullIndex, numElidedAxes, begin, inputShape);
    normalizedEnd = stopIndicesWithElidedDims(
        endMask, fullIndex, numElidedAxes, end, inputShape);
    normalizedStrides =
        stridesWithElidedDims(strides, fullIndex, numElidedAxes, inputShape);
  } else {
    for (var axis = 0; axis < inputRank; axis++) {
      normalizedBegin[axis] = startForAxis(
          beginMask, begin, strides, inputShape, axis, ellipsisMask);
      normalizedEnd[axis] =
          stopForAxis(endMask, end, strides, inputShape, axis, ellipsisMask);
      normalizedStrides[axis] = stridesForAxis(strides, axis, ellipsisMask);
    }
  }
  return {
    begin: normalizedBegin,
    end: normalizedEnd,
    strides: normalizedStrides
  };
}
// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current start value. Otherwise, insert.
function startIndicesWithElidedDims(
    beginMask, ellipsisInsertionIndex, numElidedAxes, originalBegin,
    inputShape) {
  var newIndices = inputShape.slice();
  var elidedAxes = getElidedAxes(numElidedAxes, ellipsisInsertionIndex);
  for (var axis = 0; axis < newIndices.length; axis++) {
    if (elidedAxes.indexOf(axis) > -1) {
      newIndices[axis] = 0;
    } else {
      var originalAxis =
          unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
      var originalValue = originalBegin[originalAxis];
      if (beginMask & 1 << originalAxis) {
        originalValue = 0;
      }
      newIndices[axis] = originalValue;
    }
  }
  return newIndices;
}
// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stop value. Otherwise, insert.
function stopIndicesWithElidedDims(
    endMask, ellipsisInsertionIndex, numElidedAxes, originalEnd, inputShape) {
  var newIndices = inputShape.slice();
  var elidedAxes = getElidedAxes(numElidedAxes, ellipsisInsertionIndex);
  for (var axis = 0; axis < newIndices.length; axis++) {
    if (elidedAxes.indexOf(axis) > -1) {
      newIndices[axis] = Number.MAX_SAFE_INTEGER;
    } else {
      var originalAxis =
          unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
      var originalValue = originalEnd[originalAxis];
      if (endMask & 1 << originalAxis) {
        originalValue = Number.MAX_SAFE_INTEGER;
      }
      newIndices[axis] = originalValue;
    }
  }
  for (var i = 0; i < newIndices.length; i++) {
    // Handle negative indices
    var axisSize = inputShape[i];
    if (newIndices[i] < 0) {
      newIndices[i] += axisSize;
    }
    newIndices[i] = clamp(0, newIndices[i], inputShape[i]);
  }
  return newIndices;
}
function stridesForAxis(strides, axis, ellipsisMask) {
  var stride = strides[axis];
  if (ellipsisMask & (1 << axis) || stride == null) {
    stride = 1;
  }
  return stride;
}
function startForAxis(
    beginMask, startIndices, strides, inputShape, axis, ellipsisMask) {
  // Begin with the specified index
  var start = startIndices[axis];
  var stride = strides[axis] || 1;
  // Check the axis bit from right of masked axes, or the begin index is not set
  // for the axis.
  if (beginMask & 1 << axis || ellipsisMask & 1 << axis || start == null) {
    if (stride > 0) {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axis_size-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = Number.MIN_SAFE_INTEGER;
    } else {
      // Backward iteration - use the last element.
      start = Number.MAX_SAFE_INTEGER;
    }
  }
  // Handle negative indices
  var axisSize = inputShape[axis];
  if (start < 0) {
    start += axisSize;
  }
  // Clamping
  start = clamp(0, start, axisSize - 1);
  return start;
}
function stopForAxis(
    endMask, stopIndices, strides, inputShape, axis, ellipsisMask) {
  // Begin with the specified index
  var stop = stopIndices[axis];
  var stride = strides[axis] || 1;
  // Check the axis bit from right of masked axes, or if the stop index is not
  // set for this axis.
  if (endMask & (1 << axis) || ellipsisMask & (1 << axis) || stop == null) {
    if (stride > 0) {
      // Forward iteration - use the last element. These values will get
      // clamped below
      stop = Number.MAX_SAFE_INTEGER;
    } else {
      // Backward iteration - use the first element.
      stop = Number.MIN_SAFE_INTEGER;
    }
  }
  // Handle negative indices
  var axisSize = inputShape[axis];
  if (stop < 0) {
    stop += axisSize;
  }
  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (stride > 0) {
    // Forward iteration
    stop = clamp(0, stop, axisSize);
  } else {
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
  var firstNonOneAxis = size.length;
  for (var i = 0; i < size.length; i++) {
    if (size[i] > 1) {
      firstNonOneAxis = i;
      break;
    }
  }
  for (var i = firstNonOneAxis + 1; i < size.length; i++) {
    if (begin[i] > 0 || size[i] !== shape[i]) {
      return false;
    }
  }
  return true;
}
function computeFlatOffset(begin, strides) {
  var flatOffset = begin.length > 0 ? begin[begin.length - 1] : 1;
  for (var i = 0; i < begin.length - 1; i++) {
    flatOffset += begin[i] * strides[i];
  }
  return flatOffset;
}
function parseSliceParams(x, begin, size) {
  // The following logic allows for more ergonomic calls.
  var begin_;
  var xRank = x.shape.length;
  if (typeof begin === 'number') {
    begin_ = [begin].concat(new Array(xRank - 1).fill(0));
  } else if (begin.length < xRank) {
    begin_ = begin.concat(new Array(xRank - begin.length).fill(0));
  } else {
    begin_ = begin.slice();
  }
  begin_.forEach(function(d) {
    assert(d !== -1, function() {
      return 'slice() does not support negative begin indexing.';
    });
  });
  var size_;
  if (size == null) {
    size_ = new Array(xRank).fill(-1);
  } else if (typeof size === 'number') {
    size_ = [size].concat(new Array(xRank - 1).fill(-1));
  } else if (size.length < xRank) {
    size_ = size.concat(new Array(xRank - size.length).fill(-1));
  } else {
    size_ = size;
  }
  size_ = size_.map(function(d, i) {
    if (d >= 0) {
      return d;
    } else {
      assert(d === -1, function() {
        return 'Negative size values should be exactly -1 but got ' +
            (d + ' for the slice() size at index ' + i + '.');
      });
      return x.shape[i] - begin_[i];
    }
  });
  return [begin_, size_];
}
function sliceInfo(
    xShape, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask,
    shrinkAxisMask) {
  // make a copy because it may be modified further down.
  var $begin = begin.slice();
  var $end = end.slice();
  var $strides = strides;
  if (strides == null) {
    $strides = new Array($begin.length);
  }
  var ellipsisAxes = maskToAxes(ellipsisMask);
  if (ellipsisAxes.length > 1) {
    throw new Error('Multiple ellipses in slice is not allowed.');
  }
  if (ellipsisMask !== 0 && newAxisMask !== 0) {
    throw new Error(
        'Using both ellipsisMask and newAxisMask is not yet supported.');
  }
  if (ellipsisMask !== 0 && shrinkAxisMask !== 0) {
    throw new Error(
        'Using both ellipsisMask and shrinkAxisMask is not yet supported.');
  }
  var numInterpolatedAxes = xShape.length - $begin.length;
  // Expand the dims of x based on the newAxisMask.
  var expandAxes = maskToAxes(newAxisMask);
  var newShape = xShape.slice();
  expandAxes.forEach(function(axis) {
    $begin[axis] = 0;
    $end[axis] = 1;
    newShape.splice(axis, 0, 1);
  });
  var _a = getNormalizedAxes(
          newShape, ellipsisAxes, numInterpolatedAxes, $begin, $end, $strides,
          beginMask, endMask, ellipsisMask),
      normalizedBegin = _a.begin, normalizedEnd = _a.end,
      normalizedStrides = _a.strides;
  $begin = normalizedBegin;
  $end = normalizedEnd;
  $strides = normalizedStrides;
  var shrinkAxes = maskToAxes(shrinkAxisMask);
  // Adjust the ends based on the shrink mask.
  shrinkAxes.forEach(function(axis) {
    $end[axis] = $begin[axis] + 1;
    $strides[axis] = 1;
  });
  // Figure out the output shape.
  var size = computeOutShape($begin, $end, $strides);
  // Remove the axes based on shrinkMask.
  var outShape = size.filter(function(_, axis) {
    return shrinkAxes.indexOf(axis) === -1;
  });
  var nonStrided = $strides.every(function(v) {
    return v === 1;
  });
  return {
    nonStrided: nonStrided,
    $begin: $begin,
    $end: $end,
    $strides: $strides,
    size: size,
    newShape: newShape,
    outShape: outShape
  };
}

var slice_util = {
  __proto__: null,
  assertParamsValid: assertParamsValid,
  maskToAxes: maskToAxes,
  computeOutShape: computeOutShape,
  stridesWithElidedDims: stridesWithElidedDims,
  getNormalizedAxes: getNormalizedAxes,
  startIndicesWithElidedDims: startIndicesWithElidedDims,
  stopIndicesWithElidedDims: stopIndicesWithElidedDims,
  stridesForAxis: stridesForAxis,
  startForAxis: startForAxis,
  stopForAxis: stopForAxis,
  isSliceContinous: isSliceContinous,
  computeFlatOffset: computeFlatOffset,
  parseSliceParams: parseSliceParams,
  sliceInfo: sliceInfo
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
/**
 * Serializable defines the serialization contract.
 *
 * TFJS requires serializable classes to return their className when asked
 * to avoid issues with minification.
 */
var Serializable = /** @class */ (function() {
  function Serializable() {}
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
  Serializable.prototype.getClassName = function() {
    return this.constructor.className;
  };
  /**
   * Creates an instance of T from a ConfigDict.
   *
   * This works for most descendants of serializable.  A few need to
   * provide special handling.
   * @param cls A Constructor for the class to instantiate.
   * @param config The Configuration for the object.
   */
  /** @nocollapse */
  Serializable.fromConfig = function(cls, config) {
    return new cls(config);
  };
  return Serializable;
}());
/**
 * Maps string keys to class constructors.
 *
 * Used during (de)serialization from the cross-language JSON format, which
 * requires the class name in the serialization format matches the class
 * names as used in Python, should it exist.
 */
var SerializationMap = /** @class */ (function() {
  function SerializationMap() {
    this.classNameMap = {};
  }
  /**
   * Returns the singleton instance of the map.
   */
  SerializationMap.getMap = function() {
    if (SerializationMap.instance == null) {
      SerializationMap.instance = new SerializationMap();
    }
    return SerializationMap.instance;
  };
  /**
   * Registers the class as serializable.
   */
  SerializationMap.register = function(cls) {
    SerializationMap.getMap().classNameMap[cls.className] =
        [cls, cls.fromConfig];
  };
  return SerializationMap;
}());
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
 *
 * @doc {heading: 'Models', subheading: 'Serialization', ignoreCI: true}
 */
function registerClass(cls) {
  assert(cls.className != null, function() {
    return 'Class being registered does not have the static className ' +
        'property defined.';
  });
  assert(typeof cls.className === 'string', function() {
    return 'className is required to be a string, but got type ' +
        typeof cls.className;
  });
  assert(cls.className.length > 0, function() {
    return 'Class being registered has an empty-string as its className, ' +
        'which is disallowed.';
  });
  SerializationMap.register(cls);
}

var serialization = {
  __proto__: null,
  Serializable: Serializable,
  SerializationMap: SerializationMap,
  registerClass: registerClass
};

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
var TEST_EPSILON_FLOAT32 = 1e-3;
var TEST_EPSILON_FLOAT16 = 1e-1;
function expectArraysClose(actual, expected, epsilon) {
  if (epsilon == null) {
    epsilon = testEpsilon();
  }
  return expectArraysPredicate(actual, expected, function(a, b) {
    return areClose(a, b, epsilon);
  });
}
function testEpsilon() {
  return ENGINE.backend.floatPrecision() === 32 ? TEST_EPSILON_FLOAT32 :
                                                  TEST_EPSILON_FLOAT16;
}
function expectArraysPredicate(actual, expected, predicate) {
  var checkClassType = true;
  if (isTypedArray(actual) || isTypedArray(expected)) {
    checkClassType = false;
  }
  if (isTypedArray(actual) && isTypedArray(expected)) {
    checkClassType = true;
  }
  if (checkClassType) {
    var aType = actual.constructor.name;
    var bType = expected.constructor.name;
    if (aType !== bType) {
      throw new Error(
          'Arrays are of different type. Actual: ' + aType + '. ' +
          ('Expected: ' + bType));
    }
  }
  if (Array.isArray(actual) && Array.isArray(expected)) {
    var actualShape = inferShape(actual);
    var expectedShape = inferShape(expected);
    if (!arraysEqual(actualShape, expectedShape)) {
      throw new Error(
          'Arrays have different shapes. ' +
          ('Actual: [' + actualShape + ']. Expected: [' + expectedShape + ']'));
    }
  }
  var actualFlat = isTypedArray(actual) ? actual : flatten(actual);
  var expectedFlat = isTypedArray(expected) ? expected : flatten(expected);
  if (actualFlat.length !== expectedFlat.length) {
    throw new Error(
        'Arrays have different lengths actual: ' + actualFlat.length + ' vs ' +
        ('expected: ' + expectedFlat.length + '.\n') +
        ('Actual:   ' + actualFlat + '.\n') +
        ('Expected: ' + expectedFlat + '.'));
  }
  for (var i = 0; i < expectedFlat.length; ++i) {
    var a = actualFlat[i];
    var e = expectedFlat[i];
    if (!predicate(a, e)) {
      throw new Error(
          'Arrays differ: actual[' + i + '] = ' + a + ', expected[' + i +
          '] = ' + e + '.\n' + ('Actual:   ' + actualFlat + '.\n') +
          ('Expected: ' + expectedFlat + '.'));
    }
  }
}
function expectPromiseToFail(fn, done) {
  fn().then(
      function() {
        return done.fail();
      },
      function() {
        return done();
      });
}
function expectArraysEqual(actual, expected) {
  var exp = typeof expected === 'string' || typeof expected === 'number' ||
          typeof expected === 'boolean' ?
      [expected] :
      expected;
  if (isString(actual) || isString(actual[0]) || isString(expected) ||
      isString(expected[0])) {
    // tslint:disable-next-line: triple-equals
    return expectArraysPredicate(actual, exp, function(a, b) {
      return a == b;
    });
  }
  return expectArraysPredicate(actual, expected, function(a, b) {
    return areClose(a, b, 0);
  });
}
function expectNumbersClose(a, e, epsilon) {
  if (epsilon == null) {
    epsilon = testEpsilon();
  }
  if (!areClose(a, e, epsilon)) {
    throw new Error('Numbers differ: actual === ' + a + ', expected === ' + e);
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
  for (var i = 0; i < actual.length; i++) {
    if (actual[i] < low || actual[i] > high) {
      throw new Error(
          'Value out of range:' + actual[i] + ' low: ' + low +
          ', high: ' + high);
    }
  }
}
function expectArrayBuffersEqual(actual, expected) {
  // Safari & Jasmine don't like comparing ArrayBuffers directly. Wrapping in
  // a Float32Array solves this issue.
  expect(new Float32Array(actual)).toEqual(new Float32Array(expected));
}
/** Encodes strings into utf-8 bytes. */
function encodeStrings(a) {
  for (var i = 0; i < a.length; i++) {
    var val = a[i];
    if (Array.isArray(val)) {
      encodeStrings(val);
    } else {
      a[i] = encodeString(val);
    }
  }
  return a;
}

var test_util = {
  __proto__: null,
  TEST_EPSILON_FLOAT16: TEST_EPSILON_FLOAT16,
  expectArraysClose: expectArraysClose,
  testEpsilon: testEpsilon,
  expectPromiseToFail: expectPromiseToFail,
  expectArraysEqual: expectArraysEqual,
  expectNumbersClose: expectNumbersClose,
  expectValuesInRange: expectValuesInRange,
  expectArrayBuffersEqual: expectArrayBuffersEqual,
  encodeStrings: encodeStrings
};

/** @license See the LICENSE file. */
// This code is auto-generated, do not modify this file!
var version = '0.0.0';

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
 * Enables production mode which disables correctness checks in favor of
 * performance.
 *
 * @doc {heading: 'Environment'}
 */
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
 *
 * @doc {heading: 'Environment'}
 */
function enableDebugMode() {
  env().set('DEBUG', true);
}
/** Globally disables deprecation warnings */
function disableDeprecationWarnings() {
  env().set('DEPRECATION_WARNINGS_ENABLED', false);
  console.warn('TensorFlow.js deprecation warnings have been disabled.');
}
/** Warn users about deprecated functionality. */
function deprecationWarn(msg) {
  if (env().getBool('DEPRECATION_WARNINGS_ENABLED')) {
    console.warn(
        msg + ' You can disable deprecation warnings with ' +
        'tf.disableDeprecationWarnings().');
  }
}
/**
 * Dispose all variables kept in backend engine.
 *
 * @doc {heading: 'Environment'}
 */
function disposeVariables() {
  ENGINE.disposeVariables();
}
/**
 * It returns the global engine that keeps track of all tensors and backends.
 *
 * @doc {heading: 'Environment'}
 */
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
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
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
 * - `kernelNames`: an array of unique strings with just the names of the
 * kernels in the `kernels` array.
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
 *
 * @doc {heading: 'Performance', subheading: 'Profile'}
 */
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
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
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
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
function dispose(container) {
  var tensors = getTensorsInContainer(container);
  tensors.forEach(function(tensor) {
    return tensor.dispose();
  });
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
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
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
 *
 * @doc {heading: 'Performance', subheading: 'Timing'}
 */
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
 *
 * @doc {heading: 'Backends'}
 */
function setBackend(backendName) {
  return ENGINE.setBackend(backendName);
}
/**
 * Returns a promise that resolves when the currently selected backend (or the
 * highest priority one) has initialized. Await this promise when you are using
 * a backend that has async initialization.
 *
 * @doc {heading: 'Backends'}
 */
function ready() {
  return ENGINE.ready();
}
/**
 * Returns the current backend name (cpu, webgl, etc). The backend is
 * responsible for creating tensors and executing operations on those tensors.
 *
 * @doc {heading: 'Backends'}
 */
function getBackend() {
  return ENGINE.backendName;
}
/**
 * Removes a backend and the registered factory.
 *
 * @doc {heading: 'Backends'}
 */
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
 *
 * @doc {heading: 'Backends'}
 */
function registerBackend(name, factory, priority) {
  if (priority === void 0) {
    priority = 1;
  }
  return ENGINE.registerBackend(name, factory, priority);
}
/**
 * Gets the current backend. If no backends have been initialized, this will
 * attempt to initialize the best backend. Will throw an error if the highest
 * priority backend has async initialization, in which case, you should call
 * 'await tf.ready()' before running other code.
 *
 * @doc {heading: 'Backends'}
 */
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
/**
 * Adds two `tf.Tensor`s element-wise, A + B. Supports broadcasting.
 *
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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function add_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'add');
  var $b = convertToTensor(b, 'b', 'add');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Add, inputs);
}
var add$1 = op({add_: add_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function floorDiv_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'floorDiv');
  var $b = convertToTensor(b, 'b', 'floorDiv');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(FloorDiv, inputs);
}
var floorDiv = op({floorDiv_: floorDiv_});

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
/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function div_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'div');
  var $b = convertToTensor(b, 'b', 'div');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  if ($a.dtype === 'int32' && $b.dtype === 'int32') {
    return floorDiv($a, $b);
  }
  var inputs = {a: $a, b: $b};
  var attrs = {};
  // tslint:disable-next-line: no-unnecessary-type-assertion
  return ENGINE.runKernel(RealDiv, inputs, attrs);
}
var div = op({div_: div_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function mul_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'mul');
  var $b = convertToTensor(b, 'b', 'mul');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Multiply, inputs);
}
var mul = op({mul_: mul_});

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
 * Computes absolute value element-wise: `abs(x)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.abs().print();  // or tf.abs(x)
 * ```
 * @param x The input `tf.Tensor`.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function abs_(x) {
  var $x = convertToTensor(x, 'x', 'abs');
  if ($x.dtype === 'complex64') {
    var inputs = {x: $x};
    return ENGINE.runKernel(ComplexAbs, inputs);
  } else {
    var inputs = {x: $x};
    return ENGINE.runKernel(Abs, inputs);
  }
}
var abs = op({abs_: abs_});

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
 * Computes acos of the input `tf.Tensor` element-wise: `acos(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.acos().print();  // or tf.acos(x)
 * ```
 * @param x The input tensor.
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function acos_(x) {
  var $x = convertToTensor(x, 'x', 'acos');
  var inputs = {x: $x};
  return ENGINE.runKernel(Acos, inputs);
}
var acos = op({acos_: acos_});

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
 * Computes the inverse hyperbolic cos of the input `tf.Tensor` element-wise:
 * `acosh(x)`
 *
 * ```js
 * const x = tf.tensor1d([10, 1, 3, 5.7]);
 *
 * x.acosh().print();  // or tf.acosh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function acosh_(x) {
  var $x = convertToTensor(x, 'x', 'acosh');
  var inputs = {x: $x};
  return ENGINE.runKernel(Acosh, inputs);
}
var acosh = op({acosh_: acosh_});

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
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function addN_(tensors) {
  assert(Array.isArray(tensors), function() {
    return 'The argument passed to tf.addN() must be a list of tensors';
  });
  assert(tensors.length >= 1, function() {
    return 'Must pass at least one tensor to tf.addN(), but got ' +
        ('' + tensors.length);
  });
  var $tensors = tensors.map(function(t, i) {
    return convertToTensor(t, 'tensors' + i, 'addN');
  });
  var firstTensor = $tensors[0];
  $tensors.forEach(function(t) {
    if (t.dtype !== firstTensor.dtype) {
      throw new Error(
          'All tensors passed to tf.addN() must have the same dtype');
    }
  });
  $tensors.forEach(function(t) {
    if (!arraysEqual(t.shape, firstTensor.shape)) {
      throw new Error(
          'All tensors passed to tf.addN() must have the same shape');
    }
  });
  var inputs = $tensors;
  return ENGINE.runKernel(AddN, inputs);
}
var addN = op({addN_: addN_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function all_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  var $x = convertToTensor(x, 'x', 'all', 'bool');
  var inputs = {x: $x};
  var attrs = {axis: axis, keepDims: keepDims};
  return ENGINE.runKernel(All, inputs, attrs);
}
var all = op({all_: all_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function any_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  var $x = convertToTensor(x, 'x', 'any', 'bool');
  var inputs = {x: $x};
  var attrs = {axis: axis, keepDims: keepDims};
  return ENGINE.runKernel(Any, inputs, attrs);
}
// tslint:disable-next-line:variable-name
var any = op({any_: any_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function argMax_(x, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  var $x = convertToTensor(x, 'x', 'argMax');
  var inputs = {x: $x};
  var attrs = {axis: axis};
  return ENGINE.runKernel(ArgMax, inputs, attrs);
}
var argMax = op({argMax_: argMax_});

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
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function argMin_(x, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  var $x = convertToTensor(x, 'x', 'argMin');
  var inputs = {x: $x};
  var attrs = {axis: axis};
  return ENGINE.runKernel(ArgMin, inputs, attrs);
}
var argMin = op({argMin_: argMin_});

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
 * Computes asin of the input `tf.Tensor` element-wise: `asin(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.asin().print();  // or tf.asin(x)
 * ```
 * @param x The input tensor.
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function asin_(x) {
  var $x = convertToTensor(x, 'x', 'asin');
  var inputs = {x: $x};
  return ENGINE.runKernel(Asin, inputs);
}
var asin = op({asin_: asin_});

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
 * Computes inverse hyperbolic sin of the input `tf.Tensor` element-wise:
 * `asinh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.asinh().print();  // or tf.asinh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function asinh_(x) {
  var $x = convertToTensor(x, 'x', 'asinh');
  var inputs = {x: $x};
  return ENGINE.runKernel(Asinh, inputs);
}
var asinh = op({asinh_: asinh_});

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
 * Computes atan of the input `tf.Tensor` element-wise: `atan(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.atan().print();  // or tf.atan(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function atan_(x) {
  var $x = convertToTensor(x, 'x', 'atan');
  var inputs = {x: $x};
  return ENGINE.runKernel(Atan, inputs);
}
var atan = op({atan_: atan_});

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
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function atan2_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'atan2');
  var $b = convertToTensor(b, 'b', 'atan2');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Atan2, inputs);
}
var atan2 = op({atan2_: atan2_});

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
 * Computes inverse hyperbolic tan of the input `tf.Tensor` element-wise:
 * `atanh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, .1, -.1, .7]);
 *
 * x.atanh().print();  // or tf.atanh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function atanh_(x) {
  var $x = convertToTensor(x, 'x', 'atanh');
  var inputs = {x: $x};
  return ENGINE.runKernel(Atanh, inputs);
}
var atanh = op({atanh_: atanh_});

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
/**
 *
 * @param inputShape Input tensor shape is of the following dimensions:
 *     `[batch, height, width, inChannels]`.
 * @param filterShape The filter shape is of the following dimensions:
 *     `[filterHeight, filterWidth, depth]`.
 * @param strides The strides of the sliding window for each dimension of the
 *     input tensor: `[strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat The data format of the input and output data.
 *     Defaults to 'NHWC'.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`.
 *     Defaults to `[1, 1]`. If `dilations` is a single number, then
 *     `dilationHeight == dilationWidth`.
 */
function computeDilation2DInfo(
    inputShape, filterShape, strides, pad, dataFormat, dilations) {
  if (dataFormat === void 0) {
    dataFormat = 'NHWC';
  }
  // `computerConv2DInfo` require filterShape to be in the dimension of:
  // `[filterHeight, filterWidth, depth, outDepth]`, dilation2d doesn't have
  // outDepth, it should have the same depth as the input.
  // Input shape: [batch, height, width, inChannels]
  var inputChannels = inputShape[3];
  var $filterShape = filterShape.concat([inputChannels]);
  var $dataFormat = convertConv2DDataFormat(dataFormat);
  return computeConv2DInfo(
      inputShape, $filterShape, strides, dilations, pad,
      null /* roundingMode */, null /* depthWise */, $dataFormat);
}
function computePool2DInfo(
    inShape, filterSize, strides, dilations, pad, roundingMode, dataFormat) {
  if (dataFormat === void 0) {
    dataFormat = 'channelsLast';
  }
  var _a = parseTupleParam(filterSize), filterHeight = _a[0],
      filterWidth = _a[1];
  var filterShape;
  if (dataFormat === 'channelsLast') {
    filterShape = [filterHeight, filterWidth, inShape[3], inShape[3]];
  } else if (dataFormat === 'channelsFirst') {
    filterShape = [filterHeight, filterWidth, inShape[1], inShape[1]];
  } else {
    throw new Error('Unknown dataFormat ' + dataFormat);
  }
  return computeConv2DInfo(
      inShape, filterShape, strides, dilations, pad, roundingMode, false,
      dataFormat);
}
/**
 * Computes the information for a forward pass of a pooling3D operation.
 */
function computePool3DInfo(
    inShape, filterSize, strides, dilations, pad, roundingMode, dataFormat) {
  if (dataFormat === void 0) {
    dataFormat = 'NDHWC';
  }
  var _a = parse3TupleParam(filterSize), filterDepth = _a[0],
      filterHeight = _a[1], filterWidth = _a[2];
  var filterShape;
  var $dataFormat;
  if (dataFormat === 'NDHWC') {
    $dataFormat = 'channelsLast';
    filterShape =
        [filterDepth, filterHeight, filterWidth, inShape[4], inShape[4]];
  } else if (dataFormat === 'NCDHW') {
    $dataFormat = 'channelsFirst';
    filterShape =
        [filterDepth, filterHeight, filterWidth, inShape[1], inShape[1]];
  } else {
    throw new Error('Unknown dataFormat ' + dataFormat);
  }
  return computeConv3DInfo(
      inShape, filterShape, strides, dilations, pad, false, $dataFormat,
      roundingMode);
}
/**
 * Computes the information for a forward pass of a convolution/pooling
 * operation.
 */
function computeConv2DInfo(
    inShape, filterShape, strides, dilations, pad, roundingMode, depthwise,
    dataFormat) {
  if (depthwise === void 0) {
    depthwise = false;
  }
  if (dataFormat === void 0) {
    dataFormat = 'channelsLast';
  }
  var _a = [-1, -1, -1, -1], batchSize = _a[0], inHeight = _a[1],
      inWidth = _a[2], inChannels = _a[3];
  if (dataFormat === 'channelsLast') {
    batchSize = inShape[0], inHeight = inShape[1], inWidth = inShape[2],
    inChannels = inShape[3];
  } else if (dataFormat === 'channelsFirst') {
    batchSize = inShape[0], inChannels = inShape[1], inHeight = inShape[2],
    inWidth = inShape[3];
  } else {
    throw new Error('Unknown dataFormat ' + dataFormat);
  }
  var filterHeight = filterShape[0], filterWidth = filterShape[1],
      filterChannels = filterShape[3];
  var _b = parseTupleParam(strides), strideHeight = _b[0], strideWidth = _b[1];
  var _c = parseTupleParam(dilations), dilationHeight = _c[0],
      dilationWidth = _c[1];
  var effectiveFilterHeight =
      getEffectiveFilterSize(filterHeight, dilationHeight);
  var effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
  var _d = getPadAndOutInfo(
          pad, inHeight, inWidth, strideHeight, strideWidth,
          effectiveFilterHeight, effectiveFilterWidth, roundingMode,
          dataFormat),
      padInfo = _d.padInfo, outHeight = _d.outHeight, outWidth = _d.outWidth;
  var outChannels = depthwise ? filterChannels * inChannels : filterChannels;
  var outShape;
  if (dataFormat === 'channelsFirst') {
    outShape = [batchSize, outChannels, outHeight, outWidth];
  } else if (dataFormat === 'channelsLast') {
    outShape = [batchSize, outHeight, outWidth, outChannels];
  }
  return {
    batchSize: batchSize,
    dataFormat: dataFormat,
    inHeight: inHeight,
    inWidth: inWidth,
    inChannels: inChannels,
    outHeight: outHeight,
    outWidth: outWidth,
    outChannels: outChannels,
    padInfo: padInfo,
    strideHeight: strideHeight,
    strideWidth: strideWidth,
    filterHeight: filterHeight,
    filterWidth: filterWidth,
    effectiveFilterHeight: effectiveFilterHeight,
    effectiveFilterWidth: effectiveFilterWidth,
    dilationHeight: dilationHeight,
    dilationWidth: dilationWidth,
    inShape: inShape,
    outShape: outShape,
    filterShape: filterShape
  };
}
/**
 * Computes the information for a forward pass of a 3D convolution/pooling
 * operation.
 */
function computeConv3DInfo(
    inShape, filterShape, strides, dilations, pad, depthwise, dataFormat,
    roundingMode) {
  if (depthwise === void 0) {
    depthwise = false;
  }
  if (dataFormat === void 0) {
    dataFormat = 'channelsLast';
  }
  var _a = [-1, -1, -1, -1, -1], batchSize = _a[0], inDepth = _a[1],
      inHeight = _a[2], inWidth = _a[3], inChannels = _a[4];
  if (dataFormat === 'channelsLast') {
    batchSize = inShape[0], inDepth = inShape[1], inHeight = inShape[2],
    inWidth = inShape[3], inChannels = inShape[4];
  } else if (dataFormat === 'channelsFirst') {
    batchSize = inShape[0], inChannels = inShape[1], inDepth = inShape[2],
    inHeight = inShape[3], inWidth = inShape[4];
  } else {
    throw new Error('Unknown dataFormat ' + dataFormat);
  }
  var filterDepth = filterShape[0], filterHeight = filterShape[1],
      filterWidth = filterShape[2], filterChannels = filterShape[4];
  var _b = parse3TupleParam(strides), strideDepth = _b[0], strideHeight = _b[1],
      strideWidth = _b[2];
  var _c = parse3TupleParam(dilations), dilationDepth = _c[0],
      dilationHeight = _c[1], dilationWidth = _c[2];
  var effectiveFilterDepth = getEffectiveFilterSize(filterDepth, dilationDepth);
  var effectiveFilterHeight =
      getEffectiveFilterSize(filterHeight, dilationHeight);
  var effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
  var _d = get3DPadAndOutInfo(
          pad, inDepth, inHeight, inWidth, strideDepth, strideHeight,
          strideWidth, effectiveFilterDepth, effectiveFilterHeight,
          effectiveFilterWidth, roundingMode),
      padInfo = _d.padInfo, outDepth = _d.outDepth, outHeight = _d.outHeight,
      outWidth = _d.outWidth;
  var outChannels = depthwise ? filterChannels * inChannels : filterChannels;
  var outShape;
  if (dataFormat === 'channelsFirst') {
    outShape = [batchSize, outChannels, outDepth, outHeight, outWidth];
  } else if (dataFormat === 'channelsLast') {
    outShape = [batchSize, outDepth, outHeight, outWidth, outChannels];
  }
  return {
    batchSize: batchSize,
    dataFormat: dataFormat,
    inDepth: inDepth,
    inHeight: inHeight,
    inWidth: inWidth,
    inChannels: inChannels,
    outDepth: outDepth,
    outHeight: outHeight,
    outWidth: outWidth,
    outChannels: outChannels,
    padInfo: padInfo,
    strideDepth: strideDepth,
    strideHeight: strideHeight,
    strideWidth: strideWidth,
    filterDepth: filterDepth,
    filterHeight: filterHeight,
    filterWidth: filterWidth,
    effectiveFilterDepth: effectiveFilterDepth,
    effectiveFilterHeight: effectiveFilterHeight,
    effectiveFilterWidth: effectiveFilterWidth,
    dilationDepth: dilationDepth,
    dilationHeight: dilationHeight,
    dilationWidth: dilationWidth,
    inShape: inShape,
    outShape: outShape,
    filterShape: filterShape
  };
}
function computeOutputShape2D(
    inShape, fieldSize, stride, zeroPad, roundingMode) {
  if (zeroPad == null) {
    zeroPad = computeDefaultPad(inShape, fieldSize, stride);
  }
  var inputRows = inShape[0];
  var inputCols = inShape[1];
  var outputRows =
      round((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  var outputCols =
      round((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  return [outputRows, outputCols];
}
function computeOutputShape4D(
    inShape, fieldSize, outChannels, stride, zeroPad, roundingMode) {
  if (zeroPad == null) {
    zeroPad = computeDefaultPad(inShape, fieldSize, stride);
  }
  var inputDepth = inShape[0];
  var inputRows = inShape[1];
  var inputCols = inShape[2];
  var outputDepths =
      round((inputDepth - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  var outputRows =
      round((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  var outputCols =
      round((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  return [outputDepths, outputRows, outputCols, outChannels];
}
function computeDefaultPad(inputShape, fieldSize, stride, dilation) {
  if (dilation === void 0) {
    dilation = 1;
  }
  var effectiveFieldSize = getEffectiveFilterSize(fieldSize, dilation);
  return Math.floor(
      (inputShape[0] * (stride - 1) - stride + effectiveFieldSize) / 2);
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
function getPadAndOutInfo(
    pad, inHeight, inWidth, strideHeight, strideWidth, filterHeight,
    filterWidth, roundingMode, dataFormat) {
  var padInfo;
  var outHeight;
  var outWidth;
  if (typeof pad === 'number') {
    var padType = (pad === 0) ? 'VALID' : 'NUMBER';
    padInfo = {top: pad, bottom: pad, left: pad, right: pad, type: padType};
    var outShape = computeOutputShape2D(
        [inHeight, inWidth], filterHeight, strideHeight, pad, roundingMode);
    outHeight = outShape[0];
    outWidth = outShape[1];
  } else if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideHeight);
    outWidth = Math.ceil(inWidth / strideWidth);
    var padAlongHeight =
        Math.max(0, (outHeight - 1) * strideHeight + filterHeight - inHeight);
    var padAlongWidth =
        Math.max(0, (outWidth - 1) * strideWidth + filterWidth - inWidth);
    var top_1 = Math.floor(padAlongHeight / 2);
    var bottom = padAlongHeight - top_1;
    var left = Math.floor(padAlongWidth / 2);
    var right = padAlongWidth - left;
    padInfo =
        {top: top_1, bottom: bottom, left: left, right: right, type: 'SAME'};
  } else if (pad === 'valid') {
    padInfo = {top: 0, bottom: 0, left: 0, right: 0, type: 'VALID'};
    outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
    outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
  } else if (typeof pad === 'object') {
    var top_2 = dataFormat === 'channelsLast' ? pad[1][0] : pad[2][0];
    var bottom = dataFormat === 'channelsLast' ? pad[1][1] : pad[2][1];
    var left = dataFormat === 'channelsLast' ? pad[2][0] : pad[3][0];
    var right = dataFormat === 'channelsLast' ? pad[2][1] : pad[3][1];
    var padType = (top_2 === 0 && bottom === 0 && left === 0 && right === 0) ?
        'VALID' :
        'EXPLICIT';
    padInfo =
        {top: top_2, bottom: bottom, left: left, right: right, type: padType};
    outHeight = round(
        (inHeight - filterHeight + top_2 + bottom) / strideHeight + 1,
        roundingMode);
    outWidth = round(
        (inWidth - filterWidth + left + right) / strideWidth + 1, roundingMode);
  } else {
    throw Error('Unknown padding parameter: ' + pad);
  }
  return {padInfo: padInfo, outHeight: outHeight, outWidth: outWidth};
}
function get3DPadAndOutInfo(
    pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth,
    filterDepth, filterHeight, filterWidth, roundingMode) {
  var padInfo;
  var outDepth;
  var outHeight;
  var outWidth;
  if (typeof pad === 'number') {
    var padType = (pad === 0) ? 'VALID' : 'NUMBER';
    padInfo = {
      top: pad,
      bottom: pad,
      left: pad,
      right: pad,
      front: pad,
      back: pad,
      type: padType
    };
    var outShape = computeOutputShape4D(
        [inDepth, inHeight, inWidth, 1], filterDepth, 1, strideDepth, pad,
        roundingMode);
    outDepth = outShape[0];
    outHeight = outShape[1];
    outWidth = outShape[2];
  } else if (pad === 'same') {
    outDepth = Math.ceil(inDepth / strideDepth);
    outHeight = Math.ceil(inHeight / strideHeight);
    outWidth = Math.ceil(inWidth / strideWidth);
    var padAlongDepth = (outDepth - 1) * strideDepth + filterDepth - inDepth;
    var padAlongHeight =
        (outHeight - 1) * strideHeight + filterHeight - inHeight;
    var padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
    var front = Math.floor(padAlongDepth / 2);
    var back = padAlongDepth - front;
    var top_3 = Math.floor(padAlongHeight / 2);
    var bottom = padAlongHeight - top_3;
    var left = Math.floor(padAlongWidth / 2);
    var right = padAlongWidth - left;
    padInfo = {
      top: top_3,
      bottom: bottom,
      left: left,
      right: right,
      front: front,
      back: back,
      type: 'SAME'
    };
  } else if (pad === 'valid') {
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
  } else {
    throw Error('Unknown padding parameter: ' + pad);
  }
  return {
    padInfo: padInfo,
    outDepth: outDepth,
    outHeight: outHeight,
    outWidth: outWidth
  };
}
/**
 * Rounds a value depending on the rounding mode
 * @param value
 * @param roundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function round(value, roundingMode) {
  if (!roundingMode) {
    return Math.trunc(value);
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
      throw new Error('Unknown roundingMode ' + roundingMode);
  }
}
function tupleValuesAreOne(param) {
  var _a = parseTupleParam(param), dimA = _a[0], dimB = _a[1], dimC = _a[2];
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
  } else if (dataFormat === 'NCHW') {
    return 'channelsFirst';
  } else {
    throw new Error('Unknown dataFormat ' + dataFormat);
  }
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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function reshape_(x, shape) {
  var $x = convertToTensor(x, 'x', 'reshape', 'string_or_numeric');
  var inputs = {x: $x};
  var attrs = {shape: shape};
  return ENGINE.runKernel(Reshape, inputs, attrs);
}
var reshape = op({reshape_: reshape_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function avgPool_(x, filterSize, strides, pad, dimRoundingMode) {
  var $x = convertToTensor(x, 'x', 'avgPool', 'float32');
  var dilations = 1;
  assert(eitherStridesOrDilationsAreOne(strides, dilations), function() {
    return 'Error in avgPool: Either strides or dilations must be 1. ' +
        ('Got strides ' + strides + ' and dilations \'' + dilations + '\'');
  });
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  assert(x4D.rank === 4, function() {
    return 'Error in avgPool: x must be rank 4 but got rank ' + x4D.rank + '.';
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in avgPool: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  var inputs = {x: x4D};
  var attrs = {
    filterSize: filterSize,
    strides: strides,
    pad: pad,
    dimRoundingMode: dimRoundingMode
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(AvgPool, inputs, attrs);
  res = cast(res, $x.dtype);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var avgPool = op({avgPool_: avgPool_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function avgPool3d_(x, filterSize, strides, pad, dimRoundingMode, dataFormat) {
  if (dataFormat === void 0) {
    dataFormat = 'NDHWC';
  }
  var $x = convertToTensor(x, 'x', 'avgPool3d', 'float32');
  var x5D = $x;
  var reshapedTo5D = false;
  if ($x.rank === 4) {
    reshapedTo5D = true;
    x5D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]);
  }
  assert(x5D.rank === 5, function() {
    return 'Error in avgPool3d: x must be rank 5 but got rank ' + x5D.rank +
        '.';
  });
  assert(dataFormat === 'NDHWC', function() {
    return 'Error in avgPool3d: Only NDHWC is currently supported, ' +
        ('but got dataFormat of ' + dataFormat);
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in avgPool3d: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  var inputs = {x: x5D};
  var attrs = {
    filterSize: filterSize,
    strides: strides,
    pad: pad,
    dimRoundingMode: dimRoundingMode,
    dataFormat: dataFormat
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(AvgPool3D, inputs, attrs);
  res = cast(res, x5D.dtype);
  if (reshapedTo5D) {
    return reshape(
        res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
  }
  return res;
}
var avgPool3d = op({avgPool3d_: avgPool3d_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function concat_(tensors, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  assert(tensors.length >= 1, function() {
    return 'Pass at least one tensor to concat';
  });
  var $tensors =
      convertToTensorArray(tensors, 'tensors', 'concat', 'string_or_numeric');
  if ($tensors[0].dtype === 'complex64') {
    $tensors.forEach(function(tensor) {
      if (tensor.dtype !== 'complex64') {
        throw new Error(
            'Cannot concatenate complex64 tensors with a tensor\n          with dtype ' +
            tensor.dtype + '. ');
      }
    });
  }
  if ($tensors.length === 1) {
    return clone($tensors[0]);
  }
  var inputs = $tensors;
  var attr = {axis: axis};
  return ENGINE.runKernel(Concat, inputs, attr);
}
var concat = op({concat_: concat_});

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
 * Computes sigmoid element-wise, `1 / (1 + exp(-x))`
 *
 * ```js
 * const x = tf.tensor1d([0, -1, 2, -3]);
 *
 * x.sigmoid().print();  // or tf.sigmoid(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function sigmoid_(x) {
  var $x = convertToTensor(x, 'x', 'sigmoid');
  var inputs = {x: $x};
  return ENGINE.runKernel(Sigmoid, inputs);
}
var sigmoid = op({sigmoid_: sigmoid_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function slice_(x, begin, size) {
  var $x = convertToTensor(x, 'x', 'slice', 'string_or_numeric');
  if ($x.rank === 0) {
    throw new Error('Slicing scalar is not possible');
  }
  var inputs = {x: $x};
  var attrs = {begin: begin, size: size};
  return ENGINE.runKernel(Slice, inputs, attrs);
}
var slice = op({slice_: slice_});

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
 * Computes hyperbolic tangent of the input `tf.Tensor` element-wise: `tanh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, 70]);
 *
 * x.tanh().print();  // or tf.tanh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function tanh_(x) {
  var $x = convertToTensor(x, 'x', 'tanh');
  var inputs = {x: $x};
  return ENGINE.runKernel(Tanh, inputs);
}
var tanh$1 = op({tanh_: tanh_});

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
 *
 * @doc {heading: 'Operations', subheading: 'RNN'}
 */
function basicLSTMCell_(forgetBias, lstmKernel, lstmBias, data, c, h) {
  var $forgetBias = convertToTensor(forgetBias, 'forgetBias', 'basicLSTMCell');
  var $lstmKernel = convertToTensor(lstmKernel, 'lstmKernel', 'basicLSTMCell');
  var $lstmBias = convertToTensor(lstmBias, 'lstmBias', 'basicLSTMCell');
  var $data = convertToTensor(data, 'data', 'basicLSTMCell');
  var $c = convertToTensor(c, 'c', 'basicLSTMCell');
  var $h = convertToTensor(h, 'h', 'basicLSTMCell');
  var combined = concat([$data, $h], 1);
  var weighted = matMul(combined, $lstmKernel);
  var res = add$1(weighted, $lstmBias);
  // i = input_gate, j = new_input, f = forget_gate, o = output_gate
  var batchSize = res.shape[0];
  var sliceCols = res.shape[1] / 4;
  var sliceSize = [batchSize, sliceCols];
  var i = slice(res, [0, 0], sliceSize);
  var j = slice(res, [0, sliceCols], sliceSize);
  var f = slice(res, [0, sliceCols * 2], sliceSize);
  var o = slice(res, [0, sliceCols * 3], sliceSize);
  var newC = add$1(
      mul(sigmoid(i), tanh$1(j)), mul($c, sigmoid(add$1($forgetBias, f))));
  var newH = mul(tanh$1(newC), sigmoid(o));
  return [newC, newH];
}
var basicLSTMCell = op({basicLSTMCell_: basicLSTMCell_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function batchToSpaceND_(x, blockShape, crops) {
  var $x = convertToTensor(x, 'x', 'batchToSpaceND');
  var prod = blockShape.reduce(function(a, b) {
    return a * b;
  });
  assert($x.rank >= 1 + blockShape.length, function() {
    return 'input rank is ' + $x.rank +
        ' but should be > than blockShape.length ' + blockShape.length;
  });
  assert(crops.length === blockShape.length, function() {
    return 'crops.length is ' + crops.length +
        ' but should be equal to blockShape.length  ' + blockShape.length;
  });
  assert($x.shape[0] % prod === 0, function() {
    return 'input tensor batch is ' + $x.shape[0] +
        ' but is not divisible by the product of ' +
        ('the elements of blockShape ' + blockShape.join(' * ') +
         ' === ' + prod);
  });
  var inputs = {x: $x};
  var attrs = {blockShape: blockShape, crops: crops};
  return ENGINE.runKernel(BatchToSpaceND, inputs, attrs);
}
var batchToSpaceND = op({batchToSpaceND_: batchToSpaceND_});

function xAs4D(x) {
  var x4D;
  if (x.rank === 0 || x.rank === 1) {
    x4D = reshape(x, [1, 1, 1, x.size]);
  } else if (x.rank === 2) {
    x4D = reshape(x, [1, 1, x.shape[0], x.shape[1]]);
  } else if (x.rank === 3) {
    x4D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]]);
  } else {
    x4D = x;
  }
  return x4D;
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
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function batchNorm_(x, mean, variance, offset, scale, varianceEpsilon) {
  if (varianceEpsilon == null) {
    varianceEpsilon = 0.001;
  }
  var $x = convertToTensor(x, 'x', 'batchNorm');
  var $mean = convertToTensor(mean, 'mean', 'batchNorm');
  var $variance = convertToTensor(variance, 'variance', 'batchNorm');
  var $scale;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  var $offset;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  assert($mean.rank === $variance.rank, function() {
    return 'Batch normalization gradient requires mean and variance to have ' +
        'equal ranks.';
  });
  assert($offset == null || $mean.rank === $offset.rank, function() {
    return 'Batch normalization gradient requires mean and offset to have ' +
        'equal ranks.';
  });
  assert($scale == null || $mean.rank === $scale.rank, function() {
    return 'Batch normalization gradient requires mean and scale to have ' +
        'equal ranks.';
  });
  var x4D = xAs4D($x);
  var inputs = {
    x: x4D,
    scale: $scale,
    offset: $offset,
    mean: $mean,
    variance: $variance
  };
  var attrs = {varianceEpsilon: varianceEpsilon};
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(FusedBatchNorm, inputs, attrs);
  return reshape(res, $x.shape);
}
var batchNorm = op({batchNorm_: batchNorm_});

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
  var $x = convertToTensor(x, 'x', 'batchNorm');
  var $mean = convertToTensor(mean, 'mean', 'batchNorm');
  var $variance = convertToTensor(variance, 'variance', 'batchNorm');
  var $scale;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  var $offset;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  assert($x.rank === 2, function() {
    return 'Error in batchNorm2D: x must be rank 2 but got rank ' +
        ($x.rank + '.');
  });
  assert($mean.rank === 2 || $mean.rank === 1, function() {
    return 'Error in batchNorm2D: mean must be rank 2 or rank 1 but ' +
        ('got rank ' + $mean.rank + '.');
  });
  assert($variance.rank === 2 || $variance.rank === 1, function() {
    return 'Error in batchNorm2D: variance must be rank 2 or rank 1 ' +
        ('but got rank ' + $variance.rank + '.');
  });
  if ($scale != null) {
    assert($scale.rank === 2 || $scale.rank === 1, function() {
      return 'Error in batchNorm2D: scale must be rank 2 or rank 1 ' +
          ('but got rank ' + $scale.rank + '.');
    });
  }
  if ($offset != null) {
    assert($offset.rank === 2 || $offset.rank === 1, function() {
      return 'Error in batchNorm2D: offset must be rank 2 or rank 1 ' +
          ('but got rank ' + $offset.rank + '.');
    });
  }
  return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}
var batchNorm2d = op({batchNorm2d_: batchNorm2d_});

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
  var $x = convertToTensor(x, 'x', 'batchNorm');
  var $mean = convertToTensor(mean, 'mean', 'batchNorm');
  var $variance = convertToTensor(variance, 'variance', 'batchNorm');
  var $scale;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  var $offset;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  assert($x.rank === 3, function() {
    return 'Error in batchNorm3D: x must be rank 3 but got rank ' +
        ($x.rank + '.');
  });
  assert($mean.rank === 3 || $mean.rank === 1, function() {
    return 'Error in batchNorm3D: mean must be rank 3 or rank 1 but ' +
        ('got rank ' + $mean.rank + '.');
  });
  assert($variance.rank === 3 || $variance.rank === 1, function() {
    return 'Error in batchNorm3D: variance must be rank 3 or rank 1 ' +
        ('but got rank ' + $variance.rank + '.');
  });
  if ($scale != null) {
    assert($scale.rank === 3 || $scale.rank === 1, function() {
      return 'Error in batchNorm3D: scale must be rank 3 or rank 1 ' +
          ('but got rank ' + $scale.rank + '.');
    });
  }
  if ($offset != null) {
    assert($offset.rank === 3 || $offset.rank === 1, function() {
      return 'Error in batchNorm3D: offset must be rank 3 or rank 1 ' +
          ('but got rank ' + $offset.rank + '.');
    });
  }
  return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}
var batchNorm3d = op({batchNorm3d_: batchNorm3d_});

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
  var $x = convertToTensor(x, 'x', 'batchNorm');
  var $mean = convertToTensor(mean, 'mean', 'batchNorm');
  var $variance = convertToTensor(variance, 'variance', 'batchNorm');
  var $scale;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  var $offset;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  assert($x.rank === 4, function() {
    return 'Error in batchNorm4D: x must be rank 4 but got rank ' +
        ($x.rank + '.');
  });
  assert($mean.rank === 4 || $mean.rank === 1, function() {
    return 'Error in batchNorm4D: mean must be rank 4 or rank 1 but ' +
        ('got rank ' + $mean.rank + '.');
  });
  assert($variance.rank === 4 || $variance.rank === 1, function() {
    return 'Error in batchNorm4D: variance must be rank 4 or rank 1 ' +
        ('but got rank ' + $variance.rank + '.');
  });
  if ($scale != null) {
    assert($scale.rank === 4 || $scale.rank === 1, function() {
      return 'Error in batchNorm4D: scale must be rank 4 or rank 1 ' +
          ('but got rank ' + $scale.rank + '.');
    });
  }
  if ($offset != null) {
    assert($offset.rank === 4 || $offset.rank === 1, function() {
      return 'Error in batchNorm4D: offset must be rank 4 or rank 1 ' +
          ('but got rank ' + $offset.rank + '.');
    });
  }
  return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}
var batchNorm4d = op({batchNorm4d_: batchNorm4d_});

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
/**
 * Outputs a vector with length `size` and the same dtype as `weights`.
 *
 * If `weights` are empty, then index `i` stores the number of times the value
 * `i` is counted in `x`. If `weights` are non-empty, then index `i` stores the
 * sum of the value in `weights` at each index where the corresponding value in
 * `x` is `i`.
 *
 * Values in `x` outside of the range [0, size) are ignored.
 *
 * @param x The input int tensor, rank 1.
 * @param weights The weights tensor, must have the same shape as x, or a
 *     length-0 Tensor, in which case it acts as all weights equal to 1.
 * @param size Non-negative integer.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function bincount_(x, weights, size) {
  var $x = convertToTensor(x, 'x', 'bincount');
  var $weights = convertToTensor(weights, 'weights', 'bincount');
  assert($x.dtype === 'int32', function() {
    return 'Error in bincount: input ' +
        ('dtype must be int32, but got ' + $x.dtype);
  });
  assert(size >= 0, function() {
    return 'size must be non-negative, but got ' + size + '.';
  });
  assert($weights.size === $x.size || $weights.size === 0, function() {
    return 'Error in bincount: weights must have the same size as input or' +
        ('0-length, but got input shape: ' + $x.shape + ', weights shape: ') +
        ($weights.shape + '.');
  });
  var inputs = {x: $x, weights: $weights};
  var attrs = {size: size};
  return ENGINE.runKernel(Bincount, inputs, attrs);
}
var bincount = op({bincount_: bincount_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function broadcastTo_(x, shape) {
  var input = convertToTensor(x, 'broadcastTo', 'x');
  var xShape = input.shape;
  if (shape.some(function(d) {
        return !(d > 0) || d % 1 !== 0;
      })) {
    throw new Error('broadcastTo(): Invalid broadcast shape [' + shape + '].');
  }
  if (shape.length < input.rank) {
    throw new Error(
        'broadcastTo(): shape.length=' + shape.length +
        ' < input.rank=' + input.rank + '.');
  }
  if (shape.length > input.rank) {
    var newShape = input.shape.slice();
    while (newShape.length < shape.length) {
      newShape.unshift(1);
    }
    input = reshape(input, newShape);
  }
  var inputShape = input.shape;
  var reps = Array.from(shape);
  for (var i = shape.length - 1; i >= 0; i--) {
    if (inputShape[i] === shape[i]) {
      reps[i] = 1;
    } else if (input.shape[i] !== 1) {
      throw new Error(
          'broadcastTo(): [' + xShape + '] cannot be broadcast to [' + shape +
          '].');
    }
  }
  var axes = reps.map(function(n, i) {
                   return n > 1 ? i : -1;
                 })
                 .filter(function(i) {
                   return i >= 0;
                 });
  if (axes.length === 0) {
    return clone(input);
  }
  // TODO call broadcastTo kernel directly once backends implement broadcstTo
  var inputs = {x: input};
  var attrs = {reps: reps};
  return ENGINE.runKernel(Tile, inputs, attrs);
}
var broadcastTo = op({broadcastTo_: broadcastTo_});

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
 * Computes ceiling of input `tf.Tensor` element-wise: `ceil(x)`
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.ceil().print();  // or tf.ceil(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function ceil_(x) {
  var $x = convertToTensor(x, 'x', 'ceil');
  var inputs = {x: $x};
  return ENGINE.runKernel(Ceil, inputs);
}
var ceil = op({ceil_: ceil_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function clipByValue_(x, clipValueMin, clipValueMax) {
  var $x = convertToTensor(x, 'x', 'clipByValue');
  assert((clipValueMin <= clipValueMax), function() {
    return 'Error in clip: min (' + clipValueMin + ') must be ' +
        ('less than or equal to max (' + clipValueMax + ').');
  });
  var inputs = {x: $x};
  var attrs = {clipValueMin: clipValueMin, clipValueMax: clipValueMax};
  return ENGINE.runKernel(ClipByValue, inputs, attrs);
}
var clipByValue = op({clipByValue_: clipByValue_});

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
var concat1d = op({concat1d_: concat1d_});

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
var concat2d = op({concat2d_: concat2d_});

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
var concat3d = op({concat3d_: concat3d_});

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
var concat4d = op({concat4d_: concat4d_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function conv2d_(
    x, filter, strides, pad, dataFormat, dilations, dimRoundingMode) {
  if (dataFormat === void 0) {
    dataFormat = 'NHWC';
  }
  if (dilations === void 0) {
    dilations = [1, 1];
  }
  var $x = convertToTensor(x, 'x', 'conv2d');
  var $filter = convertToTensor(filter, 'filter', 'conv2d');
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  assert(x4D.rank === 4, function() {
    return 'Error in conv2d: input must be rank 4, but got rank ' + x4D.rank +
        '.';
  });
  assert($filter.rank === 4, function() {
    return 'Error in conv2d: filter must be rank 4, but got rank ' +
        ($filter.rank + '.');
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in conv2d: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  var inDepth = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
  assert(inDepth === $filter.shape[2], function() {
    return 'Error in conv2d: depth of input (' + inDepth + ') must match ' +
        ('input depth for filter ' + $filter.shape[2] + '.');
  });
  assert(eitherStridesOrDilationsAreOne(strides, dilations), function() {
    return 'Error in conv2D: Either strides or dilations must be 1. ' +
        ('Got strides ' + strides + ' and dilations \'' + dilations + '\'');
  });
  var inputs = {x: x4D, filter: $filter};
  var attrs = {
    strides: strides,
    pad: pad,
    dataFormat: dataFormat,
    dilations: dilations,
    dimRoundingMode: dimRoundingMode
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(Conv2D, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var conv2d = op({conv2d_: conv2d_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function conv1d_(
    x, filter, stride, pad, dataFormat, dilation, dimRoundingMode) {
  if (dataFormat === void 0) {
    dataFormat = 'NWC';
  }
  if (dilation === void 0) {
    dilation = 1;
  }
  var $x = convertToTensor(x, 'x', 'conv1d');
  var $filter = convertToTensor(filter, 'filter', 'conv1d');
  var x3D = $x;
  var reshapedTo3D = false;
  if ($x.rank === 2) {
    reshapedTo3D = true;
    x3D = reshape($x, [1, $x.shape[0], $x.shape[1]]);
  }
  assert(x3D.rank === 3, function() {
    return 'Error in conv1d: input must be rank 3, but got rank ' + x3D.rank +
        '.';
  });
  assert($filter.rank === 3, function() {
    return 'Error in conv1d: filter must be rank 3, but got rank ' +
        ($filter.rank + '.');
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in conv1d: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  assert(x3D.shape[2] === $filter.shape[1], function() {
    return 'Error in conv1d: depth of input (' + x3D.shape[2] +
        ') must match ' + ('input depth for filter ' + $filter.shape[1] + '.');
  });
  assert(eitherStridesOrDilationsAreOne(stride, dilation), function() {
    return 'Error in conv1D: Either stride or dilation must be 1. ' +
        ('Got stride ' + stride + ' and dilation \'' + dilation + '\'');
  });
  assert(dataFormat === 'NWC', function() {
    return 'Error in conv1d: got dataFormat of ' + dataFormat +
        ' but only NWC is currently supported.';
  });
  var filter4D = reshape(
      $filter, [1, $filter.shape[0], $filter.shape[1], $filter.shape[2]]);
  var input4D = reshape(x3D, [x3D.shape[0], 1, x3D.shape[1], x3D.shape[2]]);
  var strides = [1, stride];
  var dilations = [1, dilation];
  var conv2dDataFormat = 'NHWC';
  var res = conv2d(
      input4D, filter4D, strides, pad, conv2dDataFormat, dilations,
      dimRoundingMode);
  if (reshapedTo3D) {
    return reshape(res, [res.shape[2], res.shape[3]]);
  }
  return reshape(res, [res.shape[0], res.shape[2], res.shape[3]]);
}
var conv1d = op({conv1d_: conv1d_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function conv2DBackpropInput_(
    xShape, dy, filter, strides, pad, dataFormat, dimRoundingMode) {
  if (dataFormat === void 0) {
    dataFormat = 'NHWC';
  }
  assert(xShape.length === dy.rank, function() {
    return 'Length of inShape ' +
        ('(' + xShape.length + ') and rank of dy (' + dy.rank + ') must match');
  });
  var xShape4D = xShape;
  var dy4D = dy;
  var reshapedTo4D = false;
  if (dy.rank === 3) {
    reshapedTo4D = true;
    dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
    xShape4D = [1, xShape[0], xShape[1], xShape[2]];
  }
  assert(xShape4D.length === 4, function() {
    return 'Error in conv2dDerInput: inShape must be length 4, but got length ' +
        (xShape4D.length + '.');
  });
  assert(dy4D.rank === 4, function() {
    return 'Error in conv2dDerInput: dy must be rank 4, but got ' +
        ('rank ' + dy4D.rank);
  });
  assert(filter.rank === 4, function() {
    return 'Error in conv2dDerInput: filter must be rank 4, but got ' +
        ('rank ' + filter.rank);
  });
  var inDepth = dataFormat === 'NHWC' ? xShape4D[3] : xShape4D[1];
  var outDepth = dataFormat === 'NHWC' ? dy4D.shape[3] : dy4D.shape[1];
  assert(inDepth === filter.shape[2], function() {
    return 'Error in conv2dDerInput: depth of input (' + inDepth + ') must ' +
        ('match input depth for filter ' + filter.shape[2] + '.');
  });
  assert(outDepth === filter.shape[3], function() {
    return 'Error in conv2dDerInput: depth of output (' + outDepth + ') must ' +
        ('match output depth for filter ' + filter.shape[3] + '.');
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in conv2dDerInput: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  var inputs = {dy: dy4D, filter: filter};
  var attrs = {
    strides: strides,
    pad: pad,
    dataFormat: dataFormat,
    dimRoundingMode: dimRoundingMode,
    inputShape: xShape4D
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(Conv2DBackpropInput, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var conv2DBackpropInput = op({conv2DBackpropInput_: conv2DBackpropInput_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function conv2dTranspose_(
    x, filter, outputShape, strides, pad, dimRoundingMode) {
  var $x = convertToTensor(x, 'x', 'conv2dTranspose');
  var $filter = convertToTensor(filter, 'filter', 'conv2dTranspose');
  return conv2DBackpropInput(
      outputShape, $x, $filter, strides, pad, 'NHWC', dimRoundingMode);
}
var conv2dTranspose = op({conv2dTranspose_: conv2dTranspose_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function conv3d_(x, filter, strides, pad, dataFormat, dilations) {
  if (dataFormat === void 0) {
    dataFormat = 'NDHWC';
  }
  if (dilations === void 0) {
    dilations = [1, 1, 1];
  }
  var $x = convertToTensor(x, 'x', 'conv3d');
  var $filter = convertToTensor(filter, 'filter', 'conv3d');
  var x5D = $x;
  var reshapedTo5D = false;
  if ($x.rank === 4) {
    reshapedTo5D = true;
    x5D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]);
  }
  assert(x5D.rank === 5, function() {
    return 'Error in conv3d: input must be rank 5, but got rank ' + x5D.rank +
        '.';
  });
  assert($filter.rank === 5, function() {
    return 'Error in conv3d: filter must be rank 5, but got rank ' +
        ($filter.rank + '.');
  });
  assert(x5D.shape[4] === $filter.shape[3], function() {
    return 'Error in conv3d: depth of input (' + x5D.shape[4] +
        ') must match ' + ('input depth for filter ' + $filter.shape[3] + '.');
  });
  assert(eitherStridesOrDilationsAreOne(strides, dilations), function() {
    return 'Error in conv3D: Either strides or dilations must be 1. ' +
        ('Got strides ' + strides + ' and dilations \'' + dilations + '\'');
  });
  assert(dataFormat === 'NDHWC', function() {
    return 'Error in conv3d: got dataFormat of ' + dataFormat +
        ' but only NDHWC is currently supported.';
  });
  var inputs = {x: x5D, filter: $filter};
  var attrs = {
    strides: strides,
    pad: pad,
    dataFormat: dataFormat,
    dilations: dilations
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(Conv3D, inputs, attrs);
  if (reshapedTo5D) {
    return reshape(
        res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
  }
  return res;
}
var conv3d = op({conv3d_: conv3d_});

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
function conv3DBackpropInput_(xShape, dy, filter, strides, pad) {
  assert(xShape.length === dy.rank, function() {
    return 'Length of inShape ' +
        ('(' + xShape.length + ') and rank of dy (' + dy.rank + ') must match');
  });
  var xShape5D = xShape;
  var dy5D = dy;
  var reshapedTo5D = false;
  if (dy.rank === 4) {
    reshapedTo5D = true;
    dy5D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]]);
    xShape5D = [1, xShape[0], xShape[1], xShape[2], xShape[3]];
  }
  var inDepth = xShape5D[4];
  var outDepth = dy5D.shape[4];
  assert(xShape5D.length === 5, function() {
    return 'Error in conv3dDerInput: inShape must be length 5, but got length ' +
        (xShape5D.length + '.');
  });
  assert(dy5D.rank === 5, function() {
    return 'Error in conv3dDerInput: dy must be rank 5, but got ' +
        ('rank ' + dy5D.rank);
  });
  assert(filter.rank === 5, function() {
    return 'Error in conv3dDerInput: filter must be rank 5, but got ' +
        ('rank ' + filter.rank);
  });
  assert(inDepth === filter.shape[3], function() {
    return 'Error in conv3dDerInput: depth of input (' + inDepth + ') must ' +
        ('match input depth for filter ' + filter.shape[3] + '.');
  });
  assert(outDepth === filter.shape[4], function() {
    return 'Error in conv3dDerInput: depth of output (' + outDepth + ') must ' +
        ('match output depth for filter ' + filter.shape[4] + '.');
  });
  var inputs = {dy: dy5D, filter: filter};
  var attrs = {pad: pad, strides: strides, inputShape: xShape5D};
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(Conv3DBackpropInputV2, inputs, attrs);
  if (reshapedTo5D) {
    return reshape(
        res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
  }
  return res;
}
var conv3DBackpropInput = op({conv3DBackpropInput_: conv3DBackpropInput_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function conv3dTranspose_(x, filter, outputShape, strides, pad) {
  var $x = convertToTensor(x, 'x', 'conv3dTranspose');
  var $filter = convertToTensor(filter, 'filter', 'conv3dTranspose');
  return conv3DBackpropInput(outputShape, $x, $filter, strides, pad);
}
var conv3dTranspose = op({conv3dTranspose_: conv3dTranspose_});

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
 * Computes cos of the input `tf.Tensor` element-wise: `cos(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.cos().print();  // or tf.cos(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function cos_(x) {
  var $x = convertToTensor(x, 'x', 'cos');
  var inputs = {x: $x};
  return ENGINE.runKernel(Cos, inputs);
}
var cos = op({cos_: cos_});

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
 * Computes hyperbolic cos of the input `tf.Tensor` element-wise: `cosh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.cosh().print();  // or tf.cosh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function cosh_(x) {
  var $x = convertToTensor(x, 'x', 'cosh');
  var inputs = {x: $x};
  return ENGINE.runKernel(Cosh, inputs);
}
var cosh = op({cosh_: cosh_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Scan'}
 */
function cumsum_(x, axis, exclusive, reverse) {
  if (axis === void 0) {
    axis = 0;
  }
  if (exclusive === void 0) {
    exclusive = false;
  }
  if (reverse === void 0) {
    reverse = false;
  }
  var $x = convertToTensor(x, 'x', 'cumsum');
  var inputs = {x: $x};
  var attrs = {axis: axis, exclusive: exclusive, reverse: reverse};
  return ENGINE.runKernel(Cumsum, inputs, attrs);
}
var cumsum = op({cumsum_: cumsum_});

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
/**
 * Outputs a vector with length `size` and the same dtype as `weights`.
 *
 * If `weights` are empty, then index `i` stores the number of times the value
 * `i` is counted in `x`. If `weights` are non-empty, then index `i` stores the
 * sum of the value in `weights` at each index where the corresponding value in
 * `x` is `i`.
 *
 * Values in `x` outside of the range [0, size) are ignored.
 *
 * @param x The input int tensor, rank 1 or rank 2.
 * @param weights The weights tensor, must have the same shape as x, or a
 *     length-0 Tensor, in which case it acts as all weights equal to 1.
 * @param size Non-negative integer.
 * @param binaryOutput Optional. Whether the kernel should count the appearance
 *     or number of occurrences. Defaults to False.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function denseBincount_(x, weights, size, binaryOutput) {
  if (binaryOutput === void 0) {
    binaryOutput = false;
  }
  var $x = convertToTensor(x, 'x', 'denseBincount');
  var $weights = convertToTensor(weights, 'weights', 'denseBincount');
  assert($x.dtype === 'int32', function() {
    return 'Error in denseBincount: input ' +
        ('dtype must be int32, but got ' + $x.dtype);
  });
  assert($x.rank <= 2, function() {
    return 'Error in denseBincount: input must be at most rank 2, but got ' +
        ('rank ' + $x.rank + '.');
  });
  assert(size >= 0, function() {
    return 'size must be non-negative, but got ' + size + '.';
  });
  assert($weights.size === $x.size || $weights.size === 0, function() {
    return 'Error in denseBincount: weights must have the same shape as x or ' +
        ('0-length, but got x shape: ' + $x.shape + ', weights shape: ') +
        ($weights.shape + '.');
  });
  var inputs = {x: $x, weights: $weights};
  var attrs = {size: size, binaryOutput: binaryOutput};
  return ENGINE.runKernel(DenseBincount, inputs, attrs);
}
var denseBincount = op({denseBincount_: denseBincount_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function depthToSpace_(x, blockSize, dataFormat) {
  if (dataFormat === void 0) {
    dataFormat = 'NHWC';
  }
  var $x = convertToTensor(x, 'x', 'depthToSpace');
  var inputHeight = (dataFormat === 'NHWC') ? $x.shape[1] : $x.shape[2];
  var inputWidth = (dataFormat === 'NHWC') ? $x.shape[2] : $x.shape[3];
  var inputDepth = (dataFormat === 'NHWC') ? $x.shape[3] : $x.shape[1];
  assert(inputHeight * blockSize >= 0, function() {
    return 'Negative dimension size caused by overflow when multiplying\n    ' +
        inputHeight + ' and ' + blockSize +
        '  for depthToSpace with input shape\n    ' + $x.shape;
  });
  assert(inputWidth * blockSize >= 0, function() {
    return 'Negative dimension size caused by overflow when multiplying\n    ' +
        inputWidth + ' and ' + blockSize +
        ' for depthToSpace with input shape\n        ' + $x.shape;
  });
  assert((inputDepth % (blockSize * blockSize) === 0), function() {
    return 'Dimension size must be evenly divisible by ' +
        blockSize * blockSize + ' but is ' + inputDepth +
        ' for depthToSpace with input shape ' + $x.shape;
  });
  var inputs = {x: $x};
  var attrs = {blockSize: blockSize, dataFormat: dataFormat};
  return ENGINE.runKernel(DepthToSpace, inputs, attrs);
}
var depthToSpace = op({depthToSpace_: depthToSpace_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function depthwiseConv2d_(
    x, filter, strides, pad, dataFormat, dilations, dimRoundingMode) {
  if (dataFormat === void 0) {
    dataFormat = 'NHWC';
  }
  if (dilations === void 0) {
    dilations = [1, 1];
  }
  var $x = convertToTensor(x, 'x', 'depthwiseConv2d');
  var $filter = convertToTensor(filter, 'filter', 'depthwiseConv2d');
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  assert(x4D.rank === 4, function() {
    return 'Error in depthwiseConv2d: input must be rank 4, but got ' +
        ('rank ' + x4D.rank + '.');
  });
  assert($filter.rank === 4, function() {
    return 'Error in depthwiseConv2d: filter must be rank 4, but got rank ' +
        ($filter.rank + '.');
  });
  assert(x4D.shape[3] === $filter.shape[2], function() {
    return 'Error in depthwiseConv2d: number of input channels ' +
        ('(' + x4D.shape[3] + ') must match the inChannels dimension in ') +
        ('filter ' + $filter.shape[2] + '.');
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in depthwiseConv2d: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  var inputs = {x: x4D, filter: $filter};
  var attrs = {
    strides: strides,
    pad: pad,
    dataFormat: dataFormat,
    dilations: dilations,
    dimRoundingMode: dimRoundingMode
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(DepthwiseConv2dNative, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var depthwiseConv2d = op({depthwiseConv2d_: depthwiseConv2d_});

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
  var $x = convertToTensor(x, 'x', 'diag');
  var inputs = {x: $x};
  return ENGINE.runKernel(Diag, inputs);
}
var diag = op({diag_: diag_});

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
/**
 * Computes the grayscale dilation over the input `x`.
 *
 * @param x The input tensor, rank 3 or rank 4 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filter The filter tensor, rank 3, of shape
 *     `[filterHeight, filterWidth, depth]`.
 * @param strides The strides of the sliding window for each dimension of the
 *     input tensor: `[strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat Specify the data format of the input and output data.
 *      Defaults to 'NHWC'. Only 'NHWC' is currently supported. With the
 *      default format "NHWC", the data is stored in the order of: [batch,
 *      height, width, channels].
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     for atrous morphological dilation. Defaults to `[1, 1]`. If `dilations`
 *     is a single number, then `dilationHeight == dilationWidth`. If it is
 *     greater than 1, then all values of `strides` must be 1.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function dilation2d_(x, filter, strides, pad, dilations, dataFormat) {
  if (dilations === void 0) {
    dilations = [1, 1];
  }
  if (dataFormat === void 0) {
    dataFormat = 'NHWC';
  }
  var $x = convertToTensor(x, 'x', 'dilation2d');
  var $filter = convertToTensor(filter, 'filter', 'dilation2d');
  assert($x.rank === 3 || $x.rank === 4, function() {
    return 'Error in dilation2d: input must be rank 3 or 4, but got rank ' +
        ($x.rank + '.');
  });
  assert($filter.rank === 3, function() {
    return 'Error in dilation2d: filter must be rank 3, but got rank ' +
        ($filter.rank + '.');
  });
  assert(dataFormat === 'NHWC', function() {
    return 'Error in dilation2d: Only NHWC is currently supported, ' +
        ('but got dataFormat of ' + dataFormat);
  });
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    reshapedTo4D = true;
  }
  var inputs = {x: x4D, filter: $filter};
  var attrs = {strides: strides, pad: pad, dilations: dilations};
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(Dilation2D, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var dilation2d = op({dilation2d_: dilation2d_});

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
  var inRank = inShape.length;
  var dims = [];
  for (var i = 0; i < inRank; i++) {
    var dim = inRank - 1 - i;
    var a = inShape[dim] || 1;
    var b = outShape[outShape.length - 1 - i] || 1;
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
  var result = [];
  for (var i = 0; i < outShape.length; i++) {
    var inDim = inShape[inShape.length - i - 1];
    var outAxis = outShape.length - i - 1;
    var outDim = outShape[outAxis];
    if (inDim == null || (inDim === 1 && outDim > 1)) {
      result.unshift(outAxis);
    }
  }
  return result;
}
function assertAndGetBroadcastShape(shapeA, shapeB) {
  var result = [];
  var l = Math.max(shapeA.length, shapeB.length);
  for (var i = 0; i < l; i++) {
    var a = shapeA[shapeA.length - i - 1];
    if (a == null) {
      a = 1;
    }
    var b = shapeB[shapeB.length - i - 1];
    if (b == null) {
      b = 1;
    }
    if (a === 1) {
      result.unshift(b);
    } else if (b === 1) {
      result.unshift(a);
    } else if (a !== b) {
      var errMsg = 'Operands could not be broadcast together with shapes ' +
          (shapeA + ' and ' + shapeB + '.');
      throw Error(errMsg);
    } else {
      result.unshift(a);
    }
  }
  return result;
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
/**
 * Returns the truth value of (a == b) element-wise. Supports broadcasting.
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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function equal_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'equal');
  var $b = convertToTensor(b, 'b', 'equal');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Equal, inputs);
}
var equal = op({equal_: equal_});

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
 * @param b A tensor with the same dtype as `a` and with shape that is
 *     compatible with `a`.
 * @return A tensor with same dtype as `a` and `b`, and shape that is
 *     broadcastable from `a` and `b`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function where_(condition, a, b) {
  var $a = convertToTensor(a, 'a', 'where');
  var $b = convertToTensor(b, 'b', 'where');
  var $condition = convertToTensor(condition, 'condition', 'where', 'bool');
  // TODO: move this logic to forward function when the broadcastTo op is
  // implemented in WASM.
  // Find the broadcastable shape for $a and $b.
  var broadcastShape = assertAndGetBroadcastShape($a.shape, $b.shape);
  var $broadcastedA = broadcastTo($a, broadcastShape);
  var $broadcastedB = broadcastTo($b, broadcastShape);
  if ($condition.rank === 1) {
    // If condition rank is 1, then the first dimension must match the size of
    // condition.
    assert($condition.shape[0] === $a.shape[0], function() {
      return 'The first dimension of `a` must match the size of `condition`.';
    });
  }
  if ($condition.rank !== 1) {
    // A must have the same shape as condition.
    assertShapesMatch(
        $condition.shape, $broadcastedB.shape, 'Error in where: ');
  }
  var inputs = {condition: $condition, t: $broadcastedA, e: $broadcastedB};
  return ENGINE.runKernel(Select, inputs);
}
var where = op({where_: where_});

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
 * Creates a `tf.Tensor` with all elements set to 0 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.zerosLike(x).print();
 * ```
 *
 * @param x The tensor of required shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function zerosLike_(x) {
  var $x = convertToTensor(x, 'x', 'zerosLike');
  var inputs = {x: $x};
  return ENGINE.runKernel(ZerosLike, inputs);
}
var zerosLike = op({zerosLike_: zerosLike_});

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
/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting. Return 0
 * if denominator is 0.
 *
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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function divNoNan_(a, b) {
  var _a;
  // TODO: Make this into its own kernel.
  var $a = convertToTensor(a, 'a', 'div');
  var $b = convertToTensor(b, 'b', 'div');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  var divResult = div($a, $b);
  var zeros = zerosLike(divResult);
  var bEqualsZero = equal($b, zeros);
  return where(bEqualsZero, zeros, divResult);
}
var divNoNan = op({divNoNan_: divNoNan_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function dot_(t1, t2) {
  var $t1 = convertToTensor(t1, 't1', 'dot');
  var $t2 = convertToTensor(t2, 't2', 'dot');
  assert(
      ($t1.rank === 1 || $t1.rank === 2) && ($t2.rank === 1 || $t2.rank === 2),
      function() {
        return 'Error in dot: inputs must all be rank 1 or 2, but got ranks ' +
            ($t1.rank + ' and ' + $t2.rank + '.');
      });
  var t1Inner = ($t1.rank === 1 ? $t1.size : $t1.shape[1]);
  var t2Inner = ($t2.rank === 1 ? $t2.size : $t2.shape[0]);
  assert(t1Inner === t2Inner, function() {
    return 'Error in dot: inner dimensions of inputs must match, but got ' +
        (t1Inner + ' and ' + t2Inner + '.');
  });
  if ($t1.rank === 1 && $t2.rank === 1) {
    var t12D = reshape($t1, [1, -1]);
    var t22D = reshape($t2, [-1, 1]);
    var t1t2 = matMul(t12D, t22D);
    return reshape(t1t2, []);
  } else if ($t1.rank === 1 && $t2.rank === 2) {
    var t12D = reshape($t1, [1, -1]);
    var t22D = reshape($t2, [$t2.shape[0], $t2.shape[1]]);
    var t1t2 = matMul(t12D, t22D);
    return reshape(t1t2, [t1t2.size]);
  } else if ($t1.rank === 2 && $t2.rank === 1) {
    var t22D = reshape($t2, [-1, 1]);
    var t1t2 = matMul($t1, t22D);
    return reshape(t1t2, [t1t2.size]);
  } else {
    var t22D = reshape($t2, [$t2.shape[0], $t2.shape[1]]);
    var t1t2 = matMul($t1, t22D);
    return t1t2;
  }
}
var dot = op({dot_: dot_});

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
/**
 * Computes exponential linear element-wise: `x > 0 ? e ^ x - 1 : 0`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 1, -3, 2]);
 *
 * x.elu().print();  // or tf.elu(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function elu_(x) {
  var $x = convertToTensor(x, 'x', 'elu');
  var inputs = {x: $x};
  return ENGINE.runKernel(Elu, inputs);
}
var elu = op({elu_: elu_});

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
 * Computes gause error function of the input `tf.Tensor` element-wise:
 * `erf(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, .1, -.1, .7]);
 *
 * x.erf().print(); // or tf.erf(x);
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function erf_(x) {
  var $x = convertToTensor(x, 'x', 'erf');
  assert($x.dtype === 'int32' || $x.dtype === 'float32', function() {
    return 'Input dtype must be `int32` or `float32`.';
  });
  if ($x.dtype === 'int32') {
    $x = cast($x, 'float32');
  }
  var inputs = {x: $x};
  return ENGINE.runKernel(Erf, inputs);
}
var erf = op({erf_: erf_});

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
 * Computes exponential of the input `tf.Tensor` element-wise. `e ^ x`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, -3]);
 *
 * x.exp().print();  // or tf.exp(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function exp_(x) {
  var $x = convertToTensor(x, 'x', 'exp');
  var inputs = {x: $x};
  return ENGINE.runKernel(Exp, inputs);
}
var exp = op({exp_: exp_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function expandDims_(x, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  var $x = convertToTensor(x, 'x', 'expandDims', 'string_or_numeric');
  assert(axis <= $x.rank, function() {
    return 'Axis must be <= rank of the tensor';
  });
  var inputs = {input: $x};
  var attrs = {dim: axis};
  return ENGINE.runKernel(ExpandDims, inputs, attrs);
}
var expandDims = op({expandDims_: expandDims_});

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
 * Computes exponential of the input `tf.Tensor` minus one element-wise.
 * `e ^ x - 1`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, -3]);
 *
 * x.expm1().print();  // or tf.expm1(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function expm1_(x) {
  var $x = convertToTensor(x, 'x', 'expm1');
  var inputs = {x: $x};
  return ENGINE.runKernel(Expm1, inputs);
}
var expm1 = op({expm1_: expm1_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function tile_(x, reps) {
  var $x = convertToTensor(x, 'x', 'tile', 'string_or_numeric');
  assert($x.rank === reps.length, function() {
    return 'Error in transpose: rank of input ' + $x.rank + ' ' +
        ('must match length of reps ' + reps + '.');
  });
  var inputs = {x: $x};
  var attrs = {reps: reps};
  return ENGINE.runKernel(Tile, inputs, attrs);
}
var tile = op({tile_: tile_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function eye_(numRows, numColumns, batchShape, dtype) {
  if (dtype === void 0) {
    dtype = 'float32';
  }
  if (numColumns == null) {
    numColumns = numRows;
  }
  var buff = buffer([numRows, numColumns], dtype);
  var n = numRows <= numColumns ? numRows : numColumns;
  for (var i = 0; i < n; ++i) {
    buff.set(1, i, i);
  }
  var out = reshape(buff.toTensor(), [numRows, numColumns]);
  if (batchShape == null) {
    return out;
  } else {
    if (batchShape.length === 1) {
      return tile(expandDims(out, 0), [batchShape[0], 1, 1]);
    } else if (batchShape.length === 2) {
      // tslint:disable-next-line:no-unnecessary-type-assertion
      return tile(
          expandDims(expandDims(out, 0), 0),
          [batchShape[0], batchShape[1], 1, 1]);
    } else if (batchShape.length === 3) {
      // tslint:disable-next-line:no-unnecessary-type-assertion
      return tile(
          expandDims(expandDims(expandDims(out, 0), 0), 0),
          [batchShape[0], batchShape[1], batchShape[2], 1, 1]);
    } else {
      throw new Error(
          'eye() currently supports only 1D and 2D ' +
          (
              // tslint:disable-next-line:no-any
              'batchShapes, but received ' + batchShape.length + 'D.'));
    }
  }
}
var eye = op({eye_: eye_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function fill(shape, value, dtype) {
  var attrs = {shape: shape, value: value, dtype: dtype};
  return ENGINE.runKernel(Fill, {}, attrs);
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
 * Computes floor of input `tf.Tensor` element-wise: `floor(x)`.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.floor().print();  // or tf.floor(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function floor_(x) {
  var $x = convertToTensor(x, 'x', 'floor');
  var inputs = {x: $x};
  return ENGINE.runKernel(Floor, inputs);
}
var floor = op({floor_: floor_});

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
 * @param batchDims Optional. The number of batch dimensions. It must be less
 *     than or equal to rank(indices). Defaults to 0.
 *     The output tensor will have shape of
 *     `x.shape[:axis] + indices.shape[batchDims:] + x.shape[axis + 1:]`
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function gather_(x, indices, axis, batchDims) {
  if (axis === void 0) {
    axis = 0;
  }
  if (batchDims === void 0) {
    batchDims = 0;
  }
  var $x = convertToTensor(x, 'x', 'gather');
  var $indices = convertToTensor(indices, 'indices', 'gather', 'int32');
  var inputs = {x: $x, indices: $indices};
  var attrs = {axis: axis, batchDims: batchDims};
  return ENGINE.runKernel(GatherV2, inputs, attrs);
}
var gather = op({gather_: gather_});

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
/**
 * Returns the truth value of (a > b) element-wise. Supports broadcasting.
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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function greater_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'greater');
  var $b = convertToTensor(b, 'b', 'greater');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Greater, inputs);
}
var greater = op({greater_: greater_});

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
/**
 * Returns the truth value of (a >= b) element-wise. Supports broadcasting.
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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function greaterEqual_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'greaterEqual');
  var $b = convertToTensor(b, 'b', 'greaterEqual');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(GreaterEqual, inputs);
}
var greaterEqual = op({greaterEqual_: greaterEqual_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function imag_(input) {
  var $input = convertToTensor(input, 'input', 'imag');
  var inputs = {input: $input};
  return ENGINE.runKernel(Imag, inputs);
}
var imag = op({imag_: imag_});

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
 * Returns which elements of x are finite.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isFinite().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function isFinite_(x) {
  var $x = convertToTensor(x, 'x', 'isFinite');
  var inputs = {x: $x};
  return ENGINE.runKernel(IsFinite, inputs);
}
var isFinite$1 = op({isFinite_: isFinite_});

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
 * Returns which elements of x are Infinity or -Infinity.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isInf().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function isInf_(x) {
  var $x = convertToTensor(x, 'x', 'isInf');
  var inputs = {x: $x};
  return ENGINE.runKernel(IsInf, inputs);
}
var isInf = op({isInf_: isInf_});

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
 * RReturns which elements of x are NaN.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isNaN().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function isNaN_(x) {
  var $x = convertToTensor(x, 'x', 'isNaN');
  var inputs = {x: $x};
  return ENGINE.runKernel(IsNan, inputs);
}
var isNaN$1 = op({isNaN_: isNaN_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function leakyRelu_(x, alpha) {
  if (alpha === void 0) {
    alpha = 0.2;
  }
  var $x = convertToTensor(x, 'x', 'leakyRelu');
  var inputs = {x: $x};
  var attrs = {alpha: alpha};
  return ENGINE.runKernel(LeakyRelu, inputs, attrs);
}
var leakyRelu = op({leakyRelu_: leakyRelu_});

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
/**
 * Returns the truth value of (a < b) element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([2, 2, 2]);
 *
 * a.less(b).print();
 * ```
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function less_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'less');
  var $b = convertToTensor(b, 'b', 'less');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Less, inputs);
}
var less = op({less_: less_});

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
/**
 * Returns the truth value of (a <= b) element-wise. Supports broadcasting.
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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function lessEqual_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'lessEqual');
  var $b = convertToTensor(b, 'b', 'lessEqual');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(LessEqual, inputs);
}
var lessEqual = op({lessEqual_: lessEqual_});

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
 * Return an evenly spaced sequence of numbers over the given interval.
 *
 * ```js
 * tf.linspace(0, 9, 10).print();
 * ```
 * @param start The start value of the sequence.
 * @param stop The end value of the sequence.
 * @param num The number of values to generate.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function linspace(start, stop, num) {
  if (num <= 0) {
    throw new Error('The number of values should be positive.');
  }
  var attrs = {start: start, stop: stop, num: num};
  return ENGINE.runKernel(LinSpace, {}, attrs);
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
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function localResponseNormalization_(x, depthRadius, bias, alpha, beta) {
  if (depthRadius === void 0) {
    depthRadius = 5;
  }
  if (bias === void 0) {
    bias = 1;
  }
  if (alpha === void 0) {
    alpha = 1;
  }
  if (beta === void 0) {
    beta = 0.5;
  }
  var $x = convertToTensor(x, 'x', 'localResponseNormalization');
  assert($x.rank === 4 || $x.rank === 3, function() {
    return 'Error in localResponseNormalization: x must be rank 3 or 4 but got\n               rank ' +
        $x.rank + '.';
  });
  assert(isInt(depthRadius), function() {
    return 'Error in localResponseNormalization: depthRadius must be an ' +
        ('integer but got depthRadius ' + depthRadius + '.');
  });
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  var inputs = {x: x4D};
  var attrs = {depthRadius: depthRadius, bias: bias, alpha: alpha, beta: beta};
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(LRN, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  } else {
    return res;
  }
}
var localResponseNormalization =
    op({localResponseNormalization_: localResponseNormalization_});

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
 * Computes natural logarithm of the input `tf.Tensor` element-wise: `ln(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.E]);
 *
 * x.log().print();  // or tf.log(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function log_(x) {
  var $x = convertToTensor(x, 'x', 'log');
  var inputs = {x: $x};
  return ENGINE.runKernel(Log, inputs);
}
var log = op({log_: log_});

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
 * Computes natural logarithm of the input `tf.Tensor` plus one
 * element-wise: `ln(1 + x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.E - 1]);
 *
 * x.log1p().print();  // or tf.log1p(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function log1p_(x) {
  var $x = convertToTensor(x, 'x', 'log1p');
  var inputs = {x: $x};
  return ENGINE.runKernel(Log1p, inputs);
}
var log1p = op({log1p_: log1p_});

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
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function grad(f) {
  assert(isFunction(f), function() {
    return 'The f passed in grad(f) must be a function';
  });
  return function(x, dy) {
    // x can be of any dtype, thus null as the last argument.
    var $x = convertToTensor(x, 'x', 'tf.grad', 'string_or_numeric');
    var $dy = (dy != null) ? convertToTensor(dy, 'dy', 'tf.grad') : null;
    return ENGINE.tidy(function() {
      var _a = ENGINE.gradients(function() {
        return f($x);
      }, [$x], $dy), value = _a.value, grads = _a.grads;
      if ($dy != null) {
        assertShapesMatch(
            value.shape, $dy.shape,
            'The shape of dy passed in grad(f)(x, dy) must match the shape ' +
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
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function grads(f) {
  assert(isFunction(f), function() {
    return 'The f passed in grads(f) must be a function';
  });
  return function(args, dy) {
    assert(Array.isArray(args), function() {
      return 'The args passed in grads(f)(args) must be an array ' +
          'of `Tensor`s or `TensorLike`s';
    });
    // args can be of any dtype, thus null as the last argument.
    var $args =
        convertToTensorArray(args, 'args', 'tf.grads', 'string_or_numeric');
    var $dy = (dy != null) ? convertToTensor(dy, 'dy', 'tf.grads') : null;
    return ENGINE.tidy(function() {
      var _a = ENGINE.gradients(function() {
        return f.apply(void 0, $args);
      }, $args, $dy), value = _a.value, grads = _a.grads;
      if ($dy != null) {
        assertShapesMatch(
            value.shape, $dy.shape,
            'The shape of dy passed in grads(f)([x1,...], dy) must ' +
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
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function valueAndGrad(f) {
  assert(isFunction(f), function() {
    return 'The f passed in valueAndGrad(f) must be a function';
  });
  return function(x, dy) {
    assert(x instanceof Tensor, function() {
      return 'The x passed in valueAndGrad(f)(x) must be a tensor';
    });
    assert(dy == null || dy instanceof Tensor, function() {
      return 'The dy passed in valueAndGrad(f)(x, dy) must be a tensor';
    });
    var _a = ENGINE.gradients(function() {
      return f(x);
    }, [x], dy), grads = _a.grads, value = _a.value;
    checkGrads(grads);
    return {grad: grads[0], value: value};
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
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function valueAndGrads(f) {
  assert(isFunction(f), function() {
    return 'The f passed in valueAndGrads(f) must be a function';
  });
  return function(args, dy) {
    assert(
        Array.isArray(args) && args.every(function(arg) {
          return arg instanceof Tensor;
        }),
        function() {
          return 'The args passed in valueAndGrads(f)(args) must be array of ' +
              'tensors';
        });
    assert(dy == null || dy instanceof Tensor, function() {
      return 'The dy passed in valueAndGrads(f)(args, dy) must be a tensor';
    });
    var res = ENGINE.gradients(function() {
      return f.apply(void 0, args);
    }, args, dy);
    if (dy != null) {
      assertShapesMatch(
          res.value.shape, dy.shape,
          'The shape of dy passed in valueAndGrads(f)([x1,...], dy) must ' +
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
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function variableGrads(f, varList) {
  assert(isFunction(f), function() {
    return 'The f passed in variableGrads(f) must be a function';
  });
  assert(
      varList == null || Array.isArray(varList) && varList.every(function(v) {
        return v instanceof Variable;
      }),
      function() {
        return 'The varList passed in variableGrads(f, varList) must be an array ' +
            'of variables';
      });
  var specifiedVarList = varList != null;
  if (!specifiedVarList) {
    // Get all of the trainable variables.
    varList = [];
    for (var varName in ENGINE.registeredVariables) {
      varList.push(ENGINE.registeredVariables[varName]);
    }
  }
  var specifiedNonTrainable = specifiedVarList ?
      varList.filter(function(variable) {
        return !variable.trainable;
      }) :
      null;
  // Prune non-trainable variables.
  var originalVarCount = varList.length;
  varList = varList.filter(function(variable) {
    return variable.trainable;
  });
  assert(varList.length > 0, function() {
    return 'variableGrads() expects at least one of the input variables to ' +
        ('be trainable, but none of the ' + originalVarCount +
         ' variables is ') +
        'trainable.';
  });
  var allowNoGradients = true;
  var _a = ENGINE.gradients(f, varList, null, allowNoGradients),
      value = _a.value, grads = _a.grads;
  assert(
      grads.some(function(g) {
        return g != null;
      }),
      function() {
        return 'Cannot find a connection between any variable and the result of ' +
            'the loss function y=f(x). Please make sure the operations that ' +
            'use variables are inside the function f passed to minimize().';
      });
  assert(value.rank === 0, function() {
    return 'The f passed in variableGrads(f) must return a scalar, but it ' +
        ('returned a rank-' + value.rank + ' tensor');
  });
  var namedGrads = {};
  varList.forEach(function(v, i) {
    if (grads[i] != null) {
      namedGrads[v.name] = grads[i];
    }
  });
  if (specifiedNonTrainable != null) {
    // If varList is explicitly provided and contains non-trainable values,
    // add them to the returned gradients with `null` values.
    specifiedNonTrainable.forEach(function(v) {
      return namedGrads[v.name] = null;
    });
  }
  return {value: value, grads: namedGrads};
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
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
function customGrad(f) {
  return ENGINE.customGrad(f);
}
function checkGrads(grads) {
  var numNullGradients = grads
                             .filter(function(g) {
                               return g == null;
                             })
                             .length;
  if (numNullGradients > 0) {
    throw new Error(
        'Cannot compute gradient of y=f(x) with respect to x. Make sure that\n    the f you passed encloses all operations that lead from x to y.');
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
 * Computes `-1 * x` element-wise.
 *
 * ```js
 * const x = tf.tensor2d([1, 2, -2, 0], [2, 2]);
 *
 * x.neg().print();  // or tf.neg(x)
 * ```
 *
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function neg_(x) {
  var $x = convertToTensor(x, 'x', 'neg');
  var inputs = {x: $x};
  return ENGINE.runKernel(Neg, inputs);
}
var neg = op({neg_: neg_});

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
 * Computes softplus of the input `tf.Tensor` element-wise: `log(exp(x) + 1)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.softplus().print();  // or tf.softplus(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function softplus_(x) {
  var $x = convertToTensor(x, 'x', 'softplus');
  var inputs = {x: $x};
  return ENGINE.runKernel(Softplus, inputs);
}
var softplus = op({softplus_: softplus_});

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
 * Computes log sigmoid of the input `tf.Tensor` element-wise:
 * `logSigmoid(x)`. For numerical stability, we use `-tf.softplus(-x)`.
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.logSigmoid().print();  // or tf.logSigmoid(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function logSigmoid_(x) {
  var $x = convertToTensor(x, 'x', 'logSigmoid');
  // Use a custom gradient to maintain previous implementation.
  // There is no LogSigmoid kernel in TF so we can't use engine.runKernel
  // directly
  var customOp = customGrad(function(x) {
    // TODO(yassogba) we can remove the chained softplus call here only
    // after backends have modualrized softplus at which point we can call
    // engine runKernel(..., Sotfplus, ...) directly.
    var value = neg(softplus(neg(x)));
    var gradFunc = function(dy) {
      var derX = mul(dy, sigmoid(neg(x)));
      return derX;
    };
    return {value: value, gradFunc: gradFunc};
  });
  return customOp($x);
}
var logSigmoid = op({logSigmoid_: logSigmoid_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function max_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  var $x = convertToTensor(x, 'x', 'max');
  var inputs = {x: $x};
  var attrs = {reductionIndices: axis, keepDims: keepDims};
  return ENGINE.runKernel(Max, inputs, attrs);
}
var max = op({max_: max_});

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
/**
 * Subtracts two `tf.Tensor`s element-wise, A - B. Supports broadcasting.
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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function sub_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'sub');
  var $b = convertToTensor(b, 'b', 'sub');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Sub, inputs);
}
var sub = op({sub_: sub_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function sum_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  var $x = convertToTensor(x, 'x', 'sum');
  if ($x.dtype === 'bool') {
    $x = cast($x, 'int32');
  }
  var inputs = {x: $x};
  var attrs = {axis: axis, keepDims: keepDims};
  return ENGINE.runKernel(Sum, inputs, attrs);
}
var sum$1 = op({sum_: sum_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function logSoftmax_(logits, axis) {
  if (axis === void 0) {
    axis = -1;
  }
  var $logits = convertToTensor(logits, 'logits', 'logSoftmax');
  if (axis === -1) {
    axis = $logits.rank - 1;
  }
  if (axis !== $logits.rank - 1) {
    throw Error(
        'Log Softmax along a non-last dimension is not yet supported. ' +
        ('Logits was rank ' + $logits.rank + ' and axis was ' + axis));
  }
  // const forward: ForwardFunc<Tensor> = (backend, save) => {
  //   const keepDims = true;
  //   const xMax = max(logits, axis, true);
  //   const shifted = sub(logits, xMax);
  //   const value =
  //       sub(cast(shifted, 'float32'), log(sum(exp(shifted), axis,
  //       keepDims)));
  //   save([value]);
  //   return value;
  // };
  // Use a custom gradient for numerical stability.
  var customOp = customGrad(function(logits, save) {
    var keepDims = true;
    var xMax = max(logits, axis, true);
    var shifted = sub(logits, xMax);
    var value =
        sub(cast(shifted, 'float32'), log(sum$1(exp(shifted), axis, keepDims)));
    save([value]);
    var gradFunc = function(dy, saved) {
      var value = saved[0];
      var keepDims = true;
      var softmax = exp(value);
      return sub(dy, mul(sum$1(dy, axis, keepDims), softmax));
    };
    return {value: value, gradFunc: gradFunc};
  });
  return customOp($logits);
  // TODO Use Engine.runKernel when CPU/WebGL/WASM backends implement this.
  // const inputs: LogSoftmaxInputs = {logits: $logits};
  // const attrs: LogSoftmaxAttrs = {axis};
  // return ENGINE.runKernel(
  //            LogSoftmax, inputs as {} as NamedTensorMap,
  //            attrs as {} as NamedAttrMap);
}
var logSoftmax = op({logSoftmax_: logSoftmax_});

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
  for (var i = 0; i < axes.length; ++i) {
    if (axes[axes.length - i - 1] !== rank - 1 - i) {
      return false;
    }
  }
  return true;
}
function combineLocations(outputLoc, reduceLoc, axes) {
  var rank = outputLoc.length + reduceLoc.length;
  var loc = [];
  var outIdx = 0;
  var reduceIdx = 0;
  for (var dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) === -1) {
      loc.push(outputLoc[outIdx++]);
    } else {
      loc.push(reduceLoc[reduceIdx++]);
    }
  }
  return loc;
}
function computeOutAndReduceShapes(aShape, axes) {
  var outShape = [];
  var rank = aShape.length;
  for (var dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) === -1) {
      outShape.push(aShape[dim]);
    }
  }
  var reduceShape = axes.map(function(dim) {
    return aShape[dim];
  });
  return [outShape, reduceShape];
}
function expandShapeToKeepDim(shape, axes) {
  var reduceSubShape = axes.map(function(x) {
    return 1;
  });
  return combineLocations(shape, reduceSubShape, axes);
}
function assertAxesAreInnerMostDims(msg, axes, rank) {
  assert(axesAreInnerMostDims(axes, rank), function() {
    return msg + ' supports only inner-most axes for now. ' +
        ('Got axes ' + axes + ' and rank-' + rank + ' input.');
  });
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
  var result = [];
  for (var i = 0; i < rank; ++i) {
    if (axes.indexOf(i) === -1) {
      result.push(i);
    }
  }
  axes.forEach(function(axis) {
    return result.push(axis);
  });
  return result;
}
/** Returns the axes permutation that undoes the original permutation. */
function getUndoAxesPermutation(axes) {
  return axes
      .map(function(axis, i) {
        return [i, axis];
      })
      .sort(function(a, b) {
        return a[1] - b[1];
      })
      .map(function(x) {
        return x[0];
      });
}
function getInnerMostAxes(numAxes, rank) {
  var res = [];
  for (var i = rank - numAxes; i < rank; ++i) {
    res.push(i);
  }
  return res;
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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function logSumExp_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  var $x = convertToTensor(x, 'x', 'logSumExp');
  var axes = parseAxisParam(axis, $x.shape);
  var xMax = max($x, axes, true /* keepDims */);
  var a = sub($x, xMax);
  var b = exp(a);
  var c = sum$1(b, axes);
  var d = log(c);
  var res = add$1(reshape(xMax, d.shape), d);
  if (keepDims) {
    var newShape = expandShapeToKeepDim(res.shape, axes);
    return reshape(res, newShape);
  }
  return res;
}
var logSumExp = op({logSumExp_: logSumExp_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function logicalAnd_(a, b) {
  var $a = convertToTensor(a, 'a', 'logicalAnd', 'bool');
  var $b = convertToTensor(b, 'b', 'logicalAnd', 'bool');
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(LogicalAnd, inputs);
}
var logicalAnd = op({logicalAnd_: logicalAnd_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function logicalNot_(x) {
  var $x = convertToTensor(x, 'x', 'logicalNot', 'bool');
  var inputs = {x: $x};
  return ENGINE.runKernel(LogicalNot, inputs);
}
var logicalNot = op({logicalNot_: logicalNot_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function logicalOr_(a, b) {
  var $a = convertToTensor(a, 'a', 'logicalOr', 'bool');
  var $b = convertToTensor(b, 'b', 'logicalOr', 'bool');
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(LogicalOr, inputs);
}
var logicalOr = op({logicalOr_: logicalOr_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function logicalXor_(a, b) {
  var $a = convertToTensor(a, 'a', 'logicalXor', 'bool');
  var $b = convertToTensor(b, 'b', 'logicalXor', 'bool');
  assertAndGetBroadcastShape($a.shape, $b.shape);
  // x ^ y = (x | y) & ~(x & y)
  return logicalAnd(logicalOr(a, b), logicalNot(logicalAnd(a, b)));
}
var logicalXor = op({logicalXor_: logicalXor_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function maxPool_(x, filterSize, strides, pad, dimRoundingMode) {
  var $x = convertToTensor(x, 'x', 'maxPool');
  var dilations = 1;
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  assert(x4D.rank === 4, function() {
    return 'Error in maxPool: input must be rank 4 but got rank ' + x4D.rank +
        '.';
  });
  assert(eitherStridesOrDilationsAreOne(strides, dilations), function() {
    return 'Error in maxPool: Either strides or dilations must be 1. ' +
        ('Got strides ' + strides + ' and dilations \'' + dilations + '\'');
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in maxPool: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  var inputs = {x: x4D};
  var attrs = {
    filterSize: filterSize,
    strides: strides,
    pad: pad,
    dimRoundingMode: dimRoundingMode
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(MaxPool, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var maxPool = op({maxPool_: maxPool_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function maxPool3d_(x, filterSize, strides, pad, dimRoundingMode, dataFormat) {
  if (filterSize === void 0) {
    filterSize = [1, 1, 1];
  }
  if (dataFormat === void 0) {
    dataFormat = 'NDHWC';
  }
  var $x = convertToTensor(x, 'x', 'maxPool3d');
  var x5D = $x;
  var reshapedTo5D = false;
  if ($x.rank === 4) {
    reshapedTo5D = true;
    x5D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]);
  }
  assert(x5D.rank === 5, function() {
    return 'Error in maxPool3d: x must be rank 5 but got rank ' + x5D.rank +
        '.';
  });
  assert(dataFormat === 'NDHWC', function() {
    return 'Error in maxPool3d: Only NDHWC is currently supported, ' +
        ('but got dataFormat of ' + dataFormat);
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in maxPool3d: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  var inputs = {x: x5D};
  var attrs = {
    filterSize: filterSize,
    strides: strides,
    pad: pad,
    dimRoundingMode: dimRoundingMode,
    dataFormat: dataFormat
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(MaxPool3D, inputs, attrs);
  if (reshapedTo5D) {
    return reshape(
        res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
  }
  return res;
}
var maxPool3d = op({maxPool3d_: maxPool3d_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function maxPoolWithArgmax_(x, filterSize, strides, pad, includeBatchInIndex) {
  if (includeBatchInIndex === void 0) {
    includeBatchInIndex = false;
  }
  var $x = convertToTensor(x, 'x', 'maxPoolWithArgmax');
  var inputs = {x: $x};
  var attrs = {
    filterSize: filterSize,
    strides: strides,
    pad: pad,
    includeBatchInIndex: includeBatchInIndex
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var result = ENGINE.runKernel(MaxPoolWithArgmax, inputs, attrs);
  return {result: result[0], indexes: result[1]};
}
var maxPoolWithArgmax = op({maxPoolWithArgmax_: maxPoolWithArgmax_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function maximum_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'maximum');
  var $b = convertToTensor(b, 'b', 'maximum');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  if ($a.dtype === 'bool') {
    $a = cast($a, 'int32');
    $b = cast($b, 'int32');
  }
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Maximum, inputs);
}
var maximum = op({maximum_: maximum_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function mean_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  var $x = convertToTensor(x, 'x', 'mean');
  var inputs = {x: $x};
  var attrs = {axis: axis, keepDims: keepDims};
  return ENGINE.runKernel(Mean, inputs, attrs);
}
var mean = op({mean_: mean_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function min_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  var $x = convertToTensor(x, 'x', 'min');
  var inputs = {x: $x};
  var attrs = {axis: axis, keepDims: keepDims};
  // tslint:disable-next-line: no-unnecessary-type-assertion
  return ENGINE.runKernel(Min, inputs, attrs);
}
var min = op({min_: min_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function minimum_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'minimum');
  var $b = convertToTensor(b, 'b', 'minimum');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  if ($a.dtype === 'bool') {
    $a = cast($a, 'int32');
    $b = cast($b, 'int32');
  }
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Minimum, inputs);
}
var minimum = op({minimum_: minimum_});

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
/**
 * Pads a `tf.Tensor` using mirror padding.
 *
 * This operation implements the `REFLECT` and `SYMMETRIC` modes of pad.
 *
 * ```js
 * const x = tf.range(0, 9).reshape([1, 1, 3, 3]);
 * x.mirrorPad([[0, 0], [0, 0], [2, 2], [2, 2]], 'reflect').print();
 * ```
 * @param x The tensor to pad.
 * @param paddings An array of length `R` (the rank of the tensor), where
 * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
 * specifying how much to pad along each dimension of the tensor.
 * In "reflect" mode, the padded regions do not include the borders,
 * while in "symmetric" mode the padded regions do include the borders.
 * For example, if the input is `[1, 2, 3]` and paddings is `[0, 2]`,
 * then the output is `[1, 2, 3, 2, 1]` in "reflect" mode, and
 * `[1, 2, 3, 3, 2]` in "symmetric" mode.
 * If `mode` is "reflect" then both `paddings[D, 0]` and `paddings[D, 1]`
 * must be no greater than `x.shape[D] - 1`. If mode is "symmetric"
 * then both `paddings[D, 0]` and `paddings[D, 1]` must be no greater than
 * `x.shape[D]`
 * @param mode String to specify padding mode. Can be `'reflect' | 'symmetric'`
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function mirrorPad_(x, paddings, mode) {
  assert(mode === 'reflect' || mode === 'symmetric', function() {
    return 'Invalid mode. Mode must be either reflect or symmetric. ' +
        ('Got ' + mode + '.');
  });
  var $x = convertToTensor(x, 'x', 'mirrorPad');
  if ($x.rank === 0) {
    throw new Error(
        'mirrorPad(scalar) is not defined. ' +
        'Pass non-scalar to mirrorPad');
  }
  assert(paddings.length === $x.rank, function() {
    return 'Padding doesn\'t match input. Must be ' + $x.rank + '. ' +
        ('Got ' + paddings.length + '.');
  });
  var shapeOffset = mode === 'reflect' ? 1 : 0;
  var _loop_1 = function(i) {
    assert(paddings[i].length === 2, function() {
      return 'Invalid number of paddings. Must be length of 2 each.';
    });
    assert(
        paddings[i][0] >= 0 && paddings[i][0] <= $x.shape[i] - shapeOffset &&
            paddings[i][1] >= 0 && paddings[i][1] <= $x.shape[i] - shapeOffset,
        function() {
          return 'Padding in dimension ' + i +
              ' cannot be greater than or equal ' +
              ('to ' + ($x.shape[i] - shapeOffset) +
               ' or less than 0 for input of ') +
              ('shape ' + $x.shape);
        });
  };
  for (var i = 0; i < $x.rank; i++) {
    _loop_1(i);
  }
  var attrs = {paddings: paddings, mode: mode};
  var inputs = {x: $x};
  return ENGINE.runKernel(MirrorPad, inputs, attrs);
}
var mirrorPad = op({mirrorPad_: mirrorPad_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function mod_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'mod');
  var $b = convertToTensor(b, 'b', 'mod');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(Mod, inputs);
}
var mod = op({mod_: mod_});

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
 * Computes square of `x` element-wise: `x ^ 2`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.sqrt(2), -1]);
 *
 * x.square().print();  // or tf.square(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function square_(x) {
  var $x = convertToTensor(x, 'x', 'square');
  var attrs = {};
  return ENGINE.runKernel('Square', {x: $x}, attrs);
}
var square = op({square_: square_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function moments_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  x = convertToTensor(x, 'x', 'moments');
  var axes = parseAxisParam(axis, x.shape);
  var xMean = mean(x, axes, keepDims);
  var keepDimsShape = xMean.shape;
  if (!keepDims) {
    keepDimsShape = expandShapeToKeepDim(xMean.shape, axes);
  }
  var devSquared =
      square(sub(cast(x, 'float32'), reshape(xMean, keepDimsShape)));
  var variance = mean(devSquared, axes, keepDims);
  return {mean: xMean, variance: variance};
}
var moments = op({moments_: moments_});

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
 *
 * @doc {heading: 'Operations', subheading: 'RNN'}
 */
function multiRNNCell_(lstmCells, data, c, h) {
  var $data = convertToTensor(data, 'data', 'multiRNNCell');
  var $c = convertToTensorArray(c, 'c', 'multiRNNCell');
  var $h = convertToTensorArray(h, 'h', 'multiRNNCell');
  var input = $data;
  var newStates = [];
  for (var i = 0; i < lstmCells.length; i++) {
    var output = lstmCells[i](input, $c[i], $h[i]);
    newStates.push(output[0]);
    newStates.push(output[1]);
    input = output[1];
  }
  var newC = [];
  var newH = [];
  for (var i = 0; i < newStates.length; i += 2) {
    newC.push(newStates[i]);
    newH.push(newStates[i + 1]);
  }
  return [newC, newH];
}
var multiRNNCell = op({multiRNNCell_: multiRNNCell_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function multinomial_(logits, numSamples, seed, normalized) {
  if (normalized === void 0) {
    normalized = false;
  }
  var $logits = convertToTensor(logits, 'logits', 'multinomial');
  var numOutcomes = $logits.size;
  var origRank = $logits.rank;
  if (numOutcomes < 2) {
    throw new Error(
        'Error in multinomial: you need at least 2 outcomes, but got ' +
        (numOutcomes + '.'));
  }
  if (origRank > 2) {
    throw new Error('Rank of probabilities must be 1 or 2, but is ' + origRank);
  }
  // TODO(lina128): Investigate correct seed behavior. The code seems not allow
  // setting see to 0.
  seed = seed || Math.random();
  // The kernel only accepts (and returns) rank 2 tensors.
  var logits2D = origRank === 1 ? reshape($logits, [1, -1]) : $logits;
  var inputs = {logits: logits2D};
  var attrs = {numSamples: numSamples, seed: seed, normalized: normalized};
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(Multinomial, inputs, attrs);
  // tslint:disable-next-line:no-unnecessary-type-assertion
  return origRank === 1 ? reshape(res, [res.size]) : res;
}
var multinomial = op({multinomial_: multinomial_});

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
/**
 * Returns the truth value of (a != b) element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([0, 2, 3]);
 *
 * a.notEqual(b).print();
 * ```
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function notEqual_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'notEqual');
  var $b = convertToTensor(b, 'b', 'notEqual');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  return ENGINE.runKernel(NotEqual, inputs);
}
var notEqual = op({notEqual_: notEqual_});

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
 * Creates a `tf.Tensor` with all elements set to 0.
 *
 * ```js
 * tf.zeros([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Can
 *     be 'float32', 'int32' or 'bool'. Defaults to 'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function zeros(shape, dtype) {
  if (dtype === void 0) {
    dtype = 'float32';
  }
  if (dtype === 'complex64') {
    var real = zeros(shape, 'float32');
    var imag = zeros(shape, 'float32');
    return complex(real, imag);
  }
  var values = makeZerosTypedArray(sizeFromShape(shape), dtype);
  return ENGINE.makeTensor(values, shape, dtype);
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
 * Creates a `tf.Tensor` with all elements set to 1.
 *
 * ```js
 * tf.ones([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Defaults to
 *     'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function ones$1(shape, dtype) {
  if (dtype === void 0) {
    dtype = 'float32';
  }
  if (dtype === 'complex64') {
    var real = ones$1(shape, 'float32');
    var imag = zeros(shape, 'float32');
    return complex(real, imag);
  }
  var values = makeOnesTypedArray(sizeFromShape(shape), dtype);
  return ENGINE.makeTensor(values, shape, dtype);
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
 * Creates a `tf.Tensor` with all elements set to 1 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.onesLike(x).print();
 * ```
 * @param x A tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function onesLike_(x) {
  var $x = convertToTensor(x, 'x', 'onesLike');
  var inputs = {x: $x};
  return ENGINE.runKernel(OnesLike, inputs);
}
var onesLike = op({onesLike_: onesLike_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function outerProduct_(v1, v2) {
  var $v1 = convertToTensor(v1, 'v1', 'outerProduct');
  var $v2 = convertToTensor(v2, 'v2', 'outerProduct');
  assert($v1.rank === 1 && $v2.rank === 1, function() {
    return 'Error in outerProduct: inputs must be rank 1, but got ranks ' +
        ($v1.rank + ' and ' + $v2.rank + '.');
  });
  var v12D = reshape($v1, [-1, 1]);
  var v22D = reshape($v2, [1, -1]);
  return matMul(v12D, v22D);
}
var outerProduct = op({outerProduct_: outerProduct_});

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
/**
 * Pads a `tf.Tensor` with a given value and paddings.
 *
 * This operation implements `CONSTANT` mode. For `REFLECT` and `SYMMETRIC`,
 * refer to `tf.mirrorPad`
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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function pad_(x, paddings, constantValue) {
  if (constantValue === void 0) {
    constantValue = 0;
  }
  var $x = convertToTensor(x, 'x', 'pad');
  if ($x.rank === 0) {
    throw new Error('pad(scalar) is not defined. Pass non-scalar to pad');
  }
  var attrs = {paddings: paddings, constantValue: constantValue};
  var inputs = {x: $x};
  return ENGINE.runKernel(PadV2, inputs, attrs);
}
var pad = op({pad_: pad_});

/**
 * Pads a `tf.Tensor1D` with a given value and paddings. See `pad` for details.
 */
function pad1d_(x, paddings, constantValue) {
  if (constantValue === void 0) {
    constantValue = 0;
  }
  assert(paddings.length === 2, function() {
    return 'Invalid number of paddings. Must be length of 2.';
  });
  return pad(x, [paddings], constantValue);
}
var pad1d = op({pad1d_: pad1d_});

/**
 * Pads a `tf.Tensor2D` with a given value and paddings. See `pad` for details.
 */
function pad2d_(x, paddings, constantValue) {
  if (constantValue === void 0) {
    constantValue = 0;
  }
  assert(
      paddings.length === 2 && paddings[0].length === 2 &&
          paddings[1].length === 2,
      function() {
        return 'Invalid number of paddings. Must be length of 2 each.';
      });
  return pad(x, paddings, constantValue);
}
var pad2d = op({pad2d_: pad2d_});

/**
 * Pads a `tf.Tensor3D` with a given value and paddings. See `pad` for details.
 */
function pad3d_(x, paddings, constantValue) {
  if (constantValue === void 0) {
    constantValue = 0;
  }
  assert(
      paddings.length === 3 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2,
      function() {
        return 'Invalid number of paddings. Must be length of 2 each.';
      });
  return pad(x, paddings, constantValue);
}
var pad3d = op({pad3d_: pad3d_});

/**
 * Pads a `tf.Tensor4D` with a given value and paddings. See `pad` for details.
 */
function pad4d_(x, paddings, constantValue) {
  if (constantValue === void 0) {
    constantValue = 0;
  }
  assert(
      paddings.length === 4 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2 &&
          paddings[3].length === 2,
      function() {
        return 'Invalid number of paddings. Must be length of 2 each.';
      });
  return pad(x, paddings, constantValue);
}
var pad4d = op({pad4d_: pad4d_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function spaceToBatchND_(x, blockShape, paddings) {
  var $x = convertToTensor(x, 'x', 'spaceToBatchND');
  assert($x.rank >= 1 + blockShape.length, function() {
    return 'input rank ' + $x.rank + ' should be > than [blockShape] ' +
        blockShape.length;
  });
  assert(paddings.length === blockShape.length, function() {
    return 'paddings.shape[0] ' + paddings.length +
        ' must be equal to [blockShape] ' + blockShape.length;
  });
  assert($x.shape.reduce(function(a, b, i) {
    if (i > 0 && i <= blockShape.length) {
      return a &&
          ((b + paddings[i - 1][0] + paddings[i - 1][1]) % blockShape[i - 1] ===
           0);
    }
    return a;
  }, true), function() {
    return 'input spatial dimensions ' + $x.shape.slice(1) + ' with paddings ' +
        paddings.toString() + ' must be divisible by blockShapes ' +
        blockShape.toString();
  });
  var inputs = {x: $x};
  var attrs = {blockShape: blockShape, paddings: paddings};
  return ENGINE.runKernel(SpaceToBatchND, inputs, attrs);
}
var spaceToBatchND = op({spaceToBatchND_: spaceToBatchND_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
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
  var $x = convertToTensor(input, 'x', 'maxPool');
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  assert(eitherStridesOrDilationsAreOne(strides, dilations), function() {
    return 'Error in pool: Either strides or dilations must be 1. ' +
        ('Got strides ' + strides + ' and dilations \'' + dilations + '\'');
  });
  var convInfo =
      computePool2DInfo(x4D.shape, windowShape, strides, dilations, pad);
  var dilation = [convInfo.dilationHeight, convInfo.dilationWidth];
  // The following implementation does batchToSpace(pool(spaceToBatch(x)))
  // whenever dilation > 1 since the TF kernels do not support dilation > 1.
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L1037
  var basePadding;
  if (pad === 'same') {
    basePadding = withSpaceToBatchBasePaddings(
        [convInfo.filterHeight, convInfo.filterWidth], dilation);
  } else {
    basePadding = [[0, 0], [0, 0]];
  }
  var isDilationOne = dilation[0] === 1 && dilation[1] === 1;
  var _a = requiredSpaceToBatchPaddings(
          [convInfo.inHeight, convInfo.inWidth], dilation, basePadding),
      adjustedPadding = _a[0], adjustedCrops = _a[1];
  var convertedPad = isDilationOne ? pad : 'valid';
  var convertedX =
      isDilationOne ? x4D : spaceToBatchND(x4D, dilation, adjustedPadding);
  var forwardOp = poolingType === 'avg' ? function() {
    return avgPool(convertedX, windowShape, strides, convertedPad);
  } : function() {
    return maxPool(convertedX, windowShape, strides, convertedPad);
  };
  var y = forwardOp();
  var res = isDilationOne ? y : batchToSpaceND(y, dilation, adjustedCrops);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
// Helper function to compute crops and paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/array_ops.py#L2184
function requiredSpaceToBatchPaddings(inputShape, blockShape, basePadding) {
  var padStart = basePadding.map(function(b) {
    return b[0];
  });
  var origPadEnd = basePadding.map(function(b) {
    return b[1];
  });
  var fullInputShape = inputShape.concat(padStart, origPadEnd);
  var padEndExtra = blockShape.map(function(b, i) {
    return (b - fullInputShape[i] % b) % b;
  });
  var padEnd = origPadEnd.map(function(s, i) {
    return s + padEndExtra[i];
  });
  var paddings = blockShape.map(function(_, i) {
    return [padStart[i], padEnd[i]];
  });
  var crops = blockShape.map(function(_, i) {
    return [0, padEndExtra[i]];
  });
  return [paddings, crops];
}
// Helper function to compute base paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L524
function withSpaceToBatchBasePaddings(filterShape, dilation) {
  // Spatial dimensions of the filters and the upsampled filters in which we
  // introduce (rate - 1) zeros between consecutive filter values.
  var dilatedFilterShape = filterShape.map(function(s, i) {
    return s + (s - 1) * (dilation[i] - 1);
  });
  var padExtraShape = dilatedFilterShape.map(function(s) {
    return s - 1;
  });
  // When padding is odd, we pad more at end, following the same
  // convention as conv2d.
  var padExtraStart = padExtraShape.map(function(s) {
    return Math.floor(s / 2);
  });
  var padExtraEnd = padExtraShape.map(function(s, i) {
    return s - padExtraStart[i];
  });
  return padExtraShape.map(function(_, i) {
    return [padExtraStart[i], padExtraEnd[i]];
  });
}
var pool = op({pool_: pool_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function pow_(base, exp) {
  var _a;
  var $base = convertToTensor(base, 'base', 'pow');
  var $exp = convertToTensor(exp, 'exp', 'pow');
  _a = makeTypesMatch($base, $exp), $base = _a[0], $exp = _a[1];
  var inputs = {a: $base, b: $exp};
  return ENGINE.runKernel(Pow, inputs);
}
var pow = op({pow_: pow_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function prelu_(x, alpha) {
  var $x = convertToTensor(x, 'x', 'prelu');
  var $alpha = convertToTensor(alpha, 'alpha', 'prelu');
  var inputs = {x: $x, alpha: $alpha};
  return ENGINE.runKernel(Prelu, inputs);
}
var prelu = op({prelu_: prelu_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function prod_(x, axis, keepDims) {
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  var $x = convertToTensor(x, 'x', 'prod');
  if ($x.dtype === 'bool') {
    // bool is not an allowed type for the underlying kernel.
    $x = cast($x, 'int32');
  }
  var inputs = {x: $x};
  var attrs = {axis: axis, keepDims: keepDims};
  return ENGINE.runKernel(Prod, inputs, attrs);
}
var prod = op({prod_: prod_});

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
  var size = sizeFromShape(shape);
  var values = null;
  if (dtype == null || dtype === 'float32') {
    values = new Float32Array(size);
  } else if (dtype === 'int32') {
    values = new Int32Array(size);
  } else if (dtype === 'bool') {
    values = new Uint8Array(size);
  } else {
    throw new Error('Unknown data type ' + dtype);
  }
  for (var i = 0; i < size; i++) {
    values[i] = randFunction();
  }
  return ENGINE.makeTensor(values, shape, dtype);
}
var rand = op({rand_: rand_});

var commonjsGlobal = typeof globalThis !== 'undefined' ?
    globalThis :
    typeof window !== 'undefined' ?
    window :
    typeof global !== 'undefined' ? global :
                                    typeof self !== 'undefined' ? self : {};

function createCommonjsModule(fn, module) {
  return module = {exports: {}}, fn(module, module.exports), module.exports;
}

var alea = createCommonjsModule(function(module) {
  // A port of an algorithm by Johannes Baagøe <baagoe@baagoe.com>, 2010
  // http://baagoe.com/en/RandomMusings/javascript/
  // https://github.com/nquinlan/better-random-numbers-for-javascript-mirror
  // Original work is under MIT license -

  // Copyright (C) 2010 by Johannes Baagøe <baagoe@baagoe.org>
  //
  // Permission is hereby granted, free of charge, to any person obtaining a
  // copy of this software and associated documentation files (the "Software"),
  // to deal in the Software without restriction, including without limitation
  // the rights to use, copy, modify, merge, publish, distribute, sublicense,
  // and/or sell copies of the Software, and to permit persons to whom the
  // Software is furnished to do so, subject to the following conditions:
  //
  // The above copyright notice and this permission notice shall be included in
  // all copies or substantial portions of the Software.
  //
  // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  // FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  // DEALINGS IN THE SOFTWARE.



  (function(global, module, define) {
    function Alea(seed) {
      var me = this, mash = Mash();

      me.next = function() {
        var t = 2091639 * me.s0 + me.c * 2.3283064365386963e-10;  // 2^-32
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
      if (me.s0 < 0) {
        me.s0 += 1;
      }
      me.s1 -= mash(seed);
      if (me.s1 < 0) {
        me.s1 += 1;
      }
      me.s2 -= mash(seed);
      if (me.s2 < 0) {
        me.s2 += 1;
      }
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
      var xg = new Alea(seed), state = opts && opts.state, prng = xg.next;
      prng.int32 = function() {
        return (xg.next() * 0x100000000) | 0;
      };
      prng.double = function() {
        return prng() +
            (prng() * 0x200000 | 0) * 1.1102230246251565e-16;  // 2^-53
      };
      prng.quick = prng;
      if (state) {
        if (typeof (state) == 'object') copy(state, xg);
        prng.state = function() {
          return copy(xg, {});
        };
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
          n += h * 0x100000000;  // 2^32
        }
        return (n >>> 0) * 2.3283064365386963e-10;  // 2^-32
      };

      return mash;
    }


    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() {
        return impl;
      });
    } else {
      this.alea = impl;
    }
  })(
      commonjsGlobal,
      module,                           // present in node.js
      (typeof undefined) == 'function'  // present with an AMD loader
  );
});

var xor128 = createCommonjsModule(function(module) {
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
      var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
        return (xg.next() >>> 0) / 0x100000000;
      };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (typeof (state) == 'object') copy(state, xg);
        prng.state = function() {
          return copy(xg, {});
        };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() {
        return impl;
      });
    } else {
      this.xor128 = impl;
    }
  })(
      commonjsGlobal,
      module,                           // present in node.js
      (typeof undefined) == 'function'  // present with an AMD loader
  );
});

var xorwow = createCommonjsModule(function(module) {
  // A Javascript implementaion of the "xorwow" prng algorithm by
  // George Marsaglia.  See http://www.jstatsoft.org/v08/i14/paper

  (function(global, module, define) {
    function XorGen(seed) {
      var me = this, strseed = '';

      // Set up generator function.
      me.next = function() {
        var t = (me.x ^ (me.x >>> 2));
        me.x = me.y;
        me.y = me.z;
        me.z = me.w;
        me.w = me.v;
        return (me.d = (me.d + 362437 | 0)) +
            (me.v = (me.v ^ (me.v << 4)) ^ (t ^ (t << 1))) |
            0;
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
      var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
        return (xg.next() >>> 0) / 0x100000000;
      };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (typeof (state) == 'object') copy(state, xg);
        prng.state = function() {
          return copy(xg, {});
        };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() {
        return impl;
      });
    } else {
      this.xorwow = impl;
    }
  })(
      commonjsGlobal,
      module,                           // present in node.js
      (typeof undefined) == 'function'  // present with an AMD loader
  );
});

var xorshift7 = createCommonjsModule(function(module) {
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
        t = X[i];
        t ^= (t >>> 7);
        v = t ^ (t << 24);
        t = X[(i + 1) & 7];
        v ^= t ^ (t >>> 10);
        t = X[(i + 3) & 7];
        v ^= t ^ (t >>> 3);
        t = X[(i + 4) & 7];
        v ^= t ^ (t << 7);
        t = X[(i + 7) & 7];
        t = t ^ (t << 13);
        v ^= t ^ (t << 9);
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
            X[j & 7] =
                (X[j & 7] << 15) ^ (seed.charCodeAt(j) + X[(j + 1) & 7] << 13);
          }
        }
        // Enforce an array length of 8, not all zeroes.
        while (X.length < 8) X.push(0);
        for (j = 0; j < 8 && X[j] === 0; ++j)
          ;
        if (j == 8)
          w = X[7] = -1;
        else
          w = X[j];

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
      var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
        return (xg.next() >>> 0) / 0x100000000;
      };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (state.x) copy(state, xg);
        prng.state = function() {
          return copy(xg, {});
        };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() {
        return impl;
      });
    } else {
      this.xorshift7 = impl;
    }
  })(
      commonjsGlobal,
      module,                           // present in node.js
      (typeof undefined) == 'function'  // present with an AMD loader
  );
});

var xor4096 = createCommonjsModule(function(module) {
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
        var w = me.w, X = me.X, i = me.i, t, v;
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
        // Storing state as object members is faster than using closure
        // variables.
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
      var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
        return (xg.next() >>> 0) / 0x100000000;
      };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (state.X) copy(state, xg);
        prng.state = function() {
          return copy(xg, {});
        };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() {
        return impl;
      });
    } else {
      this.xor4096 = impl;
    }
  })(
      commonjsGlobal,                   // window object or global
      module,                           // present in node.js
      (typeof undefined) == 'function'  // present with an AMD loader
  );
});

var tychei = createCommonjsModule(function(module) {
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
      var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
        return (xg.next() >>> 0) / 0x100000000;
      };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (typeof (state) == 'object') copy(state, xg);
        prng.state = function() {
          return copy(xg, {});
        };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() {
        return impl;
      });
    } else {
      this.tychei = impl;
    }
  })(
      commonjsGlobal,
      module,                           // present in node.js
      (typeof undefined) == 'function'  // present with an AMD loader
  );
});

var seedrandom = createCommonjsModule(function(module) {
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

  (function(pool, math) {
    //
    // The following constants are related to IEEE 754 limits.
    //
    var global = this,
        width = 256,  // each RC4 output is 0 <= x < 256
        chunks = 6,   // at least six RC4 outputs for each double
        digits = 52,  // there are 52 significant digits in a double
        rngname =
            'random',  // rngname: name for Math.random and Math.seedrandom
        startdenom = math.pow(width, chunks),
        significance = math.pow(2, digits), overflow = significance * 2,
        mask = width - 1,
        nodecrypto;  // node.js crypto module, initialized at the bottom.

    //
    // seedrandom()
    // This is the seedrandom function described above.
    //
    function seedrandom(seed, options, callback) {
      var key = [];
      options = (options == true) ? {entropy: true} : (options || {});

      // Flatten the seed string or build one from local entropy if needed.
      var shortseed = mixkey(
          flatten(
              options.entropy ? [seed, tostring(pool)] :
                                (seed == null) ? autoseed() : seed,
              3),
          key);

      // Use the seed to initialize an ARC4 generator.
      var arc4 = new ARC4(key);

      // This function returns a random double in [0, 1) that contains
      // randomness in every bit of the mantissa of the IEEE 754 value.
      var prng = function() {
        var n = arc4.g(chunks),     // Start with a numerator n < 2 ^ 48
            d = startdenom,         //   and denominator d = 2 ^ 48.
            x = 0;                  //   and no 'extra last byte'.
        while (n < significance) {  // Fill up all significant digits by
          n = (n + x) * width;      //   shifting numerator and
          d *= width;               //   denominator and generating a
          x = arc4.g(1);            //   new least-significant-byte.
        }
        while (n >= overflow) {  // To avoid rounding up, before adding
          n /= 2;                //   last byte, shift everything
          d /= 2;                //   right using integer math until
          x >>>= 1;              //   we have exactly the desired bits.
        }
        return (n + x) / d;  // Form the number within [0, 1).
      };

      prng.int32 = function() {
        return arc4.g(4) | 0;
      };
      prng.quick = function() {
        return arc4.g(4) / 0x100000000;
      };
      prng.double = prng;

      // Mix the randomness into accumulated entropy.
      mixkey(tostring(arc4.S), pool);

      // Calling convention: what to return as a function of prng, seed,
      // is_math.
      return (
          options.pass || callback ||
          function(prng, seed, is_math_call, state) {
            if (state) {
              // Load the arc4 state from the given state if it has an S array.
              if (state.S) {
                copy(state, arc4);
              }
              // Only provide the .state method if requested via options.state.
              prng.state = function() {
                return copy(arc4, {});
              };
            }

            // If called as a method of Math (Math.seedrandom()), mutate
            // Math.random because that is how seedrandom.js has worked since
            // v1.0.
            if (is_math_call) {
              math[rngname] = prng;
              return seed;
            }

            // Otherwise, it is a newer calling convention, so return the
            // prng directly.
            else
              return prng;
          })(
          prng, shortseed,
          'global' in options ? options.global : (this == math), options.state);
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
      var t, keylen = key.length, me = this, i = 0, j = me.i = me.j = 0,
             s = me.S = [];

      // The empty key [] is treated as [0].
      if (!keylen) {
        key = [keylen++];
      }

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
        var t, r = 0, i = me.i, j = me.j, s = me.S;
        while (count--) {
          t = s[i = mask & (i + 1)];
          r = r * width +
              s[mask & ((s[i] = s[j = mask & (j + t)]) + (s[j] = t))];
        }
        me.i = i;
        me.j = j;
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
          try {
            result.push(flatten(obj[prop], depth - 1));
          } catch (e) {
          }
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
        var browser = global.navigator, plugins = browser && browser.plugins;
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
    if (module.exports) {
      module.exports = seedrandom;
      // When in node.js, try using crypto package for autoseeding.
      try {
        nodecrypto = require('crypto');
      } catch (ex) {
      }
    }

    // End anonymous scope, and pass initial values.
  })(
      [],   // pool: entropy pool starts empty
      Math  // math: package containing random, pow, and seedrandom
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
// https://en.wikipedia.org/wiki/Marsaglia_polar_method
var MPRandGauss = /** @class */ (function() {
  function MPRandGauss(mean, stdDeviation, dtype, truncated, seed) {
    this.mean = mean;
    this.stdDev = stdDeviation;
    this.dtype = dtype;
    this.nextVal = NaN;
    this.truncated = truncated;
    if (this.truncated) {
      this.upper = this.mean + this.stdDev * 2;
      this.lower = this.mean - this.stdDev * 2;
    }
    var seedValue = seed ? seed : Math.random();
    this.random = seedrandom_1(seedValue.toString());
  }
  /** Returns next sample from a Gaussian distribution. */
  MPRandGauss.prototype.nextValue = function() {
    if (!isNaN(this.nextVal)) {
      var value = this.nextVal;
      this.nextVal = NaN;
      return value;
    }
    var resultX, resultY;
    var isValid = false;
    while (!isValid) {
      var v1 = void 0, v2 = void 0, s = void 0;
      do {
        v1 = 2 * this.random() - 1;
        v2 = 2 * this.random() - 1;
        s = v1 * v1 + v2 * v2;
      } while (s >= 1 || s === 0);
      var mul = Math.sqrt(-2.0 * Math.log(s) / s);
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
  };
  /** Handles proper rounding for non-floating-point numbers. */
  MPRandGauss.prototype.convertValue = function(value) {
    if (this.dtype == null || this.dtype === 'float32') {
      return value;
    }
    return Math.round(value);
  };
  /** Returns true if less than 2-standard-deviations from the mean. */
  MPRandGauss.prototype.isValidTruncated = function(value) {
    return value <= this.upper && value >= this.lower;
  };
  return MPRandGauss;
}());
// Marsaglia, George, and Wai Wan Tsang. 2000. "A Simple Method for Generating
// Gamma Variables."
var RandGamma = /** @class */ (function() {
  function RandGamma(alpha, beta, dtype, seed) {
    this.alpha = alpha;
    this.beta = 1 / beta;  // convert rate to scale parameter
    this.dtype = dtype;
    var seedValue = seed ? seed : Math.random();
    this.randu = seedrandom_1(seedValue.toString());
    this.randn = new MPRandGauss(0, 1, dtype, false, this.randu());
    if (alpha < 1) {
      this.d = alpha + (2 / 3);
    } else {
      this.d = alpha - (1 / 3);
    }
    this.c = 1 / Math.sqrt(9 * this.d);
  }
  /** Returns next sample from a gamma distribution. */
  RandGamma.prototype.nextValue = function() {
    var x2, v0, v1, x, u, v;
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
  };
  /** Handles proper rounding for non-floating-point numbers. */
  RandGamma.prototype.convertValue = function(value) {
    if (this.dtype === 'float32') {
      return value;
    }
    return Math.round(value);
  };
  return RandGamma;
}());
var UniformRandom = /** @class */ (function() {
  function UniformRandom(min, max, dtype, seed) {
    var _this = this;
    if (min === void 0) {
      min = 0;
    }
    if (max === void 0) {
      max = 1;
    }
    /** Handles proper rounding for non floating point numbers. */
    this.canReturnFloat = function() {
      return (_this.dtype == null || _this.dtype === 'float32');
    };
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
      throw new Error(
          'The difference between ' + min + ' - ' + max +
          ' <= 1 and dtype is not float');
    }
    this.random = seedrandom_1(seed);
  }
  UniformRandom.prototype.convertValue = function(value) {
    if (this.canReturnFloat()) {
      return value;
    }
    return Math.round(value);
  };
  UniformRandom.prototype.nextValue = function() {
    return this.convertValue(this.min + this.range * this.random());
  };
  return UniformRandom;
}());

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
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomGamma_(shape, alpha, beta, dtype, seed) {
  if (beta === void 0) {
    beta = 1;
  }
  if (dtype === void 0) {
    dtype = 'float32';
  }
  if (beta == null) {
    beta = 1;
  }
  if (dtype == null) {
    dtype = 'float32';
  }
  if (dtype !== 'float32' && dtype !== 'int32') {
    throw new Error('Unsupported data type ' + dtype);
  }
  var rgamma = new RandGamma(alpha, beta, dtype, seed);
  var res = buffer(shape, dtype);
  for (var i = 0; i < res.values.length; i++) {
    res.values[i] = rgamma.nextValue();
  }
  return res.toTensor();
}
var randomGamma = op({randomGamma_: randomGamma_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomNormal_(shape, mean, stdDev, dtype, seed) {
  if (mean === void 0) {
    mean = 0;
  }
  if (stdDev === void 0) {
    stdDev = 1;
  }
  if (dtype != null && dtype === 'bool') {
    throw new Error('Unsupported data type ' + dtype);
  }
  var randGauss =
      new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
  var res = buffer(shape, dtype);
  for (var i = 0; i < res.values.length; i++) {
    res.values[i] = randGauss.nextValue();
  }
  return res.toTensor();
}
var randomNormal = op({randomNormal_: randomNormal_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomUniform_(shape, minval, maxval, dtype, seed) {
  if (minval === void 0) {
    minval = 0;
  }
  if (maxval === void 0) {
    maxval = 1;
  }
  if (dtype === void 0) {
    dtype = 'float32';
  }
  var res = buffer(shape, dtype);
  var random = new UniformRandom(minval, maxval, null, seed);
  for (var i = 0; i < res.values.length; i++) {
    res.values[i] = random.nextValue();
  }
  return res.toTensor();
}
var randomUniform = op({randomUniform_: randomUniform_});

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
 * Creates a new `tf.Tensor1D` filled with the numbers in the range provided.
 *
 * The tensor is a is half-open interval meaning it includes start, but
 * excludes stop. Decrementing ranges and negative step values are also
 * supported.sv
 *
 *
 * ```js
 * tf.range(0, 9, 2).print();
 * ```
 *
 * @param start An integer start value
 * @param stop An integer stop value
 * @param step An integer increment (will default to 1 or -1)
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function range(start, stop, step, dtype) {
  if (step === void 0) {
    step = 1;
  }
  if (dtype === void 0) {
    dtype = 'float32';
  }
  if (step === 0) {
    throw new Error('Cannot have a step of zero');
  }
  var attrs = {start: start, stop: stop, step: step, dtype: dtype};
  return ENGINE.runKernel(Range, {} /* inputs */, attrs);
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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function real_(input) {
  var $input = convertToTensor(input, 'input', 'real');
  var inputs = {input: $input};
  return ENGINE.runKernel(Real, inputs);
}
var real = op({real_: real_});

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
 * Computes reciprocal of x element-wise: `1 / x`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, 2]);
 *
 * x.reciprocal().print();  // or tf.reciprocal(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function reciprocal_(x) {
  var $x = convertToTensor(x, 'x', 'reciprocal');
  var inputs = {x: $x};
  return ENGINE.runKernel(Reciprocal, inputs);
}
var reciprocal = op({reciprocal_: reciprocal_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function relu_(x) {
  var $x = convertToTensor(x, 'x', 'relu');
  var inputs = {x: $x};
  return ENGINE.runKernel(Relu, inputs);
}
var relu = op({relu_: relu_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function relu6_(x) {
  var $x = convertToTensor(x, 'x', 'relu6');
  var inputs = {x: $x};
  return ENGINE.runKernel(Relu6, inputs);
}
var relu6 = op({relu6_: relu6_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function reverse_(x, axis) {
  var $x = convertToTensor(x, 'x', 'reverse');
  var inputs = {x: $x};
  var attrs = {dims: axis};
  return ENGINE.runKernel(Reverse, inputs, attrs);
}
var reverse = op({reverse_: reverse_});

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
/**
 * Reverses a `tf.Tensor1D`.
 *
 * @param x The input tensor.
 */
function reverse1d_(x) {
  var $x = convertToTensor(x, 'x', 'reverse');
  assert($x.rank === 1, function() {
    return 'Error in reverse1D: x must be rank 1 but got rank ' + $x.rank + '.';
  });
  return reverse($x, 0);
}
var reverse1d = op({reverse1d_: reverse1d_});

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
/**
 * Reverses a `tf.Tensor2D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse2d_(x, axis) {
  var $x = convertToTensor(x, 'x', 'reverse');
  assert($x.rank === 2, function() {
    return 'Error in reverse2D: x must be rank 2 but got rank ' + $x.rank + '.';
  });
  return reverse($x, axis);
}
var reverse2d = op({reverse2d_: reverse2d_});

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
/**
 * Reverses a `tf.Tensor3D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse3d_(x, axis) {
  var $x = convertToTensor(x, 'x', 'reverse');
  assert($x.rank === 3, function() {
    return 'Error in reverse3D: x must be rank 3 but got rank ' + $x.rank + '.';
  });
  return reverse($x, axis);
}
var reverse3d = op({reverse3d_: reverse3d_});

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
/**
 * Reverses a `tf.Tensor4D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse4d_(x, axis) {
  var $x = convertToTensor(x, 'x', 'reverse');
  assert($x.rank === 4, function() {
    return 'Error in reverse4D: x must be rank 4 but got rank ' + $x.rank + '.';
  });
  return reverse($x, axis);
}
var reverse4d = op({reverse4d_: reverse4d_});

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
 * Computes round of input `tf.Tensor` element-wise: `round(x)`.
 * It implements banker's rounding.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.round().print();  // or tf.round(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function round_(x) {
  var $x = convertToTensor(x, 'x', 'round');
  var inputs = {x: $x};
  return ENGINE.runKernel(Round, inputs);
}
var round$1 = op({round_: round_});

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
 * Computes reciprocal of square root of the input `tf.Tensor` element-wise:
 * `y = 1 / sqrt(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 4, -1]);
 *
 * x.rsqrt().print();  // or tf.rsqrt(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function rsqrt_(x) {
  var $x = convertToTensor(x, 'x', 'rsqrt');
  var inputs = {x: $x};
  return ENGINE.runKernel(Rsqrt, inputs);
}
var rsqrt = op({rsqrt_: rsqrt_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function scalar(value, dtype) {
  if (((isTypedArray(value) && dtype !== 'string') || Array.isArray(value)) &&
      dtype !== 'complex64') {
    throw new Error(
        'Error creating a new Scalar: value must be a primitive ' +
        '(number|boolean|string)');
  }
  if (dtype === 'string' && isTypedArray(value) &&
      !(value instanceof Uint8Array)) {
    throw new Error(
        'When making a scalar from encoded string, ' +
        'the value must be `Uint8Array`.');
  }
  var shape = [];
  var inferredShape = [];
  return makeTensor(value, shape, inferredShape, dtype);
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
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function selu_(x) {
  var $x = convertToTensor(x, 'x', 'selu');
  var inputs = {x: $x};
  return ENGINE.runKernel(Selu, inputs);
}
var selu = op({selu_: selu_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function separableConv2d_(
    x, depthwiseFilter, pointwiseFilter, strides, pad, dilation, dataFormat) {
  if (dilation === void 0) {
    dilation = [1, 1];
  }
  if (dataFormat === void 0) {
    dataFormat = 'NHWC';
  }
  var $x = convertToTensor(x, 'x', 'separableConv2d');
  var $depthwiseFilter =
      convertToTensor(depthwiseFilter, 'depthwiseFilter', 'separableConv2d');
  var $pointwiseFilter =
      convertToTensor(pointwiseFilter, 'pointwiseFilter', 'separableConv2d');
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  if (dataFormat === 'NCHW') {
    throw new Error(
        'separableConv2d currently does not support dataFormat NCHW; only ' +
        'NHWC is supported');
  }
  assert(x4D.rank === 4, function() {
    return 'Error in separableConv2d: input must be rank 4, but got ' +
        ('rank ' + x4D.rank + '.');
  });
  assert($depthwiseFilter.rank === 4, function() {
    return 'Error in separableConv2d: depthwise filter must be rank 4, but ' +
        ('got rank ' + $depthwiseFilter.rank + '.');
  });
  assert($pointwiseFilter.rank === 4, function() {
    return 'Error in separableConv2d: pointwise filter must be rank 4, but ' +
        ('got rank ' + $depthwiseFilter.rank + '.');
  });
  assert($pointwiseFilter.shape[0] === 1, function() {
    return 'Error in separableConv2d: the first dimension of pointwise filter ' +
        (' must be 1, but got ' + $pointwiseFilter.shape[0] + '.');
  });
  assert($pointwiseFilter.shape[1] === 1, function() {
    return 'Error in separableConv2d: the second dimension of pointwise ' +
        ('filter must be 1, but got ' + $pointwiseFilter.shape[1] + '.');
  });
  var inChannels = $depthwiseFilter.shape[2];
  var channelMultiplier = $depthwiseFilter.shape[3];
  assert($pointwiseFilter.shape[2] === inChannels * channelMultiplier, function() {
    return 'Error in separableConv2d: the third dimension of pointwise filter ' +
        ('must be ' + inChannels * channelMultiplier + ', ') +
        ('but got ' + $pointwiseFilter.shape[2] + '.');
  });
  var depthwise = depthwiseConv2d(
      x4D, $depthwiseFilter, strides, pad, dataFormat, dilation);
  var pointwiseStride = 1;
  var res =
      conv2d(depthwise, $pointwiseFilter, pointwiseStride, 'valid', dataFormat);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var separableConv2d = op({separableConv2d_: separableConv2d_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function setdiff1dAsync_(x, y) {
  return __awaiter(this, void 0, void 0, function() {
    var $x, $y, xVals, yVals, ySet, outputSize, i, buffer, indices, i, p;
    return __generator(this, function(_a) {
      switch (_a.label) {
        case 0:
          $x = convertToTensor(x, 'x', 'setdiff1d');
          $y = convertToTensor(y, 'y', 'setdiff1d');
          assert($x.dtype === $y.dtype, function() {
            return 'x and y should have the same dtype, but got x (' +
                $x.dtype + ') and y (' + $y.dtype + ').';
          });
          assert($x.rank === 1, function() {
            return 'x should be 1D tensor, but got x (' + $x.shape + ').';
          });
          assert($y.rank === 1, function() {
            return 'y should be 1D tensor, but got y (' + $y.shape + ').';
          });
          return [4 /*yield*/, $x.data()];
        case 1:
          xVals = _a.sent();
          return [4 /*yield*/, $y.data()];
        case 2:
          yVals = _a.sent();
          ySet = new Set(yVals);
          outputSize = 0;
          for (i = 0; i < xVals.length; i++) {
            if (!ySet.has(xVals[i])) {
              outputSize++;
            }
          }
          buffer = new TensorBuffer([outputSize], $x.dtype);
          indices = new TensorBuffer([outputSize], 'int32');
          for (i = 0, p = 0; i < xVals.length; i++) {
            if (!ySet.has(xVals[i])) {
              buffer.values[p] = xVals[i];
              indices.values[p] = i;
              p++;
            }
          }
          return [2 /*return*/, [buffer.toTensor(), indices.toTensor()]];
      }
    });
  });
}
var setdiff1dAsync = setdiff1dAsync_;

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
 * Returns an element-wise indication of the sign of a number.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3, NaN, 0]);
 *
 * x.sign().print();  // or tf.sign(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function sign_(x) {
  var $x = convertToTensor(x, 'x', 'sign');
  var inputs = {x: $x};
  return ENGINE.runKernel(Sign, inputs);
}
var sign = op({sign_: sign_});

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
 * Computes sin of the input Tensor element-wise: `sin(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.sin().print();  // or tf.sin(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function sin_(x) {
  var $x = convertToTensor(x, 'x', 'sin');
  var inputs = {x: $x};
  return ENGINE.runKernel(Sin, inputs);
}
var sin = op({sin_: sin_});

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
 * Computes hyperbolic sin of the input `tf.Tensor` element-wise: `sinh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.sinh().print();  // or tf.sinh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function sinh_(x) {
  var $x = convertToTensor(x, 'x', 'sinh');
  var inputs = {x: $x};
  return ENGINE.runKernel(Sinh, inputs);
}
var sinh = op({sinh_: sinh_});

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
 * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
 * of length `size`. See `slice` for details.
 */
function slice1d_(x, begin, size) {
  var $x = convertToTensor(x, 'x', 'slice1d');
  assert($x.rank === 1, function() {
    return 'slice1d expects a rank-1 tensor, but got a rank-' + $x.rank +
        ' tensor';
  });
  return slice($x, [begin], [size]);
}
var slice1d = op({slice1d_: slice1d_});

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
 * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice2d_(x, begin, size) {
  var $x = convertToTensor(x, 'x', 'slice2d');
  assert($x.rank === 2, function() {
    return 'slice2d expects a rank-2 tensor, but got a rank-' + $x.rank +
        ' tensor';
  });
  return slice($x, begin, size);
}
var slice2d = op({slice2d_: slice2d_});

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
 * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice3d_(x, begin, size) {
  var $x = convertToTensor(x, 'x', 'slice3d');
  assert($x.rank === 3, function() {
    return 'slice3d expects a rank-3 tensor, but got a rank-' + $x.rank +
        ' tensor';
  });
  return slice($x, begin, size);
}
var slice3d = op({slice3d_: slice3d_});

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
 * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice4d_(x, begin, size) {
  var $x = convertToTensor(x, 'x', 'slice4d');
  assert($x.rank === 4, function() {
    return 'slice4d expects a rank-4 tensor, but got a rank-' + $x.rank +
        ' tensor';
  });
  return slice($x, begin, size);
}
var slice4d = op({slice4d_: slice4d_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function softmax_(logits, dim) {
  if (dim === void 0) {
    dim = -1;
  }
  var $logits = convertToTensor(logits, 'logits', 'softmax', 'float32');
  if (dim === -1) {
    dim = $logits.rank - 1;
  }
  if (dim !== $logits.rank - 1) {
    throw Error(
        'Softmax along a non-last dimension is not yet supported. ' +
        ('Logits was rank ' + $logits.rank + ' and dim was ' + dim));
  }
  var inputs = {logits: $logits};
  var attrs = {dim: dim};
  return ENGINE.runKernel(Softmax, inputs, attrs);
}
var softmax = op({softmax_: softmax_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function fft_(input) {
  assert(input.dtype === 'complex64', function() {
    return 'The dtype for tf.spectral.fft() must be complex64 ' +
        ('but got ' + input.dtype + '.');
  });
  var inputs = {input: input};
  return ENGINE.runKernel(FFT, inputs);
}
var fft = op({fft_: fft_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function ifft_(input) {
  assert(input.dtype === 'complex64', function() {
    return 'The dtype for tf.spectral.ifft() must be complex64 ' +
        ('but got ' + input.dtype + '.');
  });
  var inputs = {input: input};
  return ENGINE.runKernel(IFFT, inputs);
}
var ifft = op({ifft_: ifft_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function irfft_(input) {
  var innerDimensionSize = input.shape[input.shape.length - 1];
  var batch = input.size / innerDimensionSize;
  var ret;
  if (innerDimensionSize <= 2) {
    var complexInput = reshape(input, [batch, innerDimensionSize]);
    ret = ifft(complexInput);
  } else {
    // The length of unique components of the DFT of a real-valued signal
    // is 2 * (input_len - 1)
    var outputShape = [batch, 2 * (innerDimensionSize - 1)];
    var realInput = reshape(real(input), [batch, innerDimensionSize]);
    var imagInput = reshape(imag(input), [batch, innerDimensionSize]);
    var realConjugate =
        reverse(slice(realInput, [0, 1], [batch, innerDimensionSize - 2]), 1);
    var imagConjugate = mul(
        reverse(slice(imagInput, [0, 1], [batch, innerDimensionSize - 2]), 1),
        scalar(-1));
    var r = concat([realInput, realConjugate], 1);
    var i = concat([imagInput, imagConjugate], 1);
    var complexInput = reshape(complex(r, i), [outputShape[0], outputShape[1]]);
    ret = ifft(complexInput);
  }
  ret = real(ret);
  // reshape the result if the input is 3D tensor.
  if (input.rank === 3 && input.shape[0] !== 0) {
    var temp = ret;
    var batch_1 = input.shape[0];
    ret = reshape(ret, [batch_1, ret.shape[0] / batch_1, ret.shape[1]]);
    temp.dispose();
  }
  return ret;
}
var irfft = op({irfft_: irfft_});

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
 * Can contain one -1 indicating that dimension is to be inferred.
 * @param axis The dimension along which to split. Defaults to 0 (the first
 * dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function split_(x, numOrSizeSplits, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  var $x = convertToTensor(x, 'x', 'split');
  var inputs = {x: $x};
  var attr = {numOrSizeSplits: numOrSizeSplits, axis: axis};
  return ENGINE.runKernel(SplitV, inputs, attr);
}
var split = op({split_: split_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function rfft_(input, fftLength) {
  assert(input.dtype === 'float32', function() {
    return 'The dtype for rfft() must be real value but got ' + input.dtype;
  });
  var innerDimensionSize = input.shape[input.shape.length - 1];
  var batch = input.size / innerDimensionSize;
  var adjustedInput;
  if (fftLength != null && fftLength < innerDimensionSize) {
    // Need to crop
    var begin = input.shape.map(function(v) {
      return 0;
    });
    var size = input.shape.map(function(v) {
      return v;
    });
    size[input.shape.length - 1] = fftLength;
    adjustedInput = slice(input, begin, size);
    innerDimensionSize = fftLength;
  } else if (fftLength != null && fftLength > innerDimensionSize) {
    // Need to pad with zeros
    var zerosShape = input.shape.map(function(v) {
      return v;
    });
    zerosShape[input.shape.length - 1] = fftLength - innerDimensionSize;
    adjustedInput = concat([input, zeros(zerosShape)], input.shape.length - 1);
    innerDimensionSize = fftLength;
  } else {
    adjustedInput = input;
  }
  // Complement the input with zero imaginary numbers.
  var zerosInput = zerosLike(adjustedInput);
  var complexInput =
      reshape(complex(adjustedInput, zerosInput), [batch, innerDimensionSize]);
  var ret = fft(complexInput);
  // Exclude complex conjugations. These conjugations are put symmetrically.
  var half = Math.floor(innerDimensionSize / 2) + 1;
  var realValues = real(ret);
  var imagValues = imag(ret);
  var realComplexConjugate = split(
      realValues, [half, innerDimensionSize - half],
      realValues.shape.length - 1);
  var imagComplexConjugate = split(
      imagValues, [half, innerDimensionSize - half],
      imagValues.shape.length - 1);
  var outputShape = adjustedInput.shape.slice();
  outputShape[adjustedInput.shape.length - 1] = half;
  return reshape(
      complex(realComplexConjugate[0], imagComplexConjugate[0]), outputShape);
}
var rfft = op({rfft_: rfft_});

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
 * Computes square root of the input `tf.Tensor` element-wise: `y = sqrt(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 4, -1]);
 *
 * x.sqrt().print();  // or tf.sqrt(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function sqrt_(x) {
  var $x = convertToTensor(x, 'x', 'sqrt');
  var inputs = {x: $x};
  return ENGINE.runKernel(Sqrt, inputs);
}
var sqrt = op({sqrt_: sqrt_});

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
/**
 * Returns (a - b) * (a - b) element-wise.
 * Supports broadcasting.
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
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function squaredDifference_(a, b) {
  var _a;
  var $a = convertToTensor(a, 'a', 'squaredDifference');
  var $b = convertToTensor(b, 'b', 'squaredDifference');
  _a = makeTypesMatch($a, $b), $a = _a[0], $b = _a[1];
  assertAndGetBroadcastShape($a.shape, $b.shape);
  var inputs = {a: $a, b: $b};
  var attrs = {};
  return ENGINE.runKernel(SquaredDifference, inputs, attrs);
}
var squaredDifference = op({squaredDifference_: squaredDifference_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function squeeze_(x, axis) {
  var $x = convertToTensor(x, 'x', 'squeeze');
  return reshape($x, squeezeShape($x.shape, axis).newShape);
}
var squeeze = op({squeeze_: squeeze_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function stack_(tensors, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  var $tensors =
      convertToTensorArray(tensors, 'tensors', 'stack', 'string_or_numeric');
  assert($tensors.length >= 1, function() {
    return 'Pass at least one tensor to tf.stack';
  });
  if ($tensors.length > 0) {
    assert(axis <= $tensors[0].rank, function() {
      return 'Axis must be <= rank of the tensor';
    });
  }
  var inputs = $tensors;
  var attrs = {axis: axis};
  return ENGINE.runKernel(Pack, inputs, attrs);
}
var stack = op({stack_: stack_});

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
 * Computes step of the input `tf.Tensor` element-wise: `x > 0 ? 1 : alpha * x`
 *
 * ```js
 * const x = tf.tensor1d([0, 2, -1, -3]);
 *
 * x.step(.5).print();  // or tf.step(x, .5)
 * ```
 * @param x The input tensor.
 * @param alpha The gradient when input is negative.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function step_(x, alpha) {
  if (alpha === void 0) {
    alpha = 0.0;
  }
  var $x = convertToTensor(x, 'x', 'step');
  var inputs = {x: $x};
  var attrs = {alpha: alpha};
  return ENGINE.runKernel(Step, inputs, attrs);
}
var step = op({step_: step_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function stridedSlice_(
    x, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask,
    shrinkAxisMask) {
  if (beginMask === void 0) {
    beginMask = 0;
  }
  if (endMask === void 0) {
    endMask = 0;
  }
  if (ellipsisMask === void 0) {
    ellipsisMask = 0;
  }
  if (newAxisMask === void 0) {
    newAxisMask = 0;
  }
  if (shrinkAxisMask === void 0) {
    shrinkAxisMask = 0;
  }
  var $x = convertToTensor(x, 'x', 'stridedSlice');
  var inputs = {x: $x};
  var attrs = {
    begin: begin,
    end: end,
    strides: strides,
    beginMask: beginMask,
    endMask: endMask,
    ellipsisMask: ellipsisMask,
    newAxisMask: newAxisMask,
    shrinkAxisMask: shrinkAxisMask
  };
  return ENGINE.runKernel(StridedSlice, inputs, attrs);
}
var stridedSlice = op({stridedSlice_: stridedSlice_});

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
 * Computes tan of the input `tf.Tensor` element-wise, `tan(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.tan().print();  // or tf.tan(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function tan_(x) {
  var $x = convertToTensor(x, 'x', 'tan');
  var inputs = {x: $x};
  return ENGINE.runKernel(Tan, inputs);
}
var tan = op({tan_: tan_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor1d(values, dtype) {
  assertNonNull(values);
  var inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 1) {
    throw new Error('tensor1d() requires values to be a flat/TypedArray');
  }
  var shape = null;
  return makeTensor(values, shape, inferredShape, dtype);
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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor2d(values, shape, dtype) {
  assertNonNull(values);
  if (shape != null && shape.length !== 2) {
    throw new Error('tensor2d() requires shape to have two numbers');
  }
  var inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 2 && inferredShape.length !== 1) {
    throw new Error(
        'tensor2d() requires values to be number[][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor2d() requires shape to be provided when `values` ' +
        'are a flat/TypedArray');
  }
  return makeTensor(values, shape, inferredShape, dtype);
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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor4d(values, shape, dtype) {
  assertNonNull(values);
  if (shape != null && shape.length !== 4) {
    throw new Error('tensor4d() requires shape to have four numbers');
  }
  var inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 4 && inferredShape.length !== 1) {
    throw new Error(
        'tensor4d() requires values to be number[][][][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor4d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  return makeTensor(values, shape, inferredShape, dtype);
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
 * Creates rank-5 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor5d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor5d([[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]).print();
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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor5d(values, shape, dtype) {
  assertNonNull(values);
  if (shape != null && shape.length !== 5) {
    throw new Error('tensor5d() requires shape to have five numbers');
  }
  var inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 5 && inferredShape.length !== 1) {
    throw new Error(
        'tensor5d() requires values to be ' +
        'number[][][][][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor5d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  return makeTensor(values, shape, inferredShape, dtype);
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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor6d(values, shape, dtype) {
  assertNonNull(values);
  if (shape != null && shape.length !== 6) {
    throw new Error('tensor6d() requires shape to have six numbers');
  }
  var inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 6 && inferredShape.length !== 1) {
    throw new Error(
        'tensor6d() requires values to be number[][][][][][] or ' +
        'flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor6d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  shape = shape || inferredShape;
  return makeTensor(values, shape, inferredShape, dtype);
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
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function topk_(x, k, sorted) {
  if (k === void 0) {
    k = 1;
  }
  if (sorted === void 0) {
    sorted = true;
  }
  var $x = convertToTensor(x, 'x', 'topk');
  if ($x.rank === 0) {
    throw new Error('topk() expects the input to be of rank 1 or higher');
  }
  var lastDim = $x.shape[$x.shape.length - 1];
  if (k > lastDim) {
    throw new Error(
        '\'k\' passed to topk() must be <= the last dimension (' + lastDim +
        ') ' + ('but got ' + k));
  }
  var inputs = {x: $x};
  var attrs = {k: k, sorted: sorted};
  var _a = ENGINE.runKernel(TopK, inputs, attrs), values = _a[0],
      indices = _a[1];
  return {values: values, indices: indices};
}
var topk = op({topk_: topk_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function truncatedNormal_(shape, mean, stdDev, dtype, seed) {
  if (mean === void 0) {
    mean = 0;
  }
  if (stdDev === void 0) {
    stdDev = 1;
  }
  if (dtype != null && dtype === 'bool') {
    throw new Error('Unsupported data type $ { dtype }');
  }
  var randGauss =
      new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
  var res = buffer(shape, dtype);
  for (var i = 0; i < res.values.length; i++) {
    res.values[i] = randGauss.nextValue();
  }
  return res.toTensor();
}
var truncatedNormal = op({truncatedNormal_: truncatedNormal_});

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
/**
 * Finds unique elements along an axis of a tensor.
 *
 * It returns a tensor `values` containing all of the unique elements along the
 * `axis` of the given tensor `x` in the same order that they occur along the
 * `axis` in `x`; `x` does not need to be sorted. It also returns a tensor
 * `indices` the same size as the number of the elements in `x` along the `axis`
 * dimension. It contains the index in the unique output `values`.
 *
 * ```js
 * // A 1-D tensor
 * const a = tf.tensor1d([1, 1, 2, 4, 4, 4, 7, 8, 8]);
 * const {values, indices} = tf.unique(a);
 * values.print();   // [1, 2, 4, 7, 8,]
 * indices.print();  // [0, 0, 1, 2, 2, 2, 3, 4, 4]
 * ```
 *
 * ```js
 * // A 2-D tensor with axis=0
 * //
 * // 'a' is: [[1, 0, 0],
 * //          [1, 0, 0],
 * //          [2, 0, 0]]
 * const a = tf.tensor2d([[1, 0, 0], [1, 0, 0], [2, 0, 0]]);
 * const {values, indices} = tf.unique(a, 0)
 * values.print();   // [[1, 0, 0],
 *                   //  [2, 0, 0]]
 * indices.print();  // [0, 0, 1]
 * ```
 *
 * ```js
 * // A 2-D tensor with axis=1
 * //
 * // 'a' is: [[1, 0, 0],
 * //          [1, 0, 0],
 * //          [2, 0, 0]]
 * const a = tf.tensor2d([[1, 0, 0], [1, 0, 0], [2, 0, 0]]);
 * const {values, indices} = tf.unique(a, 1)
 * values.print();   // [[1, 0],
 *                   //  [1, 0],
 *                   //  [2, 0]]
 * indices.print();  // [0, 1, 1]
 * ```
 * @param x A tensor (int32, string, bool).
 * @param axis The axis of the tensor to find the unique elements.
 * @returns [uniqueElements, indices] (see above for details)
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function unique_(x, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  var $x = convertToTensor(x, 'x', 'unique', 'string_or_numeric');
  assert($x.rank > 0, function() {
    return 'The input tensor must be at least 1D';
  });
  var inputs = {x: $x};
  var attrs = {axis: axis};
  var _a = ENGINE.runKernel(Unique, inputs, attrs), values = _a[0],
      indices = _a[1];
  return {values: values, indices: indices};
}
var unique = op({unique_: unique_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Segment'}
 */
function unsortedSegmentSum_(x, segmentIds, numSegments) {
  var $x = convertToTensor(x, 'x', 'unsortedSegmentSum');
  var $segmentIds =
      convertToTensor(segmentIds, 'segmentIds', 'unsortedSegmentSum', 'int32');
  assert(isInt(numSegments), function() {
    return 'numSegments must be of dtype int';
  });
  var inputs = {x: $x, segmentIds: $segmentIds};
  var attrs = {numSegments: numSegments};
  return ENGINE.runKernel(UnsortedSegmentSum, inputs, attrs);
}
var unsortedSegmentSum = op({unsortedSegmentSum_: unsortedSegmentSum_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function unstack_(x, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  var $x = convertToTensor(x, 'x', 'unstack', 'string_or_numeric');
  assert(axis >= -$x.shape.length && axis < $x.shape.length, function() {
    return 'Axis = ' + axis + ' is not in [-' + $x.shape.length + ', ' +
        $x.shape.length + ')';
  });
  var inputs = {value: $x};
  var attrs = {axis: axis};
  return ENGINE.runKernel(Unpack, inputs, attrs);
}
var unstack = op({unstack_: unstack_});

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
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function variable(initialValue, trainable, name, dtype) {
  if (trainable === void 0) {
    trainable = true;
  }
  return ENGINE.makeVariable(initialValue, trainable, name, dtype);
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
function whereImpl(condShape, condVals) {
  var indices = [];
  for (var i = 0; i < condVals.length; i++) {
    if (condVals[i]) {
      indices.push(i);
    }
  }
  var inBuffer = buffer(condShape, 'int32');
  var out = buffer([indices.length, condShape.length], 'int32');
  for (var i = 0; i < indices.length; i++) {
    var loc = inBuffer.indexToLoc(indices[i]);
    var offset = i * condShape.length;
    out.values.set(loc, offset);
  }
  return out.toTensor();
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
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function whereAsync_(condition) {
  return __awaiter(this, void 0, void 0, function() {
    var $condition, vals, res;
    return __generator(this, function(_a) {
      switch (_a.label) {
        case 0:
          $condition =
              convertToTensor(condition, 'condition', 'whereAsync', 'bool');
          return [4 /*yield*/, $condition.data()];
        case 1:
          vals = _a.sent();
          res = whereImpl($condition.shape, vals);
          if (condition !== $condition) {
            $condition.dispose();
          }
          return [2 /*return*/, res];
      }
    });
  });
}
var whereAsync = whereAsync_;

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
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function booleanMaskAsync_(tensor, mask, axis) {
  return __awaiter(this, void 0, void 0, function() {
    var $tensor, $mask, axisFrom, maskDim, tensorShape, leadingSize, i,
        targetTensorShape, reshapedTensor, reshapedMask, positivePositions,
        indices, res;
    return __generator(this, function(_a) {
      switch (_a.label) {
        case 0:
          $tensor = convertToTensor(tensor, 'tensor', 'boolMask');
          $mask = convertToTensor(mask, 'mask', 'boolMask', 'bool');
          axisFrom = axis == null ? 0 : axis;
          maskDim = $mask.rank;
          tensorShape = $tensor.shape;
          assert(maskDim > 0, function() {
            return 'mask cannot be scalar';
          });
          assertShapesMatch(
              tensorShape.slice(axisFrom, axisFrom + maskDim), $mask.shape,
              'mask\'s shape must match the first K dimensions of tensor\'s shape,');
          leadingSize = 1;
          for (i = axisFrom; i < axisFrom + maskDim; i++) {
            leadingSize *= tensorShape[i];
          }
          targetTensorShape =
              tensorShape.slice(0, axisFrom)
                  .concat([leadingSize], tensorShape.slice(axisFrom + maskDim));
          reshapedTensor = reshape($tensor, targetTensorShape);
          reshapedMask = reshape($mask, [-1]);
          return [4 /*yield*/, whereAsync(reshapedMask)];
        case 1:
          positivePositions = _a.sent();
          indices = squeeze(positivePositions, [1]);
          res = gather(reshapedTensor, indices, axisFrom);
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
          return [2 /*return*/, res];
      }
    });
  });
}
var booleanMaskAsync = booleanMaskAsync_;

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
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function norm_(x, ord, axis, keepDims) {
  if (ord === void 0) {
    ord = 'euclidean';
  }
  if (axis === void 0) {
    axis = null;
  }
  if (keepDims === void 0) {
    keepDims = false;
  }
  x = convertToTensor(x, 'x', 'norm');
  var norm = normImpl(x, ord, axis);
  var keepDimsShape = norm.shape;
  if (keepDims) {
    var axes = parseAxisParam(axis, x.shape);
    keepDimsShape = expandShapeToKeepDim(norm.shape, axes);
  }
  return reshape(norm, keepDimsShape);
}
function normImpl(x, p, axis) {
  if (axis === void 0) {
    axis = null;
  }
  if (x.rank === 0) {
    return abs(x);
  }
  // consider vector when no axis is specified
  if (x.rank !== 1 && axis === null) {
    return normImpl(reshape(x, [-1]), p, axis);
  }
  // vector
  if (x.rank === 1 || typeof axis === 'number' ||
      Array.isArray(axis) && axis.length === 1) {
    if (p === 1) {
      return sum$1(abs(x), axis);
    }
    if (p === Infinity) {
      return max(abs(x), axis);
    }
    if (p === -Infinity) {
      return min(abs(x), axis);
    }
    if (p === 'euclidean' || p === 2) {
      // norm(x, 2) = sum(abs(xi) ^ 2) ^ 1/2
      return sqrt(sum$1(pow(abs(x), scalar(2, 'int32')), axis));
    }
    throw new Error('Error in norm: invalid ord value: ' + p);
  }
  // matrix (assumption axis[0] < axis[1])
  if (Array.isArray(axis) && axis.length === 2) {
    if (p === 1) {
      return max(sum$1(abs(x), axis[0]), axis[1] - 1);
    }
    if (p === Infinity) {
      return max(sum$1(abs(x), axis[1]), axis[0]);
    }
    if (p === -Infinity) {
      return min(sum$1(abs(x), axis[1]), axis[0]);
    }
    if (p === 'fro' || p === 'euclidean') {
      // norm(x) = sqrt(sum(pow(x, 2)))
      return sqrt(sum$1(square(x), axis));
    }
    throw new Error('Error in norm: invalid ord value: ' + p);
  }
  throw new Error('Error in norm: invalid axis: ' + axis);
}
var norm = op({norm_: norm_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Moving Average'}
 */
function movingAverage_(v, x, decay, step, zeroDebias) {
  if (zeroDebias === void 0) {
    zeroDebias = true;
  }
  var $v = convertToTensor(v, 'v', 'movingAverage');
  var $x = convertToTensor(x, 'x', 'movingAverage');
  var $decay = convertToTensor(decay, 'decay', 'movingAverage');
  assertTypesMatch($v, $x);
  assert(arraysEqual($v.shape, $x.shape), function() {
    return 'Shape mismatch in v and x';
  });
  var one = scalar(1);
  var oneMinusDecay = sub(one, $decay);
  var update = mul(sub($x, $v), oneMinusDecay);
  if (zeroDebias) {
    assert(step != null, function() {
      return 'When using zeroDebias: true, step is required.';
    });
    var $step = convertToTensor(step, 'step', 'movingAverage');
    update = div(update, sub(one, pow($decay, $step)));
  }
  return add$1($v, update);
}
var movingAverage = op({movingAverage_: movingAverage_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function scatterND_(indices, updates, shape) {
  var $indices = convertToTensor(indices, 'indices', 'scatterND', 'int32');
  var $updates = convertToTensor(updates, 'updates', 'scatterND');
  validateInput($updates, $indices, shape);
  var inputs = {indices: $indices, updates: $updates};
  var attrs = {shape: shape};
  // tslint:disable-next-line: no-unnecessary-type-assertion
  return ENGINE.runKernel(ScatterNd, inputs, attrs);
}
var scatterND = op({scatterND_: scatterND_});

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
function validateInput$1(
    sparseIndices, sparseValues, outputShape, defaultValues) {
  if (sparseIndices.dtype !== 'int32') {
    throw new Error(
        'tf.sparseToDense() expects the indices to be int32 type,' +
        (' but the dtype was ' + sparseIndices.dtype + '.'));
  }
  if (sparseIndices.rank > 2) {
    throw new Error(
        'sparseIndices should be a scalar, vector, or matrix,' +
        (' but got shape ' + sparseIndices.shape + '.'));
  }
  var numElems = sparseIndices.rank > 0 ? sparseIndices.shape[0] : 1;
  var numDims = sparseIndices.rank > 1 ? sparseIndices.shape[1] : 1;
  if (outputShape.length !== numDims) {
    throw new Error(
        'outputShape has incorrect number of elements:,' +
        (' ' + outputShape.length + ', should be: ' + numDims + '.'));
  }
  var numValues = sparseValues.size;
  if (!(sparseValues.rank === 0 ||
        sparseValues.rank === 1 && numValues === numElems)) {
    throw new Error(
        'sparseValues has incorrect shape ' +
        (sparseValues.shape + ', should be [] or [' + numElems + ']'));
  }
  if (sparseValues.dtype !== defaultValues.dtype) {
    throw new Error('sparseValues.dtype must match defaultValues.dtype');
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
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function sparseToDense_(
    sparseIndices, sparseValues, outputShape, defaultValue) {
  if (defaultValue === void 0) {
    defaultValue = 0;
  }
  var $sparseIndices =
      convertToTensor(sparseIndices, 'sparseIndices', 'sparseToDense', 'int32');
  var $sparseValues =
      convertToTensor(sparseValues, 'sparseValues', 'sparseToDense');
  var $defaultValue = convertToTensor(
      defaultValue, 'defaultValue', 'sparseToDense', $sparseValues.dtype);
  validateInput$1($sparseIndices, $sparseValues, outputShape, $defaultValue);
  var inputs = {
    sparseIndices: $sparseIndices,
    sparseValues: $sparseValues,
    defaultValue: $defaultValue
  };
  var attrs = {outputShape: outputShape};
  return ENGINE.runKernel(SparseToDense, inputs, attrs);
}
var sparseToDense = op({sparseToDense_: sparseToDense_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function gatherND_(x, indices) {
  var $indices = convertToTensor(indices, 'indices', 'gatherND', 'int32');
  var $x = convertToTensor(x, 'x', 'gatherND');
  var inputs = {params: $x, indices: $indices};
  return ENGINE.runKernel(GatherNd, inputs);
}
var gatherND = op({gatherND_: gatherND_});

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
    var newDimension = [];
    for (var i = 0; i < x.shape.length; i++) {
      if (noiseShape[i] == null && x.shape[i] != null) {
        newDimension.push(x.shape[i]);
      } else {
        newDimension.push(noiseShape[i]);
      }
    }
    return newDimension;
  }
  return noiseShape;
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
 *
 * @doc {heading: 'Operations', subheading: 'Dropout'}
 */
function dropout_(x, rate, noiseShape, seed) {
  var $x = convertToTensor(x, 'x', 'dropout');
  assert($x.dtype === 'float32', function() {
    return 'x has to be a floating point tensor since it\'s going to be ' +
        ('scaled, but got a ' + $x.dtype + ' tensor instead.');
  });
  assert(rate >= 0 && rate < 1, function() {
    return 'rate must be a float in the range [0, 1), but got ' + rate + '.';
  });
  if (rate === 0) {
    return x instanceof Tensor ? $x.clone() : $x;
  }
  var $noiseShape = getNoiseShape($x, noiseShape);
  var keepProb = 1 - rate;
  var multiplier = div(
      floor(add$1(randomUniform($noiseShape, 0, 1, 'float32', seed), keepProb)),
      keepProb);
  return mul($x, multiplier);
}
var dropout = op({dropout_: dropout_});

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
function enclosingPowerOfTwo(value) {
  // Return 2**N for integer N such that 2**N >= value.
  return Math.floor(Math.pow(2, Math.ceil(Math.log(value) / Math.log(2.0))));
}
function cosineWindow(windowLength, a, b) {
  var even = 1 - windowLength % 2;
  var newValues = new Float32Array(windowLength);
  for (var i = 0; i < windowLength; ++i) {
    var cosArg = (2.0 * Math.PI * i) / (windowLength + even - 1);
    newValues[i] = a - b * Math.cos(cosArg);
  }
  return tensor1d(newValues, 'float32');
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
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function inTopKAsync_(predictions, targets, k) {
  if (k === void 0) {
    k = 1;
  }
  return __awaiter(this, void 0, void 0, function() {
    var $predictions, $targets, lastDim, predictionsVals, targetsVals, _a,
        batch, size, precision, b, offset, vals, valAndInd, i, i;
    return __generator(this, function(_b) {
      switch (_b.label) {
        case 0:
          $predictions = convertToTensor(predictions, 'predictions', 'inTopK');
          $targets = convertToTensor(targets, 'targets', 'inTopK');
          assert($predictions.rank > 1, function() {
            return 'inTopK() expects the predictions to be of rank 2 or higher, ' +
                ('but got ' + $predictions.rank);
          });
          assert($predictions.rank - 1 === $targets.rank, function() {
            return 'predictions rank should be 1 larger than ' +
                'targets rank, but got predictions rank ' +
                ($predictions.rank + ' and targets rank ' + $targets.rank);
          });
          assertShapesMatch(
              $predictions.shape.slice(0, $predictions.shape.length - 1),
              $targets.shape,
              'predictions\'s shape should be align with the targets\' shape, ' +
                  'except the last dimension.');
          lastDim = $predictions.shape[$predictions.shape.length - 1];
          assert(k > 0 && k <= lastDim, function() {
            return '\'k\' passed to inTopK() must be > 0 && <= the predictions last ' +
                ('dimension (' + lastDim + '), but got ' + k);
          });
          return [4 /*yield*/, $predictions.data()];
        case 1:
          predictionsVals = _b.sent();
          return [4 /*yield*/, $targets.data()];
        case 2:
          targetsVals = _b.sent();
          _a = [predictionsVals.length / lastDim, lastDim], batch = _a[0],
          size = _a[1];
          precision = getTypedArrayFromDType('bool', batch);
          for (b = 0; b < batch; b++) {
            offset = b * size;
            vals = predictionsVals.subarray(offset, offset + size);
            valAndInd = [];
            for (i = 0; i < vals.length; i++) {
              valAndInd.push({value: vals[i], index: i});
            }
            valAndInd.sort(function(a, b) {
              return b.value - a.value;
            });
            precision[b] = 0;
            for (i = 0; i < k; i++) {
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
          return [2 /*return*/, tensor(precision, $targets.shape, 'bool')];
      }
    });
  });
}
var inTopKAsync = inTopKAsync_;

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function conv2DBackpropFilter_(
    x, dy, filterShape, strides, pad, dataFormat, dimRoundingMode) {
  if (dataFormat === void 0) {
    dataFormat = 'NHWC';
  }
  var x4D = x;
  if (x.rank === 3) {
    x4D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]]);
  }
  var dy4D = dy;
  if (dy4D.rank === 3) {
    dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
  }
  assert(x4D.rank === 4, function() {
    return 'Error in conv2dDerFilter: input must be rank 4, but got shape ' +
        (x4D.shape + '.');
  });
  assert(dy4D.rank === 4, function() {
    return 'Error in conv2dDerFilter: dy must be rank 4, but got shape ' +
        (dy4D.shape + '.');
  });
  assert(filterShape.length === 4, function() {
    return 'Error in conv2dDerFilter: filterShape must be length 4, but got ' +
        (filterShape + '.');
  });
  var inDepth = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
  var outDepth = dataFormat === 'NHWC' ? dy4D.shape[3] : dy4D.shape[1];
  assert(inDepth === filterShape[2], function() {
    return 'Error in conv2dDerFilter: depth of input ' + inDepth + ') must ' +
        ('match input depth in filter (' + filterShape[2] + '.');
  });
  assert(outDepth === filterShape[3], function() {
    return 'Error in conv2dDerFilter: depth of dy (' + outDepth + ') must ' +
        ('match output depth for filter (' + filterShape[3] + ').');
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in conv2dDerFilter: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  var inputs = {x: x4D, dy: dy4D};
  var attrs = {
    strides: strides,
    pad: pad,
    dataFormat: dataFormat,
    dimRoundingMode: dimRoundingMode,
    filterShape: filterShape
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  return ENGINE.runKernel(Conv2DBackpropFilter, inputs, attrs);
}
var conv2DBackpropFilter = op({conv2DBackpropFilter_: conv2DBackpropFilter_});

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
function getFusedDyActivation(dy, y, activation) {
  if (activation == null || activation === 'linear') {
    return dy;
  }
  if (activation === 'relu') {
    return mul(dy, step(y));
  }
  throw new Error(
      'Cannot compute gradient for fused activation ' + activation + '.');
}
// Returns gradient for fused bias.
function getFusedBiasGradient(bias, dyActivation) {
  var res = dyActivation;
  var reduceAxes = getReductionAxes(bias.shape, dyActivation.shape);
  if (reduceAxes.length > 0) {
    res = sum$1(res, reduceAxes);
  }
  return reshape(res, bias.shape);
}
function applyActivation(
    x, activation, preluActivationWeights, leakyreluAlpha) {
  if (activation === 'linear') {
    return x;
  } else if (activation === 'relu') {
    return relu(x);
  } else if (activation === 'elu') {
    return elu(x);
  } else if (activation === 'relu6') {
    return relu6(x);
  } else if (activation === 'prelu') {
    return prelu(x, preluActivationWeights);
  } else if (activation === 'leakyrelu') {
    return leakyRelu(x, leakyreluAlpha);
  }
  throw new Error('Unknown fused activation ' + activation + '.');
}
// Whether we should call fused ops.
var shouldFuse = function(gradientDepth, activation) {
  var gradientMode = gradientDepth > 0;
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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`) to be
 *     applied
 *      after biasAdd.
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 * @param leakyreluAlpha Optional. Alpha to be applied as part of a `leakyrelu`
 *     activation.
 */
function fusedConv2d_(_a) {
  var x = _a.x, filter = _a.filter, strides = _a.strides, pad = _a.pad,
      _b = _a.dataFormat, dataFormat = _b === void 0 ? 'NHWC' : _b,
      _c = _a.dilations, dilations = _c === void 0 ? [1, 1] : _c,
      dimRoundingMode = _a.dimRoundingMode, bias = _a.bias, _d = _a.activation,
      activation = _d === void 0 ? 'linear' : _d,
      preluActivationWeights = _a.preluActivationWeights,
      leakyreluAlpha = _a.leakyreluAlpha;
  activation = activation || 'linear';
  if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
    var result =
        conv2d(x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
    if (bias != null) {
      result = add$1(result, bias);
    }
    return applyActivation(
        result, activation, preluActivationWeights, leakyreluAlpha);
  }
  var $x = convertToTensor(x, 'x', 'conv2d');
  var $filter = convertToTensor(filter, 'filter', 'conv2d');
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  assert(x4D.rank === 4, function() {
    return 'Error in fused conv2d: input must be rank 4, but got rank ' +
        (x4D.rank + '.');
  });
  assert($filter.rank === 4, function() {
    return 'Error in fused conv2d: filter must be rank 4, but got rank ' +
        ($filter.rank + '.');
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in fused conv2d: pad must be an integer when using, ' +
          ('dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad + '.');
    });
  }
  assert(x4D.shape[3] === $filter.shape[2], function() {
    return 'Error in conv2d: depth of input (' + x4D.shape[3] +
        ') must match ' + ('input depth for filter ' + $filter.shape[2] + '.');
  });
  assert(eitherStridesOrDilationsAreOne(strides, dilations), function() {
    return 'Error in conv2D: Either strides or dilations must be 1. ' +
        ('Got strides ' + strides + ' and dilations \'' + dilations + '\'');
  });
  assert(dataFormat === 'NHWC', function() {
    return 'Error in conv2d: got dataFormat of ' + dataFormat +
        ' but only NHWC is currently supported.';
  });
  var convInfo = computeConv2DInfo(
      x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode);
  var $bias;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused conv2d');
    $bias = makeTypesMatch($bias, $x)[0];
    assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
  }
  var $preluActivationWeights;
  if (preluActivationWeights != null) {
    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused conv2d');
  }
  var grad = function(dy, saved) {
    var _a = saved, $filter = _a[0], x4D = _a[1], y = _a[2], $bias = _a[3];
    var dyActivation = getFusedDyActivation(dy, y, activation);
    assert(tupleValuesAreOne(dilations), function() {
      return 'Error in gradient of fused conv2D: ' +
          'dilation rates greater than 1 ' +
          ('are not yet supported in gradients. Got dilations \'' + dilations +
           '\'');
    });
    var xDer =
        conv2DBackpropInput(x4D.shape, dyActivation, $filter, strides, pad);
    var filterDer =
        conv2DBackpropFilter(x4D, dyActivation, $filter.shape, strides, pad);
    var der = [xDer, filterDer];
    if ($bias != null) {
      var biasDer = getFusedBiasGradient($bias, dyActivation);
      der.push(biasDer);
    }
    return der;
  };
  var inputs = {
    x: x4D,
    filter: $filter,
    bias: $bias,
    preluActivationWeights: $preluActivationWeights
  };
  var attrs = {
    strides: strides,
    pad: pad,
    dataFormat: dataFormat,
    dilations: dilations,
    dimRoundingMode: dimRoundingMode,
    activation: activation,
    leakyreluAlpha: leakyreluAlpha
  };
  // Depending on the the params passed in we will have different number of
  // inputs and thus a a different number of elements in the gradient.
  if (bias == null) {
    var customOp = customGrad(function(x4D, filter, save) {
      var res =
          // tslint:disable-next-line: no-unnecessary-type-assertion
          ENGINE.runKernel(FusedConv2D, inputs, attrs);
      save([filter, x4D, res]);
      if (reshapedTo4D) {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
      }
      return {value: res, gradFunc: grad};
    });
    return customOp(x4D, $filter);
  } else {
    var customOpWithBias = customGrad(function(x4D, filter, bias, save) {
      var res = ENGINE.runKernel(FusedConv2D, inputs, attrs);
      save([filter, x4D, res, bias]);
      if (reshapedTo4D) {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
      }
      return {value: res, gradFunc: grad};
    });
    return customOpWithBias(x4D, $filter, $bias);
  }
}
var conv2d$1 = op({fusedConv2d_: fusedConv2d_});

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
function depthwiseConv2dNativeBackpropFilter_(
    x, dy, filterShape, strides, pad, dilations, dimRoundingMode) {
  if (dilations === void 0) {
    dilations = [1, 1];
  }
  var x4D = x;
  if (x.rank === 3) {
    x4D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]]);
  }
  var dy4D = dy;
  if (dy4D.rank === 3) {
    dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
  }
  var inputs = {x: x4D, dy: dy4D};
  var attrs = {
    strides: strides,
    pad: pad,
    dimRoundingMode: dimRoundingMode,
    dilations: dilations,
    filterShape: filterShape
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  return ENGINE.runKernel(DepthwiseConv2dNativeBackpropFilter, inputs, attrs);
}
var depthwiseConv2dNativeBackpropFilter = op({
  depthwiseConv2dNativeBackpropFilter_: depthwiseConv2dNativeBackpropFilter_
});

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
function depthwiseConv2dNativeBackpropInput_(
    xShape, dy, filter, strides, pad, dilations, dimRoundingMode) {
  if (dilations === void 0) {
    dilations = [1, 1];
  }
  var dy4D = dy;
  var reshapedTo4D = false;
  if (dy.rank === 3) {
    reshapedTo4D = true;
    dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
  }
  var inputs = {dy: dy4D, filter: filter};
  var attrs = {
    strides: strides,
    pad: pad,
    dimRoundingMode: dimRoundingMode,
    dilations: dilations,
    inputShape: xShape
  };
  var res =
      // tslint:disable-next-line: no-unnecessary-type-assertion
      ENGINE.runKernel(DepthwiseConv2dNativeBackpropInput, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var depthwiseConv2dNativeBackpropInput = op(
    {depthwiseConv2dNativeBackpropInput_: depthwiseConv2dNativeBackpropInput_});

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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`).
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 * @param leakyreluAlpha Optional. Alpha to be applied as part of a `leakyrelu`
 *     activation.
 */
function fusedDepthwiseConv2d_(_a) {
  var x = _a.x, filter = _a.filter, strides = _a.strides, pad = _a.pad,
      _b = _a.dataFormat, dataFormat = _b === void 0 ? 'NHWC' : _b,
      _c = _a.dilations, dilations = _c === void 0 ? [1, 1] : _c,
      dimRoundingMode = _a.dimRoundingMode, bias = _a.bias, _d = _a.activation,
      activation = _d === void 0 ? 'linear' : _d,
      preluActivationWeights = _a.preluActivationWeights,
      leakyreluAlpha = _a.leakyreluAlpha;
  if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
    var result = depthwiseConv2d(
        x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
    if (bias != null) {
      result = add$1(result, bias);
    }
    return applyActivation(
        result, activation, preluActivationWeights, leakyreluAlpha);
  }
  var $x = convertToTensor(x, 'x', 'depthwiseConv2d');
  var $filter = convertToTensor(filter, 'filter', 'depthwiseConv2d');
  var x4D = $x;
  var reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  assert(x4D.rank === 4, function() {
    return 'Error in fused depthwiseConv2d: input must be rank 4, but got ' +
        ('rank ' + x4D.rank + '.');
  });
  assert($filter.rank === 4, function() {
    return 'Error in fused depthwiseConv2d: filter must be rank 4, ' +
        ('but got rank ' + $filter.rank + '.');
  });
  assert(x4D.shape[3] === $filter.shape[2], function() {
    return 'Error in fused depthwiseConv2d: number of input channels ' +
        ('(' + x4D.shape[3] + ') must match the inChannels dimension in ') +
        ('filter ' + $filter.shape[2] + '.');
  });
  if (dilations == null) {
    dilations = [1, 1];
  }
  assert(eitherStridesOrDilationsAreOne(strides, dilations), function() {
    return 'Error in fused depthwiseConv2d: Either strides or dilations must ' +
        ('be 1. Got strides ' + strides + ' and dilations \'' + dilations +
         '\'');
  });
  if (dimRoundingMode != null) {
    assert(isInt(pad), function() {
      return 'Error in fused depthwiseConv2d: pad must be an integer when ' +
          ('using dimRoundingMode ' + dimRoundingMode + ' but got pad ' + pad +
           '.');
    });
  }
  var convInfo = computeConv2DInfo(
      x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode,
      true /* depthwise */);
  var $bias;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused conv2d');
    $bias = makeTypesMatch($bias, $x)[0];
    assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
  }
  var $preluActivationWeights;
  if (preluActivationWeights != null) {
    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused depthwiseConv2d');
  }
  var grad = function(dy, saved) {
    assert(tupleValuesAreOne(dilations), function() {
      return 'Error in gradient of fused depthwiseConv2d: dilation rates ' +
          'greater than 1 are not yet supported. Got dilations ' +
          ('\'' + dilations + '\'');
    });
    var $filter = saved[0], x4D = saved[1], y = saved[2], bias = saved[3];
    var dyActivation = getFusedDyActivation(dy, y, activation);
    var xDer = depthwiseConv2dNativeBackpropInput(
        x4D.shape, dyActivation, $filter, strides, pad, dilations,
        dimRoundingMode);
    var filterDer = depthwiseConv2dNativeBackpropFilter(
        x4D, dyActivation, $filter.shape, strides, pad, dilations,
        dimRoundingMode);
    if (bias != null) {
      var biasDer = getFusedBiasGradient($bias, dyActivation);
      return [xDer, filterDer, biasDer];
    }
    return [xDer, filterDer];
  };
  var inputs = {
    x: x4D,
    filter: $filter,
    bias: $bias,
    preluActivationWeights: $preluActivationWeights
  };
  var attrs = {
    strides: strides,
    pad: pad,
    dataFormat: dataFormat,
    dilations: dilations,
    dimRoundingMode: dimRoundingMode,
    activation: activation,
    leakyreluAlpha: leakyreluAlpha
  };
  // Depending on the the params passed in we will have different number of
  // inputs and thus a a different number of elements in the gradient.
  if (bias == null) {
    var customOp = customGrad(function(x4D, filter, save) {
      // tslint:disable-next-line: no-unnecessary-type-assertion
      var res = ENGINE.runKernel(FusedDepthwiseConv2D, inputs, attrs);
      save([filter, x4D, res]);
      if (reshapedTo4D) {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
      }
      return {value: res, gradFunc: grad};
    });
    return customOp(x4D, $filter);
  } else {
    var customOpWithBias = customGrad(function(x4D, filter, bias, save) {
      // tslint:disable-next-line: no-unnecessary-type-assertion
      var res = ENGINE.runKernel(FusedDepthwiseConv2D, inputs, attrs);
      save([filter, x4D, res, bias]);
      if (reshapedTo4D) {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
      }
      return {value: res, gradFunc: grad};
    });
    return customOpWithBias(x4D, $filter, $bias);
  }
}
var depthwiseConv2d$1 = op({fusedDepthwiseConv2d_: fusedDepthwiseConv2d_});

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
 * - `leakyreluAlpha` Alpha of leakyrelu.
 */
function fusedMatMul_(_a) {
  var _b;
  var a = _a.a, b = _a.b, _c = _a.transposeA,
      transposeA = _c === void 0 ? false : _c, _d = _a.transposeB,
      transposeB = _d === void 0 ? false : _d, bias = _a.bias,
      _e = _a.activation, activation = _e === void 0 ? 'linear' : _e,
      preluActivationWeights = _a.preluActivationWeights,
      leakyreluAlpha = _a.leakyreluAlpha;
  if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
    var result = matMul(a, b, transposeA, transposeB);
    if (bias != null) {
      result = add$1(result, bias);
    }
    return applyActivation(
        result, activation, preluActivationWeights, leakyreluAlpha);
  }
  var $a = convertToTensor(a, 'a', 'fused matMul');
  var $b = convertToTensor(b, 'b', 'fused matMul');
  _b = makeTypesMatch($a, $b), $a = _b[0], $b = _b[1];
  var innerShapeA = transposeA ? $a.shape[$a.rank - 2] : $a.shape[$a.rank - 1];
  var innerShapeB = transposeB ? $b.shape[$b.rank - 1] : $b.shape[$b.rank - 2];
  var outerShapeA = transposeA ? $a.shape[$a.rank - 1] : $a.shape[$a.rank - 2];
  var outerShapeB = transposeB ? $b.shape[$b.rank - 2] : $b.shape[$b.rank - 1];
  var outerDimsA = $a.shape.slice(0, -2);
  var outerDimsB = $b.shape.slice(0, -2);
  var batchDimA = sizeFromShape(outerDimsA);
  var batchDimB = sizeFromShape(outerDimsB);
  assert($a.rank >= 2 && $b.rank >= 2 && $a.rank === $b.rank, function() {
    return 'Error in fused matMul: inputs must have the same rank of at ' +
        ('least 2, got ranks ' + $a.rank + ' and ' + $b.rank + '.');
  });
  assert(arraysEqual(outerDimsA, outerDimsB), function() {
    return 'Error in fused matMul: outer dimensions (' + outerDimsA +
        ') and (' +
        (outerDimsB + ') of Tensors with shapes ' + $a.shape + ' and ') +
        ($b.shape + ' must match.');
  });
  assert(innerShapeA === innerShapeB, function() {
    return 'Error in fused matMul: inner shapes (' + innerShapeA + ') and (' +
        (innerShapeB + ') of Tensors with shapes ' + $a.shape + ' and ') +
        ($b.shape + ' and transposeA=' + transposeA) +
        (' and transposeB=' + transposeB + ' must match.');
  });
  var outShape = $a.shape.slice(0, -2).concat([outerShapeA, outerShapeB]);
  var a3D = transposeA ? reshape($a, [batchDimA, innerShapeA, outerShapeA]) :
                         reshape($a, [batchDimA, outerShapeA, innerShapeA]);
  var b3D = transposeB ? reshape($b, [batchDimB, outerShapeB, innerShapeB]) :
                         reshape($b, [batchDimB, innerShapeB, outerShapeB]);
  var $bias;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused matMul');
    $bias = makeTypesMatch($bias, $a)[0];
    assertAndGetBroadcastShape(outShape, $bias.shape);
  }
  var $preluActivationWeights;
  if (preluActivationWeights != null) {
    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused matMul');
  }
  var grad = function(dy, saved) {
    var a3D = saved[0], b3D = saved[1], y = saved[2], $bias = saved[3];
    // we reshape dy because the result of the forward is not
    // necessarily going to be a 3d tensor due to a reshape done at the end of
    // the customOp.
    var dyActivation =
        getFusedDyActivation(reshape(dy, y.shape), y, activation);
    var aDer;
    var bDer;
    if (!transposeA && !transposeB) {
      aDer = matMul(dyActivation, b3D, false, true);
      bDer = matMul(a3D, dyActivation, true, false);
    } else if (!transposeA && transposeB) {
      aDer = matMul(dyActivation, b3D, false, false);
      bDer = matMul(dyActivation, a3D, true, false);
    } else if (transposeA && !transposeB) {
      aDer = matMul(b3D, dyActivation, false, true);
      bDer = matMul(a3D, dyActivation, false, false);
    } else {
      aDer = matMul(b3D, dyActivation, true, true);
      bDer = matMul(dyActivation, a3D, true, true);
    }
    if (bias != null) {
      var biasDer = getFusedBiasGradient($bias, dyActivation);
      return [aDer, bDer, biasDer];
    } else {
      return [aDer, bDer];
    }
  };
  var inputs = {
    a: a3D,
    b: b3D,
    bias: $bias,
    preluActivationWeights: $preluActivationWeights
  };
  var attrs = {
    transposeA: transposeA,
    transposeB: transposeB,
    activation: activation,
    leakyreluAlpha: leakyreluAlpha
  };
  // Depending on the the params passed in we will have different number of
  // inputs and thus a a different number of elements in the gradient.
  if (bias == null) {
    var customOp = customGrad(function(a3D, b3D, save) {
      var res =
          // tslint:disable-next-line: no-unnecessary-type-assertion
          ENGINE.runKernel(_FusedMatMul, inputs, attrs);
      save([a3D, b3D, res]);
      return {value: reshape(res, outShape), gradFunc: grad};
    });
    return customOp(a3D, b3D);
  } else {
    var customOpWithBias = customGrad(function(a3D, b3D, $bias, save) {
      var res =
          // tslint:disable-next-line: no-unnecessary-type-assertion
          ENGINE.runKernel(_FusedMatMul, inputs, attrs);
      save([a3D, b3D, res, $bias]);
      return {value: reshape(res, outShape), gradFunc: grad};
    });
    return customOpWithBias(a3D, b3D, $bias);
  }
}
var matMul$1 = op({fusedMatMul_: fusedMatMul_});

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

var fused_ops = {
  __proto__: null,
  conv2d: conv2d$1,
  depthwiseConv2d: depthwiseConv2d$1,
  matMul: matMul$1
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
/**
 * Generate a hamming window.
 *
 * See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
 *
 * ```js
 * tf.signal.hammingWindow(10).print();
 * ```
 * @param The length of window
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function hammingWindow_(windowLength) {
  return cosineWindow(windowLength, 0.54, 0.46);
}
var hammingWindow = op({hammingWindow_: hammingWindow_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function hannWindow_(windowLength) {
  return cosineWindow(windowLength, 0.5, 0.5);
}
var hannWindow = op({hannWindow_: hannWindow_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function frame_(signal, frameLength, frameStep, padEnd, padValue) {
  if (padEnd === void 0) {
    padEnd = false;
  }
  if (padValue === void 0) {
    padValue = 0;
  }
  var start = 0;
  var output = [];
  while (start + frameLength <= signal.size) {
    output.push(slice(signal, start, frameLength));
    start += frameStep;
  }
  if (padEnd) {
    while (start < signal.size) {
      var padLen = (start + frameLength) - signal.size;
      var pad = concat([
        slice(signal, start, frameLength - padLen), fill([padLen], padValue)
      ]);
      output.push(pad);
      start += frameStep;
    }
  }
  if (output.length === 0) {
    return tensor2d([], [0, frameLength]);
  }
  return reshape(concat(output), [output.length, frameLength]);
}
var frame = op({frame_: frame_});

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
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function stft_(signal, frameLength, frameStep, fftLength, windowFn) {
  if (windowFn === void 0) {
    windowFn = hannWindow;
  }
  if (fftLength == null) {
    fftLength = enclosingPowerOfTwo(frameLength);
  }
  var framedSignal = frame(signal, frameLength, frameStep);
  var windowedSignal = mul(framedSignal, windowFn(frameLength));
  var output = [];
  for (var i = 0; i < framedSignal.shape[0]; i++) {
    output.push(
        rfft(slice(windowedSignal, [i, 0], [1, frameLength]), fftLength));
  }
  return concat(output);
}
var stft = op({stft_: stft_});

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
/**
 * Extracts crops from the input image tensor and resizes them using bilinear
 * sampling or nearest neighbor sampling (possibly with aspect ratio change)
 * to a common output size specified by cropSize.
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
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function cropAndResize_(
    image, boxes, boxInd, cropSize, method, extrapolationValue) {
  if (method === void 0) {
    method = 'bilinear';
  }
  if (extrapolationValue === void 0) {
    extrapolationValue = 0;
  }
  var $image = convertToTensor(image, 'image', 'cropAndResize');
  var $boxes = convertToTensor(boxes, 'boxes', 'cropAndResize', 'float32');
  var $boxInd = convertToTensor(boxInd, 'boxInd', 'cropAndResize', 'int32');
  var numBoxes = $boxes.shape[0];
  assert($image.rank === 4, function() {
    return 'Error in cropAndResize: image must be rank 4,' +
        ('but got rank ' + $image.rank + '.');
  });
  assert($boxes.rank === 2 && $boxes.shape[1] === 4, function() {
    return 'Error in cropAndResize: boxes must be have size [' + numBoxes +
        ',4] ' + ('but had shape ' + $boxes.shape + '.');
  });
  assert($boxInd.rank === 1 && $boxInd.shape[0] === numBoxes, function() {
    return 'Error in cropAndResize: boxInd must be have size [' + numBoxes +
        '] ' + ('but had shape ' + $boxes.shape + '.');
  });
  assert(cropSize.length === 2, function() {
    return 'Error in cropAndResize: cropSize must be of length 2, but got ' +
        ('length ' + cropSize.length + '.');
  });
  assert(cropSize[0] >= 1 && cropSize[1] >= 1, function() {
    return 'cropSize must be atleast [1,1], but was ' + cropSize;
  });
  assert(method === 'bilinear' || method === 'nearest', function() {
    return 'method must be bilinear or nearest, but was ' + method;
  });
  var inputs = {image: $image, boxes: $boxes, boxInd: $boxInd};
  var attrs = {
    method: method,
    extrapolationValue: extrapolationValue,
    cropSize: cropSize
  };
  var res = ENGINE.runKernel(CropAndResize, inputs, attrs);
  return res;
}
var cropAndResize = op({cropAndResize_: cropAndResize_});

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
/**
 * Flips the image left to right. Currently available in the CPU, WebGL, and
 * WASM backends.
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 */
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
function flipLeftRight_(image) {
  var $image = convertToTensor(image, 'image', 'flipLeftRight', 'float32');
  assert($image.rank === 4, function() {
    return 'Error in flipLeftRight: image must be rank 4,' +
        ('but got rank ' + $image.rank + '.');
  });
  var inputs = {image: $image};
  var res = ENGINE.runKernel(FlipLeftRight, inputs, {});
  return res;
}
var flipLeftRight = op({flipLeftRight_: flipLeftRight_});

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
/**
 * Rotates the input image tensor counter-clockwise with an optional offset
 * center of rotation. Currently available in the CPU, WebGL, and WASM backends.
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 * @param radians The amount of rotation.
 * @param fillValue The value to fill in the empty space leftover
 *     after rotation. Can be either a single grayscale value (0-255), or an
 *     array of three numbers `[red, green, blue]` specifying the red, green,
 *     and blue channels. Defaults to `0` (black).
 * @param center The center of rotation. Can be either a single value (0-1), or
 *     an array of two numbers `[centerX, centerY]`. Defaults to `0.5` (rotates
 *     the image around its center).
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function rotateWithOffset_(image, radians, fillValue, center) {
  if (fillValue === void 0) {
    fillValue = 0;
  }
  if (center === void 0) {
    center = 0.5;
  }
  var $image = convertToTensor(image, 'image', 'rotateWithOffset', 'float32');
  assert($image.rank === 4, function() {
    return 'Error in rotateWithOffset: image must be rank 4,' +
        ('but got rank ' + $image.rank + '.');
  });
  var inputs = {image: $image};
  var attrs = {radians: radians, fillValue: fillValue, center: center};
  var res = ENGINE.runKernel(RotateWithOffset, inputs, attrs);
  return res;
}
var rotateWithOffset = op({rotateWithOffset_: rotateWithOffset_});

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
function nonMaxSuppSanityCheck(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma) {
  if (iouThreshold == null) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold == null) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  if (softNmsSigma == null) {
    softNmsSigma = 0.0;
  }
  var numBoxes = boxes.shape[0];
  maxOutputSize = Math.min(maxOutputSize, numBoxes);
  assert(0 <= iouThreshold && iouThreshold <= 1, function() {
    return 'iouThreshold must be in [0, 1], but was \'' + iouThreshold + '\'';
  });
  assert(boxes.rank === 2, function() {
    return 'boxes must be a 2D tensor, but was of rank \'' + boxes.rank + '\'';
  });
  assert(boxes.shape[1] === 4, function() {
    return 'boxes must have 4 columns, but 2nd dimension was ' + boxes.shape[1];
  });
  assert(scores.rank === 1, function() {
    return 'scores must be a 1D tensor';
  });
  assert(scores.shape[0] === numBoxes, function() {
    return 'scores has incompatible shape with boxes. Expected ' + numBoxes +
        ', ' + ('but was ' + scores.shape[0]);
  });
  assert(0 <= softNmsSigma && softNmsSigma <= 1, function() {
    return 'softNmsSigma must be in [0, 1], but was \'' + softNmsSigma + '\'';
  });
  return {
    maxOutputSize: maxOutputSize,
    iouThreshold: iouThreshold,
    scoreThreshold: scoreThreshold,
    softNmsSigma: softNmsSigma
  };
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
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function nonMaxSuppression_(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
  if (iouThreshold === void 0) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold === void 0) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  var $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppression');
  var $scores = convertToTensor(scores, 'scores', 'nonMaxSuppression');
  var inputs = nonMaxSuppSanityCheck(
      $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold);
  maxOutputSize = inputs.maxOutputSize;
  iouThreshold = inputs.iouThreshold;
  scoreThreshold = inputs.scoreThreshold;
  var attrs = {
    maxOutputSize: maxOutputSize,
    iouThreshold: iouThreshold,
    scoreThreshold: scoreThreshold
  };
  return ENGINE.runKernel(
      NonMaxSuppressionV3, {boxes: $boxes, scores: $scores}, attrs);
}
var nonMaxSuppression = op({nonMaxSuppression_: nonMaxSuppression_});

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
  var index = binarySearch(arr, element, comparator);
  var insertionPoint = index < 0 ? -(index + 1) : index;
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
  var left = 0;
  var right = arr.length;
  var middle = 0;
  var found = false;
  while (left < right) {
    middle = left + ((right - left) >>> 1);
    var compareResult = comparator(target, arr[middle]);
    if (compareResult > 0) {
      left = middle + 1;
    } else {
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
function nonMaxSuppressionV3Impl(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
  return nonMaxSuppressionImpl_(
      boxes, scores, maxOutputSize, iouThreshold, scoreThreshold,
      0 /* softNmsSigma */);
}
function nonMaxSuppressionV4Impl(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold,
    padToMaxOutputSize) {
  return nonMaxSuppressionImpl_(
      boxes, scores, maxOutputSize, iouThreshold, scoreThreshold,
      0 /* softNmsSigma */, false /* returnScoresTensor */,
      padToMaxOutputSize /* padToMaxOutputSize */, true
      /* returnValidOutputs */);
}
function nonMaxSuppressionV5Impl(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma) {
  return nonMaxSuppressionImpl_(
      boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma,
      true /* returnScoresTensor */);
}
function nonMaxSuppressionImpl_(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma,
    returnScoresTensor, padToMaxOutputSize, returnValidOutputs) {
  if (returnScoresTensor === void 0) {
    returnScoresTensor = false;
  }
  if (padToMaxOutputSize === void 0) {
    padToMaxOutputSize = false;
  }
  if (returnValidOutputs === void 0) {
    returnValidOutputs = false;
  }
  // The list is sorted in ascending order, so that we can always pop the
  // candidate with the largest score in O(1) time.
  var candidates = [];
  for (var i = 0; i < scores.length; i++) {
    if (scores[i] > scoreThreshold) {
      candidates.push({score: scores[i], boxIndex: i, suppressBeginIndex: 0});
    }
  }
  candidates.sort(ascendingComparator);
  // If softNmsSigma is 0, the outcome of this algorithm is exactly same as
  // before.
  var scale = softNmsSigma > 0 ? (-0.5 / softNmsSigma) : 0.0;
  var selectedIndices = [];
  var selectedScores = [];
  while (selectedIndices.length < maxOutputSize && candidates.length > 0) {
    var candidate = candidates.pop();
    var originalScore = candidate.score, boxIndex = candidate.boxIndex,
        suppressBeginIndex = candidate.suppressBeginIndex;
    if (originalScore < scoreThreshold) {
      break;
    }
    // Overlapping boxes are likely to have similar scores, therefore we
    // iterate through the previously selected boxes backwards in order to
    // see if candidate's score should be suppressed. We use
    // suppressBeginIndex to track and ensure a candidate can be suppressed
    // by a selected box no more than once. Also, if the overlap exceeds
    // iouThreshold, we simply ignore the candidate.
    var ignoreCandidate = false;
    for (var j = selectedIndices.length - 1; j >= suppressBeginIndex; --j) {
      var iou = intersectionOverUnion(boxes, boxIndex, selectedIndices[j]);
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
      } else if (candidate.score > scoreThreshold) {
        // Candidate's score is suppressed but is still high enough to be
        // considered, so add back to the candidates list.
        binaryInsert(candidates, candidate, ascendingComparator);
      }
    }
  }
  // NonMaxSuppressionV4 feature: padding output to maxOutputSize.
  var validOutputs = selectedIndices.length;
  var elemsToPad = maxOutputSize - validOutputs;
  if (padToMaxOutputSize && elemsToPad > 0) {
    selectedIndices.push.apply(selectedIndices, new Array(elemsToPad).fill(0));
    selectedScores.push.apply(selectedScores, new Array(elemsToPad).fill(0.0));
  }
  var result = {selectedIndices: selectedIndices};
  if (returnScoresTensor) {
    result['selectedScores'] = selectedScores;
  }
  if (returnValidOutputs) {
    result['validOutputs'] = validOutputs;
  }
  return result;
}
function intersectionOverUnion(boxes, i, j) {
  var iCoord = boxes.subarray(i * 4, i * 4 + 4);
  var jCoord = boxes.subarray(j * 4, j * 4 + 4);
  var yminI = Math.min(iCoord[0], iCoord[2]);
  var xminI = Math.min(iCoord[1], iCoord[3]);
  var ymaxI = Math.max(iCoord[0], iCoord[2]);
  var xmaxI = Math.max(iCoord[1], iCoord[3]);
  var yminJ = Math.min(jCoord[0], jCoord[2]);
  var xminJ = Math.min(jCoord[1], jCoord[3]);
  var ymaxJ = Math.max(jCoord[0], jCoord[2]);
  var xmaxJ = Math.max(jCoord[1], jCoord[3]);
  var areaI = (ymaxI - yminI) * (xmaxI - xminI);
  var areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
  if (areaI <= 0 || areaJ <= 0) {
    return 0.0;
  }
  var intersectionYmin = Math.max(yminI, yminJ);
  var intersectionXmin = Math.max(xminI, xminJ);
  var intersectionYmax = Math.min(ymaxI, ymaxJ);
  var intersectionXmax = Math.min(xmaxI, xmaxJ);
  var intersectionArea = Math.max(intersectionYmax - intersectionYmin, 0.0) *
      Math.max(intersectionXmax - intersectionXmin, 0.0);
  return intersectionArea / (areaI + areaJ - intersectionArea);
}
// A Gaussian penalty function, this method always returns values in [0, 1].
// The weight is a function of similarity, the more overlap two boxes are, the
// smaller the weight is, meaning highly overlapping boxe will be significantly
// penalized. On the other hand, a non-overlapping box will not be penalized.
function suppressWeight(iouThreshold, scale, iou) {
  var weight = Math.exp(scale * iou * iou);
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
/**
 * Performs non maximum suppression of bounding boxes based on
 * iou (intersection over union).
 *
 * This is the async version of `nonMaxSuppression`
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
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function nonMaxSuppressionAsync_(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
  if (iouThreshold === void 0) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold === void 0) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  return __awaiter(this, void 0, void 0, function() {
    var $boxes, $scores, inputs, boxesAndScores, boxesVals, scoresVals,
        selectedIndices;
    return __generator(this, function(_a) {
      switch (_a.label) {
        case 0:
          $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
          $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');
          inputs = nonMaxSuppSanityCheck(
              $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold);
          maxOutputSize = inputs.maxOutputSize;
          iouThreshold = inputs.iouThreshold;
          scoreThreshold = inputs.scoreThreshold;
          return [4 /*yield*/, Promise.all([$boxes.data(), $scores.data()])];
        case 1:
          boxesAndScores = _a.sent();
          boxesVals = boxesAndScores[0];
          scoresVals = boxesAndScores[1];
          selectedIndices = nonMaxSuppressionV3Impl(
                                boxesVals, scoresVals, maxOutputSize,
                                iouThreshold, scoreThreshold)
                                .selectedIndices;
          if ($boxes !== boxes) {
            $boxes.dispose();
          }
          if ($scores !== scores) {
            $scores.dispose();
          }
          return [2 /*return*/, tensor1d(selectedIndices, 'int32')];
      }
    });
  });
}
var nonMaxSuppressionAsync = nonMaxSuppressionAsync_;

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
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function nonMaxSuppressionWithScore_(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma) {
  if (iouThreshold === void 0) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold === void 0) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  if (softNmsSigma === void 0) {
    softNmsSigma = 0.0;
  }
  var $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppression');
  var $scores = convertToTensor(scores, 'scores', 'nonMaxSuppression');
  var params = nonMaxSuppSanityCheck(
      $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold,
      softNmsSigma);
  maxOutputSize = params.maxOutputSize;
  iouThreshold = params.iouThreshold;
  scoreThreshold = params.scoreThreshold;
  softNmsSigma = params.softNmsSigma;
  var inputs = {boxes: $boxes, scores: $scores};
  var attrs = {
    maxOutputSize: maxOutputSize,
    iouThreshold: iouThreshold,
    scoreThreshold: scoreThreshold,
    softNmsSigma: softNmsSigma
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var result = ENGINE.runKernel(NonMaxSuppressionV5, inputs, attrs);
  return {selectedIndices: result[0], selectedScores: result[1]};
}
var nonMaxSuppressionWithScore =
    op({nonMaxSuppressionWithScore_: nonMaxSuppressionWithScore_});

/**
 * Asynchronously performs non maximum suppression of bounding boxes based on
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
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function nonMaxSuppressionWithScoreAsync_(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma) {
  if (iouThreshold === void 0) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold === void 0) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  if (softNmsSigma === void 0) {
    softNmsSigma = 0.0;
  }
  return __awaiter(this, void 0, void 0, function() {
    var $boxes, $scores, params, boxesAndScores, boxesVals, scoresVals, _a,
        selectedIndices, selectedScores;
    return __generator(this, function(_b) {
      switch (_b.label) {
        case 0:
          $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
          $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');
          params = nonMaxSuppSanityCheck(
              $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold,
              softNmsSigma);
          maxOutputSize = params.maxOutputSize;
          iouThreshold = params.iouThreshold;
          scoreThreshold = params.scoreThreshold;
          softNmsSigma = params.softNmsSigma;
          return [4 /*yield*/, Promise.all([$boxes.data(), $scores.data()])];
        case 1:
          boxesAndScores = _b.sent();
          boxesVals = boxesAndScores[0];
          scoresVals = boxesAndScores[1];
          _a = nonMaxSuppressionV5Impl(
              boxesVals, scoresVals, maxOutputSize, iouThreshold,
              scoreThreshold, softNmsSigma),
          selectedIndices = _a.selectedIndices,
          selectedScores = _a.selectedScores;
          if ($boxes !== boxes) {
            $boxes.dispose();
          }
          if ($scores !== scores) {
            $scores.dispose();
          }
          return [
            2 /*return*/, {
              selectedIndices: tensor1d(selectedIndices, 'int32'),
              selectedScores: tensor1d(selectedScores)
            }
          ];
      }
    });
  });
}
var nonMaxSuppressionWithScoreAsync = nonMaxSuppressionWithScoreAsync_;

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
/**
 * Asynchronously performs non maximum suppression of bounding boxes based on
 * iou (intersection over union), with an option to pad results.
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
 * @param padToMaxOutputSize Defalts to false. If true, size of output
 *     `selectedIndices` is padded to maxOutputSize.
 * @return A map with the following properties:
 *     - selectedIndices: A 1D tensor with the selected box indices.
 *     - validOutputs: A scalar denoting how many elements in `selectedIndices`
 *       are valid. Valid elements occur first, then padding.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function nonMaxSuppressionPadded_(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold,
    padToMaxOutputSize) {
  if (iouThreshold === void 0) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold === void 0) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  if (padToMaxOutputSize === void 0) {
    padToMaxOutputSize = false;
  }
  var $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppression');
  var $scores = convertToTensor(scores, 'scores', 'nonMaxSuppression');
  var params = nonMaxSuppSanityCheck(
      $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold,
      null /* softNmsSigma */);
  var $maxOutputSize = params.maxOutputSize;
  var $iouThreshold = params.iouThreshold;
  var $scoreThreshold = params.scoreThreshold;
  var inputs = {boxes: $boxes, scores: $scores};
  var attrs = {
    maxOutputSize: $maxOutputSize,
    iouThreshold: $iouThreshold,
    scoreThreshold: $scoreThreshold,
    padToMaxOutputSize: padToMaxOutputSize
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var result = ENGINE.runKernel(NonMaxSuppressionV4, inputs, attrs);
  return {selectedIndices: result[0], validOutputs: result[1]};
}
var nonMaxSuppressionPadded =
    op({nonMaxSuppressionPadded_: nonMaxSuppressionPadded_});

/**
 * Asynchronously performs non maximum suppression of bounding boxes based on
 * iou (intersection over union), with an option to pad results.
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
 * @param padToMaxOutputSize Defalts to false. If true, size of output
 *     `selectedIndices` is padded to maxOutputSize.
 * @return A map with the following properties:
 *     - selectedIndices: A 1D tensor with the selected box indices.
 *     - validOutputs: A scalar denoting how many elements in `selectedIndices`
 *       are valid. Valid elements occur first, then padding.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function nonMaxSuppressionPaddedAsync_(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold,
    padToMaxOutputSize) {
  if (iouThreshold === void 0) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold === void 0) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  if (padToMaxOutputSize === void 0) {
    padToMaxOutputSize = false;
  }
  return __awaiter(this, void 0, void 0, function() {
    var $boxes, $scores, params, $maxOutputSize, $iouThreshold, $scoreThreshold,
        _a, boxesVals, scoresVals, _b, selectedIndices, validOutputs;
    return __generator(this, function(_c) {
      switch (_c.label) {
        case 0:
          $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
          $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');
          params = nonMaxSuppSanityCheck(
              $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold,
              null /* softNmsSigma */);
          $maxOutputSize = params.maxOutputSize;
          $iouThreshold = params.iouThreshold;
          $scoreThreshold = params.scoreThreshold;
          return [4 /*yield*/, Promise.all([$boxes.data(), $scores.data()])];
        case 1:
          _a = _c.sent(), boxesVals = _a[0], scoresVals = _a[1];
          _b = nonMaxSuppressionV4Impl(
              boxesVals, scoresVals, $maxOutputSize, $iouThreshold,
              $scoreThreshold, padToMaxOutputSize),
          selectedIndices = _b.selectedIndices, validOutputs = _b.validOutputs;
          if ($boxes !== boxes) {
            $boxes.dispose();
          }
          if ($scores !== scores) {
            $scores.dispose();
          }
          return [
            2 /*return*/, {
              selectedIndices: tensor1d(selectedIndices, 'int32'),
              validOutputs: scalar(validOutputs, 'int32')
            }
          ];
      }
    });
  });
}
var nonMaxSuppressionPaddedAsync = nonMaxSuppressionPaddedAsync_;

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
/**
 * Bilinear resize a single 3D image or a batch of 3D images to a new shape.
 *
 * @param images The images, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param size The new shape `[newHeight, newWidth]` to resize the
 *     images to. Each channel is resized individually.
 * @param alignCorners Defaults to `false`. If true, rescale
 *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
 *     corners of images and resized images. If false, rescale by
 *     `new_height / height`. Treat similarly the width dimension.
 * @param halfPixelCenters Defaults to `false`. Whether to assume pixel centers
 *     are at 0.5, which would make the floating point coordinates of the top
 *     left pixel 0.5, 0.5.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function resizeBilinear_(images, size, alignCorners, halfPixelCenters) {
  if (alignCorners === void 0) {
    alignCorners = false;
  }
  if (halfPixelCenters === void 0) {
    halfPixelCenters = false;
  }
  var $images = convertToTensor(images, 'images', 'resizeBilinear');
  assert($images.rank === 3 || $images.rank === 4, function() {
    return 'Error in resizeBilinear: x must be rank 3 or 4, but got ' +
        ('rank ' + $images.rank + '.');
  });
  assert(size.length === 2, function() {
    return 'Error in resizeBilinear: new shape must 2D, but got shape ' +
        (size + '.');
  });
  assert(halfPixelCenters === false || alignCorners === false, function() {
    return 'Error in resizeBilinear: If halfPixelCenters is true, ' +
        'alignCorners must be false.';
  });
  var batchImages = $images;
  var reshapedTo4D = false;
  if ($images.rank === 3) {
    reshapedTo4D = true;
    batchImages = reshape(
        $images, [1, $images.shape[0], $images.shape[1], $images.shape[2]]);
  }
  var inputs = {images: batchImages};
  var attrs = {
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters,
    size: size
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(ResizeBilinear, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var resizeBilinear = op({resizeBilinear_: resizeBilinear_});

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
 * @param halfPixelCenters Defaults to `false`. Whether to assumes pixels are of
 *      half the actual dimensions, and yields more accurate resizes. This flag
 *      would also make the floating point coordinates of the top left pixel
 *      0.5, 0.5.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function resizeNearestNeighbor_(images, size, alignCorners, halfPixelCenters) {
  if (alignCorners === void 0) {
    alignCorners = false;
  }
  if (halfPixelCenters === void 0) {
    halfPixelCenters = false;
  }
  var $images = convertToTensor(images, 'images', 'resizeNearestNeighbor');
  assert($images.rank === 3 || $images.rank === 4, function() {
    return 'Error in resizeNearestNeighbor: x must be rank 3 or 4, but got ' +
        ('rank ' + $images.rank + '.');
  });
  assert(size.length === 2, function() {
    return 'Error in resizeNearestNeighbor: new shape must 2D, but got shape ' +
        (size + '.');
  });
  assert($images.dtype === 'float32' || $images.dtype === 'int32', function() {
    return '`images` must have `int32` or `float32` as dtype';
  });
  assert(halfPixelCenters === false || alignCorners === false, function() {
    return 'Error in resizeNearestNeighbor: If halfPixelCenters is true, ' +
        'alignCorners must be false.';
  });
  var batchImages = $images;
  var reshapedTo4D = false;
  if ($images.rank === 3) {
    reshapedTo4D = true;
    batchImages = reshape(
        $images, [1, $images.shape[0], $images.shape[1], $images.shape[2]]);
  }
  var inputs = {images: batchImages};
  var attrs = {
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters,
    size: size
  };
  // tslint:disable-next-line: no-unnecessary-type-assertion
  var res = ENGINE.runKernel(ResizeNearestNeighbor, inputs, attrs);
  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
  }
  return res;
}
var resizeNearestNeighbor =
    op({resizeNearestNeighbor_: resizeNearestNeighbor_});

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
 *
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
function bandPart_(a, numLower, numUpper) {
  assert(numLower % 1 === 0, function() {
    return 'bandPart(): numLower must be an integer, got ' + numLower + '.';
  });
  assert(numUpper % 1 === 0, function() {
    return 'bandPart(): numUpper must be an integer, got ' + numUpper + '.';
  });
  var $a = convertToTensor(a, 'a', 'bandPart');
  assert($a.rank >= 2, function() {
    return 'bandPart(): Rank must be at least 2, got ' + $a.rank + '.';
  });
  var shape = $a.shape;
  var _a = $a.shape.slice(-2), M = _a[0], N = _a[1];
  if (!(numLower <= M)) {
    throw new Error(
        'bandPart(): numLower (' + numLower + ')' +
        (' must not be greater than the number of rows (' + M + ').'));
  }
  if (!(numUpper <= N)) {
    throw new Error(
        'bandPart(): numUpper (' + numUpper + ')' +
        (' must not be greater than the number of columns (' + N + ').'));
  }
  if (numLower < 0) {
    numLower = M;
  }
  if (numUpper < 0) {
    numUpper = N;
  }
  var i = reshape(range(0, M, 1, 'int32'), [-1, 1]);
  var j = range(0, N, 1, 'int32');
  var ij = sub(i, j);
  var inBand = logicalAnd(
      lessEqual(ij, scalar(+numLower, 'int32')),
      greaterEqual(ij, scalar(-numUpper, 'int32')));
  var zero = zeros([M, N], $a.dtype);
  return reshape(
      stack(unstack(reshape($a, [-1, M, N])).map(function(mat) {
        return where(inBand, mat, zero);
      })),
      shape);
}
var bandPart = op({bandPart_: bandPart_});

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
 *
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
function gramSchmidt_(xs) {
  var inputIsTensor2D;
  if (Array.isArray(xs)) {
    inputIsTensor2D = false;
    assert(xs != null && xs.length > 0, function() {
      return 'Gram-Schmidt process: input must not be null, undefined, or ' +
          'empty';
    });
    var dim_1 = xs[0].shape[0];
    var _loop_1 = function(i) {
      assert(xs[i].shape[0] === dim_1, function() {
        return 'Gram-Schmidt: Non-unique lengths found in the input vectors: ' +
            ('(' + xs[i].shape[0] + ' vs. ' + dim_1 + ')');
      });
    };
    for (var i = 1; i < xs.length; ++i) {
      _loop_1(i);
    }
  } else {
    inputIsTensor2D = true;
    xs = split(xs, xs.shape[0], 0).map(function(x) {
      return squeeze(x, [0]);
    });
  }
  assert(xs.length <= xs[0].shape[0], function() {
    return 'Gram-Schmidt: Number of vectors (' + xs.length + ') exceeds ' +
        ('number of dimensions (' + xs[0].shape[0] + ').');
  });
  var ys = [];
  var xs1d = xs;
  var _loop_2 = function(i) {
    ys.push(ENGINE.tidy(function() {
      var x = xs1d[i];
      if (i > 0) {
        for (var j = 0; j < i; ++j) {
          var proj = mul(sum$1(mul(ys[j], x)), ys[j]);
          x = sub(x, proj);
        }
      }
      return div(x, norm(x, 'euclidean'));
    }));
  };
  for (var i = 0; i < xs.length; ++i) {
    _loop_2(i);
  }
  if (inputIsTensor2D) {
    return stack(ys, 0);
  } else {
    return ys;
  }
}
var gramSchmidt = op({gramSchmidt_: gramSchmidt_});

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
 *
 * @doc {heading:'Operations',
 *       subheading:'Linear Algebra',
 *       namespace:'linalg'}
 */
function qr_(x, fullMatrices) {
  if (fullMatrices === void 0) {
    fullMatrices = false;
  }
  assert(x.rank >= 2, function() {
    return 'qr() requires input tensor to have a rank >= 2, but got rank ' +
        x.rank;
  });
  if (x.rank === 2) {
    return qr2d(x, fullMatrices);
  } else {
    // Rank > 2.
    // TODO(cais): Below we split the input into individual 2D tensors,
    //   perform QR decomposition on them and then stack the results back
    //   together. We should explore whether this can be parallelized.
    var outerDimsProd =
        x.shape.slice(0, x.shape.length - 2).reduce(function(value, prev) {
          return value * prev;
        });
    var x2ds = unstack(
        reshape(
            x,
            [
              outerDimsProd, x.shape[x.shape.length - 2],
              x.shape[x.shape.length - 1]
            ]),
        0);
    var q2ds_1 = [];
    var r2ds_1 = [];
    x2ds.forEach(function(x2d) {
      var _a = qr2d(x2d, fullMatrices), q2d = _a[0], r2d = _a[1];
      q2ds_1.push(q2d);
      r2ds_1.push(r2d);
    });
    var q = reshape(stack(q2ds_1, 0), x.shape);
    var r = reshape(stack(r2ds_1, 0), x.shape);
    return [q, r];
  }
}
function qr2d(x, fullMatrices) {
  if (fullMatrices === void 0) {
    fullMatrices = false;
  }
  return ENGINE.tidy(function() {
    assert(x.shape.length === 2, function() {
      return 'qr2d() requires a 2D Tensor, but got a ' + x.shape.length +
          'D Tensor.';
    });
    var m = x.shape[0];
    var n = x.shape[1];
    var q = eye(m);    // Orthogonal transform so far.
    var r = clone(x);  // Transformed matrix so far.
    var one2D = tensor2d([[1]], [1, 1]);
    var w = clone(one2D);
    var iters = m >= n ? n : m;
    var _loop_1 = function(j) {
      var _a;
      // This tidy within the for-loop ensures we clean up temporary
      // tensors as soon as they are no longer needed.
      var rTemp = r;
      var wTemp = w;
      var qTemp = q;
      _a = ENGINE.tidy(function() {
        // Find H = I - tau * w * w', to put zeros below R(j, j).
        var rjEnd1 = slice(r, [j, j], [m - j, 1]);
        var normX = norm(rjEnd1);
        var rjj = slice(r, [j, j], [1, 1]);
        // The sign() function returns 0 on 0, which causes division by zero.
        var s = where(greater(rjj, 0), tensor2d([[-1]]), tensor2d([[1]]));
        var u1 = sub(rjj, mul(s, normX));
        var wPre = div(rjEnd1, u1);
        if (wPre.shape[0] === 1) {
          w = clone(one2D);
        } else {
          w = concat(
              [one2D, slice(wPre, [1, 0], [wPre.shape[0] - 1, wPre.shape[1]])],
              0);
        }
        var tau = neg(div(matMul(s, u1), normX));
        // -- R := HR, Q := QH.
        var rjEndAll = slice(r, [j, 0], [m - j, n]);
        var tauTimesW = mul(tau, w);
        var wT = transpose(w);
        if (j === 0) {
          r = sub(rjEndAll, matMul(tauTimesW, matMul(wT, rjEndAll)));
        } else {
          var rTimesTau =
              sub(rjEndAll, matMul(tauTimesW, matMul(wT, rjEndAll)));
          r = concat([slice(r, [0, 0], [j, n]), rTimesTau], 0);
        }
        var tawTimesWT = transpose(tauTimesW);
        var qAllJEnd = slice(q, [0, j], [m, q.shape[1] - j]);
        if (j === 0) {
          q = sub(qAllJEnd, matMul(matMul(qAllJEnd, w), tawTimesWT));
        } else {
          var qTimesTau =
              sub(qAllJEnd, matMul(matMul(qAllJEnd, w), tawTimesWT));
          q = concat([slice(q, [0, 0], [m, j]), qTimesTau], 1);
        }
        return [w, r, q];
      }),
      w = _a[0], r = _a[1], q = _a[2];
      dispose([rTemp, wTemp, qTemp]);
    };
    for (var j = 0; j < iters; ++j) {
      _loop_1(j);
    }
    if (!fullMatrices && m > n) {
      q = slice(q, [0, 0], [m, n]);
      r = slice(r, [0, 0], [n, n]);
    }
    return [q, r];
  });
}
var qr = op({qr_: qr_});

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
(function(Reduction) {
Reduction[Reduction['NONE'] = 0] = 'NONE';
Reduction[Reduction['MEAN'] = 1] = 'MEAN';
Reduction[Reduction['SUM'] = 2] = 'SUM';
Reduction[Reduction['SUM_BY_NONZERO_WEIGHTS'] = 3] = 'SUM_BY_NONZERO_WEIGHTS';
})(exports.Reduction || (exports.Reduction = {}));

/**
 * Computes the weighted loss between two tensors.
 *
 * @param losses Tensor of shape `[batch_size, d1, ... dN]`.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `losses`, and must be broadcastable to `losses` (i.e., all
 *    dimensions must be either `1`, or the same as the corresponding
 *    `losses` dimension).
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function computeWeightedLoss_(losses, weights, reduction) {
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $losses = convertToTensor(losses, 'losses', 'computeWeightedLoss');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'computeWeightedLoss');
  }
  var weightedLoss = ($weights == null) ? $losses : mul($losses, $weights);
  if (reduction === exports.Reduction.NONE) {
    return weightedLoss;
  }
  if (reduction === exports.Reduction.SUM) {
    return sum$1(weightedLoss);
  }
  if (reduction === exports.Reduction.MEAN) {
    if ($weights == null) {
      return mean(weightedLoss);
    } else {
      var broadcastFactor = $losses.size / $weights.size;
      var result = div(sum$1(weightedLoss), sum$1($weights));
      return broadcastFactor > 1 ? div(result, scalar(broadcastFactor)) :
                                   result;
    }
  }
  if (reduction === exports.Reduction.SUM_BY_NONZERO_WEIGHTS) {
    if ($weights == null) {
      return div(sum$1(weightedLoss), scalar($losses.size));
    } else {
      var broadcastedWeights = mul($weights, ones$1($losses.shape));
      var numNonZeros =
          cast(sum$1(notEqual(broadcastedWeights, scalar(0))), 'float32');
      return div(sum$1(weightedLoss), numNonZeros);
    }
  }
  throw Error('Unknown reduction: ' + reduction);
}
var computeWeightedLoss = op({computeWeightedLoss_: computeWeightedLoss_});

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
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function absoluteDifference_(labels, predictions, weights, reduction) {
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $labels = convertToTensor(labels, 'labels', 'absoluteDifference');
  var $predictions =
      convertToTensor(predictions, 'predictions', 'absoluteDifference');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'absoluteDifference');
  }
  assertShapesMatch(
      $labels.shape, $predictions.shape, 'Error in absoluteDifference: ');
  var losses = abs(sub($labels, $predictions));
  return computeWeightedLoss(losses, $weights, reduction);
}
var absoluteDifference = op({absoluteDifference_: absoluteDifference_});

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
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function cosineDistance_(labels, predictions, axis, weights, reduction) {
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $labels = convertToTensor(labels, 'labels', 'cosineDistance');
  var $predictions =
      convertToTensor(predictions, 'predictions', 'cosineDistance');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'cosineDistance');
  }
  assertShapesMatch(
      $labels.shape, $predictions.shape, 'Error in cosineDistance: ');
  var one = scalar(1);
  var losses = sub(one, sum$1(mul($labels, $predictions), axis, true));
  return computeWeightedLoss(losses, $weights, reduction);
}
var cosineDistance = op({cosineDistance_: cosineDistance_});

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
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function hingeLoss_(labels, predictions, weights, reduction) {
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $labels = convertToTensor(labels, 'labels', 'hingeLoss');
  var $predictions = convertToTensor(predictions, 'predictions', 'hingeLoss');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'hingeLoss');
  }
  assertShapesMatch($labels.shape, $predictions.shape, 'Error in hingeLoss: ');
  var one = scalar(1);
  // Convert binary labels to (-1, 1)
  $labels = sub(mul(scalar(2), $labels), one);
  var losses = relu(sub(one, mul($labels, $predictions)));
  return computeWeightedLoss(losses, $weights, reduction);
}
var hingeLoss = op({hingeLoss_: hingeLoss_});

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
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function huberLoss_(labels, predictions, weights, delta, reduction) {
  if (delta === void 0) {
    delta = 1.0;
  }
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $labels = convertToTensor(labels, 'labels', 'huberLoss');
  var $predictions = convertToTensor(predictions, 'predictions', 'huberLoss');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'huberLoss');
  }
  assertShapesMatch($labels.shape, $predictions.shape, 'Error in huberLoss: ');
  var deltaScalar = scalar(delta);
  var error = abs(sub($predictions, $labels));
  var quadratic = minimum(error, deltaScalar);
  var linear = sub(error, quadratic);
  var losses =
      add$1(mul(scalar(0.5), square(quadratic)), mul(deltaScalar, linear));
  return computeWeightedLoss(losses, $weights, reduction);
}
var huberLoss = op({huberLoss_: huberLoss_});

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
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function logLoss_(labels, predictions, weights, epsilon, reduction) {
  if (epsilon === void 0) {
    epsilon = 1e-7;
  }
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $labels = convertToTensor(labels, 'labels', 'logLoss');
  var $predictions = convertToTensor(predictions, 'predictions', 'logLoss');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'logLoss');
  }
  assertShapesMatch($labels.shape, $predictions.shape, 'Error in logLoss: ');
  var one = scalar(1);
  var epsilonScalar = scalar(epsilon);
  var l1 = neg(mul($labels, log(add$1($predictions, epsilonScalar))));
  var l2 =
      mul(sub(one, $labels), log(add$1(sub(one, $predictions), epsilonScalar)));
  var losses = sub(l1, l2);
  return computeWeightedLoss(losses, $weights, reduction);
}
var logLoss = op({logLoss_: logLoss_});

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
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function meanSquaredError_(labels, predictions, weights, reduction) {
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $labels = convertToTensor(labels, 'labels', 'meanSquaredError');
  var $predictions =
      convertToTensor(predictions, 'predictions', 'meanSquaredError');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'meanSquaredError');
  }
  assertShapesMatch(
      $labels.shape, $predictions.shape, 'Error in meanSquaredError: ');
  var losses = squaredDifference($labels, $predictions);
  return computeWeightedLoss(losses, $weights, reduction);
}
var meanSquaredError = op({meanSquaredError_: meanSquaredError_});

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
function sigmoidCrossEntropyWithLogits_(labels, logits) {
  var $labels =
      convertToTensor(labels, 'labels', 'sigmoidCrossEntropyWithLogits');
  var $logits =
      convertToTensor(logits, 'logits', 'sigmoidCrossEntropyWithLogits');
  assertShapesMatch(
      $labels.shape, $logits.shape, 'Error in sigmoidCrossEntropyWithLogits: ');
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
  var maxOutput = relu($logits);
  var outputXTarget = mul($logits, $labels);
  var sigmoidOutput = log1p(exp(neg(abs($logits))));
  return add$1(sub(maxOutput, outputXTarget), sigmoidOutput);
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
 *
 * @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' }
 */
function sigmoidCrossEntropy_(
    multiClassLabels, logits, weights, labelSmoothing, reduction) {
  if (labelSmoothing === void 0) {
    labelSmoothing = 0;
  }
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $multiClassLabels = convertToTensor(
      multiClassLabels, 'multiClassLabels', 'sigmoidCrossEntropy');
  var $logits = convertToTensor(logits, 'logits', 'sigmoidCrossEntropy');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'sigmoidCrossEntropy');
  }
  assertShapesMatch(
      $multiClassLabels.shape, $logits.shape, 'Error in sigmoidCrossEntropy: ');
  if (labelSmoothing > 0) {
    var labelSmoothingScalar = scalar(labelSmoothing);
    var one = scalar(1);
    var half = scalar(0.5);
    $multiClassLabels = add$1(
        mul($multiClassLabels, sub(one, labelSmoothingScalar)),
        mul(half, labelSmoothingScalar));
  }
  var losses = sigmoidCrossEntropyWithLogits_($multiClassLabels, $logits);
  return computeWeightedLoss(losses, $weights, reduction);
}
var sigmoidCrossEntropy = op({sigmoidCrossEntropy_: sigmoidCrossEntropy_});

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
function softmaxCrossEntropyWithLogits_(labels, logits, dim) {
  if (dim === void 0) {
    dim = -1;
  }
  if (dim === -1) {
    dim = logits.rank - 1;
  }
  if (dim !== logits.rank - 1) {
    throw Error(
        'Softmax cross entropy along a non-last dimension is not yet ' +
        ('supported. Labels / logits was rank ' + logits.rank + ' ') +
        ('and dim was ' + dim));
  }
  // Use a custom gradient for numerical stability.
  var customOp = customGrad(function(labels, logits, save) {
    // Reference:
    //   1. http://cs231n.github.io/linear-classify/#softmax
    //   2. https://blog.feedly.com/tricks-of-the-trade-logsumexp/
    var keepDims = true;
    var lse = logSumExp(logits, [dim], keepDims);
    var logResult = sub(cast(logits, 'float32'), lse);
    save([labels, logResult]);
    var costVector = neg(mul(logResult, labels));
    var value = sum$1(costVector, [dim]);
    var gradFunc = function(dy, saved) {
      var labels = saved[0], logResult = saved[1];
      var dyShape = expandShapeToKeepDim(dy.shape, [dim]);
      return [
        mul(reshape(dy, dyShape), sub(cast(labels, 'float32'), exp(logResult))),
        mul(reshape(dy, dyShape), sub(exp(logResult), cast(labels, 'float32'))),
      ];
    };
    return {value: value, gradFunc: gradFunc};
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
 *
 * @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' }
 */
function softmaxCrossEntropy_(
    onehotLabels, logits, weights, labelSmoothing, reduction) {
  if (labelSmoothing === void 0) {
    labelSmoothing = 0;
  }
  if (reduction === void 0) {
    reduction = exports.Reduction.SUM_BY_NONZERO_WEIGHTS;
  }
  var $onehotLabels =
      convertToTensor(onehotLabels, 'onehotLabels', 'softmaxCrossEntropy');
  var $logits = convertToTensor(logits, 'logits', 'softmaxCrossEntropy');
  var $weights = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'softmaxCrossEntropy');
  }
  assertShapesMatch(
      $onehotLabels.shape, $logits.shape, 'Error in softmaxCrossEntropy: ');
  if (labelSmoothing > 0) {
    var labelSmoothingScalar = scalar(labelSmoothing);
    var one = scalar(1);
    var numClasses = scalar($onehotLabels.shape[1]);
    $onehotLabels = add$1(
        mul($onehotLabels, sub(one, labelSmoothingScalar)),
        div(labelSmoothingScalar, numClasses));
  }
  var losses = softmaxCrossEntropyWithLogits_($onehotLabels, $logits);
  return computeWeightedLoss(losses, $weights, reduction);
}
var softmaxCrossEntropy = op({softmaxCrossEntropy_: softmaxCrossEntropy_});

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
var spectral = {fft: fft, ifft: ifft, rfft: rfft, irfft: irfft};
var signal = {
  hammingWindow: hammingWindow,
  hannWindow: hannWindow,
  frame: frame,
  stft: stft,
};
var image = {
  flipLeftRight: flipLeftRight,
  resizeNearestNeighbor: resizeNearestNeighbor,
  resizeBilinear: resizeBilinear,
  rotateWithOffset: rotateWithOffset,
  cropAndResize: cropAndResize,
  nonMaxSuppression: nonMaxSuppression,
  nonMaxSuppressionAsync: nonMaxSuppressionAsync,
  nonMaxSuppressionWithScore: nonMaxSuppressionWithScore,
  nonMaxSuppressionWithScoreAsync: nonMaxSuppressionWithScoreAsync,
  nonMaxSuppressionPadded: nonMaxSuppressionPadded,
  nonMaxSuppressionPaddedAsync: nonMaxSuppressionPaddedAsync
};
var linalg = {bandPart: bandPart, gramSchmidt: gramSchmidt, qr: qr};
var losses = {
  absoluteDifference: absoluteDifference,
  computeWeightedLoss: computeWeightedLoss,
  cosineDistance: cosineDistance,
  hingeLoss: hingeLoss,
  huberLoss: huberLoss,
  logLoss: logLoss,
  meanSquaredError: meanSquaredError,
  sigmoidCrossEntropy: sigmoidCrossEntropy,
  softmaxCrossEntropy: softmaxCrossEntropy
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
/** @doc {heading: 'Training', subheading: 'Classes', namespace: 'train'} */
var Optimizer = /** @class */ (function(_super) {
  __extends(Optimizer, _super);
  function Optimizer() {
    return _super !== null && _super.apply(this, arguments) || this;
  }
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
   *
   * @doc {heading: 'Training', subheading: 'Optimizers'}
   */
  Optimizer.prototype.minimize = function(f, returnCost, varList) {
    if (returnCost === void 0) {
      returnCost = false;
    }
    var _a = this.computeGradients(f, varList), value = _a.value,
        grads = _a.grads;
    if (varList != null) {
      var gradArray = varList.map(function(v) {
        return ({name: v.name, tensor: grads[v.name]});
      });
      this.applyGradients(gradArray);
    } else {
      this.applyGradients(grads);
    }
    // Dispose gradients.
    dispose(grads);
    if (returnCost) {
      return value;
    } else {
      value.dispose();
      return null;
    }
  };
  Object.defineProperty(Optimizer.prototype, 'iterations', {
    /**
     * The number of iterations that this optimizer instance has been invoked
     * for.
     */
    get: function() {
      if (this.iterations_ == null) {
        this.iterations_ = 0;
      }
      return this.iterations_;
    },
    enumerable: true,
    configurable: true
  });
  Optimizer.prototype.incrementIterations = function() {
    this.iterations_ = this.iterations + 1;
  };
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
   *
   * @doc {heading: 'Training', subheading: 'Optimizers'}
   */
  Optimizer.prototype.computeGradients = function(f, varList) {
    return variableGrads(f, varList);
  };
  /**
   * Dispose the variables (if any) owned by this optimizer instance.
   */
  Optimizer.prototype.dispose = function() {
    if (this.iterations_ != null) {
      dispose(this.iterations_);
    }
  };
  Optimizer.prototype.saveIterations = function() {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        if (this.iterations_ == null) {
          this.iterations_ = 0;
        }
        return [
          2 /*return*/, {
            name: 'iter',
            // TODO(cais): Use 'int64' type when available.
            tensor: scalar(this.iterations_, 'int32')
          }
        ];
      });
    });
  };
  Optimizer.prototype.getWeights = function() {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        throw new Error(
            'getWeights() is not implemented for this optimizer yet.');
      });
    });
  };
  Optimizer.prototype.setWeights = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        throw new Error(
            'setWeights() is not implemented for this optimizer class ' +
            ('' + this.getClassName()));
      });
    });
  };
  /**
   * Extract the first element of the weight values and set it
   * as the iterations counter variable of this instance of optimizer.
   *
   * @param weightValues
   * @returns Weight values with the first element consumed and excluded.
   */
  Optimizer.prototype.extractIterations = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      var _a;
      return __generator(this, function(_b) {
        switch (_b.label) {
          case 0:
            _a = this;
            return [4 /*yield*/, weightValues[0].tensor.data()];
          case 1:
            _a.iterations_ = (_b.sent())[0];
            return [2 /*return*/, weightValues.slice(1)];
        }
      });
    });
  };
  return Optimizer;
}(Serializable));
Object.defineProperty(Optimizer, Symbol.hasInstance, {
  value: function(instance) {
    return instance.minimize != null && instance.computeGradients != null &&
        instance.applyGradients != null;
  }
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
/** @doclink Optimizer */
var AdadeltaOptimizer = /** @class */ (function(_super) {
  __extends(AdadeltaOptimizer, _super);
  function AdadeltaOptimizer(learningRate, rho, epsilon) {
    if (epsilon === void 0) {
      epsilon = null;
    }
    var _this = _super.call(this) || this;
    _this.learningRate = learningRate;
    _this.rho = rho;
    _this.epsilon = epsilon;
    _this.accumulatedGrads = [];
    _this.accumulatedUpdates = [];
    if (epsilon == null) {
      _this.epsilon = ENGINE.backend.epsilon();
    }
    return _this;
  }
  AdadeltaOptimizer.prototype.applyGradients = function(variableGradients) {
    var _this = this;
    var variableNames = Array.isArray(variableGradients) ?
        variableGradients.map(function(item) {
          return item.name;
        }) :
        Object.keys(variableGradients);
    variableNames.forEach(function(name, i) {
      var value = ENGINE.registeredVariables[name];
      var trainable = false;
      if (_this.accumulatedGrads[i] == null) {
        _this.accumulatedGrads[i] = {
          originalName: name + '/accum_grad',
          variable: tidy(function() {
            return zerosLike(value).variable(trainable);
          })
        };
      }
      if (_this.accumulatedUpdates[i] == null) {
        _this.accumulatedUpdates[i] = {
          originalName: name + '/accum_var',
          variable: tidy(function() {
            return zerosLike(value).variable(trainable);
          })
        };
      }
      var gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }
      var accumulatedGrad = _this.accumulatedGrads[i].variable;
      var accumulatedUpdate = _this.accumulatedUpdates[i].variable;
      tidy(function() {
        var newAccumulatedGrad = add$1(
            mul(accumulatedGrad, _this.rho),
            mul(square(gradient), 1 - _this.rho));
        var updates =
            mul(div(sqrt(add$1(accumulatedUpdate, _this.epsilon)),
                    sqrt(add$1(accumulatedGrad, _this.epsilon))),
                gradient);
        var newAccumulatedUpdate = add$1(
            mul(accumulatedUpdate, _this.rho),
            mul(square(updates), 1 - _this.rho));
        accumulatedGrad.assign(newAccumulatedGrad);
        accumulatedUpdate.assign(newAccumulatedUpdate);
        var newValue = add$1(mul(updates, -_this.learningRate), value);
        value.assign(newValue);
      });
    });
    this.incrementIterations();
  };
  AdadeltaOptimizer.prototype.dispose = function() {
    if (this.accumulatedUpdates != null) {
      dispose(this.accumulatedGrads.map(function(v) {
        return v.variable;
      }));
      dispose(this.accumulatedUpdates.map(function(v) {
        return v.variable;
      }));
    }
  };
  AdadeltaOptimizer.prototype.getWeights = function() {
    return __awaiter(this, void 0, void 0, function() {
      var variables;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            variables = this.accumulatedGrads.concat(this.accumulatedUpdates);
            return [4 /*yield*/, this.saveIterations()];
          case 1:
            return [
              2 /*return*/, [_a.sent()].concat(variables.map(function(v) {
                return ({name: v.originalName, tensor: v.variable});
              }))
            ];
        }
      });
    });
  };
  AdadeltaOptimizer.prototype.setWeights = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      var variableCount, trainable;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.extractIterations(weightValues)];
          case 1:
            weightValues = _a.sent();
            variableCount = weightValues.length / 2;
            trainable = false;
            this.accumulatedGrads =
                weightValues.slice(0, variableCount).map(function(v) {
                  return ({
                    originalName: v.name,
                    variable: v.tensor.variable(trainable)
                  });
                });
            this.accumulatedUpdates =
                weightValues.slice(variableCount, variableCount * 2)
                    .map(function(v) {
                      return ({
                        originalName: v.name,
                        variable: v.tensor.variable(trainable)
                      });
                    });
            return [2 /*return*/];
        }
      });
    });
  };
  AdadeltaOptimizer.prototype.getConfig = function() {
    return {
      'learningRate': this.learningRate,
      'rho': this.rho,
      'epsilon': this.epsilon
    };
  };
  /** @nocollapse */
  AdadeltaOptimizer.fromConfig = function(cls, config) {
    return new cls(config['learningRate'], config['rho'], config['epsilon']);
  };
  /** @nocollapse */
  AdadeltaOptimizer.className =
      'Adadelta';  // Name matters for Python compatibility.
  return AdadeltaOptimizer;
}(Optimizer));
registerClass(AdadeltaOptimizer);

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
/** @doclink Optimizer */
var AdagradOptimizer = /** @class */ (function(_super) {
  __extends(AdagradOptimizer, _super);
  function AdagradOptimizer(learningRate, initialAccumulatorValue) {
    if (initialAccumulatorValue === void 0) {
      initialAccumulatorValue = 0.1;
    }
    var _this = _super.call(this) || this;
    _this.learningRate = learningRate;
    _this.initialAccumulatorValue = initialAccumulatorValue;
    _this.accumulatedGrads = [];
    return _this;
  }
  AdagradOptimizer.prototype.applyGradients = function(variableGradients) {
    var _this = this;
    var variableNames = Array.isArray(variableGradients) ?
        variableGradients.map(function(item) {
          return item.name;
        }) :
        Object.keys(variableGradients);
    variableNames.forEach(function(name, i) {
      var value = ENGINE.registeredVariables[name];
      if (_this.accumulatedGrads[i] == null) {
        var trainable_1 = false;
        _this.accumulatedGrads[i] = {
          originalName: name + '/accumulator',
          variable: tidy(function() {
            return fill(value.shape, _this.initialAccumulatorValue)
                .variable(trainable_1);
          })
        };
      }
      var gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }
      var accumulatedGrad = _this.accumulatedGrads[i].variable;
      tidy(function() {
        var newAccumulatedGrad = add$1(accumulatedGrad, square(gradient));
        accumulatedGrad.assign(newAccumulatedGrad);
        var newValue = add$1(
            mul(div(gradient,
                    sqrt(add$1(newAccumulatedGrad, ENGINE.backend.epsilon()))),
                -_this.learningRate),
            value);
        value.assign(newValue);
      });
    });
    this.incrementIterations();
  };
  AdagradOptimizer.prototype.dispose = function() {
    if (this.accumulatedGrads != null) {
      dispose(this.accumulatedGrads.map(function(v) {
        return v.variable;
      }));
    }
  };
  AdagradOptimizer.prototype.getWeights = function() {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.saveIterations()];
          case 1:
            // Order matters for Python compatibility.
            return [
              2 /*return*/,
              [_a.sent()].concat(this.accumulatedGrads.map(function(v) {
                return ({name: v.originalName, tensor: v.variable});
              }))
            ];
        }
      });
    });
  };
  AdagradOptimizer.prototype.setWeights = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      var trainable;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.extractIterations(weightValues)];
          case 1:
            weightValues = _a.sent();
            trainable = false;
            this.accumulatedGrads = weightValues.map(function(v) {
              return ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
              });
            });
            return [2 /*return*/];
        }
      });
    });
  };
  AdagradOptimizer.prototype.getConfig = function() {
    return {
      'learningRate': this.learningRate,
      'initialAccumulatorValue': this.initialAccumulatorValue,
    };
  };
  /** @nocollapse */
  AdagradOptimizer.fromConfig = function(cls, config) {
    return new cls(config['learningRate'], config['initialAccumulatorValue']);
  };
  /** @nocollapse */
  AdagradOptimizer.className =
      'Adagrad';  // Note: Name matters for Python compatibility.
  return AdagradOptimizer;
}(Optimizer));
registerClass(AdagradOptimizer);

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
var AdamOptimizer = /** @class */ (function(_super) {
  __extends(AdamOptimizer, _super);
  function AdamOptimizer(learningRate, beta1, beta2, epsilon) {
    if (epsilon === void 0) {
      epsilon = null;
    }
    var _this = _super.call(this) || this;
    _this.learningRate = learningRate;
    _this.beta1 = beta1;
    _this.beta2 = beta2;
    _this.epsilon = epsilon;
    _this.accumulatedFirstMoment = [];
    _this.accumulatedSecondMoment = [];
    tidy(function() {
      // accB* will be updated by batch.
      _this.accBeta1 = scalar(beta1).variable();
      _this.accBeta2 = scalar(beta2).variable();
    });
    if (epsilon == null) {
      _this.epsilon = ENGINE.backend.epsilon();
    }
    return _this;
  }
  AdamOptimizer.prototype.applyGradients = function(variableGradients) {
    var _this = this;
    var varNames = Array.isArray(variableGradients) ?
        variableGradients.map(function(v) {
          return v.name;
        }) :
        Object.keys(variableGradients);
    tidy(function() {
      var oneMinusAccBeta1 = sub(1, _this.accBeta1);
      var oneMinusAccBeta2 = sub(1, _this.accBeta2);
      varNames.forEach(function(name, i) {
        var value = ENGINE.registeredVariables[name];
        var trainable = false;
        if (_this.accumulatedFirstMoment[i] == null) {
          _this.accumulatedFirstMoment[i] = {
            originalName: name + '/m',
            variable: tidy(function() {
              return zerosLike(value).variable(trainable);
            })
          };
        }
        if (_this.accumulatedSecondMoment[i] == null) {
          _this.accumulatedSecondMoment[i] = {
            originalName: name + '/v',
            variable: tidy(function() {
              return zerosLike(value).variable(trainable);
            })
          };
        }
        var gradient = Array.isArray(variableGradients) ?
            variableGradients[i].tensor :
            variableGradients[name];
        if (gradient == null) {
          return;
        }
        var firstMoment = _this.accumulatedFirstMoment[i].variable;
        var secondMoment = _this.accumulatedSecondMoment[i].variable;
        var newFirstMoment = add$1(
            mul(firstMoment, _this.beta1), mul(gradient, 1 - _this.beta1));
        var newSecondMoment = add$1(
            mul(secondMoment, _this.beta2),
            mul(square(gradient), 1 - _this.beta2));
        var biasCorrectedFirstMoment = div(newFirstMoment, oneMinusAccBeta1);
        var biasCorrectedSecondMoment = div(newSecondMoment, oneMinusAccBeta2);
        firstMoment.assign(newFirstMoment);
        secondMoment.assign(newSecondMoment);
        var newValue = add$1(
            mul(div(biasCorrectedFirstMoment,
                    add$1(sqrt(biasCorrectedSecondMoment), _this.epsilon)),
                -_this.learningRate),
            value);
        value.assign(newValue);
      });
      _this.accBeta1.assign(mul(_this.accBeta1, _this.beta1));
      _this.accBeta2.assign(mul(_this.accBeta2, _this.beta2));
    });
    this.incrementIterations();
  };
  AdamOptimizer.prototype.dispose = function() {
    this.accBeta1.dispose();
    this.accBeta2.dispose();
    if (this.accumulatedFirstMoment != null) {
      dispose(this.accumulatedFirstMoment.map(function(v) {
        return v.variable;
      }));
    }
    if (this.accumulatedSecondMoment != null) {
      dispose(this.accumulatedSecondMoment.map(function(v) {
        return v.variable;
      }));
    }
  };
  AdamOptimizer.prototype.getWeights = function() {
    return __awaiter(this, void 0, void 0, function() {
      var variables;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            variables = this.accumulatedFirstMoment.concat(
                this.accumulatedSecondMoment);
            return [4 /*yield*/, this.saveIterations()];
          case 1:
            return [
              2 /*return*/, [_a.sent()].concat(variables.map(function(v) {
                return ({name: v.originalName, tensor: v.variable});
              }))
            ];
        }
      });
    });
  };
  AdamOptimizer.prototype.setWeights = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      var variableCount, trainable;
      var _this = this;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.extractIterations(weightValues)];
          case 1:
            weightValues = _a.sent();
            tidy(function() {
              _this.accBeta1.assign(pow(_this.beta1, _this.iterations_ + 1));
              _this.accBeta2.assign(pow(_this.beta2, _this.iterations_ + 1));
            });
            variableCount = weightValues.length / 2;
            trainable = false;
            this.accumulatedFirstMoment =
                weightValues.slice(0, variableCount).map(function(v) {
                  return ({
                    originalName: v.name,
                    variable: v.tensor.variable(trainable)
                  });
                });
            this.accumulatedSecondMoment =
                weightValues.slice(variableCount, variableCount * 2)
                    .map(function(v) {
                      return ({
                        originalName: v.name,
                        variable: v.tensor.variable(trainable)
                      });
                    });
            return [2 /*return*/];
        }
      });
    });
  };
  AdamOptimizer.prototype.getConfig = function() {
    return {
      'learningRate': this.learningRate,
      'beta1': this.beta1,
      'beta2': this.beta2,
      'epsilon': this.epsilon,
    };
  };
  /** @nocollapse */
  AdamOptimizer.fromConfig = function(cls, config) {
    return new cls(
        config['learningRate'], config['beta1'], config['beta2'],
        config['epsilon']);
  };
  /** @nocollapse */
  AdamOptimizer.className =
      'Adam';  // Note: Name matters for Python compatibility.
  return AdamOptimizer;
}(Optimizer));
registerClass(AdamOptimizer);

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
var AdamaxOptimizer = /** @class */ (function(_super) {
  __extends(AdamaxOptimizer, _super);
  function AdamaxOptimizer(learningRate, beta1, beta2, epsilon, decay) {
    if (epsilon === void 0) {
      epsilon = null;
    }
    if (decay === void 0) {
      decay = 0.0;
    }
    var _this = _super.call(this) || this;
    _this.learningRate = learningRate;
    _this.beta1 = beta1;
    _this.beta2 = beta2;
    _this.epsilon = epsilon;
    _this.decay = decay;
    _this.accumulatedFirstMoment = [];
    _this.accumulatedWeightedInfNorm = [];
    tidy(function() {
      _this.iteration = scalar(0).variable();
      _this.accBeta1 = scalar(beta1).variable();
    });
    if (epsilon == null) {
      _this.epsilon = ENGINE.backend.epsilon();
    }
    return _this;
  }
  AdamaxOptimizer.prototype.applyGradients = function(variableGradients) {
    var _this = this;
    var variableNames = Array.isArray(variableGradients) ?
        variableGradients.map(function(item) {
          return item.name;
        }) :
        Object.keys(variableGradients);
    tidy(function() {
      var oneMinusAccBeta1 = sub(1, _this.accBeta1);
      var lr =
          div(-_this.learningRate, add$1(mul(_this.iteration, _this.decay), 1));
      variableNames.forEach(function(name, i) {
        var value = ENGINE.registeredVariables[name];
        var trainable = false;
        if (_this.accumulatedFirstMoment[i] == null) {
          _this.accumulatedFirstMoment[i] = {
            originalName: name + '/m',
            variable: zerosLike(value).variable(trainable)
          };
        }
        if (_this.accumulatedWeightedInfNorm[i] == null) {
          _this.accumulatedWeightedInfNorm[i] = {
            originalName: name + '/v',
            variable: zerosLike(value).variable(trainable)
          };
        }
        var gradient = Array.isArray(variableGradients) ?
            variableGradients[i].tensor :
            variableGradients[name];
        if (gradient == null) {
          return;
        }
        var firstMoment = _this.accumulatedFirstMoment[i].variable;
        var weightedInfNorm = _this.accumulatedWeightedInfNorm[i].variable;
        var newFirstMoment = add$1(
            mul(firstMoment, _this.beta1), mul(gradient, 1 - _this.beta1));
        var ut0 = mul(weightedInfNorm, _this.beta2);
        var ut1 = abs(gradient);
        var newWeightedInfNorm = maximum(ut0, ut1);
        firstMoment.assign(newFirstMoment);
        weightedInfNorm.assign(newWeightedInfNorm);
        var newValue = add$1(
            mul(div(lr, oneMinusAccBeta1),
                div(newFirstMoment, add$1(newWeightedInfNorm, _this.epsilon))),
            value);
        value.assign(newValue);
      });
      _this.iteration.assign(add$1(_this.iteration, 1));
      _this.accBeta1.assign(mul(_this.accBeta1, _this.beta1));
    });
    this.incrementIterations();
  };
  AdamaxOptimizer.prototype.dispose = function() {
    this.accBeta1.dispose();
    this.iteration.dispose();
    if (this.accumulatedFirstMoment != null) {
      dispose(this.accumulatedFirstMoment.map(function(v) {
        return v.variable;
      }));
    }
    if (this.accumulatedWeightedInfNorm != null) {
      dispose(this.accumulatedWeightedInfNorm.map(function(v) {
        return v.variable;
      }));
    }
  };
  AdamaxOptimizer.prototype.getWeights = function() {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        throw new Error('getWeights() is not implemented for Adamax yet.');
      });
    });
  };
  AdamaxOptimizer.prototype.setWeights = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        throw new Error('setWeights() is not implemented for Adamax yet.');
      });
    });
  };
  AdamaxOptimizer.prototype.getConfig = function() {
    return {
      'learningRate': this.learningRate,
      'beta1': this.beta1,
      'beta2': this.beta2,
      'epsilon': this.epsilon,
      'decay': this.decay
    };
  };
  /** @nocollapse */
  AdamaxOptimizer.fromConfig = function(cls, config) {
    return new cls(
        config['learningRate'], config['beta1'], config['beta2'],
        config['epsilon'], config['decay']);
  };
  /** @nocollapse */
  AdamaxOptimizer.className =
      'Adamax';  // Note: Name matters for Python compatbility.
  return AdamaxOptimizer;
}(Optimizer));
registerClass(AdamaxOptimizer);

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
/** @doclink Optimizer */
var SGDOptimizer = /** @class */ (function(_super) {
  __extends(SGDOptimizer, _super);
  function SGDOptimizer(learningRate) {
    var _this = _super.call(this) || this;
    _this.learningRate = learningRate;
    _this.setLearningRate(learningRate);
    return _this;
  }
  SGDOptimizer.prototype.applyGradients = function(variableGradients) {
    var _this = this;
    var varNames = Array.isArray(variableGradients) ?
        variableGradients.map(function(v) {
          return v.name;
        }) :
        Object.keys(variableGradients);
    varNames.forEach(function(name, i) {
      var gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }
      var value = ENGINE.registeredVariables[name];
      tidy(function() {
        var newValue = add$1(mul(_this.c, gradient), value);
        value.assign(newValue);
      });
    });
    this.incrementIterations();
  };
  /**
   * Sets the learning rate of the optimizer.
   */
  SGDOptimizer.prototype.setLearningRate = function(learningRate) {
    this.learningRate = learningRate;
    if (this.c != null) {
      this.c.dispose();
    }
    this.c = keep(scalar(-learningRate));
  };
  SGDOptimizer.prototype.dispose = function() {
    this.c.dispose();
  };
  SGDOptimizer.prototype.getWeights = function() {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.saveIterations()];
          case 1:
            return [2 /*return*/, [_a.sent()]];
        }
      });
    });
  };
  SGDOptimizer.prototype.setWeights = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.extractIterations(weightValues)];
          case 1:
            weightValues = _a.sent();
            if (weightValues.length !== 0) {
              throw new Error('SGD optimizer does not have settable weights.');
            }
            return [2 /*return*/];
        }
      });
    });
  };
  SGDOptimizer.prototype.getConfig = function() {
    return {'learningRate': this.learningRate};
  };
  /** @nocollapse */
  SGDOptimizer.fromConfig = function(cls, config) {
    return new cls(config['learningRate']);
  };
  /** @nocollapse */
  SGDOptimizer.className =
      'SGD';  // Note: Name matters for Python compatibility.
  return SGDOptimizer;
}(Optimizer));
registerClass(SGDOptimizer);

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
/** @doclink Optimizer */
var MomentumOptimizer = /** @class */ (function(_super) {
  __extends(MomentumOptimizer, _super);
  function MomentumOptimizer(learningRate, momentum, useNesterov) {
    if (useNesterov === void 0) {
      useNesterov = false;
    }
    var _this = _super.call(this, learningRate) || this;
    _this.learningRate = learningRate;
    _this.momentum = momentum;
    _this.useNesterov = useNesterov;
    _this.accumulations = [];
    _this.m = scalar(_this.momentum);
    return _this;
  }
  MomentumOptimizer.prototype.applyGradients = function(variableGradients) {
    var _this = this;
    var variableNames = Array.isArray(variableGradients) ?
        variableGradients.map(function(item) {
          return item.name;
        }) :
        Object.keys(variableGradients);
    variableNames.forEach(function(name, i) {
      var value = ENGINE.registeredVariables[name];
      if (_this.accumulations[i] == null) {
        var trainable_1 = false;
        _this.accumulations[i] = {
          originalName: name + '/momentum',
          variable: tidy(function() {
            return zerosLike(value).variable(trainable_1);
          })
        };
      }
      var accumulation = _this.accumulations[i].variable;
      var gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }
      tidy(function() {
        var newValue;
        var newAccumulation = add$1(mul(_this.m, accumulation), gradient);
        if (_this.useNesterov) {
          newValue = add$1(
              mul(_this.c, add$1(gradient, mul(newAccumulation, _this.m))),
              value);
        } else {
          newValue = add$1(mul(_this.c, newAccumulation), value);
        }
        accumulation.assign(newAccumulation);
        value.assign(newValue);
      });
    });
    this.incrementIterations();
  };
  MomentumOptimizer.prototype.dispose = function() {
    this.m.dispose();
    if (this.accumulations != null) {
      dispose(this.accumulations.map(function(v) {
        return v.variable;
      }));
    }
  };
  /**
   * Sets the momentum of the optimizer.
   *
   * @param momentum
   */
  MomentumOptimizer.prototype.setMomentum = function(momentum) {
    this.momentum = momentum;
  };
  MomentumOptimizer.prototype.getWeights = function() {
    return __awaiter(this, void 0, void 0, function() {
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.saveIterations()];
          case 1:
            // Order matters for Python compatibility.
            return [
              2 /*return*/,
              [_a.sent()].concat(this.accumulations.map(function(v) {
                return ({name: v.originalName, tensor: v.variable});
              }))
            ];
        }
      });
    });
  };
  MomentumOptimizer.prototype.setWeights = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      var trainable;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.extractIterations(weightValues)];
          case 1:
            weightValues = _a.sent();
            trainable = false;
            this.accumulations = weightValues.map(function(v) {
              return ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
              });
            });
            return [2 /*return*/];
        }
      });
    });
  };
  MomentumOptimizer.prototype.getConfig = function() {
    return {
      'learningRate': this.learningRate,
      'momentum': this.momentum,
      'useNesterov': this.useNesterov
    };
  };
  /** @nocollapse */
  MomentumOptimizer.fromConfig = function(cls, config) {
    return new cls(
        config['learningRate'], config['momentum'], config['useNesterov']);
  };
  /** @nocollapse */
  MomentumOptimizer.className =
      'Momentum';  // Name matters for Python compatibility.
  return MomentumOptimizer;
}(SGDOptimizer));
registerClass(MomentumOptimizer);

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
/** @doclink Optimizer */
var RMSPropOptimizer = /** @class */ (function(_super) {
  __extends(RMSPropOptimizer, _super);
  function RMSPropOptimizer(learningRate, decay, momentum, epsilon, centered) {
    if (decay === void 0) {
      decay = 0.9;
    }
    if (momentum === void 0) {
      momentum = 0.0;
    }
    if (epsilon === void 0) {
      epsilon = null;
    }
    if (centered === void 0) {
      centered = false;
    }
    var _this = _super.call(this) || this;
    _this.learningRate = learningRate;
    _this.decay = decay;
    _this.momentum = momentum;
    _this.epsilon = epsilon;
    _this.accumulatedMeanSquares = [];
    _this.accumulatedMoments = [];
    _this.accumulatedMeanGrads = [];
    _this.centered = centered;
    if (epsilon == null) {
      _this.epsilon = ENGINE.backend.epsilon();
    }
    if (learningRate == null) {
      throw new Error('learningRate for RMSPropOptimizer must be defined.');
    }
    return _this;
  }
  RMSPropOptimizer.prototype.applyGradients = function(variableGradients) {
    var _this = this;
    var variableNames = Array.isArray(variableGradients) ?
        variableGradients.map(function(item) {
          return item.name;
        }) :
        Object.keys(variableGradients);
    variableNames.forEach(function(name, i) {
      var value = ENGINE.registeredVariables[name];
      var trainable = false;
      if (_this.accumulatedMeanSquares[i] == null) {
        _this.accumulatedMeanSquares[i] = {
          originalName: name + '/rms',
          variable: tidy(function() {
            return zerosLike(value).variable(trainable);
          })
        };
      }
      if (_this.accumulatedMoments[i] == null) {
        _this.accumulatedMoments[i] = {
          originalName: name + '/momentum',
          variable: tidy(function() {
            return zerosLike(value).variable(trainable);
          })
        };
      }
      if (_this.accumulatedMeanGrads[i] == null && _this.centered) {
        _this.accumulatedMeanGrads[i] = {
          originalName: name + '/mg',
          variable: tidy(function() {
            return zerosLike(value).variable(trainable);
          })
        };
      }
      var gradient = Array.isArray(variableGradients) ?
          variableGradients[i].tensor :
          variableGradients[name];
      if (gradient == null) {
        return;
      }
      var accumulatedMeanSquare = _this.accumulatedMeanSquares[i].variable;
      var accumulatedMoments = _this.accumulatedMoments[i].variable;
      tidy(function() {
        var newAccumulatedMeanSquare = add$1(
            mul(accumulatedMeanSquare, _this.decay),
            mul(square(gradient), 1 - _this.decay));
        if (_this.centered) {
          var accumulatedMeanGrad = _this.accumulatedMeanGrads[i].variable;
          // Centered gradient
          var newAccumulatedMeanGrad = add$1(
              mul(accumulatedMeanGrad, _this.decay),
              mul(gradient, 1 - _this.decay));
          var gradContribution =
              div(mul(gradient, _this.learningRate),
                  sqrt(sub(
                      newAccumulatedMeanSquare,
                      add$1(square(newAccumulatedMeanGrad), _this.epsilon))));
          var newAccumulatedMoments =
              add$1(mul(accumulatedMoments, _this.momentum), gradContribution);
          accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
          accumulatedMeanGrad.assign(newAccumulatedMeanGrad);
          accumulatedMoments.assign(newAccumulatedMoments);
          var newValue = sub(value, newAccumulatedMoments);
          value.assign(newValue);
        } else {
          // Plain gradient
          var newAccumulatedMeanSquare_1 = add$1(
              mul(accumulatedMeanSquare, _this.decay),
              mul(square(gradient), 1 - _this.decay));
          var newAccumulatedMoments = add$1(
              mul(accumulatedMoments, _this.momentum),
              div(mul(gradient, _this.learningRate),
                  sqrt(add$1(newAccumulatedMeanSquare_1, _this.epsilon))));
          accumulatedMeanSquare.assign(newAccumulatedMeanSquare_1);
          accumulatedMoments.assign(newAccumulatedMoments);
          var newValue = sub(value, newAccumulatedMoments);
          value.assign(newValue);
        }
      });
    });
    this.incrementIterations();
  };
  RMSPropOptimizer.prototype.dispose = function() {
    if (this.accumulatedMeanSquares != null) {
      dispose(this.accumulatedMeanSquares.map(function(v) {
        return v.variable;
      }));
    }
    if (this.accumulatedMeanGrads != null && this.centered) {
      dispose(this.accumulatedMeanGrads.map(function(v) {
        return v.variable;
      }));
    }
    if (this.accumulatedMoments != null) {
      dispose(this.accumulatedMoments.map(function(v) {
        return v.variable;
      }));
    }
  };
  RMSPropOptimizer.prototype.getWeights = function() {
    return __awaiter(this, void 0, void 0, function() {
      var variables;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            variables =
                this.accumulatedMeanSquares.concat(this.accumulatedMoments);
            if (this.centered) {
              variables.push.apply(variables, this.accumulatedMeanGrads);
            }
            return [4 /*yield*/, this.saveIterations()];
          case 1:
            return [
              2 /*return*/, [_a.sent()].concat(variables.map(function(v) {
                return ({name: v.originalName, tensor: v.variable});
              }))
            ];
        }
      });
    });
  };
  RMSPropOptimizer.prototype.setWeights = function(weightValues) {
    return __awaiter(this, void 0, void 0, function() {
      var variableCount, trainable;
      return __generator(this, function(_a) {
        switch (_a.label) {
          case 0:
            return [4 /*yield*/, this.extractIterations(weightValues)];
          case 1:
            weightValues = _a.sent();
            variableCount = this.centered ? weightValues.length / 3 :
                                            weightValues.length / 2;
            trainable = false;
            this.accumulatedMeanSquares =
                weightValues.slice(0, variableCount).map(function(v) {
                  return ({
                    originalName: v.name,
                    variable: v.tensor.variable(trainable)
                  });
                });
            this.accumulatedMoments =
                weightValues.slice(variableCount, variableCount * 2)
                    .map(function(v) {
                      return ({
                        originalName: v.name,
                        variable: v.tensor.variable(trainable)
                      });
                    });
            if (this.centered) {
              this.accumulatedMeanGrads =
                  weightValues.slice(variableCount * 2, variableCount * 3)
                      .map(function(v) {
                        return ({
                          originalName: v.name,
                          variable: v.tensor.variable(trainable)
                        });
                      });
            }
            return [2 /*return*/];
        }
      });
    });
  };
  RMSPropOptimizer.prototype.getConfig = function() {
    return {
      'learningRate': this.learningRate,
      'decay': this.decay,
      'momentum': this.momentum,
      'epsilon': this.epsilon,
      'centered': this.centered
    };
  };
  /** @nocollapse */
  RMSPropOptimizer.fromConfig = function(cls, config) {
    return new cls(
        config['learningRate'], config['decay'], config['momentum'],
        config['epsilon'], config['centered']);
  };
  /** @nocollapse */
  RMSPropOptimizer.className =
      'RMSProp';  // Note: Name matters for Python compatibility.
  return RMSPropOptimizer;
}(Optimizer));
registerClass(RMSPropOptimizer);

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
var OptimizerConstructors = /** @class */ (function() {
  function OptimizerConstructors() {}
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
   *
   * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
   */
  OptimizerConstructors.sgd = function(learningRate) {
    return new SGDOptimizer(learningRate);
  };
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
   *
   * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
   */
  OptimizerConstructors.momentum = function(
      learningRate, momentum, useNesterov) {
    if (useNesterov === void 0) {
      useNesterov = false;
    }
    return new MomentumOptimizer(learningRate, momentum, useNesterov);
  };
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
   *
   * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
   */
  OptimizerConstructors.rmsprop = function(
      learningRate, decay, momentum, epsilon, centered) {
    if (decay === void 0) {
      decay = .9;
    }
    if (momentum === void 0) {
      momentum = 0.0;
    }
    if (epsilon === void 0) {
      epsilon = null;
    }
    if (centered === void 0) {
      centered = false;
    }
    return new RMSPropOptimizer(
        learningRate, decay, momentum, epsilon, centered);
  };
  /**
   * Constructs a `tf.AdamOptimizer` that uses the Adam algorithm.
   * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
   *
   * @param learningRate The learning rate to use for the Adam gradient
   * descent algorithm.
   * @param beta1 The exponential decay rate for the 1st moment estimates.
   * @param beta2 The exponential decay rate for the 2nd moment estimates.
   * @param epsilon A small constant for numerical stability.
   *
   * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
   */
  OptimizerConstructors.adam = function(learningRate, beta1, beta2, epsilon) {
    if (learningRate === void 0) {
      learningRate = 0.001;
    }
    if (beta1 === void 0) {
      beta1 = 0.9;
    }
    if (beta2 === void 0) {
      beta2 = 0.999;
    }
    if (epsilon === void 0) {
      epsilon = null;
    }
    return new AdamOptimizer(learningRate, beta1, beta2, epsilon);
  };
  /**
   * Constructs a `tf.AdadeltaOptimizer` that uses the Adadelta algorithm.
   * See [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
   *
   * @param learningRate The learning rate to use for the Adadelta gradient
   * descent algorithm.
   * @param rho The learning rate decay over each update.
   * @param epsilon A constant epsilon used to better condition the grad
   * update.
   *
   * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
   */
  OptimizerConstructors.adadelta = function(learningRate, rho, epsilon) {
    if (learningRate === void 0) {
      learningRate = .001;
    }
    if (rho === void 0) {
      rho = .95;
    }
    if (epsilon === void 0) {
      epsilon = null;
    }
    return new AdadeltaOptimizer(learningRate, rho, epsilon);
  };
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
   *
   * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
   */
  OptimizerConstructors.adamax = function(
      learningRate, beta1, beta2, epsilon, decay) {
    if (learningRate === void 0) {
      learningRate = 0.002;
    }
    if (beta1 === void 0) {
      beta1 = 0.9;
    }
    if (beta2 === void 0) {
      beta2 = 0.999;
    }
    if (epsilon === void 0) {
      epsilon = null;
    }
    if (decay === void 0) {
      decay = 0.0;
    }
    return new AdamaxOptimizer(learningRate, beta1, beta2, epsilon, decay);
  };
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
   *
   * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
   */
  OptimizerConstructors.adagrad = function(
      learningRate, initialAccumulatorValue) {
    if (initialAccumulatorValue === void 0) {
      initialAccumulatorValue = 0.1;
    }
    return new AdagradOptimizer(learningRate, initialAccumulatorValue);
  };
  return OptimizerConstructors;
}());

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
var train = {
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
 * Copyright 2017 Google LLC. All Rights Reserved.
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
var delayCallback = (function() {
  if (typeof requestAnimationFrame !== 'undefined') {
    return requestAnimationFrame;
  } else if (typeof setImmediate !== 'undefined') {
    return setImmediate;
  }
  return function(f) {
    return f();
  };  // no delays
})();
/**
 * Returns a promise that resolve when a requestAnimationFrame has completed.
 *
 * On Node.js this uses setImmediate instead of requestAnimationFrame.
 *
 * This is simply a sugar method so that users can do the following:
 * `await tf.nextFrame();`
 *
 * @doc {heading: 'Performance', subheading: 'Timing'}
 */
function nextFrame() {
  return new Promise(function(resolve) {
    return delayCallback(function() {
      return resolve();
    });
  });
}

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
  var rank = shapes[0].length;
  shapes.forEach(function(shape, i) {
    assert(shape.length === rank, function() {
      return 'Error in concat' + rank + 'D: rank of tensors[' + i +
          '] must be the same ' + ('as the rank of the rest (' + rank + ')');
    });
  });
  assert(axis >= 0 && axis < rank, function() {
    return 'Error in concat' + rank + 'D: axis must be between 0 and ' +
        (rank - 1) + '.';
  });
  var firstShape = shapes[0];
  shapes.forEach(function(shape, i) {
    for (var r = 0; r < rank; r++) {
      assert((r === axis) || (shape[r] === firstShape[r]), function() {
        return 'Error in concat' + rank + 'D: Shape of tensors[' + i + '] (' +
            shape + ') ' +
            ('does not match the shape of the rest (' + firstShape + ') ') +
            ('along the non-concatenated axis ' + i + '.');
      });
    }
  });
}
function computeOutShape$1(shapes, axis) {
  var outputShape = shapes[0].slice();
  for (var i = 1; i < shapes.length; i++) {
    outputShape[axis] += shapes[i][axis];
  }
  return outputShape;
}

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
var PARALLELIZE_THRESHOLD = 30;
function computeOptimalWindowSize(inSize) {
  if (inSize <= PARALLELIZE_THRESHOLD) {
    return inSize;
  }
  return nearestDivisor(inSize, Math.floor(Math.sqrt(inSize)));
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
// Returns the image center in pixels.
function getImageCenter(center, imageHeight, imageWidth) {
  var centerX = imageWidth * (typeof center === 'number' ? center : center[0]);
  var centerY = imageHeight * (typeof center === 'number' ? center : center[1]);
  return [centerX, centerY];
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
function getReshaped(inputShape, blockShape, prod, batchToSpace) {
  if (batchToSpace === void 0) {
    batchToSpace = true;
  }
  var reshaped = [];
  if (batchToSpace) {
    reshaped = reshaped.concat(blockShape.slice(0));
    reshaped.push(inputShape[0] / prod);
    reshaped = reshaped.concat(inputShape.slice(1));
  } else {
    reshaped = reshaped.concat(inputShape[0]);
    var spatialLength = blockShape.length;
    for (var i = 0; i < spatialLength; ++i) {
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
function getPermuted(reshapedRank, blockShapeRank, batchToSpace) {
  if (batchToSpace === void 0) {
    batchToSpace = true;
  }
  var permuted = [];
  if (batchToSpace) {
    permuted.push(blockShapeRank);
    for (var i = blockShapeRank + 1; i < reshapedRank; ++i) {
      if (i <= 2 * blockShapeRank) {
        permuted.push(i);
        permuted.push(i - (blockShapeRank + 1));
      } else {
        permuted.push(i);
      }
    }
  } else {
    var permutedBeforeBatch = [];
    var permutedAfterBatch = [];
    for (var i = 1; i < reshapedRank; ++i) {
      if (i >= blockShapeRank * 2 + 1 || i % 2 === 1) {
        permutedAfterBatch.push(i);
      } else {
        permutedBeforeBatch.push(i);
      }
    }
    permuted.push.apply(permuted, permutedBeforeBatch);
    permuted.push(0);
    permuted.push.apply(permuted, permutedAfterBatch);
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
function getReshapedPermuted(inputShape, blockShape, prod, batchToSpace) {
  if (batchToSpace === void 0) {
    batchToSpace = true;
  }
  var reshapedPermuted = [];
  if (batchToSpace) {
    reshapedPermuted.push(inputShape[0] / prod);
  } else {
    reshapedPermuted.push(inputShape[0] * prod);
  }
  for (var i = 1; i < inputShape.length; ++i) {
    if (i <= blockShape.length) {
      if (batchToSpace) {
        reshapedPermuted.push(blockShape[i - 1] * inputShape[i]);
      } else {
        reshapedPermuted.push(inputShape[i] / blockShape[i - 1]);
      }
    } else {
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
  var sliceBeginCoords = [0];
  for (var i = 0; i < blockShape; ++i) {
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
  var sliceSize = uncroppedShape.slice(0, 1);
  for (var i = 0; i < blockShape; ++i) {
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
var SELU_SCALEALPHA = 1.7580993408473768599402175208123;
var SELU_SCALE = 1.0507009873554804934193349852946;

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
var ERF_P = 0.3275911;
var ERF_A1 = 0.254829592;
var ERF_A2 = -0.284496736;
var ERF_A3 = 1.421413741;
var ERF_A4 = -1.453152027;
var ERF_A5 = 1.061405429;

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
function warn() {
  var msg = [];
  for (var _i = 0; _i < arguments.length; _i++) {
    msg[_i] = arguments[_i];
  }
  if (!env().getBool('IS_TEST')) {
    console.warn.apply(console, msg);
  }
}
function log$1() {
  var msg = [];
  for (var _i = 0; _i < arguments.length; _i++) {
    msg[_i] = arguments[_i];
  }
  if (!env().getBool('IS_TEST')) {
    console.log.apply(console, msg);
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
    throw new Error(
        'Cannot merge real and imag arrays of different lengths. real:' +
        (real.length + ', imag: ' + imag.length + '.'));
  }
  var result = new Float32Array(real.length * 2);
  for (var i = 0; i < result.length; i += 2) {
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
  var real = new Float32Array(complex.length / 2);
  var imag = new Float32Array(complex.length / 2);
  for (var i = 0; i < complex.length; i += 2) {
    real[i / 2] = complex[i];
    imag[i / 2] = complex[i + 1];
  }
  return {real: real, imag: imag};
}
/**
 * Extracts even indexed complex values in the given array.
 * @param complex The complex tensor values
 */
function complexWithEvenIndex(complex) {
  var len = Math.ceil(complex.length / 4);
  var real = new Float32Array(len);
  var imag = new Float32Array(len);
  for (var i = 0; i < complex.length; i += 4) {
    real[Math.floor(i / 4)] = complex[i];
    imag[Math.floor(i / 4)] = complex[i + 1];
  }
  return {real: real, imag: imag};
}
/**
 * Extracts odd indexed comple values in the given array.
 * @param complex The complex tensor values
 */
function complexWithOddIndex(complex) {
  var len = Math.floor(complex.length / 4);
  var real = new Float32Array(len);
  var imag = new Float32Array(len);
  for (var i = 2; i < complex.length; i += 4) {
    real[Math.floor(i / 4)] = complex[i];
    imag[Math.floor(i / 4)] = complex[i + 1];
  }
  return {real: real, imag: imag};
}
/**
 * Get the map representing a complex value in the given array.
 * @param complex The complex tensor values.
 * @param index An index of the target complex value.
 */
function getComplexWithIndex(complex, index) {
  var real = complex[index * 2];
  var imag = complex[index * 2 + 1];
  return {real: real, imag: imag};
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
  var real = new Float32Array(n / 2);
  var imag = new Float32Array(n / 2);
  for (var i = 0; i < Math.ceil(n / 2); i++) {
    var x = (inverse ? 2 : -2) * Math.PI * (i / n);
    real[i] = Math.cos(x);
    imag[i] = Math.sin(x);
  }
  return {real: real, imag: imag};
}
/**
 * Make the exponent term used by FFT.
 */
function exponent(k, n, inverse) {
  var x = (inverse ? 2 : -2) * Math.PI * (k / n);
  var real = Math.cos(x);
  var imag = Math.sin(x);
  return {real: real, imag: imag};
}

/**
 * Prepare the split size array. When the input is a number, the axis is evenly
 * divided among the split size. When the input contains the negative value, the
 * rest of the axis is allocated toward that.
 */
function prepareSplitSize(x, numOrSizeSplits, axis) {
  if (axis === void 0) {
    axis = 0;
  }
  var splitSizes = [];
  if (typeof (numOrSizeSplits) === 'number') {
    assert(x.shape[axis] % numOrSizeSplits === 0, function() {
      return 'Number of splits must evenly divide the axis.';
    });
    splitSizes =
        new Array(numOrSizeSplits).fill(x.shape[axis] / numOrSizeSplits);
  } else {
    var numOfNegs = numOrSizeSplits.reduce(function(count, value) {
      if (value === -1) {
        count += 1;
      }
      return count;
    }, 0);
    assert(numOfNegs <= 1, function() {
      return 'There should be only one negative value in split array.';
    });
    var negIndex = numOrSizeSplits.indexOf(-1);
    // Allow the number of split array to be -1, which indicates the rest
    // of dimension is allocated to that split.
    if (negIndex !== -1) {
      var total = numOrSizeSplits.reduce(function(a, b) {
        return b > 0 ? a + b : a;
      });
      numOrSizeSplits[negIndex] = x.shape[axis] - total;
    }
    assert(
        x.shape[axis] === numOrSizeSplits.reduce(function(a, b) {
          return a + b;
        }),
        function() {
          return 'The sum of sizes must match the size of the axis dimension.';
        });
    splitSizes = numOrSizeSplits;
  }
  return splitSizes;
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
function segOpComputeOptimalWindowSize(inSize, numSegments) {
  var done = false;
  var res;
  if (inSize <= PARALLELIZE_THRESHOLD) {
    res = inSize;
    done = true;
  } else {
    res = nearestDivisor(inSize, Math.floor(Math.sqrt(inSize)));
  }
  while (!done) {
    if (res > numSegments || res === inSize) {
      done = true;
    } else {
      res = nearestDivisor(inSize, res + 1);
    }
  }
  return res;
}
function computeOutShape$2(aShape, axis, numSegments) {
  var outShape = [];
  var rank = aShape.length;
  for (var dim = 0; dim < rank; dim++) {
    if (dim !== axis) {
      outShape.push(aShape[dim]);
    } else {
      outShape.push(numSegments);
    }
  }
  return outShape;
}
function collectGatherOpShapeInfo(x, indices, axis, batchDims) {
  var indicesRank = indices.shape.length;
  var xRank = x.shape.length;
  if (batchDims !== 0) {
    if (batchDims < -indicesRank || batchDims > indicesRank) {
      throw new Error(
          'Expect batchDims in the range of [-' + indicesRank + ', ' +
          indicesRank + '], but got ' + batchDims);
    }
  }
  if (batchDims < 0) {
    batchDims += indicesRank;
  }
  if (batchDims > xRank) {
    throw new Error(
        'batchDims (' + batchDims + ') must be less than rank(x) (\n    ' +
        xRank + ').');
  }
  if (axis < batchDims) {
    throw new Error(
        'batchDims (' + batchDims + ') must be less than or equal to axis (' +
        axis + ').');
  }
  for (var i = 0; i < batchDims; ++i) {
    if (x.shape[i] !== indices.shape[i]) {
      throw new Error(
          'x.shape[' + i + ']: ' + x.shape[i] +
          ' should be equal to indices.shape[' + i + ']: ' + indices.shape[i] +
          '.');
    }
  }
  var dimSize = x.shape[axis];
  var outputShape = [];
  var batchSize = 1;
  var outerSize = 1;
  var sliceSize = 1;
  for (var i = 0; i < batchDims; ++i) {
    outputShape.push(x.shape[i]);
    batchSize *= x.shape[i];
  }
  for (var i = batchDims; i < axis; i++) {
    outputShape.push(x.shape[i]);
    outerSize *= x.shape[i];
  }
  for (var i = batchDims; i < indicesRank; i++) {
    outputShape.push(indices.shape[i]);
  }
  for (var i = axis + 1; i < xRank; i++) {
    outputShape.push(x.shape[i]);
    sliceSize *= x.shape[i];
  }
  return {
    batchSize: batchSize,
    sliceSize: sliceSize,
    outerSize: outerSize,
    dimSize: dimSize,
    outputShape: outputShape
  };
}

var segment_util = {
  __proto__: null,
  segOpComputeOptimalWindowSize: segOpComputeOptimalWindowSize,
  computeOutShape: computeOutShape$2,
  collectGatherOpShapeInfo: collectGatherOpShapeInfo
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
function fromUint8ToStringArray(vals) {
  try {
    // Decode the bytes into string.
    return vals.map(function(val) {
      return decodeString(val);
    });
  } catch (err) {
    throw new Error(
        'Failed to decode encoded string bytes into utf-8, error: ' + err);
  }
}
function fromStringArrayToUint8(strings) {
  return strings.map(function(s) {
    return encodeString(s);
  });
}

var backend_util = {
  __proto__: null,
  slice_util: slice_util,
  segment_util: segment_util,
  fromUint8ToStringArray: fromUint8ToStringArray,
  fromStringArrayToUint8: fromStringArrayToUint8,
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
  computeOutShape: computeOutShape$1,
  computeDilation2DInfo: computeDilation2DInfo,
  computePool2DInfo: computePool2DInfo,
  computePool3DInfo: computePool3DInfo,
  computeConv2DInfo: computeConv2DInfo,
  computeConv3DInfo: computeConv3DInfo,
  computeDefaultPad: computeDefaultPad,
  tupleValuesAreOne: tupleValuesAreOne,
  eitherStridesOrDilationsAreOne: eitherStridesOrDilationsAreOne,
  convertConv2DDataFormat: convertConv2DDataFormat,
  getFusedDyActivation: getFusedDyActivation,
  getFusedBiasGradient: getFusedBiasGradient,
  applyActivation: applyActivation,
  shouldFuse: shouldFuse,
  PARALLELIZE_THRESHOLD: PARALLELIZE_THRESHOLD,
  computeOptimalWindowSize: computeOptimalWindowSize,
  getImageCenter: getImageCenter,
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
  exponent: exponent,
  prepareSplitSize: prepareSplitSize
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

var kernel_impls = {
  __proto__: null,
  nonMaxSuppressionV3Impl: nonMaxSuppressionV3Impl,
  nonMaxSuppressionV4Impl: nonMaxSuppressionV4Impl,
  nonMaxSuppressionV5Impl: nonMaxSuppressionV5Impl,
  whereImpl: whereImpl
};

exports.Abs = Abs;
exports.Acos = Acos;
exports.Acosh = Acosh;
exports.AdadeltaOptimizer = AdadeltaOptimizer;
exports.AdagradOptimizer = AdagradOptimizer;
exports.AdamOptimizer = AdamOptimizer;
exports.AdamaxOptimizer = AdamaxOptimizer;
exports.Add = Add;
exports.AddN = AddN;
exports.All = All;
exports.Any = Any;
exports.ArgMax = ArgMax;
exports.ArgMin = ArgMin;
exports.Asin = Asin;
exports.Asinh = Asinh;
exports.Atan = Atan;
exports.Atan2 = Atan2;
exports.Atanh = Atanh;
exports.AvgPool = AvgPool;
exports.AvgPool3D = AvgPool3D;
exports.AvgPool3DGrad = AvgPool3DGrad;
exports.AvgPoolGrad = AvgPoolGrad;
exports.BatchMatMul = BatchMatMul;
exports.BatchToSpaceND = BatchToSpaceND;
exports.Bincount = Bincount;
exports.BroadcastTo = BroadcastTo;
exports.Cast = Cast;
exports.Ceil = Ceil;
exports.ClipByValue = ClipByValue;
exports.Complex = Complex;
exports.ComplexAbs = ComplexAbs;
exports.Concat = Concat;
exports.Conv2D = Conv2D;
exports.Conv2DBackpropFilter = Conv2DBackpropFilter;
exports.Conv2DBackpropInput = Conv2DBackpropInput;
exports.Conv3D = Conv3D;
exports.Conv3DBackpropFilterV2 = Conv3DBackpropFilterV2;
exports.Conv3DBackpropInputV2 = Conv3DBackpropInputV2;
exports.Cos = Cos;
exports.Cosh = Cosh;
exports.CropAndResize = CropAndResize;
exports.Cumsum = Cumsum;
exports.DataStorage = DataStorage;
exports.DenseBincount = DenseBincount;
exports.DepthToSpace = DepthToSpace;
exports.DepthwiseConv2dNative = DepthwiseConv2dNative;
exports.DepthwiseConv2dNativeBackpropFilter =
    DepthwiseConv2dNativeBackpropFilter;
exports.DepthwiseConv2dNativeBackpropInput = DepthwiseConv2dNativeBackpropInput;
exports.Diag = Diag;
exports.Dilation2D = Dilation2D;
exports.Dilation2DBackpropFilter = Dilation2DBackpropFilter;
exports.Dilation2DBackpropInput = Dilation2DBackpropInput;
exports.Elu = Elu;
exports.EluGrad = EluGrad;
exports.Environment = Environment;
exports.Equal = Equal;
exports.Erf = Erf;
exports.Exp = Exp;
exports.ExpandDims = ExpandDims;
exports.Expm1 = Expm1;
exports.FFT = FFT;
exports.Fill = Fill;
exports.FlipLeftRight = FlipLeftRight;
exports.Floor = Floor;
exports.FloorDiv = FloorDiv;
exports.FromPixels = FromPixels;
exports.FusedBatchNorm = FusedBatchNorm;
exports.FusedConv2D = FusedConv2D;
exports.FusedDepthwiseConv2D = FusedDepthwiseConv2D;
exports.GatherNd = GatherNd;
exports.GatherV2 = GatherV2;
exports.Greater = Greater;
exports.GreaterEqual = GreaterEqual;
exports.IFFT = IFFT;
exports.Identity = Identity;
exports.Imag = Imag;
exports.IsFinite = IsFinite;
exports.IsInf = IsInf;
exports.IsNan = IsNan;
exports.KernelBackend = KernelBackend;
exports.LRN = LRN;
exports.LRNGrad = LRNGrad;
exports.LeakyRelu = LeakyRelu;
exports.Less = Less;
exports.LessEqual = LessEqual;
exports.LinSpace = LinSpace;
exports.Log = Log;
exports.Log1p = Log1p;
exports.LogSoftmax = LogSoftmax;
exports.LogicalAnd = LogicalAnd;
exports.LogicalNot = LogicalNot;
exports.LogicalOr = LogicalOr;
exports.Max = Max;
exports.MaxPool = MaxPool;
exports.MaxPool3D = MaxPool3D;
exports.MaxPool3DGrad = MaxPool3DGrad;
exports.MaxPoolGrad = MaxPoolGrad;
exports.MaxPoolWithArgmax = MaxPoolWithArgmax;
exports.Maximum = Maximum;
exports.Mean = Mean;
exports.Min = Min;
exports.Minimum = Minimum;
exports.MirrorPad = MirrorPad;
exports.Mod = Mod;
exports.MomentumOptimizer = MomentumOptimizer;
exports.Multinomial = Multinomial;
exports.Multiply = Multiply;
exports.Neg = Neg;
exports.NonMaxSuppressionV3 = NonMaxSuppressionV3;
exports.NonMaxSuppressionV4 = NonMaxSuppressionV4;
exports.NonMaxSuppressionV5 = NonMaxSuppressionV5;
exports.NotEqual = NotEqual;
exports.OP_SCOPE_SUFFIX = OP_SCOPE_SUFFIX;
exports.OneHot = OneHot;
exports.OnesLike = OnesLike;
exports.Optimizer = Optimizer;
exports.Pack = Pack;
exports.PadV2 = PadV2;
exports.Pool = Pool;
exports.Pow = Pow;
exports.Prelu = Prelu;
exports.Prod = Prod;
exports.RMSPropOptimizer = RMSPropOptimizer;
exports.Range = Range;
exports.Real = Real;
exports.RealDiv = RealDiv;
exports.Reciprocal = Reciprocal;
exports.Relu = Relu;
exports.Relu6 = Relu6;
exports.Reshape = Reshape;
exports.ResizeBilinear = ResizeBilinear;
exports.ResizeBilinearGrad = ResizeBilinearGrad;
exports.ResizeNearestNeighbor = ResizeNearestNeighbor;
exports.ResizeNearestNeighborGrad = ResizeNearestNeighborGrad;
exports.Reverse = Reverse;
exports.RotateWithOffset = RotateWithOffset;
exports.Round = Round;
exports.Rsqrt = Rsqrt;
exports.SGDOptimizer = SGDOptimizer;
exports.ScatterNd = ScatterNd;
exports.Select = Select;
exports.Selu = Selu;
exports.Sigmoid = Sigmoid;
exports.Sign = Sign;
exports.Sin = Sin;
exports.Sinh = Sinh;
exports.Slice = Slice;
exports.Softmax = Softmax;
exports.Softplus = Softplus;
exports.SpaceToBatchND = SpaceToBatchND;
exports.SparseToDense = SparseToDense;
exports.SplitV = SplitV;
exports.Sqrt = Sqrt;
exports.Square = Square;
exports.SquaredDifference = SquaredDifference;
exports.Step = Step;
exports.StridedSlice = StridedSlice;
exports.Sub = Sub;
exports.Sum = Sum;
exports.Tan = Tan;
exports.Tanh = Tanh;
exports.Tensor = Tensor;
exports.TensorBuffer = TensorBuffer;
exports.Tile = Tile;
exports.TopK = TopK;
exports.Transpose = Transpose;
exports.Unique = Unique;
exports.Unpack = Unpack;
exports.UnsortedSegmentSum = UnsortedSegmentSum;
exports.Variable = Variable;
exports.ZerosLike = ZerosLike;
exports._FusedMatMul = _FusedMatMul;
exports.abs = abs;
exports.acos = acos;
exports.acosh = acosh;
exports.add = add$1;
exports.addN = addN;
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
exports.batchToSpaceND = batchToSpaceND;
exports.bincount = bincount;
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
exports.copyRegisteredKernels = copyRegisteredKernels;
exports.cos = cos;
exports.cosh = cosh;
exports.cosineWindow = cosineWindow;
exports.cumsum = cumsum;
exports.customGrad = customGrad;
exports.denseBincount = denseBincount;
exports.deprecationWarn = deprecationWarn;
exports.depthToSpace = depthToSpace;
exports.depthwiseConv2d = depthwiseConv2d;
exports.device_util = device_util;
exports.diag = diag;
exports.dilation2d = dilation2d;
exports.disableDeprecationWarnings = disableDeprecationWarnings;
exports.dispose = dispose;
exports.disposeVariables = disposeVariables;
exports.div = div;
exports.divNoNan = divNoNan;
exports.dot = dot;
exports.dropout = dropout;
exports.elu = elu;
exports.enableDebugMode = enableDebugMode;
exports.enableProdMode = enableProdMode;
exports.enclosingPowerOfTwo = enclosingPowerOfTwo;
exports.engine = engine;
exports.env = env;
exports.equal = equal;
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
exports.ifft = ifft;
exports.imag = imag;
exports.image = image;
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
exports.linalg = linalg;
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
exports.losses = losses;
exports.matMul = matMul;
exports.math = math;
exports.max = max;
exports.maxPool = maxPool;
exports.maxPool3d = maxPool3d;
exports.maxPoolWithArgmax = maxPoolWithArgmax;
exports.maximum = maximum;
exports.mean = mean;
exports.memory = memory;
exports.min = min;
exports.minimum = minimum;
exports.mirrorPad = mirrorPad;
exports.mod = mod;
exports.moments = moments;
exports.movingAverage = movingAverage;
exports.mul = mul;
exports.multiRNNCell = multiRNNCell;
exports.multinomial = multinomial;
exports.neg = neg;
exports.nextFrame = nextFrame;
exports.norm = norm;
exports.notEqual = notEqual;
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
exports.round = round$1;
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
exports.signal = signal;
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
exports.spectral = spectral;
exports.split = split;
exports.sqrt = sqrt;
exports.square = square;
exports.squaredDifference = squaredDifference;
exports.squeeze = squeeze;
exports.stack = stack;
exports.step = step;
exports.stridedSlice = stridedSlice;
exports.sub = sub;
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
exports.unique = unique;
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

Object.defineProperty(exports, '__esModule', {value: true});
  })));
//# sourceMappingURL=tf-core.js.map
