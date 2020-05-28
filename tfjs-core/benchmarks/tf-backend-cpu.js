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
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core'), require('seedrandom')) :
  typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core', 'seedrandom'], factory) :
  (global = global || self, factory(global.tf = global.tf || {}, global.tf, global.seedrandom));
}(this, (function (exports, tf, seedrandom) { 'use strict';

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
  function maxImpl(aVals, reduceSize, outShape, dtype) {
      var vals = tf.util.getTypedArrayFromDType(dtype, tf.util.sizeFromShape(outShape));
      for (var i = 0; i < vals.length; ++i) {
          var offset = i * reduceSize;
          var max = aVals[offset];
          for (var j = 0; j < reduceSize; ++j) {
              var value = aVals[offset + j];
              if (value > max) {
                  max = value;
              }
          }
          vals[i] = max;
      }
      return vals;
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
  function transposeImpl(xVals, xShape, dtype, perm, newShape) {
      var xRank = xShape.length;
      var xSize = tf.util.sizeFromShape(xShape);
      var xStrides = tf.util.computeStrides(xShape);
      var newStrides = tf.util.computeStrides(newShape);
      var result = tf.util.getTypedArrayFromDType(dtype, tf.util.sizeFromShape(newShape));
      for (var i = 0; i < xSize; ++i) {
          var loc = tf.util.indexToLoc(i, xRank, xStrides);
          // Permute location.
          var newLoc = new Array(loc.length);
          for (var i_1 = 0; i_1 < newLoc.length; i_1++) {
              newLoc[i_1] = loc[perm[i_1]];
          }
          var newIndex = tf.util.locToIndex(newLoc, xRank, newStrides);
          result[newIndex] = xVals[i];
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

  var shared = {
    __proto__: null,
    maxImpl: maxImpl,
    transposeImpl: transposeImpl
  };

  /*! *****************************************************************************
  Copyright (c) Microsoft Corporation. All rights reserved.
  Licensed under the Apache License, Version 2.0 (the "License"); you may not use
  this file except in compliance with the License. You may obtain a copy of the
  License at http://www.apache.org/licenses/LICENSE-2.0

  THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
  WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
  MERCHANTABLITY OR NON-INFRINGEMENT.

  See the Apache Version 2.0 License for specific language governing permissions
  and limitations under the License.
  ***************************************************************************** */
  /* global Reflect, Promise */

  var extendStatics = function(d, b) {
      extendStatics = Object.setPrototypeOf ||
          ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
          function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
      return extendStatics(d, b);
  };

  function __extends(d, b) {
      extendStatics(d, b);
      function __() { this.constructor = d; }
      d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
  }

  function __awaiter(thisArg, _arguments, P, generator) {
      function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
      return new (P || (P = Promise))(function (resolve, reject) {
          function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
          function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
          function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
          step((generator = generator.apply(thisArg, _arguments || [])).next());
      });
  }

  function __generator(thisArg, body) {
      var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
      return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
      function verb(n) { return function (v) { return step([n, v]); }; }
      function step(op) {
          if (f) throw new TypeError("Generator is already executing.");
          while (_) try {
              if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
              if (y = 0, t) op = [op[0] & 2, t.value];
              switch (op[0]) {
                  case 0: case 1: t = op; break;
                  case 4: _.label++; return { value: op[1], done: false };
                  case 5: _.label++; y = op[1]; op = [0]; continue;
                  case 7: op = _.ops.pop(); _.trys.pop(); continue;
                  default:
                      if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                      if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                      if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                      if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                      if (t[2]) _.ops.pop();
                      _.trys.pop(); continue;
              }
              op = body.call(thisArg, _);
          } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
          if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
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
  function assertNotComplex(tensor, opName) {
      if (!Array.isArray(tensor)) {
          tensor = [tensor];
      }
      tensor.forEach(function (t) {
          if (t != null) {
              tf.util.assert(t.dtype !== 'complex64', function () { return opName + " does not support complex64 tensors in the CPU backend."; });
          }
      });
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
  function pool(xValues, xShape, dtype, strides, convInfo, poolType) {
      var strideHeight = convInfo.strideHeight;
      var strideWidth = convInfo.strideWidth;
      var dilationHeight = convInfo.dilationHeight;
      var dilationWidth = convInfo.dilationWidth;
      var effectiveFilterHeight = convInfo.effectiveFilterHeight;
      var effectiveFilterWidth = convInfo.effectiveFilterWidth;
      var padTop = convInfo.padInfo.top;
      var padLeft = convInfo.padInfo.left;
      var initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
          Number.POSITIVE_INFINITY);
      var output = tf.buffer(convInfo.outShape, dtype);
      var outputVals = output.values;
      var outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] * convInfo.outShape[3];
      var outputRowStrides = convInfo.outShape[2] * convInfo.outShape[3];
      var outputColStrides = convInfo.outShape[3];
      for (var b = 0; b < convInfo.batchSize; ++b) {
          var outputBatchOffset = b * outputBatchStrides;
          var inputBatchOffset = b * strides[0];
          for (var d = 0; d < convInfo.inChannels; ++d) {
              for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                  var xRCorner = yR * strideHeight - padTop;
                  var xRMin = Math.max(0, xRCorner);
                  var xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                  var outputRowOffset = outputBatchOffset + yR * outputRowStrides;
                  for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                      var xCCorner = yC * strideWidth - padLeft;
                      var xCMin = Math.max(0, xCCorner);
                      var xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                      var minMaxValue = initialValue;
                      var avgValue = 0;
                      var count = 0;
                      for (var xR = xRMin; xR < xRMax; xR += dilationHeight) {
                          var xROffset = inputBatchOffset + xR * strides[1];
                          for (var xC = xCMin; xC < xCMax; xC += dilationWidth) {
                              var xCOffset = xROffset + xC * strides[2];
                              var pixel = xValues[xCOffset + d];
                              if ((poolType === 'max' && pixel > minMaxValue)) {
                                  minMaxValue = pixel;
                              }
                              else if (poolType === 'avg') {
                                  avgValue += pixel;
                                  count++;
                              }
                          }
                          if (isNaN(minMaxValue)) {
                              break;
                          }
                      }
                      var outputOffset = outputRowOffset + yC * outputColStrides + d;
                      outputVals[outputOffset] =
                          poolType === 'avg' ? avgValue / count : minMaxValue;
                  }
              }
          }
      }
      return output;
  }
  function maxPoolPositions(xValues, xShape, dtype, convInfo, flattenPositions, includeBatchInIndex) {
      if (flattenPositions === void 0) { flattenPositions = false; }
      if (includeBatchInIndex === void 0) { includeBatchInIndex = false; }
      var maxPositions = tf.buffer(convInfo.outShape, 'int32');
      var strideHeight = convInfo.strideHeight;
      var strideWidth = convInfo.strideWidth;
      var dilationHeight = convInfo.dilationHeight;
      var dilationWidth = convInfo.dilationWidth;
      var effectiveFilterHeight = convInfo.effectiveFilterHeight;
      var effectiveFilterWidth = convInfo.effectiveFilterWidth;
      var padTop = convInfo.padInfo.top;
      var padLeft = convInfo.padInfo.left;
      var xBuf = tf.buffer(xShape, dtype, xValues);
      for (var b = 0; b < convInfo.batchSize; ++b) {
          for (var d = 0; d < convInfo.inChannels; ++d) {
              for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                  var xRCorner = yR * strideHeight - padTop;
                  var xRMin = xRCorner;
                  while (xRMin < 0) {
                      xRMin += dilationHeight;
                  }
                  // const xRMin = Math.max(0, xRCorner);
                  var xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                  for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                      var xCCorner = yC * strideWidth - padLeft;
                      var xCMin = xCCorner;
                      while (xCMin < 0) {
                          xCMin += dilationWidth;
                      }
                      var xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                      var maxValue = Number.NEGATIVE_INFINITY;
                      var maxPosition = -1;
                      for (var xR = xRMin; xR < xRMax; xR += dilationHeight) {
                          var wR = xR - xRCorner;
                          for (var xC = xCMin; xC < xCMax; xC += dilationWidth) {
                              var wC = xC - xCCorner;
                              var pixel = xBuf.get(b, xR, xC, d);
                              if (pixel > maxValue) {
                                  maxValue = pixel;
                                  if (flattenPositions) {
                                      maxPosition = includeBatchInIndex ?
                                          ((b * convInfo.inHeight + xR) * convInfo.inWidth + xC) *
                                              convInfo.inChannels +
                                              d :
                                          (xR * convInfo.inWidth + xC) * convInfo.inChannels + d;
                                  }
                                  else {
                                      maxPosition = wR * effectiveFilterWidth + wC;
                                  }
                              }
                          }
                      }
                      maxPositions.set(maxPosition, b, yR, yC, d);
                  }
              }
          }
      }
      return maxPositions;
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
  var nonMaxSuppressionV3 = tf.kernel_impls.nonMaxSuppressionV3;
  var split = tf.kernel_impls.split;
  var tile = tf.kernel_impls.tile;
  var topkImpl = tf.kernel_impls.topkImpl;
  var whereImpl = tf.kernel_impls.whereImpl;
  function mapActivation(backend, x, activation, preluActivationWeights) {
      if (activation === 'linear') {
          return backend.linear(x);
      }
      else if (activation === 'relu') {
          return backend.relu(x);
      }
      else if (activation === 'elu') {
          return backend.elu(x);
      }
      else if (activation === 'relu6') {
          return backend.relu6(x);
      }
      else if (activation === 'prelu') {
          return backend.prelu(x, preluActivationWeights);
      }
      throw new Error("Activation " + activation + " has not been implemented for the CPU backend.");
  }
  var MathBackendCPU = /** @class */ (function (_super) {
      __extends(MathBackendCPU, _super);
      function MathBackendCPU() {
          var _this = _super.call(this) || this;
          _this.blockSize = 48;
          _this.firstUse = true;
          _this.data = new tf.DataStorage(_this, tf.engine());
          return _this;
      }
      MathBackendCPU.prototype.write = function (values, shape, dtype) {
          if (this.firstUse) {
              this.firstUse = false;
              if (tf.env().get('IS_NODE')) {
                  tf.backend_util.warn('\n============================\n' +
                      'Hi there 👋. Looks like you are running TensorFlow.js in ' +
                      'Node.js. To speed things up dramatically, install our node ' +
                      'backend, which binds to TensorFlow C++, by running ' +
                      'npm i @tensorflow/tfjs-node, ' +
                      'or npm i @tensorflow/tfjs-node-gpu if you have CUDA. ' +
                      'Then call require(\'@tensorflow/tfjs-node\'); (-gpu ' +
                      'suffix for CUDA) at the start of your program. ' +
                      'Visit https://github.com/tensorflow/tfjs-node for more details.' +
                      '\n============================');
              }
          }
          var dataId = {};
          this.data.set(dataId, { values: values, dtype: dtype });
          return dataId;
      };
      MathBackendCPU.prototype.move = function (dataId, values, shape, dtype) {
          this.data.set(dataId, { values: values, dtype: dtype });
      };
      MathBackendCPU.prototype.numDataIds = function () {
          return this.data.numDataIds();
      };
      MathBackendCPU.prototype.read = function (dataId) {
          return __awaiter(this, void 0, void 0, function () {
              return __generator(this, function (_a) {
                  return [2 /*return*/, this.readSync(dataId)];
              });
          });
      };
      MathBackendCPU.prototype.readSync = function (dataId) {
          var _a = this.data.get(dataId), dtype = _a.dtype, complexTensors = _a.complexTensors;
          if (dtype === 'complex64') {
              var realValues = this.readSync(complexTensors.real.dataId);
              var imagValues = this.readSync(complexTensors.imag.dataId);
              return tf.backend_util.mergeRealAndImagArrays(realValues, imagValues);
          }
          return this.data.get(dataId).values;
      };
      MathBackendCPU.prototype.bufferSync = function (t) {
          var data = this.readSync(t.dataId);
          var decodedData = data;
          if (t.dtype === 'string') {
              try {
                  // Decode the bytes into string.
                  decodedData = data.map(function (d) { return tf.util.decodeString(d); });
              }
              catch (_a) {
                  throw new Error('Failed to decode encoded string bytes into utf-8');
              }
          }
          return tf.buffer(t.shape, t.dtype, decodedData);
      };
      MathBackendCPU.prototype.makeOutput = function (values, shape, dtype) {
          var dataId = this.write(values, shape, dtype);
          return tf.engine().makeTensorFromDataId(dataId, shape, dtype, this);
      };
      MathBackendCPU.prototype.disposeData = function (dataId) {
          if (this.data.has(dataId)) {
              var complexTensors = this.data.get(dataId).complexTensors;
              if (complexTensors != null) {
                  complexTensors.real.dispose();
                  complexTensors.imag.dispose();
              }
              this.data.delete(dataId);
          }
      };
      MathBackendCPU.prototype.time = function (f) {
          return __awaiter(this, void 0, void 0, function () {
              var start, kernelMs;
              return __generator(this, function (_a) {
                  start = tf.util.now();
                  f();
                  kernelMs = tf.util.now() - start;
                  return [2 /*return*/, { kernelMs: kernelMs }];
              });
          });
      };
      MathBackendCPU.prototype.memory = function () {
          return {
              // Unreliable due to automatic gc. The numbers above are cumulative.
              unreliable: true,
              reasons: ['The reported memory is an upper bound. Due to automatic garbage ' +
                      'collection, the true allocated memory may be less.']
          };
      };
      MathBackendCPU.prototype.complex = function (real, imag) {
          var result = this.makeOutput(null, real.shape, 'complex64');
          var resultData = this.data.get(result.dataId);
          // The backend owns the reference to the underlying real and imaginary
          // clones. These will explicitly get disposed when the complex tensor is
          // disposed.
          resultData.complexTensors = {
              real: tf.engine().keep(real.clone()),
              imag: tf.engine().keep(imag.clone())
          };
          return result;
      };
      MathBackendCPU.prototype.real = function (input) {
          var resultData = this.data.get(input.dataId);
          return resultData.complexTensors.real.clone();
      };
      MathBackendCPU.prototype.imag = function (input) {
          var resultData = this.data.get(input.dataId);
          return resultData.complexTensors.imag.clone();
      };
      MathBackendCPU.prototype.slice = function (x, begin, size) {
          assertNotComplex(x, 'slice');
          var isContinous = tf.slice_util.isSliceContinous(x.shape, begin, size);
          if (isContinous) {
              var flatOffset = tf.slice_util.computeFlatOffset(begin, x.strides);
              var length_1 = tf.util.sizeFromShape(size);
              var vals = this.readSync(x.dataId);
              return tf.tensor(vals.subarray(flatOffset, flatOffset + length_1), size, x.dtype);
          }
          var buffer = tf.buffer(size, x.dtype);
          var xBuf = this.bufferSync(x);
          for (var i = 0; i < buffer.size; ++i) {
              var loc = buffer.indexToLoc(i);
              var xLoc = loc.map(function (idx, j) { return idx + begin[j]; });
              buffer.values[i] = xBuf.get.apply(xBuf, xLoc);
          }
          return buffer.toTensor();
      };
      MathBackendCPU.prototype.stridedSlice = function (x, begin, end, strides) {
          assertNotComplex(x, 'stridedSlice');
          var outShape = tf.slice_util.computeOutShape(begin, end, strides);
          if (outShape.some(function (axis) { return axis === 0; })) {
              return tf.tensor([], outShape);
          }
          var buffer = tf.buffer(outShape, x.dtype);
          var xBuf = this.bufferSync(x);
          for (var i = 0; i < buffer.size; i++) {
              var loc = buffer.indexToLoc(i);
              var newLoc = new Array(loc.length);
              for (var j = 0; j < newLoc.length; j++) {
                  newLoc[j] = loc[j] * strides[j] + begin[j];
              }
              buffer.set.apply(buffer, [xBuf.get.apply(xBuf, newLoc)].concat(loc));
          }
          return buffer.toTensor();
      };
      MathBackendCPU.prototype.diag = function (x) {
          var xVals = this.readSync(x.dataId);
          var buffer = tf.buffer([x.size, x.size], x.dtype);
          var vals = buffer.values;
          for (var i = 0; i < xVals.length; i++) {
              vals[i * x.size + i] = xVals[i];
          }
          return buffer.toTensor();
      };
      MathBackendCPU.prototype.unstack = function (x, axis) {
          var num = x.shape[axis];
          var outShape = new Array(x.rank - 1);
          var outIndex = 0;
          for (var i = 0; i < x.rank; i++) {
              if (i !== axis) {
                  outShape[outIndex++] = x.shape[i];
              }
          }
          var begin = new Array(x.rank).fill(0);
          var size = x.shape.slice();
          size[axis] = 1;
          var res = new Array(num);
          for (var i = 0; i < res.length; i++) {
              begin[axis] = i;
              res[i] = this.slice(x, begin, size).reshape(outShape);
          }
          return res;
      };
      MathBackendCPU.prototype.reverse = function (x, axis) {
          assertNotComplex(x, 'reverse');
          var buffer = tf.buffer(x.shape, x.dtype);
          var xBuf = this.bufferSync(x);
          var _loop_1 = function (i) {
              var outLoc = buffer.indexToLoc(i);
              var inLoc = outLoc.slice();
              axis.forEach(function (ax) { return inLoc[ax] = x.shape[ax] - 1 - inLoc[ax]; });
              buffer.set.apply(buffer, [xBuf.get.apply(xBuf, inLoc)].concat(outLoc));
          };
          for (var i = 0; i < buffer.size; i++) {
              _loop_1(i);
          }
          return buffer.toTensor();
      };
      MathBackendCPU.prototype.concat = function (tensors, axis) {
          var _this = this;
          if (tensors[0].dtype === 'complex64') {
              var reals = tensors.map(function (t) { return tf.real(t); });
              var imags = tensors.map(function (t) { return tf.imag(t); });
              return tf.complex(this.concat(reals, axis), this.concat(imags, axis));
          }
          var tensors2D = tensors.map(function (t) {
              var innerSize = tf.util.sizeFromShape(t.shape.slice(axis));
              return t.as2D(-1, innerSize);
          });
          var outShape = tf.backend_util.computeOutShape(tensors2D.map(function (t) { return t.shape; }), 1 /* axis
            */);
          var values = tf.buffer(outShape, tensors[0].dtype)
              .values;
          if (tensors2D[0].shape[0] === 1) {
              // Use built-in TypedArray.set() method for speed.
              var offset_1 = 0;
              tensors2D.forEach(function (t) {
                  values.set(_this.readSync(t.dataId), offset_1);
                  offset_1 += t.size;
              });
          }
          else {
              var colOffset_1 = 0;
              tensors2D.forEach(function (t) {
                  var tVals = _this.readSync(t.dataId);
                  var tIdx = 0;
                  for (var row = 0; row < t.shape[0]; ++row) {
                      var resIdx = row * outShape[1] + colOffset_1;
                      for (var col = 0; col < t.shape[1]; ++col) {
                          values[resIdx + col] = tVals[tIdx++];
                      }
                  }
                  colOffset_1 += t.shape[1];
              });
          }
          var finalOutShape = tf.backend_util.computeOutShape(tensors.map(function (t) { return t.shape; }), axis);
          return tf.tensor(values, finalOutShape, tensors[0].dtype);
      };
      MathBackendCPU.prototype.neg = function (x) {
          assertNotComplex(x, 'neg');
          return this.multiply(tf.scalar(-1), x);
      };
      MathBackendCPU.prototype.add = function (a, b) {
          if (a.dtype === 'complex64' || b.dtype === 'complex64') {
              return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), function (aReal, aImag, bReal, bImag) {
                  return { real: aReal + bReal, imag: aImag + bImag };
              });
          }
          return this.broadcastedBinaryOp(a, b, tf.upcastType(a.dtype, b.dtype), function (aValue, bValue) { return aValue + bValue; });
      };
      MathBackendCPU.prototype.addN = function (tensors) {
          var _this = this;
          assertNotComplex(tensors, 'addN');
          var vals = tensors.map(function (t) { return _this.readSync(t.dataId); });
          var result = tf.buffer(tensors[0].shape, tensors[0].dtype);
          var resultVals = result.values;
          for (var i = 0; i < tensors.length; i++) {
              var currVals = vals[i];
              for (var j = 0; j < resultVals.length; j++) {
                  resultVals[j] += currVals[j];
              }
          }
          return result.toTensor();
      };
      MathBackendCPU.prototype.softmax = function (logits, dim) {
          var axes = tf.util.parseAxisParam([dim], logits.shape);
          // TODO(annxingyuan): Call maxImpl rather than op as part of softmax kernel
          // modularization.
          var maxLogit = tf.max(logits, axes);
          var expandedShape = tf.backend_util.expandShapeToKeepDim(maxLogit.shape, axes);
          var a = this.subtract(logits, maxLogit.reshape(expandedShape));
          var b = this.exp(a);
          var sumExp = this.sum(b, axes).reshape(expandedShape);
          // TODO(annxingyuan): Call divImpl rather than op as part of softmax
          // kernel modularization.
          return tf.div(b, sumExp);
      };
      MathBackendCPU.prototype.subtract = function (a, b) {
          if (a.dtype === 'complex64' || b.dtype === 'complex64') {
              return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), function (aReal, aImag, bReal, bImag) {
                  return { real: aReal - bReal, imag: aImag - bImag };
              });
          }
          return this.broadcastedBinaryOp(a, b, tf.upcastType(a.dtype, b.dtype), function (aValue, bValue) { return aValue - bValue; });
      };
      MathBackendCPU.prototype.pow = function (a, b) {
          assertNotComplex([a, b], 'pow');
          return this.broadcastedBinaryOp(a, b, a.dtype, function (aValue, bValue) { return Math.pow(aValue, bValue); });
      };
      MathBackendCPU.prototype.batchMatMul = function (a, b, transposeA, transposeB) {
          assertNotComplex([a, b], 'matMul');
          var sharedDim = transposeA ? a.shape[1] : a.shape[2];
          var leftDim = transposeA ? a.shape[2] : a.shape[1];
          var rightDim = transposeB ? b.shape[1] : b.shape[2];
          var batchDim = a.shape[0];
          var aValues = this.readSync(a.dataId);
          var bValues = this.readSync(b.dataId);
          var _a = transposeA ?
              [a.strides[0], 1, a.strides[1]] :
              [a.strides[0], a.strides[1], 1], aBatch = _a[0], aOuterStep = _a[1], aInnerStep = _a[2];
          var _b = transposeB ?
              [1, b.strides[1], b.strides[0]] :
              [b.strides[1], 1, b.strides[0]], bInnerStep = _b[0], bOuterStep = _b[1], bBatch = _b[2];
          var size = leftDim * rightDim;
          var result = tf.buffer([batchDim, leftDim, rightDim], a.dtype);
          var resVals = result.values;
          var blockSize = this.blockSize;
          for (var b_1 = 0; b_1 < batchDim; b_1++) {
              for (var i0 = 0; i0 < leftDim; i0 += blockSize) {
                  for (var j0 = 0; j0 < rightDim; j0 += blockSize) {
                      for (var k0 = 0; k0 < sharedDim; k0 += blockSize) {
                          // for when blockSize doesn't evenly divide the input
                          var iBlock = Math.min(i0 + blockSize, leftDim);
                          var jBlock = Math.min(j0 + blockSize, rightDim);
                          var kBlock = Math.min(k0 + blockSize, sharedDim);
                          for (var i = i0; i < iBlock; i++) {
                              for (var j = j0; j < jBlock; j++) {
                                  var sum = 0.0;
                                  for (var k = k0; k < kBlock; k++) {
                                      sum += aValues[b_1 * aBatch + i * aOuterStep + k * aInnerStep] *
                                          bValues[k * bInnerStep + j * bOuterStep + b_1 * bBatch];
                                  }
                                  resVals[b_1 * size + (i * rightDim + j)] += sum;
                              }
                          }
                      }
                  }
              }
          }
          return result.toTensor();
      };
      MathBackendCPU.prototype.fusedBatchMatMul = function (_a) {
          var a = _a.a, b = _a.b, transposeA = _a.transposeA, transposeB = _a.transposeB, bias = _a.bias, activation = _a.activation, preluActivationWeights = _a.preluActivationWeights;
          var result = this.batchMatMul(a, b, transposeA, transposeB);
          if (bias) {
              result = this.add(result, bias);
          }
          if (activation) {
              result =
                  mapActivation(this, result, activation, preluActivationWeights);
          }
          return result;
      };
      MathBackendCPU.prototype.multiply = function (a, b) {
          if (a.dtype === 'complex64' || b.dtype === 'complex64') {
              return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), function (aReal, aImag, bReal, bImag) {
                  return {
                      real: aReal * bReal - aImag * bImag,
                      imag: aReal * bImag + aImag * bReal
                  };
              });
          }
          return this.broadcastedBinaryOp(a, b, tf.upcastType(a.dtype, b.dtype), function (aValue, bValue) { return aValue * bValue; });
      };
      MathBackendCPU.prototype.floorDiv = function (a, b) {
          assertNotComplex([a, b], 'floorDiv');
          var op = function (a, b) { return Math.floor(a / b); };
          var outputDtype = 'int32';
          return this.broadcastedBinaryOp(a, b, outputDtype, op);
      };
      MathBackendCPU.prototype.sum = function (x, axes) {
          assertNotComplex(x, 'sum');
          tf.backend_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
          var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
          var resultDtype = tf.upcastType(x.dtype, 'int32');
          var result = tf.zeros(outShape, resultDtype);
          var reduceSize = tf.util.sizeFromShape(reduceShape);
          var vals = this.readSync(result.dataId);
          var aVals = this.readSync(x.dataId);
          for (var i = 0; i < vals.length; ++i) {
              var offset = i * reduceSize;
              var sum = 0;
              for (var j = 0; j < reduceSize; ++j) {
                  sum += aVals[offset + j];
              }
              vals[i] = sum;
          }
          return result;
      };
      MathBackendCPU.prototype.prod = function (x, axes) {
          assertNotComplex(x, 'sum');
          var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
          var resultDtype = tf.upcastType(x.dtype, 'int32');
          var result = tf.zeros(outShape, resultDtype);
          var reduceSize = tf.util.sizeFromShape(reduceShape);
          var vals = this.readSync(result.dataId);
          var aVals = this.readSync(x.dataId);
          for (var i = 0; i < vals.length; ++i) {
              var offset = i * reduceSize;
              var prod = 1;
              for (var j = 0; j < reduceSize; ++j) {
                  prod *= aVals[offset + j];
              }
              vals[i] = prod;
          }
          return result;
      };
      MathBackendCPU.prototype.unsortedSegmentSum = function (x, segmentIds, numSegments) {
          assertNotComplex(x, 'unsortedSegmentSum');
          var res = [];
          // Reshape the segment id's so that they can be broadcast with
          // x. The new shape should be [segmentIds.shape, 1, ..., 1]
          var numIters = x.rank - segmentIds.rank;
          for (var i = 0; i < numIters; ++i) {
              segmentIds = segmentIds.expandDims(i + 1);
          }
          for (var i = 0; i < numSegments; ++i) {
              var segmentId = tf.scalar(i, 'int32');
              var mask = tf.equal(segmentId, segmentIds).asType('float32');
              var sum = mask.mul(x).sum(0);
              res.push(sum);
          }
          return tf.stack(res);
      };
      MathBackendCPU.prototype.argMin = function (x, axis) {
          assertNotComplex(x, 'argMin');
          var axes = [axis];
          tf.backend_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
          var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
          var result = tf.zeros(outShape, 'int32');
          var reduceSize = tf.util.sizeFromShape(reduceShape);
          var vals = this.readSync(result.dataId);
          var aVals = this.readSync(x.dataId);
          for (var i = 0; i < vals.length; ++i) {
              var offset = i * reduceSize;
              var min = aVals[offset];
              var minIndex = 0;
              for (var j = 0; j < reduceSize; ++j) {
                  var value = aVals[offset + j];
                  if (value < min) {
                      min = value;
                      minIndex = j;
                  }
              }
              vals[i] = minIndex;
          }
          return result;
      };
      MathBackendCPU.prototype.argMax = function (x, axis) {
          assertNotComplex(x, 'argMax');
          var axes = [axis];
          tf.backend_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
          var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
          var result = tf.zeros(outShape, 'int32');
          var reduceSize = tf.util.sizeFromShape(reduceShape);
          var vals = this.readSync(result.dataId);
          var aVals = this.readSync(x.dataId);
          for (var i = 0; i < vals.length; ++i) {
              var offset = i * reduceSize;
              var max_1 = aVals[offset];
              var maxIndex = 0;
              for (var j = 0; j < reduceSize; ++j) {
                  var value = aVals[offset + j];
                  if (value > max_1) {
                      max_1 = value;
                      maxIndex = j;
                  }
              }
              vals[i] = maxIndex;
          }
          return result;
      };
      MathBackendCPU.prototype.cumsum = function (x, axis, exclusive, reverse) {
          assertNotComplex(x, 'cumsum');
          if (axis !== x.rank - 1) {
              throw new Error("backend.cumsum in CPU expects an inner-most axis=" + (x.rank - 1) + " " +
                  ("but got axis=" + axis));
          }
          var resultDtype = tf.upcastType(x.dtype, 'int32');
          var result = tf.zeros(x.shape, resultDtype);
          var vals = this.readSync(result.dataId);
          var aVals = this.readSync(x.dataId);
          var finalDim = x.shape[x.rank - 1];
          var indexAdjuster = reverse ?
              function (i, j) { return i + finalDim - j - 1; } :
              function (i, j) { return i + j; };
          for (var i = 0; i < aVals.length; i += finalDim) {
              for (var j = 0; j < finalDim; j++) {
                  var idx = indexAdjuster(i, j);
                  if (j === 0) {
                      vals[idx] = exclusive ? 0 : aVals[idx];
                  }
                  else {
                      var prevIdx = indexAdjuster(i, j - 1);
                      vals[idx] = exclusive ? aVals[prevIdx] + vals[prevIdx] :
                          aVals[idx] + vals[prevIdx];
                  }
              }
          }
          return result;
      };
      MathBackendCPU.prototype.equal = function (a, b) {
          assertNotComplex([a, b], 'equal');
          return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
              return (aVal === bVal) ? 1 : 0;
          });
      };
      MathBackendCPU.prototype.notEqual = function (a, b) {
          assertNotComplex([a, b], 'notEqual');
          return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
              return (aVal !== bVal) ? 1 : 0;
          });
      };
      MathBackendCPU.prototype.less = function (a, b) {
          assertNotComplex([a, b], 'less');
          return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
              return (aVal < bVal) ? 1 : 0;
          });
      };
      MathBackendCPU.prototype.lessEqual = function (a, b) {
          assertNotComplex([a, b], 'lessEqual');
          return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
              return (aVal <= bVal) ? 1 : 0;
          });
      };
      MathBackendCPU.prototype.greater = function (a, b) {
          assertNotComplex([a, b], 'greater');
          return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
              return (aVal > bVal) ? 1 : 0;
          });
      };
      MathBackendCPU.prototype.greaterEqual = function (a, b) {
          assertNotComplex([a, b], 'greaterEqual');
          return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
              return (aVal >= bVal) ? 1 : 0;
          });
      };
      MathBackendCPU.prototype.logicalNot = function (x) {
          assertNotComplex(x, 'logicalNot');
          var values = this.readSync(x.dataId);
          var newValues = new Uint8Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              newValues[i] = values[i] ? 0 : 1;
          }
          return this.makeOutput(newValues, x.shape, 'bool');
      };
      MathBackendCPU.prototype.logicalAnd = function (a, b) {
          assertNotComplex([a, b], 'logicalAnd');
          return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
              return aVal && bVal;
          });
      };
      MathBackendCPU.prototype.logicalOr = function (a, b) {
          assertNotComplex([a, b], 'logicalOr');
          return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
              return aVal || bVal;
          });
      };
      MathBackendCPU.prototype.select = function (condition, a, b) {
          assertNotComplex([condition, a, b], 'select');
          var values = this.readSync(condition.dataId);
          var aValues = this.readSync(a.dataId);
          var bValues = this.readSync(b.dataId);
          var result = tf.zeros(a.shape, tf.upcastType(a.dtype, b.dtype));
          var newValues = this.readSync(result.dataId);
          var index = 0;
          var offset = condition.rank === 0 || condition.rank > 1 || a.rank === 1 ?
              1 :
              tf.util.sizeFromShape(a.shape.slice(1));
          for (var i = 0; i < values.length; i++) {
              for (var j = 0; j < offset; j++) {
                  if (values[i] === 1) {
                      newValues[index++] = aValues[i];
                  }
                  else {
                      newValues[index++] = bValues[i];
                  }
              }
          }
          return result;
      };
      MathBackendCPU.prototype.where = function (condition) {
          assertNotComplex([condition], 'where');
          var condVals = this.readSync(condition.dataId);
          return whereImpl(condition.shape, condVals);
      };
      MathBackendCPU.prototype.topk = function (x, k, sorted) {
          assertNotComplex(x, 'topk');
          var xVals = this.readSync(x.dataId);
          return topkImpl(xVals, x.shape, x.dtype, k, sorted);
      };
      MathBackendCPU.prototype.min = function (x, axes) {
          assertNotComplex(x, 'min');
          tf.backend_util.assertAxesAreInnerMostDims('min', axes, x.rank);
          var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
          var result = tf.zeros(outShape, x.dtype);
          var reduceSize = tf.util.sizeFromShape(reduceShape);
          var vals = this.readSync(result.dataId);
          var aVals = this.readSync(x.dataId);
          for (var i = 0; i < vals.length; ++i) {
              var offset = i * reduceSize;
              var min = aVals[offset];
              for (var j = 0; j < reduceSize; ++j) {
                  var value = aVals[offset + j];
                  if (value < min) {
                      min = value;
                  }
              }
              vals[i] = min;
          }
          return result;
      };
      MathBackendCPU.prototype.minimum = function (a, b) {
          assertNotComplex([a, b], 'minimum');
          return this.broadcastedBinaryOp(a, b, a.dtype, function (aVal, bVal) { return Math.min(aVal, bVal); });
      };
      MathBackendCPU.prototype.mod = function (a, b) {
          assertNotComplex([a, b], 'mod');
          return this.broadcastedBinaryOp(a, b, a.dtype, function (aVal, bVal) {
              var rem = aVal % bVal;
              if ((aVal < 0 && bVal < 0) || (aVal >= 0 && bVal >= 0)) {
                  return rem;
              }
              else {
                  return (rem + bVal) % bVal;
              }
          });
      };
      MathBackendCPU.prototype.maximum = function (a, b) {
          assertNotComplex([a, b], 'maximum');
          return this.broadcastedBinaryOp(a, b, a.dtype, function (aVal, bVal) { return Math.max(aVal, bVal); });
      };
      MathBackendCPU.prototype.all = function (x, axes) {
          assertNotComplex(x, 'all');
          tf.backend_util.assertAxesAreInnerMostDims('all', axes, x.rank);
          var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
          var result = tf.zeros(outShape, x.dtype);
          var reduceSize = tf.util.sizeFromShape(reduceShape);
          var vals = this.readSync(result.dataId);
          var aVals = this.readSync(x.dataId);
          for (var i = 0; i < vals.length; ++i) {
              var offset = i * reduceSize;
              var all = aVals[offset];
              for (var j = 0; j < reduceSize; ++j) {
                  var value = aVals[offset + j];
                  all = all && value;
              }
              vals[i] = all;
          }
          return result;
      };
      MathBackendCPU.prototype.any = function (x, axes) {
          assertNotComplex(x, 'any');
          tf.backend_util.assertAxesAreInnerMostDims('any', axes, x.rank);
          var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
          var result = tf.zeros(outShape, x.dtype);
          var reduceSize = tf.util.sizeFromShape(reduceShape);
          var vals = this.readSync(result.dataId);
          var aVals = this.readSync(x.dataId);
          for (var i = 0; i < vals.length; ++i) {
              var offset = i * reduceSize;
              var anyVal = aVals[offset];
              for (var j = 0; j < reduceSize; ++j) {
                  var value = aVals[offset + j];
                  anyVal = anyVal || value;
              }
              vals[i] = anyVal;
          }
          return result;
      };
      MathBackendCPU.prototype.squaredDifference = function (a, b) {
          assertNotComplex([a, b], 'squaredDifference');
          return this.broadcastedBinaryOp(a, b, a.dtype, function (aVal, bVal) {
              var diff = aVal - bVal;
              return diff * diff;
          });
      };
      MathBackendCPU.prototype.ceil = function (x) {
          assertNotComplex(x, 'ceil');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              newValues[i] = Math.ceil(values[i]);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.floor = function (x) {
          assertNotComplex(x, 'floor');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              newValues[i] = Math.floor(values[i]);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.sign = function (x) {
          assertNotComplex(x, 'x');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              if (values[i] < 0) {
                  newValues[i] = -1;
              }
              else if (values[i] > 0) {
                  newValues[i] = 1;
              }
              else {
                  newValues[i] = 0;
              }
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.isNaN = function (x) {
          assertNotComplex(x, 'x');
          var values = this.readSync(x.dataId);
          var newValues = new Uint8Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              if (Number.isNaN(values[i])) {
                  newValues[i] = 1;
              }
          }
          return this.makeOutput(newValues, x.shape, 'bool');
      };
      MathBackendCPU.prototype.isInf = function (x) {
          assertNotComplex(x, 'x');
          var values = this.readSync(x.dataId);
          var newValues = new Uint8Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              if (Math.abs(values[i]) === Infinity) {
                  newValues[i] = 1;
              }
          }
          return this.makeOutput(newValues, x.shape, 'bool');
      };
      MathBackendCPU.prototype.isFinite = function (x) {
          assertNotComplex(x, 'x');
          var values = this.readSync(x.dataId);
          var newValues = new Uint8Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              if (Number.isFinite(values[i])) {
                  newValues[i] = 1;
              }
          }
          return this.makeOutput(newValues, x.shape, 'bool');
      };
      MathBackendCPU.prototype.round = function (x) {
          assertNotComplex(x, 'round');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              // The algorithm is based on banker's rounding.
              var base = Math.floor(values[i]);
              if (values[i] - base < 0.5) {
                  newValues[i] = Math.floor(values[i]);
              }
              else if (values[i] - base > 0.5) {
                  newValues[i] = Math.ceil(values[i]);
              }
              else {
                  if (base % 2.0 === 0.0) {
                      newValues[i] = base;
                  }
                  else {
                      newValues[i] = base + 1.0;
                  }
              }
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.exp = function (x) {
          assertNotComplex(x, 'exp');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              newValues[i] = Math.exp(values[i]);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.expm1 = function (x) {
          assertNotComplex(x, 'expm1');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              newValues[i] = Math.expm1(values[i]);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.log = function (x) {
          assertNotComplex(x, 'log');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              var value = values[i];
              newValues[i] = Math.log(value);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.log1p = function (x) {
          assertNotComplex(x, 'log1p');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              var value = values[i];
              newValues[i] = Math.log1p(value);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.sqrt = function (x) {
          assertNotComplex(x, 'sqrt');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              var value = values[i];
              newValues[i] = Math.sqrt(value);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.rsqrt = function (x) {
          assertNotComplex(x, 'rsqrt');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              var value = values[i];
              newValues[i] = 1 / Math.sqrt(value);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.reciprocal = function (x) {
          assertNotComplex(x, 'reciprocal');
          var values = this.readSync(x.dataId);
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              newValues[i] = 1 / values[i];
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.linear = function (x) {
          return x;
      };
      MathBackendCPU.prototype.relu = function (x) {
          assertNotComplex(x, 'relu');
          var res = tf.zeros(x.shape, x.dtype);
          var resVals = this.readSync(res.dataId);
          var inVals = this.readSync(x.dataId);
          for (var i = 0; i < inVals.length; ++i) {
              resVals[i] = Math.max(0, inVals[i]);
          }
          return res;
      };
      MathBackendCPU.prototype.relu6 = function (x) {
          assertNotComplex(x, 'relu');
          var res = tf.zeros(x.shape, x.dtype);
          var resVals = this.readSync(res.dataId);
          var inVals = this.readSync(x.dataId);
          for (var i = 0; i < inVals.length; ++i) {
              resVals[i] = Math.min(Math.max(0, inVals[i]), 6);
          }
          return res;
      };
      MathBackendCPU.prototype.prelu = function (x, a) {
          assertNotComplex([x, a], 'prelu');
          return this.broadcastedBinaryOp(x, a, x.dtype, function (xValue, aValue) { return xValue < 0 ? aValue * xValue : xValue; });
      };
      MathBackendCPU.prototype.elu = function (x) {
          assertNotComplex(x, 'elu');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              var v = values[i];
              if (v >= 0) {
                  resultValues[i] = v;
              }
              else {
                  resultValues[i] = (Math.exp(v) - 1);
              }
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.eluDer = function (dy, y) {
          assertNotComplex([dy, y], 'eluDer');
          var resultValues = new Float32Array(y.size);
          var values = this.readSync(y.dataId);
          var dyValues = this.readSync(dy.dataId);
          for (var i = 0; i < values.length; ++i) {
              var v = values[i];
              if (v >= 1) {
                  resultValues[i] = dyValues[i];
              }
              else {
                  resultValues[i] = dyValues[i] * (v + 1);
              }
          }
          return this.makeOutput(resultValues, y.shape, 'float32');
      };
      MathBackendCPU.prototype.selu = function (x) {
          assertNotComplex(x, 'selu');
          // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
          // see: https://arxiv.org/abs/1706.02515
          var scaleAlpha = tf.backend_util.SELU_SCALEALPHA;
          var scale = tf.backend_util.SELU_SCALE;
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              var v = values[i];
              if (v >= 0) {
                  resultValues[i] = scale * v;
              }
              else {
                  resultValues[i] = scaleAlpha * (Math.exp(v) - 1);
              }
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.clip = function (x, min, max) {
          assertNotComplex(x, 'clip');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              var v = values[i];
              resultValues[i] = v > max ? max : (v < min ? min : v);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.abs = function (x) {
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.abs(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.complexAbs = function (x) {
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < x.size; ++i) {
              var real = values[i * 2];
              var imag = values[i * 2 + 1];
              resultValues[i] = Math.hypot(real, imag);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.int = function (x) {
          assertNotComplex(x, 'int');
          var resultValues = new Int32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = values[i];
          }
          return this.makeOutput(resultValues, x.shape, 'int32');
      };
      MathBackendCPU.prototype.sigmoid = function (x) {
          assertNotComplex(x, 'sigmoid');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = 1 / (1 + Math.exp(-values[i]));
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.softplus = function (x) {
          assertNotComplex(x, 'softplus');
          // mirrors the implementation of tf.nn.softplus: https://goo.gl/vkcvwX
          // epsilon is the difference between 1.0 and the next representable float.
          // For a single precision 32 bit float this should be 2^-23, see:
          // https://math.byu.edu/~schow/work/IEEEFloatingPoint.htm
          var epsilon = 1.1920928955078125e-7;
          var threshold = Math.log(epsilon) + 2.0;
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              // Value above which exp(x) may overflow, but softplus(x) == x
              // is within machine epsilon.
              var tooLarge = values[i] > -threshold;
              // Value below which exp(x) may underflow, but softplus(x) == exp(x)
              // is within machine epsilon.
              var tooSmall = values[i] < threshold;
              var expX = Math.exp(values[i]);
              var result = void 0;
              if (tooSmall) {
                  result = expX;
              }
              else if (tooLarge) {
                  result = values[i];
              }
              else {
                  result = Math.log(1.0 + expX);
              }
              resultValues[i] = result;
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.sin = function (x) {
          assertNotComplex(x, 'sin');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.sin(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.cos = function (x) {
          assertNotComplex(x, 'cos');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.cos(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.tan = function (x) {
          assertNotComplex(x, 'tan');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.tan(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.asin = function (x) {
          assertNotComplex(x, 'asin');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.asin(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.acos = function (x) {
          assertNotComplex(x, 'acos');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.acos(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.atan = function (x) {
          assertNotComplex(x, 'atan');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.atan(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.atan2 = function (a, b) {
          assertNotComplex([a, b], 'atan2');
          return this.broadcastedBinaryOp(a, b, a.dtype, function (aValue, bValue) { return Math.atan2(aValue, bValue); });
      };
      MathBackendCPU.prototype.sinh = function (x) {
          assertNotComplex(x, 'sinh');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.sinh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.cosh = function (x) {
          assertNotComplex(x, 'cosh');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.cosh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.tanh = function (x) {
          assertNotComplex(x, 'tanh');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = tf.util.tanh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.asinh = function (x) {
          assertNotComplex(x, 'asinh');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.asinh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.acosh = function (x) {
          assertNotComplex(x, 'acosh');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.acosh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.atanh = function (x) {
          assertNotComplex(x, 'atanh');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              resultValues[i] = Math.atanh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.erf = function (x) {
          assertNotComplex(x, 'erf');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          var p = tf.backend_util.ERF_P;
          var a1 = tf.backend_util.ERF_A1;
          var a2 = tf.backend_util.ERF_A2;
          var a3 = tf.backend_util.ERF_A3;
          var a4 = tf.backend_util.ERF_A4;
          var a5 = tf.backend_util.ERF_A5;
          for (var i = 0; i < values.length; ++i) {
              var sign = Math.sign(values[i]);
              var v = Math.abs(values[i]);
              var t = 1.0 / (1.0 + p * v);
              resultValues[i] = sign *
                  (1.0 -
                      (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                          Math.exp(-v * v));
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.step = function (x, alpha) {
          if (alpha === void 0) { alpha = 0; }
          assertNotComplex(x, 'step');
          var resultValues = new Float32Array(x.size);
          var values = this.readSync(x.dataId);
          for (var i = 0; i < values.length; ++i) {
              var value = values[i];
              if (isNaN(value)) {
                  resultValues[i] = NaN;
              }
              else {
                  resultValues[i] = value > 0 ? 1 : alpha;
              }
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      };
      MathBackendCPU.prototype.fusedConv2d = function (_a) {
          var input = _a.input, filter = _a.filter, convInfo = _a.convInfo, bias = _a.bias, activation = _a.activation, preluActivationWeights = _a.preluActivationWeights;
          var result = this.conv2d(input, filter, convInfo);
          if (bias) {
              result = this.add(result, bias);
          }
          if (activation) {
              result =
                  mapActivation(this, result, activation, preluActivationWeights);
          }
          return result;
      };
      MathBackendCPU.prototype.conv2d = function (x, filter, convInfo) {
          assertNotComplex([x, filter], 'conv2d');
          var filterHeight = convInfo.filterHeight;
          var filterWidth = convInfo.filterWidth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var padLeft = convInfo.padInfo.left;
          var padTop = convInfo.padInfo.top;
          var isChannelsLast = convInfo.dataFormat === 'channelsLast';
          var y = tf.buffer(convInfo.outShape, x.dtype);
          var xBatchStride = x.strides[0];
          var xRowStride = isChannelsLast ? x.strides[1] : x.strides[2];
          var xColStride = isChannelsLast ? x.strides[2] : 1;
          var xChannelStride = isChannelsLast ? 1 : x.strides[1];
          var yBatchStride = y.strides[0];
          var yRowStride = isChannelsLast ? y.strides[1] : y.strides[2];
          var yColStride = isChannelsLast ? y.strides[2] : 1;
          var yChannelStride = isChannelsLast ? 1 : y.strides[1];
          var xVals = this.readSync(x.dataId);
          var wVals = this.readSync(filter.dataId);
          var yVals = y.values;
          for (var b = 0; b < convInfo.batchSize; ++b) {
              var xOffset1 = b * xBatchStride;
              var yOffset1 = b * yBatchStride;
              for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                  var yOffset2 = yOffset1 + yR * yRowStride;
                  var xRCorner = yR * convInfo.strideHeight - padTop;
                  for (var wR = 0; wR < filterHeight; wR++) {
                      var xR = xRCorner + wR * dilationHeight;
                      if (xR < 0 || xR >= convInfo.inHeight) {
                          continue;
                      }
                      var wOffset1 = wR * filter.strides[0];
                      var xOffset2 = xOffset1 + xR * xRowStride;
                      for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                          var yOffset3 = yOffset2 + yC * yColStride;
                          var xCCorner = yC * convInfo.strideWidth - padLeft;
                          for (var wC = 0; wC < filterWidth; wC++) {
                              var xC = xCCorner + wC * dilationWidth;
                              if (xC < 0 || xC >= convInfo.inWidth) {
                                  continue;
                              }
                              var wOffset2 = wOffset1 + wC * filter.strides[1];
                              var xOffset3 = xOffset2 + xC * xColStride;
                              var wOffset3 = wOffset2;
                              for (var d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                  var xVal = xVals[xOffset3 + d1 * xChannelStride];
                                  for (var d2 = 0; d2 < convInfo.outChannels; ++d2) {
                                      yVals[yOffset3 + d2 * yChannelStride] +=
                                          xVal * wVals[wOffset3 + d2];
                                  }
                                  wOffset3 += convInfo.outChannels;
                              }
                          }
                      }
                  }
              }
          }
          return y.toTensor();
      };
      MathBackendCPU.prototype.conv3d = function (x, filter, convInfo) {
          var filterDepth = convInfo.filterDepth;
          var filterHeight = convInfo.filterHeight;
          var filterWidth = convInfo.filterWidth;
          var dilationDepth = convInfo.dilationDepth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var padFront = convInfo.padInfo.front;
          var padLeft = convInfo.padInfo.left;
          var padTop = convInfo.padInfo.top;
          var y = tf.buffer(convInfo.outShape, x.dtype);
          var xVals = this.readSync(x.dataId);
          var wVals = this.readSync(filter.dataId);
          var yVals = y.values;
          for (var b = 0; b < convInfo.batchSize; ++b) {
              var xOffset1 = b * x.strides[0];
              var yOffset1 = b * y.strides[0];
              for (var yF = 0; yF < convInfo.outDepth; ++yF) {
                  var yOffset2 = yOffset1 + yF * y.strides[1];
                  var xFCorner = yF * convInfo.strideDepth - padFront;
                  for (var wF = 0; wF < filterDepth; wF++) {
                      var xF = xFCorner + wF * dilationDepth;
                      if (xF < 0 || xF >= convInfo.inDepth) {
                          continue;
                      }
                      var wOffset1 = wF * filter.strides[0];
                      var xOffset2 = xOffset1 + xF * x.strides[1];
                      for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                          var yOffset3 = yOffset2 + yR * y.strides[2];
                          var xRCorner = yR * convInfo.strideHeight - padTop;
                          for (var wR = 0; wR < filterHeight; wR++) {
                              var xR = xRCorner + wR * dilationHeight;
                              if (xR < 0 || xR >= convInfo.inHeight) {
                                  continue;
                              }
                              var wOffset2 = wOffset1 + wR * filter.strides[1];
                              var xOffset3 = xOffset2 + xR * x.strides[2];
                              for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                                  var yOffset4 = yOffset3 + yC * convInfo.outChannels;
                                  var xCCorner = yC * convInfo.strideWidth - padLeft;
                                  for (var wC = 0; wC < filterWidth; wC++) {
                                      var xC = xCCorner + wC * dilationWidth;
                                      if (xC < 0 || xC >= convInfo.inWidth) {
                                          continue;
                                      }
                                      var wOffset3 = wOffset2 + wC * filter.strides[2];
                                      var xOffset4 = xOffset3 + xC * convInfo.inChannels;
                                      var wOffset4 = wOffset3;
                                      for (var d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                          var xVal = xVals[xOffset4 + d1];
                                          for (var d2 = 0; d2 < convInfo.outChannels; ++d2) {
                                              yVals[yOffset4 + d2] += xVal * wVals[wOffset4 + d2];
                                          }
                                          wOffset4 += convInfo.outChannels;
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          }
          return y.toTensor();
      };
      MathBackendCPU.prototype.conv2dDerInput = function (dy, filter, convInfo) {
          assertNotComplex([dy, filter], 'conv2dDerInput');
          var dx = tf.buffer(convInfo.inShape, 'float32');
          var dxValues = dx.values;
          var dyValues = this.readSync(dy.dataId);
          var fltValues = this.readSync(filter.dataId);
          var _a = filter.strides, fltS0 = _a[0], fltS1 = _a[1], fltS2 = _a[2];
          var batchSize = convInfo.batchSize, filterHeight = convInfo.filterHeight, filterWidth = convInfo.filterWidth, inChannels = convInfo.inChannels, inHeight = convInfo.inHeight, inWidth = convInfo.inWidth, outChannels = convInfo.outChannels, outHeight = convInfo.outHeight, outWidth = convInfo.outWidth, strideHeight = convInfo.strideHeight, strideWidth = convInfo.strideWidth, dataFormat = convInfo.dataFormat;
          var topPad = filterHeight - 1 - convInfo.padInfo.top;
          var leftPad = filterWidth - 1 - convInfo.padInfo.left;
          var isChannelsLast = dataFormat === 'channelsLast';
          var xBatchStride = dx.strides[0];
          var xRowStride = isChannelsLast ? dx.strides[1] : dx.strides[2];
          var xColStride = isChannelsLast ? dx.strides[2] : 1;
          var xChannelStride = isChannelsLast ? 1 : dx.strides[1];
          var yBatchStride = dy.strides[0];
          var yRowStride = isChannelsLast ? dy.strides[1] : dy.strides[2];
          var yColStride = isChannelsLast ? dy.strides[2] : 1;
          var yChannelStride = isChannelsLast ? 1 : dy.strides[1];
          for (var b = 0; b < batchSize; ++b) {
              for (var d1 = 0; d1 < inChannels; ++d1) {
                  for (var xR = 0; xR < inHeight; ++xR) {
                      var xRCorner = xR - topPad;
                      var xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                      var yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                      for (var xC = 0; xC < inWidth; ++xC) {
                          var xCCorner = xC - leftPad;
                          var xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                          var yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                          var dotProd = 0;
                          for (var yR = xRMin; yR < yRMax; ++yR) {
                              var wR = yR * strideHeight - xRCorner;
                              for (var yC = xCMin; yC < yCMax; ++yC) {
                                  var wC = yC * strideWidth - xCCorner;
                                  var dyOffset = yBatchStride * b + yRowStride * yR + yColStride * yC;
                                  var fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                      fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                                  for (var d2 = 0; d2 < outChannels; ++d2) {
                                      var pixel = dyValues[dyOffset + yChannelStride * d2];
                                      var weight = fltValues[fltOffset + d2];
                                      dotProd += pixel * weight;
                                  }
                              }
                          }
                          var dxOffset = xBatchStride * b + xRowStride * xR +
                              xColStride * xC + xChannelStride * d1;
                          dxValues[dxOffset] = dotProd;
                      }
                  }
              }
          }
          return dx.toTensor();
      };
      MathBackendCPU.prototype.conv3dDerInput = function (dy, filter, convInfo) {
          var dx = tf.buffer(convInfo.inShape, 'float32');
          var dxValues = dx.values;
          var _a = dx.strides, dxS0 = _a[0], dxS1 = _a[1], dxS2 = _a[2], dxS3 = _a[3];
          var dyValues = this.readSync(dy.dataId);
          var _b = dy.strides, dyS0 = _b[0], dyS1 = _b[1], dyS2 = _b[2], dyS3 = _b[3];
          var fltValues = this.readSync(filter.dataId);
          var _c = filter.strides, fltS0 = _c[0], fltS1 = _c[1], fltS2 = _c[2], fltS3 = _c[3];
          var batchSize = convInfo.batchSize, filterDepth = convInfo.filterDepth, filterHeight = convInfo.filterHeight, filterWidth = convInfo.filterWidth, inChannels = convInfo.inChannels, inDepth = convInfo.inDepth, inHeight = convInfo.inHeight, inWidth = convInfo.inWidth, outChannels = convInfo.outChannels, outDepth = convInfo.outDepth, outHeight = convInfo.outHeight, outWidth = convInfo.outWidth, strideDepth = convInfo.strideDepth, strideHeight = convInfo.strideHeight, strideWidth = convInfo.strideWidth;
          var frontPad = filterDepth - 1 - convInfo.padInfo.front;
          var topPad = filterHeight - 1 - convInfo.padInfo.top;
          var leftPad = filterWidth - 1 - convInfo.padInfo.left;
          for (var b = 0; b < batchSize; ++b) {
              for (var d1 = 0; d1 < inChannels; ++d1) {
                  // Frames of depth
                  for (var xF = 0; xF < inDepth; ++xF) {
                      var xFCorner = xF - frontPad;
                      var xFMin = Math.max(0, Math.ceil(xFCorner / strideDepth));
                      var yFMax = Math.min(outDepth, (filterDepth + xFCorner) / strideDepth);
                      // Rows as per standard 2d matrix notation
                      for (var xR = 0; xR < inHeight; ++xR) {
                          var xRCorner = xR - topPad;
                          var xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                          var yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                          // Columns as per standard 2d matrix notation
                          for (var xC = 0; xC < inWidth; ++xC) {
                              var xCCorner = xC - leftPad;
                              var xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                              var yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                              var dotProd = 0;
                              for (var yF = xFMin; yF < yFMax; ++yF) {
                                  var wF = yF * strideDepth - xFCorner;
                                  for (var yR = xRMin; yR < yRMax; ++yR) {
                                      var wR = yR * strideHeight - xRCorner;
                                      for (var yC = xCMin; yC < yCMax; ++yC) {
                                          var wC = yC * strideWidth - xCCorner;
                                          var dyOffset = dyS0 * b + dyS1 * yF + dyS2 * yR + dyS3 * yC;
                                          var fltOffset = fltS0 * (filterDepth - 1 - wF) +
                                              fltS1 * (filterHeight - 1 - wR) +
                                              fltS2 * (filterWidth - 1 - wC) + fltS3 * d1;
                                          for (var d2 = 0; d2 < outChannels; ++d2) {
                                              var pixel = dyValues[dyOffset + d2];
                                              var weight = fltValues[fltOffset + d2];
                                              dotProd += pixel * weight;
                                          }
                                      }
                                  }
                              }
                              dxValues[dxS0 * b + dxS1 * xF + dxS2 * xR + dxS3 * xC + d1] =
                                  dotProd;
                          }
                      }
                  }
              }
          }
          return dx.toTensor();
      };
      MathBackendCPU.prototype.conv2dDerFilter = function (x, dy, convInfo) {
          assertNotComplex([x, dy], 'conv2dDerFilter');
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var filterHeight = convInfo.filterHeight;
          var filterWidth = convInfo.filterWidth;
          var isChannelsLast = convInfo.dataFormat === 'channelsLast';
          var dW = tf.buffer(convInfo.filterShape, 'float32');
          var leftPad = convInfo.padInfo.left;
          var topPad = convInfo.padInfo.top;
          var xBuf = this.bufferSync(x);
          var dyBuf = this.bufferSync(dy);
          for (var wR = 0; wR < filterHeight; ++wR) {
              var yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
              var yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
              for (var wC = 0; wC < filterWidth; ++wC) {
                  var yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                  var yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                  for (var d1 = 0; d1 < convInfo.inChannels; ++d1) {
                      for (var d2 = 0; d2 < convInfo.outChannels; ++d2) {
                          // Need to convolve.
                          var dotProd = 0;
                          for (var b = 0; b < convInfo.batchSize; ++b) {
                              for (var yR = yRMin; yR < yRMax; ++yR) {
                                  var xR = wR + yR * strideHeight - topPad;
                                  for (var yC = yCMin; yC < yCMax; ++yC) {
                                      var xC = wC + yC * strideWidth - leftPad;
                                      if (isChannelsLast) {
                                          dotProd +=
                                              xBuf.get(b, xR, xC, d1) * dyBuf.get(b, yR, yC, d2);
                                      }
                                      else {
                                          dotProd +=
                                              xBuf.get(b, d1, xR, xC) * dyBuf.get(b, d2, yR, yC);
                                      }
                                  }
                              }
                          }
                          dW.set(dotProd, wR, wC, d1, d2);
                      }
                  }
              }
          }
          return dW.toTensor();
      };
      MathBackendCPU.prototype.conv3dDerFilter = function (x, dy, convInfo) {
          var strideDepth = convInfo.strideDepth;
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var filterDepth = convInfo.filterDepth;
          var filterHeight = convInfo.filterHeight;
          var filterWidth = convInfo.filterWidth;
          var dw = tf.buffer(convInfo.filterShape, 'float32');
          var dwValues = dw.values;
          var _a = dw.strides, dwS0 = _a[0], dwS1 = _a[1], dwS2 = _a[2], dwS3 = _a[3];
          var dyValues = this.readSync(dy.dataId);
          var _b = dy.strides, dyS0 = _b[0], dyS1 = _b[1], dyS2 = _b[2], dyS3 = _b[3];
          var xValues = this.readSync(x.dataId);
          var _c = x.strides, xS0 = _c[0], xS1 = _c[1], xS2 = _c[2], xS3 = _c[3];
          var frontPad = convInfo.padInfo.front;
          var leftPad = convInfo.padInfo.left;
          var topPad = convInfo.padInfo.top;
          for (var wF = 0; wF < filterDepth; ++wF) {
              var yFMin = Math.max(0, Math.ceil((frontPad - wF) / strideDepth));
              var yFMax = Math.min(convInfo.outDepth, (convInfo.inDepth + frontPad - wF) / strideDepth);
              var wOffset1 = wF * dwS0;
              for (var wR = 0; wR < filterHeight; ++wR) {
                  var yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
                  var yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
                  var wOffset2 = wR * dwS1 + wOffset1;
                  for (var wC = 0; wC < filterWidth; ++wC) {
                      var yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                      var yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                      var wOffset3 = wC * dwS2 + wOffset2;
                      for (var d1 = 0; d1 < convInfo.inChannels; ++d1) {
                          var wOffset4 = d1 * dwS3 + wOffset3;
                          for (var d2 = 0; d2 < convInfo.outChannels; ++d2) {
                              var dotProd = 0;
                              for (var b = 0; b < convInfo.batchSize; ++b) {
                                  var xOffset1 = b * xS0;
                                  var yOffset1 = b * dyS0;
                                  for (var yF = yFMin; yF < yFMax; ++yF) {
                                      var xF = wF + yF * strideDepth - frontPad;
                                      var xOffset2 = xF * xS1 + xOffset1;
                                      var yOffset2 = yF * dyS1 + yOffset1;
                                      for (var yR = yRMin; yR < yRMax; ++yR) {
                                          var xR = wR + yR * strideHeight - topPad;
                                          var xOffset3 = xR * xS2 + xOffset2;
                                          var yOffset3 = yR * dyS2 + yOffset2;
                                          for (var yC = yCMin; yC < yCMax; ++yC) {
                                              var xC = wC + yC * strideWidth - leftPad;
                                              var xOffset4 = xC * xS3 + xOffset3;
                                              var yOffset4 = yC * dyS3 + yOffset3;
                                              dotProd +=
                                                  xValues[xOffset4 + d1] * dyValues[yOffset4 + d2];
                                          }
                                      }
                                  }
                              }
                              dwValues[wOffset4 + d2] = dotProd;
                          }
                      }
                  }
              }
          }
          return dw.toTensor();
      };
      MathBackendCPU.prototype.fusedDepthwiseConv2D = function (_a) {
          var input = _a.input, filter = _a.filter, convInfo = _a.convInfo, bias = _a.bias, activation = _a.activation, preluActivationWeights = _a.preluActivationWeights;
          var result = this.depthwiseConv2D(input, filter, convInfo);
          if (bias) {
              result = this.add(result, bias);
          }
          if (activation) {
              result =
                  mapActivation(this, result, activation, preluActivationWeights);
          }
          return result;
      };
      MathBackendCPU.prototype.depthwiseConv2D = function (x, filter, convInfo) {
          assertNotComplex([x, filter], 'depthwiseConv2D');
          var filterHeight = convInfo.filterHeight;
          var filterWidth = convInfo.filterWidth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var padLeft = convInfo.padInfo.left;
          var padTop = convInfo.padInfo.top;
          var chMul = convInfo.outChannels / convInfo.inChannels;
          var y = tf.buffer(convInfo.outShape, x.dtype);
          var xVals = this.readSync(x.dataId);
          var wVals = this.readSync(filter.dataId);
          var yVals = y.values;
          for (var b = 0; b < convInfo.batchSize; ++b) {
              var xOffset1 = b * x.strides[0];
              var yOffset1 = b * y.strides[0];
              for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                  var yOffset2 = yOffset1 + yR * y.strides[1];
                  var xRCorner = yR * convInfo.strideHeight - padLeft;
                  for (var wR = 0; wR < filterHeight; ++wR) {
                      var xR = xRCorner + wR * dilationHeight;
                      if (xR < 0 || xR >= convInfo.inHeight) {
                          continue;
                      }
                      var wOffset1 = wR * filter.strides[0];
                      var xOffset2 = xOffset1 + xR * x.strides[1];
                      for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                          var yOffset3 = yOffset2 + yC * y.strides[2];
                          var xCCorner = yC * convInfo.strideWidth - padTop;
                          for (var wC = 0; wC < filterWidth; ++wC) {
                              var xC = xCCorner + wC * dilationWidth;
                              if (xC < 0 || xC >= convInfo.inWidth) {
                                  continue;
                              }
                              var wOffset2 = wOffset1 + wC * filter.strides[1];
                              var xOffset3 = xOffset2 + xC * convInfo.inChannels;
                              var yOffset4 = yOffset3;
                              var wOffset3 = wOffset2;
                              for (var d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                  var xVal = xVals[xOffset3 + d1];
                                  for (var q = 0; q < chMul; ++q) {
                                      yVals[yOffset4 + q] += xVal * wVals[wOffset3 + q];
                                  }
                                  yOffset4 += chMul;
                                  wOffset3 += chMul;
                              }
                          }
                      }
                  }
              }
          }
          return y.toTensor();
      };
      MathBackendCPU.prototype.depthwiseConv2DDerInput = function (dy, filter, convInfo) {
          assertNotComplex([dy, filter], 'depthwiseConv2DDerInput');
          var dx = tf.buffer(convInfo.inShape, 'float32');
          var dxValues = dx.values;
          var _a = dx.strides, dxS0 = _a[0], dxS1 = _a[1], dxS2 = _a[2];
          var dyValues = this.readSync(dy.dataId);
          var _b = dy.strides, dyS0 = _b[0], dyS1 = _b[1], dyS2 = _b[2];
          var fltValues = this.readSync(filter.dataId);
          var _c = filter.strides, fltS0 = _c[0], fltS1 = _c[1], fltS2 = _c[2];
          var batchSize = convInfo.batchSize, filterHeight = convInfo.filterHeight, filterWidth = convInfo.filterWidth, inChannels = convInfo.inChannels, inHeight = convInfo.inHeight, inWidth = convInfo.inWidth, outChannels = convInfo.outChannels, outHeight = convInfo.outHeight, outWidth = convInfo.outWidth, strideHeight = convInfo.strideHeight, strideWidth = convInfo.strideWidth;
          var topPad = filterHeight - 1 - convInfo.padInfo.top;
          var leftPad = filterWidth - 1 - convInfo.padInfo.left;
          var chMul = outChannels / inChannels;
          for (var b = 0; b < batchSize; ++b) {
              for (var d1 = 0; d1 < inChannels; ++d1) {
                  for (var xR = 0; xR < inHeight; ++xR) {
                      var xRCorner = xR - topPad;
                      var xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                      var yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                      for (var xC = 0; xC < inWidth; ++xC) {
                          var xCCorner = xC - leftPad;
                          var xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                          var yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                          var dotProd = 0;
                          for (var yR = xRMin; yR < yRMax; ++yR) {
                              var wR = yR * strideHeight - xRCorner;
                              for (var yC = xCMin; yC < yCMax; ++yC) {
                                  var wC = yC * strideWidth - xCCorner;
                                  var dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                                  var fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                      fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                                  for (var dm = 0; dm < chMul; ++dm) {
                                      var d2 = d1 * chMul + dm;
                                      var pixel = dyValues[dyOffset + d2];
                                      var weight = fltValues[fltOffset + dm];
                                      dotProd += pixel * weight;
                                  }
                              }
                          }
                          dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
                      }
                  }
              }
          }
          return dx.toTensor();
      };
      MathBackendCPU.prototype.depthwiseConv2DDerFilter = function (x, dy, convInfo) {
          assertNotComplex([x, dy], 'depthwiseConv2DDerFilter');
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var filterHeight = convInfo.filterHeight;
          var filterWidth = convInfo.filterWidth;
          var dW = tf.buffer(convInfo.filterShape, 'float32');
          var leftPad = convInfo.padInfo.left;
          var topPad = convInfo.padInfo.top;
          var chMul = convInfo.outChannels / convInfo.inChannels;
          var xBuf = this.bufferSync(x);
          var dyBuf = this.bufferSync(dy);
          for (var wR = 0; wR < filterHeight; ++wR) {
              var yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
              var yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
              for (var wC = 0; wC < filterWidth; ++wC) {
                  var yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                  var yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                  for (var d2 = 0; d2 < convInfo.outChannels; ++d2) {
                      var d1 = Math.trunc(d2 / chMul);
                      var dm = d2 % chMul;
                      var dotProd = 0;
                      for (var b = 0; b < convInfo.batchSize; ++b) {
                          for (var yR = yRMin; yR < yRMax; ++yR) {
                              var xR = wR + yR * strideHeight - topPad;
                              for (var yC = yCMin; yC < yCMax; ++yC) {
                                  var xC = wC + yC * strideWidth - leftPad;
                                  dotProd += xBuf.get(b, xR, xC, d1) * dyBuf.get(b, yR, yC, d2);
                              }
                          }
                      }
                      dW.set(dotProd, wR, wC, d1, dm);
                  }
              }
          }
          return dW.toTensor();
      };
      MathBackendCPU.prototype.tile = function (x, reps) {
          assertNotComplex(x, 'tile');
          return tile(this.bufferSync(x), reps);
      };
      MathBackendCPU.prototype.pad = function (x, paddings, constantValue) {
          assertNotComplex(x, 'pad');
          var outShape = paddings.map(function (p, i) { return p[0] /* beforePad */ + x.shape[i] + p[1]; } /* afterPad */);
          var start = paddings.map(function (p) { return p[0]; });
          var xBuffer = this.bufferSync(x);
          var buffer = tf.buffer(outShape, x.dtype);
          if (constantValue !== 0) {
              buffer.values.fill(constantValue);
          }
          for (var i = 0; i < x.size; i++) {
              var coords = xBuffer.indexToLoc(i);
              var outCoords = coords.map(function (c, i) { return c + start[i]; });
              buffer.set.apply(buffer, [xBuffer.get.apply(xBuffer, coords)].concat(outCoords));
          }
          return buffer.toTensor();
      };
      MathBackendCPU.prototype.gather = function (x, indices, axis) {
          assertNotComplex([x, indices], 'gather');
          var newShape = x.shape.slice();
          var indicesValues = this.readSync(indices.dataId);
          newShape[axis] = indicesValues.length;
          var result = tf.buffer(newShape, x.dtype);
          var xBuf = this.bufferSync(x);
          for (var i = 0; i < result.size; ++i) {
              var newLoc = result.indexToLoc(i);
              var originalLoc = newLoc.slice();
              originalLoc[axis] = indicesValues[newLoc[axis]];
              var originalIndex = xBuf.locToIndex(originalLoc);
              result.values[i] = xBuf.values[originalIndex];
          }
          return result.toTensor();
      };
      MathBackendCPU.prototype.batchToSpaceND = function (x, blockShape, crops) {
          assertNotComplex([x], 'batchToSpaceND');
          var prod = blockShape.reduce(function (a, b) { return a * b; });
          var reshaped = tf.backend_util.getReshaped(x.shape, blockShape, prod);
          var permuted = tf.backend_util.getPermuted(reshaped.length, blockShape.length);
          var reshapedPermuted = tf.backend_util.getReshapedPermuted(x.shape, blockShape, prod);
          var sliceBeginCoords = tf.backend_util.getSliceBeginCoords(crops, blockShape.length);
          var sliceSize = tf.backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);
          return tf.transpose(x.reshape(reshaped), permuted)
              .reshape(reshapedPermuted)
              .slice(sliceBeginCoords, sliceSize);
      };
      MathBackendCPU.prototype.spaceToBatchND = function (x, blockShape, paddings) {
          assertNotComplex([x], 'spaceToBatchND');
          var prod = blockShape.reduce(function (a, b) { return a * b; });
          var completePaddings = [[0, 0]];
          completePaddings.push.apply(completePaddings, paddings);
          for (var i = 1 + blockShape.length; i < x.shape.length; ++i) {
              completePaddings.push([0, 0]);
          }
          var paddedX = x.pad(completePaddings);
          var reshapedPaddedShape = tf.backend_util.getReshaped(paddedX.shape, blockShape, prod, false);
          var permutedReshapedPaddedPermutation = tf.backend_util.getPermuted(reshapedPaddedShape.length, blockShape.length, false);
          var flattenShape = tf.backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);
          return tf.transpose(paddedX.reshape(reshapedPaddedShape), permutedReshapedPaddedPermutation)
              .reshape(flattenShape);
      };
      MathBackendCPU.prototype.maxPool = function (x, convInfo) {
          assertNotComplex(x, 'maxPool');
          var xValues = this.readSync(x.dataId);
          return pool(xValues, x.shape, x.dtype, x.strides, convInfo, 'max')
              .toTensor();
      };
      MathBackendCPU.prototype.maxPoolBackprop = function (dy, x, y, convInfo) {
          assertNotComplex([x, y], 'maxPoolBackprop');
          var xValues = this.readSync(x.dataId);
          var maxPosBuf = tf.buffer(convInfo.outShape, x.dtype, maxPoolPositions(xValues, x.shape, x.dtype, convInfo).values);
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var effectiveFilterHeight = convInfo.effectiveFilterHeight;
          var effectiveFilterWidth = convInfo.effectiveFilterWidth;
          var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          var dx = tf.buffer(x.shape, 'float32');
          var dyBuf = this.bufferSync(dy);
          for (var b = 0; b < convInfo.batchSize; ++b) {
              for (var d = 0; d < convInfo.inChannels; ++d) {
                  for (var dxR = 0; dxR < convInfo.inHeight; ++dxR) {
                      for (var dxC = 0; dxC < convInfo.inWidth; ++dxC) {
                          // Shader code begins.
                          var dyRCorner = dxR - padTop;
                          var dyCCorner = dxC - padLeft;
                          var dotProd = 0;
                          for (var wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
                              var dyR = (dyRCorner + wR) / strideHeight;
                              if (dyR < 0 || dyR >= convInfo.outHeight ||
                                  Math.floor(dyR) !== dyR) {
                                  continue;
                              }
                              for (var wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                                  var dyC = (dyCCorner + wC) / strideWidth;
                                  if (dyC < 0 || dyC >= convInfo.outWidth ||
                                      Math.floor(dyC) !== dyC) {
                                      continue;
                                  }
                                  var maxPos = effectiveFilterHeight * effectiveFilterWidth -
                                      1 - maxPosBuf.get(b, dyR, dyC, d);
                                  var curPos = wR * effectiveFilterWidth + wC;
                                  var mask = maxPos === curPos ? 1 : 0;
                                  if (mask === 0) {
                                      continue;
                                  }
                                  var pixel = dyBuf.get(b, dyR, dyC, d);
                                  dotProd += pixel * mask;
                              }
                          }
                          dx.set(dotProd, b, dxR, dxC, d);
                      }
                  }
              }
          }
          return dx.toTensor();
      };
      MathBackendCPU.prototype.avgPoolBackprop = function (dy, x, convInfo) {
          assertNotComplex([dy, x], 'avgPoolBackprop');
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var filterHeight = convInfo.filterHeight;
          var filterWidth = convInfo.filterWidth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var effectiveFilterHeight = convInfo.effectiveFilterHeight;
          var effectiveFilterWidth = convInfo.effectiveFilterWidth;
          var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          var dx = tf.buffer(x.shape, 'float32');
          var avgMultiplier = 1 / (filterHeight * filterWidth);
          var dyBuf = this.bufferSync(dy);
          for (var b = 0; b < convInfo.batchSize; ++b) {
              for (var d = 0; d < convInfo.inChannels; ++d) {
                  for (var dxR = 0; dxR < convInfo.inHeight; ++dxR) {
                      for (var dxC = 0; dxC < convInfo.inWidth; ++dxC) {
                          // Shader code begins.
                          var dyRCorner = dxR - padTop;
                          var dyCCorner = dxC - padLeft;
                          var dotProd = 0;
                          for (var wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
                              var dyR = (dyRCorner + wR) / strideHeight;
                              if (dyR < 0 || dyR >= convInfo.outHeight ||
                                  Math.floor(dyR) !== dyR) {
                                  continue;
                              }
                              for (var wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                                  var dyC = (dyCCorner + wC) / strideWidth;
                                  if (dyC < 0 || dyC >= convInfo.outWidth ||
                                      Math.floor(dyC) !== dyC) {
                                      continue;
                                  }
                                  var pixel = dyBuf.get(b, dyR, dyC, d);
                                  dotProd += pixel;
                              }
                          }
                          dx.set(dotProd * avgMultiplier, b, dxR, dxC, d);
                      }
                  }
              }
          }
          return dx.toTensor();
      };
      MathBackendCPU.prototype.pool3d = function (x, convInfo, poolType) {
          assertNotComplex(x, 'pool3d');
          var strideDepth = convInfo.strideDepth;
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var dilationDepth = convInfo.dilationDepth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var effectiveFilterDepth = convInfo.effectiveFilterDepth;
          var effectiveFilterHeight = convInfo.effectiveFilterHeight;
          var effectiveFilterWidth = convInfo.effectiveFilterWidth;
          var padFront = convInfo.padInfo.front;
          var padTop = convInfo.padInfo.top;
          var padLeft = convInfo.padInfo.left;
          var initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
              Number.POSITIVE_INFINITY);
          var xValues = this.readSync(x.dataId);
          var output = tf.buffer(convInfo.outShape, x.dtype);
          var outputVals = output.values;
          var outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] *
              convInfo.outShape[3] * convInfo.outShape[4];
          var outputDepthStrides = convInfo.outShape[2] * convInfo.outShape[3] * convInfo.outShape[4];
          var outputRowStrides = convInfo.outShape[3] * convInfo.outShape[4];
          var outputColStrides = convInfo.outShape[4];
          for (var batch = 0; batch < convInfo.batchSize; ++batch) {
              var outputBatchOffset = batch * outputBatchStrides;
              var inputBatchOffset = batch * x.strides[0];
              for (var channel = 0; channel < convInfo.inChannels; ++channel) {
                  for (var yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                      var xDepthCorner = yDepth * strideDepth - padFront;
                      var xDepthMin = xDepthCorner;
                      while (xDepthMin < 0) {
                          xDepthMin += dilationDepth;
                      }
                      var xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                      var outputDepthOffset = outputBatchOffset + yDepth * outputDepthStrides;
                      for (var yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                          var xRowCorner = yRow * strideHeight - padTop;
                          var xRowMin = xRowCorner;
                          while (xRowMin < 0) {
                              xRowMin += dilationHeight;
                          }
                          var xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                          var outputRowOffset = outputDepthOffset + yRow * outputRowStrides;
                          for (var yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                              var xColCorner = yCol * strideWidth - padLeft;
                              var xColMin = xColCorner;
                              while (xColMin < 0) {
                                  xColMin += dilationWidth;
                              }
                              var xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                              // Shader code begins
                              var outputColOffset = outputRowOffset + yCol * outputColStrides;
                              var minMaxValue = initialValue;
                              var avgValue = 0;
                              var count = 0;
                              for (var xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                                  var xDepthOffset = inputBatchOffset + xDepth * x.strides[1];
                                  for (var xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                      var xRowOffset = xDepthOffset + xRow * x.strides[2];
                                      for (var xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                          var xColOffset = xRowOffset + xCol * x.strides[3];
                                          var pixel = xValues[xColOffset + channel];
                                          if ((poolType === 'max' && pixel > minMaxValue)) {
                                              minMaxValue = pixel;
                                          }
                                          else if (poolType === 'avg') {
                                              avgValue += pixel;
                                              count++;
                                          }
                                          if (isNaN(minMaxValue)) {
                                              break;
                                          }
                                      }
                                      if (isNaN(minMaxValue)) {
                                          break;
                                      }
                                  }
                                  if (isNaN(minMaxValue)) {
                                      break;
                                  }
                              }
                              var outputOffset = outputColOffset + channel;
                              outputVals[outputOffset] =
                                  poolType === 'avg' ? avgValue / count : minMaxValue;
                          }
                      }
                  }
              }
          }
          return output.toTensor();
      };
      MathBackendCPU.prototype.avgPool3d = function (x, convInfo) {
          assertNotComplex(x, 'avgPool3d');
          return this.pool3d(x, convInfo, 'avg').toFloat();
      };
      MathBackendCPU.prototype.avgPool3dBackprop = function (dy, x, convInfo) {
          assertNotComplex([dy, x], 'avgPool3dBackprop');
          var strideDepth = convInfo.strideDepth;
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var filterDepth = convInfo.filterDepth;
          var filterHeight = convInfo.filterHeight;
          var filterWidth = convInfo.filterWidth;
          var dilationDepth = convInfo.dilationDepth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var effectiveFilterDepth = convInfo.effectiveFilterDepth;
          var effectiveFilterHeight = convInfo.effectiveFilterHeight;
          var effectiveFilterWidth = convInfo.effectiveFilterWidth;
          var padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
          var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          var dx = tf.buffer(x.shape, 'float32');
          var avgMultiplier = 1 / (filterDepth * filterHeight * filterWidth);
          var dyBuf = this.bufferSync(dy);
          for (var batch = 0; batch < convInfo.batchSize; ++batch) {
              for (var channel = 0; channel < convInfo.inChannels; ++channel) {
                  for (var dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
                      for (var dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
                          for (var dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
                              // Shader code begins.
                              var dyDepthCorner = dxDepth - padFront;
                              var dyRowCorner = dxRow - padTop;
                              var dyColCorner = dxCol - padLeft;
                              var dotProd = 0;
                              for (var wDepth = 0; wDepth < effectiveFilterDepth; wDepth += dilationDepth) {
                                  var dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                                  if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                                      Math.floor(dyDepth) !== dyDepth) {
                                      continue;
                                  }
                                  for (var wRow = 0; wRow < effectiveFilterHeight; wRow += dilationHeight) {
                                      var dyRow = (dyRowCorner + wRow) / strideHeight;
                                      if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                                          Math.floor(dyRow) !== dyRow) {
                                          continue;
                                      }
                                      for (var wCol = 0; wCol < effectiveFilterWidth; wCol += dilationWidth) {
                                          var dyCol = (dyColCorner + wCol) / strideWidth;
                                          if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                                              Math.floor(dyCol) !== dyCol) {
                                              continue;
                                          }
                                          var pixel = dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                          dotProd += pixel;
                                      }
                                  }
                              }
                              dx.set(dotProd * avgMultiplier, batch, dxDepth, dxRow, dxCol, channel);
                          }
                      }
                  }
              }
          }
          return dx.toTensor();
      };
      MathBackendCPU.prototype.maxPool3d = function (x, convInfo) {
          assertNotComplex(x, 'maxPool3d');
          return this.pool3d(x, convInfo, 'max').toFloat();
      };
      MathBackendCPU.prototype.maxPool3dPositions = function (x, convInfo) {
          var maxPositions = tf.buffer(convInfo.outShape, 'int32');
          var strideDepth = convInfo.strideDepth;
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var dilationDepth = convInfo.dilationDepth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var effectiveFilterDepth = convInfo.effectiveFilterDepth;
          var effectiveFilterHeight = convInfo.effectiveFilterHeight;
          var effectiveFilterWidth = convInfo.effectiveFilterWidth;
          var padFront = convInfo.padInfo.front;
          var padTop = convInfo.padInfo.top;
          var padLeft = convInfo.padInfo.left;
          var xBuf = this.bufferSync(x);
          for (var batch = 0; batch < convInfo.batchSize; ++batch) {
              for (var channel = 0; channel < convInfo.inChannels; ++channel) {
                  for (var yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                      var xDepthCorner = yDepth * strideDepth - padFront;
                      var xDepthMin = xDepthCorner;
                      while (xDepthMin < 0) {
                          xDepthMin += dilationDepth;
                      }
                      var xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                      for (var yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                          var xRowCorner = yRow * strideHeight - padTop;
                          var xRowMin = xRowCorner;
                          while (xRowMin < 0) {
                              xRowMin += dilationHeight;
                          }
                          var xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                          for (var yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                              var xColCorner = yCol * strideWidth - padLeft;
                              var xColMin = xColCorner;
                              while (xColMin < 0) {
                                  xColMin += dilationWidth;
                              }
                              var xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                              // Shader code begins
                              var maxValue = Number.NEGATIVE_INFINITY;
                              var maxPosition = -1;
                              for (var xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                                  var wDepth = xDepth - xDepthCorner;
                                  for (var xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                      var wRow = xRow - xRowCorner;
                                      for (var xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                          var wCol = xCol - xColCorner;
                                          var pixel = xBuf.get(batch, xDepth, xRow, xCol, channel);
                                          if (pixel >= maxValue) {
                                              maxValue = pixel;
                                              maxPosition = wDepth * effectiveFilterHeight *
                                                  effectiveFilterWidth +
                                                  wRow * effectiveFilterHeight + wCol;
                                          }
                                      }
                                  }
                              }
                              maxPositions.set(maxPosition, batch, yDepth, yRow, yCol, channel);
                          }
                      }
                  }
              }
          }
          return maxPositions.toTensor();
      };
      MathBackendCPU.prototype.maxPool3dBackprop = function (dy, x, y, convInfo) {
          assertNotComplex([x, y], 'maxPool3dBackprop');
          var maxPositions = this.maxPool3dPositions(x, convInfo);
          var strideDepth = convInfo.strideDepth;
          var strideHeight = convInfo.strideHeight;
          var strideWidth = convInfo.strideWidth;
          var dilationDepth = convInfo.dilationDepth;
          var dilationHeight = convInfo.dilationHeight;
          var dilationWidth = convInfo.dilationWidth;
          var effectiveFilterDepth = convInfo.effectiveFilterDepth;
          var effectiveFilterHeight = convInfo.effectiveFilterHeight;
          var effectiveFilterWidth = convInfo.effectiveFilterWidth;
          var padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
          var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          var dx = tf.buffer(x.shape, 'float32');
          var maxPosBuf = this.bufferSync(maxPositions);
          var dyBuf = this.bufferSync(dy);
          for (var batch = 0; batch < convInfo.batchSize; ++batch) {
              for (var channel = 0; channel < convInfo.inChannels; ++channel) {
                  for (var dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
                      for (var dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
                          for (var dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
                              // Shader code begins
                              var dyDepthCorner = dxDepth - padFront;
                              var dyRowCorner = dxRow - padTop;
                              var dyColCorner = dxCol - padLeft;
                              var dotProd = 0;
                              for (var wDepth = 0; wDepth < effectiveFilterDepth; wDepth += dilationDepth) {
                                  var dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                                  if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                                      Math.floor(dyDepth) !== dyDepth) {
                                      continue;
                                  }
                                  for (var wRow = 0; wRow < effectiveFilterHeight; wRow += dilationHeight) {
                                      var dyRow = (dyRowCorner + wRow) / strideHeight;
                                      if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                                          Math.floor(dyRow) !== dyRow) {
                                          continue;
                                      }
                                      for (var wCol = 0; wCol < effectiveFilterWidth; wCol += dilationWidth) {
                                          var dyCol = (dyColCorner + wCol) / strideWidth;
                                          if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                                              Math.floor(dyCol) !== dyCol) {
                                              continue;
                                          }
                                          var maxPos = effectiveFilterDepth *
                                              effectiveFilterHeight * effectiveFilterWidth -
                                              1 -
                                              maxPosBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                          var curPos = wDepth * effectiveFilterHeight * effectiveFilterWidth +
                                              wRow * effectiveFilterWidth + wCol;
                                          var mask = maxPos === curPos ? 1 : 0;
                                          if (mask === 0) {
                                              continue;
                                          }
                                          var pixel = dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                          dotProd += pixel * mask;
                                      }
                                  }
                              }
                              dx.set(dotProd, batch, dxDepth, dxRow, dxCol, channel);
                          }
                      }
                  }
              }
          }
          return dx.toTensor();
      };
      MathBackendCPU.prototype.cast = function (x, dtype) {
          return tf.backend_util.castTensor(x, dtype, this);
      };
      MathBackendCPU.prototype.reshape = function (x, shape) {
          return tf.backend_util.reshapeTensor(x, shape);
      };
      MathBackendCPU.prototype.avgPool = function (x, convInfo) {
          assertNotComplex(x, 'avgPool');
          assertNotComplex(x, 'maxPool');
          var xValues = this.readSync(x.dataId);
          return pool(xValues, x.shape, x.dtype, x.strides, convInfo, 'avg')
              .toTensor()
              .toFloat();
      };
      MathBackendCPU.prototype.resizeBilinear = function (x, newHeight, newWidth, alignCorners) {
          assertNotComplex(x, 'resizeBilinear');
          var _a = x.shape, batch = _a[0], oldHeight = _a[1], oldWidth = _a[2], numChannels = _a[3];
          var xValues = this.readSync(x.dataId);
          var result = new Float32Array(tf.util.sizeFromShape([batch, newHeight, newWidth, numChannels]));
          var effectiveInputSize = [
              (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
              (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
          ];
          var effectiveOutputSize = [
              (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
              (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
          ];
          var outputIdx = 0;
          var effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
          var effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
          for (var b = 0; b < batch; b++) {
              for (var r = 0; r < newHeight; r++) {
                  var sourceFracRow = effectiveRowSizeRatio * r;
                  var sourceRowFloor = Math.floor(sourceFracRow);
                  var rowFrac = sourceFracRow - sourceRowFloor;
                  var sourceRowCeil = Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
                  var topRowOffset = b * x.strides[0] + sourceRowFloor * x.strides[1];
                  var botRowOffset = b * x.strides[0] + sourceRowCeil * x.strides[1];
                  for (var c = 0; c < newWidth; c++) {
                      var sourceFracCol = effectiveColSizeRatio * c;
                      var sourceColFloor = Math.floor(sourceFracCol);
                      var colFrac = sourceFracCol - sourceColFloor;
                      var sourceColCeil = Math.min(oldWidth - 1, Math.ceil(sourceFracCol));
                      var topLeftOffest = topRowOffset + sourceColFloor * x.strides[2];
                      var botLeftOffset = botRowOffset + sourceColFloor * x.strides[2];
                      var topRightOffset = topRowOffset + sourceColCeil * x.strides[2];
                      var botRightOffest = botRowOffset + sourceColCeil * x.strides[2];
                      for (var d = 0; d < numChannels; d++) {
                          // Begin shader.
                          // Compute the fractional index of the source.
                          var topLeft = xValues[topLeftOffest + d];
                          var bottomLeft = xValues[botLeftOffset + d];
                          var topRight = xValues[topRightOffset + d];
                          var bottomRight = xValues[botRightOffest + d];
                          var top_1 = topLeft + (topRight - topLeft) * colFrac;
                          var bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                          var newValue = top_1 + (bottom - top_1) * rowFrac;
                          result[outputIdx++] = newValue;
                      }
                  }
              }
          }
          return tf.tensor(result, [batch, newHeight, newWidth, numChannels]);
      };
      MathBackendCPU.prototype.resizeBilinearBackprop = function (dy, x, alignCorners) {
          assertNotComplex([dy, x], 'resizeBilinearBackprop');
          var _a = x.shape, batch = _a[0], xHeight = _a[1], xWidth = _a[2], depth = _a[3];
          var _b = dy.shape, yHeight = _b[1], yWidth = _b[2];
          var output = new Float32Array(batch * xHeight * xWidth * depth);
          // In the backwards pass, we want to find the pixels that were generated
          // for each pixel in the input image the forward pass and add the
          // corresponding coefficient from dy to the gradient (with some
          // interpolation).
          var effectiveXSize = [
              (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
              (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
          ];
          var effectiveYSize = [
              (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
              (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
          ];
          var heightScale = effectiveXSize[0] / effectiveYSize[0];
          var widthScale = effectiveXSize[1] / effectiveYSize[1];
          // Reference implementation
          // tslint:disable-next-line:max-line-length
          // https://github.com/tensorflow/tensorflow/blob/3039375c86a5bbc9610c7725dcaa95d635f87ba2/tensorflow/core/kernels/resize_bilinear_op.cc#L275
          var dyValues = this.readSync(dy.dataId);
          var offset = 0;
          for (var b = 0; b < batch; b++) {
              var bOffset = b * x.strides[0];
              for (var r = 0; r < yHeight; r++) {
                  var dxR = r * heightScale;
                  var topDxRIndex = Math.floor(dxR);
                  var bottomDxRIndex = Math.min(Math.ceil(dxR), xHeight - 1);
                  var topDxROffset = bOffset + topDxRIndex * x.strides[1];
                  var bottomDxROffset = bOffset + bottomDxRIndex * x.strides[1];
                  var dxRLerp = dxR - topDxRIndex;
                  var inverseDxRLerp = 1.0 - dxRLerp;
                  for (var c = 0; c < yWidth; c++) {
                      var dxC = c * widthScale;
                      var leftDxCIndex = Math.floor(dxC);
                      var rightDxCIndex = Math.min(Math.ceil(dxC), xWidth - 1);
                      var dxCLerp = dxC - leftDxCIndex;
                      var inverseDxCLerp = 1.0 - dxCLerp;
                      var topLeftRCOffset = topDxROffset + leftDxCIndex * x.strides[2];
                      var topRightRCOffset = topDxROffset + rightDxCIndex * x.strides[2];
                      var bottomLeftRCOffset = bottomDxROffset + leftDxCIndex * x.strides[2];
                      var bottomRightRCOffset = bottomDxROffset + rightDxCIndex * x.strides[2];
                      var inverseDxRLerpTimesInverseDxCLerp = inverseDxRLerp * inverseDxCLerp;
                      var inverseDxRLerpTimesDxCLerp = inverseDxRLerp * dxCLerp;
                      var dxRLerpTimesInverseDxCLerp = dxRLerp * inverseDxCLerp;
                      var dxRLerpTimesDxCLerp = dxRLerp * dxCLerp;
                      for (var d = 0; d < depth; d++) {
                          var dyVal = dyValues[offset++];
                          output[topLeftRCOffset + d] +=
                              dyVal * inverseDxRLerpTimesInverseDxCLerp;
                          output[topRightRCOffset + d] += dyVal * inverseDxRLerpTimesDxCLerp;
                          output[bottomLeftRCOffset + d] +=
                              dyVal * dxRLerpTimesInverseDxCLerp;
                          output[bottomRightRCOffset + d] += dyVal * dxRLerpTimesDxCLerp;
                      }
                  }
              }
          }
          return tf.tensor4d(output, [batch, xWidth, xHeight, depth], x.dtype);
      };
      MathBackendCPU.prototype.resizeNearestNeighbor = function (x, newHeight, newWidth, alignCorners) {
          assertNotComplex(x, 'resizeNearestNeighbor');
          var _a = x.shape, batch = _a[0], oldHeight = _a[1], oldWidth = _a[2], numChannels = _a[3];
          var xValues = this.readSync(x.dataId);
          var output = new Float32Array(batch * newHeight * newWidth * numChannels);
          var effectiveInputSize = [
              (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
              (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
          ];
          var effectiveOutputSize = [
              (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
              (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
          ];
          var effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
          var effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
          var outputOffset = 0;
          for (var b = 0; b < batch; b++) {
              var batchOffset = b * x.strides[0];
              for (var r = 0; r < newHeight; r++) {
                  var sourceFracRow = effectiveRowSizeRatio * r;
                  var sourceNearestRow = Math.min(oldHeight - 1, alignCorners ? Math.round(sourceFracRow) :
                      Math.floor(sourceFracRow));
                  var rowOffset = batchOffset + sourceNearestRow * x.strides[1];
                  for (var c = 0; c < newWidth; c++) {
                      var sourceFracCol = effectiveColSizeRatio * c;
                      var sourceNearestCol = Math.min(oldWidth - 1, alignCorners ? Math.round(sourceFracCol) :
                          Math.floor(sourceFracCol));
                      var colOffset = rowOffset + sourceNearestCol * x.strides[2];
                      for (var d = 0; d < numChannels; d++) {
                          // Begin shader.
                          // Compute the fractional index of the source.
                          var newVal = xValues[colOffset + d];
                          output[outputOffset++] = newVal;
                      }
                  }
              }
          }
          return tf.tensor(output, [batch, newHeight, newWidth, numChannels], x.dtype);
      };
      MathBackendCPU.prototype.resizeNearestNeighborBackprop = function (dy, x, alignCorners) {
          assertNotComplex([dy, x], 'resizeNearestNeighborBackprop');
          var _a = x.shape, batch = _a[0], xHeight = _a[1], xWidth = _a[2], depth = _a[3];
          var _b = dy.shape, yHeight = _b[1], yWidth = _b[2];
          var output = new Float32Array(batch * xHeight * xWidth * depth);
          var dyValues = this.readSync(dy.dataId);
          // In the backwards pass, we want to find the pixels that were generated
          // for each pixel in the input image the forward pass
          var effectiveXSize = [
              (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
              (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
          ];
          var effectiveYSize = [
              (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
              (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
          ];
          var heightScale = effectiveXSize[0] / effectiveYSize[0];
          var widthScale = effectiveXSize[1] / effectiveYSize[1];
          var invHeightScale = 1 / heightScale;
          var invWidthScale = 1 / widthScale;
          // This defines the size of the window of values around a particular
          // index in dy that we want to search for contributions to dx.
          var winHeight = (Math.ceil(invHeightScale) * 2) + 2;
          var winWidth = (Math.ceil(invWidthScale) * 2) + 2;
          // Loop over the output space.
          for (var b = 0; b < batch; b++) {
              var batchOffset = b * x.strides[0];
              for (var r = 0; r < xHeight; r++) {
                  var rowOffset = batchOffset + r * x.strides[1];
                  // Compute bounds for where in dy we will look
                  var startRLerp = Math.floor(r * invHeightScale);
                  var startDyR = Math.floor(startRLerp - (winHeight / 2));
                  for (var c = 0; c < xWidth; c++) {
                      var colOffset = rowOffset + c * x.strides[2];
                      // Compute bounds for where in dy we will look
                      var startCLerp = Math.floor(c * invWidthScale);
                      var startDyC = Math.floor(startCLerp - (winWidth / 2));
                      for (var d = 0; d < depth; d++) {
                          var accum = 0;
                          // loop over dy
                          for (var dyRIndex = 0; dyRIndex < winHeight; dyRIndex++) {
                              var dyR = dyRIndex + startDyR;
                              // Guard against the window exceeding the bounds of dy
                              if (dyR < 0 || dyR >= yHeight) {
                                  continue;
                              }
                              var dyROffset = batchOffset + dyR * dy.strides[1];
                              var sourceFracRow = dyR * heightScale;
                              var sourceNearestRow = Math.min(xHeight - 1, alignCorners ? Math.round(sourceFracRow) :
                                  Math.floor(sourceFracRow));
                              if (r !== sourceNearestRow) {
                                  continue;
                              }
                              for (var dyCIndex = 0; dyCIndex < winWidth; dyCIndex++) {
                                  var dyC = dyCIndex + startDyC;
                                  // Guard against the window exceeding the bounds of dy
                                  if (dyC < 0 || dyC >= yWidth) {
                                      continue;
                                  }
                                  var dyCOffset = dyROffset + dyC * dy.strides[2];
                                  var sourceFracCol = dyC * widthScale;
                                  var sourceNearestCol = Math.min(xWidth - 1, alignCorners ? Math.round(sourceFracCol) :
                                      Math.floor(sourceFracCol));
                                  if (c === sourceNearestCol) {
                                      accum += dyValues[dyCOffset + d];
                                  }
                              }
                          }
                          output[colOffset + d] = accum;
                      }
                  }
              }
          }
          return tf.tensor4d(output, x.shape, x.dtype);
      };
      MathBackendCPU.prototype.batchNorm = function (x, mean, variance, offset, scale, varianceEpsilon) {
          assertNotComplex([x, mean, variance, scale, offset], 'batchNorm');
          var xVals = this.readSync(x.dataId);
          var mVals = this.readSync(mean.dataId);
          var varVals = this.readSync(variance.dataId);
          var sVals = scale ? this.readSync(scale.dataId) :
              new Float32Array([1]);
          var offVals = offset ? this.readSync(offset.dataId) :
              new Float32Array([0]);
          var outVals = new Float32Array(xVals.length);
          var offValsLength = offVals.length;
          var sValsLength = sVals.length;
          var varValsLength = varVals.length;
          var mValsLength = mVals.length;
          var offi = 0;
          var mi = 0;
          var si = 0;
          var vi = 0;
          for (var i = 0; i < xVals.length; ++i) {
              outVals[i] = offVals[offi++] +
                  (xVals[i] - mVals[mi++]) * sVals[si++] /
                      Math.sqrt(varVals[vi++] + varianceEpsilon);
              if (offi >= offValsLength) {
                  offi = 0;
              }
              if (mi >= mValsLength) {
                  mi = 0;
              }
              if (si >= sValsLength) {
                  si = 0;
              }
              if (vi >= varValsLength) {
                  vi = 0;
              }
          }
          return tf.tensor4d(outVals, x.shape);
      };
      MathBackendCPU.prototype.localResponseNormalization4D = function (x, depthRadius, bias, alpha, beta) {
          assertNotComplex(x, 'localResponseNormalization4D');
          var channels = x.shape[3];
          var maxD = channels - 1;
          var xValues = this.readSync(x.dataId);
          var size = x.size;
          var result = new Float32Array(size);
          function sumAcrossChannels(offset) {
              var currentChannel = offset % channels;
              var beginSumOffset = offset - currentChannel + Math.max(0, currentChannel - depthRadius);
              var endSumOffset = offset - currentChannel +
                  Math.min(currentChannel + depthRadius, maxD);
              var sum = 0.0;
              for (; beginSumOffset <= endSumOffset; beginSumOffset++) {
                  var z = xValues[beginSumOffset];
                  sum += z * z;
              }
              return sum;
          }
          for (var offset = 0; offset < size; offset++) {
              var sum = sumAcrossChannels(offset);
              var val = xValues[offset] * Math.pow(bias + alpha * sum, -beta);
              result[offset] = val;
          }
          return tf.tensor4d(result, x.shape);
      };
      MathBackendCPU.prototype.LRNGrad = function (dy, inputImage, outputImage, depthRadius, bias, alpha, beta) {
          assertNotComplex(dy, 'LRNGrad');
          var channels = dy.shape[3];
          var dyValues = this.readSync(dy.dataId);
          var inputImageValues = this.readSync(inputImage.dataId);
          var outputImageValues = this.readSync(outputImage.dataId);
          var result = new Float32Array(dy.size);
          var size = dy.size;
          for (var offset = 0; offset < size; offset++) {
              var currentChannel = offset % channels;
              var depthBegin = (offset - currentChannel) + Math.max(0, currentChannel - depthRadius);
              var depthEnd = (offset - currentChannel) +
                  Math.min(channels, currentChannel + depthRadius + 1);
              var norm = 0;
              for (var k = depthBegin; k < depthEnd; k++) {
                  norm += Math.pow(inputImageValues[k], 2);
              }
              norm = alpha * norm + bias;
              for (var k = depthBegin; k < depthEnd; k++) {
                  var dyi = -2 * alpha * beta * inputImageValues[k] *
                      outputImageValues[offset] / norm;
                  if (offset === k) {
                      dyi += Math.pow(norm, -beta);
                  }
                  dyi *= dyValues[offset];
                  result[k] += dyi;
              }
          }
          return tf.tensor4d(result, dy.shape);
      };
      MathBackendCPU.prototype.multinomial = function (logits, normalized, numSamples, seed) {
          assertNotComplex(logits, 'multinomial');
          var probabilities = normalized ? logits : tf.softmax(logits);
          var batchSize = probabilities.shape[0];
          var numEvents = probabilities.shape[1];
          var res = tf.zeros([batchSize, numSamples], 'int32');
          var resVals = this.readSync(res.dataId);
          var probVals = this.readSync(probabilities.dataId);
          for (var b = 0; b < batchSize; ++b) {
              var offset = b * numEvents;
              // The cdf won't include the last event. It will be implicit if no other
              // event happened.
              var cdf = new Float32Array(numEvents - 1);
              cdf[0] = probVals[offset];
              for (var event_1 = 1; event_1 < cdf.length; ++event_1) {
                  cdf[event_1] = cdf[event_1 - 1] + probVals[offset + event_1];
              }
              var random = seedrandom.alea(seed.toString());
              var outOffset = b * numSamples;
              for (var sampleId = 0; sampleId < numSamples; ++sampleId) {
                  var r = random();
                  // Assume last event happened by default.
                  resVals[outOffset + sampleId] = cdf.length;
                  for (var event_2 = 0; event_2 < cdf.length; event_2++) {
                      if (r < cdf[event_2]) {
                          resVals[outOffset + sampleId] = event_2;
                          break;
                      }
                  }
              }
          }
          return res;
      };
      MathBackendCPU.prototype.oneHot = function (indices, depth, onValue, offValue) {
          assertNotComplex(indices, 'oneHot');
          var res = new Float32Array(indices.size * depth);
          res.fill(offValue);
          var indicesVal = this.readSync(indices.dataId);
          for (var event_3 = 0; event_3 < indices.size; ++event_3) {
              if (indicesVal[event_3] >= 0 && indicesVal[event_3] < depth) {
                  res[event_3 * depth + indicesVal[event_3]] = onValue;
              }
          }
          return tf.tensor2d(res, [indices.size, depth], 'int32');
      };
      MathBackendCPU.prototype.nonMaxSuppression = function (boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
          assertNotComplex(boxes, 'nonMaxSuppression');
          var boxesVals = this.readSync(boxes.dataId);
          var scoresVals = this.readSync(scores.dataId);
          return nonMaxSuppressionV3(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
      };
      MathBackendCPU.prototype.fft = function (x) {
          return this.fftBatch(x, false);
      };
      MathBackendCPU.prototype.ifft = function (x) {
          return this.fftBatch(x, true);
      };
      /**
       * Calculate FFT of inner most elements of batch tensor.
       */
      MathBackendCPU.prototype.fftBatch = function (x, inverse) {
          var batch = x.shape[0];
          var innerDim = x.shape[1];
          // Collects real and imaginary values separately.
          var realResult = tf.buffer(x.shape, 'float32');
          var imagResult = tf.buffer(x.shape, 'float32');
          var real = tf.real(x).as2D(batch, innerDim);
          var imag = tf.imag(x).as2D(batch, innerDim);
          for (var b = 0; b < batch; b++) {
              // TODO: Support slice ops for complex type.
              var r = real.slice([b, 0], [1, innerDim]);
              var i = imag.slice([b, 0], [1, innerDim]);
              var input = tf.complex(r, i);
              // Run FFT by batch element.
              var res = this.readSync(this.fftImpl(input, inverse).dataId);
              for (var d = 0; d < innerDim; d++) {
                  var c = tf.backend_util.getComplexWithIndex(res, d);
                  realResult.values[b * innerDim + d] = c.real;
                  imagResult.values[b * innerDim + d] = c.imag;
              }
          }
          var t = tf.complex(realResult.toTensor(), imagResult.toTensor());
          return t.as2D(batch, innerDim);
      };
      MathBackendCPU.prototype.fftImpl = function (x, inverse) {
          var x1D = x.as1D();
          var n = x1D.size;
          if (this.isExponentOf2(n)) {
              var result = this.fftRadix2(x1D, n, inverse).as2D(x.shape[0], x.shape[1]);
              if (inverse) {
                  result = tf.complex(tf.real(result).div(tf.scalar(n)), tf.imag(result).div(tf.scalar(n)));
              }
              return result;
          }
          else {
              var data = this.readSync(x.dataId);
              var rawOutput = this.fourierTransformByMatmul(data, n, inverse);
              var output = tf.backend_util.splitRealAndImagArrays(rawOutput);
              return tf.complex(output.real, output.imag).as2D(x.shape[0], x.shape[1]);
          }
      };
      MathBackendCPU.prototype.isExponentOf2 = function (size) {
          return (size & size - 1) === 0;
      };
      // FFT using Cooley-Tukey algorithm on radix 2 dimensional input.
      MathBackendCPU.prototype.fftRadix2 = function (input, size, inverse) {
          if (size === 1) {
              return input;
          }
          var data = this.readSync(input.dataId);
          var half = size / 2;
          var evenComplex = tf.backend_util.complexWithEvenIndex(data);
          var evenTensor = tf.complex(evenComplex.real, evenComplex.imag).as1D();
          var oddComplex = tf.backend_util.complexWithOddIndex(data);
          var oddTensor = tf.complex(oddComplex.real, oddComplex.imag).as1D();
          // Recursive call for half part of original input.
          evenTensor = this.fftRadix2(evenTensor, half, inverse);
          oddTensor = this.fftRadix2(oddTensor, half, inverse);
          var e = tf.backend_util.exponents(size, inverse);
          var exponent = tf.complex(e.real, e.imag).mul(oddTensor);
          var addPart = evenTensor.add(exponent);
          var subPart = evenTensor.sub(exponent);
          var realTensor = tf.real(addPart).concat(tf.real(subPart));
          var imagTensor = tf.imag(addPart).concat(tf.imag(subPart));
          return tf.complex(realTensor, imagTensor).as1D();
      };
      // Calculate fourier transform by multplying sinusoid matrix.
      MathBackendCPU.prototype.fourierTransformByMatmul = function (data, size, inverse) {
          var ret = new Float32Array(size * 2);
          // TODO: Use matmul instead once it supports complex64 type.
          for (var r = 0; r < size; r++) {
              var real = 0.0;
              var imag = 0.0;
              for (var c = 0; c < size; c++) {
                  var e = tf.backend_util.exponent(r * c, size, inverse);
                  var term = tf.backend_util.getComplexWithIndex(data, c);
                  real += term.real * e.real - term.imag * e.imag;
                  imag += term.real * e.imag + term.imag * e.real;
              }
              if (inverse) {
                  real /= size;
                  imag /= size;
              }
              tf.backend_util.assignToTypedArray(ret, real, imag, r);
          }
          return ret;
      };
      MathBackendCPU.prototype.depthToSpace = function (x, blockSize, dataFormat) {
          tf.util.assert(dataFormat === 'NHWC', function () { return "Only NHWC dataFormat supported on CPU for depthToSpace. Got " + dataFormat; });
          tf.util.assert(blockSize > 1, function () {
              return "blockSize should be > 1 for depthToSpace, but was: " + blockSize;
          });
          var batchSize = x.shape[0];
          var inputHeight = x.shape[1];
          var inputWidth = x.shape[2];
          var inputDepth = x.shape[3];
          var outputHeight = inputHeight * blockSize;
          var outputWidth = inputWidth * blockSize;
          var outputDepth = inputDepth / (blockSize * blockSize);
          var xValues = this.readSync(x.dataId);
          var result = new Float32Array(batchSize * outputHeight * outputWidth * outputDepth);
          var outputIdx = 0;
          for (var b = 0; b < batchSize; ++b) {
              for (var h = 0; h < outputHeight; ++h) {
                  var inH = Math.floor(h / blockSize);
                  var offsetH = (h % blockSize);
                  for (var w = 0; w < outputWidth; ++w) {
                      var inW = Math.floor(w / blockSize);
                      var offsetW = (w % blockSize);
                      var offsetD = (offsetH * blockSize + offsetW) * outputDepth;
                      for (var d = 0; d < outputDepth; ++d) {
                          var inD = d + offsetD;
                          var inputIdx = inD + inputDepth * (inW + inputWidth * (inH + inputHeight * b));
                          result[outputIdx++] = xValues[inputIdx];
                      }
                  }
              }
          }
          return tf.tensor4d(result, [batchSize, outputHeight, outputWidth, outputDepth]);
      };
      MathBackendCPU.prototype.broadcastedBinaryOp = function (a, b, dtype, op) {
          var newShape = tf.backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
          var result = tf.buffer(newShape, dtype);
          var aVals = this.readSync(a.dataId);
          var bVals = this.readSync(b.dataId);
          var aBroadcastDims = tf.backend_util.getBroadcastDims(a.shape, newShape);
          var bBroadcastDims = tf.backend_util.getBroadcastDims(b.shape, newShape);
          var resVals = result.values;
          if (aBroadcastDims.length + bBroadcastDims.length === 0) {
              for (var i = 0; i < resVals.length; ++i) {
                  resVals[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
              }
          }
          else {
              var aBuf = this.bufferSync(a);
              var bBuf = this.bufferSync(b);
              var _loop_2 = function (i) {
                  var loc = result.indexToLoc(i);
                  var aLoc = loc.slice(-a.rank);
                  aBroadcastDims.forEach(function (d) { return aLoc[d] = 0; });
                  var aIndex = aBuf.locToIndex(aLoc);
                  var bLoc = loc.slice(-b.rank);
                  bBroadcastDims.forEach(function (d) { return bLoc[d] = 0; });
                  var bIndex = bBuf.locToIndex(bLoc);
                  resVals[i] = op(aVals[aIndex], bVals[bIndex]);
              };
              for (var i = 0; i < resVals.length; ++i) {
                  _loop_2(i);
              }
          }
          return result.toTensor();
      };
      MathBackendCPU.prototype.broadcastedBinaryComplexOp = function (a, b, op) {
          var newShape = tf.backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
          var realResult = tf.buffer(newShape, 'float32');
          var imagResult = tf.buffer(newShape, 'float32');
          var aVals = this.readSync(a.dataId);
          var bVals = this.readSync(b.dataId);
          var aBroadcastDims = tf.backend_util.getBroadcastDims(a.shape, newShape);
          var bBroadcastDims = tf.backend_util.getBroadcastDims(b.shape, newShape);
          var realVals = realResult.values;
          var imagVals = imagResult.values;
          if (aBroadcastDims.length + bBroadcastDims.length === 0) {
              for (var i = 0; i < realVals.length; i++) {
                  var aIdx = i % aVals.length;
                  var bIdx = i % bVals.length;
                  var result = op(aVals[aIdx * 2], aVals[aIdx * 2 + 1], bVals[bIdx * 2], bVals[bIdx * 2 + 1]);
                  realVals[i] = result.real;
                  imagVals[i] = result.imag;
              }
          }
          else {
              var aRealBuf = this.bufferSync(this.data.get(a.dataId).complexTensors.real);
              var bRealBuf = this.bufferSync(this.data.get(b.dataId).complexTensors.real);
              var _loop_3 = function (i) {
                  var loc = realResult.indexToLoc(i);
                  var aLoc = loc.slice(-a.rank);
                  aBroadcastDims.forEach(function (d) { return aLoc[d] = 0; });
                  var aIndex = aRealBuf.locToIndex(aLoc);
                  var bLoc = loc.slice(-b.rank);
                  bBroadcastDims.forEach(function (d) { return bLoc[d] = 0; });
                  var bIndex = bRealBuf.locToIndex(bLoc);
                  var opResult = op(aVals[aIndex * 2], aVals[aIndex * 2 + 1], bVals[bIndex * 2], bVals[bIndex * 2 + 1]);
                  realVals[i] = opResult.real;
                  imagVals[i] = opResult.imag;
              };
              for (var i = 0; i < realVals.length; i++) {
                  _loop_3(i);
              }
          }
          return this.complex(realResult.toTensor(), imagResult.toTensor());
      };
      MathBackendCPU.prototype.split = function (x, sizeSplits, axis) {
          return split(x, sizeSplits, axis);
      };
      MathBackendCPU.prototype.dispose = function () { };
      MathBackendCPU.prototype.floatPrecision = function () {
          return 32;
      };
      /** Returns the smallest representable number.  */
      MathBackendCPU.prototype.epsilon = function () {
          return _super.prototype.epsilon.call(this);
      };
      MathBackendCPU.prototype.cropAndResize = function (images, boxes, boxIndex, cropSize, method, extrapolationValue) {
          var _a = images.shape, batch = _a[0], imageHeight = _a[1], imageWidth = _a[2], numChannels = _a[3];
          var numBoxes = boxes.shape[0];
          var cropHeight = cropSize[0], cropWidth = cropSize[1];
          var output = tf.buffer([numBoxes, cropHeight, cropWidth, numChannels], 'float32');
          var boxVals = this.readSync(boxes.dataId);
          var boxIndVals = this.readSync(boxIndex.dataId);
          var imageVals = this.readSync(images.dataId);
          var inStride = images.strides; // to calculate flat indexes into image
          var outStride = output.strides; // to calculate flat indexes into output
          // Reference implementation
          // tslint:disable-next-line:max-line-length
          // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op.cc
          for (var b = 0; b < numBoxes; b++) {
              var startInd = b * 4;
              var y1 = boxVals[startInd];
              var x1 = boxVals[startInd + 1];
              var y2 = boxVals[startInd + 2];
              var x2 = boxVals[startInd + 3];
              var bInd = boxIndVals[b];
              if (bInd >= batch) {
                  continue;
              }
              var heightScale = (cropHeight > 1) ?
                  (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) :
                  0;
              var widthScale = (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;
              for (var y = 0; y < cropHeight; y++) {
                  var yInd = (cropHeight > 1) ?
                      y1 * (imageHeight - 1) + y * (heightScale) :
                      0.5 * (y1 + y2) * (imageHeight - 1);
                  if (yInd < 0 || yInd > imageHeight - 1) {
                      for (var x = 0; x < cropWidth; x++) {
                          for (var c = 0; c < numChannels; c++) {
                              var ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                              output.values[ind] = extrapolationValue;
                          }
                      }
                      continue;
                  }
                  if (method === 'bilinear') {
                      var topInd = Math.floor(yInd);
                      var bottomInd = Math.ceil(yInd);
                      var yLerp = yInd - topInd;
                      for (var x = 0; x < cropWidth; x++) {
                          var xInd = (cropWidth > 1) ?
                              x1 * (imageWidth - 1) + x * widthScale :
                              0.5 * (x1 + x2) * (imageWidth - 1);
                          if (xInd < 0 || xInd > imageWidth - 1) {
                              for (var c = 0; c < numChannels; c++) {
                                  var ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                                  output.values[ind] = extrapolationValue;
                              }
                              continue;
                          }
                          var leftInd = Math.floor(xInd);
                          var rightInd = Math.ceil(xInd);
                          var xLerp = xInd - leftInd;
                          for (var c = 0; c < numChannels; c++) {
                              var ind = c + leftInd * inStride[2] + topInd * inStride[1] +
                                  bInd * inStride[0];
                              var topLeft = imageVals[ind];
                              ind = c + rightInd * inStride[2] + topInd * inStride[1] +
                                  bInd * inStride[0];
                              var topRight = imageVals[ind];
                              ind = c + leftInd * inStride[2] + bottomInd * inStride[1] +
                                  bInd * inStride[0];
                              var bottomLeft = imageVals[ind];
                              ind = c + rightInd * inStride[2] + bottomInd * inStride[1] +
                                  bInd * inStride[0];
                              var bottomRight = imageVals[ind];
                              var top_2 = topLeft + (topRight - topLeft) * xLerp;
                              var bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
                              ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                              output.values[ind] = top_2 + ((bottom - top_2) * yLerp);
                          }
                      }
                  }
                  else { // method == "nearest"
                      for (var x = 0; x < cropWidth; ++x) {
                          var xInd = (cropWidth > 1) ?
                              x1 * (imageWidth - 1) + x * widthScale :
                              0.5 * (x1 + x2) * (imageWidth - 1);
                          if (xInd < 0 || xInd > imageWidth - 1) {
                              for (var c = 0; c < numChannels; c++) {
                                  var ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                                  output.values[ind] = extrapolationValue;
                              }
                              continue;
                          }
                          var closestX = Math.round(xInd);
                          var closestY = Math.round(yInd);
                          for (var c = 0; c < numChannels; c++) {
                              var inInd = c + closestX * inStride[2] +
                                  closestY * inStride[1] + bInd * inStride[0];
                              var outInd = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                              output.values[outInd] = imageVals[inInd];
                          }
                      }
                  }
              }
          }
          return output.toTensor();
      };
      MathBackendCPU.prototype.sparseToDense = function (sparseIndices, sparseValues, outputShape, defaultValue) {
          var _a = tf.backend_util.calculateShapes(sparseValues, sparseIndices, outputShape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, sliceSize = _a.sliceSize, strides = _a.strides, outputSize = _a.outputSize;
          var sumDupeIndices = false;
          return this.scatter(sparseIndices, sparseValues, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices);
      };
      MathBackendCPU.prototype.gatherND = function (x, indices) {
          var indicesShape = indices.shape;
          var sliceRank = indicesShape[indicesShape.length - 1];
          var _a = tf.backend_util.prepareAndValidate(x, indices), resultShape = _a[0], numSlices = _a[1], sliceSize = _a[2], strides = _a[3];
          if (numSlices === 0) {
              return tf.tensor([], resultShape, x.dtype);
          }
          var buffer = new tf.TensorBuffer([numSlices, sliceSize], x.dtype);
          var indicesData = this.readSync(indices.dataId);
          var xData = this.readSync(x.dataId);
          for (var i = 0; i < numSlices; i++) {
              var index = [];
              var flattenIndex = 0;
              for (var j = 0; j < sliceRank; j++) {
                  var dim = indicesData[i * sliceRank + j];
                  flattenIndex += dim * strides[j];
                  index.push(dim);
              }
              if (flattenIndex < 0 || flattenIndex >= x.size / sliceSize) {
                  throw new Error("Invalid indices: " + index + " does not index into " + x.shape);
              }
              for (var k = 0; k < sliceSize; k++) {
                  buffer.values[i * sliceSize + k] = xData[flattenIndex * sliceSize + k];
              }
          }
          return buffer.toTensor().reshape(resultShape);
      };
      MathBackendCPU.prototype.scatterND = function (indices, updates, shape) {
          var _a = tf.backend_util.calculateShapes(updates, indices, shape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, sliceSize = _a.sliceSize, strides = _a.strides, outputSize = _a.outputSize;
          var defaultValue = tf.scalar(0);
          var sumDupeIndices = true;
          return this.scatter(indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices);
      };
      MathBackendCPU.prototype.fill = function (shape, value, dtype) {
          dtype = dtype || tf.util.inferDtype(value);
          var values = tf.util.getArrayFromDType(dtype, tf.util.sizeFromShape(shape));
          values.fill(value);
          return tf.engine().makeTensor(values, shape, dtype, this);
      };
      MathBackendCPU.prototype.onesLike = function (x) {
          if (x.dtype === 'string') {
              throw new Error('onesLike is not supported for string tensors');
          }
          else {
              return this.fill(x.shape, 1, x.dtype);
          }
      };
      MathBackendCPU.prototype.zerosLike = function (x) {
          var values = tf.util.getArrayFromDType(x.dtype, tf.util.sizeFromShape(x.shape));
          return this.makeOutput(values, x.shape, x.dtype);
      };
      MathBackendCPU.prototype.linspace = function (start, stop, num) {
          return tf.backend_util.linspaceImpl(start, stop, num);
      };
      MathBackendCPU.prototype.scatter = function (indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices) {
          var flattenShape = [outputSize / sliceSize, sliceSize];
          var indicesData = this.readSync(indices.dataId);
          var updatesData = this.readSync(updates.dataId);
          if (outputSize === 0) {
              return tf.tensor([], shape, updates.dtype);
          }
          var buffer = new tf.TensorBuffer(flattenShape, updates.dtype);
          buffer.values.fill(this.readSync(defaultValue.dataId)[0]);
          for (var i = 0; i < numUpdates; i++) {
              var index = [];
              var flattenIndex = 0;
              for (var j = 0; j < sliceRank; j++) {
                  var dim = indicesData[i * sliceRank + j];
                  index.push(dim);
                  flattenIndex += dim * strides[j];
              }
              if (flattenIndex < 0 || flattenIndex >= outputSize / sliceSize) {
                  throw new Error("Invalid indices: " + index + " does not index into " + shape);
              }
              for (var k = 0; k < sliceSize; k++) {
                  if (sumDupeIndices) {
                      buffer.values[flattenIndex * sliceSize + k] +=
                          updatesData[i * sliceSize + k];
                  }
                  else {
                      buffer.values[flattenIndex * sliceSize + k] = updates.rank === 0 ?
                          updatesData[0] :
                          updatesData[i * sliceSize + k];
                  }
              }
          }
          return buffer.toTensor().reshape(shape);
      };
      return MathBackendCPU;
  }(tf.KernelBackend));

  /** @license See the LICENSE file. */
  // This code is auto-generated, do not modify this file!
  var version = '0.0.0';

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
  function createBinaryKernelConfig(name, op) {
      return {
          kernelName: name,
          backendName: 'cpu',
          kernelFunc: function (_a) {
              var inputs = _a.inputs, backend = _a.backend;
              var _b = inputs, a = _b.a, b = _b.b;
              var cpuBackend = backend;
              assertNotComplex([a, b], name);
              var aVals = cpuBackend.data.get(a.dataId).values;
              var bVals = cpuBackend.data.get(b.dataId).values;
              var _c = op(a.shape, b.shape, aVals, bVals, a.dtype), resultData = _c[0], resultShape = _c[1];
              var dataId = cpuBackend.write(resultData, resultShape, a.dtype);
              return { dataId: dataId, shape: resultShape, dtype: a.dtype };
          }
      };
  }
  function createBinaryKernelImpl(op) {
      return function (aShape, bShape, aVals, bVals, dtype) {
          var newShape = tf.backend_util.assertAndGetBroadcastShape(aShape, bShape);
          var resultRank = newShape.length;
          var resultStrides = tf.util.computeStrides(newShape);
          var resultSize = tf.util.sizeFromShape(newShape);
          var result = tf.util.getTypedArrayFromDType(dtype, resultSize);
          var aRank = aShape.length;
          var bRank = bShape.length;
          var aStrides = tf.util.computeStrides(aShape);
          var bStrides = tf.util.computeStrides(bShape);
          var aBroadcastDims = tf.backend_util.getBroadcastDims(aShape, newShape);
          var bBroadcastDims = tf.backend_util.getBroadcastDims(bShape, newShape);
          if (aBroadcastDims.length + bBroadcastDims.length === 0) {
              for (var i = 0; i < result.length; ++i) {
                  result[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
              }
          }
          else {
              var _loop_1 = function (i) {
                  var loc = tf.util.indexToLoc(i, resultRank, resultStrides);
                  var aLoc = loc.slice(-aRank);
                  aBroadcastDims.forEach(function (d) { return aLoc[d] = 0; });
                  var aIndex = tf.util.locToIndex(aLoc, aRank, aStrides);
                  var bLoc = loc.slice(-bRank);
                  bBroadcastDims.forEach(function (d) { return bLoc[d] = 0; });
                  var bIndex = tf.util.locToIndex(bLoc, bRank, bStrides);
                  result[i] = op(aVals[aIndex], bVals[bIndex]);
              };
              for (var i = 0; i < result.length; ++i) {
                  _loop_1(i);
              }
          }
          return [result, newShape];
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
  var divImpl = createBinaryKernelImpl(function (a, b) { return a / b; });

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
  var divConfig = createBinaryKernelConfig(tf.Div, divImpl);

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
  var maxConfig = {
      kernelName: tf.Max,
      backendName: 'cpu',
      kernelFunc: function (_a) {
          var inputs = _a.inputs, attrs = _a.attrs, backend = _a.backend;
          var x = inputs.x;
          var reductionIndices = attrs.reductionIndices;
          var cpuBackend = backend;
          var xShape = x.shape;
          var xRank = xShape.length;
          var origAxes = tf.util.parseAxisParam(reductionIndices, xShape);
          var axes = origAxes;
          var permutedAxes = tf.backend_util.getAxesPermutation(axes, xRank);
          var xVals = cpuBackend.data.get(x.dataId).values;
          if (permutedAxes != null) {
              var newShape = new Array(xRank);
              for (var i = 0; i < newShape.length; i++) {
                  newShape[i] = xShape[permutedAxes[i]];
              }
              xVals = transposeImpl(xVals, xShape, x.dtype, permutedAxes, newShape);
              axes = tf.backend_util.getInnerMostAxes(axes.length, xRank);
              xShape = newShape;
          }
          assertNotComplex(x, 'max');
          tf.backend_util.assertAxesAreInnerMostDims('max', axes, xRank);
          var _b = tf.backend_util.computeOutAndReduceShapes(xShape, axes), maxOutShape = _b[0], reduceShape = _b[1];
          var reduceSize = tf.util.sizeFromShape(reduceShape);
          var result = maxImpl(xVals, reduceSize, maxOutShape, x.dtype);
          var dataId = cpuBackend.write(result, maxOutShape, x.dtype);
          return { dataId: dataId, shape: maxOutShape, dtype: x.dtype };
      }
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
  function maxPoolWithArgmaxImpl(xValues, xShape, dtype, includeBatchInIndex, convInfo) {
      var strides = tf.util.computeStrides(xShape);
      var maxPools = pool(xValues, xShape, dtype, strides, convInfo, 'max');
      var maxPositions = maxPoolPositions(xValues, xShape, dtype, convInfo, true, includeBatchInIndex);
      return [maxPools.values, maxPositions.values];
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
  var maxPoolWithArgmaxConfig = {
      kernelName: tf.MaxPoolWithArgmax,
      backendName: 'cpu',
      kernelFunc: function (_a) {
          var inputs = _a.inputs, attrs = _a.attrs, backend = _a.backend;
          var x = inputs.x;
          var _b = attrs, filterSize = _b.filterSize, strides = _b.strides, pad = _b.pad, includeBatchInIndex = _b.includeBatchInIndex;
          var cpuBackend = backend;
          assertNotComplex(x, 'MaxPoolWithArgmax');
          var values = cpuBackend.data.get(x.dataId).values;
          var convInfo = tf.backend_util.computePool2DInfo(x.shape, filterSize, strides, [1, 1], pad);
          var _c = maxPoolWithArgmaxImpl(values, x.shape, x.dtype, includeBatchInIndex, convInfo), pooled = _c[0], indexes = _c[1];
          var pooledDataId = cpuBackend.write(pooled, convInfo.outShape, x.dtype);
          var indexesDataId = cpuBackend.write(indexes, convInfo.outShape, x.dtype);
          return [
              { dataId: pooledDataId, shape: convInfo.outShape, dtype: x.dtype },
              { dataId: indexesDataId, shape: convInfo.outShape, dtype: 'int32' }
          ];
      }
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
  var nonMaxSuppressionV5 = tf.kernel_impls.nonMaxSuppressionV5;
  var nonMaxSuppressionV5Config = {
      kernelName: tf.NonMaxSuppressionV5,
      backendName: 'cpu',
      kernelFunc: function (_a) {
          var inputs = _a.inputs, backend = _a.backend, attrs = _a.attrs;
          var _b = inputs, boxes = _b.boxes, scores = _b.scores;
          var _c = attrs, maxOutputSize = _c.maxOutputSize, iouThreshold = _c.iouThreshold, scoreThreshold = _c.scoreThreshold, softNmsSigma = _c.softNmsSigma;
          var cpuBackend = backend;
          assertNotComplex(boxes, 'NonMaxSuppressionWithScore');
          var boxesVals = cpuBackend.data.get(boxes.dataId).values;
          var scoresVals = cpuBackend.data.get(scores.dataId).values;
          var maxOutputSizeVal = maxOutputSize;
          var iouThresholdVal = iouThreshold;
          var scoreThresholdVal = scoreThreshold;
          var softNmsSigmaVal = softNmsSigma;
          var _d = nonMaxSuppressionV5(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal, softNmsSigmaVal), selectedIndices = _d.selectedIndices, selectedScores = _d.selectedScores;
          return [selectedIndices, selectedScores];
      }
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
  var squareConfig = {
      kernelName: tf.Square,
      backendName: 'cpu',
      kernelFunc: function (_a) {
          var inputs = _a.inputs, backend = _a.backend;
          var x = inputs.x;
          var cpuBackend = backend;
          assertNotComplex(x, 'square');
          var values = cpuBackend.data.get(x.dataId).values;
          var newValues = new Float32Array(values.length);
          for (var i = 0; i < values.length; ++i) {
              var value = values[i];
              newValues[i] = value * value;
          }
          var dataId = cpuBackend.write(newValues, x.shape, x.dtype);
          return { dataId: dataId, shape: x.shape, dtype: x.dtype };
      }
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
  var squaredDifferenceImpl = createBinaryKernelImpl(function (aVal, bVal) {
      var diff = aVal - bVal;
      return diff * diff;
  });
  var squaredDifferenceConfig = createBinaryKernelConfig(tf.SquaredDifference, squaredDifferenceImpl);

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
  var transposeConfig = {
      kernelName: tf.Transpose,
      backendName: 'cpu',
      kernelFunc: function (_a) {
          var inputs = _a.inputs, attrs = _a.attrs, backend = _a.backend;
          var x = inputs.x;
          var perm = attrs.perm;
          var cpuBackend = backend;
          assertNotComplex(x, 'transpose');
          var xRank = x.shape.length;
          var newShape = new Array(xRank);
          for (var i = 0; i < newShape.length; i++) {
              newShape[i] = x.shape[perm[i]];
          }
          var values = cpuBackend.data.get(x.dataId).values;
          var result = transposeImpl(values, x.shape, x.dtype, perm, newShape);
          var dataId = cpuBackend.write(result, newShape, x.dtype);
          return { dataId: dataId, shape: newShape, dtype: x.dtype };
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
  // List all kernel configs here
  var kernelConfigs = [
      nonMaxSuppressionV5Config, squareConfig, squaredDifferenceConfig, divConfig,
      transposeConfig, maxPoolWithArgmaxConfig, maxConfig
  ];
  for (var _i = 0, kernelConfigs_1 = kernelConfigs; _i < kernelConfigs_1.length; _i++) {
      var kernelConfig = kernelConfigs_1[_i];
      tf.registerKernel(kernelConfig);
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
  // Side effects for default initialization of MathBackendCPU
  tf.registerBackend('cpu', function () { return new MathBackendCPU(); }, 1 /* priority */);

  exports.MathBackendCPU = MathBackendCPU;
  exports.shared = shared;
  exports.version_cpu = version;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-backend-cpu.js.map
