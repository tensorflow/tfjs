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
      tensor.forEach(t => {
          if (t != null) {
              tf.util.assert(t.dtype !== 'complex64', () => `${opName} does not support complex64 tensors.`);
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
      const strideHeight = convInfo.strideHeight;
      const strideWidth = convInfo.strideWidth;
      const dilationHeight = convInfo.dilationHeight;
      const dilationWidth = convInfo.dilationWidth;
      const effectiveFilterHeight = convInfo.effectiveFilterHeight;
      const effectiveFilterWidth = convInfo.effectiveFilterWidth;
      const padTop = convInfo.padInfo.top;
      const padLeft = convInfo.padInfo.left;
      const initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
          Number.POSITIVE_INFINITY);
      const output = tf.buffer(convInfo.outShape, dtype);
      const outputVals = output.values;
      const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] * convInfo.outShape[3];
      const outputRowStrides = convInfo.outShape[2] * convInfo.outShape[3];
      const outputColStrides = convInfo.outShape[3];
      for (let b = 0; b < convInfo.batchSize; ++b) {
          const outputBatchOffset = b * outputBatchStrides;
          const inputBatchOffset = b * strides[0];
          for (let d = 0; d < convInfo.inChannels; ++d) {
              for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                  const xRCorner = yR * strideHeight - padTop;
                  const xRMin = Math.max(0, xRCorner);
                  const xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                  const outputRowOffset = outputBatchOffset + yR * outputRowStrides;
                  for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                      const xCCorner = yC * strideWidth - padLeft;
                      const xCMin = Math.max(0, xCCorner);
                      const xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                      let minMaxValue = initialValue;
                      let avgValue = 0;
                      let count = 0;
                      for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
                          const xROffset = inputBatchOffset + xR * strides[1];
                          for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                              const xCOffset = xROffset + xC * strides[2];
                              const pixel = xValues[xCOffset + d];
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
                      const outputOffset = outputRowOffset + yC * outputColStrides + d;
                      outputVals[outputOffset] =
                          poolType === 'avg' ? avgValue / count : minMaxValue;
                  }
              }
          }
      }
      return output;
  }
  function maxPoolPositions(xValues, xShape, dtype, convInfo, flattenPositions = false, includeBatchInIndex = false) {
      const maxPositions = tf.buffer(convInfo.outShape, 'int32');
      const strideHeight = convInfo.strideHeight;
      const strideWidth = convInfo.strideWidth;
      const dilationHeight = convInfo.dilationHeight;
      const dilationWidth = convInfo.dilationWidth;
      const effectiveFilterHeight = convInfo.effectiveFilterHeight;
      const effectiveFilterWidth = convInfo.effectiveFilterWidth;
      const padTop = convInfo.padInfo.top;
      const padLeft = convInfo.padInfo.left;
      const xBuf = tf.buffer(xShape, dtype, xValues);
      for (let b = 0; b < convInfo.batchSize; ++b) {
          for (let d = 0; d < convInfo.inChannels; ++d) {
              for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                  const xRCorner = yR * strideHeight - padTop;
                  let xRMin = xRCorner;
                  while (xRMin < 0) {
                      xRMin += dilationHeight;
                  }
                  // const xRMin = Math.max(0, xRCorner);
                  const xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                  for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                      const xCCorner = yC * strideWidth - padLeft;
                      let xCMin = xCCorner;
                      while (xCMin < 0) {
                          xCMin += dilationWidth;
                      }
                      const xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                      let maxValue = Number.NEGATIVE_INFINITY;
                      let maxPosition = -1;
                      for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
                          const wR = xR - xRCorner;
                          for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                              const wC = xC - xCCorner;
                              const pixel = xBuf.get(b, xR, xC, d);
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
  const nonMaxSuppressionV3 = tf.kernel_impls.nonMaxSuppressionV3;
  const split = tf.kernel_impls.split;
  const tile = tf.kernel_impls.tile;
  const topkImpl = tf.kernel_impls.topkImpl;
  const whereImpl = tf.kernel_impls.whereImpl;
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
      throw new Error(`Activation ${activation} has not been implemented for the CPU backend.`);
  }
  class MathBackendCPU extends tf.KernelBackend {
      constructor() {
          super();
          this.blockSize = 48;
          this.firstUse = true;
          this.data = new tf.DataStorage(this, tf.engine());
      }
      write(values, shape, dtype) {
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
          const dataId = {};
          this.data.set(dataId, { values, dtype });
          return dataId;
      }
      move(dataId, values, shape, dtype) {
          this.data.set(dataId, { values, dtype });
      }
      numDataIds() {
          return this.data.numDataIds();
      }
      async read(dataId) {
          return this.readSync(dataId);
      }
      readSync(dataId) {
          const { dtype, complexTensors } = this.data.get(dataId);
          if (dtype === 'complex64') {
              const realValues = this.readSync(complexTensors.real.dataId);
              const imagValues = this.readSync(complexTensors.imag.dataId);
              return tf.backend_util.mergeRealAndImagArrays(realValues, imagValues);
          }
          return this.data.get(dataId).values;
      }
      bufferSync(t) {
          const data = this.readSync(t.dataId);
          let decodedData = data;
          if (t.dtype === 'string') {
              try {
                  // Decode the bytes into string.
                  decodedData = data.map(d => tf.util.decodeString(d));
              }
              catch (_a) {
                  throw new Error('Failed to decode encoded string bytes into utf-8');
              }
          }
          return tf.buffer(t.shape, t.dtype, decodedData);
      }
      makeOutput(values, shape, dtype) {
          const dataId = this.write(values, shape, dtype);
          return tf.engine().makeTensorFromDataId(dataId, shape, dtype, this);
      }
      disposeData(dataId) {
          if (this.data.has(dataId)) {
              const { complexTensors } = this.data.get(dataId);
              if (complexTensors != null) {
                  complexTensors.real.dispose();
                  complexTensors.imag.dispose();
              }
              this.data.delete(dataId);
          }
      }
      async time(f) {
          const start = tf.util.now();
          f();
          const kernelMs = tf.util.now() - start;
          return { kernelMs };
      }
      memory() {
          return {
              // Unreliable due to automatic gc. The numbers above are cumulative.
              unreliable: true,
              reasons: ['The reported memory is an upper bound. Due to automatic garbage ' +
                      'collection, the true allocated memory may be less.']
          };
      }
      complex(real, imag) {
          const result = this.makeOutput(null, real.shape, 'complex64');
          const resultData = this.data.get(result.dataId);
          // The backend owns the reference to the underlying real and imaginary
          // clones. These will explicitly get disposed when the complex tensor is
          // disposed.
          resultData.complexTensors = {
              real: tf.engine().keep(real.clone()),
              imag: tf.engine().keep(imag.clone())
          };
          return result;
      }
      real(input) {
          const resultData = this.data.get(input.dataId);
          return resultData.complexTensors.real.clone();
      }
      imag(input) {
          const resultData = this.data.get(input.dataId);
          return resultData.complexTensors.imag.clone();
      }
      slice(x, begin, size) {
          assertNotComplex(x, 'slice');
          const isContinous = tf.slice_util.isSliceContinous(x.shape, begin, size);
          if (isContinous) {
              const flatOffset = tf.slice_util.computeFlatOffset(begin, x.strides);
              const length = tf.util.sizeFromShape(size);
              const vals = this.readSync(x.dataId);
              return tf.tensor(vals.subarray(flatOffset, flatOffset + length), size, x.dtype);
          }
          const buffer = tf.buffer(size, x.dtype);
          const xBuf = this.bufferSync(x);
          for (let i = 0; i < buffer.size; ++i) {
              const loc = buffer.indexToLoc(i);
              const xLoc = loc.map((idx, j) => idx + begin[j]);
              buffer.values[i] = xBuf.get(...xLoc);
          }
          return buffer.toTensor();
      }
      stridedSlice(x, begin, end, strides) {
          assertNotComplex(x, 'stridedSlice');
          const outShape = tf.slice_util.computeOutShape(begin, end, strides);
          if (outShape.some(axis => axis === 0)) {
              return tf.tensor([], outShape);
          }
          const buffer = tf.buffer(outShape, x.dtype);
          const xBuf = this.bufferSync(x);
          for (let i = 0; i < buffer.size; i++) {
              const loc = buffer.indexToLoc(i);
              const newLoc = new Array(loc.length);
              for (let j = 0; j < newLoc.length; j++) {
                  newLoc[j] = loc[j] * strides[j] + begin[j];
              }
              buffer.set(xBuf.get(...newLoc), ...loc);
          }
          return buffer.toTensor();
      }
      diag(x) {
          const xVals = this.readSync(x.dataId);
          const buffer = tf.buffer([x.size, x.size], x.dtype);
          const vals = buffer.values;
          for (let i = 0; i < xVals.length; i++) {
              vals[i * x.size + i] = xVals[i];
          }
          return buffer.toTensor();
      }
      unstack(x, axis) {
          const num = x.shape[axis];
          const outShape = new Array(x.rank - 1);
          let outIndex = 0;
          for (let i = 0; i < x.rank; i++) {
              if (i !== axis) {
                  outShape[outIndex++] = x.shape[i];
              }
          }
          const begin = new Array(x.rank).fill(0);
          const size = x.shape.slice();
          size[axis] = 1;
          const res = new Array(num);
          for (let i = 0; i < res.length; i++) {
              begin[axis] = i;
              res[i] = this.slice(x, begin, size).reshape(outShape);
          }
          return res;
      }
      reverse(x, axis) {
          assertNotComplex(x, 'reverse');
          const buffer = tf.buffer(x.shape, x.dtype);
          const xBuf = this.bufferSync(x);
          for (let i = 0; i < buffer.size; i++) {
              const outLoc = buffer.indexToLoc(i);
              const inLoc = outLoc.slice();
              axis.forEach(ax => inLoc[ax] = x.shape[ax] - 1 - inLoc[ax]);
              buffer.set(xBuf.get(...inLoc), ...outLoc);
          }
          return buffer.toTensor();
      }
      concat(tensors, axis) {
          if (tensors[0].dtype === 'complex64') {
              const reals = tensors.map((t) => tf.real(t));
              const imags = tensors.map((t) => tf.imag(t));
              return tf.complex(this.concat(reals, axis), this.concat(imags, axis));
          }
          const tensors2D = tensors.map(t => {
              const innerSize = tf.util.sizeFromShape(t.shape.slice(axis));
              return t.as2D(-1, innerSize);
          });
          const outShape = tf.backend_util.computeOutShape(tensors2D.map(t => t.shape), 1 /* axis
            */);
          const values = tf.buffer(outShape, tensors[0].dtype)
              .values;
          if (tensors2D[0].shape[0] === 1) {
              // Use built-in TypedArray.set() method for speed.
              let offset = 0;
              tensors2D.forEach(t => {
                  values.set(this.readSync(t.dataId), offset);
                  offset += t.size;
              });
          }
          else {
              let colOffset = 0;
              tensors2D.forEach(t => {
                  const tVals = this.readSync(t.dataId);
                  let tIdx = 0;
                  for (let row = 0; row < t.shape[0]; ++row) {
                      const resIdx = row * outShape[1] + colOffset;
                      for (let col = 0; col < t.shape[1]; ++col) {
                          values[resIdx + col] = tVals[tIdx++];
                      }
                  }
                  colOffset += t.shape[1];
              });
          }
          const finalOutShape = tf.backend_util.computeOutShape(tensors.map(t => t.shape), axis);
          return tf.tensor(values, finalOutShape, tensors[0].dtype);
      }
      neg(x) {
          assertNotComplex(x, 'neg');
          return this.multiply(tf.scalar(-1), x);
      }
      add(a, b) {
          if (a.dtype === 'complex64' || b.dtype === 'complex64') {
              return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), (aReal, aImag, bReal, bImag) => {
                  return { real: aReal + bReal, imag: aImag + bImag };
              });
          }
          return this.broadcastedBinaryOp(a, b, tf.upcastType(a.dtype, b.dtype), (aValue, bValue) => aValue + bValue);
      }
      addN(tensors) {
          assertNotComplex(tensors, 'addN');
          const vals = tensors.map(t => this.readSync(t.dataId));
          const result = tf.buffer(tensors[0].shape, tensors[0].dtype);
          const resultVals = result.values;
          for (let i = 0; i < tensors.length; i++) {
              const currVals = vals[i];
              for (let j = 0; j < resultVals.length; j++) {
                  resultVals[j] += currVals[j];
              }
          }
          return result.toTensor();
      }
      softmax(logits, dim) {
          const axes = tf.util.parseAxisParam([dim], logits.shape);
          const maxLogit = this.max(logits, axes);
          const expandedShape = tf.backend_util.expandShapeToKeepDim(maxLogit.shape, axes);
          const a = this.subtract(logits, maxLogit.reshape(expandedShape));
          const b = this.exp(a);
          const sumExp = this.sum(b, axes).reshape(expandedShape);
          // TODO(annxingyuan): Call divImpl rather than op as part of softmax
          // kernel modularization.
          return tf.div(b, sumExp);
      }
      subtract(a, b) {
          if (a.dtype === 'complex64' || b.dtype === 'complex64') {
              return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), (aReal, aImag, bReal, bImag) => {
                  return { real: aReal - bReal, imag: aImag - bImag };
              });
          }
          return this.broadcastedBinaryOp(a, b, tf.upcastType(a.dtype, b.dtype), (aValue, bValue) => aValue - bValue);
      }
      pow(a, b) {
          assertNotComplex([a, b], 'pow');
          return this.broadcastedBinaryOp(a, b, a.dtype, (aValue, bValue) => Math.pow(aValue, bValue));
      }
      batchMatMul(a, b, transposeA, transposeB) {
          assertNotComplex([a, b], 'matMul');
          const sharedDim = transposeA ? a.shape[1] : a.shape[2];
          const leftDim = transposeA ? a.shape[2] : a.shape[1];
          const rightDim = transposeB ? b.shape[1] : b.shape[2];
          const batchDim = a.shape[0];
          const aValues = this.readSync(a.dataId);
          const bValues = this.readSync(b.dataId);
          const [aBatch, aOuterStep, aInnerStep] = transposeA ?
              [a.strides[0], 1, a.strides[1]] :
              [a.strides[0], a.strides[1], 1];
          const [bInnerStep, bOuterStep, bBatch] = transposeB ?
              [1, b.strides[1], b.strides[0]] :
              [b.strides[1], 1, b.strides[0]];
          const size = leftDim * rightDim;
          const result = tf.buffer([batchDim, leftDim, rightDim], a.dtype);
          const resVals = result.values;
          const blockSize = this.blockSize;
          for (let b = 0; b < batchDim; b++) {
              for (let i0 = 0; i0 < leftDim; i0 += blockSize) {
                  for (let j0 = 0; j0 < rightDim; j0 += blockSize) {
                      for (let k0 = 0; k0 < sharedDim; k0 += blockSize) {
                          // for when blockSize doesn't evenly divide the input
                          const iBlock = Math.min(i0 + blockSize, leftDim);
                          const jBlock = Math.min(j0 + blockSize, rightDim);
                          const kBlock = Math.min(k0 + blockSize, sharedDim);
                          for (let i = i0; i < iBlock; i++) {
                              for (let j = j0; j < jBlock; j++) {
                                  let sum = 0.0;
                                  for (let k = k0; k < kBlock; k++) {
                                      sum += aValues[b * aBatch + i * aOuterStep + k * aInnerStep] *
                                          bValues[k * bInnerStep + j * bOuterStep + b * bBatch];
                                  }
                                  resVals[b * size + (i * rightDim + j)] += sum;
                              }
                          }
                      }
                  }
              }
          }
          return result.toTensor();
      }
      fusedBatchMatMul({ a, b, transposeA, transposeB, bias, activation, preluActivationWeights }) {
          let result = this.batchMatMul(a, b, transposeA, transposeB);
          if (bias) {
              result = this.add(result, bias);
          }
          if (activation) {
              result =
                  mapActivation(this, result, activation, preluActivationWeights);
          }
          return result;
      }
      multiply(a, b) {
          if (a.dtype === 'complex64' || b.dtype === 'complex64') {
              return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), (aReal, aImag, bReal, bImag) => {
                  return {
                      real: aReal * bReal - aImag * bImag,
                      imag: aReal * bImag + aImag * bReal
                  };
              });
          }
          return this.broadcastedBinaryOp(a, b, tf.upcastType(a.dtype, b.dtype), (aValue, bValue) => aValue * bValue);
      }
      floorDiv(a, b) {
          assertNotComplex([a, b], 'floorDiv');
          const op = (a, b) => Math.floor(a / b);
          const outputDtype = 'int32';
          return this.broadcastedBinaryOp(a, b, outputDtype, op);
      }
      sum(x, axes) {
          assertNotComplex(x, 'sum');
          tf.backend_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const resultDtype = tf.upcastType(x.dtype, 'int32');
          const result = tf.zeros(outShape, resultDtype);
          const reduceSize = tf.util.sizeFromShape(reduceShape);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          for (let i = 0; i < vals.length; ++i) {
              const offset = i * reduceSize;
              let sum = 0;
              for (let j = 0; j < reduceSize; ++j) {
                  sum += aVals[offset + j];
              }
              vals[i] = sum;
          }
          return result;
      }
      prod(x, axes) {
          assertNotComplex(x, 'sum');
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const resultDtype = tf.upcastType(x.dtype, 'int32');
          const result = tf.zeros(outShape, resultDtype);
          const reduceSize = tf.util.sizeFromShape(reduceShape);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          for (let i = 0; i < vals.length; ++i) {
              const offset = i * reduceSize;
              let prod = 1;
              for (let j = 0; j < reduceSize; ++j) {
                  prod *= aVals[offset + j];
              }
              vals[i] = prod;
          }
          return result;
      }
      unsortedSegmentSum(x, segmentIds, numSegments) {
          assertNotComplex(x, 'unsortedSegmentSum');
          const res = [];
          // Reshape the segment id's so that they can be broadcast with
          // x. The new shape should be [segmentIds.shape, 1, ..., 1]
          const numIters = x.rank - segmentIds.rank;
          for (let i = 0; i < numIters; ++i) {
              segmentIds = segmentIds.expandDims(i + 1);
          }
          for (let i = 0; i < numSegments; ++i) {
              const segmentId = tf.scalar(i, 'int32');
              const mask = tf.equal(segmentId, segmentIds).asType('float32');
              const sum = mask.mul(x).sum(0);
              res.push(sum);
          }
          return tf.stack(res);
      }
      argMin(x, axis) {
          assertNotComplex(x, 'argMin');
          const axes = [axis];
          tf.backend_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const result = tf.zeros(outShape, 'int32');
          const reduceSize = tf.util.sizeFromShape(reduceShape);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          for (let i = 0; i < vals.length; ++i) {
              const offset = i * reduceSize;
              let min = aVals[offset];
              let minIndex = 0;
              for (let j = 0; j < reduceSize; ++j) {
                  const value = aVals[offset + j];
                  if (value < min) {
                      min = value;
                      minIndex = j;
                  }
              }
              vals[i] = minIndex;
          }
          return result;
      }
      argMax(x, axis) {
          assertNotComplex(x, 'argMax');
          const axes = [axis];
          tf.backend_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const result = tf.zeros(outShape, 'int32');
          const reduceSize = tf.util.sizeFromShape(reduceShape);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          for (let i = 0; i < vals.length; ++i) {
              const offset = i * reduceSize;
              let max = aVals[offset];
              let maxIndex = 0;
              for (let j = 0; j < reduceSize; ++j) {
                  const value = aVals[offset + j];
                  if (value > max) {
                      max = value;
                      maxIndex = j;
                  }
              }
              vals[i] = maxIndex;
          }
          return result;
      }
      cumsum(x, axis, exclusive, reverse) {
          assertNotComplex(x, 'cumsum');
          if (axis !== x.rank - 1) {
              throw new Error(`backend.cumsum in CPU expects an inner-most axis=${x.rank - 1} ` +
                  `but got axis=${axis}`);
          }
          const resultDtype = tf.upcastType(x.dtype, 'int32');
          const result = tf.zeros(x.shape, resultDtype);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          const finalDim = x.shape[x.rank - 1];
          const indexAdjuster = reverse ?
              (i, j) => i + finalDim - j - 1 :
              (i, j) => i + j;
          for (let i = 0; i < aVals.length; i += finalDim) {
              for (let j = 0; j < finalDim; j++) {
                  const idx = indexAdjuster(i, j);
                  if (j === 0) {
                      vals[idx] = exclusive ? 0 : aVals[idx];
                  }
                  else {
                      const prevIdx = indexAdjuster(i, j - 1);
                      vals[idx] = exclusive ? aVals[prevIdx] + vals[prevIdx] :
                          aVals[idx] + vals[prevIdx];
                  }
              }
          }
          return result;
      }
      equal(a, b) {
          assertNotComplex([a, b], 'equal');
          return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
              return (aVal === bVal) ? 1 : 0;
          });
      }
      notEqual(a, b) {
          assertNotComplex([a, b], 'notEqual');
          return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
              return (aVal !== bVal) ? 1 : 0;
          });
      }
      less(a, b) {
          assertNotComplex([a, b], 'less');
          return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
              return (aVal < bVal) ? 1 : 0;
          });
      }
      lessEqual(a, b) {
          assertNotComplex([a, b], 'lessEqual');
          return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
              return (aVal <= bVal) ? 1 : 0;
          });
      }
      greater(a, b) {
          assertNotComplex([a, b], 'greater');
          return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
              return (aVal > bVal) ? 1 : 0;
          });
      }
      greaterEqual(a, b) {
          assertNotComplex([a, b], 'greaterEqual');
          return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
              return (aVal >= bVal) ? 1 : 0;
          });
      }
      logicalNot(x) {
          assertNotComplex(x, 'logicalNot');
          const values = this.readSync(x.dataId);
          const newValues = new Uint8Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              newValues[i] = values[i] ? 0 : 1;
          }
          return this.makeOutput(newValues, x.shape, 'bool');
      }
      logicalAnd(a, b) {
          assertNotComplex([a, b], 'logicalAnd');
          return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
              return aVal && bVal;
          });
      }
      logicalOr(a, b) {
          assertNotComplex([a, b], 'logicalOr');
          return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
              return aVal || bVal;
          });
      }
      select(condition, a, b) {
          assertNotComplex([condition, a, b], 'select');
          const values = this.readSync(condition.dataId);
          const aValues = this.readSync(a.dataId);
          const bValues = this.readSync(b.dataId);
          const result = tf.zeros(a.shape, tf.upcastType(a.dtype, b.dtype));
          const newValues = this.readSync(result.dataId);
          let index = 0;
          const offset = condition.rank === 0 || condition.rank > 1 || a.rank === 1 ?
              1 :
              tf.util.sizeFromShape(a.shape.slice(1));
          for (let i = 0; i < values.length; i++) {
              for (let j = 0; j < offset; j++) {
                  if (values[i] === 1) {
                      newValues[index++] = aValues[i];
                  }
                  else {
                      newValues[index++] = bValues[i];
                  }
              }
          }
          return result;
      }
      where(condition) {
          assertNotComplex([condition], 'where');
          const condVals = this.readSync(condition.dataId);
          return whereImpl(condition.shape, condVals);
      }
      topk(x, k, sorted) {
          assertNotComplex(x, 'topk');
          const xVals = this.readSync(x.dataId);
          return topkImpl(xVals, x.shape, x.dtype, k, sorted);
      }
      min(x, axes) {
          assertNotComplex(x, 'min');
          tf.backend_util.assertAxesAreInnerMostDims('min', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const result = tf.zeros(outShape, x.dtype);
          const reduceSize = tf.util.sizeFromShape(reduceShape);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          for (let i = 0; i < vals.length; ++i) {
              const offset = i * reduceSize;
              let min = aVals[offset];
              for (let j = 0; j < reduceSize; ++j) {
                  const value = aVals[offset + j];
                  if (value < min) {
                      min = value;
                  }
              }
              vals[i] = min;
          }
          return result;
      }
      minimum(a, b) {
          assertNotComplex([a, b], 'minimum');
          return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => Math.min(aVal, bVal));
      }
      mod(a, b) {
          assertNotComplex([a, b], 'mod');
          return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => {
              const rem = aVal % bVal;
              if ((aVal < 0 && bVal < 0) || (aVal >= 0 && bVal >= 0)) {
                  return rem;
              }
              else {
                  return (rem + bVal) % bVal;
              }
          });
      }
      max(x, axes) {
          assertNotComplex(x, 'max');
          tf.backend_util.assertAxesAreInnerMostDims('max', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const result = tf.zeros(outShape, x.dtype);
          const reduceSize = tf.util.sizeFromShape(reduceShape);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          for (let i = 0; i < vals.length; ++i) {
              const offset = i * reduceSize;
              let max = aVals[offset];
              for (let j = 0; j < reduceSize; ++j) {
                  const value = aVals[offset + j];
                  if (value > max) {
                      max = value;
                  }
              }
              vals[i] = max;
          }
          return result;
      }
      maximum(a, b) {
          assertNotComplex([a, b], 'maximum');
          return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => Math.max(aVal, bVal));
      }
      all(x, axes) {
          assertNotComplex(x, 'all');
          tf.backend_util.assertAxesAreInnerMostDims('all', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const result = tf.zeros(outShape, x.dtype);
          const reduceSize = tf.util.sizeFromShape(reduceShape);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          for (let i = 0; i < vals.length; ++i) {
              const offset = i * reduceSize;
              let all = aVals[offset];
              for (let j = 0; j < reduceSize; ++j) {
                  const value = aVals[offset + j];
                  all = all && value;
              }
              vals[i] = all;
          }
          return result;
      }
      any(x, axes) {
          assertNotComplex(x, 'any');
          tf.backend_util.assertAxesAreInnerMostDims('any', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const result = tf.zeros(outShape, x.dtype);
          const reduceSize = tf.util.sizeFromShape(reduceShape);
          const vals = this.readSync(result.dataId);
          const aVals = this.readSync(x.dataId);
          for (let i = 0; i < vals.length; ++i) {
              const offset = i * reduceSize;
              let anyVal = aVals[offset];
              for (let j = 0; j < reduceSize; ++j) {
                  const value = aVals[offset + j];
                  anyVal = anyVal || value;
              }
              vals[i] = anyVal;
          }
          return result;
      }
      squaredDifference(a, b) {
          assertNotComplex([a, b], 'squaredDifference');
          return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => {
              const diff = aVal - bVal;
              return diff * diff;
          });
      }
      ceil(x) {
          assertNotComplex(x, 'ceil');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              newValues[i] = Math.ceil(values[i]);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      floor(x) {
          assertNotComplex(x, 'floor');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              newValues[i] = Math.floor(values[i]);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      sign(x) {
          assertNotComplex(x, 'x');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
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
      }
      isNaN(x) {
          assertNotComplex(x, 'x');
          const values = this.readSync(x.dataId);
          const newValues = new Uint8Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              if (Number.isNaN(values[i])) {
                  newValues[i] = 1;
              }
          }
          return this.makeOutput(newValues, x.shape, 'bool');
      }
      isInf(x) {
          assertNotComplex(x, 'x');
          const values = this.readSync(x.dataId);
          const newValues = new Uint8Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              if (Math.abs(values[i]) === Infinity) {
                  newValues[i] = 1;
              }
          }
          return this.makeOutput(newValues, x.shape, 'bool');
      }
      isFinite(x) {
          assertNotComplex(x, 'x');
          const values = this.readSync(x.dataId);
          const newValues = new Uint8Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              if (Number.isFinite(values[i])) {
                  newValues[i] = 1;
              }
          }
          return this.makeOutput(newValues, x.shape, 'bool');
      }
      round(x) {
          assertNotComplex(x, 'round');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              // The algorithm is based on banker's rounding.
              const base = Math.floor(values[i]);
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
      }
      exp(x) {
          assertNotComplex(x, 'exp');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              newValues[i] = Math.exp(values[i]);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      expm1(x) {
          assertNotComplex(x, 'expm1');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              newValues[i] = Math.expm1(values[i]);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      log(x) {
          assertNotComplex(x, 'log');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              const value = values[i];
              newValues[i] = Math.log(value);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      log1p(x) {
          assertNotComplex(x, 'log1p');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              const value = values[i];
              newValues[i] = Math.log1p(value);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      sqrt(x) {
          assertNotComplex(x, 'sqrt');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              const value = values[i];
              newValues[i] = Math.sqrt(value);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      rsqrt(x) {
          assertNotComplex(x, 'rsqrt');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              const value = values[i];
              newValues[i] = 1 / Math.sqrt(value);
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      reciprocal(x) {
          assertNotComplex(x, 'reciprocal');
          const values = this.readSync(x.dataId);
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              newValues[i] = 1 / values[i];
          }
          return this.makeOutput(newValues, x.shape, 'float32');
      }
      linear(x) {
          return x;
      }
      relu(x) {
          assertNotComplex(x, 'relu');
          const res = tf.zeros(x.shape, x.dtype);
          const resVals = this.readSync(res.dataId);
          const inVals = this.readSync(x.dataId);
          for (let i = 0; i < inVals.length; ++i) {
              resVals[i] = Math.max(0, inVals[i]);
          }
          return res;
      }
      relu6(x) {
          assertNotComplex(x, 'relu');
          const res = tf.zeros(x.shape, x.dtype);
          const resVals = this.readSync(res.dataId);
          const inVals = this.readSync(x.dataId);
          for (let i = 0; i < inVals.length; ++i) {
              resVals[i] = Math.min(Math.max(0, inVals[i]), 6);
          }
          return res;
      }
      prelu(x, a) {
          assertNotComplex([x, a], 'prelu');
          return this.broadcastedBinaryOp(x, a, x.dtype, (xValue, aValue) => xValue < 0 ? aValue * xValue : xValue);
      }
      elu(x) {
          assertNotComplex(x, 'elu');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              const v = values[i];
              if (v >= 0) {
                  resultValues[i] = v;
              }
              else {
                  resultValues[i] = (Math.exp(v) - 1);
              }
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      eluDer(dy, y) {
          assertNotComplex([dy, y], 'eluDer');
          const resultValues = new Float32Array(y.size);
          const values = this.readSync(y.dataId);
          const dyValues = this.readSync(dy.dataId);
          for (let i = 0; i < values.length; ++i) {
              const v = values[i];
              if (v >= 1) {
                  resultValues[i] = dyValues[i];
              }
              else {
                  resultValues[i] = dyValues[i] * (v + 1);
              }
          }
          return this.makeOutput(resultValues, y.shape, 'float32');
      }
      selu(x) {
          assertNotComplex(x, 'selu');
          // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
          // see: https://arxiv.org/abs/1706.02515
          const scaleAlpha = tf.backend_util.SELU_SCALEALPHA;
          const scale = tf.backend_util.SELU_SCALE;
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              const v = values[i];
              if (v >= 0) {
                  resultValues[i] = scale * v;
              }
              else {
                  resultValues[i] = scaleAlpha * (Math.exp(v) - 1);
              }
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      clip(x, min, max) {
          assertNotComplex(x, 'clip');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              const v = values[i];
              resultValues[i] = v > max ? max : (v < min ? min : v);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      abs(x) {
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.abs(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      complexAbs(x) {
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < x.size; ++i) {
              const real = values[i * 2];
              const imag = values[i * 2 + 1];
              resultValues[i] = Math.hypot(real, imag);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      int(x) {
          assertNotComplex(x, 'int');
          const resultValues = new Int32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = values[i];
          }
          return this.makeOutput(resultValues, x.shape, 'int32');
      }
      sigmoid(x) {
          assertNotComplex(x, 'sigmoid');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = 1 / (1 + Math.exp(-values[i]));
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      softplus(x) {
          assertNotComplex(x, 'softplus');
          // mirrors the implementation of tf.nn.softplus: https://goo.gl/vkcvwX
          // epsilon is the difference between 1.0 and the next representable float.
          // For a single precision 32 bit float this should be 2^-23, see:
          // https://math.byu.edu/~schow/work/IEEEFloatingPoint.htm
          const epsilon = 1.1920928955078125e-7;
          const threshold = Math.log(epsilon) + 2.0;
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              // Value above which exp(x) may overflow, but softplus(x) == x
              // is within machine epsilon.
              const tooLarge = values[i] > -threshold;
              // Value below which exp(x) may underflow, but softplus(x) == exp(x)
              // is within machine epsilon.
              const tooSmall = values[i] < threshold;
              const expX = Math.exp(values[i]);
              let result;
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
      }
      sin(x) {
          assertNotComplex(x, 'sin');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.sin(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      cos(x) {
          assertNotComplex(x, 'cos');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.cos(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      tan(x) {
          assertNotComplex(x, 'tan');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.tan(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      asin(x) {
          assertNotComplex(x, 'asin');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.asin(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      acos(x) {
          assertNotComplex(x, 'acos');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.acos(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      atan(x) {
          assertNotComplex(x, 'atan');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.atan(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      atan2(a, b) {
          assertNotComplex([a, b], 'atan2');
          return this.broadcastedBinaryOp(a, b, a.dtype, (aValue, bValue) => Math.atan2(aValue, bValue));
      }
      sinh(x) {
          assertNotComplex(x, 'sinh');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.sinh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      cosh(x) {
          assertNotComplex(x, 'cosh');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.cosh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      tanh(x) {
          assertNotComplex(x, 'tanh');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = tf.util.tanh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      asinh(x) {
          assertNotComplex(x, 'asinh');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.asinh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      acosh(x) {
          assertNotComplex(x, 'acosh');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.acosh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      atanh(x) {
          assertNotComplex(x, 'atanh');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              resultValues[i] = Math.atanh(values[i]);
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      erf(x) {
          assertNotComplex(x, 'erf');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          const p = tf.backend_util.ERF_P;
          const a1 = tf.backend_util.ERF_A1;
          const a2 = tf.backend_util.ERF_A2;
          const a3 = tf.backend_util.ERF_A3;
          const a4 = tf.backend_util.ERF_A4;
          const a5 = tf.backend_util.ERF_A5;
          for (let i = 0; i < values.length; ++i) {
              const sign = Math.sign(values[i]);
              const v = Math.abs(values[i]);
              const t = 1.0 / (1.0 + p * v);
              resultValues[i] = sign *
                  (1.0 -
                      (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                          Math.exp(-v * v));
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      step(x, alpha = 0) {
          assertNotComplex(x, 'step');
          const resultValues = new Float32Array(x.size);
          const values = this.readSync(x.dataId);
          for (let i = 0; i < values.length; ++i) {
              const value = values[i];
              if (isNaN(value)) {
                  resultValues[i] = NaN;
              }
              else {
                  resultValues[i] = value > 0 ? 1 : alpha;
              }
          }
          return this.makeOutput(resultValues, x.shape, 'float32');
      }
      fusedConv2d({ input, filter, convInfo, bias, activation, preluActivationWeights }) {
          let result = this.conv2d(input, filter, convInfo);
          if (bias) {
              result = this.add(result, bias);
          }
          if (activation) {
              result =
                  mapActivation(this, result, activation, preluActivationWeights);
          }
          return result;
      }
      conv2d(x, filter, convInfo) {
          assertNotComplex([x, filter], 'conv2d');
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const padLeft = convInfo.padInfo.left;
          const padTop = convInfo.padInfo.top;
          const isChannelsLast = convInfo.dataFormat === 'channelsLast';
          const y = tf.buffer(convInfo.outShape, x.dtype);
          const xBatchStride = x.strides[0];
          const xRowStride = isChannelsLast ? x.strides[1] : x.strides[2];
          const xColStride = isChannelsLast ? x.strides[2] : 1;
          const xChannelStride = isChannelsLast ? 1 : x.strides[1];
          const yBatchStride = y.strides[0];
          const yRowStride = isChannelsLast ? y.strides[1] : y.strides[2];
          const yColStride = isChannelsLast ? y.strides[2] : 1;
          const yChannelStride = isChannelsLast ? 1 : y.strides[1];
          const xVals = this.readSync(x.dataId);
          const wVals = this.readSync(filter.dataId);
          const yVals = y.values;
          for (let b = 0; b < convInfo.batchSize; ++b) {
              const xOffset1 = b * xBatchStride;
              const yOffset1 = b * yBatchStride;
              for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                  const yOffset2 = yOffset1 + yR * yRowStride;
                  const xRCorner = yR * convInfo.strideHeight - padTop;
                  for (let wR = 0; wR < filterHeight; wR++) {
                      const xR = xRCorner + wR * dilationHeight;
                      if (xR < 0 || xR >= convInfo.inHeight) {
                          continue;
                      }
                      const wOffset1 = wR * filter.strides[0];
                      const xOffset2 = xOffset1 + xR * xRowStride;
                      for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                          const yOffset3 = yOffset2 + yC * yColStride;
                          const xCCorner = yC * convInfo.strideWidth - padLeft;
                          for (let wC = 0; wC < filterWidth; wC++) {
                              const xC = xCCorner + wC * dilationWidth;
                              if (xC < 0 || xC >= convInfo.inWidth) {
                                  continue;
                              }
                              const wOffset2 = wOffset1 + wC * filter.strides[1];
                              const xOffset3 = xOffset2 + xC * xColStride;
                              let wOffset3 = wOffset2;
                              for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                  const xVal = xVals[xOffset3 + d1 * xChannelStride];
                                  for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
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
      }
      conv3d(x, filter, convInfo) {
          const filterDepth = convInfo.filterDepth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const padFront = convInfo.padInfo.front;
          const padLeft = convInfo.padInfo.left;
          const padTop = convInfo.padInfo.top;
          const y = tf.buffer(convInfo.outShape, x.dtype);
          const xVals = this.readSync(x.dataId);
          const wVals = this.readSync(filter.dataId);
          const yVals = y.values;
          for (let b = 0; b < convInfo.batchSize; ++b) {
              const xOffset1 = b * x.strides[0];
              const yOffset1 = b * y.strides[0];
              for (let yF = 0; yF < convInfo.outDepth; ++yF) {
                  const yOffset2 = yOffset1 + yF * y.strides[1];
                  const xFCorner = yF * convInfo.strideDepth - padFront;
                  for (let wF = 0; wF < filterDepth; wF++) {
                      const xF = xFCorner + wF * dilationDepth;
                      if (xF < 0 || xF >= convInfo.inDepth) {
                          continue;
                      }
                      const wOffset1 = wF * filter.strides[0];
                      const xOffset2 = xOffset1 + xF * x.strides[1];
                      for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                          const yOffset3 = yOffset2 + yR * y.strides[2];
                          const xRCorner = yR * convInfo.strideHeight - padTop;
                          for (let wR = 0; wR < filterHeight; wR++) {
                              const xR = xRCorner + wR * dilationHeight;
                              if (xR < 0 || xR >= convInfo.inHeight) {
                                  continue;
                              }
                              const wOffset2 = wOffset1 + wR * filter.strides[1];
                              const xOffset3 = xOffset2 + xR * x.strides[2];
                              for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                                  const yOffset4 = yOffset3 + yC * convInfo.outChannels;
                                  const xCCorner = yC * convInfo.strideWidth - padLeft;
                                  for (let wC = 0; wC < filterWidth; wC++) {
                                      const xC = xCCorner + wC * dilationWidth;
                                      if (xC < 0 || xC >= convInfo.inWidth) {
                                          continue;
                                      }
                                      const wOffset3 = wOffset2 + wC * filter.strides[2];
                                      const xOffset4 = xOffset3 + xC * convInfo.inChannels;
                                      let wOffset4 = wOffset3;
                                      for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                          const xVal = xVals[xOffset4 + d1];
                                          for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
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
      }
      conv2dDerInput(dy, filter, convInfo) {
          assertNotComplex([dy, filter], 'conv2dDerInput');
          const dx = tf.buffer(convInfo.inShape, 'float32');
          const dxValues = dx.values;
          const dyValues = this.readSync(dy.dataId);
          const fltValues = this.readSync(filter.dataId);
          const [fltS0, fltS1, fltS2] = filter.strides;
          const { batchSize, filterHeight, filterWidth, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, strideHeight, strideWidth, dataFormat } = convInfo;
          const topPad = filterHeight - 1 - convInfo.padInfo.top;
          const leftPad = filterWidth - 1 - convInfo.padInfo.left;
          const isChannelsLast = dataFormat === 'channelsLast';
          const xBatchStride = dx.strides[0];
          const xRowStride = isChannelsLast ? dx.strides[1] : dx.strides[2];
          const xColStride = isChannelsLast ? dx.strides[2] : 1;
          const xChannelStride = isChannelsLast ? 1 : dx.strides[1];
          const yBatchStride = dy.strides[0];
          const yRowStride = isChannelsLast ? dy.strides[1] : dy.strides[2];
          const yColStride = isChannelsLast ? dy.strides[2] : 1;
          const yChannelStride = isChannelsLast ? 1 : dy.strides[1];
          for (let b = 0; b < batchSize; ++b) {
              for (let d1 = 0; d1 < inChannels; ++d1) {
                  for (let xR = 0; xR < inHeight; ++xR) {
                      const xRCorner = xR - topPad;
                      const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                      const yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                      for (let xC = 0; xC < inWidth; ++xC) {
                          const xCCorner = xC - leftPad;
                          const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                          const yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                          let dotProd = 0;
                          for (let yR = xRMin; yR < yRMax; ++yR) {
                              const wR = yR * strideHeight - xRCorner;
                              for (let yC = xCMin; yC < yCMax; ++yC) {
                                  const wC = yC * strideWidth - xCCorner;
                                  const dyOffset = yBatchStride * b + yRowStride * yR + yColStride * yC;
                                  const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                      fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                                  for (let d2 = 0; d2 < outChannels; ++d2) {
                                      const pixel = dyValues[dyOffset + yChannelStride * d2];
                                      const weight = fltValues[fltOffset + d2];
                                      dotProd += pixel * weight;
                                  }
                              }
                          }
                          const dxOffset = xBatchStride * b + xRowStride * xR +
                              xColStride * xC + xChannelStride * d1;
                          dxValues[dxOffset] = dotProd;
                      }
                  }
              }
          }
          return dx.toTensor();
      }
      conv3dDerInput(dy, filter, convInfo) {
          const dx = tf.buffer(convInfo.inShape, 'float32');
          const dxValues = dx.values;
          const [dxS0, dxS1, dxS2, dxS3] = dx.strides;
          const dyValues = this.readSync(dy.dataId);
          const [dyS0, dyS1, dyS2, dyS3] = dy.strides;
          const fltValues = this.readSync(filter.dataId);
          const [fltS0, fltS1, fltS2, fltS3] = filter.strides;
          const { batchSize, filterDepth, filterHeight, filterWidth, inChannels, inDepth, inHeight, inWidth, outChannels, outDepth, outHeight, outWidth, strideDepth, strideHeight, strideWidth } = convInfo;
          const frontPad = filterDepth - 1 - convInfo.padInfo.front;
          const topPad = filterHeight - 1 - convInfo.padInfo.top;
          const leftPad = filterWidth - 1 - convInfo.padInfo.left;
          for (let b = 0; b < batchSize; ++b) {
              for (let d1 = 0; d1 < inChannels; ++d1) {
                  // Frames of depth
                  for (let xF = 0; xF < inDepth; ++xF) {
                      const xFCorner = xF - frontPad;
                      const xFMin = Math.max(0, Math.ceil(xFCorner / strideDepth));
                      const yFMax = Math.min(outDepth, (filterDepth + xFCorner) / strideDepth);
                      // Rows as per standard 2d matrix notation
                      for (let xR = 0; xR < inHeight; ++xR) {
                          const xRCorner = xR - topPad;
                          const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                          const yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                          // Columns as per standard 2d matrix notation
                          for (let xC = 0; xC < inWidth; ++xC) {
                              const xCCorner = xC - leftPad;
                              const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                              const yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                              let dotProd = 0;
                              for (let yF = xFMin; yF < yFMax; ++yF) {
                                  const wF = yF * strideDepth - xFCorner;
                                  for (let yR = xRMin; yR < yRMax; ++yR) {
                                      const wR = yR * strideHeight - xRCorner;
                                      for (let yC = xCMin; yC < yCMax; ++yC) {
                                          const wC = yC * strideWidth - xCCorner;
                                          const dyOffset = dyS0 * b + dyS1 * yF + dyS2 * yR + dyS3 * yC;
                                          const fltOffset = fltS0 * (filterDepth - 1 - wF) +
                                              fltS1 * (filterHeight - 1 - wR) +
                                              fltS2 * (filterWidth - 1 - wC) + fltS3 * d1;
                                          for (let d2 = 0; d2 < outChannels; ++d2) {
                                              const pixel = dyValues[dyOffset + d2];
                                              const weight = fltValues[fltOffset + d2];
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
      }
      conv2dDerFilter(x, dy, convInfo) {
          assertNotComplex([x, dy], 'conv2dDerFilter');
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const isChannelsLast = convInfo.dataFormat === 'channelsLast';
          const dW = tf.buffer(convInfo.filterShape, 'float32');
          const leftPad = convInfo.padInfo.left;
          const topPad = convInfo.padInfo.top;
          const xBuf = this.bufferSync(x);
          const dyBuf = this.bufferSync(dy);
          for (let wR = 0; wR < filterHeight; ++wR) {
              const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
              const yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
              for (let wC = 0; wC < filterWidth; ++wC) {
                  const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                  const yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                  for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                      for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                          // Need to convolve.
                          let dotProd = 0;
                          for (let b = 0; b < convInfo.batchSize; ++b) {
                              for (let yR = yRMin; yR < yRMax; ++yR) {
                                  const xR = wR + yR * strideHeight - topPad;
                                  for (let yC = yCMin; yC < yCMax; ++yC) {
                                      const xC = wC + yC * strideWidth - leftPad;
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
      }
      conv3dDerFilter(x, dy, convInfo) {
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const filterDepth = convInfo.filterDepth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const dw = tf.buffer(convInfo.filterShape, 'float32');
          const dwValues = dw.values;
          const [dwS0, dwS1, dwS2, dwS3] = dw.strides;
          const dyValues = this.readSync(dy.dataId);
          const [dyS0, dyS1, dyS2, dyS3] = dy.strides;
          const xValues = this.readSync(x.dataId);
          const [xS0, xS1, xS2, xS3] = x.strides;
          const frontPad = convInfo.padInfo.front;
          const leftPad = convInfo.padInfo.left;
          const topPad = convInfo.padInfo.top;
          for (let wF = 0; wF < filterDepth; ++wF) {
              const yFMin = Math.max(0, Math.ceil((frontPad - wF) / strideDepth));
              const yFMax = Math.min(convInfo.outDepth, (convInfo.inDepth + frontPad - wF) / strideDepth);
              const wOffset1 = wF * dwS0;
              for (let wR = 0; wR < filterHeight; ++wR) {
                  const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
                  const yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
                  const wOffset2 = wR * dwS1 + wOffset1;
                  for (let wC = 0; wC < filterWidth; ++wC) {
                      const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                      const yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                      const wOffset3 = wC * dwS2 + wOffset2;
                      for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                          const wOffset4 = d1 * dwS3 + wOffset3;
                          for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                              let dotProd = 0;
                              for (let b = 0; b < convInfo.batchSize; ++b) {
                                  const xOffset1 = b * xS0;
                                  const yOffset1 = b * dyS0;
                                  for (let yF = yFMin; yF < yFMax; ++yF) {
                                      const xF = wF + yF * strideDepth - frontPad;
                                      const xOffset2 = xF * xS1 + xOffset1;
                                      const yOffset2 = yF * dyS1 + yOffset1;
                                      for (let yR = yRMin; yR < yRMax; ++yR) {
                                          const xR = wR + yR * strideHeight - topPad;
                                          const xOffset3 = xR * xS2 + xOffset2;
                                          const yOffset3 = yR * dyS2 + yOffset2;
                                          for (let yC = yCMin; yC < yCMax; ++yC) {
                                              const xC = wC + yC * strideWidth - leftPad;
                                              const xOffset4 = xC * xS3 + xOffset3;
                                              const yOffset4 = yC * dyS3 + yOffset3;
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
      }
      fusedDepthwiseConv2D({ input, filter, convInfo, bias, activation, preluActivationWeights }) {
          let result = this.depthwiseConv2D(input, filter, convInfo);
          if (bias) {
              result = this.add(result, bias);
          }
          if (activation) {
              result =
                  mapActivation(this, result, activation, preluActivationWeights);
          }
          return result;
      }
      depthwiseConv2D(x, filter, convInfo) {
          assertNotComplex([x, filter], 'depthwiseConv2D');
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const padLeft = convInfo.padInfo.left;
          const padTop = convInfo.padInfo.top;
          const chMul = convInfo.outChannels / convInfo.inChannels;
          const y = tf.buffer(convInfo.outShape, x.dtype);
          const xVals = this.readSync(x.dataId);
          const wVals = this.readSync(filter.dataId);
          const yVals = y.values;
          for (let b = 0; b < convInfo.batchSize; ++b) {
              const xOffset1 = b * x.strides[0];
              const yOffset1 = b * y.strides[0];
              for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                  const yOffset2 = yOffset1 + yR * y.strides[1];
                  const xRCorner = yR * convInfo.strideHeight - padLeft;
                  for (let wR = 0; wR < filterHeight; ++wR) {
                      const xR = xRCorner + wR * dilationHeight;
                      if (xR < 0 || xR >= convInfo.inHeight) {
                          continue;
                      }
                      const wOffset1 = wR * filter.strides[0];
                      const xOffset2 = xOffset1 + xR * x.strides[1];
                      for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                          const yOffset3 = yOffset2 + yC * y.strides[2];
                          const xCCorner = yC * convInfo.strideWidth - padTop;
                          for (let wC = 0; wC < filterWidth; ++wC) {
                              const xC = xCCorner + wC * dilationWidth;
                              if (xC < 0 || xC >= convInfo.inWidth) {
                                  continue;
                              }
                              const wOffset2 = wOffset1 + wC * filter.strides[1];
                              const xOffset3 = xOffset2 + xC * convInfo.inChannels;
                              let yOffset4 = yOffset3;
                              let wOffset3 = wOffset2;
                              for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                  const xVal = xVals[xOffset3 + d1];
                                  for (let q = 0; q < chMul; ++q) {
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
      }
      depthwiseConv2DDerInput(dy, filter, convInfo) {
          assertNotComplex([dy, filter], 'depthwiseConv2DDerInput');
          const dx = tf.buffer(convInfo.inShape, 'float32');
          const dxValues = dx.values;
          const [dxS0, dxS1, dxS2] = dx.strides;
          const dyValues = this.readSync(dy.dataId);
          const [dyS0, dyS1, dyS2] = dy.strides;
          const fltValues = this.readSync(filter.dataId);
          const [fltS0, fltS1, fltS2] = filter.strides;
          const { batchSize, filterHeight, filterWidth, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, strideHeight, strideWidth } = convInfo;
          const topPad = filterHeight - 1 - convInfo.padInfo.top;
          const leftPad = filterWidth - 1 - convInfo.padInfo.left;
          const chMul = outChannels / inChannels;
          for (let b = 0; b < batchSize; ++b) {
              for (let d1 = 0; d1 < inChannels; ++d1) {
                  for (let xR = 0; xR < inHeight; ++xR) {
                      const xRCorner = xR - topPad;
                      const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                      const yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                      for (let xC = 0; xC < inWidth; ++xC) {
                          const xCCorner = xC - leftPad;
                          const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                          const yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                          let dotProd = 0;
                          for (let yR = xRMin; yR < yRMax; ++yR) {
                              const wR = yR * strideHeight - xRCorner;
                              for (let yC = xCMin; yC < yCMax; ++yC) {
                                  const wC = yC * strideWidth - xCCorner;
                                  const dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                                  const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                      fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                                  for (let dm = 0; dm < chMul; ++dm) {
                                      const d2 = d1 * chMul + dm;
                                      const pixel = dyValues[dyOffset + d2];
                                      const weight = fltValues[fltOffset + dm];
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
      }
      depthwiseConv2DDerFilter(x, dy, convInfo) {
          assertNotComplex([x, dy], 'depthwiseConv2DDerFilter');
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const dW = tf.buffer(convInfo.filterShape, 'float32');
          const leftPad = convInfo.padInfo.left;
          const topPad = convInfo.padInfo.top;
          const chMul = convInfo.outChannels / convInfo.inChannels;
          const xBuf = this.bufferSync(x);
          const dyBuf = this.bufferSync(dy);
          for (let wR = 0; wR < filterHeight; ++wR) {
              const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
              const yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
              for (let wC = 0; wC < filterWidth; ++wC) {
                  const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                  const yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                  for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                      const d1 = Math.trunc(d2 / chMul);
                      const dm = d2 % chMul;
                      let dotProd = 0;
                      for (let b = 0; b < convInfo.batchSize; ++b) {
                          for (let yR = yRMin; yR < yRMax; ++yR) {
                              const xR = wR + yR * strideHeight - topPad;
                              for (let yC = yCMin; yC < yCMax; ++yC) {
                                  const xC = wC + yC * strideWidth - leftPad;
                                  dotProd += xBuf.get(b, xR, xC, d1) * dyBuf.get(b, yR, yC, d2);
                              }
                          }
                      }
                      dW.set(dotProd, wR, wC, d1, dm);
                  }
              }
          }
          return dW.toTensor();
      }
      tile(x, reps) {
          assertNotComplex(x, 'tile');
          return tile(this.bufferSync(x), reps);
      }
      pad(x, paddings, constantValue) {
          assertNotComplex(x, 'pad');
          const outShape = paddings.map((p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
          const start = paddings.map(p => p[0]);
          const xBuffer = this.bufferSync(x);
          const buffer = tf.buffer(outShape, x.dtype);
          if (constantValue !== 0) {
              buffer.values.fill(constantValue);
          }
          for (let i = 0; i < x.size; i++) {
              const coords = xBuffer.indexToLoc(i);
              const outCoords = coords.map((c, i) => c + start[i]);
              buffer.set(xBuffer.get(...coords), ...outCoords);
          }
          return buffer.toTensor();
      }
      gather(x, indices, axis) {
          assertNotComplex([x, indices], 'gather');
          const newShape = x.shape.slice();
          const indicesValues = this.readSync(indices.dataId);
          newShape[axis] = indicesValues.length;
          const result = tf.buffer(newShape, x.dtype);
          const xBuf = this.bufferSync(x);
          for (let i = 0; i < result.size; ++i) {
              const newLoc = result.indexToLoc(i);
              const originalLoc = newLoc.slice();
              originalLoc[axis] = indicesValues[newLoc[axis]];
              const originalIndex = xBuf.locToIndex(originalLoc);
              result.values[i] = xBuf.values[originalIndex];
          }
          return result.toTensor();
      }
      batchToSpaceND(x, blockShape, crops) {
          assertNotComplex([x], 'batchToSpaceND');
          const prod = blockShape.reduce((a, b) => a * b);
          const reshaped = tf.backend_util.getReshaped(x.shape, blockShape, prod);
          const permuted = tf.backend_util.getPermuted(reshaped.length, blockShape.length);
          const reshapedPermuted = tf.backend_util.getReshapedPermuted(x.shape, blockShape, prod);
          const sliceBeginCoords = tf.backend_util.getSliceBeginCoords(crops, blockShape.length);
          const sliceSize = tf.backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);
          return tf.transpose(x.reshape(reshaped), permuted)
              .reshape(reshapedPermuted)
              .slice(sliceBeginCoords, sliceSize);
      }
      spaceToBatchND(x, blockShape, paddings) {
          assertNotComplex([x], 'spaceToBatchND');
          const prod = blockShape.reduce((a, b) => a * b);
          const completePaddings = [[0, 0]];
          completePaddings.push(...paddings);
          for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
              completePaddings.push([0, 0]);
          }
          const paddedX = x.pad(completePaddings);
          const reshapedPaddedShape = tf.backend_util.getReshaped(paddedX.shape, blockShape, prod, false);
          const permutedReshapedPaddedPermutation = tf.backend_util.getPermuted(reshapedPaddedShape.length, blockShape.length, false);
          const flattenShape = tf.backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);
          return tf.transpose(paddedX.reshape(reshapedPaddedShape), permutedReshapedPaddedPermutation)
              .reshape(flattenShape);
      }
      maxPool(x, convInfo) {
          assertNotComplex(x, 'maxPool');
          const xValues = this.readSync(x.dataId);
          return pool(xValues, x.shape, x.dtype, x.strides, convInfo, 'max')
              .toTensor();
      }
      maxPoolBackprop(dy, x, y, convInfo) {
          assertNotComplex([x, y], 'maxPoolBackprop');
          const xValues = this.readSync(x.dataId);
          const maxPosBuf = tf.buffer(convInfo.outShape, x.dtype, maxPoolPositions(xValues, x.shape, x.dtype, convInfo).values);
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          const dx = tf.buffer(x.shape, 'float32');
          const dyBuf = this.bufferSync(dy);
          for (let b = 0; b < convInfo.batchSize; ++b) {
              for (let d = 0; d < convInfo.inChannels; ++d) {
                  for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
                      for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
                          // Shader code begins.
                          const dyRCorner = dxR - padTop;
                          const dyCCorner = dxC - padLeft;
                          let dotProd = 0;
                          for (let wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
                              const dyR = (dyRCorner + wR) / strideHeight;
                              if (dyR < 0 || dyR >= convInfo.outHeight ||
                                  Math.floor(dyR) !== dyR) {
                                  continue;
                              }
                              for (let wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                                  const dyC = (dyCCorner + wC) / strideWidth;
                                  if (dyC < 0 || dyC >= convInfo.outWidth ||
                                      Math.floor(dyC) !== dyC) {
                                      continue;
                                  }
                                  const maxPos = effectiveFilterHeight * effectiveFilterWidth -
                                      1 - maxPosBuf.get(b, dyR, dyC, d);
                                  const curPos = wR * effectiveFilterWidth + wC;
                                  const mask = maxPos === curPos ? 1 : 0;
                                  if (mask === 0) {
                                      continue;
                                  }
                                  const pixel = dyBuf.get(b, dyR, dyC, d);
                                  dotProd += pixel * mask;
                              }
                          }
                          dx.set(dotProd, b, dxR, dxC, d);
                      }
                  }
              }
          }
          return dx.toTensor();
      }
      avgPoolBackprop(dy, x, convInfo) {
          assertNotComplex([dy, x], 'avgPoolBackprop');
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          const dx = tf.buffer(x.shape, 'float32');
          const avgMultiplier = 1 / (filterHeight * filterWidth);
          const dyBuf = this.bufferSync(dy);
          for (let b = 0; b < convInfo.batchSize; ++b) {
              for (let d = 0; d < convInfo.inChannels; ++d) {
                  for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
                      for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
                          // Shader code begins.
                          const dyRCorner = dxR - padTop;
                          const dyCCorner = dxC - padLeft;
                          let dotProd = 0;
                          for (let wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
                              const dyR = (dyRCorner + wR) / strideHeight;
                              if (dyR < 0 || dyR >= convInfo.outHeight ||
                                  Math.floor(dyR) !== dyR) {
                                  continue;
                              }
                              for (let wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                                  const dyC = (dyCCorner + wC) / strideWidth;
                                  if (dyC < 0 || dyC >= convInfo.outWidth ||
                                      Math.floor(dyC) !== dyC) {
                                      continue;
                                  }
                                  const pixel = dyBuf.get(b, dyR, dyC, d);
                                  dotProd += pixel;
                              }
                          }
                          dx.set(dotProd * avgMultiplier, b, dxR, dxC, d);
                      }
                  }
              }
          }
          return dx.toTensor();
      }
      pool3d(x, convInfo, poolType) {
          assertNotComplex(x, 'pool3d');
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterDepth = convInfo.effectiveFilterDepth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padFront = convInfo.padInfo.front;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          const initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
              Number.POSITIVE_INFINITY);
          const xValues = this.readSync(x.dataId);
          const output = tf.buffer(convInfo.outShape, x.dtype);
          const outputVals = output.values;
          const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] *
              convInfo.outShape[3] * convInfo.outShape[4];
          const outputDepthStrides = convInfo.outShape[2] * convInfo.outShape[3] * convInfo.outShape[4];
          const outputRowStrides = convInfo.outShape[3] * convInfo.outShape[4];
          const outputColStrides = convInfo.outShape[4];
          for (let batch = 0; batch < convInfo.batchSize; ++batch) {
              const outputBatchOffset = batch * outputBatchStrides;
              const inputBatchOffset = batch * x.strides[0];
              for (let channel = 0; channel < convInfo.inChannels; ++channel) {
                  for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                      const xDepthCorner = yDepth * strideDepth - padFront;
                      let xDepthMin = xDepthCorner;
                      while (xDepthMin < 0) {
                          xDepthMin += dilationDepth;
                      }
                      const xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                      const outputDepthOffset = outputBatchOffset + yDepth * outputDepthStrides;
                      for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                          const xRowCorner = yRow * strideHeight - padTop;
                          let xRowMin = xRowCorner;
                          while (xRowMin < 0) {
                              xRowMin += dilationHeight;
                          }
                          const xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                          const outputRowOffset = outputDepthOffset + yRow * outputRowStrides;
                          for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                              const xColCorner = yCol * strideWidth - padLeft;
                              let xColMin = xColCorner;
                              while (xColMin < 0) {
                                  xColMin += dilationWidth;
                              }
                              const xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                              // Shader code begins
                              const outputColOffset = outputRowOffset + yCol * outputColStrides;
                              let minMaxValue = initialValue;
                              let avgValue = 0;
                              let count = 0;
                              for (let xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                                  const xDepthOffset = inputBatchOffset + xDepth * x.strides[1];
                                  for (let xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                      const xRowOffset = xDepthOffset + xRow * x.strides[2];
                                      for (let xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                          const xColOffset = xRowOffset + xCol * x.strides[3];
                                          const pixel = xValues[xColOffset + channel];
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
                              const outputOffset = outputColOffset + channel;
                              outputVals[outputOffset] =
                                  poolType === 'avg' ? avgValue / count : minMaxValue;
                          }
                      }
                  }
              }
          }
          return output.toTensor();
      }
      avgPool3d(x, convInfo) {
          assertNotComplex(x, 'avgPool3d');
          return this.pool3d(x, convInfo, 'avg').toFloat();
      }
      avgPool3dBackprop(dy, x, convInfo) {
          assertNotComplex([dy, x], 'avgPool3dBackprop');
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const filterDepth = convInfo.filterDepth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterDepth = convInfo.effectiveFilterDepth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
          const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          const dx = tf.buffer(x.shape, 'float32');
          const avgMultiplier = 1 / (filterDepth * filterHeight * filterWidth);
          const dyBuf = this.bufferSync(dy);
          for (let batch = 0; batch < convInfo.batchSize; ++batch) {
              for (let channel = 0; channel < convInfo.inChannels; ++channel) {
                  for (let dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
                      for (let dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
                          for (let dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
                              // Shader code begins.
                              const dyDepthCorner = dxDepth - padFront;
                              const dyRowCorner = dxRow - padTop;
                              const dyColCorner = dxCol - padLeft;
                              let dotProd = 0;
                              for (let wDepth = 0; wDepth < effectiveFilterDepth; wDepth += dilationDepth) {
                                  const dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                                  if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                                      Math.floor(dyDepth) !== dyDepth) {
                                      continue;
                                  }
                                  for (let wRow = 0; wRow < effectiveFilterHeight; wRow += dilationHeight) {
                                      const dyRow = (dyRowCorner + wRow) / strideHeight;
                                      if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                                          Math.floor(dyRow) !== dyRow) {
                                          continue;
                                      }
                                      for (let wCol = 0; wCol < effectiveFilterWidth; wCol += dilationWidth) {
                                          const dyCol = (dyColCorner + wCol) / strideWidth;
                                          if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                                              Math.floor(dyCol) !== dyCol) {
                                              continue;
                                          }
                                          const pixel = dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
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
      }
      maxPool3d(x, convInfo) {
          assertNotComplex(x, 'maxPool3d');
          return this.pool3d(x, convInfo, 'max').toFloat();
      }
      maxPool3dPositions(x, convInfo) {
          const maxPositions = tf.buffer(convInfo.outShape, 'int32');
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterDepth = convInfo.effectiveFilterDepth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padFront = convInfo.padInfo.front;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          const xBuf = this.bufferSync(x);
          for (let batch = 0; batch < convInfo.batchSize; ++batch) {
              for (let channel = 0; channel < convInfo.inChannels; ++channel) {
                  for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                      const xDepthCorner = yDepth * strideDepth - padFront;
                      let xDepthMin = xDepthCorner;
                      while (xDepthMin < 0) {
                          xDepthMin += dilationDepth;
                      }
                      const xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                      for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                          const xRowCorner = yRow * strideHeight - padTop;
                          let xRowMin = xRowCorner;
                          while (xRowMin < 0) {
                              xRowMin += dilationHeight;
                          }
                          const xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                          for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                              const xColCorner = yCol * strideWidth - padLeft;
                              let xColMin = xColCorner;
                              while (xColMin < 0) {
                                  xColMin += dilationWidth;
                              }
                              const xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                              // Shader code begins
                              let maxValue = Number.NEGATIVE_INFINITY;
                              let maxPosition = -1;
                              for (let xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                                  const wDepth = xDepth - xDepthCorner;
                                  for (let xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                      const wRow = xRow - xRowCorner;
                                      for (let xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                          const wCol = xCol - xColCorner;
                                          const pixel = xBuf.get(batch, xDepth, xRow, xCol, channel);
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
      }
      maxPool3dBackprop(dy, x, y, convInfo) {
          assertNotComplex([x, y], 'maxPool3dBackprop');
          const maxPositions = this.maxPool3dPositions(x, convInfo);
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterDepth = convInfo.effectiveFilterDepth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
          const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          const dx = tf.buffer(x.shape, 'float32');
          const maxPosBuf = this.bufferSync(maxPositions);
          const dyBuf = this.bufferSync(dy);
          for (let batch = 0; batch < convInfo.batchSize; ++batch) {
              for (let channel = 0; channel < convInfo.inChannels; ++channel) {
                  for (let dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
                      for (let dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
                          for (let dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
                              // Shader code begins
                              const dyDepthCorner = dxDepth - padFront;
                              const dyRowCorner = dxRow - padTop;
                              const dyColCorner = dxCol - padLeft;
                              let dotProd = 0;
                              for (let wDepth = 0; wDepth < effectiveFilterDepth; wDepth += dilationDepth) {
                                  const dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                                  if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                                      Math.floor(dyDepth) !== dyDepth) {
                                      continue;
                                  }
                                  for (let wRow = 0; wRow < effectiveFilterHeight; wRow += dilationHeight) {
                                      const dyRow = (dyRowCorner + wRow) / strideHeight;
                                      if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                                          Math.floor(dyRow) !== dyRow) {
                                          continue;
                                      }
                                      for (let wCol = 0; wCol < effectiveFilterWidth; wCol += dilationWidth) {
                                          const dyCol = (dyColCorner + wCol) / strideWidth;
                                          if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                                              Math.floor(dyCol) !== dyCol) {
                                              continue;
                                          }
                                          const maxPos = effectiveFilterDepth *
                                              effectiveFilterHeight * effectiveFilterWidth -
                                              1 -
                                              maxPosBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                          const curPos = wDepth * effectiveFilterHeight * effectiveFilterWidth +
                                              wRow * effectiveFilterWidth + wCol;
                                          const mask = maxPos === curPos ? 1 : 0;
                                          if (mask === 0) {
                                              continue;
                                          }
                                          const pixel = dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
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
      }
      cast(x, dtype) {
          return tf.backend_util.castTensor(x, dtype, this);
      }
      reshape(x, shape) {
          return tf.backend_util.reshapeTensor(x, shape);
      }
      avgPool(x, convInfo) {
          assertNotComplex(x, 'avgPool');
          assertNotComplex(x, 'maxPool');
          const xValues = this.readSync(x.dataId);
          return pool(xValues, x.shape, x.dtype, x.strides, convInfo, 'avg')
              .toTensor()
              .toFloat();
      }
      resizeBilinear(x, newHeight, newWidth, alignCorners) {
          assertNotComplex(x, 'resizeBilinear');
          const [batch, oldHeight, oldWidth, numChannels] = x.shape;
          const xValues = this.readSync(x.dataId);
          const result = new Float32Array(tf.util.sizeFromShape([batch, newHeight, newWidth, numChannels]));
          const effectiveInputSize = [
              (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
              (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
          ];
          const effectiveOutputSize = [
              (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
              (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
          ];
          let outputIdx = 0;
          const effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
          const effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
          for (let b = 0; b < batch; b++) {
              for (let r = 0; r < newHeight; r++) {
                  const sourceFracRow = effectiveRowSizeRatio * r;
                  const sourceRowFloor = Math.floor(sourceFracRow);
                  const rowFrac = sourceFracRow - sourceRowFloor;
                  const sourceRowCeil = Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
                  const topRowOffset = b * x.strides[0] + sourceRowFloor * x.strides[1];
                  const botRowOffset = b * x.strides[0] + sourceRowCeil * x.strides[1];
                  for (let c = 0; c < newWidth; c++) {
                      const sourceFracCol = effectiveColSizeRatio * c;
                      const sourceColFloor = Math.floor(sourceFracCol);
                      const colFrac = sourceFracCol - sourceColFloor;
                      const sourceColCeil = Math.min(oldWidth - 1, Math.ceil(sourceFracCol));
                      const topLeftOffest = topRowOffset + sourceColFloor * x.strides[2];
                      const botLeftOffset = botRowOffset + sourceColFloor * x.strides[2];
                      const topRightOffset = topRowOffset + sourceColCeil * x.strides[2];
                      const botRightOffest = botRowOffset + sourceColCeil * x.strides[2];
                      for (let d = 0; d < numChannels; d++) {
                          // Begin shader.
                          // Compute the fractional index of the source.
                          const topLeft = xValues[topLeftOffest + d];
                          const bottomLeft = xValues[botLeftOffset + d];
                          const topRight = xValues[topRightOffset + d];
                          const bottomRight = xValues[botRightOffest + d];
                          const top = topLeft + (topRight - topLeft) * colFrac;
                          const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                          const newValue = top + (bottom - top) * rowFrac;
                          result[outputIdx++] = newValue;
                      }
                  }
              }
          }
          return tf.tensor(result, [batch, newHeight, newWidth, numChannels]);
      }
      resizeBilinearBackprop(dy, x, alignCorners) {
          assertNotComplex([dy, x], 'resizeBilinearBackprop');
          const [batch, xHeight, xWidth, depth] = x.shape;
          const [, yHeight, yWidth] = dy.shape;
          const output = new Float32Array(batch * xHeight * xWidth * depth);
          // In the backwards pass, we want to find the pixels that were generated
          // for each pixel in the input image the forward pass and add the
          // corresponding coefficient from dy to the gradient (with some
          // interpolation).
          const effectiveXSize = [
              (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
              (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
          ];
          const effectiveYSize = [
              (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
              (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
          ];
          const heightScale = effectiveXSize[0] / effectiveYSize[0];
          const widthScale = effectiveXSize[1] / effectiveYSize[1];
          // Reference implementation
          // tslint:disable-next-line:max-line-length
          // https://github.com/tensorflow/tensorflow/blob/3039375c86a5bbc9610c7725dcaa95d635f87ba2/tensorflow/core/kernels/resize_bilinear_op.cc#L275
          const dyValues = this.readSync(dy.dataId);
          let offset = 0;
          for (let b = 0; b < batch; b++) {
              const bOffset = b * x.strides[0];
              for (let r = 0; r < yHeight; r++) {
                  const dxR = r * heightScale;
                  const topDxRIndex = Math.floor(dxR);
                  const bottomDxRIndex = Math.min(Math.ceil(dxR), xHeight - 1);
                  const topDxROffset = bOffset + topDxRIndex * x.strides[1];
                  const bottomDxROffset = bOffset + bottomDxRIndex * x.strides[1];
                  const dxRLerp = dxR - topDxRIndex;
                  const inverseDxRLerp = 1.0 - dxRLerp;
                  for (let c = 0; c < yWidth; c++) {
                      const dxC = c * widthScale;
                      const leftDxCIndex = Math.floor(dxC);
                      const rightDxCIndex = Math.min(Math.ceil(dxC), xWidth - 1);
                      const dxCLerp = dxC - leftDxCIndex;
                      const inverseDxCLerp = 1.0 - dxCLerp;
                      const topLeftRCOffset = topDxROffset + leftDxCIndex * x.strides[2];
                      const topRightRCOffset = topDxROffset + rightDxCIndex * x.strides[2];
                      const bottomLeftRCOffset = bottomDxROffset + leftDxCIndex * x.strides[2];
                      const bottomRightRCOffset = bottomDxROffset + rightDxCIndex * x.strides[2];
                      const inverseDxRLerpTimesInverseDxCLerp = inverseDxRLerp * inverseDxCLerp;
                      const inverseDxRLerpTimesDxCLerp = inverseDxRLerp * dxCLerp;
                      const dxRLerpTimesInverseDxCLerp = dxRLerp * inverseDxCLerp;
                      const dxRLerpTimesDxCLerp = dxRLerp * dxCLerp;
                      for (let d = 0; d < depth; d++) {
                          const dyVal = dyValues[offset++];
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
      }
      resizeNearestNeighbor(x, newHeight, newWidth, alignCorners) {
          assertNotComplex(x, 'resizeNearestNeighbor');
          const [batch, oldHeight, oldWidth, numChannels] = x.shape;
          const xValues = this.readSync(x.dataId);
          const output = new Float32Array(batch * newHeight * newWidth * numChannels);
          const effectiveInputSize = [
              (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
              (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
          ];
          const effectiveOutputSize = [
              (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
              (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
          ];
          const effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
          const effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
          let outputOffset = 0;
          for (let b = 0; b < batch; b++) {
              const batchOffset = b * x.strides[0];
              for (let r = 0; r < newHeight; r++) {
                  const sourceFracRow = effectiveRowSizeRatio * r;
                  const sourceNearestRow = Math.min(oldHeight - 1, alignCorners ? Math.round(sourceFracRow) :
                      Math.floor(sourceFracRow));
                  const rowOffset = batchOffset + sourceNearestRow * x.strides[1];
                  for (let c = 0; c < newWidth; c++) {
                      const sourceFracCol = effectiveColSizeRatio * c;
                      const sourceNearestCol = Math.min(oldWidth - 1, alignCorners ? Math.round(sourceFracCol) :
                          Math.floor(sourceFracCol));
                      const colOffset = rowOffset + sourceNearestCol * x.strides[2];
                      for (let d = 0; d < numChannels; d++) {
                          // Begin shader.
                          // Compute the fractional index of the source.
                          const newVal = xValues[colOffset + d];
                          output[outputOffset++] = newVal;
                      }
                  }
              }
          }
          return tf.tensor(output, [batch, newHeight, newWidth, numChannels], x.dtype);
      }
      resizeNearestNeighborBackprop(dy, x, alignCorners) {
          assertNotComplex([dy, x], 'resizeNearestNeighborBackprop');
          const [batch, xHeight, xWidth, depth] = x.shape;
          const [, yHeight, yWidth] = dy.shape;
          const output = new Float32Array(batch * xHeight * xWidth * depth);
          const dyValues = this.readSync(dy.dataId);
          // In the backwards pass, we want to find the pixels that were generated
          // for each pixel in the input image the forward pass
          const effectiveXSize = [
              (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
              (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
          ];
          const effectiveYSize = [
              (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
              (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
          ];
          const heightScale = effectiveXSize[0] / effectiveYSize[0];
          const widthScale = effectiveXSize[1] / effectiveYSize[1];
          const invHeightScale = 1 / heightScale;
          const invWidthScale = 1 / widthScale;
          // This defines the size of the window of values around a particular
          // index in dy that we want to search for contributions to dx.
          const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
          const winWidth = (Math.ceil(invWidthScale) * 2) + 2;
          // Loop over the output space.
          for (let b = 0; b < batch; b++) {
              const batchOffset = b * x.strides[0];
              for (let r = 0; r < xHeight; r++) {
                  const rowOffset = batchOffset + r * x.strides[1];
                  // Compute bounds for where in dy we will look
                  const startRLerp = Math.floor(r * invHeightScale);
                  const startDyR = Math.floor(startRLerp - (winHeight / 2));
                  for (let c = 0; c < xWidth; c++) {
                      const colOffset = rowOffset + c * x.strides[2];
                      // Compute bounds for where in dy we will look
                      const startCLerp = Math.floor(c * invWidthScale);
                      const startDyC = Math.floor(startCLerp - (winWidth / 2));
                      for (let d = 0; d < depth; d++) {
                          let accum = 0;
                          // loop over dy
                          for (let dyRIndex = 0; dyRIndex < winHeight; dyRIndex++) {
                              const dyR = dyRIndex + startDyR;
                              // Guard against the window exceeding the bounds of dy
                              if (dyR < 0 || dyR >= yHeight) {
                                  continue;
                              }
                              const dyROffset = batchOffset + dyR * dy.strides[1];
                              const sourceFracRow = dyR * heightScale;
                              const sourceNearestRow = Math.min(xHeight - 1, alignCorners ? Math.round(sourceFracRow) :
                                  Math.floor(sourceFracRow));
                              if (r !== sourceNearestRow) {
                                  continue;
                              }
                              for (let dyCIndex = 0; dyCIndex < winWidth; dyCIndex++) {
                                  const dyC = dyCIndex + startDyC;
                                  // Guard against the window exceeding the bounds of dy
                                  if (dyC < 0 || dyC >= yWidth) {
                                      continue;
                                  }
                                  const dyCOffset = dyROffset + dyC * dy.strides[2];
                                  const sourceFracCol = dyC * widthScale;
                                  const sourceNearestCol = Math.min(xWidth - 1, alignCorners ? Math.round(sourceFracCol) :
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
      }
      batchNormalization(x, mean, variance, varianceEpsilon, scale, offset) {
          assertNotComplex([x, mean, variance, scale, offset], 'batchNorm');
          const xVals = this.readSync(x.dataId);
          const mVals = this.readSync(mean.dataId);
          const varVals = this.readSync(variance.dataId);
          const sVals = scale ? this.readSync(scale.dataId) :
              new Float32Array([1]);
          const offVals = offset ? this.readSync(offset.dataId) :
              new Float32Array([0]);
          const outVals = new Float32Array(xVals.length);
          const offValsLength = offVals.length;
          const sValsLength = sVals.length;
          const varValsLength = varVals.length;
          const mValsLength = mVals.length;
          let offi = 0;
          let mi = 0;
          let si = 0;
          let vi = 0;
          for (let i = 0; i < xVals.length; ++i) {
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
      }
      localResponseNormalization4D(x, depthRadius, bias, alpha, beta) {
          assertNotComplex(x, 'localResponseNormalization4D');
          const channels = x.shape[3];
          const maxD = channels - 1;
          const xValues = this.readSync(x.dataId);
          const size = x.size;
          const result = new Float32Array(size);
          function sumAcrossChannels(offset) {
              const currentChannel = offset % channels;
              let beginSumOffset = offset - currentChannel + Math.max(0, currentChannel - depthRadius);
              const endSumOffset = offset - currentChannel +
                  Math.min(currentChannel + depthRadius, maxD);
              let sum = 0.0;
              for (; beginSumOffset <= endSumOffset; beginSumOffset++) {
                  const z = xValues[beginSumOffset];
                  sum += z * z;
              }
              return sum;
          }
          for (let offset = 0; offset < size; offset++) {
              const sum = sumAcrossChannels(offset);
              const val = xValues[offset] * Math.pow(bias + alpha * sum, -beta);
              result[offset] = val;
          }
          return tf.tensor4d(result, x.shape);
      }
      LRNGrad(dy, inputImage, outputImage, depthRadius, bias, alpha, beta) {
          assertNotComplex(dy, 'LRNGrad');
          const channels = dy.shape[3];
          const dyValues = this.readSync(dy.dataId);
          const inputImageValues = this.readSync(inputImage.dataId);
          const outputImageValues = this.readSync(outputImage.dataId);
          const result = new Float32Array(dy.size);
          const size = dy.size;
          for (let offset = 0; offset < size; offset++) {
              const currentChannel = offset % channels;
              const depthBegin = (offset - currentChannel) + Math.max(0, currentChannel - depthRadius);
              const depthEnd = (offset - currentChannel) +
                  Math.min(channels, currentChannel + depthRadius + 1);
              let norm = 0;
              for (let k = depthBegin; k < depthEnd; k++) {
                  norm += Math.pow(inputImageValues[k], 2);
              }
              norm = alpha * norm + bias;
              for (let k = depthBegin; k < depthEnd; k++) {
                  let dyi = -2 * alpha * beta * inputImageValues[k] *
                      outputImageValues[offset] / norm;
                  if (offset === k) {
                      dyi += Math.pow(norm, -beta);
                  }
                  dyi *= dyValues[offset];
                  result[k] += dyi;
              }
          }
          return tf.tensor4d(result, dy.shape);
      }
      multinomial(logits, normalized, numSamples, seed) {
          assertNotComplex(logits, 'multinomial');
          const probabilities = normalized ? logits : tf.softmax(logits);
          const batchSize = probabilities.shape[0];
          const numEvents = probabilities.shape[1];
          const res = tf.zeros([batchSize, numSamples], 'int32');
          const resVals = this.readSync(res.dataId);
          const probVals = this.readSync(probabilities.dataId);
          for (let b = 0; b < batchSize; ++b) {
              const offset = b * numEvents;
              // The cdf won't include the last event. It will be implicit if no other
              // event happened.
              const cdf = new Float32Array(numEvents - 1);
              cdf[0] = probVals[offset];
              for (let event = 1; event < cdf.length; ++event) {
                  cdf[event] = cdf[event - 1] + probVals[offset + event];
              }
              const random = seedrandom.alea(seed.toString());
              const outOffset = b * numSamples;
              for (let sampleId = 0; sampleId < numSamples; ++sampleId) {
                  const r = random();
                  // Assume last event happened by default.
                  resVals[outOffset + sampleId] = cdf.length;
                  for (let event = 0; event < cdf.length; event++) {
                      if (r < cdf[event]) {
                          resVals[outOffset + sampleId] = event;
                          break;
                      }
                  }
              }
          }
          return res;
      }
      oneHot(indices, depth, onValue, offValue) {
          assertNotComplex(indices, 'oneHot');
          const res = new Float32Array(indices.size * depth);
          res.fill(offValue);
          const indicesVal = this.readSync(indices.dataId);
          for (let event = 0; event < indices.size; ++event) {
              if (indicesVal[event] >= 0 && indicesVal[event] < depth) {
                  res[event * depth + indicesVal[event]] = onValue;
              }
          }
          return tf.tensor2d(res, [indices.size, depth], 'int32');
      }
      nonMaxSuppression(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
          assertNotComplex(boxes, 'nonMaxSuppression');
          const boxesVals = this.readSync(boxes.dataId);
          const scoresVals = this.readSync(scores.dataId);
          return nonMaxSuppressionV3(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
      }
      fft(x) {
          return this.fftBatch(x, false);
      }
      ifft(x) {
          return this.fftBatch(x, true);
      }
      /**
       * Calculate FFT of inner most elements of batch tensor.
       */
      fftBatch(x, inverse) {
          const batch = x.shape[0];
          const innerDim = x.shape[1];
          // Collects real and imaginary values separately.
          const realResult = tf.buffer(x.shape, 'float32');
          const imagResult = tf.buffer(x.shape, 'float32');
          const real = tf.real(x).as2D(batch, innerDim);
          const imag = tf.imag(x).as2D(batch, innerDim);
          for (let b = 0; b < batch; b++) {
              // TODO: Support slice ops for complex type.
              const r = real.slice([b, 0], [1, innerDim]);
              const i = imag.slice([b, 0], [1, innerDim]);
              const input = tf.complex(r, i);
              // Run FFT by batch element.
              const res = this.readSync(this.fftImpl(input, inverse).dataId);
              for (let d = 0; d < innerDim; d++) {
                  const c = tf.backend_util.getComplexWithIndex(res, d);
                  realResult.values[b * innerDim + d] = c.real;
                  imagResult.values[b * innerDim + d] = c.imag;
              }
          }
          const t = tf.complex(realResult.toTensor(), imagResult.toTensor());
          return t.as2D(batch, innerDim);
      }
      fftImpl(x, inverse) {
          const x1D = x.as1D();
          const n = x1D.size;
          if (this.isExponentOf2(n)) {
              let result = this.fftRadix2(x1D, n, inverse).as2D(x.shape[0], x.shape[1]);
              if (inverse) {
                  result = tf.complex(tf.real(result).div(tf.scalar(n)), tf.imag(result).div(tf.scalar(n)));
              }
              return result;
          }
          else {
              const data = this.readSync(x.dataId);
              const rawOutput = this.fourierTransformByMatmul(data, n, inverse);
              const output = tf.backend_util.splitRealAndImagArrays(rawOutput);
              return tf.complex(output.real, output.imag).as2D(x.shape[0], x.shape[1]);
          }
      }
      isExponentOf2(size) {
          return (size & size - 1) === 0;
      }
      // FFT using Cooley-Tukey algorithm on radix 2 dimensional input.
      fftRadix2(input, size, inverse) {
          if (size === 1) {
              return input;
          }
          const data = this.readSync(input.dataId);
          const half = size / 2;
          const evenComplex = tf.backend_util.complexWithEvenIndex(data);
          let evenTensor = tf.complex(evenComplex.real, evenComplex.imag).as1D();
          const oddComplex = tf.backend_util.complexWithOddIndex(data);
          let oddTensor = tf.complex(oddComplex.real, oddComplex.imag).as1D();
          // Recursive call for half part of original input.
          evenTensor = this.fftRadix2(evenTensor, half, inverse);
          oddTensor = this.fftRadix2(oddTensor, half, inverse);
          const e = tf.backend_util.exponents(size, inverse);
          const exponent = tf.complex(e.real, e.imag).mul(oddTensor);
          const addPart = evenTensor.add(exponent);
          const subPart = evenTensor.sub(exponent);
          const realTensor = tf.real(addPart).concat(tf.real(subPart));
          const imagTensor = tf.imag(addPart).concat(tf.imag(subPart));
          return tf.complex(realTensor, imagTensor).as1D();
      }
      // Calculate fourier transform by multplying sinusoid matrix.
      fourierTransformByMatmul(data, size, inverse) {
          const ret = new Float32Array(size * 2);
          // TODO: Use matmul instead once it supports complex64 type.
          for (let r = 0; r < size; r++) {
              let real = 0.0;
              let imag = 0.0;
              for (let c = 0; c < size; c++) {
                  const e = tf.backend_util.exponent(r * c, size, inverse);
                  const term = tf.backend_util.getComplexWithIndex(data, c);
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
      }
      depthToSpace(x, blockSize, dataFormat) {
          tf.util.assert(dataFormat === 'NHWC', () => `Only NHWC dataFormat supported on CPU for depthToSpace. Got ${dataFormat}`);
          tf.util.assert(blockSize > 1, () => `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);
          const batchSize = x.shape[0];
          const inputHeight = x.shape[1];
          const inputWidth = x.shape[2];
          const inputDepth = x.shape[3];
          const outputHeight = inputHeight * blockSize;
          const outputWidth = inputWidth * blockSize;
          const outputDepth = inputDepth / (blockSize * blockSize);
          const xValues = this.readSync(x.dataId);
          const result = new Float32Array(batchSize * outputHeight * outputWidth * outputDepth);
          let outputIdx = 0;
          for (let b = 0; b < batchSize; ++b) {
              for (let h = 0; h < outputHeight; ++h) {
                  const inH = Math.floor(h / blockSize);
                  const offsetH = (h % blockSize);
                  for (let w = 0; w < outputWidth; ++w) {
                      const inW = Math.floor(w / blockSize);
                      const offsetW = (w % blockSize);
                      const offsetD = (offsetH * blockSize + offsetW) * outputDepth;
                      for (let d = 0; d < outputDepth; ++d) {
                          const inD = d + offsetD;
                          const inputIdx = inD + inputDepth * (inW + inputWidth * (inH + inputHeight * b));
                          result[outputIdx++] = xValues[inputIdx];
                      }
                  }
              }
          }
          return tf.tensor4d(result, [batchSize, outputHeight, outputWidth, outputDepth]);
      }
      broadcastedBinaryOp(a, b, dtype, op) {
          const newShape = tf.backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
          const result = tf.buffer(newShape, dtype);
          const aVals = this.readSync(a.dataId);
          const bVals = this.readSync(b.dataId);
          const aBroadcastDims = tf.backend_util.getBroadcastDims(a.shape, newShape);
          const bBroadcastDims = tf.backend_util.getBroadcastDims(b.shape, newShape);
          const resVals = result.values;
          if (aBroadcastDims.length + bBroadcastDims.length === 0) {
              for (let i = 0; i < resVals.length; ++i) {
                  resVals[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
              }
          }
          else {
              const aBuf = this.bufferSync(a);
              const bBuf = this.bufferSync(b);
              for (let i = 0; i < resVals.length; ++i) {
                  const loc = result.indexToLoc(i);
                  const aLoc = loc.slice(-a.rank);
                  aBroadcastDims.forEach(d => aLoc[d] = 0);
                  const aIndex = aBuf.locToIndex(aLoc);
                  const bLoc = loc.slice(-b.rank);
                  bBroadcastDims.forEach(d => bLoc[d] = 0);
                  const bIndex = bBuf.locToIndex(bLoc);
                  resVals[i] = op(aVals[aIndex], bVals[bIndex]);
              }
          }
          return result.toTensor();
      }
      broadcastedBinaryComplexOp(a, b, op) {
          const newShape = tf.backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
          const realResult = tf.buffer(newShape, 'float32');
          const imagResult = tf.buffer(newShape, 'float32');
          const aVals = this.readSync(a.dataId);
          const bVals = this.readSync(b.dataId);
          const aBroadcastDims = tf.backend_util.getBroadcastDims(a.shape, newShape);
          const bBroadcastDims = tf.backend_util.getBroadcastDims(b.shape, newShape);
          const realVals = realResult.values;
          const imagVals = imagResult.values;
          if (aBroadcastDims.length + bBroadcastDims.length === 0) {
              for (let i = 0; i < realVals.length; i++) {
                  const aIdx = i % aVals.length;
                  const bIdx = i % bVals.length;
                  const result = op(aVals[aIdx * 2], aVals[aIdx * 2 + 1], bVals[bIdx * 2], bVals[bIdx * 2 + 1]);
                  realVals[i] = result.real;
                  imagVals[i] = result.imag;
              }
          }
          else {
              const aRealBuf = this.bufferSync(this.data.get(a.dataId).complexTensors.real);
              const bRealBuf = this.bufferSync(this.data.get(b.dataId).complexTensors.real);
              for (let i = 0; i < realVals.length; i++) {
                  const loc = realResult.indexToLoc(i);
                  const aLoc = loc.slice(-a.rank);
                  aBroadcastDims.forEach(d => aLoc[d] = 0);
                  const aIndex = aRealBuf.locToIndex(aLoc);
                  const bLoc = loc.slice(-b.rank);
                  bBroadcastDims.forEach(d => bLoc[d] = 0);
                  const bIndex = bRealBuf.locToIndex(bLoc);
                  const opResult = op(aVals[aIndex * 2], aVals[aIndex * 2 + 1], bVals[bIndex * 2], bVals[bIndex * 2 + 1]);
                  realVals[i] = opResult.real;
                  imagVals[i] = opResult.imag;
              }
          }
          return this.complex(realResult.toTensor(), imagResult.toTensor());
      }
      split(x, sizeSplits, axis) {
          return split(x, sizeSplits, axis);
      }
      dispose() { }
      floatPrecision() {
          return 32;
      }
      /** Returns the smallest representable number.  */
      epsilon() {
          return super.epsilon();
      }
      cropAndResize(images, boxes, boxIndex, cropSize, method, extrapolationValue) {
          const [batch, imageHeight, imageWidth, numChannels] = images.shape;
          const numBoxes = boxes.shape[0];
          const [cropHeight, cropWidth] = cropSize;
          const output = tf.buffer([numBoxes, cropHeight, cropWidth, numChannels], 'float32');
          const boxVals = this.readSync(boxes.dataId);
          const boxIndVals = this.readSync(boxIndex.dataId);
          const imageVals = this.readSync(images.dataId);
          const inStride = images.strides; // to calculate flat indexes into image
          const outStride = output.strides; // to calculate flat indexes into output
          // Reference implementation
          // tslint:disable-next-line:max-line-length
          // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op.cc
          for (let b = 0; b < numBoxes; b++) {
              const startInd = b * 4;
              const y1 = boxVals[startInd];
              const x1 = boxVals[startInd + 1];
              const y2 = boxVals[startInd + 2];
              const x2 = boxVals[startInd + 3];
              const bInd = boxIndVals[b];
              if (bInd >= batch) {
                  continue;
              }
              const heightScale = (cropHeight > 1) ?
                  (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) :
                  0;
              const widthScale = (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;
              for (let y = 0; y < cropHeight; y++) {
                  const yInd = (cropHeight > 1) ?
                      y1 * (imageHeight - 1) + y * (heightScale) :
                      0.5 * (y1 + y2) * (imageHeight - 1);
                  if (yInd < 0 || yInd > imageHeight - 1) {
                      for (let x = 0; x < cropWidth; x++) {
                          for (let c = 0; c < numChannels; c++) {
                              const ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                              output.values[ind] = extrapolationValue;
                          }
                      }
                      continue;
                  }
                  if (method === 'bilinear') {
                      const topInd = Math.floor(yInd);
                      const bottomInd = Math.ceil(yInd);
                      const yLerp = yInd - topInd;
                      for (let x = 0; x < cropWidth; x++) {
                          const xInd = (cropWidth > 1) ?
                              x1 * (imageWidth - 1) + x * widthScale :
                              0.5 * (x1 + x2) * (imageWidth - 1);
                          if (xInd < 0 || xInd > imageWidth - 1) {
                              for (let c = 0; c < numChannels; c++) {
                                  const ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                                  output.values[ind] = extrapolationValue;
                              }
                              continue;
                          }
                          const leftInd = Math.floor(xInd);
                          const rightInd = Math.ceil(xInd);
                          const xLerp = xInd - leftInd;
                          for (let c = 0; c < numChannels; c++) {
                              let ind = c + leftInd * inStride[2] + topInd * inStride[1] +
                                  bInd * inStride[0];
                              const topLeft = imageVals[ind];
                              ind = c + rightInd * inStride[2] + topInd * inStride[1] +
                                  bInd * inStride[0];
                              const topRight = imageVals[ind];
                              ind = c + leftInd * inStride[2] + bottomInd * inStride[1] +
                                  bInd * inStride[0];
                              const bottomLeft = imageVals[ind];
                              ind = c + rightInd * inStride[2] + bottomInd * inStride[1] +
                                  bInd * inStride[0];
                              const bottomRight = imageVals[ind];
                              const top = topLeft + (topRight - topLeft) * xLerp;
                              const bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
                              ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                              output.values[ind] = top + ((bottom - top) * yLerp);
                          }
                      }
                  }
                  else { // method == "nearest"
                      for (let x = 0; x < cropWidth; ++x) {
                          const xInd = (cropWidth > 1) ?
                              x1 * (imageWidth - 1) + x * widthScale :
                              0.5 * (x1 + x2) * (imageWidth - 1);
                          if (xInd < 0 || xInd > imageWidth - 1) {
                              for (let c = 0; c < numChannels; c++) {
                                  const ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                                  output.values[ind] = extrapolationValue;
                              }
                              continue;
                          }
                          const closestX = Math.round(xInd);
                          const closestY = Math.round(yInd);
                          for (let c = 0; c < numChannels; c++) {
                              const inInd = c + closestX * inStride[2] +
                                  closestY * inStride[1] + bInd * inStride[0];
                              const outInd = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                              output.values[outInd] = imageVals[inInd];
                          }
                      }
                  }
              }
          }
          return output.toTensor();
      }
      sparseToDense(sparseIndices, sparseValues, outputShape, defaultValue) {
          const { sliceRank, numUpdates, sliceSize, strides, outputSize } = tf.backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);
          const sumDupeIndices = false;
          return this.scatter(sparseIndices, sparseValues, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices);
      }
      gatherND(x, indices) {
          const indicesShape = indices.shape;
          const sliceRank = indicesShape[indicesShape.length - 1];
          const [resultShape, numSlices, sliceSize, strides] = tf.backend_util.prepareAndValidate(x, indices);
          if (numSlices === 0) {
              return tf.tensor([], resultShape, x.dtype);
          }
          const buffer = new tf.TensorBuffer([numSlices, sliceSize], x.dtype);
          const indicesData = this.readSync(indices.dataId);
          const xData = this.readSync(x.dataId);
          for (let i = 0; i < numSlices; i++) {
              const index = [];
              let flattenIndex = 0;
              for (let j = 0; j < sliceRank; j++) {
                  const dim = indicesData[i * sliceRank + j];
                  flattenIndex += dim * strides[j];
                  index.push(dim);
              }
              if (flattenIndex < 0 || flattenIndex >= x.size / sliceSize) {
                  throw new Error(`Invalid indices: ${index} does not index into ${x.shape}`);
              }
              for (let k = 0; k < sliceSize; k++) {
                  buffer.values[i * sliceSize + k] = xData[flattenIndex * sliceSize + k];
              }
          }
          return buffer.toTensor().reshape(resultShape);
      }
      scatterND(indices, updates, shape) {
          const { sliceRank, numUpdates, sliceSize, strides, outputSize } = tf.backend_util.calculateShapes(updates, indices, shape);
          const defaultValue = tf.scalar(0);
          const sumDupeIndices = true;
          return this.scatter(indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices);
      }
      fill(shape, value, dtype) {
          dtype = dtype || tf.util.inferDtype(value);
          const values = tf.util.getArrayFromDType(dtype, tf.util.sizeFromShape(shape));
          values.fill(value);
          return tf.engine().makeTensor(values, shape, dtype, this);
      }
      onesLike(x) {
          if (x.dtype === 'string') {
              throw new Error('onesLike is not supported for string tensors');
          }
          else {
              return this.fill(x.shape, 1, x.dtype);
          }
      }
      zerosLike(x) {
          const values = tf.util.getArrayFromDType(x.dtype, tf.util.sizeFromShape(x.shape));
          return this.makeOutput(values, x.shape, x.dtype);
      }
      linspace(start, stop, num) {
          return tf.backend_util.linspaceImpl(start, stop, num);
      }
      scatter(indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices) {
          const flattenShape = [outputSize / sliceSize, sliceSize];
          const indicesData = this.readSync(indices.dataId);
          const updatesData = this.readSync(updates.dataId);
          if (outputSize === 0) {
              return tf.tensor([], shape, updates.dtype);
          }
          const buffer = new tf.TensorBuffer(flattenShape, updates.dtype);
          buffer.values.fill(this.readSync(defaultValue.dataId)[0]);
          for (let i = 0; i < numUpdates; i++) {
              const index = [];
              let flattenIndex = 0;
              for (let j = 0; j < sliceRank; j++) {
                  const dim = indicesData[i * sliceRank + j];
                  index.push(dim);
                  flattenIndex += dim * strides[j];
              }
              if (flattenIndex < 0 || flattenIndex >= outputSize / sliceSize) {
                  throw new Error(`Invalid indices: ${index} does not index into ${shape}`);
              }
              for (let k = 0; k < sliceSize; k++) {
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
  function createBinaryKernelConfig(name, op) {
      return {
          kernelName: name,
          backendName: 'cpu',
          kernelFunc: ({ inputs, backend }) => {
              const { a, b } = inputs;
              const cpuBackend = backend;
              assertNotComplex([a, b], name);
              const aVals = cpuBackend.data.get(a.dataId).values;
              const bVals = cpuBackend.data.get(b.dataId).values;
              const [resultData, resultShape] = op(a.shape, b.shape, aVals, bVals, a.dtype);
              const dataId = cpuBackend.write(resultData, resultShape, a.dtype);
              return { dataId, shape: resultShape, dtype: a.dtype };
          }
      };
  }
  function createBinaryKernelImpl(op) {
      return (aShape, bShape, aVals, bVals, dtype) => {
          const newShape = tf.backend_util.assertAndGetBroadcastShape(aShape, bShape);
          const resultRank = newShape.length;
          const resultStrides = tf.util.computeStrides(newShape);
          const resultSize = tf.util.sizeFromShape(newShape);
          const result = tf.util.getTypedArrayFromDType(dtype, resultSize);
          const aRank = aShape.length;
          const bRank = bShape.length;
          const aStrides = tf.util.computeStrides(aShape);
          const bStrides = tf.util.computeStrides(bShape);
          const aBroadcastDims = tf.backend_util.getBroadcastDims(aShape, newShape);
          const bBroadcastDims = tf.backend_util.getBroadcastDims(bShape, newShape);
          if (aBroadcastDims.length + bBroadcastDims.length === 0) {
              for (let i = 0; i < result.length; ++i) {
                  result[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
              }
          }
          else {
              for (let i = 0; i < result.length; ++i) {
                  const loc = tf.util.indexToLoc(i, resultRank, resultStrides);
                  const aLoc = loc.slice(-aRank);
                  aBroadcastDims.forEach(d => aLoc[d] = 0);
                  const aIndex = tf.util.locToIndex(aLoc, aRank, aStrides);
                  const bLoc = loc.slice(-bRank);
                  bBroadcastDims.forEach(d => bLoc[d] = 0);
                  const bIndex = tf.util.locToIndex(bLoc, bRank, bStrides);
                  result[i] = op(aVals[aIndex], bVals[bIndex]);
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
  const divImpl = createBinaryKernelImpl((a, b) => a / b);

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
  const divConfig = createBinaryKernelConfig(tf.Div, divImpl);

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
      const strides = tf.util.computeStrides(xShape);
      const maxPools = pool(xValues, xShape, dtype, strides, convInfo, 'max');
      const maxPositions = maxPoolPositions(xValues, xShape, dtype, convInfo, true, includeBatchInIndex);
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
  const maxPoolWithArgmaxConfig = {
      kernelName: tf.MaxPoolWithArgmax,
      backendName: 'cpu',
      kernelFunc: ({ inputs, attrs, backend }) => {
          const { x } = inputs;
          const { filterSize, strides, pad, includeBatchInIndex } = attrs;
          const cpuBackend = backend;
          assertNotComplex(x, 'MaxPoolWithArgmax');
          const values = cpuBackend.data.get(x.dataId).values;
          const convInfo = tf.backend_util.computePool2DInfo(x.shape, filterSize, strides, [1, 1], pad);
          const [pooled, indexes] = maxPoolWithArgmaxImpl(values, x.shape, x.dtype, includeBatchInIndex, convInfo);
          const pooledDataId = cpuBackend.write(pooled, convInfo.outShape, x.dtype);
          const indexesDataId = cpuBackend.write(indexes, convInfo.outShape, x.dtype);
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
  const nonMaxSuppressionV5 = tf.kernel_impls.nonMaxSuppressionV5;
  const nonMaxSuppressionV5Config = {
      kernelName: tf.NonMaxSuppressionV5,
      backendName: 'cpu',
      kernelFunc: ({ inputs, backend, attrs }) => {
          const { boxes, scores } = inputs;
          const { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma } = attrs;
          const cpuBackend = backend;
          assertNotComplex(boxes, 'NonMaxSuppressionWithScore');
          const boxesVals = cpuBackend.data.get(boxes.dataId).values;
          const scoresVals = cpuBackend.data.get(scores.dataId).values;
          const maxOutputSizeVal = maxOutputSize;
          const iouThresholdVal = iouThreshold;
          const scoreThresholdVal = scoreThreshold;
          const softNmsSigmaVal = softNmsSigma;
          const { selectedIndices, selectedScores } = nonMaxSuppressionV5(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal, softNmsSigmaVal);
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
  const squareConfig = {
      kernelName: tf.Square,
      backendName: 'cpu',
      kernelFunc: ({ inputs, backend }) => {
          const { x } = inputs;
          const cpuBackend = backend;
          assertNotComplex(x, 'square');
          const values = cpuBackend.data.get(x.dataId).values;
          const newValues = new Float32Array(values.length);
          for (let i = 0; i < values.length; ++i) {
              const value = values[i];
              newValues[i] = value * value;
          }
          const dataId = cpuBackend.write(newValues, x.shape, x.dtype);
          return { dataId, shape: x.shape, dtype: x.dtype };
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
  const squaredDifferenceImpl = createBinaryKernelImpl((aVal, bVal) => {
      const diff = aVal - bVal;
      return diff * diff;
  });
  const squaredDifferenceConfig = createBinaryKernelConfig(tf.SquaredDifference, squaredDifferenceImpl);

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
      const xSize = tf.util.sizeFromShape(xShape);
      const xRank = xShape.length;
      const xStrides = tf.util.computeStrides(xShape);
      const newStrides = tf.util.computeStrides(newShape);
      const result = tf.util.getTypedArrayFromDType(dtype, tf.util.sizeFromShape(newShape));
      for (let i = 0; i < xSize; ++i) {
          const loc = tf.util.indexToLoc(i, xRank, xStrides);
          // Permute location.
          const newLoc = new Array(loc.length);
          for (let i = 0; i < newLoc.length; i++) {
              newLoc[i] = loc[perm[i]];
          }
          const newIndex = tf.util.locToIndex(newLoc, xRank, newStrides);
          result[newIndex] = xVals[i];
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
  const transposeConfig = {
      kernelName: tf.Transpose,
      backendName: 'cpu',
      kernelFunc: ({ inputs, attrs, backend }) => {
          const { x } = inputs;
          const { perm } = attrs;
          const cpuBackend = backend;
          assertNotComplex(x, 'transpose');
          const xRank = x.shape.length;
          const newShape = new Array(xRank);
          for (let i = 0; i < newShape.length; i++) {
              newShape[i] = x.shape[perm[i]];
          }
          const values = cpuBackend.data.get(x.dataId).values;
          const result = transposeImpl(values, x.shape, x.dtype, perm, newShape);
          const dataId = cpuBackend.write(result, newShape, x.dtype);
          return { dataId, shape: newShape, dtype: x.dtype };
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
  const kernelConfigs = [
      nonMaxSuppressionV5Config, squareConfig, squaredDifferenceConfig, divConfig,
      transposeConfig, maxPoolWithArgmaxConfig
  ];
  for (const kernelConfig of kernelConfigs) {
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
  tf.registerBackend('cpu', () => new MathBackendCPU(), 1 /* priority */);

  exports.MathBackendCPU = MathBackendCPU;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-backend-cpu.es2017.js.map
