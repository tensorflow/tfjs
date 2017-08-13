/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as conv_util from '../math/conv_util';
import * as util from '../util';

import * as concat3d_util from './concat3d_util';
import * as copy2D_util from './copy2d_util';
import {MatrixOrientation, NDArrayMath} from './math';
import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from './ndarray';

export class NDArrayMathCPU extends NDArrayMath {
  constructor(safeMode = false) {
    super(safeMode);
  }

  protected cloneInternal<T extends NDArray>(ndarray: T): T {
    return NDArray.make<T>(
        ndarray.shape, {values: new Float32Array(ndarray.getValues())});
  }

  protected reshapeInternal<T1 extends NDArray, T2 extends NDArray>(
      ndarray: T1, newShape: number[]): T2 {
    return this.cloneInternal(ndarray).reshape<T2>(newShape);
  }

  protected slice2DInternal(
      input: Array2D, beginRowCol: [number, number],
      sizeRowCol: [number, number]): Array2D {
    const result = Array2D.zeros(sizeRowCol);
    this.copy2DInternal(
        input, beginRowCol, sizeRowCol, result, [0, 0], sizeRowCol);
    return result;
  }

  protected copy2DInternal(
      source: Array2D, sourceBeginRowCol: [number, number],
      sourceSizeRowCol: [number, number], dest: Array2D,
      destBeginRowCol: [number, number],
      destSizeRowCol: [number, number]): void {
    copy2D_util.validateShapes(sourceSizeRowCol, destSizeRowCol);
    const srcValues = source.getValues();
    const dstValues = dest.getValues();
    const n = sourceSizeRowCol[0] * sourceSizeRowCol[1];
    for (let i = 0; i < n; ++i) {
      const srcRow = sourceBeginRowCol[0] + Math.floor(i / sourceSizeRowCol[1]);
      const srcCol = sourceBeginRowCol[1] + (i % sourceSizeRowCol[1]);
      const srcOff = srcRow * source.shape[1] + srcCol;
      const dstRow = destBeginRowCol[0] + Math.floor(i / destSizeRowCol[1]);
      const dstCol = destBeginRowCol[1] + (i % destSizeRowCol[1]);
      const dstOff = dstRow * dest.shape[1] + dstCol;
      dstValues[dstOff] = srcValues[srcOff];
    }
  }

  protected concat3DInternal(x1: Array3D, x2: Array3D, axis: number): Array3D {
    const outputShape =
        concat3d_util.computeConcat3DOutputShape(x1.shape, x2.shape, axis);

    const values = Array3D.zeros(outputShape);

    for (let i = 0; i < outputShape[0]; i++) {
      for (let j = 0; j < outputShape[1]; j++) {
        for (let k = 0; k < outputShape[2]; k++) {
          // Shader begins.
          const index: [number, number, number] = [i, j, k];
          let value: number;
          if (index[axis] < x1.shape[axis]) {
            value = x1.get(i, j, k);
          } else {
            index[axis] -= x1.shape[axis];
            const [i2, j2, k2] = index;
            value = x2.get(i2, j2, k2);
          }

          values.set(value, i, j, k);
        }
      }
    }

    return values;
  }

  protected scalarPlusArrayInternal<T extends NDArray>(c: Scalar, a: T): T {
    const resultValues = new Float32Array(a.size);
    const aValues = a.getValues();
    const cVal = c.get();
    for (let i = 0; i < resultValues.length; ++i) {
      resultValues[i] = cVal + aValues[i];
    }
    return NDArray.make<T>(a.shape, {values: resultValues});
  }

  protected scaledArrayAddInternal<T extends NDArray>(
      c1: Scalar, a: T, c2: Scalar, b: T) {
    const cValues = new Float32Array(a.size);
    const aValues = a.getValues();
    const bValues = b.getValues();
    const c1Val = c1.get();
    const c2Val = c2.get();
    for (let i = 0; i < cValues.length; ++i) {
      cValues[i] = c1Val * aValues[i] + c2Val * bValues[i];
    }
    return NDArray.make<T>(a.shape, {values: cValues});
  }

  protected scalarTimesArrayInternal<T extends NDArray>(c: Scalar, a: T): T {
    const newValues = new Float32Array(a.size);
    const aValues = a.getValues();
    const cVal = c.get();
    for (let i = 0; i < aValues.length; ++i) {
      newValues[i] = cVal * aValues[i];
    }
    return NDArray.make<T>(a.shape, {values: newValues});
  }

  protected scalarMinusArrayInternal<T extends NDArray>(c: Scalar, a: T): T {
    const negA = this.negInternal(a);
    const result = this.scalarPlusArrayInternal(c, negA);

    negA.dispose();

    return result;
  }

  protected arrayMinusScalarInternal<T extends NDArray>(a: T, c: Scalar): T {
    const negC = this.negInternal(c);
    const result = this.scalarPlusArrayInternal(negC, a);

    negC.dispose();

    return result;
  }

  protected negInternal<T extends NDArray>(a: T): T {
    return this.scalarTimesArrayInternal(Scalar.NEG_ONE, a);
  }

  protected addInternal<T extends NDArray>(a: T, b: T): T {
    return this.scaledArrayAddInternal<T>(Scalar.ONE, a, Scalar.ONE, b);
  }

  protected subInternal<T extends NDArray>(a: T, b: T): T {
    return this.scaledArrayAddInternal<T>(Scalar.ONE, a, Scalar.NEG_ONE, b);
  }

  protected matMulInternal(
      a: Array2D, b: Array2D, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Array2D {
    const sharedDim =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];

    const leftDim =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[0] : a.shape[1];
    const rightDim =
        (bOrientation === MatrixOrientation.REGULAR) ? b.shape[1] : b.shape[0];

    const normalGetter = (matrix: Array2D, i: number, j: number) =>
        matrix.get(i, j);
    const transposedGetter = (matrix: Array2D, i: number, j: number) =>
        matrix.get(j, i);

    const aGetter = (aOrientation === MatrixOrientation.REGULAR) ?
        normalGetter :
        transposedGetter;
    const bGetter = (bOrientation === MatrixOrientation.REGULAR) ?
        normalGetter :
        transposedGetter;
    const values = new Float32Array(leftDim * rightDim);
    let index = 0;

    for (let i = 0; i < leftDim; ++i) {
      for (let j = 0; j < rightDim; ++j) {
        let sum = 0;
        for (let k = 0; k < sharedDim; ++k) {
          // TODO: optimize CPU matmul.
          sum += aGetter(a, i, k) * bGetter(b, k, j);
        }
        values[index++] = sum;
      }
    }
    return Array2D.new([leftDim, rightDim], values);
  }

  protected elementWiseMulInternal<T extends NDArray>(a: T, b: T): T {
    const newValues = new Float32Array(a.size);
    const aValues = a.getValues();
    const bValues = b.getValues();
    for (let i = 0; i < aValues.length; ++i) {
      newValues[i] = aValues[i] * bValues[i];
    }
    return NDArray.make<T>(a.shape, {values: newValues});
  }

  protected elementWiseMulBroadcastInternal(a: Array2D, b: Array2D): Array2D {
    const maxRow = Math.max(a.shape[0], b.shape[0]);
    const maxCol = Math.max(a.shape[1], b.shape[1]);

    const values = new Float32Array(maxRow * maxCol);
    let index = 0;
    for (let row = 0; row < maxRow; row++) {
      for (let col = 0; col < maxCol; col++) {
        values[index++] = a.get(row % a.shape[0], col % a.shape[1]) *
            b.get(row % b.shape[0], col % b.shape[1]);
      }
    }
    return Array2D.new([maxRow, maxCol], values);
  }

  protected divideInternal<T extends NDArray>(a: T, b: T): T {
    const newValues = new Float32Array(a.size);
    const aValues = a.getValues();
    const bValues = b.getValues();
    for (let i = 0; i < aValues.length; ++i) {
      newValues[i] = aValues[i] / bValues[i];
    }
    return NDArray.make<T>(a.shape, {values: newValues});
  }

  protected scalarDividedByArrayInternal<T extends NDArray>(c: Scalar, a: T):
      T {
    const newValues = new Float32Array(a.size);
    const aValues = a.getValues();
    const cValue = c.get();
    for (let i = 0; i < aValues.length; ++i) {
      newValues[i] = cValue / aValues[i];
    }
    return NDArray.make<T>(a.shape, {values: newValues});
  }

  protected arrayDividedByScalarInternal<T extends NDArray>(a: T, c: Scalar):
      T {
    const newValues = new Float32Array(a.size);
    const aValues = a.getValues();
    const cValue = c.get();
    for (let i = 0; i < aValues.length; ++i) {
      newValues[i] = aValues[i] / cValue;
    }
    return NDArray.make<T>(a.shape, {values: newValues});
  }

  protected sumInternal(ndarray: NDArray): Scalar {
    let sum = 0;
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      sum += values[i];
    }
    return Scalar.new(sum);
  }

  protected argMinInternal(ndarray: NDArray): Scalar {
    let min = Number.MAX_VALUE;
    let minIndex = -1;
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      if (isNaN(value)) {
        return Scalar.new(NaN);
      }
      if (value < min) {
        min = value;
        minIndex = i;
      }
    }
    return Scalar.new(minIndex);
  }

  protected argMaxInternal(ndarray: NDArray): Scalar {
    let max = Number.NEGATIVE_INFINITY;
    let maxIndex = -1;
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      if (isNaN(value)) {
        return Scalar.new(NaN);
      }
      if (value > max) {
        max = value;
        maxIndex = i;
      }
    }
    return Scalar.new(maxIndex);
  }

  protected argMaxEqualsInternal(x1: NDArray, x2: NDArray): Scalar {
    const argMax1 = this.argMaxInternal(x1).get();
    const argMax2 = this.argMaxInternal(x2).get();
    if (isNaN(argMax1) || isNaN(argMax2)) {
      return Scalar.new(NaN);
    }
    return Scalar.new(+(argMax1 === argMax2));
  }

  protected topKInternal(ndarray: NDArray, k: number):
      {values: Array1D, indices: Array1D} {
    const values = ndarray.getValues();
    const valuesAndIndices: Array<{value: number, index: number}> = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(k);
    const topkIndices = new Float32Array(k);
    for (let i = 0; i < k; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }
    return {values: Array1D.new(topkValues), indices: Array1D.new(topkIndices)};
  }

  protected minInternal(ndarray: NDArray): Scalar {
    const values = ndarray.getValues();
    let min = values[0];
    for (let i = 1; i < values.length; ++i) {
      const value = values[i];
      if (isNaN(value)) {
        return Scalar.new(NaN);
      }
      if (value < min) {
        min = value;
      }
    }
    return Scalar.new(min);
  }

  protected maxInternal(ndarray: NDArray): Scalar {
    const values = ndarray.getValues();
    let max = values[0];
    for (let i = 1; i < values.length; ++i) {
      const value = values[i];
      if (isNaN(value)) {
        return Scalar.new(NaN);
      }
      if (value > max) {
        max = value;
      }
    }
    return Scalar.new(max);
  }

  protected expInternal<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.exp(values[i]);
    }
    return NDArray.make<T>(ndarray.shape, {values: newValues});
  }

  protected logInternal<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.log(value);
    }
    return NDArray.make<T>(ndarray.shape, {values: newValues});
  }

  protected logSumExpInternal(ndarray: NDArray): Scalar {
    const xMax = this.max(ndarray);
    const a = this.arrayMinusScalar(ndarray, xMax);
    const b = this.exp(a);
    const c = this.sum(b);
    const d = this.log(c);
    const result = this.add(xMax, d);

    xMax.dispose();
    a.dispose();
    b.dispose();
    c.dispose();
    d.dispose();

    return result;
  }

  protected reluInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.max(0, values[i]);
    }
    return NDArray.make<T>(ndarray.shape, {values: resultValues});
  }

  protected sigmoidInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = 1 / (1 + Math.exp(-values[i]));
    }
    return NDArray.make<T>(ndarray.shape, {values: resultValues});
  }

  protected tanhInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = util.tanh(values[i]);
    }
    return NDArray.make<T>(ndarray.shape, {values: resultValues});
  }

  protected sinInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sin(values[i]);
    }
    return NDArray.make<T>(ndarray.shape, {values: resultValues});
  }

  protected stepInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      resultValues[i] = value > 0 ? 1 : (value < 0 ? 0 : value);
    }
    return NDArray.make<T>(ndarray.shape, {values: resultValues});
  }

  /**
   * image is of shape [r, c, d1].
   * weights is of shape [F, F, d1, d2].
   */
  protected conv2dInternal(
      x: Array3D, weights: Array4D, biases: Array1D|null, stride: number,
      pad: number): Array3D {
    const [xRows, xCols, inputDepth] = x.shape;
    const fieldSize = weights.shape[0];
    const outputDepth = weights.shape[3];
    const outputShape = conv_util.computeOutputShape3D(
        [xRows, xCols, inputDepth], fieldSize, outputDepth, stride, pad);
    const y = Array3D.zeros(outputShape);
    for (let d2 = 0; d2 < outputDepth; ++d2) {
      for (let yR = 0; yR < y.shape[0]; ++yR) {
        const xRCorner = yR * stride - pad;
        const xRMin = Math.max(0, xRCorner);
        const xRMax = Math.min(xRows, fieldSize + xRCorner);
        for (let yC = 0; yC < y.shape[1]; ++yC) {
          const xCCorner = yC * stride - pad;
          const xCMin = Math.max(0, xCCorner);
          const xCMax = Math.min(xCols, fieldSize + xCCorner);
          let dotProd = 0;
          for (let xR = xRMin; xR < xRMax; ++xR) {
            const wR = xR - xRCorner;
            for (let xC = xCMin; xC < xCMax; ++xC) {
              const wC = xC - xCCorner;
              for (let d1 = 0; d1 < inputDepth; ++d1) {
                const pixel = x.get(xR, xC, d1);
                const weight = weights.get(wR, wC, d1, d2);
                dotProd += pixel * weight;
              }
            }
          }
          const bias = (biases != null) ? biases.get(d2) : 0;
          y.set(dotProd + bias, yR, yC, d2);
        }
      }
    }
    return y;
  }

  protected conv2dBackPropInternal(
      x: Array3D, dy: Array3D, weights: Array4D, stride: number,
      pad: number): {dx: Array3D, dw: Array4D, db: Array1D} {
    const fSize = weights.shape[0];
    const dw = this.conv2dDerWeights(x, dy, fSize, stride, pad);
    const db = this.conv2dDerBias(dy);
    const dx = this.conv2dTransposeInternal(dy, weights, null, stride, pad);
    return {dx, db, dw};
  }

  /**
   * image is of shape [r, c, d1].
   * weights is of shape [F, F, d1, d2].
   */
  protected conv2dTransposeInternal(
      x: Array3D, weights: Array4D, biases: Array1D|null, origStride: number,
      origPad: number): Array3D {
    const fSize = weights.shape[0];
    const pad = fSize - 1 - origPad;
    const origInputDepth = weights.shape[2];
    const origOutputDepth = weights.shape[3];
    const xRows = x.shape[0];
    const xCols = x.shape[1];

    // Dilate the input.
    const xRowsDilated = (xRows - 1) * origStride + 1;
    const xColsDilated = (xCols - 1) * origStride + 1;

    const outputShape = conv_util.computeOutputShape3D(
        [xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1,
        pad);
    const y = Array3D.zeros(outputShape);
    for (let d2 = 0; d2 < origInputDepth; ++d2) {
      for (let yR = 0; yR < y.shape[0]; ++yR) {
        const xRCorner = yR - pad;
        const xRMin = Math.max(0, Math.ceil(xRCorner / origStride));
        const xRMax = Math.min(xRows, (fSize + xRCorner) / origStride);

        for (let yC = 0; yC < y.shape[1]; ++yC) {
          const xCCorner = yC - pad;
          const xCMin = Math.max(0, Math.ceil(xCCorner / origStride));
          const xCMax = Math.min(xCols, (fSize + xCCorner) / origStride);

          let dotProd = 0;
          for (let xR = xRMin; xR < xRMax; ++xR) {
            const wR = xR * origStride - xRCorner;

            for (let xC = xCMin; xC < xCMax; ++xC) {
              const wC = xC * origStride - xCCorner;

              for (let d1 = 0; d1 < origOutputDepth; ++d1) {
                const pixel = x.get(xR, xC, d1);
                const weight =
                    weights.get(fSize - 1 - wR, fSize - 1 - wC, d2, d1);
                dotProd += pixel * weight;
              }
            }
          }
          const bias = biases != null ? biases.get(d2) : 0;
          y.set(dotProd + bias, yR, yC, d2);
        }
      }
    }
    return y;
  }

  /**
   * image is of shape [r, c, d1].
   * weights is of shape [F, F, d1, d2].
   */
  protected conv2dTransposeShaderLike(
      x: Array3D, origWeights: Array4D, origStride: number,
      origPad: number): Array3D {
    const fSize = origWeights.shape[0];
    const pad = fSize - 1 - origPad;
    const origInputDepth = origWeights.shape[2];
    const origOutputDepth = origWeights.shape[3];
    const xRows = x.shape[0];
    const xCols = x.shape[1];

    // Dilate the input.
    const xRowsDilated = (xRows - 1) * origStride + 1;
    const xColsDilated = (xCols - 1) * origStride + 1;

    const outputShape = conv_util.computeOutputShape3D(
        [xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1,
        pad);
    const y = Array3D.zeros(outputShape);

    for (let d2 = 0; d2 < origInputDepth; ++d2) {
      for (let yR = 0; yR < y.shape[0]; ++yR) {
        for (let yC = 0; yC < y.shape[1]; ++yC) {
          // Shader code begins.
          const xRCorner = yR - pad;
          const xCCorner = yC - pad;
          let dotProd = 0;
          for (let wR = 0; wR < fSize; ++wR) {
            const xR = (xRCorner + wR) / origStride;
            if (xR < 0 || xR >= xRows || Math.floor(xR) !== xR) {
              continue;
            }
            for (let wC = 0; wC < fSize; ++wC) {
              const xC = (xCCorner + wC) / origStride;
              if (xC < 0 || xC >= xCols || Math.floor(xC) !== xC) {
                continue;
              }
              for (let d1 = 0; d1 < origOutputDepth; ++d1) {
                const pixel = x.get(xR, xC, d1);
                const weight =
                    origWeights.get(fSize - 1 - wR, fSize - 1 - wC, d2, d1);
                dotProd += pixel * weight;
              }
            }
          }
          y.set(dotProd, yR, yC, d2);
        }
      }
    }
    return y;
  }

  conv2dDerWeights(
      x: Array3D, dY: Array3D, fSize: number, stride: number,
      zeroPad: number): Array4D {
    const inputDepth = x.shape[2];
    const outputDepth = dY.shape[2];
    const weightsShape =
        conv_util.computeWeightsShape4D(inputDepth, outputDepth, fSize);
    const dW = Array4D.zeros(weightsShape);

    const yNumRows = dY.shape[0];
    const yNumCols = dY.shape[1];
    const xNumRows = x.shape[0];
    const xNumCols = x.shape[1];

    for (let wR = 0; wR < fSize; ++wR) {
      const yRMin = Math.max(0, Math.ceil((zeroPad - wR) / stride));
      const yRMax = Math.min(yNumRows, (xNumRows + zeroPad - wR) / stride);

      for (let wC = 0; wC < fSize; ++wC) {
        const yCMin = Math.max(0, Math.ceil((zeroPad - wC) / stride));
        const yCMax = Math.min(yNumCols, (xNumCols + zeroPad - wC) / stride);

        for (let d1 = 0; d1 < inputDepth; ++d1) {
          for (let d2 = 0; d2 < outputDepth; ++d2) {
            // Need to convolve.
            let dotProd = 0;
            for (let yR = yRMin; yR < yRMax; ++yR) {
              const xR = wR + yR * stride - zeroPad;
              for (let yC = yCMin; yC < yCMax; ++yC) {
                const xC = wC + yC * stride - zeroPad;
                dotProd += x.get(xR, xC, d1) * dY.get(yR, yC, d2);
              }
            }
            dW.set(dotProd, wR, wC, d1, d2);
          }
        }
      }
    }
    return dW;
  }

  conv2dDerBias(dY: Array3D): Array1D {
    const outputDepth = dY.shape[2];
    const numRows = dY.shape[0];
    const numCols = dY.shape[1];
    const values = new Float32Array(outputDepth);
    for (let d2 = 0; d2 < outputDepth; ++d2) {
      let sum = 0;
      for (let r = 0; r < numRows; ++r) {
        for (let c = 0; c < numCols; ++c) {
          sum += dY.get(r, c, d2);
        }
      }
      values[d2] = sum;
    }
    return Array1D.new(values);
  }

  protected switchDimInternal<T extends NDArray>(t: T, newDim: number[]): T {
    const newShape: number[] = new Array(t.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = t.shape[newDim[i]];
    }
    const resultValues = new Float32Array(t.size);
    const values = t.getValues();
    const result = NDArray.make<T>(newShape, {values: resultValues});
    for (let i = 0; i < t.size; ++i) {
      const loc = t.indexToLoc(i);

      // Permute location.
      const newLoc: number[] = new Array(loc.length);
      for (let i = 0; i < newLoc.length; i++) {
        newLoc[i] = loc[newDim[i]];
      }

      const newIndex = result.locToIndex(newLoc);
      resultValues[newIndex] = values[i];
    }
    return result;
  }

  private pool(
      x: Array3D, fSize: number, stride: number, pad: number,
      poolType: 'max'|'min'|'avg') {
    const [xRows, xCols, depth] = x.shape;
    const outputShape = conv_util.computeOutputShape3D(
        [xRows, xCols, depth], fSize, depth, stride, pad);
    const y = Array3D.zeros(outputShape);
    for (let d = 0; d < depth; ++d) {
      for (let yR = 0; yR < y.shape[0]; ++yR) {
        const xRCorner = yR * stride - pad;
        const xRMin = Math.max(0, xRCorner);
        const xRMax = Math.min(xRows, fSize + xRCorner);
        for (let yC = 0; yC < y.shape[1]; ++yC) {
          const xCCorner = yC * stride - pad;
          const xCMin = Math.max(0, xCCorner);
          const xCMax = Math.min(xCols, fSize + xCCorner);


          let minMaxValue =
              (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                                    Number.POSITIVE_INFINITY);
          let avgValue = 0;

          for (let xR = xRMin; xR < xRMax; ++xR) {
            for (let xC = xCMin; xC < xCMax; ++xC) {
              const pixel = x.get(xR, xC, d);
              if (isNaN(pixel)) {
                minMaxValue = NaN;
                avgValue = NaN;
                break;
              }
              if ((poolType === 'max' && pixel > minMaxValue) ||
                  (poolType === 'min' && pixel < minMaxValue)) {
                minMaxValue = pixel;
              } else if (poolType === 'avg') {
                avgValue += pixel / (fSize * fSize);
              }
            }
            if (isNaN(minMaxValue)) {
              break;
            }
          }
          y.set(poolType === 'avg' ? avgValue : minMaxValue, yR, yC, d);
        }
      }
    }
    return y;
  }

  protected maxPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    return this.pool(x, fSize, stride, pad, 'max');
  }

  maxPoolPositions(x: Array3D, fSize: number, stride: number, pad: number) {
    const [xRows, xCols, depth] = x.shape;
    const outputShape =
        conv_util.computeOutputShape3D(x.shape, fSize, depth, stride, pad);
    const maxPositions = Array3D.zeros(outputShape);
    for (let d = 0; d < depth; ++d) {
      for (let yR = 0; yR < outputShape[0]; ++yR) {
        const xRCorner = yR * stride - pad;
        const xRMin = Math.max(0, xRCorner);
        const xRMax = Math.min(xRows, fSize + xRCorner);
        for (let yC = 0; yC < outputShape[1]; ++yC) {
          const xCCorner = yC * stride - pad;
          const xCMin = Math.max(0, xCCorner);
          const xCMax = Math.min(xCols, fSize + xCCorner);
          let maxValue = Number.NEGATIVE_INFINITY;
          let maxPosition = -1;
          for (let xR = xRMin; xR < xRMax; ++xR) {
            const wR = xR - xRCorner;
            for (let xC = xCMin; xC < xCMax; ++xC) {
              const wC = xC - xCCorner;
              const pixel = x.get(xR, xC, d);
              if (pixel > maxValue) {
                maxValue = pixel;
                maxPosition = wR * fSize + wC;
              }
            }
          }
          maxPositions.set(maxPosition, yR, yC, d);
        }
      }
    }
    return maxPositions;
  }

  protected maxPoolBackpropInternal(
      dy: Array3D, x: Array3D, fSize: number, origStride: number,
      origPad: number): Array3D {
    const maxPositions = this.maxPoolPositions(x, fSize, origStride, origPad);
    const pad = fSize - 1 - origPad;
    const [dyRows, dyCols, depth] = dy.shape;

    // Dilate the input.
    const dyRowsDilated = (dyRows - 1) * origStride + 1;
    const dxColsDilated = (dyCols - 1) * origStride + 1;

    const outputShape = conv_util.computeOutputShape3D(
        [dyRowsDilated, dxColsDilated, depth], fSize, depth, 1, pad);
    const dx = Array3D.zeros(outputShape);

    for (let d = 0; d < depth; ++d) {
      for (let dxR = 0; dxR < dx.shape[0]; ++dxR) {
        for (let dxC = 0; dxC < dx.shape[1]; ++dxC) {
          // Shader code begins.
          const dyRCorner = dxR - pad;
          const dyCCorner = dxC - pad;
          let dotProd = 0;
          for (let wR = 0; wR < fSize; ++wR) {
            const dyR = (dyRCorner + wR) / origStride;
            if (dyR < 0 || dyR >= dyRows || Math.floor(dyR) !== dyR) {
              continue;
            }
            for (let wC = 0; wC < fSize; ++wC) {
              const dyC = (dyCCorner + wC) / origStride;
              if (dyC < 0 || dyC >= dyCols || Math.floor(dyC) !== dyC) {
                continue;
              }
              const maxPos = fSize * fSize - 1 - maxPositions.get(dyR, dyC, d);
              const curPos = wR * fSize + wC;

              const mask = maxPos === curPos ? 1 : 0;
              if (mask === 0) {
                continue;
              }

              const pixel = dy.get(dyR, dyC, d);
              dotProd += pixel * mask;
            }
          }
          dx.set(dotProd, dxR, dxC, d);
        }
      }
    }
    return dx;
  }

  protected minPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    return this.pool(x, fSize, stride, pad, 'min');
  }

  protected avgPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    return this.pool(x, fSize, stride, pad, 'avg');
  }

  protected resizeBilinear3DInternal(
      x: Array3D, newShape2D: [number, number],
      alignCorners: boolean): Array3D {
    const output = Array3D.zeros([newShape2D[0], newShape2D[1], x.shape[2]]);

    const effectiveInputSize =
        alignCorners ? [x.shape[0] - 1, x.shape[1] - 1, x.shape[2]] : x.shape;
    const effectiveOutputSize = alignCorners ?
        [output.shape[0] - 1, output.shape[1] - 1, output.shape[2]] :
        output.shape;
    for (let r = 0; r < output.shape[0]; r++) {
      for (let c = 0; c < output.shape[1]; c++) {
        for (let d = 0; d < output.shape[2]; d++) {
          // Begin shader.

          // Compute the fractional index of the source.
          const sourceFracRow =
              (effectiveInputSize[0]) * r / (effectiveOutputSize[0]);
          const sourceFracCol =
              (effectiveInputSize[1]) * c / (effectiveOutputSize[1]);

          const sourceRowFloor = Math.floor(sourceFracRow);
          const sourceRowCeil =
              Math.min(x.shape[0] - 1, Math.ceil(sourceFracRow));
          const sourceColFloor = Math.floor(sourceFracCol);
          const sourceColCeil =
              Math.min(x.shape[1] - 1, Math.ceil(sourceFracCol));

          const topLeft = x.get(sourceRowFloor, sourceColFloor, d);
          const bottomLeft = x.get(sourceRowCeil, sourceColFloor, d);
          const topRight = x.get(sourceRowFloor, sourceColCeil, d);
          const bottomRight = x.get(sourceRowCeil, sourceColCeil, d);

          const rowFrac = sourceFracRow - sourceRowFloor;
          const colFrac = sourceFracCol - sourceColFloor;

          const top = topLeft + (topRight - topLeft) * colFrac;
          const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
          const newValue = top + (bottom - top) * rowFrac;

          output.set(newValue, r, c, d);
        }
      }
    }

    return output;
  }

  protected batchNormalization3DInternal(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon = .001, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D {
    const xValues = x.getValues();
    const meanValues = mean.getValues();
    const varianceValues = variance.getValues();
    const scaleValues = scale ? scale.getValues() : new Float32Array([1]);
    const offsetValues = offset ? offset.getValues() : new Float32Array([0]);
    const outValues = new Float32Array(xValues.length);

    for (let i = 0; i < xValues.length; i++) {
      outValues[i] = offsetValues[i % offsetValues.length] +
          (xValues[i] - meanValues[i % meanValues.length]) *
              scaleValues[i % scaleValues.length] /
              Math.sqrt(
                  varianceValues[i % varianceValues.length] + varianceEpsilon);
    }
    return NDArray.make<Array3D>(x.shape, {values: outValues});
  }
}
