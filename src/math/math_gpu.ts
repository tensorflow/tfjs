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

import * as util from '../util';

import * as concat3d_util from './concat3d_util';
import * as conv_util from './conv_util';
import {MatrixOrientation, NDArrayMath} from './math';
import * as ndarray from './ndarray';
import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from './ndarray';
import * as addscaledmat_gpu from './webgl/addscaledmat_gpu';
import * as addsubmuldiv_gpu from './webgl/addsubmuldiv_gpu';
import {OperandType} from './webgl/addsubmuldiv_gpu';
import {ArgMaxEqualsProgram} from './webgl/argmaxequals_gpu';
import {ArgMinMaxProgram} from './webgl/argminmax_gpu';
import * as avg_pool_gpu from './webgl/avg_pool_gpu';
import * as batchnorm_gpu from './webgl/batchnorm_gpu';
import * as concat3d_gpu from './webgl/concat3d_gpu';
import * as conv_backprop_gpu from './webgl/conv_backprop_gpu';
import * as conv_gpu from './webgl/conv_gpu';
import * as copy_gpu from './webgl/copy_gpu';
import {GPGPUContext} from './webgl/gpgpu_context';
import {GPGPUProgram, GPGPUBinary} from './webgl/gpgpu_math';
import * as gpgpu_math from './webgl/gpgpu_math';
import * as gpgpu_util from './webgl/gpgpu_util';
import {LogSumExpProgram} from './webgl/logsumexp_gpu';
import * as max_pool_backprop_gpu from './webgl/max_pool_backprop_gpu';
import * as max_pool_gpu from './webgl/max_pool_gpu';
import * as min_pool_gpu from './webgl/min_pool_gpu';
import {MinMaxProgram} from './webgl/minmax_gpu';
import {MatMulProgram} from './webgl/mulmat_gpu';
import * as pool_gpu from './webgl/pool_gpu';
import {ReduceSumProgram} from './webgl/reducesum_gpu';
import * as reshape_gpu from './webgl/reshape_gpu';
import * as resize_bilinear_gpu from './webgl/resize_bilinear_gpu';
import * as shader_compiler from './webgl/shader_compiler';
import {TextureManager} from './webgl/texture_manager';
import * as webgl_util from './webgl/webgl_util';
import {UnaryOpProgram, UnaryOp} from './webgl/unaryop_gpu';

const BATCHNORM_PROG = 'batchnorm';

const COPY_PROG = 'copy';
const CONCAT_PROG = 'concat';

// Matrix algebra.
const ADD_SCALED_MAT_PROG = 'addscaledmat';

// Element-wise ops.
const RESHAPE_PROG = 'reshape';
const ADD_SUM_MUL_DIV_PROG = 'addsummuldiv';

// Convolution.
const CONV2D_PROG = 'conv';
const CONV2D_TRANSPOSE_PROG = 'conv_transpose';
const CONV2D_DERW_PROG = 'conv_derw';
const CONV2D_DERB_PROG = 'conv_derb';
const MAX_POOL_PROG = 'maxpool';
const MAX_POOL_POSITIONS_PROG = 'maxpool_posn';
const MAX_POOL_BACKPROP_PROG = 'maxpool_backprop';
const MIN_POOL_PROG = 'minpool';
const AVG_POOL_PROG = 'avgpool';

const RESIZE_BILINEAR_PROG = 'resizebilin';

function makeCopyProgramName(
    sourceShapeRowCol: [number, number], sourceSizeRowCol: [number, number],
    destSizeRowCol: [number, number]): string {
  const shapeName = `${sourceShapeRowCol[0]}_${sourceShapeRowCol[1]}`;
  const srcSizeName = `${sourceSizeRowCol[0]}_${sourceSizeRowCol[1]}`;
  const dstSizeName = `${destSizeRowCol[0]}_${destSizeRowCol[1]}`;
  return `${COPY_PROG}_${shapeName}_${srcSizeName}_${dstSizeName}`;
}

export class NDArrayMathGPU extends NDArrayMath {
  private gpgpu: GPGPUContext;
  private textureManager: TextureManager;
  private programCache: {[key: string]: WebGLProgram} = {};
  private binaryCache: {[key: string]: GPGPUBinary<NDArray, NDArray>} = {};
  private gpgpuCreatedLocally: boolean;

  constructor(gpgpu?: GPGPUContext, safeMode = true) {
    super(safeMode);
    if (gpgpu == null) {
      const gl = gpgpu_util.createWebGLContext();
      this.gpgpu = new GPGPUContext(gl);
      this.gpgpuCreatedLocally = true;
    } else {
      this.gpgpu = gpgpu;
      this.gpgpuCreatedLocally = false;
    }

    this.textureManager = new TextureManager(this.gpgpu);

    ndarray.initializeGPU(this.gpgpu, this.textureManager);
  }

  getGPGPUContext(): GPGPUContext {
    return this.gpgpu;
  }

  protected cloneInternal<T extends NDArray>(ndarray: T): T {
    const textureShapeRC = ndarray.getTextureShapeRC();
    const program = this.getAndSaveProgram(
        makeCopyProgramName(textureShapeRC, textureShapeRC, textureShapeRC),
        () => copy_gpu.getFragmentShaderSource(
            textureShapeRC, textureShapeRC, textureShapeRC));

    const resultTexture = this.textureManager.acquireTexture(textureShapeRC);

    copy_gpu.copy(
        this.gpgpu, program, ndarray.getTexture(), textureShapeRC, [0, 0],
        textureShapeRC, resultTexture, textureShapeRC, [0, 0], textureShapeRC);

    return NDArray.make<T>(
        ndarray.shape, {texture: resultTexture, textureShapeRC});
  }

  protected reshapeInternal<T1 extends NDArray, T2 extends NDArray>(
      ndarray: T1, newShape: number[]): T2 {
    let newTexShape: [number, number];

    switch (newShape.length) {
      case 0:
        newTexShape = [1, 1];
        break;
      case 1:
        newTexShape = [newShape[0], 1];
        break;
      case 2:
        newTexShape = [newShape[0], newShape[1]];
        break;
      case 3:
        newTexShape = [newShape[0], newShape[1] * newShape[2]];
        break;
      default:
        throw Error(
            `Reshapes into ${newShape.length}-dim ndarray is not yet ` +
            `supported on GPU`);
    }

    const actualTexShape = ndarray.getTextureShapeRC(newTexShape);
    let clonedArray: T1;
    if (!util.arraysEqual(actualTexShape, newTexShape)) {
      clonedArray = this.reshapeTexture(ndarray, newTexShape);
    } else {
      clonedArray = this.cloneInternal(ndarray);
    }
    return clonedArray.reshape<T2>(newShape);
  }

  protected slice2DInternal(
      input: Array2D, beginRowCol: [number, number],
      sizeRowCol: [number, number]): Array2D {
    const result = NDArray.make<Array2D>(sizeRowCol, {
      texture: this.textureManager.acquireTexture(sizeRowCol),
      textureShapeRC: sizeRowCol
    });
    this.copy2DInternal(
        input, beginRowCol, sizeRowCol, result, [0, 0], sizeRowCol);
    return result;
  }

  protected copy2DInternal(
      source: Array2D, sourceBeginRowCol: [number, number],
      sourceSizeRowCol: [number, number], dest: Array2D,
      destBeginRowCol: [number, number],
      destSizeRowCol: [number, number]): void {
    const sourceShapeRC = source.getTextureShapeRC();
    const destShapeRC = dest.getTextureShapeRC();
    const program = this.getAndSaveProgram(
        makeCopyProgramName(sourceShapeRC, sourceSizeRowCol, destSizeRowCol),
        () => copy_gpu.getFragmentShaderSource(
            sourceShapeRC, sourceSizeRowCol, destSizeRowCol));

    copy_gpu.copy(
        this.gpgpu, program, source.getTexture(), sourceShapeRC,
        sourceBeginRowCol, sourceSizeRowCol, dest.getTexture(), destShapeRC,
        destBeginRowCol, destSizeRowCol);
  }

  protected concat3DInternal(x1: Array3D, x2: Array3D, axis: number): Array3D {
    const x1TexShapeRC: [number, number] =
        conv_util.computeTexShapeFrom3D(x1.shape);
    const x2TexShapeRC: [number, number] =
        conv_util.computeTexShapeFrom3D(x2.shape);

    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    const actualX1TexShape = x1.getTextureShapeRC(x1TexShapeRC);
    let cleanupX1 = false;
    if (!util.arraysEqual(actualX1TexShape, x1TexShapeRC)) {
      x1 = this.reshapeTexture(x1, x1TexShapeRC);
      cleanupX1 = true;
    }
    const actualX2TexShape = x2.getTextureShapeRC(x2TexShapeRC);
    let cleanupX2 = false;
    if (!util.arraysEqual(actualX2TexShape, x2TexShapeRC)) {
      x2 = this.reshapeTexture(x2, x2TexShapeRC);
      cleanupX2 = true;
    }

    const resultShapeRCD =
        concat3d_util.computeConcat3DOutputShape(x1.shape, x2.shape, axis);

    const program = this.getAndSaveProgram(
        `${CONCAT_PROG}_${x1.shape}_${x2.shape}_${axis}`,
        () => concat3d_gpu.getFragmentShaderSource(
            x1.shape, x2.shape, resultShapeRCD, axis));

    const resultTexShape = conv_util.computeTexShapeFrom3D(resultShapeRCD);
    const resultTex = this.textureManager.acquireTexture(resultTexShape);

    concat3d_gpu.concat3D(
        this.gpgpu, program, x1.getTexture(), x2.getTexture(), resultTex,
        resultTexShape);

    if (cleanupX1) {
      x1.dispose();
    }

    if (cleanupX2) {
      x2.dispose();
    }

    return NDArray.make<Array3D>(
        resultShapeRCD, {texture: resultTex, textureShapeRC: resultTexShape});
  }

  protected scalarPlusArrayInternal<T extends NDArray>(c: Scalar, a: T): T {
    return this.addSubMulDiv(
        c, a, a.shape, OperandType.SCALAR, '+', OperandType.MATRIX) as T;
  }

  protected arrayMinusScalarInternal<T extends NDArray>(a: T, c: Scalar): T {
    return this.addSubMulDiv(
        a, c, a.shape, OperandType.MATRIX, '-', OperandType.SCALAR) as T;
  }

  protected scalarMinusArrayInternal<T extends NDArray>(c: Scalar, a: T): T {
    return this.addSubMulDiv(
        c, a, a.shape, OperandType.SCALAR, '-', OperandType.MATRIX) as T;
  }

  protected scaledArrayAddInternal<T extends NDArray>(
      c1: Scalar, a: T, c2: Scalar, b: T) {
    let cleanupB = false;
    if (!this.doGPUShapesMatch(a, b)) {
      b = this.reshapeTexture(b, a.getTextureShapeRC());
      cleanupB = true;
    }

    const program = this.getAndSaveProgram(
        ADD_SCALED_MAT_PROG, () => addscaledmat_gpu.getFragmentShaderSource());

    const textureShapeRC = a.getTextureShapeRC();
    const resultTexture = this.textureManager.acquireTexture(textureShapeRC);

    addscaledmat_gpu.addScaledMatrices(
        this.gpgpu, program, a.getTexture(), b.getTexture(), textureShapeRC[0],
        textureShapeRC[1], c1.getTexture(), c2.getTexture(), resultTexture);

    if (cleanupB) {
      b.dispose();
    }
    // Bring the result back to the original shape.
    return NDArray.make<T>(a.shape, {texture: resultTexture, textureShapeRC});
  }

  protected scalarTimesArrayInternal<T extends NDArray>(c: Scalar, a: T): T {
    return this.addSubMulDiv(
        c, a, a.shape, OperandType.SCALAR, '*', OperandType.MATRIX) as T;
  }

  protected negInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, UnaryOp.NEG);
    return this.compileAndRun<T, T>(program, [a]);
  }

  private makeOutputArray<T extends NDArray>(shape: number[]): T {
    const textureShapeRC =
        webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, shape);
    const texture = this.textureManager.acquireTexture(textureShapeRC);
    return NDArray.make<T>(shape, {texture, textureShapeRC});
  }

  private compileAndRun<T extends NDArray, K extends NDArray>(
      program: GPGPUProgram, inputs: T[]): K {
    const output = this.makeOutputArray<K>(program.outputShape);
    const key = gpgpu_math.makeShaderKey(program, inputs, output);
    const binary = this.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(this.gpgpu, program, inputs, output);
    });
    gpgpu_math.runProgram(binary, inputs, output);
    return output;
  }

  private reshapeTexture<T extends NDArray>(a: T, newTextureShape: [
    number, number
  ]): T {
    const aTexShape = a.getTextureShapeRC();

    const program = this.getAndSaveProgram(
        RESHAPE_PROG, () => reshape_gpu.getFragmentShaderSource());

    const resultTexture = this.textureManager.acquireTexture(newTextureShape);
    reshape_gpu.reshape(
        this.gpgpu, program, a.getTexture(), aTexShape[0], aTexShape[1],
        resultTexture, newTextureShape[0], newTextureShape[1]);

    return NDArray.make<T>(
        a.shape, {texture: resultTexture, textureShapeRC: newTextureShape});
  }

  protected matMulInternal(
      a: Array2D, b: Array2D, aOrientation: MatrixOrientation,
      bOrientation: MatrixOrientation): Array2D {
    const program = new MatMulProgram(a.shape, b.shape, aOrientation,
        bOrientation);
    return this.compileAndRun<Array2D, Array2D>(program, [a, b]);
  }

  protected elementWiseMulInternal<T extends NDArray>(a: T, b: T): T {
    return this.addSubMulDiv(
        a, b, a.shape, OperandType.MATRIX, '*', OperandType.MATRIX) as T;
  }

  protected elementWiseMulBroadcastInternal(a: Array2D, b: Array2D): Array2D {
    throw new Error('Not yet implemented!');
  }

  protected batchNormalization3DInternal(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon: number, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D {
    const xTexShape = x.getTextureShapeRC();

    let cleanupMean = false;
    const preferredMeanTexShape: [number, number] =
        mean.rank === 1 ? [1, mean.size] : xTexShape;
    let meanTexShape = mean.getTextureShapeRC(preferredMeanTexShape);
    if (!util.arraysEqual(meanTexShape, preferredMeanTexShape)) {
      mean = this.reshapeTexture(mean, preferredMeanTexShape);
      meanTexShape = preferredMeanTexShape;
      cleanupMean = true;
    }

    let cleanupVariance = false;
    const preferredVarianceTexShape: [number, number] =
        variance.rank === 1 ? [1, variance.size] : xTexShape;
    let varianceTexShape = variance.getTextureShapeRC(preferredMeanTexShape);
    if (!util.arraysEqual(varianceTexShape, preferredVarianceTexShape)) {
      variance = this.reshapeTexture(variance, preferredVarianceTexShape);
      varianceTexShape = preferredVarianceTexShape;
      cleanupVariance = true;
    }

    let scaleTexShape: [number, number]|null = null;
    let cleanupScale = false;
    if (scale != null) {
      const preferredScaleTexShape: [number, number] =
          scale.rank === 1 ? [1, scale.size] : xTexShape;

      scaleTexShape = scale.getTextureShapeRC(preferredScaleTexShape);
      if (!util.arraysEqual(scaleTexShape, preferredScaleTexShape)) {
        scale = this.reshapeTexture(scale, preferredScaleTexShape);
        scaleTexShape = preferredScaleTexShape;
        cleanupScale = true;
      }
    }

    let offsetTexShape: [number, number]|null = null;
    let cleanupOffset = false;
    if (offset != null) {
      const preferredOffsetTexShape: [number, number] =
          offset.rank === 1 ? [1, offset.size] : xTexShape;

      offsetTexShape = offset.getTextureShapeRC(preferredOffsetTexShape);
      if (!util.arraysEqual(offsetTexShape, preferredOffsetTexShape)) {
        offset = this.reshapeTexture(offset, preferredOffsetTexShape);
        offsetTexShape = preferredOffsetTexShape;
        cleanupOffset = true;
      }
    }

    const resultTexShape: [number, number] = x.getTextureShapeRC();

    const program = this.getAndSaveProgram(
        `${BATCHNORM_PROG}_${xTexShape}_${meanTexShape}_${varianceTexShape}_` +
            `${scaleTexShape!}_${offsetTexShape!}_${varianceEpsilon}`,
        () => batchnorm_gpu.getFragmentShaderSource(
            xTexShape, meanTexShape, varianceTexShape, offsetTexShape,
            scaleTexShape, varianceEpsilon));

    const resultTexture = this.textureManager.acquireTexture(resultTexShape);

    batchnorm_gpu.batchNormalization(
        this.gpgpu, program, x.getTexture(), xTexShape, mean.getTexture(),
        meanTexShape, variance.getTexture(), varianceTexShape,
        offset != null ? offset.getTexture() : null,
        offset != null ? offsetTexShape : null,
        scale != null ? scale.getTexture() : null,
        scale != null ? scaleTexShape : null, resultTexture, resultTexShape);

    if (cleanupMean) {
      mean.dispose();
    }
    if (cleanupVariance) {
      variance.dispose();
    }
    if (cleanupScale) {
      scale!.dispose();
    }
    if (cleanupOffset) {
      offset!.dispose();
    }

    return NDArray.make<Array3D>(
        x.shape, {texture: resultTexture, textureShapeRC: resultTexShape});
  }

  protected switchDimInternal<T extends NDArray>(a: T, newDim: number[]): T {
    throw new Error('Not yet implemented!');
  }

  protected sumInternal(a: NDArray): Scalar {
    const program = new ReduceSumProgram(a.size);
    return this.compileAndRun(program, [a]);
  }

  protected argMinInternal(a: NDArray): Scalar {
    const program = new ArgMinMaxProgram(a.size, 'min');
    return this.compileAndRun(program, [a]);
  }

  protected argMaxInternal(a: NDArray): Scalar {
    const program = new ArgMinMaxProgram(a.size, 'max');
    return this.compileAndRun(program, [a]);
  }

  protected argMaxEqualsInternal(x1: NDArray, x2: NDArray): Scalar {
    const program = new ArgMaxEqualsProgram(x1.size, x2.size);
    return this.compileAndRun(program, [x1, x2]);
  }

  protected topKInternal(ndarray: NDArray, k: number):
      {values: Array1D, indices: Array1D} {
    throw new Error('topK GPU not yet implemented!');
  }

  protected minInternal(a: NDArray): Scalar {
    const program = new MinMaxProgram(a.size, 'min');
    return this.compileAndRun(program, [a]);
  }

  protected maxInternal(a: NDArray): Scalar {
    const program = new MinMaxProgram(a.size, 'max');
    return this.compileAndRun(program, [a]);
  }

  protected divideInternal<T extends NDArray>(a: T, b: T): T {
    return this.addSubMulDiv(
        a, b, a.shape, OperandType.MATRIX, '/', OperandType.MATRIX) as T;
  }

  protected scalarDividedByArrayInternal<T extends NDArray>(c: Scalar, a: T):
      T {
    return this.addSubMulDiv(
               c, a, a.shape, OperandType.SCALAR, '/', OperandType.MATRIX) as T;
  }

  protected arrayDividedByScalarInternal<T extends NDArray>(a: T, c: Scalar):
      T {
    return this.addSubMulDiv(
               a, c, a.shape, OperandType.MATRIX, '/', OperandType.SCALAR) as T;
  }

  protected addInternal<T extends NDArray>(a: T, b: T): T {
    return this.addSubMulDiv(
        a, b, a.shape, OperandType.MATRIX, '+', OperandType.MATRIX) as T;
  }

  protected subInternal<T extends NDArray>(a: T, b: T): T {
    return this.addSubMulDiv(
        a, b, a.shape, OperandType.MATRIX, '-', OperandType.MATRIX) as T;
  }

  protected logSumExpInternal(a: NDArray): Scalar {
    const program = new LogSumExpProgram(a.size);
    return this.compileAndRun(program, [a]);
  }

  protected expInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, UnaryOp.EXP);
    return this.compileAndRun<T, T>(program, [a]);
  }

  protected logInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, UnaryOp.LOG);
    return this.compileAndRun<T, T>(program, [a]);
  }

  protected reluInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, UnaryOp.RELU);
    return this.compileAndRun<T, T>(program, [a]);
  }

  protected sigmoidInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, UnaryOp.SIGMOID);
    return this.compileAndRun<T, T>(program, [a]);
  }

  protected tanhInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, UnaryOp.TANH);
    return this.compileAndRun<T, T>(program, [a]);
  }

  protected sinInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, UnaryOp.SIN);
    return this.compileAndRun<T, T>(program, [a]);
  }

  protected stepInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, UnaryOp.STEP);
    return this.compileAndRun<T, T>(program, [a]);
  }

  protected conv2dInternal(
      x: Array3D, weights: Array4D, biases: Array1D|null, stride: number,
      zeroPad: number): Array3D {
    const fieldSize = weights.shape[0];
    const inputDepth = weights.shape[2];
    const outputDepth = weights.shape[3];
    const progKey = [
      CONV2D_PROG, x.shape, outputDepth, fieldSize, stride, biases != null
    ].join('_');
    const program = this.getAndSaveProgram(progKey, () => {
      return conv_gpu.getFragmentShaderSource(
          x.shape, outputDepth, fieldSize, stride, zeroPad, biases != null);
    });

    const xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
    const wTexShape =
        conv_util.computeWeightsTexShape(inputDepth, outputDepth, fieldSize);
    const biasTexShape = conv_util.computeBiasesTexShape(outputDepth);

    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    const actualXTexShape = x.getTextureShapeRC(xTexShape);
    let cleanupX = false;
    if (!util.arraysEqual(actualXTexShape, xTexShape)) {
      x = this.reshapeTexture(x, xTexShape);
      cleanupX = true;
    }

    let cleanupW = false;
    const actualWTexShape = weights.getTextureShapeRC(wTexShape);
    if (!util.arraysEqual(actualWTexShape, wTexShape)) {
      weights = this.reshapeTexture(weights, wTexShape);
      cleanupW = true;
    }

    let cleanupB = false;
    if (biases != null) {
      const actualBTexShape = biases.getTextureShapeRC(biasTexShape);
      if (!util.arraysEqual(actualBTexShape, biasTexShape)) {
        biases = this.reshapeTexture(biases, biasTexShape);
        cleanupB = true;
      }
    }

    const resultShape = conv_util.computeOutputShape3D(
        x.shape, fieldSize, outputDepth, stride, zeroPad);
    const resultTexShape = conv_util.computeTexShapeFrom3D(resultShape);
    const resultTex = this.textureManager.acquireTexture(resultTexShape);

    conv_gpu.convolve(
        this.gpgpu, program, x.getTexture(), weights.getTexture(),
        biases != null ? biases.getTexture() : null, resultTex, resultTexShape);

    if (cleanupX) {
      x.dispose();
    }
    if (cleanupW) {
      weights.dispose();
    }
    if (cleanupB && biases != null) {
      biases.dispose();
    }

    return NDArray.make<Array3D>(
        resultShape, {texture: resultTex, textureShapeRC: resultTexShape});
  }

  protected conv2dBackPropInternal(
      x: Array3D, dy: Array3D, weights: Array4D, stride: number,
      pad: number): {dx: Array3D, dw: Array4D, db: Array1D} {
    const fSize = weights.shape[0];
    const inputDepth = weights.shape[2];
    const outputDepth = weights.shape[3];
    const xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
    const wTexShape =
        conv_util.computeWeightsTexShape(inputDepth, outputDepth, fSize);
    const yTexShape = conv_util.computeTexShapeFrom3D(dy.shape);

    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    let cleanupX = false;
    const actualXTexShape = x.getTextureShapeRC(xTexShape);
    if (!util.arraysEqual(actualXTexShape, xTexShape)) {
      x = this.reshapeTexture(x, xTexShape);
      cleanupX = true;
    }

    let cleanupW = false;
    const actualWTexShape = weights.getTextureShapeRC(wTexShape);
    if (!util.arraysEqual(actualWTexShape, wTexShape)) {
      weights = this.reshapeTexture(weights, wTexShape);
      cleanupW = true;
    }

    let cleanupY = false;
    const actualYTexShape = dy.getTextureShapeRC(yTexShape);
    if (!util.arraysEqual(actualYTexShape, yTexShape)) {
      dy = this.reshapeTexture(dy, yTexShape);
      cleanupY = true;
    }

    const dw = this.conv2dDerWeights(x, dy, fSize, stride, pad);
    const db = this.conv2dDerBias(dy);
    const dx = this.conv2dTransposeInternal(
        dy, weights, null /** biases */, stride, pad);

    if (cleanupX) {
      x.dispose();
    }
    if (cleanupW) {
      weights.dispose();
    }
    if (cleanupY) {
      dy.dispose();
    }
    return {dx, db, dw};
  }

  protected conv2dTransposeInternal(
      x: Array3D, weights: Array4D, biases: Array1D|null, origStride: number,
      origPad: number): Array3D {
    const origInputDepth = weights.shape[2];
    const origOutputDepth = weights.shape[3];
    const fieldSize = weights.shape[0];

    const progKey = [
      CONV2D_TRANSPOSE_PROG, x.shape, fieldSize, origInputDepth, origStride,
      origPad, biases != null
    ].join('_');
    const program = this.getAndSaveProgram(progKey, () => {
      return conv_backprop_gpu.getFragmentShaderConvTransposeSource(
          x.shape, fieldSize, origInputDepth, origStride, origPad,
          biases != null);
    });

    const xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
    const wTexShape = conv_util.computeWeightsTexShape(
        origInputDepth, origOutputDepth, fieldSize);
    const biasTexShape = conv_util.computeBiasesTexShape(origInputDepth);

    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    const actualXTexShape = x.getTextureShapeRC(xTexShape);
    let cleanupX = false;
    if (!util.arraysEqual(actualXTexShape, xTexShape)) {
      x = this.reshapeTexture(x, xTexShape);
      cleanupX = true;
    }

    let cleanupW = false;
    const actualWTexShape = weights.getTextureShapeRC(wTexShape);
    if (!util.arraysEqual(actualWTexShape, wTexShape)) {
      weights = this.reshapeTexture(weights, wTexShape);
      cleanupW = true;
    }

    let cleanupB = false;
    if (biases != null) {
      const actualBiasTexShape = biases.getTextureShapeRC(biasTexShape);
      if (!util.arraysEqual(actualBiasTexShape, biasTexShape)) {
        biases = this.reshapeTexture(biases, biasTexShape);
        cleanupB = true;
      }
    }

    // Figure out the output shape by dilating the input.
    const dilatedRC =
        conv_util.computeDilatedRC([x.shape[0], x.shape[1]], origStride);
    const pad = fieldSize - 1 - origPad;
    const resultShape = conv_util.computeOutputShape3D(
        [dilatedRC[0], dilatedRC[1], origOutputDepth], fieldSize,
        origInputDepth, 1, pad);
    const resultTexShape = conv_util.computeTexShapeFrom3D(resultShape);
    const resultTex = this.textureManager.acquireTexture(resultTexShape);

    conv_backprop_gpu.convTranspose(
        this.gpgpu, program, x.getTexture(), weights.getTexture(),
        biases != null ? biases.getTexture() : null, resultTex, resultTexShape);

    if (cleanupX) {
      x.dispose();
    }
    if (cleanupW) {
      weights.dispose();
    }
    if (cleanupB) {
      biases!.dispose();
    }

    return NDArray.make<Array3D>(
        resultShape, {texture: resultTex, textureShapeRC: resultTexShape});
  }

  conv2dDerWeights(
      x: Array3D, dY: Array3D, fSize: number, stride: number,
      zeroPad: number): Array4D {
    const inputDepth = x.shape[2];
    const outputDepth = dY.shape[2];
    const progKey = [
      CONV2D_DERW_PROG, x.shape, fSize, outputDepth, stride, zeroPad
    ].join('_');
    const program = this.getAndSaveProgram(progKey, () => {
      return conv_backprop_gpu.getFragmentShaderDerWeightsSource(
          x.shape, fSize, outputDepth, stride, zeroPad);
    });

    const xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
    const yShape = conv_util.computeOutputShape3D(
        x.shape, fSize, outputDepth, stride, zeroPad);
    const yTexShape = conv_util.computeTexShapeFrom3D(yShape);

    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    const actualXTexShape = x.getTextureShapeRC(xTexShape);
    let cleanupX = false;
    if (!util.arraysEqual(actualXTexShape, xTexShape)) {
      x = this.reshapeTexture(x, xTexShape);
      cleanupX = true;
    }

    let cleanupY = false;
    const actualYTexShape = dY.getTextureShapeRC(yTexShape);
    if (!util.arraysEqual(actualYTexShape, yTexShape)) {
      dY = this.reshapeTexture(dY, yTexShape);
      cleanupY = true;
    }

    const resultTexShape =
        conv_util.computeWeightsTexShape(inputDepth, outputDepth, fSize);
    const resultTex = this.textureManager.acquireTexture(resultTexShape);

    conv_backprop_gpu.derWeights(
        this.gpgpu, program, x.getTexture(), dY.getTexture(), resultTex,
        resultTexShape);

    if (cleanupX) {
      x.dispose();
    }
    if (cleanupY) {
      dY.dispose();
    }

    const weightsShape =
        conv_util.computeWeightsShape4D(inputDepth, outputDepth, fSize);
    return NDArray.make<Array4D>(
        weightsShape, {texture: resultTex, textureShapeRC: resultTexShape});
  }

  conv2dDerBias(dY: Array3D): Array1D {
    const outputDepth = dY.shape[2];
    const progKey = [CONV2D_DERB_PROG, dY.shape].join('_');
    const program = this.getAndSaveProgram(progKey, () => {
      return conv_backprop_gpu.getFragmentShaderDerBiasSource(dY.shape);
    });
    const yTexShape = conv_util.computeTexShapeFrom3D(dY.shape);

    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    let cleanupY = false;
    const actualYTexShape = dY.getTextureShapeRC(yTexShape);
    if (!util.arraysEqual(actualYTexShape, yTexShape)) {
      dY = this.reshapeTexture(dY, yTexShape);
      cleanupY = true;
    }

    const resultTexShape = conv_util.computeBiasesTexShape(outputDepth);
    const resultTex = this.textureManager.acquireTexture(resultTexShape);

    conv_backprop_gpu.derBias(
        this.gpgpu, program, dY.getTexture(), resultTex, resultTexShape);

    if (cleanupY) {
      dY.dispose();
    }

    return NDArray.make<Array1D>(
        [outputDepth], {texture: resultTex, textureShapeRC: resultTexShape});
  }

  private pool(
      program: WebGLProgram, x: Array3D, fSize: number, stride: number,
      pad: number): Array3D {
    const xTexShape = conv_util.computeTexShapeFrom3D(x.shape);

    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    const actualXTexShape = x.getTextureShapeRC(xTexShape);
    let cleanupX = false;
    if (!util.arraysEqual(actualXTexShape, xTexShape)) {
      x = this.reshapeTexture(x, xTexShape);
      cleanupX = true;
    }

    const resultShape =
        conv_util.computeOutputShape3D(x.shape, fSize, x.shape[2], stride, pad);
    const resultTexShape = conv_util.computeTexShapeFrom3D(resultShape);
    const poolResultTex = this.textureManager.acquireTexture(resultTexShape);

    pool_gpu.poolCommon(
        this.gpgpu, program, x.getTexture(), poolResultTex, resultTexShape);

    if (cleanupX) {
      x.dispose();
    }

    return NDArray.make<Array3D>(
        resultShape, {texture: poolResultTex, textureShapeRC: resultTexShape});
  }

  protected maxPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    const maxPoolProgKey =
        [MAX_POOL_PROG, x.shape, fSize, stride, pad].join('_');
    const maxPoolProgram = this.getAndSaveProgram(maxPoolProgKey, () => {
      return max_pool_gpu.getFragmentShaderMaxPoolSource(
          x.shape, fSize, stride, pad);
    });

    return this.pool(maxPoolProgram, x, fSize, stride, pad);
  }

  protected minPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    const minPoolProgKey =
        [MIN_POOL_PROG, x.shape, fSize, stride, pad].join('_');
    const minPoolProgram = this.getAndSaveProgram(minPoolProgKey, () => {
      return min_pool_gpu.getFragmentShaderMinPoolSource(
          x.shape, fSize, stride, pad);
    });

    return this.pool(minPoolProgram, x, fSize, stride, pad);
  }

  protected avgPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    const avgPoolProgKey =
        [AVG_POOL_PROG, x.shape, fSize, stride, pad].join('_');
    const avgPoolProgram = this.getAndSaveProgram(avgPoolProgKey, () => {
      return avg_pool_gpu.getFragmentShaderAvgPoolSource(
          x.shape, fSize, stride, pad);
    });

    return this.pool(avgPoolProgram, x, fSize, stride, pad);
  }

  protected maxPoolBackpropInternal(
      dy: Array3D, x: Array3D, fSize: number, origStride: number,
      origPad: number): Array3D {
    const maxPoolPositionsProgKey = [
      MAX_POOL_POSITIONS_PROG, x.shape, fSize, origStride, origPad
    ].join('_');
    const maxPoolPositionsProgram =
        this.getAndSaveProgram(maxPoolPositionsProgKey, () => {
          return max_pool_gpu.getFragmentShaderMaxPoolPositionsSource(
              x.shape, fSize, origStride, origPad);
        });

    const maxPoolResultShape = conv_util.computeOutputShape3D(
        x.shape, fSize, x.shape[2], origStride, origPad);
    const maxPoolResultTexShape =
        conv_util.computeTexShapeFrom3D(maxPoolResultShape);
    const maxPoolPositionsResultTex =
        this.textureManager.acquireTexture(maxPoolResultTexShape);
    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    const xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
    const actualXTexShape = x.getTextureShapeRC(xTexShape);
    let cleanupX = false;
    if (!util.arraysEqual(actualXTexShape, xTexShape)) {
      x = this.reshapeTexture(x, xTexShape);
      cleanupX = true;
    }

    max_pool_gpu.maxPoolCommon(
        this.gpgpu, maxPoolPositionsProgram, x.getTexture(),
        maxPoolPositionsResultTex, maxPoolResultTexShape);

    const maxPoolBackpropProgKey = [
      MAX_POOL_BACKPROP_PROG, dy.shape, fSize, origStride, origPad
    ].join('_');
    const program = this.getAndSaveProgram(maxPoolBackpropProgKey, () => {
      return max_pool_backprop_gpu.getFragmentShaderMaxPoolBackprop(
          dy.shape, fSize, origStride, origPad);
    });

    const dyTexShape = conv_util.computeTexShapeFrom3D(dy.shape);

    // If the texture shapes doesn't match the shapes that shaders expect,
    // do physical texture reshapes on the GPU.
    const actualDyTexShape = dy.getTextureShapeRC(dyTexShape);
    let cleanupDy = false;
    if (!util.arraysEqual(actualDyTexShape, dyTexShape)) {
      dy = this.reshapeTexture(dy, dyTexShape);
      cleanupDy = true;
    }

    const dilatedDyRC =
        conv_util.computeDilatedRC([dy.shape[0], dy.shape[1]], origStride);
    const pad = fSize - 1 - origPad;
    const resultShapeRCD = conv_util.computeOutputShape3D(
        [dilatedDyRC[0], dilatedDyRC[1], dy.shape[2]], fSize, dy.shape[2], 1,
        pad);
    const resultTexShape = conv_util.computeTexShapeFrom3D(resultShapeRCD);
    const resultTex = this.textureManager.acquireTexture(resultTexShape);

    max_pool_backprop_gpu.maxPoolBackprop(
        this.gpgpu, program, dy.getTexture(), maxPoolPositionsResultTex,
        resultTex, resultTexShape);

    if (cleanupDy) {
      dy.dispose();
    }

    if (cleanupX) {
      x.dispose();
    }

    this.textureManager.releaseTexture(
        maxPoolPositionsResultTex, maxPoolResultTexShape);

    return NDArray.make<Array3D>(
        resultShapeRCD, {texture: resultTex, textureShapeRC: resultTexShape});
  }

  protected resizeBilinear3DInternal(
      x: Array3D, newShape2D: [number, number],
      alignCorners: boolean): Array3D {
    const programKey =
        [RESIZE_BILINEAR_PROG, x.shape, newShape2D, alignCorners].join('_');

    const newShapeRCD: [number, number, number] =
        [newShape2D[0], newShape2D[1], x.shape[2]];
    const resultTexShape = conv_util.computeTexShapeFrom3D(newShapeRCD);

    const program = this.getAndSaveProgram(
        programKey,
        () => resize_bilinear_gpu.getFragmentShaderSource(
            x.shape, newShape2D, alignCorners));

    const resultTexture = this.textureManager.acquireTexture(resultTexShape);

    resize_bilinear_gpu.resizeBilinear(
        this.gpgpu, program, x.getTexture(), resultTexture, resultTexShape);

    return NDArray.make<Array3D>(
        newShapeRCD, {texture: resultTexture, textureShapeRC: resultTexShape});
  }

  private getAndSaveBinary<K extends NDArray, T extends NDArray>(
      key: string,
      getBinary: () => GPGPUBinary<K,T>):
      GPGPUBinary<K,T> {
    if (!(key in this.binaryCache)) {
      this.binaryCache[key] = getBinary();
    }
    return this.binaryCache[key] as GPGPUBinary<K, T>;
  }

  private getAndSaveProgram(programKey: string, getShaderSource: () => string):
      WebGLProgram {
    if (!(programKey in this.programCache)) {
      this.programCache[programKey] =
          this.gpgpu.createProgram(getShaderSource());
    }
    return this.programCache[programKey];
  }

  private addSubMulDiv(
      a: NDArray, b: NDArray, resultShape: number[],
      operandA: addsubmuldiv_gpu.OperandType,
      opType: addsubmuldiv_gpu.Operation,
      operandB: addsubmuldiv_gpu.OperandType): NDArray {
    let cleanupB = false;

    const aOrientation = MatrixOrientation.REGULAR;
    let bOrientation = MatrixOrientation.REGULAR;

    let logicalBTexShape: [number, number];

    if (operandA === OperandType.MATRIX && operandB === OperandType.MATRIX) {
      util.assertShapesMatch(a.shape, b.shape);

      if (a.inGPU()) {
        // Prefer B to have the shape of A.
        b.getTextureShapeRC(a.getTextureShapeRC());
      } else if (b.inGPU()) {
        // Prefer A to have the shape of B.
        a.getTextureShapeRC(b.getTextureShapeRC());
      }

      const aTexShape = a.getTextureShapeRC();
      const bTexShape = b.getTextureShapeRC();
      logicalBTexShape = bTexShape;

      if (a.rank === 1) {
        // When dealing with vectors, we can sample in transposed way without
        // the need to do physical reshape.
        if (!util.arraysEqual(bTexShape, aTexShape)) {
          bOrientation = MatrixOrientation.TRANSPOSED;
          logicalBTexShape = [bTexShape[1], bTexShape[0]];
        }
      }

      if (!util.arraysEqual(aTexShape, logicalBTexShape)) {
        b = this.reshapeTexture(b, aTexShape);
        bOrientation = MatrixOrientation.REGULAR;
        logicalBTexShape = b.getTextureShapeRC();
        cleanupB = true;
      }
    } else {
      logicalBTexShape = b.getTextureShapeRC();
    }

    const aTexShape = a.getTextureShapeRC();
    const bTexShape = b.getTextureShapeRC();

    const programKey = [
      ADD_SUM_MUL_DIV_PROG, operandA, aOrientation, opType, operandB,
      bOrientation
    ].join('_');
    const program = this.getAndSaveProgram(
        programKey,
        () => addsubmuldiv_gpu.getFragmentShaderSource(
            operandA, aOrientation, opType, operandB, bOrientation));

    const resultTextureShape: [number, number] = [
      Math.max(aTexShape[0], logicalBTexShape[0]),
      Math.max(aTexShape[1], logicalBTexShape[1])
    ];

    const resultTexture =
        this.textureManager.acquireTexture(resultTextureShape);

    addsubmuldiv_gpu.addSubMulDiv(
        this.gpgpu, program, a.getTexture(), aTexShape, b.getTexture(),
        bTexShape, resultTexture, resultTextureShape);

    if (cleanupB) {
      b.dispose();
    }

    return NDArray.make(
        resultShape,
        {texture: resultTexture, textureShapeRC: resultTextureShape});
  }

  private doGPUShapesMatch(a: NDArray, b: NDArray): boolean {
    util.assertShapesMatch(a.shape, b.shape);
    if (a.inGPU()) {
      // Prefer B to have the shape of A.
      b.getTextureShapeRC(a.getTextureShapeRC());
    } else if (b.inGPU()) {
      // Prefer A to have the shape of B.
      a.getTextureShapeRC(b.getTextureShapeRC());
    }
    return util.arraysEqual(a.getTextureShapeRC(), b.getTextureShapeRC());
  }

  getTextureManager(): TextureManager {
    return this.textureManager;
  }

  dispose() {
    for (const programKey in this.programCache) {
      if (this.programCache.hasOwnProperty(programKey)) {
        this.gpgpu.deleteProgram(this.programCache[programKey]);
      }
    }
    for (const key in this.binaryCache) {
      this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
    }
    this.textureManager.dispose();

    if (this.gpgpuCreatedLocally) {
      this.gpgpu.dispose();
    }
  }
}
