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

import * as util from '../util';

import * as axis_util from './axis_util';
import {Conv2DInfo} from './conv_util';
import {MatrixOrientation, NDArrayMath, SumTypes, SumTypesMap} from './math';
import * as ndarray from './ndarray';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, DataTypes, NDArray, Scalar} from './ndarray';
import * as reduce_util from './reduce_util';
import {AddScaledMatProgram} from './webgl/addscaledmat_gpu';
import {ArgMinMaxProgram} from './webgl/argminmax_gpu';
import {BatchNormProgram} from './webgl/batchnorm_gpu';
import * as binaryop_gpu from './webgl/binaryop_gpu';
import {BinaryOpProgram} from './webgl/binaryop_gpu';
import {ClipProgram} from './webgl/clip_gpu';
import {ConcatProgram} from './webgl/concat_gpu';
// tslint:disable-next-line:max-line-length
import {Conv2DDerBiasProgram, Conv2DDerFilterProgram, Conv2DDerInputProgram} from './webgl/conv_backprop_gpu';
import {Conv2DProgram} from './webgl/conv_gpu';
import {DepthwiseConv2DProgram} from './webgl/conv_gpu_depthwise';
import {Copy2DProgram} from './webgl/copy_gpu';
import {GPGPUContext} from './webgl/gpgpu_context';
import * as gpgpu_math from './webgl/gpgpu_math';
import {GPGPUBinary, GPGPUProgram} from './webgl/gpgpu_math';
import * as gpgpu_util from './webgl/gpgpu_util';
import {MaxPool2DBackpropProgram} from './webgl/max_pool_backprop_gpu';
import {MatMulProgram} from './webgl/mulmat_gpu';
import {MultinomialProgram} from './webgl/multinomial_gpu';
import {OneHotProgram} from './webgl/onehot_gpu';
import {Pool2DProgram} from './webgl/pool_gpu';
import {ReduceProgram} from './webgl/reduce_gpu';
import {ResizeBilinear3DProgram} from './webgl/resize_bilinear_gpu';
import {SliceProgram} from './webgl/slice_gpu';
import {TextureManager} from './webgl/texture_manager';
import {TileProgram} from './webgl/tile_gpu';
import {TransposeProgram} from './webgl/transpose_gpu';
import * as unary_op from './webgl/unaryop_gpu';
import {UnaryOpProgram} from './webgl/unaryop_gpu';
import * as webgl_util from './webgl/webgl_util';

export class NDArrayMathGPU extends NDArrayMath {
  private gpgpu: GPGPUContext;
  private textureManager: TextureManager;
  private binaryCache: {[key: string]: GPGPUBinary} = {};
  private gpgpuCreatedLocally: boolean;

  constructor(gpgpu?: GPGPUContext, safeMode = false) {
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

  protected cloneInternal<G extends keyof DataTypes, T extends NDArray<G>>(
      a: T): T {
    const texShape = a.getTextureShapeRC();
    // Pretend the source was in logical shape that matches the texture shape.
    const source = a.as2D(texShape[0], texShape[1]);
    // Do the same for output.
    const output = this.makeOutputArray<G, Array2D<G>>(texShape, a.dtype);
    this.copy2D(source, [0, 0], texShape, output, [0, 0], texShape);
    // Get back to the original logical shape.
    return output.reshape(a.shape) as T;
  }

  protected slice1DInternal(input: Array1D, begin: number, size: number):
      Array1D {
    const program = new SliceProgram([size]);
    const customSetup = program.getCustomSetupFunc([begin]);
    return this.compileAndRun(program, [input], null, customSetup);
  }

  protected slice2DInternal(input: Array2D, begin: [number, number], size: [
    number, number
  ]): Array2D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [input], null, customSetup);
  }

  protected slice3DInternal(
      input: Array3D, begin: [number, number, number],
      size: [number, number, number]): Array3D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [input], null, customSetup);
  }

  protected slice4DInternal(
      input: Array4D, begin: [number, number, number, number],
      size: [number, number, number, number]): Array4D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [input], null, customSetup);
  }

  protected copy2DInternal(
      source: Array2D, sourceBeginRowCol: [number, number],
      sourceSizeRowCol: [number, number], dest: Array2D,
      destBeginRowCol: [number, number],
      destSizeRowCol: [number, number]): void {
    const program = new Copy2DProgram(sourceSizeRowCol[1], destSizeRowCol[1]);
    const customSetup = program.getCustomSetupFunc(
        sourceBeginRowCol, destBeginRowCol, destSizeRowCol);
    this.compileAndRun(program, [source], dest, customSetup);
  }

  protected concat1DInternal(a: Array1D, b: Array1D): Array1D {
    const program = new ConcatProgram(a.shape, b.shape, 0);
    return this.compileAndRun(program, [a, b]);
  }

  protected concat2DInternal(a: Array2D, b: Array2D, axis: number): Array2D {
    const program = new ConcatProgram(a.shape, b.shape, axis);
    return this.compileAndRun(program, [a, b]);
  }

  protected concat3DInternal(x1: Array3D, x2: Array3D, axis: number): Array3D {
    const program = new ConcatProgram(x1.shape, x2.shape, axis);
    return this.compileAndRun(program, [x1, x2]);
  }

  protected concat4DInternal(x1: Array4D, x2: Array4D, axis: number): Array4D {
    const program = new ConcatProgram(x1.shape, x2.shape, axis);
    return this.compileAndRun(program, [x1, x2]);
  }

  protected scaledArrayAddInternal<T extends NDArray>(
      c1: Scalar, a: T, c2: Scalar, b: T): T {
    const program = new AddScaledMatProgram(a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b, c1, c2]) as T;
  }

  protected negInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.NEG);
    return this.compileAndRun(program, [a]) as T;
  }

  private makeOutputArray<G extends keyof DataTypes, T extends NDArray<G>>(
      shape: number[], dtype: G): T {
    const textureShapeRC =
        webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, shape);
    const texture = this.textureManager.acquireTexture(textureShapeRC);
    return NDArray.make(shape, {texture, textureShapeRC}, dtype) as T;
  }

  private compileAndRun<T extends NDArray, K extends NDArray>(
      program: GPGPUProgram, inputs: T[], output?: K,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void):
      K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }
    const key = gpgpu_math.makeShaderKey(program, inputs, output);
    const binary = this.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(this.gpgpu, program, inputs, output);
    });
    gpgpu_math.runProgram(binary, inputs, output, customSetup);
    return output;
  }

  protected matMulInternal(
      a: Array2D, b: Array2D, aOrientation: MatrixOrientation,
      bOrientation: MatrixOrientation): Array2D {
    const program =
        new MatMulProgram(a.shape, b.shape, aOrientation, bOrientation);
    return this.compileAndRun<Array2D, Array2D>(program, [a, b]);
  }

  protected multiplyInternal<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]) as T;
  }

  protected batchNormalization2DInternal(
      x: Array2D, mean: Array2D|Array1D, variance: Array2D|Array1D,
      varianceEpsilon: number, scale?: Array2D|Array1D,
      offset?: Array2D|Array1D): Array2D {
    const inputs = [x, mean, variance];

    let offsetShape = null;
    if (offset != null) {
      offsetShape = offset.shape;
      inputs.push(offset);
    }

    let scaleShape = null;
    if (scale != null) {
      scaleShape = scale.shape;
      inputs.push(scale);
    }

    const program = new BatchNormProgram(
        x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
        varianceEpsilon);
    return this.compileAndRun(program, inputs);
  }

  protected batchNormalization3DInternal(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon: number, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D {
    const inputs = [x, mean, variance];

    let offsetShape = null;
    if (offset != null) {
      offsetShape = offset.shape;
      inputs.push(offset);
    }

    let scaleShape = null;
    if (scale != null) {
      scaleShape = scale.shape;
      inputs.push(scale);
    }

    const program = new BatchNormProgram(
        x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
        varianceEpsilon);
    return this.compileAndRun(program, inputs);
  }

  protected tileInternal<D extends keyof DataTypes, T extends NDArray<D>>(
      a: T, reps: number[]): T {
    const program = new TileProgram(a.shape, reps);
    return this.compileAndRun(program, [a]);
  }

  protected transposeInternal<D extends keyof DataTypes, T extends NDArray<D>>(
      a: T, perm: number[]): T {
    const program = new TransposeProgram(a.shape, perm);
    return this.compileAndRun(program, [a]);
  }

  private reduce<D extends keyof DataTypes>(
      a: Array2D, reduceType: 'max'|'min'|'sum', dtype: D): Array2D<D> {
    const batchSize = a.shape[0];
    const inSize = a.shape[1];
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {windowSize, inSize, batchSize};
    const program = new ReduceProgram(reduceInfo, reduceType);
    const [rows, cols] = program.outputShape;
    const output =
        this.makeOutputArray(program.outputShape, dtype).as2D(rows, cols);
    this.compileAndRun(program, [a], output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.reduce(output, reduceType, dtype);
  }

  private argReduce(
      a: Array2D, reduceType: 'max'|'min',
      bestIndicesA: Array2D = null): Array2D<'int32'> {
    let batchSize = a.shape[0];
    let inSize = a.shape[1];
    if (bestIndicesA != null) {
      batchSize = bestIndicesA.shape[0];
      inSize = bestIndicesA.shape[1];
    }
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {windowSize, inSize, batchSize};
    const program =
        new ArgMinMaxProgram(reduceInfo, reduceType, bestIndicesA == null);
    const [rows, cols] = program.outputShape;
    const output =
        this.makeOutputArray(program.outputShape, 'int32').as2D(rows, cols);
    const inputs = [a];
    if (bestIndicesA != null) {
      inputs.push(bestIndicesA);
    }
    this.compileAndRun(program, inputs, output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.argReduce(a, reduceType, output);
  }

  protected sumInternal<T extends keyof DataTypes>(
      a: NDArray<T>, axes: number[]): NDArray<SumTypes[T]> {
    axis_util.assertAxesAreInnerMostDims('sum', axes, a.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(a.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = a.as2D(-1, inSize);
    const outputDType = SumTypesMap[a.dtype];
    return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
  }

  protected argMinInternal(a: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMin', axes, a.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(a.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = a.as2D(-1, inSize);
    return this.argReduce(a2D, 'min').reshape(outShape);
  }

  protected argMaxInternal(a: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMax', axes, a.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(a.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = a.as2D(-1, inSize);
    return this.argReduce(a2D, 'max').reshape(outShape);
  }

  protected equalInternal(x: NDArray, y: NDArray): NDArray<'bool'> {
    const program = new BinaryOpProgram(binaryop_gpu.EQUAL, x.shape, y.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [x, y], output);
  }

  protected topKInternal(ndarray: NDArray, k: number):
      {values: Array1D, indices: Array1D} {
    throw new Error('topK GPU not yet implemented!');
  }

  protected minInternal<G extends keyof DataTypes>(
      a: NDArray<G>, axes: number[]): NDArray<G> {
    axis_util.assertAxesAreInnerMostDims('min', axes, a.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(a.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = a.as2D(-1, inSize);
    return this.reduce(a2D, 'min', a2D.dtype).reshape(outShape);
  }

  protected maxInternal<G extends keyof DataTypes>(
      a: NDArray<G>, axes: number[]): NDArray<G> {
    axis_util.assertAxesAreInnerMostDims('max', axes, a.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(a.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = a.as2D(-1, inSize);
    return this.reduce(a2D, 'max', a2D.dtype).reshape(outShape);
  }

  protected divideInternal(a: NDArray, b: NDArray): NDArray<'float32'> {
    const program = new BinaryOpProgram(binaryop_gpu.DIV, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'float32');
    return this.compileAndRun<NDArray, NDArray<'float32'>>(
        program, [a, b], output);
  }

  protected addInternal<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b]);
  }

  protected subtractInternal<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b]);
  }

  protected ceilInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.CEIL);
    return this.compileAndRun(program, [a]) as T;
  }

  protected floorInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.FLOOR);
    return this.compileAndRun(program, [a]) as T;
  }

  protected expInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.EXP);
    return this.compileAndRun(program, [a]) as T;
  }

  protected logInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.LOG);
    return this.compileAndRun(program, [a]) as T;
  }

  protected sqrtInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.SQRT);
    return this.compileAndRun(program, [a]) as T;
  }

  protected squareInternal<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQUARE);
    return this.compileAndRun(program, [x]) as T;
  }

  protected reluInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.RELU);
    return this.compileAndRun(program, [a]) as T;
  }

  protected eluInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ELU);
    return this.compileAndRun(program, [a]) as T;
  }

  protected eluDerInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ELU_DER);
    return this.compileAndRun(program, [a]) as T;
  }

  protected leakyReluInternal<T extends NDArray>(a: T, alpha: number): T {
    const program = new UnaryOpProgram(a.shape, unary_op.LEAKY_RELU(alpha));
    return this.compileAndRun(program, [a]) as T;
  }

  protected clipInternal<T extends NDArray>(a: T, min: number, max: number): T {
    const program = new ClipProgram(a.shape, min, max);
    return this.compileAndRun(program, [a]) as T;
  }

  protected absInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ABS);
    return this.compileAndRun(program, [a]) as T;
  }

  protected sigmoidInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.SIGMOID);
    return this.compileAndRun(program, [a]) as T;
  }

  protected sinInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.SIN);
    return this.compileAndRun(program, [a]) as T;
  }

  protected cosInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.COS);
    return this.compileAndRun(program, [a]) as T;
  }

  protected tanInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.TAN);
    return this.compileAndRun(program, [a]) as T;
  }

  protected asinInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ASIN);
    return this.compileAndRun(program, [a]) as T;
  }

  protected acosInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ACOS);
    return this.compileAndRun(program, [a]) as T;
  }

  protected atanInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ATAN);
    return this.compileAndRun(program, [a]) as T;
  }

  protected sinhInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.SINH);
    return this.compileAndRun(program, [a]) as T;
  }

  protected coshInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.COSH);
    return this.compileAndRun(program, [a]) as T;
  }

  protected tanhInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.TANH);
    return this.compileAndRun(program, [a]) as T;
  }

  protected stepInternal<T extends NDArray>(a: T, alpha: number): T {
    const program = new UnaryOpProgram(a.shape, unary_op.STEP(alpha));
    return this.compileAndRun(program, [a]) as T;
  }

  protected conv2dInternal(
      x: Array4D, filter: Array4D, bias: Array1D|null,
      convInfo: Conv2DInfo): Array4D {
    const program = new Conv2DProgram(convInfo, bias != null);
    const inputs = bias != null ? [x, filter, bias] : [x, filter];
    return this.compileAndRun(program, inputs);
  }

  protected conv2dDerInputInternal(
      dy: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Conv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  protected conv2dDerFilterInternal(
      x: Array4D, dY: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Conv2DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dY]);
  }

  protected conv2dDerBiasInternal(dY: Array4D): Array1D {
    const program = new Conv2DDerBiasProgram(dY.shape);
    return this.compileAndRun(program, [dY]);
  }

  protected depthwiseConv2DInternal(
      input: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new DepthwiseConv2DProgram(convInfo);
    return this.compileAndRun(program, [input, filter]);
  }

  protected maxPoolInternal(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, 'max', false);
    return this.compileAndRun(program, [x]);
  }

  protected minPoolInternal(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, 'min', false);
    return this.compileAndRun(program, [x]);
  }

  protected avgPoolInternal(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, 'avg', false);
    return this.compileAndRun(program, [x]);
  }

  protected maxPoolBackpropInternal(
      dy: Array4D, x: Array4D, convInfo: Conv2DInfo): Array4D {
    const getPositions = true;
    const maxPoolPositionsProgram =
        new Pool2DProgram(convInfo, 'max', getPositions);
    const maxPoolPositions: Array4D =
        this.compileAndRun(maxPoolPositionsProgram, [x]);

    const maxPoolBackPropProgram = new MaxPool2DBackpropProgram(convInfo);

    const result =
        this.compileAndRun(maxPoolBackPropProgram, [dy, maxPoolPositions]);
    maxPoolPositions.dispose();
    return result as Array4D;
  }

  protected resizeBilinear3DInternal(
      x: Array3D, newShape2D: [number, number],
      alignCorners: boolean): Array3D {
    const program =
        new ResizeBilinear3DProgram(x.shape, newShape2D, alignCorners);
    return this.compileAndRun(program, [x]);
  }

  protected multinomialInternal(
      probs: Array2D, numSamples: number, seed: number): Array2D<'int32'> {
    const batchSize = probs.shape[0];
    const numOutcomes = probs.shape[1];
    const program = new MultinomialProgram(batchSize, numOutcomes, numSamples);
    const output =
        this.makeOutputArray(program.outputShape, 'int32') as Array2D<'int32'>;
    const customSetup = program.getCustomSetupFunc(seed);
    return this.compileAndRun(program, [probs], output, customSetup);
  }

  protected oneHotInternal(
      indices: Array1D, depth: number, onValue: number,
      offValue: number): Array2D {
    const program = new OneHotProgram(indices.size, depth, onValue, offValue);
    return this.compileAndRun(program, [indices]);
  }

  private getAndSaveBinary(key: string, getBinary: () => GPGPUBinary):
      GPGPUBinary {
    if (!(key in this.binaryCache)) {
      this.binaryCache[key] = getBinary();
    }
    return this.binaryCache[key];
  }

  getTextureManager(): TextureManager {
    return this.textureManager;
  }

  dispose() {
    for (const key in this.binaryCache) {
      this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
    }
    this.textureManager.dispose();

    if (this.gpgpuCreatedLocally) {
      this.gpgpu.dispose();
    }
  }
}
