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

import {ENV} from '../../environment';
import * as util from '../../util';
import {TypedArray} from '../../util';
import * as axis_util from '../axis_util';
import {Conv2DInfo} from '../conv_util';
import {NDArrayMath} from '../math';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, DataTypes, NDArray} from '../ndarray';
import * as reduce_util from '../reduce_util';
import {SumTypes, SumTypesMap} from '../types';

import {BACKEND_REGISTRY, MathBackend} from './backend';
import {MatrixOrientation} from './types/matmul';
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
import {ArrayData, GPGPUBinary, GPGPUProgram} from './webgl/gpgpu_math';
import * as gpgpu_util from './webgl/gpgpu_util';
import {MaxPool2DBackpropProgram} from './webgl/max_pool_backprop_gpu';
import {MatMulProgram} from './webgl/mulmat_gpu';
import {MultinomialProgram} from './webgl/multinomial_gpu';
import {OneHotProgram} from './webgl/onehot_gpu';
import {Pool2DProgram} from './webgl/pool_gpu';
import {ReduceProgram} from './webgl/reduce_gpu';
import {ResizeBilinear3DProgram} from './webgl/resize_bilinear_gpu';
import {SliceProgram} from './webgl/slice_gpu';
import {TextureData, TextureType} from './webgl/tex_util';
import {TextureManager} from './webgl/texture_manager';
import {TileProgram} from './webgl/tile_gpu';
import {TransposeProgram} from './webgl/transpose_gpu';
import * as unary_op from './webgl/unaryop_gpu';
import {UnaryOpProgram} from './webgl/unaryop_gpu';
import * as webgl_util from './webgl/webgl_util';

export class MathBackendWebGL implements MathBackend {
  private texData: {[id: number]: TextureData} = {};

  writePixels(
      id: number,
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): void {
    const shape: [number, number, number] =
        [pixels.height, pixels.width, numChannels];
    const texShape: [number, number] = [shape[0], shape[1]];
    const texture = this.textureManager.acquireTexture(texShape);
    this.gpgpu.uploadPixelDataToTexture(texture, pixels);
    this.texData[id] = {
      texture,
      textureType: TextureType.RGBA_COLOR,
      texShape,
      numChannels,
      dtype: 'int32'
    };
  }
  write<T extends keyof DataTypes>(
      id: number, values: DataTypes[T], dtype: T, shape: number[]): void {
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, shape);
    const texture = this.textureManager.acquireTexture(texShape);
    const textureType = TextureType.DEFAULT;
    this.texData[id] = {texture, textureType, texShape, dtype};

    if (values != null) {
      this.gpgpu.uploadMatrixToTexture(
          texture, texShape[0],
          // TODO(smilkov): Propagate the original typed array to gpgpu.
          texShape[1], typedArrayToFloat32(values, dtype));
    }
  }
  readSync<T extends keyof DataTypes>(id: number): DataTypes[T] {
    let values: Float32Array;
    const {texture, textureType, texShape, numChannels, dtype} =
        this.texData[id];
    if (textureType === TextureType.DEFAULT) {
      values = this.gpgpu.downloadMatrixFromTexture(
          texture, texShape[0], texShape[1]);
    } else {
      values = this.gpgpu.downloadMatrixFromRGBAColorTexture(
          texture, texShape[0], texShape[1], numChannels);
    }
    return float32ToTypedArray(values, dtype);
  }
  async read<T extends keyof DataTypes>(id: number): Promise<DataTypes[T]> {
    const {texture, textureType, texShape} = this.texData[id];
    if (ENV.get('WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED') &&
        textureType === TextureType.DEFAULT) {
      return this.gpgpu.downloadMatrixFromTextureAsync(
          texture, texShape[0], texShape[1]);
    }

    if (!ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')) {
      return await this.readSync(id);
    }

    // Construct an empty query. We're just interested in getting a callback
    // when the GPU command queue has executed until this point in time.
    const queryFn = () => {};
    await this.gpgpu.runQuery(queryFn);
    return this.readSync(id);
  }
  disposeData(id: number): void {
    if (id in this.texData) {
      const {texture, texShape} = this.texData[id];
      this.textureManager.releaseTexture(texture, texShape);
      delete this.texData[id];
    }
  }

  private gpgpu: GPGPUContext;
  private textureManager: TextureManager;
  private binaryCache: {[key: string]: GPGPUBinary} = {};
  private gpgpuCreatedLocally: boolean;

  constructor(gpgpu?: GPGPUContext) {
    if (gpgpu == null) {
      const gl = gpgpu_util.createWebGLContext();
      this.gpgpu = new GPGPUContext(gl);
      this.gpgpuCreatedLocally = true;
    } else {
      this.gpgpu = gpgpu;
      this.gpgpuCreatedLocally = false;
    }
    this.textureManager = new TextureManager(this.gpgpu);
  }

  getGPGPUContext(): GPGPUContext {
    return this.gpgpu;
  }

  clone<G extends keyof DataTypes, T extends NDArray<G>>(x: T): T {
    const {texShape} = this.texData[x.id];
    // Pretend the source was in logical shape that matches the texture shape.
    const source = x.as2D(texShape[0], texShape[1]);
    // Do the same for output.
    const output = this.makeOutputArray<G, Array2D<G>>(texShape, x.dtype);
    this.copy2D(source, [0, 0], texShape, output, [0, 0], texShape);
    // Get back to the original logical shape.
    return output.reshape(x.shape) as T;
  }

  slice1D(x: Array1D, begin: number, size: number): Array1D {
    const program = new SliceProgram([size]);
    const customSetup = program.getCustomSetupFunc([begin]);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  slice2D(x: Array2D, begin: [number, number], size: [number, number]):
      Array2D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  slice3D(x: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  slice4D(x: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  private copy2D(
      source: Array2D, sourceBeginRowCol: [number, number],
      sourceSizeRowCol: [number, number], dest: Array2D,
      destBeginRowCol: [number, number],
      destSizeRowCol: [number, number]): void {
    const program = new Copy2DProgram(sourceSizeRowCol[1], destSizeRowCol[1]);
    const customSetup = program.getCustomSetupFunc(
        sourceBeginRowCol, destBeginRowCol, destSizeRowCol);
    this.compileAndRun(program, [source], dest, customSetup);
  }

  concat1D(a: Array1D, b: Array1D): Array1D {
    const program = new ConcatProgram(a.shape, b.shape, 0);
    return this.compileAndRun(program, [a, b]);
  }

  concat2D(a: Array2D, b: Array2D, axis: number): Array2D {
    const program = new ConcatProgram(a.shape, b.shape, axis);
    return this.compileAndRun(program, [a, b]);
  }

  concat3D(a: Array3D, b: Array3D, axis: number): Array3D {
    const program = new ConcatProgram(a.shape, b.shape, axis);
    return this.compileAndRun(program, [a, b]);
  }

  concat4D(a: Array4D, b: Array4D, axis: number): Array4D {
    const program = new ConcatProgram(a.shape, b.shape, axis);
    return this.compileAndRun(program, [a, b]);
  }

  neg<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.NEG);
    return this.compileAndRun(program, [x]) as T;
  }

  matMul(
      a: Array2D, b: Array2D, aOrientation: MatrixOrientation,
      bOrientation: MatrixOrientation): Array2D {
    const program =
        new MatMulProgram(a.shape, b.shape, aOrientation, bOrientation);
    return this.compileAndRun<Array2D, Array2D>(program, [a, b]);
  }

  multiply<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]) as T;
  }

  batchNormalization2D(
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

  batchNormalization3D(
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

  tile<D extends keyof DataTypes, T extends NDArray<D>>(x: T, reps: number[]):
      T {
    const program = new TileProgram(x.shape, reps);
    return this.compileAndRun(program, [x]);
  }

  transpose<D extends keyof DataTypes, T extends NDArray<D>>(
      x: T, perm: number[]): T {
    const program = new TransposeProgram(x.shape, perm);
    return this.compileAndRun(program, [x]);
  }

  private reduce<D extends keyof DataTypes>(
      x: Array2D, reduceType: 'max'|'min'|'sum', dtype: D): Array2D<D> {
    const batchSize = x.shape[0];
    const inSize = x.shape[1];
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {windowSize, inSize, batchSize};
    const program = new ReduceProgram(reduceInfo, reduceType);
    const [rows, cols] = program.outputShape;
    const output =
        this.makeOutputArray(program.outputShape, dtype).as2D(rows, cols);
    this.compileAndRun(program, [x], output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.reduce(output, reduceType, dtype);
  }

  private argReduce(
      x: Array2D, reduceType: 'max'|'min',
      bestIndicesA: Array2D = null): Array2D<'int32'> {
    let batchSize = x.shape[0];
    let inSize = x.shape[1];
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
    const inputs = [x];
    if (bestIndicesA != null) {
      inputs.push(bestIndicesA);
    }
    this.compileAndRun(program, inputs, output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.argReduce(x, reduceType, output);
  }

  sum<T extends keyof DataTypes>(x: NDArray<T>, axes: number[]):
      NDArray<SumTypes[T]> {
    axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    const outputDType = SumTypesMap[x.dtype];
    return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
  }

  argMin(x: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.argReduce(a2D, 'min').reshape(outShape);
  }

  argMax(x: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.argReduce(a2D, 'max').reshape(outShape);
  }

  equal(a: NDArray, b: NDArray): NDArray<'bool'> {
    const program = new BinaryOpProgram(binaryop_gpu.EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  topKValues<D extends keyof DataTypes, T extends NDArray<D>>(x: T, k: number):
      Array1D<D> {
    throw new Error('topKValues GPU not yet implemented!');
  }

  topKIndices(x: NDArray, k: number): Array1D<'int32'> {
    throw new Error('topKIndices GPU not yet implemented!');
  }

  min<G extends keyof DataTypes>(x: NDArray<G>, axes: number[]): NDArray<G> {
    axis_util.assertAxesAreInnerMostDims('min', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'min', a2D.dtype).reshape(outShape);
  }

  max<G extends keyof DataTypes>(x: NDArray<G>, axes: number[]): NDArray<G> {
    axis_util.assertAxesAreInnerMostDims('max', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'max', a2D.dtype).reshape(outShape);
  }

  divide(a: NDArray, b: NDArray): NDArray<'float32'> {
    const program = new BinaryOpProgram(binaryop_gpu.DIV, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'float32');
    return this.compileAndRun<NDArray, NDArray<'float32'>>(
        program, [a, b], output);
  }

  add<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b]);
  }

  subtract<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b]);
  }

  pow<T extends NDArray>(a: T, b: NDArray<'int32'>): T {
    const program = new BinaryOpProgram(binaryop_gpu.POW, a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b]);
  }

  ceil<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.CEIL);
    return this.compileAndRun(program, [x]) as T;
  }

  floor<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.FLOOR);
    return this.compileAndRun(program, [x]) as T;
  }

  exp<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.EXP);
    return this.compileAndRun(program, [x]) as T;
  }

  log<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOG);
    return this.compileAndRun(program, [x]) as T;
  }

  sqrt<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQRT);
    return this.compileAndRun(program, [x]) as T;
  }

  square<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQUARE);
    return this.compileAndRun(program, [x]) as T;
  }

  relu<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RELU);
    return this.compileAndRun(program, [x]) as T;
  }

  elu<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ELU);
    return this.compileAndRun(program, [x]) as T;
  }

  eluDer<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ELU_DER);
    return this.compileAndRun(program, [x]) as T;
  }

  selu<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SELU);
    return this.compileAndRun(program, [x]) as T;
  }

  leakyRelu<T extends NDArray>(x: T, alpha: number): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LEAKY_RELU(alpha));
    return this.compileAndRun(program, [x]) as T;
  }

  clip<T extends NDArray>(x: T, min: number, max: number): T {
    const program = new ClipProgram(x.shape, min, max);
    return this.compileAndRun(program, [x]) as T;
  }

  abs<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ABS);
    return this.compileAndRun(program, [x]) as T;
  }

  sigmoid<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGMOID);
    return this.compileAndRun(program, [x]) as T;
  }

  sin<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIN);
    return this.compileAndRun(program, [x]) as T;
  }

  cos<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COS);
    return this.compileAndRun(program, [x]) as T;
  }

  tan<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TAN);
    return this.compileAndRun(program, [x]) as T;
  }

  asin<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ASIN);
    return this.compileAndRun(program, [x]) as T;
  }

  acos<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ACOS);
    return this.compileAndRun(program, [x]) as T;
  }

  atan<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATAN);
    return this.compileAndRun(program, [x]) as T;
  }

  sinh<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SINH);
    return this.compileAndRun(program, [x]) as T;
  }

  cosh<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COSH);
    return this.compileAndRun(program, [x]) as T;
  }

  tanh<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TANH);
    return this.compileAndRun(program, [x]) as T;
  }

  step<T extends NDArray>(x: T, alpha: number): T {
    const program = new UnaryOpProgram(x.shape, unary_op.STEP(alpha));
    return this.compileAndRun(program, [x]) as T;
  }

  conv2d(x: Array4D, filter: Array4D, bias: Array1D|null, convInfo: Conv2DInfo):
      Array4D {
    const program = new Conv2DProgram(convInfo, bias != null);
    const inputs = bias != null ? [x, filter, bias] : [x, filter];
    return this.compileAndRun(program, inputs);
  }

  conv2dDerInput(dy: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Conv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  conv2dDerFilter(x: Array4D, dy: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Conv2DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dy]);
  }

  conv2dDerBias(dy: Array4D): Array1D {
    const program = new Conv2DDerBiasProgram(dy.shape);
    return this.compileAndRun(program, [dy]);
  }

  depthwiseConv2D(x: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new DepthwiseConv2DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
  }

  maxPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, 'max', false);
    return this.compileAndRun(program, [x]);
  }

  minPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, 'min', false);
    return this.compileAndRun(program, [x]);
  }

  avgPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, 'avg', false);
    return this.compileAndRun(program, [x]);
  }

  maxPoolBackprop(dy: Array4D, x: Array4D, convInfo: Conv2DInfo): Array4D {
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

  resizeBilinear3D(
      x: Array3D, newShape2D: [number, number],
      alignCorners: boolean): Array3D {
    const program =
        new ResizeBilinear3DProgram(x.shape, newShape2D, alignCorners);
    return this.compileAndRun(program, [x]);
  }

  multinomial(probs: Array2D, numSamples: number, seed: number):
      Array2D<'int32'> {
    const batchSize = probs.shape[0];
    const numOutcomes = probs.shape[1];
    const program = new MultinomialProgram(batchSize, numOutcomes, numSamples);
    const output =
        this.makeOutputArray(program.outputShape, 'int32') as Array2D<'int32'>;
    const customSetup = program.getCustomSetupFunc(seed);
    return this.compileAndRun(program, [probs], output, customSetup);
  }

  oneHot(indices: Array1D, depth: number, onValue: number, offValue: number):
      Array2D {
    const program = new OneHotProgram(indices.size, depth, onValue, offValue);
    return this.compileAndRun(program, [indices]);
  }

  private makeOutputArray<G extends keyof DataTypes, T extends NDArray<G>>(
      shape: number[], dtype: G): T {
    return NDArray.make(shape, {}, dtype) as T;
  }

  private compileAndRun<T extends NDArray, K extends NDArray>(
      program: GPGPUProgram, inputs: T[], output?: K,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void):
      K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }
    const inputsData: Array<ArrayData<T>> = inputs.map(input => {
      return {array: input, texData: this.texData[input.id]};
    });
    const outputData = {array: output, texData: this.texData[output.id]};
    const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
    const binary = this.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(
          this.gpgpu, program, inputsData, outputData);
    });
    gpgpu_math.runProgram(binary, inputsData, outputData, customSetup);
    return output;
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

BACKEND_REGISTRY['webgl'] = new MathBackendWebGL();

// TODO(nsthorat): Deprecate this once we export non-abstract NDArrayMath.
export class NDArrayMathGPU extends NDArrayMath {
  constructor(gpgpu?: GPGPUContext, safeMode = false) {
    super(new MathBackendWebGL(gpgpu), safeMode);
  }

  getGPGPUContext(): GPGPUContext {
    return (this.backendEngine.getBackend() as MathBackendWebGL)
        .getGPGPUContext();
  }

  getTextureManager(): TextureManager {
    return (this.backendEngine.getBackend() as MathBackendWebGL)
        .getTextureManager();
  }
}

function float32ToTypedArray<T extends keyof DataTypes>(
    a: Float32Array, dtype: T): DataTypes[T] {
  if (dtype === 'float32') {
    return a;
  } else if (dtype === 'int32' || dtype === 'bool') {
    const result = (dtype === 'int32') ? new Int32Array(a.length) :
                                         new Uint8Array(a.length);
    for (let i = 0; i < result.length; ++i) {
      let val = a[i];
      val = isNaN(val) ? util.getNaN(dtype) : Math.round(val);
      result[i] = val;
    }
    return result;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

function typedArrayToFloat32(
    a: TypedArray, dtype: keyof DataTypes): Float32Array {
  if (a instanceof Float32Array) {
    return a;
  } else {
    const res = new Float32Array(a.length);
    for (let i = 0; i < res.length; i++) {
      const val = a[i];
      res[i] = util.isValNaN(val, dtype) ? NaN : val;
    }
    return res;
  }
}
