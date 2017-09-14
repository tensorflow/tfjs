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

import {ConvInfo} from './conv_util';
import {MatrixOrientation, NDArrayMath} from './math';
import * as ndarray from './ndarray';
import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from './ndarray';
import {AddScaledMatProgram} from './webgl/addscaledmat_gpu';
import {ArgMaxEqualsProgram} from './webgl/argmaxequals_gpu';
import {ArgMinMaxProgram} from './webgl/argminmax_gpu';
import {BatchNormProgram} from './webgl/batchnorm_gpu';
import * as binaryop_gpu from './webgl/binaryop_gpu';
import {BinaryOpProgram} from './webgl/binaryop_gpu';
import {Concat3DProgram} from './webgl/concat3d_gpu';
// tslint:disable-next-line:max-line-length
import {Conv2DDerBiasProgram, Conv2DDerInputProgram, Conv2DDerWeightsProgram} from './webgl/conv_backprop_gpu';
import {Conv2DProgram} from './webgl/conv_gpu';
import {Copy2DProgram} from './webgl/copy_gpu';
import {GPGPUContext} from './webgl/gpgpu_context';
import * as gpgpu_math from './webgl/gpgpu_math';
import {GPGPUBinary, GPGPUProgram} from './webgl/gpgpu_math';
import * as gpgpu_util from './webgl/gpgpu_util';
import {LogSumExpProgram} from './webgl/logsumexp_gpu';
import {MaxPool2DBackpropProgram} from './webgl/max_pool_backprop_gpu';
import {MinMaxProgram} from './webgl/minmax_gpu';
import {MatMulProgram} from './webgl/mulmat_gpu';
import {Pool2DProgram} from './webgl/pool_gpu';
import {ReduceSumProgram} from './webgl/reducesum_gpu';
import {ResizeBilinear3DProgram} from './webgl/resize_bilinear_gpu';
import {TextureManager} from './webgl/texture_manager';
import * as unary_op from './webgl/unaryop_gpu';
import {UnaryOpProgram} from './webgl/unaryop_gpu';
import * as webgl_util from './webgl/webgl_util';

export class NDArrayMathGPU extends NDArrayMath {
  private gpgpu: GPGPUContext;
  private textureManager: TextureManager;
  private binaryCache: {[key: string]: GPGPUBinary} = {};
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
    const texShape = ndarray.getTextureShapeRC();
    // Pretend the source was in logical shape that matches the texture shape.
    const source = ndarray.as2D(texShape[0], texShape[1]);
    // Do the same for output.
    const output = this.makeOutputArray<Array2D>(texShape);
    this.copy2D(source, [0, 0], texShape, output, [0, 0], texShape);
    // Get back to the original logical shape.
    return output.reshape(ndarray.shape);
  }

  protected slice2DInternal(
      input: Array2D, beginRowCol: [number, number],
      sizeRowCol: [number, number]): Array2D {
    const result = this.makeOutputArray<Array2D>(sizeRowCol);
    this.copy2DInternal(
        input, beginRowCol, sizeRowCol, result, [0, 0], sizeRowCol);
    return result;
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

  protected concat3DInternal(x1: Array3D, x2: Array3D, axis: number): Array3D {
    const program = new Concat3DProgram(x1.shape, x2.shape, axis);
    return this.compileAndRun(program, [x1, x2]);
  }

  protected scaledArrayAddInternal<T extends NDArray>(
      c1: Scalar, a: T, c2: Scalar, b: T) {
    const program = new AddScaledMatProgram(a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b, c1, c2]);
  }

  protected negInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.NEG);
    return this.compileAndRun<T, T>(program, [a]);
  }

  private makeOutputArray<T extends NDArray>(shape: number[]): T {
    const textureShapeRC =
        webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, shape);
    const texture = this.textureManager.acquireTexture(textureShapeRC);
    return NDArray.make<T>(shape, {texture, textureShapeRC});
  }

  private compileAndRun<T extends NDArray, K extends NDArray>(
      program: GPGPUProgram, inputs: T[], output?: K,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void):
      K {
    if (output == null) {
      output = this.makeOutputArray<K>(program.outputShape);
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
    return this.compileAndRun<T, T>(program, [a, b]);
  }

  protected batchNormalization3DInternal(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon: number|null, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D {
    const inputs = [x, mean, variance];

    if (varianceEpsilon == null) {
      varianceEpsilon = 0.000001;
    }

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
    const program = new BinaryOpProgram(binaryop_gpu.DIV, a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b]);
  }

  protected addInternal<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b]);
  }

  protected subInternal<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    return this.compileAndRun<NDArray, T>(program, [a, b]);
  }

  protected logSumExpInternal(a: NDArray): Scalar {
    const program = new LogSumExpProgram(a.size);
    return this.compileAndRun(program, [a]);
  }

  protected expInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.EXP);
    return this.compileAndRun(program, [a]);
  }

  protected logInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.LOG);
    return this.compileAndRun(program, [a]);
  }

  protected sqrtInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.SQRT);
    return this.compileAndRun(program, [a]);
  }

  protected reluInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.RELU);
    return this.compileAndRun(program, [a]);
  }

  protected absInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ABS);
    return this.compileAndRun(program, [a]);
  }

  protected sigmoidInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.SIGMOID);
    return this.compileAndRun<T, T>(program, [a]);
  }

  protected sinInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.SIN);
    return this.compileAndRun(program, [a]);
  }

  protected cosInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.COS);
    return this.compileAndRun(program, [a]);
  }

  protected tanInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.TAN);
    return this.compileAndRun(program, [a]);
  }

  protected asinInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ASIN);
    return this.compileAndRun(program, [a]);
  }

  protected acosInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ACOS);
    return this.compileAndRun(program, [a]);
  }

  protected atanInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.ATAN);
    return this.compileAndRun(program, [a]);
  }

  protected sinhInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.SINH);
    return this.compileAndRun(program, [a]);
  }

  protected coshInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.COSH);
    return this.compileAndRun(program, [a]);
  }

  protected tanhInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.TANH);
    return this.compileAndRun(program, [a]);
  }


  protected stepInternal<T extends NDArray>(a: T): T {
    const program = new UnaryOpProgram(a.shape, unary_op.STEP);
    return this.compileAndRun(program, [a]);
  }

  protected conv2dInternal(
      x: Array3D, filter: Array4D, bias: Array1D|null,
      convInfo: ConvInfo): Array3D {
    const program = new Conv2DProgram(convInfo, bias != null);
    const inputs = bias != null ? [x, filter, bias] : [x, filter];
    return this.compileAndRun(program, inputs);
  }

  protected conv2dDerInputInternal(
      dy: Array3D, filter: Array4D, convInfo: ConvInfo): Array3D {
    const program = new Conv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  protected conv2dDerFilterInternal(
      x: Array3D, dY: Array3D, convInfo: ConvInfo): Array4D {
    const program = new Conv2DDerWeightsProgram(convInfo);
    return this.compileAndRun(program, [x, dY]);
  }

  protected conv2dDerBiasInternal(dY: Array3D): Array1D {
    const program = new Conv2DDerBiasProgram(dY.shape);
    return this.compileAndRun(program, [dY]);
  }

  protected maxPoolInternal(x: Array3D, convInfo: ConvInfo): Array3D {
    const program = new Pool2DProgram(convInfo, 'max', false);
    return this.compileAndRun(program, [x]);
  }

  protected minPoolInternal(x: Array3D, convInfo: ConvInfo): Array3D {
    const program = new Pool2DProgram(convInfo, 'min', false);
    return this.compileAndRun(program, [x]);
  }

  protected avgPoolInternal(x: Array3D, convInfo: ConvInfo): Array3D {
    const program = new Pool2DProgram(convInfo, 'avg', false);
    return this.compileAndRun(program, [x]);
  }

  protected maxPoolBackpropInternal(
      dy: Array3D, x: Array3D, convInfo: ConvInfo): Array3D {
    const getPositions = true;
    const maxPoolPositionsProgram =
        new Pool2DProgram(convInfo, 'max', getPositions);
    const maxPoolPositions: Array3D =
        this.compileAndRun(maxPoolPositionsProgram, [x]);

    const maxPoolBackPropProgram = new MaxPool2DBackpropProgram(convInfo);

    const result =
        this.compileAndRun(maxPoolBackPropProgram, [dy, maxPoolPositions]);
    maxPoolPositions.dispose();
    return result as Array3D;
  }

  protected resizeBilinear3DInternal(
      x: Array3D, newShape2D: [number, number],
      alignCorners: boolean): Array3D {
    const program =
        new ResizeBilinear3DProgram(x.shape, newShape2D, alignCorners);
    return this.compileAndRun(program, [x]);
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
