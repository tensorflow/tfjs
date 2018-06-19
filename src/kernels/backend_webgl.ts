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

import {MemoryInfo, TimingInfo} from '../engine';
import {ENV} from '../environment';
import * as axis_util from '../ops/axis_util';
import {Conv2DInfo} from '../ops/conv_util';
import * as ops from '../ops/ops';
import * as reduce_util from '../ops/reduce_util';
import * as segment_util from '../ops/segment_util';
import {getStridedSlicedInfo} from '../ops/slice_util';
// tslint:disable-next-line:max-line-length
import {DataId, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import * as types from '../types';
import {DataType, DataTypeMap, RecursiveArray, TypedArray} from '../types';
import * as util from '../util';

import {KernelBackend} from './backend';
import * as backend_util from './backend_util';
import {ArgMinMaxProgram} from './webgl/argminmax_gpu';
import {AvgPool2DBackpropProgram} from './webgl/avg_pool_backprop_gpu';
import {BatchNormProgram} from './webgl/batchnorm_gpu';
import * as binaryop_gpu from './webgl/binaryop_gpu';
import {BinaryOpProgram} from './webgl/binaryop_gpu';
import {ClipProgram} from './webgl/clip_gpu';
import {ConcatProgram} from './webgl/concat_gpu';
// tslint:disable-next-line:max-line-length
import {Conv2DDerFilterProgram, Conv2DDerInputProgram} from './webgl/conv_backprop_gpu';
// tslint:disable-next-line:max-line-length
import {DepthwiseConv2DDerFilterProgram, DepthwiseConv2DDerInputProgram} from './webgl/conv_backprop_gpu_depthwise';
import {Conv2DProgram} from './webgl/conv_gpu';
import {DepthwiseConv2DProgram} from './webgl/conv_gpu_depthwise';
import {CumSumProgram} from './webgl/cumsum_gpu';
import {EncodeFloatProgram} from './webgl/encode_float_gpu';
import {FromPixelsProgram} from './webgl/from_pixels_gpu';
import {GatherProgram} from './webgl/gather_gpu';
import {GPGPUContext} from './webgl/gpgpu_context';
import * as gpgpu_math from './webgl/gpgpu_math';
import {GPGPUBinary, GPGPUProgram, TensorData} from './webgl/gpgpu_math';
import * as gpgpu_util from './webgl/gpgpu_util';
import {WhereProgram} from './webgl/logical_gpu';
import {LRNProgram} from './webgl/lrn_gpu';
import {MaxPool2DBackpropProgram} from './webgl/max_pool_backprop_gpu';
import {MatMulProgram} from './webgl/mulmat_gpu';
import {MultinomialProgram} from './webgl/multinomial_gpu';
import {OneHotProgram} from './webgl/onehot_gpu';
import {PadProgram} from './webgl/pad_gpu';
import {Pool2DProgram} from './webgl/pool_gpu';
import {ReduceProgram} from './webgl/reduce_gpu';
// tslint:disable-next-line:max-line-length
import {ResizeBilinearBackpropProgram} from './webgl/resize_bilinear_backprop_gpu';
import {ResizeBilinearProgram} from './webgl/resize_bilinear_gpu';
// tslint:disable-next-line:max-line-length
import {ResizeNearestNeighborProgram} from './webgl/resize_nearest_neighbor_gpu';
import {ReverseProgram} from './webgl/reverse_gpu';
import {SegmentOpProgram} from './webgl/segment_gpu';
import {SliceProgram} from './webgl/slice_gpu';
import {StridedSliceProgram} from './webgl/strided_slice_gpu';
import {TextureData, TextureUsage} from './webgl/tex_util';
import {TextureManager} from './webgl/texture_manager';
import {TileProgram} from './webgl/tile_gpu';
import {TransposeProgram} from './webgl/transpose_gpu';
import * as unary_op from './webgl/unaryop_gpu';
import {UnaryOpProgram} from './webgl/unaryop_gpu';
import {WebGLQuery} from './webgl/webgl_types';
import * as webgl_util from './webgl/webgl_util';

type TimerNode = RecursiveArray<Promise<number>>|Promise<number>;
export interface CPUTimerQuery {
  startMs: number;
  endMs?: number;
}

export interface WebGLMemoryInfo extends MemoryInfo {
  numBytesInGPU: number;
  unreliable: boolean;
}

export interface WebGLTimingInfo extends TimingInfo {
  uploadWaitMs: number;
  downloadWaitMs: number;
}

// Empirically determined constant used to decide the number of bytes on GPU
// before we start paging. The bytes are this constant * screen area * dpi.
const BEFORE_PAGING_CONSTANT = 300;
// Tensors with size <= than this will be uploaded as uniforms, not textures.
export const SIZE_UPLOAD_UNIFORM = 32;

export class MathBackendWebGL implements KernelBackend {
  private texData = new WeakMap<DataId, TextureData>();
  // Maps data ids that have a pending read operation, to list of subscribers.
  private pendingRead = new WeakMap<DataId, Array<(arr: TypedArray) => void>>();
  // List of data ids that are scheduled for disposal, but are waiting on a
  // pending read operation.
  private pendingDisposal = new WeakSet<DataId>();
  // List of data ids that are currently residing on gpu memory. Sorted with
  // least recently used being first.
  private lruDataGPU: DataId[] = [];
  private numBytesInGPU = 0;
  /**
   * Number of bytes allocated on the GPU before we start moving data to cpu.
   * Moving avoids gpu memory leaks and relies on JS's garbage collector.
   */
  private NUM_BYTES_BEFORE_PAGING: number;

  private canvas: HTMLCanvasElement;
  private fromPixelsCanvas: HTMLCanvasElement;

  private programTimersStack: TimerNode[];
  private activeTimers: TimerNode[];
  // Accumulated time spent (including blocking) in uploading data to webgl.
  private uploadWaitMs = 0;
  // Accumulated time spent (including blocking in downloading data from webgl.
  private downloadWaitMs = 0;

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    if (this.texData.has(dataId)) {
      throw new Error('Data buffer is already registered');
    }
    this.texData.set(dataId, {
      shape,
      dtype,
      values: null,
      texture: null,
      texShape: null,
      usage: TextureUsage.RENDER
    });
  }
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error('MathBackendWebGL.writePixels(): pixels can not be null');
    }
    const texShape: [number, number] = [pixels.height, pixels.width];
    const outShape = [pixels.height, pixels.width, numChannels];

    if (pixels instanceof HTMLVideoElement) {
      if (this.fromPixelsCanvas == null) {
        if (!ENV.get('IS_BROWSER')) {
          throw new Error(
              'Can\'t read pixels from HTMLImageElement outside the browser.');
        }
        if (document.readyState !== 'complete') {
          throw new Error(
              'The DOM is not ready yet. Please call tf.fromPixels() ' +
              'once the DOM is ready. One way to do that is to add an event ' +
              'listener for `DOMContentLoaded` on the document object');
        }
        this.fromPixelsCanvas = document.createElement('canvas');
      }
      this.fromPixelsCanvas.width = pixels.width;
      this.fromPixelsCanvas.height = pixels.height;
      this.fromPixelsCanvas.getContext('2d').drawImage(
          pixels, 0, 0, pixels.width, pixels.height);
      pixels = this.fromPixelsCanvas;
    }
    const tempPixelArray = Tensor.make(texShape, {}, 'int32');

    // This is a byte texture with pixels.
    this.texData.get(tempPixelArray.dataId).usage = TextureUsage.PIXELS;
    this.gpgpu.uploadPixelDataToTexture(
        this.getTexture(tempPixelArray.dataId), pixels);
    const program = new FromPixelsProgram(outShape);
    const res = this.compileAndRun(program, [tempPixelArray]);

    tempPixelArray.dispose();

    return res as Tensor3D;
  }
  write(dataId: DataId, values: TypedArray): void {
    if (values == null) {
      throw new Error('MathBackendWebGL.write(): values can not be null');
    }
    this.throwIfNoData(dataId);

    const texData = this.texData.get(dataId);
    const {texture, texShape, usage} = texData;
    if (texture != null) {
      // Release the old texture.
      this.releaseTexture(dataId, texture, texShape, usage);
      texData.texture = null;
      texData.texShape = null;
    }
    texData.usage = TextureUsage.UPLOAD;
    texData.values = values;

    if (!this.delayedStorage) {
      this.uploadToGPU(dataId);
    }
  }
  readSync(dataId: DataId): TypedArray {
    this.throwIfNoData(dataId);
    const texData = this.texData.get(dataId);
    const {shape, texture, values, texShape, dtype} = texData;
    if (values != null) {
      this.cacheOnCPU(dataId);
      return values;
    }
    const shouldTimeProgram = this.activeTimers != null;
    let start: number;
    if (shouldTimeProgram) {
      start = performance.now();
    }

    let float32Values;
    if (ENV.get('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
      float32Values = this.gpgpu.downloadFloat32MatrixFromOutputTexture(
          texture, texShape[0], texShape[1]);
    } else {
      const tmpTarget = Tensor.make(shape, {});
      this.texData.get(tmpTarget.dataId).usage = TextureUsage.DOWNLOAD;

      const tmpInput = Tensor.make(shape, {dataId}, dtype);
      const program = new EncodeFloatProgram(shape);
      const res = this.compileAndRun(program, [tmpInput], tmpTarget);

      const tmpData = this.texData.get(tmpTarget.dataId);
      float32Values =
          this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(
              tmpData.texture, tmpData.texShape[0], tmpData.texShape[1]);

      res.dispose();
      tmpInput.dispose();
      tmpTarget.dispose();
    }

    if (shouldTimeProgram) {
      this.downloadWaitMs += performance.now() - start;
    }
    this.cacheOnCPU(dataId, float32Values);
    return texData.values;
  }
  async read(dataId: DataId): Promise<TypedArray> {
    if (this.pendingRead.has(dataId)) {
      const subscribers = this.pendingRead.get(dataId);
      return new Promise<TypedArray>(resolve => subscribers.push(resolve));
    }
    this.throwIfNoData(dataId);
    const texData = this.texData.get(dataId);
    const {texture, values, texShape} = texData;
    if (values != null) {
      this.cacheOnCPU(dataId);
      return values;
    }
    if (ENV.get('WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED')) {
      const float32Values = await this.gpgpu.downloadMatrixFromTextureAsync(
          texture, texShape[0], texShape[1]);
      this.cacheOnCPU(dataId, float32Values);
      return texData.values;
    }

    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 0) {
      return this.readSync(dataId);
    }

    this.pendingRead.set(dataId, []);
    // Construct an empty query. We're just interested in getting a callback
    // when the GPU command queue has executed until this point in time.
    await this.gpgpu.runQuery(() => {});
    const subscribers = this.pendingRead.get(dataId);
    this.pendingRead.delete(dataId);
    const vals = this.readSync(dataId);
    // Notify all pending reads.
    subscribers.forEach(resolve => resolve(vals));
    if (this.pendingDisposal.has(dataId)) {
      this.pendingDisposal.delete(dataId);
      this.disposeData(dataId);
    }
    return vals;
  }

  async time(f: () => void): Promise<WebGLTimingInfo> {
    const oldActiveTimers = this.activeTimers;
    const newActiveTimers: TimerNode[] = [];

    let outerMostTime = false;
    if (this.programTimersStack == null) {
      this.programTimersStack = newActiveTimers;
      outerMostTime = true;
    } else {
      this.activeTimers.push(newActiveTimers);
    }
    this.activeTimers = newActiveTimers;

    f();

    const flattenedActiveTimers = util.flatten(this.activeTimers);
    this.activeTimers = oldActiveTimers;

    if (outerMostTime) {
      this.programTimersStack = null;
    }

    const kernelMs = await Promise.all(flattenedActiveTimers).then(results => {
      let sum = 0;
      results.forEach(result => sum += result);
      return sum;
    });
    const res: WebGLTimingInfo = {
      uploadWaitMs: this.uploadWaitMs,
      downloadWaitMs: this.downloadWaitMs,
      kernelMs,
      wallMs: null  // will be filled by the engine
    };
    this.uploadWaitMs = 0;
    this.downloadWaitMs = 0;
    return res;
  }
  memory(): WebGLMemoryInfo {
    return {unreliable: false, numBytesInGPU: this.numBytesInGPU} as
        WebGLMemoryInfo;
  }

  private startTimer(): WebGLQuery|CPUTimerQuery {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      return this.gpgpu.beginQuery();
    }
    return {startMs: performance.now(), endMs: null};
  }

  private endTimer(query: WebGLQuery|CPUTimerQuery): WebGLQuery|
      {startMs: number, endMs: number} {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      this.gpgpu.endQuery();
      return query;
    }
    (query as CPUTimerQuery).endMs = performance.now();
    return query;
  }

  private async getQueryTime(query: WebGLQuery|CPUTimerQuery): Promise<number> {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      return this.gpgpu.pollQueryTime(query);
    }
    const timerQuery = query as CPUTimerQuery;
    return timerQuery.endMs - timerQuery.startMs;
  }

  disposeData(dataId: DataId): void {
    if (this.pendingDisposal.has(dataId)) {
      return;
    }
    if (this.pendingRead.has(dataId)) {
      this.pendingDisposal.add(dataId);
      return;
    }
    if (this.texData.has(dataId)) {
      const {texture, texShape, usage} = this.texData.get(dataId);
      if (texture != null) {
        this.releaseTexture(dataId, texture, texShape, usage);
      }
      this.texData.delete(dataId);
    }
  }

  getTexture(dataId: DataId): WebGLTexture {
    this.uploadToGPU(dataId);
    return this.texData.get(dataId).texture;
  }

  private textureManager: TextureManager;
  private binaryCache: {[key: string]: GPGPUBinary} = {};
  private gpgpuCreatedLocally: boolean;

  constructor(private gpgpu?: GPGPUContext, private delayedStorage = true) {
    if (ENV.get('WEBGL_VERSION') < 1) {
      throw new Error('WebGL is not supported on this device');
    }
    if (ENV.get('IS_BROWSER')) {
      this.canvas = document.createElement('canvas');
    }
    if (gpgpu == null) {
      this.gpgpu = new GPGPUContext(gpgpu_util.createWebGLContext(this.canvas));
      this.gpgpuCreatedLocally = true;
    } else {
      this.gpgpuCreatedLocally = false;
    }
    // Use the device screen's resolution as a heuristic to decide on the
    // maximum memory allocated on the GPU before starting to page.
    this.NUM_BYTES_BEFORE_PAGING =
        (window.screen.height * window.screen.width * window.devicePixelRatio) *
        BEFORE_PAGING_CONSTANT;
    this.textureManager = new TextureManager(this.gpgpu);
  }

  getGPGPUContext(): GPGPUContext {
    return this.gpgpu;
  }
  getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number): T {
    const [beginIndex, size] =
        getStridedSlicedInfo(x.shape, begin, end, strides, beginMask, endMask);

    if (size.some(axis => axis === 0)) {
      return ops.tensor([], size) as T;
    }

    const program = new StridedSliceProgram(beginIndex, strides, size);
    return this.compileAndRun(program, [x]);
  }

  reverse<T extends Tensor>(x: T, axis: number[]): T {
    const program = new ReverseProgram(x.shape, axis);
    return this.compileAndRun(program, [x]);
  }

  // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
  concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    const program = new ConcatProgram(a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  neg<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.NEG);
    return this.compileAndRun(program, [x]) as T;
  }

  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D {
    const program = new MatMulProgram(a.shape, b.shape, transposeA, transposeB);
    return this.compileAndRun<Tensor2D, Tensor2D>(program, [a, b]);
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as Tensor;
    return this.compileAndRun(program, [a, b], output) as Tensor;
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
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

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const program = new LRNProgram(x.shape, radius, bias, alpha, beta);
    return this.compileAndRun(program, [x]);
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    const program = new TileProgram(x.shape, reps);
    return this.compileAndRun(program, [x]);
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    const program = new PadProgram(x.shape, paddings, constantValue);
    return this.compileAndRun(program, [x]);
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    const program = new TransposeProgram(x.shape, perm);
    return this.compileAndRun(program, [x]);
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    const program = new GatherProgram(x.shape, indices.size, axis);
    return this.compileAndRun(program, [x, indices]);
  }

  private reduce(
      x: Tensor2D, reduceType: 'all'|'max'|'min'|'sum',
      dtype: DataType): Tensor2D {
    const batchSize = x.shape[0];
    const inSize = x.shape[1];
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {windowSize, inSize, batchSize};
    const program = new ReduceProgram(reduceInfo, reduceType);
    const [rows, cols] = program.outputShape;
    const output = this.makeOutputArray<Tensor2D>([rows, cols], dtype);
    this.compileAndRun(program, [x], output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.reduce(output, reduceType, dtype);
  }

  private argReduce(
      x: Tensor2D, reduceType: 'max'|'min',
      bestIndicesA: Tensor2D = null): Tensor2D {
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
    const output = this.makeOutputArray<Tensor2D>([rows, cols], 'int32');
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

  sum(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    const outputDType = types.sumOutType(x.dtype);
    return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
  }

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    let axis = 0;
    const permutation = axis_util.getAxesPermutation([axis], x.rank);
    let permutedX = x;
    if (permutation != null) {
      permutedX = x.transpose(permutation);
      axis = axis_util.getInnerMostAxes(1, x.rank)[0];
    }

    const outShape =
        segment_util.computeOutShape(permutedX.shape, axis, numSegments);
    const inSize = util.sizeFromShape([permutedX.shape[axis]]);
    const a2D = permutedX.as2D(-1, inSize);
    const outputDType = types.sumOutType(x.dtype);
    let result =
        this.segOpCompute(
                a2D, 'unsortedSegmentSum', segmentIds, outputDType, numSegments)
            .reshape(outShape);
    if (permutation != null) {
      result = result.transpose(axis_util.getUndoAxesPermutation(permutation));
    }
    return result;
  }

  private segOpCompute(
      x: Tensor2D, segOpType: 'unsortedSegmentSum', segmentIds: Tensor1D,
      dtype: DataType, numSegments: number): Tensor2D {
    const batchSize = x.shape[0];
    const inSize = x.shape[1];
    const windowSize =
        segment_util.segOpComputeOptimalWindowSize(inSize, numSegments);
    const segOpInfo = {windowSize, inSize, batchSize, numSegments};
    const program = new SegmentOpProgram(segOpInfo, segOpType);
    const [rows, cols] = program.outputShape;
    const output = this.makeOutputArray<Tensor2D>([rows, cols], dtype);
    this.compileAndRun(program, [x, segmentIds], output);
    // No need to run another GPGPU program.
    if (output.shape[1] === numSegments) {
      return output;
    }
    segmentIds = ops.range(0, numSegments).tile([inSize / windowSize]);
    return this.segOpCompute(output, segOpType, segmentIds, dtype, numSegments);
  }

  argMin(x: Tensor, axis: number): Tensor {
    const axes = [axis];
    axis_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.argReduce(a2D, 'min').reshape(outShape);
  }

  argMax(x: Tensor, axis: number): Tensor {
    const axes = [axis];
    axis_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.argReduce(a2D, 'max').reshape(outShape);
  }

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    if (axis !== x.rank - 1) {
      throw new Error(
          `WebGL cumsum shader expects an inner-most axis=${x.rank - 1} ` +
          `but got axis=${axis}`);
    }
    const program = new CumSumProgram(x.shape, exclusive, reverse);
    return this.compileAndRun(program, [x]);
  }

  equal(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.NOT_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  less(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.LESS, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.LESS_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  greater(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.GREATER, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.GREATER_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  logicalNot<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOGICAL_NOT);
    return this.compileAndRun(program, [x]) as T;
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.LOGICAL_AND, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.LOGICAL_OR, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  where(condition: Tensor, a: Tensor, b: Tensor, dtype: DataType): Tensor {
    const program = new WhereProgram(condition.rank, a.shape, a.rank);
    const output = this.makeOutputArray(program.outputShape, dtype);
    return this.compileAndRun(program, [condition, a, b], output);
  }

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    throw new Error('topKValues GPU not yet implemented!');
  }

  topKIndices(x: Tensor, k: number): Tensor1D {
    throw new Error('topKIndices GPU not yet implemented!');
  }

  min(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('min', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'min', a2D.dtype).reshape(outShape);
  }

  minimum(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MIN, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  mod(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MOD, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  max(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('max', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'max', a2D.dtype).reshape(outShape);
  }

  maximum(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MAX, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  all(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('all', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'all', a2D.dtype).reshape(outShape);
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.SQUARED_DIFFERENCE, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  realDivide(a: Tensor, b: Tensor): Tensor {
    const op = binaryop_gpu.DIV;
    const outputDtype = 'float32';
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, outputDtype);
    return this.compileAndRun<Tensor, Tensor>(program, [a, b], output);
  }

  floorDiv(a: Tensor, b: Tensor): Tensor {
    const op = binaryop_gpu.INT_DIV;
    const outputDtype = 'int32';
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, outputDtype);
    return this.compileAndRun<Tensor, Tensor>(program, [a, b], output);
  }

  add(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as Tensor;
    return this.compileAndRun<Tensor, Tensor>(program, [a, b], output);
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as Tensor;
    return this.compileAndRun<Tensor, Tensor>(program, [a, b], output);
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    const program = new BinaryOpProgram(binaryop_gpu.POW, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as T;
    return this.compileAndRun<Tensor, T>(program, [a, b], output);
  }

  ceil<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.CEIL);
    return this.compileAndRun(program, [x]) as T;
  }

  floor<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.FLOOR);
    return this.compileAndRun(program, [x]) as T;
  }

  sign<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGN);
    return this.compileAndRun(program, [x]) as T;
  }

  round<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ROUND);
    return this.compileAndRun(program, [x]) as T;
  }

  exp<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.EXP);
    return this.compileAndRun(program, [x]) as T;
  }

  expm1<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.EXPM1);
    return this.compileAndRun(program, [x]) as T;
  }

  log<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOG);
    return this.compileAndRun(program, [x]) as T;
  }

  log1p<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOG1P);
    return this.compileAndRun(program, [x]) as T;
  }

  sqrt<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQRT);
    return this.compileAndRun(program, [x]) as T;
  }

  rsqrt<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RSQRT);
    return this.compileAndRun(program, [x]) as T;
  }

  square<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQUARE);
    return this.compileAndRun(program, [x]) as T;
  }

  reciprocal<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RECIPROCAL);
    return this.compileAndRun(program, [x]) as T;
  }

  relu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RELU);
    return this.compileAndRun(program, [x]) as T;
  }

  elu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ELU);
    return this.compileAndRun(program, [x]) as T;
  }

  eluDer<T extends Tensor>(dy: T, y: T): T {
    const program =
        new BinaryOpProgram(binaryop_gpu.ELU_DER, dy.shape, y.shape);
    return this.compileAndRun(program, [dy, y]) as T;
  }

  selu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SELU);
    return this.compileAndRun(program, [x]) as T;
  }

  int<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TO_INT);
    const output = this.makeOutputArray(program.outputShape, 'int32');
    return this.compileAndRun(program, [x], output) as T;
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    const program = new ClipProgram(x.shape, min, max);
    return this.compileAndRun(program, [x]) as T;
  }

  abs<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ABS);
    return this.compileAndRun(program, [x]) as T;
  }

  sigmoid<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGMOID);
    return this.compileAndRun(program, [x]) as T;
  }

  softplus<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SOFTPLUS);
    return this.compileAndRun(program, [x]) as T;
  }

  sin<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIN);
    return this.compileAndRun(program, [x]) as T;
  }

  cos<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COS);
    return this.compileAndRun(program, [x]) as T;
  }

  tan<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TAN);
    return this.compileAndRun(program, [x]) as T;
  }

  asin<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ASIN);
    return this.compileAndRun(program, [x]) as T;
  }

  acos<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ACOS);
    return this.compileAndRun(program, [x]) as T;
  }

  atan<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATAN);
    return this.compileAndRun(program, [x]) as T;
  }

  atan2<T extends Tensor>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.ATAN2, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]) as T;
  }

  sinh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SINH);
    return this.compileAndRun(program, [x]) as T;
  }

  cosh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COSH);
    return this.compileAndRun(program, [x]) as T;
  }

  tanh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TANH);
    return this.compileAndRun(program, [x]) as T;
  }

  asinh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ASINH);
    return this.compileAndRun(program, [x]) as T;
  }

  acosh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ACOSH);
    return this.compileAndRun(program, [x]) as T;
  }

  atanh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATANH);
    return this.compileAndRun(program, [x]) as T;
  }

  erf<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ERF);
    return this.compileAndRun(program, [x]) as T;
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    const program = new UnaryOpProgram(x.shape, unary_op.STEP(alpha));
    return this.compileAndRun(program, [x]) as T;
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new Conv2DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
  }

  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const program = new Conv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new Conv2DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dy]);
  }

  depthwiseConv2D(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const program = new DepthwiseConv2DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
  }

  depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const program = new DepthwiseConv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  depthwiseConv2DDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const program = new DepthwiseConv2DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dy]);
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new Pool2DProgram(convInfo, 'max', false);
    const output =
        this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;
    return this.compileAndRun(program, [x], output);
  }

  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new Pool2DProgram(convInfo, 'avg', false);
    const output = this.makeOutputArray(program.outputShape, 'float32');
    return this.compileAndRun(program, [x], output) as Tensor4D;
  }

  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const getPositions = true;
    const maxPoolPositionsProgram =
        new Pool2DProgram(convInfo, 'max', getPositions);
    const maxPoolPositions: Tensor4D =
        this.compileAndRun(maxPoolPositionsProgram, [x]);

    const maxPoolBackPropProgram = new MaxPool2DBackpropProgram(convInfo);
    const output =
        this.makeOutputArray(maxPoolBackPropProgram.outputShape, x.dtype);
    const result = this.compileAndRun(
        maxPoolBackPropProgram, [dy, maxPoolPositions], output);
    maxPoolPositions.dispose();
    return result as Tensor4D;
  }

  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const avgPoolBackpropProgram = new AvgPool2DBackpropProgram(convInfo);
    const output =
        this.makeOutputArray(avgPoolBackpropProgram.outputShape, x.dtype);
    return this.compileAndRun(avgPoolBackpropProgram, [dy], output) as Tensor4D;
  }

  cast<T extends Tensor<types.Rank>>(x: T, dtype: DataType): T {
    return backend_util.castTensor(x, dtype, this);
  }

  reshape<T extends Tensor<types.Rank>, R extends types.Rank>(
      x: T, shape: types.ShapeMap[R]): Tensor<R> {
    return backend_util.reshapeTensor(x, shape);
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const program =
        new ResizeBilinearProgram(x.shape, newHeight, newWidth, alignCorners);
    return this.compileAndRun(program, [x]);
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean):
      Tensor4D {
    const program = new ResizeBilinearBackpropProgram(dy, x, alignCorners);

    return this.compileAndRun(program, [dy]);
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const program = new ResizeNearestNeighborProgram(
        x.shape, newHeight, newWidth, alignCorners);
    return this.compileAndRun(program, [x]);
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    const probs = normalized ? logits : ops.softmax(logits);
    const batchSize = probs.shape[0];
    const numOutcomes = probs.shape[1];
    const program = new MultinomialProgram(batchSize, numOutcomes, numSamples);
    const output =
        this.makeOutputArray(program.outputShape, 'int32') as Tensor2D;
    const customSetup = program.getCustomSetupFunc(seed);
    return this.compileAndRun(program, [probs], output, customSetup);
  }

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    const program = new OneHotProgram(indices.size, depth, onValue, offValue);
    return this.compileAndRun(program, [indices]);
  }

  private makeOutputArray<T extends Tensor>(shape: number[], dtype: DataType):
      T {
    return Tensor.make(shape, {}, dtype) as T;
  }

  private compileAndRun<T extends Tensor, K extends Tensor>(
      program: GPGPUProgram, inputs: T[], output?: K,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void):
      K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }
    const inputsData: Array<TensorData<T>> = inputs.map(tensor => {
      const texData = this.texData.get(tensor.dataId);
      // Upload small tensors that live on the CPU as uniforms, not as textures.
      if (texData.texture == null && tensor.size <= SIZE_UPLOAD_UNIFORM) {
        return {tensor, texData: null, isUniform: true};
      }
      this.uploadToGPU(tensor.dataId);
      return {tensor, texData, isUniform: false};
    });
    this.uploadToGPU(output.dataId);
    const outputData = {
      tensor: output,
      texData: this.texData.get(output.dataId),
      isUniform: false
    };
    const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
    const binary = this.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(
          this.gpgpu, program, inputsData, outputData);
    });
    const shouldTimeProgram = this.activeTimers != null;
    let query: WebGLQuery|CPUTimerQuery;
    if (shouldTimeProgram) {
      query = this.startTimer();
    }

    gpgpu_math.runProgram(binary, inputsData, outputData, customSetup);

    if (this.numBytesInGPU > this.NUM_BYTES_BEFORE_PAGING) {
      let numBytesToPage = this.numBytesInGPU - this.NUM_BYTES_BEFORE_PAGING;
      while (numBytesToPage > 0) {
        const dataId = this.lruDataGPU.shift();
        const {shape, dtype} = this.texData.get(dataId);
        numBytesToPage -= this.computeBytes(shape, dtype);
        this.read(dataId);
      }
    }

    if (shouldTimeProgram) {
      query = this.endTimer(query);
      this.activeTimers.push(this.getQueryTime(query));
    }
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

  private disposed = false;

  dispose() {
    if (this.disposed) {
      return;
    }
    for (const key in this.binaryCache) {
      this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
    }
    this.textureManager.dispose();
    this.canvas.remove();
    if (this.fromPixelsCanvas != null) {
      this.fromPixelsCanvas.remove();
    }
    if (this.gpgpuCreatedLocally) {
      this.gpgpu.dispose();
    }
    this.disposed = true;
  }

  private throwIfNoData(dataId: DataId) {
    if (!this.texData.has(dataId)) {
      throw new Error(
          `WebGL backend: No data found for this tensor. ` +
          `Did you change your backend in the middle of the program? ` +
          `New backends can't use Tensors created with previous backends`);
    }
  }

  private uploadToGPU(dataId: DataId): void {
    this.throwIfNoData(dataId);
    const texData = this.texData.get(dataId);
    const {shape, values, texture, dtype, usage} = texData;
    if (texture != null) {
      // Array is already on GPU. No-op.
      // Touching the texture.
      this.lruDataGPU.splice(this.lruDataGPU.indexOf(dataId), 1);
      this.lruDataGPU.push(dataId);
      return;
    }
    const shouldTimeProgram = this.activeTimers != null;
    let start: number;
    if (shouldTimeProgram) {
      start = performance.now();
    }
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, shape);
    texData.texShape = texShape;
    const newTexture = this.acquireTexture(dataId, texShape, usage);
    texData.texture = newTexture;
    if (values != null) {
      this.gpgpu.uploadMatrixToTexture(
          newTexture, texShape[0],
          // TODO(smilkov): Propagate the original typed array to gpgpu.
          texShape[1], typedArrayToFloat32(values, dtype));
      // Once uploaded, don't store the values on cpu.
      texData.values = null;
      if (shouldTimeProgram) {
        this.uploadWaitMs += performance.now() - start;
      }
    }
  }

  private cacheOnCPU(dataId: DataId, float32Values?: Float32Array) {
    // In delayed storage mode, when the user reads data, we don't keep a copy
    // on the gpu, to minimize likelihood of memory leak. We re-upload to gpu
    // the next time a gpgpu program needs the texture.
    const dontKeepCopyOnGPU = this.delayedStorage;
    const texData = this.texData.get(dataId);
    const {texture, texShape, dtype, usage} = texData;
    if (dontKeepCopyOnGPU && texture != null) {
      this.releaseTexture(dataId, texture, texShape, usage);
      texData.texture = null;
      texData.texShape = null;
    }
    if (float32Values != null) {
      texData.values = float32ToTypedArray(float32Values, dtype);
    }
  }

  private releaseTexture(
      dataId: DataId, texture: WebGLTexture, texShape: [number, number],
      texType: TextureUsage) {
    const {shape, dtype} = this.texData.get(dataId);
    const idx = this.lruDataGPU.indexOf(dataId);
    if (idx >= 0) {
      this.lruDataGPU.splice(idx, 1);
    }
    this.numBytesInGPU -= this.computeBytes(shape, dtype);
    this.textureManager.releaseTexture(texture, texShape, texType);
  }

  private acquireTexture(
      dataId: DataId, texShape: [number, number],
      texType: TextureUsage): WebGLTexture {
    const {shape, dtype} = this.texData.get(dataId);
    this.lruDataGPU.push(dataId);
    this.numBytesInGPU += this.computeBytes(shape, dtype);
    return this.textureManager.acquireTexture(texShape, texType);
  }

  private computeBytes(shape: number[], dtype: DataType) {
    return util.sizeFromShape(shape) * util.bytesPerElement(dtype);
  }
}

if (ENV.get('IS_BROWSER')) {
  ENV.registerBackend('webgl', () => new MathBackendWebGL(), 2 /* priority */);
}

function float32ToTypedArray<D extends DataType>(
    a: Float32Array, dtype: D): DataTypeMap[D] {
  if (dtype === 'float32') {
    return a;
  } else if (dtype === 'int32' || dtype === 'bool') {
    const result = (dtype === 'int32') ? new Int32Array(a.length) :
                                         new Uint8Array(a.length);
    for (let i = 0; i < result.length; ++i) {
      result[i] = Math.round(a[i]);
    }
    return result;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

function typedArrayToFloat32<D extends DataType>(
    a: DataTypeMap[D], dtype: D): Float32Array {
  return (a instanceof Float32Array) ? a : new Float32Array(a);
}
