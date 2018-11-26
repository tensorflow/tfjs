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

import {getWebGLContext} from '../canvas_util';
import {MemoryInfo, TimingInfo} from '../engine';
import {ENV} from '../environment';
import {tidy} from '../globals';
import {warn} from '../log';
import * as array_ops_util from '../ops/array_ops_util';
import * as axis_util from '../ops/axis_util';
import {computeOutShape} from '../ops/concat_util';
import {Conv2DInfo} from '../ops/conv_util';
import * as gather_nd_util from '../ops/gather_nd_util';
import * as reduce_util from '../ops/reduce_util';
import * as scatter_nd_util from '../ops/scatter_nd_util';
import * as segment_util from '../ops/segment_util';
import {getStridedSlicedInfo} from '../ops/slice_util';
import {softmax} from '../ops/softmax';
import {range, scalar, tensor} from '../ops/tensor_ops';
import {DataId, Scalar, setTensorTracker, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {DataType, DataTypeMap, Rank, RecursiveArray, ShapeMap, sumOutType, TypedArray, upcastType} from '../types';
import * as util from '../util';
import {getTypedArrayFromDType, sizeFromShape} from '../util';

import {DataMover, DataStorage, KernelBackend} from './backend';
import * as backend_util from './backend_util';
import {mergeRealAndImagArrays} from './complex_util';
import {nonMaxSuppressionImpl} from './non_max_suppression_impl';
import {split} from './split_shared';
import {topkImpl} from './topk_impl';
import {ArgMinMaxProgram} from './webgl/argminmax_gpu';
import {AvgPool2DBackpropProgram} from './webgl/avg_pool_backprop_gpu';
import {BatchNormProgram} from './webgl/batchnorm_gpu';
import {BatchNormPackedProgram} from './webgl/batchnorm_packed_gpu';
import * as binaryop_complex_gpu from './webgl/binaryop_complex_gpu';
import {BinaryOpComplexProgram} from './webgl/binaryop_complex_gpu';
import * as binaryop_gpu from './webgl/binaryop_gpu';
import {BinaryOpProgram} from './webgl/binaryop_gpu';
import {ClipProgram} from './webgl/clip_gpu';
import {ComplexAbsProgram} from './webgl/complex_abs_gpu';
import {ConcatProgram} from './webgl/concat_gpu';
import {Conv2DDerFilterProgram, Conv2DDerInputProgram} from './webgl/conv_backprop_gpu';
import {DepthwiseConv2DDerFilterProgram, DepthwiseConv2DDerInputProgram} from './webgl/conv_backprop_gpu_depthwise';
import {Conv2DProgram} from './webgl/conv_gpu';
import {DepthwiseConv2DProgram} from './webgl/conv_gpu_depthwise';
import {CropAndResizeProgram} from './webgl/crop_and_resize_gpu';
import {CumSumProgram} from './webgl/cumsum_gpu';
import {DepthToSpaceProgram} from './webgl/depth_to_space_gpu';
import {EncodeFloatProgram} from './webgl/encode_float_gpu';
import * as fft_gpu from './webgl/fft_gpu';
import {FFTProgram} from './webgl/fft_gpu';
import {FromPixelsProgram} from './webgl/from_pixels_gpu';
import {GatherProgram} from './webgl/gather_gpu';
import {GatherNDProgram} from './webgl/gather_nd_gpu';
import {GPGPUContext} from './webgl/gpgpu_context';
import * as gpgpu_math from './webgl/gpgpu_math';
import {GPGPUBinary, GPGPUProgram, TensorData} from './webgl/gpgpu_math';
import {Im2ColProgram} from './webgl/im2col_gpu';
import {LRNProgram} from './webgl/lrn_gpu';
import {LRNGradProgram} from './webgl/lrn_grad_gpu';
import {MaxPool2DBackpropProgram} from './webgl/max_pool_backprop_gpu';
import {MatMulProgram} from './webgl/mulmat_gpu';
import {MatMulPackedProgram} from './webgl/mulmat_packed_gpu';
import {MultinomialProgram} from './webgl/multinomial_gpu';
import {OneHotProgram} from './webgl/onehot_gpu';
import {PackProgram} from './webgl/pack_gpu';
import {PadProgram} from './webgl/pad_gpu';
import {Pool2DProgram} from './webgl/pool_gpu';
import {ReduceProgram} from './webgl/reduce_gpu';
import {ReshapePackedProgram} from './webgl/reshape_packed_gpu';
import {ResizeBilinearBackpropProgram} from './webgl/resize_bilinear_backprop_gpu';
import {ResizeBilinearProgram} from './webgl/resize_bilinear_gpu';
import {ResizeNearestNeigborBackpropProgram} from './webgl/resize_nearest_neighbor_backprop_gpu';
import {ResizeNearestNeighborProgram} from './webgl/resize_nearest_neighbor_gpu';
import {ReverseProgram} from './webgl/reverse_gpu';
import {ScatterProgram} from './webgl/scatter_gpu';
import {SegmentOpProgram} from './webgl/segment_gpu';
import {SelectProgram} from './webgl/select_gpu';
import {SliceProgram} from './webgl/slice_gpu';
import {StridedSliceProgram} from './webgl/strided_slice_gpu';
import {TextureData, TextureUsage} from './webgl/tex_util';
import {TextureManager} from './webgl/texture_manager';
import {TileProgram} from './webgl/tile_gpu';
import {TransposeProgram} from './webgl/transpose_gpu';
import * as unary_op from './webgl/unaryop_gpu';
import {UnaryOpProgram} from './webgl/unaryop_gpu';
import {UnpackProgram} from './webgl/unpack_gpu';
import * as webgl_util from './webgl/webgl_util';
import {whereImpl} from './where_impl';

type KernelInfo = {
  name: string; query: Promise<number>;
};

export type TimerNode = RecursiveArray<KernelInfo>|KernelInfo;
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

// Combines a dataId, a shape, and a dtype without a Tensor object so that
// programs can be executed without a full Tensor object.
export interface TensorHandle {
  dataId: DataId;
  shape: number[];
  dtype: DataType;
}

// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD = 10;
// Empirically determined constant used to decide the number of bytes on GPU
// before we start paging. The bytes are this constant * screen area * dpi.
const BEFORE_PAGING_CONSTANT = 300;
// Tensors with size <= than this will be uploaded as uniforms, not textures.
export const SIZE_UPLOAD_UNIFORM = 4;
// Empirically determined minimal shared dimension in matmul before we forward
// to a.mul(b).sum() in order to take advantage of GPU parallelism. See
// https://github.com/tensorflow/tfjs-core/pull/1379 for benchmarks.
export const MATMUL_SHARED_DIM_THRESHOLD = 1000;

export class MathBackendWebGL implements KernelBackend {
  private texData: DataStorage<TextureData>;
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
  private fromPixels2DContext: CanvasRenderingContext2D;

  private programTimersStack: TimerNode[];
  private activeTimers: TimerNode[];
  // Accumulated time spent (including blocking) in uploading data to webgl.
  private uploadWaitMs = 0;
  // Accumulated time spent (including blocking in downloading data from webgl.
  private downloadWaitMs = 0;
  private cpuBackend: KernelBackend;

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    if (this.texData.has(dataId)) {
      throw new Error('Data buffer is already registered');
    }
    this.texData.set(dataId, {
      shape,
      dtype,
      values: null,
      texture: null,
      complexTensors: null,
      texShape: null,
      usage: TextureUsage.RENDER,
      isPacked: false
    });
  }

  setDataMover(dataMover: DataMover): void {
    this.texData = new DataStorage(dataMover);
  }

  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error('pixels passed to tf.fromPixels() can not be null');
    }
    const texShape: [number, number] = [pixels.height, pixels.width];
    const outShape = [pixels.height, pixels.width, numChannels];

    if (!(pixels instanceof HTMLVideoElement) &&
        !(pixels instanceof HTMLImageElement) &&
        !(pixels instanceof HTMLCanvasElement) &&
        !(pixels instanceof ImageData)) {
      throw new Error(
          'pixels passed to tf.fromPixels() must be either an ' +
          `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement or ` +
          `ImageData, but was ${(pixels as {}).constructor.name}`);
    }
    if (pixels instanceof HTMLVideoElement) {
      if (this.fromPixels2DContext == null) {
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
        this.fromPixels2DContext =
            document.createElement('canvas').getContext('2d');
      }
      this.fromPixels2DContext.canvas.width = pixels.width;
      this.fromPixels2DContext.canvas.height = pixels.height;
      this.fromPixels2DContext.drawImage(
          pixels, 0, 0, pixels.width, pixels.height);
      pixels = this.fromPixels2DContext.canvas;
    }
    const tempPixelHandle = this.makeTensorHandle(texShape, 'int32');
    // This is a byte texture with pixels.
    this.texData.get(tempPixelHandle.dataId).usage = TextureUsage.PIXELS;
    this.gpgpu.uploadPixelDataToTexture(
        this.getTexture(tempPixelHandle.dataId), pixels);
    const program = new FromPixelsProgram(outShape);
    const res = this.compileAndRun(program, [tempPixelHandle]);

    this.disposeData(tempPixelHandle.dataId);

    return res as Tensor3D;
  }

  private makeTensorHandle(shape: number[], dtype: DataType): TensorHandle {
    const dataId = {};
    this.register(dataId, shape, dtype);
    return {dataId, shape, dtype};
  }

  write(dataId: DataId, values: TypedArray): void {
    if (values == null) {
      throw new Error('MathBackendWebGL.write(): values can not be null');
    }
    const texData = this.texData.get(dataId);
    const {texture, texShape, usage, dtype, isPacked} = texData;
    if (dtype === 'complex64') {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }

    if (texture != null) {
      // Release the old texture.
      this.releaseTexture(dataId, texture, texShape, usage, isPacked);
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
    const texData = this.texData.get(dataId);
    const {values, dtype, complexTensors} = texData;
    if (values != null) {
      return this.convertAndCacheOnCPU(dataId);
    }
    const shouldTimeProgram = this.activeTimers != null;
    let start: number;
    if (shouldTimeProgram) {
      start = performance.now();
    }

    let result: Float32Array;
    if (dtype === 'complex64') {
      const realValues = complexTensors.real.dataSync() as Float32Array;
      const imagValues = complexTensors.imag.dataSync() as Float32Array;
      result = mergeRealAndImagArrays(realValues, imagValues);
    } else {
      result = this.getValuesFromTexture(dataId);
    }

    if (shouldTimeProgram) {
      this.downloadWaitMs += performance.now() - start;
    }
    return this.convertAndCacheOnCPU(dataId, result);
  }

  async read(dataId: DataId): Promise<TypedArray> {
    if (this.pendingRead.has(dataId)) {
      const subscribers = this.pendingRead.get(dataId);
      return new Promise<TypedArray>(resolve => subscribers.push(resolve));
    }
    const texData = this.texData.get(dataId);
    const {texture, values, texShape} = texData;
    if (values != null) {
      return this.convertAndCacheOnCPU(dataId);
    }

    this.pendingRead.set(dataId, []);

    if (!ENV.get('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
        ENV.get('WEBGL_VERSION') === 2) {
      throw new Error(
          `tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and ` +
          `WEBGL_VERSION=2 not yet supported.`);
    }

    // Possibly copy the texture into a buffer before inserting a fence.
    const bufferOrTexture = this.gpgpu.maybeCreateBufferFromTexture(
        texture, texShape[0], texShape[1]);

    // Create a fence and wait for it to resolve.
    await this.gpgpu.createAndWaitForFence();

    // Download the values from the GPU.
    let vals: Float32Array;
    if (bufferOrTexture instanceof WebGLTexture) {
      vals = this.getValuesFromTexture(dataId);
    } else {
      vals = this.gpgpu.downloadFloat32MatrixFromBuffer(
          bufferOrTexture, texShape[0], texShape[1]);
    }
    const dTypeVals = this.convertAndCacheOnCPU(dataId, vals);

    const subscribers = this.pendingRead.get(dataId);
    this.pendingRead.delete(dataId);

    // Notify all pending reads.
    subscribers.forEach(resolve => resolve(dTypeVals));
    if (this.pendingDisposal.has(dataId)) {
      this.pendingDisposal.delete(dataId);
      this.disposeData(dataId);
    }
    return dTypeVals;
  }

  private getValuesFromTexture(dataId: DataId): Float32Array {
    const {shape, dtype, texture, texShape} = this.texData.get(dataId);
    if (ENV.get('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
      if (this.texData.get(dataId).isPacked) {
        const batch = util.sizeFromShape(shape.slice(0, shape.length - 2));
        const rows = shape.length > 1 ? shape[shape.length - 2] : 1;
        const cols = shape[shape.length - 1];
        return this.gpgpu.downloadMatrixFromPackedTexture(
            texture, batch, rows, cols, texShape[0], texShape[1]);
      } else {
        return this.gpgpu.downloadFloat32MatrixFromOutputTexture(
            texture, texShape[0], texShape[1]);
      }
    }

    const tmpTarget = this.makeTensorHandle(shape, 'float32') as TensorHandle &
        {size: number};
    tmpTarget.size = sizeFromShape(shape);
    this.texData.get(tmpTarget.dataId).usage = TextureUsage.DOWNLOAD;
    const program = new EncodeFloatProgram(shape);
    const pageToCpu = false;
    this.compileAndRun(
        program, [{shape, dtype, dataId}], tmpTarget, null, pageToCpu);
    const tmpData = this.texData.get(tmpTarget.dataId);
    const vals = this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(
        tmpData.texture, tmpData.texShape[0], tmpData.texShape[1]);
    this.disposeData(tmpTarget.dataId);

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

    // needing to split these up because util.flatten only accepts certain types
    const flattenedActiveTimerQueries =
        util.flatten(this.activeTimers.map((d: KernelInfo) => d.query))
            .filter(d => d != null);
    const flattenedActiveTimerNames =
        util.flatten(this.activeTimers.map((d: KernelInfo) => d.name))
            .filter(d => d != null);

    this.activeTimers = oldActiveTimers;

    if (outerMostTime) {
      this.programTimersStack = null;
    }

    const kernelMs = await Promise.all(flattenedActiveTimerQueries);

    const res: WebGLTimingInfo = {
      uploadWaitMs: this.uploadWaitMs,
      downloadWaitMs: this.downloadWaitMs,
      kernelMs: util.sum(kernelMs),
      getExtraProfileInfo: () =>
          kernelMs.map((d, i) => ({name: flattenedActiveTimerNames[i], ms: d}))
              .map(d => `${d.name}: ${d.ms}`)
              .join(', '),
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

  private endTimer(query: WebGLQuery|CPUTimerQuery): WebGLQuery|CPUTimerQuery {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      this.gpgpu.endQuery();
      return query;
    }
    (query as CPUTimerQuery).endMs = performance.now();
    return query;
  }

  private async getQueryTime(query: WebGLQuery|CPUTimerQuery): Promise<number> {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      return this.gpgpu.waitForQueryAndGetTime(query as WebGLQuery);
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
      const {texture, texShape, usage, complexTensors, isPacked} =
          this.texData.get(dataId);
      if (texture != null) {
        this.releaseTexture(dataId, texture, texShape, usage, isPacked);
      }
      if (complexTensors != null) {
        complexTensors.real.dispose();
        complexTensors.imag.dispose();
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

    if (gpgpu == null) {
      const gl = getWebGLContext(ENV.get('WEBGL_VERSION'));
      this.gpgpu = new GPGPUContext(gl);
      this.canvas = gl.canvas;
      this.gpgpuCreatedLocally = true;
    } else {
      this.gpgpuCreatedLocally = false;
      this.canvas = gpgpu.gl.canvas;
    }
    if (ENV.get('WEBGL_PAGING_ENABLED')) {
      // Use the device screen's resolution as a heuristic to decide on the
      // maximum memory allocated on the GPU before starting to page.
      this.NUM_BYTES_BEFORE_PAGING =
          (window.screen.height * window.screen.width *
           window.devicePixelRatio) *
          BEFORE_PAGING_CONSTANT;
    }
    this.textureManager = new TextureManager(this.gpgpu);
  }

  private getCPUBackend(): KernelBackend|null {
    if (!ENV.get('WEBGL_CPU_FORWARD')) {
      return null;
    }

    if (this.cpuBackend == null) {
      this.cpuBackend = ENV.findBackend('cpu');
    }

    return this.cpuBackend;
  }

  /*
  Tests whether all the inputs to an op are small and on the CPU. This heuristic
  determines when it would be faster to execute a kernel on the CPU. WebGL
  kernels opt into running this check and forwarding when appropriate.
  TODO(https://github.com/tensorflow/tfjs/issues/872): Develop a more
  sustainable strategy for optimizing backend execution of ops.
   */
  private shouldExecuteOnCPU(
      inputs: Tensor[], sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD): boolean {
    return this.getCPUBackend() != null &&
        inputs.every(
            input => this.texData.get(input.dataId).texture == null &&
                input.size < sizeThreshold);
  }

  getGPGPUContext(): GPGPUContext {
    return this.gpgpu;
  }
  getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  complex<T extends Tensor>(real: T, imag: T): T {
    const result = this.makeOutputArray(real.shape, 'complex64') as T;
    const resultData = this.texData.get(result.dataId);
    // The backend owns the reference to the underlying real and imaginary
    // clones. These will explicitly get disposed when the complex tensor is
    // disposed.
    resultData.complexTensors = {
      real: ENV.engine.keep(real.clone()),
      imag: ENV.engine.keep(imag.clone())
    };

    return result;
  }
  real<T extends Tensor>(input: T): T {
    const resultData = this.texData.get(input.dataId);
    return resultData.complexTensors.real.clone() as T;
  }
  imag<T extends Tensor>(input: T): T {
    const resultData = this.texData.get(input.dataId);
    return resultData.complexTensors.imag.clone() as T;
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    if (this.shouldExecuteOnCPU([x])) {
      return this.cpuBackend.slice(x, begin, size);
    }

    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number, ellipsisMask: number,
      newAxisMask: number, shrinkAxisMask: number): T {
    if (this.shouldExecuteOnCPU([x])) {
      return this.cpuBackend.stridedSlice(
          x, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask,
          shrinkAxisMask);
    }

    const [beginIndex, size, shrinkAxis] = getStridedSlicedInfo(
        x.shape, begin, end, strides, beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask);

    const shape = size.filter((v, index) => shrinkAxis.indexOf(index) === -1);
    if (shape.some(axis => axis === 0)) {
      return tensor([], shape) as T;
    }

    const program =
        new StridedSliceProgram(beginIndex, strides, size, shrinkAxis);
    return this.compileAndRun(program, [x]);
  }

  reverse<T extends Tensor>(x: T, axis: number[]): T {
    const program = new ReverseProgram(x.shape, axis);
    return this.compileAndRun(program, [x]);
  }

  private concat2Tensors<T extends Tensor>(a: T, b: T, axis: number): T {
    // Any concat of n-dimensional tensors across any axis can be reduced to
    // a concatenation of two-dimensional tensors across the axis 1 by first
    // partitioning the axes of the original tensors into those less than the
    // axis to be concatenated and the rest. Then reshape the tensors
    // into a two-dimensional tensor by collapsing these two sets of axes and
    // concatenate the resulting matrices across the axis 1, finally reshaping
    // the result to have the proper shape.
    const outShape = computeOutShape([a.shape, b.shape], axis);
    const a2D = a.as2D(-1, sizeFromShape(a.shape.slice(axis)));
    const b2D = b.as2D(-1, sizeFromShape(b.shape.slice(axis)));
    const program = new ConcatProgram(a2D.shape, b2D.shape);
    const res = this.compileAndRun(program, [a2D, b2D]) as Tensor;
    return res.reshape(outShape) as T;
  }

  concat(tensors: Tensor[], axis: number): Tensor {
    if (this.shouldExecuteOnCPU(tensors)) {
      return this.cpuBackend.concat(tensors, axis);
    }

    if (tensors.length === 1) {
      return tensors[0];
    }
    let result = tensors[0];
    for (let i = 1; i < tensors.length; ++i) {
      result = this.concat2Tensors(result, tensors[i], axis);
    }
    return result;
  }

  neg<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.NEG);
    return this.compileAndRun(program, [x]) as T;
  }

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D {
    const outerShapeA = transposeA ? a.shape[2] : a.shape[1];
    const outerShapeB = transposeB ? b.shape[1] : b.shape[2];
    const sharedDim = transposeA ? a.shape[1] : a.shape[2];
    const [batch, , ] = a.shape;

    // Since the matrices are vectors, it is faster to call mul().sum()
    // because sum() is O(sqrt(N)) due to divide-and-conquer.
    if ((outerShapeA === 1 || outerShapeB === 1) &&
        sharedDim > MATMUL_SHARED_DIM_THRESHOLD) {
      if (transposeA) {
        a = a.transpose([0, 2, 1]);
      }
      if (transposeB) {
        b = b.transpose([0, 2, 1]);
      }

      const a3D = outerShapeB === 1 ? a : a.as3D(batch, sharedDim, 1);
      const axis = outerShapeB === 1 ? 2 : 1;
      const b3D = outerShapeB === 1 ? b.as3D(batch, 1, sharedDim) : b;
      return this.multiply(a3D, b3D).sum(axis, true /* keepDims */);
    }

    // TODO(https://github.com/tensorflow/tfjs/issues/693): Support 3D tensors
    if (batch === 1) {
      const aSqueezed = a.as2D(a.shape[1], a.shape[2]);
      const bSqueezed = b.as2D(b.shape[1], b.shape[2]);

      const program = new MatMulPackedProgram(
          aSqueezed.shape, bSqueezed.shape, [outerShapeA, outerShapeB],
          transposeA, transposeB);
      const result =
          this.compileAndRun<Tensor2D>(program, [aSqueezed, bSqueezed]);

      return result.reshape([1, result.shape[0], result.shape[1]]);
    } else {
      return this.compileAndRun(
          new MatMulProgram(a.shape, b.shape, transposeA, transposeB), [a, b]);
    }
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64') {
      const aData = this.texData.get(a.dataId);
      const bData = this.texData.get(b.dataId);

      const realProgram = new BinaryOpComplexProgram(
          binaryop_complex_gpu.COMPLEX_MULTIPLY.REAL, a.shape, b.shape);
      const imagProgram = new BinaryOpComplexProgram(
          binaryop_complex_gpu.COMPLEX_MULTIPLY.IMAG, a.shape, b.shape);

      const inputs = [
        this.makeComplexComponentTensorHandle(a, aData.complexTensors.real),
        this.makeComplexComponentTensorHandle(a, aData.complexTensors.imag),
        this.makeComplexComponentTensorHandle(b, bData.complexTensors.real),
        this.makeComplexComponentTensorHandle(b, bData.complexTensors.imag)
      ];
      const real = this.compileAndRun<Tensor>(realProgram, inputs);
      const imag = this.compileAndRun<Tensor>(imagProgram, inputs);

      const complex = this.complex(real, imag);
      real.dispose();
      imag.dispose();
      return complex;
    }

    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.multiply(a, b);
    }

    const program = new BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, a.dtype) as Tensor;
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

    if (ENV.get('WEBGL_PACK_BATCHNORMALIZATION')) {
      const batchNormPackedProgram = new BatchNormPackedProgram(
          x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
          varianceEpsilon);
      return this.compileAndRun<Tensor4D>(batchNormPackedProgram, inputs);
    }

    const batchNormProgram = new BatchNormProgram(
        x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
        varianceEpsilon);
    return this.compileAndRun(batchNormProgram, inputs);
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const program = new LRNProgram(x.shape, radius, bias, alpha, beta);
    return this.compileAndRun(program, [x]);
  }

  LRNGrad(
      dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D,
      depthRadius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const program =
        new LRNGradProgram(inputImage.shape, depthRadius, bias, alpha, beta);
    return this.compileAndRun(program, [inputImage, outputImage, dy]);
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

  batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T {
    util.assert(
        x.rank <= 4,
        'batchToSpaceND for rank > 4 with a WebGL backend not implemented yet');
    const prod = blockShape.reduce((a, b) => a * b);

    const reshaped = array_ops_util.getReshaped(x.shape, blockShape, prod);
    const permuted =
        array_ops_util.getPermuted(reshaped.length, blockShape.length);
    const reshapedPermuted =
        array_ops_util.getReshapedPermuted(x.shape, blockShape, prod);
    const sliceBeginCoords =
        array_ops_util.getSliceBeginCoords(crops, blockShape.length);
    const sliceSize =
        array_ops_util.getSliceSize(reshapedPermuted, crops, blockShape.length);

    return x.reshape(reshaped)
               .transpose(permuted)
               .reshape(reshapedPermuted)
               .slice(sliceBeginCoords, sliceSize) as T;
  }

  spaceToBatchND<T extends Tensor>(
      x: T, blockShape: number[], paddings: Array<[number, number]>): T {
    util.assert(
        x.rank <= 4,
        'spaceToBatchND for rank > 4 with a WebGL backend not implemented yet');

    const prod = blockShape.reduce((a, b) => a * b);

    const completePaddings: Array<[number, number]> = [[0, 0]];
    completePaddings.push(...paddings);
    for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
      completePaddings.push([0, 0]);
    }

    const paddedX = x.pad(completePaddings);

    const reshapedPaddedShape =
        array_ops_util.getReshaped(paddedX.shape, blockShape, prod, false);

    const permutedReshapedPaddedPermutation = array_ops_util.getPermuted(
        reshapedPaddedShape.length, blockShape.length, false);

    const flattenShape = array_ops_util.getReshapedPermuted(
        paddedX.shape, blockShape, prod, false);

    return paddedX.reshape(reshapedPaddedShape)
               .transpose(permutedReshapedPaddedPermutation)
               .reshape(flattenShape) as T;
  }

  private reduce(
      x: Tensor2D, reduceType: 'all'|'any'|'max'|'min'|'sum'|'prod',
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
    const outputDType = sumOutType(x.dtype);
    return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
  }

  prod(x: Tensor, axes: number[]): Tensor {
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    const outputDType = sumOutType(x.dtype);
    return this.reduce(a2D, 'prod', outputDType).reshape(outShape);
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
    const outputDType = sumOutType(x.dtype);
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
    segmentIds = range(0, numSegments).tile([inSize / windowSize]);
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
    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.less(a, b);
    }

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
    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.greater(a, b);
    }

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

  select(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    const program = new SelectProgram(condition.rank, a.shape, a.rank);
    const output =
        this.makeOutputArray(program.outputShape, upcastType(a.dtype, b.dtype));
    return this.compileAndRun(program, [condition, a, b], output);
  }

  where(condition: Tensor): Tensor2D {
    warn(
        'tf.where() in webgl locks the UI thread. ' +
        'Call tf.whereAsync() instead');
    const condVals = condition.dataSync();
    return whereImpl(condition.shape, condVals);
  }

  topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T] {
    const xVals = x.dataSync();
    return topkImpl(xVals, x.shape, x.dtype, k, sorted);
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
    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.minimum(a, b);
    }

    const program = new BinaryOpProgram(binaryop_gpu.MIN, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  mod(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MOD, a.shape, b.shape);
    const customSetup = program.getCustomSetupFunc();
    return this.compileAndRun(program, [a, b], null, customSetup);
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
    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.maximum(a, b);
    }

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

  any(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('any', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'any', a2D.dtype).reshape(outShape);
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
    return this.compileAndRun<Tensor>(program, [a, b], output);
  }

  floorDiv(a: Tensor, b: Tensor): Tensor {
    const op = binaryop_gpu.INT_DIV;
    const outputDtype = 'int32';
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, outputDtype);
    return this.compileAndRun<Tensor>(program, [a, b], output);
  }

  add(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' && b.dtype === 'complex64') {
      return this.complexSeparableBinaryOp(a, b, binaryop_gpu.ADD);
    }

    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, upcastType(a.dtype, b.dtype)) as Tensor;
    return this.compileAndRun<Tensor>(program, [a, b], output);
  }

  /**
   * Computes a complex binary operation that can be decomposed into a simple
   * binary operation on both the real and imagary parts.
   */
  private complexSeparableBinaryOp(a: Tensor, b: Tensor, op: string): Tensor {
    const aData = this.texData.get(a.dataId);
    const bData = this.texData.get(b.dataId);

    const [real, imag] = [
      [aData.complexTensors.real, bData.complexTensors.real],
      [aData.complexTensors.imag, bData.complexTensors.imag]
    ].map(complexParts => {
      const [aPart, bPart] = complexParts;

      const program = new BinaryOpProgram(op, a.shape, b.shape);
      const output = this.makeOutputArray(
                         program.outputShape,
                         upcastType(aPart.dtype, bPart.dtype)) as Tensor;

      const aHandle = this.makeComplexComponentTensorHandle(a, aPart);
      const bHandle = this.makeComplexComponentTensorHandle(b, bPart);

      return this.compileAndRun<Tensor>(program, [aHandle, bHandle], output);
    });

    const complex = this.complex(real, imag);
    real.dispose();
    imag.dispose();
    return complex;
  }

  // Returns a TensorHandle with the complex shape and the dataId of the
  // underlying part. We need to do this because a reshaped complex tensor is
  // not reflected in its parts.
  private makeComplexComponentTensorHandle(
      complexTensor: Tensor, complexPart: Tensor): TensorHandle {
    return {
      dataId: complexPart.dataId,
      dtype: complexPart.dtype,
      shape: complexTensor.shape
    };
  }

  addN<T extends Tensor>(tensors: T[]): T {
    let res = tensors[0];
    for (let i = 1; i < tensors.length; i++) {
      res = this.add(res, tensors[i]) as T;
    }
    return res;
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' && b.dtype === 'complex64') {
      return this.complexSeparableBinaryOp(a, b, binaryop_gpu.SUB);
    }

    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.subtract(a, b);
    }

    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, upcastType(a.dtype, b.dtype)) as Tensor;
    return this.compileAndRun<Tensor>(program, [a, b], output);
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    const program = new BinaryOpProgram(binaryop_gpu.POW, a.shape, b.shape);
    const customSetup = program.getCustomSetupFunc();
    const output = this.makeOutputArray(
                       program.outputShape, upcastType(a.dtype, b.dtype)) as T;
    return this.compileAndRun<T>(program, [a, b], output, customSetup);
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
    const customSetup = program.getCustomSetupFunc();
    return this.compileAndRun(program, [x], null, customSetup) as T;
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

  complexAbs<T extends Tensor>(x: T): T {
    const xData = this.texData.get(x.dataId);

    const program = new ComplexAbsProgram(x.shape);
    const inputs = [
      this.makeComplexComponentTensorHandle(x, xData.complexTensors.real),
      this.makeComplexComponentTensorHandle(x, xData.complexTensors.imag),
    ];

    return this.compileAndRun<Tensor>(program, inputs) as T;
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
    const customSetup = program.getCustomSetupFunc();
    return this.compileAndRun(program, [x], null, customSetup) as T;
  }

  atanh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATANH);
    const customSetup = program.getCustomSetupFunc();
    return this.compileAndRun(program, [x], null, customSetup) as T;
  }

  erf<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ERF);
    return this.compileAndRun(program, [x]) as T;
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    const program = new UnaryOpProgram(x.shape, unary_op.STEP(alpha));
    return this.compileAndRun(program, [x]) as T;
  }

  conv2dWithIm2Row(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    // Rearranges conv2d input so each block to be convolved over forms the
    // column of a new matrix with shape [filterWidth * filterHeight *
    // inChannels, outHeight * outWidth]. The filter is also rearranged so each
    // output channel forms a row of a new matrix with shape [outChannels,
    // filterWidth * filterHeight * inChannels]. The convolution is then
    // computed by multiplying these matrices and reshaping the result.
    const {
      filterWidth,
      filterHeight,
      inChannels,
      outWidth,
      outHeight,
    } = convInfo;

    const sharedDim = filterWidth * filterHeight * inChannels;
    const numCols = outHeight * outWidth;
    const x2ColShape = [sharedDim, numCols];

    const xSqueezed = x.squeeze([0]);
    const w2Row = filter.reshape([sharedDim, -1]) as Tensor2D;

    const im2ColProgram =
        new Im2ColProgram(x2ColShape, xSqueezed.shape, convInfo);
    const im2Col = this.compileAndRun<Tensor2D>(im2ColProgram, [xSqueezed]);

    const matmulProgram = new MatMulPackedProgram(
        im2Col.shape, w2Row.shape, [numCols, convInfo.outChannels], true,
        false);
    const product =
        this.compileAndRun<Tensor4D>(matmulProgram, [im2Col, w2Row]);

    return product.reshape([1, outHeight, outWidth, convInfo.outChannels]);
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (ENV.get('WEBGL_CONV_IM2COL') && x.shape[0] === 1) {
      return this.conv2dWithIm2Row(x, filter, convInfo);
    }
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

  cast<T extends Tensor>(x: T, dtype: DataType): T {
    return backend_util.castTensor(x, dtype, this);
  }

  reshape<R extends Rank>(x: Tensor, shape: ShapeMap[R]): Tensor<R> {
    if (this.texData.get(x.dataId).isPacked &&
        !webgl_util.isReshapeFree(x.shape, shape)) {
      return this.packedReshape(x, shape);
    }
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

  resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D {
    const program =
        new ResizeNearestNeigborBackpropProgram(dy, x, alignCorners);
    return this.compileAndRun(program, [dy]);
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    const probs = normalized ? logits : softmax(logits);
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

  nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold: number): Tensor1D {
    warn(
        'tf.nonMaxSuppression() in webgl locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');
    const boxesVals = boxes.dataSync();
    const scoresVals = scores.dataSync();
    return nonMaxSuppressionImpl(
        boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
  }

  cropAndResize(
      image: Tensor4D, boxes: Tensor2D, boxIndex: Tensor1D,
      cropSize: [number, number], method: 'bilinear'|'nearest',
      extrapolationValue: number): Tensor4D {
    const program = new CropAndResizeProgram(
        image.shape, boxes.shape, cropSize, method, extrapolationValue);
    return this.compileAndRun(program, [image, boxes, boxIndex]);
  }

  depthToSpace(x: Tensor4D, blockSize: number, dataFormat: 'NHWC'|'NCHW'):
      Tensor4D {
    util.assert(
        blockSize > 1,
        `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);

    const batchSize = x.shape[0];
    const inputHeight = (dataFormat === 'NHWC') ? x.shape[1] : x.shape[2];
    const inputWidth = (dataFormat === 'NHWC') ? x.shape[2] : x.shape[3];
    const inputDepth = (dataFormat === 'NHWC') ? x.shape[3] : x.shape[1];

    const outputHeight = inputHeight * blockSize;
    const outputWidth = inputWidth * blockSize;
    const outputDepth = inputDepth / (blockSize * blockSize);

    const outputShape = (dataFormat === 'NHWC') ?
        [batchSize, outputHeight, outputWidth, outputDepth] :
        [batchSize, outputDepth, outputHeight, outputWidth];

    const program = new DepthToSpaceProgram(outputShape, blockSize, dataFormat);
    return this.compileAndRun(program, [x]);
  }

  split<T extends Tensor>(x: T, sizeSplits: number[], axis: number): T[] {
    return split(x, sizeSplits, axis);
  }

  scatterND<R extends Rank>(
      indices: Tensor, updates: Tensor, shape: ShapeMap[R]): Tensor<R> {
    const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
        scatter_nd_util.calculateShapes(updates, indices, shape);

    const flattenShape = [outputSize / sliceSize, sliceSize];
    const flattenIndices = indices.reshape([numUpdates, sliceRank]);
    const flattenX = updates.reshape([numUpdates, sliceSize]);

    if (outputSize === 0) {
      return backend_util.reshapeTensor(tensor([]), shape);
    }
    const defaultValue = scalar(0);
    const program = new ScatterProgram(
        numUpdates, sliceRank, flattenIndices.rank, flattenX.rank, strides,
        flattenShape);
    return (this.compileAndRun(
                program, [flattenX, flattenIndices, defaultValue]) as Tensor)
        .reshape(shape);
  }

  sparseToDense<R extends Rank>(
      sparseIndices: Tensor, sparseValues: Tensor, outputShape: ShapeMap[R],
      defaultValue: Scalar): Tensor<R> {
    const {sliceRank, numUpdates, strides, outputSize} =
        scatter_nd_util.calculateShapes(
            sparseValues, sparseIndices, outputShape);

    const sumDupeIndices = false;
    const program = new ScatterProgram(
        numUpdates, sliceRank, sparseIndices.rank, sparseValues.rank, strides,
        [outputSize, 1], sumDupeIndices);
    return (this.compileAndRun(
                program, [sparseValues, sparseIndices, defaultValue]) as Tensor)
        .reshape(outputShape);
  }

  fft(x: Tensor2D): Tensor2D {
    const inverse = false;
    return this.fftImpl(x, inverse);
  }

  ifft(x: Tensor2D): Tensor2D {
    const inverse = true;
    return this.fftImpl(x, inverse);
  }

  private fftImpl(x: Tensor2D, inverse: boolean): Tensor2D {
    const xData = this.texData.get(x.dataId);

    const realProgram =
        new FFTProgram(fft_gpu.COMPLEX_FFT.REAL, x.shape, inverse);
    const imagProgram =
        new FFTProgram(fft_gpu.COMPLEX_FFT.IMAG, x.shape, inverse);
    const inputs = [
      this.makeComplexComponentTensorHandle(x, xData.complexTensors.real),
      this.makeComplexComponentTensorHandle(x, xData.complexTensors.imag),
    ];

    const real = this.compileAndRun<Tensor>(realProgram, inputs);
    const imag = this.compileAndRun<Tensor>(imagProgram, inputs);
    const complex = this.complex(real, imag).as2D(x.shape[0], x.shape[1]);
    real.dispose();
    imag.dispose();
    return complex;
  }

  gatherND(x: Tensor, indices: Tensor): Tensor<Rank> {
    const indicesShape = indices.shape;
    const sliceRank = indicesShape[indicesShape.length - 1];

    const [resultShape, numSlices, sliceSize, strides] =
        gather_nd_util.prepareAndValidate(x, indices);

    const flattenIndices = indices.reshape([numSlices, sliceRank]);
    const flattenX = x.reshape([x.size / sliceSize, sliceSize]);
    const program =
        new GatherNDProgram(sliceRank, strides, [numSlices, sliceSize]);
    return (this.compileAndRun(program, [flattenX, flattenIndices]) as Tensor)
        .reshape(resultShape);
  }

  private makeOutputArray<T extends Tensor>(shape: number[], dtype: DataType):
      T {
    return Tensor.make(shape, {}, dtype) as T;
  }

  private makePackedTensor<T extends Tensor>(shape: number[]): T {
    const packedTensor = Tensor.make(shape, {});
    this.texData.get(packedTensor.dataId).isPacked = true;
    return packedTensor as T;
  }

  private unpackTensor<T extends Tensor>(input: T): T {
    const program = new UnpackProgram(input.shape);
    return this.compileAndRun(
        program, [input], Tensor.make(program.outputShape, {}));
  }

  private getBatchDim(shape: number[], dimsToSkip = 2): number {
    return util.sizeFromShape(shape.slice(0, shape.length - dimsToSkip));
  }

  private getRowsCols(shape: number[]): [number, number] {
    if (shape.length === 0) {
      throw Error('Cannot get rows and columns of an empty shape array.');
    }

    return [
      shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]
    ];
  }

  private packedReshape<R extends Rank>(input: Tensor, afterShape: ShapeMap[R]):
      Tensor<R> {
    const inputAs3D = input.reshape(
        [this.getBatchDim(input.shape), ...this.getRowsCols(input.shape)]);
    const afterShapeAs3D =
        [this.getBatchDim(afterShape), ...this.getRowsCols(afterShape)];
    const program = new ReshapePackedProgram(
        afterShapeAs3D as [number, number, number],
        inputAs3D.shape as [number, number, number]);
    return this.compileAndRun<Tensor<R>>(program, [inputAs3D])
        .reshape(afterShape);
  }

  public compileAndRun<
      K extends {dtype: DataType, size: number, dataId: {}, shape: number[]}>(
      program: GPGPUProgram, inputs: TensorHandle[], output?: K,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void,
      pageToCpu = true): K {
    if (output == null) {
      if (program.usesPackedTextures) {
        output = this.makePackedTensor(program.outputShape) as {} as K;
      } else {
        output = this.makeOutputArray(program.outputShape, inputs[0].dtype) as
            {} as K;
      }
    }
    if (output.size === 0) {
      // Short-circuit the computation since the result is empty (has 0 in its
      // shape).
      this.texData.get(output.dataId).values =
          getTypedArrayFromDType(output.dtype, 0);
      return output;
    }

    const inputsData: TensorData[] = inputs.map(input => {
      if (input.dtype === 'complex64') {
        throw new Error(
            `GPGPUProgram does not support complex64 input. For complex64 ` +
            `dtypes, please separate the program into real and imaginary ` +
            `parts.`);
      }

      let texData = this.texData.get(input.dataId);

      if (texData.texture == null) {
        // Upload small tensors that live on the CPU as uniforms, not as
        // textures. Do this only when the environment supports 32bit floats due
        // to problems when comparing 16bit floats with 32bit floats.
        // TODO(https://github.com/tensorflow/tfjs/issues/821): Make it possible
        // for packed shaders to sample from uniforms.
        if (!(!texData.isPacked && program.usesPackedTextures) &&
            util.sizeFromShape(input.shape) <=
                ENV.get('WEBGL_SIZE_UPLOAD_UNIFORM')) {
          return {
            shape: input.shape,
            texData: null,
            isUniform: true,
            uniformValues: this.readSync(input.dataId)
          };
        }

        // This ensures that if a packed program's inputs have not yet been
        // uploaded to the GPU, they get uploaded as packed right off the bat.
        if (program.usesPackedTextures) {
          texData.isPacked = true;
          texData.shape = input.shape;
        }
      } else if (texData.isPacked !== !!program.usesPackedTextures) {
        let preProcessProgram: UnpackProgram|PackProgram;
        let processedInput: Tensor;

        // Explicitly specifying output tensors because compileAndRun assumes
        // that programs that take packed inputs produce a packed output.
        if (texData.isPacked) {
          preProcessProgram = new UnpackProgram(input.shape);
          processedInput = this.compileAndRun(
              preProcessProgram, [input],
              Tensor.make(preProcessProgram.outputShape, {}));
        } else {
          preProcessProgram = new PackProgram(input.shape);
          processedInput = this.compileAndRun(
              preProcessProgram, [input], this.makePackedTensor(input.shape));
        }

        texData = this.texData.get(processedInput.dataId);
        input = processedInput;
      } else if (
          texData.isPacked &&
          !webgl_util.isReshapeFree(texData.shape, input.shape)) {
        // This is a special, temporary case where a texture exists for a tensor
        // but the shapes are incompatible (due to packing constraints) because
        // the tensor did not have a chance to go through the packed reshape
        // shader. This only happens when we reshape the *same* tensor to form
        // *distinct* inputs to an op, e.g. dotting a vector with itself. This
        // case will disappear once packed uploading is the default.

        // Temporarily disable delayedStorage so the texture isn't removed from
        // the original input
        this.delayedStorage = false;
        const inputValues = (input as Tensor).dataSync();
        this.delayedStorage = true;

        input = Tensor.make(input.shape, {values: inputValues}, input.dtype);
        texData = this.texData.get(input.dataId);
        texData.isPacked = true;
      }

      this.uploadToGPU(input.dataId);
      return {shape: input.shape, texData, isUniform: false};
    });

    this.uploadToGPU(output.dataId);
    const outputData = {
      shape: output.shape,
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

    if (ENV.get('WEBGL_PAGING_ENABLED') && pageToCpu &&
        this.numBytesInGPU > this.NUM_BYTES_BEFORE_PAGING) {
      let numBytesToPage = this.numBytesInGPU - this.NUM_BYTES_BEFORE_PAGING;
      while (numBytesToPage > 0 && this.lruDataGPU.length > 0) {
        const dataId = this.lruDataGPU.shift();
        const {shape, dtype} = this.texData.get(dataId);
        numBytesToPage -= this.computeBytes(shape, dtype);
        this.read(dataId);
      }
    }

    if (shouldTimeProgram) {
      query = this.endTimer(query);
      this.activeTimers.push(
          {name: program.constructor.name, query: this.getQueryTime(query)});
    }

    if (!ENV.get('WEBGL_LAZILY_UNPACK') &&
        this.texData.get(output.dataId).isPacked && !program.isPackShader) {
      return this.unpackTensor(output as {} as Tensor) as {} as K;
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
    if (this.fromPixels2DContext != null) {
      this.fromPixels2DContext.canvas.remove();
    }
    if (this.gpgpuCreatedLocally) {
      this.gpgpu.dispose();
    }
    this.disposed = true;
  }

  floatPrecision(): number {
    return tidy(() => {
      if (this.abs(scalar(1e-8)).get() > 0) {
        return 32;
      }
      return 16;
    });
  }

  private uploadToGPU(dataId: DataId): void {
    const texData = this.texData.get(dataId);
    const {shape, values, texture, dtype, usage, isPacked} = texData;
    if (texture != null) {
      // Array is already on GPU. No-op.
      // Touching the texture.
      if (ENV.get('WEBGL_PAGING_ENABLED')) {
        const index = this.lruDataGPU.indexOf(dataId);
        if (index >= 0) {
          this.lruDataGPU.splice(this.lruDataGPU.indexOf(dataId), 1);
          this.lruDataGPU.push(dataId);
        }
      }
      return;
    }
    const shouldTimeProgram = this.activeTimers != null;
    let start: number;
    if (shouldTimeProgram) {
      start = performance.now();
    }
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape(shape, isPacked);
    texData.texShape = texShape;
    const newTexture = this.acquireTexture(dataId, texShape, usage, isPacked);
    texData.texture = newTexture;
    if (values != null) {
      // TODO(smilkov): Propagate the original typed array to gpgpu.
      if (isPacked) {
        const batch = this.getBatchDim(shape);
        let rows = 1, cols = 1;
        if (shape.length) {
          [rows, cols] = this.getRowsCols(shape);
        }
        this.gpgpu.uploadMatrixToPackedTexture(
            newTexture, batch, rows, cols, texShape[0], texShape[1],
            typedArrayToFloat32(values, dtype));
      } else {
        this.gpgpu.uploadMatrixToTexture(
            newTexture, texShape[0], texShape[1],
            typedArrayToFloat32(values, dtype));
      }
      // Once uploaded, don't store the values on cpu.
      texData.values = null;
      if (shouldTimeProgram) {
        this.uploadWaitMs += performance.now() - start;
      }
    }
  }

  private convertAndCacheOnCPU(dataId: DataId, float32Values?: Float32Array):
      TypedArray {
    // In delayed storage mode, when the user reads data, we don't keep a
    // copy on the gpu, to minimize likelihood of memory leak. We re-upload
    // to gpu the next time a gpgpu program needs the texture.
    const dontKeepCopyOnGPU = this.delayedStorage;
    const texData = this.texData.get(dataId);
    const {texture, texShape, dtype, usage, isPacked} = texData;
    if (dontKeepCopyOnGPU && texture != null) {
      this.releaseTexture(dataId, texture, texShape, usage, isPacked);
      texData.texture = null;
      texData.texShape = null;
    }
    texData.usage = TextureUsage.UPLOAD;
    if (float32Values != null) {
      texData.values = float32ToTypedArray(float32Values, dtype);
    }
    return texData.values;
  }

  private releaseTexture(
      dataId: DataId, texture: WebGLTexture, texShape: [number, number],
      texType: TextureUsage, isPacked: boolean) {
    const {shape, dtype} = this.texData.get(dataId);

    if (ENV.get('WEBGL_PAGING_ENABLED')) {
      const idx = this.lruDataGPU.indexOf(dataId);
      if (idx >= 0) {
        this.lruDataGPU.splice(idx, 1);
      }
    }
    this.numBytesInGPU -= this.computeBytes(shape, dtype);
    this.textureManager.releaseTexture(texture, texShape, texType, isPacked);
  }

  private acquireTexture(
      dataId: DataId, texShape: [number, number], texType: TextureUsage,
      isPacked: boolean): WebGLTexture {
    const {shape, dtype} = this.texData.get(dataId);
    if (ENV.get('WEBGL_PAGING_ENABLED')) {
      this.lruDataGPU.push(dataId);
    }
    this.numBytesInGPU += this.computeBytes(shape, dtype);
    return this.textureManager.acquireTexture(texShape, texType, isPacked);
  }

  private computeBytes(shape: number[], dtype: DataType) {
    return util.sizeFromShape(shape) * util.bytesPerElement(dtype);
  }
}

if (ENV.get('IS_BROWSER')) {
  ENV.registerBackend(
      'webgl', () => new MathBackendWebGL(), 2 /* priority */,
      setTensorTracker);
}

function float32ToTypedArray<D extends DataType>(
    a: Float32Array, dtype: D): DataTypeMap[D] {
  if (dtype === 'float32' || dtype === 'complex64') {
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
