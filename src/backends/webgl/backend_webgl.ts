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

// Import webgl flags.
import './flags_webgl';

import * as device_util from '../../device_util';
import {ENGINE, MemoryInfo, TimingInfo} from '../../engine';
import {ENV} from '../../environment';
import {tidy} from '../../globals';
import {warn} from '../../log';
import {buffer} from '../../ops/array_ops';
import * as array_ops_util from '../../ops/array_ops_util';
import * as axis_util from '../../ops/axis_util';
import {complex, imag, real} from '../../ops/complex_ops';
import {computeOutShape} from '../../ops/concat_util';
import {Conv2DInfo, Conv3DInfo} from '../../ops/conv_util';
import {Activation, FusedBatchMatMulConfig} from '../../ops/fused_util';
import * as gather_nd_util from '../../ops/gather_nd_util';
import * as reduce_util from '../../ops/reduce_util';
import * as scatter_nd_util from '../../ops/scatter_nd_util';
import * as segment_util from '../../ops/segment_util';
import {computeFlatOffset, getStridedSlicedInfo, isSliceContinous} from '../../ops/slice_util';
import {softmax} from '../../ops/softmax';
import {range, scalar, tensor} from '../../ops/tensor_ops';
import {DataId, Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D} from '../../tensor';
import {BackendValues, DataType, DataTypeMap, NumericDataType, PixelData, Rank, RecursiveArray, ShapeMap, sumOutType, TypedArray, upcastType} from '../../types';
import * as util from '../../util';
import {getArrayFromDType, getTypedArrayFromDType, inferDtype, sizeFromShape} from '../../util';
import {DataStorage, EPSILON_FLOAT16, EPSILON_FLOAT32, KernelBackend} from '../backend';
import * as backend_util from '../backend_util';
import {mergeRealAndImagArrays} from '../complex_util';
import {nonMaxSuppressionImpl} from '../non_max_suppression_impl';
import {split} from '../split_shared';
import {tile} from '../tile_impl';
import {topkImpl} from '../topk_impl';
import {whereImpl} from '../where_impl';

import {AddNProgram} from './addn_gpu';
import {AddNPackedProgram} from './addn_packed_gpu';
import {ArgMinMaxProgram} from './argminmax_gpu';
import {ArgMinMaxPackedProgram} from './argminmax_packed_gpu';
import {AvgPool2DBackpropProgram, AvgPool3DBackpropProgram} from './avg_pool_backprop_gpu';
import {BatchNormProgram} from './batchnorm_gpu';
import {BatchNormPackedProgram} from './batchnorm_packed_gpu';
import * as binaryop_complex_gpu from './binaryop_complex_gpu';
import {BinaryOpComplexProgram} from './binaryop_complex_gpu';
import * as binaryop_gpu from './binaryop_gpu';
import {BinaryOpProgram} from './binaryop_gpu';
import * as binaryop_packed_gpu from './binaryop_packed_gpu';
import {BinaryOpPackedProgram} from './binaryop_packed_gpu';
import {createCanvas, getWebGLContext} from './canvas_util';
import {ClipProgram} from './clip_gpu';
import {ClipPackedProgram} from './clip_packed_gpu';
import {ComplexAbsProgram} from './complex_abs_gpu';
import {ConcatProgram} from './concat_gpu';
import {ConcatPackedProgram} from './concat_packed_gpu';
import {Conv2DDerFilterProgram, Conv2DDerInputProgram, Conv3DDerFilterProgram, Conv3DDerInputProgram} from './conv_backprop_gpu';
import {DepthwiseConv2DDerFilterProgram, DepthwiseConv2DDerInputProgram} from './conv_backprop_gpu_depthwise';
import {Conv2DProgram, Conv3DProgram} from './conv_gpu';
import {DepthwiseConv2DProgram} from './conv_gpu_depthwise';
import {DepthwiseConvPacked2DProgram} from './conv_packed_gpu_depthwise';
import {CropAndResizeProgram} from './crop_and_resize_gpu';
import {CumSumProgram} from './cumsum_gpu';
import {DecodeMatrixProgram} from './decode_matrix_gpu';
import {DecodeMatrixPackedProgram} from './decode_matrix_packed_gpu';
import {DepthToSpaceProgram} from './depth_to_space_gpu';
import {DiagProgram} from './diag_gpu';
import {EncodeFloatProgram} from './encode_float_gpu';
import {EncodeFloatPackedProgram} from './encode_float_packed_gpu';
import {EncodeMatrixProgram} from './encode_matrix_gpu';
import {EncodeMatrixPackedProgram} from './encode_matrix_packed_gpu';
import * as fft_gpu from './fft_gpu';
import {FFTProgram} from './fft_gpu';
import {FillProgram} from './fill_gpu';
import {FromPixelsProgram} from './from_pixels_gpu';
import {FromPixelsPackedProgram} from './from_pixels_packed_gpu';
import {GatherProgram} from './gather_gpu';
import {GatherNDProgram} from './gather_nd_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {GPGPUBinary, GPGPUProgram, TensorData} from './gpgpu_math';
import {Im2ColPackedProgram} from './im2col_packed_gpu';
import {LRNProgram} from './lrn_gpu';
import {LRNGradProgram} from './lrn_grad_gpu';
import {LRNPackedProgram} from './lrn_packed_gpu';
import {MaxPool2DBackpropProgram, MaxPool3DBackpropProgram} from './max_pool_backprop_gpu';
import {MatMulPackedProgram} from './mulmat_packed_gpu';
import {MultinomialProgram} from './multinomial_gpu';
import {OneHotProgram} from './onehot_gpu';
import {PackProgram} from './pack_gpu';
import {PadProgram} from './pad_gpu';
import {PadPackedProgram} from './pad_packed_gpu';
import {Pool2DProgram, Pool3DProgram} from './pool_gpu';
import {ReduceProgram} from './reduce_gpu';
import {ReshapePackedProgram} from './reshape_packed_gpu';
import {ResizeBilinearBackpropProgram} from './resize_bilinear_backprop_gpu';
import {ResizeBilinearProgram} from './resize_bilinear_gpu';
import {ResizeBilinearPackedProgram} from './resize_bilinear_packed_gpu';
import {ResizeNearestNeigborBackpropProgram} from './resize_nearest_neighbor_backprop_gpu';
import {ResizeNearestNeighborProgram} from './resize_nearest_neighbor_gpu';
import {ReverseProgram} from './reverse_gpu';
import {ReversePackedProgram} from './reverse_packed_gpu';
import {ScatterProgram} from './scatter_gpu';
import {SegmentOpProgram} from './segment_gpu';
import {SelectProgram} from './select_gpu';
import {SliceProgram} from './slice_gpu';
import {SlicePackedProgram} from './slice_packed_gpu';
import {StridedSliceProgram} from './strided_slice_gpu';
import * as tex_util from './tex_util';
import {TextureData, TextureUsage} from './tex_util';
import {TextureManager} from './texture_manager';
import {TileProgram} from './tile_gpu';
import {TransposeProgram} from './transpose_gpu';
import {TransposePackedProgram} from './transpose_packed_gpu';
import * as unary_op from './unaryop_gpu';
import {UnaryOpProgram} from './unaryop_gpu';
import * as unary_packed_op from './unaryop_packed_gpu';
import {UnaryOpPackedProgram} from './unaryop_packed_gpu';
import {UnpackProgram} from './unpack_gpu';
import * as webgl_util from './webgl_util';

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

const binaryCaches: {[webGLVersion: string]: {[key: string]: GPGPUBinary}} = {};

function getBinaryCache(webGLVersion: number) {
  if (webGLVersion in binaryCaches) {
    return binaryCaches[webGLVersion];
  }
  binaryCaches[webGLVersion] = {};
  return binaryCaches[webGLVersion];
}

function mapActivationToShaderProgram(
    activation: Activation, packed = false): string {
  if (activation === 'linear') {
    if (packed) {
      return unary_packed_op.LINEAR;
    }
    return unary_op.LINEAR;
  } else if (activation === 'relu') {
    if (packed) {
      return unary_packed_op.RELU;
    }
    return unary_op.RELU;
  } else if (activation === 'prelu') {
    if (packed) {
      return binaryop_packed_gpu.PRELU;
    }
    return binaryop_gpu.PRELU;
  }
  throw new Error(`Activation ${
      activation} has not been implemented for the WebGL backend.`);
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
const CPU_HANDOFF_SIZE_THRESHOLD = 128;

// Empirically determined constant used to decide the number of MB on GPU
// before we warn about high memory use. The MB are this constant * screen area
// * dpi / 1024 / 1024.
const BEFORE_PAGING_CONSTANT = 600;
function numMBBeforeWarning(): number {
  if (ENV.global.screen == null) {
    return 1024;  // 1 GB.
  }
  return (ENV.global.screen.height * ENV.global.screen.width *
          window.devicePixelRatio) *
      BEFORE_PAGING_CONSTANT / 1024 / 1024;
}

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
  // Used to count the number of 'shallow' sliced tensors that point to the
  // same data id.
  private dataRefCount = new WeakMap<DataId, number>();
  private numBytesInGPU = 0;

  private canvas: HTMLCanvasElement;
  private fromPixels2DContext: CanvasRenderingContext2D|
      OffscreenCanvasRenderingContext2D;

  private programTimersStack: TimerNode[];
  private activeTimers: TimerNode[];
  // Accumulated time spent (including blocking) in uploading data to webgl.
  private uploadWaitMs = 0;
  // Accumulated time spent (including blocking in downloading data from webgl.
  private downloadWaitMs = 0;
  private cpuBackend: KernelBackend;

  // Number of bits of precision of this backend.
  private floatPrecisionValue: 32|16;

  private textureManager: TextureManager;
  private binaryCache: {[key: string]: GPGPUBinary};
  private gpgpuCreatedLocally: boolean;
  private numMBBeforeWarning: number;
  private warnedAboutMemory = false;

  constructor(private gpgpu?: GPGPUContext) {
    if (!ENV.getBool('HAS_WEBGL')) {
      throw new Error('WebGL is not supported on this device');
    }

    if (gpgpu == null) {
      const gl = getWebGLContext(ENV.getNumber('WEBGL_VERSION'));
      this.binaryCache = getBinaryCache(ENV.getNumber('WEBGL_VERSION'));
      this.gpgpu = new GPGPUContext(gl);
      this.canvas = gl.canvas;
      this.gpgpuCreatedLocally = true;
    } else {
      this.binaryCache = {};
      this.gpgpuCreatedLocally = false;
      this.canvas = gpgpu.gl.canvas;
    }
    this.textureManager = new TextureManager(this.gpgpu);
    this.numMBBeforeWarning = numMBBeforeWarning();

    this.texData = new DataStorage(this, ENGINE);
  }

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    if (this.texData.has(dataId)) {
      throw new Error('Data buffer is already registered');
    }
    this.texData.set(dataId, {shape, dtype});
  }

  fromPixels(
      pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() can not be null');
    }
    const texShape: [number, number] = [pixels.height, pixels.width];
    const outShape = [pixels.height, pixels.width, numChannels];

    const isCanvas = (typeof (OffscreenCanvas) !== 'undefined' &&
                      pixels instanceof OffscreenCanvas) ||
        (typeof (HTMLCanvasElement) !== 'undefined' &&
         pixels instanceof HTMLCanvasElement);
    const isPixelData = (pixels as PixelData).data instanceof Uint8Array;
    const isImageData =
        typeof (ImageData) !== 'undefined' && pixels instanceof ImageData;
    const isVideo = typeof (HTMLVideoElement) !== 'undefined' &&
        pixels instanceof HTMLVideoElement;
    const isImage = typeof (HTMLImageElement) !== 'undefined' &&
        pixels instanceof HTMLImageElement;

    if (!isCanvas && !isPixelData && !isImageData && !isVideo && !isImage) {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() must be either an ' +
          `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
          `in browser, or OffscreenCanvas, ImageData in webworker` +
          ` or {data: Uint32Array, width: number, height: number}, ` +
          `but was ${(pixels as {}).constructor.name}`);
    }

    if (isImage || isVideo) {
      if (this.fromPixels2DContext == null) {
        if (document.readyState !== 'complete') {
          throw new Error(
              'The DOM is not ready yet. Please call ' +
              'tf.browser.fromPixels() once the DOM is ready. One way to ' +
              'do that is to add an event listener for `DOMContentLoaded` ' +
              'on the document object');
        }
        //@ts-ignore
        this.fromPixels2DContext =
            createCanvas(ENV.getNumber('WEBGL_VERSION')).getContext('2d');
      }
      this.fromPixels2DContext.canvas.width = pixels.width;
      this.fromPixels2DContext.canvas.height = pixels.height;
      this.fromPixels2DContext.drawImage(
          pixels as HTMLVideoElement, 0, 0, pixels.width, pixels.height);
      //@ts-ignore
      pixels = this.fromPixels2DContext.canvas;
    }

    const tempPixelHandle = this.makeTensorHandle(texShape, 'int32');
    // This is a byte texture with pixels.
    this.texData.get(tempPixelHandle.dataId).usage = TextureUsage.PIXELS;
    this.gpgpu.uploadPixelDataToTexture(
        this.getTexture(tempPixelHandle.dataId), pixels as ImageData);
    let program, res;
    if (ENV.getBool('WEBGL_PACK')) {
      program = new FromPixelsPackedProgram(outShape);
      const packedOutput =
          this.makePackedTensor(program.outputShape, tempPixelHandle.dtype);
      res = this.compileAndRun(program, [tempPixelHandle], packedOutput);
    } else {
      program = new FromPixelsProgram(outShape);
      res = this.compileAndRun(program, [tempPixelHandle]);
    }

    this.disposeData(tempPixelHandle.dataId);

    return res as Tensor3D;
  }

  private makeTensorHandle(shape: number[], dtype: DataType): TensorHandle {
    const dataId = {};
    this.register(dataId, shape, dtype);
    return {dataId, shape, dtype};
  }

  write(dataId: DataId, values: BackendValues): void {
    if (values == null) {
      throw new Error('MathBackendWebGL.write(): values can not be null');
    }

    if (ENV.getBool('DEBUG')) {
      for (let i = 0; i < values.length; i++) {
        const num = values[i] as number;
        if (!webgl_util.canBeRepresented(num)) {
          throw Error(`The value ${num} cannot be represented on this device.`);
        }
      }
    }

    const texData = this.texData.get(dataId);
    const {dtype} = texData;
    if (dtype === 'complex64') {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }

    this.releaseGPUData(dataId);
    texData.usage = TextureUsage.UPLOAD;
    texData.values = values;
  }

  readSync(dataId: DataId): BackendValues {
    const texData = this.texData.get(dataId);
    const {values, dtype, complexTensors, slice, shape} = texData;
    if (slice != null) {
      const program = new UnaryOpProgram(shape, unary_op.CLONE);
      const res = this.compileAndRun(program, [{dataId, shape, dtype}]);
      const data = this.readSync(res.dataId);
      (res as Tensor).dispose();
      return data;
    }
    if (values != null) {
      return this.convertAndCacheOnCPU(dataId);
    }
    if (dtype === 'string') {
      return values;
    }
    const shouldTimeProgram = this.activeTimers != null;
    let start: number;
    if (shouldTimeProgram) {
      start = util.now();
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
      this.downloadWaitMs += util.now() - start;
    }
    return this.convertAndCacheOnCPU(dataId, result);
  }

  async read(dataId: DataId): Promise<BackendValues> {
    if (this.pendingRead.has(dataId)) {
      const subscribers = this.pendingRead.get(dataId);
      return new Promise<TypedArray>(resolve => subscribers.push(resolve));
    }
    const texData = this.texData.get(dataId);
    const {values, shape, slice, dtype, complexTensors} = texData;

    if (slice != null) {
      const program = new UnaryOpProgram(shape, unary_op.CLONE);
      const res = this.compileAndRun(program, [{dataId, shape, dtype}]);
      const data = this.read(res.dataId);
      (res as Tensor).dispose();
      return data;
    }

    if (values != null) {
      return this.convertAndCacheOnCPU(dataId);
    }

    if (!ENV.getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
        ENV.getNumber('WEBGL_VERSION') === 2) {
      throw new Error(
          `tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and ` +
          `WEBGL_VERSION=2 not yet supported.`);
    }

    let buffer = null;
    if (dtype !== 'complex64' && ENV.get('WEBGL_BUFFER_SUPPORTED')) {
      // Possibly copy the texture into a buffer before inserting a fence.
      const tmpTarget = this.decode(dataId);

      dataId = tmpTarget.dataId;
      const tmpData = this.texData.get(tmpTarget.dataId);

      buffer = this.gpgpu.createBufferFromTexture(
          tmpData.texture, ...tex_util.getDenseTexShape(shape));
    }

    this.pendingRead.set(dataId, []);

    if (dtype !== 'complex64') {
      // Create a fence and wait for it to resolve.
      await this.gpgpu.createAndWaitForFence();
    }

    // Download the values from the GPU.
    let vals: Float32Array;
    if (dtype === 'complex64') {
      const ps =
          Promise.all([complexTensors.real.data(), complexTensors.imag.data()]);
      const [realValues, imagValues] = await ps;
      vals = mergeRealAndImagArrays(
          realValues as Float32Array, imagValues as Float32Array);
    } else if (buffer == null) {
      vals = this.getValuesFromTexture(dataId);
    } else {
      const size = util.sizeFromShape(shape);

      vals = this.gpgpu.downloadFloat32MatrixFromBuffer(buffer, size);
      this.disposeData(dataId);
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
    const {shape, dtype, isPacked} = this.texData.get(dataId);
    const size = util.sizeFromShape(shape);
    if (ENV.getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
      const tmpTarget = this.decode(dataId);
      const tmpData = this.texData.get(tmpTarget.dataId);
      const vals = this.gpgpu
                       .downloadMatrixFromPackedTexture(
                           tmpData.texture, ...tex_util.getDenseTexShape(shape))
                       .subarray(0, size);

      this.disposeData(tmpTarget.dataId);

      return vals;
    }

    const shouldUsePackedProgram =
        ENV.getBool('WEBGL_PACK') && isPacked === true;
    const outputShape =
        shouldUsePackedProgram ? webgl_util.getShapeAs3D(shape) : shape;
    const tmpTarget =
        this.makeTensorHandle(outputShape, 'float32') as TensorHandle &
        {size: number};
    tmpTarget.size = sizeFromShape(shape);
    this.texData.get(tmpTarget.dataId).usage = TextureUsage.DOWNLOAD;

    const output = tidy(() => {
      const program = shouldUsePackedProgram ?
          new EncodeFloatPackedProgram(
              outputShape as [number, number, number]) :
          new EncodeFloatProgram(outputShape);

      return this.compileAndRun(
          program, [{shape: outputShape, dtype, dataId}], tmpTarget, null);
    });

    const tmpData = this.texData.get(output.dataId);
    const vals =
        this.gpgpu
            .downloadByteEncodedFloatMatrixFromOutputTexture(
                tmpData.texture, tmpData.texShape[0], tmpData.texShape[1])
            .subarray(0, size);
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
    if (ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      return this.gpgpu.beginQuery();
    }
    return {startMs: util.now(), endMs: null};
  }

  private endTimer(query: WebGLQuery|CPUTimerQuery): WebGLQuery|CPUTimerQuery {
    if (ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      this.gpgpu.endQuery();
      return query;
    }
    (query as CPUTimerQuery).endMs = util.now();
    return query;
  }

  private async getQueryTime(query: WebGLQuery|CPUTimerQuery): Promise<number> {
    if (ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
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
    // No-op if already disposed.
    if (!this.texData.has(dataId)) {
      return;
    }

    this.releaseGPUData(dataId);
    const {complexTensors} = this.texData.get(dataId);
    if (complexTensors != null) {
      complexTensors.real.dispose();
      complexTensors.imag.dispose();
    }
    this.texData.delete(dataId);
  }

  private releaseGPUData(dataId: DataId): void {
    const {texture, dtype, texShape, usage, isPacked, slice} =
        this.texData.get(dataId);
    const key = slice && slice.origDataId || dataId;
    const refCount = this.dataRefCount.get(key);
    if (refCount > 1) {
      this.dataRefCount.set(key, refCount - 1);
    } else {
      this.dataRefCount.delete(key);
      if (texture != null) {
        this.numBytesInGPU -= this.computeBytes(texShape, dtype);
        this.textureManager.releaseTexture(texture, texShape, usage, isPacked);
      }
    }
    const texData = this.texData.get(dataId);
    texData.texture = null;
    texData.texShape = null;
    texData.isPacked = false;
    texData.slice = null;
  }

  getTexture(dataId: DataId): WebGLTexture {
    this.uploadToGPU(dataId);
    return this.texData.get(dataId).texture;
  }

  private getCPUBackend(): KernelBackend|null {
    if (!ENV.getBool('WEBGL_CPU_FORWARD')) {
      return null;
    }

    if (this.cpuBackend == null) {
      this.cpuBackend = ENGINE.findBackend('cpu');
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

  complex<T extends Tensor>(real: T, imag: T): T {
    const result = this.makeOutputArray(real.shape, 'complex64') as T;
    const resultData = this.texData.get(result.dataId);
    // The backend owns the reference to the underlying real and imaginary
    // clones. These will explicitly get disposed when the complex tensor is
    // disposed.
    resultData.complexTensors = {
      real: ENGINE.keep(real.clone()),
      imag: ENGINE.keep(imag.clone())
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
    // Short-circuit computation if the slice is zero-sized.
    if (util.sizeFromShape(size) === 0) {
      return tensor([], size, x.dtype) as T;
    }
    const {isPacked} = this.texData.get(x.dataId);
    const isContinous = isSliceContinous(x.shape, begin, size);
    if (isPacked || !isContinous) {
      const program = ENV.getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
          new SlicePackedProgram(size) :
          new SliceProgram(size);
      const customSetup = program.getCustomSetupFunc(begin);
      return this.compileAndRun(program, [x], null, customSetup);
    }
    this.uploadToGPU(x.dataId);
    return this.shallowSlice(x, begin, size) as T;
  }

  private shallowSlice(x: Tensor, begin: number[], size: number[]): Tensor {
    const xTexData = this.texData.get(x.dataId);
    const t = Tensor.make(size, {}, x.dtype, this);
    const newTexData = this.texData.get(t.dataId);
    // Copy texture data from the original tensor.
    Object.assign(newTexData, xTexData);
    newTexData.shape = size;
    newTexData.dtype = x.dtype;
    let flatOffset = computeFlatOffset(begin, x.strides);
    if (xTexData.slice) {
      // We are slicing an already sliced tensor, so we have to accumulate
      // the offset.
      flatOffset += xTexData.slice.flatOffset;
    }
    newTexData.slice = {
      flatOffset,
      // Point to the original dataId, which is used to do ref counting.
      origDataId: xTexData.slice && xTexData.slice.origDataId || x.dataId
    };

    // Increase the ref count for that data bucket.
    const refCount = this.dataRefCount.get(newTexData.slice.origDataId) || 1;
    this.dataRefCount.set(newTexData.slice.origDataId, refCount + 1);

    return t;
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
    const program = ENV.getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
        new ReversePackedProgram(x.shape, axis) :
        new ReverseProgram(x.shape, axis);
    return this.compileAndRun(program, [x]);
  }

  concat(tensors: Tensor[], axis: number): Tensor {
    if (tensors[0].dtype === 'complex64') {
      const reals = tensors.map((t) => real(t));
      const imags = tensors.map((t) => imag(t));
      return complex(this.concat(reals, axis), this.concat(imags, axis));
    }
    if (this.shouldExecuteOnCPU(tensors)) {
      return this.cpuBackend.concat(tensors, axis);
    }

    if (tensors.length === 1) {
      return tensors[0];
    }
    if (tensors.length > ENV.getNumber('WEBGL_MAX_TEXTURES_IN_SHADER')) {
      const midIndex = Math.floor(tensors.length / 2);
      const leftSide = this.concat(tensors.slice(0, midIndex), axis);
      const rightSide = this.concat(tensors.slice(midIndex), axis);
      return this.concat([leftSide, rightSide], axis);
    }
    if (ENV.getBool('WEBGL_PACK_ARRAY_OPERATIONS') && tensors[0].rank > 1) {
      const program = new ConcatPackedProgram(tensors.map(t => t.shape), axis);
      return this.compileAndRun(program, tensors);
    }
    // Any concat of n-dimensional tensors across any axis can be reduced to
    // a concatenation of two-dimensional tensors across the axis 1 by first
    // partitioning the axes of the original tensors into those less than the
    // axis to be concatenated and the rest. Then reshape the tensors
    // into a two-dimensional tensor by collapsing these two sets of axes and
    // concatenate the resulting matrices across the axis 1, finally reshaping
    // the result to have the proper shape.
    const outShape = computeOutShape(tensors.map(t => t.shape), axis);
    const tensors2D =
        tensors.map(t => t.as2D(-1, sizeFromShape(t.shape.slice(axis))));
    const program = new ConcatProgram(tensors2D.map(t => t.shape));
    const res = this.compileAndRun(program, tensors2D) as Tensor;
    return res.reshape(outShape);
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

    const dtype = upcastType(a.dtype, b.dtype);

    const program = new MatMulPackedProgram(
        a.shape, [batch, outerShapeA, outerShapeB], transposeA, transposeB);
    const output =
        this.makePackedTensor(program.outputShape, dtype) as Tensor3D;
    return this.compileAndRun<Tensor3D>(program, [a, b], output);
  }

  fusedBatchMatMul(
      {a, b, transposeA, transposeB, bias, activation, preluActivationWeights}:
          FusedBatchMatMulConfig): Tensor3D {
    const outerShapeA = transposeA ? a.shape[2] : a.shape[1];
    const outerShapeB = transposeB ? b.shape[1] : b.shape[2];
    const [batch, , ] = a.shape;

    const dtype = upcastType(a.dtype, b.dtype);

    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const fusedActivation =
        activation ? mapActivationToShaderProgram(activation, true) : null;
    const program = new MatMulPackedProgram(
        a.shape, [batch, outerShapeA, outerShapeB], transposeA, transposeB,
        hasBias, fusedActivation, hasPreluActivationWeights);
    const output =
        this.makePackedTensor(program.outputShape, dtype) as Tensor3D;
    const inputs: TensorHandle[] = [a, b];
    if (bias) {
      inputs.push(bias);
    }
    if (preluActivationWeights) {
      inputs.push(preluActivationWeights);
    }
    return this.compileAndRun<Tensor3D>(program, inputs, output);
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
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_gpu.MUL, a.dtype);
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

    if (ENV.getBool('WEBGL_PACK_NORMALIZATION')) {
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
    const program = ENV.getBool('WEBGL_PACK_NORMALIZATION') ?
        new LRNPackedProgram(x.shape, radius, bias, alpha, beta) :
        new LRNProgram(x.shape, radius, bias, alpha, beta);
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
    if (x.dtype === 'string') {
      const data = this.readSync(x.dataId) as Uint8Array[];
      const decodedData = data.map(d => util.decodeString(d));
      const buf = buffer(x.shape, x.dtype, decodedData);
      return tile(buf, reps) as T;
    }
    const program = new TileProgram(x.shape, reps);
    return this.compileAndRun(program, [x]);
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    const program = ENV.getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
        new PadPackedProgram(x.shape, paddings, constantValue) :
        new PadProgram(x.shape, paddings, constantValue);
    return this.compileAndRun(program, [x]);
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    if (this.shouldExecuteOnCPU([x])) {
      return this.cpuBackend.transpose(x, perm);
    }
    const program = ENV.getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
        new TransposePackedProgram(x.shape, perm) :
        new TransposeProgram(x.shape, perm);
    return this.compileAndRun(program, [x]);
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    if (this.shouldExecuteOnCPU([x, indices])) {
      return this.cpuBackend.gather(x, indices, axis);
    }
    const program = new GatherProgram(x.shape, indices.size, axis);
    return this.compileAndRun(program, [x, indices]);
  }

  batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T {
    util.assert(
        x.rank <= 4,
        () => 'batchToSpaceND for rank > 4 with a WebGL backend not ' +
            'implemented yet');
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
        () => 'spaceToBatchND for rank > 4 with a WebGL backend not ' +
            'implemented yet');

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

  private argReducePacked(
      x: Tensor, reduceType: 'max'|'min', bestIndicesA: Tensor = null): Tensor {
    const inShape = bestIndicesA != null ? bestIndicesA.shape : x.shape;
    const inSize = inShape[inShape.length - 1];
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const program = new ArgMinMaxPackedProgram(
        inShape, windowSize, reduceType, bestIndicesA == null);
    const output = this.makePackedTensor(program.outputShape, 'int32');
    const inputs = bestIndicesA == null ? [x] : [x, bestIndicesA];
    this.compileAndRun(program, inputs, output);
    if (output.rank === x.rank) {
      return this.argReducePacked(x, reduceType, output);
    }
    return output;
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
    if (this.shouldExecuteOnCPU([x])) {
      return this.cpuBackend.prod(x, axes);
    }

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

  private argMinMaxReduce(x: Tensor, axis: number, reduceType: 'min'|'max'):
      Tensor {
    const axes = [axis];
    axis_util.assertAxesAreInnerMostDims(
        'arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes,
        x.rank);
    if (!ENV.getBool('WEBGL_PACK_REDUCE') || x.rank <= 2) {
      const [outShape, reduceShape] =
          axis_util.computeOutAndReduceShapes(x.shape, axes);
      const inSize = util.sizeFromShape(reduceShape);
      const a2D = x.as2D(-1, inSize);
      return this.argReduce(a2D, reduceType).reshape(outShape);
    }
    return this.argReducePacked(x, reduceType);
  }

  argMin(x: Tensor, axis: number): Tensor {
    return this.argMinMaxReduce(x, axis, 'min');
  }

  argMax(x: Tensor, axis: number): Tensor {
    return this.argMinMaxReduce(x, axis, 'max');
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
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.EQUAL, 'bool');
    }
    const program = new BinaryOpProgram(binaryop_gpu.EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.NOT_EQUAL, 'bool');
    }
    const program =
        new BinaryOpProgram(binaryop_gpu.NOT_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  less(a: Tensor, b: Tensor): Tensor {
    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.less(a, b);
    }

    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.LESS, 'bool');
    }

    const program = new BinaryOpProgram(binaryop_gpu.LESS, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.LESS_EQUAL, 'bool');
    }
    const program =
        new BinaryOpProgram(binaryop_gpu.LESS_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  greater(a: Tensor, b: Tensor): Tensor {
    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.greater(a, b);
    }

    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.GREATER, 'bool');
    }

    const program = new BinaryOpProgram(binaryop_gpu.GREATER, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(
          a, b, binaryop_packed_gpu.GREATER_EQUAL, 'bool');
    }
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
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.LOGICAL_AND, 'bool');
    }
    const program =
        new BinaryOpProgram(binaryop_gpu.LOGICAL_AND, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.LOGICAL_OR, 'bool');
    }
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
    return topkImpl(xVals, x.shape, x.dtype as NumericDataType, k, sorted);
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

    const program = ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.MIN, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.MIN, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  mod(a: Tensor, b: Tensor): Tensor {
    const program = ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.MOD, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.MOD, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  max(x: Tensor, axes: number[]): Tensor {
    if (this.shouldExecuteOnCPU([x])) {
      return this.cpuBackend.max(x, axes);
    }

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

    const program = ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.MAX, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.MAX, a.shape, b.shape);
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
    const program = ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(
            binaryop_gpu.SQUARED_DIFFERENCE, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.SQUARED_DIFFERENCE, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  realDivide(a: Tensor, b: Tensor): Tensor {
    const op = binaryop_gpu.DIV;
    const outputDtype = 'float32';
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      const checkOutOfBounds = true;
      return this.packedBinaryOp(
          a, b, binaryop_packed_gpu.DIV, outputDtype, checkOutOfBounds);
    }
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, outputDtype);
    return this.compileAndRun<Tensor>(program, [a, b], output);
  }

  floorDiv(a: Tensor, b: Tensor): Tensor {
    const op = binaryop_gpu.INT_DIV;
    const outputDtype = 'int32';
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(
          a, b, binaryop_packed_gpu.INT_DIV, outputDtype);
    }
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, outputDtype);
    return this.compileAndRun<Tensor>(program, [a, b], output);
  }

  add(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' && b.dtype === 'complex64') {
      return this.complexSeparableBinaryOp(a, b, binaryop_gpu.ADD);
    }

    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.add(a, b);
    }

    const dtype = upcastType(a.dtype, b.dtype);
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_gpu.ADD, dtype);
    }
    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, dtype) as Tensor;
    return this.compileAndRun<Tensor>(program, [a, b], output);
  }

  private packedBinaryOp(
      a: TensorHandle, b: TensorHandle, op: string, dtype: DataType,
      checkOutOfBounds = false) {
    const program =
        new BinaryOpPackedProgram(op, a.shape, b.shape, checkOutOfBounds);
    const output = this.makePackedTensor(program.outputShape, dtype) as Tensor;
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

      const aHandle = this.makeComplexComponentTensorHandle(a, aPart);
      const bHandle = this.makeComplexComponentTensorHandle(b, bPart);

      const program = new BinaryOpProgram(op, a.shape, b.shape);
      const output = this.makeOutputArray(
                         program.outputShape,
                         upcastType(aPart.dtype, bPart.dtype)) as Tensor;

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
    if (tensors.length === 1) {
      return tensors[0];
    }

    // Limit the number of uploaded textures for optimization.
    if (tensors.length > ENV.get('WEBGL_MAX_TEXTURES_IN_SHADER')) {
      const midIndex = Math.floor(tensors.length / 2);
      const leftSide = this.addN(tensors.slice(0, midIndex));
      const rightSide = this.addN(tensors.slice(midIndex));
      return this.addN([leftSide, rightSide]);
    }

    const dtype =
        tensors.map(t => t.dtype).reduce((d1, d2) => upcastType(d1, d2));
    const shapes = tensors.map(t => t.shape);
    // We can make sure shapes are identical in op level.
    const usePackedOp = ENV.getBool('WEBGL_PACK');
    const program = usePackedOp ?
        new AddNPackedProgram(tensors[0].shape, shapes) :
        new AddNProgram(tensors[0].shape, shapes);
    const output = usePackedOp ?
        this.makePackedTensor(program.outputShape, dtype) as T :
        this.makeOutputArray(program.outputShape, dtype) as T;
    return this.compileAndRun<T>(program, tensors, output);
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' && b.dtype === 'complex64') {
      return this.complexSeparableBinaryOp(a, b, binaryop_gpu.SUB);
    }

    if (this.shouldExecuteOnCPU([a, b])) {
      return this.cpuBackend.subtract(a, b);
    }
    const dtype = upcastType(a.dtype, b.dtype);
    if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_gpu.SUB, a.dtype);
    }
    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, dtype) as Tensor;
    return this.compileAndRun<Tensor>(program, [a, b], output);
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    const usePackedOp = ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
    const program = usePackedOp ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.POW, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.POW, a.shape, b.shape);
    const dtype = upcastType(a.dtype, b.dtype);
    const output = usePackedOp ?
        this.makePackedTensor(program.outputShape, dtype) as T :
        this.makeOutputArray(program.outputShape, dtype) as T;
    return this.compileAndRun<T>(program, [a, b], output);
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

  isNaN<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.IS_NAN);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [x], output) as T;
  }
  isInf<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.IS_INF);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [x], output) as T;
  }
  isFinite<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.IS_FINITE);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [x], output) as T;
  }

  round<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ROUND);
    return this.compileAndRun(program, [x]) as T;
  }

  exp<T extends Tensor>(x: T): T {
    let program: UnaryOpProgram|UnaryOpPackedProgram;
    if (ENV.getBool('WEBGL_PACK')) {
      program = new UnaryOpPackedProgram(x.shape, unary_op.EXP);
    } else {
      program = new UnaryOpProgram(x.shape, unary_op.EXP);
    }
    return this.compileAndRun(program, [x]) as T;
  }

  expm1<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.EXPM1);
    return this.compileAndRun(program, [x]) as T;
  }

  log<T extends Tensor>(x: T): T {
    let program: UnaryOpProgram|UnaryOpPackedProgram;
    if (ENV.getBool('WEBGL_PACK')) {
      program = new UnaryOpPackedProgram(x.shape, unary_packed_op.LOG);
    } else {
      program = new UnaryOpProgram(x.shape, unary_op.LOG);
    }
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
    if (this.shouldExecuteOnCPU([x])) {
      return this.cpuBackend.rsqrt(x);
    }
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
    let program: UnaryOpProgram|UnaryOpPackedProgram;
    if (ENV.getBool('WEBGL_PACK')) {
      program = new UnaryOpPackedProgram(x.shape, unary_packed_op.RELU);
    } else {
      program = new UnaryOpProgram(x.shape, unary_op.RELU);
    }
    return this.compileAndRun(program, [x]) as T;
  }

  prelu<T extends Tensor>(x: T, alpha: T): T {
    const program = ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(
            binaryop_packed_gpu.PRELU, x.shape, alpha.shape) :
        new BinaryOpProgram(binaryop_gpu.PRELU, x.shape, alpha.shape);
    return this.compileAndRun(program, [x, alpha]) as T;
  }

  elu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ELU);
    return this.compileAndRun(program, [x]) as T;
  }

  eluDer<T extends Tensor>(dy: T, y: T): T {
    const program = ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(
            binaryop_packed_gpu.ELU_DER, dy.shape, y.shape) :
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
    let program;
    if (ENV.getBool('WEBGL_PACK_CLIP')) {
      program = new ClipPackedProgram(x.shape);
    } else {
      program = new ClipProgram(x.shape);
    }
    const customSetup = program.getCustomSetupFunc(min, max);
    return this.compileAndRun(program, [x], null, customSetup) as T;
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
    const program = ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.ATAN2, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.ATAN2, a.shape, b.shape);
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

  private conv2dByMatMul(
      x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo, bias?: Tensor4D,
      activation?: Activation, preluActivationWeights?: Tensor): Tensor4D {
    // Reshapes conv2D input to 2D tensors, uses matMul and then reshape the
    // result from 2D to 4D.
    const xShape = x.shape;
    const xTexData = this.texData.get(x.dataId);
    const sharedMatMulDim = convInfo.inChannels;
    const outerShapeX = xShape[0] * xShape[1] * xShape[2];
    const outerShapeFilter = convInfo.outChannels;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const transposeA = false;
    const transposeB = false;

    // TODO: Once reduction ops are packed, batchMatMul will always be packed
    // and we can remove this condition.
    const batchMatMulWillBeUnpacked =
        (outerShapeX === 1 || outerShapeFilter === 1) &&
        sharedMatMulDim > MATMUL_SHARED_DIM_THRESHOLD;
    const reshapeWillBeExpensive = xShape[2] % 2 !== 0 && !!xTexData.isPacked;

    if (batchMatMulWillBeUnpacked || !ENV.getBool('WEBGL_LAZILY_UNPACK') ||
        !ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS') ||
        !reshapeWillBeExpensive) {
      const targetShape = isChannelsLast ? xShape[0] * xShape[1] * xShape[2] :
                                           xShape[0] * xShape[2] * xShape[3];
      const xReshaped =
          this.reshape(x, [1, targetShape, convInfo.inChannels]) as Tensor3D;
      const filterReshaped =
          this.reshape(
              filter, [1, convInfo.inChannels, convInfo.outChannels]) as
          Tensor3D;

      return this.reshape<Rank.R4>(
          this.fusedBatchMatMul({
            a: xReshaped,
            b: filterReshaped,
            transposeA,
            transposeB,
            bias,
            activation,
            preluActivationWeights
          }),
          convInfo.outShape);
    }

    // Following optimization is specific to packed |x| with odd row count
    // (For example, in channelLast mode, 'row count' refers to x.shape[2]):
    // we avoid expensive packed 2x2 reshape by padding row count to next,
    // even number. When x.shape[2] is odd, the result of packed batchMatMul is
    // the same (has the same texture layout and and values in the texture) as
    // it is for even x.shape[2] + 1. We make the odd-rows tensor to look like
    // even-rows tensor before the operation and, after the batchMatMul,
    // fix the even-rows result to have odd number of rows.
    const targetShape = isChannelsLast ?
        xShape[0] * xShape[1] * (xShape[2] + 1) :
        xShape[0] * xShape[2] * (xShape[3] + 1);
    const xReshaped = Tensor.make(
                          [1, targetShape, convInfo.inChannels],
                          {dataId: x.dataId}, x.dtype, this) as Tensor3D;

    // xTexData.shape gets referenced from GPGPUBinary.inShapeInfos.
    // Decrementing row count, after batchMatMul->...->compileProgram leads to
    // invalid row count within the reference in GPGPUBinary.inShapeInfos.
    // Alternative fix would be to provide a copy to GPGPUBinary.inShapeInfos
    // in compileProgram method, but that would affect compilation of all
    // programs - instead, provide a copy here, with even row count, before
    // calling batchMatMul->...->compileProgram and after that, the original
    // xTexData.shape is restored.
    const originalXTexDataShape = xTexData.shape;
    xTexData.shape = xTexData.shape.slice();
    xTexData.shape[xTexData.shape.length - 2]++;
    util.assert(
        webgl_util.isReshapeFree(xTexData.shape, xReshaped.shape),
        () => `packed reshape ${xTexData.shape} to ${
            xReshaped.shape} isn't free`);
    const filterReshaped =
        this.reshape(filter, [1, convInfo.inChannels, convInfo.outChannels]) as
        Tensor3D;

    const pointwiseConv = this.fusedBatchMatMul({
      a: xReshaped,
      b: filterReshaped,
      transposeA,
      transposeB,
      bias,
      activation,
      preluActivationWeights
    });
    const pointwiseConvTexData = this.texData.get(pointwiseConv.dataId);
    util.assert(
        pointwiseConvTexData.isPacked,
        () => 'batchMatMul result is expected to be packed');
    // Restore the input shape to original.
    xTexData.shape = originalXTexDataShape;
    // Set the output shape - there is no need for expensive reshape as data
    // layout is already correct.
    pointwiseConvTexData.shape = convInfo.outShape;
    return Tensor.make(
               convInfo.outShape, {dataId: pointwiseConv.dataId},
               pointwiseConv.dtype, this) as Tensor4D;
  }

  private conv2dWithIm2Row(
      x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo, bias?: Tensor4D,
      activation?: Activation, preluActivationWeights?: Tensor): Tensor4D {
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
      dataFormat
    } = convInfo;

    const isChannelsLast = dataFormat === 'channelsLast';

    const sharedDim = filterWidth * filterHeight * inChannels;
    const numCols = outHeight * outWidth;
    const x2ColShape = [sharedDim, numCols];
    const transposeA = true;
    const transposeB = false;

    const xSqueezed = x.squeeze([0]);
    const w2Row = filter.reshape([1, sharedDim, -1]) as Tensor3D;

    const im2ColProgram =
        new Im2ColPackedProgram(x2ColShape, xSqueezed.shape, convInfo);
    const im2Col =
        this.compileAndRun<Tensor2D>(im2ColProgram, [xSqueezed]).reshape([
          1, x2ColShape[0], x2ColShape[1]
        ]) as Tensor3D;

    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const fusedActivation =
        activation ? mapActivationToShaderProgram(activation, true) : null;
    const matmulProgram = new MatMulPackedProgram(
        im2Col.shape, [1, numCols, convInfo.outChannels], transposeA,
        transposeB, hasBias, fusedActivation, hasPreluActivationWeights);
    const inputs: TensorHandle[] = [im2Col, w2Row];
    if (bias) {
      inputs.push(bias);
    }
    if (hasPreluActivationWeights) {
      inputs.push(preluActivationWeights);
    }
    const product = this.compileAndRun<Tensor4D>(matmulProgram, inputs);

    if (isChannelsLast) {
      return product.reshape([1, outHeight, outWidth, convInfo.outChannels]);
    } else {
      return product.reshape([1, convInfo.outChannels, outHeight, outWidth]);
    }
  }

  fusedConv2d(
      x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo, bias?: Tensor4D,
      activation?: Activation, preluActivationWeights?: Tensor): Tensor4D {
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
        (convInfo.padInfo.type === 'SAME' ||
         convInfo.padInfo.type === 'VALID')) {
      return this.conv2dByMatMul(
          x, filter, convInfo, bias, activation, preluActivationWeights);
    }
    if (ENV.getBool('WEBGL_CONV_IM2COL') && x.shape[0] === 1) {
      return this.conv2dWithIm2Row(
          x, filter, convInfo, bias, activation, preluActivationWeights);
    }

    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const fusedActivation =
        activation ? mapActivationToShaderProgram(activation, false) : null;
    const program = new Conv2DProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
    const inputs: TensorHandle[] = [x, filter];
    if (bias) {
      inputs.push(bias);
    }
    if (preluActivationWeights) {
      inputs.push(preluActivationWeights);
    }
    return this.compileAndRun(program, inputs);
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
        (convInfo.padInfo.type === 'SAME' ||
         convInfo.padInfo.type === 'VALID')) {
      return this.conv2dByMatMul(x, filter, convInfo);
    }
    if (ENV.getBool('WEBGL_CONV_IM2COL') && x.shape[0] === 1) {
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
    let program: DepthwiseConv2DProgram|DepthwiseConvPacked2DProgram;
    if (ENV.getBool('WEBGL_PACK_DEPTHWISECONV') && convInfo.strideWidth <= 2 &&
        convInfo.outChannels / convInfo.inChannels === 1) {
      program = new DepthwiseConvPacked2DProgram(convInfo);
      return this.compileAndRun(
          program, [x, filter],
          this.makePackedTensor(convInfo.outShape, x.dtype));
    }

    program = new DepthwiseConv2DProgram(convInfo);
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

  conv3d(x: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    const program = new Conv3DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
  }

  conv3dDerInput(dy: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo):
      Tensor5D {
    const program = new Conv3DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  conv3dDerFilter(x: Tensor5D, dy: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    const program = new Conv3DDerFilterProgram(convInfo);
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

  unstack(x: Tensor, axis: number): Tensor[] {
    const num = x.shape[axis];
    const outShape: number[] = new Array(x.rank - 1);
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

  avgPool3d(x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    const program = new Pool3DProgram(convInfo, 'avg', false);
    const output = this.makeOutputArray(program.outputShape, 'float32');
    return this.compileAndRun(program, [x], output) as Tensor5D;
  }

  avgPool3dBackprop(dy: Tensor5D, x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    const avgPool3dBackpropProgram = new AvgPool3DBackpropProgram(convInfo);
    const output =
        this.makeOutputArray(avgPool3dBackpropProgram.outputShape, x.dtype);
    return this.compileAndRun(avgPool3dBackpropProgram, [dy], output) as
        Tensor5D;
  }

  maxPool3d(x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    const program = new Pool3DProgram(convInfo, 'max', false);
    const output = this.makeOutputArray(program.outputShape, 'float32');
    return this.compileAndRun(program, [x], output) as Tensor5D;
  }

  maxPool3dBackprop(
      dy: Tensor5D, x: Tensor5D, y: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    const getPositions = true;
    const maxPool3dPositionsProgram =
        new Pool3DProgram(convInfo, 'max', getPositions);
    const maxPool3dPositions: Tensor5D =
        this.compileAndRun(maxPool3dPositionsProgram, [x]);
    const maxPool3dBackPropProgram = new MaxPool3DBackpropProgram(convInfo);
    const output =
        this.makeOutputArray(maxPool3dBackPropProgram.outputShape, x.dtype);
    const result = this.compileAndRun(
        maxPool3dBackPropProgram, [dy, maxPool3dPositions], output);
    maxPool3dPositions.dispose();
    return result as Tensor5D;
  }

  reshape<R extends Rank>(x: Tensor, shape: ShapeMap[R]): Tensor<R> {
    const texData = this.texData.get(x.dataId);
    if (texData.isPacked && !webgl_util.isReshapeFree(x.shape, shape) &&
        !(texData.texture !== null &&
          webgl_util.isReshapeFree(texData.shape, shape))) {
      return this.packedReshape(x, shape);
    }
    return backend_util.reshapeTensor(x, shape);
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const program = ENV.getBool('WEBGL_PACK_IMAGE_OPERATIONS') ?
        new ResizeBilinearPackedProgram(
            x.shape, newHeight, newWidth, alignCorners) :
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

  diag(x: Tensor): Tensor {
    const program = new DiagProgram(x.size);
    return this.compileAndRun(program, [x]);
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
        () =>
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

  gatherND(x: Tensor, indices: Tensor): Tensor {
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

  fill<R extends Rank>(
      shape: ShapeMap[R], value: number|string, dtype?: DataType): Tensor<R> {
    dtype = dtype || inferDtype(value);

    if (dtype === 'string') {
      // String type should be handled in CPU memory.
      const values = getArrayFromDType(dtype, sizeFromShape(shape));
      values.fill(value as string);
      return Tensor.make(shape, {values}, dtype);
    } else {
      const program = new FillProgram(shape, value as number);
      const customSetup = program.getCustomSetupFunc(value as number);
      const output = this.makeOutputArray(shape, dtype) as Tensor<R>;
      return this.compileAndRun(program, [], output, customSetup) as Tensor<R>;
    }
  }

  onesLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    if (x.dtype === 'string') {
      throw new Error('onesLike is not supported under string dtype');
    } else {
      // TODO(cais, smilkov): Add WebGL shader for onesLike:
      //   https://github.com/tensorflow/tfjs/issues/1293
      return this.fill(x.shape, 1, x.dtype);
    }
  }

  zerosLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    return this.fill(x.shape, x.dtype === 'string' ? '' : 0, x.dtype);
  }

  linspace(start: number, stop: number, num: number): Tensor1D {
    // TODO: Use CPU implementation due to the precision problem in Safari.
    return backend_util.linspaceImpl(start, stop, num);
  }

  private makeOutputArray<T extends Tensor>(shape: number[], dtype: DataType):
      T {
    return Tensor.make(shape, {}, dtype, this) as T;
  }

  private makePackedTensor<T extends Tensor, D extends DataType = 'float32'>(
      shape: number[], dtype?: D): T {
    const packedTensor = Tensor.make(shape, {}, dtype, this);
    this.texData.get(packedTensor.dataId).isPacked = true;
    return packedTensor as T;
  }

  private unpackTensor<T extends Tensor>(input: T|TensorHandle): T {
    const program = new UnpackProgram(input.shape);
    return this.compileAndRun(
        program, [input],
        Tensor.make(program.outputShape, {}, input.dtype, this));
  }

  private packTensor<T extends Tensor>(input: T|TensorHandle): T {
    const program = new PackProgram(input.shape);
    return this.compileAndRun(
        program, [input], this.makePackedTensor(input.shape, input.dtype), null,
        true);
  }

  private packedReshape<R extends Rank>(input: Tensor, afterShape: ShapeMap[R]):
      Tensor<R> {
    const inputAs3D = input.reshape([
      webgl_util.getBatchDim(input.shape),
      ...webgl_util.getRowsCols(input.shape)
    ]);
    const afterShapeAs3D = [
      webgl_util.getBatchDim(afterShape), ...webgl_util.getRowsCols(afterShape)
    ];
    const program = new ReshapePackedProgram(
        afterShapeAs3D as [number, number, number],
        inputAs3D.shape as [number, number, number]);
    return this.compileAndRun<Tensor<R>>(program, [inputAs3D])
        .reshape(afterShape);
  }

  private decode(dataId: DataId): TensorHandle {
    const texData = this.texData.get(dataId);
    const {isPacked, shape, dtype} = texData;
    const shapeAs3D =
        webgl_util.getShapeAs3D(shape) as [number, number, number];
    const denseTexShape = tex_util.getDenseTexShape(shape);

    const tmpTarget = this.makeTensorHandle(shape, 'float32') as TensorHandle &
        {size: number};
    this.texData.get(tmpTarget.dataId).isPacked = true;
    this.texData.get(tmpTarget.dataId).dtype = dtype;
    this.texData.get(tmpTarget.dataId).texShape =
        denseTexShape.map(
            d => d * 2) as [number, number];  // To undo the effect of isPacked
                                              // being set to true.

    let program;
    if (isPacked) {
      program = new DecodeMatrixPackedProgram(shapeAs3D, denseTexShape);
    } else {
      program = new DecodeMatrixProgram(shapeAs3D, denseTexShape);
    }

    this.compileAndRun(
        program, [{shape: shapeAs3D, dtype, dataId}], tmpTarget, null, true);
    return tmpTarget;
  }

  public compileAndRun<
      K extends {dtype: DataType, size: number, dataId: {}, shape: number[]}>(
      program: GPGPUProgram, inputs: TensorHandle[], output?: K,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void,
      preventEagerUnpackingOfOutput = false): K {
    if (output == null) {
      if (program.usesPackedTextures) {
        output = this.makePackedTensor(program.outputShape, inputs[0].dtype) as
            {} as K;
      } else {
        output = this.makeOutputArray(program.outputShape, inputs[0].dtype) as
            {} as K;
      }
    }

    if (output.size === 0) {
      // Short-circuit the computation since the result is empty (has 0 in its
      // shape).
      this.texData.get(output.dataId).values =
          getTypedArrayFromDType(output.dtype as 'float32', 0);
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
        if (!program.usesPackedTextures &&
            util.sizeFromShape(input.shape) <=
                ENV.getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')) {
          // Upload small tensors that live on the CPU as uniforms, not as
          // textures. Do this only when the environment supports 32bit floats
          // due to problems when comparing 16bit floats with 32bit floats.
          // TODO(https://github.com/tensorflow/tfjs/issues/821): Make it
          // possible for packed shaders to sample from uniforms.
          return {
            shape: input.shape,
            texData: null,
            isUniform: true,
            uniformValues: texData.values as TypedArray
          };
        }

        // This ensures that if a packed program's inputs have not yet been
        // uploaded to the GPU, they get uploaded as packed right off the bat.
        if (program.usesPackedTextures) {
          texData.isPacked = true;
          texData.shape = input.shape;
        }
      } else if (!!texData.isPacked !== !!program.usesPackedTextures) {
        input = texData.isPacked ? this.unpackTensor(input) :
                                   this.packTensor(input);
        texData = this.texData.get(input.dataId);
      } else if (
          texData.isPacked &&
          !webgl_util.isReshapeFree(texData.shape, input.shape)) {
        // This is a special case where a texture exists for a tensor
        // but the shapes are incompatible (due to packing constraints) because
        // the tensor did not have a chance to go through the packed reshape
        // shader. This only happens when we reshape the *same* tensor to form
        // *distinct* inputs to an op, e.g. dotting a vector with itself. This
        // case will disappear once packed uploading is the default.

        const savedInput = input;
        const targetShape = input.shape;

        input.shape = texData.shape;
        input = this.packedReshape(input as Tensor, targetShape);
        texData = this.texData.get(input.dataId);

        savedInput.shape = targetShape;
      }

      this.uploadToGPU(input.dataId);
      return {shape: input.shape, texData, isUniform: false};
    });

    this.uploadToGPU(output.dataId);
    const outputData: TensorData = {
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

    gpgpu_math.runProgram(
        this.gpgpu, binary, inputsData, outputData, customSetup);

    if (shouldTimeProgram) {
      query = this.endTimer(query);
      this.activeTimers.push(
          {name: program.constructor.name, query: this.getQueryTime(query)});
    }

    if (!ENV.getBool('WEBGL_LAZILY_UNPACK') &&
        this.texData.get(output.dataId).isPacked &&
        preventEagerUnpackingOfOutput === false) {
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
    this.textureManager.dispose();
    if (this.canvas != null && this.canvas.remove != null) {
      this.canvas.remove();
    } else {
      this.canvas = null;
    }
    if (this.fromPixels2DContext != null &&
        //@ts-ignore
        this.fromPixels2DContext.canvas.remove) {
      //@ts-ignore
      this.fromPixels2DContext.canvas.remove();
    }
    if (this.gpgpuCreatedLocally) {
      this.gpgpu.program = null;
      this.gpgpu.dispose();
    }
    this.disposed = true;
  }

  floatPrecision(): 16|32 {
    if (this.floatPrecisionValue == null) {
      this.floatPrecisionValue = tidy(() => {
        // Momentarily switching DEBUG flag to false so we don't throw an error
        // trying to upload a small value.
        const debugFlag = ENV.getBool('DEBUG');
        ENV.set('DEBUG', false);
        const underflowCheckValue = this.abs(scalar(1e-8)).dataSync()[0];
        ENV.set('DEBUG', debugFlag);

        if (underflowCheckValue > 0) {
          return 32;
        }
        return 16;
      });
    }
    return this.floatPrecisionValue;
  }
  /** Returns the smallest representable number.  */
  epsilon(): number {
    return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
  }

  private uploadToGPU(dataId: DataId): void {
    const texData = this.texData.get(dataId);
    const {shape, dtype, values, texture, usage, isPacked} = texData;

    if (texture != null) {
      // Array is already on GPU. No-op.
      return;
    }
    const shouldTimeProgram = this.activeTimers != null;
    let start: number;
    if (shouldTimeProgram) {
      start = util.now();
    }

    let texShape = texData.texShape;
    if (texShape == null) {
      texShape = webgl_util.getTextureShapeFromLogicalShape(shape, isPacked);
      texData.texShape = texShape;
    }

    if (values != null) {
      const shapeAs3D = webgl_util.getShapeAs3D(shape);

      let program;
      let width = texShape[1], height = texShape[0];
      const isByteArray = values instanceof Uint8Array;

      if (isPacked) {
        [width, height] = tex_util.getPackedMatrixTextureShapeWidthHeight(
            texShape[0], texShape[1]);
        program = new EncodeMatrixPackedProgram(
            shapeAs3D, [height, width], isByteArray);
      } else {
        program =
            new EncodeMatrixProgram(shapeAs3D, [height, width], isByteArray);
      }

      const tempDenseInputHandle =
          this.makeTensorHandle([height, width], dtype);
      if (isByteArray) {
        this.texData.get(tempDenseInputHandle.dataId).usage =
            TextureUsage.PIXELS;
      } else {
        this.texData.get(tempDenseInputHandle.dataId).usage =
            TextureUsage.UPLOAD;
      }
      this.gpgpu.uploadDenseMatrixToTexture(
          this.getTexture(tempDenseInputHandle.dataId), width, height,
          values as TypedArray);

      const encodedOutputTarget =
          this.makeTensorHandle(
              program.outputShape, tempDenseInputHandle.dtype) as TensorHandle &
          {size: number};
      encodedOutputTarget.size = sizeFromShape(program.outputShape);
      this.texData.get(encodedOutputTarget.dataId).isPacked = isPacked;
      this.compileAndRun(program, [tempDenseInputHandle], encodedOutputTarget);

      // Have the original texture assume the identity of the encoded output.
      const outputTexData = this.texData.get(encodedOutputTarget.dataId);
      texData.texture = outputTexData.texture;
      texData.texShape = outputTexData.texShape;
      texData.isPacked = outputTexData.isPacked;
      texData.usage = outputTexData.usage;

      this.disposeData(tempDenseInputHandle.dataId);
      this.texData.delete(encodedOutputTarget.dataId);

      // Once uploaded, don't store the values on cpu.
      texData.values = null;
      if (shouldTimeProgram) {
        this.uploadWaitMs += util.now() - start;
      }
    } else {
      const newTexture = this.acquireTexture(texShape, usage, dtype, isPacked);
      texData.texture = newTexture;
    }
  }

  private convertAndCacheOnCPU(dataId: DataId, float32Values?: Float32Array):
      TypedArray {
    const texData = this.texData.get(dataId);
    const {dtype} = texData;

    this.releaseGPUData(dataId);

    if (float32Values != null) {
      texData.values = float32ToTypedArray(float32Values, dtype as 'float32');
    }
    return texData.values as TypedArray;
  }

  private acquireTexture(
      texShape: [number, number], texType: TextureUsage, dtype: DataType,
      isPacked: boolean): WebGLTexture {
    this.numBytesInGPU += this.computeBytes(texShape, dtype);
    if (!this.warnedAboutMemory &&
        this.numBytesInGPU > this.numMBBeforeWarning * 1024 * 1024) {
      const mb = (this.numBytesInGPU / 1024 / 1024).toFixed(2);
      this.warnedAboutMemory = true;
      console.warn(
          `High memory usage in GPU: ${mb} MB, ` +
          `most likely due to a memory leak`);
    }
    return this.textureManager.acquireTexture(texShape, texType, isPacked);
  }

  private computeBytes(shape: [number, number], dtype: DataType) {
    return shape[0] * shape[1] * util.bytesPerElement(dtype);
  }
}

if (device_util.isBrowser()) {
  ENGINE.registerBackend(
      'webgl', () => new MathBackendWebGL(), 2 /* priority */);
}

function float32ToTypedArray<D extends NumericDataType>(
    a: Float32Array, dtype: D): DataTypeMap[D] {
  if (dtype === 'float32' || dtype === 'complex64') {
    return a as DataTypeMap[D];
  } else if (dtype === 'int32' || dtype === 'bool') {
    const result = (dtype === 'int32') ? new Int32Array(a.length) :
                                         new Uint8Array(a.length);
    for (let i = 0; i < result.length; ++i) {
      result[i] = Math.round(a[i]);
    }
    return result as DataTypeMap[D];
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}
