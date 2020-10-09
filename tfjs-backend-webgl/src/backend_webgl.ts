/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import {complex, DataId, div, engine, env, imag, max, MemoryInfo, range, real, RecursiveArray, reshape, scalar, softmax, tensor, tidy, TimingInfo, transpose} from '@tensorflow/tfjs-core';
import {backend_util, buffer, kernel_impls, slice_util, util} from '@tensorflow/tfjs-core';
import {DataStorage, DataType, KernelBackend, NumericDataType, Rank, Scalar, ShapeMap, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D, TensorInfo, TypedArray, upcastType} from '@tensorflow/tfjs-core';

import {addImplCPU, ceilImplCPU, expImplCPU, expm1ImplCPU, floorImplCPU, logImplCPU, multiplyImplCPU, rsqrtImplCPU, simpleAbsImplCPU, sliceImplCPU, subImplCPU} from './kernel_utils/shared';

const {segment_util} = backend_util;
const split = kernel_impls.split;
const tile = kernel_impls.tile;
const topkImpl = kernel_impls.topkImpl;
const whereImpl = kernel_impls.whereImpl;

import {AddNProgram} from './addn_gpu';
import {AddNPackedProgram} from './addn_packed_gpu';
import {ArgMinMaxProgram} from './argminmax_gpu';
import {ArgMinMaxPackedProgram} from './argminmax_packed_gpu';
import {AvgPool3DBackpropProgram} from './avg_pool_backprop_gpu';
import * as binaryop_complex_gpu from './binaryop_complex_gpu';
import {BinaryOpComplexProgram} from './binaryop_complex_gpu';
import * as binaryop_gpu from './binaryop_gpu';
import {BinaryOpProgram} from './binaryop_gpu';
import * as binaryop_packed_gpu from './binaryop_packed_gpu';
import {BinaryOpPackedProgram} from './binaryop_packed_gpu';
import {getWebGLContext} from './canvas_util';
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
import {GatherProgram} from './gather_gpu';
import {GatherNDProgram} from './gather_nd_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {GPGPUBinary, GPGPUProgram, TensorData} from './gpgpu_math';
import {Im2ColPackedProgram} from './im2col_packed_gpu';
import {LRNProgram} from './lrn_gpu';
import {LRNGradProgram} from './lrn_grad_gpu';
import {LRNPackedProgram} from './lrn_packed_gpu';
import {MaxPool3DBackpropProgram} from './max_pool_backprop_gpu';
import {MatMulPackedProgram} from './mulmat_packed_gpu';
import {MultinomialProgram} from './multinomial_gpu';
import {OneHotProgram} from './onehot_gpu';
import {PackProgram} from './pack_gpu';
import {PadProgram} from './pad_gpu';
import {PadPackedProgram} from './pad_packed_gpu';
import {Pool3DProgram} from './pool_gpu';
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
import * as unary_op from './unaryop_gpu';
import {UnaryOpProgram} from './unaryop_gpu';
import * as unary_packed_op from './unaryop_packed_gpu';
import {UnaryOpPackedProgram} from './unaryop_packed_gpu';
import {UnpackProgram} from './unpack_gpu';
import * as webgl_util from './webgl_util';
import {BackendValues} from '@tensorflow/tfjs-core';

export const EPSILON_FLOAT32 = 1e-7;
export const EPSILON_FLOAT16 = 1e-4;

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
  // Tracks the total number of bytes allocated on the GPU, accounting for the
  // physical texture type.
  numBytesInGPUAllocated: number;
  // Tracks byte size of textures that were created and then made available for
  // reuse (disposed).
  numBytesInGPUFree: number;
  unreliable: boolean;
}

export interface WebGLTimingInfo extends TimingInfo {
  uploadWaitMs: number;
  downloadWaitMs: number;
}

const binaryCaches: {[webGLVersion: string]: {[key: string]: GPGPUBinary}} = {};

export function getBinaryCache(webGLVersion: number) {
  if (webGLVersion in binaryCaches) {
    return binaryCaches[webGLVersion];
  }
  binaryCaches[webGLVersion] = {};
  return binaryCaches[webGLVersion];
}

function mapActivationToShaderProgram(
    activation: backend_util.Activation, packed = false): string {
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
  } else if (activation === 'elu') {
    if (packed) {
      return unary_packed_op.ELU;
    }
    return unary_op.ELU;
  } else if (activation === 'relu6') {
    if (packed) {
      return unary_packed_op.RELU6;
    }
    return unary_op.RELU6;
  } else if (activation === 'prelu') {
    if (packed) {
      return binaryop_packed_gpu.PRELU;
    }
    return binaryop_gpu.PRELU;
  }
  throw new Error(`Activation ${
      activation} has not been implemented for the WebGL backend.`);
}

// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD = 128;

// Empirically determined constant used to decide the number of MB on GPU
// before we warn about high memory use. The MB are this constant * screen area
// * dpi / 1024 / 1024.
const BEFORE_PAGING_CONSTANT = 600;
function numMBBeforeWarning(): number {
  if (env().global.screen == null) {
    return 1024;  // 1 GB.
  }
  return (env().global.screen.height * env().global.screen.width *
          window.devicePixelRatio) *
      BEFORE_PAGING_CONSTANT / 1024 / 1024;
}

// Empirically determined minimal shared dimension in matmul before we forward
// to a.mul(b).sum() in order to take advantage of GPU parallelism. See
// https://github.com/tensorflow/tfjs-core/pull/1379 for benchmarks.
export const MATMUL_SHARED_DIM_THRESHOLD = 1000;

export class MathBackendWebGL extends KernelBackend {
  texData: DataStorage<TextureData>;
  gpgpu: GPGPUContext;

  // Maps data ids that have a pending read operation, to list of subscribers.
  private pendingRead = new WeakMap<DataId, Array<(arr: TypedArray) => void>>();
  // List of data ids that are scheduled for disposal, but are waiting on a
  // pending read operation.
  private pendingDisposal = new WeakSet<DataId>();
  // Used to count the number of 'shallow' sliced tensors that point to the
  // same data id.
  private dataRefCount = new WeakMap<DataId, number>();
  private numBytesInGPU = 0;

  private canvas: HTMLCanvasElement|OffscreenCanvas;

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
  private warnedAboutCPUBackend = false;

  constructor(gpgpu?: GPGPUContext) {
    super();
    if (!env().getBool('HAS_WEBGL')) {
      throw new Error('WebGL is not supported on this device');
    }

    if (gpgpu == null) {
      const gl = getWebGLContext(env().getNumber('WEBGL_VERSION'));
      this.binaryCache = getBinaryCache(env().getNumber('WEBGL_VERSION'));
      this.gpgpu = new GPGPUContext(gl);
      this.canvas = gl.canvas;
      this.gpgpuCreatedLocally = true;
    } else {
      this.gpgpu = gpgpu;
      this.binaryCache = {};
      this.gpgpuCreatedLocally = false;
      this.canvas = gpgpu.gl.canvas;
    }
    this.textureManager = new TextureManager(this.gpgpu);
    this.numMBBeforeWarning = numMBBeforeWarning();

    this.texData = new DataStorage(this, engine());
  }

  numDataIds() {
    return this.texData.numDataIds() +
        (this.cpuBackend ? this.cpuBackend.numDataIds() : 0) -
        this.pendingDeletes;
  }

  write(values: BackendValues, shape: number[], dtype: DataType): DataId {
    if (env().getBool('WEBGL_CHECK_NUMERICAL_PROBLEMS') ||
        env().getBool('DEBUG')) {
      this.checkNumericalProblems(values);
    }
    if (dtype === 'complex64' && values != null) {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }
    const dataId = {};
    this.texData.set(
        dataId,
        {shape, dtype, values, usage: TextureUsage.UPLOAD, refCount: 1});
    return dataId;
  }

  /** Increase refCount of a `TextureData`. */
  incRef(dataId: DataId): void {
    const texData = this.texData.get(dataId);
    texData.refCount++;
  }

  /** Decrease refCount of a `TextureData`. */
  decRef(dataId: DataId): void {
    if (this.texData.has(dataId)) {
      const texData = this.texData.get(dataId);
      texData.refCount--;
    }
  }

  move(dataId: DataId, values: BackendValues, shape: number[], dtype: DataType):
      void {
    if (env().getBool('DEBUG')) {
      this.checkNumericalProblems(values);
    }
    if (dtype === 'complex64') {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }
    this.texData.set(
        dataId,
        {shape, dtype, values, usage: TextureUsage.UPLOAD, refCount: 1});
  }

  disposeIntermediateTensorInfo(tensorInfo: TensorInfo): void {
    const dataId = tensorInfo.dataId;

    if (this.texData.has(dataId)) {
      const textureData = this.texData.get(dataId);

      textureData.refCount--;

      if (textureData.refCount < 1) {
        this.disposeData(dataId);
      }
    }
  }

  readSync(dataId: DataId): BackendValues {
    const texData = this.texData.get(dataId);
    const {values, dtype, complexTensors, slice, shape, isPacked} = texData;

    // The presence of `slice` indicates this tensor is a shallow slice of a
    // different tensor, and is using that original tensor's texture. Run
    // `clone` in order to copy that texture and read from it.
    if (slice != null) {
      let program;
      if (isPacked) {
        program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
      } else {
        program = new UnaryOpProgram(shape, unary_op.CLONE);
      }
      const res =
          this.runWebGLProgram(program, [{dataId, shape, dtype}], dtype);
      const data = this.readSync(res.dataId);
      this.disposeIntermediateTensorInfo(res);
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
      result = backend_util.mergeRealAndImagArrays(realValues, imagValues);
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
    const {values, shape, slice, dtype, complexTensors, isPacked} = texData;

    // The presence of `slice` indicates this tensor is a shallow slice of a
    // different tensor, and is using that original tensor's texture. Run
    // `clone` in order to copy that texture and read from it.
    if (slice != null) {
      let program;
      if (isPacked) {
        program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
      } else {
        program = new UnaryOpProgram(shape, unary_op.CLONE);
      }
      const res =
          this.runWebGLProgram(program, [{dataId, shape, dtype}], dtype);
      const data = this.read(res.dataId);
      this.disposeIntermediateTensorInfo(res);
      return data;
    }

    if (values != null) {
      return this.convertAndCacheOnCPU(dataId);
    }

    if (!env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
        env().getNumber('WEBGL_VERSION') === 2) {
      throw new Error(
          `tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and ` +
          `WEBGL_VERSION=2 not yet supported.`);
    }

    let buffer = null;
    let tmpDownloadTarget: TensorInfo;

    if (dtype !== 'complex64' && env().get('WEBGL_BUFFER_SUPPORTED')) {
      // Possibly copy the texture into a buffer before inserting a fence.
      tmpDownloadTarget = this.decode(dataId);
      const tmpData = this.texData.get(tmpDownloadTarget.dataId);

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
      const ps = await Promise.all(
          [complexTensors.real.data(), complexTensors.imag.data()]);
      const realValues = ps[0];
      const imagValues = ps[1];
      vals = backend_util.mergeRealAndImagArrays(
          realValues as Float32Array, imagValues as Float32Array);
    } else if (buffer == null) {
      vals = this.getValuesFromTexture(dataId);
    } else {
      const size = util.sizeFromShape(shape);
      vals = this.gpgpu.downloadFloat32MatrixFromBuffer(buffer, size);
    }
    if (tmpDownloadTarget != null) {
      this.disposeIntermediateTensorInfo(tmpDownloadTarget);
    }
    const dTypeVals = this.convertAndCacheOnCPU(dataId, vals);

    const subscribers = this.pendingRead.get(dataId);
    this.pendingRead.delete(dataId);

    // Notify all pending reads.
    subscribers.forEach(resolve => resolve(dTypeVals));
    if (this.pendingDisposal.has(dataId)) {
      this.pendingDisposal.delete(dataId);
      this.disposeData(dataId);
      this.pendingDeletes--;
    }
    return dTypeVals;
  }

  private checkNumericalProblems(values: BackendValues): void {
    if (values == null) {
      return;
    }
    for (let i = 0; i < values.length; i++) {
      const num = values[i] as number;
      if (!webgl_util.canBeRepresented(num)) {
        if (env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
          throw Error(
              `The value ${num} cannot be represented with your ` +
              `current settings. Consider enabling float32 rendering: ` +
              `'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`);
        }
        throw Error(`The value ${num} cannot be represented on this device.`);
      }
    }
  }

  private getValuesFromTexture(dataId: DataId): Float32Array {
    const {shape, dtype, isPacked} = this.texData.get(dataId);
    const size = util.sizeFromShape(shape);
    if (env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
      const tmpTarget = this.decode(dataId);
      const tmpData = this.texData.get(tmpTarget.dataId);
      const vals = this.gpgpu
                       .downloadMatrixFromPackedTexture(
                           tmpData.texture, ...tex_util.getDenseTexShape(shape))
                       .subarray(0, size);

      this.disposeIntermediateTensorInfo(tmpTarget);

      return vals;
    }

    const shouldUsePackedProgram =
        env().getBool('WEBGL_PACK') && isPacked === true;
    const outputShape =
        shouldUsePackedProgram ? webgl_util.getShapeAs3D(shape) : shape;
    const program = shouldUsePackedProgram ?
        new EncodeFloatPackedProgram(outputShape as [number, number, number]) :
        new EncodeFloatProgram(outputShape);
    const output = this.runWebGLProgram(
        program, [{shape: outputShape, dtype, dataId}], 'float32');
    const tmpData = this.texData.get(output.dataId);
    const vals =
        this.gpgpu
            .downloadByteEncodedFloatMatrixFromOutputTexture(
                tmpData.texture, tmpData.texShape[0], tmpData.texShape[1])
            .subarray(0, size);
    this.disposeIntermediateTensorInfo(output);

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

    const res: WebGLTimingInfo = {
      uploadWaitMs: this.uploadWaitMs,
      downloadWaitMs: this.downloadWaitMs,
      kernelMs: null,
      wallMs: null  // will be filled by the engine
    };

    if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
      const kernelMs = await Promise.all(flattenedActiveTimerQueries);

      res['kernelMs'] = util.sum(kernelMs);
      res['getExtraProfileInfo'] = () =>
          kernelMs.map((d, i) => ({name: flattenedActiveTimerNames[i], ms: d}))
              .map(d => `${d.name}: ${d.ms}`)
              .join(', ');
    } else {
      res['kernelMs'] = {
        error: 'WebGL query timers are not supported in this environment.'
      };
    }

    this.uploadWaitMs = 0;
    this.downloadWaitMs = 0;
    return res;
  }
  memory(): WebGLMemoryInfo {
    return {
      unreliable: false,
      numBytesInGPU: this.numBytesInGPU,
      numBytesInGPUAllocated: this.textureManager.numBytesAllocated,
      numBytesInGPUFree: this.textureManager.numBytesFree
    } as WebGLMemoryInfo;
  }

  private startTimer(): WebGLQuery|CPUTimerQuery {
    if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
      return this.gpgpu.beginQuery();
    }
    return {startMs: util.now(), endMs: null};
  }

  private endTimer(query: WebGLQuery|CPUTimerQuery): WebGLQuery|CPUTimerQuery {
    if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
      this.gpgpu.endQuery();
      return query;
    }
    (query as CPUTimerQuery).endMs = util.now();
    return query;
  }

  private async getQueryTime(query: WebGLQuery|CPUTimerQuery): Promise<number> {
    if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
      return this.gpgpu.waitForQueryAndGetTime(query as WebGLQuery);
    }
    const timerQuery = query as CPUTimerQuery;
    return timerQuery.endMs - timerQuery.startMs;
  }

  private pendingDeletes = 0;

  disposeData(dataId: DataId): void {
    if (this.pendingDisposal.has(dataId)) {
      return;
    }
    if (this.pendingRead.has(dataId)) {
      this.pendingDisposal.add(dataId);
      this.pendingDeletes++;
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

  /**
   * Returns internal information for the specific data bucket. Used in unit
   * tests.
   */
  getDataInfo(dataId: DataId): TextureData {
    return this.texData.get(dataId);
  }

  private getCPUBackend(): KernelBackend|null {
    if (!env().getBool('WEBGL_CPU_FORWARD')) {
      return null;
    }

    if (this.cpuBackend == null) {
      this.cpuBackend = engine().findBackend('cpu');
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
  shouldExecuteOnCPU(
      inputs: TensorInfo[],
      sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD): boolean {
    const cpuBackend = this.getCPUBackend();
    if (!this.warnedAboutCPUBackend && cpuBackend == null) {
      console.warn(
          'Your application contains ops that are small enough to be ' +
          'executed on the CPU backend, however the CPU backend cannot ' +
          'be found. Consider importing the CPU backend ' +
          '(@tensorflow/tfjs-backend-cpu) for better performance.');

      this.warnedAboutCPUBackend = true;
    }

    return cpuBackend != null &&
        inputs.every(
            input => this.texData.get(input.dataId).texture == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
  }

  getGPGPUContext(): GPGPUContext {
    return this.gpgpu;
  }

  complex<T extends Tensor>(real: T, imag: T): T {
    const result = this.makeOutput(real.shape, 'complex64');
    const resultData = this.texData.get(result.dataId);
    // The backend owns the reference to the underlying real and imaginary
    // clones. These will explicitly get disposed when the complex tensor is
    // disposed.
    resultData.complexTensors = {
      real: engine().keep(real.clone()),
      imag: engine().keep(imag.clone())
    };

    return result as T;
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
      const outValues = sliceImplCPU(
          this.texData.get(x.dataId).values as TypedArray, begin, size, x.shape,
          x.dtype);
      return this.makeOutput(size, x.dtype, outValues);
    }
    // Short-circuit computation if the slice is zero-sized.
    if (util.sizeFromShape(size) === 0) {
      return tensor([], size, x.dtype) as T;
    }
    const {isPacked} = this.texData.get(x.dataId);
    const isContinous = slice_util.isSliceContinous(x.shape, begin, size);
    if (isPacked || !isContinous) {
      const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
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
    const t = this.makeOutput(size, x.dtype);
    const newTexData = this.texData.get(t.dataId);
    // Copy texture data from the original tensor.
    Object.assign(newTexData, xTexData);
    newTexData.shape = size;
    newTexData.dtype = x.dtype;
    let flatOffset = slice_util.computeFlatOffset(begin, x.strides);
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
      x: T, begin: number[], end: number[], strides: number[]): T {
    const cpuRes = this.tryRunOnCpuOrThrow(
        [x], () => this.cpuBackend.stridedSlice(x, begin, end, strides));
    if (cpuRes) {
      return cpuRes;
    }

    const outShape = slice_util.computeOutShape(begin, end, strides);

    if (outShape.some(axis => axis === 0)) {
      return tensor([], outShape) as T;
    }

    const program = new StridedSliceProgram(begin, strides, outShape);
    return this.compileAndRun(program, [x]);
  }

  reverse<T extends Tensor>(x: T, axis: number[]): T {
    const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
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

    if (tensors.length === 1) {
      return tensors[0];
    }
    if (tensors.length > env().getNumber('WEBGL_MAX_TEXTURES_IN_SHADER')) {
      const midIndex = Math.floor(tensors.length / 2);
      const leftSide = this.concat(tensors.slice(0, midIndex), axis);
      const rightSide = this.concat(tensors.slice(midIndex), axis);
      return this.concat([leftSide, rightSide], axis);
    }
    if (env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') && tensors[0].rank > 1) {
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
    const outShape =
        backend_util.computeOutShape(tensors.map(t => t.shape), axis);
    const tensors2D =
        tensors.map(t => t.as2D(-1, util.sizeFromShape(t.shape.slice(axis))));
    const program = new ConcatProgram(tensors2D.map(t => t.shape));
    const res: Tensor = this.compileAndRun(program, tensors2D);
    return res.reshape(outShape);
  }

  neg<T extends Tensor>(x: T): T {
    const cpuRes = this.tryRunOnCpuOrThrow([x], () => this.cpuBackend.neg(x));
    if (cpuRes) {
      return cpuRes;
    }

    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
      return this.packedUnaryOp(x, unary_op.NEG, x.dtype) as T;
    }
    const program = new UnaryOpProgram(x.shape, unary_op.NEG);
    return this.compileAndRun(program, [x]);
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
        a = transpose(a, [0, 2, 1]);
      }
      if (transposeB) {
        b = transpose(b, [0, 2, 1]);
      }

      const a3D = outerShapeB === 1 ? a : a.as3D(batch, sharedDim, 1);
      const axis = outerShapeB === 1 ? 2 : 1;
      const b3D = outerShapeB === 1 ? b.as3D(batch, 1, sharedDim) : b;
      return this.multiply(a3D, b3D).sum(axis, true /* keepDims */);
    }

    const dtype = upcastType(a.dtype, b.dtype);

    const program = new MatMulPackedProgram(
        a.shape, [batch, outerShapeA, outerShapeB], transposeA, transposeB);
    return this.compileAndRun<Tensor3D>(program, [a, b], dtype);
  }

  fusedBatchMatMul(
      {a, b, transposeA, transposeB, bias, activation, preluActivationWeights}:
          backend_util.FusedBatchMatMulConfig): Tensor3D {
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
    const inputs: TensorInfo[] = [a, b];
    if (bias) {
      inputs.push(bias);
    }
    if (preluActivationWeights) {
      inputs.push(preluActivationWeights);
    }
    return this.compileAndRun<Tensor3D>(program, inputs, dtype);
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
        this.makeComplexComponentTensorInfo(a, aData.complexTensors.real),
        this.makeComplexComponentTensorInfo(a, aData.complexTensors.imag),
        this.makeComplexComponentTensorInfo(b, bData.complexTensors.real),
        this.makeComplexComponentTensorInfo(b, bData.complexTensors.imag)
      ];
      const real = this.compileAndRun<Tensor>(realProgram, inputs);
      const imag = this.compileAndRun<Tensor>(imagProgram, inputs);

      const complex = this.complex(real, imag);
      real.dispose();
      imag.dispose();
      return complex;
    }

    const dtype = upcastType(a.dtype, b.dtype);
    if (this.shouldExecuteOnCPU([a, b])) {
      const aData = this.texData.get(a.dataId);
      const bData = this.texData.get(b.dataId);
      const [outValues, outShape] = multiplyImplCPU(
          a.shape, b.shape, aData.values as TypedArray,
          bData.values as TypedArray, dtype);
      return this.makeOutput(outShape, dtype, outValues);
    }
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_gpu.MUL, a.dtype);
    }
    const program = new BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], a.dtype);
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const program = env().getBool('WEBGL_PACK_NORMALIZATION') ?
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
    const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
        new PadPackedProgram(x.shape, paddings, constantValue) :
        new PadProgram(x.shape, paddings, constantValue);
    return this.compileAndRun(program, [x]);
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    const cpuRes = this.tryRunOnCpuOrThrow(
        [x, indices], () => this.cpuBackend.gather(x, indices, axis));
    if (cpuRes) {
      return cpuRes;
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

    const reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
    const permuted =
        backend_util.getPermuted(reshaped.length, blockShape.length);
    const reshapedPermuted =
        backend_util.getReshapedPermuted(x.shape, blockShape, prod);
    const sliceBeginCoords =
        backend_util.getSliceBeginCoords(crops, blockShape.length);
    const sliceSize =
        backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);

    return transpose(x.reshape(reshaped), permuted)
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
        backend_util.getReshaped(paddedX.shape, blockShape, prod, false);

    const permutedReshapedPaddedPermutation = backend_util.getPermuted(
        reshapedPaddedShape.length, blockShape.length, false);

    const flattenShape = backend_util.getReshapedPermuted(
        paddedX.shape, blockShape, prod, false);

    const paddedXT = transpose(
        paddedX.reshape(reshapedPaddedShape),
        permutedReshapedPaddedPermutation);
    return reshape(paddedXT, flattenShape) as T;
  }

  private reduce(
      x: Tensor2D, reduceType: 'all'|'any'|'max'|'min'|'sum'|'prod',
      dtype: DataType): Tensor2D {
    const batchSize = x.shape[0];
    const inSize = x.shape[1];
    const windowSize = backend_util.computeOptimalWindowSize(inSize);
    const outSize = Math.ceil(inSize / windowSize);
    const reduceInfo = {windowSize, inSize, batchSize, outSize};
    const program = new ReduceProgram(reduceInfo, reduceType);
    const output = this.compileAndRun<Tensor2D>(program, [x], dtype);
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
    const windowSize = backend_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {
      windowSize,
      inSize,
      batchSize,
      outSize: Math.ceil(inSize / windowSize)
    };
    const program =
        new ArgMinMaxProgram(reduceInfo, reduceType, bestIndicesA == null);
    const inputs = [x];
    if (bestIndicesA != null) {
      inputs.push(bestIndicesA);
    }
    const output = this.compileAndRun<Tensor2D>(program, inputs, 'int32');
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
    const windowSize = backend_util.computeOptimalWindowSize(inSize);
    const program = new ArgMinMaxPackedProgram(
        inShape, windowSize, reduceType, bestIndicesA == null);
    const inputs = bestIndicesA == null ? [x] : [x, bestIndicesA];
    const output = this.compileAndRun<Tensor>(program, inputs, 'int32');
    if (output.rank === x.rank) {
      return this.argReducePacked(x, reduceType, output);
    }
    return output;
  }

  sum(x: Tensor, axes: number[]): Tensor {
    backend_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    const outputDType = tf.sumOutType(x.dtype);
    return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
  }

  prod(x: Tensor, axes: number[]): Tensor {
    const cpuRes =
        this.tryRunOnCpuOrThrow([x], () => this.cpuBackend.prod(x, axes));
    if (cpuRes) {
      return cpuRes;
    }

    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    const outputDType = tf.sumOutType(x.dtype);
    return this.reduce(a2D, 'prod', outputDType).reshape(outShape);
  }

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    let axis = 0;
    const permutation = backend_util.getAxesPermutation([axis], x.rank);
    let permutedX = x;
    if (permutation != null) {
      permutedX = transpose(x, permutation);
      axis = backend_util.getInnerMostAxes(1, x.rank)[0];
    }

    const outShape =
        segment_util.computeOutShape(permutedX.shape, axis, numSegments);
    const inSize = util.sizeFromShape([permutedX.shape[axis]]);
    const a2D = permutedX.as2D(-1, inSize);
    const outputDType = tf.sumOutType(x.dtype);
    let result =
        this.segOpCompute(
                a2D, 'unsortedSegmentSum', segmentIds, outputDType, numSegments)
            .reshape(outShape);
    if (permutation != null) {
      result =
          transpose(result, backend_util.getUndoAxesPermutation(permutation));
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
    const output =
        this.compileAndRun<Tensor2D>(program, [x, segmentIds], dtype);
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
    backend_util.assertAxesAreInnerMostDims(
        'arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes,
        x.rank);
    if (!env().getBool('WEBGL_PACK_REDUCE') || x.rank <= 2) {
      const [outShape, reduceShape] =
          backend_util.computeOutAndReduceShapes(x.shape, axes);
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
    const size = x.shape[axis];
    let result = x;
    // Use cumsum parallel algorithm, ref:
    // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    for (let i = 0; i <= Math.ceil(Math.log2(size)) - 1; i++) {
      const program = new CumSumProgram(x.shape, false, reverse);
      const customSetup = program.getCustomSetupFunc(i);
      const prevResult = result;
      result = this.compileAndRun(program, [result], result.dtype, customSetup);
      prevResult.dispose();
    }
    // For exclusive cumsum, shift the end result in the direction of sum and
    // add 0 to the front index.
    if (exclusive) {
      const program = new CumSumProgram(x.shape, exclusive, reverse);
      const prevResult = result;
      result = this.compileAndRun(program, [result]);
      prevResult.dispose();
    }

    return result;
  }

  equal(a: Tensor, b: Tensor): Tensor {
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.EQUAL, 'bool');
    }
    const program = new BinaryOpProgram(binaryop_gpu.EQUAL, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], 'bool');
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.NOT_EQUAL, 'bool');
    }
    const program =
        new BinaryOpProgram(binaryop_gpu.NOT_EQUAL, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], 'bool');
  }

  less(a: Tensor, b: Tensor): Tensor {
    const cpuRes =
        this.tryRunOnCpuOrThrow([a, b], () => this.cpuBackend.less(a, b));
    if (cpuRes) {
      return cpuRes;
    }

    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.LESS, 'bool');
    }

    const program = new BinaryOpProgram(binaryop_gpu.LESS, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], 'bool');
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.LESS_EQUAL, 'bool');
    }
    const program =
        new BinaryOpProgram(binaryop_gpu.LESS_EQUAL, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], 'bool');
  }

  greater(a: Tensor, b: Tensor): Tensor {
    const cpuRes =
        this.tryRunOnCpuOrThrow([a, b], () => this.cpuBackend.greater(a, b));
    if (cpuRes) {
      return cpuRes;
    }

    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.GREATER, 'bool');
    }

    const program = new BinaryOpProgram(binaryop_gpu.GREATER, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], 'bool');
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(
          a, b, binaryop_packed_gpu.GREATER_EQUAL, 'bool');
    }
    const program =
        new BinaryOpProgram(binaryop_gpu.GREATER_EQUAL, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], 'bool');
  }

  logicalNot<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOGICAL_NOT);
    return this.compileAndRun(program, [x]);
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.LOGICAL_AND, 'bool');
    }
    const program =
        new BinaryOpProgram(binaryop_gpu.LOGICAL_AND, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], 'bool');
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_packed_gpu.LOGICAL_OR, 'bool');
    }
    const program =
        new BinaryOpProgram(binaryop_gpu.LOGICAL_OR, a.shape, b.shape);
    return this.compileAndRun(program, [a, b], 'bool');
  }

  select(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    const program = new SelectProgram(condition.rank, a.shape, a.rank);
    return this.compileAndRun(
        program, [condition, a, b], upcastType(a.dtype, b.dtype));
  }

  where(condition: Tensor): Tensor2D {
    backend_util.warn(
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
    backend_util.assertAxesAreInnerMostDims('min', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'min', a2D.dtype).reshape(outShape);
  }

  minimum(a: Tensor, b: Tensor): Tensor {
    const cpuRes =
        this.tryRunOnCpuOrThrow([a, b], () => this.cpuBackend.minimum(a, b));
    if (cpuRes) {
      return cpuRes;
    }

    const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.MIN, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.MIN, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  mod(a: Tensor, b: Tensor): Tensor {
    const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.MOD, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.MOD, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  maximum(a: Tensor, b: Tensor): Tensor {
    const cpuRes =
        this.tryRunOnCpuOrThrow([a, b], () => this.cpuBackend.maximum(a, b));
    if (cpuRes) {
      return cpuRes;
    }

    const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.MAX, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.MAX, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  all(x: Tensor, axes: number[]): Tensor {
    backend_util.assertAxesAreInnerMostDims('all', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'all', a2D.dtype).reshape(outShape);
  }

  any(x: Tensor, axes: number[]): Tensor {
    backend_util.assertAxesAreInnerMostDims('any', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'any', a2D.dtype).reshape(outShape);
  }

  floorDiv(a: Tensor, b: Tensor): Tensor {
    const op = binaryop_gpu.INT_DIV;
    const outputDtype = 'int32';
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(
          a, b, binaryop_packed_gpu.INT_DIV, outputDtype);
    }
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    return this.compileAndRun<Tensor>(program, [a, b], outputDtype);
  }

  add(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' && b.dtype === 'complex64') {
      return this.complexSeparableBinaryOp(a, b, binaryop_gpu.ADD);
    }

    const dtype = upcastType(a.dtype, b.dtype);
    if (this.shouldExecuteOnCPU([a, b])) {
      const aData = this.texData.get(a.dataId);
      const bData = this.texData.get(b.dataId);
      const [outValues, outShape] = addImplCPU(
          a.shape, b.shape, aData.values as TypedArray,
          bData.values as TypedArray, dtype);
      return this.makeOutput(outShape, dtype, outValues);
    }

    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_gpu.ADD, dtype);
    }
    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    return this.compileAndRun<Tensor>(program, [a, b], dtype);
  }

  private packedUnaryOp(x: TensorInfo, op: string, dtype: DataType) {
    const program = new UnaryOpPackedProgram(x.shape, op);
    return this.compileAndRun<Tensor>(program, [x], dtype);
  }

  private packedBinaryOp(
      a: TensorInfo, b: TensorInfo, op: string, dtype: DataType,
      checkOutOfBounds = false) {
    const program =
        new BinaryOpPackedProgram(op, a.shape, b.shape, checkOutOfBounds);
    return this.compileAndRun<Tensor>(program, [a, b], dtype);
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

      const aHandle = this.makeComplexComponentTensorInfo(a, aPart);
      const bHandle = this.makeComplexComponentTensorInfo(b, bPart);

      const program = new BinaryOpProgram(op, a.shape, b.shape);
      return this.compileAndRun<Tensor>(
          program, [aHandle, bHandle], upcastType(aPart.dtype, bPart.dtype));
    });

    const complex = this.complex(real, imag);
    real.dispose();
    imag.dispose();
    return complex;
  }

  // Returns a TensorInfo with the complex shape and the dataId of the
  // underlying part. We need to do this because a reshaped complex tensor is
  // not reflected in its parts.
  private makeComplexComponentTensorInfo(
      complexTensor: Tensor, complexPart: Tensor): TensorInfo {
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
    if (tensors.length > env().get('WEBGL_MAX_TEXTURES_IN_SHADER')) {
      const midIndex = Math.floor(tensors.length / 2);
      const leftSide = this.addN(tensors.slice(0, midIndex));
      const rightSide = this.addN(tensors.slice(midIndex));
      return this.addN([leftSide, rightSide]);
    }

    const dtype =
        tensors.map(t => t.dtype).reduce((d1, d2) => upcastType(d1, d2));
    const shapes = tensors.map(t => t.shape);
    // We can make sure shapes are identical in op level.
    const usePackedOp = env().getBool('WEBGL_PACK');
    const program = usePackedOp ?
        new AddNPackedProgram(tensors[0].shape, shapes) :
        new AddNProgram(tensors[0].shape, shapes);
    return this.compileAndRun<T>(program, tensors, dtype);
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' && b.dtype === 'complex64') {
      return this.complexSeparableBinaryOp(a, b, binaryop_gpu.SUB);
    }

    const dtype = upcastType(a.dtype, b.dtype);
    if (this.shouldExecuteOnCPU([a, b])) {
      const aData = this.texData.get(a.dataId);
      const bData = this.texData.get(b.dataId);
      const [outValues, outShape] = subImplCPU(
          a.shape, b.shape, aData.values as TypedArray,
          bData.values as TypedArray, dtype);
      return this.makeOutput(outShape, dtype, outValues);
    }
    if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
      return this.packedBinaryOp(a, b, binaryop_gpu.SUB, a.dtype);
    }
    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    return this.compileAndRun<Tensor>(program, [a, b], dtype);
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    const usePackedOp = env().getBool('WEBGL_PACK_BINARY_OPERATIONS');
    const program = usePackedOp ?
        new BinaryOpPackedProgram(binaryop_packed_gpu.POW, a.shape, b.shape) :
        new BinaryOpProgram(binaryop_gpu.POW, a.shape, b.shape);
    const dtype = upcastType(a.dtype, b.dtype);
    return this.compileAndRun<T>(program, [a, b], dtype);
  }

  ceil<T extends Tensor>(x: T): T {
    if (this.shouldExecuteOnCPU([x])) {
      const outValues =
          ceilImplCPU(this.texData.get(x.dataId).values as TypedArray, x.dtype);
      return this.makeOutput(x.shape, x.dtype, outValues);
    }

    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
      return this.packedUnaryOp(x, unary_op.CEIL, x.dtype) as T;
    }

    const program = new UnaryOpProgram(x.shape, unary_op.CEIL);
    return this.compileAndRun(program, [x]);
  }

  floor<T extends Tensor>(x: T): T {
    if (this.shouldExecuteOnCPU([x])) {
      const outValues = floorImplCPU(
          this.texData.get(x.dataId).values as TypedArray, x.dtype);
      return this.makeOutput(x.shape, x.dtype, outValues);
    }

    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
      return this.packedUnaryOp(x, unary_op.FLOOR, x.dtype) as T;
    }

    const program = new UnaryOpProgram(x.shape, unary_op.FLOOR);
    return this.compileAndRun(program, [x]);
  }

  sign<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGN);
    return this.compileAndRun(program, [x]);
  }

  isNaN<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.IS_NAN);
    return this.compileAndRun(program, [x], 'bool');
  }
  isInf<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.IS_INF);
    return this.compileAndRun(program, [x], 'bool');
  }
  isFinite<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.IS_FINITE);
    return this.compileAndRun(program, [x], 'bool');
  }

  round<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ROUND);
    return this.compileAndRun(program, [x]);
  }

  exp<T extends Tensor>(x: T): T {
    if (this.shouldExecuteOnCPU([x])) {
      const outValues =
          expImplCPU(this.texData.get(x.dataId).values as TypedArray, x.dtype);
      return this.makeOutput(x.shape, x.dtype, outValues);
    }

    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
      return this.packedUnaryOp(x, unary_op.EXP, x.dtype) as T;
    }

    const program = new UnaryOpProgram(x.shape, unary_op.EXP);
    return this.compileAndRun(program, [x]);
  }

  expm1<T extends Tensor>(x: T): T {
    if (this.shouldExecuteOnCPU([x])) {
      const outValues = expm1ImplCPU(
          this.texData.get(x.dataId).values as TypedArray, x.dtype);
      return this.makeOutput(x.shape, x.dtype, outValues);
    }

    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
      return this.packedUnaryOp(x, unary_op.EXPM1, x.dtype) as T;
    }

    const program = new UnaryOpProgram(x.shape, unary_op.EXPM1);
    return this.compileAndRun(program, [x]);
  }

  softmax<T extends Tensor>(logits: T, dim: number): T {
    const axes = util.parseAxisParam([dim], logits.shape);
    // TODO(annxingyuan): Call maxImpl rather than op as part of softmax kernel
    // modularization.
    const maxLogit = max(logits, axes);
    const expandedShape =
        backend_util.expandShapeToKeepDim(maxLogit.shape, axes);
    const a = this.subtract(logits, maxLogit.reshape(expandedShape));
    const b = this.exp(a);
    const sumExp = this.sum(b, axes).reshape(expandedShape);

    // TODO(annxingyuan): Call divImpl rather than op as part of softmax kernel
    // modularization.
    return div(b, sumExp);
  }

  log<T extends Tensor>(x: T): T {
    if (this.shouldExecuteOnCPU([x])) {
      const outValues =
          logImplCPU(this.texData.get(x.dataId).values as TypedArray, x.dtype);
      return this.makeOutput(x.shape, x.dtype, outValues);
    }

    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
      return this.packedUnaryOp(x, unary_packed_op.LOG, x.dtype) as T;
    }

    const program = new UnaryOpProgram(x.shape, unary_op.LOG);
    return this.compileAndRun(program, [x]);
  }

  log1p<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOG1P);
    return this.compileAndRun(program, [x]);
  }

  sqrt<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQRT);
    return this.compileAndRun(program, [x]);
  }

  rsqrt<T extends Tensor>(x: T): T {
    if (this.shouldExecuteOnCPU([x])) {
      const outValues = rsqrtImplCPU(
          this.texData.get(x.dataId).values as TypedArray, x.dtype);
      return this.makeOutput(x.shape, x.dtype, outValues);
    }
    const program = new UnaryOpProgram(x.shape, unary_op.RSQRT);
    return this.compileAndRun(program, [x]);
  }

  reciprocal<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RECIPROCAL);
    return this.compileAndRun(program, [x]);
  }

  relu<T extends Tensor>(x: T): T {
    let program: UnaryOpProgram|UnaryOpPackedProgram;
    if (env().getBool('WEBGL_PACK')) {
      program = new UnaryOpPackedProgram(x.shape, unary_packed_op.RELU);
    } else {
      program = new UnaryOpProgram(x.shape, unary_op.RELU);
    }
    return this.compileAndRun(program, [x]);
  }

  relu6<T extends Tensor>(x: T): T {
    let program: UnaryOpProgram|UnaryOpPackedProgram;
    if (env().getBool('WEBGL_PACK')) {
      program = new UnaryOpPackedProgram(x.shape, unary_packed_op.RELU6);
    } else {
      program = new UnaryOpProgram(x.shape, unary_op.RELU6);
    }
    return this.compileAndRun(program, [x]);
  }

  prelu<T extends Tensor>(x: T, alpha: T): T {
    const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(
            binaryop_packed_gpu.PRELU, x.shape, alpha.shape) :
        new BinaryOpProgram(binaryop_gpu.PRELU, x.shape, alpha.shape);
    return this.compileAndRun(program, [x, alpha]);
  }

  elu<T extends Tensor>(x: T): T {
    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
      return this.packedUnaryOp(x, unary_packed_op.ELU, x.dtype) as T;
    }
    const program = new UnaryOpProgram(x.shape, unary_op.ELU);
    return this.compileAndRun(program, [x]);
  }

  eluDer<T extends Tensor>(dy: T, y: T): T {
    const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(
            binaryop_packed_gpu.ELU_DER, dy.shape, y.shape) :
        new BinaryOpProgram(binaryop_gpu.ELU_DER, dy.shape, y.shape);
    return this.compileAndRun(program, [dy, y]);
  }

  selu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SELU);
    return this.compileAndRun(program, [x]);
  }

  int<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TO_INT);
    return this.compileAndRun(program, [x], 'int32');
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    let program;
    if (env().getBool('WEBGL_PACK_CLIP')) {
      program = new ClipPackedProgram(x.shape);
    } else {
      program = new ClipProgram(x.shape);
    }
    const customSetup = program.getCustomSetupFunc(min, max);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  abs<T extends Tensor>(x: T): T {
    // TODO: handle cases when x is complex.
    if (this.shouldExecuteOnCPU([x]) && x.dtype !== 'complex64') {
      const outValues =
          simpleAbsImplCPU(this.texData.get(x.dataId).values as TypedArray);
      return this.makeOutput(x.shape, x.dtype, outValues);
    }

    if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
      return this.packedUnaryOp(x, unary_op.ABS, x.dtype) as T;
    }

    const program = new UnaryOpProgram(x.shape, unary_op.ABS);
    return this.compileAndRun(program, [x]);
  }

  complexAbs<T extends Tensor>(x: T): T {
    const xData = this.texData.get(x.dataId);

    const program = new ComplexAbsProgram(x.shape);
    const inputs = [
      this.makeComplexComponentTensorInfo(x, xData.complexTensors.real),
      this.makeComplexComponentTensorInfo(x, xData.complexTensors.imag),
    ];

    return this.compileAndRun<Tensor>(program, inputs) as T;
  }

  sigmoid<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGMOID);
    return this.compileAndRun(program, [x]);
  }

  softplus<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SOFTPLUS);
    return this.compileAndRun(program, [x]);
  }

  asin<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ASIN);
    return this.compileAndRun(program, [x]);
  }

  acos<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ACOS);
    return this.compileAndRun(program, [x]);
  }

  atan<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATAN);
    return this.compileAndRun(program, [x]);
  }

  sinh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SINH);
    return this.compileAndRun(program, [x]);
  }

  cosh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COSH);
    return this.compileAndRun(program, [x]);
  }

  tanh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TANH);
    return this.compileAndRun(program, [x]);
  }

  asinh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ASINH);
    return this.compileAndRun(program, [x]);
  }

  acosh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ACOSH);
    return this.compileAndRun(program, [x]);
  }

  atanh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATANH);
    return this.compileAndRun(program, [x]);
  }

  erf<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ERF);
    return this.compileAndRun(program, [x]);
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    const program = new UnaryOpProgram(x.shape, unary_op.STEP(alpha));
    return this.compileAndRun(program, [x]);
  }

  private conv2dByMatMul(
      x: Tensor4D, filter: Tensor4D, convInfo: backend_util.Conv2DInfo,
      bias?: Tensor, activation?: backend_util.Activation,
      preluActivationWeights?: Tensor): Tensor4D {
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

    if (batchMatMulWillBeUnpacked || !env().getBool('WEBGL_LAZILY_UNPACK') ||
        !env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ||
        !reshapeWillBeExpensive) {
      const targetShape = isChannelsLast ? xShape[0] * xShape[1] * xShape[2] :
                                           xShape[0] * xShape[2] * xShape[3];
      const xReshaped = reshape(x, [1, targetShape, convInfo.inChannels]);
      const filterReshaped =
          reshape(filter, [1, convInfo.inChannels, convInfo.outChannels]);

      const result = this.fusedBatchMatMul({
        a: xReshaped as Tensor3D,
        b: filterReshaped as Tensor3D,
        transposeA,
        transposeB,
        bias,
        activation,
        preluActivationWeights
      });
      return reshape(result, convInfo.outShape);
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
    const xReshaped: TensorInfo = {
      dataId: x.dataId,
      shape: [1, targetShape, convInfo.inChannels],
      dtype: x.dtype
    };
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
        reshape(filter, [1, convInfo.inChannels, convInfo.outChannels]);

    const pointwiseConv = this.fusedBatchMatMul({
      a: xReshaped as Tensor3D,
      b: filterReshaped as Tensor3D,
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
    return engine().makeTensorFromDataId(
               pointwiseConv.dataId, convInfo.outShape, pointwiseConv.dtype) as
        Tensor4D;
  }

  private conv2dWithIm2Row(
      x: Tensor4D, filter: Tensor4D, convInfo: backend_util.Conv2DInfo,
      bias?: Tensor, activation?: backend_util.Activation,
      preluActivationWeights?: Tensor): Tensor4D {
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
    const w2Row = filter.reshape([1, sharedDim, -1]);

    const im2ColProgram =
        new Im2ColPackedProgram(x2ColShape, xSqueezed.shape, convInfo);
    const im2Col: Tensor3D =
        this.compileAndRun<Tensor2D>(im2ColProgram, [xSqueezed]).reshape([
          1, x2ColShape[0], x2ColShape[1]
        ]);

    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const fusedActivation =
        activation ? mapActivationToShaderProgram(activation, true) : null;
    const matmulProgram = new MatMulPackedProgram(
        im2Col.shape, [1, numCols, convInfo.outChannels], transposeA,
        transposeB, hasBias, fusedActivation, hasPreluActivationWeights);
    const inputs: TensorInfo[] = [im2Col, w2Row];
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
      {input, filter, convInfo, bias, activation, preluActivationWeights}:
          backend_util.FusedConv2DConfig): Tensor4D {
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
        (convInfo.padInfo.type === 'SAME' ||
         convInfo.padInfo.type === 'VALID')) {
      return this.conv2dByMatMul(
          input, filter, convInfo, bias, activation, preluActivationWeights);
    }
    if (env().getBool('WEBGL_CONV_IM2COL') && input.shape[0] === 1) {
      return this.conv2dWithIm2Row(
          input, filter, convInfo, bias, activation, preluActivationWeights);
    }

    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const fusedActivation =
        activation ? mapActivationToShaderProgram(activation, false) : null;
    const program = new Conv2DProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
    const inputs: TensorInfo[] = [input, filter];
    if (bias) {
      inputs.push(bias);
    }
    if (preluActivationWeights) {
      inputs.push(preluActivationWeights);
    }
    return this.compileAndRun(program, inputs);
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: backend_util.Conv2DInfo):
      Tensor4D {
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
        (convInfo.padInfo.type === 'SAME' ||
         convInfo.padInfo.type === 'VALID')) {
      return this.conv2dByMatMul(x, filter, convInfo);
    }
    if (env().getBool('WEBGL_CONV_IM2COL') && x.shape[0] === 1) {
      return this.conv2dWithIm2Row(x, filter, convInfo);
    }
    const program = new Conv2DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
  }

  conv2dDerInput(
      dy: Tensor4D, filter: Tensor4D,
      convInfo: backend_util.Conv2DInfo): Tensor4D {
    const program = new Conv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: backend_util.Conv2DInfo):
      Tensor4D {
    const program = new Conv2DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dy]);
  }

  fusedDepthwiseConv2D(
      {input, filter, convInfo, bias, activation, preluActivationWeights}:
          backend_util.FusedConv2DConfig): Tensor4D {
    const shouldPackDepthwiseConv = env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
        convInfo.strideWidth <= 2 &&
        convInfo.outChannels / convInfo.inChannels === 1;
    const fusedActivation = activation ?
        mapActivationToShaderProgram(activation, shouldPackDepthwiseConv) :
        null;
    const inputs: Tensor[] = [input, filter];

    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (hasBias) {
      inputs.push(bias);
    }
    if (hasPreluActivationWeights) {
      inputs.push(preluActivationWeights);
    }

    let program: DepthwiseConv2DProgram|DepthwiseConvPacked2DProgram;
    if (shouldPackDepthwiseConv) {
      program = new DepthwiseConvPacked2DProgram(
          convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
      return this.compileAndRun(program, inputs);
    }

    program = new DepthwiseConv2DProgram(
        convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
    return this.compileAndRun(program, inputs);
  }

  depthwiseConv2D(
      x: Tensor4D, filter: Tensor4D,
      convInfo: backend_util.Conv2DInfo): Tensor4D {
    let program: DepthwiseConv2DProgram|DepthwiseConvPacked2DProgram;
    if (env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
        convInfo.strideWidth <= 2 &&
        convInfo.outChannels / convInfo.inChannels === 1) {
      program = new DepthwiseConvPacked2DProgram(convInfo);
      return this.compileAndRun(program, [x, filter]);
    }

    program = new DepthwiseConv2DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
  }

  depthwiseConv2DDerInput(
      dy: Tensor4D, filter: Tensor4D,
      convInfo: backend_util.Conv2DInfo): Tensor4D {
    const program = new DepthwiseConv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  depthwiseConv2DDerFilter(
      x: Tensor4D, dy: Tensor4D, convInfo: backend_util.Conv2DInfo): Tensor4D {
    const program = new DepthwiseConv2DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dy]);
  }

  conv3d(x: Tensor5D, filter: Tensor5D, convInfo: backend_util.Conv3DInfo):
      Tensor5D {
    const program = new Conv3DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
  }

  conv3dDerInput(
      dy: Tensor5D, filter: Tensor5D,
      convInfo: backend_util.Conv3DInfo): Tensor5D {
    const program = new Conv3DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  conv3dDerFilter(x: Tensor5D, dy: Tensor5D, convInfo: backend_util.Conv3DInfo):
      Tensor5D {
    const program = new Conv3DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dy]);
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

  avgPool3d(x: Tensor5D, convInfo: backend_util.Conv3DInfo): Tensor5D {
    const program = new Pool3DProgram(convInfo, 'avg', false);
    return this.compileAndRun(program, [x], 'float32');
  }

  avgPool3dBackprop(
      dy: Tensor5D, x: Tensor5D, convInfo: backend_util.Conv3DInfo): Tensor5D {
    const avgPool3dBackpropProgram = new AvgPool3DBackpropProgram(convInfo);
    return this.compileAndRun(avgPool3dBackpropProgram, [dy], x.dtype);
  }

  maxPool3d(x: Tensor5D, convInfo: backend_util.Conv3DInfo): Tensor5D {
    const program = new Pool3DProgram(convInfo, 'max', false);
    return this.compileAndRun(program, [x], 'float32');
  }

  maxPool3dBackprop(
      dy: Tensor5D, x: Tensor5D, y: Tensor5D,
      convInfo: backend_util.Conv3DInfo): Tensor5D {
    const getPositions = true;
    const maxPool3dPositionsProgram =
        new Pool3DProgram(convInfo, 'max', getPositions);
    const maxPool3dPositions: Tensor5D =
        this.compileAndRun(maxPool3dPositionsProgram, [x]);
    const maxPool3dBackPropProgram = new MaxPool3DBackpropProgram(convInfo);
    const result = this.compileAndRun(
        maxPool3dBackPropProgram, [dy, maxPool3dPositions], x.dtype);
    maxPool3dPositions.dispose();
    return result as Tensor5D;
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const program = env().getBool('WEBGL_PACK_IMAGE_OPERATIONS') ?
        new ResizeBilinearPackedProgram(
            x.shape, newHeight, newWidth, alignCorners) :
        new ResizeBilinearProgram(x.shape, newHeight, newWidth, alignCorners);
    return this.compileAndRun(program, [x], 'float32');
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
    const customSetup = program.getCustomSetupFunc(seed);
    return this.compileAndRun(program, [probs], 'int32', customSetup);
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

  cropAndResize(
      image: Tensor4D, boxes: Tensor2D, boxIndex: Tensor1D,
      cropSize: [number, number], method: 'bilinear'|'nearest',
      extrapolationValue: number): Tensor4D {
    const program = new CropAndResizeProgram(
        image.shape, boxes.shape, cropSize, method, extrapolationValue);
    return this.compileAndRun(program, [image, boxes, boxIndex], 'float32');
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
        backend_util.calculateShapes(updates, indices, shape);

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
    const res: Tensor =
        this.compileAndRun(program, [flattenX, flattenIndices, defaultValue]);
    return res.reshape(shape);
  }

  sparseToDense<R extends Rank>(
      sparseIndices: Tensor, sparseValues: Tensor, outputShape: ShapeMap[R],
      defaultValue: Scalar): Tensor<R> {
    const {sliceRank, numUpdates, strides, outputSize} =
        backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);

    const sumDupeIndices = false;
    const program = new ScatterProgram(
        numUpdates, sliceRank, sparseIndices.rank, sparseValues.rank, strides,
        [outputSize, 1], sumDupeIndices);
    const res: Tensor = this.compileAndRun(
        program, [sparseValues, sparseIndices, defaultValue]);
    return res.reshape(outputShape);
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
      this.makeComplexComponentTensorInfo(x, xData.complexTensors.real),
      this.makeComplexComponentTensorInfo(x, xData.complexTensors.imag),
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
        backend_util.prepareAndValidate(x, indices);

    const flattenIndices = indices.reshape([numSlices, sliceRank]);
    const flattenX = x.reshape([x.size / sliceSize, sliceSize]);
    const program =
        new GatherNDProgram(sliceRank, strides, [numSlices, sliceSize]);
    const res: Tensor = this.compileAndRun(program, [flattenX, flattenIndices]);
    return res.reshape(resultShape);
  }

  fill<R extends Rank>(
      shape: ShapeMap[R], value: number|string, dtype?: DataType): Tensor<R> {
    dtype = dtype || util.inferDtype(value);

    if (dtype === 'string') {
      // String type should be handled in CPU memory.
      const values = util.getArrayFromDType(dtype, util.sizeFromShape(shape));
      values.fill(value as string);
      return engine().makeTensor(values, shape, dtype, this) as Tensor<R>;
    } else {
      const program = new FillProgram(shape, value as number);
      const customSetup = program.getCustomSetupFunc(value as number);
      return this.compileAndRun(program, [], dtype, customSetup);
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

  makeTensorInfo(shape: number[], dtype: DataType, values?: BackendValues):
      TensorInfo {
    const dataId = this.write(values, shape, dtype);
    this.texData.get(dataId).usage = null;
    return {dataId, shape, dtype};
  }

  private makeOutput<T extends Tensor>(
      shape: number[], dtype: DataType, values?: BackendValues): T {
    const {dataId} = this.makeTensorInfo(shape, dtype, values);
    return engine().makeTensorFromDataId(dataId, shape, dtype, this) as T;
  }

  private unpackTensor(input: TensorInfo): TensorInfo {
    const program = new UnpackProgram(input.shape);
    return this.runWebGLProgram(program, [input], input.dtype);
  }

  private packTensor(input: TensorInfo): TensorInfo {
    const program = new PackProgram(input.shape);
    const preventEagerUnpackingOutput = true;
    return this.runWebGLProgram(
        program, [input], input.dtype, null /* customSetup */,
        preventEagerUnpackingOutput);
  }

  private packedReshape(input: TensorInfo, afterShape: number[]): TensorInfo {
    const input3DShape = [
      webgl_util.getBatchDim(input.shape),
      ...webgl_util.getRowsCols(input.shape)
    ] as [number, number, number];
    const input3D: TensorInfo = {
      dtype: input.dtype,
      shape: input3DShape,
      dataId: input.dataId
    };
    const afterShapeAs3D = [
      webgl_util.getBatchDim(afterShape), ...webgl_util.getRowsCols(afterShape)
    ] as [number, number, number];

    const program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
    const preventEagerUnpackingOfOutput = true;
    const output = this.runWebGLProgram(
        program, [input3D], input.dtype, null /* customSetup */,
        preventEagerUnpackingOfOutput);
    return {dataId: output.dataId, shape: afterShape, dtype: output.dtype};
  }

  private decode(dataId: DataId): TensorInfo {
    const texData = this.texData.get(dataId);
    const {isPacked, shape, dtype} = texData;
    const shapeAs3D =
        webgl_util.getShapeAs3D(shape) as [number, number, number];
    let program;
    if (isPacked) {
      program = new DecodeMatrixPackedProgram(shapeAs3D);
    } else {
      program = new DecodeMatrixProgram(shapeAs3D);
    }
    const preventEagerUnpackingOfOutput = true;
    const out = this.runWebGLProgram(
        program, [{shape: shapeAs3D, dtype, dataId}], dtype,
        null /* customSetup */, preventEagerUnpackingOfOutput);
    return {dtype, shape, dataId: out.dataId};
  }

  runWebGLProgram(
      program: GPGPUProgram, inputs: TensorInfo[], outputDtype: DataType,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void,
      preventEagerUnpackingOfOutput = false): TensorInfo {
    const output = this.makeTensorInfo(program.outputShape, outputDtype);
    const outData = this.texData.get(output.dataId);
    if (program.packedOutput) {
      outData.isPacked = true;
    }
    if (program.outPackingScheme === tex_util.PackingScheme.DENSE) {
      const texelShape = tex_util.getDenseTexShape(program.outputShape);
      // For a densely packed output, we explicitly set texShape
      // so it doesn't get assigned later according to our typical packing
      // scheme wherein a single texel can only contain values from adjacent
      // rows/cols.
      outData.texShape = texelShape.map(d => d * 2) as [number, number];
    }
    if (program.outTexUsage != null) {
      outData.usage = program.outTexUsage;
    }
    if (util.sizeFromShape(output.shape) === 0) {
      // Short-circuit the computation since the result is empty (has 0 in its
      // shape).
      outData.values =
          util.getTypedArrayFromDType(output.dtype as 'float32', 0);
      return output;
    }

    const dataToDispose: TensorInfo[] = [];
    const inputsData: TensorData[] = inputs.map(input => {
      if (input.dtype === 'complex64') {
        throw new Error(
            `GPGPUProgram does not support complex64 input. For complex64 ` +
            `dtypes, please separate the program into real and imaginary ` +
            `parts.`);
      }

      let texData = this.texData.get(input.dataId);

      if (texData.texture == null) {
        if (!program.packedInputs &&
            util.sizeFromShape(input.shape) <=
                env().getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')) {
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
        if (program.packedInputs) {
          texData.isPacked = true;
          texData.shape = input.shape;
        }
      } else if (!!texData.isPacked !== !!program.packedInputs) {
        input = texData.isPacked ? this.unpackTensor(input) :
                                   this.packTensor(input);
        dataToDispose.push(input);
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
        dataToDispose.push(input);
        texData = this.texData.get(input.dataId);

        savedInput.shape = targetShape;
      }

      this.uploadToGPU(input.dataId);
      return {shape: input.shape, texData, isUniform: false};
    });

    this.uploadToGPU(output.dataId);
    const outputData:
        TensorData = {shape: output.shape, texData: outData, isUniform: false};
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

    dataToDispose.forEach(info => this.disposeIntermediateTensorInfo(info));

    if (shouldTimeProgram) {
      query = this.endTimer(query);
      this.activeTimers.push(
          {name: program.constructor.name, query: this.getQueryTime(query)});
    }

    if (!env().getBool('WEBGL_LAZILY_UNPACK') && outData.isPacked &&
        preventEagerUnpackingOfOutput === false) {
      const unpacked = this.unpackTensor(output);
      this.disposeIntermediateTensorInfo(output);
      return unpacked;
    }
    return output;
  }

  compileAndRun<K extends TensorInfo>(
      program: GPGPUProgram, inputs: TensorInfo[], outputDtype?: DataType,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void,
      preventEagerUnpackingOfOutput = false): K {
    outputDtype = outputDtype || inputs[0].dtype;
    const outInfo = this.runWebGLProgram(
        program, inputs, outputDtype, customSetup,
        preventEagerUnpackingOfOutput);
    return engine().makeTensorFromDataId(
               outInfo.dataId, outInfo.shape, outInfo.dtype) as {} as K;
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
    // Avoid disposing the compiled webgl programs during unit testing because
    // it slows down test execution.
    if (!env().getBool('IS_TEST')) {
      const allKeys = Object.keys(this.binaryCache);
      allKeys.forEach(key => {
        this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
        delete this.binaryCache[key];
      });
    }
    this.textureManager.dispose();
    if (this.canvas != null &&
        (typeof (HTMLCanvasElement) !== 'undefined' &&
         this.canvas instanceof HTMLCanvasElement)) {
      this.canvas.remove();
    } else {
      this.canvas = null;
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
        if (!env().get('WEBGL_RENDER_FLOAT32_ENABLED')) {
          // Momentarily switching DEBUG flag to false so we don't throw an
          // error trying to upload a small value.
          const debugFlag = env().getBool('DEBUG');
          env().set('DEBUG', false);
          const underflowCheckValue = this.abs(scalar(1e-8)).dataSync()[0];
          env().set('DEBUG', debugFlag);

          if (underflowCheckValue > 0) {
            return 32;
          }
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

      const tempDenseInputHandle = this.makeTensorInfo([height, width], dtype);
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

      // We want the output to remain packed regardless of the value of
      // WEBGL_PACK.
      const preventEagerUnpacking = true;
      const encodedOutputTarget = this.runWebGLProgram(
          program, [tempDenseInputHandle], dtype, null, preventEagerUnpacking);

      // Have the original texture assume the identity of the encoded output.
      const outputTexData = this.texData.get(encodedOutputTarget.dataId);
      texData.texture = outputTexData.texture;
      texData.texShape = outputTexData.texShape;
      texData.isPacked = outputTexData.isPacked;
      texData.usage = outputTexData.usage;

      this.disposeIntermediateTensorInfo(tempDenseInputHandle);
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

  private tryRunOnCpuOrThrow<T extends Tensor>(
      inputs: TensorInfo[], fn: () => T): T|null {
    if (this.shouldExecuteOnCPU(inputs)) {
      try {
        return fn();
      } catch (e) {
        if (env().getBool('IS_TEST')) {
          throw new Error('CPU forwarding failed');
        }
      }
    }
    return null;
  }
}

function float32ToTypedArray<D extends NumericDataType>(
    a: Float32Array, dtype: D): tf.DataTypeMap[D] {
  if (dtype === 'float32' || dtype === 'complex64') {
    return a as tf.DataTypeMap[D];
  } else if (dtype === 'int32' || dtype === 'bool') {
    const result = (dtype === 'int32') ? new Int32Array(a.length) :
                                         new Uint8Array(a.length);
    for (let i = 0; i < result.length; ++i) {
      result[i] = Math.round(a[i]);
    }
    return result as tf.DataTypeMap[D];
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}
