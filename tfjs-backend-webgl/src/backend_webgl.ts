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
import {backend_util, BackendValues, buffer, DataId, DataStorage, DataToGPUWebGLOption, DataType, engine, env, GPUData, kernel_impls, KernelBackend, MemoryInfo, nextFrame, NumericDataType, Rank, RecursiveArray, scalar, ShapeMap, Tensor, Tensor2D, TensorBuffer, TensorInfo, tidy, TimingInfo, TypedArray, util, WebGLData} from '@tensorflow/tfjs-core';
import {getWebGLContext} from './canvas_util';
import {DecodeMatrixProgram} from './decode_matrix_gpu';
import {DecodeMatrixPackedProgram} from './decode_matrix_packed_gpu';
import {EncodeFloatProgram} from './encode_float_gpu';
import {EncodeFloatPackedProgram} from './encode_float_packed_gpu';
import {EncodeMatrixProgram} from './encode_matrix_gpu';
import {EncodeMatrixPackedProgram} from './encode_matrix_packed_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {getUniformLocations, GPGPUBinary, GPGPUProgram, TensorData} from './gpgpu_math';
import {simpleAbsImplCPU} from './kernel_utils/shared';
import {PackProgram} from './pack_gpu';
import {ReshapePackedProgram} from './reshape_packed_gpu';
import * as tex_util from './tex_util';
import {Texture, TextureData, TextureUsage} from './tex_util';
import {TextureManager} from './texture_manager';
import * as unary_op from './unaryop_gpu';
import {UnaryOpProgram} from './unaryop_gpu';
import {UnaryOpPackedProgram} from './unaryop_packed_gpu';
import {UnpackProgram} from './unpack_gpu';
import * as webgl_util from './webgl_util';

const whereImpl = kernel_impls.whereImpl;

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

// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD =
    env().getNumber('CPU_HANDOFF_SIZE_THRESHOLD');

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

export class MathBackendWebGL extends KernelBackend {
  texData: DataStorage<TextureData>;
  gpgpu: GPGPUContext;

  private static nextDataId = 0;
  private nextDataId(): number {
    return MathBackendWebGL.nextDataId++;
  }
  // Maps data ids that have a pending read operation, to list of subscribers.
  private pendingRead = new WeakMap<DataId, Array<(arr: TypedArray) => void>>();
  // List of data ids that are scheduled for disposal, but are waiting on a
  // pending read operation.
  private pendingDisposal = new WeakSet<DataId>();

  // Used to count the number of 'shallow' sliced tensors that point to the
  // same data id.
  dataRefCount = new WeakMap<DataId, number>();
  private numBytesInGPU = 0;

  private canvas: HTMLCanvasElement|OffscreenCanvas;

  private programTimersStack: TimerNode[];
  private activeTimers: TimerNode[];
  // Accumulated time spent (including blocking) in uploading data to webgl.
  private uploadWaitMs = 0;
  // Accumulated time spent (including blocking in downloading data from webgl.
  private downloadWaitMs = 0;

  // record the last manual GL Flush time.
  private lastGlFlushTime = 0;

  // Number of bits of precision of this backend.
  private floatPrecisionValue: 32|16;

  private textureManager: TextureManager;
  private binaryCache: {[key: string]: GPGPUBinary};
  private gpgpuCreatedLocally: boolean;
  private numMBBeforeWarning: number;
  private warnedAboutMemory = false;

  constructor(gpuResource?: GPGPUContext|HTMLCanvasElement|OffscreenCanvas) {
    super();
    if (!env().getBool('HAS_WEBGL')) {
      throw new Error('WebGL is not supported on this device');
    }

    let newGPGPU;
    if (gpuResource != null) {
      if (gpuResource instanceof GPGPUContext) {
        newGPGPU = gpuResource;
      } else {
        const gl =
            getWebGLContext(env().getNumber('WEBGL_VERSION'), gpuResource);
        newGPGPU = new GPGPUContext(gl);
      }
      this.binaryCache = {};
      this.gpgpuCreatedLocally = false;
    } else {
      const gl = getWebGLContext(env().getNumber('WEBGL_VERSION'));
      newGPGPU = new GPGPUContext(gl);
      this.binaryCache = getBinaryCache(env().getNumber('WEBGL_VERSION'));
      this.gpgpuCreatedLocally = true;
    }

    this.gpgpu = newGPGPU;
    this.canvas = this.gpgpu.gl.canvas;
    this.textureManager = new TextureManager(this.gpgpu);
    this.numMBBeforeWarning = numMBBeforeWarning();
    this.texData = new DataStorage(this, engine());
  }

  override numDataIds() {
    return this.texData.numDataIds() - this.pendingDeletes;
  }

  // Writes a new entry to the data store with a WebGL texture, and registers it
  // to the texture manager.
  writeTexture(
      texture: WebGLTexture, shape: number[], dtype: DataType,
      texHeight: number, texWidth: number, channels: string): DataId {
    // Temporarily create an tensor info to make the texture compatible with
    // the runWebGLProgram's input.
    const input = this.makeTensorInfo(shape, dtype);
    const inData = this.texData.get(input.dataId);
    // Even though the input texture could be unpacked or dense packed, it is
    // always considered as unpacked for EncodeMatrixProgram.
    inData.isPacked = false;

    // Bind texture to the input tensor.
    inData.texture = {texture, texShape: [texHeight, texWidth]};
    inData.texShape = [texHeight, texWidth];

    const shapeAs3D = webgl_util.getShapeAs3D(shape);
    const program =
        new EncodeMatrixProgram(shapeAs3D, false /* isByteArray */, channels);
    const output =
        this.runWebGLProgram(program, [input], dtype, [[texHeight, texWidth]]);
    output.shape = shape;

    // Unbind the texture from the input tensor to avoid the texture being
    // released.
    inData.texture = null;
    this.disposeIntermediateTensorInfo(input);

    return output.dataId;
  }

  override write(values: BackendValues, shape: number[], dtype: DataType):
      DataId {
    if (env().getBool('WEBGL_CHECK_NUMERICAL_PROBLEMS') ||
        env().getBool('DEBUG')) {
      this.checkNumericalProblems(values);
    }
    if (dtype === 'complex64' && values != null) {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }
    const dataId = {id: this.nextDataId()};
    this.texData.set(
        dataId,
        {shape, dtype, values, usage: TextureUsage.UPLOAD, refCount: 1});
    return dataId;
  }

  /** Return refCount of a `TensorData`. */
  override refCount(dataId: DataId): number {
    if (this.texData.has(dataId)) {
      const tensorData = this.texData.get(dataId);
      return tensorData.refCount;
    }
    return 0;
  }

  /** Increase refCount of a `TextureData`. */
  override incRef(dataId: DataId): void {
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

  override move(
      dataId: DataId, values: BackendValues, shape: number[], dtype: DataType,
      refCount: number): void {
    if (env().getBool('DEBUG')) {
      this.checkNumericalProblems(values);
    }
    if (dtype === 'complex64') {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }
    this.texData.set(
        dataId, {shape, dtype, values, usage: TextureUsage.UPLOAD, refCount});
  }

  disposeIntermediateTensorInfo(tensorInfo: TensorInfo): void {
    this.disposeData(tensorInfo.dataId);
  }

  override readSync(dataId: DataId): BackendValues {
    const texData = this.texData.get(dataId);
    const {values, dtype, complexTensorInfos, slice, shape, isPacked} = texData;

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
      const realValues =
          this.readSync(complexTensorInfos.real.dataId) as Float32Array;
      const imagValues =
          this.readSync(complexTensorInfos.imag.dataId) as Float32Array;
      result = backend_util.mergeRealAndImagArrays(realValues, imagValues);
    } else {
      result = this.getValuesFromTexture(dataId);
    }

    if (shouldTimeProgram) {
      this.downloadWaitMs += util.now() - start;
    }
    return this.convertAndCacheOnCPU(dataId, result);
  }

  override async read(dataId: DataId): Promise<BackendValues> {
    if (this.pendingRead.has(dataId)) {
      const subscribers = this.pendingRead.get(dataId);
      return new Promise<TypedArray>(resolve => subscribers.push(resolve));
    }
    const texData = this.texData.get(dataId);
    const {values, shape, slice, dtype, complexTensorInfos, isPacked} = texData;

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

    if (env().getBool('DEBUG')) {
      // getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') caused a blocking GPU call.
      // For performance reason, only check it for debugging. In production,
      // it doesn't handle this use case anyway, so behavior is not changed.
      if (!env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
          env().getNumber('WEBGL_VERSION') === 2) {
        throw new Error(
            `tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and ` +
            `WEBGL_VERSION=2 not yet supported.`);
      }
    }

    let buffer: WebGLBuffer = null;
    let tmpDownloadTarget: TensorInfo;

    if (dtype !== 'complex64' && env().get('WEBGL_BUFFER_SUPPORTED')) {
      // Possibly copy the texture into a buffer before inserting a fence.
      tmpDownloadTarget = this.decode(dataId);
      const tmpData = this.texData.get(tmpDownloadTarget.dataId);

      buffer = this.gpgpu.createBufferFromTexture(
          tmpData.texture.texture, ...tex_util.getDenseTexShape(shape));
    }

    this.pendingRead.set(dataId, []);

    if (dtype !== 'complex64') {
      // Create a fence and wait for it to resolve.
      await this.gpgpu.createAndWaitForFence();
    }

    // Download the values from the GPU.
    let vals: Float32Array;
    if (dtype === 'complex64') {
      const ps = await Promise.all([
        this.read(complexTensorInfos.real.dataId),
        this.read(complexTensorInfos.imag.dataId)
      ]);

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
    if (buffer != null) {
      const gl = this.gpgpu.gl;
      webgl_util.callAndCheck(gl, () => gl.deleteBuffer(buffer));
    }
    const dTypeVals = this.convertAndCacheOnCPU(dataId, vals);

    const subscribers = this.pendingRead.get(dataId);
    this.pendingRead.delete(dataId);

    // Notify all pending reads.
    subscribers.forEach(resolve => resolve(dTypeVals));
    if (this.pendingDisposal.has(dataId)) {
      this.pendingDisposal.delete(dataId);
      if (this.disposeData(dataId)) {
        engine().removeDataId(dataId, this);
      }
      this.pendingDeletes--;
    }
    return dTypeVals;
  }

  /**
   * Read tensor to a new texture that is densely packed for ease of use.
   * @param dataId The source tensor.
   * @param options
   *     customTexShape: Optional. If set, will use the user defined texture
   *     shape to create the texture.
   */
  override readToGPU(dataId: DataId, options: DataToGPUWebGLOption = {}):
      GPUData {
    const texData = this.texData.get(dataId);
    const {values, shape, slice, dtype, isPacked, texture} = texData;

    if (dtype === 'complex64') {
      throw new Error('Does not support reading texture for complex64 dtype.');
    }

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
      const gpuResouorce = this.readToGPU(res, options);
      this.disposeIntermediateTensorInfo(res);
      return gpuResouorce;
    }

    if (texture == null) {
      if (values != null) {
        throw new Error('Data is not on GPU but on CPU.');
      } else {
        throw new Error('There is no data on GPU or CPU.');
      }
    }

    // Decode the texture so that it is stored densely (using four channels).
    const tmpTarget = this.decode(dataId, options.customTexShape);

    // Make engine track this tensor, so that we can dispose it later.
    const tensorRef = engine().makeTensorFromTensorInfo(tmpTarget);

    const tmpData = this.texData.get(tmpTarget.dataId);
    return {tensorRef, ...tmpData.texture};
  }

  bufferSync<R extends Rank, D extends DataType>(t: TensorInfo):
      TensorBuffer<R, D> {
    const data = this.readSync(t.dataId);
    if (t.dtype === 'string') {
      try {
        // Decode the bytes into string.
        const strings = (data as Uint8Array[]).map(d => util.decodeString(d));
        return buffer(t.shape as ShapeMap[R], t.dtype, strings) as
            TensorBuffer<R, D>;
      } catch {
        throw new Error('Failed to decode encoded string bytes into utf-8');
      }
    }
    return buffer(t.shape as ShapeMap[R], t.dtype, data as TypedArray) as
        TensorBuffer<R, D>;
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
      const vals =
          this.gpgpu
              .downloadMatrixFromPackedTexture(
                  tmpData.texture.texture, ...tex_util.getDenseTexShape(shape))
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
    const vals = this.gpgpu
                     .downloadByteEncodedFloatMatrixFromOutputTexture(
                         tmpData.texture.texture, tmpData.texShape[0],
                         tmpData.texShape[1])
                     .subarray(0, size);
    this.disposeIntermediateTensorInfo(output);

    return vals;
  }

  override timerAvailable(): boolean {
    return env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0;
  }

  override time(f: () => void): Promise<WebGLTimingInfo> {
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

    return (async () => {
      if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') >
          0) {
        const kernelMs = await Promise.all(flattenedActiveTimerQueries);

        res['kernelMs'] = util.sum(kernelMs);
        res['getExtraProfileInfo'] = () =>
            kernelMs
                .map((d, i) => ({name: flattenedActiveTimerNames[i], ms: d}))
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
    })();
  }
  override memory(): WebGLMemoryInfo {
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

  /**
   * Decrease the RefCount on the dataId and dispose the memory if the dataId
   * has 0 refCount. If there are pending read on the data, the disposal would
   * added to the pending delete queue. Return true if the dataId is removed
   * from backend or the backend does not contain the dataId, false if the
   * dataId is not removed. Memory may or may not be released even when dataId
   * is removed, which also depends on dataRefCount, see `releaseGPU`.
   * @param dataId
   * @oaram force Optional, remove the data regardless of refCount
   */
  override disposeData(dataId: DataId, force = false): boolean {
    if (this.pendingDisposal.has(dataId)) {
      return false;
    }

    // No-op if already disposed.
    if (!this.texData.has(dataId)) {
      return true;
    }

    // if force flag is set, change refCount to 0, this would ensure disposal
    // when added to the pendingDisposal queue. Memory may or may not be
    // released, which also depends on dataRefCount, see `releaseGPU`.
    if (force) {
      this.texData.get(dataId).refCount = 0;
    } else {
      this.texData.get(dataId).refCount--;
    }

    if (!force && this.texData.get(dataId).refCount > 0) {
      return false;
    }

    if (this.pendingRead.has(dataId)) {
      this.pendingDisposal.add(dataId);
      this.pendingDeletes++;
      return false;
    }

    this.releaseGPUData(dataId);
    const {complexTensorInfos} = this.texData.get(dataId);
    if (complexTensorInfos != null) {
      this.disposeData(complexTensorInfos.real.dataId, force);
      this.disposeData(complexTensorInfos.imag.dataId, force);
    }

    this.texData.delete(dataId);

    return true;
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
    return this.texData.get(dataId).texture.texture;
  }

  /**
   * Returns internal information for the specific data bucket. Used in unit
   * tests.
   */
  getDataInfo(dataId: DataId): TextureData {
    return this.texData.get(dataId);
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
    return env().getBool('WEBGL_CPU_FORWARD') &&
        inputs.every(
            input => this.texData.get(input.dataId).texture == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
  }

  getGPGPUContext(): GPGPUContext {
    return this.gpgpu;
  }

  where(condition: Tensor): Tensor2D {
    backend_util.warn(
        'tf.where() in webgl locks the UI thread. ' +
        'Call tf.whereAsync() instead');
    const condVals = condition.dataSync();
    return whereImpl(condition.shape, condVals);
  }

  private packedUnaryOp(x: TensorInfo, op: string, dtype: DataType) {
    const program = new UnaryOpPackedProgram(x.shape, op);
    const outInfo = this.compileAndRun(program, [x], dtype);
    return engine().makeTensorFromTensorInfo(outInfo);
  }

  // TODO(msoulanille) remove this once the backend has been modularized
  // a copy is needed here to break a circular dependency.
  // Also remove the op from unary_op.
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
    const outInfo = this.compileAndRun(program, [x]);
    return engine().makeTensorFromTensorInfo(outInfo) as T;
  }

  makeTensorInfo(
      shape: number[], dtype: DataType,
      values?: BackendValues|string[]): TensorInfo {
    let dataId;
    if (dtype === 'string' && values != null && values.length > 0 &&
        util.isString(values[0])) {
      const encodedValues =
          (values as unknown as string[]).map(d => util.encodeString(d));

      dataId = this.write(encodedValues, shape, dtype);
    } else {
      dataId = this.write(values as TypedArray, shape, dtype);
    }

    this.texData.get(dataId).usage = null;
    return {dataId, shape, dtype};
  }

  private makeOutput<T extends Tensor>(
      shape: number[], dtype: DataType, values?: BackendValues): T {
    return engine().makeTensorFromTensorInfo(
               this.makeTensorInfo(shape, dtype, values), this) as T;
  }

  unpackTensor(input: TensorInfo): TensorInfo {
    const program = new UnpackProgram(input.shape);
    return this.runWebGLProgram(program, [input], input.dtype);
  }

  packTensor(input: TensorInfo): TensorInfo {
    const program = new PackProgram(input.shape);
    const preventEagerUnpackingOutput = true;
    return this.runWebGLProgram(
        program, [input], input.dtype, null /* customUniformValues */,
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
    const customValues = [input3DShape];
    const output = this.runWebGLProgram(
        program, [input3D], input.dtype, customValues,
        preventEagerUnpackingOfOutput);
    return {dataId: output.dataId, shape: afterShape, dtype: output.dtype};
  }

  private decode(dataId: DataId, customTexShape?: [number, number]):
      TensorInfo {
    const texData = this.texData.get(dataId);
    const {isPacked, shape, dtype} = texData;
    if (customTexShape != null) {
      const size = util.sizeFromShape(shape);
      const texSize = customTexShape[0] * customTexShape[1] * 4;
      util.assert(
          size <= texSize,
          () => 'customTexShape is too small. ' +
              'Row * Column * 4 should be equal or larger than the ' +
              'size of the tensor data.');
    }
    const shapeAs3D =
        webgl_util.getShapeAs3D(shape) as [number, number, number];
    let program;
    if (isPacked) {
      program = new DecodeMatrixPackedProgram(shapeAs3D);
    } else {
      program = new DecodeMatrixProgram(shapeAs3D);
    }
    const preventEagerUnpackingOfOutput = true;
    const customValues =
        [customTexShape != null ? customTexShape :
                                  tex_util.getDenseTexShape(shapeAs3D)];
    const out = this.runWebGLProgram(
        program, [{shape: shapeAs3D, dtype, dataId}], dtype, customValues,
        preventEagerUnpackingOfOutput, customTexShape);
    return {dtype, shape, dataId: out.dataId};
  }

  runWebGLProgram(
      program: GPGPUProgram, inputs: TensorInfo[], outputDtype: DataType,
      customUniformValues?: number[][], preventEagerUnpackingOfOutput = false,
      customTexShape?: [number, number]): TensorInfo {
    const output = this.makeTensorInfo(program.outputShape, outputDtype);
    const outData = this.texData.get(output.dataId);
    if (program.packedOutput) {
      outData.isPacked = true;
    }
    if (program.outPackingScheme === tex_util.PackingScheme.DENSE) {
      const texelShape = customTexShape != null ?
          customTexShape :
          tex_util.getDenseTexShape(program.outputShape);
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
      }

      this.uploadToGPU(input.dataId);
      if (!!texData.isPacked !== !!program.packedInputs) {
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

    if (!env().get('ENGINE_COMPILE_ONLY')) {
      gpgpu_math.runProgram(
          this.gpgpu, binary, inputsData, outputData, customUniformValues);
    }

    dataToDispose.forEach(info => this.disposeIntermediateTensorInfo(info));

    if (shouldTimeProgram) {
      query = this.endTimer(query);
      this.activeTimers.push(
          {name: program.constructor.name, query: this.getQueryTime(query)});
    }

    const glFlushThreshold = env().get('WEBGL_FLUSH_THRESHOLD');
    // Manually GL flush requested
    if (glFlushThreshold > 0) {
      const time = util.now();
      if ((time - this.lastGlFlushTime) > glFlushThreshold) {
        this.gpgpu.gl.flush();
        this.lastGlFlushTime = time;
      }
    }

    if (!env().getBool('WEBGL_LAZILY_UNPACK') && outData.isPacked &&
        preventEagerUnpackingOfOutput === false) {
      const unpacked = this.unpackTensor(output);
      this.disposeIntermediateTensorInfo(output);
      return unpacked;
    }
    return output;
  }

  compileAndRun(
      program: GPGPUProgram, inputs: TensorInfo[], outputDtype?: DataType,
      customUniformValues?: number[][],
      preventEagerUnpackingOfOutput = false): TensorInfo {
    outputDtype = outputDtype || inputs[0].dtype;
    const outInfo = this.runWebGLProgram(
        program, inputs, outputDtype, customUniformValues,
        preventEagerUnpackingOfOutput);
    return outInfo;
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

  override dispose() {
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

  override floatPrecision(): 16|32 {
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
  override epsilon(): number {
    return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
  }

  uploadToGPU(dataId: DataId): void {
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
      // This texShape may not be the final texture shape. For packed or dense
      // textures, the texShape will be changed when textures are created.
      texShape = webgl_util.getTextureShapeFromLogicalShape(shape, isPacked);
      texData.texShape = texShape;
    }

    if (values != null) {
      const shapeAs3D = webgl_util.getShapeAs3D(shape);

      let program;
      let width = texShape[1], height = texShape[0];
      const isByteArray =
          values instanceof Uint8Array || values instanceof Uint8ClampedArray;

      // texture for float array is PhysicalTextureType.PACKED_2X2_FLOAT32, we
      // need to make sure the upload uses the same packed size
      if (isPacked || !isByteArray) {
        [width, height] = tex_util.getPackedMatrixTextureShapeWidthHeight(
            texShape[0], texShape[1]);
      }

      if (isPacked) {
        program = new EncodeMatrixPackedProgram(shapeAs3D, isByteArray);
      } else {
        program = new EncodeMatrixProgram(shapeAs3D, isByteArray);
      }

      // TexShape for float array needs to be the original shape, which byte
      // array needs to be packed size. This allow the data upload shape to be
      // matched with texture creation logic.
      const tempDenseInputTexShape: [number, number] =
          isByteArray ? [height, width] : texShape;
      const tempDenseInputHandle =
          this.makeTensorInfo(tempDenseInputTexShape, dtype);
      const tempDenseInputTexData =
          this.texData.get(tempDenseInputHandle.dataId);
      if (isByteArray) {
        tempDenseInputTexData.usage = TextureUsage.PIXELS;
      } else {
        tempDenseInputTexData.usage = TextureUsage.UPLOAD;
      }
      tempDenseInputTexData.texShape = tempDenseInputTexShape;
      this.gpgpu.uploadDenseMatrixToTexture(
          this.getTexture(tempDenseInputHandle.dataId), width, height,
          values as TypedArray);

      const customValues = [[height, width]];
      // We want the output to remain packed regardless of the value of
      // WEBGL_PACK.
      const preventEagerUnpacking = true;
      const encodedOutputTarget = this.runWebGLProgram(
          program, [tempDenseInputHandle], dtype, customValues,
          preventEagerUnpacking);

      // Have the original texture assume the identity of the encoded output.
      const outputTexData = this.texData.get(encodedOutputTarget.dataId);
      texData.texShape = outputTexData.texShape;
      texData.isPacked = outputTexData.isPacked;
      texData.usage = outputTexData.usage;

      if (!env().get('ENGINE_COMPILE_ONLY')) {
        texData.texture = outputTexData.texture;
        // Once uploaded, don't store the values on cpu.
        texData.values = null;
        this.texData.delete(encodedOutputTarget.dataId);
      } else {
        this.disposeData(encodedOutputTarget.dataId);
      }

      this.disposeIntermediateTensorInfo(tempDenseInputHandle);

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

    if (float32Values != null) {
      texData.values = float32ToTypedArray(float32Values, dtype as 'float32');
    }
    return texData.values as TypedArray;
  }

  private acquireTexture(
      texShape: [number, number], texType: TextureUsage, dtype: DataType,
      isPacked: boolean): Texture {
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

  checkCompileCompletion() {
    for (const [, binary] of Object.entries(this.binaryCache)) {
      this.checkCompletion_(binary);
    }
  }

  async checkCompileCompletionAsync(): Promise<boolean[]> {
    const ps = [];
    if (this.gpgpu.parallelCompilationExtension) {
      for (const [, binary] of Object.entries(this.binaryCache)) {
        ps.push(this.checkCompletionAsync_(binary));
      }
      return Promise.all(ps);
    } else {
      for (const [, binary] of Object.entries(this.binaryCache)) {
        const p: Promise<boolean> = new Promise((resolve) => {
          try {
            this.checkCompletion_(binary);
            resolve(true);
          } catch (error) {
            throw error;
          }
        });
        ps.push(p);
      }
      return Promise.all(ps);
    }
  }

  private async checkCompletionAsync_(binary: GPGPUBinary): Promise<boolean> {
    if (this.gpgpu.gl.getProgramParameter(
            binary.webGLProgram,
            this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)) {
      return this.checkCompletion_(binary);
    } else {
      await nextFrame();
      return this.checkCompletionAsync_(binary);
    }
  }

  private checkCompletion_(binary: GPGPUBinary): boolean {
    if (this.gpgpu.gl.getProgramParameter(
            binary.webGLProgram, this.gpgpu.gl.LINK_STATUS) === false) {
      console.log(this.gpgpu.gl.getProgramInfoLog(binary.webGLProgram));
      if (this.gpgpu.gl.getShaderParameter(
              binary.fragmentShader, this.gpgpu.gl.COMPILE_STATUS) === false) {
        webgl_util.logShaderSourceAndInfoLog(
            binary.source,
            this.gpgpu.gl.getShaderInfoLog(binary.fragmentShader));
        throw new Error('Failed to compile fragment shader.');
      }
      throw new Error('Failed to link vertex and fragment shaders.');
    }
    return true;
  }

  getUniformLocations() {
    for (const binary of Object.values(this.binaryCache)) {
      // TODO: Iterating through all binaries to build VAOs is supposed to be in
      // a seperate function, like 'setVaos'. However, to avoid breaking changes
      // for the users using parallel compile feature now, buildVao is silently
      // added here.
      this.gpgpu.buildVao(binary.webGLProgram);

      const {
        variablesLocations,
        customUniformLocations,
        infLoc,
        nanLoc,
        outShapeLocation,
        outShapeStridesLocation,
        outTexShapeLocation
      } = getUniformLocations(this.gpgpu, binary.program, binary.webGLProgram);
      binary.variablesLocations = variablesLocations;
      binary.customUniformLocations = customUniformLocations;
      binary.infLoc = infLoc;
      binary.nanLoc = nanLoc;
      binary.outShapeLocation = outShapeLocation;
      binary.outShapeStridesLocation = outShapeStridesLocation;
      binary.outTexShapeLocation = outTexShapeLocation;
    }
  }

  /**
   * Create a TF.js tensor out of an existing WebGL texture. A new texture will
   * be created.
   */
  override createTensorFromGPUData(
      values: WebGLData, shape: number[], dtype: DataType): Tensor {
    values.channels = values.channels || 'RGBA';
    const {texture, height, width, channels} = values;
    const backend = engine().backend as MathBackendWebGL;

    // Have to throw an error, otherwise WebGL just warns and returns wrong
    // values.
    if (!backend.gpgpu.gl.isTexture(texture)) {
      throw new Error(
          `The texture is invalid. Also, please make sure the texture and ` +
          `the TFJS WebGL backend are using the same canvas. If you want to ` +
          `use your own custom canvas, you have to create and use the custom ` +
          `TFJS WebGL backend created from the canvas through ` +
          `'new tf.MathBackendWebGL(customCanvas)'.`);
    }

    const dataId =
        backend.writeTexture(texture, shape, dtype, height, width, channels);
    return engine().makeTensorFromDataId(dataId, shape, dtype, backend);
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
