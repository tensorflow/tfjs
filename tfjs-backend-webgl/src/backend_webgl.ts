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
import {backend_util, BackendValues, buffer, DataId, DataStorage, DataType, DataValues, engine, env, kernel_impls, KernelBackend, MemoryInfo, NumericDataType, Rank, RecursiveArray, scalar, ShapeMap, Tensor, Tensor2D, TensorBuffer, TensorInfo, tidy, TimingInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {getWebGLContext} from './canvas_util';
import {DecodeMatrixProgram} from './decode_matrix_gpu';
import {DecodeMatrixPackedProgram} from './decode_matrix_packed_gpu';
import {EncodeFloatProgram} from './encode_float_gpu';
import {EncodeFloatPackedProgram} from './encode_float_packed_gpu';
import {EncodeMatrixProgram} from './encode_matrix_gpu';
import {EncodeMatrixPackedProgram} from './encode_matrix_packed_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {GPGPUBinary, GPGPUProgram, TensorData} from './gpgpu_math';
import {simpleAbsImplCPU} from './kernel_utils/shared';
import {PackProgram} from './pack_gpu';
import {ReshapePackedProgram} from './reshape_packed_gpu';
import * as tex_util from './tex_util';
import {TextureData, TextureUsage} from './tex_util';
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
  dataRefCount = new WeakMap<DataId, number>();
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
    this.texData.set(dataId, {
      shape,
      dtype,
      values,
      usage: TextureUsage.UPLOAD,
      refCount: 1,
      complexParentRefCount: 0
    });
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
    this.texData.set(dataId, {
      shape,
      dtype,
      values,
      usage: TextureUsage.UPLOAD,
      refCount: 1,
      complexParentRefCount: 0
    });
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

  async read(dataId: DataId): Promise<BackendValues> {
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

  bufferSync<R extends Rank>(t: TensorInfo): TensorBuffer<R> {
    const data = this.readSync(t.dataId);
    let decodedData = data as DataValues;
    if (t.dtype === 'string') {
      try {
        // Decode the bytes into string.
        decodedData = (data as Uint8Array[]).map(d => util.decodeString(d));
      } catch {
        throw new Error('Failed to decode encoded string bytes into utf-8');
      }
    }
    return buffer(t.shape as ShapeMap[R], t.dtype, decodedData) as
        TensorBuffer<R>;
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

    // Trying to dispose a textureData that has a 'kept' refCount, e.g. trying
    // to dispose a tensor whose data bucket is shared with a complex tensor. In
    // this case we are removing a reference to the textureData, but we
    // shouldn't actually dispose the texture.
    if (this.texData.get(dataId).complexParentRefCount > 0) {
      this.texData.get(dataId).refCount--;
      return;
    }

    this.releaseGPUData(dataId);
    const {complexTensorInfos} = this.texData.get(dataId);
    if (complexTensorInfos != null) {
      this.texData.get(complexTensorInfos.real.dataId).complexParentRefCount--;
      this.disposeIntermediateTensorInfo(complexTensorInfos.real);

      this.texData.get(complexTensorInfos.imag.dataId).complexParentRefCount--;
      this.disposeIntermediateTensorInfo(complexTensorInfos.imag);
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
    if (!env().getBool('IS_TEST') && !this.warnedAboutCPUBackend &&
        cpuBackend == null) {
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

  where(condition: Tensor): Tensor2D {
    backend_util.warn(
        'tf.where() in webgl locks the UI thread. ' +
        'Call tf.whereAsync() instead');
    const condVals = condition.dataSync();
    return whereImpl(condition.shape, condVals);
  }

  private packedUnaryOp(x: TensorInfo, op: string, dtype: DataType) {
    const program = new UnaryOpPackedProgram(x.shape, op);
    return this.compileAndRun<Tensor>(program, [x], dtype);
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
    return this.compileAndRun(program, [x]);
  }

  makeTensorInfo(
      shape: number[], dtype: DataType,
      values?: BackendValues|string[]): TensorInfo {
    let dataId;
    if (dtype === 'string' && values != null && values.length > 0 &&
        util.isString(values[0])) {
      const encodedValues =
          (values as {} as string[]).map(d => util.encodeString(d));

      dataId = this.write(encodedValues, shape, dtype);
    } else {
      dataId = this.write(values as TypedArray, shape, dtype);
    }

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
