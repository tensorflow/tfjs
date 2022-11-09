/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import './flags_webgpu';

import {backend_util, buffer, DataStorage, DataType, engine, env, GPUData, KernelBackend, Rank, RecursiveArray, ShapeMap, TensorBuffer, TensorInfo, TimingInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {AdapterInfo} from './adapter_info';
import {BufferManager} from './buffer_manager';
import {TextureManager} from './texture_manager';
import * as webgpu_program from './webgpu_program';
import * as webgpu_util from './webgpu_util';

export interface WebGPUMemoryInfo extends backend_util.MemoryInfo {
  numBytesInGPU: number;
  numBytesAllocatedInGPU: number;
  unreliable: boolean;
}

export type BufferInfo = {
  size: number,
  usage: GPUBufferUsageFlags,
  buffer: GPUBuffer
};

export type TextureInfo = {
  width: number,
  height: number,
  format: GPUTextureFormat,
  usage: GPUTextureUsageFlags,
  texture: GPUTexture|GPUExternalTexture
};

type TensorData = {
  values: backend_util.BackendValues,
  dtype: DataType,
  shape: number[],
  refCount: number,
  resourceInfo?: BufferInfo|TextureInfo,
  // For complex numbers, the real and imaginary parts are stored as their own
  // individual tensors, with a parent joining the two with the
  // complexTensorInfos field.
  complexTensorInfos?: {real: TensorInfo, imag: TensorInfo}
};

interface DataId {}

export type WebGPUKernelInfo = {
  name: string; query: Promise<number>;
};

export type TimerNode = RecursiveArray<WebGPUKernelInfo>|WebGPUKernelInfo;

export interface WebGPUTimingInfo extends TimingInfo {
  uploadWaitMs: number;
  downloadWaitMs: number;
}

type ProgramUniform = Array<{type: string; data: number[]}>;

// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD =
    env().getNumber('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD');

// Reshape dispatch, not to exceed device limits.
const reshapeDispatch =
    (device: GPUDevice,
     program: webgpu_program.WebGPUProgram): [number, number, number] => {
      const MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE =
          device.limits.maxComputeWorkgroupsPerDimension;
      const layout = program['dispatchLayout'];
      const dispatch = program['dispatch'];
      if (dispatch.every((d) => d <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE)) {
        return dispatch;
      }

      util.assert(
          dispatch[0] > MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE &&
              layout.y === undefined && layout.z === undefined,
          () => 'Dispatch size exceeds WebGPU limits in Y or Z dimension.');

      let dispatchAverage = Math.ceil(Math.sqrt(dispatch[0]));
      if (dispatchAverage > MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE) {
        dispatchAverage = Math.ceil(Math.cbrt(dispatch[0]));
        util.assert(
            dispatchAverage <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE,
            () => 'Total dispatch size exceeds WebGPU maximum.');
        return [dispatchAverage, dispatchAverage, dispatchAverage];
      } else {
        return [dispatchAverage, dispatchAverage, 1];
      }
    };

export class WebGPUBackend extends KernelBackend {
  bufferManager: BufferManager;
  adapterInfo: AdapterInfo;
  device: GPUDevice;
  queue: GPUQueue;
  tensorMap: DataStorage<TensorData>;
  textureManager: TextureManager;
  thresholdToIncreaseWorkgroups: number;

  private activeTimers: TimerNode[];
  private currentCommandEncoder: GPUCommandEncoder;
  private currentComputePass: GPUComputePassEncoder;
  private commandQueueOwnedIds = new WeakSet<DataId>();
  private dispatchNumberInEncoder = 0;
  private disposed = false;
  private downloadWaitMs = 0;
  private dummyCanvas: HTMLCanvasElement;
  private dummyContext: GPUCanvasContext;
  private tensorDataPendingDisposal: DataId[] = [];
  private static nextDataId = 0;
  private pipelineCache: {[key: string]: GPUComputePipeline};
  private programTimersStack: TimerNode[];
  private querySet: GPUQuerySet;
  private stagingPendingDisposal: BufferInfo[] = [];
  private supportTimeQuery: boolean;
  private uniformPendingDisposal: BufferInfo[] = [];
  private uploadWaitMs = 0;

  private nextDataId(): number {
    return WebGPUBackend.nextDataId++;
  }

  constructor(device: GPUDevice, adapterInfo?: GPUAdapterInfo) {
    super();
    if (!webgpu_util.isWebGPUSupported()) {
      throw new Error('WebGPU is not supported on this device');
    }
    this.pipelineCache = {};
    this.device = device;
    this.queue = device.queue;
    this.currentCommandEncoder = null;
    this.currentComputePass = null;
    this.supportTimeQuery =
        device.features.has('timestamp-query-inside-passes');
    this.adapterInfo = new AdapterInfo(adapterInfo);
    this.thresholdToIncreaseWorkgroups =
        this.adapterInfo.intelGPUGeneration >= 12 ? 16 : 8;

    this.bufferManager = new BufferManager(this.device);
    this.textureManager = new TextureManager(this.device);
    this.tensorMap = new DataStorage(this, engine());
    if (this.supportTimeQuery) {
      this.querySet = this.device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
    }

    // Profiling tools like PIX needs this dummy canvas to
    // trigger capturing a frame.
    if (env().getBool('WEBGPU_USE_PROFILE_TOOL')) {
      this.dummyCanvas = document.createElement('canvas');
      this.dummyCanvas.width = 1;
      this.dummyCanvas.height = 1;

      this.dummyContext = this.dummyCanvas.getContext('webgpu');
      this.dummyContext.configure({
        device,
        format: 'bgra8unorm',
      });

      document.body.appendChild(this.dummyCanvas);
    }
  }

  override floatPrecision(): 32 {
    return 32;
  }

  defaultGpuBufferUsage(): number {
    return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST;
  }

  /**
   * Dispose the memory if the dataId has 0 refCount. Return true if the memory
   * is released or memory is not managed in this backend, false if memory is
   * not cleared.
   * @param dataId
   * @oaram force Optional, remove the data regardless of refCount
   */
  override disposeData(dataId: DataId, force = false): boolean {
    if (this.tensorDataPendingDisposal.indexOf(dataId) >= 0) {
      return false;
    }
    if (!this.tensorMap.has(dataId)) {
      return true;
    }

    const tensorData = this.tensorMap.get(dataId);
    this.decRef(dataId);
    if (!force && tensorData.refCount > 0) {
      return false;
    }

    // complex is never in commandQueueOwnedIds
    if (this.commandQueueOwnedIds.has(dataId)) {
      this.tensorDataPendingDisposal.push(dataId);
      return false;
    }

    const {complexTensorInfos} = this.tensorMap.get(dataId);
    if (complexTensorInfos != null) {
      this.disposeData(complexTensorInfos.real.dataId, force);
      this.disposeData(complexTensorInfos.imag.dataId, force);
    }

    this.releaseResource(dataId);
    this.tensorMap.delete(dataId);

    return true;
  }

  override memory(): WebGPUMemoryInfo {
    return {
      numBytesInGPU: this.bufferManager.numBytesUsed,
      numBytesAllocatedInGPU: this.bufferManager.numBytesAllocated,
      unreliable: false
    } as WebGPUMemoryInfo;
  }

  releaseResource(dataId: DataId) {
    const tensorData = this.tensorMap.get(dataId);
    if (!tensorData || !tensorData.resourceInfo) {
      return;
    }
    if ('texture' in tensorData.resourceInfo) {
      const textureInfo = tensorData.resourceInfo;
      if (textureInfo.texture instanceof GPUTexture) {
        this.textureManager.releaseTexture(
            textureInfo.texture, textureInfo.width, textureInfo.height,
            textureInfo.format, textureInfo.usage);
      }
      textureInfo.texture = null;
    } else {
      const bufferInfo = tensorData.resourceInfo;
      this.bufferManager.releaseBuffer(
          bufferInfo.buffer, bufferInfo.size, bufferInfo.usage);
      bufferInfo.buffer = null;
    }
    tensorData.resourceInfo = null;
  }

  /** Return refCount of a `TensorData`. */
  override refCount(dataId: DataId): number {
    if (this.tensorMap.has(dataId)) {
      const tensorData = this.tensorMap.get(dataId);
      return tensorData.refCount;
    }
    return 0;
  }

  /** Increase refCount of a `TensorData`. */
  override incRef(dataId: DataId): void {
    const tensorData = this.tensorMap.get(dataId);
    tensorData.refCount++;
  }

  /** Decrease refCount of a `TensorData`. */
  decRef(dataId: DataId): void {
    if (this.tensorMap.has(dataId)) {
      const tensorData = this.tensorMap.get(dataId);
      tensorData.refCount--;
    }
  }

  override write(values: backend_util.BackendValues, shape: number[],
      dtype: DataType): DataId {
    if (dtype === 'complex64' && values != null) {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }
    const dataId = {id: this.nextDataId()};
    this.tensorMap.set(dataId, {dtype, shape, values, refCount: 1});
    return dataId;
  }

  override move(
      dataId: DataId, values: backend_util.BackendValues, shape: number[],
      dtype: DataType, refCount: number): void {
    if (dtype === 'complex64') {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }
    this.tensorMap.set(dataId, {dtype, shape, values, refCount});
  }

  submitQueue() {
    this.ensureComputePassEnded();
    this.queue.submit([this.currentCommandEncoder.finish()]);
    this.currentCommandEncoder = null;
    this.dispatchNumberInEncoder = 0;

    this.commandQueueOwnedIds = new WeakSet<DataId>();

    this.tensorDataPendingDisposal.forEach(d => {
      this.releaseResource(d);
      this.tensorMap.delete(d);
    });
    this.uniformPendingDisposal.forEach(
        d => this.bufferManager.releaseBuffer(d.buffer, d.size, d.usage));
    this.stagingPendingDisposal.forEach(
        d => this.bufferManager.releaseUploadBuffer(d.buffer, d.size, d.usage));

    this.tensorDataPendingDisposal = [];
    this.uniformPendingDisposal = [];
    this.stagingPendingDisposal = [];
  }

  ensureCommandEncoderReady() {
    if (!this.currentCommandEncoder) {
      this.currentCommandEncoder = this.device.createCommandEncoder();
    }
  }

  ensureComputePassEnded() {
    if (this.currentComputePass) {
      this.currentComputePass.end();
      this.currentComputePass = null;
    }
  }

  getComputePass() {
    if (!this.currentComputePass) {
      this.currentComputePass = this.currentCommandEncoder.beginComputePass();
    }
    return this.currentComputePass;
  }

  public async getBufferData(buffer: GPUBuffer, size: number):
      Promise<backend_util.BackendValues> {
    const staging = this.bufferManager.acquireBuffer(
        size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.ensureCommandEncoderReady();
    this.ensureComputePassEnded();
    this.currentCommandEncoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.submitQueue();

    await staging.mapAsync(GPUMapMode.READ);
    const values = staging.getMappedRange().slice(0);

    staging.unmap();
    if (staging != null) {
      this.bufferManager.releaseBuffer(
          staging, size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    }

    // Need to get texture from swapChain to enable profiling tool
    // to capture a frame
    if (env().getBool('WEBGPU_USE_PROFILE_TOOL')) {
      util.assert(
          this.dummyContext !== undefined,
          () => `Fail to get context for profiling tool`);
      this.dummyContext.getCurrentTexture();
    }

    return values as backend_util.BackendValues;
  }

  private convertAndCacheOnCPU(dataId: DataId, data: backend_util.TypedArray):
      backend_util.TypedArray {
    const tensorData = this.tensorMap.get(dataId);
    this.releaseResource(dataId);
    tensorData.values = data;
    return tensorData.values;
  }

  // TODO: Remove once this is fixed:
  // https://github.com/tensorflow/tfjs/issues/1595
  override readSync(dataId: object): backend_util.BackendValues {
    const tensorData = this.tensorMap.get(dataId);
    const {values} = tensorData;

    if (values == null) {
      throw new Error(
          'WebGPU readSync is only available for CPU-resident tensors.');
    }

    return values;
  }

  override async read(dataId: object): Promise<backend_util.BackendValues> {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const tensorData = this.tensorMap.get(dataId);

    const {values} = tensorData;

    if (values != null) {
      // TODO(xing.xu@intel.com): Merge backend_util.BackendValues and
      // backend_util.TypedArray.
      return this.convertAndCacheOnCPU(
                 dataId, values as backend_util.TypedArray) as
          backend_util.BackendValues;
    }

    // Download the values from the GPU.
    let vals: backend_util.BackendValues;
    if (tensorData.dtype === 'complex64') {
      const ps = await Promise.all([
        this.read(tensorData.complexTensorInfos.real.dataId),
        this.read(tensorData.complexTensorInfos.imag.dataId)
      ]);

      const realValues = ps[0];
      const imagValues = ps[1];
      vals = backend_util.mergeRealAndImagArrays(
          realValues as Float32Array, imagValues as Float32Array);
    } else {
      const bufferInfo = tensorData.resourceInfo as BufferInfo;
      const data = await this.getBufferData(bufferInfo.buffer, bufferInfo.size);
      vals = webgpu_util.ArrayBufferToTypedArray(
          data as ArrayBuffer, tensorData.dtype);
    }
    this.convertAndCacheOnCPU(dataId, vals);
    return vals;
  }

  /**
   * Read tensor to a new GPUBuffer.
   * @param dataId The source tensor.
   */
  override readToGPU(dataId: DataId): GPUData {
    const srcTensorData = this.tensorMap.get(dataId);
    const {values, dtype, shape, resourceInfo} = srcTensorData;

    if (dtype === 'complex64') {
      throw new Error('Does not support reading buffer for complex64 dtype.');
    }

    if (resourceInfo == null) {
      if (values != null) {
        throw new Error('Data is not on GPU but on CPU.');
      } else {
        throw new Error('There is no data on GPU or CPU.');
      }
    }

    const size = (resourceInfo as BufferInfo).size;
    const buffer = this.bufferManager.acquireBuffer(size, resourceInfo.usage);
    this.ensureCommandEncoderReady();
    this.ensureComputePassEnded();
    this.currentCommandEncoder.copyBufferToBuffer(
        (resourceInfo as BufferInfo).buffer, 0, buffer, 0, size);
    this.submitQueue();

    const tensorInfo = this.makeTensorInfo(shape, dtype);
    // Make engine track this tensor, so that we can dispose it later.
    const tensorRef = engine().makeTensorFromTensorInfo(tensorInfo);

    const tensorData = this.tensorMap.get(tensorInfo.dataId);
    tensorData
        .resourceInfo = {size, usage: this.defaultGpuBufferUsage(), buffer};

    return {tensorRef, buffer, bufSize: size};
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

  override async time(f: () => void): Promise<WebGPUTimingInfo> {
    if (!this.supportTimeQuery) {
      console.warn(
          `This device doesn't support timestamp-query-inside-passes extension. ` +
          `Start Chrome browser with flag ` +
          `--disable-dawn-features=disallow_unsafe_apis then try again. ` +
          `Otherwise, zero will be shown for the kernel time when profiling ` +
          `mode is enabled. Using performance.now is not workable for webgpu ` +
          `since it doesn't support synchronous data read from GPU.`);
    }
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

    const flattenedActiveTimerQueries =
        util.flatten(this.activeTimers.map((d: WebGPUKernelInfo) => d.query))
            .filter(d => d != null);
    const flattenedActiveTimerNames =
        util.flatten(this.activeTimers.map((d: WebGPUKernelInfo) => d.name))
            .filter(d => d != null);

    this.activeTimers = oldActiveTimers;

    if (outerMostTime) {
      this.programTimersStack = null;
    }
    const res: WebGPUTimingInfo = {
      uploadWaitMs: this.uploadWaitMs,
      downloadWaitMs: this.downloadWaitMs,
      kernelMs: null,
      wallMs: null
    };

    const kernelMs = await Promise.all(flattenedActiveTimerQueries);
    res['kernelMs'] = util.sum(kernelMs);
    res['getExtraProfileInfo'] = () =>
        kernelMs.map((d, i) => ({name: flattenedActiveTimerNames[i], ms: d}))
            .map(d => `${d.name}: ${d.ms}`)
            .join(', ');
    this.uploadWaitMs = 0;
    this.downloadWaitMs = 0;
    return res;
  }

  makeTensorInfo(
      shape: number[], dtype: DataType,
      values?: backend_util.BackendValues|string[]): TensorInfo {
    if (dtype === 'string' && values != null && values.length > 0 &&
        util.isString(values[0])) {
      values = (values as {} as string[]).map(d => util.encodeString(d));
    }
    const dataId =
        this.write(values as backend_util.BackendValues, shape, dtype);
    return {dataId, shape, dtype};
  }

  private tensorToBinding(tensor?: TensorInfo): GPUBindingResource {
    if (!tensor) {
      return null;
    }

    const tensorData = this.tensorMap.get(tensor.dataId);
    if ('texture' in tensorData.resourceInfo) {
      const info = tensorData.resourceInfo;
      if (info.texture instanceof GPUExternalTexture) {
        return info.texture;
      } else {
        return info.texture.createView();
      }
    }
    const bufferInfo = tensorData.resourceInfo;
    return {offset: 0, size: bufferInfo.size, buffer: bufferInfo.buffer};
  }

  async getQueryTime(query: GPUQuerySet): Promise<number> {
    if (this.supportTimeQuery) {
      return this.getTimeFromQuerySet(query);
    } else {
      return 0;
    }
  }

  uploadToGPU(dataId: DataId): void {
    const tensorData = this.tensorMap.get(dataId);
    // Already on the GPU.
    if (tensorData.resourceInfo) {
      return;
    }

    const size = webgpu_util.GPUBytesPerElement(tensorData.dtype) *
        util.sizeFromShape(tensorData.shape);
    const buffer =
        this.bufferManager.acquireBuffer(size, this.defaultGpuBufferUsage());

    tensorData
        .resourceInfo = {size, usage: this.defaultGpuBufferUsage(), buffer};
    if (tensorData.values) {
      const stagingBuffer = this.bufferManager.acquireUploadBuffer(
          size, GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC);
      const arrayBuffer = stagingBuffer.getMappedRange();
      if (tensorData.dtype === 'int32' || tensorData.dtype === 'bool') {
        new Int32Array(arrayBuffer).set(tensorData.values as TypedArray);
      } else {
        new Float32Array(arrayBuffer).set(tensorData.values as Float32Array);
      }
      stagingBuffer.unmap();
      this.ensureCommandEncoderReady();
      this.ensureComputePassEnded();
      this.currentCommandEncoder.copyBufferToBuffer(
          stagingBuffer, 0, buffer, 0, size);

      const stagingInfo = {
        size,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        buffer: stagingBuffer
      };
      this.stagingPendingDisposal.push(stagingInfo);
      // TODO: WebGPU doesn't support read data synchronously from GPU to CPU.
      // So it will report error when switching backend from WebGPU to others.
      // There are two situations: 1) swithcing the backend after running a
      // model; 2) swithcing the backend within the model. Temporarilly keep the
      // values on CPU to solve the first issue.
      // tensorData.values = null;
    }
  }

  private makeUniforms(programUniform: ProgramUniform): GPUBindingResource {
    let currentOffset = 0;
    let preLength = 0;
    const offsets: number[] = [];
    programUniform.forEach((d) => {
      if (d.data.length === 0) {
        d.data = [1];
      }
      // https://www.w3.org/TR/WGSL/#alignof
      let baseAlignment: number;
      switch (d.data.length) {
        case 1:
          baseAlignment = 4;
          break;
        case 2:
          baseAlignment = 8;
          break;
        case 3:
          baseAlignment = 16;
          break;
        case 4:
          baseAlignment = 16;
          break;
        case 5:
          baseAlignment = 16;
          break;
        case 6:
          baseAlignment = 16;
          break;
        default:
          util.assert(false, () => `Unsupported ${d.data.length}D shape`);
      }

      if (preLength === 5 || preLength === 6) {
        baseAlignment = 16;
      }
      currentOffset = Math.ceil(currentOffset / baseAlignment) * baseAlignment;
      preLength = d.data.length;
      offsets.push(currentOffset);
      currentOffset += d.data.length * 4;
    });

    const arrayBuffer = new ArrayBuffer(currentOffset);
    programUniform.forEach((d, i) => {
      const offset = offsets[i];
      if (d.type === 'int32') {
        new Int32Array(arrayBuffer, offset, d.data.length).set(d.data);
      } else if (d.type === 'uint32') {
        new Uint32Array(arrayBuffer, offset, d.data.length).set(d.data);
      } else {
        new Float32Array(arrayBuffer, offset, d.data.length).set(d.data);
      }
    });

    const uniformBuffer = this.bufferManager.acquireBuffer(
        currentOffset, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
    this.queue.writeBuffer(uniformBuffer, 0, arrayBuffer, 0, currentOffset);

    const uniformInfo = {
      size: currentOffset,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
      buffer: uniformBuffer
    };
    this.uniformPendingDisposal.push(uniformInfo);

    return {offset: 0, size: currentOffset, buffer: uniformBuffer};
  }

  public runWebGPUProgram(
      program: webgpu_program.WebGPUProgram, inputs: TensorInfo[],
      outputDtype: DataType, programDefinedUniform?: ProgramUniform,
      output?: TensorInfo): TensorInfo {
    if (!output) {
      output = this.makeTensorInfo(program.outputShape, outputDtype);
    }
    if (util.sizeFromShape(output.shape) === 0) {
      // Short-circuit the computation since the result is empty (has 0 in its
      // shape).
      this.tensorMap.get(output.dataId).values =
          util.getTypedArrayFromDType(output.dtype as 'float32', 0);
      return output;
    }
    this.uploadToGPU(output.dataId);
    program.dispatch = reshapeDispatch(this.device, program);

    // There are six kinds of uniforms: NAN, INFINITY, shapes, shape strides,
    // program size, program defined uniforms.
    let programUniform: ProgramUniform = [];
    let bufferShapes: number[][] = [];
    if (!program.isFromPixels) {
      programUniform.push(
          {type: 'float32', data: [NaN]}, {type: 'float32', data: [Infinity]});
      bufferShapes = inputs.concat(output).map(d => d.shape);
      const uniformsType = 'int32';
      bufferShapes.map(d => {
        programUniform.push({type: uniformsType, data: d});
      });
      const strides = util.computeStrides(output.shape);
      programUniform.push({type: uniformsType, data: strides});
      if (program.size) {
        const size = util.sizeFromShape(program.outputShape);
        programUniform.push(
            {type: uniformsType, data: [program.isVec4 ? size / 4 : size]});
      }
    }

    const inputsData = inputs.map((input: TensorInfo, i: number) => {
      if (input.dtype === 'complex64') {
        throw new Error(
            `GPGPUProgram does not support complex64 input. For complex64 ` +
            `dtypes, please separate the program into real and imaginary ` +
            `parts.`);
      }
      this.uploadToGPU(input.dataId);

      return {
        // Returning dtype from tensorMap because it reflects dtype
        // of underlying buffer, rather than abstract dtype.
        dtype: this.tensorMap.get(input.dataId).dtype,
        shape: input.shape,
        name: program.variableNames[i]
      };
    });

    const key =
        webgpu_program.makeShaderKey(program, bufferShapes, inputsData, output);

    let pipeline;
    if (key in this.pipelineCache) {
      pipeline = this.pipelineCache[key];
    } else {
      pipeline = webgpu_program.compileProgram(
          this.device, program, inputsData, output);
      this.pipelineCache[key] = pipeline;
    }

    if (programDefinedUniform) {
      programUniform = [...programUniform, ...programDefinedUniform];
    }
    const bindings = [
      this.tensorToBinding(output), ...inputs.map(t => this.tensorToBinding(t)),
      this.makeUniforms(programUniform)
    ];

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((b, i) => ({binding: i, resource: b})),
    });

    this.ensureCommandEncoderReady();
    const pass = this.getComputePass();
    const shouldTimeProgram = this.activeTimers != null;
    if (shouldTimeProgram) {
      if (this.supportTimeQuery) {
        // tslint:disable-next-line:no-any
        (pass as any).writeTimestamp(this.querySet, 0);
      }
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
        program.dispatch[0], program.dispatch[1], program.dispatch[2]);
    if (shouldTimeProgram) {
      if (this.supportTimeQuery) {
        // tslint:disable-next-line:no-any
        (pass as any).writeTimestamp(this.querySet, 1);
      }
    }
    this.dispatchNumberInEncoder++;

    inputs.forEach(input => {
      this.commandQueueOwnedIds.add(input.dataId);
    });
    this.commandQueueOwnedIds.add(output.dataId);

    if (env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE') as
        number <= this.dispatchNumberInEncoder) {
      this.submitQueue();
    }

    if (shouldTimeProgram) {
      this.activeTimers.push({
        name: program.constructor.name,
        query: this.getQueryTime(this.querySet)
      });
    }
    return output;
  }

  async getTimeFromQuerySet(querySet: GPUQuerySet) {
    const queryBuffer = this.bufferManager.acquireBuffer(
        16, GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE);
    const dst = this.bufferManager.acquireBuffer(
        16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    this.ensureCommandEncoderReady();
    this.ensureComputePassEnded();
    this.currentCommandEncoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);
    this.currentCommandEncoder.copyBufferToBuffer(queryBuffer, 0, dst, 0, 16);
    this.submitQueue();
    await dst.mapAsync(GPUMapMode.READ);
    const arrayBuf = new BigUint64Array(dst.getMappedRange());
    const timeElapsedNanos = Number((arrayBuf[1] - arrayBuf[0]));
    dst.unmap();
    this.bufferManager.releaseBuffer(
        dst, 16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.bufferManager.releaseBuffer(
        queryBuffer, 16,
        GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE);
    // Return milliseconds.
    return timeElapsedNanos / 1000000;
  }

  shouldExecuteOnCPU(
      inputs: TensorInfo[],
      sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD): boolean {
    return env().getBool('WEBGPU_CPU_FORWARD') &&
        inputs.every(
            input => this.tensorMap.get(input.dataId).resourceInfo == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
  }

  override numDataIds() {
    return this.tensorMap.numDataIds() - this.tensorDataPendingDisposal.length;
  }

  override dispose() {
    if (this.disposed) {
      return;
    }
    this.bufferManager.dispose();
    this.textureManager.dispose();
    this.disposed = true;
  }
}
