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

import {backend_util, buffer, DataStorage, DataType, DataValues, engine, env, KernelBackend, Rank, RecursiveArray, ShapeMap, TensorBuffer, TensorInfo, TimingInfo, util} from '@tensorflow/tfjs-core';

import {BufferManager} from './buffer_manager';
import {FromPixelsImportProgram} from './kernels/FromPixels_utils/from_pixels_import_webgpu';
import {FromPixelsProgram} from './kernels/FromPixels_utils/from_pixels_webgpu';
import * as webgpu_program from './kernels/webgpu_program';
import * as webgpu_util from './webgpu_util';
import {WebGPULayout} from './webgpu_util';

export interface WebGPUMemoryInfo extends backend_util.MemoryInfo {
  numBytesInGPU: number;
  numBytesAllocatedInGPU: number;
  unreliable: boolean;
}

type BufferInfo = {
  byteSize: number,
  usage: GPUBufferUsageFlags,
  buffer?: GPUBuffer
};

type TensorBufferInfo = {
  values: backend_util.BackendValues,
  dtype: DataType,
  bufferInfo: BufferInfo,
  refCount: number,
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

// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD =
    env().getNumber('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD');

export class WebGPUBackend extends KernelBackend {
  device: GPUDevice;
  queue: GPUQueue;
  currentCommandEncoder: GPUCommandEncoder;
  currentComputePass: GPUComputePassEncoder;
  tensorMap: DataStorage<TensorBufferInfo>;
  supportTimeQuery: boolean;
  dummyCanvas: HTMLCanvasElement;
  dummyContext: GPUCanvasContext;

  private static nextDataId = 0;
  private nextDataId(): number {
    return WebGPUBackend.nextDataId++;
  }
  private commandQueueOwnedIds = new WeakSet<DataId>();
  private layoutCache: {[key: number]: WebGPULayout};
  private pipelineCache: {[key: string]: GPUComputePipeline};
  private bufferManager: BufferManager;

  private tensorDisposalQueue: DataId[] = [];
  private uniformDisposalQueue: BufferInfo[] = [];
  private stagingDisposalQueue: BufferInfo[] = [];

  private disposed = false;

  private programTimersStack: TimerNode[];
  private activeTimers: TimerNode[];
  private uploadWaitMs = 0;
  private downloadWaitMs = 0;
  private dispatchNumberInEncoder = 0;
  private querySet: GPUQuerySet;
  private fromPixelProgram?: FromPixelsProgram;
  private fromPixelImportProgram?: FromPixelsImportProgram;

  constructor(device: GPUDevice, supportTimeQuery = false) {
    super();
    if (!webgpu_util.isWebGPUSupported()) {
      throw new Error('WebGPU is not supported on this device');
    }
    this.layoutCache = {};
    this.pipelineCache = {};
    this.device = device;
    this.queue = device.queue;
    this.currentCommandEncoder = null;
    this.currentComputePass = null;
    this.supportTimeQuery = supportTimeQuery;

    this.bufferManager = new BufferManager(this.device);
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

      // TODO: @webgpu/types 0.1.6 version has a bug to support both old
      // rendring context type and webgpu context type. Use any to bypass this.
      // tslint:disable-next-line:no-any
      this.dummyContext = this.dummyCanvas.getContext('webgpu') as any;
      this.dummyContext.configure({
        device,
        format: 'bgra8unorm',
      });

      document.body.appendChild(this.dummyCanvas);
    }
  }

  floatPrecision(): 32 {
    return 32;
  }

  defaultGpuBufferUsage(): number {
    return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST;
  }

  flushDisposalQueue() {
    this.tensorDisposalQueue.forEach(d => {
      this.maybeReleaseBuffer(d);
      this.tensorMap.delete(d);
    });
    this.uniformDisposalQueue.forEach(
        d => this.bufferManager.releaseBuffer(d.buffer, d.byteSize, d.usage));
    this.stagingDisposalQueue.forEach(
        d => this.bufferManager.releaseUploadBuffer(
            d.buffer, d.byteSize, d.usage));

    this.tensorDisposalQueue = [];
    this.uniformDisposalQueue = [];
    this.stagingDisposalQueue = [];
  }

  /**
   * Dispose the memory if the dataId has 0 refCount. Return true if the memory
   * is released or memory is not managed in this backend, false if memory is
   * not cleared.
   * @param dataId
   * @oaram force Optional, remove the data regardless of refCount
   */
  disposeData(dataId: DataId, force = false): boolean {
    if (this.tensorMap.has(dataId)) {
      const data = this.tensorMap.get(dataId);
      data.refCount--;
      if (!force && data.refCount > 0) {
        return false;
      }

      if (this.commandQueueOwnedIds.has(dataId)) {
        this.tensorDisposalQueue.push(dataId);
        return false;
      } else {
        this.maybeReleaseBuffer(dataId);
      }

      const {complexTensorInfos} = this.tensorMap.get(dataId);
      if (complexTensorInfos != null) {
        this.disposeData(complexTensorInfos.real.dataId, true);
        this.disposeData(complexTensorInfos.imag.dataId, true);
      }

      this.tensorMap.delete(dataId);
    }
    return true;
  }

  memory(): WebGPUMemoryInfo {
    return {
      numBytesInGPU: this.bufferManager.numBytesUsed,
      numBytesAllocatedInGPU: this.bufferManager.numBytesAllocated,
      unreliable: false
    } as WebGPUMemoryInfo;
  }

  getBufferManager(): BufferManager {
    return this.bufferManager;
  }

  acquireBuffer(
      byteSize: number,
      usage: GPUBufferUsageFlags = this.defaultGpuBufferUsage()) {
    return this.bufferManager.acquireBuffer(byteSize, usage);
  }

  maybeReleaseBuffer(dataId: DataId) {
    const info = this.tensorMap.get(dataId);
    if (info != null && info.bufferInfo.buffer != null) {
      this.bufferManager.releaseBuffer(
          info.bufferInfo.buffer, info.bufferInfo.byteSize,
          info.bufferInfo.usage);
      info.bufferInfo.buffer = null;
    }
  }

  /** Return refCount of a `TensorData`. */
  refCount(dataId: DataId): number {
    if (this.tensorMap.has(dataId)) {
      const tensorData = this.tensorMap.get(dataId);
      return tensorData.refCount;
    }
    return 0;
  }

  /** Increase refCount of a `TensorData`. */
  incRef(dataId: DataId): void {
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

  write(values: backend_util.BackendValues, shape: number[], dtype: DataType):
      DataId {
    if (dtype === 'complex64' && values != null) {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }

    const dataId = {id: this.nextDataId()};
    const byteSize =
        util.sizeFromShape(shape) * webgpu_util.GPUBytesPerElement(dtype);

    // bool is stored in Uint8Array, converted it to Int32Array.
    if (dtype === 'bool' && values instanceof Uint8Array) {
      values = Int32Array.from(values);
    }

    this.tensorMap.set(dataId, {
      dtype,
      values,
      bufferInfo: {byteSize, usage: this.defaultGpuBufferUsage()},
      refCount: 1
    });
    return dataId;
  }

  move(
      dataId: DataId, values: backend_util.BackendValues, shape: number[],
      dtype: DataType, refCount: number): void {
    if (dtype === 'complex64') {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }
    const byteSize =
        util.sizeFromShape(shape) * webgpu_util.GPUBytesPerElement(dtype);

    this.tensorMap.set(dataId, {
      dtype,
      values,
      bufferInfo: {byteSize, usage: this.defaultGpuBufferUsage()},
      refCount
    });
  }

  submitQueue() {
    this.ensureComputePassEnded();
    this.queue.submit([this.currentCommandEncoder.finish()]);
    this.currentCommandEncoder = null;
    this.dispatchNumberInEncoder = 0;

    this.commandQueueOwnedIds = new WeakSet<DataId>();

    this.flushDisposalQueue();
  }

  getBuffer(dataId: DataId) {
    this.uploadToGPU(dataId);
    return this.tensorMap.get(dataId).bufferInfo.buffer;
  }

  getFromPixelsProgram(type: 'copyExternal'|'import'): FromPixelsProgram {
    switch (type) {
      case 'copyExternal': {
        if (!this.fromPixelProgram) {
          this.fromPixelProgram = new FromPixelsProgram();
        }
        return this.fromPixelProgram;
      }
      case 'import': {
        if (!this.fromPixelImportProgram) {
          this.fromPixelImportProgram = new FromPixelsImportProgram();
        }
        return this.fromPixelImportProgram;
      }
      default:
        util.assert(false, () => `Unsupported fromPixels shape`);
        return undefined;
    }
  }

  ensureCommandEncoderReady() {
    if (!this.currentCommandEncoder) {
      this.currentCommandEncoder = this.device.createCommandEncoder();
    }
  }

  ensureComputePassEnded() {
    if (this.currentComputePass) {
      this.currentComputePass.endPass();
      this.currentComputePass = null;
    }
  }

  getComputePass() {
    if (!this.currentComputePass) {
      this.currentComputePass = this.currentCommandEncoder.beginComputePass();
    }
    return this.currentComputePass;
  }

  private async getBufferData(info: TensorBufferInfo):
      Promise<backend_util.BackendValues> {
    if (info.values != null) {
      // Data is on the CPU.
      return info.values;
    }
    const staging = this.acquireBuffer(
        info.bufferInfo.byteSize,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.ensureCommandEncoderReady();
    this.ensureComputePassEnded();
    this.currentCommandEncoder.copyBufferToBuffer(
        info.bufferInfo.buffer, 0, staging, 0, info.bufferInfo.byteSize);
    this.submitQueue();

    await staging.mapAsync(GPUMapMode.READ);
    const values = staging.getMappedRange().slice(0);

    staging.unmap();
    if (staging != null) {
      this.bufferManager.releaseBuffer(
          staging, info.bufferInfo.byteSize,
          GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
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
    const info = this.tensorMap.get(dataId);

    this.maybeReleaseBuffer(dataId);

    info.values = data;
    return info.values;
  }

  // TODO: Remove once this is fixed:
  // https://github.com/tensorflow/tfjs/issues/1595
  readSync(dataId: object): backend_util.BackendValues {
    const texData = this.tensorMap.get(dataId);
    const {values} = texData;

    if (values == null) {
      throw new Error(
          'WebGPU readSync is only available for CPU-resident tensors.');
    }

    return values;
  }

  async read(dataId: object): Promise<backend_util.BackendValues> {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);

    const {values} = info;

    if (values != null) {
      // TODO(xing.xu@intel.com): Merge backend_util.BackendValues and
      // backend_util.TypedArray.
      return this.convertAndCacheOnCPU(
                 dataId, values as backend_util.TypedArray) as
          backend_util.BackendValues;
    }

    // Download the values from the GPU.
    let vals: backend_util.BackendValues;
    if (info.dtype === 'complex64') {
      const ps = await Promise.all([
        this.read(info.complexTensorInfos.real.dataId),
        this.read(info.complexTensorInfos.imag.dataId)
      ]);

      const realValues = ps[0];
      const imagValues = ps[1];
      vals = backend_util.mergeRealAndImagArrays(
          realValues as Float32Array, imagValues as Float32Array);
    } else {
      const data = await this.getBufferData(info);
      vals =
          webgpu_util.ArrayBufferToTypedArray(data as ArrayBuffer, info.dtype);
    }
    this.convertAndCacheOnCPU(dataId, vals);
    return vals;
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

  async time(f: () => void): Promise<WebGPUTimingInfo> {
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

  getAndSavePipeline(key: string, getPipeline: () => GPUComputePipeline) {
    if (!(key in this.pipelineCache)) {
      this.pipelineCache[key] = getPipeline();
    }
    return this.pipelineCache[key];
  }

  makeTensorInfo(
      shape: number[], dtype: DataType,
      values?: backend_util.BackendValues|string[]): TensorInfo {
    let dataId;
    if (dtype === 'string' && values != null && values.length > 0 &&
        util.isString(values[0])) {
      const encodedValues =
          (values as {} as string[]).map(d => util.encodeString(d));

      dataId = this.write(encodedValues, shape, dtype);
    } else {
      dataId = this.write(values as backend_util.BackendValues, shape, dtype);
    }
    return {dataId, shape, dtype};
  }

  private tensorToBinding(tensor?: TensorInfo): GPUBindingResource {
    if (!tensor) {
      return null;
    }

    const tensorData = this.tensorMap.get(tensor.dataId);

    return {
      offset: 0,
      size: tensorData.bufferInfo.byteSize,
      buffer: tensorData.bufferInfo.buffer
    };
  }

  async getQueryTime(query: GPUQuerySet): Promise<number> {
    if (this.supportTimeQuery) {
      return this.getTimeFromQuerySet(query);
    } else {
      return 0;
    }
  }

  uploadToGPU(dataId: DataId): void {
    const info = this.tensorMap.get(dataId);

    if (info.bufferInfo.buffer != null) {
      // Already on the GPU.
      return;
    }

    info.bufferInfo.buffer = this.acquireBuffer(info.bufferInfo.byteSize);
    if (info.values) {
      const stagingBuffer = this.bufferManager.acquireUploadBuffer(
          info.bufferInfo.byteSize,
          GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC);
      const arrayBuffer = stagingBuffer.getMappedRange();
      if (info.dtype === 'int32' || info.dtype === 'bool') {
        new Int32Array(arrayBuffer).set(info.values as Int32Array);
      } else {
        new Float32Array(arrayBuffer).set(info.values as Float32Array);
      }
      stagingBuffer.unmap();
      this.ensureCommandEncoderReady();
      this.ensureComputePassEnded();
      this.currentCommandEncoder.copyBufferToBuffer(
          stagingBuffer, 0, info.bufferInfo.buffer, 0,
          info.bufferInfo.byteSize);

      const stagingInfo = {
        byteSize: info.bufferInfo.byteSize,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        buffer: stagingBuffer
      };
      this.stagingDisposalQueue.push(stagingInfo);
      // TODO: WebGPU doesn't support read data synchronously from GPU to CPU.
      // So it will report error when switching backend from WebGPU to others.
      // There are two situations: 1) swithcing the backend after running a
      // model; 2) swithcing the backend within the model. Temporarilly keep the
      // values on CPU to solve the first issue.
      // info.values = null;
    }
  }

  private makeUniformsDataView(data: DataView): GPUBindingResource {
    const dimensionsBuffer = this.acquireBuffer(
        data.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
    this.queue.writeBuffer(dimensionsBuffer, 0, data);

    return {offset: 0, size: data.byteLength, buffer: dimensionsBuffer};
  }

  private arrayToDataView(
      arrays: Array<{type: string; data: number[]}>, length: number): DataView {
    const BYTES_PER_ELEMENT = 4;
    const uniformDataView =
        new DataView(new ArrayBuffer(length * BYTES_PER_ELEMENT));

    let dataViewIndex = 0;
    arrays.forEach(array => {
      const arrayData = array.data;

      if (array.type !== 'int32' && array.type !== 'float32' &&
          array.type !== 'uint32') {
        throw new Error(`${array.type} not supported!`);
      }

      if (array.type === 'int32') {
        arrayData.forEach(d => {
          uniformDataView.setInt32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
          dataViewIndex++;
        });
      } else if (array.type === 'uint32') {
        arrayData.forEach(d => {
          uniformDataView.setUint32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
          dataViewIndex++;
        });
      } else {
        arrayData.forEach(d => {
          uniformDataView.setFloat32(
              dataViewIndex * BYTES_PER_ELEMENT, d, true);
          dataViewIndex++;
        });
      }
    });

    return uniformDataView;
  }

  private computePadding(uniformsWithType:
                             Array<{type: string; data: number[];}>): DataView {
    let currentOffset = 0;
    let padding = 0;
    let dataViewIndex = 0;
    const dimUniformsData: Array<{type: string; data: number[];}> = [];
    uniformsWithType.forEach((d, i) => {
      if (d.data.length === 0) {
        d.data = [1];
      }
      // Complete std140 layout rules are documented here:
      // tslint:disable-next-line:max-line-length
      // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
      let baseAlignment: number;
      switch (d.data.length) {
        case 0:
          baseAlignment = 1;
          break;
        case 1:
          baseAlignment = 1;
          break;
        case 2:
          baseAlignment = 2;
          break;
        case 3:
          baseAlignment = 4;
          break;
        case 4:
          baseAlignment = 4;
          break;
        default:
          util.assert(false, () => `Unsupported ${d.data.length}D shape`);
      }

      padding = Math.ceil(currentOffset / baseAlignment) * baseAlignment -
          currentOffset;
      for (let p = 0; p < padding; ++p) {
        dimUniformsData.push({type: d.type, data: [0]});
        dataViewIndex++;
      }
      dimUniformsData.push({type: d.type, data: d.data});
      dataViewIndex = dataViewIndex + d.data.length;
      currentOffset += d.data.length + padding;
    });

    return this.arrayToDataView(dimUniformsData, dataViewIndex);
  }

  // This layout is used by all programs except fromPixel.
  private createLayout(inputEntrySize: number): WebGPULayout {
    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [];
    // Output buffer binding layout.
    bindGroupLayoutEntries.push({
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {type: 'storage' as const}
    });
    // Input buffer binding layout. Depends on variableNames length.
    for (let i = 0; i < inputEntrySize; i++) {
      bindGroupLayoutEntries.push({
        binding: i + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: 'read-only-storage' as const}
      });
    }
    bindGroupLayoutEntries.push({
      binding: inputEntrySize + 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {type: 'uniform' as const}
    });
    const bindGroupLayout =
        this.device.createBindGroupLayout({entries: bindGroupLayoutEntries});
    const pipelineLayout =
        this.device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});
    return {bindGroupLayout, pipelineLayout};
  }

  private getCachedOrCreateLayout(inputEntrySize: number): WebGPULayout {
    if (!(inputEntrySize in this.layoutCache)) {
      this.layoutCache[inputEntrySize] = this.createLayout(inputEntrySize);
    }
    return this.layoutCache[inputEntrySize];
  }

  public runWebGPUProgram(
      program: webgpu_program.WebGPUProgram, inputs: TensorInfo[],
      outputDtype: DataType,
      programUniforms?: Array<{type: string; data: number[]}>,
      output?: TensorInfo): TensorInfo {
    if (!output) {
      output = this.makeTensorInfo(program.outputShape, outputDtype);
      if (util.sizeFromShape(output.shape) === 0) {
        // Short-circuit the computation since the result is empty (has 0 in its
        // shape).
        const outData = this.tensorMap.get(output.dataId);
        outData.values =
            util.getTypedArrayFromDType(output.dtype as 'float32', 0);
        return output;
      }
      this.uploadToGPU(output.dataId);
    }

    // There are five kinds of uniforms: NAN, shapes, shape strides, program
    // size, program defined uniforms.
    let uniformsWithType: Array<{type: string; data: number[];}> =
        [{type: 'float32', data: [NaN]}];
    const bufferShapes = inputs.concat(output).map(d => d.shape);
    const uniformsType = 'int32';
    bufferShapes.map(d => {
      uniformsWithType.push({type: uniformsType, data: d});
    });
    const strides = util.computeStrides(output.shape);
    uniformsWithType.push({type: uniformsType, data: strides});
    if (program.size) {
      const size = util.sizeFromShape(program.outputShape);
      uniformsWithType.push(
          {type: uniformsType, data: [program.isVec4 ? size / 4 : size]});
    }
    if (programUniforms) {
      uniformsWithType = [...uniformsWithType, ...programUniforms];
    }

    let uniforms: GPUBindingResource = null;
    const uniformsDataView = this.computePadding(uniformsWithType);
    const uniformsByteLength = uniformsDataView.byteLength;
    uniforms = this.makeUniformsDataView(uniformsDataView);

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
    const bufferTypes = inputsData.map(d => d.dtype).concat(output.dtype);
    const broadcastDims = inputsData.map(
        d => backend_util.getBroadcastDims(d.shape, output.shape));
    const inputShapesEqualsOutShape =
        inputsData.map(d => util.arraysEqual(d.shape, output.shape)).join('_');
    const broadcastDimsKey = broadcastDims.map(d => d.join('_')).join(';');
    const key = webgpu_program.makeShaderKey(
        program, bufferShapes, bufferTypes, broadcastDimsKey,
        inputShapesEqualsOutShape);

    const {bindGroupLayout, pipelineLayout} =
        this.getCachedOrCreateLayout(program.variableNames.length);

    const pipeline = this.getAndSavePipeline(key, () => {
      return webgpu_program.compileProgram(
          this.device, program, pipelineLayout, inputsData, output);
    });

    const shouldTimeProgram = this.activeTimers != null;

    // Creating bind groups on the fly should never be a bottleneck.
    const bg = webgpu_program.makeBindGroup(
        this.device, bindGroupLayout, inputs.map(t => this.tensorToBinding(t)),
        this.tensorToBinding(output), uniforms);

    this.ensureCommandEncoderReady();
    const pass = this.getComputePass();
    if (shouldTimeProgram) {
      if (this.supportTimeQuery) {
        pass.writeTimestamp(this.querySet, 0);
      }
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatch(
        program.dispatch[0], program.dispatch[1], program.dispatch[2]);
    if (shouldTimeProgram) {
      if (this.supportTimeQuery) {
        pass.writeTimestamp(this.querySet, 1);
      }
    }
    this.dispatchNumberInEncoder++;

    inputs.forEach(input => {
      this.commandQueueOwnedIds.add(input.dataId);
    });
    this.commandQueueOwnedIds.add(output.dataId);
    if (uniforms) {
      const uniformInfo = {
        byteSize: uniformsByteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
        // tslint:disable-next-line: no-unnecessary-type-assertion
        buffer: (uniforms as GPUBufferBinding).buffer
      };
      this.uniformDisposalQueue.push(uniformInfo);
    }

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

  runFromPixelsProgram(
      program: FromPixelsProgram, output: GPUBuffer, layout: WebGPULayout,
      externalResource: GPUExternalTexture|GPUTextureView, outputId: DataId) {
    const bindGroup = this.device.createBindGroup({
      layout: layout.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: output,
          }
        },
        {
          binding: 1,
          resource: externalResource,
        },
        {
          binding: 2,
          resource: {
            buffer: program.uniform,
          }
        }
      ],
    });
    this.ensureCommandEncoderReady();
    const passEncoder = this.getComputePass();
    const shouldTimeProgram = this.activeTimers != null;
    if (shouldTimeProgram) {
      if (this.supportTimeQuery) {
        passEncoder.writeTimestamp(this.querySet, 0);
      }
    }
    passEncoder.setPipeline(program.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatch(
        program.dispatch[0], program.dispatch[1], program.dispatch[2]);
    if (shouldTimeProgram) {
      if (this.supportTimeQuery) {
        passEncoder.writeTimestamp(this.querySet, 1);
      }
    }
    this.commandQueueOwnedIds.add(outputId);
    this.submitQueue();
    if (shouldTimeProgram) {
      this.activeTimers.push({
        name: program.constructor.name,
        query: this.getQueryTime(this.querySet)
      });
    }
  }

  async getTimeFromQuerySet(querySet: GPUQuerySet) {
    const queryBuffer = this.acquireBuffer(
        16, GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE);
    const dst = this.acquireBuffer(
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
            input =>
                this.tensorMap.get(input.dataId).bufferInfo.buffer == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
  }

  numDataIds() {
    return this.tensorMap.numDataIds() - this.tensorDisposalQueue.length;
  }

  dispose() {
    if (this.disposed) {
      return;
    }
    this.bufferManager.dispose();

    if (this.fromPixelProgram) {
      this.fromPixelProgram.dispose();
    }

    if (this.fromPixelImportProgram) {
      this.fromPixelImportProgram.dispose();
    }

    this.disposed = true;
  }
}
