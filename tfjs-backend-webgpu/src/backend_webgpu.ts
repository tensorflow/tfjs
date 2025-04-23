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

import {backend_util, BackendValues, buffer, DataStorage, DataType, engine, env, GPUData, KernelBackend, Rank, RecursiveArray, ShapeMap, Tensor, TensorBuffer, TensorInfo, TimingInfo, TypedArray, util, WebGPUData} from '@tensorflow/tfjs-core';

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

type TensorData = {
  values: BackendValues,
  dtype: DataType,
  shape: number[],
  refCount: number,
  resource?: GPUBuffer|GPUTexture|GPUExternalTexture,
  // external is true means we use the resource provided by users directly
  // (without a copy), so users should be responsible for its release.
  external?: boolean,
  // For complex numbers, the real and imaginary parts are stored as their own
  // individual tensors, with a parent joining the two with the
  // complexTensorInfos field.
  complexTensorInfos?: {real: TensorInfo, imag: TensorInfo}
};

interface DataId {}

export type WebGPUKernelInfo = {
  name: string,
  query: Promise<number>,
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
  private commandEncoder: GPUCommandEncoder;
  private computePassEncoder: GPUComputePassEncoder;
  private commandQueueOwnedIds = new WeakSet<DataId>();
  private dispatchCountInPass = 0;
  private disposed = false;
  private downloadWaitMs = 0;
  private dummyCanvas: HTMLCanvasElement;
  private dummyContext: GPUCanvasContext;
  private tensorDataPendingDisposal: DataId[] = [];
  private static nextDataId = 0;
  private pipelineCache:
      {[key: string]: GPUComputePipeline|Promise<GPUComputePipeline>};
  private programTimersStack: TimerNode[];
  private queryResolveBuffer: GPUBuffer = null;
  private querySet: GPUQuerySet = null;
  private querySetCount = 2;
  private stagingPendingDisposal: GPUBuffer[] = [];
  private supportTimestampQuery: boolean;
  private uniformPendingDisposal: GPUBuffer[] = [];
  private uploadWaitMs = 0;
  private hasReadSyncWarned = false;
  private hasTimestampQueryWarned = false;

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
    this.commandEncoder = null;
    this.computePassEncoder = null;
    this.adapterInfo = new AdapterInfo(adapterInfo);
    this.supportTimestampQuery = this.device.features.has('timestamp-query');
    this.thresholdToIncreaseWorkgroups =
        this.adapterInfo.intelGPUGeneration >= 12 ? 16 : 8;

    this.bufferManager = new BufferManager(this.device);
    this.textureManager = new TextureManager(this.device);
    this.tensorMap = new DataStorage(this, engine());

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

  /**
   * Dispose the memory if the dataId has 0 refCount. Return true if the memory
   * is released or delayed in this backend, false if there are still
   * references.
   * @param dataId
   * @oaram force Optional, remove the data regardless of refCount
   */
  override disposeData(dataId: DataId, force = false): boolean {
    // No-op if already disposed.
    if (!this.tensorMap.has(dataId)) {
      return true;
    }

    const tensorData = this.tensorMap.get(dataId);
    if (force) {
      tensorData.refCount = 0;
    } else {
      tensorData.refCount--;
    }

    if (tensorData.refCount > 0) {
      return false;
    }

    if (tensorData.complexTensorInfos != null) {
      this.disposeData(tensorData.complexTensorInfos.real.dataId);
      this.disposeData(tensorData.complexTensorInfos.imag.dataId);
    }

    if (this.commandQueueOwnedIds.has(dataId)) {
      this.tensorDataPendingDisposal.push(dataId);
      return true;
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

  private releaseResource(dataId: DataId) {
    const tensorData = this.tensorMap.get(dataId);
    if (!tensorData || !tensorData.resource) {
      return;
    }

    // If tensor's resource is from external, do not release.
    if (tensorData.external) {
      tensorData.resource = null;
      return;
    }
    if (tensorData.resource instanceof GPUBuffer) {
      this.bufferManager.releaseBuffer(tensorData.resource);
    } else if (tensorData.resource instanceof GPUTexture) {
      this.textureManager.releaseTexture(tensorData.resource);
    }
    tensorData.resource = null;
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

  override write(values: BackendValues, shape: number[], dtype: DataType):
      DataId {
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
      dataId: DataId, values: BackendValues, shape: number[], dtype: DataType,
      refCount: number): void {
    if (dtype === 'complex64') {
      throw new Error(
          `Cannot write to a complex64 dtype. ` +
          `Please use tf.complex(real, imag).`);
    }
    this.tensorMap.set(dataId, {dtype, shape, values, refCount});
  }

  submitQueue() {
    this.queue.submit([this.commandEncoder.finish()]);
    this.commandEncoder = null;
    this.dispatchCountInPass = 0;

    this.commandQueueOwnedIds = new WeakSet<DataId>();

    this.tensorDataPendingDisposal.forEach(d => {
      this.releaseResource(d);
      this.tensorMap.delete(d);
    });

    this.uniformPendingDisposal.forEach(
        b => this.bufferManager.releaseBuffer(b));
    this.stagingPendingDisposal.forEach(
        b => this.bufferManager.releaseBuffer(b, false));

    this.tensorDataPendingDisposal = [];
    this.uniformPendingDisposal = [];
    this.stagingPendingDisposal = [];
  }

  ensureCommandEncoderReady() {
    if (!this.commandEncoder) {
      this.commandEncoder = this.device.createCommandEncoder();
    }
  }

  endComputePassEncoder() {
    if (this.computePassEncoder) {
      this.computePassEncoder.end();
      this.computePassEncoder = null;
    }
  }

  // Check if parallel compilation is done.
  async checkCompileCompletionAsync() {
    let pipelines: GPUComputePipeline[];
    try {
      pipelines = await Promise.all(Object.values(this.pipelineCache));
    } catch (e) {
      // TODO: Add test case to catch this exception.
      throw new Error(e.message);
    }
    Object.keys(this.pipelineCache).map((key, i) => {
      this.pipelineCache[key] = pipelines[i];
    });
  }

  public async getBufferData(buffer: GPUBuffer): Promise<ArrayBuffer> {
    if (env().getBool('WEBGPU_ENGINE_COMPILE_ONLY')) {
      console.warn(
          'The data may be invalid since WEBGPU_ENGINE_COMPILE_ONLY is true, this can only be called when WEBGPU_ENGINE_COMPILE_ONLY is false');
      return null;
    }
    const size = buffer.size;
    const stagingBuffer = this.bufferManager.acquireBuffer(
        size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.ensureCommandEncoderReady();
    this.endComputePassEncoder();
    this.commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    this.submitQueue();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const values = stagingBuffer.getMappedRange().slice(0);

    stagingBuffer.unmap();
    if (stagingBuffer != null) {
      this.bufferManager.releaseBuffer(stagingBuffer);
    }

    // Need to get texture from swapChain to enable profiling tool
    // to capture a frame
    if (env().getBool('WEBGPU_USE_PROFILE_TOOL')) {
      util.assert(
          this.dummyContext !== undefined,
          () => `Fail to get context for profiling tool`);
      this.dummyContext.getCurrentTexture();
    }

    return values;
  }

  private convertAndCacheOnCPU(dataId: DataId, data: BackendValues):
      BackendValues {
    const tensorData = this.tensorMap.get(dataId);
    tensorData.values = data;
    return tensorData.values;
  }

  override readSync(dataId: object): BackendValues {
    const tensorData = this.tensorMap.get(dataId);
    const {values, complexTensorInfos} = tensorData;

    if (values != null || tensorData.dtype === 'string') {
      return values;
    }

    if (tensorData.dtype === 'complex64') {
      const realValues =
          this.readSync(complexTensorInfos.real.dataId) as Float32Array;
      const imagValues =
          this.readSync(complexTensorInfos.imag.dataId) as Float32Array;
      const complexVals = util.convertBackendValuesAndArrayBuffer(
          backend_util.mergeRealAndImagArrays(realValues, imagValues).buffer,
          'float32');
      this.convertAndCacheOnCPU(dataId, complexVals);
      return complexVals;
    }

    if (!this.hasReadSyncWarned) {
      this.hasReadSyncWarned = true;
      console.warn(
          `The performance of synchronously reading data from GPU to CPU is ` +
          `poor on the webgpu backend, please use asynchronous APIs instead.`);
    }

    const alphaModes: GPUCanvasAlphaMode[] = ['opaque', 'premultiplied'];

    const buffer = tensorData.resource as GPUBuffer;
    const bufferSize = buffer.size;
    util.assert(
        bufferSize % 4 === 0,
        () => 'Because there is 4 bytes for ' +
            'one pixel, buffer size must be multiple of 4.');
    const pixelsSize = bufferSize / 4;
    const valsGPU = new ArrayBuffer(bufferSize);
    // TODO: adjust the reading window size according the `bufferSize`.
    const canvasWidth = 256, canvasHeight = 256;
    const stagingDeviceStorage: OffscreenCanvas[] =
        alphaModes.map(_ => new OffscreenCanvas(canvasWidth, canvasHeight));
    const stagingHostStorage = new OffscreenCanvas(canvasWidth, canvasHeight);

    this.endComputePassEncoder();
    stagingDeviceStorage
        .map((storage, index) => {
          const context = storage.getContext('webgpu');
          // TODO: use rgba8unorm format when this format is supported on Mac.
          // https://bugs.chromium.org/p/chromium/issues/detail?id=1298618
          context.configure({
            device: this.device,
            format: 'bgra8unorm',
            usage: GPUTextureUsage.COPY_DST,
            alphaMode: alphaModes[index],
          });
          return context.getCurrentTexture();
        })
        .map((texture, index) => {
          const bytesPerRow = canvasWidth * 4;
          const readDataGPUToCPU =
              (width: number, height: number, offset: number) => {
                this.ensureCommandEncoderReady();
                this.commandEncoder.copyBufferToTexture(
                    {
                      buffer,
                      bytesPerRow,
                      offset,
                    },
                    {
                      texture,
                    },
                    {
                      width,
                      height,
                    });
                this.submitQueue();

                const context = stagingHostStorage.getContext('2d', {
                  willReadFrequently: true,
                });
                context.clearRect(0, 0, width, height);
                context.drawImage(stagingDeviceStorage[index], 0, 0);
                const stagingValues =
                    context.getImageData(0, 0, width, height).data;
                const alphaMode = alphaModes[index];
                const span =
                    new Uint8ClampedArray(valsGPU, offset, width * height * 4);
                for (let k = 0; k < span.length; k += 4) {
                  if (alphaMode === 'premultiplied') {
                    span[k + 3] = stagingValues[k + 3];
                  } else {
                    const value = stagingValues[k];
                    span[k] = stagingValues[k + 2];
                    span[k + 1] = stagingValues[k + 1];
                    span[k + 2] = value;
                  }
                }
              };

          const fullyReadCount =
              Math.floor(pixelsSize / (canvasWidth * canvasHeight));
          let width = canvasWidth, height = canvasHeight, offset = 0;
          for (let i = 0; i < fullyReadCount; i++) {
            // Read the buffer data, which fully fill the whole canvas.
            readDataGPUToCPU(width, height, offset);
            offset += canvasWidth * canvasHeight * 4;
          }

          const remainSize = pixelsSize % (canvasWidth * canvasHeight);
          height = Math.floor(remainSize / canvasWidth);
          if (height > 0) {
            // Read the buffer data, which fully fill certain rows of canvas.
            readDataGPUToCPU(width, height, offset);
            offset += height * (canvasWidth * 4);
          }

          width = remainSize % canvasWidth;
          if (width > 0) {
            // Read the buffer data, which not fully fill one row of canvas.
            readDataGPUToCPU(width, 1, offset);
          }
        });

    const vals =
        util.convertBackendValuesAndArrayBuffer(valsGPU, tensorData.dtype);
    this.convertAndCacheOnCPU(dataId, vals);
    return vals;
  }

  override async read(dataId: object): Promise<BackendValues> {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const tensorData = this.tensorMap.get(dataId);

    const {values} = tensorData;

    if (values != null) {
      return values;
    }

    // Download the values from the GPU.
    let vals: BackendValues;
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
      const data = await this.getBufferData(tensorData.resource as GPUBuffer);
      vals = util.convertBackendValuesAndArrayBuffer(data, tensorData.dtype);
    }
    this.convertAndCacheOnCPU(dataId, vals);
    return vals;
  }

  // The source GPUBuffer and destination GPUBuffer have the same size and
  // usage.
  private copyBuffer(srcBuffer: GPUBuffer) {
    const size = srcBuffer.size;
    const usage = srcBuffer.usage;
    const dstBuffer = this.bufferManager.acquireBuffer(size, usage);
    this.ensureCommandEncoderReady();
    this.endComputePassEncoder();
    this.commandEncoder.copyBufferToBuffer(srcBuffer, 0, dstBuffer, 0, size);
    this.submitQueue();
    return dstBuffer;
  }

  /**
   * Create a TF.js tensor out of an existing WebGPU buffer.
   */
  override createTensorFromGPUData(
      webGPUData: WebGPUData, shape: number[], dtype: DataType): Tensor {
    let buffer = webGPUData.buffer;
    if (dtype === 'complex64') {
      throw new Error(`Cannot write to a complex64 dtype. `);
    }
    const dataId = {id: this.nextDataId()};
    this.tensorMap.set(dataId, {
      dtype,
      shape,
      values: null,
      refCount: 1,
      external: webGPUData.zeroCopy
    });
    const tensorData = this.tensorMap.get(dataId);
    const size = webgpu_util.GPUBytesPerElement(tensorData.dtype) *
        util.sizeFromShape(tensorData.shape);
    if (webGPUData.buffer.size < size) {
      throw new Error(`GPUBuffer size(${
          webGPUData.buffer.size}) is smaller than tensor size(${size})!`);
    } else if (
        (webGPUData.buffer.usage &
         (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)) !==
        (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)) {
      throw new Error(
          'GPUBuffer.usage should include GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC!');
    }

    // Do buffer copy by default.
    if (webGPUData.zeroCopy !== true) {
      buffer = this.copyBuffer(buffer);
    }
    tensorData.resource = buffer;
    return engine().makeTensorFromDataId(dataId, shape, dtype, this);
  }

  /**
   * Read tensor to a new GPUBuffer.
   * @param dataId The source tensor.
   */
  override readToGPU(dataId: DataId): GPUData {
    let srcTensorData = this.tensorMap.get(dataId);
    const {values, dtype, shape} = srcTensorData;
    let resource = srcTensorData.resource;

    if (dtype === 'complex64') {
      throw new Error('Does not support reading buffer for complex64 dtype.');
    }

    if (resource == null) {
      if (values != null) {
        this.uploadToGPU(dataId);
        srcTensorData = this.tensorMap.get(dataId);
        resource = srcTensorData.resource;
      } else {
        throw new Error('There is no data on GPU or CPU.');
      }
    }

    const srcBuffer = resource as GPUBuffer;
    const size = srcBuffer.size;
    const usage = srcBuffer.usage;
    const buffer = this.bufferManager.acquireBuffer(size, usage);
    this.ensureCommandEncoderReady();
    this.endComputePassEncoder();
    this.commandEncoder.copyBufferToBuffer(
        resource as GPUBuffer, 0, buffer, 0, size);
    this.submitQueue();

    const tensorInfo = this.makeTensorInfo(shape, dtype);
    // Make engine track this tensor, so that we can dispose it later.
    const tensorRef = engine().makeTensorFromTensorInfo(tensorInfo);

    const tensorData = this.tensorMap.get(tensorInfo.dataId);
    tensorData.resource = buffer;

    return {tensorRef, buffer};
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
    if (!this.supportTimestampQuery && !this.hasTimestampQueryWarned) {
      console.warn(
          `This device doesn't support timestamp-query extension. ` +
          `Start Chrome browser with flag ` +
          `--enable-dawn-features=allow_unsafe_apis to try it again. ` +
          `Otherwise, zero will be shown for the kernel time when profiling ` +
          `mode is enabled.`);
      this.hasTimestampQueryWarned = true;
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
      values?: BackendValues|string[]): TensorInfo {
    if (dtype === 'string' && values != null && values.length > 0 &&
        util.isString(values[0])) {
      values = (values as unknown as string[]).map(d => util.encodeString(d));
    }
    const dataId = this.write(values as BackendValues, shape, dtype);
    return {dataId, shape, dtype};
  }

  private tensorToBinding(tensor?: TensorInfo): GPUBindingResource {
    if (!tensor) {
      return null;
    }

    const tensorData = this.tensorMap.get(tensor.dataId);
    const resource = tensorData.resource;

    if (resource instanceof GPUBuffer) {
      return {buffer: resource};
    }
    if (resource instanceof GPUTexture) {
      return resource.createView();
    }
    // GPUExternalTexture
    return resource;
  }

  uploadToGPU(dataId: DataId): void {
    const tensorData = this.tensorMap.get(dataId);
    // Already on the GPU.
    if (tensorData.resource != null) {
      return;
    }

    const size = webgpu_util.GPUBytesPerElement(tensorData.dtype) *
        util.sizeFromShape(tensorData.shape);
    let buffer;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST;
    if (tensorData.values) {
      buffer = this.bufferManager.acquireBuffer(size, usage, true);
      if (buffer.mapState === 'unmapped') {
        const stagingBuffer = this.bufferManager.acquireBuffer(
            size, GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, true,
            false);
        const arrayBuffer = stagingBuffer.getMappedRange();
        if (tensorData.dtype === 'int32' || tensorData.dtype === 'bool') {
          new Int32Array(arrayBuffer).set(tensorData.values as TypedArray);
        } else {
          new Float32Array(arrayBuffer).set(tensorData.values as Float32Array);
        }
        stagingBuffer.unmap();
        this.ensureCommandEncoderReady();
        this.endComputePassEncoder();
        this.commandEncoder.copyBufferToBuffer(
            stagingBuffer, 0, buffer, 0, size);

        this.stagingPendingDisposal.push(stagingBuffer);
      } else {
        const arrayBuffer = buffer.getMappedRange();
        if (tensorData.dtype === 'int32' || tensorData.dtype === 'bool') {
          new Int32Array(arrayBuffer).set(tensorData.values as TypedArray);
        } else {
          new Float32Array(arrayBuffer).set(tensorData.values as Float32Array);
        }
        buffer.unmap();
      }

      // Once uploaded, don't store the values on cpu.
      tensorData.values = null;
    } else {
      buffer = this.bufferManager.acquireBuffer(size, usage);
    }
    tensorData.resource = buffer;
  }

  private makeUniforms(programUniform: ProgramUniform): GPUBindingResource {
    let currentOffset = 0;
    let preLength = 0;
    const offsets: number[] = [];
    let maxAlignmentOfField = 1;
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
      if (baseAlignment > maxAlignmentOfField) {
        maxAlignmentOfField = baseAlignment;
      }
      currentOffset = Math.ceil(currentOffset / baseAlignment) * baseAlignment;
      preLength = d.data.length;
      offsets.push(currentOffset);
      currentOffset += d.data.length * 4;
    });

    currentOffset =
        Math.ceil(currentOffset / maxAlignmentOfField) * maxAlignmentOfField;
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
    this.uniformPendingDisposal.push(uniformBuffer);

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

    program.shaderKey =
        webgpu_program.makeShaderKey(program, inputsData, output);

    const parallelCompilation = env().getBool('WEBGPU_ENGINE_COMPILE_ONLY');
    if (!(program.shaderKey in this.pipelineCache)) {
      this.pipelineCache[program.shaderKey] = webgpu_program.compileProgram(
          this.device, program, inputsData, output, parallelCompilation);
    }
    program.pipeline = this.pipelineCache[program.shaderKey];

    if (!parallelCompilation) {
      this.recordAndSubmit(program, output, inputs, programDefinedUniform);
    }
    return output;
  }

  private recordAndSubmit(
      program: webgpu_program.WebGPUProgram, output: TensorInfo,
      inputs: TensorInfo[], programDefinedUniform?: ProgramUniform) {
    if (program.pipeline instanceof Promise) {
      throw new Error(
          'Please call checkCompileCompletionAsync to ensure parallel compilation is done!');
    }
    // There are six kinds of uniforms: NAN, INFINITY, shapes, shape strides,
    // program size, program defined uniforms.
    let programUniform: ProgramUniform = [];
    let bufferShapes: number[][] = [];
    const uniformsType = 'int32';
    if (program.pixelsOpType == null) {
      programUniform.push(
          {type: 'float32', data: [NaN]}, {type: 'float32', data: [Infinity]});
      bufferShapes = inputs.concat(output).map(d => d.shape);
      const uniformsType = 'int32';
      bufferShapes.map(d => {
        programUniform.push({type: uniformsType, data: d});
        const strides = util.computeStrides(d);
        programUniform.push({type: uniformsType, data: strides});
      });
    } else {
      const strides = util.computeStrides(output.shape);
      programUniform.push({type: uniformsType, data: strides});
    }
    if (program.size) {
      const size = util.sizeFromShape(program.outputShape);
      programUniform.push({
        type: uniformsType,
        data: [program.outputComponent ? size / program.outputComponent : size]
      });
    }

    if (programDefinedUniform) {
      programUniform = [...programUniform, ...programDefinedUniform];
    }
    const bindings = [
      this.tensorToBinding(output), ...inputs.map(t => this.tensorToBinding(t)),
      this.makeUniforms(programUniform)
    ];

    inputs.forEach(input => {
      this.commandQueueOwnedIds.add(input.dataId);
    });
    this.commandQueueOwnedIds.add(output.dataId);

    const bindGroup = this.device.createBindGroup({
      layout: program.pipeline.getBindGroupLayout(0),
      entries: bindings.map((b, i) => ({binding: i, resource: b})),
    });

    const shouldTimeProgram = this.activeTimers != null;
    this.ensureCommandEncoderReady();

    const computePassDescriptor: GPUComputePassDescriptor = {};
    if (shouldTimeProgram && this.supportTimestampQuery) {
      this.endComputePassEncoder();
      if (this.querySet == null) {
        this.querySet = this.device.createQuerySet({
          type: 'timestamp',
          count: this.querySetCount,
        });
      }
      computePassDescriptor.timestampWrites = {
        querySet: this.querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      };
      this.computePassEncoder =
          this.commandEncoder.beginComputePass(computePassDescriptor);
    } else if (!this.computePassEncoder) {
      this.computePassEncoder =
          this.commandEncoder.beginComputePass(computePassDescriptor);
    }

    this.computePassEncoder.setPipeline(program.pipeline);
    this.computePassEncoder.setBindGroup(0, bindGroup);
    this.computePassEncoder.dispatchWorkgroups(
        program.dispatch[0], program.dispatch[1], program.dispatch[2]);
    this.dispatchCountInPass++;

    if (shouldTimeProgram ||
        env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE') as
            number <= this.dispatchCountInPass ||
        program.pixelsOpType === webgpu_program.PixelsOpType.DRAW) {
      this.endComputePassEncoder();
      if (shouldTimeProgram) {
        this.activeTimers.push(
            {name: program.constructor.name, query: this.getQueryTime()});
      } else {
        this.submitQueue();
      }
    }
  }

  async getQueryTime(): Promise<number> {
    if (!this.supportTimestampQuery) {
      return 0;
    }

    if (this.queryResolveBuffer == null) {
      this.queryResolveBuffer = this.bufferManager.acquireBuffer(
          this.querySetCount * 8,
          GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST |
              GPUBufferUsage.QUERY_RESOLVE);
    }
    this.commandEncoder.resolveQuerySet(
        this.querySet, 0, this.querySetCount, this.queryResolveBuffer, 0);

    const queryStagingBuffer = this.bufferManager.acquireBuffer(
        this.querySetCount * 8,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    this.commandEncoder.copyBufferToBuffer(
        this.queryResolveBuffer, 0, queryStagingBuffer, 0,
        this.querySetCount * 8);

    this.submitQueue();

    await queryStagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = new BigUint64Array(queryStagingBuffer.getMappedRange());
    const time = Number(arrayBuffer[1] - arrayBuffer[0]) / 1000000;
    queryStagingBuffer.unmap();
    this.bufferManager.releaseBuffer(queryStagingBuffer);
    return time;
  }

  shouldExecuteOnCPU(
      inputs: TensorInfo[],
      sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD): boolean {
    return env().getBool('WEBGPU_CPU_FORWARD') &&
        inputs.every(
            input => this.tensorMap.get(input.dataId).resource == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
  }

  override numDataIds() {
    return this.tensorMap.numDataIds() - this.tensorDataPendingDisposal.length;
  }

  override dispose() {
    if (this.disposed) {
      return;
    }
    if (this.querySet != null) {
      this.querySet.destroy();
    }
    this.bufferManager.dispose();
    this.textureManager.dispose();
    this.disposed = true;
  }
}
