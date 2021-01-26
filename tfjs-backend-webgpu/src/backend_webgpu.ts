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

/// <reference types="@webgpu/types" />

import './flags_webgpu';

import {backend_util, DataStorage, DataType, engine, env, KernelBackend, RecursiveArray, Tensor, TensorInfo, TimingInfo, util} from '@tensorflow/tfjs-core';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';

import {BufferManager} from './buffer_manager';
import {BinaryOpType, getBinaryOpString} from './kernels/binary_ops';
import {FromPixelsProgram} from './kernels/FromPixels_utils/from_pixels_webgpu';
import * as unary_op from './kernels/unary_op_webgpu';
import * as webgpu_program from './kernels/webgpu_program';
import {WebGPUBinary} from './kernels/webgpu_program';
import * as webgpu_util from './webgpu_util';

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
  bufferInfo: BufferInfo
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
const CPU_HANDOFF_SIZE_THRESHOLD = 128;

const DEFAULT_GPUBUFFER_USAGE =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class WebGPUBackend extends KernelBackend {
  device: GPUDevice;
  queue: GPUQueue;
  glslang: Glslang;
  commandQueue: GPUCommandEncoder[];
  tensorMap: DataStorage<TensorBufferInfo>;
  fromPixelProgram: FromPixelsProgram;
  supportTimeQuery: boolean;

  private commandQueueOwnedIds = new WeakSet<DataId>();
  private binaryCache: {[key: string]: WebGPUBinary};
  private bufferManager: BufferManager;

  private tensorDisposalQueue: DataId[] = [];
  private uniformDisposalQueue: BufferInfo[] = [];

  private disposed = false;

  private programTimersStack: TimerNode[];
  private activeTimers: TimerNode[];
  private uploadWaitMs = 0;
  private downloadWaitMs = 0;
  private cpuBackend: KernelBackend;
  private querySet: GPUQuerySet;

  constructor(device: GPUDevice, glslang: Glslang, supportTimeQuery = false) {
    super();
    this.binaryCache = {};
    this.device = device;
    this.queue = device.defaultQueue;
    this.commandQueue = [];
    this.glslang = glslang;
    this.supportTimeQuery = supportTimeQuery;

    this.bufferManager = new BufferManager(this.device);
    this.tensorMap = new DataStorage(this, engine());
    if (this.supportTimeQuery) {
      this.querySet = this.device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
    }
  }

  floatPrecision(): 32 {
    return 32;
  }

  flushDisposalQueue() {
    this.tensorDisposalQueue.forEach(d => {
      this.maybeReleaseBuffer(d);
      this.tensorMap.delete(d);
    });
    this.uniformDisposalQueue.forEach(
        d => this.bufferManager.releaseBuffer(d.buffer, d.byteSize, d.usage));

    this.tensorDisposalQueue = [];
    this.uniformDisposalQueue = [];
  }

  disposeData(dataId: DataId): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    if (this.commandQueueOwnedIds.has(dataId)) {
      this.tensorDisposalQueue.push(dataId);
      return;
    } else {
      this.maybeReleaseBuffer(dataId);
    }

    this.tensorMap.delete(dataId);
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
      byteSize: number, usage: GPUBufferUsageFlags = DEFAULT_GPUBUFFER_USAGE) {
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

  write(values: backend_util.BackendValues, shape: number[], dtype: DataType):
      DataId {
    const dataId = {};
    const byteSize =
        util.sizeFromShape(shape) * webgpu_util.GPUBytesPerElement(dtype);

    this.tensorMap.set(dataId, {
      dtype,
      values,
      bufferInfo: {byteSize, usage: DEFAULT_GPUBUFFER_USAGE}
    });
    return dataId;
  }

  move(
      dataId: DataId, values: backend_util.BackendValues, shape: number[],
      dtype: DataType): void {
    const byteSize =
        util.sizeFromShape(shape) * webgpu_util.GPUBytesPerElement(dtype);

    this.tensorMap.set(dataId, {
      dtype,
      values,
      bufferInfo: {byteSize, usage: DEFAULT_GPUBUFFER_USAGE}
    });
  }

  submitQueue() {
    this.queue.submit(this.commandQueue.map(enc => enc.finish()));
    this.commandQueue = [];

    this.commandQueueOwnedIds = new WeakSet<DataId>();

    this.flushDisposalQueue();
  }

  getBuffer(dataId: DataId) {
    this.uploadToGPU(dataId);
    return this.tensorMap.get(dataId).bufferInfo.buffer;
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
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
        info.bufferInfo.buffer, 0, staging, 0, info.bufferInfo.byteSize);
    this.commandQueue.push(encoder);
    this.submitQueue();

    await staging.mapAsync(GPUMapMode.READ);
    const values = staging.getMappedRange().slice(0);

    staging.unmap();
    if (staging != null) {
      this.bufferManager.releaseBuffer(
          staging, info.bufferInfo.byteSize,
          GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
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
    const data = await this.getBufferData(info);

    const dataAsTypedArray =
        webgpu_util.ArrayBufferToTypedArray(data as ArrayBuffer, info.dtype);
    this.convertAndCacheOnCPU(dataId, dataAsTypedArray);

    return dataAsTypedArray;
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

    if (this.supportTimeQuery) {
      const kernelMs = await Promise.all(flattenedActiveTimerQueries);
      res['kernelMs'] = util.sum(kernelMs);
      res['getExtraProfileInfo'] = () =>
          kernelMs.map((d, i) => ({name: flattenedActiveTimerNames[i], ms: d}))
              .map(d => `${d.name}: ${d.ms}`)
              .join(', ');
    } else {
      res['kernelMs'] = {
        error: 'WebGPU timestamp query was not supported in this environment.'
      };
    }
    this.uploadWaitMs = 0;
    this.downloadWaitMs = 0;
    return res;
  }

  getAndSavePipeline(
      key: string, getBinary: () => webgpu_program.WebGPUBinary) {
    if (!(key in this.binaryCache)) {
      this.binaryCache[key] = getBinary();
    }
    return this.binaryCache[key];
  }

  makeOutputArray<T extends Tensor>(shape: number[], dtype: DataType): T {
    const dataId = this.write(null /* values */, shape, dtype);

    return engine().makeTensorFromDataId(dataId, shape, dtype, this) as T;
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
      this.queue.writeBuffer(
          info.bufferInfo.buffer, 0, info.values as ArrayBuffer);
      info.values = null;
    }
  }

  public runWebGPUProgram(
      program: webgpu_program.WebGPUProgram, inputs: TensorInfo[],
      outputDtype: DataType, programUniforms?: number[]): TensorInfo {
    const output = this.makeTensorInfo(program.outputShape, outputDtype);

    let uniformDataLength;
    let uniforms: GPUBindingResource;
    if (program.uniforms) {
      // TODO: handle padding of program-specific uniforms
      const uniformData = new Int32Array(programUniforms);
      uniformDataLength = uniformData.byteLength;
      uniforms = this.makeUniforms(uniformData);
    }

    const inputsData = inputs.map((input: Tensor, i: number) => {
      this.uploadToGPU(input.dataId);

      return {
        // Returning dtype from tensorMap because it reflects dtype
        // of underlying buffer, rather than abstract dtype.
        dtype: this.tensorMap.get(input.dataId).dtype,
        shape: input.shape,
        name: program.variableNames[i]
      };
    });
    this.uploadToGPU(output.dataId);
    const bufferShapes = inputs.concat(output).map(d => d.shape);
    const bufferTypes = inputsData.map(d => d.dtype).concat(output.dtype);
    const key =
        webgpu_program.makeShaderKey(program, bufferShapes, bufferTypes);
    const {bindGroupLayout, pipeline} = this.getAndSavePipeline(key, () => {
      return webgpu_program.compileProgram(
          this.glslang, this.device, program, inputsData, output, uniforms);
    });

    const shouldTimeProgram = this.activeTimers != null;

    // Creating bind groups on the fly should never be a bottleneck.
    const bg = webgpu_program.makeBindGroup(
        this.device, bindGroupLayout, inputs.map(t => this.tensorToBinding(t)),
        this.tensorToBinding(output), uniforms);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
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
    pass.endPass();

    this.commandQueue.push(encoder);

    inputs.forEach(input => {
      this.commandQueueOwnedIds.add(input.dataId);
    });
    this.commandQueueOwnedIds.add(output.dataId);

    if (program.uniforms) {
      const uniformInfo = {
        byteSize: uniformDataLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
        buffer: (uniforms as GPUBufferBinding).buffer
      };
      this.uniformDisposalQueue.push(uniformInfo);
    }

    if (env().get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED')) {
      this.submitQueue();
    }

    if (shouldTimeProgram) {
      if (this.supportTimeQuery) {
        this.activeTimers.push({
          name: program.constructor.name,
          query: this.getQueryTime(this.querySet)
        });
      }
    }
    return output;
  }

  // TODO remove this once webgpu backend has been all modularized
  public compileAndRun<K extends TensorInfo>(
      program: webgpu_program.WebGPUProgram, inputs: TensorInfo[],
      output?: TensorInfo, programUniforms?: number[]): K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }
    const outInfo =
        this.runWebGPUProgram(program, inputs, output.dtype, programUniforms);
    return engine().makeTensorFromDataId(
               outInfo.dataId, outInfo.shape, outInfo.dtype) as {} as K;
  }

  async getTimeFromQuerySet(querySet: GPUQuerySet) {
    const queryBuffer = this.acquireBuffer(
        16, GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE);
    const dst = this.acquireBuffer(
        16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    const encoder = this.device.createCommandEncoder();
    // tslint:disable-next-line:no-any
    (encoder as any).resolveQuerySet(querySet, 0, 2, queryBuffer, 0);
    encoder.copyBufferToBuffer(queryBuffer, 0, dst, 0, 16);
    this.commandQueue.push(encoder);
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

  private makeUniforms(data: Uint32Array|Int32Array): GPUBindingResource {
    const dimensionsBuffer = this.acquireBuffer(
        data.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
    this.queue.writeBuffer(dimensionsBuffer, 0, data);

    return {offset: 0, size: data.byteLength, buffer: dimensionsBuffer};
  }

  private getCPUBackend(): KernelBackend|null {
    if (!env().getBool('WEBGPU_CPU_FORWARD')) {
      return null;
    }

    if (this.cpuBackend == null) {
      this.cpuBackend = engine().findBackend('cpu');
    }

    return this.cpuBackend;
  }

  shouldExecuteOnCPU(
      inputs: TensorInfo[],
      sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD): boolean {
    return this.getCPUBackend() != null &&
        inputs.every(
            input =>
                this.tensorMap.get(input.dataId).bufferInfo.buffer == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
  }

  mapActivationToShaderProgram(
      activation: backend_util.Activation, packed = false): string {
    if (activation === 'linear') {
      return unary_op.LINEAR;
    } else if (activation === 'relu') {
      return packed ? unary_op.RELU_VEC4 : unary_op.RELU;
    } else if (activation === 'elu') {
      return packed ? unary_op.ELU_VEC4 : unary_op.ELU;
    } else if (activation === 'relu6') {
      return unary_op.RELU6;
    } else if (activation === 'prelu') {
      return getBinaryOpString(BinaryOpType.PRELU, packed);
    }
    throw new Error(`Activation ${
        activation} has not been implemented for the WebGPU backend.`);
  }

  numDataIds() {
    return this.tensorMap.numDataIds();
  }

  dispose() {
    if (this.disposed) {
      return;
    }
    this.bufferManager.dispose();
    if (this.fromPixelProgram) {
      this.fromPixelProgram.dispose();
    }
    this.disposed = true;
  }
}
