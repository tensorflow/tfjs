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

import {DataMover, DataType, ENV, KernelBackend, Rank, ShapeMap, Tensor, Tensor3D, Tensor4D, util} from '@tensorflow/tfjs-core';
// How should this be imported?
import {Conv2DInfo} from '@tensorflow/tfjs-core/dist/ops/conv_util';
import * as shaderc from '@webgpu/shaderc';

import * as binary_op from './kernels/binary_op_webgpu';
import {BinaryOpProgram} from './kernels/binary_op_webgpu';
import {MatMulPackedProgram} from './kernels/matmul_packed_webgpu';
import {MatMulProgram} from './kernels/matmul_webgpu';
import {MaxPoolProgram} from './kernels/maxpool_webgpu';
import {PadProgram} from './kernels/pad_webgpu';
import * as unary_op from './kernels/unary_op_webgpu';
import {UnaryOpProgram} from './kernels/unary_op_webgpu';
import * as webgpu_program from './kernels/webgpu_program';
import {WebGPUBinary} from './kernels/webgpu_program';

type TensorInfo = {
  byteSize: number,
  values: Float32Array|Int32Array|Uint8Array,
  id: number,
  buffer: GPUBuffer
};

interface DataId {}

export class WebGPUBackend extends KernelBackend {
  device: GPUDevice;
  queue: GPUQueue;
  shaderc: shaderc.Shaderc;
  compiler: shaderc.Compiler;
  compileOpts: shaderc.CompileOptions;
  commandQueue: GPUCommandEncoder[];

  private binaryCache: {[key: string]: WebGPUBinary};

  constructor(device: GPUDevice, shaderc: shaderc.Shaderc) {
    super();
    this.binaryCache = {};
    this.device = device;
    this.queue = device.getQueue();
    this.commandQueue = [];
    this.shaderc = shaderc;
    this.compiler = new shaderc.Compiler();
    const opts = new shaderc.CompileOptions();
    opts.SetOptimizationLevel(shaderc.optimization_level.performance);
    this.compileOpts = opts;
  }

  floatPrecision(): 32 {
    return 32;
  }

  setDataMover(dataMover: DataMover): void {
    // TODO: tfjs team to implement this. Call GPUBuffer.destroy()
  }

  private tensorMap = new WeakMap<DataId, TensorInfo>();

  disposeData(dataId: DataId): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    this.destroyBuffer(info.byteSize, info.buffer);
  }

  private createBuffer(
      size: number,
      usage: GPUBufferUsage = GPUBufferUsage.STORAGE |
          GPUBufferUsage.TRANSFER_SRC | GPUBufferUsage.TRANSFER_DST) {
    return this.device.createBuffer({size, usage});
  }

  private destroyBuffer(byteSize: number, buffer: GPUBuffer) {
    // TODO: recycle deleted buffers
    buffer.destroy();
  }

  register(dataId: object, shape: number[], dtype: DataType): void {
    if (!this.tensorMap.has(dataId)) {
      const byteSize = util.sizeFromShape(shape) * util.bytesPerElement(dtype);
      const buffer = this.createBuffer(byteSize);
      this.tensorMap.set(dataId, {byteSize, values: null, id: -1, buffer});
    }
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    info.values = values;
    info.buffer.setSubData(0, values);
    this.tensorMap.set(dataId, info);
  }

  private submitQueue() {
    this.queue.submit(this.commandQueue.map(enc => enc.finish()));

    this.commandQueue = [];
  }

  private async getBufferData(info: TensorInfo): Promise<ArrayBuffer> {
    const staging = this.createBuffer(
        info.byteSize, GPUBufferUsage.TRANSFER_DST | GPUBufferUsage.MAP_READ);
    {
      const encoder = this.device.createCommandEncoder({});
      encoder.copyBufferToBuffer(info.buffer, 0, staging, 0, info.byteSize);
      this.commandQueue.push(encoder);
      this.submitQueue();
    }
    const mapped: ArrayBuffer = await staging.mapReadAsync();

    return mapped.slice(0);
  }

  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    const data = await this.getBufferData(info);

    return new Float32Array(data);
  }

  private getAndSavePipeline(
      key: string, getBinary: () => webgpu_program.WebGPUBinary) {
    if (!(key in this.binaryCache)) {
      this.binaryCache[key] = getBinary();
    }
    return this.binaryCache[key];
  }

  private makeOutputArray<T extends Tensor>(shape: number[], dtype: DataType):
      T {
    return Tensor.make(shape, {}, dtype, this) as T;
  }

  private tensorToBinding(tensor?: Tensor): webgpu_program.BindingInfo {
    if (!tensor) {
      return null;
    }

    const tensorData = this.tensorMap.get(tensor.dataId);

    return {
      resource: {
        offset: 0,
        size: tensor.size * util.bytesPerElement(tensor.dtype),
        buffer: tensorData.buffer
      }
    };
  }

  private compileAndRun<
      K extends {dtype: DataType, size: number, dataId: {}, shape: number[]}>(
      program: webgpu_program.WebGPUProgram, inputs: Tensor[], output?: Tensor,
      uniforms?: webgpu_program.BindingInfo): K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }
    const key = webgpu_program.makeShaderKey(program);
    const {bindGroupLayout, pipeline} = this.getAndSavePipeline(key, () => {
      return webgpu_program.compileProgram(
          this.compiler, this.shaderc.shader_kind.compute, this.compileOpts,
          this.device, program, inputs, output, uniforms);
    });

    // Creating bind groups on the fly should never be a bottleneck.
    const bg = webgpu_program.makeBindGroup(
        this.device, bindGroupLayout, inputs.map(t => this.tensorToBinding(t)),
        this.tensorToBinding(output), uniforms);

    const encoder = this.device.createCommandEncoder({});
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatch(
        program.dispatch[0], program.dispatch[1], program.dispatch[2]);
    pass.endPass();
    this.commandQueue.push(encoder);

    if (ENV.get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED')) {
      this.submitQueue();
    }
    return output as {} as K;
  }

  private makeUniforms(data: Uint32Array|
                       Int32Array): webgpu_program.BindingInfo {
    const dimensionsBuffer = this.createBuffer(
        data.byteLength, GPUBufferUsage.TRANSFER_DST | GPUBufferUsage.UNIFORM);
    dimensionsBuffer.setSubData(0, data);

    return {
      resource: {offset: 0, size: data.byteLength, buffer: dimensionsBuffer}
    };
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    const program = new PadProgram(x.shape, paddings, constantValue);
    return this.compileAndRun(program, [x]);
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new MaxPoolProgram(convInfo);

    const output =
        this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;

    const dimensionsData = new Int32Array([
      ...convInfo.inShape, ...convInfo.outShape,        // inShape / outShape.
      convInfo.padInfo.left, convInfo.padInfo.top,      // Padding.
      convInfo.strideWidth, convInfo.strideHeight,      // Stride.
      convInfo.dilationWidth, convInfo.dilationHeight,  // Dilation.
      convInfo.inWidth, convInfo.inHeight,              // Conv dims.
      convInfo.effectiveFilterWidth,
      convInfo.effectiveFilterHeight  // Filter dims.
    ]);
    const dimensions = this.makeUniforms(dimensionsData);

    const result = this.compileAndRun(program, [x], output, dimensions);
    this.destroyBuffer(dimensionsData.byteLength, dimensions.resource.buffer);

    return result as Tensor4D;
  }

  add(a: Tensor, b: Tensor): Tensor {
    const output = Tensor.make(a.shape, {}, a.dtype, this);
    const program = new BinaryOpProgram(binary_op.ADD, output.shape);

    return this.compileAndRun(program, [a, b], output) as Tensor;
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const output = Tensor.make(a.shape, {}, a.dtype, this);
    const program = new BinaryOpProgram(binary_op.MUL, output.shape);

    return this.compileAndRun(program, [a, b], output) as Tensor;
  }

  relu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(unary_op.RELU, x.shape);
    return this.compileAndRun(program, [x]) as T;
  }

  reshape<R extends Rank>(x: Tensor, shape: ShapeMap[R]): Tensor<R> {
    return Tensor.make(shape, {dataId: x.dataId}, x.dtype);
  }

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D {
    const outerShapeA = transposeA ? a.shape[2] : a.shape[1];
    const outerShapeB = transposeB ? b.shape[1] : b.shape[2];
    const sharedDim = transposeA ? a.shape[1] : a.shape[2];
    const [batch, , ] = a.shape;

    const output =
        Tensor.make([batch, outerShapeA, outerShapeB], {}, a.dtype, this) as
        Tensor3D;

    let program: MatMulProgram|MatMulPackedProgram;
    // TODO: We should eventually use the blocked version, but keeping around
    // the old version while we try to understand conditions under which blocked
    // is faster.
    if (ENV.get('WEBGPU_MATMUL_WORK_PER_THREAD') === 0) {
      program = new MatMulProgram(output.shape);
    } else {
      program = new MatMulPackedProgram(
          output.shape, ENV.get('WEBGPU_MATMUL_WORK_PER_THREAD') as number);
    }

    const dimensionsData =
        new Uint32Array([outerShapeA, sharedDim, outerShapeB, batch]);
    const dimensions = this.makeUniforms(dimensionsData);

    const result =
        this.compileAndRun(program, [a, b], output, dimensions) as Tensor3D;

    this.destroyBuffer(dimensionsData.byteLength, dimensions.resource.buffer);

    return result;
  }

  dispose() {
    // Backend disposal logic.
  }
}
