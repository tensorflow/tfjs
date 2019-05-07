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

import {DataMover, DataType, ENV, KernelBackend, Rank, ShapeMap, Tensor, Tensor2D, Tensor3D, Tensor4D, util} from '@tensorflow/tfjs-core';
import * as backend_util from '@tensorflow/tfjs-core/dist/backends/backend_util';
import {computeOutShape} from '@tensorflow/tfjs-core/dist/ops/concat_util';
import {Conv2DInfo} from '@tensorflow/tfjs-core/dist/ops/conv_util';
import {upcastType} from '@tensorflow/tfjs-core/dist/types';
import {assert} from '@tensorflow/tfjs-core/dist/util';
import * as shaderc from '@webgpu/shaderc';

import {ArgMinMaxProgram} from './kernels/argminmax_webgpu';
import * as binary_op from './kernels/binary_op_webgpu';
import {BinaryOpProgram} from './kernels/binary_op_webgpu';
import {ConcatProgram} from './kernels/concat_webgpu';
import {Conv2DMMProgram} from './kernels/conv2d_mm_webgpu';
import {Conv2DNaiveProgram} from './kernels/conv2d_naive_webgpu';
import {MatMulPackedProgram} from './kernels/matmul_packed_webgpu';
import {MatMulProgram} from './kernels/matmul_webgpu';
import {MaxPoolProgram} from './kernels/maxpool_webgpu';
import {PadProgram} from './kernels/pad_webgpu';
import {ResizeBilinearProgram} from './kernels/resize_bilinear_webgpu';
import {TransposeProgram} from './kernels/transpose_webgpu';
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
      programUniforms?: number[]): K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }
    let dimUniforms: number[] = [];
    const bufferShapes = inputs.concat(output).map(d => d.shape);
    let currentOffset = 0;
    bufferShapes.forEach((d, i) => {
      // Complete std140 layout rules are documented here:
      // tslint:disable-next-line:max-line-length
      // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
      let baseAlignment = 0;
      switch (d.length) {
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
          assert(false, () => `Unsupported ${d.length}D shape`);
      }

      const padding = Math.ceil(currentOffset / baseAlignment) * baseAlignment -
          currentOffset;
      for (let p = 0; p < padding; ++p) {
        dimUniforms.push(0);
      }
      dimUniforms.push(...d);
      currentOffset += d.length + padding;
    });

    // TODO: handle padding of program-specific uniforms
    if (programUniforms) {
      dimUniforms = dimUniforms.concat(programUniforms);
    }

    const uniformData = new Int32Array(dimUniforms);
    const uniforms = this.makeUniforms(uniformData);

    const key =
        webgpu_program.makeShaderKey(program, bufferShapes.map(d => d.length));
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
    this.destroyBuffer(uniformData.byteLength, uniforms.resource.buffer);
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
    const output = this.makeOutputArray(program.outputShape, x.dtype);
    return this.compileAndRun(program, [x], output);
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new MaxPoolProgram(convInfo);

    const output =
        this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;

    const dimensions = [
      convInfo.padInfo.left, convInfo.padInfo.top,      // Padding.
      convInfo.strideWidth, convInfo.strideHeight,      // Stride.
      convInfo.dilationWidth, convInfo.dilationHeight,  // Dilation.
      convInfo.inWidth, convInfo.inHeight,              // Conv dims.
      convInfo.effectiveFilterWidth,
      convInfo.effectiveFilterHeight  // Filter dims.
    ];

    return this.compileAndRun(program, [x], output, dimensions);
  }

  private binaryOp(a: Tensor, b: Tensor, op: string) {
    const dtype = upcastType(a.dtype, b.dtype);
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = Tensor.make(program.outputShape, {}, dtype) as Tensor;

    const result = this.compileAndRun(program, [a, b], output) as Tensor;
    return result;
  }

  add(a: Tensor, b: Tensor): Tensor {
    return this.binaryOp(a, b, binary_op.ADD);
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const output =
        Tensor.make(convInfo.outShape, {}, x.dtype, this) as Tensor4D;
    let program: Conv2DMMProgram|Conv2DNaiveProgram;

    const workPerThread = ENV.get('WEBGPU_CONV2D_WORK_PER_THREAD') as number;
    if (workPerThread === -1) {
      // TODO(kainino0x): This may be obsolete, but is kept for reference.
      program = new Conv2DNaiveProgram(convInfo);
    } else {
      program = new Conv2DMMProgram(convInfo, workPerThread);
    }

    const pad = convInfo.padInfo.type === 'VALID' ?
        [0, 0] :
        convInfo.padInfo.type === 'SAME' ?
        [
          -Math.floor((convInfo.filterShape[0] - 1) / 2),
          -Math.floor((convInfo.filterShape[1] - 1) / 2)
        ] :
        [convInfo.padInfo.top, convInfo.padInfo.left];

    const dimensions = [
      convInfo.filterHeight, convInfo.filterWidth, ...pad,
      convInfo.strideHeight, convInfo.strideWidth
    ];

    return this.compileAndRun(program, [x, filter], output, dimensions) as
        Tensor4D;
  }

  private argMinMaxReduce(x: Tensor, axis: number, reduceType: 'min'|'max'):
      Tensor {
    const program = new ArgMinMaxProgram(x.shape, axis, reduceType);
    const output = this.makeOutputArray(program.outputShape, 'int32') as Tensor;
    return this.compileAndRun(program, [x], output, [axis]) as Tensor;
  }

  argMin(x: Tensor, axis: number): Tensor {
    return this.argMinMaxReduce(x, axis, 'min');
  }

  argMax(x: Tensor, axis: number): Tensor {
    return this.argMinMaxReduce(x, axis, 'max');
  }
  
  concat(tensors: Tensor[], axis: number): Tensor {
    if (tensors.length === 1) {
      return tensors[0];
    }
    // Is there a maximum number of buffers that can be uploaded to a WebGPU
    // program?
    // if (tensors.length > MAX_SSBOS_FOR_WEBGPU_PROGRAM) {
    //   const midIndex = Math.floor(tensors.length / 2);
    //   const leftSide = this.concat(tensors.slice(0, midIndex), axis);
    //   const rightSide = this.concat(tensors.slice(midIndex), axis);
    //   return this.concat([leftSide, rightSide], axis);
    // }
    const outShape = computeOutShape(tensors.map(t => t.shape), axis);
    const tensors2D = tensors.map(t => t.reshape([
      util.sizeFromShape(t.shape.slice(0, axis)),
      util.sizeFromShape(t.shape.slice(axis))
    ]) as Tensor2D);
    const program = new ConcatProgram(tensors2D.map(t => t.shape));
    const res = this.compileAndRun(program, tensors2D) as Tensor;
    return res.reshape(outShape);
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    return this.binaryOp(a, b, binary_op.MUL);
  }

  floorDiv(a: Tensor, b: Tensor): Tensor {
    return this.binaryOp(a, b, binary_op.INT_DIV);
  }

  sigmoid<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGMOID);
    return this.compileAndRun(program, [x]) as T;
  }

  relu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RELU);
    return this.compileAndRun(program, [x]) as T;
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const program =
        new ResizeBilinearProgram(x.shape, newHeight, newWidth, alignCorners);

    const output =
        this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;

    return this.compileAndRun(program, [x], output) as Tensor4D;
  }

  reshape<R extends Rank>(x: Tensor, shape: ShapeMap[R]): Tensor<R> {
    return Tensor.make(shape, {dataId: x.dataId}, x.dtype);
  }

  cast<T extends Tensor>(x: T, dtype: DataType): T {
    return backend_util.castTensor(x, dtype, this);
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    const program = new TransposeProgram(x.shape, perm);
    return this.compileAndRun(program, [x]);
  }

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D {
    // TODO: Support transposed inputs.
    // const outerShapeA = transposeA ? a.shape[2] : a.shape[1];
    // const outerShapeB = transposeB ? b.shape[1] : b.shape[2];
    const outerShapeA = a.shape[1];
    const outerShapeB = b.shape[2];
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

    const result = this.compileAndRun(program, [a, b], output) as Tensor3D;

    return result;
  }

  dispose() {
    // Backend disposal logic.
  }
}
