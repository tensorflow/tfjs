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

import {backend_util, DataMover, DataType, ENV, KernelBackend, Rank, ShapeMap, Tensor, Tensor2D, Tensor3D, Tensor4D, util} from '@tensorflow/tfjs-core';
import * as shaderc from '@webgpu/shaderc';

import {BufferManager} from './buffer_manager';
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

// TODO: Delete this and import from core once new release is published.
type MemoryInfo = {
  numTensors: number; numDataBuffers: number; numBytes: number;
  unreliable?: boolean; reasons: string[];
};

export interface WebGPUMemoryInfo extends MemoryInfo {
  numBytesInGPU: number;
  unreliable: boolean;
}

type BufferInfo = {
  byteSize: number,
  usage: GPUBufferUsage,
  buffer: GPUBuffer
};

type TensorInfo = {
  values: Float32Array|Int32Array|Uint8Array,
  id: number,
  dtype: DataType,
  bufferInfo: BufferInfo
};

interface DataId {}

const DEFAULT_GPUBUFFER_USAGE =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class WebGPUBackend extends KernelBackend {
  device: GPUDevice;
  queue: GPUQueue;
  shaderc: shaderc.Shaderc;
  compiler: shaderc.Compiler;
  compileOpts: shaderc.CompileOptions;
  commandQueue: GPUCommandEncoder[];

  private commandQueueOwnedIds = new WeakSet<DataId>();
  private binaryCache: {[key: string]: WebGPUBinary};
  private fromPixels2DContext: CanvasRenderingContext2D;
  private bufferManager: BufferManager;
  private tensorMap = new WeakMap<DataId, TensorInfo>();
  private disposalQueue: BufferInfo[] = [];

  private disposed = false;

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

    this.bufferManager = new BufferManager(this.device);
  }

  floatPrecision(): 32 {
    return 32;
  }

  setDataMover(dataMover: DataMover): void {
    // TODO: tfjs team to implement this. Call GPUBuffer.destroy()
  }

  flushDisposalQueue() {
    this.disposalQueue.forEach(d => {
      this.releaseBuffer(d.buffer, d.byteSize, d.usage);
    });

    this.disposalQueue = [];
  }

  disposeData(dataId: DataId): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    if (this.commandQueueOwnedIds.has(dataId)) {
      this.disposalQueue.push(info.bufferInfo);
    } else {
      this.releaseBuffer(
          info.bufferInfo.buffer, info.bufferInfo.byteSize,
          info.bufferInfo.usage);
    }

    this.tensorMap.delete(dataId);
  }

  memory(): WebGPUMemoryInfo {
    return {
      numBytesInGPU: this.bufferManager.numBytesUsed,
      unreliable: false
    } as WebGPUMemoryInfo;
  }

  getBufferManager(): BufferManager {
    return this.bufferManager;
  }

  private acquireBuffer(
      byteSize: number, usage: GPUBufferUsage = DEFAULT_GPUBUFFER_USAGE) {
    return this.bufferManager.acquireBuffer(byteSize, usage);
  }

  private releaseBuffer(
      buffer: GPUBuffer, byteSize: number, usage: GPUBufferUsage) {
    this.bufferManager.releaseBuffer(buffer, byteSize, usage);
  }

  register(dataId: object, shape: number[], dtype: DataType): void {
    if (!this.tensorMap.has(dataId)) {
      const byteSize = util.sizeFromShape(shape) * util.bytesPerElement(dtype);
      const buffer = this.acquireBuffer(byteSize);
      this.tensorMap.set(dataId, {
        values: null,
        id: -1,
        dtype,
        bufferInfo: {byteSize, usage: DEFAULT_GPUBUFFER_USAGE, buffer}
      });
    }
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    info.values = values;
    info.bufferInfo.buffer.setSubData(0, values);
    this.tensorMap.set(dataId, info);
  }

  private submitQueue() {
    this.queue.submit(this.commandQueue.map(enc => enc.finish()));
    this.commandQueue = [];

    this.commandQueueOwnedIds = new WeakSet<DataId>();

    this.flushDisposalQueue();
  }

  private async getBufferData(info: TensorInfo): Promise<ArrayBuffer> {
    const staging = this.acquireBuffer(
        info.bufferInfo.byteSize,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const encoder = this.device.createCommandEncoder({});
    encoder.copyBufferToBuffer(
        info.bufferInfo.buffer, 0, staging, 0, info.bufferInfo.byteSize);
    this.commandQueue.push(encoder);
    this.submitQueue();

    const mapped: ArrayBuffer = await staging.mapReadAsync();
    const values = mapped.slice(0);

    staging.unmap();
    this.releaseBuffer(
        staging, info.bufferInfo.byteSize,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    return values;
  }

  private convertAndCacheOnCPU(dataId: DataId, data: backend_util.TypedArray):
      backend_util.TypedArray {
    const info = this.tensorMap.get(dataId);
    info.values = data;
    return info.values as backend_util.TypedArray;
  }

  // TODO: Remove once this is fixed:
  // https://github.com/tensorflow/tfjs/issues/1595
  readSync(dataId: object): Float32Array|Int32Array|Uint8Array {
    const texData = this.tensorMap.get(dataId);
    const {values} = texData;

    if (values == null) {
      throw new Error(
          'WebGPU readSync is only available for CPU-resident tensors.');
    }

    return values;
  }

  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    const data = await this.getBufferData(info);

    const dataAsTypedArray =
        info.dtype === 'int32' ? new Int32Array(data) : new Float32Array(data);
    this.convertAndCacheOnCPU(dataId, dataAsTypedArray);

    return dataAsTypedArray;
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
        buffer: tensorData.bufferInfo.buffer
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
      // Uniforms.
      if (d.length === 0) {
        d = [1];
      }
      // Complete std140 layout rules are documented here:
      // tslint:disable-next-line:max-line-length
      // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
      let baseAlignment: number;
      switch (d.length) {
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
          util.assert(false, () => `Unsupported ${d.length}D shape`);
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
    const inputsData =
        inputs.map((input: Tensor, i: number) => ({
                     // Returning dtype from tensorMap because it reflects dtype
                     // of underlying buffer, rather than abstract dtype.
                     dtype: this.tensorMap.get(input.dataId).dtype,
                     shape: input.shape,
                     name: program.variableNames[i]
                   }));
    const {bindGroupLayout, pipeline} = this.getAndSavePipeline(key, () => {
      return webgpu_program.compileProgram(
          this.compiler, this.shaderc.shader_kind.compute, this.compileOpts,
          this.device, program, inputsData, output, uniforms);
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

    inputs.forEach(input => {
      this.commandQueueOwnedIds.add(input.dataId);
    });
    this.commandQueueOwnedIds.add(output.dataId);

    if (ENV.get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED')) {
      this.submitQueue();
    }
    this.releaseBuffer(
        uniforms.resource.buffer, uniformData.byteLength,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
    return output as {} as K;
  }

  private makeUniforms(data: Uint32Array|
                       Int32Array): webgpu_program.BindingInfo {
    const dimensionsBuffer = this.acquireBuffer(
        data.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
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

  maxPool(x: Tensor4D, convInfo: backend_util.Conv2DInfo): Tensor4D {
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
    const dtype = backend_util.upcastType(a.dtype, b.dtype);
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = Tensor.make(program.outputShape, {}, dtype) as Tensor;

    const result = this.compileAndRun(program, [a, b], output) as Tensor;
    return result;
  }

  add(a: Tensor, b: Tensor): Tensor {
    return this.binaryOp(a, b, binary_op.ADD);
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    return this.binaryOp(a, b, binary_op.SUB);
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: backend_util.Conv2DInfo):
      Tensor4D {
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
    const outShape =
        backend_util.computeOutShape(tensors.map(t => t.shape), axis);
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

  fromPixels(
      pixels: backend_util.PixelData|ImageData|HTMLImageElement|
      HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() can not be null');
    }

    const outShape = [pixels.height, pixels.width, numChannels];
    let imageData = (pixels as ImageData | backend_util.PixelData).data;

    if (ENV.getBool('IS_BROWSER')) {
      if (!(pixels instanceof HTMLVideoElement) &&
          !(pixels instanceof HTMLImageElement) &&
          !(pixels instanceof HTMLCanvasElement) &&
          !(pixels instanceof ImageData) &&
          !((pixels as backend_util.PixelData).data instanceof Uint8Array)) {
        throw new Error(
            'pixels passed to tf.browser.fromPixels() must be either an ' +
            `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData` +
            ` or {data: Uint32Array, width: number, height: number}, ` +
            `but was ${(pixels as {}).constructor.name}`);
      }
      if (pixels instanceof HTMLVideoElement) {
        if (this.fromPixels2DContext == null) {
          this.fromPixels2DContext =
              document.createElement('canvas').getContext('2d');
          this.fromPixels2DContext.canvas.width = pixels.width;
          this.fromPixels2DContext.canvas.height = pixels.height;
        }
        this.fromPixels2DContext.drawImage(
            pixels, 0, 0, pixels.width, pixels.height);
        pixels = this.fromPixels2DContext.canvas;
      }

      // TODO: Remove this once we figure out how to upload textures directly to
      // WebGPU.
      const imageDataLivesOnGPU = pixels instanceof HTMLVideoElement ||
          pixels instanceof HTMLImageElement ||
          pixels instanceof HTMLCanvasElement;
      if (imageDataLivesOnGPU) {
        imageData = this.fromPixels2DContext
                        .getImageData(0, 0, pixels.width, pixels.height)
                        .data;
      }
    }

    // TODO: Encoding should happen on GPU once we no longer have to download
    // image data to the CPU.
    let pixelArray = imageData;
    if (numChannels != null && numChannels !== 4) {
      pixelArray = new Uint8Array(pixels.width * pixels.height * numChannels);

      for (let i = 0; i < imageData.length; i++) {
        if (i % 4 < numChannels) {
          const pixelIndex = Math.floor(i / 4);
          pixelArray[pixelIndex * numChannels + i % 4] = imageData[i];
        }
      }
    }

    const output = this.makeOutputArray(outShape, 'int32');
    this.write(output.dataId, Int32Array.from(pixelArray));
    return output as Tensor3D;
  }

  dispose() {
    if (this.disposed) {
      return;
    }
    this.bufferManager.dispose();
    this.disposed = true;
  }
}
