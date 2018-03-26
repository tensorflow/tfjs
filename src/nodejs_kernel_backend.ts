/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {scalar, tensor1d, tensor2d} from 'deeplearn';
import {BackendTimingInfo, KernelBackend} from 'deeplearn/dist/kernels/backend';
// tslint:disable-next-line:max-line-length
import {DataId, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from 'deeplearn/dist/tensor';
import {DataType, Rank, ShapeMap, upcastType} from 'deeplearn/dist/types';

import {Context, TensorHandle, TFEOpAttr, TFJSBinding} from './tfjs_binding';

type TensorInfo = {
  shape: number[],
  dtype: number
};

export class NodeJSKernelBackend implements KernelBackend {
  private shapeMap = new WeakMap<DataId, TensorInfo>();
  private handleMap = new WeakMap<DataId, TensorHandle>();
  private context: Context;

  private binding: TFJSBinding;

  constructor(binding: TFJSBinding) {
    this.binding = binding;
    this.context = new this.binding.Context();
  }

  // Returns the TF dtype for a given DataType.
  private getTFDType(dataType: DataType): number {
    switch (dataType) {
      case 'float32':
        return this.binding.TF_FLOAT;
      case 'int32':
        return this.binding.TF_INT32;
      case 'bool':
        return this.binding.TF_BOOL;
      default:
        throw new Error('Unknown dtype `${dtype}`');
    }
  }

  // Creates a new Tensor and maps the dataId to the passed in handle.
  private createOutputTensor(handle: TensorHandle): Tensor {
    const newId = {};
    this.handleMap.set(newId, handle);

    let dtype: DataType;
    switch (handle.dtype) {
      case this.binding.TF_FLOAT:
        dtype = 'float32';
        break;
      case this.binding.TF_INT32:
        dtype = 'int32';
        break;
      case this.binding.TF_BOOL:
        dtype = 'bool';
        break;
      default:
        throw new Error(`Unknown dtype enum ${handle.dtype}`);
    }
    return Tensor.make(handle.shape, {dataId: newId}, dtype);
  }

  private getInputTensors(tensors: Tensor[]): TensorHandle[] {
    const inputs: TensorHandle[] = [];
    for (let i = 0; i < tensors.length; i++) {
      inputs.push(this.handleMap.get(tensors[i].dataId));
    }
    return inputs;
  }

  private createReductionOpAttrs(tensor: Tensor): TFEOpAttr[] {
    return [
      {name: 'keep_dims', type: this.binding.TF_ATTR_BOOL, value: false},
      this.createTypeOpAttr('T', tensor.dtype),
      this.createTypeOpAttr('Tidx', 'int32')
    ];
  }

  private createTypeOpAttr(attrName: string, dtype: DataType): TFEOpAttr {
    return {
      name: attrName,
      type: this.binding.TF_ATTR_TYPE,
      value: this.getTFDType(dtype)
    };
  }

  private executeSingleInput(name: string, input: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', input.dtype)];
    return this.execute(name, opAttrs, [input]);
  }

  private execute(name: string, opAttrs: TFEOpAttr[], inputs: Tensor[]):
      Tensor {
    const output = new this.binding.TensorHandle();
    this.binding.execute(
        this.context, name, opAttrs, this.getInputTensors(inputs), output);
    return this.createOutputTensor(output);
  }

  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [
      {name: 'transpose_a', type: this.binding.TF_ATTR_BOOL, value: transposeA},
      {name: 'transpose_b', type: this.binding.TF_ATTR_BOOL, value: transposeB},
      this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))
    ];
    return this.execute('MatMul', opAttrs, [a, b]) as Tensor2D;
  }

  slice<T extends Tensor<Rank>>(x: T, begin: number[], size: number[]): T {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Index', 'int32')
    ];

    // Bind tensor values
    const beginTensor = tensor1d(begin, 'int32');
    const sizeTensor = tensor1d(size, 'int32');

    return this.execute('Slice', opAttrs, [x, beginTensor, sizeTensor]) as T;
  }

  reverse<T extends Tensor<Rank>>(a: T, axis: number[]): T {
    const opAttrs = [
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('T', a.dtype)
    ];
    const axisTensor = tensor1d(axis, 'int32');
    return this.execute('ReverseV2', opAttrs, [a, axisTensor]) as T;
  }

  concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    const opAttrs = [
      {name: 'N', type: this.binding.TF_ATTR_INT, value: 2},
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('T', a.dtype)
    ];
    const axisTensor = scalar(0, 'int32');
    return this.execute('ConcatV2', opAttrs, [a, b, axisTensor]) as Tensor2D;
  }

  neg<T extends Tensor<Rank>>(a: T): T {
    return this.executeSingleInput('Neg', a) as T;
  }

  add(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Add', opAttrs, [a, b]);
  }

  subtract(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Sub', opAttrs, [a, b]);
  }

  multiply(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Mul', opAttrs, [a, b]);
  }

  divide(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Div', opAttrs, [a, b]);
  }

  sum(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    const axisTensor = tensor1d(axes, 'int32');
    return this.execute('Sum', this.createReductionOpAttrs(x), [x, axisTensor]);
  }

  argMin(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  argMax(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }

  equal(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Equal', opAttrs, [a, b]);
  }

  notEqual(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('NotEqual', opAttrs, [a, b]);
  }

  less(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Less', opAttrs, [a, b]);
  }

  lessEqual(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('LessEqual', opAttrs, [a, b]);
  }

  greater(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Greater', opAttrs, [a, b]);
  }

  greaterEqual(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('GreaterEqual', opAttrs, [a, b]);
  }

  logicalNot<T extends Tensor<Rank>>(a: T): T {
    return this.execute('LogicalNot', [], [a]) as T;
  }

  logicalAnd(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    return this.execute('LogicalAnd', [], [a, b]);
  }

  logicalOr(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    return this.execute('LogicalOr', [], [a, b]);
  }

  logicalXor(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }

  where(
      condition: Tensor<Rank>, a: Tensor<Rank>, b: Tensor<Rank>,
      dtype: 'float32'|'int32'|'bool'): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  topKValues<T extends Tensor<Rank>>(x: T, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }
  topKIndices(x: Tensor<Rank>, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }

  min(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    const axesTensor = tensor1d(axes, 'int32');
    return this.execute('Min', this.createReductionOpAttrs(x), [x, axesTensor]);
  }

  minimum(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Minimum', opAttrs, [a, b]);
  }

  max(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    const axesTensor = tensor1d(axes, 'int32');
    return this.execute('Max', this.createReductionOpAttrs(x), [x, axesTensor]);
  }

  maximum(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Maximum', opAttrs, [a, b]);
  }

  ceil<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Ceil', x) as T;
  }

  floor<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Floor', x) as T;
  }

  pow<T extends Tensor<Rank>>(a: T, b: Tensor<Rank>): T {
    // TODO(kreeger): Tensors must be up-typed before Op execution:
    // https://github.com/tensorflow/tfjs-node/issues/32
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.execute('Pow', opAttrs, [a, b]) as T;
  }

  exp<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Exp', x) as T;
  }

  log<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Log', x) as T;
  }

  log1p<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }

  sqrt<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Sqrt', x) as T;
  }

  square<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Square', x) as T;
  }

  relu<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Relu', x) as T;
  }

  elu<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Elu', x) as T;
  }

  eluDer<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }

  selu<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Selu', x) as T;
  }

  leakyRelu<T extends Tensor<Rank>>(x: T, alpha: number): T {
    throw new Error('Method not implemented.');
  }

  prelu<T extends Tensor<Rank>>(x: T, alpha: T): T {
    throw new Error('Method not implemented.');
  }

  preluDer<T extends Tensor<Rank>>(x: T, alpha: T): T {
    throw new Error('Method not implemented.');
  }

  int<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }

  clip<T extends Tensor<Rank>>(x: T, min: number, max: number): T {
    throw new Error('Method not implemented.');
  }

  abs<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Abs', x) as T;
  }

  sigmoid<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Sigmoid', x) as T;
  }

  sin<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Sin', x) as T;
  }

  cos<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Cos', x) as T;
  }

  tan<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Tan', x) as T;
  }

  asin<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Asin', x) as T;
  }

  acos<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Acos', x) as T;
  }

  atan<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Atan', x) as T;
  }

  sinh<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Sinh', x) as T;
  }

  cosh<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Cosh', x) as T;
  }

  tanh<T extends Tensor<Rank>>(x: T): T {
    return this.executeSingleInput('Tanh', x) as T;
  }

  step<T extends Tensor<Rank>>(x: T, alpha: number): T {
    throw new Error('Method not implemented.');
  }
  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }
  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }
  conv2dDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }
  depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }
  maxPool(x: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }
  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }
  minPool(x: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }
  avgPool(x: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }
  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: {
    batchSize: number; inHeight: number; inWidth: number; inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    filterHeight: number;
    filterWidth: number;
    padInfo: {top: number; left: number; right: number; bottom: number;};
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
  }): Tensor4D {
    throw new Error('Method not implemented.');
  }

  reshape<T extends Tensor<Rank>, R extends Rank>(x: T, shape: ShapeMap[R]):
      Tensor<R> {
    const shapeTensor = tensor1d(shape, 'int32');

    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tshape', shapeTensor.dtype)
    ];
    return this.execute('Reshape', opAttrs, [x, shapeTensor]) as Tensor<R>;
  }

  cast<T extends Tensor<Rank>>(x: T, dtype: DataType): T {
    const opAttrs = [
      this.createTypeOpAttr('SrcT', x.dtype),
      this.createTypeOpAttr('DstT', dtype)
    ];
    return this.execute('Cast', opAttrs, [x]) as T;
  }

  tile<T extends Tensor<Rank>>(x: T, reps: number[]): T {
    throw new Error('Method not implemented.');
  }
  pad<T extends Tensor<Rank>>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    // Bind tensor values
    const paddingsTensor = tensor2d(paddings, [2, 2], 'int32');
    const constantTensor = scalar(constantValue, x.dtype);

    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tpaddings', paddingsTensor.dtype)
    ];

    return this.execute(
               'PadV2', opAttrs, [x, paddingsTensor, constantTensor]) as T;
  }
  transpose<T extends Tensor<Rank>>(x: T, perm: number[]): T {
    throw new Error('Method not implemented.');
  }
  gather<T extends Tensor<Rank>>(x: T, indices: Tensor1D, axis: number): T {
    throw new Error('Method not implemented.');
  }
  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    throw new Error('Method not implemented.');
  }
  batchNormalization4D(
      x: Tensor4D, mean: Tensor1D|Tensor4D, variance: Tensor1D|Tensor4D,
      varianceEpsilon: number, scale: Tensor1D|Tensor4D,
      offset: Tensor1D|Tensor4D): Tensor4D {
    throw new Error('Method not implemented.');
  }
  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number, beta: number,
      normRegion: 'acrossChannels'|'withinChannel'): Tensor4D {
    throw new Error('Method not implemented.');
  }
  multinomial(probabilities: Tensor2D, numSamples: number, seed: number):
      Tensor2D {
    throw new Error('Method not implemented.');
  }
  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    const depthTensor = scalar(depth, 'int32');
    const onValueTensor = scalar(onValue, 'int32');
    const offValueTensor = scalar(offValue, 'int32');

    const opAttrs = [
      {name: 'axis', type: this.binding.TF_ATTR_INT, value: -1},
      this.createTypeOpAttr('T', indices.dtype),
      this.createTypeOpAttr('TI', indices.dtype)
    ];

    return this.execute('OneHot', opAttrs, [
      indices, depthTensor, onValueTensor, offValueTensor
    ]) as Tensor2D;
  }
  dispose(): void {
    throw new Error('Method not implemented.');
  }
  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    return this.handleMap.get(dataId).dataSync(this.context);
  }
  readSync(dataId: object): Float32Array|Int32Array|Uint8Array {
    return this.handleMap.get(dataId).dataSync(this.context);
  }
  disposeData(dataId: object): void {
    // throw new Error('Method not implemented.');
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.shapeMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    if (!this.handleMap.has(dataId)) {
      this.handleMap.set(dataId, new this.binding.TensorHandle());
    }

    const info = this.shapeMap.get(dataId);
    this.handleMap.get(dataId).copyBuffer(info.shape, info.dtype, values);
  }

  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    throw new Error('Method not implemented.');
  }

  register(dataId: object, tShape: number[], dtype: 'float32'|'int32'|'bool'):
      void {
    if (this.shapeMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} is already registered!`);
    }
    this.shapeMap.set(dataId, {shape: tShape, dtype: this.getTFDType(dtype)});
  }

  memory(): {unreliable: boolean;} {
    throw new Error('Method not implemented.');
  }
  time(f: () => void): Promise<BackendTimingInfo> {
    throw new Error('Method not implemented.');
  }
}
