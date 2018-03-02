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
import {DataType, Rank} from 'deeplearn/dist/types';

import {Context, TensorHandle, TFEOpAttr, TFJSBinding} from './tfjs_binding';

export class NodeJSKernelBackend implements KernelBackend {
  // TODO(kreeger): Drop when 0.5.1 deeplearn is released.
  slice1D(x: Tensor1D, begin: number, size: number): Tensor1D {
    return this.slice(x, [begin], [size]);
  }
  slice2D(x: Tensor2D, begin: [number, number], size: [number, number]):
      Tensor2D {
    throw new Error('Method not implemented.');
  }
  slice3D(x: Tensor3D, begin: [number, number, number], size: [
    number, number, number
  ]): Tensor3D {
    throw new Error('Method not implemented.');
  }
  slice4D(x: Tensor4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Tensor4D {
    throw new Error('Method not implemented.');
  }
  reverse4D(a: Tensor4D, axis: number[]): Tensor4D {
    throw new Error('Method not implemented.');
  }
  pad1D(x: Tensor1D, paddings: [number, number], constantValue: number):
      Tensor1D {
    throw new Error('Method not implemented.');
  }
  pad2D(
      x: Tensor2D, paddings: [[number, number], [number, number]],
      constantValue: number): Tensor2D {
    return this.pad(x, paddings, constantValue);
  }
  // END DROP

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
        throw new Error('Unknown dtype enum `${handle.dtype}`');
    }
    return Tensor.make(handle.shape, {dataId: newId}, dtype);
  }

  private createTypeOpAttr(attrName: string, tensor: Tensor): TFEOpAttr {
    return {
      name: attrName,
      type: this.binding.TF_ATTR_TYPE,
      value: this.getTFDType(tensor.dtype)
    };
  }

  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D {
    // TODO - set attr type.
    const opAttrs = [
      {name: 'transpose_a', type: this.binding.TF_ATTR_BOOL, value: transposeA},
      {name: 'transpose_b', type: this.binding.TF_ATTR_BOOL, value: transposeB},
      this.createTypeOpAttr('T', a)
    ];
    const output = new this.binding.TensorHandle();
    this.binding.execute(
        this.context, 'MatMul', opAttrs,
        [this.handleMap.get(a.dataId), this.handleMap.get(b.dataId)], output);
    return this.createOutputTensor(output) as Tensor2D;
  }

  slice<T extends Tensor<Rank>>(x: T, begin: number[], size: number[]): T {
    // TODO - set attr type.
    const opAttrs = [
      {
        name: 'T',
        type: this.binding.TF_ATTR_TYPE,
        value: this.binding.TF_FLOAT
      },
      {
        name: 'Index',
        type: this.binding.TF_ATTR_TYPE,
        value: this.binding.TF_INT32
      }
    ];
    const output = new this.binding.TensorHandle();

    // Bind tensor values
    const beginTensor = tensor1d(begin, 'int32');
    const sizeTensor = tensor1d(size, 'int32');

    this.binding.execute(
        this.context, 'Slice', opAttrs,
        [
          this.handleMap.get(x.dataId), this.handleMap.get(beginTensor.dataId),
          this.handleMap.get(sizeTensor.dataId)
        ],
        output);
    return this.createOutputTensor(output) as T;
  }
  reverse<T extends Tensor<Rank>>(a: T, axis: number[]): T {
    throw new Error('Method not implemented.');
  }
  concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    throw new Error('Method not implemented.');
  }
  neg<T extends Tensor<Rank>>(a: T): T {
    throw new Error('Method not implemented.');
  }
  add(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  subtract(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  multiply(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  divide(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  sum(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  argMin(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  argMax(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  equal(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  notEqual(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  less(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  lessEqual(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  greater(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  greaterEqual(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  logicalNot<T extends Tensor<Rank>>(a: T): T {
    throw new Error('Method not implemented.');
  }
  logicalAnd(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  logicalOr(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
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
    throw new Error('Method not implemented.');
  }
  minimum(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  max(x: Tensor<Rank>, axes: number[]): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  maximum(a: Tensor<Rank>, b: Tensor<Rank>): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
  ceil<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  floor<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  pow<T extends Tensor<Rank>>(a: T, b: Tensor<Rank>): T {
    throw new Error('Method not implemented.');
  }
  exp<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  log<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  sqrt<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  square<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  relu<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  elu<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  eluDer<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  selu<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
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
    throw new Error('Method not implemented.');
  }
  sigmoid<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  sin<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  cos<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  tan<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  asin<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  acos<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  atan<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  sinh<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  cosh<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
  }
  tanh<T extends Tensor<Rank>>(x: T): T {
    throw new Error('Method not implemented.');
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
  tile<T extends Tensor<Rank>>(x: T, reps: number[]): T {
    throw new Error('Method not implemented.');
  }
  pad<T extends Tensor<Rank>>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    const opAttrs = [
      this.createTypeOpAttr('T', x), {
        name: 'Tpaddings',
        type: this.binding.TF_ATTR_TYPE,
        value: this.binding.TF_INT32
      }
    ];

    // Bind tensor values
    const paddingsTensor = tensor2d(paddings, [2, 2], 'int32');
    const constantTensor = scalar(constantValue, x.dtype);

    const output = new this.binding.TensorHandle();
    this.binding.execute(
        this.context, 'PadV2', opAttrs,
        [
          this.handleMap.get(x.dataId),
          this.handleMap.get(paddingsTensor.dataId),
          this.handleMap.get(constantTensor.dataId)
        ],
        output);
    return this.createOutputTensor(output) as T;
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
    throw new Error('Method not implemented.');
  }
  dispose(): void {
    throw new Error('Method not implemented.');
  }
  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    return this.handleMap.get(dataId).dataSync();
  }
  readSync(dataId: object): Float32Array|Int32Array|Uint8Array {
    return this.handleMap.get(dataId).dataSync();
  }
  disposeData(dataId: object): void {
    // throw new Error('Method not implemented.');
  }
  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    this.handleMap.get(dataId).bindBuffer(values);
  }
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    throw new Error('Method not implemented.');
  }

  register(dataId: object, shape: number[], dtype: 'float32'|'int32'|'bool'):
      void {
    if (this.handleMap.has(dataId)) {
      return;
    }
    this.handleMap.set(
        dataId, new this.binding.TensorHandle(shape, this.getTFDType(dtype)));
  }

  memory(): {unreliable: boolean;} {
    throw new Error('Method not implemented.');
  }
  time(f: () => void): Promise<BackendTimingInfo> {
    throw new Error('Method not implemented.');
  }
}
