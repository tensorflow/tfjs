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

// tslint:disable-next-line:max-line-length
import {BackendTimingInfo, DataType, fill, KernelBackend, ones, Rank, rsqrt, scalar, ShapeMap, Tensor, Tensor1D, tensor1d, Tensor2D, tensor2d, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
import {Conv2DInfo} from '@tensorflow/tfjs-core/dist/ops/conv_util';
import {upcastType} from '@tensorflow/tfjs-core/dist/types';
import {TensorMetadata, TFEOpAttr, TFJSBinding} from './tfjs_binding';

type TensorInfo = {
  shape: number[],
  dtype: number,
  values: Float32Array|Int32Array|Uint8Array,
  id: number
};

interface DataId {}

export class NodeJSKernelBackend implements KernelBackend {
  private binding: TFJSBinding;
  private tensorMap = new WeakMap<DataId, TensorInfo>();

  constructor(binding: TFJSBinding) {
    this.binding = binding;
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

  // Creates a new Tensor and maps the dataId to the passed in ID.
  private createOutputTensor(metadata: TensorMetadata): Tensor {
    const newId = {};

    this.tensorMap.set(newId, {
      shape: metadata.shape,
      dtype: metadata.dtype,
      id: metadata.id,
      values: null
    });

    let dtype: DataType;
    switch (metadata.dtype) {
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
        throw new Error(`Unknown dtype enum ${metadata.dtype}`);
    }
    return Tensor.make(metadata.shape, {dataId: newId}, dtype);
  }

  // Prepares Tensor instances for Op execution.
  private getInputTensorIds(tensors: Tensor[]): number[] {
    const ids: number[] = [];
    for (let i = 0; i < tensors.length; i++) {
      const info = this.tensorMap.get(tensors[i].dataId);
      // TODO - what about ID in this case? Handle in write()??
      if (info.values != null) {
        // Values were delayed to write into the TensorHandle. Do that before Op
        // execution and clear stored values.
        info.id =
            this.binding.createTensor(info.shape, info.dtype, info.values);
        info.values = null;
        this.tensorMap.set(tensors[i].dataId, info);
      }
      ids.push(info.id);
    }
    return ids;
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
    return this.executeSingleOutput(name, opAttrs, [input]);
  }

  private executeSingleOutput(
      name: string, opAttrs: TFEOpAttr[], inputs: Tensor[]): Tensor {
    const outputMetadata = this.binding.executeOp(
        name, opAttrs, this.getInputTensorIds(inputs), 1);
    return this.createOutputTensor(outputMetadata[0]);
  }

  private executeMultipleOutputs(
      name: string, opAttrs: TFEOpAttr[], inputs: Tensor[],
      numOutputs: number): Tensor[] {
    const outputMetadata = this.binding.executeOp(
        name, opAttrs, this.getInputTensorIds(inputs), numOutputs);
    return outputMetadata.map(m => this.createOutputTensor(m));
  }

  dispose(): void {
    throw new Error('Method not implemented.');
  }

  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    return this.readSync(dataId);
  }

  readSync(dataId: object): Float32Array|Int32Array|Uint8Array {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    if (info.values != null) {
      return info.values;
    } else {
      return this.binding.tensorDataSync(info.id);
    }
  }

  disposeData(dataId: object): void {
    const id = this.tensorMap.get(dataId).id;
    if (id != null && id >= 0) {
      this.binding.deleteTensor(id);
    }
    this.tensorMap.delete(dataId);
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    info.values = values;
    this.tensorMap.set(dataId, info);
  }

  register(dataId: object, shape: number[], dtype: DataType): void {
    if (!this.tensorMap.has(dataId)) {
      this.tensorMap.set(
          dataId, {shape, dtype: this.getTFDType(dtype), values: null, id: -1});
    }
  }

  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D {
    const opAttrs = [
      {name: 'transpose_a', type: this.binding.TF_ATTR_BOOL, value: transposeA},
      {name: 'transpose_b', type: this.binding.TF_ATTR_BOOL, value: transposeB},
      this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))
    ];
    return this.executeSingleOutput('MatMul', opAttrs, [a, b]) as Tensor2D;
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Index', 'int32')
    ];

    // Bind tensor values
    const beginTensor = tensor1d(begin, 'int32');
    const sizeTensor = tensor1d(size, 'int32');

    return this.executeSingleOutput(
               'Slice', opAttrs, [x, beginTensor, sizeTensor]) as T;
  }

  reverse<T extends Tensor>(a: T, axis: number[]): T {
    const opAttrs = [
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('T', a.dtype)
    ];
    const axisTensor = tensor1d(axis, 'int32');
    return this.executeSingleOutput('ReverseV2', opAttrs, [a, axisTensor]) as T;
  }

  concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    const opAttrs = [
      {name: 'N', type: this.binding.TF_ATTR_INT, value: 2},
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('T', a.dtype)
    ];
    // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
    const axisTensor = scalar(1, 'int32');
    return this.executeSingleOutput('ConcatV2', opAttrs, [a, b, axisTensor]) as
        Tensor2D;
  }

  neg<T extends Tensor>(a: T): T {
    return this.executeSingleInput('Neg', a) as T;
  }

  add(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Add', opAttrs, [a, b]);
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Sub', opAttrs, [a, b]);
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Mul', opAttrs, [a, b]);
  }

  divide(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Div', opAttrs, [a, b]);
  }

  sum(x: Tensor, axes: number[]): Tensor {
    const axisTensor = tensor1d(axes, 'int32');
    return this.executeSingleOutput(
        'Sum', this.createReductionOpAttrs(x), [x, axisTensor]);
  }

  argMin(x: Tensor, axis: number): Tensor {
    const axisScalar = scalar(axis, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('output_type', 'int32')
    ];
    return this.executeSingleOutput('ArgMin', opAttrs, [x, axisScalar]);
  }

  argMax(x: Tensor, axis: number): Tensor {
    const axisScalar = scalar(axis, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('output_type', 'int32')
    ];
    return this.executeSingleOutput('ArgMax', opAttrs, [x, axisScalar]);
  }

  equal(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Equal', opAttrs, [a, b]);
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('NotEqual', opAttrs, [a, b]);
  }

  less(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Less', opAttrs, [a, b]);
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('LessEqual', opAttrs, [a, b]);
  }

  greater(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Greater', opAttrs, [a, b]);
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('GreaterEqual', opAttrs, [a, b]);
  }

  logicalNot<T extends Tensor>(a: T): T {
    return this.executeSingleOutput('LogicalNot', [], [a]) as T;
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    return this.executeSingleOutput('LogicalAnd', [], [a, b]);
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    return this.executeSingleOutput('LogicalOr', [], [a, b]);
  }

  where(condition: Tensor, a: Tensor, b: Tensor, dtype: DataType): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    // 'Select' Op is where with additional inputs.
    return this.executeSingleOutput('Select', opAttrs, [condition, a, b]);
  }

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }
  topKIndices(x: Tensor, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }

  min(x: Tensor, axes: number[]): Tensor {
    const axesTensor = tensor1d(axes, 'int32');
    return this.executeSingleOutput(
        'Min', this.createReductionOpAttrs(x), [x, axesTensor]);
  }

  minimum(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Minimum', opAttrs, [a, b]);
  }

  max(x: Tensor, axes: number[]): Tensor {
    const axesTensor = tensor1d(axes, 'int32');
    return this.executeSingleOutput(
        'Max', this.createReductionOpAttrs(x), [x, axesTensor]);
  }

  maximum(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Maximum', opAttrs, [a, b]);
  }

  ceil<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Ceil', x) as T;
  }

  floor<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Floor', x) as T;
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Pow', opAttrs, [a, b]) as T;
  }

  exp<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Exp', x) as T;
  }

  log<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Log', x) as T;
  }

  log1p<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Log1p', x) as T;
  }

  sqrt<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Sqrt', x) as T;
  }

  square<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Square', x) as T;
  }

  relu<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Relu', x) as T;
  }

  elu<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Elu', x) as T;
  }

  eluDer<T extends Tensor>(dy: T, y: T): T {
    const opAttrs = [this.createTypeOpAttr('T', y.dtype)];
    return this.executeSingleOutput('EluGrad', opAttrs, [dy, y]) as T;
  }

  selu<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Selu', x) as T;
  }

  int<T extends Tensor>(x: T): T {
    throw new Error('Method not implemented.');
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    const xMin = this.minimum(x, scalar(max));
    return this.maximum(xMin, scalar(min)) as T;
  }

  abs<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Abs', x) as T;
  }

  sigmoid<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Sigmoid', x) as T;
  }

  sin<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Sin', x) as T;
  }

  cos<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Cos', x) as T;
  }

  tan<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Tan', x) as T;
  }

  asin<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Asin', x) as T;
  }

  acos<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Acos', x) as T;
  }

  atan<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Atan', x) as T;
  }

  sinh<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Sinh', x) as T;
  }

  cosh<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Cosh', x) as T;
  }

  tanh<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Tanh', x) as T;
  }

  mod(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', a.dtype)];
    return this.executeSingleOutput('FloorMod', opAttrs, [a, b]);
  }
  round<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Round', x) as T;
  }
  sign<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Sign', x) as T;
  }
  rsqrt<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Rsqrt', x) as T;
  }
  reciprocal<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Reciprocal', x) as T;
  }
  asinh<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Asinh', x) as T;
  }
  acosh<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Acosh', x) as T;
  }
  atanh<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Atanh', x) as T;
  }

  erf<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Erf', x) as T;
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', a.dtype)];
    return this.executeSingleOutput('SquaredDifference', opAttrs, [a, b]);
  }

  expm1<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Expm1', x) as T;
  }

  softplus<T extends Tensor>(x: T): T {
    return this.executeSingleInput('Softplus', x) as T;
  }

  atan2<T extends Tensor>(a: T, b: T): T {
    const opAttrs = [this.createTypeOpAttr('T', a.dtype)];
    return this.executeSingleOutput('Atan2', opAttrs, [a, b]) as T;
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    const dtype = x.dtype;
    const nans = this.isNaN(x);
    const stepNoNans = this.where(
        this.greater(x, scalar(0, dtype)), ones(x.shape),
        fill(x.shape, alpha, dtype), dtype);
    return this.where(nans, x, stepNoNans, dtype) as T;
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'use_cudnn_on_gpu', type: this.binding.TF_ATTR_BOOL, value: true},
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations},
    ];
    return this.executeSingleOutput('Conv2D', opAttrs, [x, filter]) as Tensor4D;
  }
  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', 'float32'),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'use_cudnn_on_gpu', type: this.binding.TF_ATTR_BOOL, value: true},
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations}
    ];
    const inputSizes = tensor1d(convInfo.inShape, 'int32');
    return this.executeSingleOutput(
               'Conv2DBackpropInput', opAttrs, [inputSizes, filter, dy]) as
        Tensor4D;
  }
  conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', 'float32'),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'use_cudnn_on_gpu', type: this.binding.TF_ATTR_BOOL, value: true},
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations}
    ];
    const filterSizes = tensor1d(convInfo.filterShape, 'int32');
    return this.executeSingleOutput(
               'Conv2DBackpropFilter', opAttrs, [x, filterSizes, dy]) as
        Tensor4D;
  }
  depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', input.dtype),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations}
    ];
    return this.executeSingleOutput(
               'DepthwiseConv2dNative', opAttrs, [input, filter]) as Tensor4D;
  }
  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      }
    ];
    return this.executeSingleOutput('MaxPool', opAttrs, [x]) as Tensor4D;
  }
  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding type was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
    ];
    return this.executeSingleOutput('MaxPoolGrad', opAttrs, [x, y, dy]) as
        Tensor4D;
  }

  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
    ];
    return this.executeSingleOutput('AvgPool', opAttrs, [x]) as Tensor4D;
  }
  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding type was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
    ];
    const origInputShape = tensor1d(x.shape, 'int32');
    return this.executeSingleOutput(
               'AvgPoolGrad', opAttrs, [origInputShape, dy]) as Tensor4D;
  }

  reshape<T extends Tensor, R extends Rank>(x: T, shape: ShapeMap[R]):
      Tensor<R> {
    const shapeTensor = tensor1d(shape, 'int32');

    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tshape', shapeTensor.dtype)
    ];
    return this.executeSingleOutput('Reshape', opAttrs, [x, shapeTensor]) as
        Tensor<R>;
  }

  cast<T extends Tensor>(x: T, dtype: DataType): T {
    const opAttrs = [
      this.createTypeOpAttr('SrcT', x.dtype),
      this.createTypeOpAttr('DstT', dtype)
    ];
    return this.executeSingleOutput('Cast', opAttrs, [x]) as T;
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tmultiples', 'int32')
    ];
    const multiples = tensor1d(reps, 'int32');
    return this.executeSingleOutput('Tile', opAttrs, [x, multiples]) as T;
  }
  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    // Bind tensor values
    const paddingsTensor = tensor2d(paddings, [paddings.length, 2], 'int32');
    const constantTensor = scalar(constantValue, x.dtype);

    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tpaddings', paddingsTensor.dtype)
    ];

    return this.executeSingleOutput(
               'PadV2', opAttrs, [x, paddingsTensor, constantTensor]) as T;
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    const permTensor = tensor1d(perm, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tperm', 'int32')
    ];
    return this.executeSingleOutput('Transpose', opAttrs, [x, permTensor]) as T;
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    const axisTensor = scalar(axis, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('Tparams', x.dtype),
      this.createTypeOpAttr('Tindices', indices.dtype),
      this.createTypeOpAttr('Taxis', 'int32')
    ];
    return this.executeSingleOutput(
               'GatherV2', opAttrs, [x, indices, axisTensor]) as T;
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {
        name: 'align_corners',
        type: this.binding.TF_ATTR_BOOL,
        value: alignCorners
      },
    ];
    const size = tensor1d([newHeight, newWidth], 'int32');
    return this.executeSingleOutput('ResizeBilinear', opAttrs, [x, size]) as
        Tensor4D;
  }

  resizeBilinearBackprop(
      dy: Tensor<Rank.R4>, x: Tensor<Rank.R4>,
      alignCorners: boolean): Tensor<Rank.R4> {
    // https://github.com/tensorflow/tfjs/issues/304
    throw new Error('Method not implemented.');
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {
        name: 'align_corners',
        type: this.binding.TF_ATTR_BOOL,
        value: alignCorners
      },
    ];
    const size = tensor1d([newHeight, newWidth], 'int32');
    return this.executeSingleOutput(
               'ResizeNearestNeighbor', opAttrs, [x, size]) as Tensor4D;
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor1D|Tensor4D, variance: Tensor1D|Tensor4D,
      varianceEpsilon: number, scale?: Tensor1D|Tensor4D,
      offset?: Tensor1D|Tensor4D): Tensor4D {
    if (mean.rank > 1) {
      // Fused batch norm doesn't work with high-dim mean/var/scale/offset.
      let inv = rsqrt(variance.add(scalar(varianceEpsilon)));
      if (scale != null) {
        inv = inv.mul(scale);
      }
      const xNorm = x.sub(mean).mul(inv) as Tensor4D;
      return offset != null ? xNorm.add(offset) : xNorm;
    }
    const dataFormat = 'NHWC';
    const depth = x.shape[3];
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {
        name: 'epsilon',
        type: this.binding.TF_ATTR_FLOAT,
        value: varianceEpsilon
      },
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'is_training', type: this.binding.TF_ATTR_BOOL, value: false},
    ];
    const numOutputs = 5;
    if (scale == null) {
      scale = fill([depth], 1) as Tensor1D;
    }
    if (offset == null) {
      offset = fill([depth], 0) as Tensor1D;
    }
    return this.executeMultipleOutputs(
               'FusedBatchNorm', opAttrs, [x, scale, offset, mean, variance],
               numOutputs)[0] as Tensor4D;
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'depth_radius', type: this.binding.TF_ATTR_INT, value: radius},
      {name: 'bias', type: this.binding.TF_ATTR_FLOAT, value: bias},
      {name: 'alpha', type: this.binding.TF_ATTR_FLOAT, value: alpha},
      {name: 'beta', type: this.binding.TF_ATTR_FLOAT, value: beta},
    ];
    return this.executeSingleOutput('LRN', opAttrs, [x]) as Tensor4D;
  }
  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    if (normalized) {
      throw new Error(
          'TF Node backend does not support normalized logits ' +
          'passed to multinomial');
    }
    const opAttrs = [
      this.createTypeOpAttr('T', logits.dtype),
      this.createTypeOpAttr('output_dtype', 'int32'),
      {name: 'seed', type: this.binding.TF_ATTR_INT, value: seed},
      {name: 'seed2', type: this.binding.TF_ATTR_INT, value: seed * seed},
    ];
    return this.executeSingleOutput(
               'Multinomial', opAttrs, [logits, scalar(numSamples, 'int32')]) as
        Tensor2D;
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

    return this.executeSingleOutput('OneHot', opAttrs, [
      indices, depthTensor, onValueTensor, offValueTensor
    ]) as Tensor2D;
  }

  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    throw new Error('Method not implemented.');
  }

  memory() {
    // Due to automatic garbage collection, the numbers are unreliable.
    // TODO: Since there is finalization in C, count the true
    // number of undisposed tensors.
    return {unreliable: true};
  }

  async time(f: () => void): Promise<BackendTimingInfo> {
    const start = process.hrtime();
    f();
    // hrtime() returns tuple of [seconds, nanoseconds], and we need to return
    // milliseconds.
    const elapsed = process.hrtime(start);
    return {kernelMs: elapsed[0] * 1000 + elapsed[1] / 1000000};
  }

  isNaN<T extends Tensor>(x: T): T {
    return this.executeSingleInput('IsNan', x) as T;
  }
}
