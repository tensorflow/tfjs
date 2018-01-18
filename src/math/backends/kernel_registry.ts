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

import * as util from '../../util';
import {DataType, NDArray, Scalar} from '../ndarray';

import {MathBackend} from './backend';
import {KernelInputConfig} from './tape_types';
// tslint:disable-next-line:max-line-length
import {ArgMaxInputConfig, ArgMaxNode, ArgMinInputConfig, ArgMinNode} from './types/argminmax';
// tslint:disable-next-line:max-line-length
import {BatchNorm2DInputConfig, BatchNorm2DNode, BatchNorm3DInputConfig, BatchNorm3DNode, BatchNorm4DInputConfig, BatchNorm4DNode} from './types/batchnorm';
import {BinaryInputConfig, BinaryNode} from './types/binary';
import {CastInputConfig, CastNode} from './types/cast';
// tslint:disable-next-line:max-line-length
import {Concat1DInputConfig, Concat1DNode, Concat2DInputConfig, Concat2DNode, Concat3DInputConfig, Concat3DNode, Concat4DInputConfig, Concat4DNode} from './types/concat';
// tslint:disable-next-line:max-line-length
import {Conv2DDerBiasInputConfig, Conv2DDerBiasNode, Conv2DDerFilterInputConfig, Conv2DDerFilterNode, Conv2DDerInputInputConfig, Conv2DDerInputNode, Conv2DInputConfig, Conv2DNode, DepthwiseConv2DInputConfig} from './types/conv';
import {GatherInputConfig, GatherNode} from './types/gather';
// tslint:disable-next-line:max-line-length
import {EqualInputConfig, EqualNode, LogicalInputConfig, LogicalNode, WhereInputConfig, WhereNode} from './types/logical';
import {LRN4DInputConfig, LRN4DNode} from './types/lrn';
import {MatMulInputConfig, MatMulNode} from './types/matmul';
// tslint:disable-next-line:max-line-length
import {MaximumInputConfig, MaximumNode, MaxInputConfig, MaxNode, MinimumInputConfig, MinimumNode, MinInputConfig, MinNode} from './types/minmax';
import {MultinomialInputConfig, MultinomialNode} from './types/multinomial';
import {OneHotInputConfig, OneHotNode} from './types/onehot';
// tslint:disable-next-line:max-line-length
import {Pad1DInputConfig, Pad1DNode, Pad2DInputConfig, Pad2DNode} from './types/pad';
// tslint:disable-next-line:max-line-length
import {PoolBackpropInputConfig, PoolBackpropNode, PoolInputConfig, PoolNode} from './types/pool';
import {PowInputConfig, PowNode} from './types/pow';
import {PReLUInputConfig, PReLUNode} from './types/prelu';
import {ReshapeNode} from './types/reshape';
// tslint:disable-next-line:max-line-length
import {ResizeBilinear3DInputConfig, ResizeBilinear3DNode} from './types/resize_bilinear';
import {Reverse4DInputConfig, Reverse4DNode} from './types/reverse';
// tslint:disable-next-line:max-line-length
import {Slice1DInputConfig, Slice1DNode, Slice2DInputConfig, Slice2DNode, Slice3DInputConfig, Slice3DNode, Slice4DInputConfig, Slice4DNode} from './types/slice';
import {SumInputConfig, SumNode} from './types/sum';
// tslint:disable-next-line:max-line-length
import {TopKIndicesInputConfig, TopKIndicesNode, TopKValuesInputConfig, TopKValuesNode} from './types/topk';
// tslint:disable-next-line:max-line-length
import {ClipInputConfig, ClipNode, LeakyReluInputConfig, LeakyReluNode, StepInputConfig, StepNode, TileInputConfig, TileNode, TransposeInputConfig, TransposeNode, UnaryInputConfig, UnaryNode} from './types/unary';

const KERNEL_METHODS: {
  [kernel in keyof KernelConfigRegistry]: (
      backend: MathBackend, config: KernelInputConfig) => NDArray
} = {
  // NOTE: Using {} and "return" makes VSCode run much faster.
  MatMul: (backend: MathBackend, config: MatMulInputConfig) => {
    return backend.matMul(
        config.inputs.a, config.inputs.b, config.args.aOrientation,
        config.args.bOrientation);
  },
  Clone: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.clone(config.inputs.x);
  },
  Slice1D: (backend: MathBackend, config: Slice1DInputConfig) => {
    return backend.slice1D(
        config.inputs.x, config.args.begin, config.args.size);
  },
  Slice2D: (backend: MathBackend, config: Slice2DInputConfig) => {
    return backend.slice2D(
        config.inputs.x, config.args.begin, config.args.size);
  },
  Slice3D: (backend: MathBackend, config: Slice3DInputConfig) => {
    return backend.slice3D(
        config.inputs.x, config.args.begin, config.args.size);
  },
  Slice4D: (backend: MathBackend, config: Slice4DInputConfig) => {
    return backend.slice4D(
        config.inputs.x, config.args.begin, config.args.size);
  },
  Reverse4D: (backend: MathBackend, config: Reverse4DInputConfig) => {
    return backend.reverse4D(config.inputs.x, config.args.axis);
  },
  Concat1D: (backend: MathBackend, config: Concat1DInputConfig) => {
    return backend.concat1D(config.inputs.a, config.inputs.b);
  },
  Concat2D: (backend: MathBackend, config: Concat2DInputConfig) => {
    return backend.concat2D(config.inputs.a, config.inputs.b, config.args.axis);
  },
  Concat3D: (backend: MathBackend, config: Concat3DInputConfig) => {
    return backend.concat3D(config.inputs.a, config.inputs.b, config.args.axis);
  },
  Concat4D: (backend: MathBackend, config: Concat4DInputConfig) => {
    return backend.concat4D(config.inputs.a, config.inputs.b, config.args.axis);
  },
  Neg: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.neg(config.inputs.x);
  },
  Add: (backend: MathBackend, config: BinaryInputConfig) => {
    return backend.add(config.inputs.a, config.inputs.b);
  },
  Sub: (backend: MathBackend, config: BinaryInputConfig) => {
    return backend.subtract(config.inputs.a, config.inputs.b);
  },
  Mul: (backend: MathBackend, config: BinaryInputConfig) => {
    return backend.multiply(config.inputs.a, config.inputs.b);
  },
  Div: (backend: MathBackend, config: BinaryInputConfig) => {
    return backend.divide(config.inputs.a, config.inputs.b);
  },
  Sum: (backend: MathBackend, config: SumInputConfig<'float32'|'int32'>) => {
    return backend.sum(config.inputs.x, config.args.axes);
  },
  ArgMax: (backend: MathBackend, config: ArgMaxInputConfig) => {
    return backend.argMax(config.inputs.x, config.args.axes);
  },
  ArgMin: (backend: MathBackend, config: ArgMinInputConfig) => {
    return backend.argMin(config.inputs.x, config.args.axes);
  },
  Equal: (backend: MathBackend, config: EqualInputConfig) => {
    return backend.equal(config.inputs.a, config.inputs.b);
  },
  NotEqual: (backend: MathBackend, config: EqualInputConfig) => {
    return backend.notEqual(config.inputs.a, config.inputs.b);
  },
  Less: (backend: MathBackend, config: EqualInputConfig) => {
    return backend.less(config.inputs.a, config.inputs.b);
  },
  LessEqual: (backend: MathBackend, config: EqualInputConfig) => {
    return backend.lessEqual(config.inputs.a, config.inputs.b);
  },
  Greater: (backend: MathBackend, config: EqualInputConfig) => {
    return backend.greater(config.inputs.a, config.inputs.b);
  },
  GreaterEqual: (backend: MathBackend, config: EqualInputConfig) => {
    return backend.greaterEqual(config.inputs.a, config.inputs.b);
  },
  LogicalAnd: (backend: MathBackend, config: LogicalInputConfig) => {
    return backend.logicalAnd(config.inputs.a, config.inputs.b);
  },
  LogicalOr: (backend: MathBackend, config: LogicalInputConfig) => {
    return backend.logicalOr(config.inputs.a, config.inputs.b);
  },
  Where: (backend: MathBackend, config: WhereInputConfig) => {
    return backend.where(
        config.inputs.condition, config.inputs.a, config.inputs.b,
        config.args.dtype);
  },
  TopKValues:
      (backend: MathBackend, config: TopKValuesInputConfig<NDArray>) => {
        return backend.topKValues(config.inputs.x, config.args.k);
      },
  TopKIndices: (backend: MathBackend, config: TopKIndicesInputConfig) => {
    return backend.topKIndices(config.inputs.x, config.args.k);
  },
  Min: (backend: MathBackend, config: MinInputConfig<DataType>) => {
    return backend.min(config.inputs.x, config.args.axes);
  },
  Minimum: (backend: MathBackend, config: MinimumInputConfig<DataType>) => {
    return backend.minimum(config.inputs.a, config.inputs.b);
  },
  Max: (backend: MathBackend, config: MaxInputConfig<DataType>) => {
    return backend.max(config.inputs.x, config.args.axes);
  },
  Maximum: (backend: MathBackend, config: MaximumInputConfig<DataType>) => {
    return backend.maximum(config.inputs.a, config.inputs.b);
  },
  Ceil: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.ceil(config.inputs.x);
  },
  Floor: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.floor(config.inputs.x);
  },
  Pow: (backend: MathBackend, config: PowInputConfig<NDArray>) => {
    return backend.pow(config.inputs.a, config.inputs.b);
  },
  Exp: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.exp(config.inputs.x);
  },
  Log: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.log(config.inputs.x);
  },
  Sqrt: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.sqrt(config.inputs.x);
  },
  Square: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.square(config.inputs.x);
  },
  Relu: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.relu(config.inputs.x);
  },
  Reshape: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    const x = config.inputs.x;
    const newShape = config.args.newShape;
    return NDArray.make(newShape, {dataId: x.dataId}, x.dtype);
  },
  Cast: (backend: MathBackend, config: CastInputConfig) => {
    const x = config.inputs.x;
    const newDType = config.args.newDType;

    if (!util.hasEncodingLoss(x.dtype, newDType)) {
      // We don't change the underlying data, since we cast to higher precision.
      return NDArray.make(x.shape, {dataId: x.dataId}, newDType);
    }
    if (newDType === 'int32') {
      return backend.int(x);
    } else if (newDType === 'bool') {
      return backend.notEqual(x, Scalar.new(0, x.dtype));
    } else {
      throw new Error(`Error in Cast: unknown dtype argument (${newDType})`);
    }
  },
  LeakyRelu: (backend: MathBackend, config: LeakyReluInputConfig<NDArray>) => {
    return backend.leakyRelu(config.inputs.x, config.args.alpha);
  },
  PReLU: (backend: MathBackend, config: PReLUInputConfig<NDArray>) => {
    return backend.prelu(config.inputs.x, config.inputs.alpha);
  },
  PReLUDer: (backend: MathBackend, config: PReLUInputConfig<NDArray>) => {
    return backend.preluDer(config.inputs.x, config.inputs.alpha);
  },
  Elu: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.elu(config.inputs.x);
  },
  EluDer: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.eluDer(config.inputs.x);
  },
  Selu: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.selu(config.inputs.x);
  },
  Abs: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.abs(config.inputs.x);
  },
  Sigmoid: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.sigmoid(config.inputs.x);
  },
  Step: (backend: MathBackend, config: StepInputConfig<NDArray>) => {
    return backend.step(config.inputs.x, config.args.alpha);
  },
  Sin: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.sin(config.inputs.x);
  },
  Cos: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.cos(config.inputs.x);
  },
  Tan: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.tan(config.inputs.x);
  },
  Asin: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.asin(config.inputs.x);
  },
  Acos: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.acos(config.inputs.x);
  },
  Atan: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.atan(config.inputs.x);
  },
  Sinh: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.sinh(config.inputs.x);
  },
  Cosh: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.cosh(config.inputs.x);
  },
  Tanh: (backend: MathBackend, config: UnaryInputConfig<NDArray>) => {
    return backend.tanh(config.inputs.x);
  },
  Clip: (backend: MathBackend, config: ClipInputConfig<NDArray>) => {
    return backend.clip(config.inputs.x, config.args.min, config.args.max);
  },
  Tile: (backend: MathBackend, config: TileInputConfig<NDArray>) => {
    return backend.tile(config.inputs.x, config.args.reps);
  },
  Gather: (backend: MathBackend, config: GatherInputConfig<NDArray>) => {
    return backend.gather(
        config.inputs.x, config.inputs.indices, config.args.axis);
  },
  Pad1D: (backend: MathBackend, config: Pad1DInputConfig) => {
    return backend.pad1D(
        config.inputs.x, config.args.paddings, config.args.constantValue);
  },
  Pad2D: (backend: MathBackend, config: Pad2DInputConfig) => {
    return backend.pad2D(
        config.inputs.x, config.args.paddings, config.args.constantValue);
  },
  Transpose: (backend: MathBackend, config: TransposeInputConfig<NDArray>) => {
    return backend.transpose(config.inputs.x, config.args.perm);
  },
  Conv2D: (backend: MathBackend, config: Conv2DInputConfig) => {
    return backend.conv2d(
        config.inputs.x, config.inputs.filter, config.inputs.bias,
        config.args.convInfo);
  },
  Conv2DDerInput: (backend: MathBackend, config: Conv2DDerInputInputConfig) => {
    return backend.conv2dDerInput(
        config.inputs.dy, config.inputs.filter, config.args.convInfo);
  },
  Conv2DDerFilter:
      (backend: MathBackend, config: Conv2DDerFilterInputConfig) => {
        return backend.conv2dDerFilter(
            config.inputs.x, config.inputs.dy, config.args.convInfo);
      },
  Conv2DDerBias: (backend: MathBackend, config: Conv2DDerBiasInputConfig) => {
    return backend.conv2dDerBias(config.inputs.dy);
  },
  DepthwiseConv2D:
      (backend: MathBackend, config: DepthwiseConv2DInputConfig) => {
        return backend.depthwiseConv2D(
            config.inputs.x, config.inputs.filter, config.args.convInfo);
      },
  MaxPool: (backend: MathBackend, config: PoolInputConfig) => {
    return backend.maxPool(config.inputs.x, config.args.convInfo);
  },
  MaxPoolBackprop: (backend: MathBackend, config: PoolBackpropInputConfig) => {
    return backend.maxPoolBackprop(
        config.inputs.dy, config.inputs.x, config.args.convInfo);
  },
  AvgPool: (backend: MathBackend, config: PoolInputConfig) => {
    return backend.avgPool(config.inputs.x, config.args.convInfo);
  },
  AvgPoolBackprop: (backend: MathBackend, config: PoolBackpropInputConfig) => {
    return backend.avgPoolBackprop(
        config.inputs.dy, config.inputs.x, config.args.convInfo);
  },
  MinPool: (backend: MathBackend, config: PoolInputConfig) => {
    return backend.minPool(config.inputs.x, config.args.convInfo);
  },
  ResizeBilinear3D:
      (backend: MathBackend, config: ResizeBilinear3DInputConfig) => {
        return backend.resizeBilinear3D(
            config.inputs.x, config.args.newShape2D, config.args.alignCorners);
      },
  BatchNorm4D: (backend: MathBackend, config: BatchNorm4DInputConfig) => {
    return backend.batchNormalization4D(
        config.inputs.x, config.inputs.mean, config.inputs.variance,
        config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
  },
  BatchNorm3D: (backend: MathBackend, config: BatchNorm3DInputConfig) => {
    return backend.batchNormalization3D(
        config.inputs.x, config.inputs.mean, config.inputs.variance,
        config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
  },
  BatchNorm2D: (backend: MathBackend, config: BatchNorm2DInputConfig) => {
    return backend.batchNormalization2D(
        config.inputs.x, config.inputs.mean, config.inputs.variance,
        config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
  },
  LRN4D: (backend: MathBackend, config: LRN4DInputConfig) => {
    return backend.localResponseNormalization4D(
        config.inputs.x, config.args.radius, config.args.bias,
        config.args.alpha, config.args.beta, config.args.normRegion);
  },
  Multinomial: (backend: MathBackend, config: MultinomialInputConfig) => {
    return backend.multinomial(
        config.inputs.probs, config.args.numSamples, config.args.seed);
  },
  OneHot: (backend: MathBackend, config: OneHotInputConfig) => {
    return backend.oneHot(
        config.inputs.indices, config.args.depth, config.args.onValue,
        config.args.offValue);
  }
};
export function executeKernel<K extends keyof KernelConfigRegistry, R extends
                                  KernelConfigRegistry[K]['output']>(
    backend: MathBackend, kernelName: K,
    config: KernelConfigRegistry[K]['inputAndArgs']): R {
  return KERNEL_METHODS[kernelName](backend, config) as R;
}

export interface KernelConfigRegistry {
  MatMul: MatMulNode;
  Clone: UnaryNode<NDArray>;
  Slice1D: Slice1DNode;
  Slice2D: Slice2DNode;
  Slice3D: Slice3DNode;
  Slice4D: Slice4DNode;
  Reverse4D: Reverse4DNode;
  Concat1D: Concat1DNode;
  Concat2D: Concat2DNode;
  Concat3D: Concat3DNode;
  Concat4D: Concat4DNode;
  Neg: UnaryNode<NDArray>;
  Add: BinaryNode;
  Sub: BinaryNode;
  Mul: BinaryNode;
  Div: BinaryNode;
  Sum: SumNode<DataType>;
  ArgMax: ArgMaxNode;
  ArgMin: ArgMinNode;
  Equal: EqualNode;
  NotEqual: EqualNode;
  Less: EqualNode;
  LessEqual: EqualNode;
  Greater: EqualNode;
  GreaterEqual: EqualNode;
  LogicalAnd: LogicalNode;
  LogicalOr: LogicalNode;
  Where: WhereNode;
  TopKValues: TopKValuesNode<DataType, NDArray>;
  TopKIndices: TopKIndicesNode;
  Min: MinNode<DataType>;
  Minimum: MinimumNode<DataType>;
  Max: MaxNode<DataType>;
  Maximum: MaximumNode<DataType>;
  Ceil: UnaryNode<NDArray>;
  Floor: UnaryNode<NDArray>;
  Pow: PowNode<NDArray>;
  Exp: UnaryNode<NDArray>;
  Log: UnaryNode<NDArray>;
  Sqrt: UnaryNode<NDArray>;
  Square: UnaryNode<NDArray>;
  Relu: UnaryNode<NDArray>;
  LeakyRelu: LeakyReluNode<NDArray>;
  PReLU: PReLUNode<NDArray>;
  PReLUDer: PReLUNode<NDArray>;
  Reshape: ReshapeNode;
  Cast: CastNode;
  Elu: UnaryNode<NDArray>;
  EluDer: UnaryNode<NDArray>;
  Selu: UnaryNode<NDArray>;
  Abs: UnaryNode<NDArray>;
  Sigmoid: UnaryNode<NDArray>;
  Step: StepNode<NDArray>;
  Sin: UnaryNode<NDArray>;
  Cos: UnaryNode<NDArray>;
  Tan: UnaryNode<NDArray>;
  Asin: UnaryNode<NDArray>;
  Acos: UnaryNode<NDArray>;
  Atan: UnaryNode<NDArray>;
  Sinh: UnaryNode<NDArray>;
  Cosh: UnaryNode<NDArray>;
  Tanh: UnaryNode<NDArray>;
  Clip: ClipNode<NDArray>;
  Transpose: TransposeNode<NDArray>;
  Pad1D: Pad1DNode;
  Pad2D: Pad2DNode;
  Tile: TileNode<NDArray>;
  Gather: GatherNode<NDArray>;
  Conv2D: Conv2DNode;
  Conv2DDerInput: Conv2DDerInputNode;
  Conv2DDerFilter: Conv2DDerFilterNode;
  Conv2DDerBias: Conv2DDerBiasNode;
  DepthwiseConv2D: Conv2DNode;
  MaxPool: PoolNode;
  MaxPoolBackprop: PoolBackpropNode;
  AvgPool: PoolNode;
  AvgPoolBackprop: PoolBackpropNode;
  MinPool: PoolNode;
  ResizeBilinear3D: ResizeBilinear3DNode;
  BatchNorm4D: BatchNorm4DNode;
  BatchNorm3D: BatchNorm3DNode;
  BatchNorm2D: BatchNorm2DNode;
  LRN4D: LRN4DNode;
  Multinomial: MultinomialNode;
  OneHot: OneHotNode;
}
