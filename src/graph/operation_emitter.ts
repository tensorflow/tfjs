/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {AddNode, ArgMaxEqualsNode, ArgMaxNode, Concat1DNode, Concat2DNode, Concat3DNode, Concat4DNode, Convolution2DNode, DivideNode, ExpNode, EluNode, FusedLinearCombinationNode, LogNode, MatMulNode, MaxPoolNode, MeanSquaredCostNode, MultiplyNode, Node, ReduceSumNode, ReLUNode, PReLUNode, LeakyReLUNode, ReshapeNode, SigmoidNode, SoftmaxCrossEntropyCostNode, SoftmaxNode, SquareNode, SubtractNode, TanHNode} from './graph';
import * as graph_util from './graph_util';
import {Add} from './ops/add';
import {ArgMax} from './ops/argmax';
import {ArgMaxEquals} from './ops/argmaxequals';
import {Concat1D, Concat2D, Concat3D, Concat4D} from './ops/concat';
import {Convolution2D} from './ops/convolution';
import {Divide} from './ops/divide';
// tslint:disable-next-line:max-line-length
import {ReLU, Sigmoid, Square, TanH, LeakyReLU, PReLU, Elu} from './ops/element_wise_activation';
import {MeanSquaredCost} from './ops/element_wise_cost';
import {Exp} from './ops/exp';
import {LinearCombination} from './ops/linear_combination';
import {Log} from './ops/log';
import {MatMul} from './ops/matmul';
import {MaxPool} from './ops/max_pool';
import {Multiply} from './ops/multiply';
import {Operation} from './ops/op';
import {ReduceSum} from './ops/reduce_sum';
import {Reshape} from './ops/reshape';
import {Softmax, SoftmaxCrossEntropyCost} from './ops/softmax';
import {Subtract} from './ops/subtract';

export function emitFromGraphNodes(nodes: Node[]): Operation[] {
  const ops: Operation[] = [];
  nodes.forEach(node => Array.prototype.push.apply(ops, emitOpFromNode(node)));
  return ops;
}

function emitOpFromNode(node: Node): Operation[] {
  if (node instanceof ReshapeNode) {
    return [new Reshape(node.inputs[ReshapeNode.X], node.output)];
  } else if (node instanceof MatMulNode) {
    const x1 = node.inputs[MatMulNode.X1];
    const x2 = node.inputs[MatMulNode.X2];
    return [new MatMul(x1, x2, node.output)];
  } else if (node instanceof Convolution2DNode) {
    const w = node.inputs[Convolution2DNode.W];
    const x = node.inputs[Convolution2DNode.X];
    const b = node.inputs[Convolution2DNode.B];
    return [new Convolution2D(
        w, x, b, node.output, node.fieldSize, node.outputDepth, node.stride,
        node.zeroPad)];
  } else if (node instanceof MaxPoolNode) {
    const x = node.inputs[MaxPoolNode.X];
    return [new MaxPool(
        x, node.output, node.fieldSize, node.stride, node.zeroPad)];
  } else if (node instanceof ExpNode) {
    return [new Exp(node.inputs[ExpNode.X], node.output)];
  } else if (node instanceof LogNode) {
    return [new Log(node.inputs[LogNode.X], node.output)];
  } else if (node instanceof ReLUNode) {
    return [new ReLU(node.inputs[ReLUNode.X], node.output)];
  } else if (node instanceof LeakyReLUNode) {
    return [new LeakyReLU(node.inputs[LeakyReLUNode.X], 
      node.output, node.alpha)];
  } else if (node instanceof PReLUNode) {
    return [new PReLU(node.inputs[PReLUNode.X], node.inputs[PReLUNode.ALPHA],
      node.output)];
  } else if (node instanceof EluNode) {
    return [new Elu(node.inputs[EluNode.X], node.output)];
  } else if (node instanceof TanHNode) {
    return [new TanH(node.inputs[TanHNode.X], node.output)];
  } else if (node instanceof SigmoidNode) {
    return [new Sigmoid(node.inputs[SigmoidNode.X], node.output)];
  } else if (node instanceof SoftmaxCrossEntropyCostNode) {
    const x = node.inputs[SoftmaxCrossEntropyCostNode.X];
    const target = node.inputs[SoftmaxCrossEntropyCostNode.TARGET];
    return [new SoftmaxCrossEntropyCost(x, target, node.output)];
  } else if (node instanceof SoftmaxNode) {
    return [new Softmax(node.inputs[SoftmaxNode.X], node.output)];
  } else if (node instanceof MeanSquaredCostNode) {
    const label = node.inputs[MeanSquaredCostNode.LABEL];
    const prediction = node.inputs[MeanSquaredCostNode.PREDICTION];
    return [new MeanSquaredCost(label, prediction, node.output)];
  } else if (node instanceof ArgMaxEqualsNode) {
    return [new ArgMaxEquals(
        node.inputs[ArgMaxEqualsNode.X1], node.inputs[ArgMaxEqualsNode.X2],
        node.output)];
  } else if (node instanceof ArgMaxNode) {
    return [new ArgMax(node.x, node.output)];
  } else if (node instanceof FusedLinearCombinationNode) {
    return [new LinearCombination(
        node.inputs[FusedLinearCombinationNode.T1],
        node.inputs[FusedLinearCombinationNode.T2],
        node.inputs[FusedLinearCombinationNode.C1],
        node.inputs[FusedLinearCombinationNode.C2], node.output)];
  } else if (node instanceof Concat1DNode) {
    return [new Concat1D(
        node.inputs[Concat1DNode.X1], node.inputs[Concat1DNode.X2],
        node.output)];
  } else if (node instanceof Concat2DNode) {
    return [new Concat2D(
        node.inputs[Concat2DNode.X1], node.inputs[Concat2DNode.X2], node.axis,
        node.output)];
  } else if (node instanceof Concat3DNode) {
    return [new Concat3D(
        node.inputs[Concat3DNode.X1], node.inputs[Concat3DNode.X2], node.axis,
        node.output)];
  } else if (node instanceof Concat4DNode) {
    return [new Concat4D(
        node.inputs[Concat4DNode.X1], node.inputs[Concat4DNode.X2], node.axis,
        node.output)];
  } else if (node instanceof SquareNode) {
    return [new Square(node.inputs[SquareNode.X], node.output)];
  } else if (node instanceof AddNode) {
    return [new Add(
        node.inputs[AddNode.T1], node.inputs[AddNode.T2], node.output)];
  } else if (node instanceof SubtractNode) {
    return [new Subtract(
        node.inputs[SubtractNode.T1], node.inputs[SubtractNode.T2],
        node.output)];
  } else if (node instanceof MultiplyNode) {
    return [new Multiply(
        node.inputs[MultiplyNode.T1], node.inputs[MultiplyNode.T2],
        node.output)];
  } else if (node instanceof DivideNode) {
    return [new Divide(
        node.inputs[DivideNode.T1], node.inputs[DivideNode.T2], node.output)];
  } else if (node instanceof ReduceSumNode) {
    return [new ReduceSum(node.inputs[ReduceSumNode.X], node.output)];
  } else if (graph_util.isInputNode(node)) {
    return [];
  } else {
    // tslint:disable-next-line:no-any
    throw Error(`Unsupported node type: ${node.constructor.name}`);
  }
}
