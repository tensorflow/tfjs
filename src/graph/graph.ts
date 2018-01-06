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
import {Initializer, VarianceScalingInitializer, ZerosInitializer} from '../initializers';
import * as concat_util from '../math/concat_util';
import * as conv_util from '../math/conv_util';
import {NDArray, Scalar} from '../math/ndarray';
import * as util from '../util';

/**
 * A layers sugar class around the graph that initializes variables
 * automatically for layers.
 */
export class GraphLayers {
  constructor(private g: Graph) {}

  dense(
      name: string, x: Tensor, units: number,
      activation: ((x: Tensor) => Tensor)|null = null, useBias = true,
      kernelInitializer: Initializer = new VarianceScalingInitializer(),
      biasInitializer: Initializer = new ZerosInitializer()) {
    const weights = this.g.variable(
        name + '-weights',
        kernelInitializer.initialize([x.shape[0], units], x.shape[0], units));

    let out = this.g.matmul(x, weights);

    if (useBias) {
      const bias = this.g.variable(
          name + '-bias',
          biasInitializer.initialize([units], x.shape[0], units));
      out = this.g.add(out, bias);
    }

    if (activation != null) {
      out = activation(out);
    }

    return out;
  }
}

/**
 * Graph is the primary container structure for deeplearn.js operations. Graph
 * holds the topology of operation nodes and the connectivity between them.
 */
export class Graph {
  layers: GraphLayers;

  constructor() {
    this.layers = new GraphLayers(this);
  }

  /**
   * Creates a named variable. Variables are tensors that maintain state across
   * session calls and whose values are adjusted during backpropagation
   * training.
   * @param name The name of this variable.
   * @param data The NDArray to associate with this variable tensor.
   * @return The tensor representing the variable.
   */
  variable(name: string, data: NDArray): Tensor {
    return this.addNodeAndReturnOutput(new VariableNode(this, name, data));
  }

  /**
   * Inserts a placeholder for a tensor that will be always fed. Placeholders
   * are input tensors whose values are provided by the client via feed
   * dictionaries. Placeholders are not updated as part of training; they are
   * only used as immutable input.
   * @param name The name of this placeholder.
   * @param shape The shape of the placeholder tensor.
   * @return The tensor representing the placeholder.
   */
  placeholder(name: string, shape: number[]): Tensor {
    return this.addNodeAndReturnOutput(new PlaceholderNode(this, name, shape));
  }

  /**
   * Constant value that persists across session calls.
   * @param value The value to return.
   * @return A node outputing the constant value.
   */
  constant(value: ArrayData): Tensor {
    let finalValue: NDArray;
    if (typeof value === 'number') {
      finalValue = Scalar.new(value);
    } else if (value instanceof NDArray) {
      finalValue = value;
    } else if (value instanceof Array) {
      const flatValues = util.flatten(value as number[]);
      const vals = new Float32Array(flatValues as number[]);
      finalValue = NDArray.make(util.inferShape(value), {values: vals});
    } else {
      throw new Error('unimplemented constant type.');
    }
    return this.addNodeAndReturnOutput(new ConstantNode(this, finalValue));
  }

  /**
   * Reshape the input tensor.
   * @param x The input tensor to be reshaped.
   * @param shape The shape of the output tensor.
   * @return The tensor representing the reshape operation.
   */
  reshape(x: Tensor, shape: number[]): Tensor {
    return this.addNodeAndReturnOutput(
        new ReshapeNode(this, 'Reshape', x, shape));
  }

  /**
   * Computes a fused linear combination of two tensors.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor. Same shape as t1.
   * @param c1 Coefficient of t1. Must be size 1.
   * @param c2 Coefficient of t2. Must be size 1.
   * @return The tensor representing c1*t1+c2*t2.
   */
  fusedLinearCombination(x1: Tensor, x2: Tensor, c1: Tensor, c2: Tensor):
      Tensor {
    return this.addNodeAndReturnOutput(
        new FusedLinearCombinationNode(this, x1, x2, c1, c2));
  }

  /**
   * Adds two tensors (elementwise). Broadcasts if one of the tensors is scalar.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @return The tensor representing t1+t2.
   */
  add(x1: Tensor, x2: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new AddNode(this, x1, x2));
  }

  /**
   * Subtracts two tensors (elementwise). Broadcasts if one of the tensors is
   * scalar.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @return The tensor representing t1-t2.
   */
  subtract(x1: Tensor, x2: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new SubtractNode(this, x1, x2));
  }

  /**
   * Multiply two tensors (elementwise). Broadcasts if one of the tensors is
   * scalar.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @return The tensor representing t1*t2.
   */
  multiply(x1: Tensor, x2: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new MultiplyNode(this, x1, x2));
  }

  /**
   * Divide two tensors (elementwise). Broadcasts if one of the tensors is
   * scalar.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @return The tensor representing t1 / t2.
   */
  divide(x1: Tensor, x2: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new DivideNode(this, x1, x2));
  }

  /**
   * Computes the sum of elements in the tensor.
   * @param x The input tensor.
   */
  reduceSum(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new ReduceSumNode(this, x));
  }

  /**
   * Concats two 1D tensors along a given axis.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @return The tensor representing concat of two tensors along axis.
   */
  concat1d(x1: Tensor, x2: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new Concat1DNode(this, x1, x2));
  }

  /**
   * Concats two 2D tensors along a given axis.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @param axis The axis to concatenate along.
   * @return The tensor representing concat of two tensors along axis.
   */
  concat2d(x1: Tensor, x2: Tensor, axis: number): Tensor {
    return this.addNodeAndReturnOutput(new Concat2DNode(this, x1, x2, axis));
  }

  /**
   * Concats two 3D tensors along a given axis.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @param axis The axis to concatenate along.
   * @return The tensor representing concat of two tensors along axis.
   */
  concat3d(x1: Tensor, x2: Tensor, axis: number): Tensor {
    return this.addNodeAndReturnOutput(new Concat3DNode(this, x1, x2, axis));
  }

  /**
   * Concats two 4D tensors along a given axis.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @param axis The axis to concatenate along.
   * @return The tensor representing concat of two tensors along axis.
   */
  concat4d(x1: Tensor, x2: Tensor, axis: number): Tensor {
    return this.addNodeAndReturnOutput(new Concat4DNode(this, x1, x2, axis));
  }

  /**
   * Computes the dot product between two matrices.
   * @param x1 The first input tensor.
   * @param x2 The second input tensor.
   * @return The tensor representing the dot product of x1 and x2.
   */
  matmul(x1: Tensor, x2: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new MatMulNode(this, x1, x2));
  }

  /**
   * Computes a 2D convolution.
   * @param x The input tensor to the convolution operation.
   * @param w The weight tensor used by the convolution operation.
   * @param b The bias tensor used by the convolution operation.
   * @param fieldSize The size of the convolutional kernel.
   * @param outputDepth The output depth of the convolution operation.
   * @param stride The stride of the convolution operation.
   * @param zeroPad The amount of zero padding on all sides of the input tensor.
   * @return The tensor representing the convolution operation.
   */
  conv2d(
      x: Tensor, w: Tensor, b: Tensor, fieldSize: number, outputDepth: number,
      stride = 1, zeroPad?: number): Tensor {
    return this.addNodeAndReturnOutput(new Convolution2DNode(
        this, x, w, b, fieldSize, outputDepth, stride, zeroPad));
  }

  /**
   * Computes a 2D max pool of x.
   * @param x The input tensor to the max pool operation.
   * @param fieldSize The size of the convolutional kernel.
   * @param stride The stride of the convolution operation.
   * @param zeroPad The amount of zero padding on all sides of the input tensor.
   * @return The tensor representing the max pool operation.
   */
  maxPool(x: Tensor, fieldSize: number, stride = 1, zeroPad?: number): Tensor {
    return this.addNodeAndReturnOutput(
        new MaxPoolNode(this, x, fieldSize, stride, zeroPad));
  }

  /**
   * Computes exponential of x element-wise.
   * @param x The input tensor to the exp.
   * @return The tensor representing the e ^ x operation.
   */
  exp(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new ExpNode(this, x));
  }

  /**
   * Computes log of x element-wise.
   * @param x The input tensor to the log.
   * @return The tensor representing the ln(x) operation.
   */
  log(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new LogNode(this, x));
  }

  /**
   * Computes ReLU of x element-wise.
   * @param x The input tensor to the ReLU.
   * @return The tensor representing the ReLU operation.
   */
  relu(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new ReLUNode(this, x));
  }

  /**
   * Computes LeakyReLU of x element-wise.
   * @param x The input tensor to the LeakyReLU.
   * @param alpha Negative slope coefficient.
   * @return The tensor representing the LeakyReLU operation.
   */
  leakyRelu(x: Tensor, alpha: number): Tensor {
    return this.addNodeAndReturnOutput(new LeakyReLUNode(this, x, alpha));
  }

  /**
   * Computes PReLU of x element-wise.
   * @param x The input tensor to the LeakyReLU.
   * @param alpha Negative slope coefficient tensor.
   * @return The tensor representing the PReLU operation.
   */
  prelu(x: Tensor, alpha: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new PReLUNode(this, x, alpha));
  }

  /**
   * Computes Elu of x element-wise.
   * @param x the input tensor to the Elu.
   * @return The tensor representing the Elu operation.
   */
  elu(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new EluNode(this, x));
  }

  /**
   * Computes TanH of x element-wise.
   * @param x The input tensor to the TanH.
   * @return The tensor representing the TanH operation.
   */
  tanh(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new TanHNode(this, x));
  }

  /**
   * Computes Sigmoid of x element-wise.
   * @param x The input tensor to the sigmoid.
   * @return The tensor representing the sigmoid operation.
   */
  sigmoid(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new SigmoidNode(this, x));
  }

  /**
   * Computes square of x element-wise.
   * @param x The input tensor to the square.
   */
  square(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new SquareNode(this, x));
  }

  /**
   * Computes softmax probabilities from logits.
   *
   * @param x The input logits.
   * @return The softmax probabilities.
   */
  softmax(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new SoftmaxNode(this, x));
  }

  /**
   * Creates a softmax cross-entropy cost operation in the graph.
   * @param x The input tensor to classify.
   * @param target The label tensor.
   * @return The tensor representing the softmax cross-entropy cost operation.
   */
  softmaxCrossEntropyCost(x: Tensor, target: Tensor): Tensor {
    return this.addNodeAndReturnOutput(
        new SoftmaxCrossEntropyCostNode(this, x, target));
  }

  /**
   * Creates a mean-squared cost operation in the graph.
   * @param label The label tensor.
   * @param prediction The prediction tensor.
   * @return The tensor representing the mean-squared cost operation.
   */
  meanSquaredCost(label: Tensor, prediction: Tensor) {
    return this.addNodeAndReturnOutput(
        new MeanSquaredCostNode(this, label, prediction));
  }

  /**
   * Returns the flattened index of the maximum entry in the tensor.
   * @param x The tensor with the value.
   * @return A Scalar tensor with the index of the maximum entry.
   */
  argmax(x: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new ArgMaxNode(this, x));
  }

  /**
   * Creates an argmax equals operation in the graph.
   * @param x1 First input tensor to check against.
   * @param x2 Second input tensor to check against.
   * @return The tensor representing the argmax equals operation.
   */
  argmaxEquals(x1: Tensor, x2: Tensor): Tensor {
    return this.addNodeAndReturnOutput(new ArgMaxEqualsNode(this, x1, x2));
  }

  private addNodeAndReturnOutput(node: Node): Tensor {
    this.nodes.push(node);
    node.validate();
    return node.output;
  }

  getNodes(): Node[] {
    return this.nodes;
  }

  private nodes: Node[] = [];
}

/**
 * Tensor represents the output of an operation node in the graph.
 * Tensors have no data associated with them, but maintain a shape array
 * to determine operation compatibility. All graph methods that create graph
 * operations return Tensor objects, which can be thought of as 'handles' to
 * operations.
 */
export class Tensor {
  node: Node;
  id: number;
  /**
   * @param shape The shape of this tensor, in dimension sizes.
   */
  constructor(public shape: number[]) {
    this.id = Tensor.nextID++;
  }
  private static nextID = 0;
}

/**
 * Node is the concrete base class for all operations in the graph.
 * Users generally don't need to interact directly with Node instances, but they
 * are provided for informational and introspection purposes.
 *
 * @hidden
 */
export abstract class Node {
  /**
   * @param graph The graph containing this node
   * @param name The name of this node
   * @param inputs A dictionary of named Tensors that comprise this node's
   * inputs.
   * @param output This node's output Tensor
   */
  constructor(
      public graph: Graph, public name: string,
      public inputs: {[name: string]: Tensor}, public output: Tensor) {
    this.id = Node.nextID++;
    output.node = this;
  }
  abstract validate(): void;
  id: number;
  private static nextID = 0;
}

/**
 * VariableNode represents a variable, a user-provided NDArray that's
 * adjusted during backpropagation training.
 *
 * @hidden
 */
export class VariableNode extends Node {
  constructor(graph: Graph, name: string, public data: NDArray) {
    super(graph, name, {}, new Tensor(data.shape));
  }
  validate() {
    util.assert(
        this.data != null,
        'Error adding variable op: Data for variable \'' + this.name +
            '\' is null or undefined');
  }
}

/**
 * PlaceholderNode represents a placeholder, a user-provided NDArray
 * that's used as immutable input during inference and training.
 *
 * @hidden
 */
export class PlaceholderNode extends Node {
  constructor(graph: Graph, name: string, shape: number[]) {
    super(graph, name, {}, new Tensor(shape));
  }
  validate() {}
}

/**
 * ConstantNode represents a constant value in the graph.
 *
 * @hidden
 */
export class ConstantNode extends Node {
  constructor(graph: Graph, public data: NDArray) {
    super(graph, 'Constant', {}, new Tensor(data.shape));
  }
  validate() {
    util.assert(
        this.data != null,
        'Error adding constant: data for placeholder \'' + this.name +
            '\' is null or undefined');
  }
}

/**
 * ReshapeNode represents a reshape operation in the graph.
 *
 * @hidden
 */
export class ReshapeNode extends Node {
  static readonly X = 'x';
  constructor(
      graph: Graph, public name: string, private x: Tensor,
      private shape: number[]) {
    super(graph, name, {x}, new Tensor(shape));
  }
  validate() {
    const xSize = util.sizeFromShape(this.x.shape);
    const shapeSize = util.sizeFromShape(this.shape);
    util.assert(
        xSize === shapeSize,
        `Error making reshape operation: input to reshape '${this.name}'` +
            ` of shape (${this.x.shape}) does not match size of ` +
            `requested shape ${this.shape}.`);
  }
}

/**
 * LinearCombinationNode represents a linear combination of two tensors.
 * @hidden
 */
export class FusedLinearCombinationNode extends Node {
  static readonly T1 = 't1';
  static readonly T2 = 't2';
  static readonly C1 = 'c1';
  static readonly C2 = 'c2';
  constructor(
      graph: Graph, private t1: Tensor, private t2: Tensor, private c1: Tensor,
      private c2: Tensor) {
    super(graph, 'Linear Combination', {t1, t2, c1, c2}, new Tensor(t1.shape));
  }

  validate() {
    util.assertShapesMatch(this.t1.shape, this.t2.shape);
    if (!util.isScalarShape(this.c1.shape)) {
      throw new Error(
          'Error adding fusedLinearCombination: c1 is not a scalar, got ' +
          `shape: ${this.c1.shape}`);
    }
    if (!util.isScalarShape(this.c2.shape)) {
      throw new Error(
          'Error adding fusedLinearCombination: c2 is not a scalar, got ' +
          `shape: ${this.c2.shape}`);
    }
  }
}

/**
 * @hidden
 */
export class AddNode extends Node {
  static readonly T1 = 't1';
  static readonly T2 = 't2';

  constructor(graph: Graph, private t1: Tensor, private t2: Tensor) {
    super(
        graph, 'Add', {t1, t2},
        new Tensor(util.sizeFromShape(t1.shape) === 1
            ? t2.shape
            : (t1.shape.length < t2.shape.length ? t2.shape : t1.shape)));
  }

  validate() {
    util.assert(
        util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape) ||
            (this.t1.shape.length === 2 && this.t2.shape.length === 1 &&
                this.t1.shape[1] === this.t2.shape[0]) ||
            (this.t1.shape.length === 1 && this.t2.shape.length === 2 &&
                this.t1.shape[0] === this.t2.shape[1]),
        'Error adding add operation op: one of inputs must be scalar, ' +
            `shapes ${this.t1.shape} and ${this.t2.shape} must match,` +
            'or one of them can be broadcasted (2D and 1D).');
  }
}

/**
 * @hidden
 */
export class SubtractNode extends Node {
  static readonly T1 = 't1';
  static readonly T2 = 't2';

  constructor(graph: Graph, private t1: Tensor, private t2: Tensor) {
    super(
        graph, 'Subtract', {t1, t2},
        new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape));
  }

  validate() {
    util.assert(
        util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape),
        'Error adding subtract op: one of inputs must be scalar or the ' +
            `shapes ${this.t1.shape} and ${this.t2.shape} must match.`);
  }
}

/**
 * @hidden
 */
export class MultiplyNode extends Node {
  static readonly T1 = 't1';
  static readonly T2 = 't2';

  constructor(graph: Graph, private t1: Tensor, private t2: Tensor) {
    super(
        graph, 'Multiply', {t1, t2},
        new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape));
  }

  validate() {
    util.assert(
        util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape),
        'Error adding multiply op: one of inputs must be scalar or the ' +
            `shapes ${this.t1.shape} and ${this.t2.shape} must match.`);
  }
}

/**
 * @hidden
 */
export class DivideNode extends Node {
  static readonly T1 = 't1';
  static readonly T2 = 't2';

  constructor(graph: Graph, private t1: Tensor, private t2: Tensor) {
    super(
        graph, 'Divide', {t1, t2},
        new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape));
  }

  validate() {
    util.assert(
        util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape),
        'Error adding divide op: one of inputs must be scalar or the ' +
            `shapes ${this.t1.shape} and ${this.t2.shape} must match.`);
  }
}

/**
 * @hidden
 */
export class ReduceSumNode extends Node {
  static readonly X = 'x';

  constructor(graph: Graph, x: Tensor) {
    super(graph, 'ReduceSum', {x}, new Tensor([]));
  }

  validate() {}
}

/**
 * Concat1DNode represents a 1D concatenation of two tensors along an axis.
 * @hidden
 */
export class Concat1DNode extends Node {
  static readonly X1 = 'x1';
  static readonly X2 = 'x2';
  constructor(
      graph: Graph, x1: Tensor, x2: Tensor) {
    super(
        graph, 'Concat1D', {x1, x2},
        new Tensor(concat_util.computeOutShape1D(x1.shape, x2.shape)));
  }
  validate() {}
}

/**
 * Concat2DNode represents a 2D concatenation of two tensors along an axis.
 * @hidden
 */
export class Concat2DNode extends Node {
  static readonly X1 = 'x1';
  static readonly X2 = 'x2';
  static readonly AXIS = 'axis';
  constructor(
      graph: Graph, private x1: Tensor, private x2: Tensor,
      public axis: number) {
    super(
        graph, 'Concat2D', {x1, x2},
        new Tensor(concat_util.computeOutShape(x1.shape, x2.shape, axis)));
  }
  validate() {
    concat_util.assertParams(this.x1.shape, this.x2.shape, this.axis);
  }
}

/**
 * Concat3DNode represents a 3D concatenation of two tensors along an axis.
 * @hidden
 */
export class Concat3DNode extends Node {
  static readonly X1 = 'x1';
  static readonly X2 = 'x2';
  static readonly AXIS = 'axis';
  constructor(
      graph: Graph, private x1: Tensor, private x2: Tensor,
      public axis: number) {
    super(
        graph, 'Concat3D', {x1, x2},
        new Tensor(concat_util.computeOutShape(x1.shape, x2.shape, axis)));
  }
  validate() {
    concat_util.assertParams(this.x1.shape, this.x2.shape, this.axis);
  }
}

/**
 * Concat4DNode represents a 4D concatenation of two tensors along an axis.
 * @hidden
 */
export class Concat4DNode extends Node {
  static readonly X1 = 'x1';
  static readonly X2 = 'x2';
  static readonly AXIS = 'axis';
  constructor(
      graph: Graph, private x1: Tensor, private x2: Tensor,
      public axis: number) {
    super(
        graph, 'Concat4D', {x1, x2},
        new Tensor(concat_util.computeOutShape(x1.shape, x2.shape, axis)));
  }
  validate() {
    concat_util.assertParams(this.x1.shape, this.x2.shape, this.axis);
  }
}

function getMatMulOutputShape(x1Shape: number[], x2Shape: number[]): number[] {
  if (x1Shape.length === 1 && x2Shape.length === 1) {
    return [1];
  } else if (x1Shape.length === 1 && x2Shape.length === 2) {
    return [x2Shape[1]];
  } else if (x1Shape.length === 2 && x2Shape.length === 1) {
    return [x1Shape[0]];
  }
  return [x1Shape[0], x2Shape[1]];
}

/**
 * MatMulNode represents a fully connected layer in the graph.
 * @hidden
 */
export class MatMulNode extends Node {
  static readonly X1 = 'x1';
  static readonly X2 = 'x2';
  constructor(graph: Graph, private x1: Tensor, private x2: Tensor) {
    super(
        graph, 'MatMul', {x1, x2},
        new Tensor(getMatMulOutputShape(x1.shape, x2.shape)));
  }

  validate() {
    if (this.x1.shape.length === 2 && this.x2.shape.length === 2) {
      util.assert(
          this.x1.shape[1] === this.x2.shape[0],
          'Error adding matmul op: inner shapes of matrices with shapes ' +
              `${this.x1.shape} and ${this.x2.shape} must match.`);
    } else if (this.x1.shape.length === 2 && this.x2.shape.length === 1) {
      util.assert(
          this.x1.shape[1] === this.x2.shape[0],
          'Error adding matmul op: second dimension of matrix with shape ' +
              this.x1.shape.toString() +
              ` must match size of vector with shape ${this.x2.shape}.`);
    } else if (this.x1.shape.length === 1 && this.x2.shape.length === 2) {
      util.assert(
          this.x1.shape[0] === this.x2.shape[0],
          `Error adding matmul op: size of vector with shape ${this.x1.shape}` +
              ` must match first dimension of matrix with ` +
              `shape ${this.x2.shape}.`);
    } else {
      throw new Error(
          'Error adding matmul op: inputs must be vectors or matrices.');
    }
  }
}

/**
 * Convolution2DNode represents a 2d convolution operation in the graph.
 * @hidden
 */
export class Convolution2DNode extends Node {
  static readonly X = 'x';
  static readonly W = 'w';
  static readonly B = 'b';
  constructor(
      graph: Graph, private x: Tensor, private w: Tensor, private b: Tensor,
      public fieldSize: number, public outputDepth: number, public stride = 1,
      public zeroPad?: number) {
    super(
        graph, 'Convolution 2D', {x, w, b},
        new Tensor(conv_util.computeOutputShape3D(
            x.shape as [number, number, number], fieldSize, outputDepth, stride,
            zeroPad)));
  }
  validate() {
    util.assert(
        this.x.shape.length === 3,
        'Error adding conv2d op: input must be of rank 3, but got shape: ' +
            `${this.x.shape}.`);
    util.assert(
        this.w.shape.length === 4,
        'Error adding conv2d op: weights must be of rank 4, but got shape: ' +
            `${this.w.shape}.`);
    util.assert(
        this.b.shape.length === 1,
        'Error adding conv2d op: biases must be of rank 1, but got shape: ' +
            `${this.b.shape}.`);

    util.assert(
        this.x.shape[2] === this.w.shape[2],
        `Error adding conv2d op: depth of input (${this.x.shape[2]}) ` +
            `must match input depth for weights (${this.w.shape[2]}).`);
  }
}

/**
 * MaxPoolNode represents a 2d max pool operation in the graph.
 * @hidden
 */
export class MaxPoolNode extends Node {
  static readonly X = 'x';
  constructor(
      graph: Graph, private x: Tensor, public fieldSize: number,
      public stride = 1, public zeroPad?: number) {
    super(
        graph, 'Max pool', {x},
        new Tensor(conv_util.computeOutputShape3D(
            x.shape as [number, number, number], fieldSize, x.shape[2], stride,
            zeroPad)));
  }
  validate() {
    util.assert(
        this.x.shape.length === 3,
        'Error adding maxPool op: input must be of rank 3, but got shape: ' +
            `${this.x.shape}.`);
  }
}

/**
 * ReLUNode represents a ReLU operation in the graph.
 * @hidden
 */
export class ReLUNode extends Node {
  static readonly X = 'x';
  constructor(graph: Graph, x: Tensor) {
    super(graph, 'ReLU', {x}, new Tensor(x.shape));
  }
  validate() {}
}

/**
 * LeakyReLUNode represents a LeakyReLU operation in the graph.
 * @hidden
 */
export class LeakyReLUNode extends Node {
  static readonly X = 'x';
  public alpha: number;
  constructor(graph: Graph, x: Tensor, alpha: number) {
    super(graph, 'LeakyReLU', {x}, new Tensor(x.shape));
    this.alpha = alpha;
  }
  validate() {}
}

/**
 * PReLUNode represents a PReLU operation in the graph.
 * @hidden
 */
export class PReLUNode extends Node {
  static readonly X = 'x';
  static readonly ALPHA = 'alpha';

  constructor(graph: Graph, private x: Tensor, private alpha: Tensor) {
    super(graph, 'PReLU', {x, alpha}, new Tensor(x.shape));
  }

  validate() {
    util.assert(util.arraysEqual(this.x.shape, this.alpha.shape),
      'Error adding pRelu op: the ' +
        `shapes x: ${this.x.shape} and alpha: ${this.alpha.shape} must match.`);
  }
}

export class EluNode extends Node {
  static readonly X = 'x';
  constructor(graph: Graph, x: Tensor) {
    super(graph, 'Elu', {x}, new Tensor(x.shape));
  }
  validate() {}
}

/**
 * ExpNode represents a Exponentiation operation in the graph.
 * @hidden
 */
export class ExpNode extends Node {
  static readonly X = 'x';
  constructor(graph: Graph, x: Tensor) {
    super(graph, 'Exp', {x}, new Tensor(x.shape));
  }
  validate() {}
}

/**
 * LogNode represents a Exponentiation operation in the graph.
 * @hidden
 */
export class LogNode extends Node {
  static readonly X = 'x';
  constructor(graph: Graph, x: Tensor) {
    super(graph, 'Log', {x}, new Tensor(x.shape));
  }
  validate() {}
}

/**
 * TanHNode represents a tanh operation in the graph.
 * @hidden
 */
export class TanHNode extends Node {
  static readonly X = 'x';
  constructor(graph: Graph, x: Tensor) {
    super(graph, 'TanH', {x}, new Tensor(x.shape));
  }
  validate() {}
}

/**
 * SigmoidNode represents a sigmoid operation in the graph.
 * @hidden
 */
export class SigmoidNode extends Node {
  static readonly X = 'x';
  constructor(graph: Graph, x: Tensor) {
    super(graph, 'Sigmoid', {x}, new Tensor(x.shape));
  }
  validate() {}
}

/**
 * Square node represents an element-wise square operation in the graph.
 * @hidden
 */
export class SquareNode extends Node {
  static readonly X = 'x';
  constructor(graph: Graph, x: Tensor) {
    super(graph, 'Square', {x}, new Tensor(x.shape));
  }
  validate() {}
}

/**
 * SoftmaxCrossEntropyCostNode represents a softmax cross-entropy cost operation
 * in the graph.
 * @hidden
 */
export class SoftmaxCrossEntropyCostNode extends Node {
  static readonly X = 'x';
  static readonly TARGET = 'target';
  constructor(graph: Graph, private x: Tensor, private target: Tensor) {
    super(graph, 'SoftmaxCrossEntropyCost', {x, target}, new Tensor([]));
  }
  validate() {
    util.assert(
        util.arraysEqual(this.x.shape, this.target.shape),
        `Error adding softmaxCrossEntropyCost op: x shape (${this.x.shape}) ` +
            `must match target shape (${this.target.shape}).`);
  }
}

/**
 * @hidden
 */
export class SoftmaxNode extends Node {
  static readonly X = 'x';

  constructor(graph: Graph, private x: Tensor) {
    super(graph, 'Softmax', {x}, new Tensor(x.shape));
  }
  validate() {
    util.assert(
        this.x.shape.length === 1,
        'The input to a softmax must be a 1-D tensor');
    util.assert(
        this.x.shape[0] >= 2,
        'The input to a softmax must have at least 2 values');
  }
}

/**
 * MeanSquaredCostNode represents a mean squared cost operation
 * in the graph.
 *
 * @hidden
 */
export class MeanSquaredCostNode extends Node {
  static readonly LABEL = 'label';
  static readonly PREDICTION = 'prediction';
  constructor(graph: Graph, private label: Tensor, private prediction: Tensor) {
    super(graph, 'Mean Squared Cost', {label, prediction}, new Tensor([]));
  }
  validate() {
    util.assert(
        util.arraysEqual(this.label.shape, this.prediction.shape),
        `Error adding meanSquaredCost op: label shape (${this.label.shape}) ` +
            `must match prediction shape (${this.prediction.shape}).`);
  }
}

/**
 * ArgMaxNode represents an argmax operation in the graph.
 * @hidden
 */
export class ArgMaxNode extends Node {
  static readonly X = 'x';
  constructor(graph: Graph, public x: Tensor) {
    super(graph, 'ArgMax', {x}, new Tensor([1]));
  }
  validate() {
    util.assert(
        util.sizeFromShape(this.x.shape) > 0,
        'Error adding argmax op: input tensor must have at least one entry.');
  }
}

/**
 * ArgMaxEqualsNode represents a argmax equals operation in the graph.
 * @hidden
 */
export class ArgMaxEqualsNode extends Node {
  static readonly X1 = 'x1';
  static readonly X2 = 'x2';
  constructor(graph: Graph, private x1: Tensor, private x2: Tensor) {
    super(graph, 'ArgMaxEquals', {x1, x2}, new Tensor([1]));
  }
  validate() {
    util.assert(
        util.arraysEqual(this.x1.shape, this.x2.shape),
        `Error adding ArgMaxEquals op: x1 shape (${this.x1.shape}) ` +
            `must match x2 shape (${this.x2.shape}).`);
  }
}

/**
 * @hidden
 */
export type ArrayData =
    NDArray|number|number[]|number[][]|number[][][]|number[][][][];
