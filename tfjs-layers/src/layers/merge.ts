/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * TensorFlow.js Layers: Merge Layers.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {serialization, Tensor, tidy, util} from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import {Layer, LayerArgs, SymbolicTensor} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {Shape} from '../keras_format/common';
import {l2Normalize} from '../losses';
import {Kwargs} from '../types';
import * as generic_utils from '../utils/generic_utils';
import * as mathUtils from '../utils/math_utils';
import {getExactlyOneShape} from '../utils/types_utils';

/**
 * Generic Merge layer for element-wise merge functions.
 *
 * Used to implement `Sum`, `Average`, `Concatenate`, etc.
 */
export abstract class Merge extends Layer {
  protected reshapeRequired: boolean;

  constructor(args?: LayerArgs) {
    super(args || {});
    this.supportsMasking = true;
  }

  /**
   * Logic for merging multiple tensors, to be overridden by subclasses.
   * @param inputs
   */
  protected mergeFunction(inputs: Tensor[]): Tensor {
    throw new NotImplementedError();
  }

  /**
   * Computes the shape of the result of an elementwise operation.
   *
   * @param shape1: Shape of the first tensor.
   * @param shape2: Shape of the second tensor.
   * @returns Expected output shape when an elementwise operation is carried
   *   out on 2 tensors with shapes `shape1` and `shape2`.
   * @throws ValueError: If `shape1` and `shape2` are not compatible for
   *   element-wise operations.
   */
  private computeElementwiseOpOutputShape(shape1: Shape, shape2: Shape): Shape {
    if (shape1 == null || shape2 == null) {
      return null;
    } else if (shape1.length < shape2.length) {
      return this.computeElementwiseOpOutputShape(shape2, shape1);
    } else if (shape2.length === 0) {
      return shape1;
    }
    const outputShape: Shape = shape1.slice(0, shape1.length - shape2.length);
    for (let k = 0; k < shape2.length; ++k) {
      const i = shape1[shape1.length - shape2.length + k];
      const j = shape2[k];
      if (i == null || j == null || i < 0 || j < 0) {
        outputShape.push(null);
      } else if (i === 1) {
        outputShape.push(j);
      } else if (j === 1) {
        outputShape.push(i);
      } else {
        if (i !== j) {
          throw new ValueError(
              'Operands could not be broadcast together with shapes ' +
              JSON.stringify(shape1) + ' ' + JSON.stringify(shape2));
        }
        outputShape.push(i);
      }
    }
    return outputShape;
  }

  build(inputShape: Shape|Shape[]): void {
    // Used purely for shape validation.
    if (Array.isArray(inputShape) && !Array.isArray(inputShape[0])) {
      // Make sure that inputShape is an Array of shape.
      inputShape = [getExactlyOneShape(inputShape)];
    }
    inputShape = inputShape as Shape[];
    if (inputShape.length < 2) {
      throw new ValueError(
          'A merge layer should be called on an Array of at least 2 inputs.' +
          ` Got ${inputShape.length} input(s).`);
    }

    // Make sure that there is at most one unique batch size among the input
    // shapes.
    let batchSizes: number[] = [];
    for (const shape of inputShape) {
      if (shape != null && shape[0] !== null) {
        batchSizes.push(shape[0]);
      }
    }
    batchSizes = generic_utils.unique(batchSizes);
    if (batchSizes.length > 1) {
      throw new ValueError(
          `Can not merge tensors with different batch sizes. ` +
          `Got tensors with shapes: ${JSON.stringify(inputShape)}.`);
    }

    let outputShape: Shape =
        inputShape[0] == null ? null : inputShape[0].slice(1);
    for (let i = 1; i < inputShape.length; ++i) {
      const shape = inputShape[i] == null ? null : inputShape[i].slice(1);
      outputShape = this.computeElementwiseOpOutputShape(outputShape, shape);
    }
    // If the inputs have different ranks, we have to reshape them to make them
    // broadcastable.
    const allRanks = inputShape.map(shape => shape.length);
    if (inputShape.indexOf(null) === -1 &&
        generic_utils.unique(allRanks).length === 1) {
      this.reshapeRequired = false;
    } else {
      this.reshapeRequired = true;
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = inputs as Tensor[];
      if (this.reshapeRequired) {
        const reshapedInputs: Tensor[] = [];
        const inputDims = inputs.map(input => input.rank);
        if (inputDims.indexOf(null) === -1) {
          // If ranks of all inputs are available, we simply expand each of them
          // at axis=1 until all of them have the same rank.
          const maxNDim = mathUtils.max(inputDims);
          for (let x of inputs) {
            const xNDim = x.rank;
            for (let k = 0; k < maxNDim - xNDim; ++k) {
              x = K.expandDims(x, 1);
            }
            reshapedInputs.push(x);
          }
          return this.mergeFunction(reshapedInputs);
        } else {
          // Transpose all inputs so that batch size is the last dimension.
          // [batchSize, dim1, dim2, ...] -> [dim1, dim2, ..., batchSize]
          let transposed = false;
          for (const x of inputs) {
            const xNDim = x.rank;
            if (xNDim == null) {
              const xShape = x.shape;
              const batchSize = xShape[0];
              const newShape = xShape.slice(1).concat([batchSize]);
              let xTransposed = x.reshape(
                  [batchSize].concat(mathUtils.arrayProd(xShape.slice(1))));
              xTransposed = tfc.transpose(xTransposed, [1, 0]);
              xTransposed = xTransposed.reshape(newShape);
              reshapedInputs.push(xTransposed);
              transposed = true;
            } else if (xNDim > 1) {
              const dims = mathUtils.range(1, xNDim).concat([0]);
              reshapedInputs.push(tfc.transpose(x, dims));
              transposed = true;
            } else {
              // We don't transpose inputs if they are 1D vectors or scalars.
              reshapedInputs.push(x);
            }
          }
          let y = this.mergeFunction(reshapedInputs);
          const yNDim = y.rank;
          if (transposed) {
            // If inputs have been transposed, we have to transpose the output
            // too.
            if (yNDim == null) {
              const yShape = y.shape;
              const yNDim = yShape.length;
              const batchSize = yShape[yNDim - 1];
              const newShape =
                  [batchSize].concat(yShape.slice(0, yShape.length - 1));
              y = tfc.transpose(y.reshape([-1, batchSize]), [1, 0])
                      .reshape(newShape);
            } else if (yNDim > 1) {
              const dims = [yNDim - 1].concat(mathUtils.range(0, yNDim - 1));
              y = tfc.transpose(y, dims);
            }
          }
          return y;
        }
      } else {
        return this.mergeFunction(inputs);
      }
    });
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = inputShape as Shape[];
    let outputShape: Shape;
    if (inputShape[0] == null) {
      outputShape = null;
    } else {
      outputShape = inputShape[0].slice(1);
    }
    for (let i = 1; i < inputShape.length; ++i) {
      const shape = inputShape[i] == null ? null : inputShape[i].slice(1);
      outputShape = this.computeElementwiseOpOutputShape(outputShape, shape);
    }

    let batchSizes: number[] = [];
    for (const shape of inputShape) {
      if (shape != null && shape[0] !== null) {
        batchSizes.push(shape[0]);
      }
    }
    batchSizes = generic_utils.unique(batchSizes);
    if (batchSizes.length === 1) {
      outputShape = batchSizes.concat(outputShape);
    } else {
      outputShape = [null].concat(outputShape);
    }
    return outputShape;
  }

  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor {
    return tfc.tidy(() => {
      if (mask == null) {
        return null;
      }
      if (!Array.isArray(mask)) {
        throw new ValueError('`mask` should be an Array');
      }
      if (!Array.isArray(inputs)) {
        throw new ValueError('`inputs` should be an Array');
      }
      if (mask.length !== inputs.length) {
        throw new ValueError(
            `The Array 'inputs' and 'mask' are expected to have the same ` +
            `length, but have different lengths ` +
            `(${inputs.length} vs ${mask.length})`);
      }
      if (mask.every(m => m == null)) {
        return null;
      }
      mask = mask.map(m => m == null ? m : tfc.expandDims(m, 0));
      let output = mask[0];
      for (let i = 1; i < mask.length - 1; ++i) {
        output = tfc.logicalAnd(output, mask[i]);
      }
      return output;
    });
  }
}

export class Add extends Merge {
  /** @nocollapse */
  static className = 'Add';
  constructor(args?: LayerArgs) {
    super(args);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    return tidy(() => {
      let output = inputs[0].clone();
      for (let i = 1; i < inputs.length; ++i) {
        output = tfc.add(output, inputs[i]);
      }
      return output;
    });
  }
}
serialization.registerClass(Add);

/**
 * Calculate the element-wise sum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Add` layer, by using no input argument
 *    or a single configuration argument. The resultant `Add` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const addLayer = tf.layers.add();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = addLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.add([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.add([input1, input2]).print();
 * // Gives [[11, 22], [33, 44]].
 *
 */
export function add(config?: SymbolicTensor[]|Tensor[]|LayerArgs): Layer|
    SymbolicTensor|Tensor {
  if (Array.isArray(config)) {
    const layer = new Add({});
    return layer.apply(config) as SymbolicTensor | Tensor;
  } else {
    return new Add(config);
  }
}

export class Multiply extends Merge {
  /** @nocollapse */
  static className = 'Multiply';
  constructor(args?: LayerArgs) {
    super(args);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    return tidy(() => {
      let output = inputs[0].clone();
      for (let i = 1; i < inputs.length; ++i) {
        output = tfc.mul(output, inputs[i]);
      }
      return output;
    });
  }
}
serialization.registerClass(Multiply);

/**
 * Calculate the element-wise product of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Multiply` layer, by using no input argument
 *    or a single configuration argument. The resultant `Multiply` layer can
 *    then be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const multiplyLayer = tf.layers.multiply();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = multiplyLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.multiply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.multiply([input1, input2]).print();
 * // Gives [[10, 40], [90, 160]].
 *
 */
export function multiply(config?: SymbolicTensor[]|Tensor[]|LayerArgs): Layer|
    SymbolicTensor|Tensor {
  if (Array.isArray(config)) {
    const layer = new Multiply({});
    return layer.apply(config) as SymbolicTensor | Tensor;
  } else {
    return new Multiply(config);
  }
}

export class Average extends Merge {
  /** @nocollapse */
  static className = 'Average';
  constructor(args?: LayerArgs) {
    super(args);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    return tidy(() => {
      let output = inputs[0].clone();
      for (let i = 1; i < inputs.length; ++i) {
        output = tfc.add(output, inputs[i]);
      }
      return tfc.mul(1 / inputs.length, output);
    });
  }
}
serialization.registerClass(Average);

/**
 * Calculate the element-wise arithmetic mean of inputs, which all have the same
 * shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Average` layer, by using no input argument
 *    or a single configuration argument. The resultant `Average` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const averageLayer = tf.layers.average();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = averageLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.average([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.average([input1, input2]).print();
 * // Gives [[5.5, 11], [16.5, 22]].
 *
 */
export function average(config?: SymbolicTensor[]|Tensor[]|LayerArgs): Layer|
    SymbolicTensor|Tensor {
  if (Array.isArray(config)) {
    const layer = new Average({});
    return layer.apply(config) as SymbolicTensor | Tensor;
  } else {
    return new Average(config);
  }
}

export class Maximum extends Merge {
  /** @nocollapse */
  static className = 'Maximum';
  constructor(args?: LayerArgs) {
    super(args);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    return tidy(() => {
      let output = inputs[0];
      for (let i = 1; i < inputs.length; ++i) {
        output = tfc.maximum(output, inputs[i]);
      }
      return output;
    });
  }
}
serialization.registerClass(Maximum);

/**
 * Calculate the element-wise maximum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Maximum` layer, by using no input argument
 *    or a single configuration argument. The resultant `Maximum` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const maximumLayer = tf.layers.maximum();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = maximumLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.maximum([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 20, 3, 40], [2, 2]);
 * const input2 = tf.tensor2d([10, 2, 30, 4], [2, 2]);
 * tf.layers.maximum([input1, input2]).print();
 * // Gives [[10, 20], [30, 40]].
 *
 */
export function maximum(config?: SymbolicTensor[]|Tensor[]|LayerArgs): Layer|
    SymbolicTensor|Tensor {
  if (Array.isArray(config)) {
    const layer = new Maximum({});
    return layer.apply(config) as SymbolicTensor | Tensor;
  } else {
    return new Maximum(config);
  }
}

export class Minimum extends Merge {
  /** @nocollapse */
  static className = 'Minimum';
  constructor(args?: LayerArgs) {
    super(args);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    return tidy(() => {
      let output = inputs[0];
      for (let i = 1; i < inputs.length; ++i) {
        output = tfc.minimum(output, inputs[i]);
      }
      return output;
    });
  }
}
serialization.registerClass(Minimum);

/**
 * Calculate the element-wise minimum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Minimum` layer, by using no input argument
 *    or a single configuration argument. The resultant `Minimum` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const minimumLayer = tf.layers.minimum();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = minimumLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.minimum([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 20, 3, 40], [2, 2]);
 * const input2 = tf.tensor2d([10, 2, 30, 4], [2, 2]);
 * tf.layers.minimum([input1, input2]).print();
 * // Gives [[1, 2], [3, 4]].
 *
 */
export function minimum(config?: SymbolicTensor[]|Tensor[]|LayerArgs): Layer|
    SymbolicTensor|Tensor {
  if (Array.isArray(config)) {
    const layer = new Minimum({});
    return layer.apply(config) as SymbolicTensor | Tensor;
  } else {
    return new Minimum(config);
  }
}

export declare interface ConcatenateLayerArgs extends LayerArgs {
  /**
   * Axis along which to concatenate.
   */
  axis?: number;
}

export class Concatenate extends Merge {
  /** @nocollapse */
  static className = 'Concatenate';
  readonly DEFAULT_AXIS = -1;
  private readonly axis: number;

  constructor(args?: ConcatenateLayerArgs) {
    super(args);
    if (args == null) {
      args = {};
    }
    this.axis = args.axis == null ? this.DEFAULT_AXIS : args.axis;
    this.supportsMasking = true;
    this.reshapeRequired = false;
  }

  build(inputShape: Shape|Shape[]): void {
    // Used purely for shape validation.]
    if (!(Array.isArray(inputShape) && Array.isArray(inputShape[0])) ||
        inputShape.length === 1) {
      throw new ValueError(
          'A `Concatenate` layer should be called on a list of at least 2 ' +
          'inputs');
    }
    inputShape = inputShape as Shape[];

    let allNoneShape = true;
    for (const shape of inputShape) {
      if (shape != null) {
        allNoneShape = false;
        break;
      }
    }
    if (allNoneShape) {
      return;
    }

    const shapeSet: Shape[] = [];
    for (let i = 0; i < inputShape.length; ++i) {
      const shapeWithoutConcatAxis = inputShape[i].slice();
      shapeWithoutConcatAxis.splice(this.axis, 1);
      let exists = false;
      for (const shape of shapeSet) {
        if (util.arraysEqual(shape, shapeWithoutConcatAxis)) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        shapeSet.push(shapeWithoutConcatAxis);
      }
    }
    if (shapeSet.length > 1) {
      throw new ValueError(
          'A `Concatenate` layer requires inputs with matching shapes ' +
          'except for the concat axis. Got input shapes: ' +
          JSON.stringify(inputShape));
    }
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    return tidy(() => {
      return K.concatenate(inputs, this.axis);
    });
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    if (!(Array.isArray(inputShape) && Array.isArray(inputShape[0]))) {
      throw new ValueError(
          'A `Concatenate` layer should be called on a list of inputs.');
    }
    const inputShapes = inputShape as Shape[];
    const outputShape = inputShapes[0].slice();
    const axis = this.axis < 0 ? outputShape.length + this.axis : this.axis;
    // Porting Note: the line above is because TypeScript doesn't support
    //   negative indices.
    for (const shape of inputShapes.slice(1)) {
      if (outputShape[axis] == null || shape[axis] == null) {
        outputShape[axis] = null;
        break;
      }
      outputShape[axis] += shape[axis];
    }
    return outputShape;
  }

  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor {
    if (mask == null) {
      return null;
    }
    if (!Array.isArray(mask)) {
      throw new ValueError('`mask` should be an array for Concatenate');
    }
    if (!Array.isArray(inputs)) {
      throw new ValueError('`inputs` should be an array for Concatenate');
    }
    if (mask.length !== inputs.length) {
      throw new ValueError(
          `Mismatch in the length of mask (${mask.length}) ` +
          `and the legnth of inputs (${inputs.length})`);
    }
    return tfc.tidy(() => {
      let allNullMasks = true;
      mask.forEach(m => {
        if (m != null) {
          allNullMasks = false;
          return;
        }
      });
      if (allNullMasks) {
        return null;
      }
      const outputMasks: Tensor[] = [];
      for (let i = 0; i < inputs.length; ++i) {
        if (mask[i] == null) {
          // Input is unmasked. Append all 1's to masks.
          outputMasks.push(tfc.onesLike(inputs[i]).asType('bool'));
        } else if (mask[i].rank < inputs[i].rank) {
          // Mask is smaller than the input, expand it.
          outputMasks.push(tfc.expandDims(mask[i], -1));
        } else {
          outputMasks.push(mask[i]);
        }
      }
      const concatenatedMasks = tfc.concat(outputMasks, this.axis);
      return tfc.all(concatenatedMasks, -1, false);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'axis': this.axis,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Concatenate);

/**
 * Concatenate an `Array` of inputs.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Concatenate` layer, by using no input argument
 *    or a single configuration argument. The resultant `Concatenate` layer can
 *    then be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const concatLayer = tf.layers.concatenate();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 3]});
 * const input2 = tf.input({shape: [2, 4]});
 * const output = concatLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 7], with the first dimension as the undetermined batch
 * // dimension and the last dimension as the result of concatenating the
 * // last dimensions of the two inputs.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 3]});
 * const input2 = tf.input({shape: [2, 4]});
 * const output = tf.layers.concatenate([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension and the last dimension as the result of concatenating the
 * // last dimensions of the two inputs.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
 * const input2 = tf.tensor2d([[10, 20], [30, 40]], [2, 2]);
 * tf.layers.concatenate([input1, input2]).print();
 * // Gives [[1, 2, 10, 20], [3, 4, 30, 40]].
 *
 */
export function concatenate(config?: SymbolicTensor[]|Tensor[]|
                            ConcatenateLayerArgs): Layer|SymbolicTensor|Tensor {
  if (Array.isArray(config)) {
    const layer = new Concatenate({});
    return layer.apply(config) as SymbolicTensor | Tensor;
  } else {
    return new Concatenate(config);
  }
}

export declare interface DotLayerArgs extends LayerArgs {
  /**
   * Axis or axes along which the dot product will be taken.
   *
   * Integer or an Array of integers.
   */
  axes: number|[number, number];

  /**
   * Whether to L2-normalize samples along the dot product axis
   * before taking the dot product.
   *
   * If set to `true`, the output of the dot product isthe cosine
   * proximity between the two samples.
   */
  normalize?: boolean;
}

/**
 * Interpretable potentially negative axis index.
 *
 * For example, given axis = -1, and dim = 3, this function will return 2.
 *
 * @param axis The axis index, may be a positive, zero or negative integer.
 * @param dim Total number of dimensions, a positive integer.
 * @returns A non-negative axis index equivalent to the input `axis`.
 */
function interpretAxis(axis: number, dim: number): number {
  while (axis < 0) {
    axis += dim;
  }
  return axis;
}

function batchDot(x: Tensor, y: Tensor, axes: number|[number, number]): Tensor {
  if (x.shape.length > 3 || y.shape.length > 3) {
    throw new NotImplementedError(
        'batchDot is not implemented for tensors of 4D or higher rank yet');
  }
  tfc.util.assert(
      x.shape.length >= 2,
      () => `batchDot requires the rank of x to be >= 2, ` +
          `but got ${x.shape.length}`);
  tfc.util.assert(
      x.shape.length >= 2,
      () => `batchDot requires the rank of y to be >= 2, ` +
          `but got ${y.shape.length}`);

  if (typeof axes === 'number') {
    axes = [axes, axes];
  }

  if (x.dtype === 'complex64' || y.dtype === 'complex64') {
    throw new NotImplementedError(
        'batchDot is not implemented for complex64-type Tensors yet.');
  }

  const xNDim = x.shape.length;
  const yNDim = y.shape.length;
  if (axes == null) {
    // Behave like batchMatmul by default.
    axes = [xNDim - 1, yNDim - 2];
  }
  const axesArray = axes as [number, number];

  return tfc.tidy(() => {
    let diff: number;
    if (xNDim > yNDim) {
      diff = xNDim - yNDim;
      const diffShape: Shape = [];
      for (let i = 0; i < diff; ++i) {
        diffShape.push(1);
      }
      y = y.reshape(y.shape.concat(diffShape));
    } else if (yNDim > xNDim) {
      diff = yNDim - xNDim;
      const diffShape: Shape = [];
      for (let i = 0; i < diff; ++i) {
        diffShape.push(1);
      }
      x = x.reshape(x.shape.concat(diffShape));
    } else {
      diff = 0;
    }

    let out: Tensor;
    if (x.shape.length === 2 && y.shape.length === 2) {
      if (axesArray[0] === axesArray[1]) {
        out = x.mulStrict(y).sum(axesArray[0]);
      } else {
        out = x.transpose([1, 0]).mulStrict(y).sum(axesArray[1]);
      }
    } else {
      const adjX = axesArray[0] !== x.shape.length - 1;
      const adjY = axesArray[1] === y.shape.length - 1;
      out = x.matMul(y, adjX, adjY);
    }

    if (diff > 0) {
      let idx: number;
      if (xNDim > yNDim) {
        idx = xNDim + yNDim - 3;
      } else {
        idx = xNDim - 1;
      }
      const squeezeAxes: number[] = [];
      for (let i = idx; i < idx + diff; ++i) {
        squeezeAxes.push(i);
      }
      out = out.squeeze(squeezeAxes);
    }
    if (out.shape.length === 1) {
      out = out.expandDims(1);
    }
    return out;
  });
}

export class Dot extends Merge {
  /** @nocollapse */
  static className = 'Dot';

  private axes: number|[number, number];
  private normalize: boolean;

  constructor(args: DotLayerArgs) {
    super(args);
    this.axes = args.axes;
    this.normalize = args.normalize == null ? false : args.normalize;
    this.supportsMasking = true;
    this.reshapeRequired = false;
  }

  build(inputShape: Shape|Shape[]): void {
    tfc.util.assert(
        Array.isArray(inputShape) && inputShape.length === 2 &&
            Array.isArray(inputShape[0]) && Array.isArray(inputShape[1]),
        () => 'A `Dot` layer should be called on a list of exactly 2 inputs.');
    const shape1 = inputShape[0] as Shape;
    const shape2 = inputShape[1] as Shape;
    if (shape1.length > 3 || shape2.length > 3) {
      throw new NotImplementedError(
          'Dot layer does not support tensors of 4D or higher rank yet.');
    }

    const axes = this.interpretAxes(shape1, shape2);
    if (shape1[axes[0]] !== shape2[axes[1]]) {
      throw new ValueError(
          `Dimension incompatibility: ` +
          `${shape1[axes[0]]} !== ${shape2[axes[1]]}`);
    }
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    if (inputs.length !== 2) {
      throw new ValueError(
          'A `Dot` layer must be called on exactly 2 inputs, ' +
          `but received ${inputs.length} input(s).`);
    }

    let x1 = inputs[0];
    let x2 = inputs[1];
    let axes: [number, number];
    if (!Array.isArray(this.axes)) {
      axes = [
        interpretAxis(this.axes, x1.shape.length),
        interpretAxis(this.axes, x2.shape.length)
      ];
    } else {
      axes = this.axes.map(
                 (axis, i) => interpretAxis(
                     axis, inputs[i].shape.length)) as [number, number];
    }
    if (this.normalize) {
      x1 = l2Normalize(x1, axes[0]);
      x2 = l2Normalize(x2, axes[1]);
    }
    return batchDot(x1, x2, axes);
  }

  private interpretAxes(shape1: Shape, shape2: Shape): number[] {
    let axes: number[];
    if (!Array.isArray(this.axes)) {
      // `this.axes` is a single integer.
      axes = [
        interpretAxis(this.axes, shape1.length),
        interpretAxis(this.axes, shape2.length)
      ];
    } else {
      // `this.axes` is an Array of integers.
      axes = this.axes;
    }
    return axes;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    tfc.util.assert(
        Array.isArray(inputShape) && inputShape.length === 2 &&
            Array.isArray(inputShape[0]) && Array.isArray(inputShape[1]),
        () => 'A `Dot` layer should be called on a list of exactly 2 inputs.');
    const shape1 = (inputShape[0] as Shape).slice();
    const shape2 = (inputShape[1] as Shape).slice();
    if (shape1.length > 3 || shape2.length > 3) {
      throw new NotImplementedError(
          'Dot layer does not support tensors of 4D or higher rank yet.');
    }

    const axes = this.interpretAxes(shape1, shape2);
    shape1.splice(axes[0], 1);
    shape2.splice(axes[1], 1);
    shape2.splice(0, 1);
    const outputShape = shape1.concat(shape2);
    if (outputShape.length === 1) {
      outputShape.push(1);
    }
    return outputShape;
  }

  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor {
    return null;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'axes': this.axes,
      'normalize': this.normalize
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Dot);

// TODO(cais): Add functional interfaces for the merge layers.
