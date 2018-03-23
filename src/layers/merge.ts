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

import {Tensor} from '@tensorflow/tfjs-core';
import * as _ from 'underscore';

import * as K from '../backend/deeplearnjs_backend';
import {Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {Shape} from '../types';
import * as generic_utils from '../utils/generic_utils';
import * as mathUtils from '../utils/math_utils';

/**
 * Generic merge layer for elementwise merge functions.
 *
 * Used to implement `Sum`, `Average`, etc.
 */
export class Merge extends Layer {
  protected reshapeRequired: boolean;

  constructor(config?: LayerConfig) {
    super(config || {});
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
      inputShape = [generic_utils.getExactlyOneShape(inputShape)];
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
    batchSizes = _.uniq(batchSizes);
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
    if (!_.contains(inputShape, null) && _.uniq(allRanks).length === 1) {
      this.reshapeRequired = false;
    } else {
      this.reshapeRequired = true;
    }
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    inputs = inputs as Tensor[];
    if (this.reshapeRequired) {
      const reshapedInputs: Tensor[] = [];
      const inputDims = inputs.map(input => K.ndim(input));
      if (!_.contains(inputDims, null)) {
        // If ranks of all inputs are available, we simply expand each of them
        // at axis=1 until all of them have the same rank.
        const maxNDim = _.max(inputDims);
        for (let x of inputs) {
          const xNDim = K.ndim(x);
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
          const xNDim = K.ndim(x);
          if (xNDim == null) {
            const xShape = K.shape(x);
            const batchSize = xShape[0];
            const newShape = xShape.slice(1).concat([batchSize]);
            let xTransposed = K.reshape(
                x, [batchSize].concat(mathUtils.arrayProd(xShape.slice(1))));
            xTransposed = K.permuteDimensions(xTransposed, [1, 0]);
            xTransposed = K.reshape(xTransposed, newShape);
            reshapedInputs.push(xTransposed);
            transposed = true;
          } else if (xNDim > 1) {
            const dims = _.range(1, xNDim).concat([0]);
            reshapedInputs.push(K.permuteDimensions(x, dims));
            transposed = true;
          } else {
            // We don't transpose inputs if they are 1D vectors or scalars.
            reshapedInputs.push(x);
          }
        }
        let y = this.mergeFunction(reshapedInputs);
        const yNDim = K.ndim(y);
        if (transposed) {
          // If inputs have been transposed, we have to transpose the output
          // too.
          if (yNDim == null) {
            const yShape = K.shape(y);
            const yNDim = yShape.length;
            const batchSize = yShape[yNDim - 1];
            const newShape =
                [batchSize].concat(yShape.slice(0, yShape.length - 1));
            y = K.reshape(
                K.permuteDimensions(K.reshape(y, [-1, batchSize]), [1, 0]),
                newShape);
          } else if (yNDim > 1) {
            const dims = [yNDim - 1].concat(_.range(0, yNDim - 1));
            y = K.permuteDimensions(y, dims);
          }
        }
        return y;
      }
    } else {
      return this.mergeFunction(inputs);
    }
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
    batchSizes = _.uniq(batchSizes);
    if (batchSizes.length === 1) {
      outputShape = batchSizes.concat(outputShape);
    } else {
      outputShape = [null].concat(outputShape);
    }
    return outputShape;
  }

  // TODO(cais): Implement computeMask();
}

/**
 * Layer that adds a list of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape).
 *
 */
// TODO(cais): Add examples.
export class Add extends Merge {
  constructor(config?: LayerConfig) {
    super(config);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = K.zeros(inputs[0].shape);
    for (const input of inputs) {
      output = K.add(output, input);
    }
    return output;
  }
}
generic_utils.ClassNameMap.register('Add', Add);

//  TODO(cais): Add class Subtract.

/**
 * Layer that multiplies (element-wise) an Array of inputs.
 *
 * It takes as input an Array of tensors, all of the same
 * shape, and returns a single tensor (also of the same shape).
 */
export class Multiply extends Merge {
  constructor(config?: LayerConfig) {
    super(config);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = K.ones(inputs[0].shape);
    for (const input of inputs) {
      output = K.multiply(output, input);
    }
    return output;
  }
}
generic_utils.ClassNameMap.register('Multiply', Multiply);

/**
 * Layer that averages a list of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape).
 */
export class Average extends Merge {
  constructor(config?: LayerConfig) {
    super(config);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = K.zeros(inputs[0].shape);
    for (const input of inputs) {
      output = K.add(output, input);
    }
    return K.scalarTimesArray(K.getScalar(1 / inputs.length), output);
  }
}
generic_utils.ClassNameMap.register('Average', Average);

/**
 * Layer that computes the maximum (element-wise) a list of inputs.
 *
 * It takes as input a list of tensors, all of the same shape and returns a
 * single tensor (also of the same shape).
 */
export class Maximum extends Merge {
  constructor(config?: LayerConfig) {
    super(config);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = K.zeros(inputs[0].shape);
    for (const input of inputs) {
      output = K.maximum(output, input);
    }
    return output;
  }
}
generic_utils.ClassNameMap.register('Maximum', Maximum);

/**
 * Layer that computes the minimum (element-wise) a list of inputs.
 *
 * It takes as input a list of tensors, all of the same shape and returns a
 * single tensor (also of the same shape).
 */
export class Minimum extends Merge {
  constructor(config?: LayerConfig) {
    super(config);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = K.zeros(inputs[0].shape);
    for (const input of inputs) {
      output = K.minimum(output, input);
    }
    return output;
  }
}
generic_utils.ClassNameMap.register('Minimum', Minimum);

export interface ConcatenateLayerConfig extends LayerConfig {
  /**
   * Axis along which to concatenate.
   */
  axis?: number;
}

/**
 * Layer that concatenates a list of inputs.
 *
 * It takes a list of tensors, all of the same shape except for the
 * concatenation axis, and returns a single tensor, the concatenation
 * of all inputs.
 */
export class Concatenate extends Merge {
  readonly DEFAULT_AXIS = -1;
  private readonly axis: number;

  constructor(config: ConcatenateLayerConfig) {
    super(config);
    this.axis = config.axis == null ? this.DEFAULT_AXIS : config.axis;
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
        if (_.isEqual(shape, shapeWithoutConcatAxis)) {
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
    return K.concatenate(inputs, this.axis);
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

  // TODO(cais): Implement computeMask();
  // TODO(cais): Add getConfig();
}
generic_utils.ClassNameMap.register('Concatenate', Concatenate);

// TODO(cais): Add class Dot.

// TODO(cais): Add functional interfaces for the merge layers.
