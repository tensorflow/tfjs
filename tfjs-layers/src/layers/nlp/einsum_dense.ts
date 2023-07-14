/**
 * @license
 * Copyright 2023 Google LLC.
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

/**
 *  TFJS-based einsum dense layer.
 */

/* Original source: keras/layers/core/einsum_dense.py */
import { Tensor, Tensor2D, serialization } from '@tensorflow/tfjs-core';

import { ConstraintIdentifier } from '../../constraints';
import { Layer, LayerArgs } from '../../engine/topology';
import { NotImplementedError } from '../../errors';
import { InitializerIdentifier } from '../../initializers';
import { ActivationIdentifier } from '../../keras_format/activation_config';
import { Shape } from '../../keras_format/common';
import { RegularizerIdentifier } from '../../regularizers';
import { Kwargs } from '../../types';

export declare interface EinsumDenseArgs extends LayerArgs {
  /**
   * An equation describing the einsum to perform. This equation must be a
   * valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
   * `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
   * axis expression sequence.
   */
  equation: string;

  /**
   * The expected shape of the output tensor (excluding the batch dimension and
   * any dimensions represented by ellipses). You can specify None for any
   * dimension that is unknown or can be inferred from the input shape.
   */
  outputShape: Shape;

  /**
   * Activation function to use. If you don't specify anything, no activation
   * is applied (that is, a "linear" activation: `a(x) = x`).
   */
  activation?: ActivationIdentifier;

  /**
   * A string containing the output dimension(s) to apply a bias to. Each
   * character in the `biasAxes` string should correspond to a character
   * in the output portion of the `equation` string.
   */
  biasAxes?: string;

  /**
   * Initializer for the `kernel` weights matrix.
   * Defaults to `"glorotUniform"`.
   */
  kernelInitializer?: InitializerIdentifier;

  /**
   * Initializer for the bias vector.
   * Defaults to `"zeros"`.
   */
  biasInitializer?: InitializerIdentifier;

  /**
   * Regularizer function applied to the `kernel` weights matrix.
   */
  kernelRegularizer?: RegularizerIdentifier;

  /**
   * Regularizer function applied to the bias vector.
   */
  biasRegularizer?: RegularizerIdentifier;

  /**
   * Regularizer function applied to the output of the layer (its "activation").
   */
  activityRegularizer?: RegularizerIdentifier;

  /**
   * Constraint function applied to the `kernel` weights matrix.
   */
  kernelConstraint?: ConstraintIdentifier;

  /**
   * Constraint function applied to the bias vector.
   */
  biasConstraint?: ConstraintIdentifier;
}

export declare interface EinsumDenseOptions {
  /**
   * Pass to override the configured `sequenceLength` of the layer.
   */
  sequenceLength?: number;

  /**
   * Pass `false` to not append a start value for this input.
   * Defaults to true.
   */
  addStartValue?: boolean;

  /**
   * Pass `false` to not append an end value for this input.
   * Defaults to true.
   */
  addEndValue?: boolean;
}

/**
 * A layer that uses `tf.einsum` as the backing computation.
 *
 * This layer can perform einsum calculations of arbitrary dimensionality.
 *
 * Examples:
 *
 * **Biased dense layer with einsums**
 *
 * This example shows how to instantiate a standard Keras dense layer using
 * einsum operations. This example is equivalent to
 * tf.layers.Dense({units: 64, useBias: true})`.
 *
 * const layer = new EinsumDense({
 *    equation: "ab,bc->ac", outputShape: 4, biasAxes: "c"});
 * const inputTensor = tf.input({shape: [32]});
 * const outputTensor = layer.call(inputTensor);
 * console.log(outputTensor);  // [null, 64]
 *
 * **Applying a dense layer to a sequence**
 *
 * This example shows how to instantiate a layer that applies the same dense
 * operation to every element in a sequence. Here, the `outputShape` has two
 * values (since there are two non-batch dimensions in the output); the first
 * dimension in the `outputShape` is `null`, because the sequence dimension
 * `b` has an unknown shape.
 *
 * const layer = new EinsumDense({
 *    equation: "abc,cd->abd", outputShape: [null, 64], biasAxes: "d"});
 * const inputTensor = tf.input({shape: [32, 128]});
 * const outputTensor = layer.call(inputTensor);
 * console.log(outputTensor);  // [null, 32, 64]
 *
 * **Applying a dense layer to a sequence using ellipses**
 *
 * This example shows how to instantiate a layer that applies the same dense
 * operation to every element in a sequence, but uses the ellipsis notation
 * instead of specifying the batch and sequence dimensions.
 *
 * Because we are using ellipsis notation and have specified only one axis, the
 * `outputShape` arg is a single value. When instantiated in this way, the
 * layer can handle any number of sequence dimensions - including the case
 * where no sequence dimension exists.
 *
 * const layer = new EinsumDense({
 *    equation: "...x,xy->...y", outputShape: 64, biasAxes: "y"});
 * const inputTensor = tf.input({shape: [32, 128]});
 * const outputTensor = layer.call(inputTensor);
 * console.log(outputTensor);  // [null, 32, 64]
 */
export class EinsumDense extends Layer {
  /** @nocollapse */
  static readonly className = 'EinsumDense';

  constructor(args: EinsumDenseArgs) {
    super(args);
    throw new NotImplementedError(`Not implmented yet.`);
  }

  override build(inputShape: Shape | Shape[]): void {
    throw new NotImplementedError(
      `Not implmented yet. Uses ${this.analyzeEinsumString}`);
  }

  override getConfig(): serialization.ConfigDict {
    throw new NotImplementedError(`Not implmented yet.`);
  }

  override call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor2D {
    throw new NotImplementedError(`Not implmented yet.`);
  }

  /**
   * Analyzes an einsum string to determine the required weight shape.
   */
  private analyzeEinsumString(
    equation: string,
    biasAxes: string,
    inputShape: Shape,
    outputShape: Shape
  ): [Shape, Shape, Shape] {
    throw new NotImplementedError(
      `Not implmented yet. Uses ${this.analyzeSplitString}.`);
  }

  /**
   * Analyze an pre-split einsum string to find the weight shape.
   */
  private analyzeSplitString(
    splitString: [string, string, string],
    biasAxes: string,
    inputShape: Shape,
    outputShape: Shape,
    leftElided?: boolean
  ): [Shape, Shape, Shape] {
    throw new NotImplementedError(`Not implmented yet.`);
  }
}
serialization.registerClass(EinsumDense);
