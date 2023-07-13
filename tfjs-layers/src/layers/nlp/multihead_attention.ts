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
 *  TFJS-based multi-head attention layer.
 */

/* Original source: keras/layers/attention/multi_head_attention.py */
import { Tensor, Tensor1D, Tensor2D, serialization } from '@tensorflow/tfjs-core';

import { ConstraintIdentifier } from '../../constraints';
import { Layer, LayerArgs } from '../../engine/topology';
import { NotImplementedError } from '../../errors';
import { InitializerIdentifier } from '../../initializers';
import { Shape } from '../../keras_format/common';
import { RegularizerIdentifier } from '../../regularizers';
import { Kwargs } from '../../types';

/**
 * Builds einsum equations for the attention computation.
 *
 * Query, key, value inputs after projection are expected to have the shape as:
 * `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
 * `bs` and `<non-attention dims>` are treated as `<batch dims>`.

 * The attention operations can be generalized:
 * (1) Query-key dot product:
 * `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
 * <key attention dims>, num_heads, channels) -> (<batch dims>,
 * num_heads, <query attention dims>, <key attention dims>)`
 * (2) Combination:
 * `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
 * (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
 * dims>, <query attention dims>, num_heads, channels)`
 *
 * @param rank Rank of query, key, value tensors.
 * @param attnAxes Array of axes, `[-1, rank)`,
 *    that attention will be applied to.
 * @returns Einsum equations.
 */
function buildAttentionEquation(
  rank: number, attnAxes: Array<number>
): [string, string, string] {
  throw new NotImplementedError('Not implemented yet.');
}

/**
 * Builds an einsum equation for projections inside multi-head attention.
 */
function buildProjectionEquation(
  freeDims: number, boundDims: number, outputDims: number
): [string, string, number] {
  throw new NotImplementedError('Not implemented yet.');
}

function getOutputShape(
  outputRank: number, knownLastDims: Iterable<number>
): Shape {
  throw new NotImplementedError('Not implemented yet.');
}

export declare interface MultiHeadAttentionArgs extends LayerArgs {
  /**
   * Integer. Number of attention heads.
   */
  numHeads: number,

  /**
   * Integer. Size of each attention head for query and key.
   */
  keyDim: number,

  /**
   * Integer. Size of each attention head for value.
   * Defaults to `keyDim`.
   */
  valueDim?: number,

  /**
   * Dropout probability.
   * Defaults to 0.0.
   */
  dropout?: number,

  /**
   * Whether the dense layers use bias vectors/matrices.
   * Defaults to true.
   */
  useBias?: boolean,

  /**
   * The expected shape of an output tensor, besides the batch
   * and sequence dims. If not specified, projects back to the query
   * feature dim (the query input's last dimension).
   */
  outputShape?: Shape,

  /**
   * Axes over which the attention is applied. `null` means attention over
   * all axes, but batch, heads, and features.
   */
  attentionAxes: Array<number>,

  /**
   * Initializer for dense layer kernels.
   * Defaults to `"glorotUniform"`.
   */
  kernelInitializer?: InitializerIdentifier,

  /**
   * Initializer for dense layer biases.
   * Defaults to `"zeros"`.
   */
  biasInitializer?: InitializerIdentifier,

  /**
   * Regularizer for dense layer kernels.
   */
  kernelRegularizer?: RegularizerIdentifier,

  /**
   * Regularizer for dense layer biases.
   */
  biasRegularizer?: RegularizerIdentifier,

  /**
   * Regularizer for dense layer activity.
   */
  activityRegularizer?: RegularizerIdentifier,

  /**
   * Constraint for dense layer kernels.
   */
  kernelConstraint?: ConstraintIdentifier,

  /**
   * Constraint for dense layer kernels.
   */
  biasConstraint?: ConstraintIdentifier,
}

export declare interface MultiHeadAttentionOptions {
  /**
   * Query `Tensor` of shape `(B, T, dim)`.
   */

  /**
   * Value `Tensor` of shape `(B, S, dim)`.
   */
  value: Tensor,

  /**
   * Key `Tensor` of shape `(B, S, dim)`. If not given, will use `value` for
   * both `key` and `value`, which is the most common case.
   */
  key?: Tensor,

  /**
   * A boolean mask of shape `(B, T, S)`, that prevents
   * attention to certain positions. The boolean mask specifies which
   * query elements can attend to which key elements, 1 indicates
   * attention and 0 indicates no attention. Broadcasting can happen for
   * the missing batch dimensions and the head dimension.
   */
  attentionMask?: Tensor,

  /**
   * Indicates whether the layer should behave in training mode
   * (adding dropout) or in inference mode (no dropout).
   * Will go with either using the training mode of the parent
   * layer/model, or false (inference) if there is no parent layer.
   */
  training?: boolean,

  /**
   * Indicates whether to apply a causal mask to prevent tokens from attending
   * to future tokens (e.g., used in a decoder Transformer).
   * Defaults to false.
   */
  useCausalMask?: boolean,
}

/**
 * MultiHeadAttention layer.
 *
 * This is an implementation of multi-headed attention as described in the
 * paper "Attention is all you Need" (Vaswani et al., 2017).
 * If `query`, `key,` `value` are the same, then
 * this is self-attention. Each timestep in `query` attends to the
 * corresponding sequence in `key`, and returns a fixed-width vector.
 *
 * This layer first projects `query`, `key` and `value`. These are
 * (effectively) a list of tensors of length `numAttentionHeads`, where the
 * corresponding shapes are `(batchSize, <query dimensions>, keyDim)`,
 * `(batchSize, <key/value dimensions>, keyDim)`,
 * `(batchSize, <key/value dimensions>, valueDim)`.
 *
 * Then, the query and key tensors are dot-producted and scaled. These are
 * softmaxed to obtain attention probabilities. The value tensors are then
 * interpolated by these probabilities, then concatenated back to a single
 * tensor.
 *
 * Finally, the result tensor with the last dimension as valueDim can take an
 * linear projection and return.

 * When using `MultiHeadAttention` inside a custom layer, the custom layer must
 * implement its own `build()` method and call `MultiHeadAttention`'s
 * `buildFromSignature()` there.
 * This enables weights to be restored correctly when the model is loaded.
 *
 * Examples:
 *
 * Performs 1D cross-attention over two sequence inputs with an attention mask.
 * Returns the additional attention weights over heads.
 *
 * const layer = new MultiHeadAttention({numHeads=2, keyDim=2});
 * const target = tf.keras.Input(shape=[8, 16]);
 * const source = tf.keras.Input(shape=[4, 16]);
 * const outputTensor, weights = layer.callAndReturnAttentionScores(
 *     target, {value: source});
 * console.log(outputTensor.shape);  // [null, 8, 16]
 * console.log(weights.shape);  // [null, 2, 8, 4]
 *
 * Performs 2D self-attention over a 5D input tensor on axes 2 and 3.
 *
 * const layer = new MultiHeadAttention({
 *    numHeads: 2, keyDim: 2, attentionAxes: [2, 3]});
 * const inputTensor = tf.input({shape: [5, 3, 4, 16]});
 * const outputTensor = layer.call(inputTensor, {value: inputTensor});
 * console.log(outputTensor.shape);  // [null, 5, 3, 4, 16]
 *
 * Returns:
 *    attentionOutput: The result of the computation, of shape `(B, T, E)`,
 *        where `T` is for target sequence shapes and `E` is the query input
 *        last dimension if `outputShape` is `None`. Otherwise, the
 *        multi-head outputs are projected to the shape specified by
 *        `outputShape`.
 *    attentionScores: multi-head attention coefficients over attention axes.
 */
export class MultiHeadAttention extends Layer {
  /** @nocollapse */
  static readonly className = 'MultiHeadAttention';

  constructor(args: MultiHeadAttentionArgs) {
    super(args);
    throw new NotImplementedError('Not implemented yet.');
  }

  override getConfig(): serialization.ConfigDict {
    throw new NotImplementedError('Not implemented yet.');
  }

  static override fromConfig<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>,
    config: serialization.ConfigDict
  ): T {
    throw new NotImplementedError('Not implemented yet.');
  }

  /**
   * Builds layers and variables.
   *
   * Once the method is called, this.builtFromSignature will be set to true.
   */
  private buildFromSignature(query: Tensor, value: Tensor, key?: Tensor) {
    throw new NotImplementedError(
      `Not implemented yet. Uses ${buildProjectionEquation}, ${getOutputShape},
       ${this.getCommonKwargsForSublayer}, ${this.buildAttention},
       ${this.makeOutputDense}.`);
  }

  private getCommonKwargsForSublayer(): Kwargs {
    throw new NotImplementedError('Not implemented yet.');
  }

  /**
   * Builds the output projection matrix.
   *
   * @param freeDims Number of free dimensions for einsum equation building.
   * @param commonKwargs Common keyword arguments for einsum layer.
   * @param name Name for the projection layer.
   * @returns Projection layer.
   */
  private makeOutputDense(
    freeDims: number, commonKwargs: Kwargs, name?: string
  ): Kwargs {
    throw new NotImplementedError('Not implemented yet.');
  }

  /**
   * Builds multi-head dot-product attention computations.
   *
   * This function builds attributes necessary for `_compute_attention` to
   * customize attention computation to replace the default dot-product
   * attention.
   *
   * @param rank The rank of query, key, value tensors.
   */
  private buildAttention(rank: number) {
    throw new NotImplementedError(
      `Not implemented yet. Uses ${buildAttentionEquation}.`);
  }

  private maskedSoftmax(
    attentionScores: Tensor, attentionMask?: Tensor
  ): Tensor|Tensor[] {
    throw new NotImplementedError('Not implemented yet.');
  }

  /**
   * Applies Dot-product attention with query, key, value tensors.
   *
   * This function defines the computation inside `call` with projected
   * multi-head Q, K, V inputs. Users can override this function for
   * customized attention implementation.
   *
   * @param query Projected query `Tensor` of shape `(B, T, N, key_dim)`.
   * @param key  Projected key `Tensor` of shape `(B, S, N, key_dim)`.
   * @param value Projected value `Tensor` of shape `(B, S, N, value_dim)`.
   * @param attentionMask A boolean mask of shape `(B, T, S)`, that prevents
   *    attention to certain positions. It is generally not needed if
   *    the `query` and `value` (and/or `key`) are masked.
   * @param training Boolean indicating whether the layer should behave
   *    in training mode (adding dropout) or in inference mode (doing
   *    nothing).
   * @returns attentionOutput: Multi-headed outputs of attention computation.
   * @returns attentionScores: Multi-headed attention weights.
   */
  private computeAttention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attentionMask?: Tensor,
    training?: boolean
  ): [Tensor, Tensor] {
    throw new NotImplementedError(
      `Not implemented yet. Uses ${this.maskedSoftmax}.`);
  }

  override call(
    query: Tensor, kwargs: MultiHeadAttentionOptions
  ): Tensor|Tensor2D {
    return this.callAndReturnAttentionScores(query, kwargs)[0];
  }

  /**
   * Exactly like `call` except also returns the attention scores.
   */
  callAndReturnAttentionScores(
    query: Tensor, kwargs: MultiHeadAttentionOptions
  ): [Tensor1D|Tensor2D, Tensor1D|Tensor2D] {
    throw new NotImplementedError(
      `Not implemented yet. Uses ${this.buildFromSignature},
       ${this.computeAttentionMask}, ${this.computeAttention}.`);
  }

  /**
   * Computes the attention mask.
   *
   * * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
   * * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
   * * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
   *   mask is ignored if `key` is `None` or if `key is value`.
   * * If `useCausalMask=true`, then the causal mask is computed. Its shape
   *   is [1, T, S].
   *
   * All defined masks are merged using a logical AND operation (`&`).
   *
   * In general, if the `query` and `value` are masked, then there is no need
   * to define the `attentionMask`.
   *
   * @param query Projected query `Tensor` of shape `(B, T, N, key_dim)`.
   * @param key  Projected key `Tensor` of shape `(B, S, N, key_dim)`.
   * @param value Projected value `Tensor` of shape `(B, S, N, value_dim)`.
   * @param attentionMask A boolean mask of shape `(B, T, S)`, that prevents
   *    attention to certain positions.
   * @param useCausalMask  A boolean to indicate whether to apply a causal
   *    mask to prevent tokens from attending to future tokens (e.g.,
   *    used in a decoder Transformer).
   * @returns attentionMask: A boolean mask of shape `(B, T, S)`, that prevents
   *    attention to certain positions, based on the Keras masks of the
   *    `query`, `key`, `value`, and `attentionMask` tensors, and the
   *    causal mask if `useCausalMask=true`.
   */
  private computeAttentionMask(
    query: Tensor,
    value: Tensor,
    key?: Tensor,
    attentionMask?: Tensor,
    useCausalMask?: boolean
  ): Tensor {
    throw new NotImplementedError(
      `Not implemented yet. Uses ${this.computeCasualMask}`);
  }

  /**
   * Computes a causal mask (e.g., for masked self-attention layers).
   *
   * For example, if query and value both contain sequences of length 4,
   * this function returns a boolean `Tensor` equal to:
   *
   * ```
   * [[[true,  false, false, false],
   *   [true,  true,  false, false],
   *   [true,  true,  true,  false],
   *   [true,  true,  true,  true]]]
   * ```
   *
   * @param query query `Tensor` of shape `(B, T, ...)`.
   * @param value value `Tensor` of shape `(B, S, ...)` (defaults to query).
   * @returns mask: A boolean `Tensor` of shape [1, T, S] containing a lower
   *    triangular matrix of shape [T, S].
   */
  private computeCasualMask(query: Tensor, value?: Tensor): Tensor {
    throw new NotImplementedError('Not implemented yet.');
  }

  /**
   *
   * @param inputShapes A list of [queryShape, valueShape] or
   *    [queryShape, valueShape, keyShape]. If no keyShape provided, valueShape
   *    is assumed as the keyShape.
   */
  override computeOutputShape(
    inputShapes: [Shape, Shape] | [Shape, Shape, Shape]
  ): Shape {
    throw new NotImplementedError('Not implemented yet.');
  }
}
serialization.registerClass(MultiHeadAttention);
