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
import { Tensor, einsum, linalg, logicalAnd, mul, ones, serialization, tidy, util } from '@tensorflow/tfjs-core';

import { cast, expandDims } from '../../backend/tfjs_backend';
import { Constraint, ConstraintIdentifier, getConstraint, serializeConstraint } from '../../constraints';
import { Layer, LayerArgs, SymbolicTensor } from '../../engine/topology';
import { ValueError } from '../../errors';
import { Initializer, InitializerIdentifier, getInitializer, serializeInitializer } from '../../initializers';
import { Shape } from '../../keras_format/common';
import { Regularizer, RegularizerIdentifier, getRegularizer, serializeRegularizer } from '../../regularizers';
import { Kwargs } from '../../types';
import { Softmax } from '../advanced_activations';
import { Dropout } from '../core';
import { EinsumDense } from './einsum_dense';

const _CHR_IDX = 'abcdefghijklmnopqrstuvwxyz'.split('');
/**
 * Builds einsum equations for the attention computation.
 *
 * Query, key, value inputs after projection are expected to have the shape as:
 * `(bs, <non-attention dims>, <attention dims>, numHeads, channels)`.
 * `bs` and `<non-attention dims>` are treated as `<batch dims>`.
 *
 * The attention operations can be generalized:
 * (1) Query-key dot product:
 * `(<batch dims>, <query attention dims>, numHeads, channels), (<batch dims>,
 * <key attention dims>, numHeads, channels) -> (<batch dims>,
 * numHeads, <query attention dims>, <key attention dims>)`
 * (2) Combination:
 * `(<batch dims>, numHeads, <query attention dims>, <key attention dims>),
 * (<batch dims>, <value attention dims>, numHeads, channels) -> (<batch
 * dims>, <query attention dims>, numHeads, channels)`
 *
 * @param rank Rank of query, key, value tensors.
 * @param attnAxes Array of axes, `[-1, rank)`,
 *    that attention will be applied to.
 * @returns Einsum equations.
 */
function buildAttentionEquation(
  rank: number, attnAxes: number[]
): [string, string, number] {
  const targetNotationArr = _CHR_IDX.slice(0, rank);
  // `batchDims` includes the head dim.
  const excludeIndices = [...attnAxes, rank - 1];
  const batchDims = [];
  for (const e of Array(rank).keys()) {
    if (!excludeIndices.includes(e)) {
      batchDims.push(e);
    }
  }
  let letterOffset = rank;
  let sourceNotation = '';
  for (let i = 0; i < rank; i++) {
    if (batchDims.includes(i) || i === rank - 1) {
      sourceNotation += targetNotationArr[i];
    } else {
      sourceNotation += _CHR_IDX[letterOffset];
      letterOffset++;
    }
  }

  const productNotation =
    batchDims.map(i => targetNotationArr[i]).concat(
    attnAxes.map(i => targetNotationArr[i]),
    attnAxes.map(i => sourceNotation[i]),
  ).join('');
  const targetNotation = targetNotationArr.join('');

  const dotProductEquation =
    `${sourceNotation},${targetNotation}->${productNotation}`;
  const attnScoresRank = productNotation.length;
  const combineEquation =
    `${productNotation},${sourceNotation}->${targetNotation}`;

  return [dotProductEquation, combineEquation, attnScoresRank];
}

/**
 * Builds an einsum equation for projections inside multi-head attention.
 */
function buildProjectionEquation(
  freeDims: number, boundDims: number, outputDims: number
): [string, string, number] {
  let inputStr = '';
  let kernelStr = '';
  let outputStr = '';
  let biasAxes = '';
  let letterOffset = 0;

  for (let i = 0; i < freeDims; i++) {
    const char = _CHR_IDX[i + letterOffset];
    inputStr += char;
    outputStr += char;
  }

  letterOffset += freeDims;
  for (let i = 0; i < boundDims; i++) {
    const char = _CHR_IDX[i + letterOffset];
    inputStr += char;
    kernelStr += char;
  }

  letterOffset += boundDims;
  for (let i = 0; i < outputDims; i++) {
    const char = _CHR_IDX[i + letterOffset];
    kernelStr += char;
    outputStr += char;
    biasAxes += char;
  }

  const equation = `${inputStr},${kernelStr}->${outputStr}`;
  return [equation, biasAxes, outputStr.length];
}

function getOutputShape(
  outputRank: number, knownLastDims: number[]
): Shape {
  const outputShape =
    Array(outputRank - knownLastDims.length).fill(null).concat(knownLastDims);
  return outputShape;
}

export declare interface MultiHeadAttentionArgs extends LayerArgs {
  /**
   * Integer. Number of attention heads.
   */
  numHeads: number;

  /**
   * Integer. Size of each attention head for query and key.
   */
  keyDim: number;

  /**
   * Integer. Size of each attention head for value.
   * Defaults to `keyDim`.
   */
  valueDim?: number;

  /**
   * Dropout probability.
   * Defaults to 0.0.
   */
  dropout?: number;

  /**
   * Whether the dense layers use bias vectors/matrices.
   * Defaults to true.
   */
  useBias?: boolean;

  /**
   * The expected shape of an output tensor, besides the batch
   * and sequence dims. If not specified, projects back to the query
   * feature dim (the query input's last dimension).
   */
  outputShape?: Shape;

  /**
   * Axes over which the attention is applied. `null` means attention over
   * all axes, but batch, heads, and features.
   */
  attentionAxes?: number[]|number;

  /**
   * Initializer for dense layer kernels.
   * Defaults to `"glorotUniform"`.
   */
  kernelInitializer?: Initializer|InitializerIdentifier;

  /**
   * Initializer for dense layer biases.
   * Defaults to `"zeros"`.
   */
  biasInitializer?: Initializer|InitializerIdentifier;

  /**
   * Regularizer for dense layer kernels.
   */
  kernelRegularizer?: Regularizer|RegularizerIdentifier;

  /**
   * Regularizer for dense layer biases.
   */
  biasRegularizer?: Regularizer|RegularizerIdentifier;

  /**
   * Regularizer for dense layer activity.
   */
  activityRegularizer?: Regularizer|RegularizerIdentifier;

  /**
   * Constraint for dense layer kernels.
   */
  kernelConstraint?: Constraint|ConstraintIdentifier;

  /**
   * Constraint for dense layer kernels.
   */
  biasConstraint?: Constraint|ConstraintIdentifier;
}

export declare interface MultiHeadAttentionOptions {
  /**
   * Query `Tensor` of shape `(B, T, dim)`.
   */

  /**
   * Value `Tensor` of shape `(B, S, dim)`.
   */
  value: Tensor;

  /**
   * Key `Tensor` of shape `(B, S, dim)`. If not given, will use `value` for
   * both `key` and `value`, which is the most common case.
   */
  key?: Tensor;

  /**
   * A boolean mask of shape `(B, T, S)`, that prevents
   * attention to certain positions. The boolean mask specifies which
   * query elements can attend to which key elements, 1 indicates
   * attention and 0 indicates no attention. Broadcasting can happen for
   * the missing batch dimensions and the head dimension.
   */
  attentionMask?: Tensor;

  /**
   * Indicates whether the layer should behave in training mode
   * (adding dropout) or in inference mode (no dropout).
   * Will go with either using the training mode of the parent
   * layer/model, or false (inference) if there is no parent layer.
   */
  training?: boolean;

  /**
   * Indicates whether to apply a causal mask to prevent tokens from attending
   * to future tokens (e.g., used in a decoder Transformer).
   * Defaults to false.
   */
  useCausalMask?: boolean;
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
 *
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
 * ```js
 * const layer = new MultiHeadAttention({numHeads: 2, keyDim: 2});
 * const target = tf.input({shape: [8, 16]});
 * const source = tf.input({shape: [4, 16]});
 * const outputTensor, weights = layer.callAndReturnAttentionScores(
 *     target, {value: source});
 * console.log(outputTensor.shape);  // [null, 8, 16]
 * console.log(weights.shape);  // [null, 2, 8, 4]
 * ```
 *
 * Performs 2D self-attention over a 5D input tensor on axes 2 and 3.
 *
 * ```js
 * const layer = new MultiHeadAttention({
 *    numHeads: 2, keyDim: 2, attentionAxes: [2, 3]});
 * const inputTensor = tf.input({shape: [5, 3, 4, 16]});
 * const outputTensor = layer.call(inputTensor, {value: inputTensor});
 * console.log(outputTensor.shape);  // [null, 5, 3, 4, 16]
 * ```
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

  protected readonly numHeads: number;
  protected readonly keyDim: number;
  protected readonly valueDim: number;
  protected readonly dropout: number;
  protected readonly useBias: boolean;
  protected readonly _outputShape: Shape;
  protected readonly kernelInitializer: Initializer;
  protected readonly biasInitializer: Initializer;
  protected readonly kernelRegularizer: Regularizer;
  protected readonly biasRegularizer: Regularizer;
  protected readonly kernelConstraint: Constraint;
  protected readonly biasConstraint: Constraint;
  protected dotProductEquation: string;
  protected combineEquation: string;
  protected attentionAxes: number[];
  protected builtFromSignature: boolean;
  protected softmax: Softmax;
  protected dropoutLayer: Dropout;
  protected queryShape: Shape;
  protected keyShape: Shape;
  protected valueShape: Shape;
  protected queryDense: EinsumDense;
  protected keyDense: EinsumDense;
  protected valueDense: EinsumDense;
  protected outputDense: EinsumDense;

  constructor(args: MultiHeadAttentionArgs) {
    super(args);
    this.supportsMasking = true;
    this.numHeads = args.numHeads;
    this.keyDim = args.keyDim;
    this.valueDim = args.valueDim ?? args.keyDim;
    this.dropout = args.dropout ?? 0;
    this.useBias = args.useBias ?? true;
    this._outputShape = args.outputShape;
    this.kernelInitializer = getInitializer(
      args.kernelInitializer ?? 'glorotUniform');
    this.biasInitializer = getInitializer(args.biasInitializer ?? 'zeros');
    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);
    this.activityRegularizer = getRegularizer(args.activityRegularizer);
    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.biasConstraint = getConstraint(args.biasConstraint);
    if (args.attentionAxes != null && !Array.isArray(args.attentionAxes)) {
      this.attentionAxes = [args.attentionAxes];
    } else {
      this.attentionAxes = args.attentionAxes as number[];
    }
    this.builtFromSignature = false;
    this.queryShape = null;
    this.keyShape = null;
    this.valueShape = null;
  }

  /**
   * Should be used for testing purposes only.
   */
  get _queryDense() {
    return this.queryDense;
  }

  /**
   * Should be used for testing purposes only.
   */
  get _keyDense() {
    return this.keyDense;
  }

  /**
   * Should be used for testing purposes only.
   */
  get _valueDense() {
    return this.valueDense;
  }

  /**
   * Should be used for testing purposes only.
   */
  get _outputDense() {
    return this.outputDense;
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      numHeads: this.numHeads,
      keyDim: this.keyDim,
      valueDim: this.valueDim,
      dropout: this.dropout,
      useBias: this.useBias,
      outputShape: this._outputShape,
      attentionAxes: this.attentionAxes,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      queryShape: this.queryShape,
      keyShape: this.keyShape,
      valueShape: this.valueShape,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  static override fromConfig<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>,
    config: serialization.ConfigDict
  ): T {
    // If the layer has a different build() function from the default,
    // we need to trigger the customized build to create weights.
    const queryShape = config['queryShape'] as Shape;
    const keyShape = config['keyShape'] as Shape;
    const valueShape = config['valueShape'] as Shape;
    delete config['queryShape'];
    delete config['keyShape'];
    delete config['valueShape'];

    const layer = new cls(config);
    if ([queryShape, keyShape, valueShape].includes(null)) {
        console.warn(
            'One of dimensions of the input shape is missing. It ' +
            'should have been memorized when the layer was serialized. ' +
            `${cls.toString()} is created without weights.`
        );
    } else {
      (layer as unknown as MultiHeadAttention).buildFromSignature(
        queryShape, valueShape, keyShape);
    }
    return layer;
  }

  /**
   * Builds layers and variables.
   *
   * Once the method is called, this.builtFromSignature will be set to true.
   */
  buildFromSignature(
    queryShape: Shape,
    valueShape: Shape,
    keyShape?: Shape
  ) {
    this.builtFromSignature = true;

    if (keyShape == null) {
      keyShape = valueShape;
    }

    this.queryShape = queryShape;
    this.valueShape = valueShape;
    this.keyShape = keyShape;

    // Not using SymbolicTensors since tf.input() adds a batch dimension to the
    // given shape, therefore giving the tensor the wrong rank.
    const queryRank = queryShape.length;
    const valueRank = valueShape.length;
    const keyRank = keyShape.length;

    const freeDims = queryRank - 1;
    let [einsumEquation, biasAxes, outputRank] =
      buildProjectionEquation(freeDims, 1, 2);
    this.queryDense = new EinsumDense({
      equation: einsumEquation,
      outputShape: getOutputShape(outputRank - 1, [this.numHeads, this.keyDim]),
      biasAxes: this.useBias ? biasAxes : null,
      name: 'query',
      ...this.getCommonKwargsForSublayer(),
    });

    [einsumEquation, biasAxes, outputRank] =
      buildProjectionEquation(keyRank - 1, 1, 2);
    this.keyDense = new EinsumDense({
      equation: einsumEquation,
      outputShape: getOutputShape(outputRank - 1, [this.numHeads, this.keyDim]),
      biasAxes: this.useBias ? biasAxes : null,
      name: 'key',
      ...this.getCommonKwargsForSublayer(),
    });

    [einsumEquation, biasAxes, outputRank] =
      buildProjectionEquation(valueRank - 1, 1, 2);
    this.valueDense = new EinsumDense({
      equation: einsumEquation,
      outputShape: getOutputShape(
        outputRank - 1, [this.numHeads, this.valueDim]),
      biasAxes: this.useBias ? biasAxes : null,
      name: 'value',
      ...this.getCommonKwargsForSublayer(),
    });

    // Builds the attention computations for multi-head dot product attention.
    this.buildAttention(outputRank);
    this.outputDense = this.makeOutputDense(
      freeDims,
      this.getCommonKwargsForSublayer(),
      'attentionOutput'
    );
  }

  private getCommonKwargsForSublayer(): Kwargs {
    // Create new clone of kernel/bias initializer, so that we don't reuse
    // the initializer instance, which could lead to same init value since
    // initializer is stateless.
    const kernelInitializer = getInitializer({
      className: this.kernelInitializer.getClassName(),
      config: this.kernelInitializer.getConfig(),
    });
    const biasInitializer = getInitializer({
      className: this.biasInitializer.getClassName(),
      config: this.biasInitializer.getConfig(),
    });

    const commonKwargs = {
      kernelInitializer,
      biasInitializer,
      kernelRegularizer: this.kernelRegularizer,
      biasRegularizer: this.biasRegularizer,
      activityRegularizer: this.activityRegularizer,
      kernelConstraint: this.kernelConstraint,
      biasConstraint: this.biasConstraint,
    };
    return commonKwargs;
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
  ): EinsumDense {
    let outputShape: Shape;
    if (this._outputShape) {
      if (!Array.isArray(this._outputShape)) {
        outputShape = [this._outputShape];
      } else {
        outputShape = this._outputShape;
      }
    } else {
      outputShape = [this.queryShape[this.queryShape.length - 1]];
    }

    const [einsumEquation, biasAxes, outputRank] =
      buildProjectionEquation(freeDims, 2, outputShape.length);

    return new EinsumDense({
      equation: einsumEquation,
      outputShape: getOutputShape(outputRank - 1, outputShape),
      biasAxes: this.useBias ? biasAxes : null,
      name,
      ...commonKwargs,
    });
  }

  /**
   * Builds multi-head dot-product attention computations.
   *
   * This function builds attributes necessary for `computeAttention` to
   * customize attention computation to replace the default dot-product
   * attention.
   *
   * @param rank The rank of query, key, value tensors.
   */
  protected buildAttention(rank: number) {
    if (this.attentionAxes == null) {
      this.attentionAxes = [];
      for (let i = 1; i < rank - 2; i++) {
        this.attentionAxes.push(i);
      }
    } else {
      this.attentionAxes = [...this.attentionAxes];
    }

    const [dotProductEquation, combineEquation, attnScoresRank] =
      buildAttentionEquation(rank, this.attentionAxes);
    this.dotProductEquation = dotProductEquation;
    this.combineEquation = combineEquation;

    const normAxes: number[] = [];
    const startIdx = attnScoresRank - this.attentionAxes.length;
    for (let i = startIdx; i < attnScoresRank; i++) {
      normAxes.push(i);
    }
    this.softmax = new Softmax({axis: normAxes});
    this.dropoutLayer = new Dropout({rate: this.dropout});
  }

  protected maskedSoftmax(
    attentionScores: Tensor, attentionMask?: Tensor
  ): Tensor {
    return tidy(() => {
      // Normalize the attention scores to probabilities.
      // `attentionScores` = [B, N, T, S]
      if (attentionMask != null) {
        // The expand dim happens starting from the `numHeads` dimension,
        // (<batchDims>, numHeads, <queryAttentionDims, keyAttentionDims>)
        const maskExpansionAxis = -this.attentionAxes.length * 2 - 1;
        const endIdx =
          attentionScores.shape.length - attentionMask.shape.length;
        for (let _ = 0; _ < endIdx; _++) {
          attentionMask = expandDims(attentionMask, maskExpansionAxis);
        }
      }
      return this.softmax.apply(
        attentionScores, {mask: attentionMask}) as Tensor;
    });
  }

  /**
   * Applies Dot-product attention with query, key, value tensors.
   *
   * This function defines the computation inside `call` with projected
   * multi-head Q, K, V inputs. Users can override this function for
   * customized attention implementation.
   *
   * @param query Projected query `Tensor` of shape `(B, T, N, keyDim)`.
   * @param key  Projected key `Tensor` of shape `(B, S, N, keyDim)`.
   * @param value Projected value `Tensor` of shape `(B, S, N, valueDim)`.
   * @param attentionMask A boolean mask of shape `(B, T, S)`, that prevents
   *    attention to certain positions. It is generally not needed if
   *    the `query` and `value` (and/or `key`) are masked.
   * @param training Boolean indicating whether the layer should behave
   *    in training mode (adding dropout) or in inference mode (doing
   *    nothing).
   * @returns attentionOutput: Multi-headed outputs of attention computation.
   * @returns attentionScores: Multi-headed attention weights.
   */
  protected computeAttention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attentionMask?: Tensor,
    training?: boolean
  ): [Tensor, Tensor] {
    return tidy(() => {
      // Note: Applying scalar multiply at the smaller end of einsum improves
      // XLA performance, but may introduce slight numeric differences in
      // the Transformer attention head.
      query = mul(query, 1.0 / Math.sqrt(this.keyDim));

      // Take the dot product between "query" and "key" to get the raw
      // attention scores.
      let attentionScores = einsum(this.dotProductEquation, key, query);

      attentionScores = this.maskedSoftmax(attentionScores, attentionMask);

      // This is actually dropping out entire tokens to attend to, which might
      // seem a bit unusual, but is taken from the original Transformer paper.
      const attentionScoresDropout =
        this.dropoutLayer.apply(attentionScores, {training}) as Tensor;

      // `contextLayer` = [B, T, N, H]
      const attentionOutput =
        einsum(this.combineEquation, attentionScoresDropout, value);

      return [attentionOutput, attentionScores];
    });
  }

  override apply(
    inputs: Tensor | SymbolicTensor,
    kwargs?: Kwargs
  ): Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[] {
    if (!kwargs || !kwargs['value']) {
      throw new ValueError('Must pass in `value` argument in `kwargs.`');
    }
    let newInputs: Tensor[]|SymbolicTensor[];

    newInputs = [inputs, kwargs['value']].concat(kwargs['key'] ?? []);

    // TODO(pforderique): Support mask propagation.
    return super.apply(newInputs, kwargs);
  }

  override call(
    query: Tensor, kwargs: MultiHeadAttentionOptions
  ): Tensor {
    return tidy(() => {
      return this.callAndReturnAttentionScores(query, kwargs)[0];
    });
  }

  /**
   * Exactly like `call` except also returns the attention scores.
   */
  callAndReturnAttentionScores(
    query: Tensor,
    {
      value,
      key,
      useCausalMask,
      attentionMask,
      training
    }: MultiHeadAttentionOptions
  ): [Tensor, Tensor] {
    return tidy(() => {
      if (!this.builtFromSignature) {
        this.buildFromSignature(
          query.shape,
          value.shape,
          key ? key.shape : null
        );
      }
      if (key == null) {
        key = value;
      }

      // TODO(pforderique): Support RaggedTensor inputs.

      attentionMask = this.computeAttentionMask(
        query,
        value,
        attentionMask,
        useCausalMask,
      );

      //   N = `numAttentionHeads`
      //   H = `sizePerHead`
      // `query` = [B, T, N ,H]
      query = this.queryDense.apply(query) as Tensor;

      // `key` = [B, S, N, H]
      key = this.keyDense.apply(key) as Tensor;

      // `value` = [B, S, N, H]
      value = this.valueDense.apply(value) as Tensor;

      const [attentionOutputPreDense, attentionScores] = this.computeAttention(
        query,
        key,
        value,
        attentionMask,
        training
      );
      const attentionOutput =
        this.outputDense.apply(attentionOutputPreDense) as Tensor;

      return [attentionOutput, attentionScores];
    });
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
   * @param query Projected query `Tensor` of shape `(B, T, N, keyDim)`.
   * @param key  Projected key `Tensor` of shape `(B, S, N, keyDim)`.
   * @param value Projected value `Tensor` of shape `(B, S, N, valueDim)`.
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
    attentionMask?: Tensor,
    useCausalMask = false
  ): Tensor {
    return tidy(() => {
      let autoMask: Tensor;

      const queryMask = query.kerasMask;
      const valueMask = value.kerasMask;
      if (queryMask != null) {
        autoMask = queryMask.expandDims(2); // Shape is [B, T, 1]
      }
      if (valueMask != null) {
        const mask = valueMask.expandDims(1); // Shape is [B, 1, S]
        autoMask = autoMask ? logicalAnd(autoMask, mask) : mask;
      }
      if (useCausalMask) {
        // the shape of the causal mask is [1, T, S]
        const mask = this.computeCausalMask(query, value);
        autoMask = autoMask ? logicalAnd(autoMask, mask) : mask;
      }
      if (autoMask != null) {
        // Merge attentionMask & automatic mask, to shape [B, T, S]
        attentionMask = attentionMask ?
          cast(attentionMask, 'bool').logicalAnd(autoMask) : autoMask;
      }

      return attentionMask;
    });
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
  private computeCausalMask(query: Tensor, value?: Tensor): Tensor {
    return tidy(() => {
      const qSeqLength = query.shape[1];
      const vSeqLength = value ? value.shape[1] : qSeqLength;
      // Create a lower triangular matrix.
      return linalg.bandPart(ones([1, qSeqLength, vSeqLength], 'bool'), -1, 0);
    });
  }

  /**
   *
   * @param inputShapes A list of [queryShape, valueShape] or
   *    [queryShape, valueShape, keyShape]. If no keyShape provided, valueShape
   *    is assumed as the keyShape.
   */
  override computeOutputShape(inputShapes: [Shape, Shape, Shape|null]): Shape {
    const [queryShape, valueShape, maybeKeyShape] = inputShapes;
    const keyShape = maybeKeyShape ?? valueShape;

    if (queryShape.slice(-1)[0] !== valueShape.slice(-1)[0]) {
      throw new ValueError(
        `The last dimension of 'queryShape' and 'valueShape' must be equal, ` +
        `but are ${queryShape.slice(-1)[0]}, ${valueShape.slice(-1)[0]}. ` +
        `Received: queryShape=${queryShape}, valueShape=${valueShape}`
      );
    }

    if (!util.arraysEqual(valueShape.slice(1, -1), keyShape.slice(1, -1))) {
      throw new Error(
        `All dimensions of 'value' and 'key', except the last one, must be ` +
        `equal. Received ${valueShape} and ${keyShape}`
      );
    }

    if (this._outputShape) {
      return queryShape.slice(0, -1).concat(this._outputShape);
    }

    return queryShape;
  }
}
serialization.registerClass(MultiHeadAttention);
