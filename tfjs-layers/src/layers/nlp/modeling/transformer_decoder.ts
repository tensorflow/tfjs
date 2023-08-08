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
 *  Transformer decoder block implementation based on TFJS `Layer`.
 */

/* Original source: keras_nlp/layers/modeling/transformer_decoder.py */
import { Tensor, add, serialization, tidy } from '@tensorflow/tfjs-core';

import { Activation, getActivation, serializeActivation } from '../../../activations';
import { Layer, LayerArgs, SymbolicTensor, } from '../../../engine/topology';
import { ValueError } from '../../../errors';
import { Initializer, InitializerIdentifier, getInitializer, serializeInitializer } from '../../../initializers';
import { ActivationIdentifier } from '../../../keras_format/activation_config';
import { Shape } from '../../../keras_format/common';
import { Dense, Dropout } from '../../core';
import { LayerNormalization } from '../../normalization';

import { CachedMultiHeadAttention } from './cached_multihead_attention';
import { computeCausalMask, mergePaddingAndAttentionMask } from './transformer_layer_utils';

export declare interface TransformerDecoderArgs extends LayerArgs {
  /**
   * Integer. The hidden size of feedforward network.
   */
  intermediateDim: number;

  /**
   * Integer. The number of heads in MultiHeadAttention.
   */
  numHeads: number;

  /**
   * The dropout value, shared by MultiHeadAttention and feedforward network.
   * Defaults to `0.`.
   */
  dropout?: number;

  /**
   * The activation function of feedforward network.
   * Defaults to `"relu"`.
   */
  activation?: Activation|ActivationIdentifier;

  /**
   * The eps value in layer normalization components.
   * Defaults to `1e-5`.
   */
  layerNormEpsilon?: number;

  /**
   * The kernel initializer for the dense and multiheaded attention layers.
   * Defaults to `"glorotUniform"`.
   */
  kernelInitializer?: Initializer|InitializerIdentifier;

  /**
   * The bias initializer for the dense and multiheaded attention layers.
   * Defaults to `"zeros"`.
   */
  biasInitializer?: Initializer|InitializerIdentifier;

  /**
   * If true, the inputs to the attention layer(s) and the intermediate dense
   * layer are normalized (similar to GPT-2). If set to false, outputs of
   * attention layer and intermediate dense layer are normalized
   * (similar to BERT).
   * Defaults to `false`.
   */
  normalizeFirst?: boolean;
}

export declare interface TransformerDecoderOptions {
  /**
   * decoderSequence: The decode input sequence.
   */

  /**
   * The encoder input sequence. For decoder only models (like GPT2), this
   * should be left `null`. Once the model is called without an encoderSequence,
   * you cannot call it again with encoderSequence.
   */
  encoderSequence?: Tensor|SymbolicTensor;

  /**
   * A boolean Tensor, the padding mask of decoder sequence, must be of shape
   * `[batchSize, decoderSequenceLength]`.
   */
  decoderPaddingMask?: Tensor|SymbolicTensor;

  /**
   * A boolean Tensor. Customized decoder sequence mask, must be of shape
   * `[batchSize, decoderSequenceLength, decoderSequenceLength]`.
   */
  decoderAttentionMask?: Tensor;

  /**
   * A boolean Tensor, the padding mask of encoder sequence, must be of shape
   * `[batchSize, encoderSequenceLength]`.
   */
  encoderPaddingMask?: Tensor;

  /**
   * A boolean Tensor. Customized encoder sequence mask, must be of shape
   * `[batchSize, encoderSequenceLength, encoderSequenceLength]`.
   */
  encoderAttentionMask?: Tensor;

  /**
   * A dense float Tensor. The cache of key/values pairs in the self-attention
   * layer. Has shape `[batchSize, 2, maxSeqLen, numHeads, keyDims]`.
   */
  selfAttentionCache?: Tensor;

  /**
   * Integer or Integer Tensor. The index at which to update the
   * `selfAttentionCache`. Usually, this is the index of the current token
   * being processed during decoding.
   */
  selfAttentionCacheUpdateIndex?: number;

  /**
   * A dense float Tensor. The cache of key/value pairs in the cross-attention
   * layer. Has shape `[batchSize, 2, S, numHeads, keyDims]`.
   */
  crossAttentionCache?: Tensor;

  /**
   * Integer or Integer Tensor. The index at which to update the
   * `crossAttentionCache`. Usually, this is either `0` (compute the entire
   * `crossAttentionCache`), or `null` (reuse a previously computed
   * `crossAttentionCache`).
   */
  crossAttentionCacheUpdateIndex?: number;

  /**
   * If true, a causal mask (masking out future input) is applied on the decoder
   * sequence.
   * Defaults to `true`.
   */
  useCausalMask?: boolean;
}

/**
 * Transformer decoder.
 *
 * This class follows the architecture of the transformer decoder layer in the
 * paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
 * can instantiate multiple instances of this class to stack up a decoder.
 *
 * By default, this layer will apply a causal mask to the decoder attention
 * layer. This layer will correctly compute an attention mask from an implicit
 * padding mask (for example, by passing `maskZero=true` to a
 * `tf.layers.embedding` layer). See the Masking and Padding
 * [guide](https://keras.io/guides/understanding_masking_and_padding/)
 * for more details.
 *
 * This layer can be called with either one or two inputs. The number of inputs
 * must be consistent across all calls. The options are as follows:
 *    `layer.call(decoderSequence)`: no cross-attention will be built into the
 *         decoder block. This is useful when building a "decoder-only"
 *         transformer such as GPT-2.
 *    `layer.call(decoderSequence, {encoderSequence})`: cross-attention will be
 *         built into the decoder block. This is useful when building an
 *         "encoder-decoder" transformer, such as the original transformer
 *         model described in Attention is All You Need.
 *
 * Examples:
 * ```js
 * // Create a single transformer decoder layer.
 * const decoder = new TransformerDecoder({intermediateDim: 64, numHeads: 8});
 *
 * // Create a simple model containing the decoder.
 * const decoderInput = tf.input({shape: [10, 64]});
 * const encoderInput = tf.input({shape: {[10, 64]});
 * const output = decoder.call(decoderInput, {encoderInput});
 * const model = tf.model({
 *     inputs: [decoderInput, encoderInput],
 *     outputs: output,
 * );
 *
 * // Call decoder on the inputs.
 * const decoderInputData = tf.randomUniform([2, 10, 64]);
 * const encoderInputData = tf.randomUniform([2, 10, 64]);
 * const decoderOutput = model.predict([decoderInputData, encoderInputData]);
 * ```
 *
 * References:
 *  - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
 */
export class TransformerDecoder extends Layer {
  /** @nocollapse */
  static readonly className = 'TransformerDecoder';

  protected intermediateDim: number;
  protected numHeads: number;
  protected dropout: number;
  protected activation: Activation;
  protected layerNormEpsilon: number;
  protected kernelInitializer: Initializer;
  protected biasInitializer: Initializer;
  protected normalizeFirst: boolean;
  protected decoderSequenceShape: Shape;
  protected encoderSequenceShape: Shape;

  protected selfAttentionLayer: CachedMultiHeadAttention;
  protected selfAttentionLayernorm: LayerNormalization;
  protected selfAttentionDropout: Dropout;

  protected selfCrossAttentionLayer: CachedMultiHeadAttention;
  protected selfCrossAttentionLayernorm: LayerNormalization;
  protected selfCrossAttentionDropout: Dropout;

  protected feedforwardIntermediateDense: Dense;
  protected feedforwardOutputDense: Dense;
  protected feedforwardLayernorm: LayerNormalization;
  protected feedforwardDropout: Dropout;

  constructor(args: TransformerDecoderArgs) {
    super(args);
    this.intermediateDim = args.intermediateDim;
    this.numHeads = args.numHeads;
    this.dropout = args.dropout ?? 0;
    this.activation = getActivation(args.activation ?? 'relu');
    this.layerNormEpsilon = args.layerNormEpsilon ?? 1e-05;
    this.kernelInitializer =
      getInitializer(args.kernelInitializer ?? 'glorotUniform');
    this.biasInitializer = getInitializer(args.biasInitializer ?? 'zeros');
    this.normalizeFirst = args.normalizeFirst ?? false;
  }

  /**
   *
   * @param inputShape decoderSequenceShape or
   *  [decoderSequenceShape, encoderSequenceShape]
   */
  override build(inputShape: Shape|[Shape, Shape]): void {
    if (Array.isArray(inputShape[0])) {
      // `inputShape` is of type [Shape, Shape].
      [this.decoderSequenceShape, this.encoderSequenceShape] =
        inputShape as [Shape, Shape];
    } else {
      this.decoderSequenceShape = inputShape as Shape;
    }
    // Infer the dimension of our hidden feature size from the build shape.
    const hiddenDim =
      this.decoderSequenceShape[this.decoderSequenceShape.length - 1];
    // Attention head size is `hiddenDim` over the number of heads.
    const headDim = Math.floor(hiddenDim / this.numHeads);

    // Self attention layers.
    this.selfAttentionLayer = new CachedMultiHeadAttention({
      numHeads: this.numHeads,
      keyDim: headDim,
      dropout: this.dropout,
      kernelInitializer: getInitializer(this.kernelInitializer.getClassName()),
      biasInitializer: getInitializer(this.biasInitializer.getClassName()),
    });

    this.selfAttentionLayer.buildFromSignature(
      this.decoderSequenceShape, this.decoderSequenceShape);

    this.selfAttentionLayernorm =
      new LayerNormalization({epsilon: this.layerNormEpsilon});

    this.selfAttentionLayernorm.build(this.decoderSequenceShape);
    this.selfAttentionDropout = new Dropout({rate: this.dropout});

    // Cross attention layers are optional.
    // TODO(pforderique): Add cross attention layers.

    // Feedforward layers.
    this.feedforwardIntermediateDense = new Dense({
      units: this.intermediateDim,
      activation: this.activation.getClassName() as ActivationIdentifier,
      kernelInitializer: getInitializer(this.kernelInitializer.getClassName()),
      biasInitializer: getInitializer(this.biasInitializer.getClassName()),
    });
    this.feedforwardIntermediateDense.build(this.decoderSequenceShape);
    this.feedforwardOutputDense = new Dense({
      units: hiddenDim,
      kernelInitializer: getInitializer(this.kernelInitializer.getClassName()),
      biasInitializer: getInitializer(this.biasInitializer.getClassName()),
    });
    const intermediateShape = this.decoderSequenceShape.slice();
    intermediateShape[intermediateShape.length - 1] = this.intermediateDim;
    this.feedforwardOutputDense.build(intermediateShape);
    this.feedforwardLayernorm =
      new LayerNormalization({epsilon: this.layerNormEpsilon});
    this.feedforwardLayernorm.build(this.decoderSequenceShape);
    this.feedforwardDropout = new Dropout({rate: this.dropout});
    // Create layers based on input shape.
    this.built = true;
  }

  override apply(
      decoderSequence: Tensor|SymbolicTensor,
      kwargs?: TransformerDecoderOptions): Tensor|SymbolicTensor {
    if (!this.built) {
      const decoderSequenceShape = decoderSequence.shape;
      const encoderSequenceShape =
        kwargs && kwargs.encoderSequence ? kwargs.encoderSequence.shape : null;
      this.build([decoderSequenceShape, encoderSequenceShape]);
    }
    return super.apply(decoderSequence, kwargs) as Tensor|SymbolicTensor;
  }

  override call(
      decoderSequence: Tensor, kwargs: TransformerDecoderOptions): Tensor {
    return this.callAndReturnCaches(decoderSequence, kwargs)[0];
  }

  /**
   * Forward pass of the TransformerDecoder.
   *
   * @returns One of three things, depending on call arguments:
   *   - `[outputs, null, null]`, if `selfAttentionCache` is `null`.
   *   - `[outputs, selfAttentionCache, null]`, if `selfAttentionCache` is
   *     set and the layer has no cross-attention.
   *   - `[outputs, selfAttentionCache, crossAttentionCache]`, if
   *     `selfAttentionCache` and `crossAttentionCache` are set and
   *     the layer has cross-attention.
   */
  callAndReturnCaches(
    decoderSequence: Tensor, kwargs: TransformerDecoderOptions
  ): [Tensor, Tensor, Tensor] {
    return tidy(() => {
      const hasEncoderSequence = kwargs.encoderSequence != null;
      const hasCrossAttention = this.selfCrossAttentionLayer != null;

      if (!hasCrossAttention && hasEncoderSequence) {
        throw new ValueError(
          'The number of call arguments to `TransformerDecoder` should ' +
          'not change. Use `layer.apply(decoderSequence, {encoderSequence})` ' +
          'to build a layer with cross attention, or ' +
          '`layer.apply (decoderSequence)` to build a layer without. ' +
          'This layer has been built without cross attention, but ' +
          'you are trying to call it with encoderSequence.'
        );
      } else if (hasCrossAttention && !hasEncoderSequence) {
        throw new ValueError(
          'The number of call arguments to `TransformerDecoder` should not ' +
          'change. Use `layer.apply(decoderSequence, {encoderSequence})` ' +
          'to build a layer with cross attention, or ' +
          '`layer.apply(decoderSequence)` to build a layer without. ' +
          'This layer has been built with cross attention, but ' +
          'you did not provide encoderSequence.'
        );
      }

      const hasSelfAttentionCache = kwargs.selfAttentionCache != null;
      const hasCrossAttentionCache = kwargs.crossAttentionCache != null;
      if (hasCrossAttention && (
        hasSelfAttentionCache !== hasCrossAttentionCache
      )) {
        throw new ValueError(
          'When calling `TransformerDecoder` with cross-attention (with both ' +
          '`encoderSequence` and `decoderSequence`), `selfAttentionCache` ' +
          'and `crossAttentionCache` should both be set or both be `null`.  ' +
          'One cannot be `null` while the other is not. Received: ' +
          `selfAttentionCache=${kwargs.selfAttentionCache}, ` +
          `crossAttentionCache=${kwargs.crossAttentionCache}.`
        );
      }

      const selfAttentionMask = this.computeSelfAttentionMask(
        decoderSequence,
        kwargs.decoderPaddingMask as Tensor,
        kwargs.decoderAttentionMask,
        kwargs.useCausalMask,
        kwargs.selfAttentionCache,
        kwargs.selfAttentionCacheUpdateIndex,
      );

      let x = decoderSequence; // Intermediate result.
      let selfAttentionCache = kwargs.selfAttentionCache;

      // Self attention block.
      let residual = x;
      if (this.normalizeFirst) {
        x = this.selfAttentionLayernorm.apply(x) as Tensor;
      }
      [x, selfAttentionCache] = this.selfAttentionLayer.callAndReturnCache(
        x,
        {
          value: x,
          attentionMask: selfAttentionMask,
          cache: selfAttentionCache,
          cacheUpdateIndex: kwargs.selfAttentionCacheUpdateIndex,
        }
      );
      x = this.selfAttentionDropout.apply(x) as Tensor;
      x = add(x, residual);
      if (!this.normalizeFirst) {
        x = this.selfAttentionLayernorm.apply(x) as Tensor;
      }

      // Cross attention is optional.
      // TODO(pforderique): Add cross attention logic for encoder-decoder arch.

      // Feedforward block.
      residual = x;
      if (this.normalizeFirst) {
        x = this.selfAttentionLayernorm.apply(x) as Tensor;
      }
      x = this.feedforwardIntermediateDense.apply(x) as Tensor;
      x = this.feedforwardOutputDense.apply(x) as Tensor;
      x = this.feedforwardDropout.apply(x) as Tensor;
      x = add(x, residual);
      if (!this.normalizeFirst) {
        x = this.selfAttentionLayernorm.apply(x) as Tensor;
      }

      if (selfAttentionCache != null) {
        if (hasCrossAttention) {
          return [x, selfAttentionCache, kwargs.crossAttentionCache];
        } else {
          return [x, selfAttentionCache, null];
        }
      }
      return [x, null, null];
    });
  }

  private computeSelfAttentionMask(
    decoderSequence: Tensor,
    decoderPaddingMask: Tensor,
    decoderAttentionMask: Tensor,
    useCasualMask: boolean,
    selfAttentionCache: Tensor,
    selfAttentionCacheUpdateIndex: number
  ): Tensor {
    const decoderMask = mergePaddingAndAttentionMask(
      decoderSequence, decoderPaddingMask, decoderAttentionMask);
    if(useCasualMask) {
      const batchSize = decoderSequence.shape[0];
      let inputLength = decoderSequence.shape[1];
      const outputLength = decoderSequence.shape[1];
      // We need to handle a rectangular causal mask when doing cached
      // decoding. For generative inference, `decoderSequence` will
      // generally be length 1, and `cache` will be the full generation length.
      if(selfAttentionCache != null) {
        inputLength = selfAttentionCache.shape[2];
      }

      const causalMask = computeCausalMask(
        batchSize,
        inputLength,
        outputLength,
        selfAttentionCacheUpdateIndex ?? 0
      );
      return decoderMask != null ? decoderMask.minimum(causalMask) : causalMask;
    }
    return decoderMask;
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      'intermediateDim': this.intermediateDim,
      'numHeads': this.numHeads,
      'dropout': this.dropout,
      'activation': serializeActivation(this.activation),
      'layerNormEpsilon': this.layerNormEpsilon,
      'kernelInitializer': serializeInitializer(this.kernelInitializer),
      'biasInitializer': serializeInitializer(this.biasInitializer),
      'normalizeFirst': this.normalizeFirst,
      'decoderSequenceShape': this.decoderSequenceShape,
      'encoderSequenceShape': this.encoderSequenceShape,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override computeOutputShape(decoderSequenceShape: Shape): Shape {
    return decoderSequenceShape;
  }
}
serialization.registerClass(TransformerDecoder);
