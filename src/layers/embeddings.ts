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
 * TensorFlow.js Layers: Embedding Layer.
 *
 * Original source: keras/constraints.py
 */
import {Tensor} from 'deeplearn';
import * as _ from 'underscore';

// tslint:disable:max-line-length
import * as K from '../backend/deeplearnjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Shape} from '../types';
import {ConfigDict, LayerVariable} from '../types';
import * as generic_utils from '../utils/generic_utils';
import {getExactlyOneShape} from '../utils/generic_utils';
// tslint:enable:max-line-length

export interface EmbeddingLayerConfig extends LayerConfig {
  /**
   * inputDim:
   *     int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
   */
  inputDim: number;
  /**
   * outputDim:
   *     int >= 0. Dimension of the dense embedding.
   */
  outputDim: number;
  /**
   * embeddingsInitializer:
   *     Initializer for the `embeddings` matrix
   *     (see [initializers](../initializers.md)).
   */
  embeddingsInitializer?: InitializerIdentifier|Initializer;
  /**
   * embeddingsRegularizer:
   *     Regularizer function applied to the `embeddings` matrix.
   */
  embeddingsRegularizer?: RegularizerIdentifier|Regularizer;
  /**
   * activityRegularizer:
   *     Regularizer function applied to the activation.
   */
  activityRegularizer?: RegularizerIdentifier|Regularizer;
  /**
   *  embeddingsConstraint: Constraint function applied to
   *     the `embeddings` matrix
   *     (see [constraints](../constraints.md)).
   */
  embeddingsConstraint?: ConstraintIdentifier|Constraint;
  /**
   * mask_zero: Whether or not the input value 0 is a special "padding"
   *     value that should be masked out.
   *     This is useful when using [recurrent layers](recurrent.md)
   *     which may take variable length input.
   *     If this is `True` then all subsequent layers
   *     in the model need to support masking or an exception will be raised.
   *     If mask_zero is set to True, as a consequence, index 0 cannot be
   *     used in the vocabulary (input_dim should equal size of
   *     vocabulary + 1).
   */
  maskZero?: boolean;
  /**
   * inputLength:
   *     Length of input sequences, when it is constant.
   *     This argument is required if you are going to connect
   *     `Flatten` then `Dense` layers upstream
   *     (without it, the shape of the dense outputs cannot be computed).
   */
  inputLength?: number|number[];
}

/**
 * Turns positive integers (indexes) into dense vectors of fixed size.
 * eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
 *
 * Input shape: 2D tensor with shape: `[batchSize, sequenceLength]`.
 *
 * Output shape: 3D tensor with shape: `[batchSize, sequenceLength, outputDim]`.
 */
export class Embedding extends Layer {
  private inputDim: number;
  private outputDim: number;
  private embeddingsInitializer: Initializer;
  private maskZero: boolean;
  private inputLength: number|number[];

  private embeddings: LayerVariable = null;

  readonly DEFAULT_EMBEDDINGS_INITIALIZER = 'RandomUniform';
  private readonly embeddingsRegularizer?: Regularizer;
  private readonly embeddingsConstraint?: Constraint;

  constructor(config: EmbeddingLayerConfig) {
    super(config);
    if (config.batchInputShape == null && config.inputShape == null) {
      // Porting Note: This logic is copied from Layer's constructor, since we
      // can't do exactly what the Python constructor does for Embedding().
      // Specifically, the super constructor can not be called after the
      // mutation of the `config` argument.
      let batchSize: number = null;
      if (config.batchSize != null) {
        batchSize = config.batchSize;
      }
      if (config.inputLength == null) {
        // Fix super-constructor to what it would have done if
        // 'config.inputShape' were (None, )
        this.batchInputShape = [batchSize, null];
      } else {
        // Fix super-constructor to what it would have done if
        // 'config.inputShape' were (config.inputLength, )
        this.batchInputShape =
            [batchSize].concat(generic_utils.toList(config.inputLength));
      }
    }
    this.inputDim = config.inputDim;
    this.outputDim = config.outputDim;
    this.embeddingsInitializer = getInitializer(
        config.embeddingsInitializer || this.DEFAULT_EMBEDDINGS_INITIALIZER);
    this.embeddingsRegularizer = getRegularizer(config.embeddingsRegularizer);
    this.activityRegularizer = getRegularizer(config.activityRegularizer);
    this.embeddingsConstraint = getConstraint(config.embeddingsConstraint);
    this.maskZero = config.maskZero;
    this.inputLength = config.inputLength;
  }

  public build(inputShape: Shape|Shape[]): void {
    this.embeddings = this.addWeight(
        'embeddings', [this.inputDim, this.outputDim], this.dtype,
        this.embeddingsInitializer, this.embeddingsRegularizer, true,
        this.embeddingsConstraint);
    this.built = true;
  }

  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor {
    throw new NotImplementedError(
        'computeMask has not been implemented for Embedding yet');
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    if (this.inputLength == null) {
      return [...inputShape, this.outputDim];
    }
    // inputLength can be an array if input is 3D or higher.
    const inLens: number[] = generic_utils.toList(this.inputLength);
    if (inLens.length !== inputShape.length - 1) {
      throw new ValueError(
          `"inputLength" is ${this.inputLength}, but received ` +
          `input shape has shape ${inputShape}`);
    } else {
      let i = 0;
      for (const [s1, s2] of _.zip(inLens, inputShape.slice(1))) {
        if ((s1 != null) && (s2 != null) && (s1 !== s2)) {
          throw new ValueError(
              `"inputLength" is ${this.inputLength}, but received ` +
              `input shape has shape ${inputShape}`);
        } else if (s1 == null) {
          inLens[i] = s2;
        }
        i++;
      }
    }
    return [inputShape[0], ...inLens, this.outputDim];
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    this.invokeCallHook();
    // Embedding layer accepts only a single input.
    let input = generic_utils.getExactlyOneTensor(inputs);
    if (K.dtype(input) !== 'int32') {
      input = K.cast(input, 'int32');
    }
    const output = K.gather(this.embeddings.read(), input.as1D());
    return K.reshape(
        output, getExactlyOneShape(this.computeOutputShape(input.shape)));
  }

  getConfig(): ConfigDict {
    const config = {
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      embeddingsInitializer: serializeInitializer(this.embeddingsInitializer),
      embeddingsRegularizer: serializeRegularizer(this.embeddingsRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      embeddingsConstraint: serializeConstraint(this.embeddingsConstraint),
      maskZero: this.maskZero,
      inputLength: this.inputLength
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

generic_utils.ClassNameMap.register('Embedding', Embedding);
