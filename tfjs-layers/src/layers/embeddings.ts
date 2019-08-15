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
import {notEqual, serialization, Tensor, tidy, zerosLike} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {Layer, LayerArgs} from '../engine/topology';
import {ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {Shape} from '../keras_format/common';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import * as generic_utils from '../utils/generic_utils';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/types_utils';
import {LayerVariable} from '../variables';


export declare interface EmbeddingLayerArgs extends LayerArgs {
  /**
   * Integer > 0. Size of the vocabulary, i.e. maximum integer index + 1.
   */
  inputDim: number;
  /**
   * Integer >= 0. Dimension of the dense embedding.
   */
  outputDim: number;
  /**
   * Initializer for the `embeddings` matrix.
   */
  embeddingsInitializer?: InitializerIdentifier|Initializer;
  /**
   * Regularizer function applied to the `embeddings` matrix.
   */
  embeddingsRegularizer?: RegularizerIdentifier|Regularizer;
  /**
   * Regularizer function applied to the activation.
   */
  activityRegularizer?: RegularizerIdentifier|Regularizer;
  /**
   * Constraint function applied to the `embeddings` matrix.
   */
  embeddingsConstraint?: ConstraintIdentifier|Constraint;
  /**
   * Whether the input value 0 is a special "padding" value that should be
   * masked out. This is useful when using recurrent layers which may take
   * variable length input.
   *
   * If this is `True` then all subsequent layers in the model need to support
   * masking or an exception will be raised. If maskZero is set to `True`, as a
   * consequence, index 0 cannot be used in the vocabulary (inputDim should
   * equal size of vocabulary + 1).
   */
  maskZero?: boolean;
  /**
   * Length of input sequences, when it is constant.
   *
   * This argument is required if you are going to connect `flatten` then
   * `dense` layers upstream (without it, the shape of the dense outputs cannot
   * be computed).
   */
  inputLength?: number|number[];
}

export class Embedding extends Layer {
  /** @nocollapse */
  static className = 'Embedding';
  private inputDim: number;
  private outputDim: number;
  private embeddingsInitializer: Initializer;
  private maskZero: boolean;
  private inputLength: number|number[];

  private embeddings: LayerVariable = null;

  readonly DEFAULT_EMBEDDINGS_INITIALIZER: InitializerIdentifier =
      'randomUniform';
  private readonly embeddingsRegularizer?: Regularizer;
  private readonly embeddingsConstraint?: Constraint;

  constructor(args: EmbeddingLayerArgs) {
    super(args);
    if (args.batchInputShape == null && args.inputShape == null) {
      // Porting Note: This logic is copied from Layer's constructor, since we
      // can't do exactly what the Python constructor does for Embedding().
      // Specifically, the super constructor can not be called after the
      // mutation of the `config` argument.
      let batchSize: number = null;
      if (args.batchSize != null) {
        batchSize = args.batchSize;
      }
      if (args.inputLength == null) {
        // Fix super-constructor to what it would have done if
        // 'config.inputShape' were (None, )
        this.batchInputShape = [batchSize, null];
      } else {
        // Fix super-constructor to what it would have done if
        // 'config.inputShape' were (config.inputLength, )
        this.batchInputShape =
            [batchSize].concat(generic_utils.toList(args.inputLength));
      }
    }
    this.inputDim = args.inputDim;
    generic_utils.assertPositiveInteger(this.inputDim, 'inputDim');
    this.outputDim = args.outputDim;
    generic_utils.assertPositiveInteger(this.outputDim, 'outputDim');
    this.embeddingsInitializer = getInitializer(
        args.embeddingsInitializer || this.DEFAULT_EMBEDDINGS_INITIALIZER);
    this.embeddingsRegularizer = getRegularizer(args.embeddingsRegularizer);
    this.activityRegularizer = getRegularizer(args.activityRegularizer);
    this.embeddingsConstraint = getConstraint(args.embeddingsConstraint);
    this.maskZero = args.maskZero;
    this.supportsMasking = args.maskZero;
    this.inputLength = args.inputLength;
  }

  public build(inputShape: Shape|Shape[]): void {
    this.embeddings = this.addWeight(
        'embeddings', [this.inputDim, this.outputDim], this.dtype,
        this.embeddingsInitializer, this.embeddingsRegularizer, true,
        this.embeddingsConstraint);
    this.built = true;
  }

  // Override warnOnIncompatibleInputShape because an embedding layer allows
  // the input to have varying ranks.
  protected warnOnIncompatibleInputShape(inputShape: Shape) {}

  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor {
    return tidy(() => {
      if (!this.maskZero) {
        return null;
      } else {
        inputs = getExactlyOneTensor(inputs);
        return notEqual(inputs, zerosLike(inputs as Tensor));
      }
    });
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
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
      for (let k = 0; k < inLens.length; ++k) {
        const s1 = inLens[k];
        const s2 = inputShape[k + 1];
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

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      // Embedding layer accepts only a single input.
      let input = getExactlyOneTensor(inputs);
      if (input.dtype !== 'int32') {
        input = K.cast(input, 'int32');
      }
      const output = K.gather(this.embeddings.read(), input.as1D());
      return output.reshape(
          getExactlyOneShape(this.computeOutputShape(input.shape)));
    });
  }

  getConfig(): serialization.ConfigDict {
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
serialization.registerClass(Embedding);
