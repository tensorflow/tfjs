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
 * Exported functions.
 */

// tslint:disable:max-line-length
import {doc} from '@tensorflow/tfjs-core';

import {MaxNorm, MaxNormConfig, MinMaxNorm, MinMaxNormConfig, NonNeg, UnitNorm, UnitNormConfig} from './constraints';
import {ContainerConfig, Input, InputConfig, InputLayer, InputLayerConfig, Layer, LayerConfig} from './engine/topology';
import {Model} from './engine/training';
import {Constant, ConstantConfig, GlorotNormal, GlorotUniform, HeNormal, Identity, IdentityConfig, LeCunNormal, Ones, RandomNormal, RandomNormalConfig, RandomUniform, RandomUniformConfig, SeedOnlyInitializerConfig, TruncatedNormal, TruncatedNormalConfig, VarianceScaling, VarianceScalingConfig, Zeros} from './initializers';
import {Conv1D, Conv2D, ConvLayerConfig} from './layers/convolutional';
import {DepthwiseConv2D, DepthwiseConv2DLayerConfig} from './layers/convolutional_depthwise';
import {Activation, ActivationLayerConfig, Dense, DenseLayerConfig, Dropout, DropoutLayerConfig, Flatten, RepeatVector, RepeatVectorLayerConfig} from './layers/core';
import {Embedding, EmbeddingLayerConfig} from './layers/embeddings';
import {Add, Average, Concatenate, ConcatenateLayerConfig, Maximum, Minimum, Multiply} from './layers/merge';
import {BatchNormalization, BatchNormalizationLayerConfig} from './layers/normalization';
import {AvgPooling1D, AvgPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalPooling2DLayerConfig, MaxPooling1D, MaxPooling2D, Pooling1DLayerConfig, Pooling2DLayerConfig} from './layers/pooling';
import {GRU, GRUCell, GRUCellLayerConfig, GRULayerConfig, LSTM, LSTMCell, LSTMCellLayerConfig, LSTMLayerConfig, RNN, RNNCell, RNNLayerConfig, SimpleRNN, SimpleRNNCell, SimpleRNNCellLayerConfig, SimpleRNNLayerConfig, StackedRNNCells, StackedRNNCellsConfig} from './layers/recurrent';
import {Bidirectional, BidirectionalLayerConfig, TimeDistributed, WrapperLayerConfig} from './layers/wrappers';
import {loadModelInternal, Sequential, SequentialConfig} from './models';
import {l1, L1Config, L1L2, L1L2Config, l2, L2Config} from './regularizers';
import {SymbolicTensor} from './types';

// tslint:enable:max-line-length

export class ModelExports {
  // TODO(cais): Add doc string to all the public static functions in this
  //   class; include exectuable JavaScript code snippets where applicable
  //   (b/74074458).

  // Model and related factory methods.

  /**
   * A model is a data structure that consists of `Layers` and defines inputs
   * and outputs.
   *
   * When creating a `Model`, specify its input(s) and output(s). Layers
   * are used to wire input(s) to output(s).
   *
   * For example, the following code snippet defines a model consisting of
   * two `dense` layers, with 10 and 4 units, respectively.
   *
   * ```js
   * // Define input, which has a size of 5 (not including batch dimension).
   * const input = tf.input({shape: [5]});
   *
   * // First dense layer uses relu activation.
   * const denseLayer1 = tf.layers.dense({units: 10, activation: 'relu'});
   * // Second dense layer uses softmax activation.
   * const denseLayer2 = tf.layers.dense({units: 2, activation: 'softmax'});
   *
   * // Obtain the output symbolic tensor by applying the layers on the input.
   * const output = denseLayer2.apply(denseLayer1.apply(input));
   *
   * // Create the model based on the inputs.
   * const model = tf.model({inputs: input, outputs: output});
   *
   * // The model can be used for training, evaluation and prediction.
   * // For example, the following line runs prediction with the model on
   * // some fake data.
   * model.predict(tf.ones([2, 5])).print();
   * ```
   * See also:
   *   `sequential`, `loadModel`.
   */

  @doc({heading: 'Models', subheading: 'Creation', configParamIndices: [0]})
  static model(config: ContainerConfig): Model {
    return new Model(config);
  }

  /**
   * Creates a `Sequential` model.  A sequential model is any model where the
   * outputs of one layer are the inputs to the next layer, i.e. the model
   * topology is a simple 'stack' of layers, with no branching or skipping.
   *
   * This means that the first layer passed to a Sequential model should have a
   * defined input shape. What that means is that it should have received an
   * `inputShape` or `batchInputShape` argument, or for some type of layers
   * (recurrent, Dense...) an `inputDim` argument.
   *
   * Examples:
   *
   * ```js
   * const model = tf.sequential();
   *
   * // First layer must have a defined input shape
   * model.add(tf.layers.dense({units: 32, inputShape: [50]}));
   * // Afterwards, TF.js does automatic shape inference.
   * model.add(tf.layers.dense({units: 4}));
   *
   * // Inspect the inferred shape of the model's output, which equals
   * // `[null, 4]`. The 1st dimension is the undetermined batch dimension; the
   * // 2nd is the output size of the model's last layer.
   * console.log(model.outputs[0].shape);
   * ```
   *
   * It is also possible to specify a batch size (with potentially undetermined
   * batch dimension, denoted by "null") for the first layer using the
   * `batchInputShape` key. The following example is equivalent to the above:
   *
   * ```js
   * const model = tf.sequential();
   *
   * // First layer must have a defined input shape
   * model.add(tf.layers.dense({units: 32, batchInputShape: [null, 50]}));
   * // Afterwards, TF.js does automatic shape inference.
   * model.add(tf.layers.dense({units: 4}));
   *
   * // Inspect the inferred shape of the model's output.
   * console.log(model.outputs[0].shape);
   * ```
   *
   * You can also use an `Array` of already-constructed `Layer`s to create
   * a `Sequential` model:
   *
   * ```js
   * const model = tf.sequential({
   *   layers: [tf.layers.dense({units: 32, inputShape: [50]}),
   *            tf.layers.dense({units: 4})]
   * });
   * console.log(model.outputs[0].shape);
   * ```
   */
  @doc({heading: 'Models', subheading: 'Creation', configParamIndices: [0]})
  static sequential(config?: SequentialConfig): Sequential {
    return new Sequential(config);
  }

  @doc({
    heading: 'Models',
    subheading: 'Loading',
    useDocsFrom: 'loadModelInternal'
  })
  static loadModel(modelConfigPath: string): Promise<Model> {
    return loadModelInternal(modelConfigPath);
  }

  @doc({
    heading: 'Models',
    subheading: 'Inputs',
    useDocsFrom: 'Input',
    configParamIndices: [0]
  })
  static input(config: InputConfig): SymbolicTensor {
    return Input(config);
  }

  @doc({
    heading: 'Models',
    subheading: 'Inputs',
    useDocsFrom: 'InputLayer',
    configParamIndices: [0]
  })
  static inputLayer(config: InputLayerConfig): Layer {
    return new InputLayer(config);
  }
}

export class LayerExports {
  static Layer = Layer;
  static RNNCell = RNNCell;

  // TODO(cais): Add doc string to all the public static functions in this
  //   class; include exectuable JavaScript code snippets where applicable
  //   (b/74074458).

  // Convolutional Layers.

  @doc({
    heading: 'Layers',
    subheading: 'Convolutional',
    namespace: 'layers',
    useDocsFrom: 'Conv1D',
    configParamIndices: [0]
  })
  static conv1d(config: ConvLayerConfig): Layer {
    return new Conv1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Convolutional',
    namespace: 'layers',
    useDocsFrom: 'Conv2D',
    configParamIndices: [0]
  })
  static conv2d(config: ConvLayerConfig): Layer {
    return new Conv2D(config);
  }

  // Convolutional (depthwise) Layers.

  @doc({
    heading: 'Layers',
    subheading: 'Convolutional',
    namespace: 'layers',
    useDocsFrom: 'DepthwiseConv2D',
    configParamIndices: [0]
  })
  static depthwiseConv2d(config: DepthwiseConv2DLayerConfig): Layer {
    return new DepthwiseConv2D(config);
  }

  // Core Layers.
  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Activation',
    configParamIndices: [0]
  })
  static activation(config: ActivationLayerConfig): Layer {
    return new Activation(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Dense',
    configParamIndices: [0]
  })
  static dense(config: DenseLayerConfig): Layer {
    return new Dense(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Dropout',
    configParamIndices: [0]
  })
  static dropout(config: DropoutLayerConfig): Layer {
    return new Dropout(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Flatten',
    configParamIndices: [0]
  })
  static flatten(config?: LayerConfig): Layer {
    return new Flatten(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'RepeatVector',
    configParamIndices: [0]
  })
  static repeatVector(config: RepeatVectorLayerConfig): Layer {
    return new RepeatVector(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Embedding',
    configParamIndices: [0]
  })
  static embedding(config: EmbeddingLayerConfig): Layer {
    return new Embedding(config);
  }

  // Merge Layers.

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Add',
    configParamIndices: [0]
  })
  static add(config: LayerConfig): Layer {
    return new Add(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Average',
    configParamIndices: [0]
  })
  static average(config: LayerConfig): Layer {
    return new Average(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Concatenate',
    configParamIndices: [0]
  })
  static concatenate(config: ConcatenateLayerConfig): Layer {
    return new Concatenate(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Maximum',
    configParamIndices: [0]
  })
  static maximum(config: LayerConfig): Layer {
    return new Maximum(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Minimum',
    configParamIndices: [0]
  })
  static minimum(config: LayerConfig): Layer {
    return new Minimum(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Multiply',
    configParamIndices: [0]
  })
  static multiply(config: LayerConfig): Layer {
    return new Multiply(config);
  }

  // Normalization Layers.

  @doc({
    heading: 'Layers',
    subheading: 'Normalization',
    namespace: 'layers',
    useDocsFrom: 'BatchNormalization',
    configParamIndices: [0]
  })
  static batchNormalization(config: BatchNormalizationLayerConfig): Layer {
    return new BatchNormalization(config);
  }

  // Pooling Layers.
  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'AvgPooling1D',
    configParamIndices: [0]
  })
  static avgPooling1d(config: Pooling1DLayerConfig): Layer {
    return new AvgPooling1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'AvgPooling2D',
    configParamIndices: [0]
  })
  static avgPooling2d(config: Pooling2DLayerConfig): Layer {
    return new AvgPooling2D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'GlobalAveragePooling1D',
    configParamIndices: [0]
  })
  static globalAveragePooling1d(config: LayerConfig): Layer {
    return new GlobalAveragePooling1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'GlobalAveragePooling2D',
    configParamIndices: [0]
  })
  static globalAveragePooling2d(config: GlobalPooling2DLayerConfig): Layer {
    return new GlobalAveragePooling2D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'GlobalMaxPooling1D',
    configParamIndices: [0]
  })
  static globalMaxPooling1d(config: LayerConfig): Layer {
    return new GlobalMaxPooling1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'GlobalMaxPooling2D',
    configParamIndices: [0]
  })
  static globalMaxPooling2d(config: GlobalPooling2DLayerConfig): Layer {
    return new GlobalMaxPooling2D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'MaxPooling1D',
    configParamIndices: [0]
  })
  static maxPooling1d(config: Pooling1DLayerConfig): Layer {
    return new MaxPooling1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'MaxPooling2D',
    configParamIndices: [0]
  })
  static maxPooling2d(config: Pooling2DLayerConfig): Layer {
    return new MaxPooling2D(config);
  }

  // Recurrent Layers.

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'GRU',
    configParamIndices: [0]
  })
  static gru(config: GRULayerConfig): Layer {
    return new GRU(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'GRUCell',
    configParamIndices: [0]
  })
  static gruCell(config: GRUCellLayerConfig): RNNCell {
    return new GRUCell(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'LSTM',
    configParamIndices: [0]
  })
  static lstm(config: LSTMLayerConfig): Layer {
    return new LSTM(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'LSTMCell',
    configParamIndices: [0]
  })
  static lstmCell(config: LSTMCellLayerConfig): RNNCell {
    return new LSTMCell(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'SimpleRNN',
    configParamIndices: [0]
  })
  static simpleRNN(config: SimpleRNNLayerConfig): Layer {
    return new SimpleRNN(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'SimpleRNNCell',
    configParamIndices: [0]
  })
  static simpleRNNCell(config: SimpleRNNCellLayerConfig): RNNCell {
    return new SimpleRNNCell(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'RNN',
    configParamIndices: [0]
  })
  static rnn(config: RNNLayerConfig): RNN {
    return new RNN(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'RNN',
    configParamIndices: [0]
  })
  static stackedRNNCells(config: StackedRNNCellsConfig): RNNCell {
    return new StackedRNNCells(config);
  }

  // Wrapper Layers.

  @doc({
    heading: 'Layers',
    subheading: 'Wrapper',
    namespace: 'layers',
    useDocsFrom: 'Bidirectional',
    configParamIndices: [0]
  })
  static bidirectional(config: BidirectionalLayerConfig): Layer {
    return new Bidirectional(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Wrapper',
    namespace: 'layers',
    useDocsFrom: 'TimeDistributed',
    configParamIndices: [0]
  })
  static timeDistributed(config: WrapperLayerConfig): Layer {
    return new TimeDistributed(config);
  }
}

export class ConstraintExports {
  @doc({
    heading: 'Constraints',
    namespace: 'constraints',
    useDocsFrom: 'MaxNorm',
    configParamIndices: [0]
  })
  static maxNorm(config: MaxNormConfig): MaxNorm {
    return new MaxNorm(config);
  }

  @doc({
    heading: 'Constraints',
    namespace: 'constraints',
    useDocsFrom: 'UnitNorm',
    configParamIndices: [0]
  })
  static unitNorm(config: UnitNormConfig): UnitNorm {
    return new UnitNorm(config);
  }

  @doc(
      {heading: 'Constraints', namespace: 'constraints', useDocsFrom: 'NonNeg'})
  static nonNeg(): NonNeg {
    return new NonNeg();
  }

  @doc({
    heading: 'Constraints',
    namespace: 'constraints',
    useDocsFrom: 'MinMaxNormConfig',
    configParamIndices: [0]
  })
  static minMaxNorm(config: MinMaxNormConfig): MinMaxNorm {
    return new MinMaxNorm(config);
  }
}

export class InitializerExports {
  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'Zeros'
  })
  static zeros(): Zeros {
    return new Zeros();
  }

  @doc(
      {heading: 'Initializers', namespace: 'initializers', useDocsFrom: 'Ones'})
  static ones(): Ones {
    return new Ones();
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'Constant',
    configParamIndices: [0]

  })
  static constant(config: ConstantConfig): Constant {
    return new Constant(config);
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'RandomUniform',
    configParamIndices: [0]

  })
  static randomUniform(config: RandomUniformConfig): RandomUniform {
    return new RandomUniform(config);
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'RandomNormal',
    configParamIndices: [0]

  })
  static randomNormal(config: RandomNormalConfig): RandomNormal {
    return new RandomNormal(config);
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'TruncatedNormal',
    configParamIndices: [0]

  })
  static truncatedNormal(config: TruncatedNormalConfig): TruncatedNormal {
    return new TruncatedNormal(config);
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'Identity',
    configParamIndices: [0]

  })
  static identity(config: IdentityConfig): Identity {
    return new Identity(config);
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'VarianceScaling',
    configParamIndices: [0]

  })
  static varianceScaling(config: VarianceScalingConfig): VarianceScaling {
    return new VarianceScaling(config);
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'GlorotUniform',
    configParamIndices: [0]

  })
  static glorotUniform(config: SeedOnlyInitializerConfig): GlorotUniform {
    return new GlorotUniform(config);
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'GlorotNormal',
    configParamIndices: [0]

  })
  static glorotNormal(config: SeedOnlyInitializerConfig): GlorotNormal {
    return new GlorotNormal(config);
  }

  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'HeNormal',
    configParamIndices: [0]

  })
  static heNormal(config: SeedOnlyInitializerConfig): HeNormal {
    return new HeNormal(config);
  }


  @doc({
    heading: 'Initializers',
    namespace: 'initializers',
    useDocsFrom: 'LeCunNormal',
    configParamIndices: [0]

  })
  static leCunNormal(config: SeedOnlyInitializerConfig): LeCunNormal {
    return new LeCunNormal(config);
  }
}

export class RegularizerExports {
  @doc(
      {heading: 'Regularizers', namespace: 'regularizers', useDocsFrom: 'L1L2'})
  static l1l2(config?: L1L2Config): L1L2 {
    return new L1L2(config);
  }

  @doc(
      {heading: 'Regularizers', namespace: 'regularizers', useDocsFrom: 'L1L2'})
  static l1(config?: L1Config): L1L2 {
    return l1(config);
  }

  @doc(
      {heading: 'Regularizers', namespace: 'regularizers', useDocsFrom: 'L1L2'})
  static l2(config?: L2Config): L1L2 {
    return l2(config);
  }
}
