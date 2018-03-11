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
import {doc} from 'deeplearn';

import {ContainerConfig, Input, InputConfig, InputLayer, InputLayerConfig, LayerConfig} from './engine/topology';
import {Model} from './engine/training';
import {Conv1D, Conv2D, ConvLayerConfig} from './layers/convolutional';
import {DepthwiseConv2D, DepthwiseConv2DLayerConfig} from './layers/convolutional_depthwise';
import {Activation, ActivationLayerConfig, Dense, DenseLayerConfig, Dropout, DropoutLayerConfig, Flatten, RepeatVector, RepeatVectorLayerConfig} from './layers/core';
import {Embedding, EmbeddingLayerConfig} from './layers/embeddings';
import {Add, Average, Concatenate, ConcatenateLayerConfig, Maximum, MergeLayerConfig, Minimum, Multiply} from './layers/merge';
import {BatchNormalization, BatchNormalizationLayerConfig} from './layers/normalization';
import {AvgPooling1D, AvgPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalPooling2DLayerConfig, MaxPooling1D, MaxPooling2D, Pooling1DLayerConfig, Pooling2DLayerConfig} from './layers/pooling';
import {GRU, GRUCell, GRUCellLayerConfig, GRULayerConfig, LSTM, LSTMCell, LSTMCellLayerConfig, LSTMLayerConfig, SimpleRNN, SimpleRNNCell, SimpleRNNCellLayerConfig, SimpleRNNLayerConfig} from './layers/recurrent';
import {Bidirectional, BidirectionalLayerConfig, TimeDistributed, WrapperLayerConfig} from './layers/wrappers';
import {loadModelInternal, ModelAndWeightsConfig, modelFromJSONInternal, Sequential, SequentialConfig} from './models';
import {ConfigDict, JsonValue, SymbolicTensor} from './types';

// tslint:enable:max-line-length

export class ModelExports {
  // TODO(cais): Add doc string to all the public static functions in this
  //   class; include exectuable JavaScript code snippets where applicable
  //   (b/74074458).

  // Model and related factory methods.

  @doc({
    heading: 'Models',
    subheading: 'Creation',
    useDocsFrom: 'Model',
    configParamIndices: [0]
  })
  static model(config: ContainerConfig): Model {
    return new Model(config);
  }

  @doc({
    heading: 'Models',
    subheading: 'Creation',
    useDocsFrom: 'Sequential',
    configParamIndices: [0]
  })
  static sequential(config: SequentialConfig): Sequential {
    return new Sequential(config);
  }

  @doc({
    heading: 'Models',
    subheading: 'Loading',
    useDocsFrom: 'loadModelInternal',
    configParamIndices: [0]
  })
  static async loadModel(modelAndWeights: string |
    ModelAndWeightsConfig): Promise<Model> {
    return loadModelInternal(modelAndWeights);
  }

  @doc({
    heading: 'Models',
    subheading: 'Loading',
    useDocsFrom: 'modelFromJSONInternal'
  })
  static modelFromJSON(
      jsonString: JsonValue|string, customObjects?: ConfigDict): Model {
    return modelFromJSONInternal(jsonString, customObjects);
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
  static inputLayer(config: InputLayerConfig): InputLayer {
    return new InputLayer(config);
  }
}

export class LayerExports {
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
  static conv1d(config: ConvLayerConfig): Conv1D {
    return new Conv1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Convolutional',
    namespace: 'layers',
    useDocsFrom: 'Conv2D',
    configParamIndices: [0]
  })
  static conv2d(config: ConvLayerConfig) {
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
  static depthwiseConv2d(config: DepthwiseConv2DLayerConfig): DepthwiseConv2D {
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
  static activation(config: ActivationLayerConfig): Activation {
    return new Activation(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Dense',
    configParamIndices: [0]
  })
  static dense(config: DenseLayerConfig): Dense {
    return new Dense(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Dropout',
    configParamIndices: [0]
  })
  static dropout(config: DropoutLayerConfig): Dropout {
    return new Dropout(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Flatten',
    configParamIndices: [0]
  })
  static flatten(config?: LayerConfig): Flatten {
    return new Flatten(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'RepeatVector',
    configParamIndices: [0]
  })
  static repeatVector(config: RepeatVectorLayerConfig): RepeatVector {
    return new RepeatVector(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Core',
    namespace: 'layers',
    useDocsFrom: 'Embedding',
    configParamIndices: [0]
  })
  static embedding(config: EmbeddingLayerConfig): Embedding {
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
  static add(config: MergeLayerConfig): Add {
    return new Add(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Average',
    configParamIndices: [0]
  })
  static average(config: MergeLayerConfig): Average {
    return new Average(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Concatenate',
    configParamIndices: [0]
  })
  static concatenate(config: ConcatenateLayerConfig): Concatenate {
    return new Concatenate(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Maximum',
    configParamIndices: [0]
  })
  static maximum(config: MergeLayerConfig): Maximum {
    return new Maximum(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Minimum',
    configParamIndices: [0]
  })
  static minimum(config: MergeLayerConfig): Minimum {
    return new Minimum(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Merge',
    namespace: 'layers',
    useDocsFrom: 'Multiply',
    configParamIndices: [0]
  })
  static multiply(config: MergeLayerConfig): Multiply {
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
  static batchNormalization(config: BatchNormalizationLayerConfig):
      BatchNormalization {
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
  static avgPooling1d(config: Pooling1DLayerConfig): AvgPooling1D {
    return new AvgPooling1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'AvgPooling2D',
    configParamIndices: [0]
  })
  static avgPooling2d(config: Pooling2DLayerConfig): AvgPooling2D {
    return new AvgPooling2D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'GlobalAveragePooling1D',
    configParamIndices: [0]
  })
  static globalAveragePooling1d(config: LayerConfig): GlobalAveragePooling1D {
    return new GlobalAveragePooling1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'GlobalAveragePooling2D',
    configParamIndices: [0]
  })
  static globalAveragePooling2d(config: GlobalPooling2DLayerConfig):
      GlobalAveragePooling2D {
    return new GlobalAveragePooling2D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'GlobalMaxPooling1D',
    configParamIndices: [0]
  })
  static globalMaxPooling1d(config: LayerConfig): GlobalMaxPooling1D {
    return new GlobalMaxPooling1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'GlobalMaxPooling2D',
    configParamIndices: [0]
  })
  static globalMaxPooling2d(config: GlobalPooling2DLayerConfig):
      GlobalMaxPooling2D {
    return new GlobalMaxPooling2D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'MaxPooling1D',
    configParamIndices: [0]
  })
  static maxPooling1d(config: Pooling1DLayerConfig): MaxPooling1D {
    return new MaxPooling1D(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Pooling',
    namespace: 'layers',
    useDocsFrom: 'MaxPooling2D',
    configParamIndices: [0]
  })
  static maxPooling2d(config: Pooling2DLayerConfig): MaxPooling2D {
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
  static gru(config: GRULayerConfig): GRU {
    return new GRU(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'GRUCell',
    configParamIndices: [0]
  })
  static gruCell(config: GRUCellLayerConfig): GRUCell {
    return new GRUCell(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'LSTM',
    configParamIndices: [0]
  })
  static lstm(config: LSTMLayerConfig): LSTM {
    return new LSTM(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'LSTMCell',
    configParamIndices: [0]
  })
  static lstmCell(config: LSTMCellLayerConfig): LSTMCell {
    return new LSTMCell(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'SimpleRNN',
    configParamIndices: [0]
  })
  static simpleRNN(config: SimpleRNNLayerConfig): SimpleRNN {
    return new SimpleRNN(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Recurrent',
    namespace: 'layers',
    useDocsFrom: 'SimpleRNNCell',
    configParamIndices: [0]
  })
  static simpleRNNCell(config: SimpleRNNCellLayerConfig): SimpleRNNCell {
    return new SimpleRNNCell(config);
  }

  // Wrapper Layers.

  @doc({
    heading: 'Layers',
    subheading: 'Wrapper',
    namespace: 'layers',
    useDocsFrom: 'Bidirectional',
    configParamIndices: [0]
  })
  static bidirectional(config: BidirectionalLayerConfig): Bidirectional {
    return new Bidirectional(config);
  }

  @doc({
    heading: 'Layers',
    subheading: 'Wrapper',
    namespace: 'layers',
    useDocsFrom: 'TimeDistributed',
    configParamIndices: [0]
  })
  static timeDistributed(config: WrapperLayerConfig): TimeDistributed {
    return new TimeDistributed(config);
  }
}
