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
import {modelFromJSON as modelFromJSONInternal, Sequential, SequentialConfig} from './models';
import {ConfigDict, SymbolicTensor} from './types';

// tslint:enable:max-line-length

export class ModelExports {
  // TODO(cais): Add doc string to all the public static functions in this
  //   class; include exectuable JavaScript code snippets where applicable
  //   (b/74074458).

  // Model and related factory methods.

  static model(config: ContainerConfig): Model {
    return new Model(config);
  }

  static sequential(config: SequentialConfig): Sequential {
    return new Sequential(config);
  }

  static modelFromJSON(jsonString: string, customObjects?: ConfigDict): Model {
    return modelFromJSONInternal(jsonString, customObjects);
  }

  static input(config: InputConfig): SymbolicTensor {
    return Input(config);
  }

  static inputLayer(config: InputLayerConfig): InputLayer {
    return new InputLayer(config);
  }
}

export class LayerExports {
  // TODO(cais): Add doc string to all the public static functions in this
  //   class; include exectuable JavaScript code snippets where applicable
  //   (b/74074458).

  // Convolutional Layers.

  static conv1d(config: ConvLayerConfig): Conv1D {
    return new Conv1D(config);
  }

  static conv2d(config: ConvLayerConfig) {
    return new Conv2D(config);
  }

  // Convolutional (depthwise) Layers.

  static depthwiseConv2d(config: DepthwiseConv2DLayerConfig): DepthwiseConv2D {
    return new DepthwiseConv2D(config);
  }

  // Core Layers.

  static activation(config: ActivationLayerConfig): Activation {
    return new Activation(config);
  }

  static dense(config: DenseLayerConfig): Dense {
    return new Dense(config);
  }

  static dropout(config: DropoutLayerConfig): Dropout {
    return new Dropout(config);
  }

  static flatten(config?: LayerConfig): Flatten {
    return new Flatten(config);
  }

  static repeatVector(config: RepeatVectorLayerConfig): RepeatVector {
    return new RepeatVector(config);
  }

  static embedding(config: EmbeddingLayerConfig): Embedding {
    return new Embedding(config);
  }

  // Merge Layers.

  static add(config: MergeLayerConfig): Add {
    return new Add(config);
  }

  static average(config: MergeLayerConfig): Average {
    return new Average(config);
  }

  static concatenate(config: ConcatenateLayerConfig): Concatenate {
    return new Concatenate(config);
  }

  static maximum(config: MergeLayerConfig): Maximum {
    return new Maximum(config);
  }

  static minimum(config: MergeLayerConfig): Minimum {
    return new Minimum(config);
  }

  static multiply(config: MergeLayerConfig): Multiply {
    return new Multiply(config);
  }

  // Normalization Layers.

  static batchNormalization(config: BatchNormalizationLayerConfig):
      BatchNormalization {
    return new BatchNormalization(config);
  }

  // Pooling Layers.

  static avgPooling1d(config: Pooling1DLayerConfig): AvgPooling1D {
    return new AvgPooling1D(config);
  }

  static avgPooling2d(config: Pooling2DLayerConfig): AvgPooling2D {
    return new AvgPooling2D(config);
  }

  static globalAvereagePooling1d(config: LayerConfig): GlobalAveragePooling1D {
    return new GlobalAveragePooling1D(config);
  }

  static globalAvereagePooling2d(config: GlobalPooling2DLayerConfig):
      GlobalAveragePooling2D {
    return new GlobalAveragePooling2D(config);
  }

  static globalMaxPooling1d(config: LayerConfig): GlobalMaxPooling1D {
    return new GlobalMaxPooling1D(config);
  }
  static globalMaxPooling2d(config: GlobalPooling2DLayerConfig):
      GlobalMaxPooling2D {
    return new GlobalMaxPooling2D(config);
  }

  static maxPooling1d(config: Pooling1DLayerConfig): MaxPooling1D {
    return new MaxPooling1D(config);
  }

  static maxPooling2d(config: Pooling2DLayerConfig): MaxPooling2D {
    return new MaxPooling2D(config);
  }

  // Recurrent Layers.

  static gru(config: GRULayerConfig): GRU {
    return new GRU(config);
  }

  static gruCell(config: GRUCellLayerConfig): GRUCell {
    return new GRUCell(config);
  }

  static lstm(config: LSTMLayerConfig): LSTM {
    return new LSTM(config);
  }

  static lstmCell(config: LSTMCellLayerConfig): LSTMCell {
    return new LSTMCell(config);
  }

  static simpleRNN(config: SimpleRNNLayerConfig): SimpleRNN {
    return new SimpleRNN(config);
  }

  static simpleRNNCell(config: SimpleRNNCellLayerConfig): SimpleRNNCell {
    return new SimpleRNNCell(config);
  }

  // Wrapper Layers.

  static bidirectional(config: BidirectionalLayerConfig): Bidirectional {
    return new Bidirectional(config);
  }

  static timeDistributed(config: WrapperLayerConfig): TimeDistributed {
    return new TimeDistributed(config);
  }
}
