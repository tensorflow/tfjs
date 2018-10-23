/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {InputLayer, InputLayerConfig} from './engine/input_layer';
import {Layer, LayerConfig} from './engine/topology';
import {input} from './exports';
import {ELU, ELULayerConfig, LeakyReLU, LeakyReLULayerConfig, ReLU, ReLULayerConfig, Softmax, SoftmaxLayerConfig, ThresholdedReLU, ThresholdedReLULayerConfig} from './layers/advanced_activations';
import {Conv1D, Conv2D, Conv2DTranspose, ConvLayerConfig, Cropping2D, Cropping2DLayerConfig, SeparableConv2D, SeparableConvLayerConfig, UpSampling2D, UpSampling2DLayerConfig} from './layers/convolutional';
import {DepthwiseConv2D, DepthwiseConv2DLayerConfig} from './layers/convolutional_depthwise';
import {Activation, ActivationLayerConfig, Dense, DenseLayerConfig, Dropout, DropoutLayerConfig, Flatten, Permute, PermuteLayerConfig, RepeatVector, RepeatVectorLayerConfig, Reshape, ReshapeLayerConfig} from './layers/core';
import {Embedding, EmbeddingLayerConfig} from './layers/embeddings';
import {Add, Average, Concatenate, ConcatenateLayerConfig, Dot, DotLayerConfig, Maximum, Minimum, Multiply} from './layers/merge';
import {BatchNormalization, BatchNormalizationLayerConfig} from './layers/normalization';
import {ZeroPadding2D, ZeroPadding2DLayerConfig} from './layers/padding';
import {AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalPooling2DLayerConfig, MaxPooling1D, MaxPooling2D, Pooling1DLayerConfig, Pooling2DLayerConfig} from './layers/pooling';
import {GRU, GRUCell, GRUCellLayerConfig, GRULayerConfig, LSTM, LSTMCell, LSTMCellLayerConfig, LSTMLayerConfig, RNN, RNNCell, RNNLayerConfig, SimpleRNN, SimpleRNNCell, SimpleRNNCellLayerConfig, SimpleRNNLayerConfig, StackedRNNCells, StackedRNNCellsConfig} from './layers/recurrent';
import {Bidirectional, BidirectionalLayerConfig, TimeDistributed, Wrapper, WrapperLayerConfig} from './layers/wrappers';


// TODO(cais): Add doc string to all the public static functions in this
//   class; include exectuable JavaScript code snippets where applicable
//   (b/74074458).

// Input Layer.
/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Inputs',
 *    namespace: 'layers',
 *   useDocsFrom: 'InputLayer',
 *   configParamIndices: [0]
 * }
 */
export function inputLayer(config: InputLayerConfig): Layer {
  return new InputLayer(config);
}

// Advanced Activation Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers',
 *   useDocsFrom: 'ELU',
 *   configParamIndices: [0]
 * }
 */
export function elu(config?: ELULayerConfig): Layer {
  return new ELU(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers',
 *   useDocsFrom: 'ReLU',
 *   configParamIndices: [0]
 * }
 */
export function reLU(config?: ReLULayerConfig): Layer {
  return new ReLU(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers',
 *   useDocsFrom: 'LeakyReLU',
 *   configParamIndices: [0]
 * }
 */
export function leakyReLU(config?: LeakyReLULayerConfig): Layer {
  return new LeakyReLU(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers',
 *   useDocsFrom: 'Softmax',
 *   configParamIndices: [0]
 * }
 */
export function softmax(config?: SoftmaxLayerConfig): Layer {
  return new Softmax(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers',
 *   useDocsFrom: 'ThresholdedReLU',
 *   configParamIndices: [0]
 * }
 */
export function thresholdedReLU(config?: ThresholdedReLULayerConfig): Layer {
  return new ThresholdedReLU(config);
}

// Convolutional Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Convolutional',
 *   namespace: 'layers',
 *   useDocsFrom: 'Conv1D',
 *   configParamIndices: [0]
 * }
 */
export function conv1d(config: ConvLayerConfig): Layer {
  return new Conv1D(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Convolutional',
 *   namespace: 'layers',
 *   useDocsFrom: 'Conv2D',
 *   configParamIndices: [0]
 * }
 */
export function conv2d(config: ConvLayerConfig): Layer {
  return new Conv2D(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Convolutional',
 *   namespace: 'layers',
 *   useDocsFrom: 'Conv2DTranspose',
 *   configParamIndices: [0]
 * }
 */
export function conv2dTranspose(config: ConvLayerConfig): Layer {
  return new Conv2DTranspose(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Convolutional',
 *   namespace: 'layers',
 *   useDocsFrom: 'SeparableConv2D',
 *   configParamIndices: [0]
 * }
 */
export function separableConv2d(config: SeparableConvLayerConfig): Layer {
  return new SeparableConv2D(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Convolutional',
 *   namespace: 'layers',
 *   useDocsFrom: 'Cropping2D',
 *   configParamIndices: [0]
 * }
 */
export function cropping2D(config: Cropping2DLayerConfig): Layer {
  return new Cropping2D(config);
}

/**
 * @doc{
 *   heading: 'Layers',
 *   subheading: 'Convolutional',
 *   namespace: 'layers',
 *   useDocsFrom: 'UpSampling2D',
 *   configParamIndices: [0]
 * }
 */
export function upSampling2d(config: UpSampling2DLayerConfig): Layer {
  return new UpSampling2D(config);
}

// Convolutional(depthwise) Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Convolutional',
 *   namespace: 'layers',
 *   useDocsFrom: 'DepthwiseConv2D',
 *   configParamIndices: [0]
 * }
 */

export function depthwiseConv2d(config: DepthwiseConv2DLayerConfig): Layer {
  return new DepthwiseConv2D(config);
}

// Basic Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Basic',
 *   namespace: 'layers',
 *   useDocsFrom: 'Activation',
 *   configParamIndices: [0]
 * }
 */
export function activation(config: ActivationLayerConfig): Layer {
  return new Activation(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Basic',
 *   namespace: 'layers',
 *   useDocsFrom: 'Dense',
 *   configParamIndices: [0]
 * }
 */
export function dense(config: DenseLayerConfig): Layer {
  return new Dense(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Basic',
 *   namespace: 'layers',
 *   useDocsFrom: 'Dropout',
 *   configParamIndices: [0]
 * }
 */
export function dropout(config: DropoutLayerConfig): Layer {
  return new Dropout(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Basic',
 *   namespace: 'layers',
 *   useDocsFrom: 'Flatten',
 *   configParamIndices: [0]
 * }
 */
export function flatten(config?: LayerConfig): Layer {
  return new Flatten(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Basic',
 *   namespace: 'layers',
 *   useDocsFrom: 'RepeatVector',
 *   configParamIndices: [0]
 * }
 */
export function repeatVector(config: RepeatVectorLayerConfig): Layer {
  return new RepeatVector(config);
}

/**
 * @doc{
 *   heading: 'Layers',
 *   subheading: 'Basic',
 *   namespace: 'layers',
 *   useDocsFrom: 'Reshape',
 *   configParamIndices: [0]
 * }
 */
export function reshape(config: ReshapeLayerConfig): Layer {
  return new Reshape(config);
}

/**
 * @doc{
 *   heading: 'Layers',
 *   subheading: 'Basic',
 *   namespace: 'layers',
 *   useDocsFrom: 'Permute',
 *   configParamIndices: [0]
 * }
 */
export function permute(config: PermuteLayerConfig): Layer {
  return new Permute(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Basic',
 *   namespace: 'layers',
 *    useDocsFrom: 'Embedding',
 *   configParamIndices: [0]
 * }
 */
export function embedding(config: EmbeddingLayerConfig): Layer {
  return new Embedding(config);
}

// Merge Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Merge',
 *   namespace: 'layers',
 *   useDocsFrom: 'Add',
 *   configParamIndices: [0]
 * }
 */
export function add(config?: LayerConfig): Layer {
  return new Add(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Merge',
 *   namespace: 'layers',
 *   useDocsFrom: 'Average',
 *   configParamIndices: [0]
 * }
 */
export function average(config?: LayerConfig): Layer {
  return new Average(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Merge',
 *   namespace: 'layers',
 *   useDocsFrom: 'Concatenate',
 *   configParamIndices: [0]
 * }
 */
export function concatenate(config?: ConcatenateLayerConfig): Layer {
  return new Concatenate(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Merge',
 *   namespace: 'layers',
 *   useDocsFrom: 'Maximum',
 *   configParamIndices: [0]
 * }
 */
export function maximum(config?: LayerConfig): Layer {
  return new Maximum(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Merge',
 *   namespace: 'layers',
 *   useDocsFrom: 'Minimum',
 *   configParamIndices: [0]
 * }
 */
export function minimum(config?: LayerConfig): Layer {
  return new Minimum(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Merge',
 *   namespace: 'layers',
 *   useDocsFrom: 'Multiply',
 *   configParamIndices: [0]
 * }
 */
export function multiply(config?: LayerConfig): Layer {
  return new Multiply(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Merge',
 *   namespace: 'layers',
 *   useDocsFrom: 'Dot',
 *   configParamIndices: [0]
 * }
 */
export function dot(config: DotLayerConfig): Layer {
  return new Dot(config);
}

// Normalization Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Normalization',
 *   namespace: 'layers',
 *   useDocsFrom: 'BatchNormalization',
 *   configParamIndices: [0]
 * }
 */
export function batchNormalization(config?: BatchNormalizationLayerConfig):
    Layer {
  return new BatchNormalization(config);
}

// Padding Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Padding',
 *   namespace: 'layers',
 *   useDocsFrom: 'ZeroPadding2D',
 *   configParamIndices: [0]
 * }
 */
export function zeroPadding2d(config?: ZeroPadding2DLayerConfig): Layer {
  return new ZeroPadding2D(config);
}

// Pooling Layers.
/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Pooling',
 *   namespace: 'layers',
 *   useDocsFrom: 'AveragePooling1D',
 *   configParamIndices: [0]
 * }
 */
export function averagePooling1d(config: Pooling1DLayerConfig): Layer {
  return new AveragePooling1D(config);
}
export function avgPool1d(config: Pooling1DLayerConfig): Layer {
  return averagePooling1d(config);
}
// For backwards compatibility.
// See https://github.com/tensorflow/tfjs/issues/152
export function avgPooling1d(config: Pooling1DLayerConfig): Layer {
  return averagePooling1d(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Pooling',
 *   namespace: 'layers',
 *   useDocsFrom: 'AveragePooling2D',
 *   configParamIndices: [0]
 * }
 */
export function averagePooling2d(config: Pooling2DLayerConfig): Layer {
  return new AveragePooling2D(config);
}
export function avgPool2d(config: Pooling2DLayerConfig): Layer {
  return averagePooling2d(config);
}
// For backwards compatibility.
// See https://github.com/tensorflow/tfjs/issues/152
export function avgPooling2d(config: Pooling2DLayerConfig): Layer {
  return averagePooling2d(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Pooling',
 *   namespace: 'layers',
 *   useDocsFrom: 'GlobalAveragePooling1D',
 *   configParamIndices: [0]
 * }
 */
export function globalAveragePooling1d(config: LayerConfig): Layer {
  return new GlobalAveragePooling1D(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Pooling',
 *   namespace: 'layers',
 *   useDocsFrom: 'GlobalAveragePooling2D',
 *   configParamIndices: [0]
 * }
 */
export function globalAveragePooling2d(config: GlobalPooling2DLayerConfig):
    Layer {
  return new GlobalAveragePooling2D(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Pooling',
 *   namespace: 'layers',
 *   useDocsFrom: 'GlobalMaxPooling1D',
 *   configParamIndices: [0]
 * }
 */
export function globalMaxPooling1d(config: LayerConfig): Layer {
  return new GlobalMaxPooling1D(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Pooling',
 *   namespace: 'layers',
 *   useDocsFrom: 'GlobalMaxPooling2D',
 *   configParamIndices: [0]
 * }
 */
export function globalMaxPooling2d(config: GlobalPooling2DLayerConfig): Layer {
  return new GlobalMaxPooling2D(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Pooling',
 *   namespace: 'layers',
 *   useDocsFrom: 'MaxPooling1D',
 *   configParamIndices: [0]
 * }
 */
export function maxPooling1d(config: Pooling1DLayerConfig): Layer {
  return new MaxPooling1D(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Pooling',
 *   namespace: 'layers',
 *   useDocsFrom: 'MaxPooling2D',
 *   configParamIndices: [0]
 * }
 */
export function maxPooling2d(config: Pooling2DLayerConfig): Layer {
  return new MaxPooling2D(config);
}

// Recurrent Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Recurrent',
 *   namespace: 'layers',
 *   useDocsFrom: 'GRU',
 *   configParamIndices: [0]
 * }
 */
export function gru(config: GRULayerConfig): Layer {
  return new GRU(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Recurrent',
 *   namespace: 'layers',
 *   useDocsFrom: 'GRUCell',
 *   configParamIndices: [0]
 * }
 */
export function gruCell(config: GRUCellLayerConfig): RNNCell {
  return new GRUCell(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Recurrent',
 *   namespace: 'layers',
 *   useDocsFrom: 'LSTM',
 *   configParamIndices: [0]
 * }
 */
export function lstm(config: LSTMLayerConfig): Layer {
  return new LSTM(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Recurrent',
 *   namespace: 'layers',
 *   useDocsFrom: 'LSTMCell',
 *   configParamIndices: [0]
 * }
 */
export function lstmCell(config: LSTMCellLayerConfig): RNNCell {
  return new LSTMCell(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Recurrent',
 *   namespace: 'layers',
 *   useDocsFrom: 'SimpleRNN',
 *   configParamIndices: [0]
 * }
 */
export function simpleRNN(config: SimpleRNNLayerConfig): Layer {
  return new SimpleRNN(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Recurrent',
 *   namespace: 'layers',
 *   useDocsFrom: 'SimpleRNNCell',
 *   configParamIndices: [0]
 * }
 */
export function simpleRNNCell(config: SimpleRNNCellLayerConfig): RNNCell {
  return new SimpleRNNCell(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Recurrent',
 *   namespace: 'layers',
 *   useDocsFrom: 'RNN',
 *   configParamIndices: [0]
 * }
 */
export function rnn(config: RNNLayerConfig): Layer {
  return new RNN(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Recurrent',
 *   namespace: 'layers',
 *   useDocsFrom: 'RNN',
 *   configParamIndices: [0]
 * }
 */
export function stackedRNNCells(config: StackedRNNCellsConfig): RNNCell {
  return new StackedRNNCells(config);
}

// Wrapper Layers.

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Wrapper',
 *   namespace: 'layers',
 *   useDocsFrom: 'Bidirectional',
 *   configParamIndices: [0]
 * }
 */
export function bidirectional(config: BidirectionalLayerConfig): Wrapper {
  return new Bidirectional(config);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Wrapper',
 *   namespace: 'layers',
 *   useDocsFrom: 'TimeDistributed',
 *   configParamIndices: [0]
 * }
 */
export function timeDistributed(config: WrapperLayerConfig): Layer {
  return new TimeDistributed(config);
}

// Aliases for pooling.
export const globalMaxPool1d = globalMaxPooling1d;
export const globalMaxPool2d = globalMaxPooling2d;
export const maxPool1d = maxPooling1d;
export const maxPool2d = maxPooling2d;

export {Layer, RNN, RNNCell, input /* alias for tf.input */};
