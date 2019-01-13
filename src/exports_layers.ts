/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {InputLayer, InputLayerArgs} from './engine/input_layer';
import {Layer, LayerArgs} from './engine/topology';
import {input} from './exports';
import {ELU, ELULayerArgs, LeakyReLU, LeakyReLULayerArgs, PReLU, PReLULayerArgs, ReLU, ReLULayerArgs, Softmax, SoftmaxLayerArgs, ThresholdedReLU, ThresholdedReLULayerArgs} from './layers/advanced_activations';
import {Conv1D, Conv2D, Conv2DTranspose, ConvLayerArgs, Cropping2D, Cropping2DLayerArgs, SeparableConv2D, SeparableConvLayerArgs, UpSampling2D, UpSampling2DLayerArgs} from './layers/convolutional';
import {DepthwiseConv2D, DepthwiseConv2DLayerArgs} from './layers/convolutional_depthwise';
import {Activation, ActivationLayerArgs, Dense, DenseLayerArgs, Dropout, DropoutLayerArgs, Flatten, Permute, PermuteLayerArgs, RepeatVector, RepeatVectorLayerArgs, Reshape, ReshapeLayerArgs} from './layers/core';
import {Embedding, EmbeddingLayerArgs} from './layers/embeddings';
import {Add, Average, Concatenate, ConcatenateLayerArgs, Dot, DotLayerArgs, Maximum, Minimum, Multiply} from './layers/merge';
import {BatchNormalization, BatchNormalizationLayerArgs} from './layers/normalization';
import {ZeroPadding2D, ZeroPadding2DLayerArgs} from './layers/padding';
import {AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalPooling2DLayerArgs, MaxPooling1D, MaxPooling2D, Pooling1DLayerArgs, Pooling2DLayerArgs} from './layers/pooling';
import {GRU, GRUCell, GRUCellLayerArgs, GRULayerArgs, LSTM, LSTMCell, LSTMCellLayerArgs, LSTMLayerArgs, RNN, RNNCell, RNNLayerArgs, SimpleRNN, SimpleRNNCell, SimpleRNNCellLayerArgs, SimpleRNNLayerArgs, StackedRNNCells, StackedRNNCellsArgs} from './layers/recurrent';
import {Bidirectional, BidirectionalLayerArgs, TimeDistributed, Wrapper, WrapperLayerArgs} from './layers/wrappers';


// TODO(cais): Add doc string to all the public static functions in this
//   class; include exectuable JavaScript code snippets where applicable
//   (b/74074458).

// Input Layer.
/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Inputs',
 *   namespace: 'layers',
 *   useDocsFrom: 'InputLayer',
 *   configParamIndices: [0]
 * }
 */
export function inputLayer(args: InputLayerArgs): Layer {
  return new InputLayer(args);
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
export function elu(args?: ELULayerArgs): Layer {
  return new ELU(args);
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
export function reLU(args?: ReLULayerArgs): Layer {
  return new ReLU(args);
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
export function leakyReLU(args?: LeakyReLULayerArgs): Layer {
  return new LeakyReLU(args);
}

/**
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers',
 *   useDocsFrom: 'PReLU',
 *   configParamIndices: [0]
 * }
 */
export function prelu(args?: PReLULayerArgs): Layer {
  return new PReLU(args);
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
export function softmax(args?: SoftmaxLayerArgs): Layer {
  return new Softmax(args);
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
export function thresholdedReLU(args?: ThresholdedReLULayerArgs): Layer {
  return new ThresholdedReLU(args);
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
export function conv1d(args: ConvLayerArgs): Layer {
  return new Conv1D(args);
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
export function conv2d(args: ConvLayerArgs): Layer {
  return new Conv2D(args);
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
export function conv2dTranspose(args: ConvLayerArgs): Layer {
  return new Conv2DTranspose(args);
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
export function separableConv2d(args: SeparableConvLayerArgs): Layer {
  return new SeparableConv2D(args);
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
export function cropping2D(args: Cropping2DLayerArgs): Layer {
  return new Cropping2D(args);
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
export function upSampling2d(args: UpSampling2DLayerArgs): Layer {
  return new UpSampling2D(args);
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

export function depthwiseConv2d(args: DepthwiseConv2DLayerArgs): Layer {
  return new DepthwiseConv2D(args);
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
export function activation(args: ActivationLayerArgs): Layer {
  return new Activation(args);
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
export function dense(args: DenseLayerArgs): Layer {
  return new Dense(args);
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
export function dropout(args: DropoutLayerArgs): Layer {
  return new Dropout(args);
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
export function flatten(args?: LayerArgs): Layer {
  return new Flatten(args);
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
export function repeatVector(args: RepeatVectorLayerArgs): Layer {
  return new RepeatVector(args);
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
export function reshape(args: ReshapeLayerArgs): Layer {
  return new Reshape(args);
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
export function permute(args: PermuteLayerArgs): Layer {
  return new Permute(args);
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
export function embedding(args: EmbeddingLayerArgs): Layer {
  return new Embedding(args);
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
export function add(args?: LayerArgs): Layer {
  return new Add(args);
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
export function average(args?: LayerArgs): Layer {
  return new Average(args);
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
export function concatenate(args?: ConcatenateLayerArgs): Layer {
  return new Concatenate(args);
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
export function maximum(args?: LayerArgs): Layer {
  return new Maximum(args);
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
export function minimum(args?: LayerArgs): Layer {
  return new Minimum(args);
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
export function multiply(args?: LayerArgs): Layer {
  return new Multiply(args);
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
export function dot(args: DotLayerArgs): Layer {
  return new Dot(args);
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
export function batchNormalization(args?: BatchNormalizationLayerArgs): Layer {
  return new BatchNormalization(args);
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
export function zeroPadding2d(args?: ZeroPadding2DLayerArgs): Layer {
  return new ZeroPadding2D(args);
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
export function averagePooling1d(args: Pooling1DLayerArgs): Layer {
  return new AveragePooling1D(args);
}
export function avgPool1d(args: Pooling1DLayerArgs): Layer {
  return averagePooling1d(args);
}
// For backwards compatibility.
// See https://github.com/tensorflow/tfjs/issues/152
export function avgPooling1d(args: Pooling1DLayerArgs): Layer {
  return averagePooling1d(args);
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
export function averagePooling2d(args: Pooling2DLayerArgs): Layer {
  return new AveragePooling2D(args);
}
export function avgPool2d(args: Pooling2DLayerArgs): Layer {
  return averagePooling2d(args);
}
// For backwards compatibility.
// See https://github.com/tensorflow/tfjs/issues/152
export function avgPooling2d(args: Pooling2DLayerArgs): Layer {
  return averagePooling2d(args);
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
export function globalAveragePooling1d(args?: LayerArgs): Layer {
  return new GlobalAveragePooling1D(args);
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
export function globalAveragePooling2d(args: GlobalPooling2DLayerArgs): Layer {
  return new GlobalAveragePooling2D(args);
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
export function globalMaxPooling1d(args?: LayerArgs): Layer {
  return new GlobalMaxPooling1D(args);
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
export function globalMaxPooling2d(args: GlobalPooling2DLayerArgs): Layer {
  return new GlobalMaxPooling2D(args);
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
export function maxPooling1d(args: Pooling1DLayerArgs): Layer {
  return new MaxPooling1D(args);
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
export function maxPooling2d(args: Pooling2DLayerArgs): Layer {
  return new MaxPooling2D(args);
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
export function gru(args: GRULayerArgs): Layer {
  return new GRU(args);
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
export function gruCell(args: GRUCellLayerArgs): RNNCell {
  return new GRUCell(args);
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
export function lstm(args: LSTMLayerArgs): Layer {
  return new LSTM(args);
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
export function lstmCell(args: LSTMCellLayerArgs): RNNCell {
  return new LSTMCell(args);
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
export function simpleRNN(args: SimpleRNNLayerArgs): Layer {
  return new SimpleRNN(args);
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
export function simpleRNNCell(args: SimpleRNNCellLayerArgs): RNNCell {
  return new SimpleRNNCell(args);
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
export function rnn(args: RNNLayerArgs): Layer {
  return new RNN(args);
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
export function stackedRNNCells(args: StackedRNNCellsArgs): RNNCell {
  return new StackedRNNCells(args);
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
export function bidirectional(args: BidirectionalLayerArgs): Wrapper {
  return new Bidirectional(args);
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
export function timeDistributed(args: WrapperLayerArgs): Layer {
  return new TimeDistributed(args);
}

// Aliases for pooling.
export const globalMaxPool1d = globalMaxPooling1d;
export const globalMaxPool2d = globalMaxPooling2d;
export const maxPool1d = maxPooling1d;
export const maxPool2d = maxPooling2d;

export {Layer, RNN, RNNCell, input /* alias for tf.input */};
