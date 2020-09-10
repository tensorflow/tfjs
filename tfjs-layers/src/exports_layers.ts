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
import {Conv1D, Conv2D, Conv2DTranspose, Conv3D, ConvLayerArgs, Cropping2D, Cropping2DLayerArgs, SeparableConv2D, SeparableConvLayerArgs, UpSampling2D, UpSampling2DLayerArgs} from './layers/convolutional';
import {DepthwiseConv2D, DepthwiseConv2DLayerArgs} from './layers/convolutional_depthwise';
import {ConvLSTM2D, ConvLSTM2DArgs, ConvLSTM2DCell, ConvLSTM2DCellArgs} from './layers/convolutional_recurrent';
import {Activation, ActivationLayerArgs, Dense, DenseLayerArgs, Dropout, DropoutLayerArgs, Flatten, FlattenLayerArgs, Masking, MaskingArgs, Permute, PermuteLayerArgs, RepeatVector, RepeatVectorLayerArgs, Reshape, ReshapeLayerArgs, SpatialDropout1D, SpatialDropout1DLayerConfig} from './layers/core';
import {Embedding, EmbeddingLayerArgs} from './layers/embeddings';
import {Add, Average, Concatenate, ConcatenateLayerArgs, Dot, DotLayerArgs, Maximum, Minimum, Multiply} from './layers/merge';
import {AlphaDropout, AlphaDropoutArgs, GaussianDropout, GaussianDropoutArgs, GaussianNoise, GaussianNoiseArgs} from './layers/noise';
import {BatchNormalization, BatchNormalizationLayerArgs, LayerNormalization, LayerNormalizationLayerArgs} from './layers/normalization';
import {ZeroPadding2D, ZeroPadding2DLayerArgs} from './layers/padding';
import {AveragePooling1D, AveragePooling2D, AveragePooling3D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalPooling2DLayerArgs, MaxPooling1D, MaxPooling2D, MaxPooling3D, Pooling1DLayerArgs, Pooling2DLayerArgs, Pooling3DLayerArgs} from './layers/pooling';
import {GRU, GRUCell, GRUCellLayerArgs, GRULayerArgs, LSTM, LSTMCell, LSTMCellLayerArgs, LSTMLayerArgs, RNN, RNNCell, RNNLayerArgs, SimpleRNN, SimpleRNNCell, SimpleRNNCellLayerArgs, SimpleRNNLayerArgs, StackedRNNCells, StackedRNNCellsArgs} from './layers/recurrent';
import {Bidirectional, BidirectionalLayerArgs, TimeDistributed, WrapperLayerArgs} from './layers/wrappers';

// TODO(cais): Add doc string to all the public static functions in this
//   class; include exectuable JavaScript code snippets where applicable
//   (b/74074458).

// Input Layer.
/**
 * An input layer is an entry point into a `tf.LayersModel`.
 *
 * `InputLayer` is generated automatically for `tf.Sequential`` models by
 * specifying the `inputshape` or `batchInputShape` for the first layer.  It
 * should not be specified explicitly. However, it can be useful sometimes,
 * e.g., when constructing a sequential model from a subset of another
 * sequential model's layers. Like the code snippet below shows.
 *
 * ```js
 * // Define a model which simply adds two inputs.
 * const model1 = tf.sequential();
 * model1.add(tf.layers.dense({inputShape: [4], units: 3, activation: 'relu'}));
 * model1.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
 * model1.summary();
 * model1.predict(tf.zeros([1, 4])).print();
 *
 * // Construct another model, reusing the second layer of `model1` while
 * // not using the first layer of `model1`. Note that you cannot add the second
 * // layer of `model` directly as the first layer of the new sequential model,
 * // because doing so will lead to an error related to the fact that the layer
 * // is not an input layer. Instead, you need to create an `inputLayer` and add
 * // it to the new sequential model before adding the reused layer.
 * const model2 = tf.sequential();
 * // Use an inputShape that matches the input shape of `model1`'s second
 * // layer.
 * model2.add(tf.layers.inputLayer({inputShape: [3]}));
 * model2.add(model1.layers[1]);
 * model2.summary();
 * model2.predict(tf.zeros([1, 3])).print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Inputs', namespace: 'layers'}
 */
export function inputLayer(args: InputLayerArgs): Layer {
  return new InputLayer(args);
}

// Advanced Activation Layers.

/**
 * Exponetial Linear Unit (ELU).
 *
 * It follows:
 * `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
 * `f(x) = x for x >= 0`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Fast and Accurate Deep Network Learning by Exponential Linear Units
 * (ELUs)](https://arxiv.org/abs/1511.07289v1)
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function elu(args?: ELULayerArgs): Layer {
  return new ELU(args);
}

/**
 * Rectified Linear Unit activation function.
 *
 * Input shape:
 *   Arbitrary. Use the config field `inputShape` (Array of integers, does
 *   not include the sample axis) when using this layer as the first layer
 *   in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function reLU(args?: ReLULayerArgs): Layer {
  return new ReLU(args);
}

/**
 * Leaky version of a rectified linear unit.
 *
 * It allows a small gradient when the unit is not active:
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function leakyReLU(args?: LeakyReLULayerArgs): Layer {
  return new LeakyReLU(args);
}

/**
 * Parameterized version of a leaky rectified linear unit.
 *
 * It follows
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 * wherein `alpha` is a trainable weight.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function prelu(args?: PReLULayerArgs): Layer {
  return new PReLU(args);
}

/**
 * Softmax activation layer.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function softmax(args?: SoftmaxLayerArgs): Layer {
  return new Softmax(args);
}

/**
 * Thresholded Rectified Linear Unit.
 *
 * It follows:
 * `f(x) = x for x > theta`,
 * `f(x) = 0 otherwise`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Zero-Bias Autoencoders and the Benefits of Co-Adapting
 * Features](http://arxiv.org/abs/1402.3337)
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
export function thresholdedReLU(args?: ThresholdedReLULayerArgs): Layer {
  return new ThresholdedReLU(args);
}

// Convolutional Layers.

/**
 * 1D convolution layer (e.g., temporal convolution).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input over a single spatial (or temporal) dimension
 * to produce a tensor of outputs.
 *
 * If `use_bias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model, provide an
 * `inputShape` argument `Array` or `null`.
 *
 * For example, `inputShape` would be:
 * - `[10, 128]` for sequences of 10 vectors of 128-dimensional vectors
 * - `[null, 128]` for variable-length sequences of 128-dimensional vectors.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional',  namespace: 'layers'}
 */
export function conv1d(args: ConvLayerArgs): Layer {
  return new Conv1D(args);
}

/**
 * 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input to produce a tensor of outputs.
 *
 * If `useBias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `inputShape`
 * (Array of integers, does not include the sample axis),
 * e.g. `inputShape=[128, 128, 3]` for 128x128 RGB pictures
 * in `dataFormat='channelsLast'`.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function conv2d(args: ConvLayerArgs): Layer {
  return new Conv2D(args);
}

/**
 * Transposed convolutional layer (sometimes called Deconvolution).
 *
 * The need for transposed convolutions generally arises
 * from the desire to use a transformation going in the opposite direction of
 * a normal convolution, i.e., from something that has the shape of the output
 * of some convolution to something that has the shape of its input while
 * maintaining a connectivity pattern that is compatible with said
 * convolution.
 *
 * When using this layer as the first layer in a model, provide the
 * configuration `inputShape` (`Array` of integers, does not include the
 * sample axis), e.g., `inputShape: [128, 128, 3]` for 128x128 RGB pictures in
 * `dataFormat: 'channelsLast'`.
 *
 * Input shape:
 *   4D tensor with shape:
 *   `[batch, channels, rows, cols]` if `dataFormat` is `'channelsFirst'`.
 *   or 4D tensor with shape
 *   `[batch, rows, cols, channels]` if `dataFormat` is `'channelsLast`.
 *
 * Output shape:
 *   4D tensor with shape:
 *   `[batch, filters, newRows, newCols]` if `dataFormat` is
 * `'channelsFirst'`. or 4D tensor with shape:
 *   `[batch, newRows, newCols, filters]` if `dataFormat` is `'channelsLast'`.
 *
 * References:
 *   - [A guide to convolution arithmetic for deep
 * learning](https://arxiv.org/abs/1603.07285v1)
 *   - [Deconvolutional
 * Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function conv2dTranspose(args: ConvLayerArgs): Layer {
  return new Conv2DTranspose(args);
}

/**
 * 3D convolution layer (e.g. spatial convolution over volumes).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input to produce a tensor of outputs.
 *
 * If `useBias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `inputShape`
 * (Array of integers, does not include the sample axis),
 * e.g. `inputShape=[128, 128, 128, 1]` for 128x128x128 grayscale volumes
 * in `dataFormat='channelsLast'`.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function conv3d(args: ConvLayerArgs): Layer {
  return new Conv3D(args);
}

/**
 * Depthwise separable 2D convolution.
 *
 * Separable convolution consists of first performing
 * a depthwise spatial convolution
 * (which acts on each input channel separately)
 * followed by a pointwise convolution which mixes together the resulting
 * output channels. The `depthMultiplier` argument controls how many
 * output channels are generated per input channel in the depthwise step.
 *
 * Intuitively, separable convolutions can be understood as
 * a way to factorize a convolution kernel into two smaller kernels,
 * or as an extreme version of an Inception block.
 *
 * Input shape:
 *   4D tensor with shape:
 *     `[batch, channels, rows, cols]` if data_format='channelsFirst'
 *   or 4D tensor with shape:
 *     `[batch, rows, cols, channels]` if data_format='channelsLast'.
 *
 * Output shape:
 *   4D tensor with shape:
 *     `[batch, filters, newRows, newCols]` if data_format='channelsFirst'
 *   or 4D tensor with shape:
 *     `[batch, newRows, newCols, filters]` if data_format='channelsLast'.
 *     `rows` and `cols` values might have changed due to padding.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function separableConv2d(args: SeparableConvLayerArgs): Layer {
  return new SeparableConv2D(args);
}

/**
 * Cropping layer for 2D input (e.g., image).
 *
 * This layer can crop an input
 * at the top, bottom, left and right side of an image tensor.
 *
 * Input shape:
 *   4D tensor with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, rows, cols, channels]`
 *   - If `data_format` is `"channels_first"`:
 *     `[batch, channels, rows, cols]`.
 *
 * Output shape:
 *   4D with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, croppedRows, croppedCols, channels]`
 *    - If `dataFormat` is `"channelsFirst"`:
 *     `[batch, channels, croppedRows, croppedCols]`.
 *
 * Examples
 * ```js
 *
 * const model = tf.sequential();
 * model.add(tf.layers.cropping2D({cropping:[[2, 2], [2, 2]],
 *                                inputShape: [128, 128, 3]}));
 * //now output shape is [batch, 124, 124, 3]
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function cropping2D(args: Cropping2DLayerArgs): Layer {
  return new Cropping2D(args);
}

/**
 * Upsampling layer for 2D inputs.
 *
 * Repeats the rows and columns of the data
 * by size[0] and size[1] respectively.
 *
 *
 * Input shape:
 *    4D tensor with shape:
 *     - If `dataFormat` is `"channelsLast"`:
 *         `[batch, rows, cols, channels]`
 *     - If `dataFormat` is `"channelsFirst"`:
 *        `[batch, channels, rows, cols]`
 *
 * Output shape:
 *     4D tensor with shape:
 *     - If `dataFormat` is `"channelsLast"`:
 *        `[batch, upsampledRows, upsampledCols, channels]`
 *     - If `dataFormat` is `"channelsFirst"`:
 *         `[batch, channels, upsampledRows, upsampledCols]`
 *
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function upSampling2d(args: UpSampling2DLayerArgs): Layer {
  return new UpSampling2D(args);
}

// Convolutional(depthwise) Layers.

/**
 * Depthwise separable 2D convolution.
 *
 * Depthwise Separable convolutions consists in performing just the first step
 * in a depthwise spatial convolution (which acts on each input channel
 * separately). The `depthMultplier` argument controls how many output channels
 * are generated per input channel in the depthwise step.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
export function depthwiseConv2d(args: DepthwiseConv2DLayerArgs): Layer {
  return new DepthwiseConv2D(args);
}

// Basic Layers.

/**
 * Applies an activation function to an output.
 *
 * This layer applies element-wise activation function.  Other layers, notably
 * `dense` can also apply activation functions.  Use this isolated activation
 * function to extract the values before and after the
 * activation. For instance:
 *
 * ```js
 * const input = tf.input({shape: [5]});
 * const denseLayer = tf.layers.dense({units: 1});
 * const activationLayer = tf.layers.activation({activation: 'relu6'});
 *
 * // Obtain the output symbolic tensors by applying the layers in order.
 * const denseOutput = denseLayer.apply(input);
 * const activationOutput = activationLayer.apply(denseOutput);
 *
 * // Create the model based on the inputs.
 * const model = tf.model({
 *     inputs: input,
 *     outputs: [denseOutput, activationOutput]
 * });
 *
 * // Collect both outputs and print separately.
 * const [denseOut, activationOut] = model.predict(tf.randomNormal([6, 5]));
 * denseOut.print();
 * activationOut.print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function activation(args: ActivationLayerArgs): Layer {
  return new Activation(args);
}

/**
 * Creates a dense (fully connected) layer.
 *
 * This layer implements the operation:
 *   `output = activation(dot(input, kernel) + bias)`
 *
 * `activation` is the element-wise activation function
 *   passed as the `activation` argument.
 *
 * `kernel` is a weights matrix created by the layer.
 *
 * `bias` is a bias vector created by the layer (only applicable if `useBias`
 * is `true`).
 *
 * **Input shape:**
 *
 *   nD `tf.Tensor` with shape: `(batchSize, ..., inputDim)`.
 *
 *   The most common situation would be
 *   a 2D input with shape `(batchSize, inputDim)`.
 *
 * **Output shape:**
 *
 *   nD tensor with shape: `(batchSize, ..., units)`.
 *
 *   For instance, for a 2D input with shape `(batchSize, inputDim)`,
 *   the output would have shape `(batchSize, units)`.
 *
 * Note: if the input to the layer has a rank greater than 2, then it is
 * flattened prior to the initial dot product with the kernel.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function dense(args: DenseLayerArgs): Layer {
  return new Dense(args);
}

/**
 * Applies
 * [dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) to
 * the input.
 *
 * Dropout consists in randomly setting a fraction `rate` of input units to 0 at
 * each update during training time, which helps prevent overfitting.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function dropout(args: DropoutLayerArgs): Layer {
  return new Dropout(args);
}

/**
 * Spatial 1D version of Dropout.
 *
 * This Layer type performs the same function as the Dropout layer, but it drops
 * entire 1D feature maps instead of individual elements. For example, if an
 * input example consists of 3 timesteps and the feature map for each timestep
 * has a size of 4, a `spatialDropout1d` layer may zero out the feature maps
 * of the 1st timesteps and 2nd timesteps completely while sparing all feature
 * elements of the 3rd timestep.
 *
 * If adjacent frames (timesteps) are strongly correlated (as is normally the
 * case in early convolution layers), regular dropout will not regularize the
 * activation and will otherwise just result in merely an effective learning
 * rate decrease. In this case, `spatialDropout1d` will help promote
 * independence among feature maps and should be used instead.
 *
 * **Arguments:**
 *   rate: A floating-point number >=0 and <=1. Fraction of the input elements
 *     to drop.
 *
 * **Input shape:**
 *   3D tensor with shape `(samples, timesteps, channels)`.
 *
 * **Output shape:**
 *   Same as the input shape.
 *
 * References:
 *   - [Efficient Object Localization Using Convolutional
 *      Networks](https://arxiv.org/abs/1411.4280)
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function spatialDropout1d(args: SpatialDropout1DLayerConfig): Layer {
  return new SpatialDropout1D(args);
}

/**
 * Flattens the input. Does not affect the batch size.
 *
 * A `Flatten` layer flattens each batch in its inputs to 1D (making the output
 * 2D).
 *
 * For example:
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const flattenLayer = tf.layers.flatten();
 * // Inspect the inferred output shape of the flatten layer, which
 * // equals `[null, 12]`. The 2nd dimension is 4 * 3, i.e., the result of the
 * // flattening. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(flattenLayer.apply(input).shape));
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function flatten(args?: FlattenLayerArgs): Layer {
  return new Flatten(args);
}

/**
 * Repeats the input n times in a new dimension.
 *
 * ```js
 *  const model = tf.sequential();
 *  model.add(tf.layers.repeatVector({n: 4, inputShape: [2]}));
 *  const x = tf.tensor2d([[10, 20]]);
 *  // Use the model to do inference on a data point the model hasn't see
 *  model.predict(x).print();
 *  // output shape is now [batch, 2, 4]
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function repeatVector(args: RepeatVectorLayerArgs): Layer {
  return new RepeatVector(args);
}

/**
 * Reshapes an input to a certain shape.
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const reshapeLayer = tf.layers.reshape({targetShape: [2, 6]});
 * // Inspect the inferred output shape of the Reshape layer, which
 * // equals `[null, 2, 6]`. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(reshapeLayer.apply(input).shape));
 * ```
 *
 * Input shape:
 *   Arbitrary, although all dimensions in the input shape must be fixed.
 *   Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 *
 * Output shape:
 *   [batchSize, targetShape[0], targetShape[1], ...,
 *    targetShape[targetShape.length - 1]].
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function reshape(args: ReshapeLayerArgs): Layer {
  return new Reshape(args);
}

/**
 * Permutes the dimensions of the input according to a given pattern.
 *
 * Useful for, e.g., connecting RNNs and convnets together.
 *
 * Example:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.permute({
 *   dims: [2, 1],
 *   inputShape: [10, 64]
 * }));
 * console.log(model.outputShape);
 * // Now model's output shape is [null, 64, 10], where null is the
 * // unpermuted sample (batch) dimension.
 * ```
 *
 * Input shape:
 *   Arbitrary. Use the configuration field `inputShape` when using this
 *   layer as the first layer in a model.
 *
 * Output shape:
 *   Same rank as the input shape, but with the dimensions re-ordered (i.e.,
 *   permuted) according to the `dims` configuration of this layer.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function permute(args: PermuteLayerArgs): Layer {
  return new Permute(args);
}

/**
 * Maps positive integers (indices) into dense vectors of fixed size.
 * eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
 *
 * **Input shape:** 2D tensor with shape: `[batchSize, sequenceLength]`.
 *
 * **Output shape:** 3D tensor with shape: `[batchSize, sequenceLength,
 * outputDim]`.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
export function embedding(args: EmbeddingLayerArgs): Layer {
  return new Embedding(args);
}

// Merge Layers.

/**
 * Layer that performs element-wise addition on an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape). The inputs are specified as an
 * `Array` when the `apply` method of the `Add` layer instance is called. For
 * example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const addLayer = tf.layers.add();
 * const sum = addLayer.apply([input1, input2]);
 * console.log(JSON.stringify(sum.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function add(args?: LayerArgs): Layer {
  return new Add(args);
}

/**
 * Layer that performs element-wise averaging on an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const averageLayer = tf.layers.average();
 * const average = averageLayer.apply([input1, input2]);
 * console.log(JSON.stringify(average.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function average(args?: LayerArgs): Layer {
  return new Average(args);
}

/**
 * Layer that concatenates an `Array` of inputs.
 *
 * It takes a list of tensors, all of the same shape except for the
 * concatenation axis, and returns a single tensor, the concatenation
 * of all inputs. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 3]});
 * const concatLayer = tf.layers.concatenate();
 * const output = concatLayer.apply([input1, input2]);
 * console.log(JSON.stringify(output.shape));
 * // You get [null, 2, 5], with the first dimension as the undetermined batch
 * // dimension. The last dimension (5) is the result of concatenating the
 * // last dimensions of the inputs (2 and 3).
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function concatenate(args?: ConcatenateLayerArgs): Layer {
  return new Concatenate(args);
}

/**
 * Layer that computes the element-wise maximum an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const maxLayer = tf.layers.maximum();
 * const max = maxLayer.apply([input1, input2]);
 * console.log(JSON.stringify(max.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function maximum(args?: LayerArgs): Layer {
  return new Maximum(args);
}

/**
 * Layer that computes the element-wise minimum of an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const minLayer = tf.layers.minimum();
 * const min = minLayer.apply([input1, input2]);
 * console.log(JSON.stringify(min.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function minimum(args?: LayerArgs): Layer {
  return new Minimum(args);
}

/**
 * Layer that multiplies (element-wise) an `Array` of inputs.
 *
 * It takes as input an Array of tensors, all of the same
 * shape, and returns a single tensor (also of the same shape).
 * For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const input3 = tf.input({shape: [2, 2]});
 * const multiplyLayer = tf.layers.multiply();
 * const product = multiplyLayer.apply([input1, input2, input3]);
 * console.log(product.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function multiply(args?: LayerArgs): Layer {
  return new Multiply(args);
}

/**
 * Layer that computes a dot product between samples in two tensors.
 *
 * E.g., if applied to a list of two tensors `a` and `b` both of shape
 * `[batchSize, n]`, the output will be a tensor of shape `[batchSize, 1]`,
 * where each entry at index `[i, 0]` will be the dot product between
 * `a[i, :]` and `b[i, :]`.
 *
 * Example:
 *
 * ```js
 * const dotLayer = tf.layers.dot({axes: -1});
 * const x1 = tf.tensor2d([[10, 20], [30, 40]]);
 * const x2 = tf.tensor2d([[-1, -2], [-3, -4]]);
 *
 * // Invoke the layer's apply() method in eager (imperative) mode.
 * const y = dotLayer.apply([x1, x2]);
 * y.print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
export function dot(args: DotLayerArgs): Layer {
  return new Dot(args);
}

// Normalization Layers.

/**
 * Batch normalization layer (Ioffe and Szegedy, 2014).
 *
 * Normalize the activations of the previous layer at each batch,
 * i.e. applies a transformation that maintains the mean activation
 * close to 0 and the activation standard deviation close to 1.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape` (Array of integers, does
 *   not include the sample axis) when calling the constructor of this class,
 *   if this layer is used as a first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Batch Normalization: Accelerating Deep Network Training by Reducing
 * Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
 *
 * @doc {heading: 'Layers', subheading: 'Normalization', namespace: 'layers'}
 */
export function batchNormalization(args?: BatchNormalizationLayerArgs): Layer {
  return new BatchNormalization(args);
}

/**
 * Layer-normalization layer (Ba et al., 2016).
 *
 * Normalizes the activations of the previous layer for each given example in a
 * batch independently, instead of across a batch like in `batchNormalization`.
 * In other words, this layer applies a transformation that maintanis the mean
 * activation within each example close to0 and activation variance close to 1.
 *
 * Input shape:
 *   Arbitrary. Use the argument `inputShape` when using this layer as the first
 *   layer in a model.
 *
 * Output shape:
 *   Same as input.
 *
 * References:
 *   - [Layer Normalization](https://arxiv.org/abs/1607.06450)
 *
 * @doc {heading: 'Layers', subheading: 'Normalization', namespace: 'layers'}
 */
export function layerNormalization(args?: LayerNormalizationLayerArgs): Layer {
  return new LayerNormalization(args);
}

// Padding Layers.

/**
 * Zero-padding layer for 2D input (e.g., image).
 *
 * This layer can add rows and columns of zeros
 * at the top, bottom, left and right side of an image tensor.
 *
 * Input shape:
 *   4D tensor with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, rows, cols, channels]`
 *   - If `data_format` is `"channels_first"`:
 *     `[batch, channels, rows, cols]`.
 *
 * Output shape:
 *   4D with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, paddedRows, paddedCols, channels]`
 *    - If `dataFormat` is `"channelsFirst"`:
 *     `[batch, channels, paddedRows, paddedCols]`.
 *
 * @doc {heading: 'Layers', subheading: 'Padding', namespace: 'layers'}
 */
export function zeroPadding2d(args?: ZeroPadding2DLayerArgs): Layer {
  return new ZeroPadding2D(args);
}

// Pooling Layers.

/**
 * Average pooling operation for spatial data.
 *
 * Input shape: `[batchSize, inLength, channels]`
 *
 * Output shape: `[batchSize, pooledLength, channels]`
 *
 * `tf.avgPool1d` is an alias.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
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
 * Average pooling operation for spatial data.
 *
 * Input shape:
 *  - If `dataFormat === CHANNEL_LAST`:
 *      4D tensor with shape:
 *      `[batchSize, rows, cols, channels]`
 *  - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *      `[batchSize, channels, rows, cols]`
 *
 * Output shape
 *  - If `dataFormat === CHANNEL_LAST`:
 *      4D tensor with shape:
 *      `[batchSize, pooleRows, pooledCols, channels]`
 *  - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *      `[batchSize, channels, pooleRows, pooledCols]`
 *
 * `tf.avgPool2d` is an alias.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
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
 * Average pooling operation for 3D data.
 *
 * Input shape
 *   - If `dataFormat === channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, depths, rows, cols, channels]`
 *   - If `dataFormat === channelsFirst`:
 *      4D tensor with shape:
 *       `[batchSize, channels, depths, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, pooledDepths, pooledRows, pooledCols, channels]`
 *   - If `dataFormat=channelsFirst`:
 *       5D tensor with shape:
 *       `[batchSize, channels, pooledDepths, pooledRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function averagePooling3d(args: Pooling3DLayerArgs): Layer {
  return new AveragePooling3D(args);
}
export function avgPool3d(args: Pooling3DLayerArgs): Layer {
  return averagePooling3d(args);
}
// For backwards compatibility.
// See https://github.com/tensorflow/tfjs/issues/152
export function avgPooling3d(args: Pooling3DLayerArgs): Layer {
  return averagePooling3d(args);
}

/**
 * Global average pooling operation for temporal data.
 *
 * Input Shape: 3D tensor with shape: `[batchSize, steps, features]`.
 *
 * Output Shape:2D tensor with shape: `[batchSize, features]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function globalAveragePooling1d(args?: LayerArgs): Layer {
  return new GlobalAveragePooling1D(args);
}

/**
 * Global average pooling operation for spatial data.
 *
 * Input shape:
 *   - If `dataFormat` is `CHANNEL_LAST`:
 *       4D tensor with shape: `[batchSize, rows, cols, channels]`.
 *   - If `dataFormat` is `CHANNEL_FIRST`:
 *       4D tensor with shape: `[batchSize, channels, rows, cols]`.
 *
 * Output shape:
 *   2D tensor with shape: `[batchSize, channels]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function globalAveragePooling2d(args: GlobalPooling2DLayerArgs): Layer {
  return new GlobalAveragePooling2D(args);
}

/**
 * Global max pooling operation for temporal data.
 *
 * Input Shape: 3D tensor with shape: `[batchSize, steps, features]`.
 *
 * Output Shape:2D tensor with shape: `[batchSize, features]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function globalMaxPooling1d(args?: LayerArgs): Layer {
  return new GlobalMaxPooling1D(args);
}

/**
 * Global max pooling operation for spatial data.
 *
 * Input shape:
 *   - If `dataFormat` is `CHANNEL_LAST`:
 *       4D tensor with shape: `[batchSize, rows, cols, channels]`.
 *   - If `dataFormat` is `CHANNEL_FIRST`:
 *       4D tensor with shape: `[batchSize, channels, rows, cols]`.
 *
 * Output shape:
 *   2D tensor with shape: `[batchSize, channels]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function globalMaxPooling2d(args: GlobalPooling2DLayerArgs): Layer {
  return new GlobalMaxPooling2D(args);
}

/**
 * Max pooling operation for temporal data.
 *
 * Input shape:  `[batchSize, inLength, channels]`
 *
 * Output shape: `[batchSize, pooledLength, channels]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function maxPooling1d(args: Pooling1DLayerArgs): Layer {
  return new MaxPooling1D(args);
}

/**
 * Max pooling operation for spatial data.
 *
 * Input shape
 *   - If `dataFormat === CHANNEL_LAST`:
 *       4D tensor with shape:
 *       `[batchSize, rows, cols, channels]`
 *   - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *       `[batchSize, channels, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=CHANNEL_LAST`:
 *       4D tensor with shape:
 *       `[batchSize, pooleRows, pooledCols, channels]`
 *   - If `dataFormat=CHANNEL_FIRST`:
 *       4D tensor with shape:
 *       `[batchSize, channels, pooleRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function maxPooling2d(args: Pooling2DLayerArgs): Layer {
  return new MaxPooling2D(args);
}

/**
 * Max pooling operation for 3D data.
 *
 * Input shape
 *   - If `dataFormat === channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, depths, rows, cols, channels]`
 *   - If `dataFormat === channelsFirst`:
 *      5D tensor with shape:
 *       `[batchSize, channels, depths, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, pooledDepths, pooledRows, pooledCols, channels]`
 *   - If `dataFormat=channelsFirst`:
 *       5D tensor with shape:
 *       `[batchSize, channels, pooledDepths, pooledRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
export function maxPooling3d(args: Pooling3DLayerArgs): Layer {
  return new MaxPooling3D(args);
}

// Recurrent Layers.

/**
 * Gated Recurrent Unit - Cho et al. 2014.
 *
 * This is an `RNN` layer consisting of one `GRUCell`. However, unlike
 * the underlying `GRUCell`, the `apply` method of `SimpleRNN` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const rnn = tf.layers.gru({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `GRUCell`'s number of units.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function gru(args: GRULayerArgs): Layer {
  return new GRU(args);
}

/**
 * Cell class for `GRU`.
 *
 * `GRUCell` is distinct from the `RNN` subclass `GRU` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `GRU` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.gruCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `GRUCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.gruCell({units: 4}),
 *   tf.layers.gruCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `gruCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `GRUCell`, use the
 * `tf.layers.gru`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function gruCell(args: GRUCellLayerArgs): RNNCell {
  return new GRUCell(args);
}

/**
 * Long-Short Term Memory layer - Hochreiter 1997.
 *
 * This is an `RNN` layer consisting of one `LSTMCell`. However, unlike
 * the underlying `LSTMCell`, the `apply` method of `LSTM` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const lstm = tf.layers.lstm({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = lstm.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `LSTMCell`'s number of units.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function lstm(args: LSTMLayerArgs): Layer {
  return new LSTM(args);
}

/**
 * Cell class for `LSTM`.
 *
 * `LSTMCell` is distinct from the `RNN` subclass `LSTM` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `LSTM` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.lstmCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `LSTMCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.lstmCell({units: 4}),
 *   tf.layers.lstmCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `lstmCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `LSTMCell`, use the
 * `tf.layers.lstm`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function lstmCell(args: LSTMCellLayerArgs): RNNCell {
  return new LSTMCell(args);
}

/**
 * Fully-connected RNN where the output is to be fed back to input.
 *
 * This is an `RNN` layer consisting of one `SimpleRNNCell`. However, unlike
 * the underlying `SimpleRNNCell`, the `apply` method of `SimpleRNN` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const rnn = tf.layers.simpleRNN({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `SimpleRNNCell`'s number of units.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function simpleRNN(args: SimpleRNNLayerArgs): Layer {
  return new SimpleRNN(args);
}

/**
 * Cell class for `SimpleRNN`.
 *
 * `SimpleRNNCell` is distinct from the `RNN` subclass `SimpleRNN` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `SimpleRNN` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.simpleRNNCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `SimpleRNNCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.simpleRNNCell({units: 4}),
 *   tf.layers.simpleRNNCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `SimpleRNNCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `SimpleRNNCell`, use the
 * `tf.layers.simpleRNN`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function simpleRNNCell(args: SimpleRNNCellLayerArgs): RNNCell {
  return new SimpleRNNCell(args);
}

/**
 * Convolutional LSTM layer - Xingjian Shi 2015.
 *
 * This is an `ConvRNN2D` layer consisting of one `ConvLSTM2DCell`. However,
 * unlike the underlying `ConvLSTM2DCell`, the `apply` method of `ConvLSTM2D`
 * operates on a sequence of inputs. The shape of the input (not including the
 * first, batch dimension) needs to be 4-D, with the first dimension being time
 * steps. For example:
 *
 * ```js
 * const filters = 3;
 * const kernelSize = 3;
 *
 * const batchSize = 4;
 * const sequenceLength = 2;
 * const size = 5;
 * const channels = 3;
 *
 * const inputShape = [batchSize, sequenceLength, size, size, channels];
 * const input = tf.ones(inputShape);
 *
 * const layer = tf.layers.convLstm2d({filters, kernelSize});
 *
 * const output = layer.apply(input);
 * ```
 */
/** @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'} */
export function convLstm2d(args: ConvLSTM2DArgs): ConvLSTM2D {
  return new ConvLSTM2D(args);
}

/**
 * Cell class for `ConvLSTM2D`.
 *
 * `ConvLSTM2DCell` is distinct from the `ConvRNN2D` subclass `ConvLSTM2D` in
 * that its `call` method takes the input data of only a single time step and
 * returns the cell's output at the time step, while `ConvLSTM2D` takes the
 * input data over a number of time steps. For example:
 *
 * ```js
 * const filters = 3;
 * const kernelSize = 3;
 *
 * const sequenceLength = 1;
 * const size = 5;
 * const channels = 3;
 *
 * const inputShape = [sequenceLength, size, size, channels];
 * const input = tf.ones(inputShape);
 *
 * const cell = tf.layers.convLstm2dCell({filters, kernelSize});
 *
 * cell.build(input.shape);
 *
 * const outputSize = size - kernelSize + 1;
 * const outShape = [sequenceLength, outputSize, outputSize, filters];
 *
 * const initialH = tf.zeros(outShape);
 * const initialC = tf.zeros(outShape);
 *
 * const [o, h, c] = cell.call([input, initialH, initialC], {});
 * ```
 */
/** @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'} */
export function convLstm2dCell(args: ConvLSTM2DCellArgs): ConvLSTM2DCell {
  return new ConvLSTM2DCell(args);
}

/**
 * Base class for recurrent layers.
 *
 * Input shape:
 *   3D tensor with shape `[batchSize, timeSteps, inputDim]`.
 *
 * Output shape:
 *   - if `returnState`, an Array of tensors (i.e., `tf.Tensor`s). The first
 *     tensor is the output. The remaining tensors are the states at the
 *     last time step, each with shape `[batchSize, units]`.
 *   - if `returnSequences`, the output will have shape
 *     `[batchSize, timeSteps, units]`.
 *   - else, the output will have shape `[batchSize, units]`.
 *
 * Masking:
 *   This layer supports masking for input data with a variable number
 *   of timesteps. To introduce masks to your data,
 *   use an embedding layer with the `mask_zero` parameter
 *   set to `True`.
 *
 * Notes on using statefulness in RNNs:
 *   You can set RNN layers to be 'stateful', which means that the states
 *   computed for the samples in one batch will be reused as initial states
 *   for the samples in the next batch. This assumes a one-to-one mapping
 *   between samples in different successive batches.
 *
 *   To enable statefulness:
 *     - specify `stateful: true` in the layer constructor.
 *     - specify a fixed batch size for your model, by passing
 *       if sequential model:
 *         `batchInputShape=[...]` to the first layer in your model.
 *       else for functional model with 1 or more Input layers:
 *         `batchShape=[...]` to all the first layers in your model.
 *       This is the expected shape of your inputs *including the batch size*.
 *       It should be a tuple of integers, e.g. `(32, 10, 100)`.
 *     - specify `shuffle=False` when calling fit().
 *
 *   To reset the states of your model, call `.resetStates()` on either
 *   a specific layer, or on your entire model.
 *
 * Note on specifying the initial state of RNNs
 *   You can specify the initial state of RNN layers symbolically by
 *   calling them with the option `initialState`. The value of
 *   `initialState` should be a tensor or list of tensors representing
 *   the initial state of the RNN layer.
 *
 *   You can specify the initial state of RNN layers numerically by
 *   calling `resetStates` with the keyword argument `states`. The value of
 *   `states` should be a numpy array or list of numpy arrays representing
 *   the initial state of the RNN layer.
 *
 * Note on passing external constants to RNNs
 *   You can pass "external" constants to the cell using the `constants`
 *   keyword argument of `RNN.call` method. This requires that the `cell.call`
 *   method accepts the same keyword argument `constants`. Such constants
 *   can be used to conditon the cell transformation on additional static inputs
 *   (not changing over time), a.k.a an attention mechanism.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function rnn(args: RNNLayerArgs): Layer {
  return new RNN(args);
}

/**
 * Wrapper allowing a stack of RNN cells to behave as a single cell.
 *
 * Used to implement efficient stacked RNNs.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
export function stackedRNNCells(args: StackedRNNCellsArgs): RNNCell {
  return new StackedRNNCells(args);
}

// Wrapper Layers.

/** @doc {heading: 'Layers', subheading: 'Wrapper', namespace: 'layers'} */
export function bidirectional(args: BidirectionalLayerArgs): Bidirectional {
  return new Bidirectional(args);
}

/**
 * This wrapper applies a layer to every temporal slice of an input.
 *
 * The input should be at least 3D,  and the dimension of the index `1` will be
 * considered to be the temporal dimension.
 *
 * Consider a batch of 32 samples, where each sample is a sequence of 10 vectors
 * of 16 dimensions. The batch input shape of the layer is then `[32,  10,
 * 16]`, and the `inputShape`, not including the sample dimension, is
 * `[10, 16]`.
 *
 * You can then use `TimeDistributed` to apply a `Dense` layer to each of the 10
 * timesteps, independently:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.timeDistributed({
 *   layer: tf.layers.dense({units: 8}),
 *   inputShape: [10, 16],
 * }));
 *
 * // Now model.outputShape = [null, 10, 8].
 * // The output will then have shape `[32, 10, 8]`.
 *
 * // In subsequent layers, there is no need for `inputShape`:
 * model.add(tf.layers.timeDistributed({layer: tf.layers.dense({units: 32})}));
 * console.log(JSON.stringify(model.outputs[0].shape));
 * // Now model.outputShape = [null, 10, 32].
 * ```
 *
 * The output will then have shape `[32, 10, 32]`.
 *
 * `TimeDistributed` can be used with arbitrary layers, not just `Dense`, for
 * instance a `Conv2D` layer.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.timeDistributed({
 *   layer: tf.layers.conv2d({filters: 64, kernelSize: [3, 3]}),
 *   inputShape: [10, 299, 299, 3],
 * }));
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Wrapper', namespace: 'layers'}
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

/**
 * Apply additive zero-centered Gaussian noise.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * This is useful to mitigate overfitting
 * (you could see it as a form of random data augmentation).
 * Gaussian Noise (GS) is a natural choice as corruption process
 * for real valued inputs.
 *
 * # Arguments
 *     stddev: float, standard deviation of the noise distribution.
 *
 * # Input shape
 *         Arbitrary. Use the keyword argument `input_shape`
 *         (tuple of integers, does not include the samples axis)
 *         when using this layer as the first layer in a model.
 *
 * # Output shape
 *         Same shape as input.
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
export function gaussianNoise(args: GaussianNoiseArgs): GaussianNoise {
  return new GaussianNoise(args);
}

/**
 * Apply multiplicative 1-centered Gaussian noise.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
 *      http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
export function gaussianDropout(args: GaussianDropoutArgs): GaussianDropout {
  return new GaussianDropout(args);
}

/**
 * Applies Alpha Dropout to the input.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
 * to their original values, in order to ensure the self-normalizing property
 * even after this dropout.
 * Alpha Dropout fits well to Scaled Exponential Linear Units
 * by randomly setting activations to the negative saturation value.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *   - `noise_shape`: A 1-D `Tensor` of type `int32`, representing the
 *     shape for randomly generated keep/drop flags.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
export function alphaDropout(args: AlphaDropoutArgs): AlphaDropout {
  return new AlphaDropout(args);
}

/**
 * Masks a sequence by using a mask value to skip timesteps.
 *
 * If all features for a given sample timestep are equal to `mask_value`,
 * then the sample timestep will be masked (skipped) in all downstream layers
 * (as long as they support masking).
 *
 * If any downstream layer does not support masking yet receives such
 * an input mask, an exception will be raised.
 *
 * Arguments:
 *   - `maskValue`: Either None or mask value to skip.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * @doc {heading: 'Layers', subheading: 'Mask', namespace: 'layers'}
 */
export function masking(args?: MaskingArgs): Layer {
  return new Masking(args);
}
