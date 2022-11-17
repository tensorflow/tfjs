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

import {BaseCallbackConstructor, CallbackConstructorRegistry} from './base_callbacks';
import {ContainerArgs} from './engine/container';
import {Input, InputConfig,} from './engine/input_layer';
import {SymbolicTensor} from './engine/topology';
import {LayersModel} from './engine/training';
import {Sequential, SequentialArgs} from './models';

export {loadLayersModel} from './models';

// TODO(cais): Add doc string to all the public static functions in this
//   class; include exectuable JavaScript code snippets where applicable
//   (b/74074458).

// LayersModel and related factory methods.

/**
 * A model is a data structure that consists of `Layers` and defines inputs
 * and outputs.
 *
 * The key difference between `tf.model` and `tf.sequential` is that
 * `tf.model` is more generic, supporting an arbitrary graph (without
 * cycles) of layers. `tf.sequential` is less generic and supports only a linear
 * stack of layers.
 *
 * When creating a `tf.LayersModel`, specify its input(s) and output(s). Layers
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
 * const denseLayer2 = tf.layers.dense({units: 4, activation: 'softmax'});
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
 *   `tf.sequential`, `tf.loadLayersModel`.
 *
 * @doc {heading: 'Models', subheading: 'Creation'}
 */
export function model(args: ContainerArgs): LayersModel {
  return new LayersModel(args);
}

/**
 * Creates a `tf.Sequential` model.  A sequential model is any model where the
 * outputs of one layer are the inputs to the next layer, i.e. the model
 * topology is a simple 'stack' of layers, with no branching or skipping.
 *
 * This means that the first layer passed to a `tf.Sequential` model should have
 * a defined input shape. What that means is that it should have received an
 * `inputShape` or `batchInputShape` argument, or for some type of layers
 * (recurrent, Dense...) an `inputDim` argument.
 *
 * The key difference between `tf.model` and `tf.sequential` is that
 * `tf.sequential` is less generic, supporting only a linear stack of layers.
 * `tf.model` is more generic and supports an arbitrary graph (without
 * cycles) of layers.
 *
 * Examples:
 *
 * ```js
 * const model = tf.sequential();
 *
 * // First layer must have an input shape defined.
 * model.add(tf.layers.dense({units: 32, inputShape: [50]}));
 * // Afterwards, TF.js does automatic shape inference.
 * model.add(tf.layers.dense({units: 4}));
 *
 * // Inspect the inferred shape of the model's output, which equals
 * // `[null, 4]`. The 1st dimension is the undetermined batch dimension; the
 * // 2nd is the output size of the model's last layer.
 * console.log(JSON.stringify(model.outputs[0].shape));
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
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * You can also use an `Array` of already-constructed `Layer`s to create
 * a `tf.Sequential` model:
 *
 * ```js
 * const model = tf.sequential({
 *   layers: [tf.layers.dense({units: 32, inputShape: [50]}),
 *            tf.layers.dense({units: 4})]
 * });
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * @doc {heading: 'Models', subheading: 'Creation'}
 */
export function sequential(config?: SequentialArgs): Sequential {
  return new Sequential(config);
}

/**
 * Used to instantiate an input to a model as a `tf.SymbolicTensor`.
 *
 * Users should call the `input` factory function for
 * consistency with other generator functions.
 *
 * Example:
 *
 * ```js
 * // Defines a simple logistic regression model with 32 dimensional input
 * // and 3 dimensional output.
 * const x = tf.input({shape: [32]});
 * const y = tf.layers.dense({units: 3, activation: 'softmax'}).apply(x);
 * const model = tf.model({inputs: x, outputs: y});
 * model.predict(tf.ones([2, 32])).print();
 * ```
 *
 * Note: `input` is only necessary when using `model`. When using
 * `sequential`, specify `inputShape` for the first layer or use `inputLayer`
 * as the first layer.
 *
 * @doc {heading: 'Models', subheading: 'Inputs'}
 */
export function input(config: InputConfig): SymbolicTensor {
  return Input(config);
}

export function registerCallbackConstructor(
    verbosityLevel: number,
    callbackConstructor: BaseCallbackConstructor): void {
  CallbackConstructorRegistry.registerCallbackConstructor(
      verbosityLevel, callbackConstructor);
}
