/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// Layer activation functions
import {scalar, Tensor} from 'deeplearn';

import * as K from './backend/deeplearnjs_backend';
import {ValueError} from './errors';
import {ConfigDictValue} from './types';

export type ActivationFn = (tensor: Tensor, axis?: number) => Tensor;
export type ActivationIdentifier = 'elu'|'hardsigmoid'|'linear'|'relu'|'relu6'|
    'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|string;

// TODO(cais): Consider switching arg type from string to Enum.
export function getActivation(activationType: ActivationIdentifier):
    ActivationFn {
  if (activationType == null) {
    return linear;
  } else if (activationType.toLowerCase() === 'elu') {
    return elu;
  } else if (activationType.toLowerCase() === 'hardsigmoid') {
    return hardSigmoid;
  } else if (activationType.toLowerCase() === 'linear') {
    return linear;
  } else if (activationType.toLowerCase() === 'relu') {
    return relu;
  } else if (activationType.toLowerCase() === 'relu6') {
    return relu6;
  } else if (activationType.toLowerCase() === 'selu') {
    return selu;
  } else if (activationType.toLowerCase() === 'sigmoid') {
    return sigmoid;
  } else if (activationType.toLowerCase() === 'softmax') {
    return softmax;
  } else if (activationType.toLowerCase() === 'softplus') {
    return softplus;
  } else if (activationType.toLowerCase() === 'softsign') {
    return softsign;
  } else if (activationType.toLowerCase() === 'tanh') {
    return tanh;
  } else {
    throw new ValueError(`Unsupported activation function ${activationType}`);
  }
}

/**
 * Exponential linear unit (ELU).
 * Reference: https://arxiv.org/abs/1511.07289
 * @param x: Input.
 * @param alpha: Scaling factor the negative section.
 * @return Output of the ELU activation.
 */
export function elu(x: Tensor, alpha = 1): Tensor {
  return K.elu(x, alpha);
}


/**
 * Scaled Exponential Linear Unit. (Klambauer et al., 2017).
 * Reference: Self-Normalizing Neural Networks, https://arxiv.org/abs/1706.02515
 * Notes:
 *   - To be used together with the initialization "lecunNormal".
 *   - To be used together with the dropout variant "AlphaDropout".
 *
 * @param x: Input
 * @returns Tensor with the same shape and dtype as `x`.
 */
export function selu(x: Tensor): Tensor {
  return K.selu(x);
}


// Rectified linear unit
export function relu(x: Tensor): Tensor {
  return K.relu(x);
}

/**
 * Rectified linear unit activation maxing out at 6.0.
 */
// TODO(bileschi): A new constant 6 here is being created at each invocation.
// A better pattern would be to reuse a single constant 6, created after the
// backend math has been instantiated.
export function relu6(x: Tensor): Tensor {
  return K.minimum(scalar(6.0), K.relu(x));
}

//* Linear activation (no-op) */
export function linear(x: Tensor): Tensor {
  return x;
}

/**
 * Sigmoid activation function.
 */
export function sigmoid(x: Tensor): Tensor {
  return K.sigmoid(x);
}

/**
 * Segment-wise linear approximation of sigmoid.
 */
export function hardSigmoid(x: Tensor): Tensor {
  return K.hardSigmoid(x);
}

/**
 * Softplus activation function.
 */
export function softplus(x: Tensor): Tensor {
  return K.softplus(x);
}

/**
 * Softsign activation function.
 */
export function softsign(x: Tensor): Tensor {
  return K.softsign(x);
}

/**
 * Hyperbolic tangent function.
 * @param x Input.
 * @returns Output of the hyperbolic tangent function.
 */
export function tanh(x: Tensor): Tensor {
  return K.tanh(x);
}

/**
 * Softmax activation function.
 *
 * @param x Tensor.
 * @param axis Integer, axis along which the softmax normalization is applied.
 * Invalid if < 2, as softmax across 1 (the batch dimension) is assumed to be
 * an error.
 *
 * @returns a Tensor of the same shape as x
 *
 * @throws ValueError: In case `dim(x) < 2`.
 *         NotImplementedError if input x is not ConcreteTensor.
 */
export function softmax(x: Tensor, axis: number = (-1)): Tensor {
  return K.softmax(x, axis);
}

export function serializeActivation(activation: ActivationFn): ConfigDictValue {
  return activation.name;
}
