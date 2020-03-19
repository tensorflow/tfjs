/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {ENGINE} from '../engine';
import {BACKEND_AGNOSTIC, GradConfig, KernelConfig, NamedTensorInfoMap, registerGradient, registerKernel} from '../kernel_registry';
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Creates a new tensor with the same values and shape as the specified
 * tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 *
 * x.clone().print();
 * ```
 *
 * @param x The tensor to clone.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function clone_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'clone', null);
  return ENGINE.runKernel(Clone, {x: $x}, {}) as T;
}

export const clone = op({clone_});

/**
 * Clone is generally not expected to be implemented by backends.
 */

// tslint:disable-next-line: variable-name
const Clone = 'Clone';
type CloneInputs = Pick<NamedTensorInfoMap, 'x'>;
const cloneKernelConfig: KernelConfig = {
  kernelName: Clone,
  backendName: BACKEND_AGNOSTIC,  // this is a backend agnostic kernel
  kernelFunc: ({inputs}) => {
    const {x} = inputs as CloneInputs;
    return ENGINE.makeTensorFromDataId(x.dataId, x.shape, x.dtype);
  }
};

const cloneGradientConfig: GradConfig = {
  kernelName: Clone,
  gradFunc: (dy: Tensor) => {
    return {x: () => dy.toFloat()};
  }
};

registerKernel(cloneKernelConfig);
registerGradient(cloneGradientConfig);
