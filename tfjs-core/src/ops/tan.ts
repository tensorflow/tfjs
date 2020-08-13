/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {Tan, TanInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Computes tan of the input `tf.Tensor` element-wise, `tan(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.tan().print();  // or tf.tan(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function tan_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'tan');

  const inputs: TanInputs = {x: $x};

  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.tan($x);
    save([$x]);
    return res;
  }, inputs as {} as NamedTensorMap, null /* grad */, Tan);
}
export const tan = op({tan_});
