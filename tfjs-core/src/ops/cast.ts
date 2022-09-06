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
import {Cast, CastAttrs, CastInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {DataType, TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Casts a `tf.Tensor` to a new dtype.
 *
 * ```js
 * const x = tf.tensor1d([1.5, 2.5, 3]);
 * tf.cast(x, 'int32').print();
 * ```
 * @param x The input tensor to be casted.
 * @param dtype The dtype to cast the input tensor to.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function cast_<T extends Tensor>(x: T|TensorLike, dtype: DataType): T {
  const $x = convertToTensor(x, 'x', 'cast');

  // Sanity checks.
  if (!util.isValidDtype(dtype)) {
    throw new Error(`Failed to cast to unknown dtype ${dtype}`);
  }
  if (dtype === 'string' && $x.dtype !== 'string' ||
      dtype !== 'string' && $x.dtype === 'string') {
    throw new Error('Only strings can be casted to strings');
  }

  const inputs: CastInputs = {x: $x};
  const attrs: CastAttrs = {dtype};

  return ENGINE.runKernel(
      Cast, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

export const cast = op({cast_});
