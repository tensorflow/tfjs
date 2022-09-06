/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {Scalar, Tensor1D, Tensor2D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {add} from './add';
import {concat} from './concat';
import {matMul} from './mat_mul';
import {mul} from './mul';
import {op} from './operation';
import {sigmoid} from './sigmoid';
import {slice} from './slice';
import {tanh} from './tanh';

/**
 * Computes the next state and output of a BasicLSTMCell.
 *
 * Returns `[newC, newH]`.
 *
 * Derived from tf.contrib.rnn.BasicLSTMCell.
 *
 * @param forgetBias Forget bias for the cell.
 * @param lstmKernel The weights for the cell.
 * @param lstmBias The bias for the cell.
 * @param data The input to the cell.
 * @param c Previous cell state.
 * @param h Previous cell output.
 *
 * @doc {heading: 'Operations', subheading: 'RNN'}
 */
function basicLSTMCell_(
    forgetBias: Scalar|TensorLike, lstmKernel: Tensor2D|TensorLike,
    lstmBias: Tensor1D|TensorLike, data: Tensor2D|TensorLike,
    c: Tensor2D|TensorLike, h: Tensor2D|TensorLike): [Tensor2D, Tensor2D] {
  const $forgetBias =
      convertToTensor(forgetBias, 'forgetBias', 'basicLSTMCell');
  const $lstmKernel =
      convertToTensor(lstmKernel, 'lstmKernel', 'basicLSTMCell');
  const $lstmBias = convertToTensor(lstmBias, 'lstmBias', 'basicLSTMCell');
  const $data = convertToTensor(data, 'data', 'basicLSTMCell');
  const $c = convertToTensor(c, 'c', 'basicLSTMCell');
  const $h = convertToTensor(h, 'h', 'basicLSTMCell');

  const combined = concat([$data, $h], 1);
  const weighted = matMul(combined, $lstmKernel);
  const res: Tensor2D = add(weighted, $lstmBias);

  // i = input_gate, j = new_input, f = forget_gate, o = output_gate
  const batchSize = res.shape[0];
  const sliceCols = res.shape[1] / 4;
  const sliceSize: [number, number] = [batchSize, sliceCols];
  const i = slice(res, [0, 0], sliceSize);
  const j = slice(res, [0, sliceCols], sliceSize);
  const f = slice(res, [0, sliceCols * 2], sliceSize);
  const o = slice(res, [0, sliceCols * 3], sliceSize);

  const newC: Tensor2D =
      add(mul(sigmoid(i), tanh(j)),
          mul($c, sigmoid(add($forgetBias, f)) as Tensor2D));
  const newH: Tensor2D = mul(tanh(newC), sigmoid(o));
  return [newC, newH];
}

export const basicLSTMCell = op({basicLSTMCell_});
