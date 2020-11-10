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
import {Tensor2D} from '../tensor';
import {convertToTensor, convertToTensorArray} from '../tensor_util_env';
import {TensorLike} from '../types';
import {op} from './operation';

/**
 * @docalias (data: Tensor2D, c: Tensor2D, h: Tensor2D): [Tensor2D, Tensor2D]
 */
export type LSTMCellFunc = {
  (data: Tensor2D, c: Tensor2D, h: Tensor2D): [Tensor2D, Tensor2D];
};

/**
 * Computes the next states and outputs of a stack of LSTMCells.
 *
 * Each cell output is used as input to the next cell.
 *
 * Returns `[cellState, cellOutput]`.
 *
 * Derived from tf.contrib.rn.MultiRNNCell.
 *
 * @param lstmCells Array of LSTMCell functions.
 * @param data The input to the cell.
 * @param c Array of previous cell states.
 * @param h Array of previous cell outputs.
 *
 * @doc {heading: 'Operations', subheading: 'RNN'}
 */
function multiRNNCell_(
    lstmCells: LSTMCellFunc[], data: Tensor2D|TensorLike,
    c: Array<Tensor2D|TensorLike>,
    h: Array<Tensor2D|TensorLike>): [Tensor2D[], Tensor2D[]] {
  const $data = convertToTensor(data, 'data', 'multiRNNCell');
  const $c = convertToTensorArray(c, 'c', 'multiRNNCell');
  const $h = convertToTensorArray(h, 'h', 'multiRNNCell');

  let input = $data;
  const newStates = [];
  for (let i = 0; i < lstmCells.length; i++) {
    const output = lstmCells[i](input, $c[i], $h[i]);
    newStates.push(output[0]);
    newStates.push(output[1]);
    input = output[1];
  }
  const newC: Tensor2D[] = [];
  const newH: Tensor2D[] = [];
  for (let i = 0; i < newStates.length; i += 2) {
    newC.push(newStates[i]);
    newH.push(newStates[i + 1]);
  }
  return [newC, newH];
}
export const multiRNNCell = op({multiRNNCell_});
