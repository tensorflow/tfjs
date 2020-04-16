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

import {Tensor2D} from '../tensor';
import {DataType} from '../types';

import {buffer, expandDims} from './array_ops';
import {op} from './operation';
import {tile} from './tile';

/**
 * Create an identity matrix.
 *
 * @param numRows Number of rows.
 * @param numColumns Number of columns. Defaults to `numRows`.
 * @param batchShape If provided, will add the batch shape to the beginning
 *   of the shape of the returned `tf.Tensor` by repeating the identity
 *   matrix.
 * @param dtype Data type.
 * @returns Identity matrix of the specified size and data type, possibly
 *   with batch repetition if `batchShape` is specified.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function eye_(
    numRows: number, numColumns?: number,
    batchShape?:
        [
          number
        ]|[number,
           number]|[number, number, number]|[number, number, number, number],
    dtype: DataType = 'float32'): Tensor2D {
  if (numColumns == null) {
    numColumns = numRows;
  }
  const buff = buffer([numRows, numColumns], dtype);
  const n = numRows <= numColumns ? numRows : numColumns;
  for (let i = 0; i < n; ++i) {
    buff.set(1, i, i);
  }
  const out = buff.toTensor().as2D(numRows, numColumns);
  if (batchShape == null) {
    return out;
  } else {
    if (batchShape.length === 1) {
      return tile(expandDims(out, 0), [batchShape[0], 1, 1]);
    } else if (batchShape.length === 2) {
      return tile(
          expandDims(expandDims(out, 0), 0),
          [batchShape[0], batchShape[1], 1, 1]);
    } else if (batchShape.length === 3) {
      return tile(
          expandDims(expandDims(expandDims(out, 0), 0), 0),
          [batchShape[0], batchShape[1], batchShape[2], 1, 1]);
    } else {
      throw new Error(
          `eye() currently supports only 1D and 2D ` +
          // tslint:disable-next-line:no-any
          `batchShapes, but received ${(batchShape as any).length}D.`);
    }
  }
}

export const eye = op({eye_});
