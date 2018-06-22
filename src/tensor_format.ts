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

import {DataType, TypedArray} from './types';
import {computeStrides, rightPad, sizeFromShape} from './util';

// Maximum number of values before we decide to show ellipsis.
const FORMAT_LIMIT_NUM_VALS = 20;
// Number of first and last values to show when displaying a, b,...,y, z.
const FORMAT_NUM_FIRST_LAST_VALS = 3;
// Number of significant digits to show.
const FORMAT_NUM_SIG_DIGITS = 7;

export function tensorToString(
    vals: TypedArray, shape: number[], dtype: DataType, verbose: boolean) {
  const strides = computeStrides(shape);
  const padPerCol = computeMaxSizePerColumn(vals, shape, strides);
  const rank = shape.length;
  const valsLines = subTensorToString(vals, shape, strides, padPerCol);
  const lines = ['Tensor'];
  if (verbose) {
    lines.push(`  dtype: ${dtype}`);
    lines.push(`  rank: ${rank}`);
    lines.push(`  shape: [${shape}]`);
    lines.push(`  values:`);
  }
  lines.push(valsLines.map(l => '    ' + l).join('\n'));
  return lines.join('\n');
}

function computeMaxSizePerColumn(
    vals: TypedArray, shape: number[], strides: number[]): number[] {
  const n = sizeFromShape(shape);
  const numCols = strides[strides.length - 1];
  const padPerCol = new Array(numCols).fill(0);
  const rank = shape.length;
  if (rank > 1) {
    for (let row = 0; row < n / numCols; row++) {
      const offset = row * numCols;
      for (let j = 0; j < numCols; j++) {
        padPerCol[j] =
            Math.max(padPerCol[j], valToString(vals[offset + j], 0).length);
      }
    }
  }
  return padPerCol;
}

function valToString(val: number, pad: number) {
  return rightPad(
      parseFloat(val.toFixed(FORMAT_NUM_SIG_DIGITS)).toString(), pad);
}

function subTensorToString(
    vals: TypedArray, shape: number[], strides: number[], padPerCol: number[],
    isLast = true): string[] {
  const size = shape[0];
  const rank = shape.length;
  if (rank === 0) {
    return [vals[0].toString()];
  }

  if (rank === 1) {
    if (size > FORMAT_LIMIT_NUM_VALS) {
      const firstVals =
          Array.from(vals.subarray(0, FORMAT_NUM_FIRST_LAST_VALS));
      const lastVals =
          Array.from(vals.subarray(size - FORMAT_NUM_FIRST_LAST_VALS, size));
      return [
        '[' + firstVals.map((x, i) => valToString(x, padPerCol[i])).join(', ') +
        ', ..., ' +
        lastVals
            .map(
                (x, i) => valToString(
                    x, padPerCol[size - FORMAT_NUM_FIRST_LAST_VALS + i]))
            .join(', ') +
        ']'
      ];
    }
    return [
      '[' +
      Array.from(vals).map((x, i) => valToString(x, padPerCol[i])).join(', ') +
      ']'
    ];
  }

  // The array is rank 2 or more.
  const subshape = shape.slice(1);
  const substrides = strides.slice(1);
  const stride = strides[0];
  const lines: string[] = [];
  if (size > FORMAT_LIMIT_NUM_VALS) {
    for (let i = 0; i < FORMAT_NUM_FIRST_LAST_VALS; i++) {
      const start = i * stride;
      const end = start + stride;
      lines.push(...subTensorToString(
          vals.subarray(start, end), subshape, substrides, padPerCol,
          false /* isLast */));
    }
    lines.push('...');
    for (let i = size - FORMAT_NUM_FIRST_LAST_VALS; i < size; i++) {
      const start = i * stride;
      const end = start + stride;
      lines.push(...subTensorToString(
          vals.subarray(start, end), subshape, substrides, padPerCol,
          i === size - 1 /* isLast */));
    }
  } else {
    for (let i = 0; i < size; i++) {
      const start = i * stride;
      const end = start + stride;
      lines.push(...subTensorToString(
          vals.subarray(start, end), subshape, substrides, padPerCol,
          i === size - 1 /* isLast */));
    }
  }
  const sep = rank === 2 ? ',' : '';
  lines[0] = '[' + lines[0] + sep;
  for (let i = 1; i < lines.length - 1; i++) {
    lines[i] = ' ' + lines[i] + sep;
  }
  let newLineSep = ',\n';
  for (let i = 2; i < rank; i++) {
    newLineSep += '\n';
  }
  lines[lines.length - 1] =
      ' ' + lines[lines.length - 1] + ']' + (isLast ? '' : newLineSep);
  return lines;
}
