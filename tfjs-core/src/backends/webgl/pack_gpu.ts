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

import {getChannels} from '../packing_util';

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class PackProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;

  constructor(
      outputShape:
          number[]) {  // TODO(https://github.com/tensorflow/tfjs/issues/893):
                       // Only input / output 3D tensors.
    this.outputShape = outputShape;
    const rank = outputShape.length;

    if (rank === 0) {
      this.userCode = `
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;
    } else {
      const channels = getChannels('rc', rank);
      const dtype = getCoordsDataType(rank);
      const outOfBoundsCondition =
          getOutOfBoundsCondition(rank, outputShape, channels);
      const setup = getSetup(
          rank, outputShape[outputShape.length - 1],
          outputShape[outputShape.length - 2], channels);
      const output = getOutput(outputShape, channels);

      this.userCode = `
        void main() {
          ${dtype} rc = getOutputCoords();

          if(${outOfBoundsCondition}) {
            setOutput(vec4(0));
          } else {
            ${setup}

            setOutput(vec4(${output}));
          }
        }
      `;
    }
  }
}

function getSourceCoordsArr(rank: number, dims: string[]): string[] {
  const coords = [];

  for (let row = 0; row <= 1; row++) {
    for (let col = 0; col <= 1; col++) {
      let coord = `${row === 0 ? 'r' : 'rp1'}, ${col === 0 ? 'c' : 'cp1'}`;

      for (let d = 2; d < rank; d++) {
        coord = `${dims[dims.length - 1 - d]},` + coord;
      }

      coords.push(coord);
    }
  }
  return coords;
}

function getOutOfBoundsCondition(
    rank: number, shape: number[], dims: string[]): string {
  if (rank === 1) {
    return `rc > ${shape[0]}`;
  }

  let cond = '';
  for (let i = rank - 2; i < rank; i++) {
    cond += `${dims[i]} >= ${shape[i]}`;
    if (i < rank - 1) {
      cond += '||';
    }
  }

  return cond;
}

function getSetup(
    rank: number, cols: number, rows: number, dims: string[]): string {
  if (rank === 1) {
    return '';
  }

  const innerDims = dims.slice(-2);

  return `
    int r = ${innerDims[0]};
    int c = ${innerDims[1]};
    int rp1 = r + 1;
    int cp1 = c + 1;

    bool cEdge = cp1 >= ${cols};
    bool rEdge = rp1 >= ${rows};
  `;
}

function getOutput(shape: number[], dims: string[]): string {
  const rank = shape.length;
  const sourceCoords = getSourceCoordsArr(rank, dims);
  if (rank === 1) {
    return `getA(rc),
            rc + 1 >= ${shape[0]} ? 0. : getA(rc + 1),
            0, 0`;
  }

  return `getA(${sourceCoords[0]}),
          cEdge ? 0. : getA(${sourceCoords[1]}),
          rEdge ? 0. : getA(${sourceCoords[2]}),
          rEdge || cEdge ? 0. : getA(${sourceCoords[3]})`;
}