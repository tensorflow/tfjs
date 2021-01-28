/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class SliceProgram implements WebGPUProgram {
  variableNames = ['source'];
  outputShape: number[];
  shaderKey: string;
  rank: number;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workPerThread = 1;
  workGroupSize: [number, number, number] = [16, 1, 1];
  start: number[];
  destSize: number[];

  constructor(start: number[], destSize: number[]) {
    this.outputShape = destSize;
    this.rank = destSize.length;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.start = start;
    this.destSize = destSize;
    this.shaderKey = `slice_${start}_${destSize}`;
  }

  getUserCode(): string {
    const dtype = getCoordsDataType(this.rank);
    const sourceCoords = getCoords(this.rank);

    const coordSum = this.destSize.map((_, i) => {
      return `sourceLoc.${coords[i]} = ${this.start[i]} + coords.${coords[i]};`;
    });

    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);
        ${dtype} sourceLoc;
        ${dtype} coords = getOutputCoords();
        ${coordSum.join('\n')}
        setOutput(index, getSource(${sourceCoords}));
      }
    `;
    return userCode;
  }
}

const coords = ['x', 'y', 'z', 'w', 'u', 'v'];

function getCoords(rank: number): string {
  if (rank === 1) {
    return 'sourceLoc';
  } else if (rank <= 6) {
    return coords.slice(0, rank).map(coord => `sourceLoc.${coord}`).join(',');
  } else {
    throw Error(`Slicing for rank ${rank} is not yet supported`);
  }
}
