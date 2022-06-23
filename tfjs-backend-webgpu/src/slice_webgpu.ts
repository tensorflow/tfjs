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

import {getCoordsDataType, getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class SliceProgram implements WebGPUProgram {
  variableNames = ['source'];
  uniforms: string;
  outputShape: number[];
  shaderKey: string;
  rank: number;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workPerThread = 1;
  workGroupSize: [number, number, number] = [64, 1, 1];
  start: number[];
  size = true;

  constructor(start: number[], destSize: number[]) {
    this.outputShape = destSize;
    this.rank = destSize.length;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.start = start;
    this.uniforms = `start : ${getCoordsDataType(start.length)}, `;
    this.shaderKey = 'slice';
  }

  getUserCode(): string {
    const dtype = getCoordsDataType(this.rank);
    const sourceCoords = getCoords(this.rank);
    let coordSum;
    if (this.start.length === 1) {
      coordSum = this.outputShape.map((_, i) => {
        return `sourceLoc = uniforms.start + coords;`;
      });
    } else {
      coordSum = this.outputShape.map((_, i) => {
        return `sourceLoc.${coords[i]} = uniforms.start[${i}] + coords.${
            coords[i]};`;
      });
    }

    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          var sourceLoc : ${dtype};
          let coords = getCoordsFromIndex(index);
          ${coordSum.join('\n')}
          setOutputAtIndex(index, getSource(${sourceCoords}));
        }
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
