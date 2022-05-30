/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class TileProgram implements WebGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  size = true;
  rank: number;

  constructor(aShape: number[], reps: number[]) {
    const outputShape: number[] = new Array(aShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = aShape[i] * reps[i];
    }
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.rank = this.outputShape.length;
    this.shaderKey = 'tile';
  }

  getUserCode(): string {
    const sourceCoords = getSourceCoords(this.rank, 'uniforms.');

    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          setOutputAtIndex(index, getA(${sourceCoords}));
        }
      }
    `;
    return userCode;
  }
}

function getSourceCoords(rank: number, uniformPrefix = ''): string {
  if (rank >= 5) {
    throw Error(`Tile for rank ${rank} is not yet supported`);
  }
  if (rank === 1) {
    return `(resRC % ${uniformPrefix}aShape)`;
  }

  const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
  const sourceCoords = [];
  for (let i = 0; i < rank; i++) {
    sourceCoords.push(`(${currentCoords[i]} % ${uniformPrefix}aShape[${i}])`);
  }
  return sourceCoords.join();
}
