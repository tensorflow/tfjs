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
import {util} from '@tensorflow/tfjs-core';

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class TileProgram implements WebGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  dtype: string;
  size: number;
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
    this.size = util.sizeFromShape(this.outputShape);
    this.shaderKey = 'tile';
  }

  getUserCode(): string {
    const dtype = getCoordsDataType(this.rank);
    const sourceCoords = getSourceCoords(this.rank);

    const userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        int index = int(gl_GlobalInvocationID.x);
        if (index < size) {
          setOutput(index, getA(${sourceCoords}));
        }
      }
    `;
    return userCode;
  }
}

function getSourceCoords(rank: number): string {
  if (rank >= 5) {
    throw Error(`Tile for rank ${rank} is not yet supported`);
  }
  if (rank === 1) {
    return `(resRC % aShape)`;
  }

  const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
  const sourceCoords = [];
  for (let i = 0; i < rank; i++) {
    sourceCoords.push(`(${currentCoords[i]} % aShape[${i}])`);
  }
  return sourceCoords.join();
}
