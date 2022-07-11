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

export class GatherProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[] = ['A', 'indices'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  aShape: number[];
  size = true;

  constructor(aShape: number[], outputShape: number[]) {
    this.outputShape = aShape.slice();
    this.aShape = aShape;
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `gather`;
  }

  getUserCode(): string {
    const sourceCoords = getSourceCoords(this.aShape);
    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          let indexZ = i32(getIndices(resRC.x, resRC.z));
          let inBounds = select(0.0, 1.0, indexZ >= 0 && indexZ < uniforms.aShape[2]);
          setOutputAtIndex(index, inBounds * getA(${sourceCoords}));
        }
      }
    `;
    return userCode;
  }
}

// The input and output are always flattened into rank 4 tensors.
function getSourceCoords(aShape: number[]): string {
  const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
  const sourceCoords = [];
  for (let i = 0; i < aShape.length; i++) {
    if (i === 2) {
      sourceCoords.push('indexZ');
    } else {
      sourceCoords.push(`${currentCoords[i]}`);
    }
  }
  return sourceCoords.join();
}
