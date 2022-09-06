/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export type GatherShape = [number, number, number, number];

export class GatherProgram implements GPGPUProgram {
  variableNames = ['A', 'indices'];
  outputShape: number[];
  userCode: string;
  rank: number;

  constructor(aShape: GatherShape, outputShape: GatherShape) {
    this.outputShape = outputShape;
    this.rank = outputShape.length;
    const dtype = getCoordsDataType(this.rank);
    const sourceCoords = getSourceCoords(aShape, 2);

    this.userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${aShape[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${sourceCoords}));
      }
    `;
  }
}

// The input and output are always flattened into rank 4 tensors.
function getSourceCoords(aShape: GatherShape, axis: number): string {
  const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];

  const sourceCoords = [];
  for (let i = 0; i < aShape.length; i++) {
    if (i === 2) {
      sourceCoords.push('index');
    } else {
      sourceCoords.push(`${currentCoords[i]}`);
    }
  }
  return sourceCoords.join();
}
