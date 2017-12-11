/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {MatrixOrientation} from '../types/matmul';

import {GPGPUProgram} from './gpgpu_math';

export class MatMulProgram implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  outputShape: number[];
  userCode: string;

  constructor(
      aShape: [number, number], bShape: [number, number],
      aOrient = MatrixOrientation.REGULAR,
      bOrient = MatrixOrientation.REGULAR) {
    const outerShapeA =
        (aOrient === MatrixOrientation.REGULAR) ? aShape[0] : aShape[1];
    const outerShapeB =
        (bOrient === MatrixOrientation.REGULAR) ? bShape[1] : bShape[0];
    this.outputShape = [outerShapeA, outerShapeB];

    const sharedDim =
        (aOrient === MatrixOrientation.REGULAR ? aShape[1] : aShape[0]);
    const aSnippetFromOffset = (vec4Offset: number, indexVar: string|number) =>
        (aOrient === MatrixOrientation.REGULAR) ?
        `aRow, ${indexVar} + ${vec4Offset}` :
        `${indexVar} + ${vec4Offset}, aRow`;
    const bSnippetFromOffset = (vec4Offset: number, indexVar: string|number) =>
        (bOrient === MatrixOrientation.REGULAR) ?
        `${indexVar} + ${vec4Offset}, bCol` :
        `bCol, ${indexVar} + ${vec4Offset}`;

    const sharedDimNearestVec4 = Math.floor(sharedDim / 4) * 4;
    const sharedDimVec4Remainder = sharedDim % 4;

    this.userCode = ` float dotARowBCol(int aRow, int bCol) {
      float result = 0.0;
      for (int i = 0; i < ${sharedDimNearestVec4}; i += 4) {
        vec4 a = vec4(
          getMatrixA(${aSnippetFromOffset(0, 'i')}),
          getMatrixA(${aSnippetFromOffset(1, 'i')}),
          getMatrixA(${aSnippetFromOffset(2, 'i')}),
          getMatrixA(${aSnippetFromOffset(3, 'i')})
        );
        vec4 b = vec4(
          getMatrixB(${bSnippetFromOffset(0, 'i')}),
          getMatrixB(${bSnippetFromOffset(1, 'i')}),
          getMatrixB(${bSnippetFromOffset(2, 'i')}),
          getMatrixB(${bSnippetFromOffset(3, 'i')})
        );

        result += dot(a, b);
      }

      if (${sharedDimVec4Remainder === 1}) {
        result += getMatrixA(${aSnippetFromOffset(0, sharedDimNearestVec4)}) *
          getMatrixB(${bSnippetFromOffset(0, sharedDimNearestVec4)});
      } else if (${sharedDimVec4Remainder === 2}) {
        vec2 a = vec2(
          getMatrixA(${aSnippetFromOffset(0, sharedDimNearestVec4)}),
          getMatrixA(${aSnippetFromOffset(1, sharedDimNearestVec4)})
        );
        vec2 b = vec2(
          getMatrixB(${bSnippetFromOffset(0, sharedDimNearestVec4)}),
          getMatrixB(${bSnippetFromOffset(1, sharedDimNearestVec4)})
        );
        result += dot(a, b);
      } else if (${sharedDimVec4Remainder === 3}) {
        vec3 a = vec3(
          getMatrixA(${aSnippetFromOffset(0, sharedDimNearestVec4)}),
          getMatrixA(${aSnippetFromOffset(1, sharedDimNearestVec4)}),
          getMatrixA(${aSnippetFromOffset(2, sharedDimNearestVec4)})
        );
        vec3 b = vec3(
          getMatrixB(${bSnippetFromOffset(0, sharedDimNearestVec4)}),
          getMatrixB(${bSnippetFromOffset(1, sharedDimNearestVec4)}),
          getMatrixB(${bSnippetFromOffset(2, sharedDimNearestVec4)})
        );
        result += dot(a, b);
      }

      return result;
    }

    void main() {
      ivec2 resRC = getOutputCoords();
      setOutput(dotARowBCol(resRC.x, resRC.y));
    }
    `;
  }
}
