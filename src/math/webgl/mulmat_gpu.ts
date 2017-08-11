/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {MatrixOrientation} from '../math';
import {GPGPUProgram} from './gpgpu_math';

export class MatMulProgram implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;

  constructor(aShape: [number, number], bShape: [number, number],
      aOrient = MatrixOrientation.REGULAR,
      bOrient = MatrixOrientation.REGULAR) {
    this.params = [aOrient, bOrient];

    const outerShapeA =
        (aOrient === MatrixOrientation.REGULAR) ? aShape[0] : aShape[1];
    const outerShapeB =
        (bOrient === MatrixOrientation.REGULAR) ? bShape[1] : bShape[0];
    this.outputShape = [outerShapeA, outerShapeB];

    const sharedDim =
      (aOrient === MatrixOrientation.REGULAR ? aShape[1] : aShape[0]);
    const aSnippet = (aOrient === MatrixOrientation.REGULAR) ?
        'aRow, i_float' : 'i_float, aRow';
    const bSnippet = (bOrient === MatrixOrientation.REGULAR) ?
        'i_float, bCol' : 'bCol, i_float';

    this.userCode = `
      const int sharedDim = ${sharedDim};

      float dotARowBCol(float aRow, float bCol) {
        float result = 0.0;
        for (int i = 0; i < sharedDim; i++) {
          float i_float = float(i);
          float a = getMatrixA(${aSnippet});
          float b = getMatrixB(${bSnippet});
          result += (a * b);
        }
        return result;
      }

      void main() {
        vec2 resRC = getOutputCoords();
        setOutput(dotARowBCol(resRC.x, resRC.y));
      }
    `;
  }
}

