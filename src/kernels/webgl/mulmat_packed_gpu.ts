/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

export class MatMulPackedProgram implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  outputShape: number[];
  userCode: string;

  constructor(
      aShape: [number, number], bShape: [number, number],
      outputShape: [number, number], transposeA = false, transposeB = false) {
    this.outputShape = outputShape;

    const sharedDim = transposeA ? aShape[0] : aShape[1];
    const sharedDimensionPacked = Math.ceil(sharedDim / 2);

    const aSample = transposeA ? 'i * 2, rc.x' : 'rc.x, i * 2';
    const bSample = transposeB ? 'rc.y, i * 2' : 'i * 2, rc.y';
    const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
    const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];

    this.userCode = `
      const float sharedDimension = ${sharedDimensionPacked}.0;

      vec4 dot2x2ARowBCol(ivec2 rc) {
        vec4 result = vec4(0);
        for (int i = 0; i < ${sharedDimensionPacked}; i++) {
          vec4 a = getMatrixA(${aSample});
          vec4 b = getMatrixB(${bSample});

          result += (${aSwizzle[0]} * ${bSwizzle[0]}) + (${aSwizzle[1]} * ${
        bSwizzle[1]});
        }
        return result;
      }

      void main() {
        ivec2 rc = getOutputCoords();
        gl_FragColor = dot2x2ARowBCol(rc);
      }
    `;
  }
}