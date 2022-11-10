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

import {GPGPUProgram} from './gpgpu_math';
import {assert} from './webgl_util';

export class MatMulPackedMrt2x2Program implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;
  enableShapeUniforms: false;
  mrtSupport: [number, number] = [2, 2];

  constructor(
      aShape: [number, number, number], bShape: [number, number, number],
      outputShape: [number, number, number], transposeA = false,
      transposeB = false, biasShape: number[] = null, activation: string = null,
      hasPreluActivation = false, hasLeakyreluActivation = false) {
    this.outputShape = outputShape;

    const sharedDim = bShape[1];
    const sharedDimensionPacked = Math.ceil(sharedDim / 4);

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      if (hasPreluActivation) {
        assert(false, 'PreluActivation is not supported!');
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
      } else if (hasLeakyreluActivation) {
        assert(false, 'LeakyreluActivation is not supported!');
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${activation}
        }`;
      } else {
        activationSnippet = `vec4 activation(vec4 x) {
          ${activation}
        }`;
      }

      applyActivationSnippet = `res_00 = activation(res_00);
      res_01 = activation(res_01);
      res_10 = activation(res_10);
      res_11 = activation(res_11);
      `;
    }

    let addBiasSnippet = '';
    if (biasShape != null) {
      if (biasShape.length === 1) {
        addBiasSnippet = `vec4 bias_0 = getBias(col).xyxy;
        vec4 bias_1 = getBias(col + 2).xyxy * colOOB;
        res_00 += bias_0;
        res_01 += bias_1;
        res_10 += bias_0;
        res_11 += bias_1;`;
      } else {
        assert(false, 'Matmul Bias only suppory 1D tensor!');
      }
      this.variableNames.push('bias');
    }

    if (hasPreluActivation) {
      assert(false, 'PreluActivation is not supported!');
      this.variableNames.push('preluActivationWeights');
    }

    if (hasLeakyreluActivation) {
      assert(false, 'LeakyreluActivation is not supported!');
      this.variableNames.push('leakyreluAlpha');
    }


    this.userCode = `
      ${activationSnippet}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${sharedDimensionPacked}.0;

      void main() {
        ivec3 rc = getOutputCoords();
        int row = rc.y;
        int col = rc.z;

        vec4 res_00 = vec4(0);
        vec4 res_01 = vec4(0);
        vec4 res_10 = vec4(0);
        vec4 res_11 = vec4(0);

        float rowOOB = ((row + 2) < ${
        transposeA ? aShape[2] : aShape[1]}) ? 1. : 0.;
        float colOOB = ((col + 2) < ${bShape[2]}) ? 1. : 0.;

        for (int ic = 0; ic < ${sharedDimensionPacked * 4}; ic += 4) {  // iC/4
          float icOOB = ((ic + 2) < ${bShape[1]}) ? 1. : 0.;

          ${
        transposeA ? `
          vec4 a_00 = getMatrixA(ic, row);
          vec4 a_01 = getMatrixA(ic, row + 2) * icOOB;
          vec4 a_10 = getMatrixA(ic + 2, row) * rowOOB;
          vec4 a_11 = getMatrixA(ic + 2, row + 2) * icOOB * rowOOB;
          ` :
                     `
          vec4 a_00 = getMatrixA(row, ic);
          vec4 a_01 = getMatrixA(row, ic + 2) * icOOB;
          vec4 a_10 = getMatrixA(row + 2, ic) * rowOOB;
          vec4 a_11 = getMatrixA(row + 2, ic + 2) * icOOB * rowOOB;
          `}

          vec4 b_00 = getMatrixB(ic, col);
          vec4 b_01 = getMatrixB(ic, col + 2) * colOOB;
          vec4 b_10 = getMatrixB(ic + 2, col) * icOOB;
          vec4 b_11 = getMatrixB(ic + 2, col + 2) * icOOB * colOOB;

          ${
        transposeA ? `
          res_00 += a_00.xxyy * b_00.xyxy;
          res_00 += a_00.zzww * b_00.zwzw;
          res_00 += a_10.xxyy * b_10.xyxy;
          res_00 += a_10.zzww * b_10.zwzw;

          res_01 += a_00.xxyy * b_01.xyxy;
          res_01 += a_00.zzww * b_01.zwzw;
          res_01 += a_10.xxyy * b_11.xyxy;
          res_01 += a_10.zzww * b_11.zwzw;

          res_10 += a_01.xxyy * b_00.xyxy;
          res_10 += a_01.zzww * b_00.zwzw;
          res_10 += a_11.xxyy * b_10.xyxy;
          res_10 += a_11.zzww * b_10.zwzw;

          res_11 += a_01.xxyy * b_01.xyxy;
          res_11 += a_01.zzww * b_01.zwzw;
          res_11 += a_11.xxyy * b_11.xyxy;
          res_11 += a_11.zzww * b_11.zwzw;` :
                     `
          res_00 += a_00.xxzz * b_00.xyxy;
          res_00 += a_00.yyww * b_00.zwzw;
          res_00 += a_01.xxzz * b_10.xyxy;
          res_00 += a_01.yyww * b_10.zwzw;

          res_01 += a_00.xxzz * b_01.xyxy;
          res_01 += a_00.yyww * b_01.zwzw;
          res_01 += a_01.xxzz * b_11.xyxy;
          res_01 += a_01.yyww * b_11.zwzw;

          res_10 += a_10.xxzz * b_00.xyxy;
          res_10 += a_10.yyww * b_00.zwzw;
          res_10 += a_11.xxzz * b_10.xyxy;
          res_10 += a_11.yyww * b_10.zwzw;

          res_11 += a_10.xxzz * b_01.xyxy;
          res_11 += a_10.yyww * b_01.zwzw;
          res_11 += a_11.xxzz * b_11.xyxy;
          res_11 += a_11.yyww * b_11.zwzw;`}
        }

        ${addBiasSnippet}
        ${applyActivationSnippet}

        result_00 = res_00;
        result_01 = res_01;
        result_10 = res_10;
        result_11 = res_11;
      }
    `;
  }
}
