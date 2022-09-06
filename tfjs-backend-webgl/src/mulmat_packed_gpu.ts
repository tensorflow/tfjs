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

import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';

export class MatMulPackedProgram implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(
      aShape: [number, number, number], bShape: [number, number, number],
      outputShape: [number, number, number], transposeA = false,
      transposeB = false, addBias = false, activation: string = null,
      hasPreluActivation = false, hasLeakyreluActivation = false) {
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    const sharedDim = transposeA ? aShape[1] : aShape[2];
    const sharedDimensionPacked = Math.ceil(sharedDim / 2);

    const aSample = transposeA ? 'i * 2, rc.y' : 'rc.y, i * 2';
    const bSample = transposeB ? 'rc.z, i * 2' : 'i * 2, rc.z';
    const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
    const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      if (hasPreluActivation) {
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
      } else if (hasLeakyreluActivation) {
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${activation}
        }`;
      } else {
        activationSnippet = `vec4 activation(vec4 x) {
          ${activation}
        }`;
      }

      applyActivationSnippet = `result = activation(result);`;
    }

    const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivation) {
      this.variableNames.push('preluActivationWeights');
    }

    if (hasLeakyreluActivation) {
      this.variableNames.push('leakyreluAlpha');
    }

    let batchASnippet = 'rc.x';
    let batchBSnippet = 'rc.x';
    if (aShape[0] < bShape[0]) {
      batchASnippet = `int(min(float(rc.x), ${aShape[0] - 1}.))`;
    } else if (bShape[0] < aShape[0]) {
      batchBSnippet = `int(min(float(rc.x), ${bShape[0] - 1}.))`;
    }

    this.userCode = `
      ${activationSnippet}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${sharedDimensionPacked}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        for (int i = 0; i < ${sharedDimensionPacked}; i++) {
          int batchA = ${batchASnippet};
          int batchB = ${batchBSnippet};
          vec4 a = getMatrixA(batchA, ${aSample});
          vec4 b = getMatrixB(batchB, ${bSample});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${aSwizzle[0]} * ${bSwizzle[0]});
          result += (${aSwizzle[1]} * ${bSwizzle[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${addBiasSnippet}

        ${applyActivationSnippet}

        setOutput(result);
      }
    `;
  }
}
