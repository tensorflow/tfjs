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

import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';

export class UnaryOpProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];
  enableShapeUniforms: boolean;

  constructor(aShape: number[], opSnippet: string) {
    this.outputShape = aShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);
    this.userCode = `
      float unaryOperation(float x) {
        ${opSnippet}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `;
  }
}

export const CHECK_NAN_SNIPPET = `if (isnan(x)) return x;`;

export const LINEAR = `return x;`;

export const ABS = `return abs(x);`;

export function STEP(alpha = 0.0) {
  return CHECK_NAN_SNIPPET + `
    return x > 0.0 ? 1.0 : float(${alpha});
  `;
}

export const ELU = `return (x >= 0.0) ? x : (exp(x) - 1.0);`;
export const RELU = CHECK_NAN_SNIPPET + `
  return (x < 0.0) ? 0.0 : x;
`;

export const RELU6 = CHECK_NAN_SNIPPET + `
  return (x < 0.0) ? 0.0 : min(6.0, x);
`;

export const CLONE = 'return x;';

export const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * x));`;
