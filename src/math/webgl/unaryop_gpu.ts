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

import {GPGPUProgram} from './gpgpu_math';

export class UnaryOpProgram implements GPGPUProgram {
  variableNames = ['A'];
  params: Array<{}>;
  userCode: string;
  outputShape: number[];

  constructor(aShape: number[], opSnippet: string) {
    this.outputShape = aShape;
    this.params = [opSnippet];
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

export const CHECK_NAN_SNIPPET = `
  if (isNaN(x)) {
    return x;
  }
`;

export const ABS = `
  return abs(x);
`;

export const RELU = `
  return (x < 0.0) ? 0.0 : x;
`;

export const STEP = `
  return (x == x) ? (x > 0.0 ? 1.0 : 0.0) : x;
`;

export const NEG = `
  return -x;
`;

export const EXP = `
  return exp(x);
`;

export const LOG = `
  return log(x);
`;

export const SQRT = CHECK_NAN_SNIPPET + `
  return sqrt(x);
`;

export const SIGMOID = `
  return 1.0 / (1.0 + exp(-1.0 * x));
`;

export const SIN = CHECK_NAN_SNIPPET + `
  return sin(x);
`;

export const COS = CHECK_NAN_SNIPPET + `
  return cos(x);
`;

export const TAN = `
  return tan(x);
`;

export const ASIN = CHECK_NAN_SNIPPET + `
  return asin(x);
`;

export const ACOS = CHECK_NAN_SNIPPET + `
  return acos(x);
`;

export const ATAN = CHECK_NAN_SNIPPET + `
  return atan(x);
`;

export const SINH = `
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`;

export const COSH = `
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`;

export const TANH = `
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`;
