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

import * as util from '../../util';

import {GPGPUProgram} from './gpgpu_math';

export const ADD = 'return a + b;';
export const SUB = 'return a - b;';
export const MUL = 'return a * b;';
export const DIV = 'return a / b;';

export class BinaryOpProgram implements GPGPUProgram {
  variableNames = ['A', 'B'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;
  supportsBroadcasting = true;

  constructor(op: string, aShape: number[], bShape: number[]) {
    this.params = [op];
    this.outputShape = util.assertAndGetBroadcastedShape(aShape, bShape);
    this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `;
  }
}
