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

export class ReverseProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(xShape: number[], axis: number[]) {
    this.outputShape = xShape;
    const getRevVar = (i: number) => {
      if (axis.indexOf(i) !== -1 && xShape[i] !== 1) {
        return `${xShape[i]} - coords[${i}] - 1`;
      }
      return `coords[${i}]`;
    };

    const b = getRevVar(0);
    const r = getRevVar(1);
    const c = getRevVar(2);
    const d = getRevVar(3);

    this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        float val = getX(${b}, ${r}, ${c}, ${d});
        setOutput(val);
      }
    `;
  }
}
