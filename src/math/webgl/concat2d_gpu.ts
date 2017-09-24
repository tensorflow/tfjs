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

import * as concat_util from '../concat_util';
import {GPGPUProgram} from './gpgpu_math';

export class Concat2DProgram implements GPGPUProgram {
  variableNames = ['A', 'B'];
  params: Array<{}> = [];
  outputShape: number[] = [];
  userCode: string;

  constructor(
      x1Shape: [number, number], x2Shape: [number, number], axis: number) {
    const yAxes = ['yR', 'yC'];
    const concatAxis = yAxes[axis];
    this.params = [axis];
    this.outputShape =
        concat_util.computeConcatOutputShape(x1Shape, x2Shape, axis);
    this.userCode = `
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        float value = 0.0;
        if (${concatAxis} < ${x1Shape[axis]}) {
          value = getA(yR, yC);
        } else {
          ${concatAxis} -= ${x1Shape[axis]};
          value = getB(yR, yC);
        }

        setOutput(value);
      }
    `;
  }
}
