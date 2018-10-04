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

export class UnpackProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;

    this.userCode = `
      const vec2 onePixel = 1. / vec2(${outputShape[1]}, ${outputShape[0]});

      void main() {
        ivec2 rc = getOutputCoords();
        vec2 modCoord = mod(vec2(rc.y, rc.x), 2.);
        vec4 packedInput = getA(rc.x, rc.y);

        setOutput(
          modCoord.x == 0. ?
            (modCoord.y == 0. ? packedInput.r : packedInput.b) :
            (modCoord.y == 0. ? packedInput.g : packedInput.a)
        );
      }
    `;
  }
}