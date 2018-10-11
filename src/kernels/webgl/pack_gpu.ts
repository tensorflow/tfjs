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

export class PackProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;

    this.userCode = `
      void main() {
        ivec2 rc = getOutputCoords();

        int r = rc.x;
        int c = rc.y;

        if(r >= ${outputShape[0]} || c >= ${outputShape[1]}) {
          gl_FragColor = vec4(0);
        } else {
          int rp1 = r + 1;
          int cp1 = c + 1;

          bool cEdge = cp1 >= ${outputShape[1]};
          bool rEdge = rp1 >= ${outputShape[0]};

          gl_FragColor = vec4(
              getA(r, c),
              cEdge ? 0. : getA(r, cp1),
              rEdge ? 0. : getA(rp1, c),
              rEdge || cEdge ? 0. : getA(rp1, cp1)
            );
        }
      }
    `;
  }
}
