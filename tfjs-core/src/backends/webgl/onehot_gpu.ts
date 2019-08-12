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

export class OneHotProgram implements GPGPUProgram {
  variableNames = ['indices'];
  outputShape: number[];
  userCode: string;

  // Caching uniform location for speed.
  seedLoc: WebGLUniformLocation;

  constructor(
      numIndices: number, depth: number, onValue: number, offValue: number) {
    this.outputShape = [numIndices, depth];

    this.userCode = `
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${offValue}), float(${onValue}),
                      float(index == coords.y)));
      }
    `;
  }
}
