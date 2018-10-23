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

import {getChannels, getInnerDims, getSourceCoords} from '../packing_util';

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class UnpackProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  outputShape: number[];
  userCode: string;

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;
    const rank = outputShape.length;

    const channels = getChannels('rc');
    const dtype = getCoordsDataType(rank);
    const sourceCoords = getSourceCoords(rank, channels);
    const innerDims = getInnerDims(rank, channels);
    const coords = rank === 1 ? 'rc' : innerDims.join(',');

    this.userCode = `
      void main() {
        ${dtype} rc = getOutputCoords();
        vec2 modCoord = mod(vec2(${coords}), 2.);
        vec4 packedInput = getA(${sourceCoords});

        setOutput(
          modCoord.x == 0. ?
            (modCoord.y == 0. ? packedInput.r : packedInput.g) :
            (modCoord.y == 0. ? packedInput.b : packedInput.a)
        );
      }
    `;
  }
}