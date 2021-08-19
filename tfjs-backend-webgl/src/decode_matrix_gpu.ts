/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {getGlslDifferences} from './glsl_version';
import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';
import * as shader_util from './shader_compiler_util';
import {PackingScheme} from './tex_util';

export class DecodeMatrixProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: [number, number, number];
  packedInputs = false;
  packedOutput = true;
  outPackingScheme = PackingScheme.DENSE;
  enableShapeUniforms: boolean;
  customUniforms = [{name: 'texShape', type: 'ivec2' as const }];

  constructor(outputShape: [number, number, number]) {
    const glsl = getGlslDifferences();
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    this.userCode = `
      ivec3 outCoordsFromFlatIndex(int index) {
        ${
        this.enableShapeUniforms ?
            shader_util.getOutputLogicalCoordinatesFromFlatIndexByUniform(
                ['r', 'c', 'd'], outputShape) :
            shader_util.getLogicalCoordinatesFromFlatIndex(
                ['r', 'c', 'd'], outputShape)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${glsl.output} = result;
      }
    `;
  }
}
