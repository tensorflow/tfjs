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

import {GPGPUProgram} from './gpgpu_math';

export class ResizeNearestNeighborPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[] = [];
  userCode: string;

  constructor(
      inputShape: [number, number, number, number], newHeight: number,
      newWidth: number, alignCorners: boolean, halfPixelCenters: boolean) {
    const [batch, oldHeight, oldWidth, depth] = inputShape;
    this.outputShape = [batch, newHeight, newWidth, depth];

    const effectiveInSize: [number, number] = [
      (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
      (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
    ];

    const effectiveOutSize: [number, number] = [
      (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
      (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
    ];

    // When align corners is false, we rounds the value with floor.
    const roundBase = alignCorners ? '0.5' : '0.0';
    let sourceFracIndexRC: string;
    if (halfPixelCenters) {
      sourceFracIndexRC = `max((vec3(yRC) + vec3(0.5)) * ` +
          `effectiveInputOverOutputRatioRC, vec3(0.0))`;
    } else {
      sourceFracIndexRC = `vec3(yRC) * effectiveInputOverOutputRatioRC`;
    }

    this.userCode = `
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${effectiveInSize[0] / effectiveOutSize[0]},
          ${effectiveInSize[1] / effectiveOutSize[1]},
          ${effectiveInSize[1] / effectiveOutSize[1]});
      const vec3 inputShapeRC = vec3(${oldHeight}.0, ${oldWidth}.0,
                                     ${oldWidth}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${sourceFracIndexRC};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${roundBase})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${depth - 1};
        bool hasNextRow = coords.z < ${newWidth - 1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `;
  }
}
