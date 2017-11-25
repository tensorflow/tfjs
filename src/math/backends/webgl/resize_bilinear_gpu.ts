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

export class ResizeBilinear3DProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[] = [];
  userCode: string;

  constructor(
      inputShape: [number, number, number],
      outputDimensionsRowCol: [number, number], alignCorners: boolean) {
    const depth = inputShape[2];
    this.outputShape =
        [outputDimensionsRowCol[0], outputDimensionsRowCol[1], depth];

    const effectiveInputShape = alignCorners ?
        [inputShape[0] - 1, inputShape[1] - 1, depth] :
        inputShape;

    const effectiveOutputShape = alignCorners ?
        [this.outputShape[0] - 1, this.outputShape[1] - 1, depth] :
        this.outputShape;
    this.userCode = `
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${effectiveInputShape[0] / effectiveOutputShape[0]},
          ${effectiveInputShape[1] / effectiveOutputShape[1]});
      const vec2 inputShapeRC = vec2(${inputShape[0]}.0, ${inputShape[1]}.0);

      void main() {
        ivec3 coords = getOutputCoords();
        ivec2 yRC = coords.xy;
        int d = coords.z;

        // Fractional source index.
        vec2 sourceFracIndexRC = vec2(yRC) * effectiveInputOverOutputRatioRC;

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `;
  }
}
