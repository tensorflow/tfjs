/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {GPGPUProgram} from './gpgpu_math';

export class ResizeBilinear3DProgram implements GPGPUProgram {
  variableNames = ['A'];
  params: Array<{}> = [];
  outputShape: number[] = [];
  userCode: string;

  constructor(
      inputShape: [number, number, number],
      outputDimensionsRowCol: [number, number], alignCorners: boolean) {
    const depth = inputShape[2];
    this.outputShape =
        [outputDimensionsRowCol[0], outputDimensionsRowCol[1], depth];
    this.params = [alignCorners];

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
        vec3 coords = getOutputCoords();
        vec2 yRC = coords.xy;
        float d = coords.z;

        // Fractional source index.
        vec2 sourceFracIndexRC = yRC * effectiveInputOverOutputRatioRC;

        // Compute the four integer indices.
        vec2 sourceFloorRC = floor(sourceFracIndexRC);
        vec2 sourceCeilRC = min(inputShapeRC - 1.0, ceil(sourceFracIndexRC));

        float topLeft = getA(sourceFloorRC[0], sourceFloorRC[1], d);
        float bottomLeft = getA(sourceCeilRC[0], sourceFloorRC[1], d);
        float topRight = getA(sourceFloorRC[0], sourceCeilRC[1], d);
        float bottomRight = getA(sourceCeilRC[0], sourceCeilRC[1], d);

        vec2 fracRC = sourceFracIndexRC - sourceFloorRC;

        float top = topLeft + (topRight - topLeft) * fracRC[1];
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC[1];
        float newValue = top + (bottom - top) * fracRC[0];

        setOutput(newValue);
      }
    `;
  }
}
