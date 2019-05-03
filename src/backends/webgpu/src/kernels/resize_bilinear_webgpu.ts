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

import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ResizeBilinearProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];

  constructor(
      inputShape: [number, number, number, number], newHeight: number,
      newWidth: number, alignCorners: boolean) {
    this.outputShape = [inputShape[0], newHeight, newWidth, inputShape[3]];

    this.dispatchLayout = {x: [1], y: [2], z: [0, 3]};

    this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape);

    const adjustHeight = alignCorners && newHeight > 1;
    const adjustWidth = alignCorners && newWidth > 1;

    this.userCode = `
      float getValue(int b, int r, int c, int d) {
        return x[getFlatIndex(ivec4(b, r, c, d), xShape)];
      }

      void main() {
        ivec4 coords = getOutputCoords();

        if (all(lessThan(coords, outShape))) {
          int b = coords[0];
          int d = coords[3];
          ivec2 rc = coords.yz;

          vec2 effectiveInSize = vec2(
            ${adjustHeight ? 'xShape.y - 1.0' : 'xShape.y'},
            ${adjustWidth ? 'xShape.z - 1.0' : 'xShape.z'});

          vec2 effectiveOutSize = vec2(
            ${adjustHeight ? 'outShape.y - 1.0' : 'outShape.y'},
            ${adjustWidth ? 'outShape.z - 1.0' : 'outShape.z'});

          vec2 effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          uint index = getFlatIndex(coords, outShape);

          // Fractional source index
          vec2 sourceFracIndexRC = vec2(rc) * effectiveInputOverOutputRatioRC;

          // Compute the four integer indices.
          ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);
          ivec2 sourceCeilRC = ivec2(
            min(xShape.yz - 1.0, ceil(sourceFracIndexRC)));

          float topLeft = getValue(b, sourceFloorRC.x, sourceFloorRC.y, d);
          float bottomLeft = getValue(b, sourceCeilRC.x, sourceFloorRC.y, d);
          float topRight = getValue(b, sourceFloorRC.x, sourceCeilRC.y, d);
          float bottomRight = getValue(b, sourceCeilRC.x, sourceCeilRC.y, d);

          vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

          float top = topLeft + (topRight - topLeft) * fracRC.y;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
          float newValue = top + (bottom - top) * fracRC.x;

          setOutput(index, newValue);
        }
      }
    `;
  }
}
