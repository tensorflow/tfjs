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

import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ResizeBilinearProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [4, 4, 4];
  inputShape: [number, number, number, number];
  alignCorners: boolean;

  constructor(
      inputShape: [number, number, number, number], newHeight: number,
      newWidth: number, alignCorners: boolean) {
    this.outputShape = [inputShape[0], newHeight, newWidth, inputShape[3]];

    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.inputShape = inputShape;
    this.alignCorners = alignCorners;
    this.shaderKey = `resizeBilinear_${alignCorners}`;
  }

  getUserCode(): string {
    const adjustHeight = this.alignCorners && this.outputShape[1] > 1;
    const adjustWidth = this.alignCorners && this.outputShape[2] > 1;

    const userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        if (all(lessThan(coords, ${getShapeCoords(this.outputShape)}))) {
          int b = coords[0];
          int d = coords[3];
          ivec2 rc = coords.yz;

          vec2 effectiveInSize = vec2(
            ${
        adjustHeight ? `${this.inputShape[1]} - 1.0` : `${this.inputShape[1]}`},
            ${
        adjustWidth ? `${this.inputShape[2]} - 1.0` : `${this.inputShape[2]}`});

          vec2 effectiveOutSize = vec2(
            ${
        adjustHeight ? `${this.outputShape[1]} - 1.0` :
                       `${this.outputShape[1]}`},
            ${
        adjustWidth ? `${this.outputShape[2]} - 1.0` :
                      `${this.outputShape[2]}`});

          vec2 effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          vec2 sourceFracIndexRC = vec2(rc) * effectiveInputOverOutputRatioRC;

          // Compute the four integer indices.
          ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);
          ivec2 sourceCeilRC = ivec2(
            min(vec2(${this.inputShape[1]}, ${
        this.inputShape[2]}) - 1.0, ceil(sourceFracIndexRC)));

          float topLeft = getX(b, sourceFloorRC.x, sourceFloorRC.y, d);
          float bottomLeft = getX(b, sourceCeilRC.x, sourceFloorRC.y, d);
          float topRight = getX(b, sourceFloorRC.x, sourceCeilRC.y, d);
          float bottomRight = getX(b, sourceCeilRC.x, sourceCeilRC.y, d);

          vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

          float top = topLeft + (topRight - topLeft) * fracRC.y;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
          float newValue = top + (bottom - top) * fracRC.x;

          setOutput(b, coords[1], coords[2], d, newValue);
        }
      }
    `;
    return userCode;
  }
}
