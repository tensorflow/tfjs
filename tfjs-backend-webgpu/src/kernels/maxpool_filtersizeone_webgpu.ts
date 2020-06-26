/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {backend_util} from '@tensorflow/tfjs-core';

import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class MaxPoolWithFilterSizeEqualsOneProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'ivec2 pad, stride, dilation, convDims, filterDims;';
  workGroupSize: [number, number, number] = [4, 4, 4];
  needsShapesUniforms = true;

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [0, 1], y: [2], z: [3]};

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        if (all(lessThan(coords, outShape))) {
          ivec2 xRCCorner = coords.yz * stride;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          float value = getX(batch, xRCorner, xCCorner, d);
          setOutput(batch, coords[1], coords[2], d, value);
        }
      }
    `;
    this.shaderKey = 'maxpoolv2';
  }
}
