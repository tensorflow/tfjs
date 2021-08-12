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

import {backend_util} from '@tensorflow/tfjs-core';

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class Pool2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'ivec2 pad, stride, dilation, convDims, filterDims;';
  // TODO(jiajia.qin@intel.com): Dynamically choose different workGroupSize for
  // different output shapes.
  workGroupSize: [number, number, number] = [128, 1, 1];
  poolType: 'max'|'avg';

  constructor(convInfo: backend_util.Conv2DInfo, poolType: 'max'|'avg') {
    this.outputShape = convInfo.outShape;

    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey = `pool2D_${poolType}`;
    this.poolType = poolType;
  }

  getUserCode(): string {
    let updateSnippet = `resultValue = max(value, resultValue);`;
    if (this.poolType === 'avg') {
      updateSnippet = `resultValue += value; count += 1.0;`;
    }

    let returnValue = `resultValue`;
    if (this.poolType === 'avg') {
      returnValue = `resultValue / count`;
    }

    const userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        if (coordsInBounds(coords, outShape)) {
          int batch = coords[0];
          ivec2 xRCCorner = coords.yz * stride - pad;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          float resultValue = ${
        this.poolType === 'avg' ? '0.0' : '-1.0 / 1e-20'};
          float count = 0.0;

          for (int wR = 0; wR < filterDims.x; wR += dilation.x) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= convDims.x) {
              continue;
            }

            for (int wC = 0; wC < filterDims.y; wC += dilation.y) {
              int xC = xCCorner + wC;
              if (xC < 0 || xC >= convDims.y) {
                continue;
              }

              float value = getX(batch, xR, xC, coords[3]);
              ${updateSnippet}
            }
          }

          setOutput(batch, coords[1], coords[2], coords[3], ${returnValue});
        }
      }
    `;
    return userCode;
  }
}
