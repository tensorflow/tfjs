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

import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class Pool2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'ivec2 pad, stride, dilation, convDims, filterDims;';
  // TODO(jiajia.qin@intel.com): Dynamically choose different workGroupSize and
  // workPerThead for different output shapes.
  workGroupSize: [number, number, number] = [16, 16, 1];
  workPerThread = 4;
  poolType: 'max'|'avg';

  constructor(convInfo: backend_util.Conv2DInfo, poolType: 'max'|'avg') {
    this.outputShape = convInfo.outShape;

    this.dispatchLayout = {x: [0, 1, 2], y: [3]};

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [1, this.workPerThread, 1]);

    this.shaderKey = `pool2D_${poolType}`;
    this.poolType = poolType;
  }

  getUserCode(): string {
    let updateSnippet = `resultValue[i] = max(value, resultValue[i]);`;
    if (this.poolType === 'avg') {
      updateSnippet = `resultValue[i] += value; count[i] += 1.0;`;
    }

    let returnValue = `resultValue[i]`;
    if (this.poolType === 'avg') {
      returnValue = `resultValue[i] / count[i]`;
    }

    const userCode = `
      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= convDims.x) {
          return 0.0;
        }
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        if (all(lessThan(coords, ${getShapeCoords(this.outputShape)}))) {
          int batch = coords[0];
          ivec2 xRCCorner = coords.yz * stride - pad;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          float resultValue[${this.workPerThread}];
          float count[${this.workPerThread}];
          for (int i = 0; i < ${this.workPerThread}; i++)
          {
            resultValue[i] = 0.0;
            count[i] = 0.0;
          }

          for (int wR = 0; wR < filterDims.y; wR += dilation.y) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= convDims.y) {
              continue;
            }

            for (int wC = 0; wC < filterDims.x; wC += dilation.x) {
              int xC = xCCorner + wC * dilation.x;
              for (int i = 0; i < ${this.workPerThread}; i++)
              {
                int d = coords[3] * ${this.workPerThread} + i;
                if (d < ${this.outputShape[3]})
                {
                  float value = getValue(batch, xR, xC, d);
                  ${updateSnippet}
                }
                else
                {
                  break;
                }
              }
            }
          }
          for (int i = 0; i < ${this.workPerThread}; i++)
          {
            int d = coords[3] * ${this.workPerThread} + i;
            if (d < ${this.outputShape[3]})
            {
              setOutput(batch, coords[1], coords[2], d, ${returnValue});
            }
            else
            {
              break;
            }
          }
        }
      }
    `;
    return userCode;
  }
}
