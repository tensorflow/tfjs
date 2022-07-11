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
import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class Pool2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms =
      `stride : vec2<i32>, pad : vec2<i32>, dilation : vec2<i32>, convDims : vec2<i32>, filterDims : vec2<i32>,`;
  // TODO(jiajia.qin@intel.com): Dynamically choose different workGroupSize for
  // different output shapes.
  workGroupSize: [number, number, number] = [128, 1, 1];
  poolType: 'max'|'avg';
  size = true;

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
      updateSnippet = `resultValue = resultValue + value; count = count + 1.0;`;
    }

    let returnValue = `resultValue`;
    if (this.poolType === 'avg') {
      returnValue = `resultValue / count`;
    }

    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let xRCCorner = vec2<i32>(coords.yz) * uniforms.stride - uniforms.pad;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          var resultValue = ${
        this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};
          var count = 0.0;

          for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + uniforms.dilation.x) {
            let xR = xRCorner + wR;

            if (xR < 0 || xR >= uniforms.convDims.x) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + uniforms.dilation.y) {
              let xC = xCCorner + wC;
              if (xC < 0 || xC >= uniforms.convDims.y) {
                continue;
              }

              let value = getX(batch, xR, xC, coords[3]);
              ${updateSnippet}
            }
          }

          setOutputAtIndex(index, ${returnValue});
        }
      }
    `;
    return userCode;
  }
}
