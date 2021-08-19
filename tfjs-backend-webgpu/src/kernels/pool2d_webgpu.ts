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

import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class Pool2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'ivec2 pad, stride, dilation, convDims, filterDims;';
  uniformsWgsl =
      `pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>; convDims : vec2<u32>; filterDims : vec2<u32>;`;
  // TODO(jiajia.qin@intel.com): Dynamically choose different workGroupSize for
  // different output shapes.
  workGroupSize: [number, number, number] = [128, 1, 1];
  poolType: 'max'|'avg';
  useWgsl: boolean;

  constructor(convInfo: backend_util.Conv2DInfo, poolType: 'max'|'avg') {
    this.outputShape = convInfo.outShape;

    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey = `pool2D_${poolType}`;
    this.poolType = poolType;
    this.useWgsl = getUseWgsl();
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

  getUserCodeWgsl(): string {
    let updateSnippet = `resultValue = max(value, resultValue);`;
    if (this.poolType === 'avg') {
      updateSnippet = `resultValue = resultValue + value; count = count + 1.0;`;
    }

    let returnValue = `resultValue`;
    if (this.poolType === 'avg') {
      returnValue = `resultValue / count`;
    }

    const userCode = `
    ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
    fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let coords = getOutputCoords(globalId);
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          let batch = coords[0];
          let xRCCorner = vec2<i32>(coords.yz * uniforms.stride - uniforms.pad);
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          var resultValue = ${
        this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};
          var count = 0.0;

          for (var wR = 0u; wR < uniforms.filterDims.x; wR = wR + uniforms.dilation.x) {
            let xR = xRCorner + i32(wR);

            if (xR < 0 || xR >= i32(uniforms.convDims.x)) {
              continue;
            }

            for (var wC = 0u; wC < uniforms.filterDims.y; wC = wC + uniforms.dilation.y) {
              let xC = xCCorner + i32(wC);
              if (xC < 0 || xC >= i32(uniforms.convDims.y)) {
                continue;
              }

              let value = getX(batch, u32(xR), u32(xC), coords[3]);
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
