/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

export class PoolWithFilterSizeEqualsOneProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'ivec2 pad, stride, dilation, convDims, filterDims;';
  uniformsWgsl =
      `pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>; convDims : vec2<u32>; filterDims : vec2<u32>;`;
  workGroupSize: [number, number, number] = [256, 1, 1];
  useWgsl: boolean;

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey = 'poolWithFilterSizeEqualsOne';
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const userCode = `
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
    return userCode;
  }

  getUserCodeWgsl(): string {
    const userCode = `
    ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
    fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let coords = getOutputCoords(globalId);
        let batch = coords[0];
        let d = coords[3];

        if (all(coords < uniforms.outShape)) {
          let xRCCorner = coords.yz * uniforms.stride;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          let value = getX(batch, xRCorner, xCCorner, d);
          setOutput(batch, coords[1], coords[2], d, value);
        }
      }
    `;
    return userCode;
  }
}
