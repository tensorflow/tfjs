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

import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class ResizeBilinearProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  alignCorners: boolean;
  useWgsl: boolean;

  constructor(
      inputShape: [number, number, number, number], newHeight: number,
      newWidth: number, alignCorners: boolean) {
    this.outputShape = [inputShape[0], newHeight, newWidth, inputShape[3]];

    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.alignCorners = alignCorners;
    this.shaderKey = `resizeBilinear_${alignCorners}_${
        this.outputShape[1] > 1}_${this.outputShape[2] > 1}`;
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const adjustHeight = this.alignCorners && this.outputShape[1] > 1;
    const adjustWidth = this.alignCorners && this.outputShape[2] > 1;

    const userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        if (all(lessThan(coords, outShape))) {
          int b = coords[0];
          int d = coords[3];
          ivec2 rc = coords.yz;

          vec2 effectiveInSize = vec2(
            ${adjustHeight ? `xShape.y - 1.0` : `xShape.y`},
            ${adjustWidth ? `xShape.z - 1.0` : `xShape.z`});

          vec2 effectiveOutSize = vec2(
            ${adjustHeight ? `outShape.y - 1.0` : `outShape.y`},
            ${adjustWidth ? `outShape.z - 1.0` : `outShape.z`});

          vec2 effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          vec2 sourceFracIndexRC = vec2(rc) * effectiveInputOverOutputRatioRC;

          // Compute the four integer indices.
          ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);
          ivec2 sourceCeilRC = ivec2(
            min(xShape.yz - 1.0, ceil(sourceFracIndexRC)));

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
  getUserCodeWgsl(): string {
    const adjustHeight = this.alignCorners && this.outputShape[1] > 1;
    const adjustWidth = this.alignCorners && this.outputShape[2] > 1;

    const userCode = `
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let coords = getOutputCoords(globalId);
        if (all(coords < uniforms.outShape)) {
          let b = coords[0];
          let d = coords[3];
          let rc = coords.yz;

          let effectiveInSize = vec2<f32>(
            ${
        adjustHeight ? `f32(uniforms.xShape.y) - 1.0` :
                       `f32(uniforms.xShape.y)`},
            ${
        adjustWidth ? `f32(uniforms.xShape.z) - 1.0` :
                      `f32(uniforms.xShape.z)`});

          let effectiveOutSize = vec2<f32>(
            ${
        adjustHeight ? `f32(uniforms.outShape.y) - 1.0` :
                       `f32(uniforms.outShape.y)`},
            ${
        adjustWidth ? `f32(uniforms.outShape.z) - 1.0` :
                      `f32(uniforms.outShape.z)`});

          let effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          let sourceFracIndexRC = vec2<f32>(rc) * effectiveInputOverOutputRatioRC;

          // Compute the four integer indices.
          let sourceFloorRC = vec2<u32>(sourceFracIndexRC);
          let sourceCeilRC = vec2<u32>(
            min(vec2<f32>(uniforms.xShape.yz) - vec2<f32>(1.0), ceil(sourceFracIndexRC)));

          let topLeft = getX(b, sourceFloorRC.x, sourceFloorRC.y, d);
          let bottomLeft = getX(b, sourceCeilRC.x, sourceFloorRC.y, d);
          let topRight = getX(b, sourceFloorRC.x, sourceCeilRC.y, d);
          let bottomRight = getX(b, sourceCeilRC.x, sourceCeilRC.y, d);

          let fracRC = sourceFracIndexRC - vec2<f32>(sourceFloorRC);

          let top = topLeft + (topRight - topLeft) * fracRC.y;
          let bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
          let newValue = top + (bottom - top) * fracRC.x;

          setOutput(b, coords[1], coords[2], d, newValue);
        }
      }
    `;
    return userCode;
  }
}
