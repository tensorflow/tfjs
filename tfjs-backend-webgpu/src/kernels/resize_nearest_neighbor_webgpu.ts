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

import {getGlobalIndexString, getMainHeaderString} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class ResizeNearestNeighborProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  alignCorners: boolean;
  halfPixelCenters: boolean;

  constructor(
      inputShape: [number, number, number, number], newHeight: number,
      newWidth: number, alignCorners: boolean, halfPixelCenters: boolean) {
    this.outputShape = [inputShape[0], newHeight, newWidth, inputShape[3]];

    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.alignCorners = alignCorners;
    this.halfPixelCenters = halfPixelCenters;
    this.shaderKey =
        `resizeNearest_${alignCorners}_${this.outputShape[1] > 1}_${
            this.outputShape[2] > 1}_${halfPixelCenters}`;
  }

  getUserCode(): string {
    // When align corners is false, we rounds the value with floor.
    const roundBase = this.alignCorners ? '0.5' : '0.0';
    let sourceFracIndexRC: string;
    if (this.halfPixelCenters) {
      sourceFracIndexRC =
          `max((vec2<f32>(rc) + vec2<f32>(0.5)) * effectiveInputOverOutputRatioRC` +
          `, vec2<f32>(0.0))`;
    } else {
      sourceFracIndexRC = `vec2<f32>(rc) * effectiveInputOverOutputRatioRC`;
    }

    const adjustHeight = this.alignCorners && this.outputShape[1] > 1;
    const adjustWidth = this.alignCorners && this.outputShape[2] > 1;

    const userCode = `
      ${getMainHeaderString()} {
        ${getGlobalIndexString()}
        let coords = getOutputCoords(globalId, index);
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
          let sourceFracIndexRC = ${sourceFracIndexRC};

          // Compute the coordinators of nearest neighbor point.
          let inputShapeRC = vec2<f32>(f32(uniforms.xShape.y), f32(uniforms.xShape.z));
          let sourceNearestRC = vec2<i32>(
            min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${roundBase})));
          let newValue = getX(b, sourceNearestRC.x, sourceNearestRC.y, d);

          setOutput(b, coords[1], coords[2], d, newValue);
        }
      }
    `;
    return userCode;
  }
}
