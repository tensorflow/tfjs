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
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class Conv2DDerInputProgram implements WebGPUProgram {
  variableNames = ['dy', 'W'];
  uniforms =
      'filterDims : vec2<i32>, pads : vec2<i32>, stride : vec2<i32>, outBackprop : vec4<i32>,';
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  isChannelsLast: boolean;
  size = true;

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.inShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.shaderKey = `conv2DDerInput_${this.isChannelsLast}`;
  }

  getUserCode(): string {
    const rowDim = this.isChannelsLast ? 1 : 2;
    const colDim = this.isChannelsLast ? 2 : 3;
    const channelDim = this.isChannelsLast ? 3 : 1;
    return `
    ${main('index')} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d1 = coords[${channelDim}];

        let dyCorner = vec2<i32>(coords[${rowDim}], coords[${
        colDim}]) - uniforms.pads;
        let dyRCorner = dyCorner.x;
        let dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + 1) {
          let dyR = (f32(dyRCorner) + f32(wR)) / f32(uniforms.stride.x);
          let wRPerm = uniforms.filterDims.x - 1 - wR;
          if (dyR < 0.0 || dyR >= f32(uniforms.outBackprop[1]) || fract(dyR) > 0.0 ||
              wRPerm < 0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + 1) {
            let dyC = (f32(dyCCorner) + f32(wC)) / f32(uniforms.stride.y);
            let wCPerm = uniforms.filterDims.y - 1 - wC;
            if (dyC < 0.0 || dyC >= f32(uniforms.outBackprop[2]) ||
                fract(dyC) > 0.0 || wCPerm < 0) {
              continue;
            }
            let idyC = i32(dyC);

            for (var d2 = 0; d2 < uniforms.outBackprop[3]; d2 = d2 + 1) {
              if (${this.isChannelsLast}) {
                let xValue = getDy(batch, idyR, idyC, d2);
                let wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd = dotProd + xValue * wValue;
              } else {
                let xValue = getDy(batch, d2, idyR, idyC);
                let wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd = dotProd + xValue * wValue;
              }

            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `;
  }
}

export class Conv2DDerFilterProgram implements WebGPUProgram {
  variableNames = ['x', 'dy'];
  uniforms =
      'pad : vec2<i32>, stride : vec2<i32>, batchSize : i32, outHeight : i32, outWidth : i32, inHeight : i32, inWidth : i32,';
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  isChannelsLast: boolean;
  size = true;

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.filterShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.shaderKey = `conv2DDerFilter_${this.isChannelsLast}`;
  }

  getUserCode(): string {
    return `
    ${main('index')} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let wR = coords[0];
        let wC = coords[1];
        let d1 = coords[2];
        let d2 = coords[3];

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var b = 0; b < uniforms.batchSize; b = b + 1) {
          for (var yR = 0; yR < uniforms.outHeight; yR = yR + 1) {
            let xR = wR + yR * uniforms.stride[0] - uniforms.pad[0];
            if (xR < 0 || xR >= uniforms.inHeight) {
              continue;
            }

            for (var yC = 0; yC < uniforms.outWidth; yC = yC + 1) {
              let xC = wC + yC * uniforms.stride[1] - uniforms.pad[1];

              if (xC < 0 || xC >= uniforms.inWidth) {
                continue;
              }

              if (${this.isChannelsLast}) {
                let dyValue = getDy(b, yR, yC, d2);
                let xValue = getX(b, xR, xC, d1);
                dotProd = dotProd + xValue * dyValue;
              } else {
                let dyValue = getDy(b, d2, yR, yC);
                let xValue = getX(b, d1, xR, xC);
                dotProd = dotProd + xValue * dyValue;
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `;
  }
}
