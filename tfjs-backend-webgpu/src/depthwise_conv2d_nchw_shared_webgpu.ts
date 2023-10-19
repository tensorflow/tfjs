/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {activationFnSnippet, biasActivationSnippet} from './activation_util';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export class DepthwiseConv2DNCHWSharedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = `pads : vec2<i32>, inDims : vec2<i32>,`;
  workgroupSize: [number, number, number] = [16, 16, 1];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivation: boolean;
  filterHeight: number;
  filterWidth: number;

  constructor(
      outputShape: number[], filterHeight: number, filterWidth: number,
      addBias = false, activation: backend_util.Activation = null,
      hasPreluActivation = false) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [3], y: [2], z: [0, 1]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    if (addBias) {
      this.variableNames.push('bias');
    }
    if (hasPreluActivation) {
      this.variableNames.push('preluActivationWeights');
    }

    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivation = hasPreluActivation;
    this.filterHeight = filterHeight;
    this.filterWidth = filterWidth;
    this.shaderKey = `depthwiseNCHW_${this.activation}_${this.filterHeight}_${
        this.filterWidth}`;
  }

  getUserCode(): string {
    const filterSize = this.filterWidth * this.filterHeight;
    const flatWorkgroupSize =
        this.workgroupSize[0] * this.workgroupSize[1] * this.workgroupSize[2];
    const tileAHeight = this.workgroupSize[1] + this.filterHeight - 1;
    const tileAWidth = this.workgroupSize[0] + this.filterWidth - 1;

    const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivation, false, 4)}

      var<workgroup> mm_Asub : array<array<f32, ${tileAWidth}>, ${tileAHeight}>;
      var<workgroup> mm_Bsub : array<array<f32, ${this.filterWidth}>, ${
        this.filterHeight}>;
      fn readX(batch : i32, channel : i32, row : i32, col : i32) -> f32 {
        var value = 0.0;
        if (row >=0 && row < uniforms.inDims[0] && col >=0 && col < uniforms.inDims[1])
        {
          value = getX(batch, channel, row, col);
        }
        return value;
      }

      ${main()} {
        let coords = getOutputCoords();
        let batch = coords[0];
        let xRCCorner = vec2<i32>(coords.zw) - uniforms.pads;
        let channelMul = uniforms.wShape[3];
        let d1 = coords[1] / channelMul;
        let q = coords[1] % channelMul;

        let inputRowStart = xRCCorner.x;
        let inputColStart = xRCCorner.y;

        let localRow = i32(localId.y);
        let localCol = i32(localId.x);

        // Load one tile of X into local memory.
        for (var inputRow = localRow; inputRow < ${
        tileAHeight}; inputRow = inputRow + ${this.workgroupSize[1]}) {
          for (var inputCol = localCol; inputCol < ${
        tileAWidth}; inputCol = inputCol + ${this.workgroupSize[0]}) {
            let rowOffset = inputRow - localRow;
            let colOffset = inputCol - localCol;
            mm_Asub[inputRow][inputCol] = readX(batch, d1, inputRowStart + rowOffset, inputColStart + colOffset);
          }
        }

        // Load one tile of W into local memory.
        var wIndex = i32(localIndex);
        ${
        filterSize < flatWorkgroupSize ?
            `if (wIndex < ${filterSize})` :
            `for(; wIndex < ${filterSize}; wIndex = wIndex + ${
                flatWorkgroupSize})`}

        {
          let wRow = wIndex / ${this.filterWidth};
          let wCol = wIndex % ${this.filterWidth};
          mm_Bsub[wRow][wCol] = getW(wRow, wCol, d1, q);
        }

        workgroupBarrier();

        var value = 0.0;
        for (var wR = 0; wR < ${this.filterHeight}; wR = wR + 1) {
          for (var wC = 0; wC < ${this.filterWidth}; wC = wC + 1) {
            let xVal = mm_Asub[localRow + wR][localCol + wC];
            let wVal = mm_Bsub[wR][wC];
            value = fma(xVal, wVal, value);
          }
        }
        ${biasActivationSnippet(this.addBias, this.activation)}
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
    `;
    return userCode;
  }
}
