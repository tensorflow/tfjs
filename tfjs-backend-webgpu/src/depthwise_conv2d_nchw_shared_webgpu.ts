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

import {util} from '@tensorflow/tfjs-core';
import {backend_util} from '@tensorflow/tfjs-core';

import {mapActivationToShaderProgram} from './activation_util';
import {getWorkGroupSizeString, WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export class DepthwiseConv2DNCHWSharedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = `pad : vec2<i32>, stride : vec2<i32>, dilation : vec2<i32>,
      inDims : vec2<i32>, filterHeight : i32, filterWidth : i32,
      channelMul : i32,`;
  workGroupSize: [number, number, number] = [8, 8, 1];
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
        this.dispatchLayout, this.outputShape, this.workGroupSize);

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
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation, false);
      if (this.hasPreluActivation) {
        activationSnippet =
            `fn activation(a : f32, outCoord : vec4<i32>) -> f32 {
          let b = getPreluActivationWeightsByOutputCoords(outCoord);
          ${activationOp}
        }`;
      } else {
        activationSnippet = `
          fn activation(a : f32, outCoord : vec4<i32>) -> f32 {
            ${activationOp}
          }
        `;
      }

      applyActivationSnippet = `dotProd = activation(dotProd, coords);`;
    }

    const addBiasSnippet = this.addBias ?
        'dotProd = dotProd + getBiasByOutputCoords(coords);' :
        '';

    const filterSize = this.filterWidth * this.filterHeight;
    const workGroupSize =
        this.workGroupSize[0] * this.workGroupSize[1] * this.workGroupSize[2];
    util.assert(
        filterSize <= workGroupSize,
        () => `The filterSize ${
            filterSize} must be less that or equal to workgroup size ${
            workGroupSize}`);
    const userCode = `
      ${activationSnippet}

      var<workgroup> mm_Asub : array<array<f32, 16>, 16>;
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

      fn writeResult(batch : i32, channel : i32, row : i32, col : i32,
          value : f32) {
        let coord = vec4<i32>(batch, channel, row, col);
        if (coordsInBounds4D(coord, uniforms.outShape)) {
          setOutputAtCoords(batch, channel, row, col, value);
        }
      }

      ${getWorkGroupSizeString()}
      fn main(@builtin(local_invocation_id) LocalId : vec3<u32>,
              @builtin(global_invocation_id) GlobalId : vec3<u32>,
              @builtin(local_invocation_index) LocalIndex: u32,
              @builtin(num_workgroups) NumWorkgroups: vec3<u32>) {
        localId = LocalId;
        globalId = GlobalId;
        let localIndex = i32(LocalIndex);
        numWorkgroups = NumWorkgroups;
        let coords = getOutputCoords();
        let batch = coords[0];
        let xRCCorner = vec2<i32>(coords.zw) * uniforms.stride - uniforms.pad;
        let d2 = coords[1];
        let d1 = d2 / uniforms.channelMul;
        let q = d2 - d1 * uniforms.channelMul;

        let inputRowStart = xRCCorner.x;
        let inputColStart = xRCCorner.y;

        let localRow = i32(localId.y);
        let localCol = i32(localId.x);

        // Load one tile of X into local memory.
        mm_Asub[localRow][localCol] = readX(batch, d1, inputRowStart, inputColStart);
        mm_Asub[localRow][8 + localCol] =
        readX(batch, d1, inputRowStart, inputColStart + 8);
        mm_Asub[8 + localRow][localCol] =
        readX(batch, d1, inputRowStart + 8, inputColStart);
        mm_Asub[8 + localRow][8 + localCol] =
          readX(batch, d1, inputRowStart + 8, inputColStart + 8);

        // Load one tile of W into local memory.
        if (localIndex < ${filterSize})
        {
          let wRow = localIndex / ${this.filterWidth};
          let wCol = localIndex % ${this.filterWidth};
          mm_Bsub[wRow][wCol] = getW(wRow, wCol, d1, q);
        }

        workgroupBarrier();

        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterHeight; wR = wR + 1) {
          for (var wC = 0; wC < uniforms.filterWidth; wC = wC + 1) {
            let xVal = mm_Asub[localRow + wR][localCol + wC];
            let wVal = mm_Bsub[wR][wC];
            dotProd = fma(xVal, wVal, dotProd);
          }
        }

        ${addBiasSnippet}
        ${applyActivationSnippet}
        writeResult(coords[0], coords[1], coords[2], coords[3], dotProd);
      }
    `;
    return userCode;
  }
}
