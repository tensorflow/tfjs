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

import {backend_util, util} from '@tensorflow/tfjs-core';

import {makeMatMulPackedSource} from './matmul_packed_webgpu';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch, computeWorkGroupSizeForConv2d, computeWorkPerThreadForConv2d} from './webgpu_util';

export class Conv2DDerInputMMProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms =
      'filterDims : vec2<i32>, pads : vec2<i32>, stride : vec2<i32>, outBackprop : vec4<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,';
  workGroupSize: [number, number, number];
  elementsPerThread: [number, number, number];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.inShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    this.workGroupSize =
        computeWorkGroupSizeForConv2d(this.dispatchLayout, this.outputShape);
    this.elementsPerThread =
        computeWorkPerThreadForConv2d(this.dispatchLayout, this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.elementsPerThread);

    this.shaderKey = `conv2DDerInputMM_${this.elementsPerThread}`;
  }

  getUserCode(): string {
    const matMulSource =
        makeMatMulPackedSource(this.elementsPerThread, this.workGroupSize);

    const readASnippet = `
    let outRow = row / uniforms.outShape[2];
    let outCol = row % uniforms.outShape[2];

    let WRow = col / (uniforms.filterDims[1] * uniforms.outBackprop[3]);
    let WCol = col / uniforms.outBackprop[3] % uniforms.filterDims[1];
    let xR = f32(outRow - uniforms.pads[0] + WRow) / f32(uniforms.stride[0]);
    let xC = f32(outCol - uniforms.pads[1] + WCol) / f32(uniforms.stride[1]);
    if (xR < 0.0 || xR >= f32(uniforms.outBackprop[1]) || fract(xR) > 0.0) {
      return 0.0;
    }
    if (xC < 0.0 || xC >= f32(uniforms.outBackprop[2]) || fract(xC) > 0.0) {
      return 0.0;
    }
    let coord = vec4<i32>(
        batch,
        i32(xR),
        i32(xC),
        col % uniforms.outBackprop[3]);
    return x[getIndexFromCoords4D(coord, uniforms.xShape)];`;

    const sampleA = `if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
      ${readASnippet}
    }
    return 0.0;`;

    const userCode = `
    fn mm_readA(row : i32, col : i32, globalId : vec3<u32>) -> f32 {
      var batch = i32(globalId.z);
      ${sampleA}
    }

    fn mm_readB(row : i32, col : i32, globalId : vec3<u32>) -> f32 {
      let coordX = uniforms.filterDims.x - 1 -
          row / (uniforms.filterDims[1] * uniforms.outBackprop[3]);
      let coordY = uniforms.filterDims.y - 1 -
          (row / uniforms.outBackprop[3]) % uniforms.filterDims[1];
      if (row < uniforms.dimInner && col < uniforms.dimBOuter &&
          coordX >= 0 && coordY >= 0) {
        let coord = vec4<i32>(coordX, coordY, col,
            row % uniforms.outBackprop[3]);
        return W[getIndexFromCoords4D(coord, uniforms.wShape)];
      }
      return 0.0;
    }

    fn mm_write(row : i32, col : i32, valueInput : f32, globalId : vec3<u32>) {
      if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
      {
      var batch = i32(globalId.z);
      var value = valueInput;
      let outCoord = vec4<i32>(
          batch,
          row / uniforms.outShape[2],
          row % uniforms.outShape[2],
          col);
      result[getIndexFromCoords4D(outCoord, uniforms.outShape)] = value;
      }
    }

    ${matMulSource}
  `;
    return userCode;
  }
}
