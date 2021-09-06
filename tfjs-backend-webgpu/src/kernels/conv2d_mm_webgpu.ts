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

import {backend_util, util} from '@tensorflow/tfjs-core';

import {computeDispatch, computeWorkGroupSizeForConv2d, computeWorkPerThreadForConv2d, tilesFitEvenlyIntoShape} from '../webgpu_util';
import {mapActivationToShaderProgram} from './activation_util';

import {makeMatMulPackedSource, makeMatMulPackedSourceWgsl} from './matmul_packed_webgpu';
import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class Conv2DMMProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  uniformsWgsl =
      `filterDims : vec2<i32>; pad : vec2<i32>; stride : vec2<i32>; dilation : vec2<i32>; dimAOuter : i32; dimBOuter : i32; dimInner : i32;`;
  workGroupSize: [number, number, number];
  elementsPerThread: [number, number, number];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  fitA: boolean;
  fitB: boolean;
  useWgsl: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;

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

    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }
    this.convInfo = convInfo;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    [this.fitA, this.fitB] = this.getShapeFit();
    this.shaderKey = `conv2DMM_${this.elementsPerThread}_${this.activation}_${
        this.fitA}_${this.fitB}`;
    this.useWgsl = getUseWgsl();
  }

  getShapeFit(): boolean[] {
    const tileAOuter = this.workGroupSize[1] * this.elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    util.assert(
        tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0,
        () =>
            // tslint:disable-next-line: max-line-length
        'tileInner must be multiple of workgroupsize.x and workgroupsize.y');
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const dimAOuter = this.outputShape[1] * this.outputShape[2];
    const dimBOuter = this.outputShape[3];
    const dimInner = this.convInfo.filterHeight * this.convInfo.filterWidth *
        this.convInfo.inChannels;

    return [
      tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]),
      tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter])
    ];
  }

  getUserCode(): string {
    const matMulSource = makeMatMulPackedSource(this.elementsPerThread);

    const readASnippet = `
    int outRow = row / outShape[2];
    int outCol = row % outShape[2];

    int WRow = col / (filterDims[1] * xShape[3]);
    int WCol = (col / xShape[3]) % filterDims[1];

    ivec4 coord = ivec4(
        batch,
        outRow * stride[0] + dilation[0] * WRow - pad[0],
        outCol * stride[1] + dilation[1] * WCol - pad[1],
        col % xShape[3]);
    // The bounds checking is always needed since we use it to pad zero for the
    // 'same' padding type.
    return coordsInBounds(coord, xShape) ? x[getFlatIndex(coord, xShape)] : 0;`;

    const sampleA = this.fitA ? `${readASnippet}` :
                                `if (row < dimAOuter && col < dimInner) {
      ${readASnippet}
    } else {
      return 0;
    }`;

    const sampleB = this.fitB ?
        `W[row * dimBOuter + col]` :
        `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
        W[row * dimBOuter + col] : 0`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation);
      if (this.hasPreluActivationWeights) {
        activationSnippet = `float activation(float a, ivec4 outCoord) {
                  float b = getPreluActivationWeightsAtOutCoords(outCoord);
                  ${activationOp}
                }`;
      } else {
        activationSnippet = `
                  float activation(float a, ivec4 outCoord) {
                    ${activationOp}
                  }
                `;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
    ${activationSnippet}
    ${matMulSource}

    int batch;
    int dimAOuter = outShape[1] * outShape[2];
    int dimBOuter = outShape[3];
    int dimInner = filterDims[0] * filterDims[1] * xShape[3];
    float mm_readA(int row, int col) {
      ${sampleA}
    }

    float mm_readB(int row, int col) {
      return ${sampleB};
    }

    void mm_write(int row, int col, float value) {
      ivec4 outCoord = ivec4(
          batch,
          row / outShape[2],
          row % outShape[2],
          col);
      ${addBiasSnippet}
      ${applyActivationSnippet}
      result[getFlatIndex(outCoord, outShape)] = value;
    }

    void main() {
      batch = int(gl_GlobalInvocationID.z);

      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
  `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const matMulSource =
        makeMatMulPackedSourceWgsl(this.elementsPerThread, this.workGroupSize);

    const readASnippet = `
    let outRow = row / uniforms.outShape[2];
    let outCol = row % uniforms.outShape[2];

    let WRow = col / (uniforms.filterDims[1] * uniforms.xShape[3]);
    let WCol = col / uniforms.xShape[3] % uniforms.filterDims[1];
    let coordRow = i32(outRow) * uniforms.stride[0] + uniforms.dilation[0] * i32(WRow) - uniforms.pad[0];
    if (coordRow < 0) {
      return 0.0;
    }
    let coordCol = i32(outCol) * uniforms.stride[1] + uniforms.dilation[1] * i32(WCol) - uniforms.pad[1];
    if (coordCol < 0) {
      return 0.0;
    }
    let coord = vec4<i32>(
        batch,
        coordRow,
        coordCol,
        col % uniforms.xShape[3]);
    // The bounds checking is always needed since we use it to pad zero for the
    // 'same' padding type.
    if(coordsInBounds4D(coord, uniforms.xShape)) {
      return x.numbers[getFlatIndex4D(coord, uniforms.xShape)];
    }
    return 0.0;`;

    const sampleA = this.fitA ?
        `${readASnippet}` :
        `if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
      ${readASnippet}
    }
    return 0.0;
    `;

    const sampleB = this.fitB ?
        `return W.numbers[row * uniforms.dimBOuter + col];` :
        `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
           return W.numbers[row * uniforms.dimBOuter + col];
	 }
	 return 0.0;
	 `;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, false, this.useWgsl);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a: f32, outCoord : vec4<i32>) -> f32 {
                  let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
                  ${activationOp}
                }`;
      } else {
        activationSnippet = `
                  fn activation(a : f32, outCoord : vec4<i32>) -> f32 {
                    ${activationOp}
                  }
                `;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet = this.addBias ?
        'value = value + getBiasAtOutCoordsByCoords(outCoord);' :
        '';

    const userCode = `
    ${activationSnippet}
    fn mm_readA(row : i32, col : i32, globalId : vec3<i32>) -> f32 {
      var batch = globalId.z;
      ${sampleA}
    }

    fn mm_readB(row : i32, col : i32, globalId : vec3<i32>) -> f32 {
      ${sampleB}
    }

    fn mm_write(row : i32, col : i32, valueInput : f32, globalId : vec3<i32>) {
      var batch = globalId.z;
      var value = valueInput;
      let outCoord = vec4<i32>(
          batch,
          row / uniforms.outShape[2],
          row % uniforms.outShape[2],
          col);
      ${addBiasSnippet}
      ${applyActivationSnippet}
      result.numbers[getFlatIndex4D(outCoord, uniforms.outShape)] = value;
    }
    ${matMulSource}
  `;
    return userCode;
  }
}
