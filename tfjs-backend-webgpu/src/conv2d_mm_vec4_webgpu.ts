/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {mapActivationToShaderProgram} from './activation_util';
import {makeMatMulPackedVec4Source} from './matmul_packed_vec4_webgpu';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch, tilesFitEvenlyIntoShape} from './webgpu_util';

export class Conv2DMMVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms =
      `filterDims : vec2<i32>; pad : vec2<i32>; stride : vec2<i32>; dilation : vec2<i32>;
      dimAOuter : i32; dimBOuter : i32; dimInner : i32;`;
  workGroupSize: [number, number, number] = [8, 8, 1];
  elementsPerThread: [number, number, number];
  isVec4 = true;
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  hasLeakyreluAlpha: boolean;
  tileAOuter: number;
  tileBOuter: number;
  tileInner: number;
  fitA: boolean;
  fitB: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false, hasLeakyreluAlpha = false) {
    this.outputShape = convInfo.outShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    // The first element in elementsPerThread must be 4.
    if (this.outputShape[1] === 1) {
      this.elementsPerThread = [4, 1, 1];
    } else {
      this.elementsPerThread = [4, 4, 1];
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.elementsPerThread);
    this.convInfo = convInfo;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.hasLeakyreluAlpha = hasLeakyreluAlpha;
    if (this.addBias) {
      this.variableNames.push('bias');
    }

    if (this.hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    if (this.hasLeakyreluAlpha) {
      this.variableNames.push('leakyreluAlpha');
    }

    this.tileAOuter = this.outputShape[1] === 1 ? 1 :
        this.workGroupSize[1] * this.elementsPerThread[1];
    this.tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    this.tileInner = this.tileBOuter;
    [this.fitA, this.fitB] = this.getShapeFit();
    this.shaderKey =`conv2DMMVec4_${this.activation}_${this.fitA}_${
        this.fitB}_${this.elementsPerThread}`;
  }

  getShapeFit(): boolean[] {
    const tileSizeA = [this.tileAOuter, this.tileInner];
    const tileSizeB = [this.tileInner, this.tileBOuter];
    const dimAOuter = this.outputShape[1] * this.outputShape[2];
    const dimBOuter = this.outputShape[3];
    const dimInner = this.convInfo.filterHeight * this.convInfo.filterWidth *
        this.convInfo.inChannels;
    return [
      tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]),
      tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter])
    ];
  }

  // index is used to avoid repeated definition error.
  getSampleAWithRemainder(index: number): string {
    return `let flatIndex${
        index} = getIndexFromCoords4D(coord, uniforms.xShape);
    let divBy4Remainder${index} = flatIndex${index} % 4;
    let divBy4Index${index} = flatIndex${index} / 4;
    let curData${index} = x.numbers[divBy4Index${index}];
    if (divBy4Remainder${index} == 0) {
      temp = curData${index};
    } else {
      // TODO: This could end up being a redundant load with another one in
      // the same shader invocation. Perhaps there's an opportunity for
      // optimization
      let nextData${index} = x.numbers[divBy4Index${index} + 1];
      if (divBy4Remainder${index} == 1) {
        temp = vec4<f32>(curData${index}.yzw, nextData${index}.x);
      } else if (divBy4Remainder${index} == 2) {
        temp = vec4<f32>(curData${index}.zw, nextData${index}.xy);
      } else if (divBy4Remainder${index} == 3) {
        temp = vec4<f32>(curData${index}.w, nextData${index}.xyz);
      }
    }
    `;
  }

  getUserCode(): string {
    const matMulSource = makeMatMulPackedVec4Source(this.elementsPerThread,
        this.tileAOuter, this.tileBOuter, this.tileInner);

    const remainder = this.convInfo.inChannels % 4;
    // Below code only applys to valid padding type.
    const remainderSnippet = remainder === 0 ?
        `// The bounds checking is always needed since we use it to pad zero for
          // the 'same' padding type.
          if (coordsInBounds4D(coord, uniforms.xShape)) {
            resData = x.numbers[getIndexFromCoords4D(coord, uniforms.xShape) / 4];
          } else {
            resData = vec4<f32>(0.0); }` :
        `var temp = vec4<f32>(0.0);
          ${this.getSampleAWithRemainder(1)}
          resData = temp;
          if (WCol == (uniforms.filterDims[1] - 1)) {
            coord = vec4<i32>(
              coord.x, coord.y + 1, coord.z + 1 - uniforms.filterDims[1], 0);
              ${this.getSampleAWithRemainder(2)}
            if (inChCoord == 0) {
              resData = vec4<f32>(resData.xyz, temp.x);
            } else if (inChCoord == 1) {
              resData = vec4<f32>(resData.xy, temp.xy);
            } else {
              resData = vec4<f32>(resData.x, temp.xyz);
            }
          }
          `;

    const readASnippet = `let outRow = r / uniforms.outShape[2];
        let outCol = r % uniforms.outShape[2];
        let WRow = c / (uniforms.filterDims[1] * uniforms.xShape[3]);
        let WCol = c / uniforms.xShape[3] % uniforms.filterDims[1];
        let inChCoord = c % uniforms.xShape[3];
        var coord = vec4<i32>(
            batch,
            outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0],
            outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1],
            inChCoord);
        var resData = vec4<f32>(0.0);
        ${remainderSnippet}
        return resData;`;

    const sampleA = this.fitA ?
        `${readASnippet}` :
        `if (r < uniforms.dimAOuter && c < uniforms.dimInner) {
          ${readASnippet}
         }
         return vec4<f32>(0.0);
        `;

    const sampleB = this.fitB ?
        `return W.numbers[row * uniforms.dimBOuter / 4 + col];` :
        `if(coordsInBounds2D(vec2<i32>(row, col * 4), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
           return W.numbers[row * uniforms.dimBOuter / 4 + col];
         }
         return vec4<f32>(0.0);
        `;
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, this.isVec4);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : vec4<f32>, outCoord : vec4<i32>) -> vec4<f32> {
          let b = getPreluActivationWeightsByOutputCoords(outCoord);
          ${activationOp}
        }`;
      } else if (this.hasLeakyreluAlpha) {
        activationSnippet = `fn activation(outCoord: vec4<f32>) -> vec4<f32> {
          let b = getLeakyreluAlphaByOutputCoords(outCoord);
          ${activationOp}
        }`;
        throw new Error('Leakyrelu is not supported.');
      } else {
        activationSnippet = `
        fn activation(a : vec4<f32>, outCoord : vec4<i32>) -> vec4<f32> {
          ${activationOp}
        }`;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet =
        this.addBias ? 'value = value + getBiasByOutputCoords(outCoord);' : '';

    const userCode = `
        ${activationSnippet}
        fn mm_readA(row : i32, col : i32, globalId : vec3<u32>) -> vec4<f32> {
          let r = row;
          let c = col * 4;
          var batch = i32(globalId.z);
          ${sampleA}
        }

        fn mm_readB(row : i32, col : i32, globalId : vec3<u32>) -> vec4<f32> {
          ${sampleB}
        }

        fn mm_write(row : i32, col : i32, valueInput : vec4<f32>, globalId : vec3<u32>) {
          var batch = i32(globalId.z);
          var value = valueInput;
          if (row < uniforms.dimAOuter && col * 4 < uniforms.dimBOuter)
          {
            let outCoord = vec4<i32>(
              batch,
              row / uniforms.outShape[2],
              row % uniforms.outShape[2],
              col * 4);
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutputAtCoords(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
              value);
          }
        }
        ${matMulSource}
      `;
    return userCode;
  }
}
