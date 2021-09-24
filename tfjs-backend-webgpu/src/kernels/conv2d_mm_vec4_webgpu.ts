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

import {computeDispatch, tilesFitEvenlyIntoShape} from '../webgpu_util';
import {mapActivationToShaderProgram} from './activation_util';

import {makeMatMulPackedVec4Source, makeMatMulPackedVec4SourceWgsl} from './matmul_packed_vec4_webgpu';
import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class Conv2DMMVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  uniformsWgsl =
      `filterDims : vec2<u32>; pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>;
      dimAOuter : u32; dimBOuter : u32; dimInner : u32;`;
  workGroupSize: [number, number, number];
  useWgsl: boolean;
  isVec4 = true;
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  hasLeakyreluAlpha: boolean;
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
    this.workGroupSize = [8, 8, 1];
    const elementsPerThread: [number, number, number] = [4, 4, 1];
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        elementsPerThread);
    this.convInfo = convInfo;
    this.addBias = addBias;
    this.useWgsl = getUseWgsl();
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

    [this.fitA, this.fitB] = this.getShapeFit(elementsPerThread);
    this.shaderKey =
        `conv2DMMVec4_${this.activation}_${this.fitA}_${this.fitB}`;
  }

  getShapeFit(elementsPerThread: [number, number, number]): boolean[] {
    const tileAOuter = this.workGroupSize[1] * elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * elementsPerThread[0];
    const tileInner = tileBOuter;

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
    const elementsPerThread: [number, number, number] = [4, 4, 1];
    const matMulSource = makeMatMulPackedVec4Source(elementsPerThread);

    // Below code only applys to valid padding type.
    const sampleAWithRemainder = `int flatIndex = getFlatIndex(coord, xShape);
        int divBy4Remainder = flatIndex % 4;
        int divBy4Index = flatIndex / 4;
        vec4 curData = x[divBy4Index];
        if (divBy4Remainder == 0) {
          temp = curData;
        } else {
          // TODO: This could end up being a redundant load with another one in
          // the same shader invocation. Perhaps there's an opportunity for
          // optimization
          vec4 nextData = x[divBy4Index + 1];
          if (divBy4Remainder == 1) {
            temp = vec4(curData.yzw, nextData.x);
          } else if (divBy4Remainder == 2) {
            temp = vec4(curData.zw, nextData.xy);
          } else if (divBy4Remainder == 3) {
            temp = vec4(curData.w, nextData.xyz);
          }
        }
        `;

    const remainder = this.convInfo.inChannels % 4;
    const remainderSnippet = remainder === 0 ?
        `// The bounds checking is always needed since we use it to pad zero for
        // the 'same' padding type.
        resData = coordsInBounds(coord, xShape) ?
        x[getFlatIndex(coord, xShape) / 4] : vec4(0.0);` :
        `vec4 temp = vec4(0.0);
        ${sampleAWithRemainder}
        resData = temp;
        if (WCol == (filterDims[1] - 1)) {
          coord = ivec4(
            coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
          ${sampleAWithRemainder}
          if (inChCoord == 0) {
            resData = vec4(resData.xyz, temp.x);
          } else if (inChCoord == 1) {
            resData = vec4(resData.xy, temp.xy);
          } else {
            resData = vec4(resData.x, temp.xyz);
          }
        }
        `;

    const readASnippet = `int outRow = r / outShape[2];
        int outCol = r % outShape[2];
        int WRow = c / (filterDims[1] * xShape[3]);
        int WCol = (c / xShape[3]) % filterDims[1];
        int inChCoord = c % xShape[3];
        ivec4 coord = ivec4(
            batch,
            outRow * stride[0] + dilation[0] * WRow - pad[0],
            outCol * stride[1] + dilation[1] * WCol - pad[1],
            inChCoord);
        vec4 resData = vec4(0.0);
        ${remainderSnippet}
        return resData;`;

    const sampleA =
        this.fitA ? `${readASnippet}` : `if (r < dimAOuter && c < dimInner) {
          ${readASnippet}
        } else {
          return vec4(0.0);
        }`;

    const sampleB = this.fitB ?
        `W[row * dimBOuter / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
            W[row * dimBOuter / 4 + col] : vec4(0.0)`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(
          this.activation, this.isVec4, this.useWgsl);
      if (this.hasPreluActivationWeights) {
        activationSnippet = `vec4 activation(vec4 a, ivec4 outCoord) {
          vec4 b = getPreluActivationWeightsAtOutCoords(outCoord);
          ${activationOp}
        }`;
      } else if (this.hasLeakyreluAlpha) {
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${activationOp}
        }`;
        throw new Error('Leakyrelu is not supported.');
      } else {
        activationSnippet = `
        vec4 activation(vec4 a, ivec4 outCoord) {
          ${activationOp}
        }`;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet = this.addBias ? 'ivec4 coords = getOutputCoords(); ' +
            'value += getBiasAtOutCoords(outCoord);' :
                                          '';

    const userCode = `
        ${activationSnippet}
        ${matMulSource}

        int batch;
        int dimAOuter = outShape[1] * outShape[2];
        int dimBOuter = outShape[3];
        int dimInner = filterDims[0] * filterDims[1] * xShape[3];
        vec4 mm_readA(int row, int col) {
          int r = int(row), c = int(col * 4);
          ${sampleA};
        }

        vec4 mm_readB(int row, int col) {
          return ${sampleB};
        }

        void mm_write(int row, int col, vec4 value) {
          if (row < dimAOuter && col * 4 < dimBOuter)
          {
            ivec4 outCoord = ivec4(
              batch,
              row / outShape[2],
              row % outShape[2],
              col * 4);
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutput(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
              value);
          }
        }

        void main() {
          batch = int(gl_GlobalInvocationID.z);

          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
    return userCode;
  }

  // index is used to avoid repeated definition error.
  getSampleAWithRemainderWgsl(index: number): string {
    return `let flatIndex${index} = getFlatIndex4D(coord, uniforms.xShape);
    let divBy4Remainder${index} = flatIndex${index} % 4u;
    let divBy4Index${index} = flatIndex${index} / 4u;
    let curData${index} = x.numbers[divBy4Index${index}];
    if (divBy4Remainder${index} == 0u) {
      temp = curData${index};
    } else {
      // TODO: This could end up being a redundant load with another one in
      // the same shader invocation. Perhaps there's an opportunity for
      // optimization
      let nextData${index} = x.numbers[divBy4Index${index} + 1u];
      if (divBy4Remainder${index} == 1u) {
        temp = vec4<f32>(curData${index}.yzw, nextData${index}.x);
      } elseif (divBy4Remainder${index} == 2u) {
        temp = vec4<f32>(curData${index}.zw, nextData${index}.xy);
      } elseif (divBy4Remainder${index} == 3u) {
        temp = vec4<f32>(curData${index}.w, nextData${index}.xyz);
      }
    }
    `;
  }

  getUserCodeWgsl(): string {
    const elementsPerThread: [number, number, number] = [4, 4, 1];
    const matMulSource =
        makeMatMulPackedVec4SourceWgsl(elementsPerThread, this.workGroupSize);

    const remainder = this.convInfo.inChannels % 4;
    // Below code only applys to valid padding type.
    const remainderSnippet = remainder === 0 ?
        `// The bounds checking is always needed since we use it to pad zero for
          // the 'same' padding type.
          if (coordsInBounds4D(coord, uniforms.xShape)) {
            resData = x.numbers[getFlatIndex4D(coord, uniforms.xShape) / 4u];
          } else {
            resData = vec4<f32>(0.0); }` :
        `var temp = vec4<f32>(0.0);
          ${this.getSampleAWithRemainderWgsl(1)}
          resData = temp;
          if (WCol == (uniforms.filterDims[1] - 1u)) {
            coord = vec4<u32>(
              coord.x, coord.y + 1u, coord.z + 1u - uniforms.filterDims[1], 0u);
              ${this.getSampleAWithRemainderWgsl(2)}
            if (inChCoord == 0u) {
              resData = vec4<f32>(resData.xyz, temp.x);
            } elseif (inChCoord == 1u) {
              resData = vec4<f32>(resData.xy, temp.xy);
            } else {
              resData = vec4<f32>(resData.x, temp.xyz);
            }
          }
          `;

    const readASnippet = `let outRow = r / uniforms.outShape[2];
        let outCol = r % uniforms.outShape[2];
        let WRow = c / (uniforms.filterDims[1] * uniforms.xShape[3]);
        let WCol = (c / uniforms.xShape[3]) % uniforms.filterDims[1];
        let inChCoord = c % uniforms.xShape[3];
        var coord = vec4<u32>(
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
        `return W.numbers[row * uniforms.dimBOuter / 4u + col];` :
        `if(coordsInBounds2D(vec2<u32>(row, col * 4u), vec2<u32>(uniforms.dimInner, uniforms.dimBOuter))) {
           return W.numbers[row * uniforms.dimBOuter / 4u + col];
         }
         return vec4<f32>(0.0);
        `;
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(
          this.activation, this.isVec4, this.useWgsl);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : vec4<f32>, outCoord : vec4<u32>) -> vec4<f32> {
          let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
          ${activationOp}
        }`;
      } else if (this.hasLeakyreluAlpha) {
        activationSnippet = `fn activation(a: vec4<f32>) -> vec4<f32> {
          let b = getLeakyreluAlphaAtOutCoords();
          ${activationOp}
        }`;
        throw new Error('Leakyrelu is not supported.');
      } else {
        activationSnippet = `
        fn activation(a : vec4<f32>, outCoord : vec4<u32>) -> vec4<f32> {
          ${activationOp}
        }`;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet = this.addBias ?
        'value = value + getBiasAtOutCoordsByCoords(outCoord);' :
        '';

    const userCode = `
        ${activationSnippet}
        fn mm_readA(row : u32, col : u32, globalId : vec3<u32>) -> vec4<f32> {
          let r = row;
          let c = col * 4u;
          var batch = globalId.z;
          ${sampleA}
        }

        fn mm_readB(row : u32, col : u32, globalId : vec3<u32>) -> vec4<f32> {
          ${sampleB}
        }

        fn mm_write(row : u32, col : u32, valueInput : vec4<f32>, globalId : vec3<u32>) {
          var batch = globalId.z;
          var value = valueInput;
          if (row < uniforms.dimAOuter && col * 4u < uniforms.dimBOuter)
          {
            let outCoord = vec4<u32>(
              batch,
              row / uniforms.outShape[2],
              row % uniforms.outShape[2],
              col * 4u);
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutput(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
              value);
          }
        }
        ${matMulSource}
      `;
    return userCode;
  }
}
