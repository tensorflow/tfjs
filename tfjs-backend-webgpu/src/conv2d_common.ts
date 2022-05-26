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

import {mapActivationToShaderProgram} from './activation_util';

const typeSnippet = (innerElementSize: number) => {
  switch (innerElementSize) {
    case 1:
      return 'f32';
    case 2:
      return 'vec2<f32>';
    case 3:
      return 'vec3<f32>';
    case 4:
      return 'vec4<f32>';
    default:
      throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
  }
};

export function conv2dCommonSnippet(
    isChannelsLast: boolean, fitAOuter: boolean, fitBOuter: boolean,
    fitInner: boolean, addBias = false,
    activation: backend_util.Activation = null,
    hasPreluActivationWeights = false, innerElementSizeX = 4,
    innerElementSizeW = 4, innerElementSize = 4) {
  const getXSnippet = (innerElementSize: number) => {
    switch (innerElementSize) {
      case 1:
        return 'resData = x[xIndex];';
      case 3:
        return 'resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);';
      case 4:
        return 'resData = x[xIndex / 4];';
      default:
        throw new Error(
            `innerElementSize ${innerElementSize} is not supported.`);
    }
  };
  const getWSnippet = (innerElementSize: number) => {
    switch (innerElementSize) {
      case 1:
        return 'return W[row * uniforms.wShape[3] + colIn];';
      case 4:
        return 'return W[row * uniforms.wShape[3] / 4 + colIn];';
      default:
        throw new Error(
            `innerElementSize ${innerElementSize} is not supported.`);
    }
  };
  const coordASnippet = isChannelsLast ? `
      let coord = vec4<i32>(batch, xRow, xCol, xCh);
      ` :
                                         `
      let coord = vec4<i32>(batch, xCh, xRow, xCol);
      `;

  const coordResSnippet = isChannelsLast ? `
      let outCoord = vec4<i32>(
        batch,
        row / outWidth,
        row % outWidth,
        col);
      ` :
                                           `
      let outCoord = vec4<i32>(
        batch,
        row,
        col / outWidth,
        col % outWidth);
      `;

  const xHight = isChannelsLast ? 'uniforms.xShape[1]' : 'uniforms.xShape[2]';
  const xWidth = isChannelsLast ? 'uniforms.xShape[2]' : 'uniforms.xShape[3]';
  const row = isChannelsLast ? 'row' : 'col';
  const col = isChannelsLast ? 'col' : 'row';
  const readXSnippet = `
      let inChannels = uniforms.wShape[2];
      let outWidth = ${
      isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
      let outRow = ${row} / outWidth;
      let outCol = ${row} % outWidth;

      let WRow = ${col} / (uniforms.filterDims[1] * inChannels);
      let WCol = ${col} / inChannels % uniforms.filterDims[1];
      let xRow = outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0];
      let xCol = outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1];
      let xCh = ${col} % inChannels;
      var resData = ${typeSnippet(innerElementSizeX)}(0.0);
      // The bounds checking is always needed since we use it to pad zero for
      // the 'same' padding type.
      if (xRow >= 0 && xRow < ${xHight} && xCol >= 0 && xCol < ${xWidth}) {
        ${coordASnippet}
        let xIndex = getIndexFromCoords4D(coord, uniforms.xShape);
        ${getXSnippet(innerElementSizeX)}
      }
      return resData;`;

  const sampleX = isChannelsLast ? (fitAOuter && fitInner ? `
      let col = colIn * ${innerElementSizeX};
      ${readXSnippet}` :
                                                            `
      let col = colIn * ${innerElementSizeX};
      if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${readXSnippet}
      }
      return ${typeSnippet(innerElementSizeX)}(0.0);`) :
                                   (fitInner && fitBOuter ? `
      let col = colIn * ${innerElementSizeX};
      ${readXSnippet}` :
                                                            `
      let col = colIn * ${innerElementSizeX};
      if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
        ${readXSnippet}
      }
      return ${typeSnippet(innerElementSizeX)}(0.0);`);

  const sampleW = `${getWSnippet(innerElementSizeW)}`;

  const resType = typeSnippet(innerElementSize);
  const aType = isChannelsLast ? typeSnippet(innerElementSizeX) :
                                 typeSnippet(innerElementSizeW);
  const bType = isChannelsLast ? typeSnippet(innerElementSizeW) :
                                 typeSnippet(innerElementSizeX);
  let activationSnippet = '', applyActivationSnippet = '';
  if (activation) {
    const activationOp =
        mapActivationToShaderProgram(activation, innerElementSize === 4);
    if (hasPreluActivationWeights) {
      activationSnippet =
          `fn activation(a: ${resType}, outCoord : vec4<i32>) -> ${resType} {
              let b = getPreluActivationWeightsByOutputCoords(outCoord);
              ${activationOp}
           }`;
    } else {
      activationSnippet = `
           fn activation(a : ${resType}, outCoord : vec4<i32>) -> ${resType} {
             ${activationOp}
           }`;
    }

    applyActivationSnippet = `value = activation(value, outCoord);`;
  }

  const addBiasSnippet =
      addBias ? 'value = value + getBiasByOutputCoords(outCoord);' : '';

  const userCode = `
      ${activationSnippet}
      fn mm_readA(row : i32, colIn : i32, globalId : vec3<u32>) -> ${aType} {
        var batch = i32(globalId.z);
        ${isChannelsLast ? sampleX : sampleW}
      }

      fn mm_readB(row : i32, colIn : i32, globalId : vec3<u32>) -> ${bType} {
        var batch = i32(globalId.z);
        ${isChannelsLast ? sampleW : sampleX}
      }

      fn mm_write(row : i32, colIn : i32, valueIn : ${
      resType}, globalId : vec3<u32>) {
        var col = colIn * ${innerElementSize};
        ${
      fitAOuter && fitBOuter ?
          '' :
          'if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)'}
        {
        var batch = i32(globalId.z);
        var value = valueIn;
        let outWidth = ${
      isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
        ${coordResSnippet}
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutputAtCoords(outCoord[0], outCoord[1], outCoord[2], outCoord[3], value);
        }
      }`;
  return userCode;
}
