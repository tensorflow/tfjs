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

import {backend_util, DataType} from '@tensorflow/tfjs-core';

import {symbolicallyComputeStrides} from './shader_util';

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'int';
  } else if (rank === 2) {
    return 'ivec2';
  } else if (rank === 3) {
    return 'ivec3';
  } else if (rank === 4) {
    return 'ivec4';
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

type GLSLDataType = 'float'|'int';
function mapToGlslTypes(type: DataType): GLSLDataType|DataType {
  if (type === 'float32') {
    return 'float';
  }
  if (type === 'int32') {
    return 'int';
  }
  return type;
}

interface ProgramParams {
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  workGroupSize?: [number, number, number];
  variableNames: string[];
  uniforms?: string;
  userCode: string;
}

export interface InputInfo {
  dtype: DataType;
  shape: number[];
  name: string;
}

export function makeShader(
    inputInfo: InputInfo[], outputData: {dtype: DataType, shape: number[]},
    program: ProgramParams): string {
  const prefixSnippets: string[] = [];

  if (program.workGroupSize != null) {
    prefixSnippets.push(`
      layout (local_size_x = ${program.workGroupSize[0]},
              local_size_y = ${program.workGroupSize[1]},
              local_size_z = ${program.workGroupSize[2]}) in;
    `);
  }

  // Output buffer.
  prefixSnippets.push(`
    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      ${mapToGlslTypes(outputData.dtype)} result[];
    };
  `);

  let uniformDeclaration = '';
  program.variableNames.forEach((x, i) => {
    uniformDeclaration += `${getCoordsDataType(inputInfo[i].shape.length)} ${
        x.charAt(0).toLowerCase() + x.slice(1)}Shape; `;
    prefixSnippets.push(`
      layout(std430, set = 0, binding = ${1 + i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputInfo[i].dtype)} ${x}[];
      };
    `);
  });

  uniformDeclaration +=
      `${getCoordsDataType(outputData.shape.length)} outShape; `;

  if (program.uniforms) {
    uniformDeclaration += program.uniforms;
  }

  prefixSnippets.push(`
    layout(std140, set = 0, binding = ${
      1 + program.variableNames.length}) uniform Uniforms {
      ${uniformDeclaration}
    };
  `);

  const [getOutputCoords, dispatchLayoutRank] =
      generateGetOutputCoords(program.dispatchLayout);
  const sources = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS,
    getOutputCoords,
    getSetOutputSnippet(outputData.shape.length, outputData.dtype)
  ];

  if (dispatchLayoutRank === outputData.shape.length) {
    // Input sampling snippet is only meaningful when the output isn't getting
    // implicitly reshaped (like it does in conv2d_matmul).
    const inputSamplingSnippet =
        inputInfo.map(x => getInputSamplingSnippet(x, outputData.shape))
            .join('\n');
    sources.push(inputSamplingSnippet);
  }

  sources.push(program.userCode);
  const source = sources.join('\n');
  return source;
}

const SHADER_PREFIX = `#version 450

  int idiv(int a, int b, float sign) {
    int res = a / b;
    int mod = a % b;
    if (sign < 0. && mod != 0) {
      res -= 1;
    }
    return res;
  }
`;

const SAMPLING_SNIPPETS = `
  uint getFlatIndex(uint coord, uint shape) {
    return coord;
  }

  uint getFlatIndex(ivec2 coords, ivec2 shape) {
    return uint(dot(coords, ivec2(shape.y, 1.)));
  }

  uint getFlatIndex(ivec3 coords, ivec3 shape) {
    return uint(dot(coords, ivec3(shape.y * shape.z, shape.z, 1.)));
  }

  uint getFlatIndex(ivec4 coords, ivec4 shape) {
    return uint(dot(coords, ivec4(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
  }
`;

function getSetOutputSnippet(outRank: number, outBufferType: DataType): string {
  let snippet = `void setOutput(uint flatIndex, float value) {
      result[flatIndex] = ${
      mapToGlslTypes(outBufferType) === 'int' ? 'int(value)' : 'value'};
    }
    void setOutput(uint flatIndex, int value) {
      result[flatIndex] = ${
      mapToGlslTypes(outBufferType) === 'float' ? 'float(value)' : 'value'};
    }`;

  if (outRank >= 2) {
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
    const type = getCoordsDataType(outRank);

    snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, float value) {
        uint flatIndex = getFlatIndex(${type}(${dims.join(', ')}), outShape);
        setOutput(flatIndex, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, int value) {
        uint flatIndex = getFlatIndex(${type}(${dims.join(', ')}), outShape);
        setOutput(flatIndex, value);
      }
    `;
  }

  return snippet;
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShape: number[]): string {
  let res = getSamplerFromInInfo(inInfo);

  const inShape = inInfo.shape;
  if (inShape.length <= outShape.length) {
    res += getSamplerAtOutputCoords(inInfo, outShape);
  }

  return res;
}

function getSamplerFromInInfo(inInfo: InputInfo): string {
  const texName = inInfo.name;
  const rank = inInfo.shape.length;
  const type = getCoordsDataType(rank);
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, rank);
  const inputs = dims.map(d => `int ${d}`).join(', ');

  if (rank < 1) {
    return `
      float ${funcName}() {
        return ${texName}[0];
      }
    `;
  }

  return `
    float ${funcName}(${inputs}) {
      return ${texName}[getFlatIndex(${type}(${dims.join(',')}),
        ${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape)];
    }
  `;
}

function getSamplerAtOutputCoords(
    inInfo: InputInfo, outShape: number[]): string {
  const texName = inInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);

  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';

  const inRank = inInfo.shape.length;
  const outRank = outShape.length;
  const type = getCoordsDataType(outRank);

  const broadcastDims = backend_util.getBroadcastDims(inInfo.shape, outShape);
  const rankDiff = outRank - inRank;

  let coordsSnippet = '';

  if (inRank === 0) {
    return `
      float ${funcName}() {
        return get${texFuncSnippet}();
      }
    `;
  } else {
    if (outRank < 2 && broadcastDims.length >= 1) {
      coordsSnippet = 'coords = 0;';
    } else {
      coordsSnippet =
          broadcastDims.map(d => `coords[${d + rankDiff}] = 0;`).join('\n');
    }
  }

  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    if (outRank > 1) {
      const coordsType = getCoordsDataType(inRank);
      const coordsValues =
          inInfo.shape.map((s, i) => `coords[${i + rankDiff}]`).join(', ');
      unpackedCoordsSnippet = `${coordsType}(${coordsValues})`;
    } else {
      unpackedCoordsSnippet = 'coords';
    }
  }

  return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      return ${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
      texName.charAt(0).toLowerCase() + texName.slice(1)}Shape)];
    }
  `;
}

/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
function generateGetOutputCoords(
    dispatchLayout: {x: number[], y?: number[], z?: number[]}):
    [string, number] {
  const {x, y = [], z = []} = dispatchLayout;
  let gatherDimensionsStr = '';
  const dims = [x, y, z];

  let rank = 0;

  for (let i = 0; i < dims.length; i++) {
    const arr = dims[i];

    if (arr.length === 0) {
      continue;
    }

    rank += arr.length;

    if (arr.length === 1) {
      gatherDimensionsStr += `uint d${arr[0]} = gl_GlobalInvocationID[${i}];`;
    } else {
      const strides = symbolicallyComputeStrides(arr, 'outShape');
      gatherDimensionsStr += `uint index${i} =
        gl_GlobalInvocationID[${i}];`;
      for (let j = 0; j < strides.length; j++) {
        gatherDimensionsStr += `uint d${arr[j]} = index${i} / ${strides[j]};`;

        if (j === strides.length - 1) {
          gatherDimensionsStr += `uint d${arr[j + 1]} = ` +
              `index${i} - d${arr[j]} * ${strides[j]};`;
        } else {
          gatherDimensionsStr += `index${i} -= d${arr[j]} * ${strides[j]};`;
        }
      }
    }
  }

  const dimensions = [];
  for (let i = 0; i < rank; i++) {
    dimensions.push(`d${i}`);
  }

  const dtype = getCoordsDataType(rank);
  const snippet = `${dtype} getOutputCoords() {
    ${gatherDimensionsStr}

    return ${dtype}(${dimensions.join(',')});
  }`;
  return [snippet, rank];
}
