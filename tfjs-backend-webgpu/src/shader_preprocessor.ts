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

import {backend_util, DataType, util} from '@tensorflow/tfjs-core';

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

export function getShapeCoords(dataShape: number[]): string {
  const rank = dataShape.length;
  if (rank <= 1) {
    return `int(${dataShape[0]})`;
  } else if (rank === 2) {
    return `ivec2(${dataShape[0]}, ${dataShape[1]})`;
  } else if (rank === 3) {
    return `ivec3(${dataShape[0]}, ${dataShape[1]}, ${dataShape[2]})`;
  } else if (rank === 4) {
    return `ivec4(${dataShape[0]}, ${dataShape[1]}, ${dataShape[2]}, ${
        dataShape[3]})`;
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

type GLSLDataType = 'float'|'int'|'vec4'|'ivec4'|'bvec4';
function mapToGlslTypes(type: DataType, isVec4: boolean): GLSLDataType|
    DataType {
  if (type === 'float32') {
    return isVec4 ? 'vec4' : 'float';
  } else if (type === 'int32') {
    return isVec4 ? 'ivec4' : 'int';
  } else if (type === 'bool') {
    return isVec4 ? 'bvec4' : 'bool';
  }

  return type;
}

interface ProgramParams {
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  workGroupSize?: [number, number, number];
  variableNames: string[];
  uniforms?: string;
  isVec4?: boolean;
  getUserCode: () => string;
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
      ${mapToGlslTypes(outputData.dtype, program.isVec4)} result[];
    };
  `);

  program.variableNames.forEach((x, i) => {
    prefixSnippets.push(`
      layout(std430, set = 0, binding = ${1 + i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputInfo[i].dtype, program.isVec4)} ${x}[];
      };
    `);
  });

  let uniformDeclaration = '';

  if (program.uniforms) {
    uniformDeclaration += program.uniforms;

    prefixSnippets.push(`
    layout(std140, set = 0, binding = ${
        1 + program.variableNames.length}) uniform Uniforms {
      ${uniformDeclaration}
    };
  `);
  }
  const [getOutputCoords, dispatchLayoutRank] =
      generateGetOutputCoords(outputData.shape, program.dispatchLayout);
  const getCoords = generateGetCoordsFromFlatIndex(outputData.shape);
  const sources = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS,
    getOutputCoords, getCoords,
    getSetOutputSnippet(outputData.shape, outputData.dtype, program.isVec4)
  ];

  if (dispatchLayoutRank === outputData.shape.length) {
    // Input sampling snippet is only meaningful when the output isn't getting
    // implicitly reshaped (like it does in conv2d_matmul).
    const inputSamplingSnippet =
        inputInfo
            .map(
                x => getInputSamplingSnippet(
                    x, outputData.shape, program.isVec4))
            .join('\n');
    sources.push(inputSamplingSnippet);
  }

  sources.push(program.getUserCode());
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

  // Checks whether coordinates lie within the bounds of the shape.
  bool coordsInBounds(ivec4 coord, ivec4 shape) {
    return all(greaterThanEqual(coord, ivec4(0))) &&
        all(lessThan(coord, shape));
  }

  bool coordsInBounds(ivec3 coord, ivec3 shape) {
    return all(greaterThanEqual(coord, ivec3(0))) &&
        all(lessThan(coord, shape));
  }

  bool coordsInBounds(ivec2 coord, ivec2 shape) {
    return all(greaterThanEqual(coord, ivec2(0))) &&
        all(lessThan(coord, shape));
  }
`;

const SAMPLING_SNIPPETS = `
  int getFlatIndex(int coord, int shape) {
    return coord;
  }

  int getFlatIndex(ivec2 coords, ivec2 shape) {
    return int(dot(coords, ivec2(shape.y, 1.)));
  }

  int getFlatIndex(ivec3 coords, ivec3 shape) {
    return int(dot(coords, ivec3(shape.y * shape.z, shape.z, 1.)));
  }

  int getFlatIndex(ivec4 coords, ivec4 shape) {
    return int(dot(coords, ivec4(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
  }
`;

function getSetOutputSnippet(
    outShape: number[], outBufferType: DataType, isVec4: boolean): string {
  const outRank = outShape.length;
  const glslType = mapToGlslTypes(outBufferType, isVec4);
  let snippet;
  if (isVec4) {
    snippet = `void setOutput(int flatIndex, vec4 value) {
      result[flatIndex] = ${
        glslType === 'ivec4' ?
            'ivec4(value)' :
            (glslType === 'bvec4' ? 'bvec4(value)' : 'value')};
    }
    void setOutput(int flatIndex, ivec4 value) {
      result[flatIndex] = ${
        glslType === 'vec4' ?
            'vec4(value)' :
            (glslType === 'bvec4' ? 'bvec4(value)' : 'value')};
    }`;
  } else {
    snippet = `void setOutput(int flatIndex, float value) {
      result[flatIndex] = ${
        glslType === 'int' ? 'int(value)' :
                             (glslType === 'bool' ? 'bool(value)' : 'value')};
    }
    void setOutput(int flatIndex, int value) {
      result[flatIndex] = ${
        glslType === 'float' ? 'float(value)' :
                               (glslType === 'bool' ? 'bool(value)' : 'value')};
    }`;
  }

  if (outRank >= 2) {
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
    const type = getCoordsDataType(outRank);

    if (isVec4) {
      snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, vec4 value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), ${
          getShapeCoords(outShape)});
        setOutput(flatIndex / 4, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, ivec4 value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), ${
          getShapeCoords(outShape)});
        setOutput(flatIndex / 4, value);
      }
    `;
    } else {
      snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, float value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), ${
          getShapeCoords(outShape)});
        setOutput(flatIndex, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, int value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), ${
          getShapeCoords(outShape)});
        setOutput(flatIndex, value);
      }
    `;
    }
  }

  return snippet;
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShape: number[], isVec4: boolean): string {
  let res = getSamplerFromInInfo(inInfo, isVec4);

  const inShape = inInfo.shape;
  if (inShape.length <= outShape.length) {
    res += getSamplerAtOutputCoords(inInfo, outShape, isVec4);
  }

  return res;
}

function getSamplerFromInInfo(inInfo: InputInfo, isVec4: boolean): string {
  const texName = inInfo.name;
  const rank = inInfo.shape.length;
  const type = getCoordsDataType(rank);
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, rank);
  const inputs = dims.map(d => `int ${d}`).join(', ');

  if (rank < 1) {
    if (isVec4) {
      return `
        vec4 ${funcName}() {
          return ${texName}[0];
        }
      `;
    }

    return `
      float ${funcName}() {
        return ${texName}[0];
      }
    `;
  }

  if (isVec4) {
    return `
    vec4 ${funcName}(${inputs}) {
      return ${texName}[getFlatIndex(${type}(${dims.join(',')}),
        ${getShapeCoords(inInfo.shape)}) / 4];
    }
  `;
  }

  return `
    float ${funcName}(${inputs}) {
      return float(${texName}[getFlatIndex(${type}(${dims.join(',')}),
        ${getShapeCoords(inInfo.shape)})]);
    }
  `;
}

function getSamplerAtOutputCoords(
    inInfo: InputInfo, outShape: number[], isVec4: boolean): string {
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
    if (isVec4) {
      return `
      vec4 ${funcName}() {
        return get${texFuncSnippet}();
      }

      vec4 ${funcName}(${type} coords) {
        return get${texFuncSnippet}();
      }
    `;
    }
    return `
      float ${funcName}() {
        return get${texFuncSnippet}();
      }

      float ${funcName}(${type} coords) {
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

  if (isVec4) {
    return `
      vec4 ${funcName}() {
        ${type} coords = getOutputCoords();
        ${coordsSnippet}
        return ${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)}) / 4];
      }

      vec4 ${funcName}(${type} coords) {
        ${coordsSnippet}
        return ${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)}) / 4];
      }
    `;
  }

  return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
      getShapeCoords(inInfo.shape)})]);
    }

    float ${funcName}(${type} coords) {
      ${coordsSnippet}
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
      getShapeCoords(inInfo.shape)})]);
    }
  `;
}

/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
function generateGetOutputCoords(
    outShape: number[],
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
      gatherDimensionsStr += `int d${arr[0]} =
        int(gl_GlobalInvocationID[${i}]);`;
    } else {
      const strides =
          symbolicallyComputeStrides(arr, `${getShapeCoords(outShape)}`);
      gatherDimensionsStr += `int index${i} =
        int(gl_GlobalInvocationID[${i}]);`;
      for (let j = 0; j < strides.length; j++) {
        gatherDimensionsStr += `int d${arr[j]} = index${i} / ${strides[j]};`;

        if (j === strides.length - 1) {
          gatherDimensionsStr += `int d${arr[j + 1]} = ` +
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
  let snippet = `${dtype} getOutputCoords() {
    ${gatherDimensionsStr}
  `;
  if (dimensions.length === 0) {
    snippet += `return ${dtype}(0);}`;
  } else {
    snippet += `return ${dtype}(${dimensions.join(',')});}`;
  }

  return [snippet, rank];
}

/**
 * Derives logical coordinates from a flat index. Performs integer division with
 * each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
function generateGetCoordsFromFlatIndex(shape: number[]): string {
  const rank = shape.length;

  if (rank <= 1) {
    return `int getCoordsFromFlatIndex(int index) {return index; }`;
  }

  const strides = util.computeStrides(shape);
  const dtype = getCoordsDataType(rank);
  const coords: string[] = [];
  for (let i = 0; i < rank; i++) {
    coords.push(`d${i}`);
  }

  const snippet =
      strides
          .map((stride, i) => {
            const line1 = `int ${coords[i]} = index / ${stride}`;
            const line2 = i === strides.length - 1 ?
                `int ${coords[i + 1]} = index - ${coords[i]} * ${stride}` :
                `index -= ${coords[i]} * ${stride}`;
            return `${line1}; ${line2};`;
          })
          .join('');

  return `
    ${dtype} getCoordsFromFlatIndex(int index) {
      ${snippet}
      return ${dtype}(${coords.join(',')});
    }
  `;
}
