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

import {backend_util, DataType, util} from '@tensorflow/tfjs-core';
import {symbolicallyComputeStrides} from './shader_util';

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'i32';
  } else if (rank === 2) {
    return `vec2<i32>`;
  } else if (rank === 3) {
    return `vec3<i32>`;
  } else if (rank === 4) {
    return `vec4<i32>`;
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

type WGSLDataType = 'f32'|'i32'|'vec4<f32>'|'vec4<i32>'|'vec4<bool>';
function mapToWgslTypes(type: DataType, isVec4: boolean): WGSLDataType|
    DataType {
  if (type === 'float32') {
    return isVec4 ? 'vec4<f32>' : 'f32';
  } else if (type === 'int32') {
    return isVec4 ? 'vec4<i32>' : 'i32';
  } else if (type === 'bool') {
    // Type 'bool' cannot be used in storage class,
    // https://www.w3.org/TR/WGSL/#host-shareable-types.
    return isVec4 ? 'vec4<i32>' : 'i32';
  }

  return type;
}

interface ProgramParams {
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  workGroupSize: [number, number, number];
  variableNames: string[];
  uniforms?: string;
  isVec4?: boolean;
  size?: number;
  atomic?: boolean;
  getUserCode: () => string;
}

export interface InputInfo {
  dtype: DataType;
  shape: number[];
  name: string;
}

export function getWorkGroupSizeString(): string {
  return `
  [[stage(compute), workgroup_size(workGroupSizeX, workGroupSizeY, workGroupSizeZ)]]
`;
}

export function getGlobalIndexString(nonFlatDispatch = false): string {
  if (nonFlatDispatch) {
    return 'let index = getGlobalIndex(globalId, localId);';
  } else {
    // Only used when the y/z dimension of workgroup size is 1.
    return 'let index = i32(globalId.x);';
  }
}

export function getMainHeaderString() {
  return `
  [[stage(compute), workgroup_size(workGroupSizeX, workGroupSizeY, workGroupSizeZ)]]
  fn main([[builtin(local_invocation_id)]] localId : vec3<u32>, [[builtin(global_invocation_id)]] globalId : vec3<u32>)
`;
}

export function makeShader(
    inputInfo: InputInfo[], outputData: {dtype: DataType, shape: number[]},
    program: ProgramParams, isFromPixel = false): string {
  const workGroupSizeSnippet = `
    let workGroupSizeX = ${program.workGroupSize[0]}u;
    let workGroupSizeY = ${program.workGroupSize[1]}u;
    let workGroupSizeZ = ${program.workGroupSize[2]}u;`;

  if (isFromPixel === true) {
    const getCoords = generateGetCoordsFromFlatIndex(outputData.shape);
    const outputBufferStr = `
      [[block]] struct Matrix0 {
        numbers: array<${mapToWgslTypes(outputData.dtype, program.isVec4)}>;
      };
      [[block]] struct Uniform {
        size            : i32;
        numChannels     : i32;
        outShapeStrides : vec2<i32>;
        dispatchSize    : vec3<u32>;
      };

      [[group(0), binding(0)]] var<storage, write> result : Matrix0;
      [[group(0), binding(2)]] var<uniform> uniforms: Uniform;
    `;
    return [
      SHADER_PREFIX,
      outputBufferStr,
      workGroupSizeSnippet,
      SAMPLING_SNIPPETS,
      getCoords,
      program.getUserCode(),
    ].join('\n');
  }

  const prefixSnippets: string[] = [];
  let uniformDeclaration = '[[block]] struct Uniforms { NAN : f32; ';
  program.variableNames.forEach((x, i) => {
    uniformDeclaration += `${x.charAt(0).toLowerCase() + x.slice(1)}Shape : ${
        getCoordsDataType(inputInfo[i].shape.length)}; `;
  });
  uniformDeclaration +=
      `outShape : ${getCoordsDataType(outputData.shape.length)} ; `;
  const stridesLength = outputData.shape.length - 1;
  uniformDeclaration += `
       outShapeStrides: ${getCoordsDataType(stridesLength)}; `;

  if (program.size != null) {
    uniformDeclaration += 'size : i32; ';
  }
  uniformDeclaration += 'dispatchSize : vec3<u32>; ';
  if (program.uniforms) {
    uniformDeclaration += program.uniforms;
  }
  uniformDeclaration += '};';

  prefixSnippets.push(uniformDeclaration);

  // Output buffer.
  if (program.atomic) {
    prefixSnippets.push(`
    [[block]] struct Matrix0 {
        numbers: array<atomic<i32>>;
    };

    [[group(0), binding(0)]] var<storage, read_write> result : Matrix0;
  `);
  } else {
    prefixSnippets.push(`
    [[block]] struct Matrix0 {
        numbers: array<${mapToWgslTypes(outputData.dtype, program.isVec4)}>;
    };

    [[group(0), binding(0)]] var<storage, write> result : Matrix0;
  `);
  }
  program.variableNames.forEach((x, i) => {
    prefixSnippets.push(`
    [[block]] struct Matrix${1 + i} {
      numbers: array<${mapToWgslTypes(inputInfo[i].dtype, program.isVec4)}>;
    };
    [[group(0), binding(${1 + i})]] var<storage, read> ${x} : Matrix${1 + i};
    `);
  });

  if (uniformDeclaration !== '') {
    prefixSnippets.push(`
    [[group(0), binding(${
        1 + program.variableNames.length})]] var<uniform> uniforms : Uniforms;
    `);
  }

  prefixSnippets.push(workGroupSizeSnippet);

  const [getOutputCoords, dispatchLayoutRank] =
      generateGetOutputCoords(outputData.shape, program.dispatchLayout);
  const getCoords = generateGetCoordsFromFlatIndex(outputData.shape);

  const sources = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS, getCoords,
    getOutputCoords, getOutputFlatIndexSnippet(outputData.shape.length)

  ];
  if (!program.atomic) {
    sources.push(getSetOutputSnippet(
        outputData.shape, outputData.dtype, program.isVec4));
  }
  if (dispatchLayoutRank === outputData.shape.length) {
    // Input sampling snippet is only meaningful when the output isn't getting
    // implicitly reshaped (like it does in conv2d_matmul).
    const inputSamplingSnippet =
        inputInfo
            .map(
                x => getInputSamplingSnippet(
                    x, outputData.shape, program.isVec4,
                    program.dispatchLayout.x.length ===
                        outputData.shape.length))
            .join('\n');
    sources.push(inputSamplingSnippet);
  }

  sources.push(program.getUserCode());
  const source = sources.join('\n');
  return source;
}

const SHADER_PREFIX = `
  fn idiv(a: i32, b: i32, sign: f32) -> i32 {
    var res: i32 = a / b;
    let mod: i32 = a % b;
    if (sign < 0. && mod != 0) {
      res = res - 1;
    }
    return res;
  }

  fn isNanCustom(val : f32) -> bool {
    if (val > 0.0) {
      return false;
    }
    if (val < 0.0) {
      return false;
    }
    if (val == 0.0) {
      return false;
    }
    return true;
  }

  fn isNanCustomVec4F32(val : vec4<f32>) -> vec4<f32> {
    var res = vec4<f32> (0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
      if (isNanCustom(val[i])) {
        res[i] = 1.0;
      } else {
        res[i] = 0.0;
      }
    }
    return res;
  }

  // Checks whether coordinates lie within the bounds of the shape.
  fn coordsInBounds4D(coord : vec4<i32>, shape : vec4<i32>) -> bool {
    return all(coord >= vec4<i32>(0)) &&
        all(coord < shape);
  }

  fn coordsInBounds3D(coord : vec3<i32>, shape : vec3<i32>) -> bool {
    return all(coord >= vec3<i32>(0)) &&
        all(coord < shape);
  }

  fn coordsInBounds2D(coord : vec2<i32>, shape : vec2<i32>) -> bool {
    return all(coord >= vec2<i32>(0)) &&
        all(coord < shape);
  }
  `;
const SAMPLING_SNIPPETS = `
  fn getFlatIndex1D(coord : i32, shape : i32) -> i32 {
    return coord;
  }

  fn getFlatIndex2D(coords : vec2<i32>, shape : vec2<i32>) -> i32 {
    return i32(dot(vec2<f32>(coords), vec2<f32>(f32(shape.y), 1.0)));
  }

  fn getFlatIndex3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
    return i32(dot(vec3<f32>(coords), vec3<f32>(f32(shape.y) * f32(shape.z), f32(shape.z), 1.0)));
  }

  fn getFlatIndex4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
    return i32(dot(vec4<f32>(coords), vec4<f32>(
        f32(shape.y) * f32(shape.z) * f32(shape.w), f32(shape.z) * f32(shape.w), f32(shape.w), 1.0)));
  }

  fn getGlobalIndex(globalId : vec3<u32>, localId : vec3<u32>) -> i32 {
    let localInvocationIndex = localId.z * workGroupSizeX * workGroupSizeY +
      localId.y * workGroupSizeX + localId.x;
    let workGroupID = (globalId - localId)/vec3<u32>(
      workGroupSizeX, workGroupSizeY, workGroupSizeZ);
    return i32((workGroupID.z * uniforms.dispatchSize.x * uniforms.dispatchSize.y +
      workGroupID.y * uniforms.dispatchSize.x + workGroupID.x) *
      (workGroupSizeX * workGroupSizeY * workGroupSizeZ) +
      localInvocationIndex);
  }
`;

function getOutputFlatIndexSnippet(outRank: number) {
  let snippet = '';
  switch (outRank) {
    case 0:
    case 1:
      snippet += `
        fn getOutputFlatIndex(coords : i32) -> i32 {
          return coords;
        }
        `;
      break;
    case 2:
      snippet += `
        fn getOutputFlatIndex(coords : vec2<i32>) -> i32 {
          return i32(dot(vec2<f32>(coords), vec2<f32>(f32(uniforms.outShapeStrides), 1.0)));
        }
        `;
      break;
    case 3:
      snippet += `
        fn getOutputFlatIndex(coords : vec3<i32>) -> i32 {
          return i32(dot(vec3<f32>(coords), vec3<f32>(f32(uniforms.outShapeStrides.x), f32(uniforms.outShapeStrides.y), 1.0)));
        }
        `;
      break;
    case 4:
      snippet += `
        fn getOutputFlatIndex(coords : vec4<i32>) -> i32 {
          return i32(dot(vec4<f32>(coords), vec4<f32>(
            f32(uniforms.outShapeStrides.x), f32(uniforms.outShapeStrides.y), f32(uniforms.outShapeStrides.z), 1.0)));
        }
        `;
      break;
    default:
      util.assert(false, () => `Unsupported ${outRank}D shape`);
      break;
  }
  return snippet;
}
function getSetOutputSnippet(
    outShape: number[], outBufferType: DataType, isVec4: boolean): string {
  const outRank = outShape.length;
  const wgslType = mapToWgslTypes(outBufferType, isVec4);
  let snippet;
  if (isVec4) {
    snippet = `fn setOutputFlat(flatIndex : i32, value : vec4<f32>) {
      result.numbers[flatIndex] = ${wgslType}(value);
    }
    fn setOutputFlatI32(flatIndex : i32, value : vec4<i32>) {
      result.numbers[flatIndex] = ${wgslType}(value);
    }`;
  } else {
    snippet = `fn setOutputFlat(flatIndex : i32, value : f32) {
      result.numbers[flatIndex] = ${wgslType}(value);
    }
    fn setOutputFlatI32(flatIndex : i32, value : i32) {
      result.numbers[flatIndex] = ${wgslType}(value);
    }`;
  }
  if (outRank >= 2) {
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
    const type = getCoordsDataType(outRank);

    if (isVec4) {
      snippet += `
      fn setOutput(${
          dims.map(d => `${d} : i32`).join(', ')}, value : vec4<f32>) {
        let flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutputFlat(flatIndex / 4, value);
      }
      fn setOutputI32(${
          dims.map(d => `${d} : i32`).join(', ')}, value : vec4<i32>) {
        let flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutputFlatI32(flatIndex / 4, value);
      }
    `;
    } else {
      snippet += `
      fn setOutput(${dims.map(d => `${d} : i32`).join(', ')}, value : f32) {
        let flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutputFlat(flatIndex, value);
      }
      fn setOutputI32(${dims.map(d => `${d} : i32`).join(', ')}, value : i32) {
        let flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutputFlatI32(flatIndex, value);
      }
    `;
    }
  }

  return snippet;
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShape: number[], isVec4: boolean,
    isFlatDispatchLayout: boolean): string {
  let res = getSamplerFromInInfo(inInfo, isVec4);

  const inShape = inInfo.shape;
  if (inShape.length <= outShape.length) {
    res += getSamplerAtOutputCoords(
        inInfo, outShape, isVec4, isFlatDispatchLayout);
  }

  return res;
}

function getSamplerFromInInfo(inInfo: InputInfo, isVec4: boolean): string {
  const texName = inInfo.name;
  const rank = inInfo.shape.length;
  const type = getCoordsDataType(rank);
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, rank);
  const inputs = dims.map(d => `${d} : i32`).join(', ');

  if (rank < 1) {
    if (isVec4) {
      return `
        fn ${funcName}() -> vec4<f32> {
          return vec4<f32>(${texName}.numbers[0]);
        }
      `;
    }

    return `
      fn ${funcName}() ->f32 {
        return f32(${texName}.numbers[0]);
      }
    `;
  }

  const shapeStr =
      `uniforms.${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;
  let rankStr = `${rank}D`;
  if (rank === 0) {
    rankStr = '1D';
  }

  if (isVec4) {
    return `
      fn ${funcName}(${inputs}) -> vec4<f32> {
        return vec4<f32>(${texName}.numbers[getFlatIndex${rankStr}(${type}(${
        dims.join(',')}),
          ${shapeStr}) / 4]);
      }
      `;
  }

  return `
    fn ${funcName}(${inputs}) -> f32 {
      return f32(${texName}.numbers[getFlatIndex${rankStr}(${type}(${
      dims.join(',')}),
        ${shapeStr})]);
    }
   `;
}
// TODO: Implement getXXXFromFlatIndex, use it instead of getXXXAtOutCoords when
// it's flat dispatch layout.
export function getSamplerAtOutputCoords(
    inInfo: InputInfo, outShape: number[], isVec4: boolean,
    isFlatDispatchLayout: boolean): string {
  const texName = inInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);

  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';

  const inRank = inInfo.shape.length;
  const outRank = outShape.length;
  const type = getCoordsDataType(outRank);

  // If the inShape equals the outShape and the dispatch layout is flat, we can
  // directly use |gl_GlobalInvocationID.x| as the index and don't need coords
  // conversion between these two shapes.
  if (util.arraysEqual(inInfo.shape, outShape) && isFlatDispatchLayout) {
    if (isVec4) {
      return `
        fn ${
          funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : i32) -> vec4<f32> {
          return vec4<f32>(${texName}.numbers[globalIndex]);
        }

        fn ${funcName}ByCoords(coords : ${type}) -> vec4<f32> {
          return vec4<f32>(${texName}.numbers[${
          outRank > 1 ? 'getOutputFlatIndex(coords)' : 'coords'} / 4]);
        }
        `;
    } else {
      return `
      fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : i32) -> f32 {
        return f32(${texName}.numbers[globalIndex]);
      }

      fn ${funcName}ByCoords(coords : ${type}) -> f32 {
        return f32(${texName}.numbers[${
          outRank > 1 ? 'getOutputFlatIndex(coords)' : 'coords'}]);
      }
      `;
    }
  }

  const broadcastDims = backend_util.getBroadcastDims(inInfo.shape, outShape);
  const rankDiff = outRank - inRank;

  let coordsSnippet = '';

  if (inRank === 0) {
    if (isVec4) {
      return `
      fn ${
          funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : i32) -> vec4<f32> {
        return get${texFuncSnippet}();
      }

      fn ${funcName}ByCoords(coords : ${type}) -> vec4<f32> {
        return get${texFuncSnippet}();
      }
    `;
    }
    return `
      fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : i32) -> f32{
        return get${texFuncSnippet}();
      }

      fn ${funcName}ByCoords(coords : ${type}) -> f32{
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

  const shapeStr =
      `uniforms.${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;
  const rankStr = `${inRank}D`;
  if (isVec4) {
    return `
      fn ${
        funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : i32) -> vec4<f32> {
        var coords = getOutputCoords(globalId, globalIndex);
        ${coordsSnippet}
        return ${texName}.numbers[getFlatIndex${rankStr}(${
        unpackedCoordsSnippet}, ${shapeStr}) / 4];
      }

      fn ${funcName}ByCoords(coordsIn : ${type}) -> vec4<f32> {
        var coords = coordsIn;
        ${coordsSnippet}
        return ${texName}.numbers[getFlatIndex${rankStr}(${
        unpackedCoordsSnippet}, ${shapeStr}) / 4];
      }
    `;
  }

  return `
    fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : i32) -> f32 {
      var coords = getOutputCoords(globalId, globalIndex);
      ${coordsSnippet}
      return f32(${texName}.numbers[getFlatIndex${rankStr}(${
      unpackedCoordsSnippet}, ${shapeStr})]);
    }

    fn ${funcName}ByCoords(coordsIn : ${type}) -> f32 {
      var coords = coordsIn;
      ${coordsSnippet}
      return f32(${texName}.numbers[getFlatIndex${rankStr}(${
      unpackedCoordsSnippet}, ${shapeStr})]);
    }
  `;
}

/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
export function generateGetOutputCoords(
    outShape: number[],
    dispatchLayout: {x: number[], y?: number[], z?: number[]}):
    [string, number] {
  const {x, y = [], z = []} = dispatchLayout;

  const outRank = outShape.length;
  if (x.length === outRank) {
    const dtype = getCoordsDataType(outRank);
    const snippet =
        `fn getOutputCoords(globalId : vec3<u32>, globalIndex : i32) -> ${
            dtype}{
      return getCoordsFromFlatIndex(i32(globalIndex));
    }
    `;
    return [snippet, outRank];
  }

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
      gatherDimensionsStr += `let d${arr[0]} = i32(globalId[${i}]);`;
    } else {
      const strides = symbolicallyComputeStrides(arr, 'uniforms.outShape');
      gatherDimensionsStr += `var index${i} = i32(globalId[${i}]);`;
      for (let j = 0; j < strides.length; j++) {
        gatherDimensionsStr += `let d${arr[j]} = index${i} / ${strides[j]};`;

        if (j === strides.length - 1) {
          gatherDimensionsStr += `let d${arr[j + 1]} = ` +
              `index${i} - d${arr[j]} * ${strides[j]};`;
        } else {
          gatherDimensionsStr +=
              `index${i} = index${i} - d${arr[j]} * ${strides[j]};`;
        }
      }
    }
  }

  const dimensions = [];
  for (let i = 0; i < rank; i++) {
    dimensions.push(`d${i}`);
  }

  const dtype = getCoordsDataType(rank);
  let snippet =
      `fn getOutputCoords(globalId : vec3<u32>, globalIndex : i32) -> ${dtype} {
    ${gatherDimensionsStr}
  `;
  if (dimensions.length === 0) {
    snippet += `return ${dtype}(0); }`;
  } else {
    snippet += `return ${dtype}(${dimensions.join(',')}); }`;
  }

  return [snippet, rank];
}

/**
 * Derives logical coordinates from a flat index. Performs integer division
 * with each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
function generateGetCoordsFromFlatIndex(shape: number[]): string {
  const rank = shape.length;

  if (rank <= 1) {
    return `fn getCoordsFromFlatIndex(index : i32) -> i32 { return index; }`;
  }

  const strides = util.computeStrides(shape);
  const dtype = getCoordsDataType(rank);

  const coords: string[] = [];
  for (let i = 0; i < rank; i++) {
    coords.push(`d${i}`);
  }

  if (strides.length === 1) {
    return `    fn getCoordsFromFlatIndex(index : i32) -> vec2<i32> {
      let d0 = index / uniforms.outShapeStrides; let d1 = index - d0 * uniforms.outShapeStrides;
      return vec2<i32>(d0, d1);
    }`;
  }
  const snippet = 'var index2 = index;' +
      strides
          .map((_, i) => {
            const line1 =
                `let ${coords[i]} = index2 / uniforms.outShapeStrides[${i}]`;
            const line2 = i === strides.length - 1 ?
                `let ${coords[i + 1]} = index2 - ${
                    coords[i]} * uniforms.outShapeStrides[${i}]` :
                `index2 = index2 - ${coords[i]} * uniforms.outShapeStrides[${
                    i}]`;
            return `${line1}; ${line2};`;
          })
          .join('');

  return `
    fn getCoordsFromFlatIndex(index : i32) -> ${dtype} {
      ${snippet}
      return ${dtype}(${coords.join(',')});
    }
  `;
}
