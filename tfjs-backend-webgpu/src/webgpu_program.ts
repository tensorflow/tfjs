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

import {backend_util, DataType, Rank, ShapeMap, TensorInfo, util} from '@tensorflow/tfjs-core';

import {symbolicallyComputeStrides} from './shader_util';

export interface WebGPUProgram {
  // The unique key to distinguish different shader source code.
  shaderKey: string;
  outputShape: number[];
  // dispatchLayout enumerates how tensor dimensions are distributed among
  // dispatch x,y,z dimensions.
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  // dispatch specifies geometry of thread groups - derived from dispatchLayout.
  dispatch: [number, number, number];
  variableNames: string[];
  // Describe each variable's type and must have one-one mapping with
  // variableNames. If not set, all variables type will be either f32 or
  // vec4<f32> based on isVec4 member.
  variableTypes?: string[];
  uniforms?: string;
  // Size of register cache in one dimension (assumes square cache).
  // Each thread writes to workPerThread * workPerThread locations in the output
  // buffer.
  workPerThread?: number;
  // workGroupSize.x * workGroupSize.y * workGroupSize.z = the number of threads
  // in a thread group. Individual dimensions determines thread layout within
  // the group.
  workGroupSize: [number, number, number];
  isVec4?: boolean;
  // Whether to use output size for bounds checking.
  size?: boolean;
  // Whether to use atomic built-in functions.
  atomic?: boolean;
  getUserCode: () => string;
}

export const compileProgram =
    (device: GPUDevice, program: WebGPUProgram,
     pipelineLayout: GPUPipelineLayout, inputsData: InputInfo[],
     output: TensorInfo, isFromPixel = false): GPUComputePipeline => {
      const outputData = {dtype: output.dtype, shape: output.shape};
      const source = makeShader(inputsData, outputData, program, isFromPixel);
      const module = device.createShaderModule(
          {code: source, label: program.constructor.name});
      const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {module, entryPoint: 'main'},
        label: program.constructor.name
      });

      return pipeline;
    };

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'i32';
  } else if (rank === 2) {
    return `vec2<i32>`;
  } else if (rank === 3) {
    return `vec3<i32>`;
  } else if (rank === 4) {
    return `vec4<i32>`;
  } else if (rank === 5) {
    return `vec5`;
  } else if (rank === 6) {
    return `vec6`;
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

export function getCoordsXYZ(index: number): string {
  if (index === 0) {
    return 'x';
  } else if (index === 1) {
    return 'y';
  } else if (index === 2) {
    return 'z';
  } else if (index === 3) {
    return 'w';
  } else if (index === 4) {
    return 'u';
  } else if (index === 5) {
    return 'v';
  } else {
    throw Error(`Index ${index} is not yet supported`);
  }
}

export function getMainHeaderAndGlobalIndexString(): string {
  return `
    ${getMainHeaderString()}
      let index = getGlobalIndex();
`;
}

export function getMainHeaderString(): string {
  return `
  ${getWorkGroupSizeString()}
  fn main(@builtin(local_invocation_id) LocalId : vec3<u32>,
          @builtin(global_invocation_id) GlobalId : vec3<u32>,
          @builtin(num_workgroups) NumWorkgroups: vec3<u32>) {
    localId = LocalId;
    globalId = GlobalId;
    numWorkgroups = NumWorkgroups;
`;
}

export function getWorkGroupSizeString(): string {
  return `
  @stage(compute) @workgroup_size(workGroupSizeX, workGroupSizeY, workGroupSizeZ)
`;
}

export function makeShader(
    inputInfo: InputInfo[], outputData: {dtype: DataType, shape: number[]},
    program: WebGPUProgram, isFromPixel = false): string {
  const prefixSnippets: string[] = [];
  prefixSnippets.push(`
      let workGroupSizeX = ${program.workGroupSize[0]}u;
      let workGroupSizeY = ${program.workGroupSize[1]}u;
      let workGroupSizeZ = ${program.workGroupSize[2]}u;

      var<private> localId: vec3<u32>;
      var<private> globalId: vec3<u32>;
      var<private> numWorkgroups: vec3<u32>;

      // Only used when the y/z dimension of workgroup size is 1.
      fn getGlobalIndex() -> i32 {
        ${
      isFlatDispatch(program) ?
          `  return i32(globalId.x);` :
          `  let localInvocationIndex = localId.z * workGroupSizeX * workGroupSizeY +
                   localId.y * workGroupSizeX + localId.x;
               let workGroupID = (globalId - localId)/vec3<u32>(
                   workGroupSizeX, workGroupSizeY, workGroupSizeZ);

               return i32((workGroupID.z * numWorkgroups.x * numWorkgroups.y +
                   workGroupID.y * numWorkgroups.x + workGroupID.x) *
                   (workGroupSizeX * workGroupSizeY * workGroupSizeZ) +
                   localInvocationIndex);
        `}
      }
    `);

  if (isFromPixel === true) {
    prefixSnippets.push(`
        struct Uniform {
          size            : i32,
          numChannels     : i32,
          outShapeStrides : vec2<i32>,
          dispatchSize    : vec3<u32>,
        };

        @group(0) @binding(0) var<storage, write> result: array<${
        mapToWgslTypes(outputData.dtype, program.isVec4)}>;
        @group(0) @binding(2) var<uniform> uniforms: Uniform;
      `);
    return [
      commonSnippet,
      prefixSnippets.join('\n'),
      getCoordsFromIndexSnippet(outputData.shape),
      program.getUserCode(),
    ].join('\n');
  }

  let preMemberIsStruct = false;
  let currentMemberIsStruct = false;
  let uniformDeclaration = 'struct Uniforms { NAN : f32, ';
  program.variableNames.forEach((x, i) => {
    const perDataType = getCoordsDataType(inputInfo[i].shape.length);
    if (perDataType === 'vec5' || perDataType === 'vec6') {
      currentMemberIsStruct = true;
    }
    if (preMemberIsStruct || currentMemberIsStruct) {
      uniformDeclaration += `@align(16) `;
    }
    preMemberIsStruct = currentMemberIsStruct;
    uniformDeclaration +=
        `${x.charAt(0).toLowerCase() + x.slice(1)}Shape : ${perDataType}, `;
  });
  const outputDataType = getCoordsDataType(outputData.shape.length);
  currentMemberIsStruct =
      outputDataType === 'vec5' || outputDataType === 'vec6';
  if (preMemberIsStruct || currentMemberIsStruct) {
    uniformDeclaration += `@align(16) `;
  }
  preMemberIsStruct = currentMemberIsStruct;
  uniformDeclaration += `outShape : ${outputDataType}, `;
  const stridesLength = outputData.shape.length - 1;
  const stridesDataType = getCoordsDataType(stridesLength);
  currentMemberIsStruct =
      stridesDataType === 'vec5' || stridesDataType === 'vec6';
  if (preMemberIsStruct || currentMemberIsStruct) {
    uniformDeclaration += `@align(16) `;
  }
  preMemberIsStruct = currentMemberIsStruct;
  uniformDeclaration += `
         outShapeStrides: ${stridesDataType}, `;

  if (program.size) {
    if (preMemberIsStruct) {
      uniformDeclaration += `@align(16) `;
    }
    preMemberIsStruct = false;
    uniformDeclaration += 'size : i32, ';
  }

  if (program.uniforms) {
    if (preMemberIsStruct) {
      uniformDeclaration += `@align(16) `;
    }
    uniformDeclaration += program.uniforms;
  }
  uniformDeclaration += '};';

  prefixSnippets.push(uniformDeclaration);

  // Output buffer.
  if (program.atomic) {
    prefixSnippets.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<atomic<i32>>;
    `);
  } else {
    prefixSnippets.push(`
      @group(0) @binding(0) var<storage, write> result: array<${
        mapToWgslTypes(outputData.dtype, program.isVec4)}>;
    `);
  }
  program.variableNames.forEach((x, i) => {
    prefixSnippets.push(`
      @group(0) @binding(${1 + i}) var<storage, read> ${x}: array<${
        program.variableTypes ?
            program.variableTypes[i] :
            mapToWgslTypes(inputInfo[i].dtype, program.isVec4)}>;
        `);
  });

  if (uniformDeclaration !== '') {
    prefixSnippets.push(`
      @group(0) @binding(${
        1 + program.variableNames.length}) var<uniform> uniforms: Uniforms;
      `);
  }

  const [coordsSnippet, dispatchLayoutRank] =
      getOutputCoordsSnippet(outputData.shape, program.dispatchLayout);

  const sources = [
    commonSnippet, prefixSnippets.join('\n'),
    getCoordsFromIndexSnippet(outputData.shape), coordsSnippet,
    getOutputIndexFromCoordsSnippet(outputData.shape.length)
  ];
  if (!program.atomic) {
    sources.push(
        setOutputSnippet(outputData.shape, outputData.dtype, program.isVec4));
  }
  if (dispatchLayoutRank === outputData.shape.length) {
    // Input snippet is only meaningful when the output isn't getting
    // implicitly reshaped (like it does in conv2d_matmul).
    const inputSnippet =
        inputInfo
            .map(
                (x, i) => getInputSnippet(
                    x, outputData.shape,
                    program.variableTypes ?
                        (program.variableTypes[i] === 'vec4<f32>') :
                        program.isVec4,
                    program.dispatchLayout.x.length ===
                        outputData.shape.length))
            .join('\n');
    sources.push(inputSnippet);
  }

  sources.push(program.getUserCode());
  const source = sources.join('\n');
  return source;
}

export function makeShaderKey<R extends Rank>(
    program: WebGPUProgram, shapes: Array<ShapeMap[R]>, types: string[] = [],
    broadcastDimsKey = '', inputShapesEqualsOutShape = ''): string {
  const flatDispatchString = isFlatDispatch(program) ? 'flatDispatch' : '';
  const key = program.shaderKey + '_' +
      (program.workGroupSize ? program.workGroupSize.join(',') : '') +
      shapes.map(shape => shape.length).join(',') + types.join(',') +
      program.variableNames.join(',') + broadcastDimsKey +
      inputShapesEqualsOutShape + flatDispatchString;
  return key;
}

const commonSnippet = `
  struct vec5 {x: i32, y: i32, z: i32, w: i32, u: i32};
  struct vec6 {x: i32, y: i32, z: i32, w: i32, u: i32, v: i32};

  // Checks whether coordinates lie within the bounds of the shape.
  fn coordsInBounds2D(coord : vec2<i32>, shape : vec2<i32>) -> bool {
    return all(coord >= vec2<i32>(0)) && all(coord < shape);
  }
  fn coordsInBounds3D(coord : vec3<i32>, shape : vec3<i32>) -> bool {
    return all(coord >= vec3<i32>(0)) && all(coord < shape);
  }
  fn coordsInBounds4D(coord : vec4<i32>, shape : vec4<i32>) -> bool {
    return all(coord >= vec4<i32>(0)) && all(coord < shape);
  }

  fn getIndexFromCoords1D(coord : i32, shape : i32) -> i32 {
    return coord;
  }
  fn getIndexFromCoords2D(coords : vec2<i32>, shape : vec2<i32>) -> i32 {
    return dot(coords, vec2<i32>(shape.y, 1));
  }
  fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
    return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));
  }
  fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
    return dot(coords, vec4<i32>(
        shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
  }
  fn getIndexFromCoords5D(coords : vec5, shape : vec5) -> i32 {
    let shapeStrides: vec5 = vec5(shape.y * shape.z * shape.w * shape.u, shape.z * shape.w * shape.u, shape.w * shape.u, shape.u, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u;
  }
  fn getIndexFromCoords6D(coords : vec6, shape : vec6) -> i32 {
    let shapeStrides: vec6 = vec6(shape.y * shape.z * shape.w * shape.u * shape.v, shape.z * shape.w * shape.u * shape.v, shape.w * shape.u * shape.v, shape.u * shape.v, shape.v, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u + coords.v*shapeStrides.v;
  }

  fn idiv(a: i32, b: i32, sign: f32) -> i32 {
    var res: i32 = a / b;
    let mod: i32 = a % b;
    if (sign < 0. && mod != 0) {
      res = res - 1;
    }
    return res;
  }

  // NaN defination in IEEE 754-1985 is :
  //   - sign = either 0 or 1.
  //   - biased exponent = all 1 bits.
  //   - fraction = anything except all 0 bits (since all 0 bits represents infinity).
  // https://en.wikipedia.org/wiki/IEEE_754-1985#Representation_of_non-numbers
  fn isnan(val: f32) -> bool {
    let floatToUint: u32 = bitcast<u32>(val);
    return (floatToUint & 0x7fffffffu) > 0x7f800000u;
  }
  fn isnanVec4(val : vec4<f32>) -> vec4<bool> {
    return vec4<bool>(isnan(val[0]), isnan(val[1]), isnan(val[2]), isnan(val[3]));
  }
`;

type InputInfo = {
  dtype: DataType; shape: number[]; name: string;
};
type WGSLDataType = 'f32'|'i32'|'vec4<f32>'|'vec4<i32>'|'vec4<bool>';

/**
 * Derives logical coordinates from a flat index. Performs integer division
 * with each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
function getCoordsFromIndexSnippet(shape: number[]): string {
  const rank = shape.length;

  if (rank <= 1) {
    return `fn getCoordsFromIndex(index : i32) -> i32 { return index; }`;
  }

  const strides = util.computeStrides(shape);
  const dtype = getCoordsDataType(rank);

  const coords: string[] = [];
  for (let i = 0; i < rank; i++) {
    coords.push(`d${i}`);
  }

  if (strides.length === 1) {
    return `    fn getCoordsFromIndex(index : i32) -> vec2<i32> {
      let d0 = index / uniforms.outShapeStrides; let d1 = index - d0 * uniforms.outShapeStrides;
      return vec2<i32>(d0, d1);
    }`;
  }
  let snippet;
  snippet = 'var index2 = index;' +
      strides
          .map((_, i) => {
            const line1 =
                `let ${coords[i]} = index2 / uniforms.outShapeStrides.${
                    getCoordsXYZ(i)}`;
            const line2 = i === strides.length - 1 ?
                `let ${coords[i + 1]} = index2 - ${
                    coords[i]} * uniforms.outShapeStrides.${getCoordsXYZ(i)}` :
                `index2 = index2 - ${coords[i]} * uniforms.outShapeStrides.${
                    getCoordsXYZ(i)}`;
            return `${line1}; ${line2};`;
          })
          .join('');

  return `
    fn getCoordsFromIndex(index : i32) -> ${dtype} {
      ${snippet}
      return ${dtype}(${coords.join(',')});
    }
  `;
}

function getInputAtCoordsSnippet(
    inputInfo: InputInfo, isVec4: boolean): string {
  const texName = inputInfo.name;
  const rank = inputInfo.shape.length;
  const type = getCoordsDataType(rank);
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const dims = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'].slice(0, rank);
  const inputs = dims.map(d => `${d} : i32`).join(', ');

  if (rank < 1) {
    if (isVec4) {
      return `
        fn ${funcName}() -> vec4<f32> {
          return vec4<f32>(${texName}[0]);
        }
      `;
    }

    return `
      fn ${funcName}() ->f32 {
        return f32(${texName}[0]);
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
        return vec4<f32>(${texName}[getIndexFromCoords${rankStr}(${type}(${
        dims.join(',')}),
          ${shapeStr}) / 4]);
      }
      `;
  }

  return `
    fn ${funcName}(${inputs}) -> f32 {
      return f32(${texName}[getIndexFromCoords${rankStr}(${type}(${
      dims.join(',')}),
        ${shapeStr})]);
    }
   `;
}

function getInputByOutputSnippet(
    inputInfo: InputInfo, outShape: number[], isVec4: boolean,
    isFlatDispatchLayout: boolean): string {
  const texName = inputInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);

  const funcName = 'get' + texFuncSnippet + 'ByOutput';

  const inRank = inputInfo.shape.length;
  const outRank = outShape.length;
  const type = getCoordsDataType(outRank);

  // If the inShape equals the outShape and the dispatch layout is flat, we can
  // directly use |gl_GlobalInvocationID.x| as the index and don't need coords
  // conversion between these two shapes.
  if (util.arraysEqual(inputInfo.shape, outShape) && isFlatDispatchLayout) {
    if (isVec4) {
      return `
      fn ${funcName}Index(globalIndex : i32) -> vec4<f32> {
        return vec4<f32>(${texName}[globalIndex]);
      }

      fn ${funcName}Coords(coords : ${type}) -> vec4<f32> {
        return vec4<f32>(${texName}[${
          outRank > 1 ? 'getOutputIndexFromCoords(coords)' : 'coords'} / 4]);
      }
      `;
    } else {
      return `
    fn ${funcName}Index(globalIndex : i32) -> f32 {
      return f32(${texName}[globalIndex]);
    }

    fn ${funcName}Coords(coords : ${type}) -> f32 {
      return f32(${texName}[${
          outRank > 1 ? 'getOutputIndexFromCoords(coords)' : 'coords'}]);
    }
    `;
    }
  }

  const broadcastDims =
      backend_util.getBroadcastDims(inputInfo.shape, outShape);
  const rankDiff = outRank - inRank;

  let coordsSnippet = '';

  if (inRank === 0) {
    if (isVec4) {
      return `
    fn ${funcName}Index(globalIndex : i32) -> vec4<f32> {
      return get${texFuncSnippet}();
    }

    fn ${funcName}Coords(coords : ${type}) -> vec4<f32> {
      return get${texFuncSnippet}();
    }
  `;
    }
    return `
    fn ${funcName}Index(globalIndex : i32) -> f32{
      return get${texFuncSnippet}();
    }

    fn ${funcName}Coords(coords : ${type}) -> f32{
      return get${texFuncSnippet}();
    }
  `;
  } else {
    if (outRank < 2 && broadcastDims.length >= 1) {
      coordsSnippet = 'coords = 0;';
    } else {
      coordsSnippet =
          broadcastDims.map(d => `coords.${getCoordsXYZ(d + rankDiff)} = 0;`)
              .join('\n');
    }
  }

  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    if (outRank > 1) {
      const coordsType = getCoordsDataType(inRank);
      const coordsValues =
          inputInfo.shape.map((s, i) => `coords.${getCoordsXYZ(i + rankDiff)}`)
              .join(', ');
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
    fn ${funcName}Index(globalIndex : i32) -> vec4<f32> {
      var coords = getCoordsFromIndex(globalIndex);
      ${coordsSnippet}
      return ${texName}[getIndexFromCoords${rankStr}(${
        unpackedCoordsSnippet}, ${shapeStr}) / 4];
    }

    fn ${funcName}Coords(coordsIn : ${type}) -> vec4<f32> {
      var coords = coordsIn;
      ${coordsSnippet}
      return ${texName}[getIndexFromCoords${rankStr}(${
        unpackedCoordsSnippet}, ${shapeStr}) / 4];
    }
  `;
  }

  return `
  fn ${funcName}Index(globalIndex : i32) -> f32 {
    var coords = getCoordsFromIndex(globalIndex);
    ${coordsSnippet}
    return f32(${texName}[getIndexFromCoords${rankStr}(${
      unpackedCoordsSnippet}, ${shapeStr})]);
  }

  fn ${funcName}Coords(coordsIn : ${type}) -> f32 {
    var coords = coordsIn;
    ${coordsSnippet}
    return f32(${texName}[getIndexFromCoords${rankStr}(${
      unpackedCoordsSnippet}, ${shapeStr})]);
  }
`;
}

function getInputSnippet(
    inputInfo: InputInfo, outShape: number[], isVec4: boolean,
    isFlatDispatchLayout: boolean): string {
  let res = getInputAtCoordsSnippet(inputInfo, isVec4);

  const inShape = inputInfo.shape;
  if (inShape.length <= outShape.length) {
    res += getInputByOutputSnippet(
        inputInfo, outShape, isVec4, isFlatDispatchLayout);
  }

  return res;
}

/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
function getOutputCoordsSnippet(
    outShape: number[],
    dispatchLayout: {x: number[], y?: number[], z?: number[]}):
    [string, number] {
  const {x, y = [], z = []} = dispatchLayout;

  const outRank = outShape.length;
  if (x.length === outRank) {
    const dtype = getCoordsDataType(outRank);
    const snippet = `fn getOutputCoords() -> ${dtype}{
    let globalIndex = getGlobalIndex();
    return getCoordsFromIndex(globalIndex);
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
  let snippet = `fn getOutputCoords() -> ${dtype} {
  ${gatherDimensionsStr}
`;
  if (dimensions.length === 0) {
    snippet += `return ${dtype}(0); }`;
  } else {
    snippet += `return ${dtype}(${dimensions.join(',')}); }`;
  }

  return [snippet, rank];
}

function getOutputIndexFromCoordsSnippet(outRank: number) {
  let snippet = '';
  switch (outRank) {
    case 0:
    case 1:
      snippet += `
        fn getOutputIndexFromCoords(coords : i32) -> i32 {
          return coords;
        }
        `;
      break;
    case 2:
      snippet += `
        fn getOutputIndexFromCoords(coords : vec2<i32>) -> i32 {
          return dot(coords, vec2<i32>(uniforms.outShapeStrides, 1));
        }
        `;
      break;
    case 3:
      snippet += `
        fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
          return dot(coords, vec3<i32>(uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, 1));
        }
        `;
      break;
    case 4:
      snippet += `
        fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {
          return dot(coords, vec4<i32>(
            uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, uniforms.outShapeStrides.z, 1));
        }
        `;
      break;
    case 5:
      snippet += `
        fn getOutputIndexFromCoords(coords : vec5) -> i32 {
          return coords.x * uniforms.outShapeStrides.x +
              coords.y * uniforms.outShapeStrides.y +
              coords.z * uniforms.outShapeStrides.z +
              coords.w * uniforms.outShapeStrides.w +
              coords.u;
        }
        `;
      break;
    case 6:
      snippet += `
        fn getOutputIndexFromCoords(coords : vec6) -> i32 {
          return coords.x * uniforms.outShapeStrides.x +
              coords.y * uniforms.outShapeStrides.y +
              coords.z * uniforms.outShapeStrides.z +
              coords.w * uniforms.outShapeStrides.w +
              coords.u * uniforms.outShapeStrides.u +
              coords.v;
        }
        `;
      break;
    default:
      util.assert(false, () => `Unsupported ${outRank}D shape`);
      break;
  }
  return snippet;
}

function isFlatDispatch(program: WebGPUProgram): boolean {
  return program.dispatch[1] === 1 && program.dispatch[2] === 1;
}

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

function setOutputSnippet(
    outShape: number[], outBufferType: DataType, isVec4: boolean): string {
  const outRank = outShape.length;
  const wgslType = mapToWgslTypes(outBufferType, isVec4);
  let snippet;
  if (isVec4) {
    snippet = `fn setOutputAtIndex(flatIndex : i32, value : vec4<f32>) {
      result[flatIndex] = ${wgslType}(value);
    }
    fn setOutputAtIndexI32(flatIndex : i32, value : vec4<i32>) {
      result[flatIndex] = ${wgslType}(value);
    }`;
  } else {
    snippet = `fn setOutputAtIndex(flatIndex : i32, value : f32) {
      result[flatIndex] = ${wgslType}(value);
    }
    fn setOutputAtIndexI32(flatIndex : i32, value : i32) {
      result[flatIndex] = ${wgslType}(value);
    }`;
  }
  if (outRank >= 2) {
    const dims = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'].slice(0, outRank);
    const type = getCoordsDataType(outRank);

    if (isVec4) {
      snippet += `
      fn setOutputAtCoords(${
          dims.map(d => `${d} : i32`).join(', ')}, value : vec4<f32>) {
        let flatIndex = getOutputIndexFromCoords(${type}(${dims.join(', ')}));
        setOutputAtIndex(flatIndex / 4, value);
      }
      fn setOutputAtCoordsI32(${
          dims.map(d => `${d} : i32`).join(', ')}, value : vec4<i32>) {
        let flatIndex = getOutputIndexFromCoords(${type}(${dims.join(', ')}));
        setOutputAtIndexI32(flatIndex / 4, value);
      }
    `;
    } else {
      snippet += `
      fn setOutputAtCoords(${
          dims.map(d => `${d} : i32`).join(', ')}, value : f32) {
        let flatIndex = getOutputIndexFromCoords(${type}(${dims.join(', ')}));
        setOutputAtIndex(flatIndex, value);
      }
      fn setOutputAtCoordsI32(${
          dims.map(d => `${d} : i32`).join(', ')}, value : i32) {
        let flatIndex = getOutputIndexFromCoords(${type}(${dims.join(', ')}));
        setOutputAtIndexI32(flatIndex, value);
      }
    `;
    }
  }

  return snippet;
}
