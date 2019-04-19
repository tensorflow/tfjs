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

import {DataType, util} from '@tensorflow/tfjs-core';

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'uint';
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

type GLSLDataType = 'float'|'uint';
function mapToGlslTypes(type: DataType): GLSLDataType|DataType {
  if (type === 'float32') {
    return 'float';
  }
  if (type === 'int32') {
    return 'uint';
  }
  return type;
};

export function makeShader(
    inputTypes: Array<{dtype: DataType, shape: number[]}>,
    variableNames: string[], outputData: {dtype: DataType, shape: number[]},
    userCode: string, tileSize: number): string {
  let tileSizeSnippet: string;
  if (tileSize != null) {
    tileSizeSnippet = `const uint TileSize = ${tileSize};
    layout (local_size_x = TileSize, local_size_y = TileSize, 
      local_size_z = 1) in;`;
  }
  const prefixSnippets: string[] = [];
  variableNames.forEach((x, i) => {
    prefixSnippets.push(`
      layout(std430, set = 0, binding = ${i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputTypes[i].dtype)} ${x}[];
      };
    `);
  });

  // Output buffer.
  prefixSnippets.push(`
    layout(std430, set = 0, binding = ${
      variableNames.length}) writeonly buffer ssbOut {
      float result[];
    };
  `);

  const outputSamplingSnippet = getOutputSamplingSnippet(outputData.shape);

  const source = [
    SHADER_PREFIX, tileSizeSnippet, prefixSnippets.join('\n'),
    SAMPLING_SNIPPETS, outputSamplingSnippet, SET_OUTPUT_SNIPPET, userCode
  ].join('\n');

  return source;
}

const SHADER_PREFIX = `
  #version 450
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
`;

const SET_OUTPUT_SNIPPET = `
  void setOutput(uint flatIndex, float value) {
    result[flatIndex] = value;
  }
`;

function getOutputSamplingSnippet(outShape: number[]): string {
  switch (outShape.length) {
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutput1DCoords(outShape as [number]);
    case 2:
      return getOutput2DCoords(outShape as [number, number]);
    case 3:
      return getOutput3DCoords(outShape as [number, number, number]);
    default:
      throw new Error(
          `${outShape.length}-D output sampling is not yet supported`);
  }
}

function getOutputScalarCoords() {
  return `int getOutputCoords() {
    return 0;
  }`;
}

function getOutput1DCoords(shape: [number]) {
  return `uint getOutputCoords(uint index) {
    return index;
  }`;
}

function getOutput2DCoords(shape: [number, number]) {
  // TODO: See whether using a 2D/3D dispatch to avoid division would improve
  // performance.
  return `
    ivec2 getOutputCoords(uint index) {
      uint r = index / ${shape[1]};
      uint c = index - r * ${shape[1]};
      return ivec2(r, c);
    }
  `;
}

function getOutput3DCoords(shape: [number, number, number]) {
  const strides = util.computeStrides(shape);

  return `ivec3 getOutputCoords(uint index) {
    uint d = index / ${strides[0]};
    index -= d * ${strides[0]};
    uint r = index / ${strides[1]};
    uint c = index - r * ${strides[1]};

    return ivec3(d, r, c);
  }`;
}