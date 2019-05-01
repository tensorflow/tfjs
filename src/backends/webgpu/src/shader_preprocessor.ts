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
}

interface ProgramParams {
  tileSize?: [number, number?, number?];
  variableNames: string[];
  uniforms?: string;
  userCode: string;
}

export function makeShader(
    inputTypes: Array<{dtype: DataType, shape: number[]}>,
    outputData: {dtype: DataType, shape: number[]},
    program: ProgramParams): string {
  const prefixSnippets: string[] = [];

  if (program.tileSize != null) {
    const ts = program.tileSize;

    ts[1] = ts[1] || 1;
    ts[2] = ts[2] || 1;
    prefixSnippets.push(`
      const uvec3 TileSize = uvec3(${ts[0]}, ${ts[1]}, ${ts[2]});
      layout (local_size_x = TileSize.x,
              local_size_y = TileSize.y,
              local_size_z = TileSize.z) in;
    `);
  }

  // Output buffer.
  prefixSnippets.push(`
    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      float result[];
    };
  `);

  program.variableNames.forEach((x, i) => {
    prefixSnippets.push(`
      layout(std430, set = 0, binding = ${1 + i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputTypes[i].dtype)} ${x}[];
      };
    `);
  });

  if (program.uniforms) {
    prefixSnippets.push(`
      layout(std140, set = 0, binding = ${
        1 + program.variableNames.length}) uniform Uniforms {
        ${program.uniforms}
      };
    `);
  }

  const outputSamplingSnippet = getOutputSamplingSnippet(outputData.shape);

  const source = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS,
    outputSamplingSnippet, SET_OUTPUT_SNIPPET, program.userCode
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

  uint getFlatIndex(ivec4 coords, ivec4 shape) {
    return uint(dot(coords, ivec4(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
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
    case 4:
      return getOutput4DCoords(outShape as [number, number, number, number]);
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

function getOutput4DCoords(shape: [number, number, number, number]) {
  const strides = util.computeStrides(shape);

  return `ivec4 getOutputCoords(uint index) {
    uint d1 = index / ${strides[0]};
    index -= d1 * ${strides[0]};
    uint d2 = index / ${strides[1]};
    index -= d2 * ${strides[1]};
    uint r = index / ${strides[2]};
    uint c = index - r * ${strides[2]};

    return ivec4(d1, d2, r, c);
  }`;
}
