/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {ENV} from '../../environment';
import * as broadcast_util from '../../ops/broadcast_util';
import * as util from '../../util';

import * as shader_util from './shader_compiler_util';

export type ShapeInfo = {
  logicalShape: number[],
  texShape: [number, number],
  isUniform: boolean,
  isPacked: boolean
};

export type InputInfo = {
  name: string,
  shapeInfo: ShapeInfo
};

export function makeShader(
    inputsInfo: InputInfo[], outputShape: ShapeInfo, userCode: string,
    broadcast: boolean, usesPackedTextures: boolean): string {
  let inputPrefixSnippet: string[]|string = inputsInfo.map(x => {
    const size = util.sizeFromShape(x.shapeInfo.logicalShape);
    if (x.shapeInfo.isUniform) {
      return `uniform float ${x.name}${size > 1 ? `[${size}]` : ''};`;
    }
    return `uniform sampler2D ${x.name};`;
  });
  inputPrefixSnippet = inputPrefixSnippet.join('\n');

  const inputSamplingSnippet =
      inputsInfo
          .map(
              x => getInputSamplingSnippet(
                  x, outputShape, broadcast, usesPackedTextures))
          .join('\n');
  const outTexShape = outputShape.texShape;
  let outputSamplingSnippet: string;
  let floatTextureSetOutputSnippet: string;
  let shaderPrefix = SHADER_PREFIX;

  if (outputShape.isPacked) {
    outputSamplingSnippet =
        getPackedOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
    floatTextureSetOutputSnippet = FLOAT_TEXTURE_SET_RGBA_SNIPPET;
  } else {
    outputSamplingSnippet =
        getOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
    floatTextureSetOutputSnippet = FLOAT_TEXTURE_SET_R_SNIPPET;
  }

  if (usesPackedTextures) {
    shaderPrefix += SHADER_PACKED_PREFIX;
  }

  const source = [
    shaderPrefix, FLOAT_TEXTURE_SAMPLE_SNIPPET, floatTextureSetOutputSnippet,
    inputPrefixSnippet, outputSamplingSnippet, inputSamplingSnippet, userCode
  ].join('\n');
  return source;
}

function getSamplerFromInInfo(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    case 0:
      return getSamplerScalar(inInfo);
    case 1:
      return getSampler1D(inInfo);
    case 2:
      return getSampler2D(inInfo);
    case 3:
      return getSampler3D(inInfo);
    case 4:
      return getSampler4D(inInfo);
    case 5:
      return getSampler5D(inInfo);
    case 6:
      return getSampler6D(inInfo);
    default:
      throw new Error(
          `${shape.length}-D input sampling` +
          ` is not yet supported`);
  }
}

function getPackedSamplerFromInInfo(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    case 0:
      return getPackedSamplerScalar(inInfo);
    case 1:
      return getPackedSampler1D(inInfo);
    case 2:
      return getPackedSampler2D(inInfo);
    case 3:
      return getPackedSampler3D(inInfo);
    case 4:
      return getPackedSampler4D(inInfo);
    default:
      throw new Error(
          `Packed ${shape.length}-D input sampling` +
          ` is not yet supported`);
  }
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShapeInfo: ShapeInfo, broadcast: boolean,
    usesPackedTextures = false): string {
  let res = getSamplerFlat(inInfo);
  if (usesPackedTextures) {
    res += getPackedSamplerFromInInfo(inInfo);
  } else {
    res += getSamplerFromInInfo(inInfo);
  }

  // If input and output have matching logical shapes, add
  // getTexNameAtOutCoord() method that samples the input textureSampler using
  // the output coordinates.
  if (broadcast ||
      util.arraysEqual(
          inInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape)) {
    if (usesPackedTextures) {
      res += getPackedSamplerAtOutputCoords(inInfo, outShapeInfo, broadcast);
    } else {
      res += getSamplerAtOutputCoords(inInfo, outShapeInfo, broadcast);
    }
  }
  return res;
}

function getPackedOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutputPacked1DCoords(outShape as [number], outTexShape);
    case 2:
      return getOutputPacked2DCoords(outShape as [number, number], outTexShape);
    case 3:
      return getOutputPacked3DCoords(
          outShape as [number, number, number], outTexShape);
    case 4:
      return getOutputPacked4DCoords(
          outShape as [number, number, number, number], outTexShape);
    default:
      throw new Error(
          `${outShape.length}-D packed output ` +
          `coordinate fetching is not yet supported`);
  }
}

function getOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutput1DCoords(outShape as [number], outTexShape);
    case 2:
      return getOutput2DCoords(outShape as [number, number], outTexShape);
    case 3:
      return getOutput3DCoords(
          outShape as [number, number, number], outTexShape);
    case 4:
      return getOutput4DCoords(
          outShape as [number, number, number, number], outTexShape);
    case 5:
      return getOutput5DCoords(
          outShape as [number, number, number, number, number], outTexShape);
    case 6:
      return getOutput6DCoords(
          outShape as [number, number, number, number, number, number],
          outTexShape);
    default:
      throw new Error(
          `${outShape.length}-D output sampling is not yet supported`);
  }
}

const SAMPLE_1D_SNIPPET = `
vec2 UVfrom1D(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_2D_SNIPPET = `
vec2 UVfrom2D(int texNumR, int texNumC, int numC, int row, int col) {
  int index = row * numC + col;
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_3D_SNIPPET = `
vec2 UVfrom3D(int texNumR, int texNumC, int stride0,
    int stride1, int row, int col, int depth) {
  // Explicitly use integer operations as dot() only works on floats.
  int index = row * stride0 + col * stride1 + depth;
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_4D_SNIPPET = `
vec2 UVfrom4D(int texNumR, int texNumC, int stride0,
    int stride1, int stride2, int row, int col, int depth,
    int depth2) {
  // Explicitly use integer operations as dot() only works on floats.
  int index = row * stride0 + col * stride1 + depth * stride2 + depth2;
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom4D(int texNumR, int texNumC, int texelsInBatch2,
    int texelsInBatch, int texelsInLogicalRow, int b2, int b,
    int row, int col) {
  int index = b2 * texelsInBatch2 + b * texelsInBatch +
    (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_5D_SNIPPET = `
vec2 UVfrom5D(int texNumR, int texNumC, int stride0,
    int stride1, int stride2, int stride3, int row, int col, int depth,
    int depth2, int depth3) {
  // Explicitly use integer operations as dot() only works on floats.
  int index = row * stride0 + col * stride1 +
              depth * stride2 + depth2 * stride3 + depth3;
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_6D_SNIPPET = `
vec2 UVfrom6D(int texNumR, int texNumC, int stride0,
    int stride1, int stride2, int stride3, int stride4,
    int row, int col, int depth, int depth2, int depth3, int depth4) {
  // Explicitly use integer operations as dot() only works on floats.
  int index = row * stride0 + col * stride1 + depth * stride2 + depth2 *
    stride3 + depth3 * stride4 + depth4;
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const FLOAT_TEXTURE_SAMPLE_SNIPPET = `
  float sampleTexture(sampler2D textureSampler, vec2 uv) {
    return texture2D(textureSampler, uv).r;
  }
`;

const FLOAT_TEXTURE_SET_R_SNIPPET = `
  void setOutput(float val) {
    gl_FragColor = vec4(val, 0, 0, 0);
  }
`;

const FLOAT_TEXTURE_SET_RGBA_SNIPPET = `
  void setOutput(vec4 val) {
    gl_FragColor = val;
  }
`;

let NAN_CHECKS = '';
if (ENV.get('PROD')) {
  NAN_CHECKS = `
    bool isNaN(float val) {
      return false;
    }

    bool hasNaN(vec4 values) {
      return false;
    }
  `;
} else {
  /**
   * Previous NaN check '(val < 0.0 || 0.0 < val || val == 0.0) ? false : true'
   * does not work on iOS 12
   */
  NAN_CHECKS = `
    bool isNaN(float val) {
      return (val < 1.0 || 0.0 < val || val == 0.0) ? false : true;
    }

    bool hasNaN(vec4 values) {
      return any(bvec4(
        isNaN(values.x),
        isNaN(values.y),
        isNaN(values.z),
        isNaN(values.w)
      ));
    }
  `;
}

const SHADER_PREFIX = `
  precision highp float;
  precision highp int;
  varying vec2 resultUV;
  const vec2 halfCR = vec2(0.5, 0.5);

  struct ivec5
  {
    int x;
    int y;
    int z;
    int w;
    int u;
  };

  struct ivec6
  {
    int x;
    int y;
    int z;
    int w;
    int u;
    int v;
  };

  ${NAN_CHECKS}

  float getNaN(vec4 values) {
    return dot(vec4(1), values);
  }

  int round(float value) {
    return int(floor(value + 0.5));
  }

  int imod(int x, int y) {
    return x - y * (x / y);
  }

  //Based on the work of Dave Hoskins
  //https://www.shadertoy.com/view/4djSRW
  #define HASHSCALE1 443.8975
  float random(float seed){
    vec2 p = resultUV * seed;
    vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
  }

  ${SAMPLE_1D_SNIPPET}
  ${SAMPLE_2D_SNIPPET}
  ${SAMPLE_3D_SNIPPET}
  ${SAMPLE_4D_SNIPPET}
  ${SAMPLE_5D_SNIPPET}
  ${SAMPLE_6D_SNIPPET}
`;

const SHADER_PACKED_PREFIX = `
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;

function getOutputScalarCoords() {
  return `
    int getOutputCoords() {
      return 0;
    }
  `;
}

function getOutputPacked1DCoords(
    shape: [number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  if (texShape[0] === 1) {
    return `
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${packedTexShape[1]}.0);
      }
    `;
  }

  if (texShape[1] === 1) {
    return `
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${packedTexShape[0]}.0);
      }
    `;
  }

  return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      return resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
    }
  `;
}

function getOutput1DCoords(
    shape: [number], texShape: [number, number]): string {
  if (texShape[0] === 1) {
    return `
      int getOutputCoords() {
        return int(resultUV.x * ${texShape[1]}.0);
      }
    `;
  }
  if (texShape[1] === 1) {
    return `
      int getOutputCoords() {
        return int(resultUV.y * ${texShape[0]}.0);
      }
    `;
  }
  return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      return resTexRC.x * ${texShape[1]} + resTexRC.y;
    }
  `;
}

function getOutputPacked3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const texelsInLogicalRow = Math.ceil(shape[2] / 2);
  const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[1] / 2);

  return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec3(b, r, c);
    }
  `;
}

function getOutput3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const coordsFromIndexSnippet =
      shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);

  return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec3(r, c, d);
    }
  `;
}

function getOutputPacked4DCoords(
    shape: [number, number, number, number],
    texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

  const texelsInLogicalRow = Math.ceil(shape[3] / 2);
  const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[2] / 2);
  const texelsInBatch2 = texelsInBatch * shape[1];

  return `
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      int b2 = index / ${texelsInBatch2};
      index -= b2 * ${texelsInBatch2};

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec4(b2, b, r, c);
    }
  `;
}

function getOutput4DCoords(
    shape: [number, number, number, number],
    texShape: [number, number]): string {
  const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(
      ['r', 'c', 'd', 'd2'], shape);

  return `
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec4(r, c, d, d2);
    }
  `;
}

function getOutput5DCoords(
    shape: [number, number, number, number, number],
    texShape: [number, number]): string {
  const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(
      ['r', 'c', 'd', 'd2', 'd3'], shape);

  return `
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${texShape[0]},
                             ${texShape[1]}));

      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      ${coordsFromIndexSnippet}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `;
}

function getOutput6DCoords(
    shape: [number, number, number, number, number, number],
    texShape: [number, number]): string {
  const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(
      ['r', 'c', 'd', 'd2', 'd3', 'd4'], shape);

  return `
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      ${coordsFromIndexSnippet}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `;
}

function getOutputPacked2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  if (util.arraysEqual(shape, texShape)) {
    return `
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${packedTexShape[0]}, ${
        packedTexShape[1]}));
      }
    `;
  }

  // texels needed to accommodate a logical row
  const texelsInLogicalRow = Math.ceil(shape[1] / 2);

  /**
   * getOutputCoords
   *
   * resTexRC: The rows and columns of the texels. If you move over one
   * texel to the right in the packed texture, you are moving over one column
   * (not two).
   *
   * index: The texel index
   */
  return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));

      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec2(r, c);
    }
  `;
}

function getOutput2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  if (util.arraysEqual(shape, texShape)) {
    return `
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${texShape[0]}, ${texShape[1]}));
      }
    `;
  }
  if (shape[1] === 1) {
    return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `;
  }
  if (shape[0] === 1) {
    return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `;
  }
  return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      int r = index / ${shape[1]};
      int c = index - r * ${shape[1]};
      return ivec2(r, c);
    }
  `;
}

function getPackedSamplerScalar(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  return `
    vec4 ${funcName}() {
      return texture2D(${texName}, halfCR);
    }
  `;
}

function getSamplerScalar(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  if (inputInfo.shapeInfo.isUniform) {
    return `float ${funcName}() {return ${texName};}`;
  }
  return `
    float ${funcName}() {
      return sampleTexture(${texName}, halfCR);
    }
  `;
}

function getPackedSampler1D(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

  return `
    vec4 ${funcName}(int index) {
      vec2 uv = packedUVfrom1D(
        ${packedTexShape[0]}, ${packedTexShape[1]}, index);
      return texture2D(${texName}, uv);
    }
  `;
}

function getSampler1D(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);

  return `
    float ${funcName}(int index) {
      return ${funcName}Flat(index);
    }
  `;
}

function getPackedSampler2D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;

  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texShape != null && util.arraysEqual(shape, texShape)) {
    return `
      vec4 ${funcName}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);

        return texture2D(${texName}, uv);
      }
    `;
  }

  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const valuesPerRow = Math.ceil(shape[1] / 2);

  return `
    vec4 ${funcName}(int row, int col) {
      vec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${
      packedTexShape[1]}, row, col);
      return texture2D(${texName}, uv);
    }
  `;
}

function getSampler2D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);

  const texShape = inputInfo.shapeInfo.texShape;
  if (texShape != null && util.arraysEqual(shape, texShape)) {
    const texNumR = texShape[0];
    const texNumC = texShape[1];
    return `
    float ${funcName}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
  }

  const {newShape, keptDims} = util.squeezeShape(shape);
  const squeezedShape = newShape;
  if (squeezedShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['row', 'col'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col) {
        float index = dot(vec2(row, col), vec2(${shape[1]}, 1));
        return ${funcName}Flat(round(index));
      }
    `;
  }

  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texNumC === 1) {
    // index is used directly as physical (no risk of float16 overflow).
    return `
    float ${funcName}(int row, int col) {
      float index = dot(vec2(row, col), vec2(${shape[1]}, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  if (texNumR === 1) {
    // index is used directly as physical (no risk of float16 overflow).
    return `
    float ${funcName}(int row, int col) {
      float index = dot(vec2(row, col), vec2(${shape[1]}, 1));
      vec2 uv = vec2((index + 0.5) / ${texNumC}.0, 0.5);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  return `
  float ${funcName}(int row, int col) {
    vec2 uv = UVfrom2D(${texNumR}, ${texNumC}, ${shape[1]}, row, col);
    return sampleTexture(${texName}, uv);
  }
`;
}

function getPackedSampler3D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

  if (shape[0] === 1) {
    const squeezedShape = shape.slice(1);
    const keptDims = [1, 2];
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['b', 'row', 'col'];
    return `
        ${getPackedSamplerFromInInfo(newInputInfo)}
        vec4 ${funcName}(int b, int row, int col) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
  }

  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[2] / 2);
  const texelsInBatch = valuesPerRow * Math.ceil(shape[1] / 2);

  return `
    vec4 ${funcName}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${valuesPerRow}, b, row, col);
      return texture2D(${texName}, uv);
    }
  `;
}

function getSampler3D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];

  const {newShape, keptDims} = util.squeezeShape(shape);
  const squeezedShape = newShape;
  if (squeezedShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['row', 'col', 'depth'];
    return `
        ${getSamplerFromInInfo(newInputInfo)}
        float ${funcName}(int row, int col, int depth) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth) {
        float index = dot(vec3(row, col, depth),
                          vec3(${stride0}, ${stride1}, 1));
        return ${funcName}Flat(round(index));
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texNumC === stride0) {
    // texC is used directly as physical (no risk of float16 overflow).
    return `
        float ${funcName}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${stride1}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${texNumC}.0, ${texNumR}.0);
          return sampleTexture(${texName}, uv);
        }
      `;
  }

  if (texNumC === stride1) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
    float ${funcName}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${shape[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
  }

  return `
      float ${funcName}(int row, int col, int depth) {
        vec2 uv = UVfrom3D(
            ${texNumR}, ${texNumC}, ${stride0}, ${stride1}, row, col, depth);
        return sampleTexture(${texName}, uv);
      }
  `;
}

function getPackedSampler4D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[3] / 2);
  const texelsInBatch = valuesPerRow * Math.ceil(shape[2] / 2);
  const texelsInBatch2 = texelsInBatch * shape[1];

  return `
    vec4 ${funcName}(int b2, int b, int row, int col) {
      vec2 uv = packedUVfrom4D(
        ${texNumR}, ${texNumC}, ${texelsInBatch2},
        ${texelsInBatch}, ${valuesPerRow}, b2, b, row, col);
      return texture2D(${texName}, uv);
    }
  `;
}

function getSampler4D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride2 = shape[3];
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;

  const {newShape, keptDims} = util.squeezeShape(shape);
  if (newShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, newShape);
    const params = ['row', 'col', 'depth', 'depth2'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth, int depth2) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float index = dot(vec4(row, col, depth, depth2),
                          vec4(${stride0}, ${stride1}, ${stride2}, 1));
        return ${funcName}Flat(round(index));
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texNumC === stride0) {
    // texC is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2), vec3(${stride1}, ${stride2}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (texNumC === stride2) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${shape[1] * shape[2]}, ${shape[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      vec2 uv = UVfrom4D(${texNumR}, ${texNumC}, ${stride0}, ${stride1},
          ${stride2}, row, col, depth, depth2);
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getSampler5D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride3 = shape[4];
  const stride2 = shape[3] * stride3;
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;

  const {newShape, keptDims} = util.squeezeShape(shape);
  if (newShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, newShape);
    const params = ['row', 'col', 'depth', 'depth2', 'depth3'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${stride0}, ${stride1}, ${stride2}, ${stride3})) +
          depth3;
        return ${funcName}Flat(index);
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];

  if (texNumC === stride0) {
    // texC is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(
          vec4(col, depth, depth2, depth3),
          vec4(${stride1}, ${stride2}, ${stride3}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }

  if (texNumC === stride3) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${shape[1] * shape[2] * shape[3]}, ${shape[2] * shape[3]},
            ${shape[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }

  return `
    float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
      vec2 uv = UVfrom5D(${texNumR}, ${texNumC}, ${stride0}, ${stride1},
          ${stride2}, ${stride3}, row, col, depth, depth2, depth3);
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getSampler6D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride4 = shape[5];
  const stride3 = shape[4] * stride4;
  const stride2 = shape[3] * stride3;
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;
  const {newShape, keptDims} = util.squeezeShape(shape);
  if (newShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, newShape);
    const params = ['row', 'col', 'depth', 'depth2', 'depth3', 'depth4'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${stride0}, ${stride1}, ${stride2}, ${stride3})) +
          dot(
            vec2(depth3, depth4),
            vec2(${stride4}, 1));
        return ${funcName}Flat(index);
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texNumC === stride0) {
    // texC is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(
          vec4(col, depth, depth2, depth3),
          vec4(${stride1}, ${stride2}, ${stride3}, ${stride4})) + depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (texNumC === stride4) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${shape[1] * shape[2] * shape[3] * shape[4]},
               ${shape[2] * shape[3] * shape[4]},
               ${shape[3] * shape[4]},
               ${shape[4]})) + depth3;
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      vec2 uv = UVfrom6D(${texNumR}, ${texNumC}, ${stride0}, ${stride1},
          ${stride2}, ${stride3}, ${stride4}
          ,row, col, depth, depth2, depth3, depth4);
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getSamplerFlat(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName =
      'get' + texName.charAt(0).toUpperCase() + texName.slice(1) + 'Flat';
  const inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);

  if (inputInfo.shapeInfo.isUniform) {
    if (inSize === 1) {
      return `float ${funcName}(int index) {return ${texName};}`;
    }
    return `
      float ${funcName}(int index) {
        for (int i = 0; i < ${inSize}; i++) {
          if (i == index) {
            return ${texName}[i];
          }
        }
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const tNumR = texShape[0];
  const tNumC = texShape[1];
  if (tNumC === 1 && tNumR === 1) {
    return `
      float ${funcName}(int index) {
        return sampleTexture(${texName}, halfCR);
      }
    `;
  }
  if (tNumC === 1) {
    return `
      float ${funcName}(int index) {
        vec2 uv = vec2(0.5, (float(index) + 0.5) / ${tNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (tNumR === 1) {
    return `
      float ${funcName}(int index) {
        vec2 uv = vec2((float(index) + 0.5) / ${tNumC}.0, 0.5);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int index) {
      vec2 uv = UVfrom1D(${tNumR}, ${tNumC}, index);
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getBroadcastOutputCoordsSampler(
    inputInfo: InputInfo, outShapeInfo: ShapeInfo, texFuncSnippet: string,
    funcName: string): string {
  const inRank = inputInfo.shapeInfo.logicalShape.length;
  const outRank = outShapeInfo.logicalShape.length;

  let type = 'int';
  if (outRank === 2) {
    type = 'ivec2';
  } else if (outRank === 3) {
    type = 'ivec3';
  } else if (outRank === 4) {
    type = 'ivec4';
  }
  const broadcastDims = broadcast_util.getBroadcastDims(
      inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
  const rankDiff = outRank - inRank;
  let coordsSnippet: string;
  if (inRank === 0) {
    coordsSnippet = '';
  } else if (outRank < 2 && broadcastDims.length >= 1) {
    coordsSnippet = 'coords = 0;';
  } else {
    coordsSnippet =
        broadcastDims.map(d => `coords[${d + rankDiff}] = 0;`).join('\n');
  }
  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
                                .map((s, i) => `coords[${i + rankDiff}]`)
                                .join(', ');
  }
  return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      return get${texFuncSnippet}(${unpackedCoordsSnippet});
    }
  `;
}

function getPackedSamplerAtOutputCoords(
    inputInfo: InputInfo, outShapeInfo: ShapeInfo,
    supportsBroadcasting: boolean) {
  const texName = inputInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
  const outTexShape = outShapeInfo.texShape;

  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const broadcastDims = broadcast_util.getBroadcastDims(
      inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
  const inRank = inputInfo.shapeInfo.logicalShape.length;
  const outRank = outShapeInfo.logicalShape.length;
  if (broadcastDims.length) {
    throw Error('Packed broadcast sampling is not implemented yet.');
  }

  const inTexShape = inputInfo.shapeInfo.texShape;
  if (util.arraysEqual(inTexShape, outTexShape)) {
    return `
      vec4 ${funcName}() {
        return texture2D(${texName}, resultUV);
      }
    `;
  }

  let output = `return texture2D(${texName}, uv)`;

  if (inRank === 1 && outRank > 1) {
    output = `
      vec4 sample = texture2D(${texName}, uv);
      return vec4(sample.xy, sample.xy);
    `;
  } else if (inRank === 0 && outRank > 0) {
    if (outRank === 1) {
      output = `
        vec4 sample = texture2D(${texName}, uv);
        return vec4(sample.x, sample.x, 0., 0.);
      `;
    } else {
      output = `
        vec4 sample = texture2D(${texName}, uv);
        return vec4(sample.x);
      `;
    }
  }

  return `
    vec4 ${funcName}() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      int texR = index / ${texNumC};
      int texC = index - texR * ${texNumC};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}, ${texNumR});

      ${output};
    }
  `;
}

function getSamplerAtOutputCoords(
    inputInfo: InputInfo, outShapeInfo: ShapeInfo,
    supportsBroadcasting: boolean) {
  const texName = inputInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';

  const broadcastDims = broadcast_util.getBroadcastDims(
      inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
  const inRank = inputInfo.shapeInfo.logicalShape.length;
  const outRank = outShapeInfo.logicalShape.length;
  const doBroadcast =
      supportsBroadcasting && ((outRank > inRank) || broadcastDims.length > 0);
  const broadcastOverOuter =
      broadcast_util.broadcastDimsAreOuter(broadcastDims);
  const isUniform = inputInfo.shapeInfo.isUniform;

  if (doBroadcast && !broadcastOverOuter) {
    return getBroadcastOutputCoordsSampler(
        inputInfo, outShapeInfo, texFuncSnippet, funcName);
  }

  const inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
  let broadcastSnippet = '';
  if (doBroadcast && broadcastOverOuter) {
    broadcastSnippet = `
        int mainPart = index / ${inSize};
        index -= mainPart * ${inSize};
      `;
  }

  const outTexShape = outShapeInfo.texShape;
  if (isUniform) {
    if (inSize === 1) {
      return `float ${funcName}() {return ${texName};}`;
    }
    return `
      float ${funcName}() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                              vec2(${outTexShape[0]}, ${outTexShape[1]}));
        int index = resTexRC.x * ${outTexShape[1]} + resTexRC.y;
        ${broadcastSnippet}
        return get${texFuncSnippet}Flat(index);
      }
    `;
  }

  // At this point, the input is not a uniform.
  const inTexShape = inputInfo.shapeInfo.texShape;
  if (util.arraysEqual(inTexShape, outTexShape)) {
    return `
      float ${funcName}() {
        return sampleTexture(${texName}, resultUV);
      }
    `;
  }

  return `
    float ${funcName}() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${outTexShape[0]}, ${outTexShape[1]}));
      int index = resTexRC.x * ${outTexShape[1]} + resTexRC.y;
      ${broadcastSnippet}
      int texR = index / ${inTexShape[1]};
      int texC = index - texR * ${inTexShape[1]};
      vec2 uv = (vec2(texC, texR) + halfCR) /
                 vec2(${inTexShape[1]}.0, ${inTexShape[0]}.0);

      return sampleTexture(${texName}, uv);
    }
  `;
}

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'int';
  } else if (rank === 2) {
    return 'ivec2';
  } else if (rank === 3) {
    return 'ivec3';
  } else if (rank === 4) {
    return 'ivec4';
  } else if (rank === 5) {
    return 'ivec5';
  } else if (rank === 6) {
    return 'ivec6';
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

/** Returns a new input info (a copy) that has a squeezed logical shape. */
function squeezeInputInfo(
    inInfo: InputInfo, squeezedShape: number[]): InputInfo {
  // Deep copy.
  const newInputInfo: InputInfo = JSON.parse(JSON.stringify(inInfo));
  newInputInfo.shapeInfo.logicalShape = squeezedShape;
  return newInputInfo;
}

function getSqueezedParams(params: string[], keptDims: number[]): string {
  return keptDims.map(d => params[d]).join(', ');
}