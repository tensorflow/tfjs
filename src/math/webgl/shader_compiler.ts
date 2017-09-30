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

import * as util from '../../util';

export type ShapeInfo = {
  logicalShape: number[],
  texShape: [number, number]
};

export type InputInfo = {
  name: string,
  shapeInfo: ShapeInfo
};

export function makeShader(
    inputsInfo: InputInfo[], outputShape: ShapeInfo, userCode: string,
    broadcast: boolean): string {
  const inputPrefixSnippet =
      inputsInfo.map(x => `uniform sampler2D ${x.name};`).join('\n');
  const inputSamplingSnippet =
      inputsInfo.map(x => getInputSamplingSnippet(x, outputShape, broadcast))
          .join('\n');
  const outTexShape = outputShape.texShape;
  const outputSamplingSnippet =
      getOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
  const source = [
    SHADER_PREFIX, inputPrefixSnippet, inputSamplingSnippet,
    outputSamplingSnippet, userCode
  ].join('\n');
  return source;
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShapeInfo: ShapeInfo, broadcast: boolean) {
  const shape = inInfo.shapeInfo.logicalShape;
  const texShape = inInfo.shapeInfo.texShape;
  const outTexShape = outShapeInfo.texShape;

  let res = '';
  switch (shape.length) {
    case 0:
      res += getSamplerScalar(inInfo.name);
      break;
    case 1:
      res += getSampler1D(inInfo.name, texShape);
      break;
    case 2:
      res += getSampler2D(inInfo.name, shape as [number, number], texShape);
      break;
    case 3:
      res += getSampler3D(
          inInfo.name, shape as [number, number, number], texShape);
      break;
    case 4:
      res += getSampler4D(
          inInfo.name, shape as [number, number, number, number], texShape);
      break;
    default:
      throw new Error(
          `${shape.length}-D input sampling` +
          ` is not yet supported`);
  }
  // If input and output have matching logical shapes, add
  // getTexNameAtOutCoord() method that samples the input texture using the
  // output coordinates.
  if (broadcast ||
      util.arraysEqual(
          inInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape)) {
    res +=
        getSamplerAtOutputCoords(inInfo.name, texShape, outTexShape, broadcast);
  }
  res += getSamplerFlat(inInfo.name, texShape);
  return res;
}

function getOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      // Doesn't make sense to call getOutputCoords() when output is scalar.
      return '';
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
`;

const SAMPLE_2D_SNIPPET = `
vec2 UVfrom2D(int texNumR, int texNumC, int numC, int row, int col) {
  int index = row * numC + col;
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_3D_SNIPPET = `
vec2 UVfrom3D(int texNumR, int texNumC, int stride0,
    int stride1, int row, int col, int depth) {
  int index = row * stride0 + col * stride1 + depth;
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_4D_SNIPPET = `
vec2 UVfrom4D(int texNumR, int texNumC, int stride0,
    int stride1, int stride2, int row, int col, int depth,
    int depth2) {
  int index = row * stride0 + col * stride1 + depth * stride2 + depth2;
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SHADER_PREFIX = `
  precision highp float;
  varying vec2 resultUV;
  const vec2 halfCR = vec2(0.5, 0.5);

  float sample(sampler2D texture, vec2 uv) {
    return texture2D(texture, uv).r;
  }

  void setOutput(float val) {
    gl_FragColor = vec4(val, 0, 0, 0);
  }

  bool isNaN(float val) {
    return val == val ? false : true;
  }

  bool hasNaN(vec4 values) {
    return any(notEqual(values, values));
  }

  float getNaN(vec4 values) {
    return dot(vec4(1), values);
  }

  int round(float value) {
    return int(floor(value + 0.5));
  }

  const vec2 randomConst = vec2(
    23.14069263277926, // e^pi (Gelfond's constant)
     2.665144142690225 // 2^sqrt(2) (Gelfond–Schneider constant)
  );

  float random(float seed) {
      return fract(cos(dot(resultUV * seed, randomConst)) * 12345.6789);
  }

  ${SAMPLE_1D_SNIPPET}
  ${SAMPLE_2D_SNIPPET}
  ${SAMPLE_3D_SNIPPET}
  ${SAMPLE_4D_SNIPPET}
`;

function getOutput1DCoords(
    shape: [number], texShape: [number, number]): string {
  if (texShape[0] === 1) {
    return `
      int getOutputCoords() {
        return int(gl_FragCoord.x);
      }
    `;
  }
  if (texShape[1] === 1) {
    return `
      int getOutputCoords() {
        return int(gl_FragCoord.y);
      }
    `;
  }
  return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(gl_FragCoord.yx);
      return resTexRC.x * ${texShape[1]} + resTexRC.y;
    }
  `;
}

function getOutput3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];
  return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(gl_FragCoord.yx);
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      int r = index / ${stride0};
      index -= r * ${stride0};
      int c = index / ${stride1};
      int d = index - c * ${stride1};
      return ivec3(r, c, d);
    }
  `;
}

function getOutput4DCoords(
    shape: [number, number, number, number],
    texShape: [number, number]): string {
  const stride2 = shape[3];
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;
  return `
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(gl_FragCoord.yx);
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      int r = index / ${stride0};
      index -= r * ${stride0};

      int c = index / ${stride1};
      index -= c * ${stride1};

      int d = index / ${stride2};
      int d2 = index - d * ${stride2};

      return ivec4(r, c, d, d2);
    }
  `;
}

function getOutput2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  if (util.arraysEqual(shape, texShape)) {
    return `
      ivec2 getOutputCoords() {
        return ivec2(gl_FragCoord.yx);
      }
    `;
  }
  if (shape[1] === 1) {
    return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(gl_FragCoord.yx);
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `;
  }
  if (shape[0] === 1) {
    return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(gl_FragCoord.yx);
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `;
  }
  return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(gl_FragCoord.yx);
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      int r = index / ${shape[1]};
      int c = index - r * ${shape[1]};
      return ivec2(r, c);
    }
  `;
}

function getSamplerScalar(texName: string): string {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  return `
    float ${funcName}() {
      return sample(${texName}, halfCR);
    }
  `;
}

function getSampler1D(texName: string, texShape: [number, number]): string {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const tR = texShape[0];
  const tC = texShape[1];
  if (texShape[0] === 1 && texShape[1] === 1) {
    return `
      float ${funcName}(int index) {
        return sample(${texName}, halfCR);
      }
    `;
  }
  if (texShape[1] === 1) {
    return `
      float ${funcName}(int index) {
        vec2 uv = vec2(0.5, (float(index) + 0.5) / ${tR}.0);
        return sample(${texName}, uv);
      }
    `;
  }
  if (texShape[0] === 1) {
    return `
      float ${funcName}(int index) {
        vec2 uv = vec2((float(index) + 0.5) / ${tC}.0, 0.5);
        return sample(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int index) {
      vec2 uv = UVfrom1D(${tR}, ${tC}, index);
      return sample(${texName}, uv);
    }
  `;
}

function getSampler3D(
    texName: string, shape: [number, number, number],
    texShape: [number, number]): string {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const tR = texShape[0];
  const tC = texShape[1];
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];
  if (tC === stride0) {
    return `
      float ${funcName}(int row, int col, int depth) {
        int texR = row;
        int texC = col * ${stride1} + depth;
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${tC}.0, ${tR}.0);
        return sample(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int row, int col, int depth) {
      vec2 uv = UVfrom3D(${tR}, ${tC}, ${stride0}, ${stride1}, row, col, depth);
      return sample(${texName}, uv);
    }
  `;
}

function getSampler4D(
    texName: string, shape: [number, number, number, number],
    texShape: [number, number]): string {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const tR = texShape[0];
  const tC = texShape[1];
  const stride2 = shape[3];
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;

  if (tC === stride0) {
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int texR = row;
        int texC = col * ${stride1} + depth * ${stride2} + depth2;
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${tC}.0, ${tR}.0);
        return sample(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      vec2 uv = UVfrom4D(${tR}, ${tC}, ${stride0}, ${stride1}, ${stride2},
          row, col, depth, depth2);
      return sample(${texName}, uv);
    }
  `;
}

function getSampler2D(
    texName: string, shape: [number, number],
    texShape: [number, number]): string {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const tR = texShape[0];
  const tC = texShape[1];
  if (util.arraysEqual(shape, texShape)) {
    return `
      float ${funcName}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${tC}.0, ${tR}.0);
        return sample(${texName}, uv);
      }
    `;
  }
  if (tC === 1) {
    if (shape[0] === 1) {
      return `
        float ${funcName}(int row, int col) {
          vec2 uv = vec2(0.5, (float(col) + 0.5) / ${tR}.0);
          return sample(${texName}, uv);
        }
      `;
    }
    if (shape[1] === 1) {
      return `
        float ${funcName}(int row, int col) {
          vec2 uv = vec2(0.5, (float(row) + 0.5) / ${tR}.0);
          return sample(${texName}, uv);
        }
      `;
    }
    return `
      float ${funcName}(int row, int col) {
        int index = row * ${shape[1]} + col;
        vec2 uv = vec2(0.5, (float(index) + 0.5) / ${tR}.0);
        return sample(${texName}, uv);
      }
    `;
  }
  if (tR === 1) {
    return `
      float ${funcName}(int row, int col) {
        int index = row * ${shape[1]} + col;
        vec2 uv = vec2((float(index) + 0.5) / ${tC}.0, 0.5);
        return sample(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int row, int col) {
      vec2 uv = UVfrom2D(${tR}, ${tC}, ${shape[1]}, row, col);
      return sample(${texName}, uv);
    }
  `;
}

function getSamplerFlat(texName: string, texShape: [number, number]): string {
  const funcName =
      'get' + texName.charAt(0).toUpperCase() + texName.slice(1) + 'Flat';
  const tNumR = texShape[0];
  const tNumC = texShape[1];
  if (tNumC === 1 && tNumR === 1) {
    return `
      float ${funcName}(int index) {
        return sample(${texName}, halfCR);
      }
    `;
  }
  if (tNumC === 1) {
    return `
      float ${funcName}(int index) {
        vec2 uv = vec2(0.5, (float(index) + 0.5) / ${tNumR}.0);
        return sample(${texName}, uv);
      }
    `;
  }
  if (tNumR === 1) {
    return `
      float ${funcName}(int index) {
        vec2 uv = vec2((float(index) + 0.5) / ${tNumC}.0, 0.5);
        return sample(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int index) {
      int texR = index / ${tNumC};
      int texC = index - texR * ${tNumC};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${tNumC}.0, ${tNumR}.0);
      return sample(${texName}, uv);
    }
  `;
}

function getSamplerAtOutputCoords(
    texName: string, inTexShape: [number, number],
    outTexShape: [number, number], broadcast: boolean) {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1) +
      'AtOutCoords';
  if (util.arraysEqual(inTexShape, outTexShape)) {
    return `
      float ${funcName}() {
        return sample(${texName}, resultUV);
      }
    `;
  }
  const inSize = util.sizeFromShape(inTexShape);
  let broadcastSnippet = '';
  if (broadcast) {
    broadcastSnippet = `
      int mainPart = index / ${inSize};
      index -= mainPart * ${inSize};
    `;
  }
  return `
    float ${funcName}() {
      ivec2 resTexRC = ivec2(gl_FragCoord.yx);
      int index = resTexRC.x * ${outTexShape[1]} + resTexRC.y;
      ${broadcastSnippet}
      int texR = index / ${inTexShape[1]};
      int texC = index - texR * ${inTexShape[1]};
      vec2 uv = (vec2(texC, texR) + halfCR) /
                 vec2(${inTexShape[1]}.0, ${inTexShape[0]}.0);
      return sample(${texName}, uv);
    }
  `;
}
