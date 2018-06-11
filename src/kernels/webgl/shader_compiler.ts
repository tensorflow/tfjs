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

import * as tex_util from './tex_util';

export type ShapeInfo = {
  logicalShape: number[],
  texShape: [number, number],
  isUniform: boolean
};

export type InputInfo = {
  name: string,
  shapeInfo: ShapeInfo
};

export function makeShader(
    inputsInfo: InputInfo[], outputShape: ShapeInfo, userCode: string,
    broadcast: boolean): string {
  const sampleSnippet = getSampleSnippet();
  const setOutputSnippet = getSetOutputSnippet();
  let inputPrefixSnippet: string[]|string = inputsInfo.map(x => {
    const size = util.sizeFromShape(x.shapeInfo.logicalShape);
    if (x.shapeInfo.isUniform) {
      return `uniform float ${x.name}${size > 1 ? `[${size}]` : ''};`;
    }
    return `uniform sampler2D ${x.name};`;
  });
  inputPrefixSnippet = inputPrefixSnippet.join('\n');
  const inputSamplingSnippet =
      inputsInfo.map(x => getInputSamplingSnippet(x, outputShape, broadcast))
          .join('\n');
  const outTexShape = outputShape.texShape;
  const outputSamplingSnippet =
      getOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
  const source = [
    SHADER_PREFIX, sampleSnippet, setOutputSnippet, inputPrefixSnippet,
    outputSamplingSnippet, inputSamplingSnippet, userCode
  ].join('\n');
  return source;
}

function getSampleSnippet() {
  return ENV.get('WEBGL_FLOAT_TEXTURE_ENABLED') ?
      FLOAT_TEXTURE_SAMPLE_SNIPPET :
      UNSIGNED_BYTE_TEXTURE_SAMPLE_SNIPPET;
}

function getSetOutputSnippet() {
  return ENV.get('WEBGL_FLOAT_TEXTURE_ENABLED') ?
      FLOAT_TEXTURE_SETOUTPUT_SNIPPET :
      UNSIGNED_BYTE_TEXTURE_SETOUTPUT_SNIPPET;
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
    default:
      throw new Error(
          `${shape.length}-D input sampling` +
          ` is not yet supported`);
  }
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShapeInfo: ShapeInfo, broadcast: boolean): string {
  let res = getSamplerFlat(inInfo);
  res += getSamplerFromInInfo(inInfo);

  // If input and output have matching logical shapes, add
  // getTexNameAtOutCoord() method that samples the input
  // textureSampler using the output coordinates.
  if (broadcast ||
      util.arraysEqual(
          inInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape)) {
    res += getSamplerAtOutputCoords(inInfo, outShapeInfo, broadcast);
  }
  return res;
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
  // Explicitly use integer operations as dot() only works on floats.
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
  // Explicitly use integer operations as dot() only works on floats.
  int index = row * stride0 + col * stride1 + depth * stride2 + depth2;
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

const UNSIGNED_BYTE_TEXTURE_SAMPLE_SNIPPET = `
  uniform float NaN;

  const vec4 floatDeltas = vec4(
      1.0,
      1.0 / 255.0,
      1.0 / (255.0 * 255.0),
      1.0 / (255.0 * 255.0 * 255.0)
  );
  const float minValue = ${tex_util.FLOAT_MIN}.0;
  const float maxValue = ${tex_util.FLOAT_MAX}.0;
  const float range = (maxValue - minValue) / 255.0;
  const vec2 dotRange = vec2(1.0, range);

  float sampleTexture(sampler2D textureSampler, vec2 uv) {
    vec4 sampleValue = texture2D(textureSampler, uv);
    if (all(equal(sampleValue, vec4(${tex_util.BYTE_NAN_VALUE})))) {
      return NaN;
    }

    vec4 encValue = floor(sampleValue * 255.0 + 0.5);
    float decodedValue = dot(encValue, floatDeltas);
    return dot(vec2(minValue, decodedValue), dotRange);
  }
`;

const UNSIGNED_BYTE_TEXTURE_SETOUTPUT_SNIPPET = `
  const vec4 floatPowers = vec4(
    1.0,
    255.0,
    255.0 * 255.0,
    255.0 * 255.0 * 255.0
  );
  const vec2 recipRange = vec2(1.0/range);
  const vec2 recipRange255 = vec2(1.0/(maxValue - minValue));

  void setOutput(float decodedValue) {
    if (isNaN(decodedValue)) {
      gl_FragColor = vec4(${tex_util.BYTE_NAN_VALUE});
      return;
    }

    float a = dot(vec2(decodedValue, -minValue), recipRange);
    float b = fract(a) * 255.0;
    float c = fract(b) * 255.0;
    float d = fract(c) * 255.0;
    gl_FragColor = floor(vec4(a, b, c, d)) / 255.0;

    // TODO(dsmilkov): Version above gets better accuracy but probably slower
    // than the version below. Benchmark to determine if the accuracy is worth
    // the cost.

    // float normValue = dot(vec2(decodedValue, -minValue), recipRange255);
    // vec4 f = normValue * floatPowers;
    // gl_FragColor = floor(fract(f) * 255.0) / 255.0;
  }
`;

const FLOAT_TEXTURE_SAMPLE_SNIPPET = `
  float sampleTexture(sampler2D textureSampler, vec2 uv) {
    return texture2D(textureSampler, uv).r;
  }
`;

const FLOAT_TEXTURE_SETOUTPUT_SNIPPET = `
  void setOutput(float val) {
    gl_FragColor = vec4(val, 0, 0, 0);
  }
`;

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

  bool isNaN(float val) {
    float v1 = val * val;
    float v2 = val * val;
    return v1 == v2 ? false : true;
  }

  bool hasNaN(vec4 values) {
    vec4 v1 = values * values;
    vec4 v2 = values * values;
    return any(notEqual(v1, v2));
  }

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
`;

function getOutputScalarCoords() {
  return `
    int getOutputCoords() {
      return 0;
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

function getOutput3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];
  return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
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
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
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

function getOutput5DCoords(
    shape: [number, number, number, number, number],
    texShape: [number, number]): string {
  const stride3 = shape[4];
  const stride2 = shape[3] * stride3;
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;
  return `
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${texShape[0]},
                             ${texShape[1]}));

      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      int r = index / ${stride0};
      index -= r * ${stride0};

      int c = index / ${stride1};
      index -= c * ${stride1};

      int d = index / ${stride2};
      index -= d * ${stride2};

      int d2 = index  / ${stride3};
      int d3 = index - d2 * ${stride3};

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
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

function getSampler1D(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);

  return `
    float ${funcName}(int index) {
      return ${funcName}Flat(index);
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
    return `
      float ${funcName}(int row, int col) {
        int index = row * ${shape[1]} + col;
        return ${funcName}Flat(index);
      }
    `;
  }

  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texNumC === 1) {
    return `
    float ${funcName}(int row, int col) {
      int index = row * ${shape[1]} + col;
      vec2 uv = vec2(0.5, (float(index) + 0.5) / ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  if (texNumR === 1) {
    return `
    float ${funcName}(int row, int col) {
      int index = row * ${shape[1]} + col;
      vec2 uv = vec2((float(index) + 0.5) / ${texNumC}.0, 0.5);
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
    return `
      float ${funcName}(int row, int col, int depth) {
        int index = row * ${stride0} + col * ${stride1} + depth;
        return ${funcName}Flat(index);
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texNumC === stride0) {
    return `
        float ${funcName}(int row, int col, int depth) {
          int texR = row;
          int texC = col * ${stride1} + depth;
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${texNumC}.0, ${texNumR}.0);
          return sampleTexture(${texName}, uv);
        }
      `;
  }

  if (texNumC === stride1) {
    return `
    float ${funcName}(int row, int col, int depth) {
      int texR = row * ${shape[1]} + col;
      int texC = depth;
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
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int index = row * ${stride0} + col * ${stride1} +
            depth * ${stride2} + depth2;
        return ${funcName}Flat(index);
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texNumC === stride0) {
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int texR = row;
        int texC = col * ${stride1} + depth * ${stride2} + depth2;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (texNumC === stride2) {
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int texR = row * ${shape[1] * shape[2]} + col * ${shape[2]} + depth;
        int texC = depth2;
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
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        int index = row * ${stride0} + col * ${stride1} +
            depth * ${stride2} + depth2 * ${stride3} + depth3;
        return ${funcName}Flat(index);
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];

  if (texNumC === stride0) {
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        int texC = col * ${stride1} + depth * ${stride2} +
                   depth2 * ${stride3} + depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }

  if (texNumC === stride3) {
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row * ${shape[1] * shape[2]} + col * ${shape[2]} +
                   depth * ${shape[3]} + depth2;
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
