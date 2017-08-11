/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as util from '../../util';

export type NDArrayShape = {
  shape: number[],
  texShape: [number, number];
};

export type Input = {
  name: string,
  fullShape: NDArrayShape
};

export function makeShader(
    inputs: Input[], output: NDArrayShape, userCode: string): string {
  const inputPrefixSnippet =
      inputs.map(x => `uniform sampler2D ${x.name};`).join('\n');
  const inputSamplingSnippet =
      inputs.map(x => getInputSamplingSnippet(x, output)).join('\n');
  const outTexShape = output.texShape;
  const outputSamplingSnippet =
      getOutputSamplingSnippet(output.shape, outTexShape);
  const source = [
    SHADER_PREFIX, inputPrefixSnippet, SAMPLE_1D_SNIPPET, SAMPLE_2D_SNIPPET,
    SAMPLE_3D_SNIPPET, inputSamplingSnippet, outputSamplingSnippet, userCode
  ].join('\n');
  return source;
}

function getInputSamplingSnippet(input: Input, output: NDArrayShape) {
  const fullShape = input.fullShape;
  const shape = fullShape.shape;
  const texShape = fullShape.texShape;
  const outTexShape = output.texShape;

  let res = '';
  switch (shape.length) {
    case 0:
      res += getSamplerScalar(input.name);
      break;
    case 1:
      res += getSampler1D(input.name, texShape);
      break;
    case 2:
      res += getSampler2D(input.name, shape as [number, number], texShape);
      break;
    case 3:
      res += getSampler3D(input.name, shape as [number, number, number],
          texShape);
      break;
    default:
      throw new Error(`${fullShape.shape.length}-D input sampling is not yet ` +
                      `supported`);
  }
  // If input and output have matching logical shapes, add
  // getTexNameAtOutCoord() method that samples the input texture using the
  // output coordinates.
  if (util.arraysEqual(input.fullShape.shape, output.shape)) {
    res += getSamplerAtOutputCoords(input.name, texShape, outTexShape);
  }
  res += getSamplerFlat(input.name, texShape);
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
      return getOutput3DCoords(outShape as [number, number, number],
          outTexShape);
    default:
      throw new Error(
          `${outShape.length}-D output sampling is not yet supported`);
  }
}

const SHADER_PREFIX = `
  precision highp float;
  varying vec2 resultUV;
  const vec2 halfCR = vec2(0.5, 0.5);

  void setOutput(float val) {
    gl_FragColor = vec4(val, 0, 0, 0);
  }

  bool isNaN(float val) {
    return val == val ? false : true;
  }
`;

const SAMPLE_1D_SNIPPET = `
  float sample1D(sampler2D texture, float texNumR, float texNumC, float index) {
    float texR = floor(index / texNumC);
    float texC = mod(index, texNumC);
    vec2 uv = (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
    return texture2D(texture, uv).r;
  }
`;

const SAMPLE_2D_SNIPPET = `
  float sample2D(sampler2D texture, float texNumR, float texNumC, float numC,
      float row, float col) {
    float index = dot(vec2(row, col), vec2(numC, 1.0));
    float texR = floor(index / texNumC);
    float texC = mod(index, texNumC);
    vec2 uv = (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
    return texture2D(texture, uv).r;
  }
`;

const SAMPLE_3D_SNIPPET = `
  float sample3D(sampler2D texture, float texNumR, float texNumC, float stride0,
      float stride1, float row, float col, float depth) {
    float index = dot(vec3(row, col, depth), vec3(stride0, stride1, 1.0));
    float texR = floor(index / texNumC);
    float texC = mod(index, texNumC);
    vec2 uv = (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
    return texture2D(texture, uv).r;
  }
`;

function getOutput1DCoords(
    shape: [number], texShape: [number, number]): string {
  if (texShape[0] === 1) {
    return `
      float getOutputCoords() {
        return floor(gl_FragCoord.x);
      }
    `;
  }
  if (texShape[1] === 1) {
    return `
      float getOutputCoords() {
        return floor(gl_FragCoord.y);
      }
    `;
  }
  return `
    float getOutputCoords() {
      vec2 resTexRC = floor(gl_FragCoord.yx);
      return dot(resTexRC, vec2(${texShape[1]}.0, 1.0));
    }
  `;
}

function getOutput3DCoords(shape: [number, number, number],
    texShape: [number, number]): string {
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];
  return `
    vec3 getOutputCoords() {
      vec2 resTexRC = floor(gl_FragCoord.yx);
      float index = dot(resTexRC, vec2(${texShape[1]}.0, 1.0));
      float r = floor(index / ${stride0}.0);
      index -= r * ${stride0}.0;
      float c = floor(index / ${stride1}.0);
      float d = mod(index, ${stride1}.0);
      return vec3(r, c, d);
    }
  `;
}

function getOutput2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  if (util.arraysEqual(shape, texShape)) {
    return `
      vec2 getOutputCoords() {
        return floor(gl_FragCoord.yx);
      }
    `;
  }
  return `
    vec2 getOutputCoords() {
      vec2 resTexRC = floor(gl_FragCoord.yx);
      float index = dot(resTexRC, vec2(${texShape[1]}.0, 1.0));
      float r = floor(index / ${shape[1]}.0);
      float c = mod(index, ${shape[1]}.0);
      return vec2(r, c);
    }
  `;
}

function getSamplerScalar(texName: string): string {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  return `
    float ${funcName}() {
      return texture2D(${texName}, halfCR).r;
    }
  `;
}

function getSampler1D(
    texName: string, texShape: [number, number]): string {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const tR = texShape[0];
  const tC = texShape[1];
  if (texShape[1] === 1) {
    return `
      float ${funcName}(float index) {
        vec2 uv = vec2(0.5, (index + 0.5) / ${tR}.0);
        return texture2D(${texName}, uv).r;
      }
    `;
  }
  if (texShape[0] === 1) {
    return `
      float ${funcName}(float index) {
        vec2 uv = vec2((index + 0.5) / ${tC}.0, 0.5);
        return texture2D(${texName}, uv).r;
      }
    `;
  }
  return `
    float ${funcName}(float index) {
      return sample1D(${texName}, ${tR}.0, ${tC}.0, index);
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
  return `
    float ${funcName}(float row, float col, float depth) {
      return sample3D(${texName}, ${tR}.0, ${tC}.0, ${stride0}.0, ${stride1}.0,
          row, col, depth);
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
      float ${funcName}(float row, float col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${tC}.0, ${tR}.0);
        return texture2D(${texName}, uv).r;
      }
    `;
  }
  return `
    float ${funcName}(float row, float col) {
      return sample2D(${texName}, ${tR}.0, ${tC}.0, ${shape[1]}.0, row, col);
    }
  `;
}

function getSamplerFlat(texName: string, texShape: [number, number]): string {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1) +
      'Flat';
  const tNumR = texShape[0];
  const tNumC = texShape[1];
  return `
    float ${funcName}(float index) {
      float texR = floor(index / ${tNumC}.0);
      float texC = mod(index, ${tNumC}.0);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${tNumC}.0, ${tNumR}.0);
      return texture2D(${texName}, uv).r;
    }
  `;
}

function getSamplerAtOutputCoords(texName: string, inTexShape: [number, number],
    outTexShape: [number, number]) {
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1) +
    'AtOutCoords';
  if (util.arraysEqual(inTexShape, outTexShape)) {
    return `
      float ${funcName}() {
        return texture2D(${texName}, resultUV).r;
      }
    `;
  }
  return `
    float ${funcName}() {
      vec2 resTexRC = floor(gl_FragCoord.yx);
      float index = dot(resTexRC, vec2(${outTexShape[1]}.0, 1.0));
      float texR = floor(index / ${inTexShape[1]}.0);
      float texC = mod(index, ${inTexShape[1]}.0);
      vec2 uv = (vec2(texC, texR) + halfCR) /
                 vec2(${inTexShape[1]}.0, ${inTexShape[0]}.0);
      return texture2D(${texName}, uv).r;
    }
  `;
}
