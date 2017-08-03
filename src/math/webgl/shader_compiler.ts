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
import {NDArray} from '../ndarray';

export type Input = {
  name: string; array: NDArray;
};

export function makeShaderKey(inputs: NDArray[], output: NDArray): string {
  const ins = inputs.map(x => x.shape + '_' + x.getTextureShapeRC());
  return ins.join('_') + '_' + output.shape + '_' + output.getTextureShapeRC();
}

export function makeShader(
    inputs: Input[], output: NDArray, userCode: string): string {
  const inputPrefixSnippet =
      inputs.map(x => `uniform sampler2D ${x.name};`).join('\n');
  const inputSamplingSnippet =
      inputs.map(x => getInputSamplingSnippet(x)).join('\n');
  const outTexShape = output.getTextureShapeRC();
  const outputSamplingSnippet =
      getOutputSamplingSnippet(output.shape, outTexShape);
  const source = [
    SHADER_PREFIX, inputPrefixSnippet, SAMPLE_2D_SNIPPET, inputSamplingSnippet,
    outputSamplingSnippet, userCode
  ].join('\n');
  return source;
}

function getInputSamplingSnippet(input: Input) {
  const arr = input.array;
  const shape = arr.shape;
  const texShape = arr.getTextureShapeRC(shape as [number, number]);
  switch (shape.length) {
    case 2:
      return getSampler2D(input.name, shape as [number, number], texShape);
    default:
      throw new Error(`${arr.rank}-D input sampling is not yet supported`);
  }
}

function getOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 2:
      return getOutput2DCoords(outShape as [number, number], outTexShape);
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

function getOutput2DCoords(
    shape: [number, number], texShape: [number, number]) {
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

function getSampler2D(
    texName: string, shape: [number, number], texShape: [number, number]) {
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
