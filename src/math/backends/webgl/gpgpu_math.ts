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

import {ENV} from '../../../environment';
import * as util from '../../../util';
import {NDArray} from '../../ndarray';
import {GPGPUContext} from './gpgpu_context';
import * as shader_compiler from './shader_compiler';
import {ShapeInfo} from './shader_compiler';
import {TextureData} from './tex_util';

const ATTRIBUTE_NAMES = ['uv', 'clipSpacePos'];

export interface GPGPUProgram {
  variableNames: string[];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting?: boolean;
}

export interface GPGPUBinary {
  webGLProgram: WebGLProgram;
  program: GPGPUProgram;
  uniformLocations: {[name: string]: WebGLUniformLocation};
  attributeLocations: {[name: string]: number};
  gpgpu: GPGPUContext;
  source: string;
  inShapeInfos: ShapeInfo[];
  outShapeInfo: ShapeInfo;
}

const NAN_UNIFORM_NAME = 'NaN';

function shouldUploadNaNUniform(): boolean {
  return !ENV.get('WEBGL_FLOAT_TEXTURE_ENABLED');
}

export interface ArrayData<T extends NDArray> {
  array: T;
  texData: TextureData;
}

export function compileProgram<T extends NDArray, K extends NDArray>(
    gpgpu: GPGPUContext, program: GPGPUProgram, inputs: Array<ArrayData<T>>,
    output: ArrayData<K>): GPGPUBinary {
  const userCode = program.userCode;
  const inputInfos = inputs.map((input, i) => {
    const shapeInfo = {
      logicalShape: input.array.shape,
      texShape: input.texData.texShape,
      textureType: input.texData.textureType
    };
    return {name: program.variableNames[i], shapeInfo};
  });
  const inShapeInfos = inputInfos.map(x => x.shapeInfo);
  const outShapeInfo = {
    logicalShape: output.array.shape,
    texShape: output.texData.texShape,
    textureType: output.texData.textureType
  };
  const source = shader_compiler.makeShader(
      inputInfos, outShapeInfo, userCode,
      program.supportsBroadcasting === true);

  const webGLProgram = gpgpu.createProgram(source);

  const uniformLocations: {[name: string]: WebGLUniformLocation} = {};
  for (let i = 0; i < program.variableNames.length; i++) {
    const uniformName = program.variableNames[i];
    uniformLocations[uniformName] =
        gpgpu.getUniformLocation(webGLProgram, uniformName);
  }
  const attributeLocations: {[name: string]: number} = {};
  ATTRIBUTE_NAMES.forEach(attribute => {
    attributeLocations[attribute] =
        gpgpu.getAttributeLocation(webGLProgram, attribute);
  });

  if (shouldUploadNaNUniform()) {
    uniformLocations[NAN_UNIFORM_NAME] =
        gpgpu.getUniformLocation(webGLProgram, NAN_UNIFORM_NAME);
  }

  return {
    program,
    source,
    webGLProgram,
    uniformLocations,
    attributeLocations,
    gpgpu,
    inShapeInfos,
    outShapeInfo
  };
}

function validateBinaryAndProgram(
    shapeInfos: ShapeInfo[], inputs: Array<ArrayData<NDArray>>) {
  if (shapeInfos.length !== inputs.length) {
    throw Error(
        `Binary was compiled with ${shapeInfos.length} inputs, but ` +
        `was executed with ${inputs.length} inputs`);
  }

  shapeInfos.forEach((s, i) => {
    const shapeA = s.logicalShape;
    const texShapeA = s.texShape;
    const shapeB = inputs[i].array.shape;
    const texShapeB = inputs[i].texData.texShape;

    if (!util.arraysEqual(shapeA, shapeB)) {
      throw Error(
          `Binary was compiled with different shapes than ` +
          `the current args. Shapes ${shapeA} and ${shapeB} must match`);
    }
    if (!util.arraysEqual(texShapeA, texShapeB)) {
      throw Error(
          `Binary was compiled with different texture shapes than the` +
          ` current args. Shape ${texShapeA} and ${texShapeB} must match`);
    }
  });
}

export function runProgram<T extends NDArray, K extends NDArray>(
    binary: GPGPUBinary, inputs: Array<ArrayData<T>>, output: ArrayData<K>,
    customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) =>
        void): void {
  validateBinaryAndProgram(binary.inShapeInfos, inputs);
  validateBinaryAndProgram([binary.outShapeInfo], [output]);

  const outTex = output.texData.texture;
  const outTexShape = output.texData.texShape;
  const gpgpu = binary.gpgpu;
  gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
  gpgpu.setProgram(binary.webGLProgram);
  inputs.forEach((input, i) => {
    const tex = input.texData.texture;
    const variableName = binary.program.variableNames[i];
    const variableUniformLocation = binary.uniformLocations[variableName];
    gpgpu.setInputMatrixTexture(tex, variableUniformLocation, i);
  });

  if (shouldUploadNaNUniform()) {
    gpgpu.gl.uniform1f(binary.uniformLocations[NAN_UNIFORM_NAME], NaN);
  }

  if (customSetup != null) {
    customSetup(gpgpu, binary.webGLProgram);
  }
  gpgpu.executeProgram(binary.attributeLocations);
}

export function makeShaderKey(
    program: GPGPUProgram, inputs: Array<ArrayData<NDArray>>,
    output: ArrayData<NDArray>): string {
  let keyInputs = '';
  inputs.concat(output).forEach(x => {
    keyInputs +=
        `${x.array.shape}_${x.texData.texShape}_${x.texData.textureType}`;
  });
  const keyUserCode = program.userCode;
  const keyBroadcast = (program.supportsBroadcasting === true).toString();
  let key = program.constructor.name;
  // Fast string concat. See https://jsperf.com/string-concatenation/14.
  key += '_' + keyBroadcast + '_' + keyInputs + '_' + keyUserCode;
  return key;
}
