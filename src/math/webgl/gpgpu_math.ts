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
import {NDArray} from '../ndarray';

import {GPGPUContext} from './gpgpu_context';
import * as shader_compiler from './shader_compiler';
import {ShapeInfo} from './shader_compiler';

export interface GPGPUProgram {
  variableNames: string[];
  outputShape: number[];
  params: Array<{}>;
  userCode: string;
  supportsBroadcasting?: boolean;
}

export interface GPGPUBinary {
  webGLProgram: WebGLProgram;
  program: GPGPUProgram;
  uniformLocations: {[name: string]: WebGLUniformLocation};
  gpgpu: GPGPUContext;
  source: string;
  inShapeInfos: ShapeInfo[];
  outShapeInfo: ShapeInfo;
}

export function compileProgram<T extends NDArray, K extends NDArray>(
    gpgpu: GPGPUContext, program: GPGPUProgram, inputs: T[],
    output: K): GPGPUBinary {
  const userCode = program.userCode;
  const inputInfos = inputs.map((input, i) => {
    const shapeInfo = {
      logicalShape: input.shape,
      texShape: input.getTextureShapeRC()
    };
    return {name: program.variableNames[i], shapeInfo};
  });
  const inShapeInfos = inputInfos.map(x => x.shapeInfo);
  const outShapeInfo = {
    logicalShape: output.shape,
    texShape: output.getTextureShapeRC()
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

  return {
    program,
    source,
    webGLProgram,
    uniformLocations,
    gpgpu,
    inShapeInfos,
    outShapeInfo
  };
}

function validateBinaryAndProgram(shapeInfos: ShapeInfo[], inputs: NDArray[]) {
  if (shapeInfos.length !== inputs.length) {
    throw Error(
        `Binary was compiled with ${shapeInfos.length} inputs, but ` +
        `was executed with ${inputs.length} inputs`);
  }

  shapeInfos.forEach((s, i) => {
    const shapeA = s.logicalShape;
    const texShapeA = s.texShape;
    const shapeB = inputs[i].shape;
    const texShapeB = inputs[i].getTextureShapeRC();

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
    binary: GPGPUBinary, inputs: T[], output: K,
    customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) =>
        void): void {
  validateBinaryAndProgram(binary.inShapeInfos, inputs);
  validateBinaryAndProgram([binary.outShapeInfo], [output]);

  const outTex = output.getTexture();
  const outTexShape = output.getTextureShapeRC();
  const gpgpu = binary.gpgpu;
  gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
  gpgpu.setProgram(binary.webGLProgram);
  inputs.forEach((input, i) => {
    const tex = input.getTexture();
    const variableName = binary.program.variableNames[i];
    const variableUniformLocation = binary.uniformLocations[variableName];
    gpgpu.setInputMatrixTexture(tex, variableUniformLocation, i);
  });
  if (customSetup != null) {
    customSetup(gpgpu, binary.webGLProgram);
  }
  gpgpu.executeProgram();
}

export function makeShaderKey(
    program: GPGPUProgram, inputs: NDArray[], output: NDArray): string {
  const params = program.params;
  const keyStart =
      inputs.concat(output).map(x => x.shape + '_' + x.getTextureShapeRC());
  const keyEnd = params.map(String);
  let key = [program.constructor.name];
  key.push((program.supportsBroadcasting === true).toString());
  key = key.concat(keyStart, keyEnd);
  return key.join('_');
}
