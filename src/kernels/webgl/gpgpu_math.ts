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

import {Tensor} from '../../tensor';
import * as util from '../../util';

import {GPGPUContext} from './gpgpu_context';
import * as shader_compiler from './shader_compiler';
import {InputInfo, ShapeInfo} from './shader_compiler';
import {TextureData} from './tex_util';

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
  gpgpu: GPGPUContext;
  source: string;
  inShapeInfos: ShapeInfo[];
  outShapeInfo: ShapeInfo;
}

export interface TensorData<T extends Tensor> {
  tensor: T;
  texData: TextureData;
  isUniform: boolean;
}

export function compileProgram<T extends Tensor, K extends Tensor>(
    gpgpu: GPGPUContext, program: GPGPUProgram, inputs: Array<TensorData<T>>,
    output: TensorData<K>): GPGPUBinary {
  const userCode = program.userCode;
  const inputInfos: InputInfo[] = inputs.map((input, i) => {
    const shapeInfo = {
      logicalShape: input.tensor.shape,
      texShape: input.isUniform ? null : input.texData.texShape,
      isUniform: input.isUniform
    };
    return {name: program.variableNames[i], shapeInfo};
  });
  const inShapeInfos = inputInfos.map(x => x.shapeInfo);
  const outShapeInfo = {
    logicalShape: output.tensor.shape,
    texShape: output.texData.texShape,
    isUniform: false
  };
  const source = shader_compiler.makeShader(
      inputInfos, outShapeInfo, userCode,
      program.supportsBroadcasting === true);

  const webGLProgram = gpgpu.createProgram(source);

  const uniformLocations: {[name: string]: WebGLUniformLocation} = {};
  for (let i = 0; i < program.variableNames.length; i++) {
    const uniformName = program.variableNames[i];
    const shouldThrow = false;
    uniformLocations[uniformName] =
        gpgpu.getUniformLocation(webGLProgram, uniformName, shouldThrow);
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

function validateBinaryAndProgram(
    shapeInfos: ShapeInfo[], inputs: Array<TensorData<Tensor>>) {
  if (shapeInfos.length !== inputs.length) {
    throw Error(
        `Binary was compiled with ${shapeInfos.length} inputs, but ` +
        `was executed with ${inputs.length} inputs`);
  }

  shapeInfos.forEach((s, i) => {
    const shapeA = s.logicalShape;
    const input = inputs[i];
    const shapeB = input.tensor.shape;

    if (!util.arraysEqual(shapeA, shapeB)) {
      throw Error(
          `Binary was compiled with different shapes than ` +
          `the current args. Shapes ${shapeA} and ${shapeB} must match`);
    }
    // The input is uploaded as uniform.
    if (s.isUniform && input.isUniform) {
      return;
    }

    const texShapeA = s.texShape;
    const texShapeB = input.isUniform ? null : input.texData.texShape;
    if (!util.arraysEqual(texShapeA, texShapeB)) {
      throw Error(
          `Binary was compiled with different texture shapes than the` +
          ` current args. Shape ${texShapeA} and ${texShapeB} must match`);
    }
  });
}

export function runProgram<T extends Tensor, K extends Tensor>(
    binary: GPGPUBinary, inputs: Array<TensorData<T>>, output: TensorData<K>,
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
    const variableName = binary.program.variableNames[i];
    const variableUniformLocation = binary.uniformLocations[variableName];
    if (variableUniformLocation != null) {
      if (input.isUniform) {
        if (input.tensor.size === 1) {
          gpgpu.gl.uniform1f(
              variableUniformLocation, input.tensor.dataSync()[0]);
        } else {
          let vals = input.tensor.dataSync();
          if (!(vals instanceof Float32Array)) {
            vals = new Float32Array(vals);
          }
          gpgpu.gl.uniform1fv(variableUniformLocation, vals);
        }
        return;
      }
      const tex = input.texData.texture;
      gpgpu.setInputMatrixTexture(tex, variableUniformLocation, i);
    }
  });

  if (customSetup != null) {
    customSetup(gpgpu, binary.webGLProgram);
  }
  gpgpu.executeProgram();
}

export function makeShaderKey(
    program: GPGPUProgram, inputs: Array<TensorData<Tensor>>,
    output: TensorData<Tensor>): string {
  let keyInputs = '';
  inputs.concat(output).forEach(x => {
    keyInputs +=
        `${x.tensor.shape}_${x.isUniform ? 'uniform' : x.texData.texShape}`;
  });
  const keyUserCode = program.userCode;
  const keyBroadcast = (program.supportsBroadcasting === true).toString();
  let key = program.constructor.name;
  // Fast string concat. See https://jsperf.com/string-concatenation/14.
  key += '_' + keyBroadcast + '_' + keyInputs + '_' + keyUserCode;
  return key;
}
