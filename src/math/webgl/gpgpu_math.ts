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

import {NDArray} from '../ndarray';

import {GPGPUContext} from './gpgpu_context';
import * as shader_compiler from './shader_compiler';
import * as util from '../../util';

export interface GPGPUProgram {
  variableNames: string[];
  outputShape: number[];
  params: Array<{}>;
  userCode: string;
}

export interface GPGPUBinary<T extends NDArray, K extends NDArray> {
  webGLProgram: WebGLProgram;
  program: GPGPUProgram;
  gpgpu: GPGPUContext;
  source: string;
  inputs: T[];
  output: K;
}

export function compileProgram<T extends NDArray, K extends NDArray>(
    gpgpu: GPGPUContext, program: GPGPUProgram, inputs: T[],
    output: K): GPGPUBinary<T,K> {
  const userCode = program.userCode;
  const programInputs = program.variableNames.map((x, i) => {
    const fullShape = {
      shape: inputs[i].shape,
      texShape: inputs[i].getTextureShapeRC()
    };
    return {name: x, fullShape};
  });

  const outFullShape = {
    shape: output.shape,
    texShape: output.getTextureShapeRC()
  };
  const source = shader_compiler.makeShader(programInputs, outFullShape,
      userCode);
  return {
    program,
    source,
    webGLProgram: gpgpu.createProgram(source),
    gpgpu,
    inputs,
    output
  };
}

function validateBinaryAndProgram(aArrays: NDArray[], bArrays: NDArray[]) {
  aArrays.forEach((a, i) => {
    const shapeA = a.shape;
    const texShapeA = a.getTextureShapeRC();
    const shapeB = bArrays[i].shape;
    const texShapeB = bArrays[i].getTextureShapeRC();

    if (!util.arraysEqual(shapeA, shapeB)) {
      throw Error(`Binary was compiled with different shapes than ` +
          `the current args. Shapes ${shapeA} and ${shapeB} must match`);
    }
    if (!util.arraysEqual(texShapeA, texShapeB)) {
      throw Error(`Binary was compiled with different texture shapes than the` +
          ` current args. Shape ${texShapeA} and ${texShapeB} must match`);
    }
  });
}

export function runProgram<T extends NDArray, K extends NDArray>(
    binary: GPGPUBinary<T,K>, inputs?: T[], output?: K): void {
  if (inputs == null) {
    inputs = binary.inputs;
  } else {
    validateBinaryAndProgram(binary.inputs, inputs);
  }
  if (output == null) {
    output = binary.output;
  } else {
    validateBinaryAndProgram([binary.output], [output]);
  }
  const outTex = output.getTexture();
  const outTexShape = output.getTextureShapeRC();
  const gpgpu = binary.gpgpu;
  gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
  gpgpu.setProgram(binary.webGLProgram);
  inputs.forEach((input, i) => {
    const tex = input.getTexture();
    gpgpu.setInputMatrixTexture(tex, binary.program.variableNames[i], i);
  });
  gpgpu.executeProgram();
}

export function makeShaderKey(
    program: GPGPUProgram, inputs: NDArray[],
    output: NDArray): string {
  const params = program.params;
  const keyStart =
      inputs.concat(output).map(x => x.shape + '_' + x.getTextureShapeRC());
  const keyEnd = params.map(p => p.toString());
  const key = [program.constructor.name].concat(keyStart, keyEnd);
  return key.join('_');
}
