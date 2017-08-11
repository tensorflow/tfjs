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

import {GPGPUProgram} from './gpgpu_math';

export enum UnaryOp {
  EXP, LOG, NEG, RELU, SIGMOID, STEP, SIN, TANH
}

export class UnaryOpProgram implements GPGPUProgram {
  variableNames = ['A'];
  params: Array<{}>;
  userCode: string;
  outputShape: number[];

  constructor(aShape: number[], op: UnaryOp) {
    this.outputShape = aShape;
    this.params = [op];
    this.userCode = `
      void main() {
        float v = getAAtOutCoords();
        ${getOpSnippet(op)}
        setOutput(r);
      }
    `;
  }
}

function getOpSnippet(op: UnaryOp) {
  switch(op) {
    case UnaryOp.EXP:
      return 'float r = exp(v);';
    case UnaryOp.LOG:
      return 'float r = log(v);';
    case UnaryOp.NEG:
      return 'float r = -v;';
    case UnaryOp.RELU:
      return 'float r = (v < 0.0) ? 0.0 : v;';
    case UnaryOp.SIGMOID:
      return 'float r = 1.0 / (1.0 + exp(-1.0 * v));';
    case UnaryOp.STEP:
      return 'float r = (v == v) ? (v > 0.0 ? 1.0 : 0.0) : v;';
    case UnaryOp.SIN:
      return 'float r = sin(v);';
    case UnaryOp.TANH:
      return `float e2x = exp(-2.0 * v);
              float r = (1.0 - e2x) / (1.0 + e2x);`;
    default:
      throw Error('Unrecognized unary op type ' + op);
  }
}
