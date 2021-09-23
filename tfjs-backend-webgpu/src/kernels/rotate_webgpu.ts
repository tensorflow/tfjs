/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {util} from '@tensorflow/tfjs-core';

import {getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class RotateProgram implements WebGPUProgram {
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniformsWgsl: string;
  workGroupSize: [number, number, number] = [64, 1, 1];
  size: number;
  fillSnippetWgsl: string;

  constructor(
      imageShape: [number, number, number, number],
      fillValue: number|[number, number, number]) {
    this.outputShape = imageShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.uniformsWgsl = `centerX : f32; centerY : f32; sinRadians : f32;
          cosRadians : f32;`;
    this.shaderKey = 'rotate';
    this.size = util.sizeFromShape(this.outputShape);
    this.outputShape = imageShape;

    if (typeof fillValue === 'number') {
      this.uniformsWgsl += ` fillValue : f32;`;
      this.fillSnippetWgsl = `var outputValue = uniforms.fillValue;`;
      this.shaderKey += '_float';
    } else {
      this.uniformsWgsl += ` fillValue : vec3<f32>;`;
      this.fillSnippetWgsl = `var outputValue = uniforms.fillValue[coords[3]];`;
      this.shaderKey += '_vec3';
    }
  }

  getUserCodeWgsl(): string {
    const userCode = `
        ${getMainHeaderStringWgsl()} {
          ${getGlobalIndexStringWgsl()}

          if (index < uniforms.size) {
            let coords = getOutputCoords(globalId, index);
            let coordXFloat = (f32(coords[2]) - uniforms.centerX) *
                uniforms.cosRadians - (f32(coords[1]) - uniforms.centerY) *
                uniforms.sinRadians;
            let coordYFloat = (f32(coords[2]) - uniforms.centerX) *
                uniforms.sinRadians + (f32(coords[1]) - uniforms.centerY) *
                uniforms.cosRadians;
            let coordX = i32(round(coordXFloat + uniforms.centerX));
            let coordY = i32(round(coordYFloat + uniforms.centerY));
            ${this.fillSnippetWgsl}
            if(coordX >= 0 && coordX < uniforms.xShape[2] && coordY >= 0 &&
                coordY < uniforms.xShape[1]) {
              outputValue = getX(coords[0], coordY, coordX, coords[3]);
            }
            setOutputFlat(index, outputValue);
          }
        }
      `;
    return userCode;
  }
}
