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

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class RotateProgram implements WebGPUProgram {
    outputShape: number[] = [];
    shaderKey: string;
    dispatchLayout: {x: number[]};
    dispatch: [number, number, number];
    variableNames = ['x'];
    uniforms: string;
    uniformsWgsl: string;
    workGroupSize: [number, number, number] = [64, 1, 1];
    xShape: number[];
    size: number;
    fillSnippet: string;
    fillSnippetWgsl: string;
    useWgsl: boolean;

    constructor(
        imageShape: [number, number, number, number],
        fillValue: number|[number, number, number]) {
      this.outputShape = imageShape;
      this.dispatchLayout = flatDispatchLayout(this.outputShape);
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize);
      this.uniforms = `float centerX; float centerY; float sinRadians;
          float cosRadians;`;
      this.uniformsWgsl = `centerX : f32; centerY : f32; sinRadians : f32;
          cosRadians : f32;`;
      this.xShape = imageShape;
      this.shaderKey = 'rotate';
      this.useWgsl = getUseWgsl();
      this.size = util.sizeFromShape(this.outputShape);
      this.outputShape = imageShape;

      if (typeof fillValue === 'number') {
        this.uniforms += ` float fillValue;`;
        this.uniformsWgsl += ` fillValue : f32;`;
        this.fillSnippet = `float outputValue = fillValue;`;
        this.fillSnippetWgsl = `var outputValue = uniforms.fillValue;`;
        this.shaderKey += '_float';
      } else {
        this.uniforms += ` vec3 fillValue;`;
        this.uniformsWgsl += ` fillValue : vec3<f32>;`;
        this.fillSnippet = `float outputValue = fillValue[coords[3]];`;
        this.fillSnippetWgsl =
            `var outputValue = uniforms.fillValue[coords[3]];`;
        this.shaderKey += '_vec3';
      }
    }

    getUserCode(): string {
      const userCode = `
          void main() {
            int flatIndex = getGlobalIndex();

            if (flatIndex < size) {
              ivec4 coords = getOutputCoords();
              float coordXFloat = (float(coords[2]) - centerX) * cosRadians -
                  (float(coords[1]) - centerY) * sinRadians;
              float coordYFloat = (float(coords[2]) - centerX) * sinRadians +
                  (float(coords[1]) - centerY) * cosRadians;
              int coordX = int(round(coordXFloat + centerX));
              int coordY = int(round(coordYFloat + centerY));
              ${this.fillSnippet}
              if(coordX >= 0 && coordX < xShape[2] && coordY >= 0 &&
                  coordY < xShape[1]) {
                outputValue = getX(coords[0], coordY, coordX, coords[3]);
              }
              setOutput(flatIndex, outputValue);
            }
          }
      `;
      return userCode;
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
