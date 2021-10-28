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
import {getGlobalIndexString, getMainHeaderString} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class OneHotProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['indices'];
  uniforms = `onValue : f32; offValue : f32;`;
  workGroupSize: [number, number, number] = [256, 1, 1];
  size: number;

  constructor(numIndices: number, depth: number) {
    this.outputShape = [numIndices, depth];
    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey = 'oneHot';
    this.size = util.sizeFromShape(this.outputShape);
  }

  getUserCode(): string {
    const userCode = `
      ${getMainHeaderString()} {
        ${getGlobalIndexString()}
        if (index < uniforms.size) {
          let coords = getOutputCoords(globalId, index);
          let ind = round(getIndices(coords.x));
          setOutputFlat(index, mix(uniforms.offValue, uniforms.onValue,
                      f32(i32(ind) == coords.y)));
        }
      }
    `;
    return userCode;
  }
}
