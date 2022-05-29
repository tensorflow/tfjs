/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class FillProgram implements WebGPUProgram {
  variableNames: string[] = [];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  uniforms = 'value : f32,';
  workGroupSize: [number, number, number] = [64, 1, 1];
  size = true;

  constructor(shape: number[]) {
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey = 'fill';
  }

  getUserCode(): string {
    const userCode = `
    ${getMainHeaderAndGlobalIndexString()}
      if (index < uniforms.size) {
        setOutputAtIndex(index, uniforms.value);
      }
    }
  `;
    return userCode;
  }
}
