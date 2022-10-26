/**
 * @license
 * Copyright 2022 Google LLC.
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

import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ReverseProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms: string;
  workgroupSize: [number, number, number] = [64, 1, 1];
  size = true;

  constructor(xShape: [number, number, number, number]) {
    this.outputShape = xShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.uniforms = ` axis : vec4<i32>,`;
    this.shaderKey = 'reverse';
  }

  getUserCode(): string {
    const inputCoordsSnippet = `
      // Using uniform variables as judging conditions, so the function has
      // coherent execution within all threads.
      fn getReverseInputCoords(coords : vec4<i32>) -> vec4<i32> {
        var inputCoords: vec4<i32>;
        if ((uniforms.axis[0] == 1) && (uniforms.xShape[0] != 1)) {
          inputCoords[0] = uniforms.xShape[0] - coords[0] - 1;
        } else {
          inputCoords[0] = coords[0];
        }
        if ((uniforms.axis[1] == 1) && (uniforms.xShape[1] != 1)) {
          inputCoords[1] = uniforms.xShape[1] - coords[1] - 1;
        } else {
          inputCoords[1] = coords[1];
        }
        if ((uniforms.axis[2] == 1) && (uniforms.xShape[2] != 1)) {
          inputCoords[2] = uniforms.xShape[2] - coords[2] - 1;
        } else {
          inputCoords[2] = coords[2];
        }
        if ((uniforms.axis[3] == 1) && (uniforms.xShape[3] != 1)) {
          inputCoords[3] = uniforms.xShape[3] - coords[3] - 1;
        } else {
          inputCoords[3] = coords[3];
        }

        return inputCoords;
      }
    `;
    const userCode = `
      ${inputCoordsSnippet}
      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let inputCoords = getReverseInputCoords(coords);
          setOutputAtIndex(index, getX(inputCoords[0],
              inputCoords[1], inputCoords[2], inputCoords[3]));
        }
      }
    `;
    return userCode;
  }
}
