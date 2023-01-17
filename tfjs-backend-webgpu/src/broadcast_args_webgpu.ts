/**
 * @license
 * Copyright 2023 Google LLC.
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

export class BroadcastArgsProgram implements WebGPUProgram {
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['s0', 's1'];
  uniforms = 's0Size : i32, s1Size : i32, ';
  workgroupSize: [number, number, number] = [64, 1, 1];
  size = true;

  constructor(shape: number) {
    this.outputShape = [shape];
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    this.shaderKey = 'broadcastArgs';
  }

  getUserCode(): string {
    const userCode = `
  ${main('index')} {
    if (index < uniforms.size) {
      var s0 = 1.0;
      var s1 = 1.0;
      let indexS0 = index - uniforms.size + uniforms.s0Size;
      let indexS1 = index - uniforms.size + uniforms.s1Size;
      if (indexS0 >= 0) {
        s0 = getS0(indexS0);
      }
      if (indexS1 >= 0) {
        s1 = getS1(indexS1);
      }

      if (s0 == 1.0) {
        setOutputAtIndex(index, s1);
      } else if (s1 == 1.0) {
        setOutputAtIndex(index, s0);
      } else if (s0 != s1) {
        setOutputAtIndex(index, uniforms.NAN);
      } else {
        setOutputAtIndex(index, s0);
      }
    }
  }
  `;
    return userCode;
  }
}
