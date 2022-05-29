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

import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class FlipLeftRightProgram implements WebGPUProgram {
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  size = true;

  constructor(imageShape: [number, number, number, number]) {
    this.outputShape = imageShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = 'flipLeftRight';
  }

  getUserCode(): string {
    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let coordX = uniforms.xShape[2] - coords[2] - 1;
          let outputValue = getX(coords[0], coords[1], coordX, coords[3]);
          setOutputAtIndex(index, outputValue);
        }
      }
    `;
    return userCode;
  }
}
