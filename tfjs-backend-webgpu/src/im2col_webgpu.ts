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

export class Im2ColProgram implements WebGPUProgram {
  variableNames = ['x'];
  uniforms =
      `pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, outWidth : i32, itemsPerBlockRow : i32,
       inChannels : i32,`;
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  isChannelsLast: boolean;
  size = true;

  constructor(outputShape: number[], isChannelsLast: boolean) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.isChannelsLast = isChannelsLast;
    this.shaderKey = `im2col_${this.isChannelsLast}`;
  }

  getUserCode(): string {
    const rowDim = this.isChannelsLast ? 1 : 2;
    const colDim = this.isChannelsLast ? 2 : 3;

    const row = this.isChannelsLast ? 'coords[1]' : 'coords[2]';
    const col = this.isChannelsLast ? 'coords[2]' : 'coords[1]';
    const getXSnippet = this.isChannelsLast ? 'getX(batch, xRow, xCol, ch)' :
                                              'getX(batch, ch, xRow, xCol)';

    const userCode = `
    ${main('index')} {
      let coords = getCoordsFromIndex(index);
      if(index < uniforms.size) {
        let batch = coords[0];
        let row = ${row};
        let col = ${col};
        let offsetY = (row / uniforms.outWidth) * uniforms.strides[0] - uniforms.pads[0];
        let xRow = offsetY + uniforms.dilations[0] * (col / uniforms.itemsPerBlockRow);
        var value = 0.0;
        if(xRow < uniforms.xShape[${rowDim}] && xRow >= 0) {
          let offsetX = (row % uniforms.outWidth) * uniforms.strides[1] -
              uniforms.pads[1];
          let xCol = offsetX + uniforms.dilations[1] * ((col %
              uniforms.itemsPerBlockRow) / uniforms.inChannels);
          let ch = col % uniforms.inChannels;
          if(xCol < uniforms.xShape[${colDim}] && xCol >= 0) {
            value = ${getXSnippet};
          }
        }
        setOutputAtIndex(index, value);
      }
    }
   `;
    return userCode;
  }
}
