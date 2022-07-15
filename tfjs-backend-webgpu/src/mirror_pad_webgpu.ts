/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {getCoordsDataType, getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class MirrorPadProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  uniforms = '';
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  xShape: number[];
  offset: number;
  size = true;

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      mode: 'reflect'|'symmetric') {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.xShape = xShape;
    paddings.map((_, i) => {
      this.uniforms += ` pad${i} : vec2<i32>,`;
    });
    this.offset = mode === 'reflect' ? 0 : 1;
    this.shaderKey = `mirrorPad_${mode}`;
  }

  getUserCode(): string {
    const rank = this.xShape.length;
    // The length of paddings are same with the rank of the input tensor.
    const start = this.xShape.map((_, i) => `uniforms.pad${i}[0]`).join(',');
    const end = this.xShape
                    .map(
                        (_, i) => `uniforms.pad${i}[0] + uniforms.xShape${
                            rank > 1 ? `[${i}]` : ''}`)
                    .join(',');

    const shaderStart = rank === 1 ? 'start' : 'start[i]';
    const shaderEnd = rank === 1 ? 'end' : 'end[i]';
    const shaderOutC = rank === 1 ? 'outC' : 'outC[i]';
    const dtype = getCoordsDataType(rank);
    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';

    return `
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          let start = ${dtype}(${start});
          let end = ${dtype}(${end});
          var outC = getCoordsFromIndex(index);
          for (var i = 0; i < ${rank}; i = i + 1) {
            if (${shaderOutC} < ${shaderStart}) {
              ${shaderOutC} = ${shaderStart} * 2 - ${shaderOutC} - ${
        this.offset};
            } else if(${shaderOutC} >= ${shaderEnd}) {
              ${shaderOutC} = (${shaderEnd} - 1) * 2 - ${shaderOutC} + ${
        this.offset};
            }
          }
          let coords = outC - start;
          setOutputAtIndex(index, getX(${unpackedCoords}));
        }
      }
    `;
  }
}
