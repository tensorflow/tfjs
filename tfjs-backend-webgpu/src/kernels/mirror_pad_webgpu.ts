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

import {util} from '@tensorflow/tfjs-core';

import {getCoordsDataTypeWgsl, getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class MirrorPadProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  uniformsWgsl = '';
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  xShape: number[];
  offset: number;
  size: number;

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
      this.uniformsWgsl += ` pad${i} : vec2<i32>;`;
    });
    this.offset = mode === 'reflect' ? 0 : 1;
    this.shaderKey = `mirrorPad_${mode}`;
    this.size = util.sizeFromShape(this.outputShape);
  }

  getUserCodeWgsl(): string {
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
    const dtype = getCoordsDataTypeWgsl(rank);
    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';

    return `
      ${getMainHeaderStringWgsl()} {
        ${getGlobalIndexStringWgsl()}
        let start = ${dtype}(${start});
        let end = ${dtype}(${end});
        var outC = getOutputCoords(globalId, index);
        if (index < uniforms.size) {
          for (var i = 0; i < ${rank}; i = i + 1) {
            if (${shaderOutC} < ${shaderStart}) {
              ${shaderOutC} = ${shaderStart} * 2 - ${shaderOutC} - ${
        this.offset};
            } elseif(${shaderOutC} >= ${shaderEnd}) {
              ${shaderOutC} = (${shaderEnd} - 1) * 2 - ${shaderOutC} + ${
        this.offset};
            }
          }
          let coords = outC - start;
          setOutputFlat(index, getX(${unpackedCoords}));
        }
      }
    `;
  }
}
