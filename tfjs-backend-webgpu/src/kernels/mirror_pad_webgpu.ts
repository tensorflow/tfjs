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

import {getCoordsDataType} from '../shader_preprocessor';
import {getCoordsDataTypeWgsl, getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class MirrorPadProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  uniforms = '';
  uniformsWgsl = '';
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  xShape: number[];
  offset: number;
  size: number;
  useWgsl: boolean;

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
      this.uniforms += ` ivec2 pad${i};`;
      this.uniformsWgsl += ` pad${i} : vec2<u32>;`;
    });
    this.offset = mode === 'reflect' ? 0 : 1;
    this.shaderKey = `mirrorPad_${mode}`;
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const rank = this.xShape.length;
    // The length of paddings are same with the rank of the input tensor.
    const start = this.xShape.map((_, i) => `pad${i}[0]`).join(',');
    const end =
        this.xShape
            .map((_, i) => `pad${i}[0] + xShape${rank > 1 ? `[${i}]` : ''}`)
            .join(',');

    const shaderStart = rank === 1 ? 'start' : 'start[i]';
    const shaderEnd = rank === 1 ? 'end' : 'end[i]';
    const shaderOutC = rank === 1 ? 'outC' : 'outC[i]';
    const dtype = getCoordsDataType(rank);
    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';

    return `
      ${dtype} start = ${dtype}(${start});
      ${dtype} end = ${dtype}(${end});

      void main() {
        ${dtype} outC = getOutputCoords();
        int index = int(gl_GlobalInvocationID.x);
        if (index < size)
        {
          for (int i = 0; i < ${rank}; i++) {
            if (${shaderOutC} < ${shaderStart}) {
              ${shaderOutC} = ${shaderStart} * 2 - ${shaderOutC} - ${
        this.offset};
            } else if(${shaderOutC} >= ${shaderEnd}) {
              ${shaderOutC} = (${shaderEnd} - 1) * 2 - ${shaderOutC} + ${
        this.offset};
            }
          }
          ${dtype} coords = outC - start;
          setOutput(index, getX(${unpackedCoords}));
        }
      }
    `;
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
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let start = ${dtype}(${start});
        let end = ${dtype}(${end});
        var outC = getOutputCoords(globalId);
        let index = globalId.x;
        if (index < uniforms.size)
        {
          for (var i = 0u; i < ${rank}u; i = i + 1u) {
            if (${shaderOutC} < ${shaderStart}) {
              ${shaderOutC} = ${shaderStart} * 2u - ${shaderOutC} - ${
        this.offset}u;
            } elseif(${shaderOutC} >= ${shaderEnd}) {
              ${shaderOutC} = (${shaderEnd} - 1u) * 2u - ${shaderOutC} + ${
        this.offset}u;
            }
          }
          let coords = outC - start;
          setOutputFlat(index, getX(${unpackedCoords}));
        }
      }
    `;
  }
}
