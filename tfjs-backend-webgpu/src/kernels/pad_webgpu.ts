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

import {util} from '@tensorflow/tfjs-core';

import {getCoordsDataType} from '../shader_preprocessor';
import {getCoordsDataTypeWgsl} from '../shader_preprocessor_wgsl';
import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class PadProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'float constantValue;';
  uniformsWgsl = 'constantValue : f32;';
  workGroupSize: [number, number, number] = [64, 1, 1];
  xShape: number[];
  size: number;
  useWgsl: boolean;

  constructor(xShape: number[], paddings: Array<[number, number]>) {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    paddings.map((_, i) => {
      this.uniforms += ` ivec2 pad${i};`;
      this.uniformsWgsl += ` pad${i} : vec2<u32>;`;
    });
    this.xShape = xShape;
    this.shaderKey = 'pad';
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const rank = this.xShape.length;
    const type = getCoordsDataType(rank);
    // The length of paddings are same with the rank of the input tensor.
    const start = this.xShape.map((_, i) => `pad${i}[0]`).join(',');
    const end =
        this.xShape
            .map((_, i) => `pad${i}[0] + xShape${rank > 1 ? `[${i}]` : ''}`)
            .join(',');
    const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
    const endValue = rank > 1 ? `${type}(${end})` : `${end}`;

    const leftPadCondition =
        rank > 1 ? `any(lessThan(outC, start))` : `outC < start`;
    const rightPadCondition =
        rank > 1 ? `any(greaterThanEqual(outC, end))` : `outC >= end`;

    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';

    const userCode = `
      ${type} start = ${startValue};
      ${type} end = ${endValue};

      void main() {
        int flatIndex = int(gl_GlobalInvocationID.x);

          if (flatIndex < size) {
            ${type} outC = getOutputCoords();

            if (${leftPadCondition} || ${rightPadCondition}) {
              setOutput(flatIndex, constantValue);
            } else {
              ${type} coords = outC - start;
              setOutput(flatIndex, getX(${unpackedCoords}));
            }
          }
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const rank = this.xShape.length;
    const type = getCoordsDataTypeWgsl(rank);
    // The length of paddings are same with the rank of the input tensor.
    const start = this.xShape.map((_, i) => `uniforms.pad${i}[0]`).join(',');
    const end = this.xShape
                    .map(
                        (_, i) => `uniforms.pad${i}[0] + uniforms.xShape${
                            rank > 1 ? `[${i}]` : ''}`)
                    .join(',');
    const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
    const endValue = rank > 1 ? `${type}(${end})` : `${end}`;

    const leftPadCondition = rank > 1 ? `any(outC < start)` : `outC < start`;
    const rightPadCondition = rank > 1 ? `any(outC >= end)` : `outC >= end`;

    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';

    const userCode = `
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let start = ${startValue};
        let end = ${endValue};
        let flatIndex = globalId.x;
        if (flatIndex < uniforms.size) {
          let outC = getOutputCoords(globalId);

          if (${leftPadCondition} || ${rightPadCondition}) {
            setOutputFlat(flatIndex, uniforms.constantValue);
          } else {
            let coords = outC - start;
            setOutputFlat(flatIndex, getX(${unpackedCoords}));
          }
        }
      }
    `;
    return userCode;
  }
}
