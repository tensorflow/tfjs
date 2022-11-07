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

const writeSnippet = `
  fn bincount_write(index: i32, value: f32) {
    var oldValue = atomicLoad(& (result[index]));
    var exchanged = false;
    for (; !exchanged;) {
      let newValueF32 = bitcast<f32>(oldValue) + value;
      let newValue = bitcast<i32>(newValueF32);
      let res = atomicCompareExchangeWeak(
          &(result[index]), oldValue, newValue);
      oldValue = res.old_value;
      exchanged = res.exchanged;
    }
  }
`;

const binaryWriteSnippet = `
  fn bincount_write(index: i32, value: f32) {
    result[index] = value;
  }
`;

export class BincountProgram implements WebGPUProgram {
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'binCountSize : i32,';
  workgroupSize: [number, number, number] = [64, 1, 1];
  atomic = true;
  hasWeights = true;
  binaryOutput = false;
  rank: number;

  constructor(
      shape: [number]|[number, number], hasWeights: boolean,
      binaryOutput = false) {
    this.outputShape = shape;
    this.rank = shape.length;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    this.binaryOutput = binaryOutput;
    if (binaryOutput) {
      this.atomic = false;
    }
    this.hasWeights = hasWeights;
    if (this.hasWeights) {
      this.variableNames.push('w');
    }
    this.shaderKey =
        `bincount_${this.hasWeights}_${this.binaryOutput}_${this.rank}`;
  }

  getUserCode(): string {
    const userCode = `
    ${this.binaryOutput ? binaryWriteSnippet : writeSnippet}
  ${main('index')} {
    ${
        this.rank === 1 ?
            `if (index < uniforms.xShape) {
      let indexVal = i32(getX(index));
      if (indexVal < uniforms.binCountSize) {
        let value = ${
                this.binaryOutput ?
                    1. :
                    (this.hasWeights ? 'f32(getW(index))' : '1.')};
        bincount_write(indexVal, value);
      }
    }` :
            `let coord = getCoordsFromIndex(index);
    if (coordsInBounds2D(coord, uniforms.xShape)) {
      let indexVal = i32(getX(coord[0], coord[1]));
      if (indexVal < uniforms.binCountSize) {
        let value = ${
                this.binaryOutput ?
                    1. :
                    (this.hasWeights ? 'f32(getW(coord[0], coord[1]))' : '1.')};
        bincount_write(coord.x * uniforms.binCountSize + indexVal, value);
      }
    }`}
  }
  `;
    return userCode;
  }
}
