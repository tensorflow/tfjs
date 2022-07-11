/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

export enum CumOpType {
  Prod = '*',
  Sum = '+',
}

export class CumProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number];
  // pow(i32, i32) is not supported, use pow(f32, f32) instead.
  uniforms = 'index : f32,';
  size = true;
  exclusive: boolean;
  reverse: boolean;
  op: CumOpType;

  constructor(
      op: CumOpType, shape: number[], exclusive: boolean, reverse: boolean) {
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.exclusive = exclusive;
    this.reverse = reverse;
    this.op = op;
    this.shaderKey = `cum_${this.op}_${this.exclusive}_${this.reverse}`;
  }

  getUserCode(): string {
    const rank = this.outputShape.length;
    const initVal = this.op === CumOpType.Prod ? '1.0' : '0.0';
    const val = this.exclusive ? initVal :
                                 `getX(${getCoords(rank, 'coords', this.op)})`;
    const length = this.outputShape[this.outputShape.length - 1];
    let condition = '';
    let idxString = '';
    // When exclusive is set, the cum op becomes roll op that copies the
    // value from the previous index based on the direction specified by the
    // reverse flag.
    if (this.exclusive) {
      condition = this.reverse ? `end != ${length - 1}` : 'end != 0';
      idxString = this.reverse ? 'end + 1' : 'end - 1';
    } else {
      condition = this.reverse ? `end + pow2 < ${length}` : 'end >= pow2';
      idxString = (this.reverse ? 'end + pow2' : 'end - pow2');
    }
    return `
      ${getMainHeaderAndGlobalIndexString()}
       if (index < uniforms.size) {
         var coords = getCoordsFromIndex(index);

         let end = ${getFinalCoord(rank, 'coords', this.op)};
         var val = ${val};
         let pow2 = i32(pow(2.0, uniforms.index));
         if (${condition}) {
           let idx = ${idxString};
           ${getFinalCoord(rank, 'coords', this.op)} = idx;
           val ${this.op}= getX(${getCoords(rank, 'coords', this.op)});
         }
         setOutputAtIndex(index, val);
       }
      }
    `;
  }
}

function getCoords(rank: number, name: string, op: CumOpType): string {
  if (rank === 1) {
    return `${name}`;
  } else if (rank === 2) {
    return `${name}.x, ${name}.y`;
  } else if (rank === 3) {
    return `${name}.x, ${name}.y, ${name}.z`;
  } else if (rank === 4) {
    return `${name}.x, ${name}.y, ${name}.z, ${name}.w`;
  } else {
    throw Error(`Cumulative ${op} for rank ${rank} is not yet supported`);
  }
}

function getFinalCoord(rank: number, name: string, op: CumOpType): string {
  if (rank === 1) {
    return `${name}`;
  } else if (rank === 2) {
    return `${name}.y`;
  } else if (rank === 3) {
    return `${name}.z`;
  } else if (rank === 4) {
    return `${name}.w`;
  } else {
    throw Error(`Cumulative ${op} for rank ${rank} is not yet supported`);
  }
}
