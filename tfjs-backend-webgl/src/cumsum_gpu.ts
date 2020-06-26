/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class CumSumProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  // Caching uniform location for speed.
  index: WebGLUniformLocation;

  constructor(shape: number[], exclusive: boolean, reverse: boolean) {
    this.outputShape = shape;
    const rank = shape.length;
    const val = exclusive ? '0.0' : `getX(${getCoords(rank, 'coords')})`;
    const length = shape[shape.length - 1];
    let condition = '';
    let idxString = '';
    // When exclusive is set, the cumsum op becomes roll op that copies the
    // value from the previous index based on the direction specified by the
    // reverse flag.
    if (exclusive) {
      condition = reverse ? `end != ${length - 1}` : 'end != 0';
      idxString = reverse ? 'end + 1' : 'end - 1';
    } else {
      condition = reverse ? `end + pow2 < ${length}` : 'end >= pow2';
      idxString = (reverse ? 'end + pow2' : 'end - pow2');
    }

    this.userCode = `
      uniform float index;
      void main() {
        ${getCoordsDataType(rank)} coords = getOutputCoords();
        int end = ${getFinalCoord(rank, 'coords')};
        float val = ${val};
        int pow2 = int(pow(2.0, index));
        if (${condition}) {
          int idx = ${idxString};
          ${getFinalCoord(rank, 'coords')} = idx;
          val += getX(${getCoords(rank, 'coords')});
        }
        setOutput(val);
      }
    `;
  }

  getCustomSetupFunc(index: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.index == null) {
        this.index = gpgpu.getUniformLocation(webGLProgram, 'index');
      }
      gpgpu.gl.uniform1f(this.index, index);
    };
  }
}

function getCoords(rank: number, name: string): string {
  if (rank === 1) {
    return `${name}`;
  } else if (rank === 2) {
    return `${name}.x, ${name}.y`;
  } else if (rank === 3) {
    return `${name}.x, ${name}.y, ${name}.z`;
  } else if (rank === 4) {
    return `${name}.x, ${name}.y, ${name}.z, ${name}.w`;
  } else {
    throw Error(`Cumulative sum for rank ${rank} is not yet supported`);
  }
}

function getFinalCoord(rank: number, name: string): string {
  if (rank === 1) {
    return `${name}`;
  } else if (rank === 2) {
    return `${name}.y`;
  } else if (rank === 3) {
    return `${name}.z`;
  } else if (rank === 4) {
    return `${name}.w`;
  } else {
    throw Error(`Cumulative sum for rank ${rank} is not yet supported`);
  }
}
