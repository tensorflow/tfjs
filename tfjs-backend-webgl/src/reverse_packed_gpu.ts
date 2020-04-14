/**
 * @license
 * Copyright 2019 Google LLC All Rights Reserved.
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

import {GPGPUProgram} from './gpgpu_math';
import {getChannels} from './packing_util';
import {getCoordsDataType} from './shader_compiler';

export class ReversePackedProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;
  packedInputs = true;
  packedOutput = true;

  constructor(xShape: number[], axis: number[]) {
    const rank = xShape.length;
    if (rank > 4) {
      throw new Error(
          `WebGL backend: Reverse of rank-${rank} tensor is not yet supported`);
    }
    this.outputShape = xShape;
    const channels = getChannels('rc', rank);
    const nextColumn =
        `${channels[rank - 1]} + 1 < ${this.outputShape[rank - 1]}`;
    const nextRow = `${channels[rank - 2]} + 1 < ${this.outputShape[rank - 2]}`;
    const type = getCoordsDataType(rank);
    if (rank === 1) {
      this.userCode = `
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${xShape[0]} - rc - 1),
            ${xShape[0]} - rc - 1);
          if(${nextColumn}){
              result.g = getChannel(getX(${xShape[0]} - (rc  + 1) - 1),
                ${xShape[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `;
    } else {
      this.userCode = `
        void main() {
          ${type} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${getR(channels.slice())};
          if(${nextColumn}){
            result.g = ${getG(channels.slice())};
          }
          if(${nextRow}) {
            result.b = ${getB(channels.slice())};
            if(${nextColumn}) {
              result.a = ${getA(channels.slice())};
            }
          }
          setOutput(result);
        }
    `;
    }

    function getR(channels: string[]): string {
      return getChannel(channels);
    }

    function getG(channels: string[]): string {
      channels[rank - 1] = '(' + channels[rank - 1] + ` + 1)`;
      return getChannel(channels);
    }

    function getB(channels: string[]): string {
      channels[rank - 2] = '(' + channels[rank - 2] + ` + 1)`;
      return getChannel(channels);
    }

    function getA(channels: string[]): string {
      channels[rank - 1] = '(' + channels[rank - 1] + ` + 1)`;
      channels[rank - 2] = '(' + channels[rank - 2] + ` + 1)`;
      return getChannel(channels);
    }

    function getChannel(channels: string[]): string {
      const inCoordsArray = xShape.map((_, i) => getInCoord(i, channels));
      const inCoords = inCoordsArray.join(',');
      const innerDims = inCoordsArray.slice(-2).join(',');
      return `getChannel(getX(${inCoords}), vec2(${innerDims}))`;
    }

    function getInCoord(i: number, channels1: string[]): string {
      if (axis.indexOf(i) !== -1 && xShape[i] !== 1) {
        return `${xShape[i]} - ${channels1[i]} - 1`;
      } else {
        return `${channels1[i]}`;
      }
    }
  }
}
