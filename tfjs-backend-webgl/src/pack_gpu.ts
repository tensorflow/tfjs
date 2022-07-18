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

import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';
import {getChannels} from './packing_util';
import {getCoordsDataType} from './shader_compiler';

export class PackProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;
  packedInputs = false;
  packedOutput = true;
  enableShapeUniforms: boolean;
  rank: number;

  constructor(
      outputShape:
          number[]) {  // TODO(https://github.com/tensorflow/tfjs/issues/893):
                       // Only input / output 3D tensors.
    this.outputShape = outputShape;
    this.rank = outputShape.length;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    if (this.rank === 0) {
      this.userCode = `
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;
    } else {
      const channels = getChannels('rc', this.rank);
      const dtype = getCoordsDataType(this.rank);
      const outOfBoundsCondition = this.getOutOfBoundsCondition(channels);
      const setup = this.getSetup(channels);
      const output = this.getOutput(channels);

      this.userCode = `
        void main() {
          ${dtype} rc = getOutputCoords();

          if(${outOfBoundsCondition}) {
            setOutput(vec4(0));
          } else {
            ${setup}

            setOutput(vec4(${output}));
          }
        }
      `;
    }
  }

  private getSourceCoordsArr(dims: string[]): string[] {
    const coords = [];

    for (let row = 0; row <= 1; row++) {
      for (let col = 0; col <= 1; col++) {
        let coord = `${row === 0 ? 'r' : 'rp1'}, ${col === 0 ? 'c' : 'cp1'}`;

        for (let d = 2; d < this.rank; d++) {
          coord = `${dims[dims.length - 1 - d]},` + coord;
        }

        coords.push(coord);
      }
    }
    return coords;
  }

  private getOutOfBoundsCondition(dims: string[]): string {
    if (this.rank === 1) {
      return `rc > ${
          this.enableShapeUniforms ? 'outShape' : this.outputShape[0]}`;
    }

    let cond = '';
    for (let i = this.rank - 2; i < this.rank; i++) {
      cond += `${dims[i]} >= ${
          this.enableShapeUniforms ? `outShape[${i}]` : this.outputShape[i]}`;
      if (i < this.rank - 1) {
        cond += '||';
      }
    }

    return cond;
  }

  private getSetup(dims: string[]): string {
    if (this.rank === 1) {
      return '';
    }

    const innerDims = dims.slice(-2);
    const col = this.enableShapeUniforms ? `outShape[${this.rank} - 1]` :
                                           this.outputShape[this.rank - 1];
    const row = this.enableShapeUniforms ? `outShape[${this.rank} - 2]` :
                                           this.outputShape[this.rank - 2];

    return `
      int r = ${innerDims[0]};
      int c = ${innerDims[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${col};
      bool rEdge = rp1 >= ${row};
    `;
  }

  private getOutput(dims: string[]): string {
    const sourceCoords = this.getSourceCoordsArr(dims);
    if (this.rank === 1) {
      const outShape =
          this.enableShapeUniforms ? 'outShape' : this.outputShape[0];
      return `getA(rc), (rc + 1 >= ${outShape} ? 0. : getA(rc + 1)), 0, 0`;
    }

    return `getA(${sourceCoords[0]}),
            cEdge ? 0. : getA(${sourceCoords[1]}),
            rEdge ? 0. : getA(${sourceCoords[2]}),
            rEdge || cEdge ? 0. : getA(${sourceCoords[3]})`;
  }
}
