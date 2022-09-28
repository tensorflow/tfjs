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
      const output = this.getOutput(channels);

      this.userCode = `
         void main() {
           ${dtype} rc = getOutputCoords();

           if(${outOfBoundsCondition}) {
             setOutput(vec4(0));
           } else {
             setOutput(vec4(${output}));
           }
         }
       `;
    }
  }

  private getSourceCoordsPrefix(dims: string[]): string {
    let coord = '';
    for (let d = 1; d < this.rank; d++) {
      coord = `${dims[dims.length - 1 - d]}, ` + coord;
    }
    return coord;
  }

  private getOutOfBoundsCondition(dims: string[]): string {
    if (this.rank === 1) {
      return `rc > ${
          this.enableShapeUniforms ? 'outShape' : this.outputShape[0]}`;
    }

    let cond = `${dims[this.rank - 1]} >= ${
        this.enableShapeUniforms ? `outShape[${this.rank - 1}]` :
                                   this.outputShape[this.rank - 1]}`;
    return cond;
  }

  private getOutput(dims: string[]): string {
    const sourceCoordsPrefix = this.getSourceCoordsPrefix(dims);
    const lastDim = this.rank === 1 ? 'rc' : dims[this.rank - 1];
    let lastDimSize;
    if (this.rank === 1) {
      lastDimSize = this.enableShapeUniforms ? 'outShape' : this.outputShape[0];
    } else {
      lastDimSize = this.enableShapeUniforms ? `outShape[${this.rank} - 1]` :
                                               this.outputShape[this.rank - 1];
    }

    return `getA(${sourceCoordsPrefix}${lastDim}),
     ${lastDim} + 1 >= ${lastDimSize} ? 0. : getA(${sourceCoordsPrefix}${
        lastDim} + 1),
         ${lastDim} + 2 >= ${lastDimSize} ? 0. : getA(${sourceCoordsPrefix}${
        lastDim} + 2),
         ${lastDim} + 3 >= ${lastDimSize} ? 0. : getA(${sourceCoordsPrefix}${
        lastDim} + 3)`;
  }
}
