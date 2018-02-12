/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

export class Copy2DProgram implements GPGPUProgram {
  variableNames = ['source'];
  outputShape: number[];
  userCode: string;

  constructor(srcNumCols: number, destNumCols: number) {
    this.outputShape = null;
    this.userCode = `
      uniform ivec2 sourceStart;
      uniform ivec2 destStart;

      void main() {
        ivec2 destCoords = getOutputCoords() - destStart;
        int index = destCoords.x * ${destNumCols} + destCoords.y;
        int r = index / ${srcNumCols};
        ivec2 sourceCoords = sourceStart + ivec2(r, index - r * ${srcNumCols});
        setOutput(getSource(sourceCoords.x, sourceCoords.y));
      }
    `;
  }

  getCustomSetupFunc(
      sourceStart: [number, number], destStart: [number, number],
      destSize: [number, number]) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      gpgpu.setOutputMatrixWriteRegion(
          destStart[0], destSize[0], destStart[1], destSize[1]);
      const sourceStartCRLoc =
          gpgpu.getUniformLocation(webGLProgram, 'sourceStart');
      gpgpu.gl.uniform2i(sourceStartCRLoc, sourceStart[0], sourceStart[1]);
      const destStartCRLoc =
          gpgpu.getUniformLocation(webGLProgram, 'destStart');
      gpgpu.gl.uniform2i(destStartCRLoc, destStart[0], destStart[1]);
    };
  }
}
