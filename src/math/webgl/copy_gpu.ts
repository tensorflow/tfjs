/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';

export class Copy2DProgram implements GPGPUProgram {
  variableNames = ['source'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;

  constructor(srcNumCols: number, destNumCols: number) {
    this.outputShape = null;
    this.params = [srcNumCols, destNumCols];
    this.userCode = `
      uniform vec2 sourceStart;
      uniform vec2 destStart;

      void main() {
        vec2 destCoords = getOutputCoords() - destStart;
        float index = dot(destCoords, vec2(${destNumCols}.0, 1.0));
        vec2 sourceCoords = sourceStart + vec2(
          floor(index / ${srcNumCols}.0),
          mod(index, ${srcNumCols}.0)
        );
        setOutput(getSource(sourceCoords.x, sourceCoords.y));
      }
    `;
  }

  getCustomSetupFunc(
      sourceStart: [number, number], destStart: [number, number],
      destSize: [number, number]) {
    return (gpgpu: GPGPUContext) => {
      gpgpu.setOutputMatrixWriteRegion(
          destStart[0], destSize[0], destStart[1], destSize[1]);
      const sourceStartCRLoc = gpgpu.getUniformLocation('sourceStart');
      gpgpu.gl.uniform2f(sourceStartCRLoc, sourceStart[0], sourceStart[1]);
      const destStartCRLoc = gpgpu.getUniformLocation('destStart');
      gpgpu.gl.uniform2f(destStartCRLoc, destStart[0], destStart[1]);
    };
  }
}
