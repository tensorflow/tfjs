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

import {GPGPUProgram} from './gpgpu_math';
import * as util from '../../util';

export class AddScaledMatProgram implements GPGPUProgram {
  variableNames = ['A', 'B', 'c1', 'c2'];
  params: Array<{}> = [];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting = true;

  constructor(aShape: number[], bShape: number[]) {
    this.outputShape = util.assertAndGetBroadcastedShape(aShape, bShape);
    this.userCode = `
      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        float c1 = getC1();
        float c2 = getC2();
        setOutput(dot(vec2(c1, c2), vec2(a, b)));
      }
    `;
  }
}
