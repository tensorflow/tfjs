/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import {UniformType} from './shader_compiler';

export class ClipProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];
  customUniforms = [
    {name: 'minVal', type: 'float' as UniformType},
    {name: 'maxVal', type: 'float' as UniformType}
  ];

  constructor(aShape: number[]) {
    this.outputShape = aShape;
    this.userCode = `

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `;
  }
}
