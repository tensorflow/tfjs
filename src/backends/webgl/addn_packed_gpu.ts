/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

export class AddNPackedProgram implements GPGPUProgram {
  variableNames: string[];
  outputShape: number[] = [];
  userCode: string;
  usesPackedTextures = true;

  constructor(outputShape: number[], shapes: number[][]) {
    this.outputShape = outputShape;
    this.variableNames = shapes.map((_, i) => `T${i}`);

    const snippets: string[] = [];
    // Get target elements from every input tensor.
    this.variableNames.forEach(variable => {
      snippets.push(
        `vec4 v${variable} = get${variable}AtOutCoords();`
      );
    });

    // Calculate the sum of all elements.
    const operation = this.variableNames.map(variable => {
      return `v${variable}`;
    }).join(' + ');

    this.userCode = `
      void main() {
        ${snippets.join('\n        ')}

        vec4 result = ${operation};
        setOutput(result);
      }
    `;
  }
}
