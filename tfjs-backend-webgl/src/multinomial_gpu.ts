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

import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';

export class MultinomialProgram implements GPGPUProgram {
  variableNames = ['probs'];
  outputShape: number[];
  userCode: string;

  // Caching uniform location for speed.
  seedLoc: WebGLUniformLocation;

  constructor(batchSize: number, numOutcomes: number, numSamples: number) {
    this.outputShape = [batchSize, numSamples];

    this.userCode = `
      uniform float seed;

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${numOutcomes - 1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${numOutcomes - 1}));
      }
    `;
  }

  getCustomSetupFunc(seed: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.seedLoc == null) {
        this.seedLoc = gpgpu.getUniformLocation(webGLProgram, 'seed');
      }
      gpgpu.gl.uniform1f(this.seedLoc, seed);
    };
  }
}
