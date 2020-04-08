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

import {backend_util} from '@tensorflow/tfjs-core';
import {GPGPUProgram} from './gpgpu_math';

export class ConcatProgram implements GPGPUProgram {
  variableNames: string[];
  outputShape: number[] = [];
  userCode: string;

  // Concats 2d tensors along axis=1. See comments in MathBackendWebGL.concat().
  constructor(shapes: Array<[number, number]>) {
    this.outputShape = backend_util.computeOutShape(shapes, 1 /* axis */);
    this.variableNames = shapes.map((_, i) => `T${i}`);

    const offsets: number[] = new Array(shapes.length - 1);
    offsets[0] = shapes[0][1];
    for (let i = 1; i < offsets.length; i++) {
      offsets[i] = offsets[i - 1] + shapes[i][1];
    }

    const snippets = [`if (yC < ${offsets[0]}) setOutput(getT0(yR, yC));`];
    for (let i = 1; i < offsets.length; i++) {
      const shift = offsets[i - 1];
      snippets.push(
          `else if (yC < ${offsets[i]}) ` +
          `setOutput(getT${i}(yR, yC-${shift}));`);
    }
    const lastIndex = offsets.length;
    const lastShift = offsets[offsets.length - 1];
    snippets.push(`else setOutput(getT${lastIndex}(yR, yC-${lastShift}));`);

    this.userCode = `
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${snippets.join('\n        ')}
      }
    `;
  }
}
