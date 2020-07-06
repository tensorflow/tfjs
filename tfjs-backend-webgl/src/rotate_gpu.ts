/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

export class RotateProgram implements GPGPUProgram {
  variableNames = ['Image'];
  outputShape: number[] = [];
  userCode: string;

  constructor(
      imageShape: [number, number, number, number], radians: number,
      fillValue: number|[number, number, number],
      center: number|[number, number]) {
    const imageHeight = imageShape[1];
    const imageWidth = imageShape[2];
    const sinFactor = Math.sin(-radians).toFixed(3);
    const cosFactor = Math.cos(-radians).toFixed(3);
    this.outputShape = imageShape;

    const centerX =
        (imageWidth * (typeof center === 'number' ? center : center[0]))
            .toFixed(3);
    const centerY =
        (imageHeight * (typeof center === 'number' ? center : center[1]))
            .toFixed(3);

    let fillSnippet = '';
    if (typeof fillValue === 'number') {
      fillSnippet = `float outputValue = ${fillValue.toFixed(2)};`;
    } else {
      fillSnippet = `
        vec3 fill = vec3(${fillValue.join(',')});
        float outputValue = fill[coords[3]];`;
    }

    this.userCode = `
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - ${centerX}) * ${
        cosFactor} - (float(y) - ${centerY}) * ${sinFactor};
          float coordYFloat = (float(x) - ${centerX}) * ${
        sinFactor} + (float(y) - ${centerY}) * ${cosFactor};
          int coordX = int(round(coordXFloat + ${centerX}));
          int coordY = int(round(coordYFloat + ${centerY}));
          ${fillSnippet}
          if(coordX >= 0 && coordX < ${imageWidth} && coordY >= 0 && coordY < ${
        imageHeight}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `;
  }
}
