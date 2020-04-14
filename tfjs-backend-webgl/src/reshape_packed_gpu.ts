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

import {GPGPUProgram} from './gpgpu_math';
import * as shader_util from './shader_compiler_util';

export class ReshapePackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;

  constructor(outputShape: [number, number, number], inputShape: [
    number, number, number
  ]) {
    this.outputShape = outputShape;

    let mainLoop = ``;
    for (let i = 0; i < 4; i++) {
      let thisRC = `thisRC = rc;`;
      if (i % 2 === 1) {
        thisRC += `thisRC.z += 1;`;
      }
      if (i > 1) {
        thisRC += `thisRC.y += 1;`;
      }

      mainLoop += `
        ${thisRC}
        ${i > 0 ? `if(thisRC.y < rows && thisRC.z < cols){` : ''}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${i}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${i > 0 ? '}' : ''}
      `;
    }

    this.userCode = `
      ${getReshapedInputCoords(inputShape)}
      ${shader_util.getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${outputShape[1]};
        int cols = ${outputShape[2]};

        ${mainLoop}

        setOutput(result);
      }
    `;
  }
}

function getReshapedInputCoords(shape: [number, number, number]): string {
  const coordsFromIndexSnippet =
      shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);

  return `
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${coordsFromIndexSnippet}
      return ivec3(r, c, d);
    }
  `;
}
