/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {getCoordsDataType} from './shader_preprocessor';

// Generates GLSL that computes strides.
export function symbolicallyComputeStrides(
    indicesArr: number[], variableName: string): string[] {
  if (Math.max(...indicesArr) > 3) {
    throw new Error('Cannot symbolically compute strides for rank > 4 tensor.');
  }

  const numCoords = indicesArr.length;
  const shape = indicesArr.map(d => `${variableName}[${d}]`);
  const strides = new Array(numCoords - 1);
  strides[numCoords - 2] = shape[numCoords - 1];
  for (let i = numCoords - 3; i >= 0; --i) {
    strides[i] = `(${strides[i + 1]} * ${shape[i + 1]})`;
  }

  return strides;
}

/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
export function generateGetOutputCoords(
    dispatchLayout: {x: number[], y: number[], z: number[]},
    rank: number): string {
  const dtype = getCoordsDataType(rank);
  let gatherDimensionsStr = '';
  const dims = [dispatchLayout.x, dispatchLayout.y, dispatchLayout.z];

  for (let i = 0; i < dims.length; i++) {
    const arr = dims[i];

    if (arr.length === 1) {
      gatherDimensionsStr += `uint d${arr[0]} = gl_GlobalInvocationID[${i}];`;
    } else {
      const strides = symbolicallyComputeStrides(arr, 'outShape');
      gatherDimensionsStr += `uint index${i} = 
            gl_GlobalInvocationID[${i}];`;
      for (let j = 0; j < strides.length; j++) {
        gatherDimensionsStr += `uint d${arr[j]} = index${i} / ${strides[j]};`;

        if (j === strides.length - 1) {
          gatherDimensionsStr += `uint d${arr[j + 1]} = ` +
              `index${i} - d${arr[j]} * ${strides[j]};`;
        } else {
          gatherDimensionsStr += `index${i} -= d${arr[j]} * ${strides[j]};`;
        }
      }
    }
  }

  const dimensions = [];
  for (let i = 0; i < rank; i++) {
    dimensions.push(`d${i}`);
  }

  return `${dtype} getOutputCoords() {
    ${gatherDimensionsStr}

    return ${dtype}(${dimensions.join(',')});
  }`;
}