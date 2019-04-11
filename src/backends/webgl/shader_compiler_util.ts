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

import * as util from '../../util';

/**
 * Produces GLSL code that derives logical coordinates from a flat
 * index. The code performs integer division with each stride and decrements
 * the index until the index equals the final dimension coordinate.
 */
export function getLogicalCoordinatesFromFlatIndex(
    coords: string[], shape: number[], index = 'index'): string {
  const strides = util.computeStrides(shape);
  return strides
      .map((stride, i) => {
        const line1 = `int ${coords[i]} = ${index} / ${stride}`;
        const line2 = i === strides.length - 1 ?
            `int ${coords[i + 1]} = ${index} - ${coords[i]} * ${stride}` :
            `index -= ${coords[i]} * ${stride}`;
        return `${line1}; ${line2};`;
      })
      .join('');
}

function buildVec(x: string[]): string {
  if (x.length === 1) {
    return `${x[0]}`;
  }
  return `vec${x.length}(${x.join(',')})`;
}

/**
 * Produces GLSL code that computes the dot product of the input x and y
 * vectors. Handles splitting inputs into increments of vec4s when necessary.
 */
export function dotify(x: string[], y: string[]): string {
  if (x.length !== y.length) {
    throw new Error(
        `Vectors to be dotted must be of the same length -` +
        `got ${x.length} and ${y.length}`);
  }

  const slices: string[] = [];
  const nearestVec4 = Math.floor(x.length / 4);
  const nearestVec4Remainder = x.length % 4;

  for (let i = 0; i < nearestVec4; i++) {
    const xSlice = x.slice(i * 4, i * 4 + 4);
    const ySlice = y.slice(i * 4, i * 4 + 4);
    slices.push(`${buildVec(xSlice)}, ${buildVec(ySlice)}`);
  }

  if (nearestVec4Remainder !== 0) {
    let xSlice = x.slice(nearestVec4 * 4);
    let ySlice = y.slice(nearestVec4 * 4);
    if (xSlice.length === 1) {
      xSlice = xSlice.map(d => `float(${d})`);
      ySlice = ySlice.map(d => `float(${d})`);
    }
    slices.push(`${buildVec(xSlice)}, ${buildVec(ySlice)}`);
  }

  return slices.map((d, i) => `dot(${d})`).join('+');
}
