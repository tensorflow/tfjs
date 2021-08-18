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

import {util} from '@tensorflow/tfjs-core';

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

export function getOutputLogicalCoordinatesFromFlatIndexByUniform(
    coords: string[], shape: number[], index = 'index'): string {
  const strides = util.computeStrides(shape);
  return strides
      .map((_, i) => {
        const line1 = `int ${coords[i]} = ${index} / outShapeStrides[${i}]`;
        const line2 = i === strides.length - 1 ?
            `int ${coords[i + 1]} = ${index} - ${coords[i]} * outShapeStrides[${
                i}]` :
            `index -= ${coords[i]} * outShapeStrides[${i}]`;
        return `${line1}; ${line2};`;
      })
      .join('');
}

// Produces GLSL code that computes strides.
function symbolicallyComputeStrides(
    indicesArr: number[], variableName: string): string[] {
  const numCoords = indicesArr.length;
  const shape = indicesArr.map(d => `${variableName}[${d}]`);
  const strides = new Array(numCoords - 1);
  strides[numCoords - 2] = shape[numCoords - 1];
  for (let i = numCoords - 3; i >= 0; --i) {
    strides[i] = `(${strides[i + 1]} * ${shape[i + 1]})`;
  }

  return strides;
}

export function getLogicalCoordinatesFromFlatIndexByUniform(
    coords: string[], variableName: string, index = 'index'): string {
  const indicesArray = coords.map((_, i) => i);
  const strides = symbolicallyComputeStrides(indicesArray, variableName);
  return strides
      .map((_, i) => {
        const line1 = `int ${coords[i]} = ${index} / ${strides[i]}`;
        const line2 = i === strides.length - 1 ?
            `int ${coords[i + 1]} = ${index} - ${coords[i]} * ${strides[i]}` :
            `index -= ${coords[i]} * ${strides[i]}`;
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

/**
 * Produces GLSL that computes the flat index from 3D coordinates.
 */
export function getFlatIndexFrom3D(shape: [number, number, number]): string {
  const strides = util.computeStrides(shape).map(d => d.toString());

  return `
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${strides[0]} + coords.y * ${strides[1]} + coords.z;
  }
`;
}

export function getFlatIndexFrom3DOutput(): string {
  return `
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`;
}

export const ENCODE_FLOAT_SNIPPET = `
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`;
