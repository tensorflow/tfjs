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

const arrayProduct = (arr: number[]) => {
  if (!arr.length) {
    throw new Error('Cannot compute product of empty array.');
  }
  let product = 1;
  for (let i = 0; i < arr.length; i++) {
    product *= arr[i];
  }
  return product;
};

// Computes dispatch geometry based on layout of output dimensions and tileSize.
export function computeDispatch(
    layout: {x: number[], y: number[], z: number[]}, outputShape: number[],
    tileSize: [number, number, number] = [1, 1, 1]): [number, number, number] {
  return [
    Math.ceil(arrayProduct(layout.x.map(d => outputShape[d])) / tileSize[0]),
    Math.ceil(arrayProduct(layout.y.map(d => outputShape[d])) / tileSize[1]),
    Math.ceil(arrayProduct(layout.z.map(d => outputShape[d])) / tileSize[2])
  ];
}