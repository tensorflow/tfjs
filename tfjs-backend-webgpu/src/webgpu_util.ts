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
import {DataType} from '@tensorflow/tfjs-core';

const arrayProduct = (arr: number[]) => {
  let product = 1;
  for (let i = 0; i < arr.length; i++) {
    product *= arr[i];
  }
  return product;
};

// Computes dispatch geometry based on layout of output dimensions and
// workGroupSize.
export function computeDispatch(
    layout: {x: number[], y?: number[], z?: number[]}, outputShape: number[],
    workGroupSize: [number, number, number] = [1, 1, 1],
    elementsPerThread: [number, number, number] =
        [1, 1, 1]): [number, number, number] {
  return [
    Math.ceil(
        arrayProduct(layout.x.map(d => outputShape[d])) /
        (workGroupSize[0] * elementsPerThread[0])),
    layout.y ? Math.ceil(
                   arrayProduct(layout.y.map(d => outputShape[d])) /
                   (workGroupSize[1] * elementsPerThread[1])) :
               1,
    layout.z ? Math.ceil(
                   arrayProduct(layout.z.map(d => outputShape[d])) /
                   (workGroupSize[2] * elementsPerThread[2])) :
               1
  ];
}

export function flatDispatchLayout(shape: number[]) {
  return {x: shape.map((d, i) => i)};
}

export function GPUBytesPerElement(dtype: DataType): number {
  if (dtype === 'float32' || dtype === 'int32' || dtype === 'bool') {
    return 4;
  } else if (dtype === 'complex64') {
    return 8;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

export function ArrayBufferToTypedArray(
    data: ArrayBuffer, dtype: DataType) {
  if (dtype === 'float32') {
    return new Float32Array(data);
  }
  else if (dtype === 'int32') {
    return new Int32Array(data);
  } else if (dtype === 'bool') {
    const dataAsInt32Array = new Int32Array(data);
    const boolData = new ArrayBuffer(dataAsInt32Array.length);
    const dataAsTypedArray = new Uint8Array(boolData);
    for (let i = 0; i < dataAsInt32Array.length; i++) {
      dataAsTypedArray[i] = dataAsInt32Array[i];
    }
    return dataAsTypedArray;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}
