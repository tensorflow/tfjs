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

export function tilesFitEvenlyIntoShape(
    tileSize: number[], shape: number[]): boolean {
  if (tileSize.length !== shape.length) {
    throw new Error(
        `Cannot compute whether rank ${tileSize.length}` +
        ` tiles fit evenly into rank ${shape.length} shape` +
        ` - ranks must match.`);
  }
  return shape.every(
      (dim: number, dimIdx: number) => dim % tileSize[dimIdx] === 0);
}

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

export function computeWorkGroupSizeForConv2d(
    layout: {x: number[], y?: number[], z?: number[]},
    outputShape: number[]): [number, number, number] {
  const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
  const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
  // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
  // These are experimental values. Usually, we need to adjust the work group
  // size based on the output shape. For example, when one dimension is smaller
  // than 4, it will be wasteful if we assign a larger size for this dimension,
  // which results lots of threads doing useless work and reduces parallelism
  // of hardware threads. But it is always a balance between work group size
  // and shared memory. If one dimension is too small, such as 1, shared memory
  // will won't be fully utilized.
  if (dim0 <= 4) {
    return [4, 16, 1];
  }
  if (dim1 <= 4) {
    return [16, 4, 1];
  }

  return [16, 16, 1];
}

export function computeWorkGroupSizeForMatMul(
    dimAOuter: number, dimInner: number,
    dimBOuter: number): [number, number, number] {
  // These are experimental values. Usually, we need to adjust the work group
  // size based on the input shapes to improve the EU occupancy.
  // 64 (16 x 4) is the default tile size. If one dimension can't be divisible
  // by 64, it means some threads will be idle. To improve the thread
  // utilization, reducing the work group size may be a good way.
  if (dimAOuter === 1) {
    return [64, 1, 1];
  } else if (dimBOuter === 1) {
    return [1, 64, 1];
  } else if (dimInner % 64 === 0 && dimBOuter % 64 === 0) {
    return [16, 16, 1];
  } else if (dimInner < 192 && dimBOuter < 192) {
    return [8, 8, 1];
  }

  return [16, 16, 1];
}

export function computeWorkPerThreadForConv2d(
    layout: {x: number[], y?: number[], z?: number[]},
    outputShape: number[]): [number, number, number] {
  const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
  const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
  // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
  // The following conditions correspond to the values set in
  // computeWorkGroupSizeForConv2d.
  if (dim0 <= 4) {
    return [1, 2, 1];
  }
  if (dim1 <= 4) {
    return [2, 1, 1];
  }

  return [2, 2, 1];
}

export function flatDispatchLayout(shape: number[]) {
  return {x: shape.map((d, i) => i)};
}

export function GPUBytesPerElement(dtype: DataType): number {
  if (dtype === 'float32' || dtype === 'int32' || dtype === 'bool' ||
      dtype === 'string') {
    return 4;
  } else if (dtype === 'complex64') {
    return 8;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

export function ArrayBufferToTypedArray(data: ArrayBuffer, dtype: DataType) {
  if (dtype === 'float32') {
    return new Float32Array(data);
  } else if (dtype === 'int32') {
    return new Int32Array(data);
  } else if (dtype === 'bool' || dtype === 'string') {
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
