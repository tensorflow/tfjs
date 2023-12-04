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
import {DataType, TensorInfo, util} from '@tensorflow/tfjs-core';

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
// workgroupSize.
export function computeDispatch(
    layout: {x: number[], y?: number[], z?: number[]}, outputShape: number[],
    workgroupSize: [number, number, number] = [1, 1, 1],
    elementsPerThread: [number, number, number] =
        [1, 1, 1]): [number, number, number] {
  const [dispatchX, dispatchY, dispatchZ] = [
    Math.ceil(
        arrayProduct(layout.x.map(d => outputShape[d])) /
        (workgroupSize[0] * elementsPerThread[0])),
    layout.y ? Math.ceil(
                   arrayProduct(layout.y.map(d => outputShape[d])) /
                   (workgroupSize[1] * elementsPerThread[1])) :
               1,
    layout.z ? Math.ceil(
                   arrayProduct(layout.z.map(d => outputShape[d])) /
                   (workgroupSize[2] * elementsPerThread[2])) :
               1
  ];
  return [dispatchX, dispatchY, dispatchZ];
}

export type WorkgroupInfo = {
  workgroupSize: [number, number, number],
  elementsPerThread: [number, number, number],
};

export function computeWorkgroupInfoForMatMul(
    dimAOuter: number, dimInner: number, dimBOuter: number,
    transposeA = false): WorkgroupInfo {
  // These are experimental values. Usually, we need to adjust the work group
  // size based on the input shapes to improve the EU occupancy.
  // TODO: WebGPU limits the maximum allowed shared memory size as 16K. To make
  // sure it doesn't exceed this limitations. Temporarily reduce the work group
  // size to [8, 8, 1] and the work per thread size is [4, 4, 1]. But we should
  // revisit it and find the balance between work group size and work per thread
  // size.
  const workgroupSize: [number, number, number] = [8, 8, 1];
  const elementsPerThread: [number, number, number] = [4, 4, 1];

  if (!transposeA) {
    if (dimAOuter <= 8) {
      elementsPerThread[1] = 1;
    }

    if (dimInner <= 16 && dimBOuter <= 16) {
      workgroupSize[0] = 4;
    }
  }

  return {workgroupSize, elementsPerThread};
}

export function computeWorkgroupSizeForConv2d(
    layout: {x: number[], y?: number[], z?: number[]}, outputShape: number[],
    isVec4 = false): [number, number, number] {
  if (isVec4) {
    return [8, 8, 1];
  }

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

export function computeWorkPerThreadForConv2d(
    layout: {x: number[], y?: number[], z?: number[]}, outputShape: number[],
    isVec4 = false): [number, number, number] {
  if (isVec4) {
    return [4, 4, 1];
  }

  const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
  const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
  // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
  // The following conditions correspond to the values set in
  // computeWorkgroupSizeForConv2d.
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

export function isWebGPUSupported(): boolean {
  return !!(typeof globalThis !== 'undefined' && (globalThis.navigator)
    && (globalThis.navigator.gpu));
}

export function assertNotComplex(
    tensor: TensorInfo|TensorInfo[], opName: string): void {
  if (!Array.isArray(tensor)) {
    tensor = [tensor];
  }
  tensor.forEach(t => {
    if (t != null) {
      util.assert(
          t.dtype !== 'complex64',
          () => `${opName} does not support complex64 tensors ` +
              'in the WebGPU backend.');
    }
  });
}

export enum MatMulProgramType {
  MatMulReduceProgram,
  MatMulSplitKProgram,
  MatMulSmallOutputSizeProgram,
  MatMulPackedProgram,
  MatMulMax
}
