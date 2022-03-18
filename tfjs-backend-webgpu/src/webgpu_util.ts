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
  const [dispatchX, dispatchY, dispatchZ] = [
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
  return [dispatchX, dispatchY, dispatchZ];
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
  // TODO: WebGPU limits the maximum allowed shared memory size as 16K. To make
  // sure it doesn't exceed this limitations. Temporarily reduce the work group
  // size to [8, 8, 1] and the work per thread size is [4, 4, 1]. But we should
  // revisit it and find the balance between work group size and work per thread
  // size.
  if (dimAOuter === 1) {
    return [32, 1, 1];
  } else if (dimBOuter === 1) {
    return [1, 32, 1];
  }

  return [8, 8, 1];
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

// AlignBy4
export function gpuSizeFromShape(shape: number[]): number {
  let size = 4;
  if (shape.length === 0) {
    // Scalar.
    return size;
  }
  size = Math.ceil(shape[shape.length - 1] / 4) * 4;
  for (let i = shape.length - 2; i >= 0; i--) {
    size *= shape[i];
  }
  return size;
}

export function ArrayBufferToTypedArray(data: ArrayBuffer, dtype: DataType) {
  if (dtype === 'float32') {
    return new Float32Array(data);
  } else if (dtype === 'int32') {
    return new Int32Array(data);
  } else if (dtype === 'bool' || dtype === 'string') {
    return Uint8Array.from(new Int32Array(data));
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

export function isWebGPUSupported(): boolean {
  return ((typeof window !== 'undefined') ||
          //@ts-ignore
          (typeof WorkerGlobalScope !== 'undefined')) &&
      !!navigator.gpu;
}

export interface WebGPULayout {
  bindGroupLayout: GPUBindGroupLayout;
  pipelineLayout: GPUPipelineLayout;
}

function LinearIndex(
    d_ch: number, y: number, x: number, s_ch: number, shape: number[]) {
  return y * shape[1] * shape[2] * shape[3] + x * shape[2] * shape[3] +
      s_ch * shape[3] + d_ch;
}

export function DivideRoundUp(n: number, divisor: number) {
  return Math.ceil(n / divisor);
}

export function RearrangeWeightsToOHWIOGroupO4I4(
    weightsData: Float32Array, weightsShape: number[]) {
  const out_group_size = 2;
  const dst_slices = DivideRoundUp(weightsShape[3], 4);
  const src_slices = DivideRoundUp(weightsShape[2], 4);
  const dst_groups = DivideRoundUp(dst_slices, out_group_size);
  const length = dst_groups * weightsShape[0] * weightsShape[1] * src_slices *
      out_group_size * 4 * 4;
  const dst = new Float32Array(length);

  var counter = 0;
  for (var d = 0; d < dst_groups; ++d) {
    for (var y = 0; y < weightsShape[0]; ++y) {
      for (var x = 0; x < weightsShape[1]; ++x) {
        for (var s = 0; s < src_slices; ++s) {
          for (var d_group = 0; d_group < out_group_size; ++d_group) {
            for (var j = 0; j < 4; ++j) {
              var filter = [];
              for (var i = 0; i < 4; ++i) {
                const s_ch = s * 4 + i;
                const d_ch = (d * out_group_size + d_group) * 4 + j;
                if (s_ch < weightsShape[2] && d_ch < weightsShape[3]) {
                  const f_index = LinearIndex(d_ch, y, x, s_ch, weightsShape);
                  filter[i] = weightsData[f_index];
                } else {
                  filter[i] = 0.0;
                }
              }
              dst[counter++] = filter[0];
              dst[counter++] = filter[1];
              dst[counter++] = filter[2];
              dst[counter++] = filter[3];
            }
          }
        }
      }
    }
  }
  return dst;
}

export function RearrangeWeights3x3Data(
    weightsData: Float32Array, weightsShape: number[]) {
  const src_depth = DivideRoundUp(weightsShape[2], 4);
  const length = 9 * src_depth * 4;
  const dst = new Float32Array(length);

  let counter = 0;
  for (let s = 0; s < src_depth; ++s) {
    for (let y = 0; y < 3; ++y) {
      for (let x = 0; x < 3; ++x) {
        const filter_val = [];
        for (let i = 0; i < 4; ++i) {
          const s_ch = s * 4 + i;
          if (s_ch < weightsShape[2]) {
            const f_index = LinearIndex(0, y, x, s_ch, weightsShape);
            filter_val[i] = weightsData[f_index];
          } else {
            filter_val[i] = 0.0;
          }
        }
        dst[counter++] = filter_val[0];
        dst[counter++] = filter_val[1];
        dst[counter++] = filter_val[2];
        dst[counter++] = filter_val[3];
      }
    }
  }
  return dst;
}

export enum DataLayout {
  NHWC,
  PHWC4,
  O4HWI4,
  OHW10  // for depthwise
}
