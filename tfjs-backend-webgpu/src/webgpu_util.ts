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

export const ARGMINMAXPROGRAM = 0x0101;
export const CLIPPROGRAM = 0x0102;
export const CONCATPROGRAM = 0x0103;
export const CONV2DMMPROGRAM = 0x0104;
export const CONV2DNAIVEPROGRAM = 0x0105;
export const CROPANDRESIZEPROGRAM = 0x0106;
export const DEPTHWISECONV2DPROGRAM = 0x0107;
export const FILLPROGRAM = 0x0108;
export const IM2COLPROGRAM = 0x0109;
export const MATMULPACKEDPROGRAM = 0x010a;
export const MATMULPROGRAM = 0x010b;
export const MAXPOOLWITHFILTERSIZEEQUALSONEPROGRAM = 0x010c;
export const PADPROGRAM = 0x010d;
export const POOL2DPROGRAM = 0x010e;
export const RESIZEBILINEARPROGRAM = 0x0110;
export const REDUCEPROGRAM = 0x0111;
export const SELECTPROGRAM = 0x0112;
export const SLICEPROGRAM = 0x0113;
export const STRIDEDSLICEPROGRAM = 0x0114;
export const TRANSPOSEPROGRAM = 0x0115;
export const TRANSPOSESHAREDPROGRAM = 0x0116;
export const BINARYOPPROGRAM= 0x0117;

export const AVG = 0x0201;
export const MAX = 0x0202;
export const MIN = 0x0203;

export const ABS = 0x0301;
export const EXP = 0x0302;
export const LOG= 0x0303;
export const NEG = 0x0304;
export const PRELU = 0x0305;
export const RELU = 0x0306;
export const RELU6 = 0x0307;
export const SIGMOID = 0x0308;
export const TANH = 0x0309;

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

  if ((dim1 > dim0) && (dim1 / dim0 >= 2)) {
    return [2, 4, 1];
  }
  if ((dim0 > dim1) && (dim0 / dim1 >= 2)) {
    return [4, 2, 1];
  }

  return [2, 2, 1];
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

export function ArrayBufferToTypedArray(data: ArrayBuffer, dtype: DataType) {
  if (dtype === 'float32') {
    return new Float32Array(data);
  } else if (dtype === 'int32') {
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

export const hashForActivation: {[key: string]: number;} = {
  ['linear'] : 0x0401,
  ['relu'] : 0x0402,
  ['elu'] : 0x0403,
  ['relu6'] : 0x0404,
  ['prelu'] : 0x0405
};

export const hashForReduce: {[key: string]: number;} = {
  ['min'] : 0x0406,
  ['max'] : 0x0407,
  ['sum'] : 0x0408
};

export const hashForCrop: {[key: string]: number;} = {
  ['bilinear'] : 0x0409,
  ['nearest'] : 0x040a
};

export const hashForType: {[key : string]: number;} = {
  ['float32'] : 0x040b,
  ['string'] : 0x040c,
  ['int32'] : 0x040d,
  ['bool'] : 0x040e,
  ['complex64'] : 0x040f
};