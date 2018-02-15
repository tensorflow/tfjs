/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

// Lookup table for non - utf8, with necessary escapes at(o >= 127 or o < 32)
import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from 'deeplearn';

import {tensorflow} from './index';
import {TensorMap} from './types';

export function loadRemoteProtoFile(url: string): Promise<tensorflow.GraphDef> {
  return fetch(new Request(url))
      .then(res => res.arrayBuffer())
      .then(buffer => tensorflow.GraphDef.decode(new Uint8Array(buffer)));
}


export function loadRemoteWeightFile(url: string): Promise<ArrayBuffer> {
  return fetch(new Request(url)).then(res => res.arrayBuffer());
}

export async function buildWeightMap(
    graphPromise: Promise<tensorflow.GraphDef>,
    weightPromise: Promise<ArrayBuffer>): Promise<TensorMap> {
  const tensorMap: TensorMap = {};
  return Promise.all([graphPromise, weightPromise]).then(([graph, weight]) => {
    const constNodes = graph.node.filter(node => node.op === 'Const');
    constNodes.forEach(node => {
      const index = node.attr['index'].i as number;
      const length = node.attr['length'].i as number;
      const tensor = node.attr['value'].tensor;
      tensorMap[node.name] = buildTensor(index, length, weight, tensor);
    });
    return tensorMap;
  });
}

function buildTensor(
    index: number, length: number, weight: ArrayBuffer,
    tensor: tensorflow.ITensor): NDArray {
  const dims = tensor.tensorShape.dim;
  const dimSizes = dims.map(dim => dim.size) as number[];
  switch (tensor.dtype) {
    case tensorflow.DataType.DT_INT32: {
      const values = new Int32Array(weight, index, length);
      return toNDArray(dimSizes, values, 'int32');
    }
    case tensorflow.DataType.DT_FLOAT: {
      const values = new Float32Array(weight, index, length);
      return toNDArray(dimSizes, values, 'float32');
    }
    case tensorflow.DataType.DT_BOOL: {
      const values = new Uint8Array(weight, index, length);
      return toNDArray(dimSizes, values, 'bool');
    }
    default:
      throw new Error(`tensor data type: ${tensor.dtype} is not supported`);
  }
}


function toNDArray(
    shape: number[],
    values: boolean[]|number[]|Int32Array|Float32Array|Uint8Array,
    dtype: 'float32'|'int32'|'bool'): NDArray {
  if (values instanceof Int32Array || values instanceof Float32Array ||
      values instanceof Uint8Array) {
    values = Array.prototype.slice.call(values);
  }

  if (values.length === 1) {
    const len = shape.reduce((accu, curr) => accu * curr, 1);
    values = new Array(len).fill(values[0]);
  }

  switch (shape.length) {
    case 0:
      return Scalar.new(values[0], dtype);
    case 1: {
      return Array1D.new(values, dtype);
    }
    case 2:
      return Array2D.new(shape as [number, number], values, dtype);
    case 3:
      return Array3D.new(shape as [number, number, number], values, dtype);
    case 4:
      return Array4D.new(
          shape as [number, number, number, number], values, dtype);
    default:
      throw new Error('dimension higher than 4 is not supported');
  }
}
