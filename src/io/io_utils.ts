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

import {ArrayOps} from '../ops/array_ops';
import {Tensor} from '../tensor';
import {NamedTensorMap, TypedArray} from '../types';
import {sizeFromShape} from '../util';

// tslint:disable-next-line:max-line-length
import {DTYPE_VALUE_SIZE_MAP, ModelArtifacts, ModelArtifactsInfo, WeightsManifestEntry} from './types';

/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
export async function encodeWeights(tensors: NamedTensorMap):
    Promise<{data: ArrayBuffer, specs: WeightsManifestEntry[]}> {
  // TODO(adarob, cais): Support quantization.
  const specs: WeightsManifestEntry[] = [];
  const dataPromises: Array<Promise<TypedArray>> = [];
  for (const name in tensors) {
    const t = tensors[name];

    if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool') {
      throw new Error(`Unsupported dtype in weight '${name}': ${t.dtype}`);
    }
    specs.push({name, shape: t.shape, dtype: t.dtype});
    dataPromises.push(t.data());
  }
  const tensorValues = await Promise.all(dataPromises);
  return {data: concatenateTypedArrays(tensorValues), specs};
}

/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param buffer A flat ArrayBuffer carrying the binary values of the tensors
 *   concatenated in the order specified in `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
export function decodeWeights(
    buffer: ArrayBuffer, specs: WeightsManifestEntry[]): NamedTensorMap {
  // TODO(adarob, cais): Support quantization.
  const out: NamedTensorMap = {};
  let offset = 0;
  for (const spec of specs) {
    const name = spec.name;
    const dtype = spec.dtype;
    const shape = spec.shape;

    if (spec.quantization != null) {
      throw new Error(
          `decodeWeights does not support quantization yet, but encountered ` +
          `weight '${name} with quantization.'`);
    }

    const size = sizeFromShape(shape);
    let value: Tensor;
    if (dtype === 'float32') {
      value = ArrayOps.tensor(
          new Float32Array(buffer, offset, size), shape, 'float32');
    } else if (dtype === 'int32') {
      value =
          ArrayOps.tensor(new Int32Array(buffer, offset, size), shape, 'int32');
    } else if (dtype === 'bool') {
      value =
          ArrayOps.tensor(new Uint8Array(buffer, offset, size), shape, 'bool');
    } else {
      throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
    }
    out[name] = value;

    offset += size * DTYPE_VALUE_SIZE_MAP[dtype];
  }
  return out;
}

/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
export function concatenateTypedArrays(xs: TypedArray[]): ArrayBuffer {
  // TODO(adarob, cais): Support quantization.
  if (xs === null) {
    throw new Error(`Invalid input value: ${JSON.stringify(xs)}`);
  }

  let totalByteLength = 0;
  xs.forEach(x => {
    // tslint:disable-next-line:no-any
    if (x as any instanceof Float32Array || x as any instanceof Int32Array) {
      totalByteLength += x.buffer.byteLength;
      // tslint:disable-next-line:no-any
    } else if (x as any instanceof Uint8Array) {
      totalByteLength += x.buffer.byteLength;
    } else {
      throw new Error(`Unsupported TypedArray subtype: ${x.constructor.name}`);
    }
  });

  const y = new Uint8Array(totalByteLength);
  let offset = 0;
  xs.forEach(x => {
    y.set(new Uint8Array(x.buffer), offset);
    offset += x.buffer.byteLength;
  });

  return y.buffer;
}

/**
 * Calculate the byte length of a JavaScript string.
 *
 * Note that a JavaScript string can contain wide characters, therefore the
 * length of the string is not necessarily equal to the byte length.
 *
 * @param str Input string.
 * @returns Byte length.
 */
export function stringByteLength(str: string): number {
  return new Blob([str]).size;
}

/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
export function arrayBufferToBase64String(buffer: ArrayBuffer): string {
  return btoa(String.fromCharCode.apply(null, new Uint8Array(buffer)));
}

/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
export function base64StringToArrayBuffer(str: string): ArrayBuffer {
  const s = atob(str);
  const buffer = new Uint8Array(s.length);
  for (let i = 0; i < s.length; ++i) {
    buffer.set([s.charCodeAt(i)], i);
  }
  return buffer.buffer;
}

/**
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers A number of array buffers to concatenate.
 * @returns Result of concatenating `buffers` in order.
 */
export function concatenateArrayBuffers(buffers: ArrayBuffer[]): ArrayBuffer {
  let totalByteLength = 0;
  buffers.forEach(buffer => {
    totalByteLength += buffer.byteLength;
  });

  const temp = new Uint8Array(totalByteLength);
  let offset = 0;
  buffers.forEach(buffer => {
    temp.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  });
  return temp.buffer;
}

/**
 * Get the basename of a path.
 *
 * Behaves in a way analogous to Linux's basename command.
 *
 * @param path
 */
export function basename(path: string): string {
  const SEPARATOR = '/';
  path = path.trim();
  while (path.endsWith(SEPARATOR)) {
    path = path.slice(0, path.length - 1);
  }
  const items = path.split(SEPARATOR);
  return items[items.length - 1];
}

/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
export function getModelArtifactsInfoForJSON(modelArtifacts: ModelArtifacts):
    ModelArtifactsInfo {
  if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
    throw new Error('Expected JSON model topology, received ArrayBuffer.');
  }

  return {
    dateSaved: new Date(),
    modelTopologyType: 'JSON',
    modelTopologyBytes: modelArtifacts.modelTopology == null ?
        0 :
        stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
    weightSpecsBytes: modelArtifacts.weightSpecs == null ?
        0 :
        stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
    weightDataBytes: modelArtifacts.weightData == null ?
        0 :
        modelArtifacts.weightData.byteLength,
  };
}
