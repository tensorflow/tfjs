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

import {tensor} from '../ops/tensor_ops';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {TypedArray} from '../types';
import {sizeFromShape} from '../util';
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
    const size = sizeFromShape(shape);
    let typedArray: TypedArray;

    if ('quantization' in spec) {
      const quantization = spec.quantization;
      if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
        throw new Error(
            `Weight ${spec.name} has unknown ` +
            `quantization dtype ${quantization.dtype}. ` +
            `Supported quantization dtypes are: 'uint8' and 'uint16'.`);
      }
      const quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
      const byteBuffer =
          buffer.slice(offset, offset + size * quantizationSizeFactor);
      const quantizedArray = (quantization.dtype === 'uint8') ?
          new Uint8Array(byteBuffer) :
          new Uint16Array(byteBuffer);
      if (dtype === 'float32') {
        typedArray = Float32Array.from(
            quantizedArray, v => v * quantization.scale + quantization.min);
      } else if (dtype === 'int32') {
        typedArray = Int32Array.from(
            quantizedArray,
            v => Math.round(v * quantization.scale + quantization.min));
      } else {
        throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
      }
      offset += size * quantizationSizeFactor;
    } else {
      const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
      const byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);

      if (dtype === 'float32') {
        typedArray = new Float32Array(byteBuffer);
      } else if (dtype === 'int32') {
        typedArray = new Int32Array(byteBuffer);
      } else if (dtype === 'bool') {
        typedArray = new Uint8Array(byteBuffer);
      } else {
        throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
      }
      offset += size * dtypeFactor;
    }

    let value: Tensor;
    if (dtype === 'float32') {
      value = tensor(typedArray, shape, 'float32');
    } else if (dtype === 'int32') {
      value = tensor(typedArray, shape, 'int32');
    } else if (dtype === 'bool') {
      value = tensor(typedArray, shape, 'bool');
    } else {
      throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
    }
    out[name] = value;
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

  // `normalizedXs` is here for this reason: a `TypedArray`'s `buffer'
  // can have a different byte length from that of the `TypedArray` itself,
  // for example, when the `TypedArray` is created from an offset in an
  // `ArrayBuffer`. `normliazedXs` holds `TypedArray`s whose `buffer`s match
  // the `TypedArray` in byte length. If an element of `xs` does not show
  // this property, a new `TypedArray` that satisfy this property will be
  // constructed and pushed into `normalizedXs`.
  const normalizedXs: TypedArray[] = [];
  xs.forEach((x: TypedArray) => {
    totalByteLength += x.byteLength;
    // tslint:disable:no-any
    normalizedXs.push(
        x.byteLength === x.buffer.byteLength ? x :
                                               new (x.constructor as any)(x));
    if (!(x as any instanceof Float32Array || x as any instanceof Int32Array ||
          x as any instanceof Uint8Array)) {
      throw new Error(`Unsupported TypedArray subtype: ${x.constructor.name}`);
    }
    // tslint:enable:no-any
  });

  const y = new Uint8Array(totalByteLength);
  let offset = 0;
  normalizedXs.forEach((x: TypedArray) => {
    y.set(new Uint8Array(x.buffer), offset);
    offset += x.byteLength;
  });

  return y.buffer;
}

// Use Buffer on Node.js instead of Blob/atob/btoa
const useNodeBuffer = typeof Buffer !== 'undefined' &&
    (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
     typeof btoa === 'undefined');

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
  if (useNodeBuffer) {
    return Buffer.byteLength(str);
  }
  return new Blob([str]).size;
}

/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
export function arrayBufferToBase64String(buffer: ArrayBuffer): string {
  if (useNodeBuffer) {
    return Buffer.from(buffer).toString('base64');
  }
  return btoa(String.fromCharCode.apply(null, new Uint8Array(buffer)));
}

/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
export function base64StringToArrayBuffer(str: string): ArrayBuffer {
  if (useNodeBuffer) {
    const buf = Buffer.from(str, 'base64');
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
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
  buffers.forEach((buffer: ArrayBuffer) => {
    totalByteLength += buffer.byteLength;
  });

  const temp = new Uint8Array(totalByteLength);
  let offset = 0;
  buffers.forEach((buffer: ArrayBuffer) => {
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
