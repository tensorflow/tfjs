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

import {complex} from '../ops/complex';

import {tensor} from '../ops/tensor';
import {NamedTensor, NamedTensorMap} from '../tensor_types';
import {TypedArray} from '../types';
import {sizeFromShape} from '../util';

import {DTYPE_VALUE_SIZE_MAP, ModelArtifacts, ModelArtifactsInfo, WeightGroup, WeightsManifestEntry} from './types';

/** Number of bytes reserved for the length of the string. (32bit integer). */
const NUM_BYTES_STRING_LENGTH = 4;

/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @param group Group to which the weights belong (optional).
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
export async function encodeWeights(
    tensors: NamedTensorMap|NamedTensor[], group?: WeightGroup):
    Promise<{data: ArrayBuffer, specs: WeightsManifestEntry[]}> {
  // TODO(adarob, cais): Support quantization.
  const specs: WeightsManifestEntry[] = [];
  const dataPromises: Array<Promise<TypedArray>> = [];

  const names: string[] = Array.isArray(tensors) ?
      tensors.map(tensor => tensor.name) :
      Object.keys(tensors);

  for (let i = 0; i < names.length; ++i) {
    const name = names[i];
    const t = Array.isArray(tensors) ? tensors[i].tensor : tensors[name];
    if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool' &&
        t.dtype !== 'string' && t.dtype !== 'complex64') {
      throw new Error(`Unsupported dtype in weight '${name}': ${t.dtype}`);
    }
    const spec: WeightsManifestEntry = {name, shape: t.shape, dtype: t.dtype};
    if (t.dtype === 'string') {
      const utf8bytes = new Promise<TypedArray>(async resolve => {
        const vals = await t.bytes() as Uint8Array[];
        const totalNumBytes = vals.reduce((p, c) => p + c.length, 0) +
            NUM_BYTES_STRING_LENGTH * vals.length;
        const bytes = new Uint8Array(totalNumBytes);
        let offset = 0;
        for (let i = 0; i < vals.length; i++) {
          const val = vals[i];
          const bytesOfLength =
              new Uint8Array(new Uint32Array([val.length]).buffer);
          bytes.set(bytesOfLength, offset);
          offset += NUM_BYTES_STRING_LENGTH;
          bytes.set(val, offset);
          offset += val.length;
        }
        resolve(bytes);
      });
      dataPromises.push(utf8bytes);
    } else {
      dataPromises.push(t.data());
    }
    if (group != null) {
      spec.group = group;
    }
    specs.push(spec);
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
  let float16Decode: (buffer: Uint16Array) => Float32Array | undefined;
  let offset = 0;
  for (const spec of specs) {
    const name = spec.name;
    const dtype = spec.dtype;
    const shape = spec.shape;
    const size = sizeFromShape(shape);
    let values: TypedArray|string[]|Uint8Array[];

    if ('quantization' in spec) {
      const quantization = spec.quantization;
      if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
        if (!('min' in quantization && 'scale' in quantization)) {
          throw new Error(
              `Weight ${spec.name} with quantization ${quantization.dtype} ` +
              `doesn't have corresponding metadata min and scale.`);
        }
      } else if (quantization.dtype === 'float16') {
        if (dtype !== 'float32') {
          throw new Error(
              `Weight ${spec.name} is quantized with ${quantization.dtype} ` +
              `which only supports weights of type float32 not ${dtype}.`);
        }
      } else {
        throw new Error(
            `Weight ${spec.name} has unknown ` +
            `quantization dtype ${quantization.dtype}. ` +
            `Supported quantization dtypes are: ` +
            `'uint8', 'uint16', and 'float16'.`);
      }
      const quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
      const byteBuffer =
          buffer.slice(offset, offset + size * quantizationSizeFactor);
      const quantizedArray = (quantization.dtype === 'uint8') ?
          new Uint8Array(byteBuffer) :
          new Uint16Array(byteBuffer);
      if (dtype === 'float32') {
        if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
          values = new Float32Array(quantizedArray.length);
          for (let i = 0; i < quantizedArray.length; i++) {
            const v = quantizedArray[i];
            values[i] = v * quantization.scale + quantization.min;
          }
        } else if (quantization.dtype === 'float16') {
          if (float16Decode === undefined) {
            float16Decode = getFloat16Decoder();
          }
          values = float16Decode(quantizedArray as Uint16Array);
        } else {
          throw new Error(
              `Unsupported quantization type ${quantization.dtype} ` +
              `for weight type float32.`);
        }
      } else if (dtype === 'int32') {
        if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
          throw new Error(
              `Unsupported quantization type ${quantization.dtype} ` +
              `for weight type int32.`);
        }
        values = new Int32Array(quantizedArray.length);
        for (let i = 0; i < quantizedArray.length; i++) {
          const v = quantizedArray[i];
          values[i] = Math.round(v * quantization.scale + quantization.min);
        }
      } else {
        throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
      }
      offset += size * quantizationSizeFactor;
    } else if (dtype === 'string') {
      const size = sizeFromShape(spec.shape);
      values = [];
      for (let i = 0; i < size; i++) {
        const byteLength = new Uint32Array(
            buffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
        offset += NUM_BYTES_STRING_LENGTH;
        const bytes = new Uint8Array(buffer.slice(offset, offset + byteLength));
        (values as Uint8Array[]).push(bytes);
        offset += byteLength;
      }
    } else {
      const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
      const byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);

      if (dtype === 'float32') {
        values = new Float32Array(byteBuffer);
      } else if (dtype === 'int32') {
        values = new Int32Array(byteBuffer);
      } else if (dtype === 'bool') {
        values = new Uint8Array(byteBuffer);
      } else if (dtype === 'complex64') {
        values = new Float32Array(byteBuffer);
        const real = new Float32Array(values.length / 2);
        const image = new Float32Array(values.length / 2);
        for (let i = 0; i < real.length; i++) {
          real[i] = values[i * 2];
          image[i] = values[i * 2 + 1];
        }
        const realTensor = tensor(real, shape, 'float32');
        const imageTensor = tensor(image, shape, 'float32');
        out[name] = complex(realTensor, imageTensor);
      } else {
        throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
      }
      offset += size * dtypeFactor;
    }
    if (dtype !== 'complex64') {
      out[name] = tensor(values, shape, dtype);
    }
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
  const buf = new Uint8Array(buffer);
  let s = '';
  for (let i = 0, l = buf.length; i < l; i++) {
    s += String.fromCharCode(buf[i]);
  }
  return btoa(s);
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
  if (buffers.length === 1) {
    return buffers[0];
  }

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

/**
 * Computes mantisa table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 2048 mantissa lookup values.
 */
function computeFloat16MantisaTable(): Uint32Array {
  const convertMantissa = (i: number): number => {
    let m = i << 13;
    let e = 0;

    while ((m & 0x00800000) === 0) {
      e -= 0x00800000;
      m <<= 1;
    }
    m &= ~0x00800000;
    e += 0x38800000;

    return m | e;
  };

  const mantisaTable = new Uint32Array(2048);

  mantisaTable[0] = 0;
  for (let i = 1; i < 1024; i++) {
    mantisaTable[i] = convertMantissa(i);
  }
  for (let i = 1024; i < 2048; i++) {
    mantisaTable[i] = 0x38000000 + ((i - 1024) << 13);
  }

  return mantisaTable;
}

/**
 * Computes exponent table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 64 exponent lookup values.
 */
function computeFloat16ExponentTable(): Uint32Array {
  const exponentTable = new Uint32Array(64);

  exponentTable[0] = 0;
  exponentTable[31] = 0x47800000;
  exponentTable[32] = 0x80000000;
  exponentTable[63] = 0xc7800000;
  for (let i = 1; i < 31; i++) {
    exponentTable[i] = i << 23;
  }
  for (let i = 33; i < 63; i++) {
    exponentTable[i] = 0x80000000 + ((i - 32) << 23);
  }

  return exponentTable;
}

/**
 * Computes offset table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 6d offset values.
 */
function computeFloat16OffsetTable(): Uint32Array {
  const offsetTable = new Uint32Array(64);

  for (let i = 0; i < 64; i++) {
    offsetTable[i] = 1024;
  }
  offsetTable[0] = offsetTable[32] = 0;

  return offsetTable;
}

/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
export function getFloat16Decoder(): (buffer: Uint16Array) => Float32Array {
  // Algorithm is based off of
  // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf

  // Cache lookup tables
  const mantisaTable = computeFloat16MantisaTable();
  const exponentTable = computeFloat16ExponentTable();
  const offsetTable = computeFloat16OffsetTable();

  return (quantizedArray: Uint16Array) => {
    const buffer = new ArrayBuffer(4 * quantizedArray.length);
    const bufferUint32View = new Uint32Array(buffer);
    for (let index = 0; index < quantizedArray.length; index++) {
      const float16Bits = quantizedArray[index];
      const float32Bits =
          mantisaTable[offsetTable[float16Bits >> 10] + (float16Bits & 0x3ff)] +
          exponentTable[float16Bits >> 10];
      bufferUint32View[index] = float32Bits;
    }
    return new Float32Array(buffer);
  };
}
