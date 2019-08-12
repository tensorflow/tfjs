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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {scalar, tensor1d, tensor2d} from '../ops/ops';
import {NamedTensor, NamedTensorMap} from '../tensor_types';
import {expectArraysEqual} from '../test_util';
import {expectArraysClose} from '../test_util';
import {encodeString} from '../util';
import {arrayBufferToBase64String, base64StringToArrayBuffer, basename, concatenateArrayBuffers, concatenateTypedArrays, stringByteLength} from './io_utils';
import {WeightsManifestEntry} from './types';

describe('concatenateTypedArrays', () => {
  it('Single float arrays', () => {
    const x = new Float32Array([1.1, 2.2, 3.3]);
    const buffer = concatenateTypedArrays([x]);
    expect(buffer.byteLength).toEqual(12);
    expect(new Float32Array(buffer, 0, 3)).toEqual(x);
  });

  it('Float arrays', () => {
    const x = new Float32Array([1.1, 2.2, 3.3]);
    const y = new Float32Array([-1.1, -2.2, -3.3]);
    const buffer = concatenateTypedArrays([x, y]);
    expect(buffer.byteLength).toEqual(24);
    expect(new Float32Array(buffer, 0, 3)).toEqual(x);
    expect(new Float32Array(buffer, 12, 3)).toEqual(y);
  });
  it('Single int32 arrays', () => {
    const x = new Int32Array([11, 22, 33]);
    const buffer = concatenateTypedArrays([x]);
    expect(buffer.byteLength).toEqual(12);
    expect(new Int32Array(buffer, 0, 3)).toEqual(x);
  });

  it('Int32 arrays', () => {
    const x = new Int32Array([11, 22, 33]);
    const y = new Int32Array([-11, -22, -33]);
    const buffer = concatenateTypedArrays([x, y]);
    expect(buffer.byteLength).toEqual(24);
    expect(new Int32Array(buffer, 0, 3)).toEqual(x);
    expect(new Int32Array(buffer, 12, 3)).toEqual(y);
  });

  it('Single uint8 arrays', () => {
    const x = new Uint8Array([11, 22, 33]);
    const buffer = concatenateTypedArrays([x]);
    expect(buffer.byteLength).toEqual(3);
    expect(new Uint8Array(buffer, 0, 3)).toEqual(x);
  });

  it('Uint8 arrays', () => {
    const x = new Uint8Array([11, 22, 33]);
    const y = new Uint8Array([111, 122, 133]);
    const buffer = concatenateTypedArrays([x, y]);
    expect(buffer.byteLength).toEqual(6);
    expect(new Uint8Array(buffer, 0, 3)).toEqual(x);
    expect(new Uint8Array(buffer, 3, 3)).toEqual(y);
  });

  it('Mixed Uint8, Int32 and Float32 arrays', () => {
    const x = new Uint8Array([0, 1, 1, 0]);
    const y = new Int32Array([10, 20, 30, 40]);
    const z = new Float32Array([-1.1, -2.2, -3.3, -4.4]);
    const buffer = concatenateTypedArrays([x, y, z]);
    expect(buffer.byteLength).toEqual(1 * 4 + 4 * 4 + 4 * 4);
    expect(new Uint8Array(buffer, 0, 4)).toEqual(x);
    expect(new Int32Array(buffer, 4, 4)).toEqual(y);
    expect(new Float32Array(buffer, 20, 4)).toEqual(z);
  });

  it('Concatenate Float32Arrays from SubArrays', () => {
    const x1 = new Float32Array([1.1, 2.2, 3.3]);
    const x2 = new Float32Array([-1.1, -2.2, -3.3]);
    const xConcatenated = concatenateTypedArrays([x1, x2]);
    const y1 = new Float32Array(xConcatenated, 0, 3);
    const y2 = new Float32Array(xConcatenated, 3 * 4, 3);
    // At this point, the buffer of y1 is longer than than the actual byte
    // length of y1, because of the way y1 is constructed. The same is true for
    // y2.
    expect(y1.buffer.byteLength).toEqual(6 * 4);
    expect(y2.buffer.byteLength).toEqual(6 * 4);

    const yConcatenated = concatenateTypedArrays([y1, y2]);
    expect(yConcatenated.byteLength).toEqual(6 * 4);
    expect(new Float32Array(yConcatenated, 0, 3)).toEqual(x1);
    expect(new Float32Array(yConcatenated, 3 * 4, 3)).toEqual(x2);
  });

  it('Concatenate Int32Array from SubArrays', () => {
    const x1 = new Int32Array([11, 22, 33]);
    const x2 = new Int32Array([-11, -22, -33]);
    const xConcatenated = concatenateTypedArrays([x1, x2]);
    const y1 = new Int32Array(xConcatenated, 0, 3);
    const y2 = new Int32Array(xConcatenated, 3 * 4, 3);
    // At this point, the buffer of y1 is longer than than the actual byte
    // length of y1, because of the way y1 is constructed. The same is true for
    // y2.
    expect(y1.buffer.byteLength).toEqual(6 * 4);
    expect(y2.buffer.byteLength).toEqual(6 * 4);

    const yConcatenated = concatenateTypedArrays([y1, y2]);
    expect(yConcatenated.byteLength).toEqual(6 * 4);
    expect(new Int32Array(yConcatenated, 0, 3)).toEqual(x1);
    expect(new Int32Array(yConcatenated, 3 * 4, 3)).toEqual(x2);
  });

  it('Concatenate Uint8Array from SubArrays', () => {
    const x1 = new Uint8Array([11, 22, 33]);
    const x2 = new Uint8Array([44, 55, 66]);
    const xConcatenated = concatenateTypedArrays([x1, x2]);
    const y1 = new Uint8Array(xConcatenated, 0, 3);
    const y2 = new Uint8Array(xConcatenated, 3, 3);
    // At this point, the buffer of y1 is longer than than the actual byte
    // length of y1, because of the way y1 is constructed. The same is true for
    // y2.
    expect(y1.buffer.byteLength).toEqual(6);
    expect(y2.buffer.byteLength).toEqual(6);

    const yConcatenated = concatenateTypedArrays([y1, y2]);
    expect(yConcatenated.byteLength).toEqual(6);
    expect(new Uint8Array(yConcatenated, 0, 3)).toEqual(x1);
    expect(new Uint8Array(yConcatenated, 3, 3)).toEqual(x2);
  });

  it('Concatenate mixed TypedArrays from SubArrays', () => {
    const x1 = new Uint8Array([11, 22, 33, 44]);
    const x2 = new Int32Array([-44, -55, -66]);
    const x3 = new Float32Array([1.1, 2.2, 3.3]);
    const xConcatenated = concatenateTypedArrays([x1, x2, x3]);
    const y1 = new Uint8Array(xConcatenated, 0, 4);
    const y2 = new Int32Array(xConcatenated, 4, 3);
    const y3 = new Float32Array(xConcatenated, 4 + 3 * 4, 3);
    // At this point, the buffer of y1 is longer than than the actual byte
    // length of y1, because of the way y1 is constructed. The same is true for
    // y2 and y3.
    expect(y1.buffer.byteLength).toEqual(4 + 3 * 4 + 3 * 4);
    expect(y2.buffer.byteLength).toEqual(4 + 3 * 4 + 3 * 4);
    expect(y3.buffer.byteLength).toEqual(4 + 3 * 4 + 3 * 4);

    const yConcatenated = concatenateTypedArrays([y1, y2, y3]);
    expect(yConcatenated.byteLength).toEqual(4 + 3 * 4 + 3 * 4);
    expect(new Uint8Array(yConcatenated, 0, 4)).toEqual(x1);
    expect(new Int32Array(yConcatenated, 4, 3)).toEqual(x2);
    expect(new Float32Array(yConcatenated, 4 + 3 * 4, 3)).toEqual(x3);
  });

  it('null and undefined inputs', () => {
    expect(() => concatenateTypedArrays(null)).toThrow();
    expect(() => concatenateTypedArrays(undefined)).toThrow();
  });

  it('empty input array', () => {
    expect(concatenateTypedArrays([]).byteLength).toEqual(0);
  });

  it('Unsupported dtype', () => {
    const x = new Int16Array([0, 1, 1, 0]);
    // tslint:disable-next-line:no-any
    expect(() => concatenateTypedArrays([x as any]))
        .toThrowError(/Unsupported TypedArray subtype: Int16Array/);
  });
});

describe('encodeWeights', () => {
  it('Float32 tensors as NamedTensorMap', async () => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[10, 20], [30, 40]]),
      x2: scalar(42),
      x3: tensor1d([-1.3, -3.7, 1.3, 3.7]),
    };
    const dataAndSpecs = await tf.io.encodeWeights(tensors);
    const data = dataAndSpecs.data;
    const specs = dataAndSpecs.specs;
    expect(data.byteLength).toEqual(4 * (4 + 1 + 4));
    expect(new Float32Array(data, 0, 4)).toEqual(new Float32Array([
      10, 20, 30, 40
    ]));
    expect(new Float32Array(data, 16, 1)).toEqual(new Float32Array([42]));
    expect(new Float32Array(data, 20, 4)).toEqual(new Float32Array([
      -1.3, -3.7, 1.3, 3.7
    ]));
    expect(specs).toEqual([
      {
        name: 'x1',
        dtype: 'float32',
        shape: [2, 2],
      },
      {
        name: 'x2',
        dtype: 'float32',
        shape: [],
      },
      {
        name: 'x3',
        dtype: 'float32',
        shape: [4],
      }
    ]);
  });

  it('Float32 tensors as NamedTensor array', async () => {
    const tensors: NamedTensor[] = [
      {name: 'x1234', tensor: tensor2d([[10, 20], [30, 40]])}, {
        name: 'a42',
        tensor: scalar(42),
      },
      {name: 'b41', tensor: tensor1d([-1.3, -3.7, 1.3, 3.7])}
    ];
    const dataAndSpecs = await tf.io.encodeWeights(tensors);
    const data = dataAndSpecs.data;
    const specs = dataAndSpecs.specs;
    expect(data.byteLength).toEqual(4 * (4 + 1 + 4));
    expect(new Float32Array(data, 0, 4)).toEqual(new Float32Array([
      10, 20, 30, 40
    ]));
    expect(new Float32Array(data, 16, 1)).toEqual(new Float32Array([42]));
    expect(new Float32Array(data, 20, 4)).toEqual(new Float32Array([
      -1.3, -3.7, 1.3, 3.7
    ]));
    expect(specs).toEqual([
      {
        name: 'x1234',
        dtype: 'float32',
        shape: [2, 2],
      },
      {
        name: 'a42',
        dtype: 'float32',
        shape: [],
      },
      {
        name: 'b41',
        dtype: 'float32',
        shape: [4],
      }
    ]);
  });

  it('Empty NamedTensor array', async () => {
    const tensors: NamedTensor[] = [];
    const dataAndSpecs = await tf.io.encodeWeights(tensors);
    const data = dataAndSpecs.data;
    const specs = dataAndSpecs.specs;
    expect(data.byteLength).toEqual(0);
    expect(specs).toEqual([]);
  });

  it('Int32 tensors', async () => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[10, 20], [30, 40]], [2, 2], 'int32'),
      x2: scalar(42, 'int32'),
      x3: tensor1d([-1, -3, -3, -7], 'int32'),
    };
    const dataAndSpecs = await tf.io.encodeWeights(tensors);
    const data = dataAndSpecs.data;
    const specs = dataAndSpecs.specs;
    expect(data.byteLength).toEqual(4 * (4 + 1 + 4));
    expect(new Int32Array(data, 0, 4)).toEqual(new Int32Array([
      10, 20, 30, 40
    ]));
    expect(new Int32Array(data, 16, 1)).toEqual(new Int32Array([42]));
    expect(new Int32Array(data, 20, 4)).toEqual(new Int32Array([
      -1, -3, -3, -7
    ]));
    expect(specs).toEqual([
      {
        name: 'x1',
        dtype: 'int32',
        shape: [2, 2],
      },
      {
        name: 'x2',
        dtype: 'int32',
        shape: [],
      },
      {
        name: 'x3',
        dtype: 'int32',
        shape: [4],
      }
    ]);
  });

  it('Bool tensors', async () => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[true, false], [false, true]], [2, 2], 'bool'),
      x2: scalar(false, 'bool'),
      x3: tensor1d([false, true, true, false], 'bool'),
    };
    const dataAndSpecs = await tf.io.encodeWeights(tensors);
    const data = dataAndSpecs.data;
    const specs = dataAndSpecs.specs;
    expect(data.byteLength).toEqual(4 + 1 + 4);
    expect(new Uint8Array(data, 0, 4)).toEqual(new Uint8Array([1, 0, 0, 1]));
    expect(new Uint8Array(data, 4, 1)).toEqual(new Uint8Array([0]));
    expect(new Uint8Array(data, 5, 4)).toEqual(new Uint8Array([0, 1, 1, 0]));
    expect(specs).toEqual([
      {
        name: 'x1',
        dtype: 'bool',
        shape: [2, 2],
      },
      {
        name: 'x2',
        dtype: 'bool',
        shape: [],
      },
      {
        name: 'x3',
        dtype: 'bool',
        shape: [4],
      }
    ]);
  });

  it('String tensors', async () => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([['a', 'bc'], ['def', 'g']], [2, 2]),
      x2: scalar(''),                       // Empty string.
      x3: tensor1d(['здраво', 'поздрав']),  // Cyrillic.
      x4: scalar('正常'),                   // East Asian.
      x5: scalar('hello')                   // Single string.
    };
    const dataAndSpecs = await tf.io.encodeWeights(tensors);
    const data = dataAndSpecs.data;
    const specs = dataAndSpecs.specs;
    const x1ByteLength = 7 + 4 * 4;       // 7 ascii chars + 4 ints.
    const x2ByteLength = 4;               // No chars + 1 int.
    const x3ByteLength = 13 * 2 + 2 * 4;  // 13 cyrillic letters + 2 ints.
    const x4ByteLength = 6 + 1 * 4;       // 2 east asian letters + 1 int.
    const x5ByteLength = 5 + 1 * 4;       // 5 ascii chars + 1 int.
    expect(data.byteLength)
        .toEqual(
            x1ByteLength + x2ByteLength + x3ByteLength + x4ByteLength +
            x5ByteLength);
    // x1 'a'.
    expect(new Uint32Array(data, 0, 1)[0]).toBe(1);
    expect(new Uint8Array(data, 4, 1)).toEqual(encodeString('a'));
    // x1 'bc'.
    expect(new Uint32Array(data.slice(5, 9))[0]).toBe(2);
    expect(new Uint8Array(data, 9, 2)).toEqual(encodeString('bc'));
    // x1 'def'.
    expect(new Uint32Array(data.slice(11, 15))[0]).toBe(3);
    expect(new Uint8Array(data, 15, 3)).toEqual(encodeString('def'));
    // x1 'g'.
    expect(new Uint32Array(data.slice(18, 22))[0]).toBe(1);
    expect(new Uint8Array(data, 22, 1)).toEqual(encodeString('g'));

    // x2 is empty string.
    expect(new Uint32Array(data.slice(23, 27))[0]).toBe(0);

    // x3 'здраво'.
    expect(new Uint32Array(data.slice(27, 31))[0]).toBe(12);
    expect(new Uint8Array(data, 31, 12)).toEqual(encodeString('здраво'));

    // x3 'поздрав'.
    expect(new Uint32Array(data.slice(43, 47))[0]).toBe(14);
    expect(new Uint8Array(data, 47, 14)).toEqual(encodeString('поздрав'));

    // x4 '正常'.
    expect(new Uint32Array(data.slice(61, 65))[0]).toBe(6);
    expect(new Uint8Array(data, 65, 6)).toEqual(encodeString('正常'));

    // x5 'hello'.
    expect(new Uint32Array(data.slice(71, 75))[0]).toBe(5);
    expect(new Uint8Array(data, 75, 5)).toEqual(encodeString('hello'));

    expect(specs).toEqual([
      {name: 'x1', dtype: 'string', shape: [2, 2]},
      {name: 'x2', dtype: 'string', shape: []},
      {name: 'x3', dtype: 'string', shape: [2]},
      {name: 'x4', dtype: 'string', shape: []},
      {name: 'x5', dtype: 'string', shape: []}
    ]);
  });

  it('Mixed dtype tensors', async () => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[10, 20], [30, 40]], [2, 2], 'int32'),
      x2: scalar(13.37, 'float32'),
      x3: tensor1d([true, false, false, true], 'bool'),
    };
    const dataAndSpecs = await tf.io.encodeWeights(tensors);
    const data = dataAndSpecs.data;
    const specs = dataAndSpecs.specs;
    expect(data.byteLength).toEqual(4 * 4 + 4 * 1 + 1 * 4);
    expect(new Int32Array(data, 0, 4)).toEqual(new Int32Array([
      10, 20, 30, 40
    ]));
    expect(new Float32Array(data, 16, 1)).toEqual(new Float32Array([13.37]));
    expect(new Uint8Array(data, 20, 4)).toEqual(new Uint8Array([1, 0, 0, 1]));
    expect(specs).toEqual([
      {
        name: 'x1',
        dtype: 'int32',
        shape: [2, 2],
      },
      {
        name: 'x2',
        dtype: 'float32',
        shape: [],
      },
      {
        name: 'x3',
        dtype: 'bool',
        shape: [4],
      }
    ]);
  });
});

describeWithFlags('decodeWeights', {}, () => {
  it('Mixed dtype tensors', async () => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[10, 20], [30, 40]], [2, 2], 'int32'),
      x2: scalar(13.37, 'float32'),
      x3: tensor1d([true, false, false], 'bool'),
      x4: tensor2d([['здраво', 'a'], ['b', 'c']], [2, 2], 'string'),
      x5: tensor1d([''], 'string'),  // Empty string.
      x6: scalar('hello'),           // Single string.
      y1: tensor2d([-10, -20, -30], [3, 1], 'float32'),
    };
    const dataAndSpecs = await tf.io.encodeWeights(tensors);
    const data = dataAndSpecs.data;
    const specs = dataAndSpecs.specs;
    const decoded = tf.io.decodeWeights(data, specs);
    expect(Object.keys(decoded).length).toEqual(7);
    expectArraysEqual(await decoded['x1'].data(), await tensors['x1'].data());
    expectArraysEqual(await decoded['x2'].data(), await tensors['x2'].data());
    expectArraysEqual(await decoded['x3'].data(), await tensors['x3'].data());
    expectArraysEqual(await decoded['x4'].data(), await tensors['x4'].data());
    expectArraysEqual(await decoded['x5'].data(), await tensors['x5'].data());
    expectArraysEqual(await decoded['x6'].data(), await tensors['x6'].data());
    expectArraysEqual(await decoded['y1'].data(), await tensors['y1'].data());
  });

  it('Unsupported dtype raises Error', () => {
    const buffer = new ArrayBuffer(4);
    // tslint:disable-next-line:no-any
    const specs: any = [
      {
        name: 'x',
        dtype: 'int16',
        shape: [],
      },
      {name: 'y', dtype: 'int16', shape: []}
    ];
    expect(() => tf.io.decodeWeights(buffer, specs))
        .toThrowError(/Unsupported dtype in weight \'x\': int16/);
  });

  it('support quantization uint8 weights', async () => {
    const manifestSpecs: WeightsManifestEntry[] = [
      {
        'name': 'weight0',
        'dtype': 'float32',
        'shape': [3],
        'quantization': {'min': -1, 'scale': 0.1, 'dtype': 'uint8'}
      },
      {
        'name': 'weight1',
        'dtype': 'int32',
        'shape': [3],
        'quantization': {'min': -1, 'scale': 0.1, 'dtype': 'uint8'}
      }
    ];
    const data = new Uint8Array([0, 48, 255, 0, 48, 255]);
    const decoded = tf.io.decodeWeights(data.buffer, manifestSpecs);
    const weight0 = decoded['weight0'];
    expectArraysClose(await weight0.data(), [-1, 3.8, 24.5]);
    expect(weight0.shape).toEqual([3]);
    expect(weight0.dtype).toEqual('float32');

    const weight1 = decoded['weight1'];
    expectArraysEqual(await weight1.data(), [-1, 4, 25]);
    expect(weight1.shape).toEqual([3]);
    expect(weight1.dtype).toEqual('int32');
  });

  it('support quantization uint16 weights', async () => {
    const manifestSpecs: WeightsManifestEntry[] = [
      {
        'name': 'weight0',
        'dtype': 'float32',
        'shape': [3],
        'quantization': {'min': -1, 'scale': 0.1, 'dtype': 'uint16'}
      },
      {
        'name': 'weight1',
        'dtype': 'int32',
        'shape': [3],
        'quantization': {'min': -1, 'scale': 0.1, 'dtype': 'uint16'}
      }
    ];
    const data = new Uint16Array([0, 48, 255, 0, 48, 255]);
    const decoded = tf.io.decodeWeights(data.buffer, manifestSpecs);
    const weight0 = decoded['weight0'];
    expectArraysClose(await weight0.data(), [-1, 3.8, 24.5]);
    expect(weight0.shape).toEqual([3]);
    expect(weight0.dtype).toEqual('float32');

    const weight1 = decoded['weight1'];
    expectArraysEqual(await weight1.data(), [-1, 4, 25]);
    expect(weight1.shape).toEqual([3]);
    expect(weight1.dtype).toEqual('int32');
  });
});

describe('stringByteLength', () => {
  it('ASCII only', () => {
    const str = '_Lorem ipsum 1337!';
    expect(stringByteLength(str)).toEqual(str.length);
  });

  it('Mixed narrow and wide chars', () => {
    const str = 'aЖ文1';
    expect(stringByteLength(str.slice(0, 1))).toEqual(1);
    expect(stringByteLength(str.slice(0, 2))).toEqual(3);
    expect(stringByteLength(str.slice(0, 3))).toEqual(6);
    expect(stringByteLength(str.slice(0, 4))).toEqual(7);
  });
});

describe('arrayBufferToBase64String-base64StringToArrayBuffer', () => {
  it('Round trip', () => {
    // Generate some semi-random binary data.
    const x = [];
    for (let k = 0; k < 2; ++k) {
      for (let i = 0; i < 254; ++i) {
        x.push(i + k);
      }
      for (let i = 254; i >= 0; --i) {
        x.push(i + k);
      }
    }
    const buffer = Uint8Array.from(x).buffer;
    const base64Str = arrayBufferToBase64String(buffer);
    const decoded =
        Array.from(new Uint8Array(base64StringToArrayBuffer(base64Str)));
    expect(decoded).toEqual(x);
  });
});

describe('concatenateArrayBuffers', () => {
  it('Concatenate 3 non-empty ArrayBuffers', () => {
    const buffer1 = new Uint8Array([1, 2, 3]);
    const buffer2 = new Uint8Array([11, 22, 33, 44]);
    const buffer3 = new Uint8Array([111, 222, 100]);
    const out = concatenateArrayBuffers(
        [buffer1.buffer, buffer2.buffer, buffer3.buffer]);
    expect(new Uint8Array(out)).toEqual(new Uint8Array([
      1, 2, 3, 11, 22, 33, 44, 111, 222, 100
    ]));
  });

  it('Concatenate non-empty and empty ArrayBuffers', () => {
    const buffer1 = new Uint8Array([1, 2, 3]);
    const buffer2 = new Uint8Array([11, 22, 33, 44]);
    const buffer3 = new Uint8Array([]);
    const buffer4 = new Uint8Array([150, 100, 50]);
    const out = concatenateArrayBuffers(
        [buffer1.buffer, buffer2.buffer, buffer3.buffer, buffer4.buffer]);
    expect(new Uint8Array(out)).toEqual(new Uint8Array([
      1, 2, 3, 11, 22, 33, 44, 150, 100, 50
    ]));
  });

  it('A single ArrayBuffer', () => {
    const buffer1 = new Uint8Array([1, 3, 3, 7]);
    const out = concatenateArrayBuffers([buffer1.buffer]);
    expect(new Uint8Array(out)).toEqual(buffer1);
  });

  it('Zero ArrayBuffers', () => {
    expect(new Uint8Array(concatenateArrayBuffers([])))
        .toEqual(new Uint8Array([]));
  });
});

describe('basename', () => {
  it('Paths without slashes', () => {
    expect(basename('foo.txt')).toEqual('foo.txt');
    expect(basename('bar')).toEqual('bar');
  });

  it('Paths with slashes', () => {
    expect(basename('qux/foo.txt')).toEqual('foo.txt');
    expect(basename('qux/My Model.json')).toEqual('My Model.json');
    expect(basename('foo/bar/baz')).toEqual('baz');
    expect(basename('/foo/bar/baz')).toEqual('baz');
    expect(basename('foo/bar/baz/')).toEqual('baz');
    expect(basename('foo/bar/baz//')).toEqual('baz');
  });
});
