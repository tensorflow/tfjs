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
import {scalar, tensor1d, tensor2d} from '../ops/ops';
import {NamedTensorMap} from '../tensor_types';
import {expectArraysEqual} from '../test_util';
import {arrayBufferToBase64String, base64StringToArrayBuffer, basename, concatenateArrayBuffers, concatenateTypedArrays, stringByteLength} from './io_utils';

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
  it('Float32 tensors', async done => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[10, 20], [30, 40]]),
      x2: scalar(42),
      x3: tensor1d([-1.3, -3.7, 1.3, 3.7]),
    };
    tf.io.encodeWeights(tensors)
        .then(dataAndSpecs => {
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
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Int32 tensors', async done => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[10, 20], [30, 40]], [2, 2], 'int32'),
      x2: scalar(42, 'int32'),
      x3: tensor1d([-1, -3, -3, -7], 'int32'),
    };
    tf.io.encodeWeights(tensors)
        .then(dataAndSpecs => {
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
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Bool tensors', async done => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[true, false], [false, true]], [2, 2], 'bool'),
      x2: scalar(false, 'bool'),
      x3: tensor1d([false, true, true, false], 'bool'),
    };
    tf.io.encodeWeights(tensors)
        .then(dataAndSpecs => {
          const data = dataAndSpecs.data;
          const specs = dataAndSpecs.specs;
          expect(data.byteLength).toEqual(4 + 1 + 4);
          expect(new Uint8Array(data, 0, 4)).toEqual(new Uint8Array([
            1, 0, 0, 1
          ]));
          expect(new Uint8Array(data, 4, 1)).toEqual(new Uint8Array([0]));
          expect(new Uint8Array(data, 5, 4)).toEqual(new Uint8Array([
            0, 1, 1, 0
          ]));
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
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Mixed dtype tensors', async done => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[10, 20], [30, 40]], [2, 2], 'int32'),
      x2: scalar(13.37, 'float32'),
      x3: tensor1d([true, false, false, true], 'bool'),
    };
    tf.io.encodeWeights(tensors)
        .then(dataAndSpecs => {
          const data = dataAndSpecs.data;
          const specs = dataAndSpecs.specs;
          expect(data.byteLength).toEqual(4 * 4 + 4 * 1 + 1 * 4);
          expect(new Int32Array(data, 0, 4)).toEqual(new Int32Array([
            10, 20, 30, 40
          ]));
          expect(new Float32Array(data, 16, 1))
              .toEqual(new Float32Array([13.37]));
          expect(new Uint8Array(data, 20, 4)).toEqual(new Uint8Array([
            1, 0, 0, 1
          ]));
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
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });
});

describe('decodeWeights', () => {
  it('Mixed dtype tensors', async done => {
    const tensors: NamedTensorMap = {
      x1: tensor2d([[10, 20], [30, 40]], [2, 2], 'int32'),
      x2: scalar(13.37, 'float32'),
      x3: tensor1d([true, false, false, true], 'bool'),
      y1: tensor2d([-10, -20, -30], [3, 1], 'float32'),
    };
    tf.io.encodeWeights(tensors)
        .then(dataAndSpecs => {
          const data = dataAndSpecs.data;
          const specs = dataAndSpecs.specs;
          expect(data.byteLength).toEqual(4 * 4 + 4 * 1 + 1 * 4 + 4 * 3);
          const decoded = tf.io.decodeWeights(data, specs);
          expect(Object.keys(decoded).length).toEqual(4);
          expectArraysEqual(decoded['x1'], tensors['x1']);
          expectArraysEqual(decoded['x2'], tensors['x2']);
          expectArraysEqual(decoded['x3'], tensors['x3']);
          expectArraysEqual(decoded['y1'], tensors['y1']);
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
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
