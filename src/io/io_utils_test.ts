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
import {expectArraysEqual} from '../test_util';
import {NamedTensorMap} from '../types';

import {concatenateTypedArrays} from './io_utils';

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
