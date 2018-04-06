/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for -specific types.
 */

// tslint:disable:max-line-length
import {scalar, tensor1d, zeros} from '@tensorflow/tfjs-core';

import {nameScope} from './backend/tfjs_backend';
import {NonNeg} from './constraints';
import {DType} from './types';
import {ConcreteTensor, LayerVariable, SymbolicTensor} from './types';
import {describeMathCPU} from './utils/test_utils';

// tslint:enable:max-line-length

const CT = ConcreteTensor;

/**
 * Unit tests for SymbolicTensor.
 */
describe('SymbolicTensor Test', () => {
  it('Correct dtype and shape properties', () => {
    const st1 = new SymbolicTensor(DType.float32, [4, 6], null, [], {});
    expect(st1.dtype).toEqual(DType.float32);
    expect(st1.shape).toEqual([4, 6]);
  });

  it('Correct names and ids', () => {
    const st1 = new SymbolicTensor(
        DType.float32, [2, 2], null, [], {}, 'TestSymbolicTensor');
    const st2 = new SymbolicTensor(
        DType.float32, [2, 2], null, [], {}, 'TestSymbolicTensor');
    expect(st1.name.indexOf('TestSymbolicTensor')).toEqual(0);
    expect(st2.name.indexOf('TestSymbolicTensor')).toEqual(0);
    // Explicit names of symbolic tensors should be unique.
    expect(st1 === st2).toBe(false);

    expect(st1.id).toBeGreaterThanOrEqual(0);
    expect(st2.id).toBeGreaterThanOrEqual(0);
    expect(st1.id === st2.id).toBe(false);
  });

  it('Invalid tensor name leads to error', () => {
    expect(() => new SymbolicTensor(DType.float32, [2, 2], null, [], {}, '!'))
        .toThrowError();
  });
});


/**
 * Unit tests for ConcreteTensor.
 */
describeMathCPU('ConcreteTensor Test', () => {
  it('Constructor: no explicit name', () => {
    const v1 = new CT(zeros([2]));
    expect(v1.name).toBeFalsy();
    expect(v1.dtype).toEqual(DType.float32);
    expect(v1.shape).toEqual([2]);
    expect(v1.value().dataSync()).toEqual(new Float32Array([0, 0]));

    const v2 = new CT(zeros([2, 2]));
    expect(v2.name).toBeFalsy();
    expect(v2.dtype).toEqual(DType.float32);
    expect(v2.shape).toEqual([2, 2]);
    expect(v2.value().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('Constructor: explicit name', () => {
    const v1 = new CT(zeros([]), 'foo');
    expect(v1.name.indexOf('foo')).toEqual(0);
    expect(v1.dtype).toEqual(DType.float32);
    expect(v1.shape).toEqual([]);
    expect(v1.value().dataSync()).toEqual(new Float32Array([0]));

    const v2 = new CT(zeros([2, 2, 1]));
    expect(v1.name.indexOf('foo')).toEqual(0);
    expect(v2.dtype).toEqual(DType.float32);
    expect(v2.shape).toEqual([2, 2, 1]);
    expect(v2.value().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));

    expect(v2.name === v1.name).toBe(false);
  });

  it('Read value', () => {
    const v1 = new CT(scalar(10), 'foo');
    expect(v1.value().dataSync()).toEqual(new Float32Array([10]));
  });

  it('Generates unique ID', () => {
    const v1 = new CT(scalar(1), 'foo');
    const v2 = new CT(scalar(1), 'foo');
    expect(v1.id).not.toEqual(v2.id);
  });
});


/**
 * Unit tests for Variable.
 */
describeMathCPU('Variable', () => {
  it('Variable constructor: no explicit name', () => {
    const v1 = new LayerVariable(zeros([2]));
    expect(v1.name.indexOf('Variable')).toEqual(0);
    expect(v1.dtype).toEqual(DType.float32);
    expect(v1.shape).toEqual([2]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));

    const v2 = new LayerVariable(zeros([2, 2]));
    expect(v2.name.indexOf('Variable')).toEqual(0);
    expect(v2.dtype).toEqual(DType.float32);
    expect(v2.shape).toEqual([2, 2]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));

    expect(v2.name === v1.name).toBe(false);
  });

  it('Variable constructor: explicit name', () => {
    const v1 = new LayerVariable(zeros([]), undefined, 'foo');
    expect(v1.name.indexOf('foo')).toEqual(0);
    expect(v1.dtype).toEqual(DType.float32);
    expect(v1.shape).toEqual([]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0]));

    const v2 = new LayerVariable(zeros([2, 2, 1]));
    expect(v1.name.indexOf('foo')).toEqual(0);
    expect(v2.dtype).toEqual(DType.float32);
    expect(v2.shape).toEqual([2, 2, 1]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    expect(v2.name.length).toBeGreaterThan(0);

    expect(v2.name === v1.name).toBe(false);
  });

  it('Variable constructor: explicit name with name scope', () => {
    let v1: LayerVariable;
    nameScope('barScope', () => {
      nameScope('bazScope', () => {
        v1 = new LayerVariable(scalar(0), undefined, 'foo');
      });
    });
    expect(v1.name.indexOf('barScope/bazScope/foo')).toEqual(0);
    expect(v1.dtype).toEqual(DType.float32);
    expect(v1.shape).toEqual([]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
  });

  it('Variable trainable property', () => {
    const v1 = new LayerVariable(zeros([]), null, 'foo', false);
    expect(v1.trainable).toEqual(false);
  });

  it('Variable works if name is null or undefined', () => {
    expect((new LayerVariable(zeros([]), null)).name.indexOf('Variable'))
        .toEqual(0);
    expect((new LayerVariable(zeros([]), undefined)).name.indexOf('Variable'))
        .toEqual(0);
  });

  it('int32 dtype', () => {
    expect(new LayerVariable(zeros([]), DType.int32).dtype)
        .toEqual(DType.int32);
  });

  it('bool dtype', () => {
    expect(new LayerVariable(zeros([]), DType.bool).dtype).toEqual(DType.bool);
  });

  it('Read value', () => {
    const v1 = new LayerVariable(scalar(10), null, 'foo');
    expect(v1.read().dataSync()).toEqual(new Float32Array([10]));
  });

  it('Update value: Compatible shape', () => {
    const v = new LayerVariable(tensor1d([10, -10]), null, 'bar');
    expect(v.name.indexOf('bar')).toEqual(0);
    expect(v.shape).toEqual([2]);
    expect(v.read().dataSync()).toEqual(new Float32Array([10, -10]));

    v.write(tensor1d([10, 50]));
    expect(v.name.indexOf('bar')).toEqual(0);
    expect(v.shape).toEqual([2]);
    expect(v.read().dataSync()).toEqual(new Float32Array([10, 50]));
  });

  it('Update value: w/ constraint', () => {
    const v =
        new LayerVariable(tensor1d([10, -10]), null, 'bar', true, new NonNeg());

    v.write(tensor1d([-10, 10]));
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 10]));
  });


  it('Update value: Incompatible shape', () => {
    const v = new LayerVariable(zeros([2, 2]), null, 'qux');
    expect(() => {
      v.write(zeros([4]));
    }).toThrowError();
  });

  it('Generates unique ID', () => {
    const v1 = new LayerVariable(scalar(1), null, 'foo');
    const v2 = new LayerVariable(scalar(1), null, 'foo');
    expect(v1.id).not.toEqual(v2.id);
  });

  it('Generates unique IDs for Tensors and Variables', () => {
    const v1 = new CT(scalar(1), 'foo');
    const v2 = new LayerVariable(scalar(1), null, 'foo');
    expect(v1.id).not.toEqual(v2.id);
  });
});
