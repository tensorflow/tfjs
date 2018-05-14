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
 * Unit tests for LayerVariables.
 */

// tslint:disable:max-line-length
import {randomUniform, scalar, tensor1d, zeros} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {nameScope} from './backend/tfjs_backend';
import * as tfl from './index';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from './utils/test_utils';
import * as V from './variables';
// tslint:enable:max-line-length


/**
 * Unit tests for Variable.
 */
describeMathCPU('Variable', () => {
  it('Variable constructor: no explicit name', () => {
    const v1 = new V.LayerVariable(zeros([2]));
    expect(v1.name.indexOf('Variable')).toEqual(0);
    expect(v1.dtype).toEqual('float32');
    expect(v1.shape).toEqual([2]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));

    const v2 = new V.LayerVariable(zeros([2, 2]));
    expect(v2.name.indexOf('Variable')).toEqual(0);
    expect(v2.dtype).toEqual('float32');
    expect(v2.shape).toEqual([2, 2]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));

    expect(v2.name === v1.name).toBe(false);
  });

  it('Variable constructor: explicit name', () => {
    const v1 = new V.LayerVariable(zeros([]), undefined, 'foo');
    expect(v1.name.indexOf('foo')).toEqual(0);
    expect(v1.dtype).toEqual('float32');
    expect(v1.shape).toEqual([]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0]));

    const v2 = new V.LayerVariable(zeros([2, 2, 1]));
    expect(v1.name.indexOf('foo')).toEqual(0);
    expect(v2.dtype).toEqual('float32');
    expect(v2.shape).toEqual([2, 2, 1]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    expect(v2.name.length).toBeGreaterThan(0);

    expect(v2.name === v1.name).toBe(false);
  });

  it('Variable constructor: explicit name with name scope', () => {
    let v1: V.LayerVariable;
    nameScope('barScope', () => {
      nameScope('bazScope', () => {
        v1 = new V.LayerVariable(scalar(0), undefined, 'foo');
      });
    });
    expect(v1.name.indexOf('barScope/bazScope/foo')).toEqual(0);
    expect(v1.dtype).toEqual('float32');
    expect(v1.shape).toEqual([]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
  });

  it('Variable trainable property', () => {
    const v1 = new V.LayerVariable(zeros([]), null, 'foo', false);
    expect(v1.trainable).toEqual(false);
  });

  it('Variable works if name is null or undefined', () => {
    expect((new V.LayerVariable(zeros([]), null)).name.indexOf('Variable'))
        .toEqual(0);
    expect((new V.LayerVariable(zeros([]), undefined)).name.indexOf('Variable'))
        .toEqual(0);
  });

  it('int32 dtype', () => {
    expect(new V.LayerVariable(zeros([]), 'int32').dtype).toEqual('int32');
  });

  it('bool dtype', () => {
    expect(new V.LayerVariable(zeros([]), 'bool').dtype).toEqual('bool');
  });

  it('Read value', () => {
    const v1 = new V.LayerVariable(scalar(10), null, 'foo');
    expect(v1.read().dataSync()).toEqual(new Float32Array([10]));
  });

  it('Update value: Compatible shape', () => {
    const v = new V.LayerVariable(tensor1d([10, -10]), null, 'bar');
    expect(v.name.indexOf('bar')).toEqual(0);
    expect(v.shape).toEqual([2]);
    expect(v.read().dataSync()).toEqual(new Float32Array([10, -10]));

    v.write(tensor1d([10, 50]));
    expect(v.name.indexOf('bar')).toEqual(0);
    expect(v.shape).toEqual([2]);
    expect(v.read().dataSync()).toEqual(new Float32Array([10, 50]));
  });

  it('Update value: w/ constraint', () => {
    const v = new V.LayerVariable(
        tensor1d([10, -10]), null, 'bar', true, tfl.constraints.nonNeg());

    v.write(tensor1d([-10, 10]));
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 10]));
  });


  it('Update value: Incompatible shape', () => {
    const v = new V.LayerVariable(zeros([2, 2]), null, 'qux');
    expect(() => {
      v.write(zeros([4]));
    }).toThrowError();
  });

  it('Generates unique ID', () => {
    const v1 = new V.LayerVariable(scalar(1), null, 'foo');
    const v2 = new V.LayerVariable(scalar(1), null, 'foo');
    expect(v1.id).not.toEqual(v2.id);
  });

  it('Generates unique IDs for Tensors and Variables', () => {
    const v1 = scalar(1);
    const v2 = new V.LayerVariable(scalar(1), null, 'foo');
    expect(v1.id).not.toEqual(v2.id);
  });
});

describeMathCPUAndGPU('Create Variable', () => {
  it('From Tensor, no explicit name', () => {
    const v = V.variable(zeros([2, 2]));
    expect(v.name.indexOf('Variable')).toEqual(0);
    expect(v.shape).toEqual([2, 2]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('From Tensor, no explicit name', () => {
    const v = V.variable(zeros([3]));
    expect(v.name.indexOf('Variable')).toEqual(0);
    expect(v.shape).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('From Tensor, explicit name', () => {
    const v = V.variable(zeros([3]), undefined, 'Var1');
    expect(v.name.indexOf('Var1')).toEqual(0);
    expect(v.shape).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });
});

describeMathCPUAndGPU('ZerosVariable', () => {
  it('Scalar', () => {
    const s = V.zerosVariable([], 'float32', 'Scalar');
    expect(s.name.indexOf('Scalar')).toEqual(0);
    expect(K.shape(s.read())).toEqual([]);
    expect(s.read().dataSync()).toEqual(new Float32Array([0]));
  });

  it('Vector', () => {
    const v = V.zerosVariable([3], 'float32', 'Vector');
    expect(v.name.indexOf('Vector')).toEqual(0);
    expect(K.shape(v.read())).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('Matrix', () => {
    const m = V.zerosVariable([2, 2], 'float32', 'Matrix');
    expect(m.name.indexOf('Matrix')).toEqual(0);
    expect(K.shape(m.read())).toEqual([2, 2]);
    expect(m.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('3D', () => {
    const t = V.zerosVariable([2, 2, 2], 'float32', 'Tertiary');
    expect(t.name.indexOf('Tertiary')).toEqual(0);
    expect(K.shape(t.read())).toEqual([2, 2, 2]);
    expect(t.read().dataSync()).toEqual(new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 0
    ]));
  });

  it('4D', () => {
    const q = V.zerosVariable([1, 2, 1, 3], 'float32', 'Quaternary');
    expect(q.name.indexOf('Quaternary')).toEqual(0);
    expect(K.shape(q.read())).toEqual([1, 2, 1, 3]);
    expect(q.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
  });
});

describeMathCPUAndGPU('OnesVariable', () => {
  it('Scalar', () => {
    const s = V.onesVariable([], 'float32', 'Scalar');
    expect(s.name.indexOf('Scalar')).toEqual(0);
    expect(K.shape(s.read())).toEqual([]);
    expect(s.read().dataSync()).toEqual(new Float32Array([1]));
  });
  it('Vector', () => {
    const v = V.onesVariable([3], 'float32', 'Vector');
    expect(v.name.indexOf('Vector')).toEqual(0);
    expect(K.shape(v.read())).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([1, 1, 1]));
  });
  it('Matrix', () => {
    const m = V.onesVariable([2, 2], 'float32', 'Matrix');
    expect(m.name.indexOf('Matrix')).toEqual(0);
    expect(K.shape(m.read())).toEqual([2, 2]);
    expect(m.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
  });
  it('3D', () => {
    const t = V.onesVariable([2, 2, 2], 'float32', 'Tertiary');
    expect(t.name.indexOf('Tertiary')).toEqual(0);
    expect(K.shape(t.read())).toEqual([2, 2, 2]);
    expect(t.read().dataSync()).toEqual(new Float32Array([
      1, 1, 1, 1, 1, 1, 1, 1
    ]));
  });
  it('4D', () => {
    const q = V.onesVariable([1, 2, 1, 3], 'float32', 'Quaternary');
    expect(q.name.indexOf('Quaternary')).toEqual(0);
    expect(K.shape(q.read())).toEqual([1, 2, 1, 3]);
    expect(q.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
  });
});

describeMathCPUAndGPU('ZerosLike', () => {
  it('Scalar', () => {
    const s = V.zerosLike(randomUniform([], -10, 10));
    expect(K.shape(s.read())).toEqual([]);
    expect(s.read().dataSync()).toEqual(new Float32Array([0]));
  });

  it('Vector', () => {
    const v = V.zerosLike(randomUniform([3], -10, 10));
    expect(K.shape(v.read())).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('Matrix', () => {
    const m = V.zerosLike(randomUniform([2, 2], -10, 10));
    expect(K.shape(m.read())).toEqual([2, 2]);
    expect(m.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('3D', () => {
    const t = V.zerosLike(randomUniform([2, 2, 2], -10, 10));
    expect(K.shape(t.read())).toEqual([2, 2, 2]);
    expect(t.read().dataSync()).toEqual(new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 0
    ]));
  });

  it('4D', () => {
    const q = V.zerosLike(randomUniform([1, 2, 1, 3], -10, 10));
    expect(K.shape(q.read())).toEqual([1, 2, 1, 3]);
    expect(q.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
  });
});

describeMathCPUAndGPU('OnesLike', () => {
  it('Scalar', () => {
    const s = V.onesLike(randomUniform([], -10, 10));
    expect(K.shape(s.read())).toEqual([]);
    expect(s.read().dataSync()).toEqual(new Float32Array([1]));
  });

  it('Vector', () => {
    const v = V.onesLike(randomUniform([3], -10, 10));
    expect(K.shape(v.read())).toEqual([3]);
    expect(v.read().dataSync()).toEqual(new Float32Array([1, 1, 1]));
  });

  it('Matrix', () => {
    const m = V.onesLike(randomUniform([2, 2], -10, 10));
    expect(K.shape(m.read())).toEqual([2, 2]);
    expect(m.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
  });

  it('3D', () => {
    const t = V.onesLike(randomUniform([2, 2, 2], -10, 10));
    expect(K.shape(t.read())).toEqual([2, 2, 2]);
    expect(t.read().dataSync()).toEqual(new Float32Array([
      1, 1, 1, 1, 1, 1, 1, 1
    ]));
  });

  it('4D', () => {
    const q = V.onesLike(randomUniform([1, 2, 1, 3], -10, 10));
    expect(K.shape(q.read())).toEqual([1, 2, 1, 3]);
    expect(q.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
  });
});

describeMathCPUAndGPU('eye (I-matrix builder)', () => {
  it('Variable Zero sized 2D matrix', () => {
    expect(() => V.eyeVariable(0)).toThrowError(/Shapes can not be <= 0./);
  });
  it('Variable 1 sized 2D matrix', () => {
    const I = V.eyeVariable(1);
    expect(I.shape).toEqual([1, 1]);
    expect(I.read().dataSync()).toEqual(new Float32Array([1]));
  });
  it('Variable 2 sized 2D matrix', () => {
    const I = V.eyeVariable(2);
    expect(I.shape).toEqual([2, 2]);
    expect(I.read().dataSync()).toEqual(new Float32Array([1, 0, 0, 1]));
  });
});

describeMathCPUAndGPU('Variable update', () => {
  it('Update', () => {
    const v = new V.LayerVariable(scalar(10.0));
    V.update(v, scalar(20.0));
    expectTensorsClose(v.read(), scalar(20.0));
  });
  it('Update: Incompatible shape', () => {
    const v = new V.LayerVariable(tensor1d([10.0, 20.0]));
    const x = tensor1d([10.0, 20.0, 30.0]);
    expect(() => V.update(v, x)).toThrowError();
  });
  it('UpdateAdd', () => {
    const v = new V.LayerVariable(scalar(10.0));
    V.updateAdd(v, scalar(20.0));
    expectTensorsClose(v.read(), scalar(30.0));
  });
  it('UpdateAdd: Incompatible shape', () => {
    const v = new V.LayerVariable(tensor1d([10.0, 20.0]));
    const x = tensor1d([0.0, 10.0, 20.0]);
    expect(() => V.updateAdd(v, x)).toThrowError();
  });
  it('UpdateSub', () => {
    const v = new V.LayerVariable(scalar(10.0));
    V.updateSub(v, scalar(20.0));
    const vNew = v.read();
    expectTensorsClose(vNew, scalar(-10.0));
  });
  it('UpdateSub: Incompatible shape', () => {
    const v = new V.LayerVariable(tensor1d([10.0, 20.0]));
    const x = tensor1d([0.0, 10.0, 20.0]);
    expect(() => V.updateSub(v, x)).toThrowError();
  });
});

describeMathCPUAndGPU('batchGetValue', () => {
  it('Legnth-3 Array, Mixed Tensor and Variable', () => {
    const v1 = V.variable(zeros([]));
    const v2 = V.variable(zeros([2]));
    const v3 = V.variable(zeros([2, 2]));
    const values = V.batchGetValue([v1, v2, v3]);
    expect(values.length).toEqual(3);
    expect(values[0].shape).toEqual([]);
    expect(values[0].dataSync()).toEqual(new Float32Array([0]));
    expect(values[1].shape).toEqual([2]);
    expect(values[1].dataSync()).toEqual(new Float32Array([0, 0]));
    expect(values[2].shape).toEqual([2, 2]);
    expect(values[2].dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });
});

describeMathCPUAndGPU('batchSetValue', () => {
  it('Update using Tensor values', () => {
    const v1 = V.randomUniformVariable([2], 0, 1);
    const v2 = V.randomUniformVariable([2, 2], 0, 1);
    V.batchSetValue([[v1, zeros([2])], [v2, zeros([2, 2])]]);
    expect(v1.shape).toEqual([2]);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));
    expect(v2.shape).toEqual([2, 2]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('Update using Tensor values', () => {
    const v1 = V.randomUniformVariable([], 0, 1);
    const v2 = V.randomUniformVariable([2, 2, 1], 0, 1);
    V.batchSetValue([[v1, zeros([])], [v2, zeros([2, 2, 1])]]);
    expect(v1.shape).toEqual([]);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
    expect(v2.shape).toEqual([2, 2, 1]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('Update empty Array', () => {
    V.batchSetValue([]);
  });
});
