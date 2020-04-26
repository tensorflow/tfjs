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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

function generateCaseInputs(totalSizeTensor: number, totalSizeFilter: number) {
  const inp = new Array(totalSizeTensor);
  const filt = new Array(totalSizeFilter);

  for (let i = 0; i < totalSizeTensor; i++) {
    inp[i] = i * 0.001 - totalSizeTensor * 0.001 / 2;
  }
  for (let i = 0; i < totalSizeFilter; i++) {
    const sign = i % 2 === 0 ? -1 : 1;
    filt[i] = i * 0.001 * sign;
  }

  return {input: inp, filter: filt};
}

describeWithFlags('fused matmul', ALL_ENVS, () => {
  it('fused A x B', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.fused.matMul({a, b});

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [0, 8, -3, 20]);
  });

  it('fused A x B with relu', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const transposeA = false;
    const transposeB = false;

    const c = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: null, activation: 'relu'});

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [0, 8, 0, 20]);
  });

  it('fused A x B with elu', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const transposeA = false;
    const transposeB = false;

    const c = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: null, activation: 'elu'});

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [0, 8, -0.9502, 20]);
  });

  it('fused A x B with relu6', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const transposeA = false;
    const transposeB = false;

    const c = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: null, activation: 'relu6'});

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [0, 6, 0, 6]);
  });

  it('fused A x B with prelu', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const alpha = tf.tensor2d([0.5, 0.5], [1, 2]);
    const transposeA = false;
    const transposeB = false;

    const c = tf.fused.matMul({
      a,
      b,
      transposeA,
      transposeB,
      bias: null,
      activation: 'prelu',
      preluActivationWeights: alpha
    });

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [0, 8, -1.5, 20]);
  });

  it('fused A x B with relu transpose', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [2, 3]);
    const transposeA = false;
    const transposeB = true;

    const c = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: null, activation: 'relu'});

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [0, 9, 0, 24]);
  });

  it('fused A x B with 2d bias and relu', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const c = tf.tensor2d([1, 1, 1, 1], [2, 2]);
    const transposeA = false;
    const transposeB = false;

    const d = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: c, activation: 'relu'});

    expect(d.shape).toEqual([2, 2]);
    expectArraysClose(await d.data(), [1, 9, 0, 21]);
  });

  it('fused A x B with relu and broadcasted bias', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const c = tf.tensor1d([1, 1]);
    const act: tf.fused.Activation = 'relu';
    const transposeA = false;
    const transposeB = false;

    const d = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: c, activation: act});

    expect(d.shape).toEqual([2, 2]);
    expectArraysClose(await d.data(), [1, 9, 0, 21]);
  });

  it('fused A x B with elu and broadcasted bias', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const c = tf.tensor1d([1, 1]);
    const act: tf.fused.Activation = 'elu';
    const transposeA = false;
    const transposeB = false;

    const d = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: c, activation: act});

    expect(d.shape).toEqual([2, 2]);
    expectArraysClose(await d.data(), [1, 9, -0.8647, 21]);
  });

  it('fused A x B with relu and broadcasted bias different rank', async () => {
    const a = tf.tensor3d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [2, 2, 3]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1, 0, 1, -3, 2, 2, 1], [2, 3, 2]);
    const c = tf.tensor2d([1, 2], [1, 2]);
    const act: tf.fused.Activation = 'relu';
    const transposeA = false;
    const transposeB = false;

    const d = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: c, activation: act});

    expect(d.shape).toEqual([2, 2, 2]);
    expectArraysClose(await d.data(), [2, 6, 0, 18, 0, 30, 0, 42]);
  });

  it('fused A x B with 2d bias only', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const c = tf.tensor2d([1, 1, 1, 1], [2, 2]);
    const transposeA = false;
    const transposeB = false;

    const d = tf.fused.matMul(
        {a, b, transposeA, transposeB, bias: c, activation: 'linear'});

    expect(d.shape).toEqual([2, 2]);
    expectArraysClose(await d.data(), [1, 9, -2, 21]);
  });

  it('fused A x B with relu gradient', async () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [2, 3]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);
    const transposeA = false;
    const transposeB = false;

    const grads = tf.grads((a, b) => {
      const prod = tf.matMul(a, b, transposeA, transposeB);
      return tf.relu(prod);
    });

    const fusedGrads = tf.grads((a, b) => {
      return tf.fused.matMul(
          {a, b, transposeA, transposeB, bias: null, activation: 'relu'});
    });

    const [da, db] = grads([a, b], dy);
    const [fusedDa, fusedDb] = fusedGrads([a, b], dy);
    expectArraysClose(await da.array(), await fusedDa.array());
    expectArraysClose(await db.data(), await fusedDb.array());
  });

  it('gradient with clones A x B with relu', () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [2, 3]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);
    const transposeA = false;
    const transposeB = false;

    const fusedGrads = tf.grads((a, b) => {
      return tf.fused
          .matMul({
            a: a.clone(),
            b: b.clone(),
            transposeA,
            transposeB,
            bias: null,
            activation: 'relu'
          })
          .clone();
    });

    const [fusedDa, fusedDb] = fusedGrads([a, b], dy);
    expect(fusedDa.shape).toEqual(a.shape);
    expect(fusedDb.shape).toEqual(b.shape);
  });

  it('fused A x B with relu bias gradient', async () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [2, 3]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const c = tf.tensor2d([1, 1, 1, 1], [2, 2]);
    const transposeA = false;
    const transposeB = false;

    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const grads = tf.grads((a, b, c) => {
      const prod = tf.matMul(a, b, transposeA, transposeB);
      const sum = tf.add(prod, c);
      return tf.relu(sum);
    });

    const fusedGrads = tf.grads((a, b, c) => {
      return tf.fused.matMul(
          {a, b, transposeA, transposeB, bias: c, activation: 'relu'});
    });

    const [da, db, dc] = grads([a, b, c], dy);
    const [fusedDa, fusedDb, fusedDc] = fusedGrads([a, b, c], dy);

    expectArraysClose(await da.array(), await fusedDa.array());
    expectArraysClose(await db.array(), await fusedDb.array());
    expectArraysClose(await dc.array(), await fusedDc.array());
  });

  it('fused A x B with relu bias gradient transpose', async () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [3, 2]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const c = tf.tensor2d([1, 1, 1, 1], [2, 2]);
    const transposeA = true;
    const transposeB = false;

    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const grads = tf.grads((a, b, c) => {
      const prod = tf.matMul(a, b, transposeA, transposeB);
      const sum = tf.add(prod, c);
      return tf.relu(sum);
    });

    const fusedGrads = tf.grads((a, b, c) => {
      return tf.fused.matMul(
          {a, b, transposeA, transposeB, bias: c, activation: 'relu'});
    });

    const [da, db, dc] = grads([a, b, c], dy);
    const [fusedDa, fusedDb, fusedDc] = fusedGrads([a, b, c], dy);

    expectArraysClose(await da.array(), await fusedDa.array());
    expectArraysClose(await db.array(), await fusedDb.array());
    expectArraysClose(await dc.array(), await fusedDc.array());
  });

  it('fused A x B with relu and broadcasted bias gradient', async () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [2, 3]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const c = tf.tensor2d([[1]]);
    const transposeA = false;
    const transposeB = false;

    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const grads = tf.grads((a, b, c) => {
      const prod = tf.matMul(a, b, transposeA, transposeB);
      const sum = tf.add(prod, c);
      return tf.relu(sum);
    });

    const fusedGrads = tf.grads((a, b, c) => {
      return tf.fused.matMul(
          {a, b, transposeA, transposeB, bias: c, activation: 'relu'});
    });

    const [da, db, dc] = grads([a, b, c], dy);
    const [fusedDa, fusedDb, fusedDc] = fusedGrads([a, b, c], dy);

    expectArraysClose(await da.array(), await fusedDa.array());
    expectArraysClose(await db.array(), await fusedDb.array());
    expectArraysClose(await dc.array(), await fusedDc.array());
  });
});

describeWithFlags('fused depthwiseConv2D', ALL_ENVS, () => {
  it('basic', async () => {
    const fSize = 2;
    const pad = 'valid';
    const strides = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [-0.303873, -0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.fused.depthwiseConv2d({x, filter: w, strides, pad});
    expect(result.shape).toEqual([1, 2, 2, 1]);
    const expected = [0.47737, 0.40018, 0.00859, -0.09615];
    expectArraysClose(await result.data(), expected);
  });

  it('basic with relu', async () => {
    const fSize = 2;
    const pad = 'valid';
    const strides = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [-0.303873, -0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.fused.depthwiseConv2d(
        {x, filter: w, strides, pad, activation: 'relu'});
    expect(result.shape).toEqual([1, 2, 2, 1]);
    const expected = [0.47737, 0.40018, 0.00859, 0];
    expectArraysClose(await result.data(), expected);
  });

  it('basic with broadcasted bias and relu', async () => {
    const fSize = 2;
    const pad = 'valid';
    const strides = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [-0.303873, -0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.fused.depthwiseConv2d(
        {x, filter: w, strides, pad, bias: tf.scalar(1), activation: 'relu'});
    expect(result.shape).toEqual([1, 2, 2, 1]);
    const expected = [1.47737, 1.40018, 1.00859, 0.90385];
    expectArraysClose(await result.data(), expected);
  });

  it('prelu', async () => {
    const fSize = 3;
    const pad = 'valid';
    const strides = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.149194, 0.089009, 0.654891, 0.083324, 0.537043, 0.644331, 0.563037,
          0.211859, 0.633501, 0.186427, 0.777034, 0.50001,  0.607341, 0.95303,
          0.696479, 0.050387, 0.62045,  0.728049, 0.028043, 0.437009, 0.712881,
          0.741935, 0.974474, 0.621102, 0.171411
        ],
        [1, 5, 5, inDepth]);
    const alpha = tf.tensor4d(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1, 3, 3, 1]);
    const w = tf.tensor4d(
        [
          -0.125386, -0.975199, -0.640437, -0.281895, -0.990968, -0.347208,
          -0.889702, -0.180695, -0.691992
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.fused.depthwiseConv2d({
      x,
      filter: w,
      strides,
      pad,
      activation: 'prelu',
      preluActivationWeights: alpha
    });
    expect(result.shape).toEqual([1, 3, 3, 1]);
    const expected = [
      -0.25400, -0.50118, -0.73622, -0.94068, -1.2298, -1.84585, -2.3089,
      -2.7499, -2.64077
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            tf.fused.depthwiseConv2d({x, filter, strides, pad}));
    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(
        await dx.data(),
        [-3, 2, 1, -8, 1.5, 0.5, -4, 1, 0, -3, 2, 1, -8, 1.5, 0.5, -4, 1, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    expectArraysClose(await dfilter.data(), [26, 38, 62, 74]);
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
    const bias = tf.ones([2, 2, 2, 1]);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const fusedGrads = tf.grads(
        (x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.depthwiseConv2d({
          x,
          filter: w,
          strides,
          pad,
          dataFormat: 'NHWC',
          dilations: [1, 1],
          bias: b
        }));
    const [dxFused, dfilterFused, dbiasFused] =
        fusedGrads([x, filter, bias], dy);

    const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
      const conv = tf.depthwiseConv2d(x, filter, strides, pad);
      const sum = tf.add(conv, bias);
      return sum;
    });
    const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

    expectArraysClose(await dxFused.array(), await dx.array());
    expectArraysClose(await dfilterFused.array(), await dfilter.array());
    expectArraysClose(await dbiasFused.array(), await dbias.array());
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and activation',
     async () => {
       const inputDepth = 1;
       const outputDepth = 1;
       const inputShape: [number, number, number, number] =
           [2, 3, 3, inputDepth];
       const filterSize = 2;
       const strides = 1;
       const pad = 0;

       const filterShape: [number, number, number, number] =
           [filterSize, filterSize, inputDepth, outputDepth];
       const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
       const bias = tf.ones([2, 2, 2, 1]);

       const x = tf.tensor4d(
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
       const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

       const fusedGrads = tf.grads(
           (x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.depthwiseConv2d({
             x,
             filter: w,
             strides,
             pad,
             dataFormat: 'NHWC',
             dilations: [1, 1],
             bias: b,
             activation: 'relu'
           }));
       const [dxFused, dfilterFused, dbiasFused] =
           fusedGrads([x, filter, bias], dy);

       const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
         const conv = tf.depthwiseConv2d(x, filter, strides, pad);
         const sum = tf.add(conv, bias);
         return tf.relu(sum);
       });
       const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

       expectArraysClose(await dxFused.array(), await dx.array());
       expectArraysClose(await dfilterFused.array(), await dfilter.array());
       expectArraysClose(await dbiasFused.array(), await dbias.array());
     });
});

describeWithFlags('fused conv2d', ALL_ENVS, () => {
  it('basic', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({x, filter: w, strides: stride, pad});
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [-5, 2, -11, 5, -17, 8, -23, 11, -29, 14, -35, 17, -41, 20, -47, 23];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with relu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'relu'
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [0, 2, 0, 5, 0, 8, 0, 11, 0, 14, 0, 17, 0, 20, 0, 23];

    expectArraysClose(await result.data(), expected);
  });

  it('relu with stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=1 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 1;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       // TODO(annxingyuan): Make this test work with large inputs
       // https://github.com/tensorflow/tfjs/issues/3143
       const inputData = [];
       for (let i = 0; i < xSize * xSize * inputDepth; i++) {
         inputData.push(i % 5);
       }

       const wData = [];
       for (let i = 0; i < fSize * fSize * inputDepth * outputDepth; i++) {
         wData.push(i % 5);
       }

       const x = tf.tensor4d(inputData, inputShape);
       const w = tf.tensor4d(wData, [fSize, fSize, inputDepth, outputDepth]);

       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'relu'
       });
       expect(result.shape).toEqual([1, 4, 4, 1]);
       expectArraysClose(
           await result.data(), new Float32Array([
             0, 0, 0, 0, 0, 0, 0.0022320011630654335, 0.002256002277135849,
             0.016055995598435402, 0.01836000196635723, 0.020663997158408165,
             0.014543991535902023, 0.01876799948513508, 0.020303994417190552,
             0.02184000052511692, 0.01507200300693512

           ]));
     });

  it('relu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 8;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
       const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'relu',
         bias
       });
       expect(result.shape).toEqual([1, 4, 4, 8]);
       expectArraysClose(await result.data(), new Float32Array([
                           25.75398063659668,
                           0,
                           26.857805252075195,
                           0,
                           33.961631774902344,
                           0,
                           30.065458297729492,
                           0,
                           23.118206024169922,
                           0,
                           24.212820053100586,
                           0,
                           31.307422637939453,
                           0,
                           27.402034759521484,
                           0,
                           20.482431411743164,
                           0,
                           21.567821502685547,
                           0,
                           28.653217315673828,
                           0,
                           24.73861312866211,
                           0,
                           11.078080177307129,
                           0,
                           12.130399703979492,
                           0,
                           19.182720184326172,
                           0,
                           15.235037803649902,
                           0,
                           4.6677775382995605,
                           0.31717729568481445,
                           5.697869777679443,
                           0,
                           12.727968215942383,
                           2.2569849491119385,
                           8.758066177368164,
                           4.226885795593262,
                           2.0319995880126953,
                           2.9575586318969727,
                           3.052880048751831,
                           1.9366796016693115,
                           10.073760032653809,
                           4.915799617767334,
                           6.094639778137207,
                           6.89492130279541,
                           0,
                           5.5979437828063965,
                           0.4078875780105591,
                           4.586280822753906,
                           7.419551849365234,
                           7.5746169090271,
                           3.43121600151062,
                           9.562952041625977,
                           0,
                           6.404943943023682,
                           0,
                           5.401776313781738,
                           6.5998077392578125,
                           8.398608207702637,
                           2.602976083755493,
                           10.395440101623535,
                           0,
                           21.440250396728516,
                           0,
                           20.483882904052734,
                           0,
                           23.527509689331055,
                           0,
                           25.571144104003906,
                           0,
                           24.080629348754883,
                           0,
                           23.133480072021484,
                           0,
                           26.186328887939453,
                           0,
                           28.239177703857422,
                           0,
                           26.721012115478516,
                           0,
                           25.783079147338867,
                           0,
                           28.84514808654785,
                           0,
                           30.907209396362305,
                           0,
                           18.914127349853516,
                           0,
                           17.960111618041992,
                           0,
                           21.006093978881836,
                           0,
                           23.052082061767578,
                           0,
                           17.89089584350586,
                           0,
                           16.95684814453125,
                           0,
                           20.022798538208008,
                           0,
                           22.088754653930664,
                           0,
                           19.06132698059082,
                           0,
                           18.133424758911133,
                           0,
                           21.205520629882812,
                           0,
                           23.27761459350586,
                           0,
                           20.23175811767578,
                           0,
                           19.309999465942383,
                           0,
                           22.388240814208984,
                           0,
                           24.46647834777832,
                           0,
                           13.584352493286133,
                           0,
                           12.6395845413208,
                           0,
                           15.694815635681152,
                           0,
                           17.750045776367188
                         ]));
     });

  it('prelu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 8;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
       const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
       const preluActivationWeights = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8]);

       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'prelu',
         preluActivationWeights,
         bias
       });
       expect(result.shape).toEqual([1, 4, 4, 8]);
       expectArraysClose(
           await result.data(), new Float32Array([
             25.75398063659668,   -41.61178970336914,  26.857805252075195,
             -87.63885498046875,  33.961631774902344,  -114.0812759399414,
             30.065458297729492,  -136.93893432617188, 23.118206024169922,
             -36.33102035522461,  24.212820053100586,  -77.04048156738281,
             31.307422637939453,  -98.12835693359375,  27.402034759521484,
             -115.5947265625,     20.482431411743164,  -31.050262451171875,
             21.567821502685547,  -66.44209289550781,  28.653217315673828,
             -82.17544555664062,  24.73861312866211,   -94.25041198730469,
             11.078080177307129,  -12.208478927612305, 12.130399703979492,
             -28.626232147216797, 19.182720184326172,  -25.253299713134766,
             15.235037803649902,  -18.08960723876953,  4.6677775382995605,
             0.31717729568481445, 5.697869777679443,   -2.8516759872436523,
             12.727968215942383,  2.2569849491119385,  8.758066177368164,
             4.226885795593262,   2.0319995880126953,  2.9575586318969727,
             3.052880048751831,   1.9366796016693115,  10.073760032653809,
             4.915799617767334,   6.094639778137207,   6.89492130279541,
             -0.6037763357162476, 5.5979437828063965,  0.4078875780105591,
             4.586280822753906,   7.419551849365234,   7.5746169090271,
             3.43121600151062,    9.562952041625977,   -1.4065279960632324,
             6.404943943023682,   -1.2100803852081299, 5.401776313781738,
             6.5998077392578125,  8.398608207702637,   2.602976083755493,
             10.395440101623535,  -16.418434143066406, 21.440250396728516,
             -46.38618850708008,  20.483882904052734,  -42.52848815917969,
             23.527509689331055,  -87.84530639648438,  25.571144104003906,
             -19.054208755493164, 24.080629348754883,  -54.32115936279297,
             23.133480072021484,  -55.79951477050781,  26.186328887939453,
             -106.48924255371094, 28.239177703857422,  -21.689987182617188,
             26.721012115478516,  -62.25614929199219,  25.783079147338867,
             -69.070556640625,    28.84514808654785,   -125.13325500488281,
             30.907209396362305,  -13.891133308410645, 18.914127349853516,
             -38.81135940551758,  17.960111618041992,  -29.915504455566406,
             21.006093978881836,  -70.20361328125,     23.052082061767578,
             -12.857919692993164, 17.89089584350586,   -35.771610260009766,
             16.95684814453125,   -24.949115753173828, 20.022798538208008,
             -63.39042282104492,  22.088754653930664,  -14.02528190612793,
             19.06132698059082,   -39.2921257019043,   18.133424758911133,
             -30.847349166870117, 21.205520629882812,  -71.69097137451172,
             23.27761459350586,   -15.192638397216797, 20.23175811767578,
             -42.8126335144043,   19.309999465942383,  -36.74560546875,
             22.388240814208984,  -79.99152374267578,  24.46647834777832,
             -8.556736946105957,  13.584352493286133,  -22.835901260375977,
             12.6395845413208,    -3.336000442504883,  15.694815635681152,
             -33.0570182800293,   17.750045776367188
           ]));
     });

  it('basic with bias', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      bias: tf.tensor1d([5, 6])
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [0, 8, -6, 11, -12, 14, -18, 17, -24, 20, -30, 23, -36, 26, -42, 29];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with elu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'elu'
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [-0.99326, 2, -1, 5, -1, 8, -1, 11, -1, 14, -1, 17, -1, 20, -1, 23];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with prelu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const alpha = tf.tensor3d([0.25, 0.75], [1, 1, 2]);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'prelu',
      preluActivationWeights: alpha
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [
      -1.25, 2, -2.75, 5, -4.25, 8, -5.75, 11, -7.25, 14, -8.75, 17, -10.25, 20,
      -11.75, 23
    ];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with broadcasted bias and relu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      bias: tf.scalar(5),
      activation: 'relu'
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [0, 7, 0, 10, 0, 13, 0, 16, 0, 19, 0, 22, 0, 25, 0, 28];

    expectArraysClose(await result.data(), expected);
  });

  it('im2row', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({x, filter: w, strides, pad});

    expectArraysClose(
        await result.data(),
        [10, 5, 10, 50, 25, 50, -10, -5, -10, -50, -25, -50]);
  });

  it('im2row with relu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'relu'
    });

    expectArraysClose(
        await result.data(), [10, 5, 10, 50, 25, 50, 0, 0, 0, 0, 0, 0]);
  });

  it('im2row with prelu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
    const alpha = tf.tensor3d([0.5], [1, 1, inputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'prelu',
      preluActivationWeights: alpha
    });

    expectArraysClose(
        await result.data(),
        [10, 5, 10, 50, 25, 50, -5, -2.5, -5, -25, -12.5, -25]);
  });

  it('pointwise with prelu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [1, 1];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
    const alpha = tf.tensor3d([0.5], [1, 1, inputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'prelu',
      preluActivationWeights: alpha
    });

    expectArraysClose(await result.data(), [
      10,  5,    10,  30,  15,   30,  50,  25,    50,  70,  35,    70,
      20,  10,   20,  40,  20,   40,  60,  30,    60,  80,  40,    80,
      -5,  -2.5, -5,  -15, -7.5, -15, -25, -12.5, -25, -35, -17.5, -35,
      -10, -5,   -10, -20, -10,  -20, -30, -15,   -30, -40, -20,   -40
    ]);
  });

  it('im2row with broadcasted bias and relu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      bias: tf.scalar(5),
      activation: 'relu'
    });

    expectArraysClose(
        await result.data(), [15, 10, 15, 55, 30, 55, 0, 0, 0, 0, 0, 0]);
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            tf.fused.conv2d({x, filter, strides, pad}));
    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(
        await dx.data(),
        [-3, 2, 1, -8, 1.5, 0.5, -4, 1, 0, -3, 2, 1, -8, 1.5, 0.5, -4, 1, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    expectArraysClose(await dfilter.data(), [26, 38, 62, 74]);
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
    const bias = tf.ones([2, 2, 2, 1]);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const fusedGrads =
        tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
          x,
          filter: w,
          strides,
          pad,
          dataFormat: 'NHWC',
          dilations: [1, 1],
          bias: b
        }));
    const [dxFused, dfilterFused, dbiasFused] =
        fusedGrads([x, filter, bias], dy);

    const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
      const conv = tf.conv2d(x, filter, strides, pad);
      const sum = tf.add(conv, bias);
      return sum;
    });
    const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

    expectArraysClose(await dxFused.array(), await dx.array());
    expectArraysClose(await dfilterFused.array(), await dfilter.array());
    expectArraysClose(await dbiasFused.array(), await dbias.array());
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and relu',
     async () => {
       const inputDepth = 1;
       const outputDepth = 1;
       const inputShape: [number, number, number, number] =
           [2, 3, 3, inputDepth];
       const filterSize = 2;
       const strides = 1;
       const pad = 0;

       const filterShape: [number, number, number, number] =
           [filterSize, filterSize, inputDepth, outputDepth];
       const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
       const bias = tf.ones([2, 2, 2, 1]);

       const x = tf.tensor4d(
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
       const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

       const fusedGrads =
           tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
             x,
             filter: w,
             strides,
             pad,
             dataFormat: 'NHWC',
             dilations: [1, 1],
             bias: b,
             activation: 'relu'
           }));
       const [dxFused, dfilterFused, dbiasFused] =
           fusedGrads([x, filter, bias], dy);

       const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
         const conv = tf.conv2d(x, filter, strides, pad);
         const sum = tf.add(conv, bias);
         return tf.relu(sum);
       });
       const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

       expectArraysClose(await dxFused.array(), await dx.array());
       expectArraysClose(await dfilterFused.array(), await dfilter.array());
       expectArraysClose(await dbiasFused.array(), await dbias.array());
     });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and elu', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
    const bias = tf.ones([2, 2, 2, 1]);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const fusedGrads =
        tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
          x,
          filter: w,
          strides,
          pad,
          dataFormat: 'NHWC',
          dilations: [1, 1],
          bias: b,
          activation: 'elu'
        }));
    const [dxFused, dfilterFused, dbiasFused] =
        fusedGrads([x, filter, bias], dy);

    const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
      const conv = tf.conv2d(x, filter, strides, pad);
      const sum = tf.add(conv, bias);
      return tf.elu(sum);
    });
    const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

    expectArraysClose(await dxFused.array(), await dx.array());
    expectArraysClose(await dfilterFused.array(), await dfilter.array());
    expectArraysClose(await dbiasFused.array(), await dbias.array());
  });

  it('fused matmul with relu6 and gradients', async () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [2, 3]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);
    const transposeA = false;
    const transposeB = false;

    const fusedGrads = tf.grads((a, b) => {
      return tf.fused.matMul(
          {a, b, transposeA, transposeB, bias: null, activation: 'relu6'});
    });
    const [fusedDa, fusedDb] = fusedGrads([a, b], dy);

    const grads = tf.grads((a, b) => {
      const prod = tf.matMul(a, b, transposeA, transposeB);
      return tf.relu6(prod);
    });
    const [da, db] = grads([a, b], dy);

    expectArraysClose(await da.array(), await fusedDa.array());
    expectArraysClose(await db.data(), await fusedDb.array());
  });
});
