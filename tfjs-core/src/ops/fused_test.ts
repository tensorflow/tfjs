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

function generateCaseInputsPrelu(
    totalSizeTensor: number, totalSizeFilter: number) {
  const inp = new Array(totalSizeTensor);
  const filt = new Array(totalSizeFilter);

  for (let i = 0; i < totalSizeTensor; i++) {
    inp[i] = i - totalSizeTensor / 2;
  }
  for (let i = 0; i < totalSizeFilter; i++) {
    filt[i] = i + 1;
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
       expectArraysClose(await result.data(), new Float32Array([
                           854, 431, 568, 382, 580, 427, 854, 288, 431, 568,
                           580, 289, 285, 570, 285, 258
                         ]));
     });

  it('relu stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=4 p=same', async () => {
    const inputDepth = 16;
    const xSize = 8;
    const inputShape: [number, number, number, number] =
        [1, xSize, xSize, inputDepth];
    const outputDepth = 4;
    const fSize = 3;
    const pad = 'same';
    const stride: [number, number] = [2, 2];

    const inputs = generateCaseInputs(
        1 * xSize * xSize * inputDepth,
        fSize * fSize * inputDepth * outputDepth);
    const x = tf.tensor4d(inputs.input, inputShape);
    const w =
        tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'relu'
    });
    expect(result.shape).toEqual([1, 4, 4, 4]);
    expectArraysClose(
        await result.data(), new Float32Array([
          8772360,  8794320,  8816280,  8838240,  10094856, 10121424, 10147992,
          10174560, 11417352, 11448528, 11479704, 11510880, 7493040,  7516128,
          7539216,  7562304,  19352324, 19411152, 19469980, 19528800, 20674820,
          20738256, 20801692, 20865120, 21997312, 22065360, 22133408, 22201440,
          13759920, 13807584, 13855248, 13902912, 29932280, 30027984, 30123688,
          30219360, 31254776, 31355088, 31455400, 31555680, 32577272, 32682192,
          32787112, 32892000, 20026796, 20099040, 20171284, 20243520, 16368304,
          16450528, 16532752, 16614976, 16955056, 17040352, 17125648, 17210944,
          17541808, 17630176, 17718544, 17806912, 10026272, 10086720, 10147168,
          10207616
        ]));
  });

  it('relu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=4 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 4;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
       const bias = tf.tensor1d([1, 4, 2, 3]);

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
       expect(result.shape).toEqual([1, 4, 4, 4]);
       expectArraysClose(
           await result.data(), new Float32Array([
             8772361,  8794324,  8816282,  8838243,  10094857, 10121428,
             10147994, 10174563, 11417353, 11448532, 11479706, 11510883,
             7493041,  7516132,  7539218,  7562307,  19352324, 19411156,
             19469982, 19528804, 20674820, 20738260, 20801694, 20865124,
             21997312, 22065364, 22133410, 22201444, 13759921, 13807588,
             13855250, 13902915, 29932280, 30027988, 30123690, 30219364,
             31254776, 31355092, 31455402, 31555684, 32577272, 32682196,
             32787114, 32892004, 20026796, 20099044, 20171286, 20243524,
             16368305, 16450532, 16532754, 16614979, 16955056, 17040356,
             17125650, 17210948, 17541808, 17630180, 17718546, 17806916,
             10026273, 10086724, 10147170, 10207619
           ]));
     });

  it('relu stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same', async () => {
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

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'relu'
    });
    expect(result.shape).toEqual([1, 4, 4, 8]);
    expectArraysClose(
        await result.data(), new Float32Array([
          17522760, 17544720, 17566680, 17588640, 17610600, 17632560, 17654520,
          17676480, 20163140, 20189712, 20216284, 20242848, 20269412, 20295984,
          20322556, 20349120, 22803520, 22834704, 22865888, 22897056, 22928224,
          22959408, 22990592, 23021760, 14962992, 14986080, 15009168, 15032256,
          15055344, 15078432, 15101520, 15124608, 38645816, 38704648, 38763500,
          38822304, 38881108, 38939960, 38998792, 39057600, 41286200, 41349640,
          41413100, 41476512, 41539924, 41603384, 41666824, 41730240, 43926584,
          43994624, 44062700, 44130720, 44198740, 44266816, 44334864, 44402880,
          27472168, 27519840, 27567512, 27615168, 27662824, 27710496, 27758168,
          27805824, 59768880, 59864560, 59960308, 60055968, 60151628, 60247376,
          60343056, 60438720, 62409256, 62509552, 62609908, 62710176, 62810444,
          62910800, 63011096, 63111360, 65049640, 65154544, 65259508, 65364384,
          65469260, 65574224, 65679128, 65784000, 39981352, 40053592, 40125852,
          40198080, 40270308, 40342568, 40414808, 40487040, 32654372, 32736608,
          32818844, 32901056, 32983268, 33065504, 33147740, 33229952, 33824804,
          33910112, 33995420, 34080704, 34165988, 34251296, 34336604, 34421888,
          34995236, 35083616, 35171996, 35260352, 35348708, 35437088, 35525464,
          35613824, 19992096, 20052544, 20112992, 20173440, 20233888, 20294336,
          20354784, 20415232
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
       expectArraysClose(
           await result.data(), new Float32Array([
             17522760, 17544724, 17566682, 17588644, 17610608, 17632566,
             17654524, 17676488, 20163140, 20189716, 20216286, 20242852,
             20269420, 20295990, 20322560, 20349128, 22803520, 22834708,
             22865890, 22897060, 22928232, 22959414, 22990596, 23021768,
             14962993, 14986084, 15009170, 15032259, 15055353, 15078438,
             15101525, 15124616, 38645816, 38704652, 38763504, 38822308,
             38881116, 38939968, 38998796, 39057608, 41286200, 41349644,
             41413104, 41476516, 41539932, 41603392, 41666828, 41730248,
             43926584, 43994628, 44062704, 44130724, 44198748, 44266824,
             44334868, 44402888, 27472168, 27519844, 27567514, 27615172,
             27662832, 27710502, 27758172, 27805832, 59768880, 59864564,
             59960312, 60055972, 60151636, 60247384, 60343060, 60438728,
             62409256, 62509556, 62609912, 62710180, 62810452, 62910808,
             63011100, 63111368, 65049640, 65154548, 65259512, 65364388,
             65469268, 65574232, 65679132, 65784008, 39981352, 40053596,
             40125856, 40198084, 40270316, 40342576, 40414812, 40487048,
             32654372, 32736612, 32818846, 32901060, 32983276, 33065510,
             33147744, 33229960, 33824804, 33910116, 33995424, 34080708,
             34165996, 34251304, 34336608, 34421896, 34995236, 35083620,
             35172000, 35260356, 35348716, 35437096, 35525468, 35613832,
             19992096, 20052548, 20112994, 20173444, 20233896, 20294342,
             20354788, 20415240
           ]));
     });

  it('relu stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=32 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 32;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);

       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'relu'
       });
       expect(result.shape).toEqual([1, 4, 4, 32]);
       expectArraysClose(
           await result.data(), new Float32Array([
             70025136,  70047104,  70069104,  70091040,  70112976,  70134984,
             70156944,  70178880,  70200816,  70222776,  70244784,  70266720,
             70288656,  70310656,  70332624,  70354560,  70376496,  70398464,
             70420464,  70442400,  70464336,  70486344,  70508304,  70530240,
             70552176,  70574136,  70596144,  70618080,  70640016,  70662032,
             70683984,  70705920,  80572848,  80599408,  80626032,  80652560,
             80679120,  80705736,  80732304,  80758848,  80785392,  80811960,
             80838576,  80865136,  80891664,  80918288,  80944848,  80971392,
             80997936,  81024496,  81051120,  81077648,  81104208,  81130824,
             81157392,  81183936,  81210480,  81237048,  81263664,  81290224,
             81316752,  81343376,  81369936,  81396480,  91120560,  91151728,
             91182960,  91214080,  91245264,  91276504,  91307664,  91338816,
             91369968,  91401128,  91432368,  91463552,  91494672,  91525904,
             91557072,  91588224,  91619376,  91650544,  91681776,  91712896,
             91744080,  91775320,  91806480,  91837632,  91868784,  91899944,
             91931184,  91962368,  91993488,  92024720,  92055888,  92087040,
             59782688,  59805776,  59828892,  59851968,  59875044,  59898160,
             59921248,  59944320,  59967392,  59990480,  60013596,  60036672,
             60059748,  60082864,  60105952,  60129024,  60152096,  60175184,
             60198300,  60221376,  60244452,  60267568,  60290656,  60313728,
             60336800,  60359888,  60383004,  60406080,  60429156,  60452272,
             60475360,  60498432,  154406848, 154465632, 154524544, 154583264,
             154642112, 154701024, 154759840, 154818592, 154877408, 154936224,
             154995136, 155054000, 155112704, 155171616, 155230400, 155289216,
             155348032, 155406816, 155465728, 155524432, 155583296, 155642208,
             155701024, 155759840, 155818592, 155877408, 155936320, 155995168,
             156053888, 156112800, 156171584, 156230400, 164954560, 165017952,
             165081456, 165144800, 165208272, 165271776, 165335200, 165398560,
             165461984, 165525408, 165588912, 165652400, 165715728, 165779232,
             165842624, 165906048, 165969472, 166032864, 166096368, 166159696,
             166223184, 166286688, 166350112, 166413536, 166476896, 166540320,
             166603824, 166667296, 166730640, 166794144, 166857536, 166920960,
             175502240, 175570272, 175638384, 175706336, 175774416, 175842560,
             175910560, 175978496, 176046560, 176114560, 176182704, 176250800,
             176318736, 176386848, 176454880, 176522880, 176590880, 176658912,
             176727024, 176794960, 176863056, 176931200, 176999200, 177067264,
             177135200, 177203200, 177271344, 177339456, 177407376, 177475488,
             177543520, 177611520, 109745696, 109793344, 109841056, 109888672,
             109936352, 109984056, 110031712, 110079360, 110127008, 110174664,
             110222368, 110270048, 110317664, 110365376, 110413024, 110460672,
             110508320, 110555968, 110603680, 110651296, 110698976, 110746680,
             110794336, 110841984, 110889632, 110937288, 110984992, 111032672,
             111080288, 111128000, 111175648, 111223296, 238788512, 238884160,
             238979952, 239075520, 239171280, 239267072, 239362720, 239458240,
             239554016, 239649664, 239745456, 239841232, 239936784, 240032576,
             240128224, 240223872, 240319520, 240415168, 240510960, 240606512,
             240702288, 240798080, 240893728, 240989504, 241085024, 241180672,
             241276464, 241372224, 241467792, 241563584, 241659232, 241754880,
             249336224, 249436480, 249536880, 249637024, 249737424, 249837824,
             249938080, 250038208, 250138592, 250238848, 250339248, 250439632,
             250539792, 250640192, 250740448, 250840704, 250940960, 251041216,
             251141616, 251241776, 251342160, 251442560, 251542816, 251643200,
             251743328, 251843584, 251943984, 252044384, 252144528, 252244928,
             252345184, 252445440, 259883936, 259988832, 260093808, 260198560,
             260303568, 260408576, 260513440, 260618176, 260723168, 260828032,
             260933040, 261038032, 261142800, 261247776, 261352672, 261457536,
             261562400, 261667296, 261772272, 261877040, 261982032, 262087040,
             262191904, 262296896, 262401632, 262506496, 262611504, 262716512,
             262821264, 262926272, 263031104, 263136000, 159708736, 159780928,
             159853232, 159925408, 159997648, 160069952, 160142176, 160214368,
             160286624, 160358848, 160431152, 160503408, 160575568, 160647872,
             160720064, 160792320, 160864576, 160936768, 161009072, 161081232,
             161153488, 161225792, 161298016, 161370272, 161442464, 161514688,
             161586992, 161659232, 161731408, 161803712, 161875904, 161948160,
             130370848, 130453056, 130535344, 130617488, 130699728, 130782024,
             130864224, 130946432, 131028640, 131110840, 131193136, 131275376,
             131357520, 131439808, 131522016, 131604224, 131686432, 131768640,
             131850928, 131933072, 132015312, 132097608, 132179808, 132262016,
             132344224, 132426424, 132508720, 132590960, 132673104, 132755392,
             132837600, 132919808, 135043360, 135128640, 135214000, 135299216,
             135384528, 135469888, 135555168, 135640448, 135725728, 135811008,
             135896352, 135981680, 136066912, 136152256, 136237536, 136322816,
             136408096, 136493376, 136578720, 136663952, 136749280, 136834624,
             136919904, 137005184, 137090464, 137175744, 137261088, 137346416,
             137431648, 137516992, 137602272, 137687552, 139715872, 139804224,
             139892656, 139980944, 140069328, 140157776, 140246112, 140334464,
             140422800, 140511152, 140599600, 140687984, 140776272, 140864704,
             140953056, 141041408, 141129760, 141218112, 141306544, 141394832,
             141483216, 141571664, 141660016, 141748352, 141836688, 141925040,
             142013488, 142101856, 142190160, 142278608, 142366944, 142455296,
             79787040,  79847472,  79907952,  79968384,  80028816,  80089288,
             80149728,  80210176,  80270624,  80331064,  80391536,  80451968,
             80512400,  80572880,  80633312,  80693760,  80754208,  80814640,
             80875120,  80935552,  80995984,  81056456,  81116896,  81177344,
             81237792,  81298232,  81358704,  81419136,  81479568,  81540048,
             81600480,  81660928
           ]));
     });

  it('relu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=32 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 32;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);

       const bias = tf.tensor1d([
         1, 5, 1, 3, 9, 3, 5,  8,    5,  7,  5,    7,  2,  1,    2,  4,
         6, 3, 6, 8, 4, 8, -5, -2.5, -5, -1, -1.5, -3, -2, -1.5, -5, -3
       ]);
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
       expect(result.shape).toEqual([1, 4, 4, 32]);
       expectArraysClose(
           await result.data(), new Float32Array([
             70025136,  70047112,  70069104,  70091040,  70112984,  70134984,
             70156952,  70178888,  70200824,  70222784,  70244792,  70266728,
             70288656,  70310656,  70332624,  70354560,  70376504,  70398464,
             70420472,  70442408,  70464336,  70486352,  70508296,  70530240,
             70552168,  70574136,  70596144,  70618080,  70640016,  70662032,
             70683976,  70705920,  80572848,  80599416,  80626032,  80652560,
             80679128,  80705736,  80732312,  80758856,  80785400,  80811968,
             80838584,  80865144,  80891664,  80918288,  80944848,  80971392,
             80997944,  81024496,  81051128,  81077656,  81104208,  81130832,
             81157384,  81183936,  81210472,  81237048,  81263664,  81290224,
             81316752,  81343376,  81369928,  81396480,  91120560,  91151736,
             91182960,  91214080,  91245272,  91276504,  91307672,  91338824,
             91369976,  91401136,  91432376,  91463560,  91494672,  91525904,
             91557072,  91588224,  91619384,  91650544,  91681784,  91712904,
             91744080,  91775328,  91806472,  91837632,  91868776,  91899944,
             91931184,  91962368,  91993488,  92024720,  92055880,  92087040,
             59782688,  59805780,  59828892,  59851972,  59875052,  59898164,
             59921252,  59944328,  59967396,  59990488,  60013600,  60036680,
             60059752,  60082864,  60105952,  60129028,  60152104,  60175188,
             60198304,  60221384,  60244456,  60267576,  60290652,  60313724,
             60336796,  60359888,  60383004,  60406076,  60429152,  60452272,
             60475356,  60498428,  154406848, 154465632, 154524544, 154583264,
             154642128, 154701024, 154759840, 154818592, 154877408, 154936224,
             154995136, 155054000, 155112704, 155171616, 155230400, 155289216,
             155348032, 155406816, 155465728, 155524448, 155583296, 155642208,
             155701024, 155759840, 155818592, 155877408, 155936320, 155995168,
             156053888, 156112800, 156171584, 156230400, 164954560, 165017952,
             165081456, 165144800, 165208288, 165271776, 165335200, 165398560,
             165461984, 165525408, 165588912, 165652400, 165715728, 165779232,
             165842624, 165906048, 165969472, 166032864, 166096368, 166159712,
             166223184, 166286688, 166350112, 166413536, 166476896, 166540320,
             166603824, 166667296, 166730640, 166794144, 166857536, 166920960,
             175502240, 175570272, 175638384, 175706336, 175774432, 175842560,
             175910560, 175978496, 176046560, 176114560, 176182704, 176250800,
             176318736, 176386848, 176454880, 176522880, 176590880, 176658912,
             176727024, 176794976, 176863056, 176931200, 176999200, 177067264,
             177135200, 177203200, 177271344, 177339456, 177407376, 177475488,
             177543520, 177611520, 109745696, 109793352, 109841056, 109888672,
             109936360, 109984056, 110031720, 110079368, 110127016, 110174672,
             110222376, 110270056, 110317664, 110365376, 110413024, 110460672,
             110508328, 110555968, 110603688, 110651304, 110698976, 110746688,
             110794328, 110841984, 110889624, 110937288, 110984992, 111032672,
             111080288, 111128000, 111175640, 111223296, 238788512, 238884160,
             238979952, 239075520, 239171296, 239267072, 239362720, 239458240,
             239554016, 239649664, 239745456, 239841232, 239936784, 240032576,
             240128224, 240223872, 240319520, 240415168, 240510960, 240606528,
             240702288, 240798080, 240893728, 240989504, 241085024, 241180672,
             241276464, 241372224, 241467792, 241563584, 241659232, 241754880,
             249336224, 249436480, 249536880, 249637024, 249737440, 249837824,
             249938080, 250038208, 250138592, 250238848, 250339248, 250439632,
             250539792, 250640192, 250740448, 250840704, 250940960, 251041216,
             251141616, 251241792, 251342160, 251442560, 251542816, 251643200,
             251743328, 251843584, 251943984, 252044384, 252144528, 252244928,
             252345184, 252445440, 259883936, 259988832, 260093808, 260198560,
             260303584, 260408576, 260513440, 260618176, 260723168, 260828032,
             260933040, 261038032, 261142800, 261247776, 261352672, 261457536,
             261562400, 261667296, 261772272, 261877056, 261982032, 262087040,
             262191904, 262296896, 262401632, 262506496, 262611504, 262716512,
             262821264, 262926272, 263031104, 263136000, 159708736, 159780928,
             159853232, 159925408, 159997664, 160069952, 160142176, 160214368,
             160286624, 160358848, 160431152, 160503408, 160575568, 160647872,
             160720064, 160792320, 160864576, 160936768, 161009072, 161081248,
             161153488, 161225792, 161298016, 161370272, 161442464, 161514688,
             161586992, 161659232, 161731408, 161803712, 161875904, 161948160,
             130370848, 130453064, 130535344, 130617488, 130699736, 130782024,
             130864232, 130946440, 131028648, 131110848, 131193144, 131275384,
             131357520, 131439808, 131522016, 131604224, 131686440, 131768640,
             131850936, 131933080, 132015312, 132097616, 132179800, 132262016,
             132344216, 132426424, 132508720, 132590960, 132673104, 132755392,
             132837592, 132919808, 135043360, 135128640, 135214000, 135299216,
             135384544, 135469888, 135555168, 135640448, 135725728, 135811008,
             135896352, 135981680, 136066912, 136152256, 136237536, 136322816,
             136408096, 136493376, 136578720, 136663968, 136749280, 136834624,
             136919904, 137005184, 137090464, 137175744, 137261088, 137346416,
             137431648, 137516992, 137602272, 137687552, 139715872, 139804224,
             139892656, 139980944, 140069344, 140157776, 140246112, 140334464,
             140422800, 140511152, 140599600, 140687984, 140776272, 140864704,
             140953056, 141041408, 141129760, 141218112, 141306544, 141394848,
             141483216, 141571680, 141660016, 141748352, 141836688, 141925040,
             142013488, 142101856, 142190160, 142278608, 142366944, 142455296,
             79787040,  79847480,  79907952,  79968384,  80028824,  80089288,
             80149736,  80210184,  80270632,  80331072,  80391544,  80451976,
             80512400,  80572880,  80633312,  80693760,  80754216,  80814640,
             80875128,  80935560,  80995984,  81056464,  81116888,  81177344,
             81237784,  81298232,  81358704,  81419136,  81479568,  81540048,
             81600472,  81660928
           ]));
     });

  it('prelu stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 8;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputsPrelu(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
       const preluActivationWeights = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8]);

       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'prelu',
         preluActivationWeights
       });
       expect(result.shape).toEqual([1, 4, 4, 8]);
       expectArraysClose(
           await result.data(), new Float32Array([
             -24805912,  -49715616,  -74729112,  -99846528,  -125067800,
             -150392736, -175821472, -201354240, -22165524,  -44425632,
             -66780324,  -89229696,  -111773696, -134412192, -157145264,
             -179973120, -19525136,  -39135648,  -58831536,  -78612864,
             -98479600,  -118431648, -138469072, -158592000, -10104240,
             -20260800,  -30469680,  -40730880,  -51044400,  -61410240,
             -71828400,  -82298880,  -3682824,   -7395744,   -11138760,
             -14911872,  -18715080,  -22548384,  -26411784,  -30305280,
             -1042440,   -2105760,   -3189960,   -4295040,   -5421000,
             -6567840,   -7735560,   -8924160,   1597944,    1592112,
             1586280,    1580448,    1574616,    1568784,    1562952,
             1557120,    2404944,    2403360,    2401776,    2400192,
             2398608,    2397024,    2395440,    2393856,    17440248,
             17462064,   17483880,   17505696,   17527512,   17549328,
             17571144,   17592960,   20080628,   20107056,   20133484,
             20159904,   20186324,   20212752,   20239180,   20265600,
             22721008,   22752048,   22783088,   22814112,   22845136,
             22876176,   22907216,   22938240,   14914128,   14937120,
             14960112,   14983104,   15006096,   15029088,   15052080,
             15075072,   13890896,   13923872,   13956848,   13989824,
             14022800,   14055776,   14088752,   14121728,   15061328,
             15097376,   15133424,   15169472,   15205520,   15241568,
             15277616,   15313664,   16231760,   16270880,   16310000,
             16349120,   16388240,   16427360,   16466480,   16505600,
             9584352,    9611968,    9639584,    9667200,    9694816,
             9722432,    9750048,    9777664
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

       const inputs = generateCaseInputsPrelu(
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
             -24805912,  -49715608,  -74729104,  -99846512,  -125067760,
             -150392704, -175821440, -201354176, -22165524,  -44425624,
             -66780320,  -89229680,  -111773664, -134412160, -157145248,
             -179973056, -19525136,  -39135640,  -58831528,  -78612848,
             -98479560,  -118431616, -138469040, -158591936, -10104239,
             -20260792,  -30469674,  -40730868,  -51044356,  -61410204,
             -71828368,  -82298816,  -3682823,   -7395736,   -11138754,
             -14911860,  -18715036,  -22548348,  -26411748,  -30305216,
             -1042439,   -2105752,   -3189954,   -4295028,   -5420955,
             -6567804,   -7735525,   -8924096,   1597945,    1592116,
             1586282,    1580451,    1574625,    1568790,    1562957,
             1557128,    2404945,    2403364,    2401778,    2400195,
             2398617,    2397030,    2395445,    2393864,    17440248,
             17462068,   17483882,   17505700,   17527520,   17549334,
             17571148,   17592968,   20080628,   20107060,   20133486,
             20159908,   20186332,   20212758,   20239184,   20265608,
             22721008,   22752052,   22783090,   22814116,   22845144,
             22876182,   22907220,   22938248,   14914129,   14937124,
             14960114,   14983107,   15006105,   15029094,   15052085,
             15075080,   13890897,   13923876,   13956850,   13989827,
             14022809,   14055782,   14088757,   14121736,   15061329,
             15097380,   15133426,   15169475,   15205529,   15241574,
             15277621,   15313672,   16231761,   16270884,   16310002,
             16349123,   16388249,   16427366,   16466485,   16505608,
             9584353,    9611972,    9639586,    9667203,    9694825,
             9722438,    9750053,    9777672
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
