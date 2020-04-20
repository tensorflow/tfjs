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
    inp[i] = i + 1;
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

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth, fSize * fSize * inputDepth);
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
       expect(result.shape).toEqual([1, 4, 4, 1]);
       expectArraysClose(await result.data(), new Float32Array([
                           2209560, 2543640, 2877720, 1890576, 4882200, 5216280,
                           5550360, 3475728, 7554840, 7888920, 8223000, 5060880,
                           4153744, 4302736, 4451728, 2551904
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
