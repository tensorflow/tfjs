/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

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

  it('basic with channel-wise broadcasted bias and relu', async () => {
    const strides = 1;
    const pad = 'same';
    const x = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8
        ],
        [1, 3, 3, 4]);
    const w = tf.tensor4d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [2, 2, 4, 1]);
    const bias = tf.tensor1d([0, 1, 2, 3]);

    const result = tf.fused.depthwiseConv2d({x, filter: w, strides, pad, bias});
    expect(result.shape).toEqual([1, 3, 3, 4]);
    const expected = [
      124, 167, 92,  142, 112, 117, 76,  124, 16, 28, 44, 64,
      88,  134, 134, 88,  76,  120, 154, 205, 40, 58, 80, 106,
      4,   18,  36,  31,  20,  33,  50,  71,  0,  7,  16, 27
    ];
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

  it('leakyrelu', async () => {
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
    const alpha = 0.3;
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
      activation: 'leakyrelu',
      leakyreluAlpha: alpha
    });

    expect(result.shape).toEqual([1, 3, 3, 1]);
    const expected = [
      -0.7620067596435547, -0.7517655491828918, -0.7362186312675476,
      -0.7055101990699768, -0.7378802299499512, -0.9229262471199036,
      -0.9895440340042114, -1.031226396560669, -0.8802568912506104
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('sigmoid', async () => {
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
    const w = tf.tensor4d(
        [
          -0.125386, -0.975199, -0.640437, -0.281895, -0.990968, -0.347208,
          -0.889702, -0.180695, -0.691992
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.fused.depthwiseConv2d(
        {x, filter: w, strides, pad, activation: 'sigmoid'});

    expect(result.shape).toEqual([1, 3, 3, 1]);
    const expected = [
      0.07309964, 0.07544667, 0.07914197, 0.08693069, 0.07873929, 0.04409045,
      0.03562334, 0.0311462, 0.05048907
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
