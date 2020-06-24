/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import {Rank} from '../types';

describeWithFlags('conv2dTranspose', ALL_ENVS, () => {
  it('input=2x2x1,d2=1,f=2,s=1,p=0', async () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor3d([2], inputShape);
    const w = tf.tensor4d(
        [3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);

    const result = tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
    const expected = [6, 2, 10, 0];

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('input=2x2x1,d2=1,f=2,s=1,p=0, batch=2', async () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number, number] =
        [2, 1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor4d([2, 3], inputShape);
    const w = tf.tensor4d(
        [3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);

    const result = tf.conv2dTranspose(x, w, [2, 2, 2, 1], origStride, origPad);
    const expected = [6, 2, 10, 0, 9, 3, 15, 0];

    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  // Reference (Python) TensorFlow code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // x = tf.constant(np.array([[
  //     [[-0.14656299], [0.32942239], [-1.90302866]],
  //     [[-0.06487813], [-2.02637842], [-1.83669377]],
  //     [[0.82650784], [-0.89249092], [0.01207666]]
  // ]]).astype(np.float32))
  // filt = tf.constant(np.array([
  //     [[[-0.48280062], [1.26770487]], [[-0.83083738], [0.54341856]]],
  //     [[[-0.274904], [0.73111374]], [[2.01885189], [-2.68975237]]]
  // ]).astype(np.float32))
  //
  // with tf.GradientTape() as g:
  //   g.watch(x)
  //   g.watch(filt)
  //   y = tf.keras.backend.conv2d_transpose(x, filt, [1, 4, 4, 2])
  //   print(y)
  // (x_grad, filt_grad) = g.gradient(y, [x, filt])
  //
  // print("x_grad = %s" % x_grad)
  // print("filt_grad = %s" % filt_grad)
  // ```
  it('gradient with clones input=[1,3,3,1] f=[2,2,2,1] s=1 padding=valid',
     async () => {
       const inputDepth = 1;
       const outputDepth = 2;
       const inputShape: [number, number, number, number] =
           [1, 3, 3, inputDepth];
       const filterSize = 2;
       const stride = 1;
       const pad = 'valid';

       const filterShape: [number, number, number, number] =
           [filterSize, filterSize, outputDepth, inputDepth];

       const x = tf.tensor4d(
           [[
             [[-0.14656299], [0.32942239], [-1.90302866]],
             [[-0.06487813], [-2.02637842], [-1.83669377]],
             [[0.82650784], [-0.89249092], [0.01207666]]
           ]],
           inputShape);
       const filt = tf.tensor4d(
           [
             [[[-0.48280062], [1.26770487]], [[-0.83083738], [0.54341856]]],
             [[[-0.274904], [0.73111374]], [[2.01885189], [-2.68975237]]]
           ],
           filterShape);

       const grads = tf.grads(
           (x: tf.Tensor4D, filter: tf.Tensor4D) =>
               tf.conv2dTranspose(
                     x.clone(), filter.clone(), [1, 4, 4, outputDepth], stride,
                     pad)
                   .clone());
       const dy = tf.ones([1, 4, 4, outputDepth]);
       const [xGrad, filtGrad] = grads([x, filt], dy);

       const expectedXGrad = tf.ones([1, 3, 3, 1]).mul(tf.scalar(0.2827947));
       expectArraysClose(await xGrad.data(), await expectedXGrad.data());
       const expectedFiltGrad =
           tf.ones([2, 2, 2, 1]).mul(tf.scalar(-5.70202599));
       expectArraysClose(await filtGrad.data(), await expectedFiltGrad.data());
     });

  // Reference (Python) TensorFlow code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // x = tf.constant(np.array([
  //     [[[-0.36541713], [-0.53973116]], [[0.01731674], [0.90227772]]]
  // ]).astype(np.float32))
  // filt = tf.constant(np.array([
  //     [[[-0.01423461], [-1.00267384]], [[1.61163029], [0.66302646]]],
  //     [[[-0.46900087], [-0.78649444]], [[0.87780536], [-0.84551637]]]
  // ]).astype(np.float32))
  //
  // with tf.GradientTape() as g:
  //   g.watch(x)
  //   g.watch(filt)
  //   y = tf.keras.backend.conv2d_transpose(x, filt, [1, 4, 4, 2], strides=(2,
  //   2)) print(y)
  // (x_grad, filt_grad) = g.gradient(y, [x, filt])
  //
  // print("x_grad = %s" % -x_grad)
  // print("filt_grad = %s" % -filt_grad)
  // ```
  it('gradient input=[1,2,2,1] f=[2,2,2,1] s=[2,2] padding=valid', async () => {
    const inputDepth = 1;
    const outputDepth = 2;
    const inputShape: [number, number, number, number] = [1, 2, 2, inputDepth];
    const filterSize = 2;
    const stride: [number, number] = [2, 2];
    const pad = 'valid';

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, outputDepth, inputDepth];

    const x = tf.tensor4d(
        [[[[-0.36541713], [-0.53973116]], [[0.01731674], [0.90227772]]]],
        inputShape);
    const filt = tf.tensor4d(
        [
          [[[-0.01423461], [-1.00267384]], [[1.61163029], [0.66302646]]],
          [[[-0.46900087], [-0.78649444]], [[0.87780536], [-0.84551637]]]
        ],
        filterShape);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            tf.conv2dTranspose(x, filter, [1, 4, 4, outputDepth], stride, pad));
    const dy = tf.ones([1, 4, 4, outputDepth]).mul(tf.scalar(-1));
    const [xGrad, filtGrad] = grads([x, filt], dy);

    const expectedXGrad = tf.ones([1, 2, 2, 1]).mul(tf.scalar(-0.03454196));
    expectArraysClose(await xGrad.data(), await expectedXGrad.data());
    expect(xGrad.shape).toEqual([1, 2, 2, 1]);

    const expectedFiltGrad = tf.ones([2, 2, 2, 1]).mul(tf.scalar(-0.01444618));
    expectArraysClose(await filtGrad.data(), await expectedFiltGrad.data());
    expect(filtGrad.shape).toEqual([2, 2, 2, 1]);
  });

  // Reference (Python) TensorFlow code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // x = tf.constant(np.array([[
  //     [[1.52433065], [-0.77053435], [-0.64562341]],
  //     [[0.77962889], [1.58413887], [-0.25581856]],
  //     [[-0.58966221], [0.05411662], [0.70749138]]
  // ]]).astype(np.float32))
  // filt = tf.constant(np.array([
  //     [[[0.11178388], [-0.96654977]], [[1.21021296], [0.84121729]]],
  //     [[[0.34968338], [-0.42306114]], [[1.27395733], [-1.09014535]]]
  // ]).astype(np.float32))
  //
  // with tf.GradientTape() as g:
  //   g.watch(x)
  //   g.watch(filt)
  //   y = tf.keras.backend.conv2d_transpose(
  //       x, filt, [1, 3, 3, 2], strides=(1, 1), padding='same')
  // (x_grad, filt_grad) = g.gradient(y, [x, filt])
  //
  // print("x_grad = %s" % x_grad)
  // print("filt_grad = %s" % filt_grad)
  // ```
  it('gradient input=[1,3,3,1] f=[2,2,2,1] s=[1,1] padding=same', async () => {
    const inputDepth = 1;
    const outputDepth = 2;
    const inputShape: [number, number, number, number] = [1, 3, 3, inputDepth];
    const filterSize = 2;
    const stride: [number, number] = [1, 1];
    const pad = 'same';

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, outputDepth, inputDepth];

    const x = tf.tensor4d(
        [[
          [[1.52433065], [-0.77053435], [-0.64562341]],
          [[0.77962889], [1.58413887], [-0.25581856]],
          [[-0.58966221], [0.05411662], [0.70749138]]
        ]],
        inputShape);
    const filt = tf.tensor4d(
        [
          [[[0.11178388], [-0.96654977]], [[1.21021296], [0.84121729]]],
          [[[0.34968338], [-0.42306114]], [[1.27395733], [-1.09014535]]]
        ],
        filterShape);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            tf.conv2dTranspose(x, filter, [1, 3, 3, outputDepth], stride, pad));
    const dy = tf.ones([1, 3, 3, outputDepth]);
    const [xGrad, filtGrad] = grads([x, filt], dy);

    expectArraysClose(await xGrad.array(), [[
                        [[1.30709858], [1.30709858], [-0.92814366]],
                        [[1.30709858], [1.30709858], [-0.92814366]],
                        [[1.19666437], [1.19666437], [-0.85476589]]
                      ]]);
    expectArraysClose(await filtGrad.array(), [
      [[[2.38806788], [2.38806788]], [[2.58201847], [2.58201847]]],
      [[[2.2161221], [2.2161221]], [[3.11756406], [3.11756406]]]
    ]);
  });

  // Reference (Python) TensorFlow code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // x = tf.constant(np.array([[
  //     [[1.52433065], [-0.77053435]], [[0.77962889], [1.58413887]],
  // ]]).astype(np.float32))
  // filt = tf.constant(np.array([
  //     [[[0.11178388], [-0.96654977]], [[1.21021296], [0.84121729]]],
  //     [[[0.34968338], [-0.42306114]], [[1.27395733], [-1.09014535]]]
  // ]).astype(np.float32))
  //
  // with tf.GradientTape() as g:
  //   g.watch(x)
  //   g.watch(filt)
  //   y = tf.keras.backend.conv2d_transpose(
  //       x, filt, [1, 3, 3, 2], strides=(2, 2), padding='same')
  //   print(y.shape)
  // (x_grad, filt_grad) = g.gradient(y, [x, filt])
  //
  // print("x_grad = %s" % x_grad)
  // print("filt_grad = %s" % filt_grad)
  // ```
  it('gradient input=[1,2,2,2] f=[2,2,2,1] s=[2,2] padding=same', async () => {
    const inputDepth = 2;
    const outputDepth = 2;
    const inputShape: [number, number, number, number] = [1, 2, 2, inputDepth];
    const filterSize = 2;
    const stride: [number, number] = [2, 2];
    const pad = 'same';

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, outputDepth, inputDepth];

    const x = tf.tensor4d(
        [[
          [[-1.81506593, 1.00900095], [-0.05199118, 0.26311377]],
          [[-1.18469792, -0.34780521], [2.04971242, -0.65154692]]
        ]],
        inputShape);
    const filt = tf.tensor4d(
        [
          [
            [[0.19529686, -0.79594708], [0.70314057, -0.06081263]],
            [[0.28724744, 0.88522715], [-0.51824096, -0.97120989]]
          ],
          [
            [[0.51872197, -1.17569193], [1.28316791, -0.81225092]],
            [[-0.44221532, 0.70058174], [-0.4849217, 0.03806348]]
          ]
        ],
        filterShape);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            tf.conv2dTranspose(x, filter, [1, 3, 3, outputDepth], stride, pad));
    const dy = tf.ones([1, 3, 3, outputDepth]);
    const [xGrad, filtGrad] = grads([x, filt], dy);

    expectArraysClose(await xGrad.data(), [
      1.54219678, -2.19204008, 2.70032732, -2.84470257, 0.66744391, -0.94274245,
      0.89843743, -0.85675972
    ]);
    expect(xGrad.shape).toEqual([1, 2, 2, 2]);
    expectArraysClose(await filtGrad.data(), [
      -1.00204261, 0.27276259, -1.00204261, 0.27276259, -2.99976385, 0.66119574,
      -2.99976385, 0.66119574, -1.86705711, 1.27211472, -1.86705711, 1.27211472,
      -1.81506593, 1.00900095, -1.81506593, 1.00900095
    ]);
    expect(filtGrad.shape).toEqual([2, 2, 2, 2]);
  });

  it('throws when x is not rank 3', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    // tslint:disable-next-line:no-any
    const x: any = tf.tensor2d([2, 2], [2, 1]);
    const w = tf.tensor4d(
        [3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);

    expect(() => tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
        .toThrowError();
  });

  it('throws when weights is not rank 4', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor3d([2], inputShape);
    // tslint:disable-next-line:no-any
    const w: any = tf.tensor3d([3, 1, 5, 0], [fSize, fSize, origInputDepth]);

    expect(() => tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
        .toThrowError();
  });

  it('throws when x depth does not match weights original output depth', () => {
    const origInputDepth = 1;
    const origOutputDepth = 2;
    const wrongOrigOutputDepth = 3;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor3d([2, 2], inputShape);
    const w = tf.randomNormal<Rank.R4>(
        [fSize, fSize, origInputDepth, wrongOrigOutputDepth]);

    expect(() => tf.conv2dTranspose(x, w, [2, 2, 2], origStride, origPad))
        .toThrowError();
  });

  it('throws when passed x as a non-tensor', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const w = tf.tensor4d(
        [3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);

    expect(
        () => tf.conv2dTranspose(
            {} as tf.Tensor3D, w, [2, 2, 1], origStride, origPad))
        .toThrowError(
            /Argument 'x' passed to 'conv2dTranspose' must be a Tensor/);
  });

  it('throws when passed filter as a non-tensor', () => {
    const origOutputDepth = 1;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor3d([2], inputShape);

    expect(
        () => tf.conv2dTranspose(
            x, {} as tf.Tensor4D, [2, 2, 1], origStride, origPad))
        .toThrowError(
            /Argument 'filter' passed to 'conv2dTranspose' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const origPad = 0;
    const origStride = 1;

    const x = [[[2]]];                           // 1x1x1
    const w = [[[[3]], [[1]]], [[[5]], [[0]]]];  // 2x2x1x1

    const result = tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
    const expected = [6, 2, 10, 0];

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });
});
