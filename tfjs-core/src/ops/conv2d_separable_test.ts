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
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('separableConv2d', ALL_ENVS, () => {
  it('input=1x3x3x1,f=2,s=1,d=1,p=valid,chMul=1,outDepth=2', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;
    const outDepth = 2;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const depthwiseFilter = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );
    const pointwiseFilter =
        tf.tensor4d([0.1, -0.2], [1, 1, inDepth * chMul, outDepth]);

    const result =
        tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.10702161, -0.21404321, 0.10316753, -0.20633507, 0.06704096, -0.13408193,
      0.07788632, -0.15577264
    ]);
    expect(result.shape).toEqual([1, 2, 2, outDepth]);
  });

  it('input=1x3x3x1,f=2,s=1,d=1,p=valid,chMul=1,outDepth=2 in tensor',
     async () => {
       const fSize = 2;
       const pad = 'valid';
       const stride = 1;
       const chMul = 1;
       const inDepth = 1;
       const outDepth = 2;

       const x = tf.tensor4d(
           [
             0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
             0.0741907, 0.409265, 0.351377
           ],
           [1, 3, 3, inDepth]);
       const depthwiseFilter = tf.tensor4d(
           [0.303873, 0.229223, 0.144333, 0.803373],
           [fSize, fSize, inDepth, chMul],
       );
       const pointwiseFilter =
           tf.tensor4d([0.1, -0.2], [1, 1, inDepth * chMul, outDepth]);

       const result =
           x.separableConv2d(depthwiseFilter, pointwiseFilter, stride, pad);

       expectArraysClose(await result.data(), [
         0.10702161, -0.21404321, 0.10316753, -0.20633507, 0.06704096,
         -0.13408193, 0.07788632, -0.15577264
       ]);
       expect(result.shape).toEqual([1, 2, 2, outDepth]);
     });

  it('input=1x3x3x1,f=2,s=1,d=1,p=valid,chMul=2,outDepth=2', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 2;
    const inDepth = 1;
    const outDepth = 3;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const depthwiseFilter = tf.tensor4d(
        [
          0.303873, 0.229223, 0.144333, 0.803373, -0.303873, -0.229223,
          -0.144333, -0.803373
        ],
        [fSize, fSize, inDepth, chMul],
    );
    const pointwiseFilter = tf.tensor4d(
        [0.1, -0.2, -0.1, 0.2, 0.15, 0.15], [1, 1, inDepth * chMul, outDepth]);

    const result =
        tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.00305368, 0.0140969, 0.00980358, -0.10853045, -0.06339455, -0.0699412,
      0.11010849, 0.0347524, 0.05214475, 0.10307151, 0.02221644, 0.04224815
    ]);
    expect(result.shape).toEqual([1, 2, 2, outDepth]);
  });

  it('input=1x3x3x1,f=2,s=1,d=1,p=valid,chMul=1,outDepth=2,3D input',
     async () => {
       const fSize = 2;
       const pad = 'valid';
       const stride = 1;
       const chMul = 1;
       const inDepth = 1;
       const outDepth = 2;

       const x = tf.tensor3d(
           [
             0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
             0.0741907, 0.409265, 0.351377
           ],
           [3, 3, inDepth]);
       const depthwiseFilter = tf.tensor4d(
           [0.303873, 0.229223, 0.144333, 0.803373],
           [fSize, fSize, inDepth, chMul],
       );
       const pointwiseFilter =
           tf.tensor4d([0.1, -0.2], [1, 1, inDepth * chMul, outDepth]);

       const result =
           tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

       expectArraysClose(await result.data(), [
         0.10702161, -0.21404321, 0.10316753, -0.20633507, 0.06704096,
         -0.13408193, 0.07788632, -0.15577264
       ]);
       expect(result.shape).toEqual([2, 2, outDepth]);
     });

  it('input=1x4x4x1,f=2,s=2,d=1,p=valid,chMul=1,outDepth=2', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride: [number, number] = [2, 2];
    const chMul = 1;
    const inDepth = 1;
    const outDepth = 2;

    const x = tf.tensor4d(
        [
          0.675707, 0.758567, 0.413529, 0.963967, 0.217291, 0.101335, 0.804231,
          0.329673, 0.924503, 0.728742, 0.180217, 0.210459, 0.133869, 0.650827,
          0.047613, 0.554795
        ],
        [1, 4, 4, inDepth]);
    const depthwiseFilter = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );
    const pointwiseFilter =
        tf.tensor4d([0.1, -0.2], [1, 1, inDepth * chMul, outDepth]);

    const result =
        tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.04919822, -0.09839644, 0.07275512, -0.14551024, 0.09901544, -0.19803089,
      0.05555845, -0.11111691
    ]);
    expect(result.shape).toEqual([1, 2, 2, outDepth]);
  });

  it('input=2x4x4x1,f=2,s=2,d=1,p=valid,chMul=1,outDepth=2', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride: [number, number] = [2, 2];
    const chMul = 1;
    const inDepth = 1;
    const outDepth = 2;

    const x = tf.tensor4d(
        [
          0.675707,  0.758567,  0.413529,  0.963967,  0.217291,  0.101335,
          0.804231,  0.329673,  0.924503,  0.728742,  0.180217,  0.210459,
          0.133869,  0.650827,  0.047613,  0.554795,  -0.675707, -0.758567,
          -0.413529, -0.963967, -0.217291, -0.101335, -0.804231, -0.329673,
          -0.924503, -0.728742, -0.180217, -0.210459, -0.133869, -0.650827,
          -0.047613, -0.554795
        ],
        [2, 4, 4, inDepth]);
    const depthwiseFilter = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );
    const pointwiseFilter =
        tf.tensor4d([0.1, -0.2], [1, 1, inDepth * chMul, outDepth]);

    const result =
        tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.04919822, -0.09839644, 0.07275512, -0.14551024, 0.09901544, -0.19803089,
      0.05555845, -0.11111691, -0.04919822, 0.09839644, -0.07275512, 0.14551024,
      -0.09901544, 0.19803089, -0.05555845, 0.11111691
    ]);
    expect(result.shape).toEqual([2, 2, 2, outDepth]);
  });

  it('input=1x4x4x2,f=2,s=2,d=1,p=valid,chMul=1,outDepth=2', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride: [number, number] = [2, 2];
    const chMul = 1;
    const inDepth = 2;
    const outDepth = 2;

    const x = tf.tensor4d(
        [
          0.675707,  0.758567,  0.413529,  0.963967,  0.217291,  0.101335,
          0.804231,  0.329673,  0.924503,  0.728742,  0.180217,  0.210459,
          0.133869,  0.650827,  0.047613,  0.554795,  -0.675707, -0.758567,
          -0.413529, -0.963967, -0.217291, -0.101335, -0.804231, -0.329673,
          -0.924503, -0.728742, -0.180217, -0.210459, -0.133869, -0.650827,
          -0.047613, -0.554795
        ],
        [1, 4, 4, inDepth]);
    const depthwiseFilter = tf.tensor4d(
        [
          0.303873, 0.229223, 0.144333, 0.803373, 0.98976838, 0.56597068,
          0.42654137, 0.66445535
        ],
        [fSize, fSize, inDepth, chMul],
    );
    const pointwiseFilter = tf.tensor4d(
        [0.1, -0.2, 0.05, -0.05], [1, 1, inDepth * chMul, outDepth]);

    const result =
        tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.20072255, -0.32641545, 0.08474462, -0.11823604, -0.20072255, 0.32641545,
      -0.08474462, 0.11823604
    ]);
    expect(result.shape).toEqual([1, 2, 2, outDepth]);
  });

  it('input=1x4x4x1,f=2,s=1,d=2,p=valid,chMul=1,outDepth=2', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;
    const outDepth = 2;
    const dilationRate = 2;

    const x = tf.tensor4d(
        [
          0.675707, 0.758567, 0.413529, 0.963967, 0.217291, 0.101335, 0.804231,
          0.329673, 0.924503, 0.728742, 0.180217, 0.210459, 0.133869, 0.650827,
          0.047613, 0.554795
        ],
        [1, 4, 4, inDepth]);
    const depthwiseFilter = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );
    const pointwiseFilter =
        tf.tensor4d([0.1, -0.2], [1, 1, inDepth * chMul, outDepth]);

    const result = tf.separableConv2d(
        x, depthwiseFilter, pointwiseFilter, stride, pad, dilationRate);

    expectArraysClose(await result.data(), [
      0.05783373, -0.11566745, 0.07257301, -0.14514601, 0.03079498, -0.06158997,
      0.06460048, -0.12920095
    ]);
    expect(result.shape).toEqual([1, 2, 2, outDepth]);
  });

  it('input=1x4x4x1,f=2,s=1,d=1,p=same,chMul=1,outDepth=2', async () => {
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;
    const outDepth = 2;

    const x = tf.tensor4d(
        [
          0.675707, 0.758567, 0.413529, 0.963967, 0.217291, 0.101335, 0.804231,
          0.329673, 0.924503, 0.728742, 0.180217, 0.210459, 0.133869, 0.650827,
          0.047613, 0.554795
        ],
        [1, 4, 4, inDepth]);
    const depthwiseFilter = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );
    const pointwiseFilter =
        tf.tensor4d([0.1, -0.2], [1, 1, inDepth * chMul, outDepth]);

    const result =
        tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.04919822, -0.09839644, 0.09860218, -0.19720435, 0.07275512, -0.14551024,
      0.03405062, -0.06810125, 0.08081452, -0.16162904, 0.04651042, -0.09302084,
      0.05150411, -0.10300821, 0.01305549, -0.02611098, 0.09901544, -0.19803089,
      0.03949417, -0.07898834, 0.05555845, -0.11111691, 0.0144028,  -0.02880561,
      0.01898637, -0.03797274, 0.02086828, -0.04173655, 0.01416401, -0.02832802,
      0.01685872, -0.03371745
    ]);
    expect(result.shape).toEqual([1, 4, 4, outDepth]);
  });

  it('TensorLike', async () => {
    const pad = 'valid';
    const stride = 1;
    const outDepth = 2;

    const x = [[
      [[0.230664], [0.987388], [0.0685208]],
      [[0.419224], [0.887861], [0.731641]],
      [[0.0741907], [0.409265], [0.351377]]
    ]];
    const depthwiseFilter =
        [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];
    const pointwiseFilter = [[[[0.1, -0.2]]]];

    const result =
        tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.10702161, -0.21404321, 0.10316753, -0.20633507, 0.06704096, -0.13408193,
      0.07788632, -0.15577264
    ]);
    expect(result.shape).toEqual([1, 2, 2, outDepth]);
  });

  it('TensorLike Chained', async () => {
    const pad = 'valid';
    const stride = 1;
    const outDepth = 2;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const depthwiseFilter =
        [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];
    const pointwiseFilter = [[[[0.1, -0.2]]]];

    const result =
        x.separableConv2d(depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.10702161, -0.21404321, 0.10316753, -0.20633507, 0.06704096, -0.13408193,
      0.07788632, -0.15577264
    ]);
    expect(result.shape).toEqual([1, 2, 2, outDepth]);
  });

  it('Incorrect input rank raises error', () => {
    // tslint:disable-next-line:no-any
    const x = tf.zeros([4, 4]) as any;
    const depthwiseFilter: tf.Tensor4D = tf.zeros([2, 2, 1, 3]);
    const pointwiseFilter: tf.Tensor4D = tf.zeros([1, 1, 2, 4]);
    expect(
        () =>
            tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, 1, 'valid'))
        .toThrowError(/rank 4/);
  });

  it('Incorrect depthwise filter rank raises error', () => {
    const x: tf.Tensor4D = tf.zeros([1, 4, 4, 1]);
    // tslint:disable-next-line:no-any
    const depthwiseFilter = tf.zeros([2, 2, 1]) as any;
    const pointwiseFilter: tf.Tensor4D = tf.zeros([1, 1, 2, 4]);
    expect(
        () =>
            tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, 1, 'valid'))
        .toThrowError(/rank 4/);
  });

  it('Incorrect depthwise filter rank raises error', () => {
    const x: tf.Tensor4D = tf.zeros([1, 4, 4, 1]);
    const depthwiseFilter: tf.Tensor4D = tf.zeros([2, 2, 1, 3]);
    // tslint:disable-next-line:no-any
    const pointwiseFilter = tf.zeros([1, 1, 2]) as any;
    expect(
        () =>
            tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, 1, 'valid'))
        .toThrowError(/rank 4/);
  });

  it('Incorrect point filter 1st dimension raises error', () => {
    const x: tf.Tensor4D = tf.zeros([1, 4, 4, 1]);
    const depthwiseFilter: tf.Tensor4D = tf.zeros([2, 2, 1, 3]);
    const pointwiseFilter: tf.Tensor4D = tf.zeros([2, 1, 3, 6]);
    expect(
        () =>
            tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, 1, 'valid'))
        .toThrowError(/must be 1, but got 2/);
  });

  it('Incorrect point filter 2nd dimension raises error', () => {
    const x: tf.Tensor4D = tf.zeros([1, 4, 4, 1]);
    const depthwiseFilter: tf.Tensor4D = tf.zeros([2, 2, 1, 3]);
    const pointwiseFilter: tf.Tensor4D = tf.zeros([1, 5, 3, 6]);
    expect(
        () =>
            tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, 1, 'valid'))
        .toThrowError(/must be 1, but got 5/);
  });

  it('Incorrect pointwise filter 3rd dimension raises error', () => {
    const x: tf.Tensor4D = tf.zeros([1, 4, 4, 1]);
    const depthwiseFilter: tf.Tensor4D = tf.zeros([2, 2, 1, 3]);
    const pointwiseFilter: tf.Tensor4D = tf.zeros([1, 1, 4, 6]);
    expect(
        () =>
            tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, 1, 'valid'))
        .toThrowError(/must be 3, but got 4/);
  });

  it('throws when passed x as a non-tensor', () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;
    const outDepth = 2;

    const depthwiseFilter: tf.Tensor4D =
        tf.zeros([fSize, fSize, inDepth, chMul]);
    const pointwiseFilter: tf.Tensor4D =
        tf.zeros([1, 1, inDepth * chMul, outDepth]);

    const e = /Argument 'x' passed to 'separableConv2d' must be a Tensor/;
    expect(
        () => tf.separableConv2d(
            {} as tf.Tensor3D, depthwiseFilter, pointwiseFilter, stride, pad))
        .toThrowError(e);
  });

  it('throws when passed depthwiseFilter as a non-tensor', () => {
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;
    const outDepth = 2;

    const x: tf.Tensor4D = tf.zeros([1, 3, 3, inDepth]);
    const pointwiseFilter: tf.Tensor4D =
        tf.zeros([1, 1, inDepth * chMul, outDepth]);

    const e = new RegExp(
        'Argument \'depthwiseFilter\' passed to \'separableConv2d\'' +
        ' must be a Tensor');
    expect(
        () => tf.separableConv2d(
            x, {} as tf.Tensor4D, pointwiseFilter, stride, pad))
        .toThrowError(e);
  });

  it('throws when passed pointwiseFilter as a non-tensor', () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x: tf.Tensor4D = tf.zeros([1, 3, 3, inDepth]);
    const depthwiseFilter: tf.Tensor4D =
        tf.zeros([fSize, fSize, inDepth, chMul]);

    const e = new RegExp(
        'Argument \'pointwiseFilter\' passed to \'separableConv2d\'' +
        ' must be a Tensor');
    expect(
        () => tf.separableConv2d(
            x, depthwiseFilter, {} as tf.Tensor4D, stride, pad))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const pad = 'valid';
    const stride = 1;
    const outDepth = 2;

    // 3x3x1
    const x = [
      [[0.230664], [0.987388], [0.0685208]],
      [[0.419224], [0.887861], [0.731641]],
      [[0.0741907], [0.409265], [0.351377]]
    ];
    // 2x2x1x1
    const depthwiseFilter =
        [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];
    // 1x1x1x2
    const pointwiseFilter = [[[[0.1, -0.2]]]];

    const result =
        tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, stride, pad);

    expectArraysClose(await result.data(), [
      0.10702161, -0.21404321, 0.10316753, -0.20633507, 0.06704096, -0.13408193,
      0.07788632, -0.15577264
    ]);
    expect(result.shape).toEqual([2, 2, outDepth]);
  });
});
