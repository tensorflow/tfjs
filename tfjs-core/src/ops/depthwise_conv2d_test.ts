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

describeWithFlags('depthwiseConv2D', ALL_ENVS, () => {
  it('input=1x3x3x1,f=2,s=1,d=1,p=valid,chMul=1', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 2, 2, 1]);
    const expected = [1.07022, 1.03167, 0.67041, 0.778863];
    expectArraysClose(await result.data(), expected);
  });

  it('input=1x3x3x1,f=2,s=1,d=1,p=explicit,chMul=1', async () => {
    const fSize = 2;
    const pad =
        [[0, 0], [1, 2], [0, 1], [0, 0]] as tf.backend_util.ExplicitPadding;
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 5, 3, 1]);
    const expected = [
      0.826533, 0.197560, 0.0098898, 1.070216, 1.031675, 0.126422, 0.6704096,
      0.778863, 0.273041, 0.116357, 0.204908, 0.106774, 0, 0, 0
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=1x5x5x1,f=3,s=1,d=1,p=valid,chMul=1', async () => {
    const fSize = 3;
    const pad = 'valid';
    const stride = 1;
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
          0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
          0.180695, 0.691992
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 3, 3, 1]);
    const expected = [
      2.540022, 2.505885, 2.454062, 2.351701, 2.459601, 3.076421, 3.29848,
      3.437421, 2.93419
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=1x3x3x1,f=2,s=1,d=2,p=valid,chMul=1', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const dilation = 2;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );
    // adding a dilation rate is equivalent to using a filter
    // with 0s for the dilation rate
    const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
    const wDilated = tf.tensor4d(
        [0.303873, 0, 0.229223, 0, 0, 0, 0.144333, 0, 0.803373],
        [fSizeDilated, fSizeDilated, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);

    const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad);

    expect(result.shape).toEqual(expectedResult.shape);
    expectArraysClose(await result.data(), await expectedResult.data());
  });

  it('input=1x5x5x1,f=3,s=1,d=2,p=valid,chMul=1', async () => {
    const fSize = 3;
    const pad = 'valid';
    const stride = 1;
    const dilation = 2;
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
          0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
          0.180695, 0.691992
        ],
        [fSize, fSize, inDepth, chMul],
    );
    // adding a dilation rate is equivalent to using a filter
    // with 0s for the dilation rate
    const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
    const wDilated = tf.tensor4d(
        [
          0.125386, 0, 0.975199, 0, 0.640437, 0, 0, 0, 0, 0,
          0.281895, 0, 0.990968, 0, 0.347208, 0, 0, 0, 0, 0,
          0.889702, 0, 0.180695, 0, 0.691992
        ],
        [fSizeDilated, fSizeDilated, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);

    const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad);

    expect(result.shape).toEqual(expectedResult.shape);
    expectArraysClose(await result.data(), await expectedResult.data());
  });

  it('input=1x5x5x1,f=2,s=1,d=4,p=valid,chMul=1', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const dilation = 4;
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
        [0.125386, 0.975199, 0.640437, 0.281895],
        [fSize, fSize, inDepth, chMul],
    );
    // adding a dilation rate is equivalent to using a filter
    // with 0s for the dilation rate
    const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
    const wDilated = tf.tensor4d(
        [
          0.125386, 0, 0, 0, 0.975199, 0, 0, 0,        0, 0, 0, 0,       0,
          0,        0, 0, 0, 0,        0, 0, 0.640437, 0, 0, 0, 0.281895
        ],
        [fSizeDilated, fSizeDilated, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);

    const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad);

    expect(result.shape).toEqual(expectedResult.shape);
    expectArraysClose(await result.data(), await expectedResult.data());
  });

  it('input=1x3x3x2,f=2,s=1,d=1,p=same,chMul=1', async () => {
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const chMul = 1;
    const inDepth = 2;

    const x = tf.tensor4d(
        [
          0.111057, 0.661818, 0.701979, 0.424362, 0.992854, 0.417599, 0.423036,
          0.500499, 0.368484, 0.714135, 0.456693, 0.531058, 0.636636, 0.345024,
          0.0506303, 0.789682, 0.177473, 0.793569
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [
          0.614293, 0.0648011, 0.101113, 0.452887, 0.0582746, 0.426481,
          0.872743, 0.765767
        ],
        [fSize, fSize, inDepth, chMul]);
    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 3, 3, 2]);

    const expected = [
      0.485445, 0.995389, 0.95166, 0.927856, 0.636516, 0.253547, 0.378414,
      1.10771, 0.430373, 1.23126, 0.290885, 0.372855, 0.3962, 0.379995,
      0.0490466, 0.410569, 0.10902, 0.0514242
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=1x5x5x1,f=3,s=1,d=1,p=same,chMul=1', async () => {
    const fSize = 3;
    const pad = 'same';
    const stride = 1;
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
          0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
          0.180695, 0.691992
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 5, 5, 1]);
    const expected = [
      0.684796, 1.179251, 1.680593, 0.885615, 1.152995, 1.52291,  2.540022,
      2.505885, 2.454062, 1.871258, 2.371015, 2.351701, 2.459601, 3.076421,
      1.323994, 1.985572, 3.29848,  3.437421, 2.93419,  1.823238, 1.410545,
      2.352186, 2.19622,  1.348218, 0.774635
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=1x3x3x2,f=2,s=1,d=2,p=same,chMul=1', async () => {
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const dilation = 2;
    const inDepth = 2;

    const x = tf.tensor4d(
        [
          0.111057, 0.661818, 0.701979, 0.424362, 0.992854, 0.417599, 0.423036,
          0.500499, 0.368484, 0.714135, 0.456693, 0.531058, 0.636636, 0.345024,
          0.0506303, 0.789682, 0.177473, 0.793569
        ],
        [1, 3, 3, inDepth]);

    const w: tf.Tensor4D =
        tf.stack(
              [
                tf.tensor2d(
                    [0.614293, 0.0648011, 0.101113, 0.452887], [fSize, fSize]),
                tf.tensor2d(
                    [0.0582746, 0.426481, 0.872743, 0.765767], [fSize, fSize])
              ],
              2)
            .expandDims(3);

    // adding a dilation rate is equivalent to using a filter
    // with 0s for the dilation rate
    const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
    const wDilated: tf.Tensor4D =
        tf.stack(
              [
                tf.tensor2d(
                    [0.614293, 0, 0.0648011, 0, 0, 0, 0.101113, 0, 0.452887],
                    [fSizeDilated, fSizeDilated]),
                tf.tensor2d(
                    [0.0582746, 0, 0.426481, 0, 0, 0, 0.872743, 0, 0.765767],
                    [fSizeDilated, fSizeDilated])
              ],
              2)
            .expandDims(3);

    expect(wDilated.shape).toEqual([fSizeDilated, fSizeDilated, inDepth, 1]);

    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);

    const expectedResult = tf.depthwiseConv2d(x, wDilated, stride, pad);

    expect(result.shape).toEqual(expectedResult.shape);
    expectArraysClose(await result.data(), await expectedResult.data());
  });

  it('input=1x5x5x1,f=3,s=1,d=2,p=same,chMul=1', async () => {
    const fSize = 3;
    const pad = 'valid';
    const stride = 1;
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
          0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
          0.180695, 0.691992
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 3, 3, 1]);
    const expected = [
      2.540022, 2.505885, 2.454062, 2.351701, 2.459601, 3.076421, 3.29848,
      3.437421, 2.93419
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=1x5x5x1,f=3,s=1,d=2,p=explicit,chMul=1', async () => {
    const fSize = 3;
    const pad =
        [[0, 0], [0, 0], [0, 1], [0, 1]] as tf.backend_util.ExplicitPadding;
    const stride = 1;
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
          0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
          0.180695, 0.691992
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 3, 4, 1]);
    const expected = [
      2.540022, 2.505885, 2.454062, 1.871258, 2.35170, 2.459601, 3.076421,
      1.32399, 3.298480, 3.437421, 2.93419, 1.823238
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=1x3x3x4,f=3,s=1,d=2,p=same,chMul=1', async () => {
    const fSize = 3;
    const pad = 'same';
    const stride = 1;
    const chMul = 1;
    const inDepth = 4;
    const dilation = 2;

    const x = tf.tensor4d(
        [
          0.5227615, 0.3477598, 0.5227615, 0.3477598, 0.4690094, 0.408161,
          0.4690094, 0.408161,  0.3239015, 0.2372907, 0.3239015, 0.2372907,
          0.6136674, 0.7918105, 0.6136674, 0.7918105, 0.9145211, 0.218611,
          0.9145211, 0.218611,  0.3778793, 0.2392365, 0.3778793, 0.2392365,
          0.2340134, 0.1251984, 0.2340134, 0.1251984, 0.6222534, 0.1327361,
          0.6222534, 0.1327361, 0.7697753, 0.1216059, 0.7697753, 0.1216059
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [
          0.6511372, 0.8699447, 0.6511372, 0.8699447, 0.267792,  0.9981787,
          0.267792,  0.9981787, 0.4913572, 0.3321196, 0.4913572, 0.3321196,
          0.5286497, 0.4241803, 0.5286497, 0.4241803, 0.0175446, 0.8365464,
          0.0175446, 0.8365464, 0.1768399, 0.2874831, 0.1768399, 0.2874831,
          0.0933998, 0.5764548, 0.0933998, 0.5764548, 0.0661623, 0.8850273,
          0.0661623, 0.8850273, 0.8700929, 0.205422,  0.8700929, 0.205422
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);
    expect(result.shape).toEqual([1, 3, 3, 4]);
    const expected = [
      0.7517092227935791,  0.4949187934398651,  0.7517092227935791,
      0.4949187934398651,  0.04939830303192139, 0.4589206874370575,
      0.04939830303192139, 0.4589206874370575,  0.3548273742198944,
      0.5258132815361023,  0.3548273742198944,  0.5258132815361023,
      0.0775906890630722,  0.7311626672744751,  0.0775906890630722,
      0.7311626672744751,  0.01604490540921688, 0.1828782558441162,
      0.01604490540921688, 0.1828782558441162,  0.3310448229312897,
      0.5360028743743896,  0.3310448229312897,  0.5360028743743896,
      0.4393753409385681,  0.565629243850708,   0.4393753409385681,
      0.565629243850708,   0.13651414215564728, 0.5184575319290161,
      0.13651414215564728, 0.5184575319290161,  0.5643441677093506,
      0.6942259669303894,  0.5643441677093506,  0.6942259669303894
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=1x3x3x2,f=2,s=1,p=same,chMul=2', async () => {
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const chMul = 2;
    const inDepth = 2;

    const x = tf.tensor4d(
        [
          0.675707, 0.758567, 0.413529, 0.963967, 0.217291, 0.101335, 0.804231,
          0.329673, 0.924503, 0.728742, 0.180217, 0.210459, 0.133869, 0.650827,
          0.047613, 0.554795, 0.653365, 0.442196
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [
          0.347154, 0.386692, 0.327191, 0.483784, 0.591807, 0.24263, 0.95182,
          0.174353, 0.592136, 0.623469, 0.988244, 0.660731, 0.946534, 0.0801365,
          0.864889, 0.874602
        ],
        [fSize, fSize, inDepth, chMul]);
    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 3, 3, 4]);

    const expected = [
      1.83059,   0.937125,  2.1218,   1.39024,  0.990167, 0.803472,
      1.31405,   1.14959,   0.182147, 0.196385, 0.241141, 0.188081,
      0.950656,  0.622581,  1.92451,  1.20179,  1.07422,  0.483268,
      1.36948,   1.14256,   0.449444, 0.477042, 0.505857, 0.393989,
      0.0746509, 0.0633184, 0.74101,  0.41159,  0.403195, 0.176938,
      0.602415,  0.345499,  0.226819, 0.252651, 0.144682, 0.213927
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=2x3x3x2,f=2,s=1,p=same,chMul=2', async () => {
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const chMul = 2;
    const inDepth = 2;

    const x = tf.tensor4d(
        [
          0.261945, 0.0528113, 0.656698,  0.127345,  0.610039, 0.169131,
          0.458647, 0.0988288, 0.966109,  0.0421747, 0.82035,  0.274711,
          0.359377, 0.512113,  0.689682,  0.941571,  0.31961,  0.743826,
          0.858147, 0.984766,  0.926973,  0.579597,  0.444104, 0.505969,
          0.241437, 0.937999,  0.0957074, 0.773611,  0.46023,  0.469379,
          0.363789, 0.269745,  0.486136,  0.894215,  0.794299, 0.724615
        ],
        [2, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [
          0.240347, 0.906352, 0.478657, 0.825918, 0.380769, 0.184705, 0.238241,
          0.201907, 0.294087, 0.181165, 0.191303, 0.7225, 0.430064, 0.900622,
          0.670338, 0.33478
        ],
        [fSize, fSize, inDepth, chMul]);
    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([2, 3, 3, 4]);

    const expected = [
      0.863379, 1.3119,   0.102795, 0.154853, 1.02704,   1.62173,  0.293466,
      0.261764, 0.387876, 0.701529, 0.133508, 0.338167,  0.880395, 1.28039,
      0.786492, 0.775361, 0.884845, 1.43995,  0.764374,  1.0196,   0.291162,
      0.801428, 0.273788, 0.764303, 0.348985, 0.45311,   0.469447, 0.613073,
      0.287461, 0.684128, 0.627899, 0.927844, 0.0768174, 0.28968,  0.356037,
      0.614339, 0.67138,  1.07894,  1.30747,  1.86705,   0.617971, 1.35402,
      0.860607, 1.29693,  0.242087, 0.485892, 0.331979,  0.757015, 0.410527,
      0.740235, 1.28431,  1.42516,  0.68281,  0.975185,  1.13892,  1.62237,
      0.344208, 0.561029, 0.363292, 0.911203, 0.272541,  0.419513, 0.342154,
      0.403335, 0.419286, 0.587321, 0.600655, 0.884853,  0.190907, 0.719914,
      0.346842, 0.598472
    ];
    expectArraysClose(await result.data(), expected);
  });

  it('input=2x3x3x2,f=2,s=1,d=2,p=same,chMul=2',
     async () => {
       const fSize = 2;
       const pad = 'same';
       const stride = 1;
       const inDepth = 2;
       const dilation = 2;
       const noDilation = 1;

       const x = tf.tensor4d(
           [
             0.261945, 0.0528113, 0.656698,  0.127345,  0.610039, 0.169131,
             0.458647, 0.0988288, 0.966109,  0.0421747, 0.82035,  0.274711,
             0.359377, 0.512113,  0.689682,  0.941571,  0.31961,  0.743826,
             0.858147, 0.984766,  0.926973,  0.579597,  0.444104, 0.505969,
             0.241437, 0.937999,  0.0957074, 0.773611,  0.46023,  0.469379,
             0.363789, 0.269745,  0.486136,  0.894215,  0.794299, 0.724615
           ],
           [2, 3, 3, inDepth]);

       const w = tf.stack(
                     [
                       tf.stack(
                           [
                             tf.tensor2d(
                                 [0.240347, 0.906352, 0.478657, 0.825918],
                                 [fSize, fSize]),
                             tf.tensor2d(
                                 [0.380769, 0.184705, 0.238241, 0.201907],
                                 [fSize, fSize])
                           ],
                           2),
                       tf.stack(
                           [
                             tf.tensor2d(
                                 [0.294087, 0.181165, 0.191303, 0.7225],
                                 [fSize, fSize]),
                             tf.tensor2d(
                                 [0.430064, 0.900622, 0.670338, 0.33478],
                                 [fSize, fSize])
                           ],
                           2)
                     ],
                     3) as tf.Tensor4D;

       const fSizeDilated = fSize + (fSize - 1) * (dilation - 1);
       const wDilated = tf.stack([
      tf.stack(
          [
            tf.tensor2d(
              [0.240347, 0, 0.906352, 0, 0, 0, 0.478657, 0, 0.825918],
              [fSizeDilated, fSizeDilated]),
            tf.tensor2d(
              [0.380769, 0, 0.184705, 0, 0, 0, 0.238241, 0, 0.201907],
              [fSizeDilated, fSizeDilated])
          ],
          2),
      tf.stack(
          [
            tf.tensor2d([0.294087, 0, 0.181165, 0, 0, 0, 0.191303, 0, 0.7225],
              [fSizeDilated, fSizeDilated]),
            tf.tensor2d(
              [0.430064, 0, 0.900622, 0, 0, 0, 0.670338, 0, 0.33478],
              [fSizeDilated, fSizeDilated])
          ],
          2)
    ], 3) as tf.Tensor4D;

       const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);

       const expectedResult =
           tf.depthwiseConv2d(x, wDilated, stride, pad, 'NHWC', noDilation);

       expect(result.shape).toEqual(expectedResult.shape);
       expectArraysClose(await result.data(), await expectedResult.data());
     });

  it('input=2x3x3x2,f=3,s=1,d=2,p=same,chMul=2', async () => {
    const fSize = 3;
    const pad = 'same';
    const stride = 1;
    const inDepth = 2;
    const dilation = 2;

    const x = tf.tensor4d(
        [[
          [
            [0.52276146, 0.34775984], [0.4690094, 0.40816104],
            [0.32390153, 0.23729074], [0.61366737, 0.7918105],
            [0.9145211, 0.218611], [0.37787926, 0.23923647],
            [0.23401344, 0.12519836]
          ],

          [
            [0.6222534, 0.13273609], [0.7697753, 0.12160587],
            [0.0448128, 0.94806635], [0.4199953, 0.7140714],
            [0.01420832, 0.47453713], [0.02061439, 0.37226152],
            [0.62741446, 0.23167181]
          ],

          [
            [0.7257557, 0.14352751], [0.3011638, 0.3869065],
            [0.09286129, 0.25151742], [0.7566397, 0.13099921],
            [0.65324724, 0.38959372], [0.65826, 0.7505318],
            [0.35919082, 0.85470796]
          ],

          [
            [0.24827361, 0.2826661], [0.24717247, 0.27446854],
            [0.27112448, 0.68068564], [0.11082292, 0.7948675],
            [0.41535318, 0.659986], [0.22165525, 0.18149579],
            [0.42273378, 0.9558281]
          ],

          [
            [0.943074, 0.6799041], [0.78851473, 0.07249606],
            [0.771909, 0.7925967], [0.9551083, 0.03087568],
            [0.82589805, 0.94797385], [0.5895462, 0.5045923],
            [0.9667754, 0.24292922]
          ],

          [
            [0.67123663, 0.109761], [0.04002762, 0.51942277],
            [0.37868536, 0.8467603], [0.77171385, 0.51604605],
            [0.8192849, 0.38843668], [0.19607484, 0.5591624],
            [0.45990825, 0.35768318]
          ],

          [
            [0.67443585, 0.6256168], [0.9373623, 0.6498393],
            [0.7623085, 0.13218105], [0.9349631, 0.7660191],
            [0.50054944, 0.7738123], [0.30201948, 0.525643],
            [0.30896342, 0.21111596]
          ]
        ]],
        [1, 7, 7, inDepth]);

    const w = tf.tensor4d(
        [
          [
            [[0.65113723], [0.8699447]],

            [[0.267792], [0.9981787]],

            [[0.4913572], [0.33211958]]
          ],
          [
            [[0.5286497], [0.42418027]],

            [[0.01754463], [0.8365464]],

            [[0.17683995], [0.2874831]]
          ],
          [
            [[0.09339976], [0.57645476]],

            [[0.06616235], [0.8850273]],

            [[0.87009287], [0.20542204]]
          ]
        ],
        [fSize, fSize, inDepth, 1],
    );
    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilation);

    expect(result.shape).toEqual([1, 7, 7, 2]);
    expectArraysClose(await result.data(), [
      0.19526604, 0.5378273, 0.795022,   0.9384107,  1.0860794,  0.7942326,
      0.9764694,  1.3974442, 0.5930813,  0.9848901,  0.44526684, 1.275759,
      0.572345,   1.1784878, 0.27117175, 0.773588,   0.20055711, 0.71320784,
      0.73477566, 1.8867722, 0.64123434, 1.6549369,  0.55551285, 2.0385633,
      0.24740812, 1.233143,  0.08528192, 1.6214795,  1.062326,   1.3828603,
      1.4494176,  1.1022222, 2.2350664,  2.283423,   1.5940895,  1.8871424,
      1.6627852,  2.4903212, 1.0405337,  2.0754304,  1.1508893,  1.9568737,
      0.6148571,  1.1505995, 1.1105528,  1.3823687,  1.4342139,  2.9909487,
      1.0210396,  2.6467443, 1.0563798,  3.3963797,  0.42652097, 2.274134,
      0.51121074, 2.264094,  1.1009313,  1.6042703,  1.510688,   1.2317145,
      2.025515,   2.3658662, 1.6722159,  2.0787857,  1.3785586,  2.895031,
      1.2915218,  2.2051222, 1.0423074,  2.4303207,  0.27844793, 0.84346974,
      0.25781655, 1.1208354, 0.9447272,  2.0111258,  0.3689065,  1.9052455,
      0.79137695, 2.355344,  0.5429248,  1.5593178,  0.8248403,  1.9922242,
      0.77847,    1.5032601, 0.8622418,  0.84645665, 1.6850245,  2.2958806,
      1.6242284,  1.329045,  1.6652328,  2.480535,   1.2793491,  1.2951884,
      1.0667037,  1.5720158
    ]);
  });

  it('Tensor3D is allowed', async () => {
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const chMul = 3;
    const inDepth = 2;

    const x = tf.zeros<Rank.R3>([3, 3, inDepth]);
    const w = tf.zeros<Rank.R4>([fSize, fSize, inDepth, chMul]);
    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([3, 3, inDepth * chMul]);
  });

  it('Pass null for dilations, which defaults to [1, 1]', () => {
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const chMul = 3;
    const inDepth = 2;
    const dilations: [number, number] = null;

    const x = tf.zeros<Rank.R3>([3, 3, inDepth]);
    const w = tf.zeros<Rank.R4>([fSize, fSize, inDepth, chMul]);
    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', dilations);
    expect(result.shape).toEqual([3, 3, inDepth * chMul]);
  });

  it('TensorLike', async () => {
    const pad = 'valid';
    const stride = 1;

    const x = [[
      [[0.230664], [0.987388], [0.0685208]],
      [[0.419224], [0.887861], [0.731641]],
      [[0.0741907], [0.409265], [0.351377]]
    ]];
    const w = [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];

    const result = tf.depthwiseConv2d(x, w, stride, pad);

    const expected = [1.07022, 1.03167, 0.67041, 0.778863];
    expectArraysClose(await result.data(), expected);
  });
  it('TensorLike Chained', async () => {
    const pad = 'valid';
    const stride = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];

    const result = x.depthwiseConv2d(w, stride, pad);
    expect(result.shape).toEqual([1, 2, 2, 1]);

    const expected = [1.07022, 1.03167, 0.67041, 0.778863];
    expectArraysClose(await result.data(), expected);
  });

  it('throws when passed x as a non-tensor', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const fSize = 1;
    const pad = 'same';
    const stride = 2;
    const dataFormat = 'NHWC';
    const dilation = 2;

    const w = tf.tensor4d([3], [fSize, fSize, inputDepth, outputDepth]);

    const e = /Argument 'x' passed to 'depthwiseConv2d' must be a Tensor/;
    expect(
        () => tf.depthwiseConv2d(
            {} as tf.Tensor3D, w, stride, pad, dataFormat, dilation))
        .toThrowError(e);
  });

  it('throws when passed filter as a non-tensor', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const pad = 'same';
    const stride = 2;
    const dataFormat = 'NHWC';
    const dilation = 2;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);

    const e = /Argument 'filter' passed to 'depthwiseConv2d' must be a Tensor/;
    expect(
        () => tf.depthwiseConv2d(
            x, {} as tf.Tensor4D, stride, pad, dataFormat, dilation))
        .toThrowError(e);
  });

  it('throws when input is int32', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, inDepth], 'int32');
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );

    const errRegex = /Argument 'x' passed to 'depthwiseConv2d' must be float32/;
    expect(() => tf.depthwiseConv2d(x, w, stride, pad)).toThrowError(errRegex);
  });

  it('throws when filter is int32', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [1, 2, 3, 4],
        [fSize, fSize, inDepth, chMul],
        'int32',
    );

    const errRegex =
        /Argument 'filter' passed to 'depthwiseConv2d' must be float32/;
    expect(() => tf.depthwiseConv2d(x, w, stride, pad)).toThrowError(errRegex);
  });

  it('throws when dimRoundingMode is set and pad is same', () => {
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;
    const dimRoundingMode = 'round';

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );
    expect(
        () => tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', 1, dimRoundingMode))
        .toThrowError();
  });

  it('throws when dimRoundingMode is set and pad is valid', () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;
    const dimRoundingMode = 'round';

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );
    expect(
        () => tf.depthwiseConv2d(x, w, stride, pad, 'NHWC', 1, dimRoundingMode))
        .toThrowError();
  });

  it('throws when dimRoundingMode is set and pad is a non-integer number',
     () => {
       const fSize = 2;
       const pad = 1.2;
       const stride = 1;
       const chMul = 1;
       const inDepth = 1;
       const dimRoundingMode = 'round';

       const x = tf.tensor4d(
           [
             0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
             0.0741907, 0.409265, 0.351377
           ],
           [1, 3, 3, inDepth]);
       const w = tf.tensor4d(
           [0.303873, 0.229223, 0.144333, 0.803373],
           [fSize, fSize, inDepth, chMul],
       );
       expect(
           () => tf.depthwiseConv2d(
               x, w, stride, pad, 'NHWC', 1, dimRoundingMode))
           .toThrowError();
     });

  it('throws when dimRoundingMode is set and pad is explicit by non-integer ' +
         'number',
     () => {
       const fSize = 2;
       const pad = [[0, 0], [0, 2.1], [1, 1], [0, 0]] as
           tf.backend_util.ExplicitPadding;
       const stride = 1;
       const chMul = 1;
       const inDepth = 1;
       const dimRoundingMode = 'round';

       const x = tf.tensor4d(
           [
             0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
             0.0741907, 0.409265, 0.351377
           ],
           [1, 3, 3, inDepth]);
       const w = tf.tensor4d(
           [0.303873, 0.229223, 0.144333, 0.803373],
           [fSize, fSize, inDepth, chMul],
       );
       expect(
           () => tf.depthwiseConv2d(
               x, w, stride, pad, 'NHWC', 1, dimRoundingMode))
           .toThrowError();
     });

  it('accepts a tensor-like object', async () => {
    const pad = 'valid';
    const stride = 1;
    // 1x3x3x1
    const x = [[
      [[0.230664], [0.987388], [0.0685208]],
      [[0.419224], [0.887861], [0.731641]],
      [[0.0741907], [0.409265], [0.351377]]
    ]];
    // 2x2x1x1
    const w = [[[[0.303873]], [[0.229223]]], [[[0.144333]], [[0.803373]]]];
    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 2, 2, 1]);

    const expected = [1.07022, 1.03167, 0.67041, 0.778863];
    expectArraysClose(await result.data(), expected);
  });
});

describeWithFlags('depthwiseConv2d gradients', ALL_ENVS, () => {
  let images: tf.Tensor4D;
  let filter: tf.Tensor4D;
  let result: tf.Tensor4D;
  const stride = 1;
  const pad = 'same';

  beforeEach(() => {
    // two 2x2 RGB images => 2x2x2x3
    images = tf.tensor4d([
      [[[2, 3, 1], [3, 0, 2]], [[0, 4, 1], [3, 1, 3]]],
      [[[2, 1, 0], [0, 3, 3]], [[4, 0, 1], [1, 4, 1]]]
    ]);
    // 2x2 filters, chMul = 2 => 2x2x3x2
    filter = tf.tensor4d([
      [[[1, 1], [1, 1], [0, 0]], [[0, 1], [1, 1], [1, 1]]],
      [[[1, 0], [1, 1], [0, 0]], [[0, 1], [1, 0], [0, 0]]]
    ]);
    // result of convolution operatoin
    result = tf.tensor4d([
      [
        [[2, 8, 8, 7, 2, 2], [6, 3, 1, 1, 0, 0]],
        [[0, 3, 5, 5, 3, 3], [3, 3, 1, 1, 0, 0]]
      ],
      [
        [[6, 3, 8, 4, 3, 3], [1, 0, 7, 7, 0, 0]],
        [[4, 5, 4, 4, 1, 1], [1, 1, 4, 4, 0, 0]]
      ]
    ]);
  });

  it('wrt input', async () => {
    const {value, grad} = tf.valueAndGrad(
        (x: tf.Tensor4D) => tf.depthwiseConv2d(x, filter, stride, pad))(images);

    expectArraysClose(await value.data(), await result.data());

    const expectedGrad = tf.tensor4d([
      [[[2., 2., 0.], [3., 4., 2.]], [[3., 4., 0.], [5., 7., 2.]]],
      [[[2., 2., 0.], [3., 4., 2.]], [[3., 4., 0.], [5., 7., 2.]]]
    ]);

    expectArraysClose(await grad.data(), await expectedGrad.data());
  });

  // The gradients of normal and depthwise 2D convolutions are actually the same
  // in the special case that dy = 1, so we also test the gradient of a function
  // of the output to disambiguate the two methods.
  it('wrt input, squared output', async () => {
    const grad = tf.grad(
        (x: tf.Tensor4D) =>
            tf.square(tf.depthwiseConv2d(x, filter, stride, pad)))(images);

    const expectedGrad = tf.tensor4d([
      [[[20., 30., 0.], [34., 34., 8.]], [[10., 50., 0.], [46., 44., 12.]]],
      [[[18., 24., 0.], [8., 52., 12.]], [[30., 40., 0.], [22., 76., 4.]]]
    ]);

    expectArraysClose(await grad.data(), await expectedGrad.data());
  });

  it('wrt filter', async () => {
    const {value, grad} = tf.valueAndGrad(
        (f: tf.Tensor4D) => tf.depthwiseConv2d(images, f, stride, pad))(filter);

    expectArraysClose(await value.data(), await result.data());

    const expectedGrad = tf.tensor4d([
      [[[15., 15.], [16., 16.], [12., 12.]], [[7., 7.], [8., 8.], [9., 9.]]],
      [[[8., 8.], [9., 9.], [6., 6.]], [[4., 4.], [5., 5.], [4., 4.]]]
    ]);

    expectArraysClose(await grad.data(), await expectedGrad.data());
  });

  it('gradient with clones', async () => {
    const [dx, dFilter] = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            tf.depthwiseConv2d(x.clone(), filter.clone(), stride, pad).clone())(
        [images, filter]);
    expect(dx.shape).toEqual(images.shape);
    expect(dFilter.shape).toEqual(filter.shape);
  });

  // Also disambiguate regular vs. depthwise filter gradients
  it('wrt filter, squared output', async () => {
    const grad = tf.grad(
        (f: tf.Tensor4D) =>
            tf.square(tf.depthwiseConv2d(images, f, stride, pad)))(filter);

    const expectedGrad = tf.tensor4d([
      [
        [[120., 122.], [180., 166.], [12., 12.]],
        [[20., 76.], [90., 66.], [46., 46.]]
      ],
      [
        [[86., 42.], [122., 114.], [10., 10.]],
        [[24., 54.], [80., 46.], [18., 18.]]
      ]
    ]);

    expectArraysClose(await grad.data(), await expectedGrad.data());
  });

  it('throws error on dilations > 1', () => {
    const grad = tf.grad(
        (x: tf.Tensor4D) =>
            tf.depthwiseConv2d(x, filter, stride, pad, 'NHWC', 2));

    expect(() => grad(images))
        .toThrowError(/dilation rates greater than 1 are not yet supported/);
  });

  it('wrt input, stride=2, pad=valid', async () => {
    const dx = tf.grad(
        (x: tf.Tensor4D) => tf.depthwiseConv2d(x, filter, 2, 'valid'))(images);

    expectArraysClose(await dx.data(), [
      2., 2., 0., 1., 2., 2., 1., 2., 0., 1., 1., 0.,
      2., 2., 0., 1., 2., 2., 1., 2., 0., 1., 1., 0.
    ]);
    expect(dx.shape).toEqual([2, 2, 2, 3]);
  });

  it('wrt filter, stride=2, pad=valid', async () => {
    const df = tf.grad(
        (f: tf.Tensor4D) => tf.depthwiseConv2d(images, f, 2, 'valid'))(filter);

    expectArraysClose(await df.data(), [
      4., 4., 4., 4., 1., 1., 3., 3., 3., 3., 5., 5.,
      4., 4., 4., 4., 2., 2., 4., 4., 5., 5., 4., 4.
    ]);
    expect(df.shape).toEqual([2, 2, 3, 2]);
  });

  it('gradient with clones', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);

    const f = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );

    const [dx, df] = tf.grads(
        (x: tf.Tensor4D, f: tf.Tensor4D) =>
            tf.depthwiseConv2d(x.clone(), f.clone(), stride, pad).clone())(
        [x, f]);

    expectArraysClose(await dx.data(), [
      0.303873, 0.533096, 0.229223, 0.448206, 1.480802, 1.032596, 0.144333,
      0.947706, 0.803373
    ]);
    expect(dx.shape).toEqual([1, 3, 3, 1]);

    expectArraysClose(
        await df.data(), [2.525137, 2.6754108, 1.7905407, 2.380144]);
    expect(df.shape).toEqual([2, 2, 1, 1]);
  });
});
