/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

const {expectArraysClose} = test_util;

describeWithFlags('Conv2D WebGL Implementation ', ALL_ENVS, () => {
  it('should work when width is odd and called multiple times.', async () => {
    const filter = tf.tensor4d([-1, 3, 2, 1, 3, 4, 4, -2], [1, 1, 4, 2]);
    const image = tf.tensor3d(
        [
          111, 112, 113, 114, 121, 122, 123, 124, 131, 132, 133, 134,
          211, 212, 213, 214, 221, 222, 223, 224, 231, 232, 233, 234,
          311, 312, 313, 314, 321, 322, 323, 324, 331, 332, 333, 334,

        ],
        [3, 3, 4]);

    tf.conv2d(image, filter, 1, 'valid');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [
      908, 669, 988, 729, 1068, 789, 1708, 1269, 1788, 1329, 1868, 1389, 2508,
      1869, 2588, 1929, 2668, 1989
    ];

    expectArraysClose(resultData, expected);
  });

  it('image is packed and isChannelFirst.', async () => {
    const filter = tf.tensor4d([1], [1, 1, 1, 1]);
    const image = tf.tensor3d([11, 12, 13, 21, 22, 23, 31, 32, 33], [1, 3, 3]);

    // pack image.
    tf.mul(image, 1);

    tf.conv2d(image, filter, 1, 'valid', 'NCHW');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid', 'NCHW');
    const resultData = await result.data();

    const expected = [11, 12, 13, 21, 22, 23, 31, 32, 33];

    expectArraysClose(resultData, expected);
  });

  it('image is unpacked and isChannelFirst.', async () => {
    const filter = tf.tensor4d([1], [1, 1, 1, 1]);
    const image = tf.tensor3d([11, 12, 13, 21, 22, 23, 31, 32, 33], [1, 3, 3]);

    tf.conv2d(image, filter, 1, 'valid', 'NCHW');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid', 'NCHW');
    const resultData = await result.data();

    const expected = [11, 12, 13, 21, 22, 23, 31, 32, 33];

    expectArraysClose(resultData, expected);
  });

  it('image is packed and isChannelLast.', async () => {
    const filter = tf.tensor4d([1], [1, 1, 1, 1]);
    const image = tf.tensor3d([11, 12, 13, 21, 22, 23, 31, 32, 33], [3, 3, 1]);

    // pack image.
    tf.mul(image, 1);

    tf.conv2d(image, filter, 1, 'valid');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [11, 12, 13, 21, 22, 23, 31, 32, 33];

    expectArraysClose(resultData, expected);
  });

  it('image is unpacked and isChannelLast.', async () => {
    const filter = tf.tensor4d([1], [1, 1, 1, 1]);
    const image = tf.tensor3d([11, 12, 13, 21, 22, 23, 31, 32, 33], [3, 3, 1]);

    tf.conv2d(image, filter, 1, 'valid');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [11, 12, 13, 21, 22, 23, 31, 32, 33];

    expectArraysClose(resultData, expected);
  });
});

describeWithFlags('Matmul MRT works ', ALL_ENVS, () => {
  function makeContinuousArr(size: number) {
    return Array.from(Array(size).keys());
  }

  it('basis', async () => {
    const image = tf.tensor4d(makeContinuousArr(32), [1, 2, 4, 4]);
    const filter = tf.tensor4d(makeContinuousArr(32), [1, 1, 4, 8]);
    const result = tf.conv2d(image, filter, 1, 'valid');
    // const webGLBackend = tf.backend() as MathBackendWebGL;
    // const resInfo = webGLBackend.getDataInfo(result);
    const resultData = await result.data();

    const expected = [
      112,  118,  124,  130,  136,  142,  148,  154,  304,  326,  348,
      370,  392,  414,  436,  458,  496,  534,  572,  610,  648,  686,
      724,  762,  688,  742,  796,  850,  904,  958,  1012, 1066, 880,
      950,  1020, 1090, 1160, 1230, 1300, 1370, 1072, 1158, 1244, 1330,
      1416, 1502, 1588, 1674, 1264, 1366, 1468, 1570, 1672, 1774, 1876,
      1978, 1456, 1574, 1692, 1810, 1928, 2046, 2164, 2282
    ];

    // expectArraysClose(resInfo.mrtStorage, [2, 2]);
    // expectArraysClose(resInfo.texture.texShape, [2, 2]);
    expectArraysClose(resultData, expected);
  });


  it('output has paddings', async () => {
    const image = tf.tensor4d(makeContinuousArr(28), [1, 1, 7, 4]);
    const filter = tf.tensor4d(makeContinuousArr(28), [1, 1, 4, 7]);
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [
      98,   104,  110,  116,  122,  128,  134,  266,  288,  310,
      332,  354,  376,  398,  434,  472,  510,  548,  586,  624,
      662,  602,  656,  710,  764,  818,  872,  926,  770,  840,
      910,  980,  1050, 1120, 1190, 938,  1024, 1110, 1196, 1282,
      1368, 1454, 1106, 1208, 1310, 1412, 1514, 1616, 1718
    ];

    expectArraysClose(result.shape, [1, 1, 7, 7]);
    expectArraysClose(resultData, expected);
  });

  it('only one target is used', async () => {
    const image = tf.tensor4d([1, 2, 3, 4], [1, 1, 2, 2]);
    const filter = tf.tensor4d([1, 0.1, 0.01, 0.001], [1, 1, 2, 2]);
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [
      1.0199999809265137, 0.10199999809265137, 3.0399999618530273,
      0.30400002002716064
    ];

    expectArraysClose(result.shape, [1, 1, 2, 2]);
    expectArraysClose(resultData, expected);
  });

  it('edge case', async () => {
    const image = tf.tensor4d([2], [1, 1, 1, 1]);
    const filter = tf.tensor4d([3], [1, 1, 1, 1]);
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = await result.data();

    const expected = [6];

    expectArraysClose(result.shape, [1, 1, 1, 1]);
    expectArraysClose(resultData, expected);
  });
});

describeWithFlags('Fused Matmul MRT works ', ALL_ENVS, () => {
  function makeContinuousArr(size: number) {
    return Array.from(Array(size).keys());
  }

  it('Add bias', async () => {
    const image = tf.tensor4d(makeContinuousArr(32), [1, 2, 4, 4]);
    const filter = tf.tensor4d(makeContinuousArr(32), [1, 1, 4, 8]);
    const bias = tf.tensor1d(makeContinuousArr(8));
    const result =
        tf.fused.conv2d({x: image, filter, strides: 1, pad: 'valid', bias});
    const resultData = await result.data();

    const expected = [
      112,  119,  126,  133,  140,  147,  154,  161,  304,  327,  350,
      373,  396,  419,  442,  465,  496,  535,  574,  613,  652,  691,
      730,  769,  688,  743,  798,  853,  908,  963,  1018, 1073, 880,
      951,  1022, 1093, 1164, 1235, 1306, 1377, 1072, 1159, 1246, 1333,
      1420, 1507, 1594, 1681, 1264, 1367, 1470, 1573, 1676, 1779, 1882,
      1985, 1456, 1575, 1694, 1813, 1932, 2051, 2170, 2289
    ];

    expectArraysClose(resultData, expected);
  });
});

describeWithFlags('MRT pipeline works ', ALL_ENVS, () => {
  function makeContinuousArr(size: number) {
    return Array.from(Array(size).keys());
  }

  it('Fused conv + Add ', async () => {
    const image = tf.tensor4d(makeContinuousArr(32), [1, 2, 4, 4]);
    const filter = tf.tensor4d(makeContinuousArr(32), [1, 1, 4, 8]);
    const bias = tf.tensor1d(makeContinuousArr(8));
    const convRes =
        tf.fused.conv2d({x: image, filter, strides: 1, pad: 'valid', bias});
    const t = tf.ones([1, 2, 4, 8]);
    const result = tf.add(convRes, t);
    const resultData = await result.data();

    debugger;
    const expected = [
      113,  120,  127,  134,  141,  148,  155,  162,  305,  328,  351,
      374,  397,  420,  443,  466,  497,  536,  575,  614,  653,  692,
      731,  770,  689,  744,  799,  854,  909,  964,  1019, 1074, 881,
      952,  1023, 1094, 1165, 1236, 1307, 1378, 1073, 1160, 1247, 1334,
      1421, 1508, 1595, 1682, 1265, 1368, 1471, 1574, 1677, 1780, 1883,
      1986, 1457, 1576, 1695, 1814, 1933, 2052, 2171, 2290
    ];

    expectArraysClose(resultData, expected);
  });

  it('Fused conv + Depthwise ', async () => {
    const image = tf.tensor4d(makeContinuousArr(32), [1, 2, 4, 4]);
    const filter = tf.tensor4d(makeContinuousArr(32), [1, 1, 4, 8]);
    const bias = tf.tensor1d(makeContinuousArr(8));
    const convRes =
        tf.fused.conv2d({x: image, filter, strides: 1, pad: 'valid', bias});
    const deothFilter = tf.ones([2, 2, 8, 1]) as tf.Tensor4D;
    const result = tf.depthwiseConv2d(convRes, deothFilter, 1, 'valid');
    const resultData = await result.data();

    const expected = [
      2368, 2556, 2744, 2932, 3120, 3308, 3496, 3684, 3136, 3388, 3640, 3892,
      4144, 4396, 4648, 4900, 3904, 4220, 4536, 4852, 5168, 5484, 5800, 6116
    ];

    expectArraysClose(resultData, expected);
  });
});
