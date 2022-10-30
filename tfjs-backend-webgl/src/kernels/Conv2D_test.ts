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
    // const webGLBackend = tf.backend() as MathBackendWebGL;
    // const resInfo = webGLBackend.getDataInfo(result);
    const resultData = await result.data();

    const expected = [
      98,   104,  110,  116,  122,  128,  134,  266,  288,  310,
      332,  354,  376,  398,  434,  472,  510,  548,  586,  624,
      662,  602,  656,  710,  764,  818,  872,  926,  770,  840,
      910,  980,  1050, 1120, 1190, 938,  1024, 1110, 1196, 1282,
      1368, 1454, 1106, 1208, 1310, 1412, 1514, 1616, 1718
    ];

    // expectArraysClose(resInfo.mrtStorage, [2, 2]);
    // expectArraysClose(resInfo.texture.texShape, [2, 2]);
    expectArraysClose(result.shape, [1, 1, 7, 7]);
    expectArraysClose(resultData, expected);
  });

  it('only one target is used', async () => {
    const image = tf.tensor4d([1, 2, 3, 4], [1, 1, 2, 2]);
    const filter = tf.tensor4d([1, 0.1, 0.01, 0.001], [1, 1, 2, 2]);
    const result = tf.conv2d(image, filter, 1, 'valid');
    // const webGLBackend = tf.backend() as MathBackendWebGL;
    // const resInfo = webGLBackend.getDataInfo(result);
    const resultData = await result.data();

    const expected = [
      1.0199999809265137, 0.10199999809265137, 3.0399999618530273,
      0.30400002002716064
    ];

    // expectArraysClose(resInfo.mrtStorage, [2, 2]);
    // expectArraysClose(resInfo.texture.texShape, [2, 2]);
    expectArraysClose(result.shape, [1, 1, 2, 2]);
    expectArraysClose(resultData, expected);
  });

  it('edge case', async () => {
    const image = tf.tensor4d([2], [1, 1, 1, 1]);
    const filter = tf.tensor4d([3], [1, 1, 1, 1]);
    const result = tf.conv2d(image, filter, 1, 'valid');
    // const webGLBackend = tf.backend() as MathBackendWebGL;
    // const resInfo = webGLBackend.getDataInfo(result);
    const resultData = await result.data();

    const expected = [6];

    // expectArraysClose(resInfo.mrtStorage, [2, 2]);
    // expectArraysClose(resInfo.texture.texShape, [2, 2]);
    expectArraysClose(result.shape, [1, 1, 1, 1]);
    expectArraysClose(resultData, expected);
  });
});
