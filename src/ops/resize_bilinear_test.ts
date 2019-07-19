/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

describeWithFlags('resizeBilinear', ALL_ENVS, () => {
  it('simple alignCorners=false', async () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    const output = input.resizeBilinear([3, 3], false);

    expectArraysClose(
        await output.data(), [2, 2, 2, 10 / 3, 10 / 3, 10 / 3, 4, 4, 4]);
  });

  it('simple alignCorners=true', async () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    const output = input.resizeBilinear([3, 3], true);

    expectArraysClose(await output.data(), [2, 2, 2, 3, 3, 3, 4, 4, 4]);
  });

  it('matches tensorflow w/ random numbers alignCorners=false', async () => {
    const input = tf.tensor3d(
        [
          1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395,
          1.30809779, 0.61835143, 3.49600659, 2.09230986, 0.56473997,
          0.03823943, 1.19864896
        ],
        [2, 3, 2]);
    const output = input.resizeBilinear([4, 5], false);

    expectArraysClose(await output.data(), [
      1.19074047, 0.91373104, 1.68596613, 0.05186744, 1.69034398, -0.15654698,
      0.7130264,  0.94193673, 0.38725394, 1.30809784, 0.9045459,  2.20486879,
      1.59434628, 0.89455694, 1.68591988, 0.26748738, 0.58103991, 1.00690198,
      0.21274668, 1.25337338, 0.6183514,  3.49600649, 1.50272655, 1.73724651,
      1.68149579, 0.69152176, 0.44905344, 1.07186723, 0.03823943, 1.19864893,
      0.6183514,  3.49600649, 1.50272655, 1.73724651, 1.68149579, 0.69152176,
      0.44905344, 1.07186723, 0.03823943, 1.19864893
    ]);
  });

  it('matches tensorflow w/ random numbers alignCorners=true', async () => {
    const input = tf.tensor3d(
        [
          1.56324531, 2.13817752, 1.44398421, 1.07632684, 0.59306785,
          -0.36970865, 1.62451879, 1.8367334, 1.13944798, 2.01993218,
          2.01919952, 2.67524054
        ],
        [2, 3, 2]);
    const output = input.resizeBilinear([4, 5], true);

    expectArraysClose(await output.data(), [
      1.5632453,  2.13817763, 1.50361478, 1.60725224,  1.44398427, 1.07632685,
      1.01852608, 0.35330909, 0.59306782, -0.36970866, 1.58366978, 2.03769612,
      1.46307099, 1.71427906, 1.3424722,  1.39086199,  1.20545864, 1.01806819,
      1.06844509, 0.6452744,  1.60409427, 1.93721485,  1.42252707, 1.82130599,
      1.24096,    1.70539713, 1.3923912,  1.68282723,  1.54382229, 1.66025746,
      1.62451875, 1.83673346, 1.38198328, 1.92833281,  1.13944793, 2.01993227,
      1.57932377, 2.34758639, 2.01919961, 2.67524052
    ]);
  });

  it('batch of 2, simple, alignCorners=true', async () => {
    const input = tf.tensor4d([2, 2, 4, 4, 3, 3, 5, 5], [2, 2, 2, 1]);
    const output = input.resizeBilinear([3, 3], true /* alignCorners */);

    expectArraysClose(
        await output.data(),
        [2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 5, 5, 5]);
  });

  it('target width = 1, alignCorners=true', async () => {
    const input = tf.tensor3d([
      [
        [120.68856811523438, 134.51638793945312, 83.03671264648438],
        [121.58008575439453, 113.28836059570312, 136.3172149658203],
        [79.38370513916016, 101.87127685546875, 104.54979705810547],
        [96.31678771972656, 111.77168273925781, 83.73509979248047]
      ],
      [
        [119.45088195800781, 88.98846435546875, 97.47553253173828],
        [117.5562973022461, 108.26356506347656, 99.62212371826172],
        [136.62701416015625, 94.10433197021484, 80.97366333007812],
        [83.61205291748047, 90.60148620605469, 81.82512664794922]
      ],
      [
        [103.0362777709961, 123.1098403930664, 125.62944030761719],
        [92.2915267944336, 103.15729522705078, 119.18060302734375],
        [102.93293762207031, 117.821044921875, 99.40152740478516],
        [96.32952117919922, 105.80963134765625, 104.8491439819336]
      ],
      [
        [104.87507629394531, 134.0189208984375, 111.02627563476562],
        [85.4534683227539, 107.68426513671875, 103.03722381591797],
        [89.70533752441406, 98.25298309326172, 78.42916870117188],
        [113.6744613647461, 95.8189697265625, 122.75005340576172]
      ]
    ]);

    const output = input.resizeBilinear([3, 1], true);

    const expected = [
      120.68857, 134.51639, 83.03671, 111.243576, 106.04915, 111.55249,
      104.87508, 134.01892, 111.026276
    ];
    expectArraysClose(await output.data(), expected);
    expect(output.shape).toEqual([3, 1, 3]);
  });

  it('target height = 1, alignCorners=true', async () => {
    const input = tf.tensor3d([
      [
        [120.68856811523438, 134.51638793945312, 83.03671264648438],
        [121.58008575439453, 113.28836059570312, 136.3172149658203],
        [79.38370513916016, 101.87127685546875, 104.54979705810547],
        [96.31678771972656, 111.77168273925781, 83.73509979248047]
      ],
      [
        [119.45088195800781, 88.98846435546875, 97.47553253173828],
        [117.5562973022461, 108.26356506347656, 99.62212371826172],
        [136.62701416015625, 94.10433197021484, 80.97366333007812],
        [83.61205291748047, 90.60148620605469, 81.82512664794922]
      ],
      [
        [103.0362777709961, 123.1098403930664, 125.62944030761719],
        [92.2915267944336, 103.15729522705078, 119.18060302734375],
        [102.93293762207031, 117.821044921875, 99.40152740478516],
        [96.32952117919922, 105.80963134765625, 104.8491439819336]
      ],
      [
        [104.87507629394531, 134.0189208984375, 111.02627563476562],
        [85.4534683227539, 107.68426513671875, 103.03722381591797],
        [89.70533752441406, 98.25298309326172, 78.42916870117188],
        [113.6744613647461, 95.8189697265625, 122.75005340576172]
      ]
    ]);

    const output = input.resizeBilinear([1, 3], true);

    const expected = [
      120.68857, 134.51639, 83.03671, 100.481895, 107.57982, 120.4335, 96.31679,
      111.77168, 83.7351
    ];
    expectArraysClose(await output.data(), expected);
    expect(output.shape).toEqual([1, 3, 3]);
  });

  it('throws when passed a non-tensor', () => {
    const e = /Argument 'images' passed to 'resizeBilinear' must be a Tensor/;
    expect(() => tf.image.resizeBilinear({} as tf.Tensor3D, [
      1, 1
    ])).toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const input = [[[2], [2]], [[4], [4]]];  // 2x2x1
    const output = tf.image.resizeBilinear(input, [3, 3], false);
    expectArraysClose(
        await output.data(), [2, 2, 2, 10 / 3, 10 / 3, 10 / 3, 4, 4, 4]);
  });
});

describeWithFlags('resizeBilinear gradients', ALL_ENVS, () => {
  it('greyscale: upscale, same aspect ratio', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([
      [[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]],
      [[9.0], [10.0], [11.0], [12.0]], [[13.0], [14.0], [15.0], [16.0]]
    ]);

    const size: [number, number] = [4, 4];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [6.0, 17.0, 38.0, 75.0];

    expectArraysClose(await output.data(), expected);
  });

  it('with clones, greyscale: upscale, same aspect ratio', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([
      [[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]],
      [[9.0], [10.0], [11.0], [12.0]], [[13.0], [14.0], [15.0], [16.0]]
    ]);

    const size: [number, number] = [4, 4];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeBilinear(i.clone(), size, alignCorners).clone());

    const output = g(input, dy);
    const expected = [6.0, 17.0, 38.0, 75.0];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: upscale, same aspect ratio, align corners', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([
      [[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]],
      [[9.0], [10.0], [11.0], [12.0]], [[13.0], [14.0], [15.0], [16.0]]
    ]);

    const size: [number, number] = [4, 4];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected =
        [17.333330154418945, 23.999998092651367, 44.0, 50.66666793823242];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: upscale, taller than wider', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([
      [[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]],
      [[9.0], [10.0], [11.0], [12.0]], [[13.0], [14.0], [15.0], [16.0]],
      [[17.0], [18.0], [19.0], [20.0]], [[21.0], [22.0], [23.0], [24.0]],
      [[25.0], [26.0], [27.0], [28.0]], [[29.0], [30.0], [31.0], [32.0]],
      [[33.0], [34.0], [35.0], [36.0]]
    ]);

    const size: [number, number] = [9, 4];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      25.55555534362793, 55.5555534362793, 208.44444274902344, 376.4444274902344
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: upscale, taller than wider, align corners', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([
      [[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]],
      [[9.0], [10.0], [11.0], [12.0]], [[13.0], [14.0], [15.0], [16.0]],
      [[17.0], [18.0], [19.0], [20.0]], [[21.0], [22.0], [23.0], [24.0]],
      [[25.0], [26.0], [27.0], [28.0]], [[29.0], [30.0], [31.0], [32.0]],
      [[33.0], [34.0], [35.0], [36.0]]
    ]);

    const size: [number, number] = [9, 4];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [99.0, 114.0, 219.00001525878906, 233.99998474121094];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: upscale, wider than taller', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([
      [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]],
      [[8.0], [9.0], [10.0], [11.0], [12.0], [13.0], [14.0]],
      [[15.0], [16.0], [17.0], [18.0], [19.0], [20.0], [21.0]],
      [[22.0], [23.0], [24.0], [25.0], [26.0], [27.0], [28.0]]
    ]);

    const size: [number, number] = [4, 7];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      14.428570747375488, 52.07142639160156, 98.71427917480469,
      240.78573608398438
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: upscale, wider than taller, align corners', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([
      [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]],
      [[8.0], [9.0], [10.0], [11.0], [12.0], [13.0], [14.0]],
      [[15.0], [16.0], [17.0], [18.0], [19.0], [20.0], [21.0]],
      [[22.0], [23.0], [24.0], [25.0], [26.0], [27.0], [28.0]]
    ]);

    const size: [number, number] = [4, 7];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [51.33332824707031, 70.0, 133.0, 151.66668701171875];

    expectArraysClose(await output.data(), expected);
  });

  // Downscale

  it('greyscale: downscale, same aspect ratio', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);

    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    const size: [number, number] = [2, 2];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
      0.0
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: downscale, same aspect ratio, align corners', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);

    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    const size: [number, number] = [2, 2];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
      4.0
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: downscale, taller than wider', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);

    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]);

    const size: [number, number] = [3, 2];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0, 0.0, 2.0, 0.0, 1.9999998807907104, 0.0, 2.6666665077209473, 0.0,
      2.6666665077209473, 0.0, 3.3333330154418945, 0.0, 3.333333730697632, 0.0,
      4.000000476837158, 0.0
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: downscale, taller than wider, align corners', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);

    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]);

    const size: [number, number] = [3, 2];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0, 0.0, 0.0, 2.0, 1.5, 0.0, 0.0, 2.0, 1.5, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0,
      6.0
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: downscale, wider than taller', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);

    const dy = tf.tensor3d([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]);

    const size: [number, number] = [2, 3];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0, 1.3333332538604736, 1.6666665077209473, 2.000000238418579, 0.0, 0.0,
      0.0, 0.0, 4.0, 3.3333330154418945, 3.6666665077209473, 4.000000476837158,
      0.0, 0.0, 0.0, 0.0
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: downscale, wider than taller, align corners', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);

    const dy = tf.tensor3d([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]);

    const size: [number, number] = [2, 3];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.5, 2.5,
      6.0
    ];

    expectArraysClose(await output.data(), expected);
  });

  // No Op

  it('greyscale: same size', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);

    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    const size: [number, number] = [2, 2];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [1.0, 2.0, 3.0, 4.0];

    expectArraysClose(await output.data(), expected);
  });

  it('greyscale: same size, align corners', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);

    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    const size: [number, number] = [2, 2];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [1.0, 2.0, 3.0, 4.0];

    expectArraysClose(await output.data(), expected);
  });

  // 3 channel upscale
  it('color: upscale, wider than taller', async () => {
    const input = tf.tensor3d([
      [
        [115.11029815673828, 111.90936279296875, 66.87433624267578],
        [72.03849029541016, 81.86637878417969, 119.53585815429688]
      ],
      [
        [68.555419921875, 97.49642181396484, 116.90741729736328],
        [128.69467163085938, 86.78314208984375, 104.3116683959961]
      ]
    ]);

    const dy = tf.tensor3d([
      [
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
      ],
      [
        [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0],
        [25.0, 26.0, 27.0], [28.0, 29.0, 30.0]
      ],
      [
        [31.0, 32.0, 33.0], [34.0, 35.0, 36.0], [37.0, 38.0, 39.0],
        [40.0, 41.0, 42.0], [43.0, 44.0, 45.0]
      ]
    ]);

    const size: [number, number] = [3, 5];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      15.399999618530273, 17.799999237060547, 20.19999885559082,
      56.26666259765625, 60.533329010009766, 64.79999542236328,
      80.00000762939453, 83.0, 86.0, 178.33334350585938, 183.66668701171875,
      189.00001525878906
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('color: upscale, wider than taller, align corners', async () => {
    const input = tf.tensor3d([
      [
        [115.11029815673828, 111.90936279296875, 66.87433624267578],
        [72.03849029541016, 81.86637878417969, 119.53585815429688]
      ],
      [
        [68.555419921875, 97.49642181396484, 116.90741729736328],
        [128.69467163085938, 86.78314208984375, 104.3116683959961]
      ]
    ]);

    const dy = tf.tensor3d([
      [
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
      ],
      [
        [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0],
        [25.0, 26.0, 27.0], [28.0, 29.0, 30.0]
      ],
      [
        [31.0, 32.0, 33.0], [34.0, 35.0, 36.0], [37.0, 38.0, 39.0],
        [40.0, 41.0, 42.0], [43.0, 44.0, 45.0]
      ]
    ]);

    const size: [number, number] = [3, 5];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      33.75, 37.5, 41.25, 56.25, 60.0, 63.75, 108.75, 112.5, 116.25, 131.25,
      135.0, 138.75
    ];

    expectArraysClose(await output.data(), expected);
  });

  // 3 channel downscale

  it('color: downscale, taller than wider', async () => {
    const input = tf.tensor3d([
      [
        [120.68856811523438, 134.51638793945312, 83.03671264648438],
        [121.58008575439453, 113.28836059570312, 136.3172149658203],
        [79.38370513916016, 101.87127685546875, 104.54979705810547],
        [96.31678771972656, 111.77168273925781, 83.73509979248047]
      ],
      [
        [119.45088195800781, 88.98846435546875, 97.47553253173828],
        [117.5562973022461, 108.26356506347656, 99.62212371826172],
        [136.62701416015625, 94.10433197021484, 80.97366333007812],
        [83.61205291748047, 90.60148620605469, 81.82512664794922]
      ],
      [
        [103.0362777709961, 123.1098403930664, 125.62944030761719],
        [92.2915267944336, 103.15729522705078, 119.18060302734375],
        [102.93293762207031, 117.821044921875, 99.40152740478516],
        [96.32952117919922, 105.80963134765625, 104.8491439819336]
      ],
      [
        [104.87507629394531, 134.0189208984375, 111.02627563476562],
        [85.4534683227539, 107.68426513671875, 103.03722381591797],
        [89.70533752441406, 98.25298309326172, 78.42916870117188],
        [113.6744613647461, 95.8189697265625, 122.75005340576172]
      ]
    ]);

    const dy =
        tf.tensor3d([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]);

    const size: [number, number] = [3, 1];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0,
      2.0,
      3.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.6666665077209473,
      3.3333330154418945,
      3.999999761581421,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.666666269302368,
      4.3333330154418945,
      4.999999523162842,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.6666669845581055,
      5.333333969116211,
      6.000000953674316,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ];

    expectArraysClose(await output.data(), expected);
  });

  it('color: downscale, width = 1, align corners', async () => {
    const input = tf.tensor3d([
      [
        [120.68856811523438, 134.51638793945312, 83.03671264648438],
        [121.58008575439453, 113.28836059570312, 136.3172149658203],
        [79.38370513916016, 101.87127685546875, 104.54979705810547],
        [96.31678771972656, 111.77168273925781, 83.73509979248047]
      ],
      [
        [119.45088195800781, 88.98846435546875, 97.47553253173828],
        [117.5562973022461, 108.26356506347656, 99.62212371826172],
        [136.62701416015625, 94.10433197021484, 80.97366333007812],
        [83.61205291748047, 90.60148620605469, 81.82512664794922]
      ],
      [
        [103.0362777709961, 123.1098403930664, 125.62944030761719],
        [92.2915267944336, 103.15729522705078, 119.18060302734375],
        [102.93293762207031, 117.821044921875, 99.40152740478516],
        [96.32952117919922, 105.80963134765625, 104.8491439819336]
      ],
      [
        [104.87507629394531, 134.0189208984375, 111.02627563476562],
        [85.4534683227539, 107.68426513671875, 103.03722381591797],
        [89.70533752441406, 98.25298309326172, 78.42916870117188],
        [113.6744613647461, 95.8189697265625, 122.75005340576172]
      ]
    ]);

    const dy =
        tf.tensor3d([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]);

    const size: [number, number] = [3, 1];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      2.0, 2.5, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      2.0, 2.5, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ];
    expectArraysClose(await output.data(), expected);
  });

  it('color: downscale, height = 1, align corners', async () => {
    const input = tf.tensor3d([
      [
        [120.68856811523438, 134.51638793945312, 83.03671264648438],
        [121.58008575439453, 113.28836059570312, 136.3172149658203],
        [79.38370513916016, 101.87127685546875, 104.54979705810547],
        [96.31678771972656, 111.77168273925781, 83.73509979248047]
      ],
      [
        [119.45088195800781, 88.98846435546875, 97.47553253173828],
        [117.5562973022461, 108.26356506347656, 99.62212371826172],
        [136.62701416015625, 94.10433197021484, 80.97366333007812],
        [83.61205291748047, 90.60148620605469, 81.82512664794922]
      ],
      [
        [103.0362777709961, 123.1098403930664, 125.62944030761719],
        [92.2915267944336, 103.15729522705078, 119.18060302734375],
        [102.93293762207031, 117.821044921875, 99.40152740478516],
        [96.32952117919922, 105.80963134765625, 104.8491439819336]
      ],
      [
        [104.87507629394531, 134.0189208984375, 111.02627563476562],
        [85.4534683227539, 107.68426513671875, 103.03722381591797],
        [89.70533752441406, 98.25298309326172, 78.42916870117188],
        [113.6744613647461, 95.8189697265625, 122.75005340576172]
      ]
    ]);

    const dy = tf.tensor3d([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]);

    const size: [number, number] = [1, 3];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1., 2., 3., 2., 2.5, 3., 2., 2.5, 3., 7., 8., 9., 0., 0., 0., 0.,
      0., 0., 0., 0., 0.,  0., 0., 0.,  0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0.,  0., 0., 0.,  0., 0., 0., 0., 0., 0., 0., 0.
    ];
    expectArraysClose(await output.data(), expected);
  });

  it('color: downscale, taller than wider, align corners', async () => {
    const input = tf.tensor3d([
      [
        [120.68856811523438, 134.51638793945312, 83.03671264648438],
        [121.58008575439453, 113.28836059570312, 136.3172149658203],
        [79.38370513916016, 101.87127685546875, 104.54979705810547],
        [96.31678771972656, 111.77168273925781, 83.73509979248047]
      ],
      [
        [119.45088195800781, 88.98846435546875, 97.47553253173828],
        [117.5562973022461, 108.26356506347656, 99.62212371826172],
        [136.62701416015625, 94.10433197021484, 80.97366333007812],
        [83.61205291748047, 90.60148620605469, 81.82512664794922]
      ],
      [
        [103.0362777709961, 123.1098403930664, 125.62944030761719],
        [92.2915267944336, 103.15729522705078, 119.18060302734375],
        [102.93293762207031, 117.821044921875, 99.40152740478516],
        [96.32952117919922, 105.80963134765625, 104.8491439819336]
      ],
      [
        [104.87507629394531, 134.0189208984375, 111.02627563476562],
        [85.4534683227539, 107.68426513671875, 103.03722381591797],
        [89.70533752441406, 98.25298309326172, 78.42916870117188],
        [113.6744613647461, 95.8189697265625, 122.75005340576172]
      ]
    ]);

    const dy = tf.tensor3d([
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
      [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
    ]);

    const size: [number, number] = [3, 2];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected = [
      1.0,  2.0,  3.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0,  5.0,  6.0,
      3.5,  4.0,  4.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0,  5.5,  6.0,
      3.5,  4.0,  4.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0,  5.5,  6.0,
      13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 17.0, 18.0
    ];
    expectArraysClose(await output.data(), expected);
  });

  // 3 channel no-op

  it('color: same size', async () => {
    const input = tf.tensor3d([
      [
        [115.11029815673828, 111.90936279296875, 66.87433624267578],
        [72.03849029541016, 81.86637878417969, 119.53585815429688]
      ],
      [
        [68.555419921875, 97.49642181396484, 116.90741729736328],
        [128.69467163085938, 86.78314208984375, 104.3116683959961]
      ]
    ]);

    const dy = tf.tensor3d([
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ]);

    const size: [number, number] = [2, 2];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected =
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    expectArraysClose(await output.data(), expected);
  });

  it('color: same size, align corners', async () => {
    const input = tf.tensor3d([
      [
        [115.11029815673828, 111.90936279296875, 66.87433624267578],
        [72.03849029541016, 81.86637878417969, 119.53585815429688]
      ],
      [
        [68.555419921875, 97.49642181396484, 116.90741729736328],
        [128.69467163085938, 86.78314208984375, 104.3116683959961]
      ]
    ]);

    const dy = tf.tensor3d([
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ]);

    const size: [number, number] = [2, 2];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) => tf.image.resizeBilinear(i, size, alignCorners));

    const output = g(input, dy);
    const expected =
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    expectArraysClose(await output.data(), expected);
  });
});
