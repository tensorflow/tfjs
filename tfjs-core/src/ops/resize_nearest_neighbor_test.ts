/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

describeWithFlags('resizeNearestNeighbor', ALL_ENVS, () => {
  it('simple alignCorners=false', async () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    const output = input.resizeNearestNeighbor([3, 3], false);

    expectArraysClose(await output.data(), [2, 2, 2, 2, 2, 2, 4, 4, 4]);
  });

  it('simple alignCorners=true', async () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    const output = input.resizeNearestNeighbor([3, 3], true);

    expectArraysClose(await output.data(), [2, 2, 2, 4, 4, 4, 4, 4, 4]);
  });

  it('matches tensorflow w/ random numbers alignCorners=false', async () => {
    const input = tf.tensor3d(
        [
          1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395,
          1.30809779, 0.61835143, 3.49600659, 2.09230986, 0.56473997,
          0.03823943, 1.19864896
        ],
        [2, 3, 2]);
    const output = input.resizeNearestNeighbor([4, 5], false);

    expectArraysClose(await output.data(), [
      1.19074047,   0.913731039, 1.19074047,   0.913731039,  2.01611662,
      -0.522708297, 2.01611662,  -0.522708297, 0.38725394,   1.30809784,
      1.19074047,   0.913731039, 1.19074047,   0.913731039,  2.01611662,
      -0.522708297, 2.01611662,  -0.522708297, 0.38725394,   1.30809784,
      0.61835146,   3.49600649,  0.61835146,   3.49600649,   2.09230995,
      0.564739943,  2.09230995,  0.564739943,  0.0382394306, 1.19864893,
      0.61835146,   3.49600649,  0.61835146,   3.49600649,   2.09230995,
      0.564739943,  2.09230995,  0.564739943,  0.0382394306, 1.19864893
    ]);
  });

  it('matches tensorflow w/ random numbers alignCorners=true', async () => {
    const input = tf.tensor3d(
        [
          1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395,
          1.30809779, 0.61835143, 3.49600659, 2.09230986, 0.56473997,
          0.03823943, 1.19864896
        ],
        [2, 3, 2]);
    const output = input.resizeNearestNeighbor([4, 5], true);

    expectArraysClose(await output.data(), [
      1.19074044, 0.91373104,  2.01611669, -0.52270832, 2.01611669, -0.52270832,
      0.38725395, 1.30809779,  0.38725395, 1.30809779,  1.19074044, 0.91373104,
      2.01611669, -0.52270832, 2.01611669, -0.52270832, 0.38725395, 1.30809779,
      0.38725395, 1.30809779,  0.61835143, 3.49600659,  2.09230986, 0.56473997,
      2.09230986, 0.56473997,  0.03823943, 1.19864896,  0.03823943, 1.19864896,
      0.61835143, 3.49600659,  2.09230986, 0.56473997,  2.09230986, 0.56473997,
      0.03823943, 1.19864896,  0.03823943, 1.19864896
    ]);
  });

  it('batch of 2, simple, alignCorners=true', async () => {
    const input = tf.tensor4d([2, 2, 4, 4, 3, 3, 5, 5], [2, 2, 2, 1]);
    const output = input.resizeNearestNeighbor([3, 3], true /* alignCorners */);

    expectArraysClose(
        await output.data(),
        [2, 2, 2, 4, 4, 4, 4, 4, 4, 3, 3, 3, 5, 5, 5, 5, 5, 5]);
  });

  it('throws when passed a non-tensor', () => {
    const e =
        /Argument 'images' passed to 'resizeNearestNeighbor' must be a Tensor/;
    expect(() => tf.image.resizeNearestNeighbor({} as tf.Tensor3D, [
      1, 1
    ])).toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const input = [[[2], [2]], [[4], [4]]];  // 2x2x1
    const output = tf.image.resizeNearestNeighbor(input, [3, 3], false);

    expectArraysClose(await output.data(), [2, 2, 2, 2, 2, 2, 4, 4, 4]);
  });

  it('does not throw when some output dim is 1 and alignCorners=true', () => {
    const input = tf.tensor3d([2, 2, 4, 4], [2, 2, 1]);
    expect(() => input.resizeNearestNeighbor([1, 3], true)).not.toThrow();
  });
});

describeWithFlags('resizeNearestNeighbor gradients', ALL_ENVS, () => {
  it('greyscale: upscale, same aspect ratio', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([
      [[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]],
      [[9.0], [10.0], [11.0], [12.0]], [[13.0], [14.0], [15.0], [16.0]]
    ]);

    const size: [number, number] = [4, 4];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([[[14.0], [22.0]], [[46.0], [54.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
            tf.image.resizeNearestNeighbor(i.clone(), size, alignCorners)
                .clone());

    const output = g(input, dy);
    const expected = tf.tensor3d([[[14.0], [22.0]], [[46.0], [54.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([[[14.0], [22.0]], [[46.0], [54.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([[[95.0], [115.0]], [[220.0], [236.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([[[60.0], [76.0]], [[255.0], [275.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([[[48.0], [57.0]], [[160.0], [141.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([[[33.0], [72.0]], [[117.0], [184.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  //
  // Downscaling
  //

  it('greyscale: downscale, same aspect ratio', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);
    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    const size: [number, number] = [2, 2];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0], [0.0], [2.0], [0.0]], [[0.0], [0.0], [0.0], [0.0]],
      [[3.0], [0.0], [4.0], [0.0]], [[0.0], [0.0], [0.0], [0.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0], [0.0], [0.0], [2.0]], [[0.0], [0.0], [0.0], [0.0]],
      [[0.0], [0.0], [0.0], [0.0]], [[3.0], [0.0], [0.0], [4.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0], [0.0], [2.0], [0.0]], [[3.0], [0.0], [4.0], [0.0]],
      [[5.0], [0.0], [6.0], [0.0]], [[0.0], [0.0], [0.0], [0.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0], [0.0], [0.0], [2.0]], [[0.0], [0.0], [0.0], [0.0]],
      [[3.0], [0.0], [0.0], [4.0]], [[5.0], [0.0], [0.0], [6.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  it('greyscale: downscale, taller than wider', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);
    const dy = tf.tensor3d([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]);

    const size: [number, number] = [2, 3];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0], [2.0], [3.0], [0.0]], [[0.0], [0.0], [0.0], [0.0]],
      [[4.0], [5.0], [6.0], [0.0]], [[0.0], [0.0], [0.0], [0.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  it('greyscale: downscale, taller than wider, align corners', async () => {
    const input = tf.tensor3d([
      [[100.0], [50.0], [25.0], [10.0]], [[60.0], [20.0], [80.0], [20.0]],
      [[40.0], [15.0], [200.0], [203.0]], [[40.0], [10.0], [230.0], [200.0]]
    ]);
    const dy = tf.tensor3d([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]);

    const size: [number, number] = [2, 3];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0], [0.0], [2.0], [3.0]], [[0.0], [0.0], [0.0], [0.0]],
      [[0.0], [0.0], [0.0], [0.0]], [[4.0], [0.0], [5.0], [6.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  it('greyscale: downscale, same size', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    const size: [number, number] = [2, 2];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  it('greyscale: downscale, same size, align corners', async () => {
    const input = tf.tensor3d([[[100.0], [50.0]], [[60.0], [20.0]]]);
    const dy = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    const size: [number, number] = [2, 2];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([[[1.0], [2.0]], [[3.0], [4.0]]]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  //
  // 3 channel images
  //

  it('color: upscale, wider than taller', async () => {
    const input = tf.tensor3d([
      [
        [100.26818084716797, 74.61857604980469, 81.62117767333984],
        [127.86964416503906, 85.0583267211914, 102.95439147949219]
      ],
      [
        [104.3798828125, 96.70733642578125, 92.60601043701172],
        [77.63021850585938, 68.55794525146484, 96.17212677001953]
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[69.0, 75.0, 81.0], [76.0, 80.0, 84.0]],
      [[102.0, 105.0, 108.0], [83.0, 85.0, 87.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  it('color: upscale, wider than taller, align corners', async () => {
    const input = tf.tensor3d([
      [
        [100.26818084716797, 74.61857604980469, 81.62117767333984],
        [127.86964416503906, 85.0583267211914, 102.95439147949219]
      ],
      [
        [104.3798828125, 96.70733642578125, 92.60601043701172],
        [77.63021850585938, 68.55794525146484, 96.17212677001953]
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
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[5.0, 7.0, 9.0], [30.0, 33.0, 36.0]],
      [[100.0, 104.0, 108.0], [195.0, 201.0, 207.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  it('color: downscale, taller than wider', async () => {
    const input = tf.tensor3d([
      [
        [97.98934936523438, 77.24969482421875, 113.70111846923828],
        [111.34081268310547, 113.15758514404297, 157.90521240234375],
        [105.77980041503906, 85.75989532470703, 69.62374114990234],
        [125.94231414794922, 73.11385345458984, 87.03099822998047]
      ],
      [
        [62.25117111206055, 90.23927307128906, 119.1966552734375],
        [93.55166625976562, 95.9106674194336, 115.56237030029297],
        [102.98121643066406, 98.1983413696289, 97.55982971191406],
        [86.47753143310547, 97.04051208496094, 121.50492095947266]
      ],
      [
        [92.4140853881836, 118.45619201660156, 108.0341796875],
        [126.43061065673828, 123.28077697753906, 121.03379821777344],
        [128.6694793701172, 98.47042846679688, 114.47464752197266],
        [93.31566619873047, 95.2713623046875, 102.51188659667969]
      ],
      [
        [101.55884552001953, 83.31947326660156, 119.08016204833984],
        [128.28546142578125, 92.56212615966797, 74.85054779052734],
        [88.9786148071289, 119.43685913085938, 73.06110382080078],
        [98.17908477783203, 105.54570007324219, 93.45832061767578]
      ]
    ]);

    const dy =
        tf.tensor3d([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]);

    const size: [number, number] = [3, 1];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[7.0, 8.0, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  it('color: downscale, taller than wider, align corners', async () => {
    const input = tf.tensor3d([
      [
        [97.98934936523438, 77.24969482421875, 113.70111846923828],
        [111.34081268310547, 113.15758514404297, 157.90521240234375],
        [105.77980041503906, 85.75989532470703, 69.62374114990234],
        [125.94231414794922, 73.11385345458984, 87.03099822998047]
      ],
      [
        [62.25117111206055, 90.23927307128906, 119.1966552734375],
        [93.55166625976562, 95.9106674194336, 115.56237030029297],
        [102.98121643066406, 98.1983413696289, 97.55982971191406],
        [86.47753143310547, 97.04051208496094, 121.50492095947266]
      ],
      [
        [92.4140853881836, 118.45619201660156, 108.0341796875],
        [126.43061065673828, 123.28077697753906, 121.03379821777344],
        [128.6694793701172, 98.47042846679688, 114.47464752197266],
        [93.31566619873047, 95.2713623046875, 102.51188659667969]
      ],
      [
        [101.55884552001953, 83.31947326660156, 119.08016204833984],
        [128.28546142578125, 92.56212615966797, 74.85054779052734],
        [88.9786148071289, 119.43685913085938, 73.06110382080078],
        [98.17908477783203, 105.54570007324219, 93.45832061767578]
      ]
    ]);

    const dy =
        tf.tensor3d([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]);

    const size: [number, number] = [3, 1];
    const alignCorners = true;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[7.0, 8.0, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });

  it('color: same size', async () => {
    const input = tf.tensor3d([
      [
        [100.26818084716797, 74.61857604980469, 81.62117767333984],
        [127.86964416503906, 85.0583267211914, 102.95439147949219]
      ],
      [
        [104.3798828125, 96.70733642578125, 92.60601043701172],
        [77.63021850585938, 68.55794525146484, 96.17212677001953]
      ]
    ]);

    const dy = tf.tensor3d([
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ]);

    const size: [number, number] = [2, 2];
    const alignCorners = false;
    const g = tf.grad(
        (i: tf.Tensor3D) =>
            tf.image.resizeNearestNeighbor(i, size, alignCorners));

    const output = g(input, dy);
    const expected = tf.tensor3d([
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ]);

    expectArraysClose(await output.data(), await expected.data());
    expect(output.shape).toEqual(expected.shape);
    expect(output.dtype).toBe(expected.dtype);
  });
});
