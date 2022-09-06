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
import {Tensor2D, Tensor3D} from '../../tensor';
import {expectArraysClose} from '../../test_util';

import {scalar, tensor1d, tensor2d, tensor3d} from '../ops';

describeWithFlags('bandPart', ALL_ENVS, () => {
  it('keeps tensor unchanged', async () => {
    const x: Tensor2D = tensor2d([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3]);
    expectArraysClose(
        await tf.linalg.bandPart(x, -1, -1).array(),
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]);
  });

  it('upper triangular matrix', async () => {
    const x: Tensor2D = tensor2d([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3]);
    expectArraysClose(
        await tf.linalg.bandPart(x, 0, -1).array(),
        [[1, 1, 1], [0, 1, 1], [0, 0, 1]]);
  });

  it('lower triangular matrix', async () => {
    const x: Tensor2D = tensor2d([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3]);
    expectArraysClose(
        await tf.linalg.bandPart(x, -1, 0).array(),
        [[1, 0, 0], [1, 1, 0], [1, 1, 1]]);
  });

  it('diagonal elements', async () => {
    const x: Tensor2D = tensor2d([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3]);
    expectArraysClose(
        await tf.linalg.bandPart(x, 0, 0).array(),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  });

  it('lower triangular elements', async () => {
    const x: Tensor2D = tensor2d([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3]);
    expectArraysClose(
        await tf.linalg.bandPart(x, 1, 0).array(),
        [[1, 0, 0], [1, 1, 0], [0, 1, 1]]);
  });

  it('upper triangular elements', async () => {
    const x: Tensor2D = tensor2d([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3]);
    expectArraysClose(
        await tf.linalg.bandPart(x, 0, 1).array(),
        [[1, 1, 0], [0, 1, 1], [0, 0, 1]]);
  });

  it('4X4 matrix - tensorflow python examples', async () => {
    const x: Tensor2D = tensor2d(
        [[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, 0]]);
    expectArraysClose(
        await tf.linalg.bandPart(x, 1, -1).array(),
        [[0, 1, 2, 3], [-1, 0, 1, 2], [0, -1, 0, 1], [0, 0, -1, 0]]);
    expectArraysClose(
        await tf.linalg.bandPart(x, 2, 1).array(),
        [[0, 1, 0, 0], [-1, 0, 1, 0], [-2, -1, 0, 1], [0, -2, -1, 0]]);
  });

  it('3 dimensional matrix', async () => {
    const x: Tensor3D = tensor3d([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]);
    expectArraysClose(
        await tf.linalg.bandPart(x, 0, 0).array(),
        [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]);
  });

  it('2X3X3 tensor', async () => {
    const x: Tensor3D = tensor3d(
        [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]);
    expectArraysClose(
        await tf.linalg.bandPart(x, 1, 2).array(),
        [[[1, 1, 1], [1, 1, 1], [0, 1, 1]], [[1, 1, 1], [1, 1, 1], [0, 1, 1]]]);
  });

  const la = tf.linalg;

  it('fails for scalar', async () => {
    const x = scalar(1);
    expect(() => la.bandPart(x, 1, 2)).toThrowError(/bandPart.*rank/i);
  });

  it('fails for 1D tensor', async () => {
    const x = tensor1d([1, 2, 3, 4, 5]);
    expect(() => la.bandPart(x, 1, 2)).toThrowError(/bandPart.*rank/i);
  });

  it('fails if numLower or numUpper too large', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);

    for (const numLower of [3, 5, 8, 13]) {
      for (const numUpper of [-1, 0, 1, 2]) {
        expect(() => tf.linalg.bandPart(a, numLower, numUpper))
            .toThrowError(/bandPart.*numLower/i);
      }
    }

    for (const numLower of [-1, 0, 1]) {
      for (const numUpper of [4, 5, 9]) {
        expect(() => tf.linalg.bandPart(a, numLower, numUpper))
            .toThrowError(/bandPart.*numUpper/i);
      }
    }

    for (const numLower of [3, 5, 8, 13]) {
      for (const numUpper of [4, 5, 9]) {
        expect(() => tf.linalg.bandPart(a, numLower, numUpper))
            .toThrowError(/bandPart.*(numLower|numUpper)/i);
      }
    }
  });

  it('works for 3x4 example', async () => {
    const a = tf.tensor2d([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);

    expectArraysClose(
        await la.bandPart(a, 0, 0).array(),
        [[1, 0, 0, 0], [0, 6, 0, 0], [0, 0, 11, 0]]);
    expectArraysClose(
        await la.bandPart(a, 0, 1).array(),
        [[1, 2, 0, 0], [0, 6, 7, 0], [0, 0, 11, 12]]);
    expectArraysClose(
        await la.bandPart(a, 0, 2).array(),
        [[1, 2, 3, 0], [0, 6, 7, 8], [0, 0, 11, 12]]);
    for (const numUpper of [3, 4, -1, -2]) {
      expectArraysClose(
          await la.bandPart(a, 0, numUpper).array(),
          [[1, 2, 3, 4], [0, 6, 7, 8], [0, 0, 11, 12]]);
    }

    expectArraysClose(
        await la.bandPart(a, 1, 0).array(),
        [[1, 0, 0, 0], [5, 6, 0, 0], [0, 10, 11, 0]]);
    expectArraysClose(
        await la.bandPart(a, 1, 1).array(),
        [[1, 2, 0, 0], [5, 6, 7, 0], [0, 10, 11, 12]]);
    expectArraysClose(
        await la.bandPart(a, 1, 2).array(),
        [[1, 2, 3, 0], [5, 6, 7, 8], [0, 10, 11, 12]]);
    for (const numUpper of [3, 4, -1, -2]) {
      expectArraysClose(
          await la.bandPart(a, 1, numUpper).array(),
          [[1, 2, 3, 4], [5, 6, 7, 8], [0, 10, 11, 12]]);
    }

    for (const numLower of [2, 3, -1, -2]) {
      expectArraysClose(
          await la.bandPart(a, numLower, 0).array(),
          [[1, 0, 0, 0], [5, 6, 0, 0], [9, 10, 11, 0]]);
      expectArraysClose(
          await la.bandPart(a, numLower, 1).array(),
          [[1, 2, 0, 0], [5, 6, 7, 0], [9, 10, 11, 12]]);
      expectArraysClose(
          await la.bandPart(a, numLower, 2).array(),
          [[1, 2, 3, 0], [5, 6, 7, 8], [9, 10, 11, 12]]);
      for (const numUpper of [3, 4, -1, -2]) {
        expectArraysClose(
            await la.bandPart(a, numLower, numUpper).array(),
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
      }
    }
  });
});
