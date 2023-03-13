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
import {ALL_ENVS, describeWithFlags, SYNC_BACKEND_ENVS} from '../jasmine_util';
import {encodeStrings, expectArraysClose} from '../test_util';
import {TensorLike1D} from '../types';

describeWithFlags('slice ', ALL_ENVS, () => {
  describeWithFlags('ergonomics', ALL_ENVS, () => {
    it('slices 2x2x2 array into 2x1x1 no size', async () => {
      const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
      const result = a.slice([0, 1, 1]);
      expect(result.shape).toEqual([2, 1, 1]);
      expectArraysClose(await result.data(), [4, 8]);
    });

    it('slices 2x2x2 array into 1x2x2 with scalar begin no size', async () => {
      const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
      const result = a.slice(1);
      expect(result.shape).toEqual([1, 2, 2]);
      expectArraysClose(await result.data(), [5, 6, 7, 8]);
    });

    it('slices 2x2x2 array using 2d size and 2d size', async () => {
      const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
      const result = a.slice([0, 1]);
      expect(result.shape).toEqual([2, 1, 2]);
      expectArraysClose(await result.data(), [3, 4, 7, 8]);
    });

    it('slices 2x2x2 array using negative size', async () => {
      const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
      const result = a.slice([0, 1], [-1, 1]);
      expect(result.shape).toEqual([2, 1, 2]);
      expectArraysClose(await result.data(), [3, 4, 7, 8]);
    });

    it('slices 2x2x2 array using 1d size', async () => {
      const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
      const result = a.slice(0, 1);
      expect(result.shape).toEqual([1, 2, 2]);
      expectArraysClose(await result.data(), [1, 2, 3, 4]);
    });

    it('throws when passed a non-tensor', () => {
      expect(() => tf.slice({} as tf.Tensor, 0, 0))
          .toThrowError(/Argument 'x' passed to 'slice' must be a Tensor/);
    });

    it('accepts a tensor-like object', async () => {
      const a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];  // 2x2x2
      const result = tf.slice(a, [0, 1, 1]);
      expect(result.shape).toEqual([2, 1, 1]);
      expectArraysClose(await result.data(), [4, 8]);
    });

    it('should match source tensor dtype', () => {
      const a = tf.tensor1d([1, 2, 3, 4, 5], 'int32');
      const b = a.asType('float32');

      expect(tf.slice(b, 0).dtype).toEqual('float32');
    });

    it('throws when begin is negative', async () => {
      const a = [[1, 2], [3, 4]];  // 2x2
      expect(() => tf.slice(a, [-1, 1], [
        1, 1
      ])).toThrowError(/slice\(\) does not support negative begin indexing./);
    });
  });

  describeWithFlags('shallow slicing', ALL_ENVS, () => {
    it('shallow slice an input that was cast', async () => {
      const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
      const b = a.toFloat();
      const c = b.slice(1, 1);
      expect(c.dtype).toBe('float32');
      expect(c.shape).toEqual([1, 2]);
      expectArraysClose(await c.data(), [3, 4]);
    });

    it('delayed async read of sliced tensor has no mem leak', async () => {
      const a = tf.zeros([10]);
      const b = tf.slice(a, 0, 1);
      const nBefore = tf.memory().numTensors;
      expect(nBefore).toBe(2);
      await b.data();
      const nAfter = tf.memory().numTensors;
      expect(nAfter).toBe(2);
      tf.dispose([a, b]);
      expect(tf.memory().numTensors).toBe(0);
    });
  });

  describeWithFlags('shallow slicing', SYNC_BACKEND_ENVS, () => {
    it('delayed sync read of sliced tensor has no mem leak', () => {
      const a = tf.zeros([10]);
      const b = tf.slice(a, 0, 1);
      const nBefore = tf.memory().numTensors;
      expect(nBefore).toBe(2);
      b.dataSync();
      const nAfter = tf.memory().numTensors;
      expect(nAfter).toBe(2);
      tf.dispose([a, b]);
      expect(tf.memory().numTensors).toBe(0);
    });
  });

  describeWithFlags('slice5d', ALL_ENVS, () => {
    it('slices 1x1x1x1x1 into shape 1x1x1x1x1 (effectively a copy)',
       async () => {
         const a = tf.tensor5d([[[[[5]]]]], [1, 1, 1, 1, 1]);
         const result = tf.slice(a, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]);

         expect(result.shape).toEqual([1, 1, 1, 1, 1]);
         expectArraysClose(await result.data(), [5]);
       });

    it('slices 2x2x2x2x2 array into 1x2x2x2x2 starting at [1,0,0,0,0]',
       async () => {
         const a = tf.tensor5d(
             [
               1,  2,  3,   4,   5,   6,   7,   8,   9,   10, 11,
               12, 13, 14,  15,  16,  11,  22,  33,  44,  55, 66,
               77, 88, 111, 222, 333, 444, 555, 666, 777, 888
             ],
             [2, 2, 2, 2, 2]);
         const result = tf.slice(a, [1, 0, 0, 0, 0], [1, 2, 2, 2, 2]);

         expect(result.shape).toEqual([1, 2, 2, 2, 2]);
         expectArraysClose(await result.data(), [
           11, 22, 33, 44, 55, 66, 77, 88, 111, 222, 333, 444, 555, 666, 777,
           888
         ]);
       });

    it('slices 2x2x2x2x2 array into 2x1x1x1x1 starting at [0,1,1,1,1]',
       async () => {
         const a = tf.tensor5d(
             [
               1,  2,  3,   4,   5,   6,   7,   8,   9,   10, 11,
               12, 13, 14,  15,  16,  11,  22,  33,  44,  55, 66,
               77, 88, 111, 222, 333, 444, 555, 666, 777, 888
             ],
             [2, 2, 2, 2, 2]);
         const result = tf.slice(a, [0, 1, 1, 1, 1], [2, 1, 1, 1, 1]);

         expect(result.shape).toEqual([2, 1, 1, 1, 1]);
         expectArraysClose(await result.data(), [16, 888]);
       });

    it('accepts a tensor-like object', async () => {
      const a = [[[[[5]]]]];  // 1x1x1x1x1
      const result = tf.slice(a, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]);

      expect(result.shape).toEqual([1, 1, 1, 1, 1]);
      expectArraysClose(await result.data(), [5]);
    });
  });

  describeWithFlags('slice6d', ALL_ENVS, () => {
    it('slices 1x1x1x1x1x1 into shape 1x1x1x1x1x1 (effectively a copy)',
       async () => {
         const a = tf.tensor6d([[[[[[5]]]]]], [1, 1, 1, 1, 1, 1]);
         const result = tf.slice(a, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]);

         expect(result.shape).toEqual([1, 1, 1, 1, 1, 1]);
         expectArraysClose(await result.data(), [5]);
       });

    it('slices 2x2x2x2x2x2 array into 1x2x2x2x2x2 starting at [1,0,0,0,0,0]',
       async () => {
         const a = tf.tensor6d(
             [
               31,  32,  33,   34,   35,   36,   37,   38,   39,   310,  311,
               312, 313, 314,  315,  316,  311,  322,  333,  344,  355,  366,
               377, 388, 3111, 3222, 3333, 3444, 3555, 3666, 3777, 3888,

               1,   2,   3,    4,    5,    6,    7,    8,    9,    10,   11,
               12,  13,  14,   15,   16,   11,   22,   33,   44,   55,   66,
               77,  88,  111,  222,  333,  444,  555,  666,  777,  888
             ],
             [2, 2, 2, 2, 2, 2]);
         const result = tf.slice(a, [1, 0, 0, 0, 0, 0], [1, 2, 2, 2, 2, 2]);

         expect(result.shape).toEqual([1, 2, 2, 2, 2, 2]);
         expectArraysClose(await result.data(), [
           1,  2,  3,   4,   5,   6,   7,   8,   9,   10, 11,
           12, 13, 14,  15,  16,  11,  22,  33,  44,  55, 66,
           77, 88, 111, 222, 333, 444, 555, 666, 777, 888
         ]);
       });

    it('slices 2x2x2x2x2x2 array into 2x1x1x1x1x1 starting at [0,1,1,1,1,1]',
       async () => {
         const a = tf.tensor6d(
             [
               31,  32,  33,   34,   35,   36,   37,   38,   39,   310,  311,
               312, 313, 314,  315,  316,  311,  322,  333,  344,  355,  366,
               377, 388, 3111, 3222, 3333, 3444, 3555, 3666, 3777, 3888,

               1,   2,   3,    4,    5,    6,    7,    8,    9,    10,   11,
               12,  13,  14,   15,   16,   11,   22,   33,   44,   55,   66,
               77,  88,  111,  222,  333,  444,  555,  666,  777,  888
             ],
             [2, 2, 2, 2, 2, 2]);
         const result = tf.slice(a, [0, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1]);

         expect(result.shape).toEqual([2, 1, 1, 1, 1, 1]);
         expectArraysClose(await result.data(), [3888, 888]);
       });

    it('accepts a tensor-like object', async () => {
      const a = [[[[[[5]]]]]];  // 1x1x1x1x1x1
      const result = tf.slice(a, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]);

      expect(result.shape).toEqual([1, 1, 1, 1, 1, 1]);
      expectArraysClose(await result.data(), [5]);
    });
  });

  describeWithFlags('accepts string', ALL_ENVS, () => {
    it('slices 2x2x2 array into 2x1x1 no size.', async () => {
      const a = tf.tensor3d(
          ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'],
          [2, 2, 2], 'string');
      const result = a.slice([0, 1, 1]);
      expect(result.shape).toEqual([2, 1, 1]);
      expectArraysClose(await result.data(), ['four', 'eight']);
    });

    it('slices 2x2x2 array into 1x2x2 with scalar begin no size.', async () => {
      const a = tf.tensor3d(
          ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'],
          [2, 2, 2]);
      const result = a.slice(1);
      expect(result.shape).toEqual([1, 2, 2]);
      expectArraysClose(await result.data(), ['five', 'six', 'seven', 'eight']);
    });

    it('slice encoded string.', async () => {
      const bytes =
          encodeStrings([
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'
          ]) as TensorLike1D;
      const a = tf.tensor3d(bytes, [2, 2, 2], 'string');
      const result = a.slice([0, 1, 1]);
      expect(result.shape).toEqual([2, 1, 1]);
      expectArraysClose(await result.data(), ['four', 'eight']);
    });
  });
});
