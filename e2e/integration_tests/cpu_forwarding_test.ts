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

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as tfc from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {SMOKE} from './constants';

/**
 *  This file tests cpu forwarding from webgl backend.
 */

describeWithFlags(
    `${SMOKE} cpu forwarding (webgl->cpu)`, {
      predicate: testEnv => testEnv.backendName === 'webgl' &&
          tfc.findBackend('webgl') !== null && tfc.findBackend('cpu') !== null
    },

    () => {
      let webglCpuForwardFlagSaved: boolean;

      beforeAll(() => {
        webglCpuForwardFlagSaved = tfc.env().getBool('WEBGL_CPU_FORWARD');
        tfc.env().set('WEBGL_CPU_FORWARD', true);
      });

      afterAll(() => {
        tfc.env().set('WEBGL_CPU_FORWARD', webglCpuForwardFlagSaved);
      });

      it('should work for slice.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        const result = a.slice([0, 1, 1]);
        expect(result.shape).toEqual([2, 1, 1]);
        tfc.test_util.expectArraysClose(await result.data(), [4, 8]);
      });

      it('should work for stridedSlice.', async () => {
        await tfc.setBackend('webgl');

        const t = tfc.tensor2d([
          [1, 2, 3, 4, 5],
          [2, 3, 4, 5, 6],
          [3, 4, 5, 6, 7],
          [4, 5, 6, 7, 8],
          [5, 6, 7, 8, 9],
          [6, 7, 8, 9, 10],
          [7, 8, 9, 10, 11],
          [8, 8, 9, 10, 11],
          [9, 8, 9, 10, 11],
          [10, 8, 9, 10, 11],
        ]);
        const begin = [0, 4];
        const end = [0, 5];
        const strides = [1, 1];
        const beginMask = 0;
        const endMask = 0;
        const ellipsisMask = 1;
        const output = t.stridedSlice(
            begin, end, strides, beginMask, endMask, ellipsisMask);
        expect(output.shape).toEqual([10, 1]);
        tfc.test_util.expectArraysClose(
            await output.data(), [5, 6, 7, 8, 9, 10, 11, 11, 11, 11]);
      });

      it('should work for concat.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([3]);
        const b = tfc.tensor1d([5]);

        const result = tfc.concat1d([a, b]);
        const expected = [3, 5];
        tfc.test_util.expectArraysClose(await result.data(), expected);
      });

      it('should work for neg.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1, -3, 2, 7, -4]);
        const result = tfc.neg(a);
        tfc.test_util.expectArraysClose(
            await result.data(), [-1, 3, -2, -7, 4]);
      });

      it('should work for multiply.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor2d([1, 2, -3, -4], [2, 2]);
        const b = tfc.tensor2d([5, 3, 4, -7], [2, 2]);
        const expected = [5, 6, -12, 28];
        const result = tfc.mul(a, b);

        expect(result.shape).toEqual([2, 2]);
        tfc.test_util.expectArraysClose(await result.data(), expected);
      });

      it('should work for gather.', async () => {
        await tfc.setBackend('webgl');

        const t = tfc.tensor1d([1, 2, 3]);

        const t2 = tfc.gather(t, tfc.scalar(1, 'int32'), 0);

        expect(t2.shape).toEqual([]);
        tfc.test_util.expectArraysClose(await t2.data(), [2]);
      });

      it('should work for prod.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
        const result = tfc.prod(a);
        tfc.test_util.expectArraysClose(await result.data(), 0);
      });

      it('should work for less.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1, 4, 5], 'int32');
        const b = tfc.tensor1d([2, 3, 5], 'int32');
        const res = tfc.less(a, b);

        expect(res.dtype).toBe('bool');
        tfc.test_util.expectArraysClose(await res.data(), [1, 0, 0]);
      });

      it('should work for greater.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1, 4, 5], 'int32');
        const b = tfc.tensor1d([2, 3, 5], 'int32');
        const res = tfc.greater(a, b);

        expect(res.dtype).toBe('bool');
        tfc.test_util.expectArraysClose(await res.data(), [0, 1, 0]);
      });

      it('should work for minimum.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([0.5, 3, -0.1, -4]);
        const b = tfc.tensor1d([0.2, 0.4, 0.25, 0.15]);
        const result = tfc.minimum(a, b);

        expect(result.shape).toEqual(a.shape);
        tfc.test_util.expectArraysClose(
            await result.data(), [0.2, 0.4, -0.1, -4]);
      });

      it('should work for maximum.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([0.5, 3, -0.1, -4]);
        const b = tfc.tensor1d([0.2, 0.4, 0.25, 0.15]);
        const result = tfc.maximum(a, b);

        expect(result.shape).toEqual(a.shape);
        tfc.test_util.expectArraysClose(
            await result.data(), [0.5, 3, 0.25, 0.15]);
      });

      it('should work for max.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([3, -1, 0, 100, -7, 2]);
        const r = tfc.max(a);
        tfc.test_util.expectArraysClose(await r.data(), 100);
      });

      it('should work for add.', async () => {
        await tfc.setBackend('webgl');

        const c = tfc.scalar(5);
        const a = tfc.tensor1d([1, 2, 3]);

        const result = tfc.add(c, a);

        tfc.test_util.expectArraysClose(await result.data(), [6, 7, 8]);
      });

      it('should work for sub.', async () => {
        await tfc.setBackend('webgl');

        const c = tfc.scalar(5);
        const a = tfc.tensor1d([7, 2, 3]);

        const result = tfc.sub(c, a);

        tfc.test_util.expectArraysClose(await result.data(), [-2, 3, 2]);
      });

      it('should work for ceil.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1.5, 2.1, -1.4]);
        const r = tfc.ceil(a);
        tfc.test_util.expectArraysClose(await r.data(), [2, 3, -1]);
      });

      it('should work for floor.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1.5, 2.1, -1.4]);
        const r = tfc.floor(a);

        tfc.test_util.expectArraysClose(await r.data(), [1, 2, -2]);
      });

      it('should work for exp.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1, 2, 0]);
        const r = tfc.exp(a);

        tfc.test_util.expectArraysClose(
            await r.data(), [Math.exp(1), Math.exp(2), 1]);
      });

      it('should work for expm1.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1, 2, 0]);
        const r = tfc.expm1(a);

        tfc.test_util.expectArraysClose(
            await r.data(), [Math.expm1(1), Math.expm1(2), Math.expm1(0)]);
      });

      it('should work for log.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1, 2]);
        const r = tfc.log(a);
        tfc.test_util.expectArraysClose(
            await r.data(), [Math.log(1), Math.log(2)]);
      });

      it('should work for rsqrt.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([2, 4]);
        const r = tfc.rsqrt(a);
        tfc.test_util.expectArraysClose(
            await r.data(), [1 / Math.sqrt(2), 1 / Math.sqrt(4)]);
      });

      it('should work for abs.', async () => {
        await tfc.setBackend('webgl');

        const a = tfc.tensor1d([1, -2, 0, 3, -0.1]);
        const result = tfc.abs(a);
        tfc.test_util.expectArraysClose(await result.data(), [1, 2, 0, 3, 0.1]);
      });

      it('should work for transpose.', async () => {
        await tfc.setBackend('webgl');

        const t = tfc.tensor2d([1, 11, 2, 22, 3, 33, 4, 44], [2, 4]);
        const t2 = tfc.transpose(t, [1, 0]);

        expect(t2.shape).toEqual([4, 2]);
        tfc.test_util.expectArraysClose(
            await t2.data(), [1, 3, 11, 33, 2, 4, 22, 44]);
      });
    });
