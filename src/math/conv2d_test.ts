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

import * as dl from '../index';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose} from '../test_util';
import {Rank} from './types';

describeWithFlags('conv2d', ALL_ENVS, () => {
  it('x=[2,2,1] f=[1,1,1,2] s=1 p=0', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    const w = dl.tensor4d([2], [fSize, fSize, inputDepth, outputDepth]);
    const bias = dl.tensor1d([-1]);

    const result = dl.conv2d(x, w, bias, stride, pad);

    expectArraysClose(result, [1, 3, 5, 7]);
  });

  it('x=[2,2,2,1] f=[1,1,1,1] s=1 p=0', () => {
    const inputDepth = 1;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = dl.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], inShape);
    const w = dl.tensor4d([2], [fSize, fSize, inputDepth, outputDepth]);
    const bias = dl.tensor1d([-1]);

    const result = dl.conv2d(x, w, bias, stride, pad);
    expect(result.shape).toEqual([2, 2, 2, 1]);
    const expected = [1, 3, 5, 7, 9, 11, 13, 15];

    expectArraysClose(result, expected);
  });

  it('x=[2,2,1] f=[2,2,1,1] s=1 p=0', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    const w =
        dl.tensor4d([3, 1, 5, 0], [fSize, fSize, inputDepth, outputDepth]);
    const bias = dl.tensor1d([-1]);

    const result = dl.conv2d(x, w, bias, stride, pad);
    expectArraysClose(result, [19]);
  });

  it('throws when x is not rank 3', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;

    // tslint:disable-next-line:no-any
    const x: any = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const w =
        dl.tensor4d([3, 1, 5, 0], [fSize, fSize, inputDepth, outputDepth]);
    const bias = dl.tensor1d([-1]);

    expect(() => dl.conv2d(x, w, bias, stride, pad)).toThrowError();
  });

  it('throws when weights is not rank 4', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const pad = 0;
    const stride = 1;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    // tslint:disable-next-line:no-any
    const w: any = dl.tensor3d([3, 1, 5, 0], [2, 2, 1]);
    const bias = dl.tensor1d([-1]);

    expect(() => dl.conv2d(x, w, bias, stride, pad)).toThrowError();
  });

  it('throws when biases is not rank 1', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    const w =
        dl.tensor4d([3, 1, 5, 0], [fSize, fSize, inputDepth, outputDepth]);
    // tslint:disable-next-line:no-any
    const bias: any = dl.tensor2d([2, 2, 2, 2], [2, 2]);

    expect(() => dl.conv2d(x, w, bias, stride, pad)).toThrowError();
  });

  it('throws when x depth does not match weight depth', () => {
    const inputDepth = 1;
    const wrongInputDepth = 5;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    const w =
        dl.randomNormal<Rank.R4>([fSize, fSize, wrongInputDepth, outputDepth]);
    const bias = dl.tensor1d([-1]);

    expect(() => dl.conv2d(x, w, bias, stride, pad)).toThrowError();
  });

  it('throws when dimRoundingMode is set and pad is not a number', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const dimRoundingMode = 'round';

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    const w = dl.randomNormal<Rank.R4>([fSize, fSize, inputDepth, outputDepth]);
    const bias = dl.tensor1d([-1]);

    expect(() => dl.conv2d(x, w, bias, stride, pad, dimRoundingMode))
        .toThrowError();
  });

  it('gradient input=[3,3,1] f=[2,2,1,1] s=1 p=0', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number] = [3, 3, inputDepth];
    const filterSize = 2;
    const stride = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = dl.ones<Rank.R4>(filterShape);
    const bias = dl.tensor1d([-1]);

    const x = dl.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = dl.tensor3d([3, 1, 2, 0], [2, 2, 1]);

    const grads = dl.grads(
        (x: dl.Tensor3D, filter: dl.Tensor4D, bias: dl.Tensor1D) =>
            x.conv2d(filter, bias, stride, pad));
    const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(dx, [3, 4, 1, 5, 6, 1, 2, 2, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    // TODO(nsthorat): Fix the precision for byte textures.
    expectArraysClose(dfilter, [13, 19, 31, 37], 1e-1);

    expect(dbias.shape).toEqual(bias.shape);
    expectArraysClose(dbias, [6], 1e-1);
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const stride = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = dl.ones<Rank.R4>(filterShape);

    const bias = dl.tensor1d([-1]);

    const x = dl.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = dl.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const grads = dl.grads(
        (x: dl.Tensor4D, filter: dl.Tensor4D, bias: dl.Tensor1D) =>
            x.conv2d(filter, bias, stride, pad));
    const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(
        dx, [3, 4, 1, 5, 6, 1, 2, 2, 0, 3, 4, 1, 5, 6, 1, 2, 2, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    // TODO(nsthorat): Fix the precision for byte textures.
    expectArraysClose(dfilter, [13 * 2, 19 * 2, 31 * 2, 37 * 2], 1e-1);

    expect(dbias.shape).toEqual(bias.shape);
    expectArraysClose(dbias, [12]);
  });
});
