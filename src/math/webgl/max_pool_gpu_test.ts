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

import * as test_util from '../../test_util';
import {NDArrayMathCPU} from '../math_cpu';
import {Array3D, NDArray} from '../ndarray';
import * as pool_gpu_test_util from './pool_gpu_test_util';

describe('max_pool_gpu', () => {
  function uploadMaxPoolDownload(
      a: Float32Array, xShape: [number, number, number],
      filterSizes: [number, number]|number, strides: [number, number]|number,
      zeroPad: number|'valid'|'same'): Float32Array {
    return pool_gpu_test_util.uploadPoolDownload(
        a, xShape, filterSizes, strides, zeroPad, 'max');
  }

  function compareToCPU(
      xShape: [number, number, number], fSize: number, stride: number,
      pad: number) {
    const x = NDArray.randNormal<Array3D>(xShape);

    const mathCPU = new NDArrayMathCPU();
    const yCPU = mathCPU.maxPool(x, fSize, stride, pad);
    const yGPU =
        uploadMaxPoolDownload(x.getValues(), x.shape, fSize, stride, pad);

    test_util.expectArraysClose(yGPU, yCPU.getValues(), 1e-5);
  }

  it('matches CPU on random input, d1=1,d2=1,f=2,s=1,p=0', () => {
    const depth = 1;
    const dyShape: [number, number, number] = [8, 8, depth];
    const fSize = 2;
    const stride = 1;
    const zeroPad = 0;
    compareToCPU(dyShape, fSize, stride, zeroPad);
  });

  it('matches CPU on random input, d=1,f=3,s=2,p=1', () => {
    const depth = 1;
    const inputShape: [number, number, number] = [7, 7, depth];
    const fSize = 3;
    const stride = 2;
    const zeroPad = 1;
    compareToCPU(inputShape, fSize, stride, zeroPad);
  });

  it('matches CPU on random input, d=4,f=2,s=1,p=0', () => {
    const depth = 4;
    const inputShape: [number, number, number] = [8, 8, depth];
    const fSize = 2;
    const stride = 1;
    const zeroPad = 0;
    compareToCPU(inputShape, fSize, stride, zeroPad);
  });

  it('matches CPU on random input, d=3,f=3,s=3,p=1', () => {
    const depth = 3;
    const inputShape: [number, number, number] = [7, 7, depth];
    const fSize = 3;
    const stride = 3;
    const zeroPad = 1;
    compareToCPU(inputShape, fSize, stride, zeroPad);
  });

  it('non even filter 1x2 on 3x3 input', () => {
    const x = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const res =
        uploadMaxPoolDownload(x.getValues(), x.shape, [1, 2], [1, 1], 'valid');
    expect(res).toEqual(new Float32Array([2, 3, 5, 6, 8, 9]));
  });

  it('non even filter 2x1 on 3x3 input', () => {
    const x = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const res =
        uploadMaxPoolDownload(x.getValues(), x.shape, [2, 1], [1, 1], 'valid');
    expect(res).toEqual(new Float32Array([4, 5, 6, 7, 8, 9]));
  });
});
