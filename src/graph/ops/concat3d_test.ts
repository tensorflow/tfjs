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

import * as concat_util from '../../math/concat_util';
import {NDArrayMathCPU} from '../../math/math_cpu';
import {Array3D} from '../../math/ndarray';
import {Tensor} from '../graph';
import {TensorArrayMap} from '../tensor_array_map';

import {Concat3D} from './concat3d';

describe('concat3d operation', () => {
  let math: NDArrayMathCPU;

  let x1Tensor: Tensor;
  let x2Tensor: Tensor;
  let yTensor: Tensor;
  let concatOperation: Concat3D;
  let tensorArrayMap: TensorArrayMap;

  beforeEach(() => {
    math = new NDArrayMathCPU();
    tensorArrayMap = new TensorArrayMap();
  });

  afterEach(() => {
    tensorArrayMap.disposeArray(x1Tensor);
    tensorArrayMap.disposeArray(x2Tensor);
    tensorArrayMap.disposeArray(yTensor);
  });

  it('concats tensors, axis=0', () => {
    const axis = 0;

    const x1 = Array3D.new([1, 1, 3], [1, 2, 3]);
    const x2 = Array3D.new([1, 1, 3], [4, 5, 6]);

    x1Tensor = new Tensor(x1.shape);
    x2Tensor = new Tensor(x2.shape);
    yTensor = new Tensor(
        concat_util.computeConcatOutputShape(x1.shape, x2.shape, axis));

    tensorArrayMap.set(x1Tensor, x1);
    tensorArrayMap.set(x2Tensor, x2);

    concatOperation = new Concat3D(x1Tensor, x2Tensor, axis, yTensor);

    concatOperation.feedForward(math, tensorArrayMap);

    const y = tensorArrayMap.get(yTensor);

    expect(y.shape).toEqual([2, 1, 3]);
    expect(y.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('concats tensors, axis=1', () => {
    const axis = 1;

    const x1 = Array3D.new([1, 1, 3], [1, 2, 3]);
    const x2 = Array3D.new([1, 1, 3], [4, 5, 6]);

    x1Tensor = new Tensor(x1.shape);
    x2Tensor = new Tensor(x2.shape);
    yTensor = new Tensor(
        concat_util.computeConcatOutputShape(x1.shape, x2.shape, axis));

    tensorArrayMap.set(x1Tensor, x1);
    tensorArrayMap.set(x2Tensor, x2);

    concatOperation = new Concat3D(x1Tensor, x2Tensor, axis, yTensor);

    concatOperation.feedForward(math, tensorArrayMap);

    const y = tensorArrayMap.get(yTensor);

    expect(y.shape).toEqual([1, 2, 3]);
    expect(y.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('concats tensors, axis=2', () => {
    const axis = 2;

    const x1 = Array3D.new([1, 1, 3], [1, 2, 3]);
    const x2 = Array3D.new([1, 1, 3], [4, 5, 6]);

    x1Tensor = new Tensor(x1.shape);
    x2Tensor = new Tensor(x2.shape);
    yTensor = new Tensor(
        concat_util.computeConcatOutputShape(x1.shape, x2.shape, axis));

    tensorArrayMap.set(x1Tensor, x1);
    tensorArrayMap.set(x2Tensor, x2);

    concatOperation = new Concat3D(x1Tensor, x2Tensor, axis, yTensor);

    concatOperation.feedForward(math, tensorArrayMap);

    const y = tensorArrayMap.get(yTensor);

    expect(y.shape).toEqual([1, 1, 6]);
    expect(y.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });
});
