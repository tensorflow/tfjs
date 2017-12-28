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

import {ENV} from '../environment';
import {Array1D, NDArray} from '../math/ndarray';
import {Tensor} from './graph';
import {SummedTensorArrayMap, TensorArrayMap} from './tensor_array_map';

describe('TensorArrayMap.size', () => {
  it('is 0 at construction', () => {
    expect((new TensorArrayMap()).size()).toEqual(0);
  });

  it('is 1 after add', () => {
    const map = new TensorArrayMap();
    map.set(new Tensor([]), NDArray.zeros([1]));
    expect(map.size()).toEqual(1);
  });

  it('increments for every add', () => {
    const map = new TensorArrayMap();
    for (let i = 0; i < 9; ++i) {
      map.set(new Tensor([]), NDArray.zeros([1]));
    }
    expect(map.size()).toEqual(9);
  });
});

describe('TensorArrayMap.hasNullArray', () => {
  let map: TensorArrayMap;
  let t: Tensor;
  beforeEach(() => {
    map = new TensorArrayMap();
    t = new Tensor([]);
  });

  it('returns true for null NDArray entries', () => {
    map.set(t, null);
    expect(map.hasNullArray(t)).toBe(true);
  });

  it('returns false for non-null NDArray entries', () => {
    map.set(t, NDArray.zeros([1]));
    expect(map.hasNullArray(t)).toBe(false);
  });

  it('throws for missing keys', () => {
    expect(() => map.hasNullArray(t)).toThrowError(/not in array map/);
  });
});

describe('TensorArrayMap.get', () => {
  let map: TensorArrayMap;
  let t: Tensor;
  beforeEach(() => {
    map = new TensorArrayMap();
    t = new Tensor([]);
  });

  it('returns the associated NDArray', () => {
    const nda = NDArray.zeros([1]);
    map.set(t, nda);
    expect(map.get(t)).toBe(nda);
  });

  it('throws if associated NDArray is null', () => {
    map.set(t, null);
    expect(() => map.get(t)).toThrowError(/has null array/);
  });

  it('throws for missing key', () => {
    expect(() => map.get(t)).toThrowError(/not in array map/);
  });
});

describe('TensorArrayMap.delete', () => {
  let map: TensorArrayMap;
  let t: Tensor;
  beforeEach(() => {
    map = new TensorArrayMap();
    t = new Tensor([]);
  });

  it('deletes the key from the map', () => {
    map.set(t, null);
    map.delete(t);
    expect(() => map.get(t)).toThrow();
  });

  it('is tolerant of deleting nonexistent keys', () => {
    map.set(t, null);
    map.delete(t);
    map.delete(t);
    map.delete(t);
    map.delete(t);
  });
});

describe('SummedTensorArrayMap.add', () => {
  let map: SummedTensorArrayMap;
  let t: Tensor;
  const math = ENV.math;
  beforeEach(() => {
    map = new SummedTensorArrayMap(math);
    t = new Tensor([]);
  });

  it('add sums gradients', () => {
    map.add(t, Array1D.new([1, 2, 3]));
    expect(map.get(t).dataSync()).toEqual(new Float32Array([1, 2, 3]));

    map.add(t, Array1D.new([30, 20, 10]));
    expect(map.get(t).dataSync()).toEqual(new Float32Array([31, 22, 13]));
  });
});
