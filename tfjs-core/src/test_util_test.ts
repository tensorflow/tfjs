/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {ALL_ENVS, describeWithFlags} from './jasmine_util';
import {expectArraysClose, expectArraysEqual} from './test_util';

describeWithFlags('expectArraysEqual', ALL_ENVS, () => {
  it('same arrays', () => {
    expectArraysEqual([1, 2, 3], [1, 2, 3]);
  });

  it('throws on different arrays', () => {
    expect(() => expectArraysEqual([1, 2, 3], [3, 2, 1]))
        .toThrowError(/Arrays differ/);
  });

  it('same nested arrays', () => {
    expectArraysEqual([[1, 2], [3, 4]], [[1, 2], [3, 4]]);
  });

  it('throws on different nested arrays', () => {
    expect(() => expectArraysEqual([[1, 2], [3, 4]], [[1, 2], [4, 3]]))
        .toThrowError(/Arrays differ/);
  });

  it('throws on different nested shapes', () => {
    expect(() => expectArraysEqual([[1, 2], [3, 4]], [1, 2, 3, 4]))
        .toThrowError(
            /Arrays have different shapes. Actual: \[2,2\]. Expected: \[4\]/);
  });

  it('float32 with regular array', () => {
    expectArraysEqual(new Float32Array([1, 2, 3]), [1, 2, 3]);
  });

  it('throws on different values of float32 with regular array', () => {
    expect(() => expectArraysEqual(new Float32Array([1, 2, 3]), [1, 2, 4]))
        .toThrowError(/Arrays differ/);
  });

  it('int32 with regular array', () => {
    expectArraysEqual(new Int32Array([1, 2, 3]), [1, 2, 3]);
  });

  it('throws on different values of int32 with regular array', () => {
    expect(() => expectArraysEqual(new Int32Array([1, 2, 3]), [1, 2, 4]))
        .toThrowError(/Arrays differ/);
  });

  it('throws on float32 with int32', () => {
    expect(
        () => expectArraysEqual(
            new Float32Array([1, 2, 3]), new Int32Array([1, 2, 3])))
        .toThrowError(/Arrays are of different type/);
  });

  it('throws on int32 with uint8', () => {
    expect(
        () => expectArraysEqual(
            new Int32Array([1, 2, 3]), new Uint8Array([1, 2, 3])))
        .toThrowError(/Arrays are of different type/);
  });
});

describeWithFlags('expectArraysClose', ALL_ENVS, () => {
  it('same arrays', () => {
    expectArraysClose([1, 2, 3], [1, 2, 3]);
  });

  it('throws on different arrays', () => {
    expect(() => expectArraysClose([1, 2, 3], [3, 2, 1]))
        .toThrowError(/Arrays differ/);
  });

  it('same nested arrays', () => {
    expectArraysClose([[1, 2], [3, 4]], [[1, 2], [3, 4]]);
  });

  it('throws on different nested arrays', () => {
    expect(() => expectArraysClose([[1, 2], [3, 4]], [[1, 2], [4, 3]]))
        .toThrowError(/Arrays differ/);
  });

  it('throws on different nested shapes', () => {
    expect(() => expectArraysClose([[1, 2], [3, 4]], [1, 2, 3, 4]))
        .toThrowError(
            /Arrays have different shapes. Actual: \[2,2\]. Expected: \[4\]/);
  });

  it('float32 with regular array', () => {
    expectArraysClose(new Float32Array([1, 2, 3]), [1, 2, 3]);
  });

  it('throws on different values of float32 with regular array', () => {
    expect(() => expectArraysClose(new Float32Array([1, 2, 3]), [1, 2, 4]))
        .toThrowError(/Arrays differ/);
  });

  it('int32 with regular array', () => {
    expectArraysClose(new Int32Array([1, 2, 3]), [1, 2, 3]);
  });

  it('throws on different values of int32 with regular array', () => {
    expect(() => expectArraysClose(new Int32Array([1, 2, 3]), [1, 2, 4]))
        .toThrowError(/Arrays differ/);
  });

  it('throws on float32 with int32', () => {
    expect(
        () => expectArraysClose(
            new Float32Array([1, 2, 3]), new Int32Array([1, 2, 3])))
        .toThrowError(/Arrays are of different type/);
  });

  it('throws on int32 with uint8', () => {
    expect(
        () => expectArraysClose(
            new Int32Array([1, 2, 3]), new Uint8Array([1, 2, 3])))
        .toThrowError(/Arrays are of different type/);
  });

  it('similar arrays with good epsilon', () => {
    const epsilon = 0.1;
    expectArraysClose(new Float32Array([1, 2, 3.08]), [1, 2, 3], epsilon);
  });

  it('similar arrays with bad epsilon', () => {
    const epsilon = 0.01;
    expect(
        () => expectArraysClose(
            new Float32Array([1, 2, 3.08]), [1, 2, 3], epsilon))
        .toThrowError(/Arrays differ/);
  });
});
