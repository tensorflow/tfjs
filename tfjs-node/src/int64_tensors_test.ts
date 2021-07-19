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

import {encodeInt32ArrayAsInt64, Int64Scalar} from './int64_tensors';

describe('int64 tensors', () => {
  it('positive value', () => {
    const x = new Int64Scalar(42);
    expect(x.dtype).toEqual('int64');
    const valueArray = x.valueArray;
    expect(valueArray.constructor.name).toEqual('Int32Array');
    expect(valueArray.length).toEqual(2);
    expect(valueArray[0]).toEqual(42);
    expect(valueArray[1]).toEqual(0);
  });

  it('zero value', () => {
    const x = new Int64Scalar(0);
    expect(x.dtype).toEqual('int64');
    const valueArray = x.valueArray;
    expect(valueArray.constructor.name).toEqual('Int32Array');
    expect(valueArray.length).toEqual(2);
    expect(valueArray[0]).toEqual(0);
    expect(valueArray[1]).toEqual(0);
  });

  it('negative value', () => {
    const x = new Int64Scalar(-3);
    expect(x.dtype).toEqual('int64');
    const valueArray = x.valueArray;
    expect(valueArray.constructor.name).toEqual('Int32Array');
    expect(valueArray.length).toEqual(2);
    expect(valueArray[0]).toEqual(-3);
    expect(valueArray[1]).toEqual(-1);
  });

  it('Non-integer value leads to error', () => {
    expect(() => new Int64Scalar(0.4)).toThrowError(/integer/);
    expect(() => new Int64Scalar(-3.2)).toThrowError(/integer/);
  });

  it('Out-of-bound value leads to error', () => {
    expect(() => new Int64Scalar(2147483648)).toThrowError(/bound/);
    expect(() => new Int64Scalar(2147483648 * 2)).toThrowError(/bound/);
    expect(() => new Int64Scalar(-2147483648 - 1)).toThrowError(/bound/);
  });

  it('encode int32array as int64 layout', () => {
    const input = Int32Array.from([2, 10]);
    const valueArray = encodeInt32ArrayAsInt64(input);
    expect(valueArray.length).toEqual(4);
    expect(valueArray[0]).toEqual(2);
    expect(valueArray[1]).toEqual(0);
    expect(valueArray[2]).toEqual(10);
    expect(valueArray[3]).toEqual(0);
  });
});
