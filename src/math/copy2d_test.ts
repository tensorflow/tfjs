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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Array2D} from './ndarray';

const tests: MathTests = it => {
  it('throws an error if source and dest shapes have different areas', math => {
    const source = Array2D.zeros([100, 100]);
    const dest = Array2D.zeros([100, 100]);
    const sourceSize: [number, number] = [20, 20];
    const destSize: [number, number] = [5, 5];

    expect(
        () => math.copy2D(source, [0, 0], sourceSize, dest, [0, 0], destSize))
        .toThrowError();

    source.dispose();
    dest.dispose();
  });

  it('copies a src shape into a dst shape', math => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);

    math.copy2D(source, [1, 1], [2, 3], dest, [2, 0], [3, 2]);

    test_util.expectArraysClose(
        dest.getValues(),
        new Float32Array([0, 0, 0, 0, 6, 7, 8, 10, 11, 12, 0, 0]));

    source.dispose();
    dest.dispose();
  });

  it('copies a 1x1 source to a 1x1 dest', math => {
    const source = Array2D.new([1, 1], new Float32Array([Math.PI]));
    const dest = Array2D.new([1, 1], new Float32Array([0]));

    math.copy2D(source, [0, 0], [1, 1], dest, [0, 0], [1, 1]);
    const result = dest.getValues();

    expect(result.length).toEqual(1);
    test_util.expectArraysClose(result, new Float32Array([Math.PI]));

    source.dispose();
    dest.dispose();
  });

  it('copies a 1x2 source to a 1x2 dest', math => {
    const source = Array2D.new([1, 2], new Float32Array([1, 2]));
    const dest = Array2D.new([1, 2], new Float32Array([0, 0]));

    math.copy2D(source, [0, 0], [1, 2], dest, [0, 0], [1, 2]);

    const result = dest.getValues();

    expect(result.length).toEqual(2);
    test_util.expectArraysClose(result, new Float32Array([1, 2]));

    source.dispose();
    dest.dispose();
  });

  it('copies a 2x1 source to a 2x1 dest', math => {
    const source = Array2D.new([2, 1], new Float32Array([1, 2]));
    const dest = Array2D.new([2, 1], new Float32Array([0, 0]));

    math.copy2D(source, [0, 0], [2, 1], dest, [0, 0], [2, 1]);
    const result = dest.getValues();

    expect(result.length).toEqual(2);
    test_util.expectArraysClose(result, new Float32Array([1, 2]));

    source.dispose();
    dest.dispose();
  });

  it('copies a 2x2 source to a 2x2 dest', math => {
    const source = Array2D.new([2, 2], new Float32Array([1, 2, 3, 4]));
    const dest = Array2D.new([2, 2], new Float32Array([0, 0, 0, 0]));

    math.copy2D(source, [0, 0], [2, 2], dest, [0, 0], [2, 2]);
    const result = dest.getValues();

    expect(result.length).toEqual(4);
    test_util.expectArraysClose(result, new Float32Array([1, 2, 3, 4]));

    source.dispose();
    dest.dispose();
  });

  it('copies inner 2x2 from a 4x4 source to a 2x2 dest', math => {
    const sourceVals = new Float32Array(16);
    sourceVals[5] = 10;
    sourceVals[6] = 11;
    sourceVals[9] = 12;
    sourceVals[10] = 13;

    const source = Array2D.new([4, 4], sourceVals);

    const dest = Array2D.new([2, 2], new Float32Array(4));
    math.copy2D(source, [1, 1], [2, 2], dest, [0, 0], [2, 2]);

    const result = dest.getValues();
    expect(result.length).toEqual(4);

    test_util.expectArraysClose(result, new Float32Array([10, 11, 12, 13]));

    source.dispose();
    dest.dispose();
  });

  it('copies a 1x4 row from source into a 2x2 dest', math => {
    const source = Array2D.new([1, 4], new Float32Array([1, 2, 3, 4]));
    const dest = Array2D.new([2, 2], new Float32Array(4));

    math.copy2D(source, [0, 0], [1, 4], dest, [0, 0], [2, 2]);
    const result = dest.getValues();

    expect(result.length).toEqual(4);

    test_util.expectArraysClose(result, new Float32Array([1, 2, 3, 4]));

    source.dispose();
    dest.dispose();
  });

  it('copies a 1x4 row from source into a 4x1 dest', math => {
    const source = Array2D.new([1, 4], new Float32Array([1, 2, 3, 4]));
    const dest = Array2D.new([4, 1], new Float32Array(4));

    math.copy2D(source, [0, 0], [1, 4], dest, [0, 0], [4, 1]);

    const result = dest.getValues();
    expect(result.length).toEqual(4);

    test_util.expectArraysClose(result, new Float32Array([1, 2, 3, 4]));

    source.dispose();
    dest.dispose();
  });

  it('copies a column from source into a dest row vector', math => {
    const sourceVals = new Float32Array(10 * 10);
    for (let i = 0; i < 10; ++i) {
      sourceVals[3 + (i * 10)] = i + 1;
    }

    const source = Array2D.new([10, 10], sourceVals);

    const dest = Array2D.new([1, 10], new Float32Array(10));
    math.copy2D(source, [0, 3], [10, 1], dest, [0, 0], [1, 10]);

    const result = dest.getValues();

    test_util.expectArraysClose(
        result, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));

    source.dispose();
    dest.dispose();
  });

  it('doesn\'t touch destination pixels outside of the source box', math => {
    const source = Array2D.new([1, 1], new Float32Array([1]));
    const dest = Array2D.new([1, 2], new Float32Array([Math.PI, 0]));

    math.copy2D(source, [0, 0], [1, 1], dest, [0, 1], [1, 1]);

    const result = dest.getValues();

    test_util.expectArraysClose(result, new Float32Array([Math.PI, 1]));

    source.dispose();
    dest.dispose();
  });

  it('throws when requesting out of bounds source copy', math => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);

    expect(() => math.copy2D(source, [1, 1], [10, 10], dest, [2, 0], [
      3, 2
    ])).toThrowError();

    source.dispose();
    dest.dispose();
  });

  it('throws when requesting out of bounds dest copy', math => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);

    expect(() => math.copy2D(source, [1, 1], [2, 3], dest, [2, 0], [
      3, 10
    ])).toThrowError();

    source.dispose();
    dest.dispose();
  });

  it('accumulates results from previous copies into dest texture', math => {
    const shape: [number, number] = [10, 10];
    const size: [number, number] = [10, 1];
    const sourceVals = new Float32Array(100);
    for (let i = 0; i < 100; ++i) {
      sourceVals[i] = i;
    }

    const source = Array2D.new(shape, sourceVals);
    const dest = Array2D.zeros(shape);

    for (let i = 0; i < 10; ++i) {
      const offset: [number, number] = [0, i];
      math.copy2D(source, offset, size, dest, offset, size);
    }
    const res = dest.getValues();

    test_util.expectArraysClose(res, sourceVals);
  });
};

test_util.describeMathCPU('copy2D', [tests]);
test_util.describeMathGPU('copy2D', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
