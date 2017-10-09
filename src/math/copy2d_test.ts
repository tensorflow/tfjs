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
};

test_util.describeMathCPU('copy2D', [tests]);
test_util.describeMathGPU('copy2D', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
