/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysEqual} from '../../test_util';

async function expectResult(
    result: tf.NamedTensorMap, indices: number[][], values: string[],
    shape: [number, number]) {
  expectArraysEqual(await result.indices.data(), indices);
  expectArraysEqual(await result.values.data(), values);
  expectArraysEqual(await result.shape.data(), shape);

  expect(result.indices.shape).toEqual([indices.length, 2]);
  expect(result.values.shape).toEqual([values.length]);
  expect(result.shape.shape).toEqual([2]);

  expect(result.indices.dtype).toEqual('int32');
  expect(result.values.dtype).toEqual('string');
  expect(result.shape.dtype).toEqual('int32');
}

describeWithFlags('stringSplit', ALL_ENVS, () => {
  it('white space delimiter', async () => {
    const result = tf.string.stringSplit(['pigs on the wing', 'animals'], ' ');
    await expectResult(
        result, [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]],
        ['pigs', 'on', 'the', 'wing', 'animals'], [2, 4]);
  });

  it('empty delimiter', async () => {
    const result = tf.string.stringSplit(['hello', 'hola', 'hi'], '');
    await expectResult(
        result,
        [
          [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2],
          [1, 3], [2, 0], [2, 1]
        ],
        ['h', 'e', 'l', 'l', 'o', 'h', 'o', 'l', 'a', 'h', 'i'], [3, 5]);
  });

  it('empty token', async () => {
    const result = tf.string.stringSplit(
        ['', ' a', 'b ', ' c', ' ', ' d ', '  e', 'f  ', '  g  ', '  '], ' ');
    await expectResult(
        result, [[1, 0], [2, 0], [3, 0], [5, 0], [6, 0], [7, 0], [8, 0]],
        ['a', 'b', 'c', 'd', 'e', 'f', 'g'], [10, 1]);
  });

  it('set empty token', async () => {
    const result = tf.string.stringSplit(
        ['', ' a', 'b ', ' c', ' ', ' d ', '. e', 'f .', ' .g. ', ' .'], ' .');
    await expectResult(
        result, [[1, 0], [2, 0], [3, 0], [5, 0], [6, 0], [7, 0], [8, 0]],
        ['a', 'b', 'c', 'd', 'e', 'f', 'g'], [10, 1]);
  });

  it('with delimiter', async () => {
    const input = ['hello|world', 'hello world'];
    let result = tf.string.stringSplit(input, '|');
    await expectResult(
        result, [[0, 0], [0, 1], [1, 0]], ['hello', 'world', 'hello world'],
        [2, 2]);
    result = tf.string.stringSplit(input, '| ');
    await expectResult(
        result, [[0, 0], [0, 1], [1, 0], [1, 1]],
        ['hello', 'world', 'hello', 'world'], [2, 2]);
    result =
        tf.string.stringSplit(['hello.cruel,world', 'hello cruel world'], '.,');
    await expectResult(
        result, [[0, 0], [0, 1], [0, 2], [1, 0]],
        ['hello', 'cruel', 'world', 'hello cruel world'], [2, 3]);
  });

  it('no skip empty', async () => {
    const input = ['#a', 'b#', '#c#'];
    let result = tf.string.stringSplit(input, '#', false);
    await expectResult(
        result, [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2]],
        ['', 'a', 'b', '', '', 'c', ''], [3, 3]);
    result = tf.string.stringSplit(input, '#');
    await expectResult(
        result, [[0, 0], [1, 0], [2, 0]], ['a', 'b', 'c'], [3, 1]);
  });

  it('large input does not cause an argument overflow', async () => {
    const input = 'a'.repeat(200000);
    const result = tf.string.stringSplit([input], '');
    await expectResult(
        result, Array(input.length).fill(0).map((_, i) => [0, i]),
        input.split(''), [1, input.length]);
  });
});
