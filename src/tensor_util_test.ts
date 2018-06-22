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

import * as tf from './index';
import {Tensor} from './tensor';
import {NamedTensorMap} from './tensor_types';
// tslint:disable-next-line:max-line-length
import {flattenNameArrayMap, getTensorsInContainer, isTensorInList, unflattenToNameArrayMap} from './tensor_util';

describe('tensor_util.isTensorInList', () => {
  it('not in list', () => {
    const a = tf.scalar(1);
    const list: Tensor[] = [tf.scalar(1), tf.tensor1d([1, 2, 3])];

    expect(isTensorInList(a, list)).toBe(false);
  });

  it('in list', () => {
    const a = tf.scalar(1);
    const list: Tensor[] = [tf.scalar(2), tf.tensor1d([1, 2, 3]), a];

    expect(isTensorInList(a, list)).toBe(true);
  });
});

describe('tensor_util.flattenNameArrayMap', () => {
  it('basic', () => {
    const a = tf.scalar(1);
    const b = tf.scalar(3);
    const c = tf.tensor1d([1, 2, 3]);

    const map: NamedTensorMap = {a, b, c};
    expect(flattenNameArrayMap(map, Object.keys(map))).toEqual([a, b, c]);
  });
});

describe('tensor_util.unflattenToNameArrayMap', () => {
  it('basic', () => {
    const a = tf.scalar(1);
    const b = tf.scalar(3);
    const c = tf.tensor1d([1, 2, 3]);

    expect(unflattenToNameArrayMap(['a', 'b', 'c'], [
      a, b, c
    ])).toEqual({a, b, c});
  });
});

describe('getTensorsInContainer', () => {
  it('null input returns empty tensor', () => {
    const results = getTensorsInContainer(null);

    expect(results).toEqual([]);
  });

  it('tensor input returns one element tensor', () => {
    const x = tf.scalar(1);
    const results = getTensorsInContainer(x);

    expect(results).toEqual([x]);
  });

  it('name tensor map returns flattened tensor', () => {
    const x1 = tf.scalar(1);
    const x2 = tf.scalar(3);
    const x3 = tf.scalar(4);
    const results = getTensorsInContainer({x1, x2, x3});

    expect(results).toEqual([x1, x2, x3]);
  });

  it('can extract from arbitrary depth', () => {
    const container = [
      {x: tf.scalar(1), y: tf.scalar(2)},
      [[[tf.scalar(3)]], {z: tf.scalar(4)}]
    ];
    const results = getTensorsInContainer(container);
    expect(results.length).toBe(4);
  });

  it('works with loops in container', () => {
    const container = [tf.scalar(1), tf.scalar(2), [tf.scalar(3)]];
    const innerContainer = [container];
    // tslint:disable-next-line:no-any
    container.push(innerContainer as any);
    const results = getTensorsInContainer(container);
    expect(results.length).toBe(3);
  });
});
