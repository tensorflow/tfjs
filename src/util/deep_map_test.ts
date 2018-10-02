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
 *
 * =============================================================================
 */

// tslint:disable-next-line:max-line-length
import {deepMap, deepMapAndAwaitAll, DeepMapAsyncResult, DeepMapResult, deepZip, isIterable} from './deep_map';

const integerNames = [
  'zero', 'one', 'two', 'three', ['an array representing', 'four'],
  {five: 'I am a dict'}
];

// tslint:disable-next-line:no-any
function transform(x: any): DeepMapResult {
  if (x === null) {
    return null;
  }
  if (typeof (x) === 'number') {
    return {value: integerNames[x], recurse: false};
  }

  if (isIterable(x)) {
    return {value: null, recurse: true};
  } else {
    return {value: x, recurse: false};
  }
}

// tslint:disable-next-line:no-any
function asyncTransform(x: any): DeepMapAsyncResult {
  const result = transform(x);
  return {
    value: result.value === null ? null : Promise.resolve(result.value),
    recurse: result.recurse
  };
}

describe('deepMap', () => {
  it('maps single mappable objects', () => {
    expect(deepMap(null, transform)).toEqual(null);
    expect(deepMap(1, transform)).toEqual('one');
    expect(deepMap(3, transform)).toEqual('three');
    expect(deepMap('hello', transform)).toEqual('hello');
  });
  it('maps arrays of mappable objects', () => {
    expect(deepMap([null, 1], transform)).toEqual([null, 'one']);
    expect(deepMap([1, 2, 3], transform)).toEqual(['one', 'two', 'three']);
    expect(deepMap([1, 'hello', 3, null], transform)).toEqual([
      'one', 'hello', 'three', null
    ]);
  });
  it('maps objects containing mappable fields', () => {
    expect(deepMap({a: null, b: 1}, transform)).toEqual({a: null, b: 'one'});
    expect(deepMap({a: 1, b: 2, c: 3}, transform))
        .toEqual({a: 'one', b: 'two', c: 'three'});
    expect(deepMap({a: 1, b: 'hello', c: 3, d: null}, transform))
        .toEqual({a: 'one', b: 'hello', c: 'three', d: null});
  });
  it('maps nested structures containing mappable fields', () => {
    const input = {a: 'hello', b: [2, 3, null, {ba: 0, bb: 'world'}]};
    const expected = {
      a: 'hello',
      b: ['two', 'three', null, {ba: 'zero', bb: 'world'}]
    };
    expect(deepMap(input, transform)).toEqual(expected);
  });
  it('maps elements that produce arrays or objects', () => {
    const input = {a: 'hello', b: [2, 4, null, {ba: 5, bb: 'world'}]};
    const expected = {
      a: 'hello',
      b: [
        'two', ['an array representing', 'four'], null,
        {ba: {five: 'I am a dict'}, bb: 'world'}
      ]
    };
    expect(deepMap(input, transform)).toEqual(expected);
  });
  it('handles repetitions', () => {
    const input = {a1: 4, a2: 4, b1: 5, b2: 5};
    const result = deepMap(input, transform);
    expect(result).toEqual({
      a1: ['an array representing', 'four'],
      a2: ['an array representing', 'four'],
      b1: {five: 'I am a dict'},
      b2: {five: 'I am a dict'}
    });
    const b1Mapped = result.b1;
    const b2Mapped = result.b2;
    expect(b2Mapped).toBe(b1Mapped);
  });
  it('detects and rejects cycles', () => {
    // tslint:disable-next-line:no-any
    const b: any[] = [2, 3, null, {ba: 0, bb: 'world'}];
    const c = {a: 'hello', b};
    b[4] = c;
    const input = [b, c];
    expect(() => deepMap(input, transform)).toThrowError();
  });
});

describe('asyncDeepMap', () => {
  it('Maps single mappable objects', async () => {
    expect(await deepMapAndAwaitAll(null, asyncTransform)).toEqual(null);
    expect(await deepMapAndAwaitAll(1, asyncTransform)).toEqual('one');
    expect(await deepMapAndAwaitAll(3, asyncTransform)).toEqual('three');
    expect(await deepMapAndAwaitAll('hello', asyncTransform)).toEqual('hello');
  });
  it('Maps arrays of mappable objects', async () => {
    expect(await deepMapAndAwaitAll([null, 1], asyncTransform)).toEqual([
      null, 'one'
    ]);
    expect(await deepMapAndAwaitAll([1, 2, 3], asyncTransform)).toEqual([
      'one', 'two', 'three'
    ]);
    expect(await deepMapAndAwaitAll([1, 'hello', 3, null], asyncTransform))
        .toEqual(['one', 'hello', 'three', null]);
  });
  it('Maps objects containing mappable fields', async () => {
    expect(await deepMapAndAwaitAll({a: null, b: 1}, asyncTransform))
        .toEqual({a: null, b: 'one'});
    expect(await deepMapAndAwaitAll({a: 1, b: 2, c: 3}, asyncTransform))
        .toEqual({a: 'one', b: 'two', c: 'three'});
    expect(await deepMapAndAwaitAll(
               {a: 1, b: 'hello', c: 3, d: null}, asyncTransform))
        .toEqual({a: 'one', b: 'hello', c: 'three', d: null});
  });
  it('Maps nested structures containing mappable fields', async () => {
    const input = {a: 'hello', b: [2, 3, null, {ba: 0, bb: 'world'}]};
    const expected = {
      a: 'hello',
      b: ['two', 'three', null, {ba: 'zero', bb: 'world'}]
    };
    expect(await deepMapAndAwaitAll(input, asyncTransform)).toEqual(expected);
  });
  it('maps elements that produce arrays or objects', async () => {
    const input = {a: 'hello', b: [2, 4, null, {ba: 5, bb: 'world'}]};
    const expected = {
      a: 'hello',
      b: [
        'two', ['an array representing', 'four'], null,
        {ba: {five: 'I am a dict'}, bb: 'world'}
      ]
    };
    expect(await deepMapAndAwaitAll(input, asyncTransform)).toEqual(expected);
  });
  it('handles repetitions', async () => {
    const input = {a1: 4, a2: 4, b1: 5, b2: 5};
    const result = await deepMapAndAwaitAll(input, asyncTransform);
    expect(result).toEqual({
      a1: ['an array representing', 'four'],
      a2: ['an array representing', 'four'],
      b1: {five: 'I am a dict'},
      b2: {five: 'I am a dict'}
    });
    const b1Mapped = result.b1;
    const b2Mapped = result.b2;
    expect(b2Mapped).toBe(b1Mapped);
  });
  it('detects and rejects cycles', async done => {
    try {
      // tslint:disable-next-line:no-any
      const b: any[] = [2, 3, null, {ba: 0, bb: 'world'}];
      const c = {a: 'hello', b};
      b[4] = c;
      const input = [b, c];
      await deepMapAndAwaitAll(input, asyncTransform);
      done.fail();
    } catch (e) {
      expect(e.message).toBe('Circular references are not supported.');
      done();
    }
  });
});

describe('deepZip', () => {
  it('zips arrays of primitives', () => {
    expect(deepZip([1, 2, 3])).toEqual([1, 2, 3]);
    expect(deepZip([null, 1])).toEqual([null, 1]);
    expect(deepZip([1, 'hello', 3, null])).toEqual([1, 'hello', 3, null]);
  });
  it('zips objects containing simple fields', () => {
    expect(deepZip([
      {a: 1, b: 2}, {a: 3, b: 4}
    ])).toEqual({a: [1, 3], b: [2, 4]});
    expect(deepZip([
      {a: 1, b: 'hello', c: 4, d: null}, {a: 2, b: 'world', c: 5, d: 7},
      {a: 3, b: '!', c: 6, d: null}
    ])).toEqual({
      a: [1, 2, 3],
      b: ['hello', 'world', '!'],
      c: [4, 5, 6],
      d: [null, 7, null]
    });
  });
  it('zips arrays containing simple fields', () => {
    expect(deepZip([[1, 2], [3, 4]])).toEqual([[1, 3], [2, 4]]);
  });
  it('zips nested structures', () => {
    const input = [
      {a: 'plums', b: [1, 2, null, {ba: 3, bb: 'sweet'}]},
      {a: 'icebox', b: [3, 4, 5, {ba: 6, bb: 'cold'}]}
    ];
    const expected = {
      a: ['plums', 'icebox'],
      b: [[1, 3], [2, 4], [null, 5], {ba: [3, 6], bb: ['sweet', 'cold']}]
    };
    expect(deepZip(input)).toEqual(expected);
  });
  it('detects and rejects cycles', () => {
    // tslint:disable-next-line:no-any
    const b: any[] = [2, 3, null, {ba: 0, bb: 'world'}];
    const c = {a: 'hello', b};
    b[4] = c;
    const input = [b, c];
    expect(() => deepZip(input)).toThrowError();
  });
});
