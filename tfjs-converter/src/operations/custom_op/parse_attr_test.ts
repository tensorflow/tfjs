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

import * as tensorflow from '../../data/compiled_api';
import {parseRawAttr} from '../operation_mapper';
import {ValueType} from '../types';

const rawAttrs = {
  c: {tensor: {}},
  d: {i: 3},
  e: {s: 'TkhXQw=='},
  f: {type: tensorflow.DataType.DT_FLOAT},
  g: {b: true},
  h: {f: 4.5},
  i: {list: {i: [3, 6, 0]}},
  j: {list: {f: [4.5, 5.5, 0.0]}},
  k: {list: {s: ['TkhXQw==', 'TkhXQw==', '']}},
  l: {
    list: {type: [tensorflow.DataType.DT_FLOAT, tensorflow.DataType.DT_INT32]}
  },
  m: {shape: {dim: [{name: 'a', size: 1}, {name: 'b', size: 2}]}},
  n: {
    list: {
      shape: [
        {dim: [{name: 'a', size: 1}, {name: 'b', size: 2}]},
        {dim: [{name: 'c', size: 2}, {name: 'd', size: 3}]}
      ]
    }
  },
  o: {list: {b: [true, false]}}
} as tensorflow.INameAttrList;

const attrs: {[key: string]: ValueType} = {};
for (const attrName in rawAttrs) {
  // tslint:disable-next-line:no-any
  const rawAttr = (rawAttrs as any)[attrName];
  attrs[attrName] = parseRawAttr(rawAttr);
}

describe('NodeValueImpl', () => {
  it('should parse number', () => {
    expect(attrs['d']).toEqual(3);
    expect(attrs['h']).toEqual(4.5);
  });
  it('should parse number[]', () => {
    expect(attrs['i']).toEqual([3, 6, 0]);
    expect(attrs['j']).toEqual([4.5, 5.5, 0.0]);
  });
  it('should parse string', () => {
    expect(attrs['e']).toEqual('nhwc');
  });
  it('should parse string[]', () => {
    expect(attrs['k']).toEqual(['nhwc', 'nhwc', '']);
  });
  it('should parse boolean', () => {
    expect(attrs['g']).toEqual(true);
  });
  it('should parse boolean[]', () => {
    expect(attrs['o']).toEqual([true, false]);
  });
  it('should parse dtype', () => {
    expect(attrs['f']).toEqual('float32');
  });
  it('should parse dtype[]', () => {
    expect(attrs['l']).toEqual(['float32', 'int32']);
  });
  it('should parse tensor shape', () => {
    expect(attrs['m']).toEqual([1, 2]);
  });
  it('should parse tensor shape[]', () => {
    expect(attrs['n']).toEqual([[1, 2], [2, 3]]);
  });
});
