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
import {scalar, test_util} from '@tensorflow/tfjs-core';

import * as tensorflow from '../../data/compiled_api';
import {ExecutionContext} from '../../executor/execution_context';
import {Node} from '../types';

import {NodeValueImpl} from './node_value_impl';

const NODE: Node = {
  name: 'test',
  op: 'const',
  category: 'custom',
  inputNames: ['a', 'b'],
  inputs: [],
  inputParams: {},
  attrParams: {},
  children: [],
  rawAttrs: {
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
      list:
          {type: [tensorflow.DataType.DT_FLOAT, tensorflow.DataType.DT_INT32]}
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
  }
};
const TENSOR_MAP = {
  'a': [scalar(1)],
  'b': [scalar(2)],
  'test': [scalar(3)]
};

let nodeValue: NodeValueImpl;
describe('NodeValueImpl', () => {
  beforeEach(() => {
    nodeValue =
        new NodeValueImpl(NODE, TENSOR_MAP, new ExecutionContext({}, {}));
  });
  describe('getInput', () => {
    it('should find tensor from tensormap', async () => {
      const result = nodeValue.inputs[0];
      test_util.expectArraysClose(await result.data(), [1]);

      const result2 = nodeValue.inputs[1];
      test_util.expectArraysClose(await result2.data(), [2]);
    });
  });
  describe('getAttr', () => {
    it('should parse number', () => {
      expect(nodeValue.attrs['d']).toEqual(3);
      expect(nodeValue.attrs['h']).toEqual(4.5);
    });
    it('should parse number[]', () => {
      expect(nodeValue.attrs['i']).toEqual([3, 6, 0]);
      expect(nodeValue.attrs['j']).toEqual([4.5, 5.5, 0.0]);
    });
    it('should parse string', () => {
      expect(nodeValue.attrs['e']).toEqual('nhwc');
    });
    it('should parse string[]', () => {
      expect(nodeValue.attrs['k']).toEqual(['nhwc', 'nhwc', '']);
    });
    it('should parse boolean', () => {
      expect(nodeValue.attrs['g']).toEqual(true);
    });
    it('should parse boolean[]', () => {
      expect(nodeValue.attrs['o']).toEqual([true, false]);
    });
    it('should parse dtype', () => {
      expect(nodeValue.attrs['f']).toEqual('float32');
    });
    it('should parse dtype[]', () => {
      expect(nodeValue.attrs['l']).toEqual(['float32', 'int32']);
    });
    it('should parse tensor shape', () => {
      expect(nodeValue.attrs['m']).toEqual([1, 2]);
    });
    it('should parse tensor shape[]', () => {
      expect(nodeValue.attrs['n']).toEqual([[1, 2], [2, 3]]);
    });
  });
});
