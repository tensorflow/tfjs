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
import * as dl from 'deeplearn';

import {Node} from '../index';

import {executeOp} from './graph_executor';
import {createTensorAttr} from './test_helper';

describe('graph', () => {
  let node: Node;
  const input1 = dl.Array1D.new([1]);

  beforeEach(() => {
    node = {
      name: 'input1',
      op: '',
      category: 'graph',
      inputNames: [],
      inputs: [],
      params: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('const', () => {
      it('should return input', () => {
        node.op = 'const';
        expect(executeOp(node, {input1})).toEqual(input1);
      });
    });
    describe('placeholder', () => {
      it('should return input', () => {
        node.op = 'placeholder';
        expect(executeOp(node, {input1})).toEqual(input1);
      });
    });
    describe('identity', () => {
      it('should return input', () => {
        node.inputNames = ['input'];
        node.params.x = createTensorAttr(0);
        node.op = 'identity';
        expect(executeOp(node, {input: input1})).toEqual(input1);
      });
    });
    describe('shape', () => {
      it('should return shape', () => {
        node.inputNames = ['input'];
        node.params.x = createTensorAttr(0);
        node.op = 'shape';
        expect(Array.prototype.slice.call(
                   executeOp(node, {input: input1}).dataSync()))
            .toEqual([1]);
      });
    });
  });
});
