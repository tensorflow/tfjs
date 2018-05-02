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
import * as tfc from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../../executor';
import {Node} from '../index';

import {executeOp} from './graph_executor';
// tslint:disable-next-line:max-line-length
import {createNumberAttr, createStrAttr, createTensorAttr, createTensorsAttr} from './test_helper';

describe('graph', () => {
  let node: Node;
  const input1 = [tfc.tensor1d([1])];
  const input2 = [tfc.tensor1d([1])];
  const context = new ExecutionContext({});

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
        expect(executeOp(node, {input1}, context)).toEqual(input1);
      });
    });
    describe('placeholder', () => {
      it('should return input', () => {
        node.op = 'placeholder';
        expect(executeOp(node, {input1}, context)).toEqual(input1);
      });
      it('should return default if input not set', () => {
        node.inputNames = ['input2'];
        node.op = 'placeholder';
        node.params.default = createTensorAttr(0);
        expect(executeOp(node, {input2}, context)).toEqual(input2);
      });
    });
    describe('identity', () => {
      it('should return input', () => {
        node.inputNames = ['input'];
        node.params.x = createTensorAttr(0);
        node.op = 'identity';
        expect(executeOp(node, {input: input1}, context)).toEqual(input1);
      });
    });
    describe('shape', () => {
      it('should return shape', () => {
        node.inputNames = ['input'];
        node.params.x = createTensorAttr(0);
        node.op = 'shape';
        expect(
            Array.prototype.slice.call(
                (executeOp(node, {input: input1}, context) as tfc.Tensor[])[0]
                    .dataSync()))
            .toEqual([1]);
      });
    });
    describe('noop', () => {
      it('should return empty', () => {
        node.op = 'noop';
        expect(executeOp(node, {}, context)).toEqual([]);
      });
    });
  });
  describe('print', () => {
    it('should return empty', () => {
      node.op = 'print';
      node.inputNames = ['input1', 'input2'];
      node.params.x = createTensorAttr(0);
      node.params.data = createTensorsAttr(1, 1);
      node.params.message = createStrAttr('message');
      node.params.summarize = createNumberAttr(1);
      spyOn(console, 'log');
      spyOn(console, 'warn');

      expect(executeOp(node, {input1, input2}, context)).toEqual(input1);
      expect(console.warn).toHaveBeenCalled();
      expect(console.log).toHaveBeenCalledWith('message');
      expect(console.log).toHaveBeenCalledWith([1]);
    });
  });
  describe('stopGradient', () => {
    it('should return input', () => {
      node.inputNames = ['input'];
      node.params.x = createTensorAttr(0);
      node.op = 'stopGradient';
      expect(executeOp(node, {input: input1}, context)).toEqual(input1);
    });
  });
});
