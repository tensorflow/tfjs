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
import {test_util} from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../../executor/execution_context';
import {Node} from '../types';

import {executeOp} from './graph_executor';
// tslint:disable-next-line:max-line-length
import {createNumberAttr, createStrAttr, createTensorAttr, createTensorsAttr} from './test_helper';

describe('graph', () => {
  let node: Node;
  const input1 = [tfc.tensor1d([1])];
  const input2 = [tfc.tensor1d([1])];
  const input3 = [tfc.tensor3d([1, 1, 1, 2, 2, 2], [1, 2, 3])];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'input1',
      op: '',
      category: 'graph',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('Const', () => {
      it('should return input', () => {
        node.op = 'Const';
        expect(executeOp(node, {input1}, context)).toEqual(input1);
      });
    });
    describe('placeholder', () => {
      it('should return input', () => {
        node.op = 'Placeholder';
        expect(executeOp(node, {input1}, context)).toEqual(input1);
      });
      it('should return default if input not set', () => {
        node.inputNames = ['input2'];
        node.op = 'PlaceholderWithDefault';
        node.inputParams.default = createTensorAttr(0);
        expect(executeOp(node, {input2}, context)).toEqual(input2);
      });
    });
    describe('Identity', () => {
      it('should return input', () => {
        node.inputNames = ['input'];
        node.inputParams.x = createTensorAttr(0);
        node.op = 'Identity';
        test_util.expectArraysEqual(
            (executeOp(node, {input: input1}, context) as tfc.Tensor[])[0],
            input1[0]);
      });
    });
    describe('Snapshot', () => {
      it('should return input', () => {
        node.inputNames = ['input'];
        node.inputParams.x = createTensorAttr(0);
        node.op = 'Snapshot';
        const result =
            (executeOp(node, {input: input1}, context) as tfc.Tensor[])[0];
        expect(result.rank).toEqual(input1[0].rank);
        test_util.expectArraysClose(result, [1]);
      });
    });
    describe('Shape', () => {
      it('should return shape', () => {
        node.inputNames = ['input'];
        node.inputParams.x = createTensorAttr(0);
        node.op = 'Shape';
        expect(
            Array.prototype.slice.call(
                (executeOp(node, {input: input3}, context) as tfc.Tensor[])[0]
                    .dataSync()))
            .toEqual([1, 2, 3]);
      });
    });
    describe('ShapeN', () => {
      it('should return shapeN', () => {
        node.inputNames = ['input1', 'input3'];
        node.inputParams.x = createTensorsAttr(0, 0);
        node.op = 'ShapeN';
        expect((executeOp(node, {input1, input3}, context) as tfc.Tensor[])
                   .map(t => {
                     return Array.prototype.slice.call(t.dataSync());
                   }))
            .toEqual([[1], [1, 2, 3]]);
      });
    });
    describe('Size', () => {
      it('should return size', () => {
        node.inputNames = ['input'];
        node.inputParams.x = createTensorAttr(0);
        node.op = 'Size';
        expect(
            Array.prototype.slice.call(
                (executeOp(node, {input: input3}, context) as tfc.Tensor[])[0]
                    .dataSync()))
            .toEqual([6]);
      });
    });
    describe('Rank', () => {
      it('should return rank', () => {
        node.inputNames = ['input'];
        node.inputParams.x = createTensorAttr(0);
        node.op = 'Rank';
        expect(
            Array.prototype.slice.call(
                (executeOp(node, {input: input3}, context) as tfc.Tensor[])[0]
                    .dataSync()))
            .toEqual([3]);
      });
    });
    describe('NoOp', () => {
      it('should return empty', () => {
        node.op = 'NoOp';
        expect(executeOp(node, {}, context)).toEqual([]);
      });
    });
  });
  describe('Print', () => {
    it('should return empty', () => {
      node.op = 'Print';
      node.inputNames = ['input1', 'input2'];
      node.inputParams.x = createTensorAttr(0);
      node.inputParams.data = createTensorsAttr(1, 2);
      node.attrParams.message = createStrAttr('message');
      node.attrParams.summarize = createNumberAttr(1);
      spyOn(console, 'log').and.callThrough();
      spyOn(console, 'warn').and.callThrough();

      expect(executeOp(node, {input1, input2}, context)).toEqual(input1);
      expect(console.warn).toHaveBeenCalled();
      expect(console.log).toHaveBeenCalledWith('message');
      expect(console.log).toHaveBeenCalledWith([1]);
    });
  });
  describe('StopGradient', () => {
    it('should return input', () => {
      node.inputNames = ['input'];
      node.inputParams.x = createTensorAttr(0);
      node.op = 'StopGradient';
      test_util.expectArraysClose(
          (executeOp(node, {input: input1}, context) as tfc.Tensor[])[0],
          input1[0]);
    });
  });
  describe('FakeQuantWithMinMaxVars', () => {
    it('should return input', () => {
      node.inputNames = ['input'];
      node.inputParams.x = createTensorAttr(0);
      node.op = 'FakeQuantWithMinMaxVars';
      test_util.expectArraysClose(
          (executeOp(node, {input: input1}, context) as tfc.Tensor[])[0],
          input1[0]);
    });
  });
});
