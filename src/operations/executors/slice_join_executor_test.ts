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

import {executeOp} from './slice_join_executor';
// tslint:disable-next-line:max-line-length
import {createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttr, createTensorAttr, createTensorsAttr} from './test_helper';

describe('slice join', () => {
  let node: Node;
  const input1 = dl.Scalar.new(1);
  const input2 = dl.Scalar.new(2);
  const input3 = dl.Scalar.new(3);
  describe('multi-tensor ops', () => {
    beforeEach(() => {
      node = {
        name: 'test',
        op: '',
        category: 'slice_join',
        inputNames: ['input1', 'input2', 'input3'],
        inputs: [],
        params: {
          tensors: createTensorsAttr(0, 1),
          axis: createNumberAttrFromIndex(-1)
        },
        children: []
      };
    });
    describe('executeOp', () => {
      ['concat', 'stack'].forEach(op => {
        it('should call dl.' + op, () => {
          const spy = spyOn(dl, op as 'concat');
          node.op = op;
          executeOp(node, {input1, input2, input3});

          expect(spy).toHaveBeenCalledWith([input1, input2], 3);
        });
      });
    });
  });
  describe('single-tensor ops', () => {
    beforeEach(() => {
      node = {
        name: 'test',
        op: '',
        category: 'slice_join',
        inputNames: ['input1'],
        inputs: [],
        params: {x: createTensorAttr(0)},
        children: []
      };
    });
    describe('executeOp', () => {
      it('should call dl.reverse', () => {
        spyOn(dl, 'reverse');
        node.op = 'reverse';
        node.params.axis = createNumberAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];
        executeOp(node, {input1, input2});

        expect(dl.reverse).toHaveBeenCalledWith(input1, 2);
      });

      it('should call dl.tile', () => {
        spyOn(dl, 'tile');
        node.op = 'tile';
        node.params.reps = createNumberAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];
        executeOp(node, {input1, input2});

        expect(dl.tile).toHaveBeenCalledWith(input1, 2);
      });

      it('should call dl.slice', () => {
        spyOn(dl, 'slice');
        node.op = 'slice';
        node.params.begin = createNumericArrayAttr([1]);
        node.params.size = createNumericArrayAttr([2]);
        executeOp(node, {input1});

        expect(dl.slice).toHaveBeenCalledWith(input1, [1], [2]);
      });

      it('should call dl.gather', () => {
        spyOn(dl, 'gather');
        node.op = 'gather';
        node.params.axis = createNumberAttr(1);
        node.params.indices = createTensorAttr(1);
        node.inputNames = ['input1', 'input2'];
        executeOp(node, {input1, input2});

        expect(dl.gather).toHaveBeenCalledWith(input1, input2, 1);
      });
    });
  });
});
