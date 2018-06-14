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

import {ExecutionContext} from '../../executor/execution_context';
import {Node} from '../types';

import {executeOp} from './slice_join_executor';
// tslint:disable-next-line:max-line-length
import {createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttr, createTensorAttr, createTensorsAttr} from './test_helper';

describe('slice join', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const input2 = [tfc.scalar(2)];
  const input3 = [tfc.scalar(3)];
  const input4 = [tfc.tensor1d([3])];
  const input5 = [tfc.tensor1d([3, 4])];
  const context = new ExecutionContext({});

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
      it('should call tfc.concat', () => {
        const spy = spyOn(tfc, 'concat');
        node.op = 'concat';
        executeOp(node, {input1, input2, input3}, context);

        expect(spy).toHaveBeenCalledWith([input1[0], input2[0]], 3);
      });
      it('should call tfc.unstack', () => {
        const spy = spyOn(tfc, 'unstack');
        node.op = 'unstack';
        node.params.tensor = createTensorAttr(0);
        node.params.axis = createNumberAttr(4);
        executeOp(node, {input1}, context);

        expect(spy).toHaveBeenCalledWith(input1[0], 4);
      });

      it('should call tfc.stack', () => {
        const spy = spyOn(tfc, 'stack');
        node.op = 'stack';
        node.params.tensors = createTensorsAttr(0, 0);
        node.params.axis = createNumberAttr(4);
        executeOp(node, {input1, input2, input3}, context);

        expect(spy.calls.mostRecent().args[0][0]).toEqual(input1[0]);
        expect(spy.calls.mostRecent().args[0][1]).toEqual(input2[0]);
        expect(spy.calls.mostRecent().args[0][2]).toEqual(input3[0]);
        expect(spy.calls.mostRecent().args[1]).toEqual(4);
      });

      it('should reshape tensors for tfc.stack', () => {
        const spy = spyOn(tfc, 'stack');
        node.op = 'stack';
        node.inputNames = ['input1', 'input2', 'input3', 'input4'];
        node.params.tensors = createTensorsAttr(0, 0);
        node.params.axis = createNumberAttr(4);
        executeOp(node, {input1, input2, input3, input4}, context);

        expect(spy.calls.mostRecent().args[0][0]).toEqual(input1[0]);
        expect(spy.calls.mostRecent().args[0][1]).toEqual(input2[0]);
        expect(spy.calls.mostRecent().args[0][2]).toEqual(input3[0]);
        expect(spy.calls.mostRecent().args[0][3].shape).toEqual([]);
        expect(spy.calls.mostRecent().args[1]).toEqual(4);
      });

      it('should raise error if tensors shape does not match for tfc.stack',
         () => {
           node.op = 'stack';
           node.inputNames = ['input1', 'input2', 'input3', 'input5'];
           node.params.tensors = createTensorsAttr(0, 0);
           node.params.axis = createNumberAttr(4);
           expect(
               () => executeOp(node, {input1, input2, input3, input5}, context))
               .toThrow(new Error('the input tensors shape does not match'));
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
      it('should call tfc.reverse', () => {
        spyOn(tfc, 'reverse');
        node.op = 'reverse';
        node.params.axis = createNumberAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];
        executeOp(node, {input1, input2}, context);

        expect(tfc.reverse).toHaveBeenCalledWith(input1[0], 2);
      });

      it('should call tfc.tile', () => {
        spyOn(tfc, 'tile');
        node.op = 'tile';
        node.params.reps = createNumberAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];
        executeOp(node, {input1, input2}, context);

        expect(tfc.tile).toHaveBeenCalledWith(input1[0], 2);
      });

      it('should call tfc.slice', () => {
        spyOn(tfc, 'slice');
        node.op = 'slice';
        node.params.begin = createNumericArrayAttr([1]);
        node.params.size = createNumericArrayAttr([2]);
        executeOp(node, {input1}, context);

        expect(tfc.slice).toHaveBeenCalledWith(input1[0], [1], [2]);
      });

      it('should call tfc.stridedSlice', () => {
        spyOn(tfc, 'stridedSlice');
        node.op = 'stridedSlice';
        node.params.begin = createNumericArrayAttr([1]);
        node.params.end = createNumericArrayAttr([2]);
        node.params.strides = createNumericArrayAttr([3]);
        node.params.beginMask = createNumberAttr(4);
        node.params.endMask = createNumberAttr(5);
        executeOp(node, {input1}, context);

        expect(tfc.stridedSlice)
            .toHaveBeenCalledWith(input1[0], [1], [2], [3], 4, 5);
      });

      it('should call tfc.gather', () => {
        spyOn(tfc, 'gather');
        node.op = 'gather';
        node.params.axis = createNumberAttr(1);
        node.params.indices = createTensorAttr(1);
        node.inputNames = ['input1', 'input2'];
        executeOp(node, {input1, input2}, context);

        expect(tfc.gather).toHaveBeenCalledWith(input1[0], input2[0], 1);
      });

      it('should call tfc.split', () => {
        spyOn(tfc, 'split');
        node.op = 'split';
        node.params.axis = createNumberAttrFromIndex(0);
        node.params.x = createTensorAttr(1);
        node.params.numOrSizeSplits = createNumberAttr(2);
        node.inputNames = ['input1', 'input2'];
        executeOp(node, {input1, input2}, context);

        expect(tfc.split).toHaveBeenCalledWith(input2[0], 2, 1);
      });
    });
  });
});
