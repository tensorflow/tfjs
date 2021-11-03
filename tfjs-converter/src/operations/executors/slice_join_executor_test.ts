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
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {ExecutionContext} from '../../executor/execution_context';
import * as slice_join from '../op_list/slice_join';
import {Node} from '../types';

import {executeOp} from './slice_join_executor';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';
import {createBooleanArrayAttrFromIndex, createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttrFromIndex, createTensorAttr, createTensorsAttr, validateParam} from './test_helper';

describe('slice join', () => {
  let node: Node;
  const input1 = [tfOps.scalar(1)];
  const input2 = [tfOps.scalar(2)];
  const input3 = [tfOps.scalar(3)];
  const input4 = [tfOps.tensor1d([3])];
  const input5 = [tfOps.tensor1d([3, 4])];
  const context = new ExecutionContext({}, {}, {});
  let spyOps: RecursiveSpy<typeof tfOps>;
  let spyOpsAsTfOps: typeof tfOps;

  beforeEach(() => {
    spyOps = spyOnAllFunctions(tfOps);
    spyOpsAsTfOps = spyOps as unknown as typeof tfOps;
  });

  describe('multi-tensor ops', () => {
    beforeEach(() => {
      node = {
        name: 'test',
        op: '',
        category: 'slice_join',
        inputNames: ['input1', 'input2', 'input3'],
        inputs: [],
        inputParams: {},
        attrParams: {},
        children: []
      };
    });
    describe('executeOp', () => {
      it('Concat', () => {
        node.op = 'Concat';
        node.inputParams.tensors = createTensorsAttr(1, 0);
        node.inputParams.axis = createNumberAttrFromIndex(0);
        node.attrParams.n = createNumberAttr(2);
        spyOps.concat.and.returnValue({});
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.concat).toHaveBeenCalledWith([input2[0], input3[0]], 1);
      });
      it('Concat when input length and n mismatch', () => {
        node.op = 'Concat';
        node.inputParams.tensors = createTensorsAttr(0, -1);
        node.inputParams.axis = createNumberAttrFromIndex(-1);
        node.attrParams.n = createNumberAttr(1);
        spyOps.concat.and.returnValue({});
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.concat).toHaveBeenCalledWith([input1[0]], 3);
      });
      it('should match json def for Concat', () => {
        node.op = 'Concat';
        node.inputParams.tensors = createTensorsAttr(1, 0);
        node.inputParams.axis = createNumberAttrFromIndex(0);
        node.attrParams.n = createNumberAttr(2);

        expect(validateParam(node, slice_join.json, 'Concat')).toBeTruthy();
      });
      it('ConcatV2', () => {
        node.op = 'ConcatV2';
        node.inputParams.tensors = createTensorsAttr(0, -1);
        node.inputParams.axis = createNumberAttrFromIndex(-1);
        node.attrParams.n = createNumberAttr(2);
        spyOps.concat.and.returnValue({});
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.concat).toHaveBeenCalledWith([input1[0], input2[0]], 3);
      });
      it('ConcatV2 when input length and n mismatch', () => {
        node.op = 'ConcatV2';
        node.inputParams.tensors = createTensorsAttr(0, -1);
        node.inputParams.axis = createNumberAttrFromIndex(-1);
        node.attrParams.n = createNumberAttr(1);
        spyOps.concat.and.returnValue({});
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.concat).toHaveBeenCalledWith([input1[0]], 3);
      });
      it('should match json def for ConcatV2', () => {
        node.op = 'ConcatV2';
        node.inputParams.tensors = createTensorsAttr(0, -1);
        node.inputParams.axis = createNumberAttrFromIndex(-1);
        node.attrParams.n = createNumberAttr(3);

        expect(validateParam(node, slice_join.json, 'ConcatV2')).toBeTruthy();
      });
      it('should call tfOps.unstack', () => {
        node.op = 'Unpack';
        node.inputParams.tensor = createTensorAttr(0);
        node.attrParams.axis = createNumberAttr(4);
        spyOps.unstack.and.returnValue({});

        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.unstack).toHaveBeenCalledWith(input1[0], 4);
      });
      it('should match json def for unstack', () => {
        node.op = 'Unpack';
        node.inputParams.tensor = createTensorAttr(0);
        node.attrParams.axis = createNumberAttr(4);

        expect(validateParam(node, slice_join.json)).toBeTruthy();
      });
      it('should call tfOps.stack', () => {
        node.op = 'Pack';
        node.inputParams.tensors = createTensorsAttr(0, 0);
        node.attrParams.axis = createNumberAttr(4);
        spyOps.stack.and.returnValue({});
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.stack.calls.mostRecent().args[0][0]).toEqual(input1[0]);
        expect(spyOps.stack.calls.mostRecent().args[0][1]).toEqual(input2[0]);
        expect(spyOps.stack.calls.mostRecent().args[0][2]).toEqual(input3[0]);
        expect(spyOps.stack.calls.mostRecent().args[1]).toEqual(4);
      });
      it('should match json def for unstack', () => {
        node.op = 'Pack';
        node.inputParams.tensors = createTensorsAttr(0, 0);
        node.attrParams.axis = createNumberAttr(4);

        expect(validateParam(node, slice_join.json)).toBeTruthy();
      });
      it('should reshape tensors for tfOps.stack', () => {
        node.op = 'Pack';
        node.inputNames = ['input1', 'input2', 'input3', 'input4'];
        node.inputParams.tensors = createTensorsAttr(0, 0);
        node.attrParams.axis = createNumberAttr(4);
        spyOps.stack.and.returnValue({});
        executeOp(
            node, {input1, input2, input3, input4}, context, spyOpsAsTfOps);

        expect(spyOps.stack.calls.mostRecent().args[0][0]).toEqual(input1[0]);
        expect(spyOps.stack.calls.mostRecent().args[0][1]).toEqual(input2[0]);
        expect(spyOps.stack.calls.mostRecent().args[0][2]).toEqual(input3[0]);
        expect(spyOps.stack.calls.mostRecent().args[0][3].shape).toEqual([]);
        expect(spyOps.stack.calls.mostRecent().args[1]).toEqual(4);
      });
      it('should raise error if tensors shape does not match for tfOps.stack',
         () => {
           node.op = 'Pack';
           node.inputNames = ['input1', 'input2', 'input3', 'input5'];
           node.inputParams.tensors = createTensorsAttr(0, 0);
           node.attrParams.axis = createNumberAttr(4);
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
        inputParams: {x: createTensorAttr(0)},
        attrParams: {},
        children: []
      };
    });
    describe('executeOp', () => {
      it('should call tfOps.reverse', () => {
        node.op = 'Reverse';
        node.inputParams.dims = createBooleanArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input6'];
        const input6 = [tfOps.tensor1d([false, true], 'bool')];
        spyOps.reverse.and.returnValue({});
        executeOp(node, {input1, input6}, context, spyOpsAsTfOps);

        expect(spyOps.reverse).toHaveBeenCalledWith(input1[0], [1]);
      });
      it('should match json def for reverse', () => {
        node.op = 'Reverse';
        node.inputParams.dims = createBooleanArrayAttrFromIndex(1);

        expect(validateParam(node, slice_join.json, 'Reverse')).toBeTruthy();
      });
      it('should call tfOps.reverse', () => {
        node.op = 'ReverseV2';
        node.inputParams.axis = createNumericArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input4'];
        spyOps.reverse.and.returnValue({});
        executeOp(node, {input1, input4}, context, spyOpsAsTfOps);

        expect(spyOps.reverse).toHaveBeenCalledWith(input1[0], [3]);
      });
      it('should match json def for reverse', () => {
        node.op = 'ReverseV2';
        node.inputParams.axis = createNumericArrayAttrFromIndex(1);

        expect(validateParam(node, slice_join.json, 'ReverseV2')).toBeTruthy();
      });
      it('should call tfOps.tile', () => {
        node.op = 'Tile';
        node.inputParams.reps = createNumericArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input4'];
        spyOps.tile.and.returnValue({});
        executeOp(node, {input1, input4}, context, spyOpsAsTfOps);

        expect(spyOps.tile).toHaveBeenCalledWith(input1[0], [3]);
      });
      it('should match json def for tile', () => {
        node.op = 'Tile';
        node.inputParams.reps = createNumericArrayAttrFromIndex(1);

        expect(validateParam(node, slice_join.json)).toBeTruthy();
      });
      it('should call tfOps.slice', () => {
        node.op = 'Slice';
        node.inputParams.begin = createNumericArrayAttrFromIndex(1);
        node.inputParams.size = createNumericArrayAttrFromIndex(2);
        const input6 = [tfOps.tensor1d([2], 'int32')];
        node.inputNames = ['input1', 'input6', 'input4'];
        spyOps.slice.and.returnValue({});

        executeOp(node, {input1, input6, input4}, context, spyOpsAsTfOps);

        expect(spyOps.slice).toHaveBeenCalledWith(input1[0], [2], [3]);
      });
      it('should match json def for slice', () => {
        node.op = 'Slice';
        node.inputParams.begin = createNumericArrayAttrFromIndex(1);
        node.inputParams.size = createNumericArrayAttrFromIndex(2);

        expect(validateParam(node, slice_join.json)).toBeTruthy();
      });
      it('should call tfOps.stridedSlice', () => {
        node.op = 'StridedSlice';
        node.inputParams.begin = createNumericArrayAttrFromIndex(1);
        node.inputParams.end = createNumericArrayAttrFromIndex(2);
        node.inputParams.strides = createNumericArrayAttrFromIndex(3);
        node.attrParams.beginMask = createNumberAttr(4);
        node.attrParams.endMask = createNumberAttr(5);
        node.attrParams.ellipsisMask = createNumberAttr(1);
        node.attrParams.newAxisMask = createNumberAttr(2);
        node.attrParams.shrinkAxisMask = createNumberAttr(3);
        node.inputNames = ['input1', 'input6', 'input7', 'input4'];
        const input6 = [tfOps.tensor1d([2], 'int32')];
        const input7 = [tfOps.tensor1d([3], 'int32')];
        executeOp(
            node, {input1, input6, input7, input4}, context, spyOpsAsTfOps);

        expect(spyOps.stridedSlice)
            .toHaveBeenCalledWith(input1[0], [2], [3], [3], 4, 5, 1, 2, 3);
      });
      it('should match json def for stridedSlice', () => {
        node.op = 'StridedSlice';
        node.inputParams.begin = createNumericArrayAttrFromIndex(1);
        node.inputParams.end = createNumericArrayAttrFromIndex(2);
        node.inputParams.strides = createNumericArrayAttrFromIndex(3);
        node.attrParams.beginMask = createNumberAttr(4);
        node.attrParams.endMask = createNumberAttr(5);
        node.attrParams.ellipsisMask = createNumberAttr(1);
        node.attrParams.newAxisMask = createNumberAttr(2);
        node.attrParams.shrinkAxisMask = createNumberAttr(3);

        expect(validateParam(node, slice_join.json)).toBeTruthy();
      });
      it('should call tfOps.gather', () => {
        node.op = 'Gather';
        node.inputParams.indices = createTensorAttr(1);
        const input5 = [tfOps.scalar(2, 'int32')];
        node.inputNames = ['input1', 'input5'];
        spyOps.gather.and.returnValue({});
        executeOp(node, {input1, input5, input3}, context, spyOpsAsTfOps);

        expect(spyOps.gather)
            .toHaveBeenCalledWith(
                input1[0], jasmine.objectContaining({dataId: input5[0].dataId}),
                0);
      });
      it('should match json def for gather', () => {
        node.op = 'Gather';
        node.inputParams.indices = createTensorAttr(1);

        expect(validateParam(node, slice_join.json, 'Gather')).toBeTruthy();
      });
      it('should call tfOps.gather', () => {
        node.op = 'GatherV2';
        node.inputParams.indices = createTensorAttr(1);
        node.inputParams.axis = createNumberAttrFromIndex(2);
        node.attrParams.batchDims = createNumberAttr(1);
        const input5 = [tfOps.scalar(2, 'int32')];
        node.inputNames = ['input1', 'input5', 'input3'];
        spyOps.gather.and.returnValue({});
        executeOp(node, {input1, input5, input3}, context, spyOpsAsTfOps);

        expect(spyOps.gather)
            .toHaveBeenCalledWith(
                input1[0], jasmine.objectContaining({dataId: input5[0].dataId}),
                3, 1);
      });

      it('should make indices param of int32 dtype', () => {
        node.op = 'Gather';
        node.inputParams.indices = createTensorAttr(1);
        node.inputNames = ['input1', 'input5'];
        const input5 = [tfOps.scalar(2, 'float32')];
        spyOps.gather.and.returnValue({});
        executeOp(node, {input1, input5}, context, spyOpsAsTfOps);

        expect(spyOps.gather)
            .toHaveBeenCalledWith(
                input1[0], jasmine.objectContaining({dtype: 'int32'}), 0);
      });
      it('should match json def for gather', () => {
        node.op = 'GatherV2';
        node.inputParams.indices = createTensorAttr(1);
        node.inputParams.axis = createNumberAttrFromIndex(2);
        node.attrParams.batchDims = createNumberAttr(1);

        expect(validateParam(node, slice_join.json, 'GatherV2')).toBeTruthy();
      });
      it('should call tfOps.split', () => {
        node.op = 'Split';
        node.inputParams.axis = createNumberAttrFromIndex(0);
        node.inputParams.x = createTensorAttr(1);
        node.attrParams.numOrSizeSplits = createNumberAttr(2);
        node.inputNames = ['input1', 'input2'];
        spyOps.split.and.returnValue({});
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.split).toHaveBeenCalledWith(input2[0], 2, 1);
      });
      it('should match json def for split', () => {
        node.op = 'Split';
        node.inputParams.axis = createNumberAttrFromIndex(0);
        node.inputParams.x = createTensorAttr(1);
        node.attrParams.numOrSizeSplits = createNumberAttr(2);

        expect(validateParam(node, slice_join.json, 'Split')).toBeTruthy();
      });
      it('should call tfOps.split', () => {
        node.op = 'SplitV';
        node.inputParams.x = createTensorAttr(0);
        node.inputParams.numOrSizeSplits = createNumericArrayAttrFromIndex(1);
        node.inputParams.axis = createNumberAttrFromIndex(2);
        node.inputNames = ['input1', 'input2', 'input3'];
        spyOps.split.and.returnValue({});
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.split).toHaveBeenCalledWith(input1[0], 2, 3);
      });
      it('should match json def for split', () => {
        node.op = 'SplitV';
        node.inputParams.x = createTensorAttr(0);
        node.inputParams.numOrSizeSplits = createNumericArrayAttrFromIndex(1);
        node.inputParams.axis = createNumberAttrFromIndex(2);

        expect(validateParam(node, slice_join.json, 'SplitV')).toBeTruthy();
      });
      it('should call tfOps.scatterND', () => {
        node.op = 'ScatterNd';
        node.inputParams.indices = createTensorAttr(0);
        node.inputParams.values = createTensorAttr(1);
        node.inputParams.shape = createNumericArrayAttrFromIndex(2);
        node.inputNames = ['input1', 'input2', 'input4'];
        spyOps.scatterND.and.returnValue({});
        executeOp(node, {input1, input2, input4}, context, spyOpsAsTfOps);

        expect(spyOps.scatterND).toHaveBeenCalledWith(input1[0], input2[0], [
          3
        ]);
      });
      it('should match json def for scatterND', () => {
        node.op = 'ScatterNd';
        delete node.inputParams.x;
        node.inputParams.indices = createTensorAttr(0);
        node.inputParams.values = createTensorAttr(1);
        node.inputParams.shape = createNumericArrayAttrFromIndex(2);

        expect(validateParam(node, slice_join.json)).toBeTruthy();
      });
      it('should call tfOps.gatherND', () => {
        node.op = 'GatherNd';
        node.inputParams.x = createTensorAttr(0);
        node.inputParams.indices = createTensorAttr(1);
        node.inputNames = ['input1', 'input2'];
        spyOps.gatherND.and.returnValue({});
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.gatherND).toHaveBeenCalledWith(input1[0], input2[0]);
      });
      it('should match json def for gatherND', () => {
        node.op = 'GatherNd';
        node.inputParams.x = createTensorAttr(0);
        node.inputParams.indices = createTensorAttr(1);

        expect(validateParam(node, slice_join.json)).toBeTruthy();
      });
      it('should call tfOps.sparseToDense', () => {
        node.op = 'SparseToDense';
        node.inputParams.sparseIndices = createTensorAttr(0);
        node.inputParams.outputShape = createNumericArrayAttrFromIndex(1);
        node.inputParams.sparseValues = createTensorAttr(2);
        node.inputParams.defaultValue = createTensorAttr(3);
        node.inputParams.indices = createTensorAttr(1);
        node.inputNames = ['input1', 'input4', 'input3', 'input2'];
        spyOps.sparseToDense.and.returnValue({});
        executeOp(
            node, {input1, input2, input3, input4}, context, spyOpsAsTfOps);

        expect(spyOps.sparseToDense)
            .toHaveBeenCalledWith(input1[0], input3[0], [3], input2[0]);
      });
      it('should make defaultValue of same dtype as sparseValues', () => {
        node.op = 'SparseToDense';
        node.inputParams.sparseIndices = createTensorAttr(0);
        node.inputParams.outputShape = createNumericArrayAttrFromIndex(1);
        node.inputParams.sparseValues = createTensorAttr(2);
        node.inputParams.defaultValue = createTensorAttr(3);
        node.inputParams.indices = createTensorAttr(1);
        const input5 = [tfOps.scalar(5, 'int32')];
        node.inputNames = ['input1', 'input4', 'input3', 'input5'];
        spyOps.sparseToDense.and.returnValue({});
        executeOp(
            node, {input1, input5, input3, input4}, context, spyOpsAsTfOps);

        expect(spyOps.sparseToDense)
            .toHaveBeenCalledWith(
                input1[0], input3[0], [3],
                jasmine.objectContaining({dtype: 'float32'}));
      });
      it('should match json def for sparseToDense', () => {
        node.op = 'SparseToDense';
        node.inputParams = {};
        node.inputParams.sparseIndices = createTensorAttr(0);
        node.inputParams.outputShape = createNumericArrayAttrFromIndex(1);
        node.inputParams.sparseValues = createTensorAttr(2);
        node.inputParams.defaultValue = createTensorAttr(3);

        expect(validateParam(node, slice_join.json)).toBeTruthy();
      });
    });
  });
});
