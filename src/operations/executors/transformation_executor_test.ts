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

// tslint:disable-next-line:max-line-length
import {createDtypeAttr, createNumberAttr, createNumericArrayAttrFromIndex, createStrAttr, createTensorAttr} from './test_helper';
import {executeOp} from './transformation_executor';

describe('transformation', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const input2 = [tfc.tensor1d([1, 1])];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'transformation',
      inputNames: ['input1'],
      inputs: [],
      inputParams: {x: createTensorAttr(0)},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('Cast', () => {
      it('should call tfc.cast', () => {
        spyOn(tfc, 'cast');
        node.op = 'Cast';
        node.attrParams.dtype = createDtypeAttr('float32');
        executeOp(node, {input1}, context);

        expect(tfc.cast).toHaveBeenCalledWith(input1[0], 'float32');
      });
    });
    describe('expandDExpandDimsims', () => {
      it('should call tfc.expandDims', () => {
        spyOn(tfc, 'expandDims');
        node.op = 'ExpandDims';
        node.attrParams.axis = createNumberAttr(1);
        executeOp(node, {input1}, context);

        expect(tfc.expandDims).toHaveBeenCalledWith(input1[0], 1);
      });
    });
    describe('Pad', () => {
      it('should call tfc.pad', () => {
        spyOn(tfc, 'pad');
        node.op = 'Pad';
        node.inputParams.padding = createNumericArrayAttrFromIndex(1);
        node.attrParams.constantValue = createNumberAttr(1);
        node.inputNames = ['input1', 'input3'];
        const input3 = [tfc.tensor2d([1, 1, 2, 2], [2, 2])];
        executeOp(node, {input1, input3}, context);

        expect(tfc.pad).toHaveBeenCalledWith(input1[0], [[1, 1], [2, 2]], 1);
      });
    });
    describe('PadV2', () => {
      it('should call tfc.pad', () => {
        spyOn(tfc, 'pad');
        node.op = 'PadV2';
        node.inputParams.padding = createNumericArrayAttrFromIndex(1);
        node.attrParams.constantValue = createNumberAttr(1);
        node.inputNames = ['input1', 'input3'];
        const input3 = [tfc.tensor2d([1, 1, 2, 2], [2, 2])];
        executeOp(node, {input1, input3}, context);

        expect(tfc.pad).toHaveBeenCalledWith(input1[0], [[1, 1], [2, 2]], 1);
      });
    });
    describe('Reshape', () => {
      it('should call tfc.reshape', () => {
        spyOn(tfc, 'reshape');
        node.op = 'Reshape';
        node.inputParams.shape = createNumericArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.reshape).toHaveBeenCalledWith(input1[0], [1, 1]);
      });
    });
    describe('Squeeze', () => {
      it('should call tfc.squeeze', () => {
        spyOn(tfc, 'squeeze');
        node.op = 'Squeeze';
        node.attrParams.axis = createNumberAttr(1);
        executeOp(node, {input1}, context);

        expect(tfc.squeeze).toHaveBeenCalledWith(input1[0], 1);
      });
    });
    describe('SpaceToBatchND', () => {
      it('should call tfc.spaceToBatchND', () => {
        spyOn(tfc, 'spaceToBatchND');
        node.op = 'SpaceToBatchND';
        node.inputParams.blockShape = createNumericArrayAttrFromIndex(1);
        node.inputParams.paddings = createNumericArrayAttrFromIndex(2);
        node.inputNames = ['input1', 'input2', 'input3'];
        const input2 = [tfc.tensor1d([1, 1, 2, 2])];
        const input3 = [tfc.tensor2d([1, 2, 2, 3, 2, 3, 3, 4], [4, 2])];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfc.spaceToBatchND)
            .toHaveBeenCalledWith(
                input1[0], [1, 1, 2, 2], [[1, 2], [2, 3], [2, 3], [3, 4]]);
      });
    });
    describe('BatchToSpaceND', () => {
      it('should call tfc.batchToSpaceND', () => {
        spyOn(tfc, 'batchToSpaceND');
        node.op = 'BatchToSpaceND';
        node.inputParams.blockShape = createNumericArrayAttrFromIndex(1);
        node.inputParams.crops = createNumericArrayAttrFromIndex(2);
        node.inputNames = ['input1', 'input2', 'input3'];
        const input2 = [tfc.tensor1d([1, 1, 2, 2])];
        const input3 = [tfc.tensor2d([1, 2, 2, 3, 2, 3, 3, 4], [4, 2])];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfc.batchToSpaceND)
            .toHaveBeenCalledWith(
                input1[0], [1, 1, 2, 2], [[1, 2], [2, 3], [2, 3], [3, 4]]);
      });
    });
    describe('DepthToSpace', () => {
      it('should call tfc.depthToSpace', () => {
        spyOn(tfc, 'depthToSpace');
        node.op = 'DepthToSpace';
        node.attrParams.blockSize = createNumberAttr(1);
        node.attrParams.dataFormat = createStrAttr('nhwc');
        node.inputNames = ['input1'];
        executeOp(node, {input1}, context);

        expect(tfc.depthToSpace).toHaveBeenCalledWith(input1[0], 1, 'NHWC');
      });
    });
  });
});
