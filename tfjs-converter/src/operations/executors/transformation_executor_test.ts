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
import {Node} from '../types';
import {createDtypeAttr, createNumberAttr, createNumericArrayAttrFromIndex, createStrAttr, createTensorAttr} from './test_helper';
import {executeOp} from './transformation_executor';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';

describe('transformation', () => {
  let node: Node;
  const input1 = [tfOps.scalar(1)];
  const input2 = [tfOps.tensor1d([1, 1])];
  const context = new ExecutionContext({}, {}, {});

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
    let spyOps: RecursiveSpy<typeof tfOps>;
    let spyOpsAsTfOps: typeof tfOps;

    beforeEach(() => {
      spyOps = spyOnAllFunctions(tfOps);
      spyOpsAsTfOps = spyOps as unknown as typeof tfOps;
    });

    describe('Cast', () => {
      it('should call tfOps.cast', () => {
        node.op = 'Cast';
        node.attrParams.dtype = createDtypeAttr('float32');
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.cast).toHaveBeenCalledWith(input1[0], 'float32');
      });
    });
    describe('ExpandDims', () => {
      it('should call tfOps.expandDims', () => {
        node.op = 'ExpandDims';
        node.attrParams.axis = createNumberAttr(1);
        spyOps.expandDims.and.returnValue({});
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.expandDims).toHaveBeenCalledWith(input1[0], 1);
      });
    });
    describe('MirrorPad', () => {
      it('should call tfc.mirrorPad', () => {
        node.op = 'MirrorPad';
        node.inputParams.padding = createNumericArrayAttrFromIndex(1);
        node.attrParams.mode = createStrAttr('reflect');
        node.inputNames = ['input1', 'input3'];
        const input3 = [tfOps.tensor2d([1, 1, 2, 2], [2, 2])];
        spyOps.mirrorPad.and.returnValue({});
        executeOp(node, {input1, input3}, context, spyOpsAsTfOps);

        expect(spyOps.mirrorPad)
            .toHaveBeenCalledWith(input1[0], [[1, 1], [2, 2]], 'reflect');
      });
    });
    describe('Pad', () => {
      it('should call tfOps.pad', () => {
        node.op = 'Pad';
        node.inputParams.padding = createNumericArrayAttrFromIndex(1);
        node.attrParams.constantValue = createNumberAttr(1);
        node.inputNames = ['input1', 'input3'];
        const input3 = [tfOps.tensor2d([1, 1, 2, 2], [2, 2])];
        spyOps.pad.and.returnValue({});
        executeOp(node, {input1, input3}, context, spyOpsAsTfOps);

        expect(spyOps.pad).toHaveBeenCalledWith(input1[0], [[1, 1], [2, 2]], 1);
      });
    });
    describe('PadV2', () => {
      it('should call tfOps.pad', () => {
        node.op = 'PadV2';
        node.inputParams.padding = createNumericArrayAttrFromIndex(1);
        node.attrParams.constantValue = createNumberAttr(1);
        node.inputNames = ['input1', 'input3'];
        const input3 = [tfOps.tensor2d([1, 1, 2, 2], [2, 2])];
        spyOps.pad.and.returnValue({});
        executeOp(node, {input1, input3}, context, spyOpsAsTfOps);

        expect(spyOps.pad).toHaveBeenCalledWith(input1[0], [[1, 1], [2, 2]], 1);
      });
    });
    describe('Reshape', () => {
      it('should call tfOps.reshape', () => {
        node.op = 'Reshape';
        node.inputParams.shape = createNumericArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.reshape).toHaveBeenCalledWith(input1[0], [1, 1]);
      });
    });
    describe('Squeeze', () => {
      it('should call tfOps.squeeze', () => {
        node.op = 'Squeeze';
        node.attrParams.axis = createNumberAttr(1);
        spyOps.squeeze.and.returnValue({});
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.squeeze).toHaveBeenCalledWith(input1[0], 1);
      });
    });
    describe('SpaceToBatchND', () => {
      it('should call tfOps.spaceToBatchND', () => {
        node.op = 'SpaceToBatchND';
        node.inputParams.blockShape = createNumericArrayAttrFromIndex(1);
        node.inputParams.paddings = createNumericArrayAttrFromIndex(2);
        node.inputNames = ['input1', 'input2', 'input3'];
        const input2 = [tfOps.tensor1d([1, 1, 2, 2])];
        const input3 = [tfOps.tensor2d([1, 2, 2, 3, 2, 3, 3, 4], [4, 2])];
        spyOps.spaceToBatchND.and.returnValue({});
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.spaceToBatchND)
            .toHaveBeenCalledWith(
                input1[0], [1, 1, 2, 2], [[1, 2], [2, 3], [2, 3], [3, 4]]);
      });
    });
    describe('BatchToSpaceND', () => {
      it('should call tfOps.batchToSpaceND', () => {
        node.op = 'BatchToSpaceND';
        node.inputParams.blockShape = createNumericArrayAttrFromIndex(1);
        node.inputParams.crops = createNumericArrayAttrFromIndex(2);
        node.inputNames = ['input1', 'input2', 'input3'];
        const input2 = [tfOps.tensor1d([1, 1, 2, 2])];
        const input3 = [tfOps.tensor2d([1, 2, 2, 3, 2, 3, 3, 4], [4, 2])];
        spyOps.batchToSpaceND.and.returnValue({});
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.batchToSpaceND)
            .toHaveBeenCalledWith(
                input1[0], [1, 1, 2, 2], [[1, 2], [2, 3], [2, 3], [3, 4]]);
      });
    });
    describe('DepthToSpace', () => {
      it('should call tfOps.depthToSpace', () => {
        node.op = 'DepthToSpace';
        node.attrParams.blockSize = createNumberAttr(1);
        node.attrParams.dataFormat = createStrAttr('nhwc');
        node.inputNames = ['input1'];
        spyOps.depthToSpace.and.returnValue({});
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.depthToSpace).toHaveBeenCalledWith(input1[0], 1, 'NHWC');
      });
    });
    describe('BroadcastTo', () => {
      it('should call tfOps.broadcastTo', () => {
        node.op = 'BroadcastTo';
        node.inputParams.shape = createNumericArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.broadcastTo).toHaveBeenCalledWith(input1[0], [1, 1]);
      });
    });
    describe('BroadcastArgs', () => {
      it('should call tfOps.broadcastArgs', () => {
        node.op = 'BroadcastArgs';
        node.inputParams.s0 = createTensorAttr(0);
        node.inputParams.s1 = createTensorAttr(1);
        node.inputNames = ['input1', 'input2'];
        const input1 = [tfOps.tensor1d([1, 1])];
        const input2 = [tfOps.tensor1d([1, 1])];
        spyOps.broadcastArgs.and.returnValue({});
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.broadcastArgs).toHaveBeenCalledWith(input1[0], input2[0]);
      });
    });
  });
});
