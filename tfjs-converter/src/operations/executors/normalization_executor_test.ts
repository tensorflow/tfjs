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
import * as normalization from '../op_list/normalization';
import {Node} from '../types';

import {executeOp} from './normalization_executor';
import {createBoolAttr, createNumberAttr, createNumericArrayAttrFromIndex, createTensorAttr, validateParam} from './test_helper';
import {spyOnAllFunctions, RecursiveSpy} from './spy_ops';

describe('normalization', () => {
  let node: Node;
  const input1 = [tfOps.scalar(1)];
  const context = new ExecutionContext({}, {}, {});
  let spyOps: RecursiveSpy<typeof tfOps>;
  let spyOpsAsTfOps: typeof tfOps;

  beforeEach(() => {
    spyOps = spyOnAllFunctions(tfOps);
    spyOpsAsTfOps = spyOps as unknown as typeof tfOps;
    node = {
      name: 'test',
      op: '',
      category: 'normalization',
      inputNames: ['input1'],
      inputs: [],
      inputParams: {x: createTensorAttr(0)},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('EuclideanNorm', () => {
      it('should call tfOps.euclideanNorm', () => {
        node.op = 'EuclideanNorm';
        node.inputParams['axis'] = createNumericArrayAttrFromIndex(1);
        node.attrParams.keepDims = createBoolAttr(false);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfOps.tensor1d([2])];
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.euclideanNorm).toHaveBeenCalledWith(input1[0], [2],
            false);
      });
      it('should match json def', () => {
        node.op = 'EuclideanNorm';
        delete node.inputParams.x;
        node.inputParams.axis = createNumericArrayAttrFromIndex(1);
        node.attrParams.keepDims = createBoolAttr(false);

        expect(validateParam(node, normalization.json)).toBeTruthy();
      });
    });
    describe('FusedBatchNorm', () => {
      it('should call tfOps.batchNorm', () => {
        node.op = 'FusedBatchNorm';
        node.inputParams.scale = createTensorAttr(1);
        node.inputParams.offset = createTensorAttr(2);
        node.inputParams.mean = createTensorAttr(3);
        node.inputParams.variance = createTensorAttr(4);
        node.attrParams.epsilon = createNumberAttr(5);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];
        const input2 = [tfOps.scalar(1)];
        const input3 = [tfOps.scalar(2)];
        const input4 = [tfOps.scalar(3)];
        const input5 = [tfOps.scalar(4)];
        executeOp(node, {input1, input2, input3, input4, input5}, context,
                  spyOpsAsTfOps);

        expect(spyOps.batchNorm)
            .toHaveBeenCalledWith(
                input1[0], input4[0], input5[0], input3[0], input2[0], 5);
      });
    });
    describe('FusedBatchNormV2', () => {
      it('should call tfOps.batchNorm', () => {
        node.op = 'FusedBatchNormV2';
        node.inputParams.scale = createTensorAttr(1);
        node.inputParams.offset = createTensorAttr(2);
        node.inputParams.mean = createTensorAttr(3);
        node.inputParams.variance = createTensorAttr(4);
        node.attrParams.epsilon = createNumberAttr(5);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];
        const input2 = [tfOps.scalar(1)];
        const input3 = [tfOps.scalar(2)];
        const input4 = [tfOps.scalar(3)];
        const input5 = [tfOps.scalar(4)];
        executeOp(node, {input1, input2, input3, input4, input5}, context,
                  spyOpsAsTfOps);

        expect(spyOps.batchNorm)
            .toHaveBeenCalledWith(
                input1[0], input4[0], input5[0], input3[0], input2[0], 5);
      });
    });
    describe('FusedBatchNormV3', () => {
      it('should call tfOps.batchNorm', () => {
        node.op = 'FusedBatchNormV3';
        node.inputParams.scale = createTensorAttr(1);
        node.inputParams.offset = createTensorAttr(2);
        node.inputParams.mean = createTensorAttr(3);
        node.inputParams.variance = createTensorAttr(4);
        node.attrParams.epsilon = createNumberAttr(5);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];
        const input2 = [tfOps.scalar(1)];
        const input3 = [tfOps.scalar(2)];
        const input4 = [tfOps.scalar(3)];
        const input5 = [tfOps.scalar(4)];
        executeOp(node, {input1, input2, input3, input4, input5}, context,
                  spyOpsAsTfOps);

        expect(spyOps.batchNorm)
            .toHaveBeenCalledWith(
                input1[0], input4[0], input5[0], input3[0], input2[0], 5);
      });
    });
    describe('LRN', () => {
      it('should call tfOps.localResponseNormalization', () => {
        node.op = 'LRN';
        node.attrParams.radius = createNumberAttr(1);
        node.attrParams.bias = createNumberAttr(2);
        node.attrParams.alpha = createNumberAttr(3);
        node.attrParams.beta = createNumberAttr(4);
        spyOps.localResponseNormalization.and.returnValue({});

        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.localResponseNormalization)
            .toHaveBeenCalledWith(input1[0], 1, 2, 3, 4);
      });
      it('should match json def', () => {
        node.op = 'LRN';
        node.attrParams.radius = createNumberAttr(1);
        node.attrParams.bias = createNumberAttr(2);
        node.attrParams.alpha = createNumberAttr(3);
        node.attrParams.beta = createNumberAttr(4);

        expect(validateParam(node, normalization.json)).toBeTruthy();
      });
    });

    describe('Softmax', () => {
      it('should call tfOps.softmax', () => {
        node.op = 'Softmax';
        spyOps.softmax.and.returnValue({});

        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.softmax).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'Softmax';

        expect(validateParam(node, normalization.json)).toBeTruthy();
      });
    });

    describe('LogSoftmax', () => {
      it('should call tfOps.logSoftmax', () => {
        node.op = 'LogSoftmax';
        spyOps.logSoftmax.and.returnValue({});

        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.logSoftmax).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'LogSoftmax';

        expect(validateParam(node, normalization.json)).toBeTruthy();
      });
    });
  });
});
