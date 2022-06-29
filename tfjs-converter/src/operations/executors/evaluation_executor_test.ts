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

import {executeOp} from './evaluation_executor';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';
import {createBoolAttr, createNumberAttrFromIndex, createTensorAttr} from './test_helper';

describe('evaluation', () => {
  let node: Node;
  const input1 = [tfOps.tensor1d([1])];
  const input2 = [tfOps.scalar(1)];
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'input1',
      op: '',
      category: 'evaluation',
      inputNames: ['input1', 'input2'],
      inputs: [],
      inputParams: {},
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

    describe('LowerBound', () => {
      it('should return input', () => {
        node.op = 'LowerBound';
        node.inputParams['sortedSequence'] = createTensorAttr(0);
        node.inputParams['values'] = createTensorAttr(1);
        node.inputNames = ['sortedSequence', 'values'];

        const sortedSequence = [tfOps.tensor2d(
            [0., 3., 8., 9., 10., 1., 2., 3., 4., 5.], [2, 5], 'int32')];
        const values = [tfOps.tensor2d(
            [
              9.8,
              2.1,
              4.3,
              0.1,
              6.6,
              4.5,
            ],
            [2, 3], 'float32')];
        executeOp(node, {sortedSequence, values}, context, spyOpsAsTfOps);
        expect(spyOps.lowerBound)
            .toHaveBeenCalledWith(sortedSequence[0], values[0]);
      });
    });

    describe('TopKV2', () => {
      it('should return input', () => {
        node.op = 'TopKV2';
        node.inputParams['x'] = createTensorAttr(0);
        node.inputParams['k'] = createNumberAttrFromIndex(1);
        node.attrParams['sorted'] = createBoolAttr(true);
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);
        expect(spyOps.topk).toHaveBeenCalledWith(input1[0], 1, true);
      });
    });

    describe('UpperBound', () => {
      it('should return input', () => {
        node.op = 'UpperBound';
        node.inputParams['sortedSequence'] = createTensorAttr(0);
        node.inputParams['values'] = createTensorAttr(1);
        node.inputNames = ['sortedSequence', 'values'];

        const sortedSequence = [tfOps.tensor2d(
            [0., 3., 8., 9., 10., 1., 2., 3., 4., 5.], [2, 5], 'int32')];
        const values = [tfOps.tensor2d(
            [
              9.8,
              2.1,
              4.3,
              0.1,
              6.6,
              4.5,
            ],
            [2, 3], 'float32')];
        executeOp(node, {sortedSequence, values}, context, spyOpsAsTfOps);
        expect(spyOps.upperBound)
            .toHaveBeenCalledWith(sortedSequence[0], values[0]);
      });
    });

    describe('Unique', () => {
      it('should get called correctly', () => {
        node.op = 'Unique';
        node.inputParams['x'] = createTensorAttr(0);
        executeOp(node, {input1}, context, spyOpsAsTfOps);
        expect(spyOps.unique).toHaveBeenCalledWith(input1[0]);
      });
    });

    describe('UniqueV2', () => {
      it('should get called correctly', () => {
        node.op = 'UniqueV2';
        node.inputParams['x'] = createTensorAttr(0);
        node.inputParams['axis'] = createNumberAttrFromIndex(1);
        const xInput = [tfOps.tensor2d([[1], [2]])];
        const axisInput = [tfOps.scalar(1)];
        executeOp(
            node, {'input1': xInput, 'input2': axisInput}, context,
            spyOpsAsTfOps);
        expect(spyOps.unique).toHaveBeenCalledWith(xInput[0], 1);
      });
    });
  });
});
