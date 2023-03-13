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

import {Tensor, test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {ExecutionContext} from '../../executor/execution_context';
import {Node} from '../types';

import {executeOp} from './arithmetic_executor';
import {createTensorAttr, createTensorsAttr, uncapitalize} from './test_helper';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';

describe('arithmetic', () => {
  let node: Node;
  const input1 = [tfOps.scalar(1)];
  const input2 = [tfOps.scalar(1)];
  const input3 = [tfOps.scalar(4)];
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'arithmetic',
      inputNames: ['input1', 'input2'],
      inputs: [],
      inputParams: {a: createTensorAttr(0), b: createTensorAttr(1)},
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

    (['Add', 'Mul', 'Div', 'Sub', 'Maximum', 'Minimum', 'Pow',
      'SquaredDifference', 'Mod', 'FloorDiv', 'DivNoNan'] as const)
         .forEach((op => {
           it('should call tfOps.' + op, () => {
             node.op = op;
             executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

             expect(spyOps[uncapitalize(op)])
               .toHaveBeenCalledWith(input1[0], input2[0]);
           });
         }));

    it('AddV2', async () => {

      node.op = 'AddV2';
      const res = executeOp(node, {input1, input2}, context,
                            spyOpsAsTfOps) as Tensor[];
      expect(spyOps.add).toHaveBeenCalledWith(input1[0], input2[0]);
      expect(res[0].dtype).toBe('float32');
      expect(res[0].shape).toEqual([]);
      test_util.expectArraysClose(await res[0].data(), 2);
    });

    it('AddN', async () => {
      node.op = 'AddN';
      node.inputParams = {tensors: createTensorsAttr(0, 0)};
      node.inputNames = ['input1', 'input2', 'input3'];
      const res =
        executeOp(node, {input1, input2, input3}, context,
                  spyOpsAsTfOps) as Tensor[];
      expect(spyOps.addN)
        .toHaveBeenCalledWith([input1[0], input2[0], input3[0]]);
      expect(res[0].dtype).toBe('float32');
      expect(res[0].shape).toEqual([]);
      test_util.expectArraysClose(await res[0].data(), [6]);
    });
  });
});
