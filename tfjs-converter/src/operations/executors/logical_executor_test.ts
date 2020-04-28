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

import {executeOp} from './logical_executor';
import {createTensorAttr} from './test_helper';

describe('logical', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const input2 = [tfc.scalar(2)];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'logical',
      inputNames: ['input1', 'input2'],
      inputs: [],
      inputParams: {a: createTensorAttr(0), b: createTensorAttr(1)},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    ['Equal', 'NotEqual', 'Greater', 'GreaterEqual', 'Less', 'LessEqual',
     'LogicalAnd', 'LogicalOr']
        .forEach(op => {
          it('should call tfc.' + op, () => {
            const spy =
                spyOn(tfc, op.charAt(0).toLowerCase() + op.slice(1) as 'equal');
            node.op = op;
            executeOp(node, {input1, input2}, context);

            expect(spy).toHaveBeenCalledWith(input1[0], input2[0]);
          });
        });
    describe('LogicalNot', () => {
      it('should call tfc.logicalNot', () => {
        spyOn(tfc, 'logicalNot');
        node.op = 'LogicalNot';
        executeOp(node, {input1}, context);

        expect(tfc.logicalNot).toHaveBeenCalledWith(input1[0]);
      });
    });

    describe('Select', () => {
      it('should call tfc.where', () => {
        spyOn(tfc, 'where');
        node.op = 'Select';
        node.inputNames = ['input1', 'input2', 'input3'];
        node.inputParams.condition = createTensorAttr(2);
        const input3 = [tfc.scalar(1)];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfc.where).toHaveBeenCalledWith(input3[0], input1[0], input2[0]);
      });
    });

    describe('SelectV2', () => {
      it('should call tfc.where', () => {
        spyOn(tfc, 'where');
        node.op = 'SelectV2';
        node.inputNames = ['input1', 'input2', 'input3'];
        node.inputParams.condition = createTensorAttr(2);
        const input3 = [tfc.scalar(1)];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfc.where).toHaveBeenCalledWith(input3[0], input1[0], input2[0]);
      });
    });
  });
});
