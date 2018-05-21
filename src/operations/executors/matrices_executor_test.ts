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

import {executeOp} from './matrices_executor';
// tslint:disable-next-line:max-line-length
import {createBoolAttr, createNumericArrayAttr, createTensorAttr} from './test_helper';

describe('matrices', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const input2 = [tfc.scalar(2)];
  const context = new ExecutionContext({});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'matrices',
      inputNames: ['input1', 'input2'],
      inputs: [],
      params: {a: createTensorAttr(0), b: createTensorAttr(1)},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('matMul', () => {
      it('should call tfc.matMul', () => {
        spyOn(tfc, 'matMul');
        node.op = 'matMul';
        node.params.transposeA = createBoolAttr(true);
        node.params.transposeB = createBoolAttr(false);
        executeOp(node, {input1, input2}, context);

        expect(tfc.matMul)
            .toHaveBeenCalledWith(input1[0], input2[0], true, false);
      });
    });

    describe('transpose', () => {
      it('should call tfc.transpose', () => {
        spyOn(tfc, 'transpose');
        node.op = 'transpose';
        node.inputNames = ['input1', 'input2', 'input3'];
        node.params = {
          x: createTensorAttr(0),
          perm: createNumericArrayAttr([1, 2])
        };
        executeOp(node, {input1}, context);

        expect(tfc.transpose).toHaveBeenCalledWith(input1[0], [1, 2]);
      });
    });
  });
});
