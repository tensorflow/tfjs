/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {executeOp} from './evaluation_executor';
// tslint:disable-next-line:max-line-length
import {createBoolAttr, createNumberAttrFromIndex, createTensorAttr} from './test_helper';

describe('evaluation', () => {
  let node: Node;
  const input1 = [tfc.tensor1d([1])];
  const input2 = [tfc.scalar(1)];
  const context = new ExecutionContext({}, {});

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
    describe('TopKV2', () => {
      it('should return input', () => {
        node.op = 'TopKV2';
        node.inputParams['x'] = createTensorAttr(0);
        node.inputParams['k'] = createNumberAttrFromIndex(1);
        node.attrParams['sorted'] = createBoolAttr(true);
        spyOn(tfc, 'topk').and.callThrough();
        executeOp(node, {input1, input2}, context);
        expect(tfc.topk).toHaveBeenCalledWith(input1[0], 1, true);
      });
    });
  });
});
