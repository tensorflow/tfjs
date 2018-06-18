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

import {executeOp} from './arithmetic_executor';
import {createTensorAttr} from './test_helper';

describe('arithmetic', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const input2 = [tfc.scalar(1)];
  const context = new ExecutionContext({});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'arithmetic',
      inputNames: ['input1', 'input2'],
      inputs: [],
      params: {a: createTensorAttr(0), b: createTensorAttr(1)},
      children: []
    };
  });

  describe('executeOp', () => {
    ['add', 'mul', 'div', 'sub', 'maximum', 'minimum', 'pow',
     'squaredDifference', 'mod', 'floorDiv']
        .forEach((op => {
          it('should call tfc.' + op, () => {
            const spy = spyOn(tfc, op as 'add');
            node.op = op;
            executeOp(node, {input1, input2}, context);

            expect(spy).toHaveBeenCalledWith(input1[0], input2[0]);
          });
        }));
  });
});
