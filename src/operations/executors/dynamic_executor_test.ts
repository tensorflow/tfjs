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

import {executeOp} from './dynamic_executor';
// tslint:disable-next-line:max-line-length
import {createNumberAttr, createTensorAttr} from './test_helper';

describe('dynamic', () => {
  let node: Node;
  const input1 = [tfc.tensor1d([1])];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'input1',
      op: '',
      category: 'image',
      inputNames: ['input1'],
      inputs: [],
      params: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('nonMaxSuppression', () => {
      it('should return input', () => {
        node.op = 'nonMaxSuppression';
        node.params['boxes'] = createTensorAttr(0);
        node.params['scores'] = createTensorAttr(1);
        node.params['maxOutputSize'] = createTensorAttr(2);
        node.params['iouThreshold'] = createTensorAttr(3);
        node.params['scoreThreshold'] = createNumberAttr(1);
        node.inputNames = ['input1', 'input2', 'input3', 'input4'];
        const input2 = [tfc.tensor1d([1])];
        const input3 = [tfc.tensor1d([1])];
        const input4 = [tfc.tensor1d([1])];
        spyOn(tfc.image, 'nonMaxSuppressionAsync').and.callThrough();
        const result =
            executeOp(node, {input1, input2, input3, input4}, context);
        expect(tfc.image.nonMaxSuppressionAsync)
            .toHaveBeenCalledWith(
                input1[0], input2[0], input3[0], input4[0], 1);
        expect(result instanceof Promise).toBeTruthy();
      });
    });

    describe('whereAsync', () => {
      it('should call tfc.whereAsync', async () => {
        node.op = 'whereAsync';
        const input1 = [tfc.scalar(1)];
        node.params = {'condition': createTensorAttr(0)};
        spyOn(tfc, 'whereAsync').and.callThrough();

        const result = executeOp(node, {input1}, context);
        expect(tfc.whereAsync).toHaveBeenCalledWith(input1[0]);
        expect(result instanceof Promise).toBeTruthy();
      });
    });
  });
});
