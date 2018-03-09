/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import * as dl from 'deeplearn';

import {Node} from '../index';

import {executeOp} from './normalization_executor';
import {createNumberAttr, createTensorAttr} from './test_helper';

describe('normalization', () => {
  let node: Node;
  const input1 = dl.scalar(1);

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'normalization',
      inputNames: ['input1'],
      inputs: [],
      params: {x: createTensorAttr(0)},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('batchNormalization', () => {
      it('should call dl.batchNormalization', () => {
        spyOn(dl, 'batchNormalization');
        node.op = 'batchNormalization';
        node.params.scale = createTensorAttr(1);
        node.params.offset = createTensorAttr(2);
        node.params.mean = createTensorAttr(3);
        node.params.variance = createTensorAttr(4);
        node.params.epislon = createNumberAttr(5);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];
        const input2 = dl.scalar(1);
        const input3 = dl.scalar(2);
        const input4 = dl.scalar(3);
        const input5 = dl.scalar(4);
        executeOp(node, {input1, input2, input3, input4, input5});

        expect(dl.batchNormalization)
            .toHaveBeenCalledWith(input1, input4, input5, 5, input2, input3);
      });
    });

    describe('localResponseNormalization', () => {
      it('should call dl.localResponseNormalization', () => {
        spyOn(dl, 'localResponseNormalization');
        node.op = 'localResponseNormalization';
        node.params.radius = createNumberAttr(1);
        node.params.bias = createNumberAttr(2);
        node.params.alpha = createNumberAttr(3);
        node.params.beta = createNumberAttr(4);

        executeOp(node, {input1});

        expect(dl.localResponseNormalization)
            .toHaveBeenCalledWith(input1, 1, 2, 3, 4);
      });
    });

    describe('softmax', () => {
      it('should call dl.softmax', () => {
        spyOn(dl, 'softmax');
        node.op = 'softmax';

        executeOp(node, {input1});

        expect(dl.softmax).toHaveBeenCalledWith(input1);
      });
    });
  });
});
