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

import {executeOp} from './convolution_executor';
// tslint:disable-next-line:max-line-length
import {createNumberAttr, createNumericArrayAttr, createStrAttr, createTensorAttr} from './test_helper';

describe('convolution', () => {
  let node: Node;
  const input = dl.scalar(1);

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'convolution',
      inputNames: ['input'],
      inputs: [],
      params: {x: createTensorAttr(0)},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('avgPool', () => {
      it('should call dl.avgPool', () => {
        spyOn(dl, 'avgPool');
        node.op = 'avgPool';
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);

        executeOp(node, {input});

        expect(dl.avgPool).toHaveBeenCalledWith(input, [2, 2], [2, 2], 'same');
      });
    });

    describe('maxPool', () => {
      it('should call dl.maxPool', () => {
        spyOn(dl, 'maxPool');
        node.op = 'maxPool';
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);

        executeOp(node, {input});

        expect(dl.maxPool).toHaveBeenCalledWith(input, [2, 2], [2, 2], 'same');
      });
    });
    describe('Conv2d', () => {
      it('should call dl.conv2d', () => {
        spyOn(dl, 'conv2d');
        node.op = 'conv2d';
        node.params['filter'] = createTensorAttr(1);
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');

        const input1 = dl.scalar(1.0);
        const input2 = dl.scalar(1.0);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.conv2d).toHaveBeenCalledWith(input1, input2, [2, 2], 'same');
      });
    });
    describe('conv2dTranspose', () => {
      it('should call dl.conv2dTranspose', () => {
        spyOn(dl, 'conv2dTranspose');
        node.op = 'conv2dTranspose';
        node.params['outputShape'] = createNumericArrayAttr([1, 2, 2, 2]);
        node.params['filter'] = createTensorAttr(1);
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');

        const input1 = dl.scalar(1.0);
        const input2 = dl.scalar(1.0);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.conv2dTranspose)
            .toHaveBeenCalledWith(input1, input2, [1, 2, 2, 2], [2, 2], 'same');
      });
    });
    describe('Conv1d', () => {
      it('should call dl.conv1d', () => {
        spyOn(dl, 'conv1d');
        node.op = 'conv1d';
        node.category = 'convolution';
        node.params['filter'] = createTensorAttr(1);
        node.params['stride'] = createNumberAttr(1);
        node.params['pad'] = createStrAttr('same');

        const input1 = dl.scalar(1.0);
        const input2 = dl.scalar(1.0);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.conv1d).toHaveBeenCalledWith(input1, input2, 1, 'same');
      });
    });

    describe('depthwiseConv2d', () => {
      it('should call dl.depthwiseConv2d', () => {
        spyOn(dl, 'depthwiseConv2d');
        node.op = 'depthwiseConv2d';
        node.category = 'convolution';
        node.params['input'] = createTensorAttr(0);
        node.params['filter'] = createTensorAttr(1);
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['rates'] = createNumericArrayAttr([2, 2]);
        const input1 = dl.scalar(1.0);
        const input2 = dl.scalar(1.0);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.depthwiseConv2d)
            .toHaveBeenCalledWith(input1, input2, [2, 2], 'same', [2, 2]);
      });
    });
  });
});
