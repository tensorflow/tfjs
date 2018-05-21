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

import {executeOp} from './convolution_executor';
// tslint:disable-next-line:max-line-length
import {createNumberAttr, createNumericArrayAttr, createStrAttr, createTensorAttr} from './test_helper';

describe('convolution', () => {
  let node: Node;
  const input = [tfc.scalar(1)];
  const context = new ExecutionContext({});

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
      it('should call tfc.avgPool', () => {
        spyOn(tfc, 'avgPool');
        node.op = 'avgPool';
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfc.avgPool)
            .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
      });
    });

    describe('maxPool', () => {
      it('should call tfc.maxPool', () => {
        spyOn(tfc, 'maxPool');
        node.op = 'maxPool';
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfc.maxPool)
            .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
      });
    });
    describe('Conv2d', () => {
      it('should call tfc.conv2d', () => {
        spyOn(tfc, 'conv2d');
        node.op = 'conv2d';
        node.params['filter'] = createTensorAttr(1);
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['dataFormat'] = createStrAttr('NHWC');
        node.params['dilations'] = createNumericArrayAttr([2, 2]);

        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.conv2d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
      });
    });
    describe('conv2dTranspose', () => {
      it('should call tfc.conv2dTranspose', () => {
        spyOn(tfc, 'conv2dTranspose');
        node.op = 'conv2dTranspose';
        node.params['outputShape'] = createNumericArrayAttr([1, 2, 2, 2]);
        node.params['filter'] = createTensorAttr(1);
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');

        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.conv2dTranspose)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [1, 2, 2, 2], [2, 2], 'same');
      });
    });
    describe('Conv1d', () => {
      it('should call tfc.conv1d', () => {
        spyOn(tfc, 'conv1d');
        node.op = 'conv1d';
        node.category = 'convolution';
        node.params['filter'] = createTensorAttr(1);
        node.params['stride'] = createNumberAttr(1);
        node.params['pad'] = createStrAttr('same');
        node.params['dataFormat'] = createStrAttr('NWC');
        node.params['dilation'] = createNumberAttr(1);

        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.conv1d)
            .toHaveBeenCalledWith(input1[0], input2[0], 1, 'same', 'NWC', 1);
      });
    });

    describe('depthwiseConv2d', () => {
      it('should call tfc.depthwiseConv2d', () => {
        spyOn(tfc, 'depthwiseConv2d');
        node.op = 'depthwiseConv2d';
        node.category = 'convolution';
        node.params['input'] = createTensorAttr(0);
        node.params['filter'] = createTensorAttr(1);
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['dataFormat'] = createStrAttr('NHWC');
        node.params['dilations'] = createNumericArrayAttr([2, 2]);
        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.depthwiseConv2d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
      });
    });
  });
});
