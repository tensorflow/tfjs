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
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'convolution',
      inputNames: ['input'],
      inputs: [],
      inputParams: {x: createTensorAttr(0)},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('AvgPool', () => {
      it('should call tfc.avgPool', () => {
        spyOn(tfc, 'avgPool');
        node.op = 'AvgPool';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfc.avgPool)
            .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
      });
    });

    describe('maxPool', () => {
      it('should call tfc.maxPool', () => {
        spyOn(tfc, 'maxPool');
        node.op = 'MaxPool';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfc.maxPool)
            .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
      });
    });
    describe('Conv2d', () => {
      it('should call tfc.conv2d', () => {
        spyOn(tfc, 'conv2d');
        node.op = 'Conv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([2, 2]);

        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.conv2d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
      });
    });
    describe('Conv2DBackpropInput', () => {
      it('should call tfc.conv2dTranspose', () => {
        spyOn(tfc, 'conv2dTranspose');
        node.op = 'Conv2DBackpropInput';
        node.attrParams['outputShape'] = createNumericArrayAttr([1, 2, 2, 2]);
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');

        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.conv2dTranspose)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [1, 2, 2, 2], [2, 2], 'same');
      });
    });
    describe('Conv1D', () => {
      it('should call tfc.conv1d', () => {
        spyOn(tfc, 'conv1d');
        node.op = 'Conv1D';
        node.category = 'convolution';
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['stride'] = createNumberAttr(1);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NWC');
        node.attrParams['dilation'] = createNumberAttr(1);

        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.conv1d)
            .toHaveBeenCalledWith(input1[0], input2[0], 1, 'same', 'NWC', 1);
      });
    });

    describe('DepthwiseConv2d', () => {
      it('should call tfc.depthwiseConv2d', () => {
        spyOn(tfc, 'depthwiseConv2d');
        node.op = 'DepthwiseConv2d';
        node.category = 'convolution';
        node.inputParams['input'] = createTensorAttr(0);
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([2, 2]);
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
