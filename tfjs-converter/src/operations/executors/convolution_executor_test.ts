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
import {createNumberAttr, createNumericArrayAttr, createStrArrayAttr, createStrAttr, createTensorAttr, createTensorsAttr} from './test_helper';
import {createBoolAttr} from './test_helper';

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
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);

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
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.depthwiseConv2d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
      });
    });

    describe('Conv3d', () => {
      it('should call tfc.conv3d', () => {
        spyOn(tfc, 'conv3d');
        node.op = 'Conv3D';
        node.category = 'convolution';
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 2, 1]);

        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfc.conv3d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2, 2], 'same', 'NHWC', [2, 2, 2]);
      });
    });

    describe('AvgPool3D', () => {
      it('should call tfc.avgPool3d', () => {
        spyOn(tfc, 'avgPool3d');
        node.op = 'AvgPool3D';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfc.avgPool3d)
            .toHaveBeenCalledWith(input[0], [2, 2, 2], [2, 2, 2], 'same');
      });
    });

    describe('MaxPool3D', () => {
      it('should call tfc.maxPool3d', () => {
        spyOn(tfc, 'maxPool3d');
        node.op = 'MaxPool3D';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfc.maxPool3d)
            .toHaveBeenCalledWith(input[0], [2, 2, 2], [2, 2, 2], 'same');
      });
    });

    describe('MaxPoolWithArgmax', () => {
      it('should call tfc.maxPoolWithArgmax', () => {
        spyOn(tfc, 'maxPoolWithArgmax');
        node.op = 'maxPoolWithArgmax';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 2, 1]);
        node.attrParams['dataFormat'] = createStrAttr('NDHWC');
        node.attrParams['includeBatchInIndex'] = createBoolAttr(true);
        executeOp(node, {input}, context);

        expect(tfc.maxPoolWithArgmax)
            .toHaveBeenCalledWith(
                input[0], [2, 2, 2], [2, 2, 2], 'NDHWC', 'same', true);
      });
    });

    describe('_FusedConv2d', () => {
      it('with bias and activation func', () => {
        spyOn(tfc.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'relu']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(1);
        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(2.0)];
        const input3 = [tfc.scalar(3.0)];

        node.inputNames = ['input1', 'input2', 'input3'];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfc.fused.conv2d).toHaveBeenCalledWith({
          x: input1[0],
          filter: input2[0],
          strides: [2, 2],
          pad: 'same',
          dataFormat: 'NHWC',
          dilations: [2, 2],
          bias: input3[0],
          activation: 'relu',
          preluActivationWeights: undefined
        });
      });

      it('with bias and prelu activation func', () => {
        spyOn(tfc.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'prelu']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(2);
        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(2.0)];
        const input3 = [tfc.scalar(3.0)];
        const input4 = [tfc.scalar(4.0)];
        node.inputNames = ['input1', 'input2', 'input3', 'input4'];
        executeOp(node, {input1, input2, input3, input4}, context);

        expect(tfc.fused.conv2d).toHaveBeenCalledWith({
          x: input1[0],
          filter: input2[0],
          strides: [2, 2],
          pad: 'same',
          dataFormat: 'NHWC',
          dilations: [2, 2],
          bias: input3[0],
          activation: 'prelu',
          preluActivationWeights: input4[0]
        });
      });

      it('bias add', () => {
        spyOn(tfc.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(1);
        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(2.0)];
        const input3 = [tfc.scalar(3.0)];

        node.inputNames = ['input1', 'input2', 'input3'];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfc.fused.conv2d).toHaveBeenCalledWith({
          x: input1[0],
          filter: input2[0],
          strides: [2, 2],
          pad: 'same',
          dataFormat: 'NHWC',
          dilations: [2, 2],
          bias: input3[0],
          activation: undefined,
          preluActivationWeights: undefined
        });
      });
      it('fail with batchnorm', () => {
        spyOn(tfc.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['fusedbatchnorm']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(1);
        const input1 = [tfc.scalar(1.0)];
        const input2 = [tfc.scalar(2.0)];
        const input3 = [tfc.scalar(3.0)];

        node.inputNames = ['input1', 'input2', 'input3'];
        expect(() => executeOp(node, {input1, input2, input3}, context))
            .toThrow();
      });
    });
  });
  describe('FusedDepthwiseConv2d', () => {
    it('with bias and activation func', () => {
      spyOn(tfc.fused, 'depthwiseConv2d');
      node.op = 'FusedDepthwiseConv2dNative';
      node.inputParams['filter'] = createTensorAttr(1);
      node.inputParams['args'] = createTensorsAttr(2, 0);
      node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'relu']);
      node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['pad'] = createStrAttr('same');
      node.attrParams['dataFormat'] = createStrAttr('NHWC');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['numArgs'] = createNumberAttr(1);
      const input1 = [tfc.scalar(1.0)];
      const input2 = [tfc.scalar(2.0)];
      const input3 = [tfc.scalar(3.0)];

      node.inputNames = ['input1', 'input2', 'input3'];
      executeOp(node, {input1, input2, input3}, context);

      expect(tfc.fused.depthwiseConv2d).toHaveBeenCalledWith({
        x: input1[0],
        filter: input2[0],
        strides: [2, 2],
        pad: 'same',
        dataFormat: 'NHWC',
        dilations: [2, 2],
        bias: input3[0],
        activation: 'relu',
        preluActivationWeights: undefined
      });
    });

    it('with bias and prelu activation func', () => {
      spyOn(tfc.fused, 'depthwiseConv2d');
      node.op = 'FusedDepthwiseConv2dNative';
      node.inputParams['filter'] = createTensorAttr(1);
      node.inputParams['args'] = createTensorsAttr(2, 0);
      node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'prelu']);
      node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['pad'] = createStrAttr('same');
      node.attrParams['dataFormat'] = createStrAttr('NHWC');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['numArgs'] = createNumberAttr(2);
      const input1 = [tfc.scalar(1.0)];
      const input2 = [tfc.scalar(2.0)];
      const input3 = [tfc.scalar(3.0)];
      const input4 = [tfc.scalar(4.0)];
      node.inputNames = ['input1', 'input2', 'input3', 'input4'];
      executeOp(node, {input1, input2, input3, input4}, context);

      expect(tfc.fused.depthwiseConv2d).toHaveBeenCalledWith({
        x: input1[0],
        filter: input2[0],
        strides: [2, 2],
        pad: 'same',
        dataFormat: 'NHWC',
        dilations: [2, 2],
        bias: input3[0],
        activation: 'prelu',
        preluActivationWeights: input4[0]
      });
    });

    it('bias add', () => {
      spyOn(tfc.fused, 'depthwiseConv2d');
      node.op = 'FusedDepthwiseConv2dNative';
      node.inputParams['filter'] = createTensorAttr(1);
      node.inputParams['args'] = createTensorsAttr(2, 0);
      node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd']);
      node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['pad'] = createStrAttr('same');
      node.attrParams['dataFormat'] = createStrAttr('NHWC');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['numArgs'] = createNumberAttr(1);
      const input1 = [tfc.scalar(1.0)];
      const input2 = [tfc.scalar(2.0)];
      const input3 = [tfc.scalar(3.0)];

      node.inputNames = ['input1', 'input2', 'input3'];
      executeOp(node, {input1, input2, input3}, context);

      expect(tfc.fused.depthwiseConv2d).toHaveBeenCalledWith({
        x: input1[0],
        filter: input2[0],
        strides: [2, 2],
        pad: 'same',
        dataFormat: 'NHWC',
        dilations: [2, 2],
        bias: input3[0],
        activation: undefined,
        preluActivationWeights: undefined
      });
    });
  });
});
