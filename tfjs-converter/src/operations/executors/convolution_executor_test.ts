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
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {ExecutionContext} from '../../executor/execution_context';
import {Node} from '../types';

import {executeOp} from './convolution_executor';
import {createNumberAttr, createNumericArrayAttr, createStrArrayAttr, createStrAttr, createTensorAttr, createTensorsAttr} from './test_helper';
import {createBoolAttr} from './test_helper';

describe('convolution', () => {
  let node: Node;
  const input = [tfOps.scalar(1)];
  const context = new ExecutionContext({}, {}, {});

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
      it('should call tfOps.avgPool', () => {
        spyOn(tfOps, 'avgPool');
        node.op = 'AvgPool';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfOps.avgPool)
            .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
      });
    });

    describe('maxPool', () => {
      it('should call tfOps.maxPool', () => {
        spyOn(tfOps, 'maxPool');
        node.op = 'MaxPool';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfOps.maxPool)
            .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
      });
    });
    describe('Conv2d', () => {
      it('should call tfOps.conv2d', () => {
        spyOn(tfOps, 'conv2d');
        node.op = 'Conv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);

        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfOps.conv2d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
      });
      it('should support explicit padding', () => {
        spyOn(tfOps, 'conv2d');
        node.op = 'Conv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('explicit');
        node.attrParams['explicitPaddings'] =
            createNumericArrayAttr([0, 0, 1, 1, 2, 2, 0, 0]);
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);

        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfOps.conv2d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2], [[0, 0], [1, 1], [2, 2], [0, 0]],
                'NHWC', [2, 2]);
      });
    });
    describe('Conv2DBackpropInput', () => {
      it('should call tfOps.conv2dTranspose', () => {
        spyOn(tfOps, 'conv2dTranspose');
        node.op = 'Conv2DBackpropInput';
        node.attrParams['outputShape'] = createNumericArrayAttr([1, 2, 2, 2]);
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');

        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfOps.conv2dTranspose)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [1, 2, 2, 2], [2, 2], 'same');
      });
      it('should support explicit padding', () => {
        spyOn(tfOps, 'conv2dTranspose');
        node.op = 'Conv2DBackpropInput';
        node.attrParams['outputShape'] = createNumericArrayAttr([1, 2, 2, 2]);
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('explicit');
        node.attrParams['explicitPaddings'] =
            createNumericArrayAttr([0, 0, 1, 1, 2, 2, 0, 0]);

        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfOps.conv2dTranspose)
            .toHaveBeenCalledWith(
                input1[0],
                input2[0],
                [1, 2, 2, 2],
                [2, 2],
                [[0, 0], [1, 1], [2, 2], [0, 0]],
            );
      });
    });
    describe('Conv1D', () => {
      it('should call tfOps.conv1d', () => {
        spyOn(tfOps, 'conv1d');
        node.op = 'Conv1D';
        node.category = 'convolution';
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['stride'] = createNumberAttr(1);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NWC');
        node.attrParams['dilation'] = createNumberAttr(1);

        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfOps.conv1d)
            .toHaveBeenCalledWith(input1[0], input2[0], 1, 'same', 'NWC', 1);
      });
    });

    describe('DepthwiseConv2d', () => {
      it('should call tfOps.depthwiseConv2d', () => {
        spyOn(tfOps, 'depthwiseConv2d');
        node.op = 'DepthwiseConv2d';
        node.category = 'convolution';
        node.inputParams['input'] = createTensorAttr(0);
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfOps.depthwiseConv2d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
      });
      it('support explicit padding', () => {
        spyOn(tfOps, 'depthwiseConv2d');
        node.op = 'DepthwiseConv2d';
        node.category = 'convolution';
        node.inputParams['input'] = createTensorAttr(0);
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('explicit');
        node.attrParams['explicitPaddings'] =
            createNumericArrayAttr([0, 0, 1, 1, 2, 2, 0, 0]);
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfOps.depthwiseConv2d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2], [[0, 0], [1, 1], [2, 2], [0, 0]],
                'NHWC', [2, 2]);
      });
    });

    describe('Conv3d', () => {
      it('should call tfOps.conv3d', () => {
        spyOn(tfOps, 'conv3d');
        node.op = 'Conv3D';
        node.category = 'convolution';
        node.inputParams['filter'] = createTensorAttr(1);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 2, 1]);

        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(1.0)];
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2}, context);

        expect(tfOps.conv3d)
            .toHaveBeenCalledWith(
                input1[0], input2[0], [2, 2, 2], 'same', 'NHWC', [2, 2, 2]);
      });
    });

    describe('AvgPool3D', () => {
      it('should call tfOps.avgPool3d', () => {
        spyOn(tfOps, 'avgPool3d');
        node.op = 'AvgPool3D';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfOps.avgPool3d)
            .toHaveBeenCalledWith(input[0], [2, 2, 2], [2, 2, 2], 'same');
      });
    });

    describe('MaxPool3D', () => {
      it('should call tfOps.maxPool3d', () => {
        spyOn(tfOps, 'maxPool3d');
        node.op = 'MaxPool3D';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 2, 1]);

        executeOp(node, {input}, context);

        expect(tfOps.maxPool3d)
            .toHaveBeenCalledWith(input[0], [2, 2, 2], [2, 2, 2], 'same');
      });
    });

    describe('MaxPoolWithArgmax', () => {
      it('should call tfOps.maxPoolWithArgmax', () => {
        spyOn(tfOps, 'maxPoolWithArgmax').and.returnValue({});
        node.op = 'MaxPoolWithArgmax';
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['dataFormat'] = createStrAttr('NDHWC');
        node.attrParams['includeBatchInIndex'] = createBoolAttr(true);
        executeOp(node, {input}, context);

        expect(tfOps.maxPoolWithArgmax)
            .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same', true);
      });
    });

    describe('_FusedConv2d', () => {
      it('with bias and activation func', () => {
        spyOn(tfOps.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'relu']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(1);
        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(2.0)];
        const input3 = [tfOps.scalar(3.0)];

        node.inputNames = ['input1', 'input2', 'input3'];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfOps.fused.conv2d).toHaveBeenCalledWith({
          x: input1[0],
          filter: input2[0],
          strides: [2, 2],
          pad: 'same',
          dataFormat: 'NHWC',
          dilations: [2, 2],
          bias: input3[0],
          activation: 'relu',
          preluActivationWeights: undefined,
          leakyreluAlpha: undefined
        });
      });
      it('should support explicit padding', () => {
        spyOn(tfOps.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'relu']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('explicit');
        node.attrParams['explicitPaddings'] =
            createNumericArrayAttr([0, 0, 1, 1, 2, 2, 0, 0]);
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(1);
        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(2.0)];
        const input3 = [tfOps.scalar(3.0)];

        node.inputNames = ['input1', 'input2', 'input3'];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfOps.fused.conv2d).toHaveBeenCalledWith({
          x: input1[0],
          filter: input2[0],
          strides: [2, 2],
          pad: [[0, 0], [1, 1], [2, 2], [0, 0]],
          dataFormat: 'NHWC',
          dilations: [2, 2],
          bias: input3[0],
          activation: 'relu',
          preluActivationWeights: undefined,
          leakyreluAlpha: undefined
        });
      });
      it('with bias and prelu activation func', () => {
        spyOn(tfOps.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'prelu']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(2);
        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(2.0)];
        const input3 = [tfOps.scalar(3.0)];
        const input4 = [tfOps.scalar(4.0)];
        node.inputNames = ['input1', 'input2', 'input3', 'input4'];
        executeOp(node, {input1, input2, input3, input4}, context);

        expect(tfOps.fused.conv2d).toHaveBeenCalledWith({
          x: input1[0],
          filter: input2[0],
          strides: [2, 2],
          pad: 'same',
          dataFormat: 'NHWC',
          dilations: [2, 2],
          bias: input3[0],
          activation: 'prelu',
          preluActivationWeights: input4[0],
          leakyreluAlpha: undefined
        });
      });
      it('with bias and leakyrelu activation func', () => {
        spyOn(tfOps.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] =
            createStrArrayAttr(['biasadd', 'leakyrelu']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(1);
        node.attrParams['leakyreluAlpha'] = createNumberAttr(0.3);
        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(2.0)];
        const input3 = [tfOps.scalar(3.0)];
        node.inputNames = ['input1', 'input2', 'input3'];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfOps.fused.conv2d).toHaveBeenCalledWith({
          x: input1[0],
          filter: input2[0],
          strides: [2, 2],
          pad: 'same',
          dataFormat: 'NHWC',
          dilations: [2, 2],
          bias: input3[0],
          activation: 'leakyrelu',
          preluActivationWeights: undefined,
          leakyreluAlpha: 0.3
        });
      });

      it('bias add', () => {
        spyOn(tfOps.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(1);
        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(2.0)];
        const input3 = [tfOps.scalar(3.0)];

        node.inputNames = ['input1', 'input2', 'input3'];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfOps.fused.conv2d).toHaveBeenCalledWith({
          x: input1[0],
          filter: input2[0],
          strides: [2, 2],
          pad: 'same',
          dataFormat: 'NHWC',
          dilations: [2, 2],
          bias: input3[0],
          activation: undefined,
          preluActivationWeights: undefined,
          leakyreluAlpha: undefined
        });
      });
      it('fail with batchnorm', () => {
        spyOn(tfOps.fused, 'conv2d');
        node.op = '_FusedConv2D';
        node.inputParams['filter'] = createTensorAttr(1);
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['fusedbatchnorm']);
        node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['pad'] = createStrAttr('same');
        node.attrParams['dataFormat'] = createStrAttr('NHWC');
        node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.attrParams['numArgs'] = createNumberAttr(1);
        const input1 = [tfOps.scalar(1.0)];
        const input2 = [tfOps.scalar(2.0)];
        const input3 = [tfOps.scalar(3.0)];

        node.inputNames = ['input1', 'input2', 'input3'];
        expect(() => executeOp(node, {input1, input2, input3}, context))
            .toThrow();
      });
    });
  });
  describe('FusedDepthwiseConv2d', () => {
    it('support explicit padding', () => {
      spyOn(tfOps.fused, 'depthwiseConv2d');
      node.op = 'FusedDepthwiseConv2dNative';
      node.inputParams['filter'] = createTensorAttr(1);
      node.inputParams['args'] = createTensorsAttr(2, 0);
      node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'relu']);
      node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['pad'] = createStrAttr('explicit');
      node.attrParams['explicitPaddings'] =
          createNumericArrayAttr([0, 0, 1, 1, 2, 2, 0, 0]);
      node.attrParams['dataFormat'] = createStrAttr('NHWC');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['numArgs'] = createNumberAttr(1);
      const input1 = [tfOps.scalar(1.0)];
      const input2 = [tfOps.scalar(2.0)];
      const input3 = [tfOps.scalar(3.0)];

      node.inputNames = ['input1', 'input2', 'input3'];
      executeOp(node, {input1, input2, input3}, context);

      expect(tfOps.fused.depthwiseConv2d).toHaveBeenCalledWith({
        x: input1[0],
        filter: input2[0],
        strides: [2, 2],
        pad: [[0, 0], [1, 1], [2, 2], [0, 0]],
        dataFormat: 'NHWC',
        dilations: [2, 2],
        bias: input3[0],
        activation: 'relu',
        preluActivationWeights: undefined,
        leakyreluAlpha: undefined
      });
    });
    it('with bias and activation func', () => {
      spyOn(tfOps.fused, 'depthwiseConv2d');
      node.op = 'FusedDepthwiseConv2dNative';
      node.inputParams['filter'] = createTensorAttr(1);
      node.inputParams['args'] = createTensorsAttr(2, 0);
      node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'relu']);
      node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['pad'] = createStrAttr('same');
      node.attrParams['dataFormat'] = createStrAttr('NHWC');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['numArgs'] = createNumberAttr(1);
      const input1 = [tfOps.scalar(1.0)];
      const input2 = [tfOps.scalar(2.0)];
      const input3 = [tfOps.scalar(3.0)];

      node.inputNames = ['input1', 'input2', 'input3'];
      executeOp(node, {input1, input2, input3}, context);

      expect(tfOps.fused.depthwiseConv2d).toHaveBeenCalledWith({
        x: input1[0],
        filter: input2[0],
        strides: [2, 2],
        pad: 'same',
        dataFormat: 'NHWC',
        dilations: [2, 2],
        bias: input3[0],
        activation: 'relu',
        preluActivationWeights: undefined,
        leakyreluAlpha: undefined
      });
    });
    it('with bias and prelu activation func', () => {
      spyOn(tfOps.fused, 'depthwiseConv2d');
      node.op = 'FusedDepthwiseConv2dNative';
      node.inputParams['filter'] = createTensorAttr(1);
      node.inputParams['args'] = createTensorsAttr(2, 0);
      node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'prelu']);
      node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['pad'] = createStrAttr('same');
      node.attrParams['dataFormat'] = createStrAttr('NHWC');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['numArgs'] = createNumberAttr(2);
      const input1 = [tfOps.scalar(1.0)];
      const input2 = [tfOps.scalar(2.0)];
      const input3 = [tfOps.scalar(3.0)];
      const input4 = [tfOps.scalar(4.0)];
      node.inputNames = ['input1', 'input2', 'input3', 'input4'];
      executeOp(node, {input1, input2, input3, input4}, context);

      expect(tfOps.fused.depthwiseConv2d).toHaveBeenCalledWith({
        x: input1[0],
        filter: input2[0],
        strides: [2, 2],
        pad: 'same',
        dataFormat: 'NHWC',
        dilations: [2, 2],
        bias: input3[0],
        activation: 'prelu',
        preluActivationWeights: input4[0],
        leakyreluAlpha: undefined
      });
    });
    it('with bias and leakyrelu activation func', () => {
      spyOn(tfOps.fused, 'depthwiseConv2d');
      node.op = 'FusedDepthwiseConv2dNative';
      node.inputParams['filter'] = createTensorAttr(1);
      node.inputParams['args'] = createTensorsAttr(2, 0);
      node.attrParams['fusedOps'] =
          createStrArrayAttr(['biasadd', 'leakyrelu']);
      node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['pad'] = createStrAttr('same');
      node.attrParams['dataFormat'] = createStrAttr('NHWC');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['numArgs'] = createNumberAttr(1);
      node.attrParams['leakyreluAlpha'] = createNumberAttr(0.3);
      const input1 = [tfOps.scalar(1.0)];
      const input2 = [tfOps.scalar(2.0)];
      const input3 = [tfOps.scalar(3.0)];
      node.inputNames = ['input1', 'input2', 'input3'];
      executeOp(node, {input1, input2, input3}, context);

      expect(tfOps.fused.depthwiseConv2d).toHaveBeenCalledWith({
        x: input1[0],
        filter: input2[0],
        strides: [2, 2],
        pad: 'same',
        dataFormat: 'NHWC',
        dilations: [2, 2],
        bias: input3[0],
        activation: 'leakyrelu',
        preluActivationWeights: undefined,
        leakyreluAlpha: 0.3
      });
    });

    it('bias add', () => {
      spyOn(tfOps.fused, 'depthwiseConv2d');
      node.op = 'FusedDepthwiseConv2dNative';
      node.inputParams['filter'] = createTensorAttr(1);
      node.inputParams['args'] = createTensorsAttr(2, 0);
      node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd']);
      node.attrParams['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['pad'] = createStrAttr('same');
      node.attrParams['dataFormat'] = createStrAttr('NHWC');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);
      node.attrParams['numArgs'] = createNumberAttr(1);
      const input1 = [tfOps.scalar(1.0)];
      const input2 = [tfOps.scalar(2.0)];
      const input3 = [tfOps.scalar(3.0)];

      node.inputNames = ['input1', 'input2', 'input3'];
      executeOp(node, {input1, input2, input3}, context);

      expect(tfOps.fused.depthwiseConv2d).toHaveBeenCalledWith({
        x: input1[0],
        filter: input2[0],
        strides: [2, 2],
        pad: 'same',
        dataFormat: 'NHWC',
        dilations: [2, 2],
        bias: input3[0],
        activation: undefined,
        preluActivationWeights: undefined,
        leakyreluAlpha: undefined
      });
    });
  });

  describe('dilation2d', () => {
    it('should call tfOps.dilation2d', () => {
      spyOn(tfOps, 'dilation2d');
      node.op = 'Dilation2D';
      node.inputParams['filter'] = createTensorAttr(1);
      node.attrParams['strides'] = createNumericArrayAttr([1, 1, 1, 1]);
      node.attrParams['pad'] = createStrAttr('same');
      node.attrParams['dilations'] = createNumericArrayAttr([1, 2, 2, 1]);

      const input1 = [tfOps.scalar(1.0)];
      const input2 = [tfOps.scalar(1.0)];
      node.inputNames = ['input1', 'input2'];

      executeOp(node, {input1, input2}, context);

      expect(tfOps.dilation2d)
          .toHaveBeenCalledWith(
              input1[0], input2[0], [1, 1], 'same', [2, 2], 'NHWC');
    });
  });
});
