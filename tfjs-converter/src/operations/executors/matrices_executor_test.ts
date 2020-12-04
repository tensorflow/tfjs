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

import {executeOp} from './matrices_executor';
import {createBoolAttr, createNumberAttr, createNumericArrayAttr, createStrArrayAttr, createTensorAttr, createTensorsAttr} from './test_helper';

describe('matrices', () => {
  let node: Node;
  const input1 = [tfOps.scalar(1)];
  const input2 = [tfOps.scalar(2)];
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'matrices',
      inputNames: ['input1', 'input2'],
      inputs: [],
      inputParams: {a: createTensorAttr(0), b: createTensorAttr(1)},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('MatMul', () => {
      it('should call tfOps.matMul', () => {
        spyOn(tfOps, 'matMul');
        node.op = 'MatMul';
        node.attrParams.transposeA = createBoolAttr(true);
        node.attrParams.transposeB = createBoolAttr(false);
        executeOp(node, {input1, input2}, context);

        expect(tfOps.matMul)
            .toHaveBeenCalledWith(input1[0], input2[0], true, false);
      });
    });
    describe('_FusedMatMul', () => {
      it('should call tfOps.fused.matMul', () => {
        spyOn(tfOps.fused, 'matMul');
        node.op = '_FusedMatMul';
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'relu']);
        node.attrParams['numArgs'] = createNumberAttr(1);
        node.attrParams.transposeA = createBoolAttr(true);
        node.attrParams.transposeB = createBoolAttr(false);
        const input3 = [tfOps.scalar(3.0)];
        node.inputNames = ['input1', 'input2', 'input3'];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfOps.fused.matMul).toHaveBeenCalledWith({
          a: input1[0],
          b: input2[0],
          transposeA: true,
          transposeB: false,
          bias: input3[0],
          activation: 'relu',
          preluActivationWeights: undefined,
          leakyreluAlpha: undefined
        });
      });
      it('should call tfOps.fused.matMul - prelu activation', () => {
        spyOn(tfOps.fused, 'matMul');
        node.op = '_FusedMatMul';
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] = createStrArrayAttr(['biasadd', 'prelu']);
        node.attrParams['numArgs'] = createNumberAttr(2);
        node.attrParams.transposeA = createBoolAttr(true);
        node.attrParams.transposeB = createBoolAttr(false);
        const input3 = [tfOps.scalar(3.0)];
        const input4 = [tfOps.scalar(4.0)];
        node.inputNames = ['input1', 'input2', 'input3', 'input4'];
        executeOp(node, {input1, input2, input3, input4}, context);

        expect(tfOps.fused.matMul).toHaveBeenCalledWith({
          a: input1[0],
          b: input2[0],
          transposeA: true,
          transposeB: false,
          bias: input3[0],
          activation: 'prelu',
          preluActivationWeights: input4[0],
          leakyreluAlpha: undefined
        });
      });
      it('should call tfOps.fused.matMul - leakyrelu activation', () => {
        spyOn(tfOps.fused, 'matMul');
        node.op = '_FusedMatMul';
        node.inputParams['args'] = createTensorsAttr(2, 0);
        node.attrParams['fusedOps'] =
            createStrArrayAttr(['biasadd', 'leakyrelu']);
        node.attrParams['numArgs'] = createNumberAttr(1);
        node.attrParams.transposeA = createBoolAttr(true);
        node.attrParams.transposeB = createBoolAttr(false);
        node.attrParams.leakyreluAlpha = createNumberAttr(0.3);
        const input3 = [tfOps.scalar(3.0)];
        node.inputNames = ['input1', 'input2', 'input3'];
        executeOp(node, {input1, input2, input3}, context);

        expect(tfOps.fused.matMul).toHaveBeenCalledWith({
          a: input1[0],
          b: input2[0],
          transposeA: true,
          transposeB: false,
          bias: input3[0],
          activation: 'leakyrelu',
          preluActivationWeights: undefined,
          leakyreluAlpha: 0.3
        });
      });
    });
    describe('BatchMatMul', () => {
      it('should call tfOps.matMul', () => {
        spyOn(tfOps, 'matMul');
        node.op = 'BatchMatMul';
        node.attrParams.transposeA = createBoolAttr(true);
        node.attrParams.transposeB = createBoolAttr(false);
        executeOp(node, {input1, input2}, context);

        expect(tfOps.matMul)
            .toHaveBeenCalledWith(input1[0], input2[0], true, false);
      });
    });
    describe('BatchMatMulV2', () => {
      it('should call tfOps.matMul', () => {
        spyOn(tfOps, 'matMul');
        node.op = 'BatchMatMulV2';
        node.attrParams.transposeA = createBoolAttr(true);
        node.attrParams.transposeB = createBoolAttr(false);
        executeOp(node, {input1, input2}, context);

        expect(tfOps.matMul)
            .toHaveBeenCalledWith(input1[0], input2[0], true, false);
      });
    });
    describe('Transpose', () => {
      it('should call tfOps.transpose', () => {
        spyOn(tfOps, 'transpose');
        node.op = 'Transpose';
        node.inputNames = ['input1', 'input2', 'input3'];
        node.inputParams.x = createTensorAttr(0);
        node.attrParams.perm = createNumericArrayAttr([1, 2]);
        executeOp(node, {input1}, context);

        expect(tfOps.transpose).toHaveBeenCalledWith(input1[0], [1, 2]);
      });
    });
  });
});
