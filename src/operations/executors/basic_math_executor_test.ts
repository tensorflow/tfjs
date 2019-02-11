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
import * as basic_math from '../op_list/basic_math';
import {Node, OpMapper} from '../types';

import {executeOp} from './basic_math_executor';
// tslint:disable-next-line:max-line-length
import {createNumberAttr, createNumericArrayAttrFromIndex, createTensorAttr, validateParam} from './test_helper';

describe('basic math', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'basic_math',
      inputNames: ['input1'],
      inputs: [],
      inputParams: {x: createTensorAttr(0)},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    ['Abs', 'Acos', 'Asin', 'Atan', 'Ceil', 'Cos', 'Cosh', 'Elu', 'Exp',
     'Floor', 'Log', 'Neg', 'Relu', 'Selu', 'Sigmoid', 'Sin', 'Sinh', 'Sqrt',
     'Square', 'Tanh', 'Tan', 'Sign', 'Round', 'Expm1', 'Log1p', 'Reciprocal',
     'Softplus', 'Asinh', 'Acosh', 'Atanh', 'Erf']
        .forEach(op => {
          it('should call tfc.' + op, () => {
            const spy =
                spyOn(tfc, op.charAt(0).toLowerCase() + op.slice(1) as 'abs');
            node.op = op;
            executeOp(node, {input1}, context);

            expect(spy).toHaveBeenCalledWith(input1[0]);
          });
          it('should match op def', () => {
            node.op = op;

            expect(validateParam(node, basic_math.json as OpMapper[]))
                .toBeTruthy();
          });
        });
    describe('Relu6', () => {
      it('should call tfc.clipByValue', () => {
        spyOn(tfc, 'clipByValue');
        node.op = 'Relu6';
        node.attrParams['clipValueMax'] = createNumberAttr(6);
        node.attrParams['clipValueMin'] = createNumberAttr(0);

        executeOp(node, {input1}, context);

        expect(tfc.clipByValue).toHaveBeenCalledWith(input1[0], 0, 6);
      });
      it('should match op def', () => {
        node.op = 'Relu6';
        node.attrParams['clipValueMax'] = createNumberAttr(6);
        node.attrParams['clipValueMin'] = createNumberAttr(0);

        expect(validateParam(node, basic_math.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('Prod', () => {
      it('should call tfc.prod', () => {
        spyOn(tfc, 'prod');
        node.op = 'Prod';
        node.inputParams['axes'] = createNumericArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfc.scalar(2)];
        executeOp(node, {input1, input2}, context);

        expect(tfc.prod).toHaveBeenCalledWith(input1[0], [2]);
      });
      it('should match op def', () => {
        node.op = 'Prod';
        node.inputParams['axes'] = createNumericArrayAttrFromIndex(1);

        expect(validateParam(node, basic_math.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('Rsqrt', () => {
      it('should call tfc.rsqrt', () => {
        const input1 = [tfc.scalar(1)];
        node.op = 'Rsqrt';
        spyOn(tfc, 'rsqrt').and.returnValue(input1);
        executeOp(node, {input1}, context);

        expect(tfc.rsqrt).toHaveBeenCalledWith(input1[0]);
      });
      it('should match op def', () => {
        node.op = 'Rsqrt';

        expect(validateParam(node, basic_math.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('LeakyRelu', () => {
      it('should call tfc.leakyRelu', () => {
        spyOn(tfc, 'leakyRelu');
        node.op = 'LeakyRelu';
        node.attrParams['alpha'] = createNumberAttr(1);
        node.inputNames = ['input1'];
        executeOp(node, {input1}, context);

        expect(tfc.leakyRelu).toHaveBeenCalledWith(input1[0], 1);
      });
      it('should match op def', () => {
        node.op = 'LeakyRelu';
        node.attrParams['alpha'] = createNumberAttr(1);
        expect(validateParam(node, basic_math.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('Atan2', () => {
      it('should call tfc.atan2', () => {
        spyOn(tfc, 'atan2');
        node.op = 'Atan2';
        node.inputParams['y'] = createTensorAttr(1);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfc.scalar(2)];
        executeOp(node, {input1, input2}, context);

        expect(tfc.atan2).toHaveBeenCalledWith(input1[0], input2[0]);
      });
      it('should match op def', () => {
        node.op = 'Atan2';
        node.inputParams['y'] = createTensorAttr(1);

        expect(validateParam(node, basic_math.json as OpMapper[])).toBeTruthy();
      });
    });
  });
});
