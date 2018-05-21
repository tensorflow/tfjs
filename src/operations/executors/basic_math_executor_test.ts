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

import {executeOp} from './basic_math_executor';
import {createNumberAttr, createTensorAttr} from './test_helper';

describe('basic math', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const context = new ExecutionContext({});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'basic_math',
      inputNames: ['input1'],
      inputs: [],
      params: {x: createTensorAttr(0)},
      children: []
    };
  });

  describe('executeOp', () => {
    ['abs', 'acos', 'asin', 'atan', 'ceil', 'cos', 'cosh', 'elu', 'exp',
     'floor', 'log', 'neg', 'relu', 'selu', 'sigmoid', 'sin', 'sinh', 'sqrt',
     'square', 'tanh', 'tan', 'sign', 'round', 'expm1', 'log1p', 'reciprocal',
     'softplus', 'asinh', 'acosh', 'atanh', 'erf']
        .forEach(op => {
          it('should call tfc.' + op, () => {
            const spy = spyOn(tfc, op as 'abs');
            node.op = op;
            executeOp(node, {input1}, context);

            expect(spy).toHaveBeenCalledWith(input1[0]);
          });
        });
    describe('clipByValue', () => {
      it('should call tfc.clipByValue', () => {
        spyOn(tfc, 'clipByValue');
        node.op = 'clipByValue';
        node.params['clipValueMax'] = createNumberAttr(6);
        node.params['clipValueMin'] = createNumberAttr(0);

        executeOp(node, {input1}, context);

        expect(tfc.clipByValue).toHaveBeenCalledWith(input1[0], 0, 6);
      });
    });
    describe('rsqrt', () => {
      it('should call tfc.div', () => {
        const input1 = [tfc.scalar(1)];
        node.op = 'rsqrt';
        spyOn(tfc, 'div');
        spyOn(tfc, 'sqrt').and.returnValue(input1);

        executeOp(node, {input1}, context);

        expect(tfc.sqrt).toHaveBeenCalledWith(input1[0]);
        expect(tfc.div).toHaveBeenCalledWith(jasmine.any(tfc.Tensor), input1);
      });
    });
  });
});
