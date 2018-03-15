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

import {executeOp} from './basic_math_executor';
import {createNumberAttr, createTensorAttr} from './test_helper';

describe('basic math', () => {
  let node: Node;
  const input1 = [dl.scalar(1)];

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
     'floor', 'log', 'relu', 'selu', 'sigmoid', 'sin', 'sinh', 'sqrt', 'square',
     'tanh', 'tan']
        .forEach(op => {
          it('should call dl.' + op, () => {
            const spy = spyOn(dl, op as 'abs');
            node.op = op;
            executeOp(node, {input1});

            expect(spy).toHaveBeenCalledWith(input1[0]);
          });
        });
    describe('clipByValue', () => {
      it('should call dl.clipByValue', () => {
        spyOn(dl, 'clipByValue');
        node.op = 'clipByValue';
        node.params['clipValueMax'] = createNumberAttr(6);
        node.params['clipValueMin'] = createNumberAttr(0);

        executeOp(node, {input1});

        expect(dl.clipByValue).toHaveBeenCalledWith(input1[0], 0, 6);
      });
    });
    describe('rsqrt', () => {
      it('should call dl.div', () => {
        const input1 = [dl.scalar(1)];
        node.op = 'rsqrt';
        spyOn(dl, 'div');
        spyOn(dl, 'sqrt').and.returnValue(input1);

        executeOp(node, {input1});

        expect(dl.sqrt).toHaveBeenCalledWith(input1[0]);
        expect(dl.div).toHaveBeenCalledWith(jasmine.any(dl.Tensor), input1);
      });
    });
  });
});
