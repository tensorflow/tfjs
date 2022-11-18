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
import {scalar} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {ExecutionContext} from '../../executor/execution_context';
import * as basic_math from '../op_list/basic_math';
import {Node} from '../types';

import {executeOp} from './basic_math_executor';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';
import {createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttrFromIndex, createTensorAttr, uncapitalize, validateParam} from './test_helper';

describe('basic math', () => {
  let node: Node;
  const input1 = [scalar(1)];
  const context = new ExecutionContext({}, {}, {});

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
    let spyOps: RecursiveSpy<typeof tfOps>;
    let spyOpsAsTfOps: typeof tfOps;

    beforeEach(() => {
      spyOps = spyOnAllFunctions(tfOps);
      spyOpsAsTfOps = spyOps as unknown as typeof tfOps;
    });

    ([
      'Abs',      'Acos',  'Asin',    'Atan',  'Ceil',  'Cos',   'Cosh',
      'Elu',      'Exp',   'Floor',   'Log',   'Imag',  'Neg',   'Real',
      'Relu',     'Selu',  'Sigmoid', 'Sin',   'Sinh',  'Sqrt',  'Square',
      'Tanh',     'Tan',   'Sign',    'Round', 'Expm1', 'Log1p', 'Reciprocal',
      'Softplus', 'Asinh', 'Acosh',   'Atanh', 'Erf'
    ] as const )
        .forEach(op => {
          it('should call tfOps.' + op, () => {
            node.op = op;
            spyOps[uncapitalize(op)].and.returnValue({});
            executeOp(node, {input1}, context, spyOpsAsTfOps);

            expect(spyOps[uncapitalize(op)]).toHaveBeenCalledWith(input1[0]);
          });
          it('should match op def', () => {
            node.op = op;

            expect(validateParam(node, basic_math.json)).toBeTruthy();
          });
        });
    describe('Relu6', () => {
      it('should call tfOps.relu6', () => {
        node.op = 'Relu6';

        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.relu6).toHaveBeenCalledWith(input1[0]);
      });
      it('should match op def', () => {
        node.op = 'Relu6';

        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('ClipByValue', () => {
      it('should call tfOps.clipByValue', () => {
        node.op = 'ClipByValue';
        node.inputNames = ['input1', 'input2', 'input3'];
        node.inputParams['clipValueMin'] = createNumberAttrFromIndex(1);
        node.inputParams['clipValueMax'] = createNumberAttrFromIndex(2);
        const input2 = [tfOps.scalar(2)];
        const input3 = [tfOps.scalar(3)];
        executeOp(node, {input1, input2, input3}, context, spyOpsAsTfOps);

        expect(spyOps.clipByValue).toHaveBeenCalledWith(input1[0], 2, 3);
      });
      it('should match op def', () => {
        node.op = 'ClipByValue';
        node.inputParams['clipValueMin'] = createNumberAttrFromIndex(1);
        node.inputParams['clipValueMax'] = createNumberAttrFromIndex(2);

        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('Prod', () => {
      it('should call tfOps.prod', () => {
        node.op = 'Prod';
        node.inputParams['axes'] = createNumericArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfOps.tensor1d([2])];
        spyOps.prod.and.returnValue({});
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.prod).toHaveBeenCalledWith(input1[0], [2]);
      });
      it('should match op def', () => {
        node.op = 'Prod';
        node.inputParams['axes'] = createNumericArrayAttrFromIndex(1);

        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('Rsqrt', () => {
      it('should call tfOps.rsqrt', () => {
        const input1 = [tfOps.scalar(1)];
        node.op = 'Rsqrt';
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.rsqrt).toHaveBeenCalledWith(input1[0]);
      });
      it('should match op def', () => {
        node.op = 'Rsqrt';

        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('LeakyRelu', () => {
      it('should call tfOps.leakyRelu', () => {
        node.op = 'LeakyRelu';
        node.attrParams['alpha'] = createNumberAttr(1);
        node.inputNames = ['input1'];
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.leakyRelu).toHaveBeenCalledWith(input1[0], 1);
      });
      it('should match op def', () => {
        node.op = 'LeakyRelu';
        node.attrParams['alpha'] = createNumberAttr(1);
        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('Prelu', () => {
      it('should call tfOps.Prelu', () => {
        node.op = 'Prelu';
        node.inputParams['x'] = createTensorAttr(0);
        node.inputParams['alpha'] = createTensorAttr(1);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfOps.scalar(1)];
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.prelu).toHaveBeenCalledWith(input1[0], input2[0]);
      });
      it('should match op def', () => {
        node.op = 'Prelu';
        node.inputParams['x'] = createTensorAttr(0);
        node.inputParams['alpha'] = createTensorAttr(1);
        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('Atan2', () => {
      it('should call tfOps.atan2', () => {
        node.op = 'Atan2';
        node.inputParams['y'] = createTensorAttr(1);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfOps.scalar(2)];
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.atan2).toHaveBeenCalledWith(input1[0], input2[0]);
      });
      it('should match op def', () => {
        node.op = 'Atan2';
        node.inputParams['y'] = createTensorAttr(1);

        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('ComplexAbs', () => {
      it('should call tfOps.abs', () => {
        node.op = 'ComplexAbs';
        node.inputNames = ['input1'];
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.abs).toHaveBeenCalledWith(input1[0]);
      });
      it('should match op def', () => {
        node.op = 'ComplexAbs';

        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('Complex', () => {
      it('should call tfOps.complex', () => {
        node.op = 'Complex';
        node.inputParams = {
          real: createTensorAttr(0),
          imag: createTensorAttr(1)
        };
        const input2 = [tfOps.scalar(2)];
        node.inputNames = ['input1', 'input2'];
        executeOp(node, {input1, input2}, context, spyOpsAsTfOps);

        expect(spyOps.complex).toHaveBeenCalledWith(input1[0], input2[0]);
      });
      it('should match op def', () => {
        node.op = 'Complex';
        node.inputParams = {
          real: createTensorAttr(0),
          imag: createTensorAttr(1)
        };

        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
    describe('IsNan', () => {
      it('should call tfOps.isNaN', () => {
        node.op = 'IsNan';

        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.isNaN).toHaveBeenCalledWith(input1[0]);
      });
      it('should match op def', () => {
        node.op = 'IsNan';

        expect(validateParam(node, basic_math.json)).toBeTruthy();
      });
    });
  });
});
