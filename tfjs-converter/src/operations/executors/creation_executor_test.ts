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
import * as creation from '../op_list/creation';
import {Node} from '../types';

import {executeOp} from './creation_executor';
import {createDtypeAttr, createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttrFromIndex, createTensorAttr, validateParam} from './test_helper';

describe('creation', () => {
  let node: Node;
  const input1 = [tfOps.tensor1d([1, 2, 3])];
  const input2 = [tfOps.scalar(1)];
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'creation',
      inputNames: ['input1', 'input2'],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('Fill', () => {
      it('should call tfOps.fill', () => {
        spyOn(tfOps, 'fill');
        node.op = 'Fill';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputParams['value'] = createNumberAttrFromIndex(1);
        node.attrParams['dtype'] = createDtypeAttr('int32');

        executeOp(node, {input1, input2}, context);

        expect(tfOps.fill).toHaveBeenCalledWith([1, 2, 3], 1, 'int32');
      });
      it('should match json def', () => {
        node.op = 'Fill';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputParams['value'] = createNumberAttrFromIndex(1);
        node.attrParams['dtype'] = createDtypeAttr('int32');

        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('LinSpace', () => {
      it('should call tfOps.linspace', () => {
        spyOn(tfOps, 'linspace');
        node.op = 'LinSpace';
        node.inputParams['start'] = createNumberAttrFromIndex(0);
        node.inputParams['stop'] = createNumberAttrFromIndex(1);
        node.inputParams['num'] = createNumberAttrFromIndex(2);
        node.inputNames = ['input', 'input2', 'input3'];
        const input = [tfOps.scalar(0)];
        const input3 = [tfOps.scalar(2)];
        executeOp(node, {input, input2, input3}, context);

        expect(tfOps.linspace).toHaveBeenCalledWith(0, 1, 2);
      });
      it('should match json def', () => {
        node.op = 'LinSpace';
        node.inputParams['start'] = createNumberAttrFromIndex(0);
        node.inputParams['stop'] = createNumberAttrFromIndex(1);
        node.inputParams['num'] = createNumberAttrFromIndex(2);

        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('OneHot', () => {
      it('should call tfOps.oneHot', () => {
        spyOn(tfOps, 'oneHot');
        node.op = 'OneHot';
        node.inputParams['indices'] = createTensorAttr(0);
        node.inputParams['depth'] = createNumberAttrFromIndex(1);
        node.inputParams['onValue'] = createNumberAttrFromIndex(2);
        node.inputParams['offValue'] = createNumberAttrFromIndex(3);
        node.inputNames = ['input', 'input2', 'input3', 'input4'];
        const input = [tfOps.tensor1d([0])];
        const input3 = [tfOps.scalar(2)];
        const input4 = [tfOps.scalar(3)];
        executeOp(node, {input, input2, input3, input4}, context);

        expect(tfOps.oneHot).toHaveBeenCalledWith(input[0], 1, 2, 3);
      });
      it('should match json def', () => {
        node.op = 'OneHot';
        node.inputParams['indices'] = createTensorAttr(0);
        node.inputParams['depth'] = createNumberAttrFromIndex(1);
        node.inputParams['onValue'] = createNumberAttrFromIndex(2);
        node.inputParams['offValue'] = createNumberAttrFromIndex(3);

        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('Ones', () => {
      it('should call tfOps.ones', () => {
        spyOn(tfOps, 'ones');
        node.op = 'Ones';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        executeOp(node, {input1}, context);

        expect(tfOps.ones).toHaveBeenCalledWith([1, 2, 3], 'float32');
      });
      it('should match json def', () => {
        node.op = 'Ones';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');

        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('OnesLike', () => {
      it('should call tfOps.onesLike', () => {
        spyOn(tfOps, 'onesLike');
        node.op = 'OnesLike';
        node.inputParams['x'] = createTensorAttr(0);
        executeOp(node, {input1}, context);

        expect(tfOps.onesLike).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'OnesLike';
        node.inputParams['x'] = createTensorAttr(0);

        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('Range', () => {
      it('should call tfOps.range', () => {
        spyOn(tfOps, 'range');
        node.op = 'Range';
        node.inputParams['start'] = createNumberAttrFromIndex(0);
        node.inputParams['stop'] = createNumberAttrFromIndex(1);
        node.inputParams['step'] = createNumberAttrFromIndex(2);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        node.inputNames = ['input', 'input2', 'input3'];
        const input = [tfOps.scalar(0)];
        const input3 = [tfOps.scalar(2)];
        executeOp(node, {input, input2, input3}, context);

        expect(tfOps.range).toHaveBeenCalledWith(0, 1, 2, 'float32');
      });
      it('should match json def', () => {
        node.op = 'Range';
        node.inputParams['start'] = createNumberAttrFromIndex(0);
        node.inputParams['stop'] = createNumberAttrFromIndex(1);
        node.inputParams['step'] = createNumberAttrFromIndex(2);
        node.attrParams['dtype'] = createDtypeAttr('float32');

        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('RandomUniform', () => {
      it('should call tfOps.randomUniform', () => {
        spyOn(tfOps, 'randomUniform');
        node.op = 'RandomUniform';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.attrParams['maxval'] = createNumberAttr(1);
        node.attrParams['minval'] = createNumberAttr(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        node.attrParams['seed'] = createNumberAttr(0);

        executeOp(node, {input1}, context);

        expect(tfOps.randomUniform)
            .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32');
      });
      it('should match json def', () => {
        node.op = 'RandomUniform';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.attrParams['maxval'] = createNumberAttr(1);
        node.attrParams['minval'] = createNumberAttr(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        node.attrParams['seed'] = createNumberAttr(0);

        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('TruncatedNormal', () => {
      it('should call tfOps.truncatedNormal', () => {
        spyOn(tfOps, 'truncatedNormal');
        node.op = 'TruncatedNormal';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.attrParams['stdDev'] = createNumberAttr(1);
        node.attrParams['mean'] = createNumberAttr(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        node.attrParams['seed'] = createNumberAttr(0);

        executeOp(node, {input1}, context);

        expect(tfOps.truncatedNormal)
            .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32', 0);
      });
      it('should match json def', () => {
        node.op = 'TruncatedNormal';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.attrParams['stdDev'] = createNumberAttr(1);
        node.attrParams['mean'] = createNumberAttr(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        node.attrParams['seed'] = createNumberAttr(0);

        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('Zeros', () => {
      it('should call tfOps.zeros', () => {
        spyOn(tfOps, 'zeros');
        node.op = 'Zeros';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        executeOp(node, {input1}, context);

        expect(tfOps.zeros).toHaveBeenCalledWith([1, 2, 3], 'float32');
      });
      it('should match json def', () => {
        node.op = 'Zeros';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('ZerosLike', () => {
      it('should call tfOps.zerosLike', () => {
        spyOn(tfOps, 'zerosLike');
        node.op = 'ZerosLike';
        node.inputParams['x'] = createTensorAttr(0);
        executeOp(node, {input1}, context);

        expect(tfOps.zerosLike).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'ZerosLike';
        node.inputParams['x'] = createTensorAttr(0);
        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
    describe('Multinomial', () => {
      it('should call tfOps.multinomial', () => {
        spyOn(tfOps, 'multinomial');
        node.op = 'Multinomial';
        node.inputParams['logits'] = createTensorAttr(0);
        node.inputParams['numSamples'] = createNumberAttrFromIndex(1);
        node.attrParams['seed'] = createNumberAttr(2);
        executeOp(node, {input1, input2}, context);

        expect(tfOps.multinomial).toHaveBeenCalledWith(input1[0], 1, 2);
      });
      it('should match json def', () => {
        node.op = 'Multinomial';
        node.inputParams['logits'] = createTensorAttr(0);
        node.inputParams['numSamples'] = createNumberAttrFromIndex(1);
        node.attrParams['seed'] = createNumberAttr(2);
        expect(validateParam(node, creation.json)).toBeTruthy();
      });
    });
  });
});
