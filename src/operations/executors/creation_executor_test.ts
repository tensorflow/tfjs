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
import * as creation from '../op_list/creation';
import {Node, OpMapper} from '../types';

import {executeOp} from './creation_executor';
// tslint:disable-next-line:max-line-length
import {createDtypeAttr, createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttrFromIndex, createTensorAttr, validateParam} from './test_helper';

describe('creation', () => {
  let node: Node;
  const input1 = [tfc.tensor1d([1, 2, 3])];
  const input2 = [tfc.scalar(1)];
  const context = new ExecutionContext({}, {});

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
      it('should call tfc.fill', () => {
        spyOn(tfc, 'fill');
        node.op = 'Fill';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputParams['value'] = createNumberAttrFromIndex(1);
        node.attrParams['dtype'] = createDtypeAttr('int32');

        executeOp(node, {input1, input2}, context);

        expect(tfc.fill).toHaveBeenCalledWith([1, 2, 3], 1, 'int32');
      });
      it('should match json def', () => {
        node.op = 'Fill';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputParams['value'] = createNumberAttrFromIndex(1);
        node.attrParams['dtype'] = createDtypeAttr('int32');

        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('LinSpace', () => {
      it('should call tfc.linspace', () => {
        spyOn(tfc, 'linspace');
        node.op = 'LinSpace';
        node.inputParams['start'] = createNumberAttrFromIndex(0);
        node.inputParams['stop'] = createNumberAttrFromIndex(1);
        node.inputParams['num'] = createNumberAttrFromIndex(2);
        node.inputNames = ['input', 'input2', 'input3'];
        const input = [tfc.scalar(0)];
        const input3 = [tfc.scalar(2)];
        executeOp(node, {input, input2, input3}, context);

        expect(tfc.linspace).toHaveBeenCalledWith(0, 1, 2);
      });
      it('should match json def', () => {
        node.op = 'LinSpace';
        node.inputParams['start'] = createNumberAttrFromIndex(0);
        node.inputParams['stop'] = createNumberAttrFromIndex(1);
        node.inputParams['num'] = createNumberAttrFromIndex(2);

        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('OneHot', () => {
      it('should call tfc.oneHot', () => {
        spyOn(tfc, 'oneHot');
        node.op = 'OneHot';
        node.inputParams['indices'] = createTensorAttr(0);
        node.inputParams['depth'] = createNumberAttrFromIndex(1);
        node.inputParams['onValue'] = createNumberAttrFromIndex(2);
        node.inputParams['offValue'] = createNumberAttrFromIndex(3);
        node.inputNames = ['input', 'input2', 'input3', 'input4'];
        const input = [tfc.tensor1d([0])];
        const input3 = [tfc.scalar(2)];
        const input4 = [tfc.scalar(3)];
        executeOp(node, {input, input2, input3, input4}, context);

        expect(tfc.oneHot).toHaveBeenCalledWith(input[0], 1, 2, 3);
      });
      it('should match json def', () => {
        node.op = 'OneHot';
        node.inputParams['indices'] = createTensorAttr(0);
        node.inputParams['depth'] = createNumberAttrFromIndex(1);
        node.inputParams['onValue'] = createNumberAttrFromIndex(2);
        node.inputParams['offValue'] = createNumberAttrFromIndex(3);

        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('Ones', () => {
      it('should call tfc.ones', () => {
        spyOn(tfc, 'ones');
        node.op = 'Ones';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        executeOp(node, {input1}, context);

        expect(tfc.ones).toHaveBeenCalledWith([1, 2, 3], 'float32');
      });
      it('should match json def', () => {
        node.op = 'Ones';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');

        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('OnesLike', () => {
      it('should call tfc.onesLike', () => {
        spyOn(tfc, 'onesLike');
        node.op = 'OnesLike';
        node.inputParams['x'] = createTensorAttr(0);
        executeOp(node, {input1}, context);

        expect(tfc.onesLike).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'OnesLike';
        node.inputParams['x'] = createTensorAttr(0);

        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('Range', () => {
      it('should call tfc.range', () => {
        spyOn(tfc, 'range');
        node.op = 'Range';
        node.inputParams['start'] = createNumberAttrFromIndex(0);
        node.inputParams['stop'] = createNumberAttrFromIndex(1);
        node.inputParams['step'] = createNumberAttrFromIndex(2);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        node.inputNames = ['input', 'input2', 'input3'];
        const input = [tfc.scalar(0)];
        const input3 = [tfc.scalar(2)];
        executeOp(node, {input, input2, input3}, context);

        expect(tfc.range).toHaveBeenCalledWith(0, 1, 2, 'float32');
      });
      it('should match json def', () => {
        node.op = 'Range';
        node.inputParams['start'] = createNumberAttrFromIndex(0);
        node.inputParams['stop'] = createNumberAttrFromIndex(1);
        node.inputParams['step'] = createNumberAttrFromIndex(2);
        node.attrParams['dtype'] = createDtypeAttr('float32');

        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('RandomUniform', () => {
      it('should call tfc.randomUniform', () => {
        spyOn(tfc, 'randomUniform');
        node.op = 'RandomUniform';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.attrParams['maxval'] = createNumberAttr(1);
        node.attrParams['minval'] = createNumberAttr(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        node.attrParams['seed'] = createNumberAttr(0);

        executeOp(node, {input1}, context);

        expect(tfc.randomUniform)
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

        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('TruncatedNormal', () => {
      it('should call tfc.truncatedNormal', () => {
        spyOn(tfc, 'truncatedNormal');
        node.op = 'TruncatedNormal';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.attrParams['stdDev'] = createNumberAttr(1);
        node.attrParams['mean'] = createNumberAttr(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        node.attrParams['seed'] = createNumberAttr(0);

        executeOp(node, {input1}, context);

        expect(tfc.truncatedNormal)
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

        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('Zeros', () => {
      it('should call tfc.zeros', () => {
        spyOn(tfc, 'zeros');
        node.op = 'Zeros';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        executeOp(node, {input1}, context);

        expect(tfc.zeros).toHaveBeenCalledWith([1, 2, 3], 'float32');
      });
      it('should match json def', () => {
        node.op = 'Zeros';
        node.inputParams['shape'] = createNumericArrayAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('float32');
        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('ZerosLike', () => {
      it('should call tfc.zerosLike', () => {
        spyOn(tfc, 'zerosLike');
        node.op = 'ZerosLike';
        node.inputParams['x'] = createTensorAttr(0);
        executeOp(node, {input1}, context);

        expect(tfc.zerosLike).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'ZerosLike';
        node.inputParams['x'] = createTensorAttr(0);
        expect(validateParam(node, creation.json as OpMapper[])).toBeTruthy();
      });
    });
  });
});
