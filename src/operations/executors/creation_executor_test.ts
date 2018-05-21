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

import {executeOp} from './creation_executor';
// tslint:disable-next-line:max-line-length
import {createDtypeAttr, createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttrFromIndex, createTensorAttr} from './test_helper';

describe('creation', () => {
  let node: Node;
  const input1 = [tfc.tensor1d([1, 2, 3])];
  const input2 = [tfc.scalar(1)];
  const context = new ExecutionContext({});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'creation',
      inputNames: ['input1', 'input2'],
      inputs: [],
      params: {x: createTensorAttr(0)},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('fill', () => {
      it('should call tfc.fill', () => {
        spyOn(tfc, 'fill');
        node.op = 'fill';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.params['value'] = createNumberAttrFromIndex(1);

        executeOp(node, {input1, input2}, context);

        expect(tfc.fill).toHaveBeenCalledWith([1, 2, 3], 1);
      });
    });
    describe('linspace', () => {
      it('should call tfc.linspace', () => {
        spyOn(tfc, 'linspace');
        node.op = 'linspace';
        node.params['start'] = createNumberAttrFromIndex(0);
        node.params['stop'] = createNumberAttrFromIndex(1);
        node.params['num'] = createNumberAttrFromIndex(2);
        node.inputNames = ['input', 'input2', 'input3'];
        const input = [tfc.scalar(0)];
        const input3 = [tfc.scalar(2)];
        executeOp(node, {input, input2, input3}, context);

        expect(tfc.linspace).toHaveBeenCalledWith(0, 1, 2);
      });
    });
    describe('oneHot', () => {
      it('should call tfc.oneHot', () => {
        spyOn(tfc, 'oneHot');
        node.op = 'oneHot';
        node.params['indices'] = createNumericArrayAttrFromIndex(0);
        node.params['depth'] = createNumberAttrFromIndex(1);
        node.params['onValue'] = createNumberAttrFromIndex(2);
        node.params['offValue'] = createNumberAttrFromIndex(3);
        node.inputNames = ['input', 'input2', 'input3', 'input4'];
        const input = [tfc.tensor1d([0])];
        const input3 = [tfc.scalar(2)];
        const input4 = [tfc.scalar(3)];
        executeOp(node, {input, input2, input3, input4}, context);

        expect(tfc.oneHot).toHaveBeenCalledWith([0], 1, 2, 3);
      });
    });
    describe('ones', () => {
      it('should call tfc.ones', () => {
        spyOn(tfc, 'ones');
        node.op = 'ones';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.params['dtype'] = createDtypeAttr('float32');
        executeOp(node, {input1}, context);

        expect(tfc.ones).toHaveBeenCalledWith([1, 2, 3], 'float32');
      });
    });
    describe('onesLike', () => {
      it('should call tfc.onesLike', () => {
        spyOn(tfc, 'onesLike');
        node.op = 'onesLike';
        executeOp(node, {input1}, context);

        expect(tfc.onesLike).toHaveBeenCalledWith(input1[0]);
      });
    });
    describe('range', () => {
      it('should call tfc.range', () => {
        spyOn(tfc, 'range');
        node.op = 'range';
        node.params['start'] = createNumberAttrFromIndex(0);
        node.params['stop'] = createNumberAttr(1);
        node.params['step'] = createNumberAttr(2);
        node.params['dtype'] = createDtypeAttr('float32');
        node.inputNames = ['input', 'input2', 'input3'];
        const input = [tfc.scalar(0)];
        const input3 = [tfc.scalar(2)];
        executeOp(node, {input, input2, input3}, context);

        expect(tfc.range).toHaveBeenCalledWith(0, 1, 2, 'float32');
      });
    });
    describe('randomUniform', () => {
      it('should call tfc.randomUniform', () => {
        spyOn(tfc, 'randomUniform');
        node.op = 'randomUniform';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.params['maxval'] = createNumberAttr(1);
        node.params['minval'] = createNumberAttr(0);
        node.params['dtype'] = createDtypeAttr('float32');
        node.params['seed'] = createNumberAttr(0);

        executeOp(node, {input1}, context);

        expect(tfc.randomUniform)
            .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32');
      });
    });
    describe('truncatedNormal', () => {
      it('should call tfc.truncatedNormal', () => {
        spyOn(tfc, 'truncatedNormal');
        node.op = 'truncatedNormal';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.params['stdDev'] = createNumberAttr(1);
        node.params['mean'] = createNumberAttr(0);
        node.params['dtype'] = createDtypeAttr('float32');
        node.params['seed'] = createNumberAttr(0);

        executeOp(node, {input1}, context);

        expect(tfc.truncatedNormal)
            .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32', 0);
      });
    });
    describe('zeros', () => {
      it('should call tfc.zeros', () => {
        spyOn(tfc, 'zeros');
        node.op = 'zeros';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.params['dtype'] = createDtypeAttr('float32');
        executeOp(node, {input1}, context);

        expect(tfc.zeros).toHaveBeenCalledWith([1, 2, 3], 'float32');
      });
    });
    describe('zerosLike', () => {
      it('should call tfc.zerosLike', () => {
        spyOn(tfc, 'zerosLike');
        node.op = 'zerosLike';
        executeOp(node, {input1}, context);

        expect(tfc.zerosLike).toHaveBeenCalledWith(input1[0]);
      });
    });
  });
});
