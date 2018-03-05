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

import {executeOp} from './creation_executor';
// tslint:disable-next-line:max-line-length
import {createDtypeAttr, createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttrFromIndex, createTensorAttr} from './test_helper';

describe('creation', () => {
  let node: Node;
  const input1 = dl.Tensor1D.new([1, 2, 3]);
  const input2 = dl.Scalar.new(1);

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
      it('should call dl.fill', () => {
        spyOn(dl, 'fill');
        node.op = 'fill';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.params['value'] = createNumberAttrFromIndex(1);

        executeOp(node, {input1, input2});

        expect(dl.fill).toHaveBeenCalledWith([1, 2, 3], 1);
      });
    });
    describe('linspace', () => {
      it('should call dl.linspace', () => {
        spyOn(dl, 'linspace');
        node.op = 'linspace';
        node.params['start'] = createNumberAttrFromIndex(0);
        node.params['stop'] = createNumberAttrFromIndex(1);
        node.params['num'] = createNumberAttrFromIndex(2);
        node.inputNames = ['input', 'input2', 'input3'];
        const input = dl.Scalar.new(0);
        const input3 = dl.Scalar.new(2);
        executeOp(node, {input, input2, input3});

        expect(dl.linspace).toHaveBeenCalledWith(0, 1, 2);
      });
    });
    describe('oneHot', () => {
      it('should call dl.oneHot', () => {
        spyOn(dl, 'oneHot');
        node.op = 'oneHot';
        node.params['indices'] = createNumericArrayAttrFromIndex(0);
        node.params['depth'] = createNumberAttrFromIndex(1);
        node.params['onValue'] = createNumberAttrFromIndex(2);
        node.params['offValue'] = createNumberAttrFromIndex(3);
        node.inputNames = ['input', 'input2', 'input3', 'input4'];
        const input = dl.Array1D.new([0]);
        const input3 = dl.Scalar.new(2);
        const input4 = dl.Scalar.new(3);
        executeOp(node, {input, input2, input3, input4});

        expect(dl.oneHot).toHaveBeenCalledWith([0], 1, 2, 3);
      });
    });
    describe('ones', () => {
      it('should call dl.ones', () => {
        spyOn(dl, 'ones');
        node.op = 'ones';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.params['dtype'] = createDtypeAttr('float32');
        executeOp(node, {input1});

        expect(dl.ones).toHaveBeenCalledWith([1, 2, 3], 'float32');
      });
    });
    describe('onesLike', () => {
      it('should call dl.onesLike', () => {
        spyOn(dl, 'onesLike');
        node.op = 'onesLike';
        executeOp(node, {input1});

        expect(dl.onesLike).toHaveBeenCalledWith(input1);
      });
    });
    describe('range', () => {
      it('should call dl.range', () => {
        spyOn(dl, 'range');
        node.op = 'range';
        node.params['start'] = createNumberAttrFromIndex(0);
        node.params['stop'] = createNumberAttr(1);
        node.params['step'] = createNumberAttr(2);
        node.params['dtype'] = createDtypeAttr('float32');
        node.inputNames = ['input', 'input2', 'input3'];
        const input = dl.Scalar.new(0);
        const input3 = dl.Scalar.new(2);
        executeOp(node, {input, input2, input3});

        expect(dl.range).toHaveBeenCalledWith(0, 1, 2, 'float32');
      });
    });
    describe('randomUniform', () => {
      it('should call dl.randomUniform', () => {
        spyOn(dl, 'randomUniform');
        node.op = 'randomUniform';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.params['maxval'] = createNumberAttr(1);
        node.params['minval'] = createNumberAttr(0);
        node.params['dtype'] = createDtypeAttr('float32');
        node.params['seed'] = createNumberAttr(0);

        executeOp(node, {input1});

        expect(dl.randomUniform)
            .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32');
      });
    });
    describe('truncatedNormal', () => {
      it('should call dl.truncatedNormal', () => {
        spyOn(dl, 'truncatedNormal');
        node.op = 'truncatedNormal';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.inputNames = ['input1'];
        node.params['stdDev'] = createNumberAttr(1);
        node.params['mean'] = createNumberAttr(0);
        node.params['dtype'] = createDtypeAttr('float32');
        node.params['seed'] = createNumberAttr(0);

        executeOp(node, {input1});

        expect(dl.truncatedNormal)
            .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32', 0);
      });
    });
    describe('zeros', () => {
      it('should call dl.zeros', () => {
        spyOn(dl, 'zeros');
        node.op = 'zeros';
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.params['dtype'] = createDtypeAttr('float32');
        executeOp(node, {input1});

        expect(dl.zeros).toHaveBeenCalledWith([1, 2, 3], 'float32');
      });
    });
    describe('zerosLike', () => {
      it('should call dl.zerosLike', () => {
        spyOn(dl, 'zerosLike');
        node.op = 'zerosLike';
        executeOp(node, {input1});

        expect(dl.zerosLike).toHaveBeenCalledWith(input1);
      });
    });
  });
});
