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
import * as tfc from '@tensorflow/tfjs-core';

import {Node} from '../index';
// tslint:disable-next-line:max-line-length
import {createDtypeAttr, createNumberAttr, createNumericArrayAttrFromIndex, createTensorAttr} from './test_helper';
import {executeOp} from './transformation_executor';

describe('transformation', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const input2 = [tfc.tensor1d([1, 1])];

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'transformation',
      inputNames: ['input1'],
      inputs: [],
      params: {x: createTensorAttr(0)},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('cast', () => {
      it('should call tfc.cast', () => {
        spyOn(tfc, 'cast');
        node.op = 'cast';
        node.params.dtype = createDtypeAttr('float32');
        executeOp(node, {input1});

        expect(tfc.cast).toHaveBeenCalledWith(input1[0], 'float32');
      });
    });
    describe('expandDims', () => {
      it('should call tfc.expandDims', () => {
        spyOn(tfc, 'expandDims');
        node.op = 'expandDims';
        node.params.axis = createNumberAttr(1);
        executeOp(node, {input1});

        expect(tfc.expandDims).toHaveBeenCalledWith(input1[0], 1);
      });
    });
    describe('pad', () => {
      it('should call tfc.pad', () => {
        spyOn(tfc, 'pad');
        node.op = 'pad';
        node.params.padding = createNumericArrayAttrFromIndex(1);
        node.params.constantValue = createNumberAttr(1);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(tfc.pad).toHaveBeenCalledWith(input1[0], [1, 1], 1);
      });
    });
    describe('reshape', () => {
      it('should call tfc.reshape', () => {
        spyOn(tfc, 'reshape');
        node.op = 'reshape';
        node.params.shape = createNumericArrayAttrFromIndex(1);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(tfc.reshape).toHaveBeenCalledWith(input1[0], [1, 1]);
      });
    });
    describe('squeeze', () => {
      it('should call tfc.squeeze', () => {
        spyOn(tfc, 'squeeze');
        node.op = 'squeeze';
        node.params.axis = createNumberAttr(1);
        executeOp(node, {input1});

        expect(tfc.squeeze).toHaveBeenCalledWith(input1[0], 1);
      });
    });
  });
});
