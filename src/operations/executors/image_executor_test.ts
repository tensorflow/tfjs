/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import * as image from '../op_list/image';
import {Node, OpMapper} from '../types';

import {executeOp} from './image_executor';
// tslint:disable-next-line:max-line-length
import {createBoolAttr, createNumberAttr, createNumericArrayAttrFromIndex, createStrAttr, createTensorAttr, validateParam} from './test_helper';

describe('image', () => {
  let node: Node;
  const input1 = [tfc.tensor1d([1])];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'input1',
      op: '',
      category: 'image',
      inputNames: ['input1'],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('ResizeBilinear', () => {
      it('should return input', () => {
        node.op = 'ResizeBilinear';
        node.inputParams['images'] = createTensorAttr(0);
        node.inputParams['size'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['alignCorners'] = createBoolAttr(true);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfc.tensor1d([1, 2])];
        spyOn(tfc.image, 'resizeBilinear');
        executeOp(node, {input1, input2}, context);
        expect(tfc.image.resizeBilinear)
            .toHaveBeenCalledWith(input1[0], [1, 2], true);
      });
      it('should match json def', () => {
        node.op = 'ResizeBilinear';
        node.inputParams['images'] = createTensorAttr(0);
        node.inputParams['size'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['alignCorners'] = createBoolAttr(true);

        expect(validateParam(node, image.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('ResizeNearestNeighbor', () => {
      it('should return input', () => {
        node.op = 'ResizeNearestNeighbor';
        node.inputParams['images'] = createTensorAttr(0);
        node.inputParams['size'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['alignCorners'] = createBoolAttr(true);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfc.tensor1d([1, 2])];
        spyOn(tfc.image, 'resizeNearestNeighbor');
        executeOp(node, {input1, input2}, context);
        expect(tfc.image.resizeNearestNeighbor)
            .toHaveBeenCalledWith(input1[0], [1, 2], true);
      });
      it('should match json def', () => {
        node.op = 'ResizeNearestNeighbor';
        node.inputParams['images'] = createTensorAttr(0);
        node.inputParams['size'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['alignCorners'] = createBoolAttr(true);

        expect(validateParam(node, image.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('CropAndResize', () => {
      it('should return input', () => {
        node.op = 'CropAndResize';
        node.inputParams['image'] = createTensorAttr(0);
        node.inputParams['boxes'] = createTensorAttr(1);
        node.inputParams['boxInd'] = createTensorAttr(2);
        node.inputParams['cropSize'] = createNumericArrayAttrFromIndex(3);
        node.attrParams['method'] = createStrAttr('bilinear');
        node.attrParams['extrapolationValue'] = createNumberAttr(0.5);
        node.inputNames = ['input1', 'input2', 'input3', 'input4'];

        spyOn(tfc.image, 'cropAndResize');
        const input2 = [tfc.tensor1d([2])];
        const input3 = [tfc.tensor1d([3])];
        const input4 = [tfc.tensor1d([4, 5])];

        executeOp(node, {input1, input2, input3, input4}, context);
        expect(tfc.image.cropAndResize)
            .toHaveBeenCalledWith(
                input1[0], input2[0], input3[0], [4, 5], 'bilinear', 0.5);
      });

      it('should match json def', () => {
        node.op = 'CropAndResize';
        node.inputParams['image'] = createTensorAttr(0);
        node.inputParams['boxes'] = createTensorAttr(1);
        node.inputParams['boxInd'] = createTensorAttr(2);
        node.inputParams['cropSize'] = createNumericArrayAttrFromIndex(3);
        node.attrParams['method'] = createStrAttr('bilinear');
        node.attrParams['extrapolationValue'] = createNumberAttr(0.5);
        node.inputNames = ['input1', 'input2', 'input3', 'input4'];

        expect(validateParam(node, image.json as OpMapper[])).toBeTruthy();
      });
    });
  });
});
