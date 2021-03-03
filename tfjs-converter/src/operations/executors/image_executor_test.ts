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
import * as image from '../op_list/image';
import {Node} from '../types';

import {executeOp} from './image_executor';
import {createBoolAttr, createNumberAttr, createNumericArrayAttrFromIndex, createStrAttr, createTensorAttr, validateParam} from './test_helper';

describe('image', () => {
  let node: Node;
  const input1 = [tfOps.tensor1d([1])];
  const context = new ExecutionContext({}, {}, {});

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
        node.attrParams['halfPixelCenters'] = createBoolAttr(true);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfOps.tensor1d([1, 2])];
        spyOn(tfOps.image, 'resizeBilinear');
        executeOp(node, {input1, input2}, context);
        expect(tfOps.image.resizeBilinear)
            .toHaveBeenCalledWith(input1[0], [1, 2], true, true);
      });
      it('should match json def', () => {
        node.op = 'ResizeBilinear';
        node.inputParams['images'] = createTensorAttr(0);
        node.inputParams['size'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['alignCorners'] = createBoolAttr(true);
        node.attrParams['halfPixelCenters'] = createBoolAttr(true);

        expect(validateParam(node, image.json)).toBeTruthy();
      });
    });
    describe('ResizeNearestNeighbor', () => {
      it('should return input', () => {
        node.op = 'ResizeNearestNeighbor';
        node.inputParams['images'] = createTensorAttr(0);
        node.inputParams['size'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['alignCorners'] = createBoolAttr(true);
        node.attrParams['halfPixelCenters'] = createBoolAttr(true);
        node.inputNames = ['input1', 'input2'];
        const input2 = [tfOps.tensor1d([1, 2])];
        spyOn(tfOps.image, 'resizeNearestNeighbor');
        executeOp(node, {input1, input2}, context);
        expect(tfOps.image.resizeNearestNeighbor)
            .toHaveBeenCalledWith(input1[0], [1, 2], true, true);
      });
      it('should match json def', () => {
        node.op = 'ResizeNearestNeighbor';
        node.inputParams['images'] = createTensorAttr(0);
        node.inputParams['size'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['alignCorners'] = createBoolAttr(true);
        node.attrParams['halfPixelCenters'] = createBoolAttr(true);

        expect(validateParam(node, image.json)).toBeTruthy();
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

        spyOn(tfOps.image, 'cropAndResize');
        const input2 = [tfOps.tensor1d([2])];
        const input3 = [tfOps.tensor1d([3])];
        const input4 = [tfOps.tensor1d([4, 5])];

        executeOp(node, {input1, input2, input3, input4}, context);
        expect(tfOps.image.cropAndResize)
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

        expect(validateParam(node, image.json)).toBeTruthy();
      });
    });
  });
});
