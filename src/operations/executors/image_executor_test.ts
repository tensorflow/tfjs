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

import {Node} from '../types';

import {executeOp} from './image_executor';
// tslint:disable-next-line:max-line-length
import {createBoolAttr, createNumericArrayAttr, createTensorAttr} from './test_helper';

describe('image', () => {
  let node: Node;
  const input1 = [tfc.tensor1d([1])];
  const context = new ExecutionContext({});

  beforeEach(() => {
    node = {
      name: 'input1',
      op: '',
      category: 'image',
      inputNames: ['input1'],
      inputs: [],
      params: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('resizeBilinear', () => {
      it('should return input', () => {
        node.op = 'resizeBilinear';
        node.params['images'] = createTensorAttr(0);
        node.params['size'] = createNumericArrayAttr([1, 2]);
        node.params['alignCorners'] = createBoolAttr(true);
        spyOn(tfc.image, 'resizeBilinear');
        executeOp(node, {input1}, context);
        expect(tfc.image.resizeBilinear)
            .toHaveBeenCalledWith(input1[0], [1, 2], true);
      });
    });
    describe('resizeNearestNeighbor', () => {
      it('should return input', () => {
        node.op = 'resizeNearestNeighbor';
        node.params['images'] = createTensorAttr(0);
        node.params['size'] = createNumericArrayAttr([1, 2]);
        node.params['alignCorners'] = createBoolAttr(true);
        spyOn(tfc.image, 'resizeNearestNeighbor');
        executeOp(node, {input1}, context);
        expect(tfc.image.resizeNearestNeighbor)
            .toHaveBeenCalledWith(input1[0], [1, 2], true);
      });
    });
  });
});
