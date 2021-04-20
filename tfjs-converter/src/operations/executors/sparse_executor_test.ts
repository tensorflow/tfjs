/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {Tensor} from '@tensorflow/tfjs-core';
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import {ExecutionContext} from '../../executor/execution_context';
import * as sparse from '../op_list/sparse';
import {Node} from '../types';

import {executeOp} from './sparse_executor';
import {createTensorAttr, validateParam} from './test_helper';

describe('sparse', () => {
  let node: Node;
  const inputIndices = [tfOps.tensor2d(
      [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 3], [5, 3], 'int32')];
  const inputShape = [tfOps.tensor1d([2, 3, 6], 'int32')];
  const newShape = [tfOps.tensor1d([9, -1], 'int32')];
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'sparse',
      inputNames: ['inputIndices', 'inputShape', 'newShape'],
      inputs: [],
      inputParams: {
        inputIndices: createTensorAttr(0),
        inputShape: createTensorAttr(1),
        newShape: createTensorAttr(2)
      },
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('SparseReshape', () => {
      it('should call tfOps.sparse.sparseReshape', async () => {
        spyOn(tfOps.sparse, 'sparseReshape').and.callThrough();
        node.op = 'SparseReshape';
        const result =
            executeOp(node, {inputIndices, inputShape, newShape}, context) as
            Tensor[];

        expect(tfOps.sparse.sparseReshape)
            .toHaveBeenCalledWith(inputIndices[0], inputShape[0], newShape[0]);
        expectArraysClose(
            await result[0].data(), [0, 0, 0, 1, 1, 2, 4, 2, 8, 1]);
        expectArraysClose(await result[1].data(), [9, 4]);
      });

      it('should match json def', () => {
        node.op = 'SparseReshape';

        expect(validateParam(node, sparse.json)).toBeTruthy();
      });
    });
  });
});
