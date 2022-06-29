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
import {Tensor, test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {ExecutionContext} from '../../executor/execution_context';
import * as sparse from '../op_list/sparse';
import {Node} from '../types';

import {executeOp} from './sparse_executor';
import {createTensorAttr, validateParam} from './test_helper';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';

describe('sparse', () => {
  let node: Node;
  const context = new ExecutionContext({}, {}, {});
  let spyOps: RecursiveSpy<typeof tfOps>;
  let spyOpsAsTfOps: typeof tfOps;

  beforeEach(() => {
    spyOps = spyOnAllFunctions(tfOps);
    spyOpsAsTfOps = spyOps as unknown as typeof tfOps;
    node = {
      name: 'test',
      op: '',
      category: 'sparse',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('SparseFillEmptyRows', () => {
      it('should call tfOps.sparse.sparseFillEmptyRows', async () => {
        node.op = 'SparseFillEmptyRows';
        node.inputParams = {
          indices: createTensorAttr(0),
          values: createTensorAttr(1),
          denseShape: createTensorAttr(2),
          defaultValue: createTensorAttr(3)
        };
        node.inputNames = ['indices', 'values', 'denseShape', 'defaultValue'];

        const indices = [tfOps.tensor2d(
            [0, 0, 1, 0, 1, 3, 1, 4, 3, 2, 3, 3], [6, 2], 'int32')];
        const values = [tfOps.tensor1d([0, 10, 13, 14, 32, 33], 'int32')];
        const denseShape = [tfOps.tensor1d([5, 6], 'int32')];
        const defaultValue = [tfOps.scalar(-1, 'int32')];
        const result = executeOp(
                           node, {indices, values, denseShape, defaultValue},
                           context, spyOpsAsTfOps) as Tensor[];

        expect(spyOps.sparse.sparseFillEmptyRows)
            .toHaveBeenCalledWith(
                indices[0], values[0], denseShape[0], defaultValue[0]);
        test_util.expectArraysClose(
            await result[0].data(),
            [0, 0, 1, 0, 1, 3, 1, 4, 2, 0, 3, 2, 3, 3, 4, 0]);
        test_util.expectArraysClose(
            await result[1].data(), [0, 10, 13, 14, -1, 32, 33, -1]);
        test_util.expectArraysClose(await result[2].data(), [0, 0, 1, 0, 1]);
        test_util.expectArraysClose(await result[3].data(), [0, 1, 2, 3, 5, 6]);
      });
      it('should match json def', () => {
        node.op = 'SparseFillEmptyRows';
        node.inputParams = {
          indices: createTensorAttr(0),
          values: createTensorAttr(1),
          denseShape: createTensorAttr(2),
          defaultValue: createTensorAttr(3)
        };

        expect(validateParam(node, sparse.json)).toBeTruthy();
      });
    });
    describe('SparseReshape', () => {
      it('should call tfOps.sparse.sparseReshape', async () => {
        node.op = 'SparseReshape';
        node.inputParams = {
          inputIndices: createTensorAttr(0),
          inputShape: createTensorAttr(1),
          newShape: createTensorAttr(2)
        };
        node.inputNames = ['inputIndices', 'inputShape', 'newShape'];

        const inputIndices = [tfOps.tensor2d(
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 3], [5, 3], 'int32')];
        const inputShape = [tfOps.tensor1d([2, 3, 6], 'int32')];
        const newShape = [tfOps.tensor1d([9, -1], 'int32')];
        const result =
            executeOp(node, {inputIndices, inputShape, newShape}, context,
                      spyOpsAsTfOps) as Tensor[];

        expect(spyOps.sparse.sparseReshape)
            .toHaveBeenCalledWith(inputIndices[0], inputShape[0], newShape[0]);
        test_util.expectArraysClose(
            await result[0].data(), [0, 0, 0, 1, 1, 2, 4, 2, 8, 1]);
        test_util.expectArraysClose(await result[1].data(), [9, 4]);
      });

      it('should match json def', () => {
        node.op = 'SparseReshape';
        node.inputParams = {
          inputIndices: createTensorAttr(0),
          inputShape: createTensorAttr(1),
          newShape: createTensorAttr(2)
        };

        expect(validateParam(node, sparse.json)).toBeTruthy();
      });
    });
    describe('SparseSegmentMean', () => {
      it('should call tfOps.sparse.sparseSegmentMean', async () => {
        node.op = 'SparseSegmentMean';
        node.inputParams = {
          data: createTensorAttr(0),
          indices: createTensorAttr(1),
          segmentIds: createTensorAttr(2)
        };
        node.inputNames = ['data', 'indices', 'segmentIds'];

        const data = [tfOps.tensor2d(
            [1, 2, 3, 4, -1, -2, -3, -4, 6, 7, 8, 9], [3, 4], 'float32')];
        const indices = [tfOps.tensor1d([0, 1, 2], 'int32')];
        const segmentIds = [tfOps.tensor1d([0, 1, 1], 'int32')];
        const result =
            executeOp(node, {data, indices, segmentIds}, context,
                      spyOpsAsTfOps) as Tensor[];

        expect(spyOps.sparse.sparseSegmentMean)
            .toHaveBeenCalledWith(data[0], indices[0], segmentIds[0]);
        test_util.expectArraysClose(
            await result[0].data(), [1.0, 2.0, 3.0, 4.0, 2.5, 2.5, 2.5, 2.5]);
      });
      it('should match json def', () => {
        node.op = 'SparseSegmentMean';
        node.inputParams = {
          data: createTensorAttr(0),
          indices: createTensorAttr(1),
          segmentIds: createTensorAttr(2)
        };

        expect(validateParam(node, sparse.json)).toBeTruthy();
      });
    });
    describe('SparseSegmentSum', () => {
      it('should call tfOps.sparse.sparseSegmentSum', async () => {
        node.op = 'SparseSegmentSum';
        node.inputParams = {
          data: createTensorAttr(0),
          indices: createTensorAttr(1),
          segmentIds: createTensorAttr(2)
        };
        node.inputNames = ['data', 'indices', 'segmentIds'];

        const data = [tfOps.tensor2d(
            [1, 2, 3, 4, -1, -2, -3, -4, 5, 6, 7, 8], [3, 4], 'int32')];
        const indices = [tfOps.tensor1d([0, 1], 'int32')];
        const segmentIds = [tfOps.tensor1d([0, 0], 'int32')];
        const result =
            executeOp(node, {data, indices, segmentIds}, context,
                      spyOpsAsTfOps) as Tensor[];

        expect(spyOps.sparse.sparseSegmentSum)
            .toHaveBeenCalledWith(data[0], indices[0], segmentIds[0]);
        test_util.expectArraysClose(await result[0].data(), [0, 0, 0, 0]);
      });
      it('should match json def', () => {
        node.op = 'SparseSegmentSum';
        node.inputParams = {
          data: createTensorAttr(0),
          indices: createTensorAttr(1),
          segmentIds: createTensorAttr(2)
        };

        expect(validateParam(node, sparse.json)).toBeTruthy();
      });
    });
  });
});
