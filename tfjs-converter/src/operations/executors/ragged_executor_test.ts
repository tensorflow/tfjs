/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import * as ragged from '../op_list/ragged';
import {Node} from '../types';

import {executeOp} from './ragged_executor';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';
import {createNumberAttr, createStrArrayAttr, createTensorAttr, createTensorsAttr, validateParam} from './test_helper';

describe('ragged', () => {
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
      category: 'ragged',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('RaggedGather', () => {
      const OUTPUT_RAGGED_RANK = 2;
      beforeEach(() => {
        node.op = 'RaggedGather';
        node.inputParams = {
          paramsNestedSplits: createTensorsAttr(0, 2),
          paramsDenseValues: createTensorAttr(2),
          indices: createTensorAttr(3),
        };
        node.attrParams = {
          outputRaggedRank: createNumberAttr(OUTPUT_RAGGED_RANK)
        };
      });

      it('should call tfOps.ragged.raggedGather', async () => {
        node.inputNames = [
          'paramsNestedSplits1', 'paramsNestedSplits2', 'paramsDenseValues',
          'indices'
        ];

        const paramsNestedSplits1 =
            [tfOps.tensor1d([0, 1, 3, 3, 5, 6], 'int32')];
        const paramsNestedSplits2 =
            [tfOps.tensor1d([0, 0, 2, 3, 5, 8, 9], 'int32')];
        const paramsDenseValues =
            [tfOps.tensor1d([.1, .2, .3, .4, .5, .6, .7, .8, .9])];
        const indices = [tfOps.tensor1d([2, 1, 0, 2, 3], 'int32')];

        const result = executeOp(
                           node, {
                             paramsNestedSplits1,
                             paramsNestedSplits2,
                             paramsDenseValues,
                             indices
                           },
                           context, spyOpsAsTfOps) as Tensor[];

        expect(spyOps.raggedGather)
            .toHaveBeenCalledWith(
                [paramsNestedSplits1[0], paramsNestedSplits2[0]],
                paramsDenseValues[0], indices[0], OUTPUT_RAGGED_RANK);

        test_util.expectArraysClose(await result[0].data(), [0, 0, 2, 3, 3, 5]);
        test_util.expectArraysClose(await result[1].data(), [0, 2, 3, 3, 5, 8]);

        test_util.expectArraysClose(
            await result[2].data(), [.1, .2, .3, .4, .5, .6, .7, .8]);
      });
    });

    it('should match json def', () => {
      expect(validateParam(node, ragged.json)).toBeTruthy();
    });

    describe('RaggedRange', () => {
      beforeEach(() => {
        node.op = 'RaggedRange';
        node.inputParams = {
          starts: createTensorAttr(0),
          limits: createTensorAttr(1),
          splits: createTensorAttr(2),
        };
      });

      it('should call tfOps.ragged.raggedRange', async () => {
        node.inputNames = ['starts', 'limits', 'splits'];

        const starts = [tfOps.tensor1d([0, 5, 8, 5], 'int32')];
        const limits = [tfOps.tensor1d([8, 7, 8, 1], 'int32')];
        const splits = [tfOps.tensor1d([2, 1, 1, -1], 'int32')];
        const result =
            executeOp(node, {starts, limits, splits}, context, spyOpsAsTfOps) as
            Tensor[];

        expect(spyOps.raggedRange)
            .toHaveBeenCalledWith(starts[0], limits[0], splits[0]);
        test_util.expectArraysClose(await result[0].data(), [0, 4, 6, 6, 10]);
        test_util.expectArraysClose(
            await result[1].data(), [0, 2, 4, 6, 5, 6, 5, 4, 3, 2]);
      });

      it('should match json def', () => {
        expect(validateParam(node, ragged.json)).toBeTruthy();
      });
    });

    describe('RaggedTensorToTensor', () => {
      const types = ['FIRST_DIM_SIZE', 'VALUE_ROWIDS'];
      beforeEach(() => {
        node.op = 'RaggedTensorToTensor';
        node.inputParams = {
          shape: createTensorAttr(0),
          values: createTensorAttr(1),
          defaultValue: createTensorAttr(2),
          rowPartitionTensors: createTensorsAttr(3, 0)
        };
        node.attrParams = {rowPartitionTypes: createStrArrayAttr(types)};
      });

      it('should call tfOps.ragged.raggedTensorToTensor', async () => {
        node.inputNames = [
          'shape', 'values', 'defaultValue', 'rowPartition1', 'rowPartition2'
        ];

        const shape = [tfOps.tensor1d([4, 4], 'int32')];
        const values = [tfOps.tensor1d([.1, .2, .3, .4, .5, .6, .7, .8, .9])];
        const defaultValue = [tfOps.scalar(1.5)];
        const rowPartition1 = [tfOps.scalar(4, 'int32')];
        const rowPartition2 =
            [tfOps.tensor1d([0, 0, 0, 2, 2, 2, 2, 3, 3], 'int32')];

        const result =
            executeOp(
                node,
                {shape, values, defaultValue, rowPartition1, rowPartition2},
                context, spyOpsAsTfOps) as Tensor[];

        expect(spyOps.raggedTensorToTensor)
            .toHaveBeenCalledWith(
                shape[0], values[0], defaultValue[0],
                [rowPartition1[0], rowPartition2[0]], types);
        test_util.expectArraysClose(await result[0].data(), [
          .1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5, .4, .5, .6, .7, .8, .9, 1.5, 1.5
        ]);
      });

      it('should match json def', () => {
        expect(validateParam(node, ragged.json)).toBeTruthy();
      });
    });
  });
});
