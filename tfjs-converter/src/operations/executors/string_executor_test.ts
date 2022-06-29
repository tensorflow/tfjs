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
import * as string from '../op_list/string';
import {Node} from '../types';

import {executeOp} from './string_executor';
import {createBoolAttr, createNumberAttr, createNumericArrayAttr, createStrAttr, createTensorAttr, validateParam} from './test_helper';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';

describe('string', () => {
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
      category: 'string',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: [],
      outputs: []
    };
  });

  describe('executeOp', () => {
    describe('StringNGrams', () => {
      it('should call tfOps.string.stringNGrams', async () => {
        node.op = 'StringNGrams';
        node.inputParams = {
          data: createTensorAttr(0),
          dataSplits: createTensorAttr(1)
        };
        node.attrParams = {
          separator: createStrAttr('|'),
          nGramWidths: createNumericArrayAttr([3]),
          leftPad: createStrAttr('LP'),
          rightPad: createStrAttr('RP'),
          padWidth: createNumberAttr(-1),
          preserveShortSequences: createBoolAttr(false)
        };
        node.inputNames = ['data', 'dataSplits'];
        node.outputs = ['ngrams', 'ngrams_splits'];

        const data = [tfOps.tensor1d(['a', 'b', 'c', 'd', 'e', 'f'], 'string')];
        const dataSplits = [tfOps.tensor1d([0, 4, 6], 'int32')];
        const result = executeOp(node, {data, dataSplits}, context,
                                 spyOpsAsTfOps) as Tensor[];

        expect(spyOps.string.stringNGrams)
            .toHaveBeenCalledWith(
                data[0], dataSplits[0], '|', [3], 'LP', 'RP', -1, false);
        test_util.expectArraysEqual(await result[0].data(), [
          'LP|LP|a', 'LP|a|b', 'a|b|c', 'b|c|d', 'c|d|RP', 'd|RP|RP',  // 0
          'LP|LP|e', 'LP|e|f', 'e|f|RP', 'f|RP|RP'                     // 1
        ]);
        test_util.expectArraysEqual(await result[1].data(), [0, 6, 10]);
      });
      it('should match json def', () => {
        node.op = 'StringNGrams';
        node.inputParams = {
          data: createTensorAttr(0),
          dataSplits: createTensorAttr(1)
        };
        node.outputs = ['ngrams', 'ngrams_splits'];
        expect(validateParam(node, string.json)).toBeTruthy();
      });
    });
    describe('StringSplit', () => {
      it('should call tfOps.string.stringSplit', async () => {
        node.op = 'StringSplit';
        node.inputParams = {
          input: createTensorAttr(0),
          delimiter: createTensorAttr(1)
        };
        node.attrParams = {skipEmpty: createBoolAttr(false)};
        node.inputNames = ['input', 'delimiter'];
        node.outputs = ['indices', 'values', 'shape'];

        const input = [tfOps.tensor1d(['#a', 'b#', '#c#'], 'string')];
        const delimiter = [tfOps.scalar('#', 'string')];
        const result = executeOp(node, {input, delimiter}, context,
                                 spyOpsAsTfOps) as Tensor[];

        expect(spyOps.string.stringSplit)
            .toHaveBeenCalledWith(input[0], delimiter[0], false);
        test_util.expectArraysEqual(
            await result[0].data(), [0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 2, 2]);
        test_util.expectArraysEqual(
            await result[1].data(), ['', 'a', 'b', '', '', 'c', '']);
        test_util.expectArraysEqual(await result[2].data(), [3, 3]);
      });
      it('should match json def', () => {
        node.op = 'StringSplit';
        node.inputParams = {
          input: createTensorAttr(0),
          delimiter: createTensorAttr(1)
        };
        node.outputs = ['indices', 'values', 'shape'];

        expect(validateParam(node, string.json)).toBeTruthy();
      });
    });
    describe('StringToHashBucketFast', () => {
      it('should call tfOps.string.stringToHashBucketFast', async () => {
        node.op = 'StringToHashBucketFast';
        node.inputParams = {input: createTensorAttr(0)};
        node.attrParams = {numBuckets: createNumberAttr(10)};
        node.inputNames = ['input'];

        const input = [tfOps.tensor1d(['a', 'b', 'c', 'd'], 'string')];
        const result = executeOp(node, {input}, context,
                                 spyOpsAsTfOps) as Tensor[];

        expect(spyOps.string.stringToHashBucketFast)
            .toHaveBeenCalledWith(input[0], 10);
        test_util.expectArraysClose(await result[0].data(), [9, 2, 2, 5]);
      });
      it('should match json def', () => {
        node.op = 'StringToHashBucketFast';
        node.inputParams = {input: createTensorAttr(0)};

        expect(validateParam(node, string.json)).toBeTruthy();
      });
    });
  });
});
