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
import {memory} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {ExecutionContext} from '../../executor/execution_context';
import * as dynamic from '../op_list/dynamic';
import {Node} from '../types';

import {executeOp} from './dynamic_executor';
import {createBoolAttr, createNumberAttrFromIndex, createTensorAttr, validateParam} from './test_helper';

describe('dynamic', () => {
  let node: Node;
  const input1 = [tfOps.tensor1d([1])];
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'input1',
      op: '',
      category: 'dynamic',
      inputNames: ['input1'],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('NonMaxSuppressionV2', () => {
      it('should return input', () => {
        node.op = 'NonMaxSuppressionV2';
        node.inputParams['boxes'] = createTensorAttr(0);
        node.inputParams['scores'] = createTensorAttr(1);
        node.inputParams['maxOutputSize'] = createNumberAttrFromIndex(2);
        node.inputParams['iouThreshold'] = createNumberAttrFromIndex(3);
        node.inputParams['scoreThreshold'] = createNumberAttrFromIndex(4);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];
        const input2 = [tfOps.tensor1d([1])];
        const input3 = [tfOps.tensor1d([1])];
        const input4 = [tfOps.tensor1d([1])];
        const input5 = [tfOps.tensor1d([1])];
        spyOn(tfOps.image, 'nonMaxSuppressionAsync');
        const result =
            executeOp(node, {input1, input2, input3, input4, input5}, context);
        expect(tfOps.image.nonMaxSuppressionAsync)
            .toHaveBeenCalledWith(input1[0], input2[0], 1, 1, 1);
        expect(result instanceof Promise).toBeTruthy();
      });
      it('should match json def', () => {
        node.op = 'NonMaxSuppressionV2';
        node.inputParams['boxes'] = createTensorAttr(0);
        node.inputParams['scores'] = createTensorAttr(1);
        node.inputParams['maxOutputSize'] = createNumberAttrFromIndex(2);
        node.inputParams['iouThreshold'] = createNumberAttrFromIndex(3);
        node.inputParams['scoreThreshold'] = createNumberAttrFromIndex(4);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];

        expect(validateParam(node, dynamic.json, 'NonMaxSuppressionV3'))
            .toBeTruthy();
      });
    });
    describe('NonMaxSuppressionV3', () => {
      it('should return input', () => {
        node.op = 'NonMaxSuppressionV3';
        node.inputParams['boxes'] = createTensorAttr(0);
        node.inputParams['scores'] = createTensorAttr(1);
        node.inputParams['maxOutputSize'] = createNumberAttrFromIndex(2);
        node.inputParams['iouThreshold'] = createNumberAttrFromIndex(3);
        node.inputParams['scoreThreshold'] = createNumberAttrFromIndex(4);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];
        const input2 = [tfOps.tensor1d([1])];
        const input3 = [tfOps.tensor1d([1])];
        const input4 = [tfOps.tensor1d([1])];
        const input5 = [tfOps.tensor1d([1])];
        spyOn(tfOps.image, 'nonMaxSuppressionAsync');
        const result =
            executeOp(node, {input1, input2, input3, input4, input5}, context);
        expect(tfOps.image.nonMaxSuppressionAsync)
            .toHaveBeenCalledWith(input1[0], input2[0], 1, 1, 1);
        expect(result instanceof Promise).toBeTruthy();
      });
      it('should match json def', () => {
        node.op = 'NonMaxSuppressionV3';
        node.inputParams['boxes'] = createTensorAttr(0);
        node.inputParams['scores'] = createTensorAttr(1);
        node.inputParams['maxOutputSize'] = createNumberAttrFromIndex(2);
        node.inputParams['iouThreshold'] = createNumberAttrFromIndex(3);
        node.inputParams['scoreThreshold'] = createNumberAttrFromIndex(4);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];

        expect(validateParam(node, dynamic.json, 'NonMaxSuppressionV3'))
            .toBeTruthy();
      });
    });

    describe('NonMaxSuppressionV4', () => {
      it('should return input', () => {
        node.op = 'NonMaxSuppressionV4';
        node.inputParams['boxes'] = createTensorAttr(0);
        node.inputParams['scores'] = createTensorAttr(1);
        node.inputParams['maxOutputSize'] = createNumberAttrFromIndex(2);
        node.inputParams['iouThreshold'] = createNumberAttrFromIndex(3);
        node.inputParams['scoreThreshold'] = createNumberAttrFromIndex(4);
        node.attrParams['padToMaxOutputSize'] = createBoolAttr(true);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];
        const input2 = [tfOps.tensor1d([1])];
        const input3 = [tfOps.tensor1d([1])];
        const input4 = [tfOps.tensor1d([1])];
        const input5 = [tfOps.tensor1d([1])];
        spyOn(tfOps.image, 'nonMaxSuppressionPaddedAsync').and.returnValue({});
        const result =
            executeOp(node, {input1, input2, input3, input4, input5}, context);
        expect(tfOps.image.nonMaxSuppressionPaddedAsync)
            .toHaveBeenCalledWith(input1[0], input2[0], 1, 1, 1, true);
        expect(result instanceof Promise).toBeTruthy();
      });
      it('should match json def', () => {
        node.op = 'NonMaxSuppressionV4';
        node.inputParams['boxes'] = createTensorAttr(0);
        node.inputParams['scores'] = createTensorAttr(1);
        node.inputParams['maxOutputSize'] = createNumberAttrFromIndex(2);
        node.inputParams['iouThreshold'] = createNumberAttrFromIndex(3);
        node.inputParams['scoreThreshold'] = createNumberAttrFromIndex(4);
        node.attrParams['padToMaxOutputSize'] = createBoolAttr(true);
        node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];

        expect(validateParam(node, dynamic.json, 'NonMaxSuppressionV4'))
            .toBeTruthy();
      });
    });

    describe('NonMaxSuppressionV5', () => {
      it('should return input', () => {
        node.op = 'NonMaxSuppressionV5';
        node.inputParams['boxes'] = createTensorAttr(0);
        node.inputParams['scores'] = createTensorAttr(1);
        node.inputParams['maxOutputSize'] = createNumberAttrFromIndex(2);
        node.inputParams['iouThreshold'] = createNumberAttrFromIndex(3);
        node.inputParams['scoreThreshold'] = createNumberAttrFromIndex(4);
        node.inputParams['softNmsSigma'] = createNumberAttrFromIndex(5);
        node.inputNames =
            ['input1', 'input2', 'input3', 'input4', 'input5', 'input6'];
        const input2 = [tfOps.tensor1d([1])];
        const input3 = [tfOps.tensor1d([1])];
        const input4 = [tfOps.tensor1d([1])];
        const input5 = [tfOps.tensor1d([1])];
        const input6 = [tfOps.tensor1d([1])];
        spyOn(tfOps.image, 'nonMaxSuppressionWithScoreAsync')
            .and.returnValue({});
        const result = executeOp(
            node, {input1, input2, input3, input4, input5, input6}, context);
        expect(tfOps.image.nonMaxSuppressionWithScoreAsync)
            .toHaveBeenCalledWith(input1[0], input2[0], 1, 1, 1, 1);
        expect(result instanceof Promise).toBeTruthy();
      });
      it('should match json def', () => {
        node.op = 'NonMaxSuppressionV5';
        node.inputParams['boxes'] = createTensorAttr(0);
        node.inputParams['scores'] = createTensorAttr(1);
        node.inputParams['maxOutputSize'] = createNumberAttrFromIndex(2);
        node.inputParams['iouThreshold'] = createNumberAttrFromIndex(3);
        node.inputParams['scoreThreshold'] = createNumberAttrFromIndex(4);
        node.inputParams['softNmsSigma'] = createNumberAttrFromIndex(5);
        node.inputNames =
            ['input1', 'input2', 'input3', 'input4', 'input5', 'input6'];

        expect(validateParam(node, dynamic.json, 'NonMaxSuppressionV5'))
            .toBeTruthy();
      });
    });

    describe('Where', () => {
      it('should call tfOps.whereAsync', async () => {
        node.op = 'Where';
        node.inputParams = {'condition': createTensorAttr(0)};
        const input1 = [tfOps.scalar(1)];
        spyOn(tfOps, 'whereAsync');

        const result = executeOp(node, {input1}, context);
        expect(
            (tfOps.whereAsync as jasmine.Spy).calls.mostRecent().args[0].dtype)
            .toEqual('bool');
        expect((tfOps.whereAsync as jasmine.Spy)
                   .calls.mostRecent()
                   .args[0]
                   .arraySync())
            .toEqual(1);
        expect(result instanceof Promise).toBeTruthy();
      });
      it('should match json def', () => {
        node.op = 'Where';
        node.inputParams = {'condition': createTensorAttr(0)};

        expect(validateParam(node, dynamic.json)).toBeTruthy();
      });
      it('should not have memory leak', async () => {
        node.op = 'Where';
        node.inputParams = {'condition': createTensorAttr(0)};
        const input1 = [tfOps.scalar(1)];
        spyOn(tfOps, 'whereAsync').and.callThrough();

        const prevCount = memory().numTensors;
        await executeOp(node, {input1}, context);
        const afterCount = memory().numTensors;
        expect(afterCount).toEqual(prevCount + 1);
      });
    });

    describe('ListDiff', () => {
      it('should call tfOps.setdiff1dAsync', async () => {
        node.op = 'ListDiff';
        node.inputNames = ['input1', 'input2'];
        node.inputParams = {'x': createTensorAttr(0), 'y': createTensorAttr(1)};
        const input1 = [tfOps.scalar(1)];
        const input2 = [tfOps.scalar(1)];
        spyOn(tfOps, 'setdiff1dAsync');

        const result = executeOp(node, {input1, input2}, context);
        expect(tfOps.setdiff1dAsync).toHaveBeenCalledWith(input1[0], input2[0]);
        expect(result instanceof Promise).toBeTruthy();
      });
      it('should match json def', () => {
        node.op = 'ListDiff';
        node.inputNames = ['input1', 'input2'];
        node.inputParams = {'x': createTensorAttr(0), 'y': createTensorAttr(1)};

        expect(validateParam(node, dynamic.json)).toBeTruthy();
      });
    });
  });
});
