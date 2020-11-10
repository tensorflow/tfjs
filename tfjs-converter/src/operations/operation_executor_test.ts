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

import * as tfc from '@tensorflow/tfjs-core';
import {add, mul, scalar, Tensor, test_util} from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../executor/execution_context';
import {ResourceManager} from '../executor/resource_manager';

import {deregisterOp, registerOp} from './custom_op/register';
import * as arithmetic from './executors/arithmetic_executor';
import * as basic_math from './executors/basic_math_executor';
import * as control from './executors/control_executor';
import * as convolution from './executors/convolution_executor';
import * as creation from './executors/creation_executor';
import * as dynamic from './executors/dynamic_executor';
import * as evaluation from './executors/evaluation_executor';
import * as graph from './executors/graph_executor';
import * as hash_table from './executors/hash_table_executor';
import * as image from './executors/image_executor';
import * as logical from './executors/logical_executor';
import * as matrices from './executors/matrices_executor';
import * as normalization from './executors/normalization_executor';
import * as reduction from './executors/reduction_executor';
import * as slice_join from './executors/slice_join_executor';
import * as spectral from './executors/spectral_executor';
import * as transformation from './executors/transformation_executor';
import {executeOp} from './operation_executor';
import {Node} from './types';

describe('OperationExecutor', () => {
  let node: Node;
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: 'const',
      category: 'graph',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    [arithmetic, basic_math, convolution, control, creation, dynamic,
     evaluation, image, graph, logical, matrices, normalization, reduction,
     slice_join, spectral, transformation]
        .forEach(category => {
          it('should call ' + category.CATEGORY + ' executor', () => {
            spyOn(category, 'executeOp');
            node.category = category.CATEGORY;
            executeOp(node, {}, context);
            expect(category.executeOp).toHaveBeenCalledWith(node, {}, context);
          });
        });
    [arithmetic, basic_math, convolution, creation, evaluation, image, graph,
     logical, matrices, normalization, reduction, slice_join, spectral,
     transformation]
        .forEach(category => {
          it('should call tidy around executor', () => {
            spyOn(tfc, 'tidy');
            node.category = category.CATEGORY;
            executeOp(node, {}, context);
            expect(tfc.tidy).toHaveBeenCalled();
          });
        });

    it('hash_table executor should have been called.', () => {
      const resourceManager = new ResourceManager();
      spyOn(hash_table, 'executeOp');
      node.category = hash_table.CATEGORY;
      executeOp(node, {}, context, resourceManager);
      expect(hash_table.executeOp)
          .toHaveBeenCalledWith(node, {}, context, resourceManager);
    });
  });

  describe('custom op executeOp', () => {
    it('should throw exception if custom op is not registered', () => {
      node.category = 'custom';
      expect(() => executeOp(node, {}, context))
          .toThrowError('Custom op const is not registered.');
    });
  });

  describe('custom op executeOp', () => {
    it('should call the registered custom op', async () => {
      registerOp('const', () => [scalar(1)]);
      registerOp('const2', () => [scalar(2)]);
      node.category = 'custom';
      const result = executeOp(node, {}, context) as Tensor[];
      test_util.expectArraysClose(await result[0].data(), [1]);
      deregisterOp('const');
      deregisterOp('const2');
    });

    it('should handle custom op with inputs and attrs', async () => {
      registerOp('const', (node) => {
        const a = node.inputs[0];
        const b = node.inputs[1];
        const attrC = node.attrs['c'] as Tensor;
        const attrD = node.attrs['d'] as number;
        return [add(mul(attrC.dataSync()[0], a), mul(attrD, b))];
      });

      node.category = 'custom';
      node.inputNames = ['a', 'b'];
      node.rawAttrs = {c: {tensor: {}}, d: {i: 3}};
      const result = executeOp(
                         node, {a: [scalar(1)], b: [scalar(2)], c: [scalar(2)]},
                         context) as Tensor[];
      // result = 2 * 1 + 3 * 2
      test_util.expectArraysClose(await result[0].data(), [8]);
      deregisterOp('const');
    });
  });
});
