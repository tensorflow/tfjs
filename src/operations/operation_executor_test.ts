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

import {ExecutionContext} from '../executor/execution_context';

import * as arithmetic from './executors/arithmetic_executor';
import * as basic_math from './executors/basic_math_executor';
import * as convolution from './executors/convolution_executor';
import * as creation from './executors/creation_executor';
import * as graph from './executors/graph_executor';
import * as image from './executors/image_executor';
import * as logical from './executors/logical_executor';
import * as matrices from './executors/matrices_executor';
import * as normalization from './executors/normalization_executor';
import * as reduction from './executors/reduction_executor';
import * as slice_join from './executors/slice_join_executor';
import * as transformation from './executors/transformation_executor';
import {executeOp} from './operation_executor';
import {Node} from './types';

describe('OperationExecutor', () => {
  let node: Node;
  const context = new ExecutionContext({});

  beforeEach(() => {
    node = {
      name: 'test',
      op: 'const',
      category: 'graph',
      inputNames: [],
      inputs: [],
      params: {},
      children: []
    };
  });

  describe('executeOp', () => {
    [arithmetic, basic_math, convolution, creation, image, graph, logical,
     matrices, normalization, reduction, slice_join, transformation]
        .forEach(category => {
          it('should call ' + category.CATEGORY + ' executor', () => {
            spyOn(category, 'executeOp');
            node.category = category.CATEGORY;
            executeOp(node, {}, context);
            expect(category.executeOp).toHaveBeenCalledWith(node, {}, context);
          });
        });
  });
});
