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

import {NamedTensorsMap} from '../data/types';
import {ExecutionContext} from '../executor/execution_context';

import * as arithmetic from './executors/arithmetic_executor';
import * as basicMath from './executors/basic_math_executor';
import * as control from './executors/control_executor';
import * as convolution from './executors/convolution_executor';
import * as creation from './executors/creation_executor';
import * as dynamic from './executors/dynamic_executor';
import * as evaluation from './executors/evaluation_executor';
import * as graph from './executors/graph_executor';
import * as image from './executors/image_executor';
import * as logical from './executors/logical_executor';
import * as matrices from './executors/matrices_executor';
import * as normalization from './executors/normalization_executor';
import * as reduction from './executors/reduction_executor';
import * as sliceJoin from './executors/slice_join_executor';
import * as spectral from './executors/spectral_executor';
import * as transformation from './executors/transformation_executor';
import {Node} from './types';

/**
 * Executes the op defined by the node object.
 * @param node
 * @param tensorMap contains tensors for executed nodes and weights
 */
export function executeOp(
    node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext): tfc.Tensor[]|Promise<tfc.Tensor[]> {
  const value =
      ((node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext) => {
        switch (node.category) {
          case 'arithmetic':
            return arithmetic.executeOp(node, tensorMap, context);
          case 'basic_math':
            return basicMath.executeOp(node, tensorMap, context);
          case 'control':
            return control.executeOp(node, tensorMap, context);
          case 'convolution':
            return convolution.executeOp(node, tensorMap, context);
          case 'creation':
            return creation.executeOp(node, tensorMap, context);
          case 'dynamic':
            return dynamic.executeOp(node, tensorMap, context);
          case 'evaluation':
            return evaluation.executeOp(node, tensorMap, context);
          case 'image':
            return image.executeOp(node, tensorMap, context);
          case 'graph':
            return graph.executeOp(node, tensorMap, context);
          case 'logical':
            return logical.executeOp(node, tensorMap, context);
          case 'matrices':
            return matrices.executeOp(node, tensorMap, context);
          case 'normalization':
            return normalization.executeOp(node, tensorMap, context);
          case 'reduction':
            return reduction.executeOp(node, tensorMap, context);
          case 'slice_join':
            return sliceJoin.executeOp(node, tensorMap, context);
          case 'spectral':
            return spectral.executeOp(node, tensorMap, context);
          case 'transformation':
            return transformation.executeOp(node, tensorMap, context);
          default:
            throw TypeError(`Node type ${node.op} is not implemented`);
        }
      })(node, tensorMap, context);
  if (value instanceof Promise) {
    return value.then((data) => [].concat(data));
  }
  return [].concat(value);
}
