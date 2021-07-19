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

import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {cloneTensor, getParamValue, getTensor} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext): Tensor[] => {
      switch (node.op) {
        case 'Const': {
          return tensorMap[node.name];
        }
        case 'PlaceholderWithDefault':
          const def =
              getParamValue('default', node, tensorMap, context) as Tensor;
          return [getTensor(node.name, tensorMap, context) || def];
        case 'Placeholder':
          return [getTensor(node.name, tensorMap, context)];
        case 'Identity':
        case 'StopGradient':
        case 'FakeQuantWithMinMaxVars': {  // This op is currently ignored.
          const data = getParamValue('x', node, tensorMap, context) as Tensor;
          return [cloneTensor(data)];
        }
        case 'IdentityN':
          return (getParamValue('x', node, tensorMap, context) as Tensor[])
              .map((t: Tensor) => cloneTensor(t));
        case 'Snapshot':
          const snapshot =
              (getParamValue('x', node, tensorMap, context) as Tensor);
          return [cloneTensor(snapshot)];
        case 'Shape':
          return [tfOps.tensor1d(
              (getParamValue('x', node, tensorMap, context) as Tensor).shape,
              'int32')];
        case 'ShapeN':
          return (getParamValue('x', node, tensorMap, context) as Tensor[])
              .map((t: Tensor) => tfOps.tensor1d(t.shape));
        case 'Size':
          return [tfOps.scalar(
              (getParamValue('x', node, tensorMap, context) as Tensor).size,
              'int32')];
        case 'Rank':
          return [tfOps.scalar(
              (getParamValue('x', node, tensorMap, context) as Tensor).rank,
              'int32')];
        case 'NoOp':
          return [tfOps.scalar(1)];
        case 'Print':
          const input = getParamValue('x', node, tensorMap, context) as Tensor;
          const data =
              getParamValue('data', node, tensorMap, context) as Tensor[];
          const message =
              getParamValue('message', node, tensorMap, context) as string;
          const summarize =
              getParamValue('summarize', node, tensorMap, context) as number;
          console.warn(
              'The graph has a tf.print() operation,' +
              'usually used for debugging, which slows down performance.');
          console.log(message);
          for (let i = 0; i < data.length; i++) {
            console.log(Array.prototype.slice.call(data[i].dataSync())
                            .slice(0, summarize));
          }
          return [input];

        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'graph';
