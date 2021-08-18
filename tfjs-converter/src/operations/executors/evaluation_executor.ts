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

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext):
        Tensor[] => {
          switch (node.op) {
            case 'TopKV2': {
              const x = getParamValue('x', node, tensorMap, context) as Tensor;
              const k = getParamValue('k', node, tensorMap, context) as number;
              const sorted =
                  getParamValue('sorted', node, tensorMap, context) as boolean;
              const result = tfOps.topk(x, k, sorted);
              return [result.values, result.indices];
            }
            case 'Unique': {
              const x = getParamValue('x', node, tensorMap, context) as Tensor;
              const result = tfOps.unique(x);
              return [result.values, result.indices];
            }
            case 'UniqueV2': {
              const x = getParamValue('x', node, tensorMap, context) as Tensor;
              const axis =
                  getParamValue('axis', node, tensorMap, context) as number;
              const result = tfOps.unique(x, axis);
              return [result.values, result.indices];
            }
            default:
              throw TypeError(`Node type ${node.op} is not implemented`);
          }
        };

export const CATEGORY = 'evaluation';
