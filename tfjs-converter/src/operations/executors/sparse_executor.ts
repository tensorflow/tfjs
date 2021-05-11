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

import {Scalar, Tensor, Tensor1D, Tensor2D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext): Tensor[] => {
      switch (node.op) {
        case 'SparseFillEmptyRows': {
          const {
            outputIndices,
            outputValues,
            emptyRowIndicator,
            reverseIndexMap
          } =
              tfOps.sparse.sparseFillEmptyRows(
                  getParamValue('indices', node, tensorMap, context) as
                      Tensor2D,
                  getParamValue('values', node, tensorMap, context) as Tensor1D,
                  getParamValue('denseShape', node, tensorMap, context) as
                      Tensor1D,
                  getParamValue('defaultValue', node, tensorMap, context) as
                      Scalar);
          return [
            outputIndices, outputValues, emptyRowIndicator, reverseIndexMap
          ];
        }
        case 'SparseReshape': {
          const {outputIndices, outputShape} = tfOps.sparse.sparseReshape(
              getParamValue('inputIndices', node, tensorMap, context) as
                  Tensor2D,
              getParamValue('inputShape', node, tensorMap, context) as Tensor1D,
              getParamValue('newShape', node, tensorMap, context) as Tensor1D);
          return [outputIndices, outputShape];
        }
        case 'SparseSegmentMean': {
          const outputData = tfOps.sparse.sparseSegmentMean(
              getParamValue('data', node, tensorMap, context) as Tensor,
              getParamValue('indices', node, tensorMap, context) as Tensor1D,
              getParamValue('segmentIds', node, tensorMap, context) as
                  Tensor1D);
          return [outputData];
        }
        case 'SparseSegmentSum': {
          const outputData = tfOps.sparse.sparseSegmentSum(
              getParamValue('data', node, tensorMap, context) as Tensor,
              getParamValue('indices', node, tensorMap, context) as Tensor1D,
              getParamValue('segmentIds', node, tensorMap, context) as
                  Tensor1D);
          return [outputData];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'sparse';
