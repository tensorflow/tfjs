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

import {Tensor, Tensor1D, Tensor2D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpAsyncExecutor, Node} from '../types';

import {getParamValue} from './utils';

function nmsParams(
    node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext) {
  const boxes = getParamValue('boxes', node, tensorMap, context) as Tensor;
  const scores = getParamValue('scores', node, tensorMap, context) as Tensor;
  const maxOutputSize =
      getParamValue('maxOutputSize', node, tensorMap, context) as number;
  const iouThreshold =
      getParamValue('iouThreshold', node, tensorMap, context) as number;
  const scoreThreshold =
      getParamValue('scoreThreshold', node, tensorMap, context) as number;
  const softNmsSigma =
      getParamValue('softNmsSigma', node, tensorMap, context) as number;

  return {
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
    softNmsSigma
  };
}

export const executeOp: InternalOpAsyncExecutor = async(
    node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext): Promise<Tensor[]> => {
  switch (node.op) {
    case 'NonMaxSuppressionV5': {
      const {
        boxes,
        scores,
        maxOutputSize,
        iouThreshold,
        scoreThreshold,
        softNmsSigma
      } = nmsParams(node, tensorMap, context);

      const result = await tfOps.image.nonMaxSuppressionWithScoreAsync(
          boxes as Tensor2D, scores as Tensor1D, maxOutputSize, iouThreshold,
          scoreThreshold, softNmsSigma);

      return [result.selectedIndices, result.selectedScores];
    }
    case 'NonMaxSuppressionV4': {
      const {boxes, scores, maxOutputSize, iouThreshold, scoreThreshold} =
          nmsParams(node, tensorMap, context);

      const padToMaxOutputSize =
          getParamValue('padToMaxOutputSize', node, tensorMap, context) as
          boolean;

      const result = await tfOps.image.nonMaxSuppressionPaddedAsync(
          boxes as Tensor2D, scores as Tensor1D, maxOutputSize, iouThreshold,
          scoreThreshold, padToMaxOutputSize);

      return [result.selectedIndices, result.validOutputs];
    }
    case 'NonMaxSuppressionV3':
    case 'NonMaxSuppressionV2': {
      const {boxes, scores, maxOutputSize, iouThreshold, scoreThreshold} =
          nmsParams(node, tensorMap, context);

      return [await tfOps.image.nonMaxSuppressionAsync(
          boxes as Tensor2D, scores as Tensor1D, maxOutputSize, iouThreshold,
          scoreThreshold)];
    }
    case 'Where': {
      const condition = tfOps.cast(
          (getParamValue('condition', node, tensorMap, context) as Tensor),
          'bool');
      const result = [await tfOps.whereAsync(condition)];
      condition.dispose();
      return result;
    }
    case 'ListDiff': {
      return tfOps.setdiff1dAsync(
          getParamValue('x', node, tensorMap, context) as Tensor,
          getParamValue('y', node, tensorMap, context) as Tensor);
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'dynamic';
