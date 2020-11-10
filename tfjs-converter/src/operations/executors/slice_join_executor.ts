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

import {Scalar, Tensor, Tensor1D, tidy, util} from '@tensorflow/tfjs-core';
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
        case 'ConcatV2':
        case 'Concat': {
          const n = getParamValue('n', node, tensorMap, context) as number;
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          let inputs =
              getParamValue('tensors', node, tensorMap, context) as Tensor[];
          inputs = inputs.slice(0, n);
          return [tfOps.concat(inputs, axis)];
        }
        case 'GatherV2':
        case 'Gather': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          const input = getParamValue('x', node, tensorMap, context) as Tensor;
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor1D;
          return [tfOps.gather(input, tfOps.cast(indices, 'int32'), axis)];
        }
        case 'ReverseV2':
        case 'Reverse': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const input = getParamValue('x', node, tensorMap, context) as Tensor;
          return [tfOps.reverse(input, axis)];
        }
        case 'Slice': {
          // tslint:disable-next-line:no-any
          const begin = getParamValue('begin', node, tensorMap, context) as any;
          // tslint:disable-next-line:no-any
          const size = getParamValue('size', node, tensorMap, context) as any;
          return [tfOps.slice(
              getParamValue('x', node, tensorMap, context) as Tensor, begin,
              size)];
        }
        case 'StridedSlice': {
          const begin =
              getParamValue('begin', node, tensorMap, context) as number[];
          const end =
              getParamValue('end', node, tensorMap, context) as number[];
          const strides =
              getParamValue('strides', node, tensorMap, context) as number[];
          const beginMask =
              getParamValue('beginMask', node, tensorMap, context) as number;
          const endMask =
              getParamValue('endMask', node, tensorMap, context) as number;
          const ellipsisMask =
              getParamValue('ellipsisMask', node, tensorMap, context) as number;
          const newAxisMask =
              getParamValue('newAxisMask', node, tensorMap, context) as number;
          const shrinkAxisMask =
              getParamValue('shrinkAxisMask', node, tensorMap, context) as
              number;
          const tensor = getParamValue('x', node, tensorMap, context) as Tensor;

          return [tfOps.stridedSlice(
              tensor, begin, end, strides, beginMask, endMask, ellipsisMask,
              newAxisMask, shrinkAxisMask)];
        }
        case 'Pack': {
          return tidy(() => {
            const axis =
                getParamValue('axis', node, tensorMap, context) as number;
            const tensors =
                getParamValue('tensors', node, tensorMap, context) as Tensor[];
            // Reshape the tensors to the first tensor's shape if they don't
            // match.
            const shape = tensors[0].shape;
            const squeezedShape = tfOps.squeeze(tensors[0]).shape;
            const mapped = tensors.map(tensor => {
              const sameShape = util.arraysEqual(tensor.shape, shape);
              if (!sameShape &&
                  !util.arraysEqual(
                      tfOps.squeeze(tensor).shape, squeezedShape)) {
                throw new Error('the input tensors shape does not match');
              }
              return sameShape ? tensor : tfOps.reshape(tensor, shape);
            });
            return [tfOps.stack(mapped, axis)];
          });
        }
        case 'Unpack': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          const tensor =
              getParamValue('tensor', node, tensorMap, context) as Tensor;
          return tfOps.unstack(tensor, axis);
        }
        case 'Tile': {
          const reps =
              getParamValue('reps', node, tensorMap, context) as number[];
          return [tfOps.tile(
              getParamValue('x', node, tensorMap, context) as Tensor, reps)];
        }
        case 'Split':
        case 'SplitV': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          const numOrSizeSplits =
              getParamValue('numOrSizeSplits', node, tensorMap, context) as
                  number |
              number[];
          const tensor = getParamValue('x', node, tensorMap, context) as Tensor;

          return tfOps.split(tensor, numOrSizeSplits, axis);
        }
        case 'ScatterNd': {
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor;
          const values =
              getParamValue('values', node, tensorMap, context) as Tensor;
          const shape =
              getParamValue('shape', node, tensorMap, context) as number[];
          return [tfOps.scatterND(indices, values, shape)];
        }
        case 'GatherNd': {
          const x = getParamValue('x', node, tensorMap, context) as Tensor;
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor;
          return [tfOps.gatherND(x, indices)];
        }
        case 'SparseToDense': {
          const indices =
              getParamValue('sparseIndices', node, tensorMap, context) as
              Tensor;
          const shape =
              getParamValue('outputShape', node, tensorMap, context) as
              number[];
          const sparseValues =
              getParamValue('sparseValues', node, tensorMap, context) as Tensor;
          const defaultValue =
              getParamValue('defaultValue', node, tensorMap, context) as Scalar;
          return [tfOps.sparseToDense(
              indices, sparseValues, shape,
              sparseValues.dtype === defaultValue.dtype ?
                  defaultValue :
                  tfOps.cast(defaultValue, sparseValues.dtype))];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'slice_join';
