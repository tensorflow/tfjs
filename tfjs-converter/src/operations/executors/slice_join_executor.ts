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

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor = (node: Node,
                                            tensorMap: NamedTensorsMap,
                                            context: ExecutionContext):
                                               tfc.Tensor[] => {
  switch (node.op) {
    case 'ConcatV2':
    case 'Concat': {
      const n = getParamValue('n', node, tensorMap, context) as number;
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      let inputs =
          getParamValue('tensors', node, tensorMap, context) as tfc.Tensor[];
      inputs = inputs.slice(0, n);
      return [tfc.concat(inputs, axis)];
    }
    case 'GatherV2':
    case 'Gather': {
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      const input = getParamValue('x', node, tensorMap, context) as tfc.Tensor;
      const indices =
          getParamValue('indices', node, tensorMap, context) as tfc.Tensor1D;
      return [tfc.gather(input, indices.asType('int32'), axis)];
    }
    case 'ReverseV2':
    case 'Reverse': {
      const axis = getParamValue('axis', node, tensorMap, context) as number[];
      const input = getParamValue('x', node, tensorMap, context) as tfc.Tensor;
      return [tfc.reverse(input, axis)];
    }
    case 'Slice': {
      // tslint:disable-next-line:no-any
      const begin = getParamValue('begin', node, tensorMap, context) as any;
      // tslint:disable-next-line:no-any
      const size = getParamValue('size', node, tensorMap, context) as any;
      return [tfc.slice(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, begin,
          size)];
    }
    case 'StridedSlice': {
      const begin =
          getParamValue('begin', node, tensorMap, context) as number[];
      const end = getParamValue('end', node, tensorMap, context) as number[];
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
          getParamValue('shrinkAxisMask', node, tensorMap, context) as number;
      const tensor = getParamValue('x', node, tensorMap, context) as tfc.Tensor;
      if (begin.length === 1 && tensor.shape.length > 1) {
        for (let i = 1; i < tensor.shape.length; i++) {
          begin.push(0);
          end.push(tensor.shape[i]);
          strides.push(strides[0]);
        }
      }
      return [tfc.stridedSlice(
          tensor, begin, end, strides, beginMask, endMask, ellipsisMask,
          newAxisMask, shrinkAxisMask)];
    }
    case 'Pack': {
      return tfc.tidy(() => {
        const axis = getParamValue('axis', node, tensorMap, context) as number;
        const tensors =
            getParamValue('tensors', node, tensorMap, context) as tfc.Tensor[];
        // Reshape the tensors to the first tensor's shape if they don't match.
        const shape = tensors[0].shape;
        const squeezedShape = tensors[0].squeeze().shape;
        const mapped = tensors.map(tensor => {
          const sameShape = tfc.util.arraysEqual(tensor.shape, shape);
          if (!sameShape &&
              !tfc.util.arraysEqual(tensor.squeeze().shape, squeezedShape)) {
            throw new Error('the input tensors shape does not match');
          }
          return sameShape ? tensor : tensor.reshape(shape);
        });
        return [tfc.stack(mapped, axis)];
      });
    }
    case 'Unpack': {
      return tfc.tidy(() => {
        const axis = getParamValue('axis', node, tensorMap, context) as number;
        const tensor =
            getParamValue('tensor', node, tensorMap, context) as tfc.Tensor;
        return tfc.unstack(tensor, axis);
      });
    }
    case 'Tile': {
      const reps = getParamValue('reps', node, tensorMap, context) as number[];
      return [tfc.tile(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, reps)];
    }
    case 'Split':
    case 'SplitV': {
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      const numOrSizeSplits =
          getParamValue('numOrSizeSplits', node, tensorMap, context) as number |
          number[];
      return tfc.split(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          numOrSizeSplits, axis);
    }
    case 'ScatterNd': {
      const indices =
          getParamValue('indices', node, tensorMap, context) as tfc.Tensor;
      const values =
          getParamValue('values', node, tensorMap, context) as tfc.Tensor;
      const shape =
          getParamValue('shape', node, tensorMap, context) as number[];
      return [tfc.scatterND(indices, values, shape)];
    }
    case 'GatherNd': {
      const x = getParamValue('x', node, tensorMap, context) as tfc.Tensor;
      const indices =
          getParamValue('indices', node, tensorMap, context) as tfc.Tensor;
      return [tfc.gatherND(x, indices)];
    }
    case 'SparseToDense': {
      const indices =
          getParamValue('sparseIndices', node, tensorMap, context) as
          tfc.Tensor;
      const shape =
          getParamValue('outputShape', node, tensorMap, context) as number[];
      const sparseValues =
          getParamValue('sparseValues', node, tensorMap, context) as tfc.Tensor;
      const defaultValue =
          getParamValue('defaultValue', node, tensorMap, context) as tfc.Scalar;
      return [tfc.sparseToDense(
          indices, sparseValues, shape,
          sparseValues.dtype === defaultValue.dtype ?
              defaultValue :
              defaultValue.asType(sparseValues.dtype))];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'slice_join';
