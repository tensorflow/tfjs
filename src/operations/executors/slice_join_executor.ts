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
import {Node} from '../types';

import {OpExecutor} from './types';
import {getParamValue} from './utils';

export let executeOp: OpExecutor = (node: Node, tensorMap: NamedTensorsMap,
                                    context: ExecutionContext):
                                       tfc.Tensor[] => {
  switch (node.op) {
    case 'concat': {
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      const inputs =
          getParamValue('tensors', node, tensorMap, context) as tfc.Tensor[];
      return [tfc.concat(inputs, axis)];
    }
    case 'gather': {
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      const input = getParamValue('x', node, tensorMap, context) as tfc.Tensor;
      const indices =
          getParamValue('indices', node, tensorMap, context) as tfc.Tensor1D;
      return [tfc.gather(input, indices, axis)];
    }
    case 'reverse': {
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      const input = getParamValue('x', node, tensorMap, context) as tfc.Tensor;
      return [tfc.reverse(input, axis)];
    }
    case 'slice': {
      // tslint:disable-next-line:no-any
      const begin = getParamValue('begin', node, tensorMap, context) as any;
      // tslint:disable-next-line:no-any
      const size = getParamValue('size', node, tensorMap, context) as any;
      return [tfc.slice(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, begin,
          size)];
    }
    case 'stridedSlice': {
      const begin =
          getParamValue('begin', node, tensorMap, context) as number[];
      const end = getParamValue('end', node, tensorMap, context) as number[];
      const strides =
          getParamValue('strides', node, tensorMap, context) as number[];
      const beginMask =
          getParamValue('beginMask', node, tensorMap, context) as number;
      const endMask =
          getParamValue('endMask', node, tensorMap, context) as number;
      return [tfc.stridedSlice(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, begin,
          end, strides, beginMask, endMask)];
    }
    case 'stack': {
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
    case 'unstack': {
      return tfc.tidy(() => {
        const axis = getParamValue('axis', node, tensorMap, context) as number;
        const tensor =
            getParamValue('tensor', node, tensorMap, context) as tfc.Tensor;
        return tfc.unstack(tensor, axis);
      });
    }
    case 'tile': {
      const reps = getParamValue('reps', node, tensorMap, context) as number[];
      return [tfc.tile(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, reps)];
    }
    case 'split': {
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      const numOrSizeSplits =
          getParamValue('numOrSizeSplits', node, tensorMap, context) as number;
      return tfc.split(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          numOrSizeSplits, axis);
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'slice_join';
