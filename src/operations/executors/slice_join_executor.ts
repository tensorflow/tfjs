/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as dl from 'deeplearn';
import {TensorMap} from '../../data/types';
import {Node} from '../index';
import {getParamValue} from './utils';

/**
 * Executes the op defined by the node object.
 * @param node
 * @param tensorMap contains tensors for executed nodes and weights
 */
export function executeOp(node: Node, tensorMap: TensorMap): dl.Tensor {
  switch (node.op) {
    case 'concat': {
      const axis = getParamValue('axis', node, tensorMap) as number;
      const inputs = getParamValue('tensors', node, tensorMap) as dl.Tensor[];
      return dl.concat(inputs, axis);
    }
    case 'gather': {
      const axis = getParamValue('axis', node, tensorMap) as number;
      const input = getParamValue('x', node, tensorMap) as dl.Tensor;
      const indices = getParamValue('indices', node, tensorMap) as dl.Tensor1D;
      return dl.gather(input, indices, axis);
    }
    case 'reverse': {
      const axis = getParamValue('axis', node, tensorMap) as number;
      const input = getParamValue('x', node, tensorMap) as dl.Tensor;
      return dl.reverse(input, axis);
    }
    case 'slice': {
      // tslint:disable-next-line:no-any
      const begin = getParamValue('begin', node, tensorMap) as any;
      // tslint:disable-next-line:no-any
      const size = getParamValue('size', node, tensorMap) as any;
      return dl.slice(
          getParamValue('x', node, tensorMap) as dl.Tensor, begin, size);
    }
    case 'stack': {
      const axis = getParamValue('axis', node, tensorMap) as number;
      return dl.stack(
          getParamValue('tensors', node, tensorMap) as dl.Tensor[], axis);
    }
    case 'tile': {
      const reps = getParamValue('reps', node, tensorMap) as number[];
      return dl.tile(getParamValue('x', node, tensorMap) as dl.Tensor, reps);
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
}

export const CATEGORY = 'slice_join';
