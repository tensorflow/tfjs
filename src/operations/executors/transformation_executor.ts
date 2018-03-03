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
import {Node, ValueType} from '../index';
import {getParamValue} from './utils';

/**
 * Executes the op defined by the node object.
 * @param node
 * @param tensorMap contains tensors for executed nodes and weights
 */
export function executeOp(node: Node, tensorMap: TensorMap): dl.Tensor {
  switch (node.op) {
    case 'cast': {
      return dl.cast(
          getParamValue('x', node, tensorMap) as dl.Tensor,
          getParamValue('dtype', node, tensorMap) as 'int32' | 'float32' |
              'bool');
    }
    case 'squeeze': {
      const axis = node.params['axis'].value as number[];
      return dl.squeeze(getParamValue('x', node, tensorMap) as dl.Tensor, axis);
    }

    case 'reshape': {
      return dl.reshape(
          getParamValue('x', node, tensorMap) as dl.Tensor,
          getParamValue('shape', node, tensorMap) as number[]);
    }
    case 'pad': {
      return dl.pad(
          getParamValue('x', node, tensorMap) as dl.Tensor,
          getParamValue('padding', node, tensorMap) as any,
          getParamValue('constantValue', node, tensorMap) as number);
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
}
