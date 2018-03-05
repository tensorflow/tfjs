/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {OpExecutor} from './types';
import {getParamValue} from './utils';

export let executeOp: OpExecutor =
    (node: Node, tensorMap: TensorMap): dl.Tensor => {
      switch (node.op) {
        case 'add': {
          return dl.add(
              getParamValue('a', node, tensorMap) as dl.Tensor,
              getParamValue('b', node, tensorMap) as dl.Tensor);
        }
        case 'mul':
          return dl.mul(
              getParamValue('a', node, tensorMap) as dl.Tensor,
              getParamValue('b', node, tensorMap) as dl.Tensor);
        case 'div': {
          return dl.div(
              getParamValue('a', node, tensorMap) as dl.Tensor,
              getParamValue('b', node, tensorMap) as dl.Tensor);
        }
        case 'sub': {
          return dl.sub(
              getParamValue('a', node, tensorMap) as dl.Tensor,
              getParamValue('b', node, tensorMap) as dl.Tensor);
        }
        case 'minimum': {
          return dl.minimum(
              getParamValue('a', node, tensorMap) as dl.Tensor,
              getParamValue('b', node, tensorMap) as dl.Tensor);
        }
        case 'maximum': {
          return dl.maximum(
              getParamValue('a', node, tensorMap) as dl.Tensor,
              getParamValue('b', node, tensorMap) as dl.Tensor);
        }
        case 'pow': {
          return dl.pow(
              getParamValue('a', node, tensorMap) as dl.Tensor,
              getParamValue('b', node, tensorMap) as dl.Tensor);
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'arithmetic';
