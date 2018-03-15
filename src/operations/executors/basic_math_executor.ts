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

import {NamedTensorsMap} from '../../data/index';
import {Node} from '../index';

import {OpExecutor} from './types';
import {getParamValue, getTensor} from './utils';

export let executeOp: OpExecutor =
    (node: Node, tensorMap: NamedTensorsMap): dl.Tensor[] => {
      switch (node.op) {
        case 'abs':
          return [dl.abs(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'acos':
          return [dl.acos(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'asin':
          return [dl.asin(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'atan':
          return [dl.atan(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'ceil':
          return [dl.ceil(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'cos':
          return [dl.cos(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'cosh':
          return [dl.cosh(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'elu':
          return [dl.elu(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'exp':
          return [dl.exp(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'floor':
          return [dl.floor(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'log':
          return [dl.log(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'relu':
          return [dl.relu(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'selu':
          return [dl.selu(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'sigmoid':
          return [dl.sigmoid(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'sin':
          return [dl.sin(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'sinh': {
          return [dl.sinh(getParamValue('x', node, tensorMap) as dl.Tensor)];
        }
        case 'sqrt': {
          return [dl.sqrt(getParamValue('x', node, tensorMap) as dl.Tensor)];
        }
        case 'square': {
          return [dl.square(getParamValue('x', node, tensorMap) as dl.Tensor)];
        }
        case 'tanh': {
          return [dl.tanh(getParamValue('x', node, tensorMap) as dl.Tensor)];
        }
        case 'tan':
          return [dl.tan(getParamValue('x', node, tensorMap) as dl.Tensor)];
        case 'clipByValue':
          return [dl.clipByValue(
              getParamValue('x', node, tensorMap) as dl.Tensor,
              getParamValue('clipValueMin', node, tensorMap) as number,
              getParamValue('clipValueMax', node, tensorMap) as number)];
        case 'rsqrt':
          return [dl.div(
              dl.scalar(1.0, 'float32'),
              dl.sqrt(getTensor(node.inputNames[0], tensorMap)))];

        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'basic_math';
