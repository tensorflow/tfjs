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
import {getParamValue} from './utils';

export let executeOp: OpExecutor =
    (node: Node, tensorMap: NamedTensorsMap): dl.Tensor[] => {
      switch (node.op) {
        case 'batchNormalization': {
          return [dl.batchNormalization(
              getParamValue('x', node, tensorMap) as dl.Tensor,
              getParamValue('mean', node, tensorMap) as dl.Tensor,
              getParamValue('variance', node, tensorMap) as dl.Tensor,
              getParamValue('epislon', node, tensorMap) as number,
              getParamValue('scale', node, tensorMap) as dl.Tensor,
              getParamValue('offset', node, tensorMap) as dl.Tensor)];
        }
        case 'localResponseNormalization': {
          return [dl.localResponseNormalization(
              getParamValue('x', node, tensorMap) as dl.Tensor3D | dl.Tensor4D,
              getParamValue('radius', node, tensorMap) as number,
              getParamValue('bias', node, tensorMap) as number,
              getParamValue('alpha', node, tensorMap) as number,
              getParamValue('beta', node, tensorMap) as number)];
        }
        case 'softmax': {
          return [dl.softmax(getParamValue('x', node, tensorMap) as dl.Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'normalization';
