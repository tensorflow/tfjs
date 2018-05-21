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

import {getParamValue, getTensor} from './utils';

export async function executeOp(
    node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext): Promise<tfc.Tensor[]> {
  switch (node.op) {
    case 'loopCond':
      return [getParamValue('pred', node, tensorMap, context) as tfc.Tensor];
    case 'switch': {
      const pred =
          getParamValue('pred', node, tensorMap, context) as tfc.Tensor;
      const data =
          getParamValue('data', node, tensorMap, context) as tfc.Tensor;
      // Outputs nodes :0 => false, :1 => true
      return (await pred.data())[0] ? [undefined, data] : [data, undefined];
    }
    case 'merge':
      const inputName = node.inputNames.find(
          name => getTensor(name, tensorMap, context) !== undefined);
      return inputName ? [getTensor(inputName, tensorMap, context)] : undefined;

    case 'enter':
      const frameId =
          getParamValue('frameName', node, tensorMap, context) as string;
      const data =
          getParamValue('tensor', node, tensorMap, context) as tfc.Tensor;
      context.enterFrame(frameId);
      return [data];

    case 'exit':
      const tensor =
          getParamValue('tensor', node, tensorMap, context) as tfc.Tensor;
      context.exitFrame();
      return [tensor];

    case 'nextIteration':
      const input =
          getParamValue('tensor', node, tensorMap, context) as tfc.Tensor;
      context.nextIteration();
      return [input];
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
}

export const CATEGORY = 'control';
