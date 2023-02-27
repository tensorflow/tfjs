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
import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line:no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

// import {NamedTensorsMap} from '../data/types';
import {ExecutionContext} from '../executor/execution_context';
// import {ResourceManager} from '../executor/resource_manager';

// import {NodeValueImpl} from './custom_op/node_value_impl';
// import {getRegisteredOp} from './custom_op/register';
import * as arithmetic from './executors/arithmetic_executor';
import * as basicMath from './executors/basic_math_executor';
import * as convolution from './executors/convolution_executor';
import * as graph from './executors/graph_executor';
import * as reduction from './executors/reduction_executor';
import * as transformation from './executors/transformation_executor';
// import * as control from './executors/control_executor';
// import * as creation from './executors/creation_executor';
// import * as dynamic from './executors/dynamic_executor';
// import * as evaluation from './executors/evaluation_executor';
// import * as hashTable from './executors/hash_table_executor';
// import * as image from './executors/image_executor';
// import * as logical from './executors/logical_executor';
// import * as matrices from './executors/matrices_executor';
// import * as normalization from './executors/normalization_executor';
// import * as ragged from './executors/ragged_executor';
// import * as sliceJoin from './executors/slice_join_executor';
// import * as sparse from './executors/sparse_executor';
// import * as spectral from './executors/spectral_executor';
// import * as string from './executors/string_executor';
import {parseNodeName} from './executors/utils';
import {InternalOpExecutor, Node, ValueType} from './types';

/**
 * Executes the op defined by the node object.
 * @param node
 * @param tensorMap contains tensors for executed nodes and weights
 * @param context contains tensors and information for running the current node.
 * @param resourceManager Optional. Contains global resources of the model.
 */
// export function executeOp(
//     node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
//     resourceManager?: ResourceManager, tidy = tfc.tidy): tfc.Tensor[]|
//     Promise<tfc.Tensor[]> {
//   const value =
//       ((node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext)
//       => {
//         switch (node.category) {
//           case 'arithmetic':
//             return tidy(() => arithmetic.executeOp(node, tensorMap,
//             context));
//           case 'basic_math':
//             return tidy(() => basicMath.executeOp(node, tensorMap, context));
//           case 'control':
//             return control.executeOp(node, tensorMap, context);
//           case 'convolution':
//             return tidy(() => convolution.executeOp(node, tensorMap,
//             context));
//           case 'creation':
//             return tidy(() => creation.executeOp(node, tensorMap, context));
//           case 'dynamic':
//             return dynamic.executeOp(node, tensorMap, context);
//           case 'evaluation':
//             return tidy(() => evaluation.executeOp(node, tensorMap,
//             context));
//           case 'image':
//             return tidy(() => image.executeOp(node, tensorMap, context));
//           case 'graph':
//             return tidy(() => graph.executeOp(node, tensorMap, context));
//           case 'logical':
//             return tidy(() => logical.executeOp(node, tensorMap, context));
//           case 'matrices':
//             return tidy(() => matrices.executeOp(node, tensorMap, context));
//           case 'normalization':
//             return tidy(
//                 () => normalization.executeOp(node, tensorMap, context));
//           case 'ragged':
//             return tidy(() => ragged.executeOp(node, tensorMap, context));
//           case 'reduction':
//             return tidy(() => reduction.executeOp(node, tensorMap, context));
//           case 'slice_join':
//             return tidy(() => sliceJoin.executeOp(node, tensorMap, context));
//           case 'sparse':
//             return tidy(() => sparse.executeOp(node, tensorMap, context));
//           case 'spectral':
//             return tidy(() => spectral.executeOp(node, tensorMap, context));
//           case 'string':
//             return tidy(() => string.executeOp(node, tensorMap, context));
//           case 'transformation':
//             return tidy(
//                 () => transformation.executeOp(node, tensorMap, context));
//           case 'hash_table':
//             return hashTable.executeOp(
//                 node, tensorMap, context, resourceManager);
//           case 'custom':
//             const opMapper = getRegisteredOp(node.op);
//             if (opMapper && opMapper.customExecutor) {
//               return opMapper.customExecutor(
//                   new NodeValueImpl(node, tensorMap, context));
//             } else {
//               throw TypeError(`Custom op ${node.op} is not registered.`);
//             }
//           default:
//             throw TypeError(
//                 `Unknown op '${node.op}'. File an issue at ` +
//                 `https://github.com/tensorflow/tfjs/issues so we can add it`
//                 +
//                 `, or register a custom execution with tf.registerOp()`);
//         }
//       })(node, tensorMap, context);
//   if (tfc.util.isPromise(value)) {
//     return value.then((data) => [].concat(data));
//   }
//   return [].concat(value);
// }

export type OpInput = OpInputInfo|OpInputInfo[]|OpInputValue;

export interface OpInputInfo {
  readonly _isInputInfo: true;
  readonly nodeId: number;
  readonly nodeName: string;
  readonly inputName: string;
  readonly index?: number;
  readonly postProcess?: (tensor: tfc.Tensor) => ValueType;
}

export interface OpInputValue {
  readonly _isValue: true;
  readonly value?: ValueType;
}

export class OpExecutorManager {
  readonly nodeNameToId = new Map<string, number>();
  public ops: typeof tfOps = tfOps;
  public tidy = tfc.tidy;

  public buildOpExecutor(node: Node): InternalOpExecutor {
    const exec = this.buildOpExecutorMux(node);
    return (ctx: ExecutionContext) => {
      return this.tidy(() => exec(ctx));
    };
  }

  private buildOpExecutorMux(node: Node): InternalOpExecutor {
    const builder = new OpExecutorBuilder(/*manager=*/this, node);
    switch (node.category) {
      case 'arithmetic':
        return arithmetic.buildOpExecutor(builder);
      case 'basic_math':
        return basicMath.buildOpExecutor(builder);
      case 'convolution':
        return convolution.buildOpExecutor(builder);
      case 'graph':
        return graph.buildOpExecutor(builder);
      case 'reduction':
        return reduction.buildOpExecutor(builder);
      case 'transformation':
        return transformation.buildOpExecutor(builder);
      default:
        throw TypeError(
            `Unknown op '${node.op}'. File an issue at ` +
            `https://github.com/tensorflow/tfjs/issues so we can add
                    it` +
            `, or register a custom execution with tf.registerOp()`);
    }
  }

  public registerNode(nodeName: string): number {
    const cachedId = this.nodeNameToId.get(nodeName);
    if (cachedId == null) {
      return cachedId;
    }
    const id = this.nodeNameToId.size;
    this.nodeNameToId.set(nodeName, id);
    return id;
  }
}

export class OpExecutorBuilder {
  constructor(
      public readonly manager: OpExecutorManager, public readonly node: Node) {}

  /**
   * Registers params for this node and returns the info to get the param.
   * Returns the value directly if the given name is an attr of the node.
   */
  public param(paramName: string): OpInputInfo|OpInputInfo[]|OpInputValue {
    const node = this.node;

    const inputParam = node.inputParams[paramName];
    if (inputParam && inputParam.inputIndexStart !== undefined) {
      const start = inputParam.inputIndexStart;
      const end = inputParam.inputIndexEnd === 0 ?
          undefined :
          (inputParam.inputIndexEnd === undefined ? start + 1 :
                                                    inputParam.inputIndexEnd);
      const shiftedStart = start < 0 ? node.inputNames.length + start : start;
      if (inputParam.type === 'tensor') {
        return this.inputInfo(node.inputNames[shiftedStart]);
      } else if (inputParam.type === 'tensors') {
        const inputs = node.inputNames.slice(start, end);
        return inputs.map((name) => this.inputInfo(name));
      } else {
        return {
          ...this.inputInfo(node.inputNames[shiftedStart]),
          postProcess: (tensor: Tensor) => {
            const data = tensor.dataSync();
            return inputParam.type === 'number' ?
                data[0] :
                tfc.util.toNestedArray(tensor.shape, data);
          },
        };
      }
    }
    const attrParam = node.attrParams[paramName];
    return {_isValue: true, value: attrParam && attrParam.value};
  }

  /**
   * Retrieve the tensor from tensorsMap based on input name.
   * @param name Node input name
   */
  public inputInfo(inputName: string): OpInputInfo {
    const [nodeName, index] = parseNodeName(inputName);
    const nodeId = this.manager.registerNode(nodeName);
    return {
      _isInputInfo: true,
      nodeId,
      nodeName,
      inputName,
      index,
    };
  }
}
