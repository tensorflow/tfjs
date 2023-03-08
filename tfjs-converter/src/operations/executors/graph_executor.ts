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

import {Tensor} from '@tensorflow/tfjs-core';
import {ExecutionContext} from '../../executor/execution_context';

import {cloneTensor} from './utils';

import type {OpExecutorBuilder} from '../operation_executor';
import {InternalOpExecutor} from '../types';

export function buildOpExecutor(builder: OpExecutorBuilder):
    InternalOpExecutor {
  const node = builder.node;
  const ops = builder.manager.ops;

  switch (node.op) {
    case 'Const': {
      const value$ = builder.inputInfo(node.name);
      return (ctx: ExecutionContext) => [ctx.getOpParamValue(value$)];
    }
    case 'PlaceholderWithDefault': {
      const value$ = builder.inputInfo(node.name);
      const default$ = builder.param('default');
      return (ctx: ExecutionContext) =>
                 [ctx.getOpParamValue(value$) || ctx.getOpParamValue(default$)];
    }
    case 'Placeholder': {
      const value$ = builder.inputInfo(node.name);
      return (ctx: ExecutionContext) => [ctx.getOpParamValue(value$)];
    }
    case 'Identity':
    case 'StopGradient':
    case 'FakeQuantWithMinMaxVars': {  // This op is currently ignored.
      const x$ = builder.param('x');
      return (ctx: ExecutionContext) => [cloneTensor(ctx.getOpParamValue(x$))];
    }
    case 'IdentityN': {
      const x$ = builder.param('x');
      return (ctx: ExecutionContext) =>
                 ctx.getOpParamValue<Tensor[]>(x$).map(cloneTensor);
    }
    case 'Snapshot': {
      const x$ = builder.param('x');
      return (ctx: ExecutionContext) => [cloneTensor(ctx.getOpParamValue(x$))];
    }
    case 'Shape': {
      const x$ = builder.param('x');
      return (ctx: ExecutionContext) => [ops.tensor1d(
                 ctx.getOpParamValue<Tensor>(x$).shape, 'int32')];
    }
    case 'ShapeN': {
      const x$ = builder.param('x');
      return (ctx: ExecutionContext) => ctx.getOpParamValue<Tensor[]>(x$).map(
                 (t) => ops.tensor1d(t.shape));
    }
    case 'Size': {
      const x$ = builder.param('x');
      return (ctx: ExecutionContext) => [ops.scalar(
                 ctx.getOpParamValue<Tensor>(x$).size, 'int32')];
    }
    case 'Rank': {
      const x$ = builder.param('x');
      return (ctx: ExecutionContext) => [ops.scalar(
                 ctx.getOpParamValue<Tensor>(x$).rank, 'int32')];
    }
    case 'NoOp':
      return (ctx: ExecutionContext) => [ops.scalar(1)];
    case 'Print': {
      const x$ = builder.param('x');
      const data$ = builder.param('data');
      const message$ = builder.param('message');
      const summarize$ = builder.param('summarize');
      return (ctx: ExecutionContext) => {
        console.warn(
            'The graph has a tf.print() operation,' +
            'usually used for debugging, which slows down performance.');
        console.log(ctx.getOpParamValue(message$));
        const data = ctx.getOpParamValue<Tensor[]>(data$);
        const summarize = ctx.getOpParamValue<number>(summarize$);
        for (let i = 0; i < data.length; i++) {
          console.log(Array.prototype.slice.call(data[i].dataSync())
                          .slice(0, summarize));
        }

        return [ctx.getOpParamValue(x$)];
      };
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'graph';
