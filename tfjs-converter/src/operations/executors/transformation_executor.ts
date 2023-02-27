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

import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor} from '../types';
import type {OpExecutorBuilder} from '../operation_executor';

export function buildOpExecutor(builder: OpExecutorBuilder):
    InternalOpExecutor {
  const node = builder.node;
  const ops = builder.manager.ops;
  switch (node.op) {
    case 'Cast': {
      const x$ = builder.param('x');
      const dtype$ = builder.param('dtype');
      return (ctx: ExecutionContext) => [ops.cast(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(dtype$))];
    }
    case 'ExpandDims': {
      const x$ = builder.param('x');
      const axis$ = builder.param('axis');
      return (ctx: ExecutionContext) => [ops.expandDims(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(axis$))];
    }
    case 'Squeeze': {
      const x$ = builder.param('x');
      const axis$ = builder.param('axis');
      return (ctx: ExecutionContext) => [ops.squeeze(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(axis$))];
    }

    case 'Reshape': {
      const x$ = builder.param('x');
      const shape$ = builder.param('shape');
      return (ctx: ExecutionContext) => [ops.reshape(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(shape$))];
    }
    case 'MirrorPad': {
      const x$ = builder.param('x');
      const padding$ = builder.param('padding');
      const mode$ = builder.param('mode');
      return (ctx: ExecutionContext) => [ops.mirrorPad(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(padding$),
                 ctx.getOpParamValue(mode$))];
    }
    case 'PadV2':
    case 'Pad': {
      const x$ = builder.param('x');
      const padding$ = builder.param('padding');
      const constantValue$ = builder.param('constantValue');
      return (ctx: ExecutionContext) => [ops.pad(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(padding$),
                 ctx.getOpParamValue(constantValue$))];
    }
    case 'SpaceToBatchND': {
      const x$ = builder.param('x');
      const blockShape$ = builder.param('blockShape');
      const paddings$ = builder.param('paddings');
      return (ctx: ExecutionContext) => [ops.spaceToBatchND(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(blockShape$),
                 ctx.getOpParamValue(paddings$))];
    }
    case 'BatchToSpaceND': {
      const x$ = builder.param('x');
      const blockShape$ = builder.param('blockShape');
      const crops$ = builder.param('crops');
      return (ctx: ExecutionContext) => [ops.batchToSpaceND(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(blockShape$),
                 ctx.getOpParamValue(crops$))];
    }
    case 'DepthToSpace': {
      const x$ = builder.param('x');
      const blockShape$ = builder.param('blockShape');
      const dataFormat$ = builder.param('dataFormat');
      return (ctx: ExecutionContext) => [ops.depthToSpace(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(blockShape$),
                 ctx.getOpParamValue(dataFormat$))];
    }
    case 'BroadcastTo': {
      const x$ = builder.param('x');
      const shape$ = builder.param('shape');
      return (ctx: ExecutionContext) => [ops.broadcastTo(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(shape$))];
    }
    case 'BroadcastArgs': {
      const s0$ = builder.param('s0');
      const s1$ = builder.param('s1');
      return (ctx: ExecutionContext) => [ops.broadcastTo(
                 ctx.getOpParamValue(s0$), ctx.getOpParamValue(s1$))];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'transformation';
