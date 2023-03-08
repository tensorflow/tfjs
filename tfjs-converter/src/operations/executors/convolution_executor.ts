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
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';
import type {OpExecutorBuilder, OpInput} from '../operation_executor';


import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor} from '../types';

import {getPadding} from './utils';

function fusedConvAndDepthWiseParams(
    ctx: ExecutionContext, fusedOps$: OpInput, numArgs$: OpInput,
    strides$: OpInput, pad$: OpInput, explicitPaddings$: OpInput,
    dataFormat$: OpInput, dilations$: OpInput, args$: OpInput,
    leakyreluAlpha$: OpInput) {
  const [extraOp, activationFunc] = ctx.getOpParamValue<string[]>(fusedOps$);

  const isBiasAdd = extraOp === 'biasadd';
  const noBiasAdd = !isBiasAdd;
  const isPrelu = activationFunc === 'prelu';
  const isBatchNorm = extraOp === 'fusedbatchnorm';

  const numArgs = ctx.getOpParamValue<number>(numArgs$);
  if (isBiasAdd) {
    if (isPrelu && numArgs !== 2) {
      throw new Error(
          'FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu ' +
          'must have two extra arguments: bias and alpha.');
    }
    if (!isPrelu && isBiasAdd && numArgs !== 1) {
      throw new Error(
          'FusedConv2d and DepthwiseConv2d with BiasAdd must have ' +
          'one extra argument: bias.');
    }
  }
  if (isBatchNorm) {
    throw new Error(
        'FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported');
  }
  const strides = ctx.getOpParamValue<number[]>(strides$);
  const pad = getPadding(ctx, pad$, explicitPaddings$);
  const dataFormat = ctx.getOpParamValue<string>(dataFormat$).toUpperCase();
  const dilations = ctx.getOpParamValue<number[]>(dilations$);
  let [biasArg, preluArg] = ctx.getOpParamValue<Tensor[]>(args$);
  if (noBiasAdd) {
    preluArg = biasArg;
    biasArg = undefined;
  }
  const leakyreluAlpha = ctx.getOpParamValue<number>(leakyreluAlpha$);

  return {
    strides,
    pad,
    dataFormat,
    dilations,
    biasArg,
    preluArg,
    activationFunc,
    leakyreluAlpha,
  };
}

export function buildOpExecutor(builder: OpExecutorBuilder):
    InternalOpExecutor {
  const node = builder.node;
  const ops = builder.manager.ops;
  switch (node.op) {
    case 'Conv1D': {
      const x$ = builder.param('x');
      const filter$ = builder.param('filter');
      const stride$ = builder.param('stride');
      const pad$ = builder.param('pad');
      const dataFormat$ = builder.param('dataFormat');
      const dilation$ = builder.param('dilation');

      return (ctx: ExecutionContext) => {
        return [ops.conv1d(
            ctx.getOpParamValue(x$), ctx.getOpParamValue(filter$),
            ctx.getOpParamValue(stride$), ctx.getOpParamValue(pad$),
            ctx.getOpParamValue(dataFormat$), ctx.getOpParamValue(dilation$))];
      };
    }
    case 'Conv2D': {
      const x$ = builder.param('x');
      const filter$ = builder.param('filter');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const dataFormat$ = builder.param('dataFormat');
      const dilations$ = builder.param('dilations');

      return (ctx: ExecutionContext) => {
        const strides = ctx.getOpParamValue<number[]>(strides$);
        const dilations = ctx.getOpParamValue<number[]>(dilations$);
        return [ops.conv2d(
            ctx.getOpParamValue(x$), ctx.getOpParamValue(filter$),
            [strides[1], strides[2]], ctx.getOpParamValue(pad$),
            ctx.getOpParamValue(dataFormat$), [dilations[1], dilations[2]])];
      };
    }
    case '_FusedConv2D': {
      const x$ = builder.param('x');
      const filter$ = builder.param('filter');
      const fusedOps$ = builder.param('fusedOps');
      const numArgs$ = builder.param('numArgs');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const explicitPaddings$ = builder.param('explicitPaddings');
      const dataFormat$ = builder.param('dataFormat');
      const dilations$ = builder.param('dilations');
      const args$ = builder.param('args');
      const leakyreluAlpha$ = builder.param('leakyreluAlpha');

      return (ctx: ExecutionContext) => {
        const argv = fusedConvAndDepthWiseParams(
            ctx, fusedOps$, numArgs$, strides$, pad$, explicitPaddings$,
            dataFormat$, dilations$, args$, leakyreluAlpha$);

        return [ops.fused.conv2d({
          x: ctx.getOpParamValue(x$),
          filter: ctx.getOpParamValue(filter$),
          strides: [argv.strides[1], argv.strides[2]],
          pad: argv.pad as 'valid' | 'same',
          dataFormat: argv.dataFormat as 'NHWC' | 'NCHW',
          dilations: [argv.dilations[1], argv.dilations[2]],
          bias: argv.biasArg,
          activation: argv.activationFunc as tfOps.fused.Activation,
          preluActivationWeights: argv.preluArg,
          leakyreluAlpha: argv.leakyreluAlpha,
        })];
      };
    }

    case 'FusedDepthwiseConv2dNative': {
      const x$ = builder.param('x');
      const filter$ = builder.param('filter');
      const fusedOps$ = builder.param('fusedOps');
      const numArgs$ = builder.param('numArgs');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const explicitPaddings$ = builder.param('explicitPaddings');
      const dataFormat$ = builder.param('dataFormat');
      const dilations$ = builder.param('dilations');
      const args$ = builder.param('args');
      const leakyreluAlpha$ = builder.param('leakyreluAlpha');

      return (ctx: ExecutionContext) => {
        const argv = fusedConvAndDepthWiseParams(
            ctx, fusedOps$, numArgs$, strides$, pad$, explicitPaddings$,
            dataFormat$, dilations$, args$, leakyreluAlpha$);

        return [ops.fused.depthwiseConv2d({
          x: ctx.getOpParamValue(x$),
          filter: ctx.getOpParamValue(filter$),
          strides: [argv.strides[1], argv.strides[2]],
          pad: argv.pad as 'valid' | 'same',
          dataFormat: argv.dataFormat as 'NHWC' | 'NCHW',
          dilations: [argv.dilations[1], argv.dilations[2]],
          bias: argv.biasArg,
          activation: argv.activationFunc as tfOps.fused.Activation,
          preluActivationWeights: argv.preluArg,
          leakyreluAlpha: argv.leakyreluAlpha,
        })];
      };
    }
    case 'Conv2DBackpropInput':
    case 'Conv2dTranspose': {
      const shape$ = builder.param('outputShape');
      const x$ = builder.param('x');
      const filter$ = builder.param('filter');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');


      return (ctx: ExecutionContext) => {
        const strides = ctx.getOpParamValue<number[]>(strides$);
        return [ops.conv2dTranspose(
            ctx.getOpParamValue(x$), ctx.getOpParamValue(filter$),
            ctx.getOpParamValue(shape$), [strides[1], strides[2]],
            ctx.getOpParamValue(pad$))];
      };
    }
    case 'DepthwiseConv2dNative':
    case 'DepthwiseConv2d': {
      const input$ = builder.param('input');
      const filter$ = builder.param('filter');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const dilations$ = builder.param('dilations');
      const dataFormat$ = builder.param('dataFormat');

      return (ctx: ExecutionContext) => {
        const strides = ctx.getOpParamValue<number[]>(strides$);
        const dilations = ctx.getOpParamValue<number[]>(dilations$);
        return [ops.depthwiseConv2d(
            ctx.getOpParamValue(input$), ctx.getOpParamValue(filter$),
            [strides[1], strides[2]], ctx.getOpParamValue(pad$),
            ctx.getOpParamValue<'NHWC'|'NCHW'>(dataFormat$),
            [dilations[1], dilations[2]])];
      };
    }
    case 'Conv3D': {
      const x$ = builder.param('x');
      const filter$ = builder.param('filter');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const dilations$ = builder.param('dilations');
      const dataFormat$ = builder.param('dataFormat');

      return (ctx: ExecutionContext) => {
        const strides = ctx.getOpParamValue<number[]>(strides$);
        const dilations = ctx.getOpParamValue<number[]>(dilations$);
        return [ops.conv3d(
            ctx.getOpParamValue(x$), ctx.getOpParamValue(filter$),
            [strides[1], strides[2], strides[3]], ctx.getOpParamValue(pad$),
            ctx.getOpParamValue(dataFormat$),
            [dilations[1], dilations[2], dilations[3]])];
      };
    }
    case 'AvgPool': {
      const x$ = builder.param('x');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const kernelSize$ = builder.param('kernelSize');

      return (ctx: ExecutionContext) => {
        const kernelSize = ctx.getOpParamValue<number[]>(kernelSize$);
        const strides = ctx.getOpParamValue<number[]>(strides$);
        return [ops.avgPool(
            ctx.getOpParamValue(x$), [kernelSize[1], kernelSize[2]],
            [strides[1], strides[2]], ctx.getOpParamValue(pad$))];
      };
    }
    case 'MaxPool': {
      const x$ = builder.param('x');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const kernelSize$ = builder.param('kernelSize');

      return (ctx: ExecutionContext) => {
        const kernelSize = ctx.getOpParamValue<number[]>(kernelSize$);
        const strides = ctx.getOpParamValue<number[]>(strides$);
        return [ops.maxPool(
            ctx.getOpParamValue(x$), [kernelSize[1], kernelSize[2]],
            [strides[1], strides[2]], ctx.getOpParamValue(pad$))];
      };
    }
    case 'MaxPoolWithArgmax': {
      const x$ = builder.param('x');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const kernelSize$ = builder.param('kernelSize');
      const includeBatchInIndex$ = builder.param('includeBatchInIndex');


      return (ctx: ExecutionContext) => {
        const kernelSize = ctx.getOpParamValue<number[]>(kernelSize$);
        const strides = ctx.getOpParamValue<number[]>(strides$);
        const {result, indexes} = ops.maxPoolWithArgmax(
            ctx.getOpParamValue(x$), [kernelSize[1], kernelSize[2]],
            [strides[1], strides[2]], ctx.getOpParamValue(pad$),
            ctx.getOpParamValue(includeBatchInIndex$));
        return [result, indexes];
      };
    }
    case 'AvgPool3D': {
      const x$ = builder.param('x');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const kernelSize$ = builder.param('kernelSize');

      return (ctx: ExecutionContext) => {
        const kernelSize = ctx.getOpParamValue<number[]>(kernelSize$);
        const strides = ctx.getOpParamValue<number[]>(strides$);
        return [ops.avgPool3d(
            ctx.getOpParamValue(x$),
            [kernelSize[1], kernelSize[2], kernelSize[3]],
            [strides[1], strides[2], strides[3]], ctx.getOpParamValue(pad$))];
      };
    }

    case 'MaxPool3D': {
      const x$ = builder.param('x');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const kernelSize$ = builder.param('kernelSize');

      return (ctx: ExecutionContext) => {
        const kernelSize = ctx.getOpParamValue<number[]>(kernelSize$);
        const strides = ctx.getOpParamValue<number[]>(strides$);
        return [ops.maxPool3d(
            ctx.getOpParamValue(x$),
            [kernelSize[1], kernelSize[2], kernelSize[3]],
            [strides[1], strides[2], strides[3]], ctx.getOpParamValue(pad$))];
      };
    }

    case 'Dilation2D': {
      const x$ = builder.param('x');
      const filter$ = builder.param('filter');
      const strides$ = builder.param('strides');
      const pad$ = builder.param('pad');
      const dilations$ = builder.param('dilations');

      return (ctx: ExecutionContext) => {
        const strides = ctx.getOpParamValue<number[]>(strides$);
        const dilations = ctx.getOpParamValue<number[]>(dilations$);

        // strides: [1, stride_height, stride_width, 1].
        const strideHeight = strides[1];
        const strideWidth = strides[2];

        // dilations: [1, dilation_height, dilation_width, 1].
        const dilationHeight = dilations[1];
        const dilationWidth = dilations[2];

        return [ops.dilation2d(
            ctx.getOpParamValue(x$), ctx.getOpParamValue(filter$),
            [strideHeight, strideWidth], ctx.getOpParamValue(pad$),
            [dilationHeight, dilationWidth],
            /*dataFormat=*/'NHWC')];
      };
    }

    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
}

export const CATEGORY = 'convolution';
