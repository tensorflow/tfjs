/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {fused, Tensor4D} from '@tensorflow/tfjs-core';

import {registerOp} from '../custom_op/register';

registerOp('_FusedConv2D', node => {
  const [extraOp, activationFunc] = (node.attrs['fused_ops'] || []) as string[];
  const isBiasAdd = extraOp === 'biasadd';
  const isPrelu = activationFunc === 'prelu';
  const isBatchNorm = extraOp === 'fusedbatchnorm';

  const numArgs = node.attrs['num_args'] as number;
  if (isBiasAdd) {
    if (isPrelu && numArgs !== 2) {
      throw new Error(
          'Fused Conv2d with BiasAdd and Prelu must have two ' +
          'extra arguments: bias and alpha.');
    }
    if (!isPrelu && numArgs !== 1) {
      throw new Error(
          'Fused Conv2d with BiasAdd must have one extra argument: bias.');
    }
  }
  if (isBatchNorm) {
    throw new Error('Fused Conv2d with FusedBatchNorm is not supported.');
  }
  const stride = node.attrs['strides'] as number[];
  const pad = node.attrs['padding'];
  const dataFormat =
      (node.attrs['data_format'] as string || 'NHWC').toUpperCase();
  const dilations = (node.attrs['dilations'] || [1, 1, 1, 1]) as number[];

  const x = node.inputs[0] as Tensor4D;
  const filter = node.inputs[1] as Tensor4D;
  const bias = node.inputs[2];
  const preluActivationWeights = node.inputs[3];

  return fused.conv2d({
    x,
    filter,
    strides: [stride[1], stride[2]],
    pad: pad as 'valid' | 'same',
    dataFormat: dataFormat as 'NHWC' | 'NCHW',
    dilations: [dilations[1], dilations[2]],
    bias,
    activation: activationFunc as fused.Activation,
    preluActivationWeights
  });
});
