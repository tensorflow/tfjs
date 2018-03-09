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

import {NamedTensorMap} from '../../data/index';
import {Node} from '../index';

import {OpExecutor} from './types';
import {getParamValue} from './utils';

export let executeOp: OpExecutor = (node: Node,
                                    tensorMap: NamedTensorMap): dl.Tensor => {
  switch (node.op) {
    case 'conv1d': {
      const stride = getParamValue('stride', node, tensorMap) as number;
      const pad = getParamValue('pad', node, tensorMap);
      return dl.conv1d(
          getParamValue('x', node, tensorMap) as dl.Tensor3D,
          getParamValue('filter', node, tensorMap) as dl.Tensor3D, stride,
          pad as 'valid' | 'same');
    }
    case 'conv2d': {
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      return dl.conv2d(
          getParamValue('x', node, tensorMap) as dl.Tensor3D | dl.Tensor4D,
          getParamValue('filter', node, tensorMap) as dl.Tensor4D,
          [stride[1], stride[2]], pad as 'valid' | 'same');
    }
    case 'conv2dTranspose': {
      const shape =
          getParamValue(
              'outputShape', node, tensorMap) as [number, number, number] |
          [number, number, number, number];
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      return dl.conv2dTranspose(
          getParamValue('x', node, tensorMap) as dl.Tensor3D | dl.Tensor4D,
          getParamValue('filter', node, tensorMap) as dl.Tensor4D, shape,
          [stride[1], stride[2]], pad as 'valid' | 'same');
    }
    case 'depthwiseConv2d': {
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      const rates = getParamValue('rates', node, tensorMap) as number[];
      return dl.depthwiseConv2d(
          getParamValue('input', node, tensorMap) as dl.Tensor3D | dl.Tensor4D,
          getParamValue('filter', node, tensorMap) as dl.Tensor4D,
          [stride[1], stride[2]], pad as 'valid' | 'same',
          [rates[0], rates[1]]);
    }

    case 'avgPool': {
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      const kernelSize =
          getParamValue('kernelSize', node, tensorMap) as number[];

      return dl.avgPool(
          getParamValue('x', node, tensorMap) as dl.Tensor3D | dl.Tensor4D,
          [kernelSize[1], kernelSize[2]], [stride[1], stride[2]],
          pad as 'valid' | 'same');
    }

    case 'maxPool': {
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      const kernelSize =
          getParamValue('kernelSize', node, tensorMap) as number[];

      return dl.maxPool(
          getParamValue('x', node, tensorMap) as dl.Tensor3D | dl.Tensor4D,
          [kernelSize[1], kernelSize[2]], [stride[1], stride[2]],
          pad as 'valid' | 'same');
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'convolution';
