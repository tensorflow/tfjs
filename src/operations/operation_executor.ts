/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {TensorMap} from '../data/types';
import {Node, ValueType} from './index';

export function getParamValue(
    paramName: string, node: Node, tensorMap: TensorMap): ValueType {
  const param = node.params[paramName];
  if (param.inputIndex !== undefined) {
    return param.type === 'tensor' ?
        tensorMap[node.inputNames[param.inputIndex]] :
        Array.prototype.slice.call(
            tensorMap[node.inputNames[param.inputIndex]].dataSync());
  }
  return param.value;
}

export function executeOp(node: Node, tensorMap: TensorMap): dl.Tensor {
  switch (node.op) {
    case 'add': {
      return dl.add(
          tensorMap[node.inputNames[0]], tensorMap[node.inputNames[1]]);
    }
    case 'const': {
      return tensorMap[node.name];
    }
    case 'placeholder':
      return tensorMap[node.name];
    case 'placeholderWithDefault':
      return tensorMap[node.name] || tensorMap[node.inputNames[0]];
    case 'floor':
      return dl.floor(tensorMap[node.inputNames[0]]);
    case 'mul':
      return dl.mul(
          tensorMap[node.inputNames[0]], tensorMap[node.inputNames[1]]);
    case 'matMul':
      return dl.matMul(
          tensorMap[node.inputNames[0]] as dl.Tensor2D,
          tensorMap[node.inputNames[1]] as dl.Tensor2D,
          node.params['transposeA'].value as boolean,
          node.params['transposeB'].value as boolean);
    case 'conv2D': {
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      return dl.conv2d(
          tensorMap[node.inputNames[0]] as dl.Tensor3D | dl.Tensor4D,
          tensorMap[node.inputNames[1]] as dl.Tensor4D, [stride[1], stride[2]],
          pad as 'valid' | 'same');
    }

    case 'depthwiseConv2d': {
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      const rate = getParamValue('rate', node, tensorMap) as number[];
      return dl.depthwiseConv2d(
          tensorMap[node.inputNames[0]] as dl.Tensor3D | dl.Tensor4D,
          tensorMap[node.inputNames[1]] as dl.Tensor4D, [stride[1], stride[2]],
          pad as 'valid' | 'same', [rate[0], rate[1]]);
    }

    case 'avgPool': {
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      const rate = getParamValue('rate', node, tensorMap) as number[];
      const kernelSize = getParamValue('ksize', node, tensorMap) as number[];

      return dl.avgPool(
          tensorMap[node.inputNames[0]] as dl.Tensor3D | dl.Tensor4D,
          [kernelSize[1], kernelSize[2]], [stride[1], stride[2]],
          pad as 'valid' | 'same');
    }

    case 'maxPool': {
      const stride = getParamValue('strides', node, tensorMap) as number[];
      const pad = getParamValue('pad', node, tensorMap);
      const rate = getParamValue('rate', node, tensorMap) as number[];
      const kernelSize = getParamValue('ksize', node, tensorMap) as number[];

      return dl.maxPool(
          tensorMap[node.inputNames[0]] as dl.Tensor3D | dl.Tensor4D,
          [kernelSize[1], kernelSize[2]], [stride[1], stride[2]],
          pad as 'valid' | 'same');
    }

    case 'randomUniform': {
      return dl.randomUniform(
          Array.prototype.slice.call(tensorMap[node.inputNames[0]].dataSync()),
          getParamValue('maxVal', node, tensorMap) as number,
          getParamValue('minVal', node, tensorMap) as number, 'float32');
    }

    case 'div': {
      return dl.div(
          tensorMap[node.inputNames[0]], tensorMap[node.inputNames[1]]);
    }

    case 'sigmoid': {
      return dl.sigmoid(tensorMap[node.inputNames[0]]);
    }
    case 'tanh': {
      return dl.tanh(tensorMap[node.inputNames[0]]);
    }
    case 'squeeze': {
      const axis = node.params['axis'].value as number[];
      return dl.squeeze(tensorMap[node.inputNames[0]], axis);
    }

    case 'reshape': {
      return tensorMap[node.inputNames[0]].reshape(
          Array.prototype.slice.call(tensorMap[node.inputNames[1]].dataSync()));
    }

    case 'slice': {
      // tslint:disable-next-line:no-any
      const begin = getParamValue('begin', node, tensorMap) as any;
      // tslint:disable-next-line:no-any
      const size = getParamValue('size', node, tensorMap) as any;
      return dl.slice(tensorMap[node.inputNames[0]], begin, size);
    }
    case 'fill': {
      const shape = getParamValue('begin', node, tensorMap) as number[];
      const value = getParamValue('begin', node, tensorMap) as number[];
      return dl.fill(shape, value[0]);
    }
    case 'sub': {
      return dl.sub(
          tensorMap[node.inputNames[0]], tensorMap[node.inputNames[1]]);
    }

    case 'exp': {
      return dl.exp(tensorMap[node.inputNames[0]]);
    }
    case 'relu':
      return dl.relu(tensorMap[node.inputNames[0]]);

    case 'relu6':
      return dl.clipByValue(
          tensorMap[node.inputNames[0]],
          getParamValue('min', node, tensorMap) as number,
          getParamValue('max', node, tensorMap) as number);

    case 'minimum': {
      return dl.minimum(
          tensorMap[node.inputNames[0]], tensorMap[node.inputNames[1]]);
    }
    case 'maximum': {
      return dl.maximum(
          tensorMap[node.inputNames[0]], tensorMap[node.inputNames[1]]);
    }
    case 'max': {
      const axis = getParamValue('axis', node, tensorMap) as number[];
      const keepDims = getParamValue('keepDims', node, tensorMap) as boolean;
      return dl.max(tensorMap[node.inputNames[0]], axis, keepDims);
    }
    case 'mean': {
      const axis = getParamValue('axis', node, tensorMap) as number[];
      return dl.mean(tensorMap[node.inputNames[0]], axis);
    }
    case 'fusedBatchNorm': {
      const mean = getParamValue('axis', node, tensorMap) as number[];

      return dl.batchNormalization(
          tensorMap[node.inputNames[0]],
          getParamValue('mean', node, tensorMap) as dl.Tensor,
          getParamValue('variance', node, tensorMap) as dl.Tensor,
          getParamValue('epislon', node, tensorMap) as number,
          getParamValue('scale', node, tensorMap) as dl.Tensor,
          getParamValue('offset', node, tensorMap) as dl.Tensor,
      );
    }
    case 'shape': {
      return dl.Tensor1D.new(tensorMap[node.inputNames[0]].shape, 'int32');
    }
    case 'transpose': {
      return dl.transpose(
          tensorMap[node.inputNames[3]],
          getParamValue('perms', node, tensorMap) as number[]);
    }
    case 'rsqrt':
      return dl.div(
          dl.Scalar.new(1.0, 'float32'),
          dl.sqrt(tensorMap[node.inputNames[0]]));

    case 'softmax':
      return dl.softmax(tensorMap[node.inputNames[0]]);

    case 'identity':
      return tensorMap[node.inputNames[0]];

    case 'concat': {
      const axis = getParamValue('axis', node, tensorMap) as number[];
      const inputs = node.inputNames.slice(0, -1).map(name => tensorMap[name]);
      return dl.concat(inputs, axis[0]);
    }

    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
}
