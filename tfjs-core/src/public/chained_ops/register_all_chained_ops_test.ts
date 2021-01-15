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

import '../../public/chained_ops/register_all_chained_ops';

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';

// Testing for presence of chained op in this file will allow us to more easily
// customize when we want this test to run. Currently it will run be default
// (And karma will always load the chain augmentor files). But this gives us
// flexibility to change in future.

const CHAINED_OPS = [
  'abs',
  'acos',
  'acosh',
  'add',
  'all',
  'any',
  'argMax',
  'argMin',
  'as1D',
  'as2D',
  'as3D',
  'as4D',
  'as5D',
  'asin',
  'asinh',
  'asScalar',
  'asType',
  'atan',
  'atan2',
  'atanh',
  'avgPool',
  'batchNorm',
  'batchToSpaceND',
  'broadcastTo',
  'cast',
  'ceil',
  'clipByValue',
  'concat',
  'conv1d',
  'conv2d',
  'conv2dTranspose',
  'cos',
  'cosh',
  'cumsum',
  'depthToSpace',
  'depthwiseConv2d',
  'dilation2d',
  'div',
  'divNoNan',
  'dot',
  'elu',
  'equal',
  'erf',
  'exp',
  'expandDims',
  'expm1',
  'fft',
  'flatten',
  'floor',
  'floorDiv',
  'gather',
  'greater',
  'greaterEqual',
  'ifft',
  'irfft',
  'isFinite',
  'isInf',
  'isNaN',
  'leakyRelu',
  'less',
  'lessEqual',
  'localResponseNormalization',
  'log',
  'log1p',
  'logicalAnd',
  'logicalNot',
  'logicalOr',
  'logicalXor',
  'logSigmoid',
  'logSoftmax',
  'logSumExp',
  'matMul',
  'max',
  'maximum',
  'maxPool',
  'mean',
  'min',
  'minimum',
  'mirrorPad',
  'mod',
  'mul',
  'neg',
  'norm',
  'notEqual',
  'oneHot',
  'onesLike',
  'pad',
  'pool',
  'pow',
  'prelu',
  'prod',
  'reciprocal',
  'relu',
  'relu6',
  'reshape',
  'reshapeAs',
  'resizeBilinear',
  'resizeNearestNeighbor',
  'reverse',
  'rfft',
  'round',
  'rsqrt',
  'selu',
  'separableConv2d',
  'sigmoid',
  'sign',
  'sin',
  'sinh',
  'slice',
  'softmax',
  'softplus',
  'spaceToBatchND',
  'split',
  'sqrt',
  'square',
  'square',
  'squeeze',
  'stack',
  'step',
  'stridedSlice',
  'sub',
  'sum',
  'tan',
  'tanh',
  'tile',
  'toBool',
  'toFloat',
  'toInt',
  'topk',
  'transpose',
  'unique',
  'unsortedSegmentSum',
  'unstack',
  'where',
  'zerosLike'
];

describeWithFlags('chained ops', ALL_ENVS, () => {
  it('all chained ops should exist on tensor ', async () => {
    const tensor = tf.tensor([1, 2, 3]);
    for (const opName of CHAINED_OPS) {
      //@ts-ignore
      expect(typeof tensor[opName])
          .toBe('function', `${opName} chained op not found`);
    }
  });
});
