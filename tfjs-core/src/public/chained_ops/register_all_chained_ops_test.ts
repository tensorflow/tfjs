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
  'addStrict',
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
  'depthwiseConv2D',
  'dilation2d',
  'div',
  'divNoNan',
  'divStrict',
  'dot',
  'elu',
  'equal',
  'equalStrict',
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
  'greaterEqualStrict',
  'greaterStrict',
  'ifft',
  'irfft',
  'isFinite',
  'isInf',
  'isNaN',
  'leakyRelu',
  'less',
  'lessEqual',
  'lessEqualStrict',
  'lessStrict',
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
  'maximumStrict',
  'maxPool',
  'mean',
  'min',
  'minimum',
  'minimumStrict',
  'mod',
  'modStrict',
  'mul',
  'mulStrict',
  'neg',
  'norm',
  'notEqual',
  'notEqualStrict',
  'oneHot',
  'onesLike',
  'pad',
  'pool',
  'pow',
  'powStrict',
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
  'squaredDifferenceStrict',
  'squeeze',
  'stack',
  'step',
  'stridedSlice',
  'sub',
  'subStrict',
  'sum',
  'tan',
  'tanh',
  'tile',
  'toBool',
  'toFloat',
  'toInt',
  'topk',
  'transpose',
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
