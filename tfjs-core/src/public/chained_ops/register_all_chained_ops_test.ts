/**
 * @license
 * Copyright 2018 Google LLC All Rights Reserved.
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
  'add',
  'all',
  'any',
  'atan2',
  'avgPool',
  'batchNorm',
  'batchToSpaceND',
  'broadcastTo',
  'concat',
  'conv1d',
  'conv2d',
  'cumsum',
  'conv2dTranspose',
  'depthToSpace',
  'depthwiseConv2d',
  'depthwiseConv2D',
  'dilation2d',
  'div',
  'divNoNan',
  'dot',
  'elu',
  'equal',
  'floorDiv',
  'greater',
  'greaterEqual',
  'leakyRelu',
  'less',
  'lessEqual',
  'localResponseNormalization',
  'logicalAnd',
  'logicalNot',
  'logicalOr',
  'logicalXor',
  'logSumExp',
  'matMul',
  'max',
  'maximum',
  'maxPool',
  'minimum',
  'mod',
  'mul',
  'notEqual',
  'oneHot',
  'pad',
  'pool',
  'pow',
  'prelu',
  'prod',
  'relu',
  'resizeBilinear',
  'resizeNearestNeighbor',
  'relu6',
  'reverse',
  'selu',
  'separableConv2d',
  'spaceToBatchND',
  'split',
  'square',
  'sub',
  'tile',
  'transpose',
  'where'
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
