/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {KernelConfig, registerKernel} from '@tensorflow/tfjs-core';

import {absConfig} from './kernels/Abs';
import {addConfig} from './kernels/Add';
import {divConfig} from './kernels/Div';
import {expConfig} from './kernels/Exp';
import {floorDivConfig} from './kernels/FloorDiv';
import {fromPixelsAsyncConfig} from './kernels/FromPixelsAsync';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {preluConfig} from './kernels/Prelu';
import {reluConfig} from './kernels/Relu';
import {relu6Config} from './kernels/Relu6';
import {sigmoidConfig} from './kernels/Sigmoid';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {subConfig} from './kernels/Sub';
import {tanhConfig} from './kernels/Tanh';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  absConfig,
  addConfig,
  divConfig,
  expConfig,
  fromPixelsAsyncConfig,
  fromPixelsConfig,
  floorDivConfig,
  fusedBatchNormConfig,
  fusedDepthwiseConv2DConfig,
  greaterEqualConfig,
  lessConfig,
  lessEqualConfig,
  logConfig,
  mirrorPadConfig,
  multiplyConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV5Config,
  preluConfig,
  reluConfig,
  relu6Config,
  sigmoidConfig,
  squareConfig,
  squaredDifferenceConfig,
  subConfig,
  tanhConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
