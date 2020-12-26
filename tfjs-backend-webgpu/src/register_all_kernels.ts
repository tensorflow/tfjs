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

import {addConfig} from './kernels/Add';
import {argMaxConfig} from './kernels/ArgMax';
import {castConfig} from './kernels/Cast';
import {clipByValueConfig} from './kernels/ClipByValue';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {divConfig} from './kernels/Div';
import {expandDimsConfig} from './kernels/ExpandDims';
import {floorDivConfig} from './kernels/FloorDiv';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fromPixelsAsyncConfig} from './kernels/FromPixelsAsync';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {maxPoolConfig} from './kernels/MaxPool';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {padV2Config} from './kernels/PadV2';
import {reluConfig} from './kernels/Relu';
import {reshapeConfig} from './kernels/Reshape';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {sigmoidConfig} from './kernels/Sigmoid';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {subConfig} from './kernels/Sub';
import {transposeConfig} from './kernels/Transpose';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  addConfig,
  argMaxConfig,
  castConfig,
  clipByValueConfig,
  concatConfig,
  conv2DConfig,
  depthwiseConv2dNativeConfig,
  divConfig,
  expandDimsConfig,
  fromPixelsConfig,
  fromPixelsAsyncConfig,
  floorDivConfig,
  fusedBatchNormConfig,
  fusedConv2DConfig,
  maxPoolConfig,
  mirrorPadConfig,
  multiplyConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV5Config,
  padV2Config,
  reluConfig,
  reshapeConfig,
  resizeBilinearConfig,
  sigmoidConfig,
  squareConfig,
  squaredDifferenceConfig,
  subConfig,
  transposeConfig,
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
