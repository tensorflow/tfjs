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

import {_fusedMatMulConfig} from './kernels/_FusedMatMul';
import {addConfig} from './kernels/Add';
import {atan2Config} from './kernels/Atan2';
import {avgPoolConfig} from './kernels/AvgPool';
import {avgPoolGradConfig} from './kernels/AvgPoolGrad';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchNormConfig} from './kernels/BatchNorm';
import {castConfig} from './kernels/Cast';
import {complexConfig} from './kernels/Complex';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {cosConfig} from './kernels/Cos';
import {fftConfig} from './kernels/FFT';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {identityConfig} from './kernels/Identity';
import {ifftConfig} from './kernels/IFFT';
import {imagConfig} from './kernels/Imag';
import {maxConfig} from './kernels/Max';
import {maxPoolConfig} from './kernels/MaxPool';
import {maxPoolGradConfig} from './kernels/MaxPoolGrad';
import {maxPoolWithArgmaxConfig} from './kernels/MaxPoolWithArgmax';
import {meanConfig} from './kernels/Mean';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {realConfig} from './kernels/Real';
import {realDivConfig} from './kernels/RealDiv';
import {reshapeConfig} from './kernels/Reshape';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {signConfig} from './kernels/Sign';
import {sinConfig} from './kernels/Sin';
import {sliceConfig} from './kernels/Slice';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanConfig} from './kernels/Tan';
import {tileConfig} from './kernels/Tile';
import {transposeConfig} from './kernels/Transpose';
import {uniqueConfig} from './kernels/Unique';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  _fusedMatMulConfig,
  addConfig,
  atan2Config,
  avgPoolConfig,
  avgPoolGradConfig,
  batchMatMulConfig,
  batchNormConfig,
  castConfig,
  complexConfig,
  concatConfig,
  conv2DConfig,
  cosConfig,
  fftConfig,
  flipLeftRightConfig,
  fromPixelsConfig,
  fusedConv2DConfig,
  identityConfig,
  ifftConfig,
  imagConfig,
  maxConfig,
  maxPoolConfig,
  maxPoolGradConfig,
  maxPoolWithArgmaxConfig,
  meanConfig,
  mirrorPadConfig,
  multiplyConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  realConfig,
  realDivConfig,
  reshapeConfig,
  rotateWithOffsetConfig,
  signConfig,
  sinConfig,
  sliceConfig,
  squareConfig,
  subConfig,
  sumConfig,
  squaredDifferenceConfig,
  tanConfig,
  tileConfig,
  transposeConfig,
  uniqueConfig,
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
