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
import {acosConfig} from './kernels/Acos';
import {acoshConfig} from './kernels/Acosh';
import {addConfig} from './kernels/Add';
import {asinConfig} from './kernels/Asin';
import {asinhConfig} from './kernels/Asinh';
import {atanConfig} from './kernels/Atan';
import {atan2Config} from './kernels/Atan2';
import {atanhConfig} from './kernels/Atanh';
import {avgPoolConfig} from './kernels/AvgPool';
import {avgPoolGradConfig} from './kernels/AvgPoolGrad';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchNormConfig} from './kernels/BatchNorm';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {castConfig} from './kernels/Cast';
import {complexConfig} from './kernels/Complex';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {cosConfig} from './kernels/Cos';
import {coshConfig} from './kernels/Cosh';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {eluGradConfig} from './kernels/EluGrad';
import {equalConfig} from './kernels/Equal';
import {erfConfig} from './kernels/Erf';
import {fftConfig} from './kernels/FFT';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {floorConfig} from './kernels/Floor';
import {floorDivConfig} from './kernels/FloorDiv';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {ifftConfig} from './kernels/IFFT';
import {imagConfig} from './kernels/Imag';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {logicalOrConfig} from './kernels/LogicalOr';
import {maxConfig} from './kernels/Max';
import {maxPoolConfig} from './kernels/MaxPool';
import {maxPoolGradConfig} from './kernels/MaxPoolGrad';
import {maxPoolWithArgmaxConfig} from './kernels/MaxPoolWithArgmax';
import {meanConfig} from './kernels/Mean';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {modConfig} from './kernels/Mod';
import {multiplyConfig} from './kernels/Multiply';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {oneHotConfig} from './kernels/OneHot';
import {padV2Config} from './kernels/PadV2';
import {powConfig} from './kernels/Pow';
import {preluConfig} from './kernels/Prelu';
import {realConfig} from './kernels/Real';
import {realDivConfig} from './kernels/RealDiv';
import {reshapeConfig} from './kernels/Reshape';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {sigmoidConfig} from './kernels/Sigmoid';
import {signConfig} from './kernels/Sign';
import {sinConfig} from './kernels/Sin';
import {sinhConfig} from './kernels/Sinh';
import {sliceConfig} from './kernels/Slice';
import {softplusConfig} from './kernels/Softplus';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanConfig} from './kernels/Tan';
import {tanhConfig} from './kernels/Tanh';
import {tileConfig} from './kernels/Tile';
import {transposeConfig} from './kernels/Transpose';
import {uniqueConfig} from './kernels/Unique';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  _fusedMatMulConfig,
  acosConfig,
  acoshConfig,
  addConfig,
  asinConfig,
  asinhConfig,
  atan2Config,
  atanConfig,
  atanhConfig,
  avgPoolConfig,
  avgPoolGradConfig,
  batchMatMulConfig,
  batchNormConfig,
  batchToSpaceNDConfig,
  castConfig,
  complexConfig,
  concatConfig,
  conv2DConfig,
  cosConfig,
  coshConfig,
  cropAndResizeConfig,
  eluGradConfig,
  equalConfig,
  erfConfig,
  fftConfig,
  flipLeftRightConfig,
  floorConfig,
  floorDivConfig,
  fromPixelsConfig,
  fusedConv2DConfig,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  ifftConfig,
  imagConfig,
  lessConfig,
  lessEqualConfig,
  logicalAndConfig,
  logicalOrConfig,
  maxConfig,
  maxPoolConfig,
  maxPoolGradConfig,
  maxPoolWithArgmaxConfig,
  meanConfig,
  mirrorPadConfig,
  modConfig,
  multiplyConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  oneHotConfig,
  padV2Config,
  powConfig,
  preluConfig,
  realConfig,
  realDivConfig,
  reshapeConfig,
  rotateWithOffsetConfig,
  sigmoidConfig,
  signConfig,
  sinConfig,
  sinhConfig,
  sliceConfig,
  softplusConfig,
  spaceToBatchNDConfig,
  squareConfig,
  squaredDifferenceConfig,
  subConfig,
  sumConfig,
  tanConfig,
  tanhConfig,
  tileConfig,
  transposeConfig,
  uniqueConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
