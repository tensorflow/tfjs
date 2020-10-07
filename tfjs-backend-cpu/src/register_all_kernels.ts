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
// We explicitly import the modular kernels so they get registered in the
// global registry when we compile the library. A modular build would replace
// the contents of this file and import only the kernels that are needed.
import {KernelConfig, registerKernel} from '@tensorflow/tfjs-core';

import {absConfig} from './kernels/Abs';
import {acosConfig} from './kernels/Acos';
import {acoshConfig} from './kernels/Acosh';
import {addConfig} from './kernels/Add';
import {asinConfig} from './kernels/Asin';
import {asinhConfig} from './kernels/Asinh';
import {atanConfig} from './kernels/Atan';
import {atanhConfig} from './kernels/Atanh';
import {avgPoolConfig} from './kernels/AvgPool';
import {avgPoolBackpropConfig} from './kernels/AvgPoolBackprop';
import {batchNormConfig} from './kernels/BatchNorm';
import {castConfig} from './kernels/Cast';
import {ceilConfig} from './kernels/Ceil';
import {clipConfig} from './kernels/Clip';
import {complexConfig} from './kernels/Complex';
import {concatConfig} from './kernels/Concat';
import {cosConfig} from './kernels/Cos';
import {coshConfig} from './kernels/Cosh';
import {dilation2dConfig} from './kernels/Dilation2D';
import {dilation2dBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';
import {dilation2dBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {divConfig} from './kernels/Div';
import {eluConfig} from './kernels/Elu';
import {erfConfig} from './kernels/Erf';
import {expConfig} from './kernels/Exp';
import {expm1Config} from './kernels/Expm1';
import {fftConfig} from './kernels/FFT';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {floorConfig} from './kernels/Floor';
import {identityConfig} from './kernels/Identity';
import {ifftConfig} from './kernels/IFFT';
import {imagConfig} from './kernels/Imag';
import {isFiniteConfig} from './kernels/IsFinite';
import {isInfConfig} from './kernels/IsInf';
import {isNaNConfig} from './kernels/IsNaN';
import {logConfig} from './kernels/Log';
import {log1pConfig} from './kernels/Log1p';
import {logicalNotConfig} from './kernels/LogicalNot';
import {maxConfig} from './kernels/Max';
import {maxPoolConfig} from './kernels/MaxPool';
import {maxPoolBackpropConfig} from './kernels/MaxPoolBackprop';
import {maxPoolWithArgmaxConfig} from './kernels/MaxPoolWithArgmax';
import {multiplyConfig} from './kernels/Multiply';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {padV2Config} from './kernels/PadV2';
import {realConfig} from './kernels/Real';
import {reciprocalConfig} from './kernels/Reciprocal';
import {reshapeConfig} from './kernels/Reshape';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {roundConfig} from './kernels/Round';
import {rsqrtConfig} from './kernels/Rsqrt';
import {seluConfig} from './kernels/Selu';
import {sigmoidConfig} from './kernels/Sigmoid';
import {signConfig} from './kernels/Sign';
import {sinConfig} from './kernels/Sin';
import {sinhConfig} from './kernels/Sinh';
import {sliceConfig} from './kernels/Slice';
import {softplusConfig} from './kernels/Softplus';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {sqrtConfig} from './kernels/Sqrt';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {stepConfig} from './kernels/Step';
import {subConfig} from './kernels/Sub';
import {tanConfig} from './kernels/Tan';
import {tanhConfig} from './kernels/Tanh';
import {transposeConfig} from './kernels/Transpose';
import {uniqueConfig} from './kernels/Unique';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  absConfig,
  acosConfig,
  acoshConfig,
  addConfig,
  asinConfig,
  asinhConfig,
  atanConfig,
  atanhConfig,
  avgPoolConfig,
  avgPoolBackpropConfig,
  batchNormConfig,
  castConfig,
  ceilConfig,
  clipConfig,
  complexConfig,
  concatConfig,
  cosConfig,
  coshConfig,
  dilation2dConfig,
  dilation2dBackpropInputConfig,
  dilation2dBackpropFilterConfig,
  divConfig,
  eluConfig,
  erfConfig,
  expConfig,
  expm1Config,
  fftConfig,
  flipLeftRightConfig,
  floorConfig,
  identityConfig,
  ifftConfig,
  imagConfig,
  isFiniteConfig,
  isInfConfig,
  isNaNConfig,
  logConfig,
  log1pConfig,
  logicalNotConfig,
  maxPoolConfig,
  maxPoolBackpropConfig,
  maxPoolWithArgmaxConfig,
  maxConfig,
  multiplyConfig,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  padV2Config,
  realConfig,
  reciprocalConfig,
  reshapeConfig,
  rotateWithOffsetConfig,
  roundConfig,
  rsqrtConfig,
  seluConfig,
  sigmoidConfig,
  signConfig,
  sinConfig,
  sinhConfig,
  sliceConfig,
  softplusConfig,
  spaceToBatchNDConfig,
  sqrtConfig,
  squareConfig,
  squaredDifferenceConfig,
  stepConfig,
  subConfig,
  tanConfig,
  tanhConfig,
  transposeConfig,
  uniqueConfig,
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
