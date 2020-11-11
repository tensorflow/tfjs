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

import {_fusedMatMulConfig} from './kernels/_FusedMatMul';
import {absConfig} from './kernels/Abs';
import {acosConfig} from './kernels/Acos';
import {acoshConfig} from './kernels/Acosh';
import {addConfig} from './kernels/Add';
import {addNConfig} from './kernels/AddN';
import {asinConfig} from './kernels/Asin';
import {asinhConfig} from './kernels/Asinh';
import {atanConfig} from './kernels/Atan';
import {atanhConfig} from './kernels/Atanh';
import {avgPoolConfig} from './kernels/AvgPool';
import {avgPoolGradConfig} from './kernels/AvgPoolGrad';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchNormConfig} from './kernels/BatchNorm';
import {castConfig} from './kernels/Cast';
import {ceilConfig} from './kernels/Ceil';
import {clipConfig} from './kernels/Clip';
import {complexConfig} from './kernels/Complex';
import {complexAbsConfig} from './kernels/ComplexAbs';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {conv2DBackpropFilterConfig} from './kernels/Conv2DBackpropFilter';
import {conv2DBackpropInputConfig} from './kernels/Conv2DBackpropInput';
import {conv3DConfig} from './kernels/Conv3D';
import {conv3DBackpropFilterV2Config} from './kernels/Conv3DBackpropFilterV2';
import {conv3DBackpropInputV2Config} from './kernels/Conv3DBackpropInputV2';
import {cosConfig} from './kernels/Cos';
import {coshConfig} from './kernels/Cosh';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {depthwiseConv2dNativeBackpropFilterConfig} from './kernels/DepthwiseConv2dNativeBackpropFilter';
import {depthwiseConv2dNativeBackpropInputConfig} from './kernels/DepthwiseConv2dNativeBackpropInput';
import {dilation2dConfig} from './kernels/Dilation2D';
import {dilation2dBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';
import {dilation2dBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {eluConfig} from './kernels/Elu';
import {equalConfig} from './kernels/Equal';
import {erfConfig} from './kernels/Erf';
import {expConfig} from './kernels/Exp';
import {expm1Config} from './kernels/Expm1';
import {fftConfig} from './kernels/FFT';
import {fillConfig} from './kernels/Fill';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {floorConfig} from './kernels/Floor';
import {floorDivConfig} from './kernels/FloorDiv';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {ifftConfig} from './kernels/IFFT';
import {imagConfig} from './kernels/Imag';
import {isFiniteConfig} from './kernels/IsFinite';
import {isInfConfig} from './kernels/IsInf';
import {isNaNConfig} from './kernels/IsNaN';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {log1pConfig} from './kernels/Log1p';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {logicalNotConfig} from './kernels/LogicalNot';
import {logicalOrConfig} from './kernels/LogicalOr';
import {maxConfig} from './kernels/Max';
import {maxPoolConfig} from './kernels/MaxPool';
import {maxPoolGradConfig} from './kernels/MaxPoolGrad';
import {maxPoolWithArgmaxConfig} from './kernels/MaxPoolWithArgmax';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {negConfig} from './kernels/Neg';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {oneHotConfig} from './kernels/OneHot';
import {padV2Config} from './kernels/PadV2';
import {powConfig} from './kernels/Pow';
import {preluConfig} from './kernels/Prelu';
import {prodConfig} from './kernels/Prod';
import {realConfig} from './kernels/Real';
import {realDivConfig} from './kernels/RealDiv';
import {reciprocalConfig} from './kernels/Reciprocal';
import {reluConfig} from './kernels/Relu';
import {relu6Config} from './kernels/Relu6';
import {reshapeConfig} from './kernels/Reshape';
import {reverseConfig} from './kernels/Reverse';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {roundConfig} from './kernels/Round';
import {rsqrtConfig} from './kernels/Rsqrt';
import {seluConfig} from './kernels/Selu';
import {sigmoidConfig} from './kernels/Sigmoid';
import {signConfig} from './kernels/Sign';
import {sinConfig} from './kernels/Sin';
import {sinhConfig} from './kernels/Sinh';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {softplusConfig} from './kernels/Softplus';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {sqrtConfig} from './kernels/Sqrt';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {stepConfig} from './kernels/Step';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanConfig} from './kernels/Tan';
import {tanhConfig} from './kernels/Tanh';
import {transposeConfig} from './kernels/Transpose';
import {uniqueConfig} from './kernels/Unique';
import {unpackConfig} from './kernels/Unpack';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  _fusedMatMulConfig,
  absConfig,
  acosConfig,
  acoshConfig,
  addConfig,
  addNConfig,
  asinConfig,
  asinhConfig,
  atanConfig,
  atanhConfig,
  avgPoolConfig,
  avgPoolGradConfig,
  batchMatMulConfig,
  batchNormConfig,
  castConfig,
  ceilConfig,
  clipConfig,
  complexConfig,
  complexAbsConfig,
  concatConfig,
  conv2DBackpropFilterConfig,
  conv2DBackpropInputConfig,
  conv2DConfig,
  conv3DBackpropFilterV2Config,
  conv3DBackpropInputV2Config,
  conv3DConfig,
  cosConfig,
  coshConfig,
  depthwiseConv2dNativeConfig,
  depthwiseConv2dNativeBackpropFilterConfig,
  depthwiseConv2dNativeBackpropInputConfig,
  dilation2dConfig,
  dilation2dBackpropInputConfig,
  dilation2dBackpropFilterConfig,
  realDivConfig,
  eluConfig,
  equalConfig,
  erfConfig,
  expConfig,
  expm1Config,
  fftConfig,
  fillConfig,
  flipLeftRightConfig,
  floorConfig,
  floorDivConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  ifftConfig,
  imagConfig,
  isFiniteConfig,
  isInfConfig,
  isNaNConfig,
  lessConfig,
  lessEqualConfig,
  logConfig,
  log1pConfig,
  logicalAndConfig,
  logicalNotConfig,
  logicalOrConfig,
  maxPoolConfig,
  maxPoolGradConfig,
  maxPoolWithArgmaxConfig,
  maxConfig,
  mirrorPadConfig,
  multiplyConfig,
  negConfig,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  oneHotConfig,
  padV2Config,
  powConfig,
  preluConfig,
  prodConfig,
  realConfig,
  reciprocalConfig,
  reluConfig,
  relu6Config,
  reshapeConfig,
  reverseConfig,
  rotateWithOffsetConfig,
  roundConfig,
  rsqrtConfig,
  seluConfig,
  sigmoidConfig,
  signConfig,
  sinConfig,
  sinhConfig,
  sliceConfig,
  softmaxConfig,
  softplusConfig,
  spaceToBatchNDConfig,
  sqrtConfig,
  squareConfig,
  squaredDifferenceConfig,
  stepConfig,
  subConfig,
  sumConfig,
  tanConfig,
  tanhConfig,
  transposeConfig,
  uniqueConfig,
  unpackConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
