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
import {absConfig} from './kernels/Abs';
import {addConfig} from './kernels/Add';
import {addNConfig} from './kernels/AddN';
import {argMaxConfig} from './kernels/ArgMax';
import {argMinConfig} from './kernels/ArgMin';
import {avgPoolConfig} from './kernels/AvgPool';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {castConfig} from './kernels/Cast';
import {ceilConfig} from './kernels/Ceil';
import {clipByValueConfig} from './kernels/ClipByValue';
import {complexConfig} from './kernels/Complex';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {eluConfig} from './kernels/Elu';
import {expConfig} from './kernels/Exp';
import {expandDimsConfig} from './kernels/ExpandDims';
import {expm1Config} from './kernels/Expm1';
import {fillConfig} from './kernels/Fill';
import {floorConfig} from './kernels/Floor';
import {floorDivConfig} from './kernels/FloorDiv';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {imagConfig} from './kernels/Imag';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {maxPoolConfig} from './kernels/MaxPool';
import {minConfig} from './kernels/Min';
import {minimumConfig} from './kernels/Minimum';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {negConfig} from './kernels/Neg';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {onesLikeConfig} from './kernels/OnesLike';
import {packConfig} from './kernels/Pack';
import {padV2Config} from './kernels/PadV2';
import {preluConfig} from './kernels/Prelu';
import {realConfig} from './kernels/Real';
import {realDivConfig} from './kernels/RealDiv';
import {reluConfig} from './kernels/Relu';
import {relu6Config} from './kernels/Relu6';
import {reshapeConfig} from './kernels/Reshape';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {rsqrtConfig} from './kernels/Rsqrt';
import {selectConfig} from './kernels/Select';
import {sigmoidConfig} from './kernels/Sigmoid';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {stridedSliceConfig} from './kernels/StridedSlice';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanhConfig} from './kernels/Tanh';
import {transposeConfig} from './kernels/Transpose';
import {unpackConfig} from './kernels/Unpack';
import {zerosLikeConfig} from './kernels/ZerosLike';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  _fusedMatMulConfig,
  absConfig,
  addConfig,
  addNConfig,
  argMaxConfig,
  argMinConfig,
  avgPoolConfig,
  batchMatMulConfig,
  batchToSpaceNDConfig,
  castConfig,
  ceilConfig,
  clipByValueConfig,
  complexConfig,
  concatConfig,
  conv2DConfig,
  cropAndResizeConfig,
  depthwiseConv2dNativeConfig,
  eluConfig,
  expandDimsConfig,
  expConfig,
  expm1Config,
  fillConfig,
  fromPixelsConfig,
  floorConfig,
  floorDivConfig,
  fusedBatchNormConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  imagConfig,
  lessConfig,
  lessEqualConfig,
  logConfig,
  maxConfig,
  maximumConfig,
  maxPoolConfig,
  minConfig,
  minimumConfig,
  mirrorPadConfig,
  multiplyConfig,
  negConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  onesLikeConfig,
  packConfig,
  padV2Config,
  preluConfig,
  realConfig,
  realDivConfig,
  reluConfig,
  relu6Config,
  reshapeConfig,
  resizeBilinearConfig,
  rsqrtConfig,
  selectConfig,
  sigmoidConfig,
  sliceConfig,
  stridedSliceConfig,
  softmaxConfig,
  spaceToBatchNDConfig,
  squareConfig,
  squaredDifferenceConfig,
  subConfig,
  sumConfig,
  tanhConfig,
  transposeConfig,
  unpackConfig,
  zerosLikeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
