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
import {addNConfig} from './kernels/AddN';
import {argMaxConfig} from './kernels/ArgMax';
import {argMinConfig} from './kernels/ArgMin';
import {asinConfig} from './kernels/Asin';
import {asinhConfig} from './kernels/Asinh';
import {atanConfig} from './kernels/Atan';
import {atanhConfig} from './kernels/Atanh';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {ceilConfig} from './kernels/Ceil';
import {cosConfig} from './kernels/Cos';
import {coshConfig} from './kernels/Cosh';
import {diagConfig} from './kernels/Diag';
import {dilation2dConfig} from './kernels/Dilation2D';
import {dilation2dBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';
import {dilation2dBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {eluConfig} from './kernels/Elu';
import {equalConfig} from './kernels/Equal';
import {erfConfig} from './kernels/Erf';
import {expConfig} from './kernels/Exp';
import {floorConfig} from './kernels/Floor';
import {floorDivConfig} from './kernels/FloorDiv';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {log1pConfig} from './kernels/Log1p';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {logicalNotConfig} from './kernels/LogicalNot';
import {logicalOrConfig} from './kernels/LogicalOr';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {minConfig} from './kernels/Min';
import {minimumConfig} from './kernels/Minimum';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {onesLikeConfig} from './kernels/OnesLike';
import {preluConfig} from './kernels/Prelu';
import {prodConfig} from './kernels/Prod';
import {reluConfig} from './kernels/Relu';
import {relu6Config} from './kernels/Relu6';
import {roundConfig} from './kernels/Round';
import {rsqrtConfig} from './kernels/Rsqrt';
import {seluConfig} from './kernels/Selu';
import {sigmoidConfig} from './kernels/Sigmoid';
import {signConfig} from './kernels/Sign';
import {sinConfig} from './kernels/Sin';
import {sinhConfig} from './kernels/Sinh';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {sqrtConfig} from './kernels/Sqrt';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {sumConfig} from './kernels/Sum';
import {tanConfig} from './kernels/Tan';
import {tanhConfig} from './kernels/Tanh';
import {unpackConfig} from './kernels/Unpack';
import {unsortedSegmentSumConfig} from './kernels/UnsortedSegmentSum';
import {zerosLikeConfig} from './kernels/ZerosLike';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  absConfig,
  acosConfig,
  acoshConfig,
  addConfig,
  addNConfig,
  argMaxConfig,
  argMinConfig,
  asinConfig,
  asinhConfig,
  atanConfig,
  atanhConfig,
  batchMatMulConfig,
  ceilConfig,
  cosConfig,
  coshConfig,
  diagConfig,
  dilation2dBackpropFilterConfig,
  dilation2dBackpropInputConfig,
  dilation2dConfig,
  eluConfig,
  equalConfig,
  erfConfig,
  expConfig,
  floorConfig,
  floorDivConfig,
  greaterConfig,
  greaterEqualConfig,
  lessConfig,
  lessEqualConfig,
  log1pConfig,
  logConfig,
  logicalAndConfig,
  logicalNotConfig,
  logicalOrConfig,
  maxConfig,
  maximumConfig,
  minConfig,
  minimumConfig,
  mirrorPadConfig,
  multiplyConfig,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  onesLikeConfig,
  preluConfig,
  prodConfig,
  relu6Config,
  reluConfig,
  roundConfig,
  rsqrtConfig,
  seluConfig,
  sigmoidConfig,
  signConfig,
  sinConfig,
  sinhConfig,
  sliceConfig,
  softmaxConfig,
  sqrtConfig,
  squareConfig,
  squaredDifferenceConfig,
  sumConfig,
  tanConfig,
  tanhConfig,
  unpackConfig,
  unsortedSegmentSumConfig,
  zerosLikeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
