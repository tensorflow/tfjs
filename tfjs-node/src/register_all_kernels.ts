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
import {KernelConfig, registerKernel} from '@tensorflow/tfjs';

import {_fusedMatMulConfig} from './kernels/_FusedMatMul';
import {absConfig} from './kernels/Abs';
import {acosConfig} from './kernels/Acos';
import {acoshConfig} from './kernels/Acosh';
import {addConfig} from './kernels/Add';
import {addNConfig} from './kernels/AddN';
import {allConfig} from './kernels/All';
import {anyConfig} from './kernels/Any';
import {argMaxConfig} from './kernels/ArgMax';
import {argMinConfig} from './kernels/ArgMin';
import {asinConfig} from './kernels/Asin';
import {asinhConfig} from './kernels/Asinh';
import {atanConfig} from './kernels/Atan';
import {atan2Config} from './kernels/Atan2';
import {atanhConfig} from './kernels/Atanh';
import {avgPoolConfig} from './kernels/AvgPool';
import {avgPool3DConfig} from './kernels/AvgPool3D';
import {avgPool3DGradConfig} from './kernels/AvgPool3DGrad';
import {avgPoolGradConfig} from './kernels/AvgPoolGrad';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {bincountConfig} from './kernels/Bincount';
import {broadcastArgsConfig} from './kernels/BroadcastArgs';
import {castConfig} from './kernels/Cast';
import {ceilConfig} from './kernels/Ceil';
import {clipByValueConfig} from './kernels/ClipByValue';
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
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {cumprodConfig} from './kernels/Cumprod';
import {cumsumConfig} from './kernels/Cumsum';
import {depthToSpaceConfig} from './kernels/DepthToSpace';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {depthwiseConv2dNativeBackpropFilterConfig} from './kernels/DepthwiseConv2dNativeBackpropFilter';
import {depthwiseConv2dNativeBackpropInputConfig} from './kernels/DepthwiseConv2dNativeBackpropInput';
import {diagConfig} from './kernels/Diag';
import {dilation2dConfig} from './kernels/Dilation2D';
import {dilation2dBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';
import {dilation2dBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {einsumConfig} from './kernels/Einsum';
import {eluConfig} from './kernels/Elu';
import {eluGradConfig} from './kernels/EluGrad';
import {equalConfig} from './kernels/Equal';
import {erfConfig} from './kernels/Erf';
import {expConfig} from './kernels/Exp';
import {expandDimsConfig} from './kernels/ExpandDims';
import {expm1Config} from './kernels/Expm1';
import {FFTConfig} from './kernels/FFT';
import {fillConfig} from './kernels/Fill';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {floorConfig} from './kernels/Floor';
import {floorDivConfig} from './kernels/FloorDiv';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {gatherNdConfig} from './kernels/GatherNd';
import {gatherV2Config} from './kernels/GatherV2';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {IFFTConfig} from './kernels/IFFT';
import {imagConfig} from './kernels/Imag';
import {isFiniteConfig} from './kernels/IsFinite';
import {isInfConfig} from './kernels/IsInf';
import {isNanConfig} from './kernels/IsNan';
import {leakyReluConfig} from './kernels/LeakyRelu';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {linSpaceConfig} from './kernels/LinSpace';
import {logConfig} from './kernels/Log';
import {log1pConfig} from './kernels/Log1p';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {logicalNotConfig} from './kernels/LogicalNot';
import {logicalOrConfig} from './kernels/LogicalOr';
import {LRNConfig} from './kernels/LRN';
import {LRNGradConfig} from './kernels/LRNGrad';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {maxPoolConfig} from './kernels/MaxPool';
import {maxPool3DConfig} from './kernels/MaxPool3D';
import {maxPool3DGradConfig} from './kernels/MaxPool3DGrad';
import {maxPoolGradConfig} from './kernels/MaxPoolGrad';
import {meanConfig} from './kernels/Mean';
import {minConfig} from './kernels/Min';
import {minimumConfig} from './kernels/Minimum';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {modConfig} from './kernels/Mod';
import {multinomialConfig} from './kernels/Multinomial';
import {multiplyConfig} from './kernels/Multiply';
import {negConfig} from './kernels/Neg';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {oneHotConfig} from './kernels/OneHot';
import {onesLikeConfig} from './kernels/OnesLike';
import {packConfig} from './kernels/Pack';
import {padV2Config} from './kernels/PadV2';
import {powConfig} from './kernels/Pow';
import {preluConfig} from './kernels/Prelu';
import {prodConfig} from './kernels/Prod';
import {rangeConfig} from './kernels/Range';
import {realConfig} from './kernels/Real';
import {realDivConfig} from './kernels/RealDiv';
import {reciprocalConfig} from './kernels/Reciprocal';
import {reluConfig} from './kernels/Relu';
import {relu6Config} from './kernels/Relu6';
import {reshapeConfig} from './kernels/Reshape';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {resizeBilinearGradConfig} from './kernels/ResizeBilinearGrad';
import {resizeNearestNeighborConfig} from './kernels/ResizeNearestNeighbor';
import {resizeNearestNeighborGradConfig} from './kernels/ResizeNearestNeighborGrad';
import {reverseConfig} from './kernels/Reverse';
import {roundConfig} from './kernels/Round';
import {rsqrtConfig} from './kernels/Rsqrt';
import {scatterNdConfig} from './kernels/ScatterNd';
import {selectConfig} from './kernels/Select';
import {seluConfig} from './kernels/Selu';
import {sigmoidConfig} from './kernels/Sigmoid';
import {signConfig} from './kernels/Sign';
import {sinConfig} from './kernels/Sin';
import {sinhConfig} from './kernels/Sinh';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {softplusConfig} from './kernels/Softplus';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {sparseToDenseConfig} from './kernels/SparseToDense';
import {splitVConfig} from './kernels/SplitV';
import {sqrtConfig} from './kernels/Sqrt';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {staticRegexReplaceConfig} from './kernels/StaticRegexReplace';
import {stepConfig} from './kernels/Step';
import {stridedSliceConfig} from './kernels/StridedSlice';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanConfig} from './kernels/Tan';
import {tanhConfig} from './kernels/Tanh';
import {tensorScatterUpdateConfig} from './kernels/TensorScatterUpdate';
import {tileConfig} from './kernels/Tile';
import {topKConfig} from './kernels/TopK';
import {transposeConfig} from './kernels/Transpose';
import {uniqueConfig} from './kernels/Unique';
import {unpackConfig} from './kernels/Unpack';
import {unsortedSegmentSumConfig} from './kernels/UnsortedSegmentSum';
import {zerosLikeConfig} from './kernels/ZerosLike';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  FFTConfig,
  IFFTConfig,
  LRNConfig,
  LRNGradConfig,
  _fusedMatMulConfig,
  absConfig,
  acosConfig,
  acoshConfig,
  addConfig,
  addNConfig,
  allConfig,
  anyConfig,
  argMaxConfig,
  argMinConfig,
  asinConfig,
  asinhConfig,
  atan2Config,
  atanConfig,
  atanhConfig,
  avgPool3DConfig,
  avgPool3DGradConfig,
  avgPoolConfig,
  avgPoolGradConfig,
  batchMatMulConfig,
  batchToSpaceNDConfig,
  broadcastArgsConfig,
  castConfig,
  ceilConfig,
  clipByValueConfig,
  complexAbsConfig,
  complexConfig,
  concatConfig,
  conv2DBackpropFilterConfig,
  conv2DBackpropInputConfig,
  conv2DConfig,
  conv3DBackpropFilterV2Config,
  conv3DBackpropInputV2Config,
  conv3DConfig,
  cosConfig,
  coshConfig,
  cropAndResizeConfig,
  cumprodConfig,
  cumsumConfig,
  bincountConfig,
  depthToSpaceConfig,
  depthwiseConv2dNativeBackpropFilterConfig,
  depthwiseConv2dNativeBackpropInputConfig,
  depthwiseConv2dNativeConfig,
  diagConfig,
  dilation2dBackpropFilterConfig,
  dilation2dBackpropInputConfig,
  dilation2dConfig,
  eluConfig,
  eluGradConfig,
  einsumConfig,
  equalConfig,
  erfConfig,
  expConfig,
  expandDimsConfig,
  expm1Config,
  fillConfig,
  flipLeftRightConfig,
  floorConfig,
  floorDivConfig,
  fusedBatchNormConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  gatherNdConfig,
  gatherV2Config,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  imagConfig,
  isFiniteConfig,
  isInfConfig,
  isNanConfig,
  leakyReluConfig,
  lessConfig,
  lessEqualConfig,
  linSpaceConfig,
  log1pConfig,
  logConfig,
  logicalAndConfig,
  logicalNotConfig,
  logicalOrConfig,
  maxConfig,
  maxPool3DConfig,
  maxPool3DGradConfig,
  maxPoolConfig,
  maxPoolGradConfig,
  maximumConfig,
  meanConfig,
  minConfig,
  minimumConfig,
  mirrorPadConfig,
  modConfig,
  multinomialConfig,
  multiplyConfig,
  negConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  oneHotConfig,
  onesLikeConfig,
  packConfig,
  padV2Config,
  powConfig,
  preluConfig,
  prodConfig,
  rangeConfig,
  realConfig,
  realDivConfig,
  reciprocalConfig,
  relu6Config,
  reluConfig,
  reshapeConfig,
  resizeBilinearConfig,
  resizeBilinearGradConfig,
  resizeNearestNeighborConfig,
  resizeNearestNeighborGradConfig,
  reverseConfig,
  roundConfig,
  rsqrtConfig,
  scatterNdConfig,
  selectConfig,
  seluConfig,
  sigmoidConfig,
  signConfig,
  sinConfig,
  sinhConfig,
  sliceConfig,
  softmaxConfig,
  softplusConfig,
  spaceToBatchNDConfig,
  sparseToDenseConfig,
  splitVConfig,
  sqrtConfig,
  squareConfig,
  squaredDifferenceConfig,
  staticRegexReplaceConfig,
  stepConfig,
  stridedSliceConfig,
  subConfig,
  sumConfig,
  tanConfig,
  tanhConfig,
  tensorScatterUpdateConfig,
  tileConfig,
  topKConfig,
  transposeConfig,
  uniqueConfig,
  unpackConfig,
  unsortedSegmentSumConfig,
  zerosLikeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
