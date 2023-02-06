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
import {allConfig} from './kernels/All';
import {anyConfig} from './kernels/Any';
import {argMaxConfig} from './kernels/ArgMax';
import {asinConfig} from './kernels/Asin';
import {asinhConfig} from './kernels/Asinh';
import {atanConfig} from './kernels/Atan';
import {avgPoolConfig} from './kernels/AvgPool';
import {avgPool3DConfig} from './kernels/AvgPool3D';
import {avgPool3DGradConfig} from './kernels/AvgPool3DGrad';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {broadcastArgsConfig} from './kernels/BroadcastArgs';
import {castConfig} from './kernels/Cast';
import {ceilConfig} from './kernels/Ceil';
import {clipByValueConfig} from './kernels/ClipByValue';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {conv2DBackpropInputConfig} from './kernels/Conv2DBackpropInput';
import {cosConfig} from './kernels/Cos';
import {coshConfig} from './kernels/Cosh';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {cumprodConfig} from './kernels/Cumprod';
import {cumsumConfig} from './kernels/Cumsum';
import {denseBincountConfig} from './kernels/DenseBincount';
import {depthToSpaceConfig} from './kernels/DepthToSpace';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {diagConfig} from './kernels/Diag';
import {dilation2DConfig} from './kernels/Dilation2D';
import {dilation2DBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';
import {dilation2DBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {eluConfig} from './kernels/Elu';
import {eluGradConfig} from './kernels/EluGrad';
import {equalConfig} from './kernels/Equal';
import {expConfig} from './kernels/Exp';
import {expandDimsConfig} from './kernels/ExpandDims';
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
import {isNaNConfig} from './kernels/IsNan';
import {leakyReluConfig} from './kernels/LeakyRelu';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {log1pConfig} from './kernels/Log1p';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {logicalNotConfig} from './kernels/LogicalNot';
import {logicalOrConfig} from './kernels/LogicalOr';
import {logicalXorConfig} from './kernels/LogicalXor';
import {lrnConfig} from './kernels/LRN';
import {lrnGradConfig} from './kernels/LRNGrad';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {maxPoolConfig} from './kernels/MaxPool';
import {maxPool3DConfig} from './kernels/MaxPool3D';
import {maxPool3DGradConfig} from './kernels/MaxPool3DGrad';
import {meanConfig} from './kernels/Mean';
import {minConfig} from './kernels/Min';
import {minimumConfig} from './kernels/Minimum';
import {mirrorPadConfig} from './kernels/MirrorPad';
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
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {roundConfig} from './kernels/Round';
import {rsqrtConfig} from './kernels/Rsqrt';
import {scatterNdConfig} from './kernels/ScatterNd';
import {searchSortedConfig} from './kernels/SearchSorted';
import {selectConfig} from './kernels/Select';
import {sigmoidConfig} from './kernels/Sigmoid';
import {sinConfig} from './kernels/Sin';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {sparseFillEmptyRowsConfig} from './kernels/SparseFillEmptyRows';
import {sparseReshapeConfig} from './kernels/SparseReshape';
import {sparseSegmentMeanConfig} from './kernels/SparseSegmentMean';
import {sparseSegmentSumConfig} from './kernels/SparseSegmentSum';
import {sparseToDenseConfig} from './kernels/SparseToDense';
import {splitVConfig} from './kernels/SplitV';
import {sqrtConfig} from './kernels/Sqrt';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {stepConfig} from './kernels/Step';
import {stridedSliceConfig} from './kernels/StridedSlice';
import {stringNGramsConfig} from './kernels/StringNGrams';
import {stringSplitConfig} from './kernels/StringSplit';
import {stringToHashBucketFastConfig} from './kernels/StringToHashBucketFast';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanConfig} from './kernels/Tan';
import {tanhConfig} from './kernels/Tanh';
import {tensorScatterUpdateConfig} from './kernels/TensorScatterUpdate';
import {tileConfig} from './kernels/Tile';
import {topKConfig} from './kernels/TopK';
import {transformConfig} from './kernels/Transform';
import {transposeConfig} from './kernels/Transpose';
import {unpackConfig} from './kernels/Unpack';
import {zerosLikeConfig} from './kernels/ZerosLike';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  _fusedMatMulConfig,
  absConfig,
  acosConfig,
  acoshConfig,
  addConfig,
  addNConfig,
  allConfig,
  anyConfig,
  argMaxConfig,
  asinConfig,
  asinhConfig,
  atanConfig,
  avgPoolConfig,
  avgPool3DConfig,
  avgPool3DGradConfig,
  batchMatMulConfig,
  batchToSpaceNDConfig,
  broadcastArgsConfig,
  castConfig,
  ceilConfig,
  clipByValueConfig,
  concatConfig,
  conv2DConfig,
  conv2DBackpropInputConfig,
  cosConfig,
  coshConfig,
  cropAndResizeConfig,
  cumprodConfig,
  cumsumConfig,
  denseBincountConfig,
  depthToSpaceConfig,
  depthwiseConv2dNativeConfig,
  diagConfig,
  dilation2DConfig,
  dilation2DBackpropFilterConfig,
  dilation2DBackpropInputConfig,
  eluConfig,
  eluGradConfig,
  equalConfig,
  expConfig,
  expandDimsConfig,
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
  isNaNConfig,
  leakyReluConfig,
  lessConfig,
  lessEqualConfig,
  log1pConfig,
  logConfig,
  logicalAndConfig,
  logicalNotConfig,
  logicalOrConfig,
  logicalXorConfig,
  lrnConfig,
  lrnGradConfig,
  maxConfig,
  maximumConfig,
  maxPoolConfig,
  maxPool3DConfig,
  maxPool3DGradConfig,
  meanConfig,
  minConfig,
  minimumConfig,
  mirrorPadConfig,
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
  realDivConfig,
  reciprocalConfig,
  reluConfig,
  relu6Config,
  reshapeConfig,
  resizeBilinearConfig,
  resizeBilinearGradConfig,
  resizeNearestNeighborConfig,
  resizeNearestNeighborGradConfig,
  reverseConfig,
  rotateWithOffsetConfig,
  roundConfig,
  rsqrtConfig,
  scatterNdConfig,
  searchSortedConfig,
  selectConfig,
  sigmoidConfig,
  sinConfig,
  sliceConfig,
  softmaxConfig,
  spaceToBatchNDConfig,
  sparseFillEmptyRowsConfig,
  sparseReshapeConfig,
  sparseSegmentMeanConfig,
  sparseSegmentSumConfig,
  sparseToDenseConfig,
  splitVConfig,
  sqrtConfig,
  squareConfig,
  squaredDifferenceConfig,
  stepConfig,
  stridedSliceConfig,
  stringNGramsConfig,
  stringSplitConfig,
  stringToHashBucketFastConfig,
  subConfig,
  sumConfig,
  tanConfig,
  tanhConfig,
  tensorScatterUpdateConfig,
  tileConfig,
  topKConfig,
  transformConfig,
  transposeConfig,
  unpackConfig,
  zerosLikeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
