
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

// Import shared functionality from tfjs-backend-cpu without triggering
// side effects.
// tslint:disable-next-line: no-imports-from-dist
import * as shared from '@tensorflow/tfjs-backend-cpu/dist/shared';
// tslint:disable-next-line: no-imports-from-dist
import {SimpleBinaryKernelImpl} from '@tensorflow/tfjs-backend-cpu/dist/shared';
// tslint:disable-next-line: no-imports-from-dist
import {SimpleUnaryImpl} from '@tensorflow/tfjs-backend-cpu/dist/utils/unary_types';

export type SimpleBinaryKernelImplCPU = SimpleBinaryKernelImpl;
export type SimpleUnaryKernelImplCPU = SimpleUnaryImpl;
const {
  addImpl: addImplCPU,
  bincountImpl: bincountImplCPU,
  bincountReduceImpl: bincountReduceImplCPU,
  ceilImpl: ceilImplCPU,
  concatImpl: concatImplCPU,
  equalImpl: equalImplCPU,
  expImpl: expImplCPU,
  expm1Impl: expm1ImplCPU,
  floorImpl: floorImplCPU,
  gatherNdImpl: gatherNdImplCPU,
  gatherV2Impl: gatherV2ImplCPU,
  greaterImpl: greaterImplCPU,
  greaterEqualImpl: greaterEqualImplCPU,
  lessImpl: lessImplCPU,
  lessEqualImpl: lessEqualImplCPU,
  linSpaceImpl: linSpaceImplCPU,
  logImpl: logImplCPU,
  maxImpl: maxImplCPU,
  maximumImpl: maximumImplCPU,
  minimumImpl: minimumImplCPU,
  multiplyImpl: multiplyImplCPU,
  negImpl: negImplCPU,
  notEqualImpl: notEqualImplCPU,
  prodImpl: prodImplCPU,
  rangeImpl: rangeImplCPU,
  rsqrtImpl: rsqrtImplCPU,
  sigmoidImpl: sigmoidImplCPU,
  simpleAbsImpl: simpleAbsImplCPU,
  sliceImpl: sliceImplCPU,
  sparseFillEmptyRowsImpl: sparseFillEmptyRowsImplCPU,
  sparseReshapeImpl: sparseReshapeImplCPU,
  sparseSegmentReductionImpl: sparseSegmentReductionImplCPU,
  sqrtImpl: sqrtImplCPU,
  stridedSliceImpl: stridedSliceImplCPU,
  stringNGramsImpl: stringNGramsImplCPU,
  stringSplitImpl: stringSplitImplCPU,
  stringToHashBucketFastImpl: stringToHashBucketFastImplCPU,
  subImpl: subImplCPU,
  tileImpl: tileImplCPU,
  topKImpl: topKImplCPU,
  transposeImpl: transposeImplCPU,
  uniqueImpl: uniqueImplCPU,
} = shared;

export {
  addImplCPU,
  bincountImplCPU,
  bincountReduceImplCPU,
  ceilImplCPU,
  concatImplCPU,
  equalImplCPU,
  expImplCPU,
  expm1ImplCPU,
  floorImplCPU,
  gatherNdImplCPU,
  gatherV2ImplCPU,
  greaterEqualImplCPU,
  greaterImplCPU,
  lessEqualImplCPU,
  lessImplCPU,
  linSpaceImplCPU,
  logImplCPU,
  maxImplCPU,
  maximumImplCPU,
  minimumImplCPU,
  multiplyImplCPU,
  negImplCPU,
  notEqualImplCPU,
  prodImplCPU,
  sigmoidImplCPU,
  simpleAbsImplCPU,
  sliceImplCPU,
  sparseFillEmptyRowsImplCPU,
  sparseReshapeImplCPU,
  sparseSegmentReductionImplCPU,
  sqrtImplCPU,
  stridedSliceImplCPU,
  stringNGramsImplCPU,
  stringSplitImplCPU,
  stringToHashBucketFastImplCPU,
  subImplCPU,
  rangeImplCPU,
  rsqrtImplCPU,
  tileImplCPU,
  topKImplCPU,
  transposeImplCPU,
  uniqueImplCPU,
};
