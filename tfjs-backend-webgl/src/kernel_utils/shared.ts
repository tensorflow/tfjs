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

const {
  simpleAbsImpl: simpleAbsImplCPU,
  addImpl: addImplCPU,
  ceilImpl: ceilImplCPU,
  expImpl: expImplCPU,
  expm1Impl: expm1ImplCPU,
  floorImpl: floorImplCPU,
  logImpl: logImplCPU,
  maxImpl: maxImplCPU,
  multiplyImpl: multiplyImplCPU,
  rsqrtImpl: rsqrtImplCPU,
  sliceImpl: sliceImplCPU,
  subImpl: subImplCPU,
  transposeImpl: transposeImplCPU,
  uniqueImpl: uniqueImplCPU,
} = shared;

export {
  simpleAbsImplCPU,
  addImplCPU,
  ceilImplCPU,
  expImplCPU,
  expm1ImplCPU,
  logImplCPU,
  multiplyImplCPU,
  sliceImplCPU,
  subImplCPU,
  floorImplCPU,
  maxImplCPU,
  rsqrtImplCPU,
  transposeImplCPU,
  uniqueImplCPU,
};
