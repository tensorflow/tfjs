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
import {depthToSpace} from '../../ops/depth_to_space';
import {getGlobalTensorClass, Tensor4D} from '../../tensor';
import {Rank} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    depthToSpace<T extends Tensor4D>(
        blockSize: number, dataFormat: 'NHWC'|'NCHW'): T;
  }
}

getGlobalTensorClass().prototype.depthToSpace = function<T extends Tensor4D>(
    blockSize: number, dataFormat: 'NHWC'|'NCHW'): T {
  this.throwIfDisposed();
  return depthToSpace(this, blockSize, dataFormat) as T;
};
