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

import {Identity, IdentityInputs, KernelConfig} from '@tensorflow/tfjs-core';

export const identityConfig: KernelConfig = {
  kernelName: Identity,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as IdentityInputs;
    // No need to incRef on the backend because node backend does not use
    // other kernels as itermediates. We re-use the dataId here to allow
    // core to do the appropriate book-keeping on the tensor and its clones.
    return {dataId: x.dataId, shape: x.shape, dtype: x.dtype};
  }
};
