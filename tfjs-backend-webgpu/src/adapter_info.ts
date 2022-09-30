/**
 * @license
 * Copyright 2022 Google LLC.
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

// TODO: Remove it once webgpu/types is successfully upgraded.
// https://github.com/tensorflow/tfjs/issues/6869
export interface GPUAdapterInfo {
  vendor: string;
  architecture: string;
}

export class AdapterInfo {
  private vendor: string;

  constructor(adapterInfo: GPUAdapterInfo) {
    if (adapterInfo) {
      this.vendor = adapterInfo.vendor;
    }
  }

  isIntel(): boolean {
    return this.vendor === 'intel';
  }
}
