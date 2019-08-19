/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
const os = require('os');
const name = require('../package.json').name;
const version = require('../package.json').version;

const platform = os.platform();

const CPU_DARWIN = `CPU-darwin-${version}.tar.gz`;
const CPU_LINUX = `CPU-linux-${version}.tar.gz`;
const GPU_LINUX = `GPU-linux-${version}.tar.gz`;
const CPU_WINDOWS = `CPU-windows-${version}.zip`;
const GPU_WINDOWS = `GPU-windows-${version}.zip`;

let addonName;

if (name.includes('gpu')) {
  if (platform === 'linux') {
    addonName = GPU_LINUX;
  } else if (platform === 'win32') {
    addonName = GPU_WINDOWS;
  }
} else {
  if (platform === 'linux') {
    addonName = CPU_LINUX;
  } else if (platform === 'darwin') {
    addonName = CPU_DARWIN;
  } else if (platform === 'win32') {
    addonName = CPU_WINDOWS;
  }
}

// Print out the addon tarball name so that it can be used in bash script when
// uploading the tarball to GCP bucket.
console.log(addonName);

module.exports = {
  addonName: addonName
};
