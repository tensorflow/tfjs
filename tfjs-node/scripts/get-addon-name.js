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
const fs = require('fs');
const join = require('path').join;
const name = require('../package.json').name;
const version = require('../package.json').version;

const platform = os.platform();
const CUSTOM_BINARY_FILENAME = 'custom-binary.json';
const customBinaries = loadCustomBinary();

/** Map the os.arch() to arch string in a file name */
const ARCH_MAPPING = { 'x64': 'x86_64' };
/** Map the os.platform() to the platform value in a file name */
const PLATFORM_MAPPING = {
  'darwin': 'darwin',
  'linux': 'linux',
  'win32': 'windows'
};
/** The extension of a compressed file */
const PLATFORM_EXTENSION = os.platform() === 'win32' ? 'zip' : 'tar.gz';
/**
 * Current supported type, platform and architecture combinations
 * `tf-lib` represents tensorflow shared libraries and `binding` represents
 * node binding.
 */
const ALL_SUPPORTED_COMBINATION = [
  'cpu-darwin-x86_64',
  'gpu-linux-x86_64',
  'cpu-linux-x86_64',
  'cpu-windows-x86_64',
  'gpu-windows-x86_64'
];

const type = name.includes('gpu')? 'GPU': 'CPU';
const addonName = `${type}-${PLATFORM_MAPPING[platform]}-` +
    `${version}.${PLATFORM_EXTENSION}`;

// Print out the addon tarball name so that it can be used in bash script when
// uploading the tarball to GCP bucket.
console.log(addonName);

function loadCustomBinary() {
  const cfg = join(__dirname, CUSTOM_BINARY_FILENAME);
  return fs.existsSync(cfg) ? require(cfg) : {};
}

function getCustomBinary(name) {
  return customBinaries[name];
}

module.exports = {
  addonName: addonName,
  customTFLibUri: customBinaries['tf-lib'],
  customAddon: customBinaries['addon'],
  ARCH_MAPPING,
  PLATFORM_MAPPING,
  PLATFORM_EXTENSION,
  ALL_SUPPORTED_COMBINATION
};
