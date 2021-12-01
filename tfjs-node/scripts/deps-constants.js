/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
const module_path_napi = require('../package.json').binary.module_path;
const modulePath =
    module_path_napi.replace('{napi_build_version}', process.versions.napi);

/** Version of the libtensorflow shared library to depend on. */
const LIBTENSORFLOW_VERSION = '2.7.0';

/** Map the os.arch() to arch string in a file name */
const ARCH_MAPPING = {
  'x64': 'x86_64',
  'arm64': 'arm64'
};
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
  'cpu-darwin-x86_64', 'gpu-linux-x86_64', 'cpu-linux-x86_64',
  'cpu-windows-x86_64', 'gpu-windows-x86_64'
];

/** Get the MAJOR.MINOR-only version of libtensorflow. */
function getLibTensorFlowMajorDotMinorVersion() {
  const items = LIBTENSORFLOW_VERSION.split('.');
  if (items.length < 3) {
    throw new Error(
        `Invalid version string for libtensorflow: ${LIBTENSORFLOW_VERSION}`);
  }
  return `${items[0]}.${items[1]}`;
}

// Determine constants for deps folder names and destination (build) path names.
let depsLibTensorFlowName = 'libtensorflow';
let depsLibTensorFlowFrameworkName = 'libtensorflow_framework';

let destLibTensorFlowName = depsLibTensorFlowName;
let destLibTensorFlowFrameworkName = depsLibTensorFlowFrameworkName;

if (os.platform() === 'win32') {
  depsLibTensorFlowName = 'tensorflow.dll';
  depsLibTensorFlowFrameworkName = '';  // Not supported on Windows

  destLibTensorFlowName = depsLibTensorFlowName;
  destLibTensorFlowFrameworkName = '';  // Not supported on Windows
} else if (os.platform() === 'darwin') {
  depsLibTensorFlowName += '.dylib';
  depsLibTensorFlowFrameworkName += '.dylib';

  destLibTensorFlowName = depsLibTensorFlowName;
  destLibTensorFlowFrameworkName = depsLibTensorFlowFrameworkName;
} else if (os.platform() === 'linux') {
  // Linux has a hard-coded version number, make the destination name simpler:
  depsLibTensorFlowName += `.so.${LIBTENSORFLOW_VERSION}`;
  depsLibTensorFlowFrameworkName += `.so.${LIBTENSORFLOW_VERSION}`;
  destLibTensorFlowName += '.so';
  destLibTensorFlowFrameworkName += '.so';
} else {
  throw Exception('Unsupported platform: ' + os.platform());
}

const depsPath = join(__dirname, '..', 'deps');
const depsLibPath = join(depsPath, 'lib');

const depsLibTensorFlowPath = join(depsLibPath, depsLibTensorFlowName);
const depsLibTensorFlowFrameworkPath =
    join(depsLibPath, depsLibTensorFlowFrameworkName);

// Get information for custom binary
const CUSTOM_BINARY_FILENAME = 'custom-binary.json';
function loadCustomBinary() {
  const cfg = join(__dirname, CUSTOM_BINARY_FILENAME);
  return fs.existsSync(cfg) ? require(cfg) : {};
}
const customBinaries = loadCustomBinary();

module.exports = {
  depsPath,
  depsLibPath,
  depsLibTensorFlowFrameworkName,
  depsLibTensorFlowFrameworkPath,
  depsLibTensorFlowName,
  depsLibTensorFlowPath,
  destLibTensorFlowFrameworkName,
  destLibTensorFlowName,
  getLibTensorFlowMajorDotMinorVersion,
  modulePath,
  LIBTENSORFLOW_VERSION,
  ARCH_MAPPING,
  PLATFORM_MAPPING,
  PLATFORM_EXTENSION,
  ALL_SUPPORTED_COMBINATION,
  customTFLibUri: customBinaries['tf-lib'],
  customAddon: customBinaries['addon']
};
