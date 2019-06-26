/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
const path = require('path');

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
  depsLibTensorFlowName += '.so.1.14.0';
  depsLibTensorFlowFrameworkName += '.so.1.14.0';

  destLibTensorFlowName += '.so';
  destLibTensorFlowFrameworkName += '.so';
} else {
  throw Exception('Unsupported platform: ' + os.platform());
}

const depsPath = path.join(__dirname, '..', 'deps');
const depsLibPath = path.join(depsPath, 'lib');

const depsLibTensorFlowPath = path.join(depsLibPath, depsLibTensorFlowName);
const depsLibTensorFlowFrameworkPath =
    path.join(depsLibPath, depsLibTensorFlowFrameworkName);

module.exports = {
  depsPath,
  depsLibPath,
  depsLibTensorFlowFrameworkName,
  depsLibTensorFlowFrameworkPath,
  depsLibTensorFlowName,
  depsLibTensorFlowPath,
  destLibTensorFlowFrameworkName,
  destLibTensorFlowName
};
