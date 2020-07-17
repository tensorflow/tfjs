/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

const fs = require('fs');
let path = require('path');
const rimraf = require('rimraf');
const util = require('util');
const cp = require('child_process');
const os = require('os');
const {
  depsPath,
  depsLibPath,
  depsLibTensorFlowPath,
  LIBTENSORFLOW_VERSION,
  PLATFORM_MAPPING,
  ARCH_MAPPING,
  PLATFORM_EXTENSION,
  ALL_SUPPORTED_COMBINATION,
  modulePath,
  customTFLibUri,
  customAddon
} = require('./deps-constants.js');
const resources = require('./resources');
const {addonName} = require('./get-addon-name.js');

const exists = util.promisify(fs.exists);
const mkdir = util.promisify(fs.mkdir);
const rename = util.promisify(fs.rename);
const rimrafPromise = util.promisify(rimraf);

const BASE_URI =
    'https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-';

const platform = os.platform();
// Use windows path
if (platform === 'win32') {
  path = path.win32;
}
let libType = process.argv[2] === undefined ? 'cpu' : process.argv[2];
let system = `${libType}-${PLATFORM_MAPPING[platform]}-` +
    `${ARCH_MAPPING[os.arch()]}`;
let forceDownload = process.argv[3] === undefined ? undefined : process.argv[3];

let packageJsonFile;

function setPackageJsonFile() {
  packageJsonFile =
      JSON.parse(fs.readFileSync(`${__dirname}/../package.json`).toString());
}

function updateAddonName() {
  if (customAddon !== undefined) {
    Object.assign(packageJsonFile['binary'], customAddon);
  } else {
    packageJsonFile['binary']['package_name'] = addonName;
  }
  const stringFile = JSON.stringify(packageJsonFile, null, 2);
  fs.writeFileSync((`${__dirname}/../package.json`), stringFile);
}

function revertAddonName(orig) {
  packageJsonFile['binary'] = orig;
  const stringFile = JSON.stringify(packageJsonFile, null, 2).concat('\n');
  fs.writeFileSync((`${__dirname}/../package.json`), stringFile);
}

/**
 * Returns the libtensorflow hosted path of the current platform.
 */
function getPlatformLibtensorflowUri() {
  // Exception for mac+gpu user
  if (platform === 'darwin') {
    system = `cpu-${PLATFORM_MAPPING[platform]}-${ARCH_MAPPING[os.arch()]}`;
  }

  if (customTFLibUri !== undefined) {
    return customTFLibUri;
  }

  if (platform === 'linux' && os.arch() === 'arm') {
    return 'https://storage.googleapis.com/tf-builds/libtensorflow_r1_14_linux_arm.tar.gz';
  }

  if (ALL_SUPPORTED_COMBINATION.indexOf(system) === -1) {
    throw new Error(`Unsupported system: ${libType}-${platform}-${os.arch()}`);
  }

  return `${BASE_URI}${system}-${LIBTENSORFLOW_VERSION}.${PLATFORM_EXTENSION}`;
}

/**
 * Ensures a directory exists, creates as needed.
 */
async function ensureDir(dirPath) {
  if (!await exists(dirPath)) {
    await mkdir(dirPath);
  }
}

/**
 * Deletes the deps directory if it exists, and creates a fresh deps folder.
 */
async function cleanDeps() {
  if (await exists(depsPath)) {
    await rimrafPromise(depsPath);
  }
  await mkdir(depsPath);
}

/**
 * Downloads libtensorflow and notifies via a callback when unpacked.
 */
async function downloadLibtensorflow(callback) {
  // Ensure dependencies staged directory is available:
  await ensureDir(depsPath);

  console.warn('* Downloading libtensorflow');
  resources.downloadAndUnpackResource(
      getPlatformLibtensorflowUri(), depsPath, async () => {
        if (platform === 'win32') {
          // Some windows libtensorflow zip files are missing structure and the
          // eager headers. Check, restructure, and download resources as
          // needed.
          if (!await exists(depsLibTensorFlowPath)) {
            // Verify that tensorflow.dll exists
            const libtensorflowDll = path.join(depsPath, 'tensorflow.dll');
            if (!await exists(libtensorflowDll)) {
              throw new Error('Could not find libtensorflow.dll');
            }

            await ensureDir(depsLibPath);
            await rename(libtensorflowDll, depsLibTensorFlowPath);
          }
        }
        // No other work is required on other platforms.
        if (callback !== undefined) {
          callback();
        }
      });
}

/**
 * Calls node-gyp for Node.js Tensorflow binding after lib is downloaded.
 */
async function build() {
  // Load package.json file
  setPackageJsonFile();
  // Update addon name in package.json file
  const origBinary = JSON.parse(JSON.stringify(packageJsonFile['binary']));
  updateAddonName();
  console.error('* Building TensorFlow Node.js bindings');
  let buildOption = '--fallback-to-build';
  if (customTFLibUri !== undefined && customAddon === undefined) {
    // Has custom tensorflow shared libs but no addon. Then build it from source
    buildOption = '--build-from-source';
  }
  cp.exec(`node-pre-gyp install ${buildOption}`, (err) => {
    if (err) {
      console.log('node-pre-gyp install failed with error: ' + err);
    }
    if (platform === 'win32') {
      // Move libtensorflow to module path, where tfjs_binding.node locates.
      cp.exec('node scripts/deps-stage.js symlink ' + modulePath);
    }
    revertAddonName(origBinary);
  });
}

/**
 * Ensures libtensorflow requirements are met for building the binding.
 */
async function run() {
  // First check if deps library exists:
  if (forceDownload !== 'download' && await exists(depsLibTensorFlowPath)) {
    // Library has already been downloaded, then compile and simlink:
    await build();
  } else {
    // Library has not been downloaded, download, then compile and symlink:
    await cleanDeps();
    await downloadLibtensorflow(build);
  }
}

run();
