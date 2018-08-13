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
const https = require('https');
const fs = require('fs');
let path = require('path');
const rimraf = require('rimraf');
const tar = require('tar');
const util = require('util');
const zip = require('adm-zip');

const copy = util.promisify(fs.copyFile);
const exists = util.promisify(fs.exists);
const mkdir = util.promisify(fs.mkdir);
const rename = util.promisify(fs.rename);
const rimrafPromise = util.promisify(rimraf);
const symlink = util.promisify(fs.symlink);
const unlink = util.promisify(fs.unlink);

const BASE_URI = 'https://storage.googleapis.com/tf-builds/';
const CPU_DARWIN = 'libtensorflow_r1_9_darwin.tar.gz';
const CPU_LINUX = 'libtensorflow_r1_9_linux_cpu.tar.gz';
const GPU_LINUX = 'libtensorflow_r1_9_linux_gpu.tar.gz';
const CPU_WINDOWS = 'libtensorflow_r1_9_windows_cpu.zip';

const platform = process.argv[2];
let action = process.argv[3];
let targetDir = process.argv[4];

let targetUri = BASE_URI;
let libName = 'libtensorflow.so';
if (platform === 'linux-cpu') {
  targetUri += CPU_LINUX;
} else if (platform === 'linux-gpu') {
  targetUri += GPU_LINUX;
} else if (platform === 'darwin') {
  targetUri += CPU_DARWIN;
} else if (platform.endsWith('windows')) {
  targetUri += CPU_WINDOWS;
  libName = 'tensorflow.dll';

  // Some windows machines contain a trailing " char:
  if (targetDir != undefined && targetDir.endsWith('"')) {
    targetDir = targetDir.substr(0, targetDir.length - 1);
  }

  // Windows action can have a path passed in:
  if (action.startsWith('..\\')) {
    action = action.substr(3);
  }

  // Use windows path
  path = path.win32;
} else {
  throw new Error(`Unsupported platform: ${platform}`);
}

const depsPath = path.join(__dirname, '..', 'deps');
const depsLibPath = path.join(depsPath, 'lib', libName);
const destLibPath =
    targetDir !== undefined ? path.join(targetDir, libName) : undefined;

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
 * Symlinks the extracted libtensorflow library to the destination path. If the
 * symlink fails, a copy is made.
 */
async function symlinkDepsLib() {
  if (destLibPath === undefined) {
    throw new Error('Destination path not supplied!');
  }
  try {
    await symlink(depsLibPath, destLibPath);
  } catch (e) {
    console.error(
        `  * Symlink of ${destLibPath} failed, creating a copy on disk.`);
    await copy(depsLibPath, destLibPath);
  }
}

/**
 * Moves the deps library path to the destination path.
 */
async function moveDepsLib() {
  if (destLibPath === undefined) {
    throw new Error('Destination path not supplied!');
  }
  await rename(depsLibPath, destLibPath);
}

/**
 * Downloads libtensorflow and notifies via a callback when unpacked.
 */
async function downloadLibtensorflow(callback) {
  // The deps folder and resources do not exist, download and callback as
  // needed:
  console.error('  * Downloading libtensorflow');

  // Ensure dependencies staged directory is available:
  await ensureDir(depsPath);

  const request = https.get(targetUri, response => {
    if (platform.endsWith('windows')) {
      // Windows stores builds in a zip file. Save to disk, extract, and delete
      // the downloaded archive.
      const tempFileName = path.join(__dirname, '_libtensorflow.zip');
      const outputFile = fs.createWriteStream(tempFileName);
      const request = https.get(targetUri, response => {
        response.pipe(outputFile).on('close', async () => {
          const zipFile = new zip(tempFileName);
          zipFile.extractAllTo(depsPath, true /* overwrite */);
          await unlink(tempFileName);

          if (callback !== undefined) {
            callback();
          }
        });
        request.end();
      });
    } else {
      // All other platforms use a tarball:
      response.pipe(tar.x({C: depsPath, strict: true})).on('close', () => {
        callback();
      });
    }
    request.end();
  });
}

/**
 * Ensures libtensorflow requirements are met for building the binding.
 */
async function run() {
  // Validate the action passed to the script:
  // - 'download' - Just downloads libtensorflow
  // - 'symlink'  - Downloads libtensorflow as needed, symlinks to dest.
  // - 'move'     - Downloads libtensorflow as needed, copies to dest.
  if (action === 'download') {
    // This action always re-downloads. Delete existing deps and start download.
    await cleanDeps();
    await downloadLibtensorflow();
  } else if (action === 'symlink') {
    // First check if deps library exists:
    if (await exists(depsLibPath)) {
      // Library has already been downloaded, simlink:
      await symlinkDepsLib();
    } else {
      // Library has not been downloaded, download and symlink:
      await cleanDeps();
      await downloadLibtensorflow(symlinkDepsLib);
    }
  } else if (action === 'move') {
    // Move action is used when installing this module as a package, always
    // clean, download, and move the lib.
    await cleanDeps();
    await downloadLibtensorflow(await moveDepsLib);
  } else {
    throw new Error('Invalid action: ' + action);
  }
}

run();
