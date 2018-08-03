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
const path = require('path');
const tar = require('tar');
const util = require('util');

const exists = util.promisify(fs.exists);
const mkdir = util.promisify(fs.mkdir);
const symlink = util.promisify(fs.symlink);

const BASE_URI = 'https://storage.googleapis.com/tf-builds/';
const CPU_DARWIN = 'libtensorflow_r1_9_darwin.tar.gz';
const CPU_LINUX = 'libtensorflow_r1_9_linux_cpu.tar.gz';
const GPU_LINUX = 'libtensorflow_r1_9_linux_gpu.tar.gz';

const platform = process.argv[2];
const targetDir = process.argv[3];

// TODO(kreeger): Handle windows (dll) support:
// https://github.com/tensorflow/tfjs/issues/549
let targetUri = BASE_URI;
let libName = 'libtensorflow';
if (platform === 'linux-cpu') {
  targetUri += CPU_LINUX;
  libName += '.so';
} else if (platform === 'linux-gpu') {
  targetUri += GPU_LINUX;
  libName += '.so';
} else if (platform === 'darwin') {
  targetUri += CPU_DARWIN;
  libName += '.so';
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
 * Symlinks the extracted libtensorflow library to the desired directory.
 */
async function symlinkDepsLib() {
  await symlink(depsLibPath, destLibPath);
}

/**
 * Downloads libtensorflow and optionally symlinks the library as needed.
 */
async function downloadLibtensorflow(shouldSymlink) {
  // The deps folder and resources do not exist, download and symlink as
  // needed:
  console.error('  * Downloading libtensorflow');
  const request = https.get(targetUri, response => {
    response
        .pipe(tar.x({
          C: depsPath,
        }))
        .on('close', async () => {
          if (shouldSymlink) {
            await symlinkDepsLib();
          }
        });
    request.end();
  });
}

/**
 * Ensures libtensorflow requirements are met for building the binding.
 */
async function run() {
  // Ensure dependencies staged directory is available:
  await ensureDir(depsPath);

  // This script can optionally only download and not symlink:
  if (destLibPath !== undefined) {
    if (await exists(depsLibPath)) {
      // The libtensorflow package has already been downloaded, simply simlink
      // to the destination path:
      await symlinkDepsLib();
    } else {
      // The libtensorflow package does not exist, download and symlink to the
      // destination path:
      downloadLibtensorflow(true);
    }
  } else {
    // No symlink destination path supplied, simply download libtensorflow:
    downloadLibtensorflow(false);
  }
}

run();
