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
const cp = require('child_process');
const os = require('os');
const ProgressBar = require('progress');
const {depsPath, depsLibPath} = require('./deps-constants.js');

const exists = util.promisify(fs.exists);
const mkdir = util.promisify(fs.mkdir);
const rimrafPromise = util.promisify(rimraf);
const unlink = util.promisify(fs.unlink);
const exec = util.promisify(cp.exec);

const BASE_URI = 'https://storage.googleapis.com/tf-builds/';
const CPU_DARWIN = 'libtensorflow_r1_10_darwin.tar.gz';
const CPU_LINUX = 'libtensorflow_r1_10_linux_cpu.tar.gz';
const GPU_LINUX = 'libtensorflow_r1_10_linux_gpu.tar.gz';
const CPU_WINDOWS = 'libtensorflow_r1_10_windows_cpu.zip';
const GPU_WINDOWS = 'libtensorflow_r1_10_windows_gpu.zip';

const platform = os.platform();
let libType = process.argv[2] === undefined ? 'cpu' : process.argv[2];
let forceDownload = process.argv[3] === undefined ? undefined : process.argv[3];

let targetUri = BASE_URI;

async function getTargetUri() {
  if (platform === 'linux') {
    if (libType === 'gpu') {
      targetUri += GPU_LINUX;
    } else {
      targetUri += CPU_LINUX;
    }
  } else if (platform === 'darwin') {
    targetUri += CPU_DARWIN;
  } else if (platform === 'win32') {
    // Use windows path
    path = path.win32;
    if (libType === 'gpu') {
      targetUri += GPU_WINDOWS;
    } else {
      targetUri += CPU_WINDOWS;
    }
  } else {
    throw new Error(`Unsupported platform: ${platform}`);
  }
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
  await getTargetUri();
  // The deps folder and resources do not exist, download and callback as
  // needed:
  console.error('* Downloading libtensorflow');

  // Ensure dependencies staged directory is available:
  await ensureDir(depsPath);

  const request = https.get(targetUri, response => {
    const bar = new ProgressBar('[:bar] :rate/bps :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 30,
      total: parseInt(response.headers['content-length'], 10)
    });

    if (platform === 'win32') {
      // Windows stores builds in a zip file. Save to disk, extract, and delete
      // the downloaded archive.
      const tempFileName = path.join(__dirname, '_libtensorflow.zip');
      const outputFile = fs.createWriteStream(tempFileName);
      response
          .on('data',
              (chunk) => {
                bar.tick(chunk.length);
              })
          .pipe(outputFile)
          .on('close', async () => {
            const zipFile = new zip(tempFileName);
            zipFile.extractAllTo(depsPath, true /* overwrite */);
            await unlink(tempFileName);

            if (callback !== undefined) {
              callback();
            }
          });
    } else {
      // All other platforms use a tarball:
      response
          .on('data',
              (chunk) => {
                bar.tick(chunk.length);
              })
          .pipe(tar.x({C: depsPath, strict: true}))
          .on('close', () => {
            if (callback !== undefined) {
              callback();
            }
          });
    }
    request.end();
  });
}

/**
 * Calls node-gyp for Node.js Tensorflow binding after lib is downloaded.
 */
async function build() {
  console.error('* Building TensorFlow Node.js bindings');
  cp.exec('node-gyp rebuild', (err) => {
    if (err) {
      throw new Error('node-gyp rebuild failed with: ' + err);
    }
  });
}

/**
 * Ensures libtensorflow requirements are met for building the binding.
 */
async function run() {
  // First check if deps library exists:
  if (forceDownload !== 'download' && await exists(depsLibPath)) {
    // Library has already been downloaded, then compile and simlink:
    await build();
  } else {
    // Library has not been downloaded, download, then compile and symlink:
    await cleanDeps();
    await downloadLibtensorflow(build);
  }
}

run();
