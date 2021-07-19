#!/usr/bin/env node
// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

const shell = require('shelljs');
const {exec} = require('../../scripts/test-util');
const {showDiff, getFileSizeBytes} = require('../../scripts/bundle-size-util');

// Get the bundle sizes from this change.
exec(`yarn rollup -c`, {silent: false});

const bundleFilename = 'dist/tf-backend-wasm.min.js';
const minBundleSize = getFileSizeBytes(bundleFilename);
const wasmFileName = 'dist/tfjs-backend-wasm.wasm';
const wasmSize = getFileSizeBytes(wasmFileName);

// Clone master and get the bundle size from master.
const dirName = '/tmp/tfjs-backend-wasm-bundle';
const wasmDirName = 'tfjs-backend-wasm';
exec(
    `git clone --depth=1 --single-branch ` +
        `https://github.com/tensorflow/tfjs ${dirName}`,
    {silent: true});

shell.cd(dirName);
shell.cd(wasmDirName);

exec(
    `yarn && yarn build-deps && yarn build-ci && yarn rollup -c`,
    {silent: false});

const masterMinBundleSize = getFileSizeBytes(bundleFilename);
const masterWasmSize = getFileSizeBytes(wasmFileName);

console.log(`~~~~ WASM file ~~~~`);
console.log(`==> post-gzip`)
showDiff(wasmSize.gzipFileSizeBytes, masterWasmSize.gzipFileSizeBytes);
console.log();
console.log(`==> pre-gzip`)
showDiff(wasmSize.fileSizeBytes, masterWasmSize.fileSizeBytes);
console.log();
console.log();

console.log(`~~~~ minified bundle (JavaScript) ~~~~`);
console.log(`==> post-gzip`)
showDiff(
    minBundleSize.gzipFileSizeBytes, masterMinBundleSize.gzipFileSizeBytes);
console.log();
console.log(`==> pre-gzip`)
showDiff(minBundleSize.fileSizeBytes, masterMinBundleSize.fileSizeBytes);
console.log();
console.log();

exec(`rm -r ${dirName}`);
