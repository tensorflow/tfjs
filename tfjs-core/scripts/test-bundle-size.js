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

function getFileSizeBytes(filename) {
  const gzipFilename = `${filename}.gzip}`;
  exec(`gzip -c ${filename} > ${gzipFilename}`, {silent: true});
  const fileSizeBytes =
      +(exec(`ls -l ${filename} | awk '{print $5}'`, {silent: true}));
  const gzipFileSizeBytes =
      +(exec(`ls -l ${gzipFilename} | awk '{print $5}'`, {silent: true}));
  return {fileSizeBytes, gzipFileSizeBytes};
}

// Get the bundle sizes from this change.
exec(`yarn rollup -c --ci`, {silent: true});
const minSize = getFileSizeBytes('dist/tf-core.min.js');

// Clone master and get the bundle size from master.
const dirName = '/tmp/tfjs-core-bundle';
exec(
    `git clone --depth=1 --single-branch ` +
        `https://github.com/tensorflow/tfjs-core.git ${dirName}`,
    {silent: true});

shell.cd(dirName);
exec(`yarn && yarn rollup -c --ci`, {silent: true});

const masterMinSize = getFileSizeBytes('dist/tf-core.min.js');

function showDiff(newSize, masterSize) {
  const diffBytes = newSize - masterSize;
  const diffPercent = (100 * diffBytes / masterSize).toFixed(2);
  const sign = diffBytes > 0 ? '+' : '';

  const charWidth = 7;
  const diffKiloBytes =
      (sign + (diffBytes / 1024).toFixed(2)).padStart(charWidth, ' ');
  const masterKiloBytes =
      ((masterSize / 1024).toFixed(2)).padStart(charWidth, ' ');
  const newKiloBytes = ((newSize / 1024).toFixed(2)).padStart(charWidth, ' ');

  console.log(`  diff:   ${diffKiloBytes} K  (${sign}${diffPercent}%)`);
  console.log(`  master: ${masterKiloBytes} K`);
  console.log(`  change: ${newKiloBytes} K`);
}

console.log(`~~~~minified bundle~~~~`);
console.log(`==> post-gzip`)
showDiff(minSize.gzipFileSizeBytes, masterMinSize.gzipFileSizeBytes);
console.log();
console.log(`==> pre-gzip`)
showDiff(minSize.fileSizeBytes, masterMinSize.fileSizeBytes);
console.log();
console.log();

exec(`rm -r ${dirName}`);
