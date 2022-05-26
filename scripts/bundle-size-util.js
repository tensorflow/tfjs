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
const {exec} = require('./test-util');

function getFileSizeBytes(filename) {
  const fileSizeBytes =
      +(exec(`cat ${filename} | wc -c`, {silent: true}));
  const gzipFileSizeBytes =
      +(exec(`gzip -c ${filename} | wc -c`, {silent: true}));
  return {fileSizeBytes, gzipFileSizeBytes};
}

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

exports.getFileSizeBytes = getFileSizeBytes;
exports.showDiff = showDiff;
