#!/usr/bin/env node
// Copyright 2018 Google LLC. All Rights Reserved.
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

// Run this script from the base directory (not the package directory):
// ./scripts/make-version.js DIR_NAME
// Where DIR_NAME is the directory name for the package you want to make a
// version for.
const fs = require('fs');

const dirName = process.argv[2];
const packageJsonFile = dirName + '/package.json';
if (!fs.existsSync(packageJsonFile)) {
  console.log(
      packageJsonFile, 'does not exist. Please call this script as follows:');
  console.log('./scripts/make-version.js DIR_NAME');
  process.exit(1);
}

const version = JSON.parse(fs.readFileSync(packageJsonFile, 'utf8')).version;

const versionCode = `/** @license See the LICENSE file. */

// This code is auto-generated, do not modify this file!
const version = '${version}';
export {version};
`

fs.writeFile(dirName + '/src/version.ts', versionCode, err => {
  if (err) {
    throw new Error(`Could not save version file ${version}: ${err}`);
  }
  console.log(`Version file for version ${version} saved sucessfully.`);
});

if (dirName === 'tfjs-converter') {
  const pipVersionCode = `# @license See the LICENSE file.

# This code is auto-generated, do not modify this file!
version = '${version}'
`;

  fs.writeFile(
      dirName + '/python/tensorflowjs/version.py', pipVersionCode, err => {
        if (err != null) {
          throw new Error(`Could not save pip version file ${version}: ${err}`);
        }
        console.log(
            `Version file for pip version ${version} saved sucessfully.`);
      });

  const buildFilename = dirName + '/python/BUILD';
  fs.readFile(buildFilename, 'utf-8', function(err, data) {
    if (err != null) {
      throw new Error(`Could not update the BUILD file: ${err}`);
    }

    const newValue = data.replace(
        /version\ =\ \"[0-9]+\.[0-9]+\.[0-9]+\"/g, `version = "${version}"`);
    fs.writeFileSync(buildFilename, newValue, 'utf-8');

    console.log(
        `pip version ${version} for BUILD file is updated sucessfully.`);
  });
}
