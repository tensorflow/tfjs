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
// ./scripts/tag-version.js DIR_NAME
// Where DIR_NAME is the directory name for the package you want to make a
// version for.
const fs = require('fs');

let dirName = process.argv[2];
if (dirName.endsWith('/')) {
  dirName = dirName.slice(0, -1);
}
const packageJsonFile = dirName + '/package.json';
if (!fs.existsSync(packageJsonFile)) {
  console.log(
    packageJsonFile, 'does not exist. Please call this script as follows:');
  console.log('./scripts/tag-version.js DIR_NAME');
  process.exit(1);
}

var exec = require('child_process').exec;

var version = JSON.parse(fs.readFileSync(packageJsonFile, 'utf8')).version;
var tag = `${dirName}-v${version}`;

exec(`git tag ${tag} && git push --tags`, (err, stdout, stderr) => {
  if (err) {
    throw new Error(`Could not git tag with ${tag}: ${err.message}.`);
  }
  console.log(`Successfully tagged with ${tag}.`);
});
