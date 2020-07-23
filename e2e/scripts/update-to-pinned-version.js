#!/usr/bin/env node
// Copyright 2020 Google LLC. All Rights Reserved.
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

// Run this script from the e2e root directory:
// node ./scripts/update-to-pinned-version.js VERSION
// Where VERSION is the pinned version to update to for tfjs package
// dependencies.
const fs = require('fs');

const DEPENDENCY_LIST = [
  'tfjs-core', 'tfjs-converter', 'tfjs-layers', 'tfjs-backend-cpu',
  'tfjs-backend-webgl', 'tfjs-data', 'tfjs', 'tfjs-node'
];

// Todo(linazhao): Refactor release-util.ts to import this js function.
function updateTFJSDependencyVersions(deps, pkg, parsedPkg, tfjsVersion) {
  if (deps != null) {
    for (let j = 0; j < deps.length; j++) {
      const dep = deps[j];

      // Get the current dependency package version.
      let version = '';
      const depNpmName = `@tensorflow/${dep}`;
      if (parsedPkg['dependencies'] != null &&
          parsedPkg['dependencies'][depNpmName] != null) {
        version = parsedPkg['dependencies'][depNpmName];
      } else if (
          parsedPkg['peerDependencies'] != null &&
          parsedPkg['peerDependencies'][depNpmName] != null) {
        version = parsedPkg['peerDependencies'][depNpmName];
      } else if (
          parsedPkg['devDependencies'] != null &&
          parsedPkg['devDependencies'][depNpmName] != null) {
        version = parsedPkg['devDependencies'][depNpmName];
      }
      if (version == null) {
        throw new Error(`No dependency found for ${dep}.`);
      }

      let relaxedVersionPrefix = '';
      if (version.startsWith('~') || version.startsWith('^')) {
        relaxedVersionPrefix = version.substr(0, 1);
      }
      const versionLatest = relaxedVersionPrefix + tfjsVersion;

      pkg = pkg.replace(
          new RegExp(`"${depNpmName}": "${version}"`, 'g'),
          `"${depNpmName}": "${versionLatest}"`);
    }
  }

  return pkg;
}

const latestVersion = process.argv[2];
if (!latestVersion) {
  console.log('Please pass a valid version.');
  process.exit(1);
}

const dirName = path.basename(process.cwd());
if (dirName != 'e2e') {
  console.log(
      `Expect to run this script from e2e, instead run from ${dirName}`);
  process.exit(1);
}

const packageJsonFile = './package.json';
if (!fs.existsSync(packageJsonFile)) {
  console.log(
      packageJsonFile,
      'does not exist. Please call this script from the e2e root directory.');
  process.exit(1);
}

// Update the version.
let pkg = fs.readFileSync(packageJsonFile, 'utf8');
const parsedPkg = JSON.parse(pkg);

pkg = updateTFJSDependencyVersions(
    DEPENDENCY_LIST, pkg, parsedPkg, latestVersion);

fs.writeFileSync(packageJsonFile, pkg);
