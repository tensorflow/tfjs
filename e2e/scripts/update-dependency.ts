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
// yarn update-dependency --version=VERSION
// Where VERSION is the pinned version to update to for tfjs package
// dependencies.

import * as argparse from 'argparse';
import * as fs from 'fs';

import {updateTFJSDependencyVersions} from '../../scripts/release-util';

const parser = new argparse.ArgumentParser();
parser.addArgument('--version', {
  required: true,
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

const DEPENDENCY_LIST = [
  'tfjs-core', 'tfjs-converter', 'tfjs-layers', 'tfjs-backend-cpu',
  'tfjs-backend-webgl', 'tfjs-backend-wasm', 'tfjs-data', 'tfjs', 'tfjs-node'
];

function main() {
  const args = parser.parseArgs();
  const latestVersion = args.version;

  if (!latestVersion) {
    console.log('Please pass a valid version.');
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

  process.exit(0);
}

main();
