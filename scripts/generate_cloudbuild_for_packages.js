#!/usr/bin/env node
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

const {findPackagesWithDiff, allPackages} =
    require('./find_packages_with_diff.js');
const {generateCloudbuild} = require('./generate_cloudbuild.js');
const {ArgumentParser} = require('argparse');
const yaml = require('js-yaml');
const fs = require('fs');

const parser = new ArgumentParser({
  description: 'Generate a cloudbuild file to test packages. When run' +
      ' with no arguments, tests packages affected by the current' +
      ' changes.'
});

parser.addArgument('packages', {
  type: String,
  nargs: '*',
  help: 'packages to consider as having changed',
});

parser.addArgument(['-o', '--output'], {
  type: String,
  nargs: '?',
  defaultValue: 'cloudbuild_generated.yml',
});

const args = parser.parseArgs(process.argv.slice(2));

let packages;
if (args.packages.length > 0) {
  // Test packages specified in command line args.
  packages = args.packages;
} else if (process.env['NIGHTLY']) {
  // Test all packages during the nightly build.
  packages = allPackages;
} else {
  // Test packages that have changed.
  packages = findPackagesWithDiff();
}

const cloudbuild = generateCloudbuild(packages);
fs.writeFileSync(args.output, yaml.safeDump(cloudbuild));
