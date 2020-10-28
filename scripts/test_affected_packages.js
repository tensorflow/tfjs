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

const {findPackagesWithDiff, allPackages}
      = require('./find_packages_with_diff.js');
const {generateCloudbuild} = require('./generate_cloudbuild.js');
const shell = require('shelljs');


let packages;
if (process.argv.length > 2) {
  // Test packages specified in command line args.
  packages = process.argv.slice(2);
} else if (process.env['NIGHTLY']) {
  // Test all packages during the nightly build.
  packages = allPackages;
} else {
  // Test packages that have changed.
  packages = findPackagesWithDiff();
}


generateCloudbuild(packages);
shell.exec('gcloud builds submit . --config=cloudbuild_generated.yml '
	  +`--substitutions _NIGHTLY=${process.env['NIGHTLY']}`);
