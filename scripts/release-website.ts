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

/**
 * This script creates pull requests to make release for tfjs website. Once the
 * pull request is merged, you must deploy the website.
 *
 * This script requires hub to be installed: https://hub.github.com/
 */
import chalk from 'chalk';
import * as fs from 'fs';
import * as shell from 'shelljs';

import {$, makeReleaseDir, TMP_DIR, WEBSITE_RELEASE_UNIT, updateDependency, prepareReleaseBuild, createPR} from './release-util';

export async function releaseWebsite(args: any) {
  const {phases, repo} = WEBSITE_RELEASE_UNIT;
  // Website release only has one phase.
  const phase = phases[0];
  const packages = phases[0].packages;
  const deps = phases[0].deps || [];

  const dir = `${TMP_DIR}/${repo}`;
  makeReleaseDir(dir);

  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';

  // Publishing website, another repo.
  $(`git clone ${urlBase}tensorflow/${repo} ${dir} --depth=1`);
  shell.cd(dir);

  for (let i = 0; i < packages.length; i++) {
    const packageName = packages[i];

    // Update the version.
    const packageJsonPath = `${dir}/package.json`;
    let pkg = `${fs.readFileSync(packageJsonPath)}`;
    const parsedPkg = JSON.parse(`${pkg}`);
    const latestVersion = parsedPkg.version;

    console.log(chalk.magenta.bold(
        `~~~ Processing ${packageName} (${latestVersion}) ~~~`));

    pkg = await updateDependency(deps, pkg, parsedPkg);

    fs.writeFileSync(packageJsonPath, pkg);

    prepareReleaseBuild(phase, packageName);
  }

  const timestamp = Date.now();
  const branchName = `master_${timestamp}`;

  createPR(branchName, 'master', phase.title);
}
