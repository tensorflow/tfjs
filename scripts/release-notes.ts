#!/usr/bin/env node
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as commander from 'commander';
import * as shell from 'shelljs';
import * as mkdirp from 'mkdirp';

const TMP_DIR = '/tmp/tfjs-release-notes';

interface Dependency {
  name: string;
  github: string;
  npm: string;
}

const UNION_DEPENDENCIES: Dependency[] = [
  {
    name: 'Core',
    github: 'https://github.com/tensorflow/tfjs-core',
    npm: '@tensorflow/tfjs-core'
  },
  {
    name: 'Layers',
    github: 'https://github.com/tensorflow/tfjs-layers',
    npm: '@tensorflow/tfjs-layers'
  }
];

commander.option('--startVersion <string>', 'Which version of union to use')
    .option('--endVersion <string>', 'Which version of union to use')
    .parse(process.argv);

if (commander.startVersion == null) {
  console.log('Please provide a start version.');
  process.exit(1);
}

const startVersion = 'v' + commander.startVersion;
const endVersion =
    commander.endVersion != null ? 'v' + commander.endVersion : 'HEAD';

mkdirp(TMP_DIR, (err) => {
  if (err) {
    console.log('Error creating temp dir', TMP_DIR);
    process.exit(1);
  }
});

// Remove anything that exists already in the tmp dir.
shell.exec(`rm -r ${TMP_DIR}/*`);

// Get all the commits of the union package between the versions.
const commits = shell.exec(
    `git log --pretty=format:"%H" ${startVersion}..${endVersion}`,
    {silent: true});

const commitLines = commits.stdout.split('\n').filter(line => line !== '');

// Read the union package.json from the earliest commit so we can find the
// dependencies.
const earliestCommit = commitLines[commitLines.length - 1];

const earliestUnionPackageJson = JSON.parse(
    shell.exec(`git show ${earliestCommit}:package.json`, {silent: true})
        .stdout);

// earliestUnionPackageJson.dependencies;
console.log(earliestUnionPackageJson.dependencies);

// Clone all of the dependencies into the tmp directory.
UNION_DEPENDENCIES.forEach(dependency => {
  console.log(`Cloning ${dependency.github}...`);

  const dir = `${TMP_DIR}/${dependency.name}`;
  shell.exec(`mkdir ${dir}`);
  shell.exec(`git clone ${dependency.github} ${dir}`, {silent: true});

  shell.pushd(dir, {silent: true});
  console.log('cwd', process.cwd());
  shell.popd();

  console.log();
});
