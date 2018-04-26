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
// tslint:disable-next-line:no-require-imports
const octokit = require('@octokit/rest')();
import * as readline from 'readline';
import * as fs from 'fs';
import * as util from './util';
import {$, Repo, RepoCommits, Commit} from './util';

const TMP_DIR = '/tmp/tfjs-release-notes';

commander.option('--startVersion <string>', 'Which version of union to use')
    .option('--endVersion <string>', 'Which version of union to use')
    .option('--out <string>', 'Where to write the draft markdown')
    .parse(process.argv);

if (commander.startVersion == null) {
  console.log('Please provide a start version with --startVersion.');
  process.exit(1);
}

if (commander.out == null) {
  console.log('Please provide a file to write the draft to with --out');
  process.exit(1);
}

const UNION_DEPENDENCIES: Repo[] = [
  {name: 'Core', identifier: 'tfjs-core'},
  {name: 'Layers', identifier: 'tfjs-layers'}
];

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
$(`rm -f -r ${TMP_DIR}/*`);

// Get all the commits of the union package between the versions.
const unionCommits =
    $(`git log --pretty=format:"%H" ${startVersion}..${endVersion}`);

const commitLines = unionCommits.trim().split('\n');

// Read the union package.json from the earliest commit so we can find the
// dependencies.
const earliestCommit = commitLines[commitLines.length - 1];
const earliestUnionPackageJson =
    JSON.parse($(`git show ${earliestCommit}:package.json`));
const latestCommit = commitLines[0];
const latestUnionPackageJson =
    JSON.parse($(`git show ${latestCommit}:package.json`));

const repoCommits: RepoCommits[] = [];

// Clone all of the dependencies into the tmp directory.
UNION_DEPENDENCIES.forEach(repo => {
  // Find the version of the dependency from the package.json from the
  // earliest union tag.
  const npm = '@tensorflow/' + repo.identifier;
  const repoStartVersion = earliestUnionPackageJson.dependencies[npm];
  const repoEndVersion = latestUnionPackageJson.dependencies[npm];

  console.log(
      `${repo.name}: ${repoStartVersion}` +
      ` =====> ${repoEndVersion}`);

  const dir = `${TMP_DIR}/${repo.name}`;

  // Clone the repo and find the commit from the tagged start version.
  console.log(`Cloning ${repo.identifier}...`);

  $(`mkdir ${dir}`);
  $(`git clone https://github.com/tensorflow/${repo.identifier} ${dir}`);

  const startCommit = $(`git -C ${dir} rev-list -n 1 v${repoStartVersion}`);

  console.log('Querying commits...');
  // Get subjects, bodies, emails, etc from commit metadata.
  const commitFieldQueries = ['%s', '%b', '%aE', '%H'];
  const commitFields = commitFieldQueries.map(query => {
    // Use a unique delimiter so we can split the log.
    const uniqueDelimiter = '--^^&&';
    return $(`git -C ${dir} log --pretty=format:"${query}${uniqueDelimiter}" ` +
             `v${repoStartVersion}..v${repoEndVersion}`)
        .trim()
        .split(uniqueDelimiter)
        .slice(0, -1)
        .map(str => str.trim());
  });

  const commits: Commit[] = [];
  for (let i = 0; i < commitFields[0].length; i++) {
    commits.push({
      subject: commitFields[0][i],
      body: commitFields[1][i],
      authorEmail: commitFields[2][i],
      sha: commitFields[3][i]
    });
  }

  repoCommits.push({
    repo,
    startVersion: repoStartVersion,
    endVersion: repoEndVersion,
    startCommit,
    commits
  });
});

// Ask for github token.
const rl =
    readline.createInterface({input: process.stdin, output: process.stdout});
rl.question(
    'Enter GitHub token (https://github.com/settings/tokens): ',
    token => writeReleaseNotesDraft(token));

export async function writeReleaseNotesDraft(token: string) {
  octokit.authenticate({type: 'token', token});

  const notes = await util.getReleaseNotesDraft(octokit, repoCommits);

  fs.writeFileSync(commander.out, notes);

  console.log('Done writing notes to', commander.out);

  // So the script doesn't just hang.
  process.exit(0);
}
