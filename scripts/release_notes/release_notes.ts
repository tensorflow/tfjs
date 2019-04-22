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

/**
 * Generates a draft release notes markdown file for a release. This script
 * takes a start version of the union package, and optionally an end version.
 * It then finds the matching versions for all the dependency packages, and
 * finds all commit messages between those versions.
 *
 * The release notes are grouped by repository, and then bucketed by a set of
 * tags which committers can use to organize commits into sections. See
 * DEVELOPMENT.md for more details on the available tags.
 *
 * This script will ask for your github token which is used to make requests
 * to the github API for usernames for commits as this is not stored in git
 * logs. You can generate a token for your account here;
 * https://github.com/settings/tokens
 *
 * Usage:
 *   # Release notes for all commits after tfjs union version 0.9.0.
 *   yarn release-notes --startVersion 0.9.0 --out ./draft_notes.md
 *
 *   # Release notes for all commits after version 0.9.0 up to and including
 *   # version 0.10.0.
 *   yarn release-notes --startVersion 0.9.0 --endVersion 0.10.3 \
 *       --out ./draft_notes.md
 */

import * as commander from 'commander';
import * as mkdirp from 'mkdirp';
import * as readline from 'readline';
import * as fs from 'fs';
import * as util from './util';
import {$, Commit, Repo, RepoCommits} from './util';
// tslint:disable-next-line:no-require-imports
const octokit = require('@octokit/rest')();

const OUT_FILE = 'release-notes.md';
const TMP_DIR = '/tmp/tfjs-release-notes';

const UNION_DEPENDENCIES: Repo[] = [
  {name: 'Core', identifier: 'tfjs-core'},
  {name: 'Data', identifier: 'tfjs-data'},
  {name: 'Layers', identifier: 'tfjs-layers'},
  {name: 'Converter', identifier: 'tfjs-converter'}
];

async function main() {
  const versions = $(`git tag`).split('\n');
  versions.push('HEAD');
  console.log('\x1b[33m%s\x1b[0m', 'tfjs versions');
  console.log(versions.join(', '));
  const startVersion = await util.question(`Enter the union start version: `);
  if (versions.indexOf(startVersion) === -1) {
    console.log('\x1b[31m%s\x1b[0m', `Unknown start version: ${startVersion}`);
    process.exit(1);
  }
  let endVersion = await util.question(
      `Enter the union end version (leave empty for HEAD): `);
  if (endVersion === '') {
    endVersion = 'HEAD';
  }
  if (versions.indexOf(endVersion) === -1) {
    console.log('\x1b[31m%s\x1b[0m', `Unknown end version: ${endVersion}`);
    process.exit(1);
  }

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

    const startCommit =
        $(repoStartVersion != null ?
              `git -C ${dir} rev-list -n 1 v${repoStartVersion}` :
              // Get the first commit if there are no tags yet.
              `git rev-list --max-parents=0 HEAD`);

    console.log('Querying commits...');
    // Get subjects, bodies, emails, etc from commit metadata.
    const commitFieldQueries = ['%s', '%b', '%aE', '%H'];
    const commitFields = commitFieldQueries.map(query => {
      // Use a unique delimiter so we can split the log.
      const uniqueDelimiter = '--^^&&';
      const versionQuery = repoStartVersion != null ?
          `v${repoStartVersion}..v${repoEndVersion}` :
          `#${startCommit}..v${repoEndVersion}`;
      return $(`git -C ${dir} log --pretty=format:"${query}${
                   uniqueDelimiter}" ` +
               `${versionQuery}`)
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
  const token = await util.question(
      'Enter GitHub token (https://github.com/settings/tokens): ');
  octokit.authenticate({type: 'token', token});

  const notes = await util.getReleaseNotesDraft(octokit, repoCommits);

  fs.writeFileSync(OUT_FILE, notes);

  console.log('Done writing notes to', OUT_FILE);

  // So the script doesn't just hang.
  process.exit(0);
}
main();
