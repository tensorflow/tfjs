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
 *   yarn release-notes
 */

import * as argparse from 'argparse';
import * as fs from 'fs';
import * as util from './util';
import {$, Commit, Repo, RepoCommits} from './util';
// tslint:disable-next-line:no-require-imports
const octokit = require('@octokit/rest')();

const OUT_FILE = 'release-notes.md';

const TFJS_REPOS: Repo[] = [
  {name: 'Core', identifier: 'tfjs', path: 'tfjs-core'},
  {name: 'Data', identifier: 'tfjs', path: 'tfjs-data'},
  {name: 'Layers', identifier: 'tfjs', path: 'tfjs-layers'},
  {name: 'Converter', identifier: 'tfjs', path: 'tfjs-converter'},
  {name: 'Node', identifier: 'tfjs', path: 'tfjs-node'},
  {name: 'Wasm', identifier: 'tfjs', path: 'tfjs-backend-wasm'},
  {name: 'Cpu', identifier: 'tfjs', path: 'tfjs-backend-cpu'},
  {name: 'Webgl', identifier: 'tfjs', path: 'tfjs-backend-webgl'}
];

const VIS_REPO: Repo = {
  name: 'tfjs-vis',
  identifier: 'tfjs-vis',
  path: 'tfjs-vis',
};

const RN_REPO: Repo = {
  name: 'tfjs-react-native',
  identifier: 'tfjs-react-native',
  path: 'tfjs-react-native',
};

async function askUserForVersions(validVersions: string[], packageName: string):
    Promise<{startVersion: string, endVersion: string}> {
  const YELLOW_TERMINAL_COLOR = '\x1b[33m%s\x1b[0m';
  const RED_TERMINAL_COLOR = '\x1b[31m%s\x1b[0m';

  console.log(YELLOW_TERMINAL_COLOR, packageName + ' versions');
  console.log(validVersions.join(', '));
  const startVersion = await util.question(`Enter the start version: `);
  if (validVersions.indexOf(startVersion) === -1) {
    console.log(RED_TERMINAL_COLOR, `Unknown start version: ${startVersion}`);
    process.exit(1);
  }
  const defaultVersion = validVersions[validVersions.length - 1];
  let endVersion = await util.question(
      `Enter the end version (leave empty for ${defaultVersion}): `);
  if (endVersion === '') {
    endVersion = defaultVersion;
  }
  if (validVersions.indexOf(endVersion) === -1) {
    console.log(RED_TERMINAL_COLOR, `Unknown end version: ${endVersion}`);
    process.exit(1);
  }
  return {startVersion, endVersion};
}

function getTaggedVersions(packageName: string) {
  const versions =
      $(`git tag`)
          .split('\n')
          .filter(x => new RegExp('^' + packageName + '-v([0-9])').test(x))
          .map(x => x.substring((packageName + '-v').length));
  return versions;
}

function getTagName(packageName: string, version: string) {
  return packageName + '-v' + version;
}

async function generateTfjsPackageNotes() {
  // Get union start version and end version.
  const identifier = 'tfjs';
  const versions = getTaggedVersions(identifier);
  const {startVersion, endVersion} =
      await askUserForVersions(versions, identifier);
  const startCommit =
      `git rev-list -n 1 ` + getTagName(identifier, startVersion);

  // Populate start and end for each of the tfjs packages.
  TFJS_REPOS.forEach(repo => {
    // Find the version of the dependency from the package.json from the
    // earliest tfjs tag.
    repo.startCommit = startCommit;
    repo.startVersion = startVersion;
    repo.endVersion = endVersion;
  });

  await generateNotes(TFJS_REPOS);
}

async function generateVisNotes() {
  // Get union start version and end version.
  const versions = getTaggedVersions('tfjs-vis');
  const {startVersion, endVersion} =
      await askUserForVersions(versions, 'tfjs-vis');

  // Get tfjs-vis start version and end version.
  VIS_REPO.startVersion = startVersion;
  VIS_REPO.endVersion = endVersion;
  VIS_REPO.startCommit = $(`git rev-list -n 1 ${
      getTagName(VIS_REPO.identifier, VIS_REPO.startVersion)}`);

  await generateNotes([VIS_REPO]);
}


async function generateReactNativeNotes() {
  // Get start version and end version.
  const versions = getTaggedVersions('tfjs-react-native');
  const {startVersion, endVersion} =
      await askUserForVersions(versions, 'tfjs-react-native');

  // Get tfjs-vis start version and end version.
  RN_REPO.startVersion = startVersion;
  RN_REPO.endVersion = endVersion;
  RN_REPO.startCommit = $(`git rev-list -n 1 ${
      getTagName(RN_REPO.identifier, RN_REPO.startVersion)}`);

  await generateNotes([RN_REPO]);
}


async function generateNotes(repositories: util.Repo[]) {
  const repoCommits: RepoCommits[] = [];
  // Clone all of the dependencies into the tmp directory.
  repositories.forEach(repo => {
    console.log(
        `${repo.name}: ${repo.startVersion}` +
        ` =====> ${repo.endVersion}`);

    console.log('Querying commits...');
    // Get subjects, bodies, emails, etc from commit metadata.
    const commitFieldQueries = ['%s', '%b', '%aE', '%H'];
    const commitFields = commitFieldQueries.map(query => {
      // Use a unique delimiter so we can split the log.
      const uniqueDelimiter = '--^^&&';
      const versionQuery = repo.startVersion != null ?
          `${getTagName(repo.identifier, repo.startVersion)}..` +
              `${getTagName(repo.identifier, repo.endVersion)}` :
          `#${repo.startCommit}..${
              getTagName(repo.identifier, repo.endVersion)}`;
      return $(`git log --pretty=format:"${query}${uniqueDelimiter}" ` +
               `${versionQuery}`)
          .trim()
          .split(uniqueDelimiter)
          .slice(0, -1)
          .map(str => str.trim());
    });

    const commits: Commit[] = [];
    for (let i = 0; i < commitFields[0].length; i++) {
      // Make sure the files touched contain the repo directory.
      const filesTouched =
          $(`git show --pretty="format:" --name-only ${commitFields[3][i]}`)
              .split('\n');
      let touchedDir = false;
      for (let j = 0; j < filesTouched.length; j++) {
        if (filesTouched[j].startsWith(repo.path)) {
          touchedDir = true;
          break;
        }
      }
      if (!touchedDir) {
        continue;
      }

      commits.push({
        subject: commitFields[0][i],
        body: commitFields[1][i],
        authorEmail: commitFields[2][i],
        sha: commitFields[3][i]
      });
    }

    repoCommits.push({
      repo,
      startVersion: repo.startVersion,
      endVersion: repo.endVersion,
      startCommit: repo.startCommit,
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

const parser = new argparse.ArgumentParser();

parser.addArgument('--project', {
  help:
      'Which project to generate release notes for. One of union|vis. Defaults to union.',
  defaultValue: 'union',
  choices: ['union', 'vis', 'rn']
});

const args = parser.parseArgs();

if (args.project === 'union') {
  generateTfjsPackageNotes();
} else if (args.project === 'vis') {
  generateVisNotes();
} else if (args.project === 'rn') {
  generateReactNativeNotes();
}
