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

import * as shell from 'shelljs';
import * as readline from 'readline';

const GOOGLERS_WITH_GMAIL = [
  'dsmilkov',
  'kainino0x',
  'davidsoergel',
  'pyu10055',
  'nkreeger',
  'tafsiri',
  'annxingyuan',
  'Kangyi Zhang',
  'lina128',
  'mattsoulanille',
  'jinjingforever',
];

const rl =
    readline.createInterface({input: process.stdin, output: process.stdout});

/**
 * A wrapper around shell.exec for readability.
 * @param cmd The bash command to execute.
 * @returns stdout returned by the executed bash script.
 */
export function $(cmd: string) {
  const result = shell.exec(cmd, {silent: true});
  if (result.code > 0) {
    console.log('$', cmd);
    console.log(result.stderr);
    process.exit(1);
  }
  return result.stdout.trim();
}

export async function question(questionStr: string): Promise<string> {
  return new Promise<string>(
      resolve => rl.question(questionStr, response => resolve(response)));
}

export interface Repo {
  name: string;
  identifier: string;
  path: string;
  startVersion?: string;
  startCommit?: string;
  endVersion?: string;
}

export interface RepoCommits {
  repo: Repo;
  startVersion: string;
  endVersion: string;
  startCommit: string;
  commits: Commit[];
}

export interface Commit {
  subject: string;
  body: string;
  authorEmail: string;
  sha: string;
}

export interface OctokitGetCommit {
  repos: {
    getCommit:
        (config: {owner: string, repo: string, sha: string}) => {
          data: {
            commit: {
              author: {
                name: string,
                email: string,
              }
            },
            author: {login: string},
          }
        }
  };
}

interface SectionTag {
  section: string;
  tag: string;
}

const SECTION_TAGS: SectionTag[] = [
  {section: 'Features', tag: 'FEATURE'},
  {section: 'Breaking changes', tag: 'BREAKING'},
  {section: 'Bug fixes', tag: 'BUG'}, {section: 'Performance', tag: 'PERF'},
  {section: 'Development', tag: 'DEV'}, {section: 'Documentation', tag: 'DOC'},
  {section: 'Security', tag: 'SECURITY'}, {section: 'Misc', tag: 'MISC'},
  {section: 'Internal', tag: 'INTERNAL'}
];

/**
 * Assembles the release note drafts from a set of commits.
 *
 * @param octokit An authenticated octokit object (to make github API
 *     requests).
 * It only needs to satisfy the OctokitGetCommit interface which gets commit
 * metadata.
 * @param repoCommits An object representing the metadata for commits to
 * assmemble into release notes.
 * @returns The release notes markdown draft as a string.
 */
export async function getReleaseNotesDraft(
    octokit: OctokitGetCommit, repoCommits: RepoCommits[]): Promise<string> {
  const repoNotes = [];
  for (let i = 0; i < repoCommits.length; i++) {
    const repoCommit = repoCommits[i];

    const getUsernameForCommit = async (sha: string) => {
      let result;
      try {
        result = await octokit.repos.getCommit(
            {owner: 'tensorflow', repo: 'tfjs', sha});
        return result.data.author.login;
      } catch (e) {
        console.log(`Error fetching username for commit ${sha}`);
        console.log(`Using ${result.data.commit.author.name}`);
        return result.data.commit.author.name;
      }
    };

    const tagEntries: {[tag: string]: string[]} = {};
    SECTION_TAGS.forEach(({tag}) => tagEntries[tag] = []);

    for (let j = 0; j < repoCommit.commits.length; j++) {
      const commit = repoCommit.commits[j];

      const tagsFound: Array<{tag: string, tagMessage: string}> = [];
      const bodyLines = commit.body.split('\n').map(line => line.trim());
      // Get tags for the body by finding lines that start with tags. Do
      // this without a regex for readability.
      SECTION_TAGS.forEach(({tag}) => {
        if (tag === 'INTERNAL') {
          return;
        }

        bodyLines.forEach(line => {
          // Split by word boundaries, and make sure the first word is the
          // tag.
          const split = line.split(/\b/);
          if (split[0] === tag) {
            const tagMessage = line.substring(tag.length).trim();
            tagsFound.push({tag, tagMessage});
          }
        });
      });
      // If no explicit tags, put this under misc.
      if (tagsFound.length === 0) {
        tagsFound.push({tag: 'MISC', tagMessage: ''});
      }

      const username = await getUsernameForCommit(commit.sha);
      const isExternalContributor =
          !commit.authorEmail.endsWith('@google.com') &&
          GOOGLERS_WITH_GMAIL.indexOf(username) === -1;

      const pullRequestRegexp = /\(#([0-9]+)\)/;
      const pullRequestMatch = commit.subject.match(pullRequestRegexp);

      let subject = commit.subject;
      let pullRequestNumber = null;
      if (pullRequestMatch != null) {
        subject = subject.replace(pullRequestRegexp, '').trim();
        pullRequestNumber = pullRequestMatch[1];
      }

      for (let k = 0; k < tagsFound.length; k++) {
        const {tag, tagMessage} = tagsFound[k];

        // When the tag has no message, use the subject.
        let entry;
        if (tagMessage === '') {
          entry = '- ' + subject;
        } else {
          entry = '- ' + tagMessage + ' [' + subject + ']';
        }

        // Attach the link to the pull request.
        const pullRequestSuffix = pullRequestNumber != null ?
            ` ([#${pullRequestNumber}]` +
                `(https://github.com/tensorflow/tfjs/pull/${
                    pullRequestNumber})).` :
            '';

        entry = entry.trim() + pullRequestSuffix;

        // For external contributors, we need to query github because git
        // does not contain github username metadatea.
        if (isExternalContributor) {
          entry += (!entry.endsWith('.') ? '.' : '') + ` Thanks, @${username}.`;
        }

        tagEntries[tag].push(entry);
      }
    }

    const repoLines: string[] = [];
    SECTION_TAGS.forEach(({tag, section}) => {
      if (tagEntries[tag].length !== 0) {
        const sectionNotes = tagEntries[tag].join('\n');
        repoLines.push(`### ${section}`);
        repoLines.push(sectionNotes);
      }
    });

    const repoSection = `## ${repoCommit.repo.name} ` +
        `(${repoCommit.startVersion} ==> ` +
        `${repoCommit.endVersion})\n\n` + repoLines.join('\n');
    repoNotes.push(repoSection);
  }

  return repoNotes.join('\n\n');
}
