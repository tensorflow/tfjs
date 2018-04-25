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

/**
 * A wrapper around shell.exec for readability.
 * @param cmd The bash command to execute.
 */
export function $(cmd) {
  const result = shell.exec(cmd, {silent: true});
  if (result.code > 0) {
    console.log('$', cmd);
    console.log(result.stderr);
    process.exit(1);
  }
  return result.stdout.trim();
}

export interface Repo {
  name: string;
  identifier: string;
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
          data: {author: {login: string}}
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
  {section: 'Bug fixes', tag: 'BUG'},
  {section: 'Performance improvements', tag: 'PERF'},
  {section: 'Development', tag: 'DEV'}, {section: 'Documentation', tag: 'DOC'}
];

export async function getReleaseNotesDraft(
    octokit: OctokitGetCommit, repoCommits: RepoCommits[]): Promise<string> {
  const repoNotes = [];
  for (let i = 0; i < repoCommits.length; i++) {
    const repoCommit = repoCommits[i];

    const getUsernameForCommit = async sha => {
      const result = await octokit.repos.getCommit(
          {owner: 'tensorflow', repo: repoCommit.repo.identifier, sha});
      return result.data.author.login;
    };

    const tagEntries: {[tag: string]: string[]} = {};
    SECTION_TAGS.forEach(({tag}) => tagEntries[tag] = []);

    const notes = [];
    for (let j = 0; j < repoCommit.commits.length; j++) {
      const commit = repoCommit.commits[j];

      // Get tags for the body.
      SECTION_TAGS.forEach(tag => {
        const re = new RegExp(`^${tag.tag}`, 'gm');
        console.log(commit.body.match(re));
      });

      const isExternalContributor = !commit.authorEmail.endsWith('@google.com');

      // Replace pull numbers will fully qualified path.
      const subject = commit.subject.replace(
          /\(#([0-9]+)\)/,
          `([#$1](https://github.com/tensorflow/` +
              `${repoCommit.repo.identifier}/pull/$1))`);
      let entry = '- ' + subject;

      if (isExternalContributor) {
        const username = await getUsernameForCommit(commit.sha);

        entry += (!entry.endsWith('.') ? '.' : '') + ` Thanks @${username}.`;
      }

      const trimmedBody = commit.body.trim();
      if (trimmedBody !== '') {
        entry += '\n\n' +
            trimmedBody.split('\n').map(line => '> ' + line).join('\n');
      }

      notes.push(entry);
    }

    const repoSection = `## ${repoCommit.repo.name} ` +
        `(${repoCommit.startVersion} ==> ` +
        `${repoCommit.endVersion})\n` + notes.join('\n');
    repoNotes.push(repoSection);
  }
  return repoNotes.join('\n\n');
}
