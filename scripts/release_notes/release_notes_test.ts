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

import * as release_notes from './release_notes';
import {RepoCommits, OctokitGetCommit} from './util';
import * as util from './util';
import {release} from 'os';

const fakeCommitContributors = {
  'sha1': 'fakecontributor1',
  'sha2': 'fakecontributor2',
  'sha3': 'fakecontributor3'
};

const fakeOctokit: OctokitGetCommit = {
  repos: {
    getCommit: (config: {owner: string, repo: string, sha: string}) => {
      return {data: {author: {login: fakeCommitContributors[config.sha]}}};
    }
  }
};

describe('getReleaseNotesDraft', () => {
  it('Basic draft written', done => {
    const repoCommits: RepoCommits[] = [{
      repo: {name: 'Core', identifier: 'tfjs-core'},
      startVersion: '0.9.0',
      endVersion: '0.10.0',
      startCommit: 'fakecommit',
      commits: [{
        subject: 'Add tf.toPixels. (#900)',
        body: `
          tf.toPixels is the inverse of tf.fromPixels.
          FEATURE
        `,
        authorEmail: 'test@google.com',
        sha: 'sha1'
      }]
    }];

    util.getReleaseNotesDraft(fakeOctokit, repoCommits).then(notes => {
      expect(notes).toEqual([
        '## Core (0.9.0 ==> 0.10.0)', '', '### Features',
        '- Add tf.toPixels. ([#900]' +
            '(https://github.com/tensorflow/tfjs-core/pull/900)).'
      ].join('\n'));
      done();
    });
  });

  it('Basic draft external contributor thanks them', done => {
    const repoCommits: RepoCommits[] = [{
      repo: {name: 'Core', identifier: 'tfjs-core'},
      startVersion: '0.9.0',
      endVersion: '0.10.0',
      startCommit: 'fakecommit',
      commits: [{
        subject: 'Add tf.toPixels. (#900)',
        body: `
          tf.toPixels is the inverse of tf.fromPixels.
          FEATURE
        `,
        authorEmail: 'test@gmail.com',
        sha: 'sha1'
      }]
    }];

    util.getReleaseNotesDraft(fakeOctokit, repoCommits).then(notes => {
      expect(notes).toEqual([
        '## Core (0.9.0 ==> 0.10.0)', '', '### Features',
        '- Add tf.toPixels. ([#900]' +
            '(https://github.com/tensorflow/tfjs-core/pull/900)).' +
            ' Thanks, @fakecontributor1.'
      ].join('\n'));
      done();
    });
  });

  it('Complex draft', done => {
    const repoCommits: RepoCommits[] = [
      {
        repo: {name: 'Core', identifier: 'tfjs-core'},
        startVersion: '0.9.0',
        endVersion: '0.10.0',
        startCommit: 'fakecommit',
        commits: [
          {
            subject: 'Add tf.toPixels. (#900)',
            body: `
              tf.toPixels is the inverse of tf.fromPixels.
            `,
            authorEmail: 'test@gmail.com',
            sha: 'sha1'
          },
          {
            subject: 'Improvements to matMul. (#901)',
            body: `
              Makes matmul better overall.
              FEATURE Adds transpose bit.
              PERF Improves speed of matMul by 100%.
            `,
            authorEmail: 'test@gmail.com',
            sha: 'sha2'
          }
        ]
      },
      {
        repo: {name: 'Layers', identifier: 'tfjs-layers'},
        startVersion: '0.4.0',
        endVersion: '0.5.1',
        startCommit: 'fakecommit2',
        commits: [{
          subject: 'Change API of layers. (#100)',
          body: `
            tf.toPixels is the inverse of tf.fromPixels.
            BREAKING
            DOC Update docstrings of tf.layers.dense.
            FEATURE Add automatic argument parsing.
          `,
          authorEmail: 'test@gmail.com',
          sha: 'sha1'
        }]
      }
    ];

    util.getReleaseNotesDraft(fakeOctokit, repoCommits).then(notes => {
      expect(notes).toEqual([
        '## Core (0.9.0 ==> 0.10.0)',
        '',
        '### Features',
        '- Adds transpose bit. [Improvements to matMul.] ([#901]' +
            '(https://github.com/tensorflow/tfjs-core/pull/901)).' +
            ' Thanks, @fakecontributor2.',
        '### Performance',
        '- Improves speed of matMul by 100%. [Improvements to matMul.]' +
            ' ([#901](https://github.com/tensorflow/tfjs-core/pull/901)).' +
            ' Thanks, @fakecontributor2.',
        '### Misc',
        '- Add tf.toPixels. ([#900]' +
            '(https://github.com/tensorflow/tfjs-core/pull/900)).' +
            ' Thanks, @fakecontributor1.',
        '',
        '## Layers (0.4.0 ==> 0.5.1)',
        '',
        '### Features',
        '- Add automatic argument parsing. [Change API of layers.] ' +
            '([#100](https://github.com/tensorflow/tfjs-layers/pull/100)).' +
            ' Thanks, @fakecontributor1.',

        '### Breaking changes',
        '- Change API of layers. ' +
            '([#100](https://github.com/tensorflow/tfjs-layers/pull/100)).' +
            ' Thanks, @fakecontributor1.',
        '### Documentation',
        '- Update docstrings of tf.layers.dense. [Change API of layers.] ' +
            '([#100](https://github.com/tensorflow/tfjs-layers/pull/100)).' +
            ' Thanks, @fakecontributor1.',
      ].join('\n'));
      done();
    });
  });
});
