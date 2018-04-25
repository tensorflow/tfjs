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

const fakeOctokit: OctokitGetCommit = {
  repos: {
    getCommit: (config: {owner: string, repo: string, sha: string}) => {
      return {data: {author: {login: '1'}}};
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
        sha: 'sha1234'
      }]
    }];

    util.getReleaseNotesDraft(fakeOctokit, repoCommits).then(notes => {
      console.log(notes);
      expect(notes).toEqual([
        '## Core (0.9.0 ==> 0.10.0)', '', '### Features',
        '- Add tf.toPixels. ([#900]' +
            '(https://github.com/tensorflow/tfjs-core/pull/900))'
      ].join('\n'));
      done();
    });
  });
});
