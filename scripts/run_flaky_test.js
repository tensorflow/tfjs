// Copyright 2021 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

const {exec} = require('shelljs');
const fs = require('fs');

const {FILE_NAME} = require('./run_flaky_test_helper.js');

describe('run_flaky', () => {
  it('exits with zero if the command succeeds', () => {
    expect(exec('node ./scripts/run_flaky.js "echo success"').code).toEqual(0);
  });

  it('exits with one if the command fails', () => {
    expect(exec('node ./scripts/run_flaky.js "exit 1"').code).toEqual(1);
  });

  it('exits with zero if the command eventually succeeds', () => {
    // Remove test file that was not cleaned up from a prior run.
    if (fs.existsSync(FILE_NAME)) {
      fs.unlinkSync(FILE_NAME);
    }

    // This command should fail once and then succeed the next run.
    expect(exec(
               'node ./scripts/run_flaky.js --times 2' +
               ' \'node ./scripts/run_flaky_test_helper.js\'')
               .code)
        .toEqual(0);

    // Clean up test file
    fs.unlinkSync(FILE_NAME);
  });
});
