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

describe('run_flaky', () => {
  it('exits with zero if the command succeeds', () => {
    expect(exec('node ./scripts/run_flaky.js "echo success"').code).toEqual(0);
  });

  it('exits with one if the command fails', () => {
    expect(exec('node ./scripts/run_flaky.js "exit 1"').code).toEqual(1);
  });

  it('exits with zero if the command eventually succeeds', () => {
    // Try a command that exits with a random number in [0...4] 100 times.
    // Bash's "$RANDOM" does not work in CI tests, so use "node -e" instead.
    // P(this test failing | run_flaky and this test are correct) = (4/5)**100
    expect(exec(
               'node ./scripts/run_flaky.js --times 100 \'node -e "' +
               'process.exit(Math.floor(Math.random() * 5))' +
               '"\'')
               .code)
        .toEqual(0);
  });
});
