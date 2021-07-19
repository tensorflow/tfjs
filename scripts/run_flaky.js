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

const {ArgumentParser} = require('argparse');
const {exec} = require('shelljs');


function main(args) {
  const command = args['command'];
  const times = parseInt(args['times']);

  if (times === 0) {
    throw new Error(`Flaky test asked to run zero times: '${command}'`);
  }

  console.log(`Running flaky test at most ${times} times`);
  console.log(`Command: '${command}'`);
  const exitCodes = [];
  for (let i = 0; i < times; i++) {
    console.log(`Flaky run ${i + 1} of a potential ${times} for '${command}'`);
    const exitCode = exec(command).code;
    exitCodes.push(exitCode);
    if (exitCode === 0) {
      break;
    }
  }

  const success = exitCodes[exitCodes.length - 1] === 0;
  if (!success) {
    console.error(`Flaky test failed ${times} times`);
  } else if (exitCodes.length > 1) {
    console.warn(
        `Flaky test failed ${exitCodes.length - 1} times before passing.`);
  } else {
    // Test passed with no fails
    console.log('Flaky test passed on the first run');
  }
  console.error(`Command: '${command}'`);
  console.error(`Exit codes: ${exitCodes}`);

  if (!success) {
    process.exit(1);
  }
}

const parser = new ArgumentParser(
    {description: 'Run a flaky test a number of times or until it passes'});

parser.addArgument('command', {help: 'Flaky command to run'})
parser.addArgument('--times', {
  help: 'Maximum number of times to run the command',
  defaultValue: 3,
  nargs: '?',
  type: 'int',
});

main(parser.parseArgs());
