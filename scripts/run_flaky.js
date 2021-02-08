const {ArgumentParser} = require('argparse');
const shell = require('shelljs');


function exec(command) {
  return new Promise((resolve) => {
    shell.exec(command, (exitCode) => {
      resolve(exitCode);
    });
  });
}

async function main(command, times) {
  const exitCodes = [];
  for (let i = 0; i < times; i++) {
    console.log(`Flaky run ${i + 1} of a potential ${times} for '${command}'`);
    const exitCode = await exec(command);
    exitCodes.push(exitCode)
    if (exitCode === 0) {
      break;
    }
  }
  return exitCodes;
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

const args = parser.parseArgs();
const command = args['command'];
const times = parseInt(args['times']);

if (times === 0) {
  throw new Error(`Flaky test asked to run zero times: '${command}'`);
}

console.log(`Running flaky test at most ${times} times`);
console.log(`Command: '${command}'`);
main(command, times).then(exitCodes => {
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
});
