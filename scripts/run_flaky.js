const {ArgumentParser} = require('argparse');
const shell = require('shelljs');


function exec(command) {
  return new Promise((resolve, reject) => {
    shell.exec(command, (error, stdout, stderr) => {
      if (error) {
        reject(error);
      }
      if (stderr) {
        reject(stderr);
      }
      resolve(stdout);
    });
  });
}

async function main(command, times) {
  const errors = [];
  for (let i = 0; i < times; i++) {
    try {
      await exec(command);
      return errors;
    } catch (e) {
      errors.push(e);
    }
  }
  throw new Error();
}

const parser = new ArgumentParser(
    {description: 'Run a flaky test a number of times or until it passes'});

parser.addArgument('command', {help: 'Flaky command to run'})
parser.addArgument('times', {
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

main(command, times)
    .then(errors => {
      if (errors.length > 0) {
        console.warn(`Flaky test failed ${errors.length} times before passing.`)
        console.warn(`Command: '${command}'`);
      }
    })
    .catch(() => {
      console.error(`Flaky test failed ${times} times`);
      console.warn(`Command: '${command}'`);
      process.exit(1);
    });
