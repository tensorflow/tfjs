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

// Not using argparse because the script needs to concatenate the rest of the
// arguments after the command into arguments used for the command (instead of
// for this script), including --dashed arguments. Argparse would treat dashed
// args as part of the 'run_flaky.js' command instead of as part of the command
// to be run.
const args = [...process.argv];
args.shift(); // remove node binary arg
args.shift(); // remove this command

if (args.length === 0 || args[0] === '-h') {
  console.log('usage: run_flaky.js [-h] [--times [TIMES]] command [args ...]');
  process.exit(0);
}

let times = 3;
if (args[0] === '--times') {
  args.shift();
  times = parseInt(args[0]);
  if (isNaN(times)) {
    throw new Error(`'--times' must be a number but got '${args[0]}'`);
  }
  if (times === 0) {
    throw new Error(`Flaky test asked to run zero times: '${command}'`);
  }
  args.shift();
}

const command = args.join(' ');

console.log(`Running flaky test ${times} times`);
console.log(`Command: '${command}'`);
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
