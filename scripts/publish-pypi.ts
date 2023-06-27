// publish-pypi.ts
import * as argparse from 'argparse';
import { checkoutReleaseBranch, question, $ } from './release-util';
import * as shell from 'shelljs';
import * as fs from 'fs';
import chalk from 'chalk';

const TMP_DIR = '/tmp/tfjs-pypi';

const parser = new argparse.ArgumentParser();
parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocol rather than the http protocol when cloning repos.',
});

async function getNewlyCreatedBranches(): Promise<string[]> {
  const branchesStr = $(`git branch -r --sort=-authordate --format='%(HEAD) %(refname:lstrip=-1)'`);

  const branches = branchesStr.split('\n').map((line: string) => line.trim());

  const pattern = /^tfjs_\d+\.\d+\.\d+.*$/;
  const tfjsBranches = branches.filter((branch: string) => pattern.test(branch));

  return tfjsBranches;
}

async function main() {
  const args = parser.parseArgs();

  try {
    const branches = await getNewlyCreatedBranches();
    console.log('Branches:', branches);

    const latestBranch = branches[0];
    console.log('Latest Branch:', latestBranch);

    const answer = await question(chalk.cyan.bold(`Is this the right branch '${latestBranch}' you are looking for? (y/N): `));

    if (answer.toLowerCase() === 'y') {
      checkoutReleaseBranch(latestBranch, args.git_protocol, TMP_DIR);
      const targetDir = `${TMP_DIR}/tfjs-converter/python`;
      shell.cd(targetDir);
      console.log(chalk.blue.bold('Current directory:', shell.pwd().toString()));
      
      $('bazel clean');
      $('bazel build python3_wheel');

      // Remove dist folder if it exists
      if (fs.existsSync('./dist')) {
        fs.rmdirSync('./dist', { recursive: true });
      }

      $('./build-pip-package.sh --upload ./dist');

      // Command executed successfully
      console.log('Command executed successfully');
    } else {
      console.log('Aborted.');
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

// Invoke the main function
main();
