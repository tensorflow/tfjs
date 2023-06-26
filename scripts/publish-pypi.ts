import * as argparse from 'argparse';
import { checkoutReleaseBranch, question } from './release-util';
const TMP_DIR = '/tmp/tfjs-pypi';
import * as shell from 'shelljs';
import { execSync } from 'child_process';
import chalk from 'chalk';

const parser = new argparse.ArgumentParser();
parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocol rather than the http protocol when cloning repos.',
});
parser.addArgument(['--dry'], {
  action: 'storeTrue',
  help: 'Dry run. Stage all packages in verdaccio but do not publish them to the registry.',
});

async function getNewlyCreatedBranches(): Promise<string[]> {
  const branchesStr = execSync(
    `git branch -r --sort=-authordate --format='%(HEAD) %(refname:lstrip=-1)'`,
  ).toString();

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

    const answer = await question(chalk.cyan.bold(`Is this the right branch '${latestBranch}' you are looking for? (y/n): `));

    if (answer.toLowerCase() === 'y') {
      checkoutReleaseBranch(latestBranch, args.git_protocol, TMP_DIR);
      const targetDir = `${TMP_DIR}/tfjs-converter/python`;
      shell.cd(targetDir);
      console.log(chalk.blue.bold('Current directory:', shell.pwd().toString()));
    } else {
      console.log('Aborted.');
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

// Invoke the main function
main();
