import chalk from 'chalk';
import * as readline from 'readline';
import * as shell from 'shelljs';

const rl =
  readline.createInterface({input: process.stdin, output: process.stdout});

export async function question(questionStr: string): Promise<string> {
  console.log(chalk.bold(questionStr));
  return new Promise<string>(
    resolve => rl.question('> ', response => resolve(response)));
}

/**
 * A wrapper around shell.exec for readability.
 * @param cmd The bash command to execute.
 * @returns stdout returned by the executed bash script.
 */
export function $(cmd: string) {
  const result = shell.exec(cmd, {silent: true});
  if (result.code > 0) {
    console.log('$', cmd);
    console.log(result.stderr);
    process.exit(1);
  }
  return result.stdout.trim();
}
