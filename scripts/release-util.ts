import chalk from 'chalk';
import * as readline from 'readline';
import * as shell from 'shelljs';

export interface Phase {
  // The list of packages that will be updated with this change.
  packages: string[];
  // The list of dependencies that all of the packages will update to.
  deps?: string[];
  // An ordered map of scripts, key is package name, value is an object with two
  // optional fields: `before-yarn` with scripts to run before `yarn`, and
  // `after-yarn` with scripts to run after yarn is called and before the pull
  // request is sent out.
  scripts?: {[key: string]: {[key: string]: string[]}};
  // Whether to leave the version of the package alone. Defaults to false
  // (change the version).
  leaveVersion?: boolean;
  title?: string;
}

export interface ReleaseUnit {
  // A human-readable name. Used for generating release branch.
  name: string;
  // The phases in this release unit.
  phases: Phase[];
  // The repository *only if it is not the same as tfjs*.
  repo?: string;
}

export const CORE_PHASE: Phase = {
  packages: ['tfjs-core']
};

export const LAYERS_CONVERTER_PHASE: Phase = {
  packages: ['tfjs-layers', 'tfjs-converter'],
  deps: ['tfjs-core']
};

export const DATA_PHASE: Phase = {
  packages: ['tfjs-data'],
  deps: ['tfjs-core', 'tfjs-layers']
}

export const UNION_PHASE: Phase = {
  packages: ['tfjs'],
  deps: ['tfjs-core', 'tfjs-layers', 'tfjs-converter', 'tfjs-data']
};

export const NODE_PHASE: Phase = {
  packages: ['tfjs-node', 'tfjs-node-gpu'],
  deps: ['tfjs', 'tfjs-core'],
  scripts: {'tfjs-node-gpu': {'before-yarn': ['yarn prep-gpu']}}
};

export const WASM_PHASE: Phase = {
  packages: ['tfjs-backend-wasm'],
  deps: ['tfjs-core']
};

export const VIS_PHASE: Phase = {
  packages: ['tfjs-vis']
};

export const REACT_NATIVE_PHASE: Phase = {
  packages: ['tfjs-react-native']
};

export const WEBSITE_PHASE: Phase = {
  packages: ['tfjs-website'],
  deps: ['tfjs', 'tfjs-node', 'tfjs-vis', 'tfjs-react-native'],
  scripts: {'tfjs-website': {'after-yarn': ['yarn build-prod']}},
  leaveVersion: true,
  title: 'Update website to latest dependencies.'
};

export const TFJS_RELEASE_UNIT: ReleaseUnit = {
  name: 'tfjs',
  phases: [
    CORE_PHASE, LAYERS_CONVERTER_PHASE, DATA_PHASE, UNION_PHASE, NODE_PHASE,
    WASM_PHASE
  ]
};

export const VIS_RELEASE_UNIT: ReleaseUnit = {
  name: 'vis',
  phases: [VIS_PHASE]
};

export const REACT_NATIVE_RELEASE_UNIT: ReleaseUnit = {
  name: 'react-native',
  phases: [REACT_NATIVE_PHASE]
};

export const WEBSITE_RELEASE_UNIT: ReleaseUnit = {
  name: 'website',
  phases: [WEBSITE_PHASE],
  repo: 'tfjs-website'
};

export const RELEASE_UNITS: ReleaseUnit[] = [
  TFJS_RELEASE_UNIT, VIS_RELEASE_UNIT, REACT_NATIVE_RELEASE_UNIT, WEBSITE_RELEASE_UNIT
];

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

export function printReleaseUnit(id: number) {
  const releaseUnit = RELEASE_UNITS[id];
  console.log(chalk.green(`Release unit ${id}:`));
  console.log(` packages: ${chalk.blue(releaseUnit.phases.map(
    phase => phase.packages.join(', ')).join(', '))}`);
}

export function printPhase(phases: Phase[], phaseId: number) {
  const phase = phases[phaseId];
  console.log(chalk.green(`Phase ${phaseId}:`));
  console.log(`  packages: ${chalk.blue(phase.packages.join(', '))}`);
  if (phase.deps != null) {
    console.log(`   deps: ${phase.deps.join(', ')}`);
  }
}
