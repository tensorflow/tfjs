/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * A helper script to generate empty files typically used in modularizing an op.
 * Takes two params:
 *    --op the name of the op
 *    --kernel the name of the kernel (this is optional)
 *    --chained is this op part of the chained api (optional)
 *
 * It assumes you run it from tfjs_core.
 *
 * Example
 *  npx ts-node -s scripts/touch_modular_op_files.ts --op "op_name" --kernel \
 *    "KernelName" --chained
 *
 * Generates the following files (they will be empty)
 *    tfjs_core/src/ops/op_name.ts
 *
 *  if --chained is present
 *    tfjs_core/src/public/chained_ops/op_name.ts
 *
 *  if --kernel is present
 *    tfjs_core/src/gradients/KernelName_grad.ts
 */

import * as argparse from 'argparse';
import {execSync} from 'child_process';
import * as fs from 'fs';

const parser = new argparse.ArgumentParser();

parser.addArgument('--op', {help: 'the name of the op'});
parser.addArgument('--grad', {help: 'the name of the grad.'});
parser.addArgument('--chained', {
  action: 'storeTrue',
  defaultValue: false,
  help: 'is this op part of the chained api.'
});

async function main() {
  const args = parser.parseArgs();
  console.log('Called touch_modular_op_files with args:', args);

  if (args.op != null) {
    let filePath = `./src/ops/${args.op}.ts`;
    let command = `touch ${filePath}`;
    execSync(command);

    // create a test file
    filePath = `./src/ops/${args.op}_test.ts`;
    command = `touch ${filePath}`;
    execSync(command);

    if (args.chained) {
      filePath = `./src/public/chained_ops/${args.op}.ts`;
      command = `touch ${filePath}`;
      execSync(command);
    }
  }

  if (args.grad) {
    const downcaseFirstChar = (str: string) => {
      return str.charAt(0).toLowerCase() + str.slice(1);
    };

    const gradientFileTemplate = `/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {KernelName, KernelNameAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';

export const ${downcaseFirstChar(args.grad)}GradConfig: GradConfig = {
  kernelName: KernelName,
  inputsToSave: [], // UPDATE ME
  outputsToSave: [], // UPDATE ME
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [] = saved;
    const {} = attrs as {} as KernelNameAttrs;
    return {
    };
  }
};
`;
    const filePath = `./src/gradients/${args.grad}_grad.ts`;
    fs.writeFileSync(filePath, gradientFileTemplate, {flag: 'a'});
  }
}

main();
