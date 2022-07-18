import * as argparse from 'argparse';
import * as fs from 'fs';

const parser = new argparse.ArgumentParser();

parser.addArgument(
  'json', {help: 'Path to json input file'});

parser.addArgument(
  'out', {help: 'Path to write output'});

const {json, out} = parser.parseArgs() as {
  json: string,
  out: string,
};

const jsonContents = fs.readFileSync(json, 'utf8').replace(/"/g, '\'');
const tsContents = `
/**
 * @license
 * Copyright ${new Date().getFullYear()} Google LLC. All Rights Reserved.
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

import {OpMapper} from '../types';

export const json: OpMapper[] = ${jsonContents};
`;

fs.writeFileSync(out, tsContents);
