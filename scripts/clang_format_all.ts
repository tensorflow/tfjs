/**
 * @license
 * Copyright 2022 Google LLC.
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

import {spawn} from 'child_process';
import {StringDecoder} from 'string_decoder';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import {PromiseQueue} from './promise_queue';

// @ts-ignore because clang-format does not have types.
import {getNativeBinary} from 'clang-format';

const CLANG_FORMAT = getNativeBinary() as string;

const EXCLUDE_DIRECTORY = [
  /^.*node_modules.*$/,
  /^.*dist.*$/,
  /^.*\/\..*$/, // Directories that start with '.'
];
const MATCH_FILE = [
  /^.*\.ts$/,
  /^.*\.tsx$/,
  /^.*\.js$/,
  /^.*\.jsx$/,
  /^.*\.c$/,
  /^.*\.cc$/,
  /^.*\.h$/,
];

function matchesAny(val: string, regexes: RegExp[]): boolean {
  for (const regexp of regexes) {
    if (regexp.test(val)) {
      return true;
    }
  }
  return false;
}

// Filter paths that should be clang-formatted
function filter(path: string, stats: fs.Stats): boolean {
  // Never edit yourself because that makes it difficult to find errors.
  // The sourcemap is wrong.
  if (path.endsWith('clang_format_all.ts')) {
    return false;
  }

  // Check directories separately from files because there are different
  // conditions for each.
  if (stats.isDirectory()) {
    if (!matchesAny(path, EXCLUDE_DIRECTORY)) {
      return true;
    }
  } else if (stats.isFile()) {
    return matchesAny(path, MATCH_FILE);
  }
  return false;
}

// Recursively get the files in a directory. This is essentially `find` with a
// filter.
async function* getFiles(rootPath: string, filter: (path: string, stats: fs.Stats) => boolean = () => true): AsyncGenerator<string> {
  const stats = await fs.promises.lstat(rootPath);
  if (!filter(rootPath, stats)) {
    return;
  }

  if (stats.isDirectory()) {
    const contents = await fs.promises.readdir(rootPath);
    for (const fileName of contents) {
      const filePath = path.join(rootPath, fileName);
      yield * getFiles(filePath, filter);
    }
  } else if (stats.isFile()) {
    // Already filtered above
    yield rootPath;
  }
}

async function readStream<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const outs: T[] = [];
  for await (const data of stream) {
    outs.push(data);
  }
  return outs;
}

// Run clang-format with the given args and collect stdout and stderr.
async function runClang(args: string[]): Promise<string> {
  const clang = spawn(CLANG_FORMAT, args, {
    stdio: 'pipe',
  });

  // stdio comes in Buffers, so it must be decoded.
  const decoder = new StringDecoder('utf8');

  // Read from both the stdout and stderr streams at the same time.
  const stdout = readStream<Buffer>(clang.stdout);
  const stderr = readStream<Buffer>(clang.stderr);
  const [lines, errors] = await Promise.all([stdout, stderr]);

  const decode = decoder.write.bind(decoder);
  if (errors.length > 0) {
    throw new Error(errors.map(decode).join('\n'));
  }
  return lines.map(decode).join('\n');
}

async function* batch<V>(iterator: AsyncIterator<V>, size: number): AsyncGenerator<V[]> {
  let allDone = false;
  while (!allDone) {
    const resultPromises: Array<Promise<IteratorResult<V>>> = [];

    for (let i = 0; i < size; i++) {
      resultPromises.push(iterator.next());
    }

    const results = await Promise.all(resultPromises);

    const values: V[] = [];
    for (const {value, done} of results) {
      if (done) {
        allDone = true;
        if (value != null) {
          values.push(value);
        }
        break;
      }
      values.push(value);
    }

    if (values.length > 0) {
      yield values;
    }
  }
}

async function main() {
  const files = getFiles(path.join(__dirname, '../'), filter);
  const batches = batch(files, 32)

  const threads = os.cpus().length;
  const threadQueue = new PromiseQueue<string>(threads);

  for await (const batch of batches) {
    threadQueue.add(() => {
      return runClang(['-i', '-style=file', ...batch]);
    });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

