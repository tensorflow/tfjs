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
// @ts-ignore because clang-format does not have types.
import {getNativeBinary} from 'clang-format';
import * as glob from 'glob';
import * as os from 'os';
import {StringDecoder} from 'string_decoder';

import {PromiseQueue} from './promise_queue';

const CLANG_FORMAT = getNativeBinary() as string;

function globAsync(pattern: string, options: glob.IOptions): Promise<string[]> {
  return new Promise((resolve, reject) => {
    glob.default(pattern, options, (err, matches) => {
      if (err) {
        reject(err);
        return;
      }
      resolve(matches);
    });
  });
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

function* batch<V>(iterable: Iterable<V>, size: number): Generator<V[]> {
  let values: V[] = [];
  for (let val of iterable) {
    values.push(val);
    if (values.length === size) {
      yield values;
      values = [];
    }
  }

  // Make sure to yield the last one
  if (values.length > 0) {
    yield values;
  }
}

async function main() {
  const files = await globAsync('**/*.@(t|j)s?(x)', {
    ignore: [
      '**/node_modules/**', '**/dist/**',
      '**/*_pb.js',  // Compiled protobuf files
      '**.**',       // Files that start with '.'
    ]
  });
  const batches = batch(files, 32);

  const threads = os.cpus().length;
  const threadQueue = new PromiseQueue<string>(threads);

  for (const batch of batches) {
    threadQueue.add(() => {
      return runClang(['-i', '-style=file', ...batch]);
    });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
