/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as firebase from './firebase';
import {BenchmarkLog} from './types';

function nextTick(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve));
}

// tslint:disable-next-line:no-any
export async function benchmarkAndLog<T extends any>(
    name: string, benchmark: (size: T) => Promise<number>, sizes: T[],
    sizeToParams: (size: T) => string, runCount = 100): Promise<void> {
  const logs: BenchmarkLog[] = [];

  for (let i = 0; i < sizes.length; i++) {
    const size = sizes[i];
    let averageTimeMs = 0;
    for (let j = 0; j < runCount; j++) {
      const result = await benchmark(size);
      averageTimeMs += result / runCount;
      await nextTick();
    }
    const benchmarkLog:
        BenchmarkLog = {params: sizeToParams(size), averageTimeMs};
    logs.push(benchmarkLog);
  }
  await firebase.logBenchmarkRun(name, logs);
}
