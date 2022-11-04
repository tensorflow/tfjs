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

export class PromiseQueue<T> {
  private queue: Array<() => Promise<void>> = [];
  private running = 0;

  constructor(private readonly concurrency = Infinity) {}

  add(f: () => Promise<T>) {
    return new Promise((resolve) => {
      this.queue.push(async () => {
        this.running++;
        try {
          resolve(await f());
        } finally {
          this.running--;
          this.run();
        }
      });
      this.run();
    });
  }

  run() {
    while (this.running < this.concurrency && this.queue.length > 0) {
      this.queue.shift()();
    }
  }
}
