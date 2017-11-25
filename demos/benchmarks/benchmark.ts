/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

// Maximum number of time before CPU tests don't execute during the next round.
export const LAST_RUN_CPU_CUTOFF_MS = 5000;

export interface BenchmarkRunGroup {
  name: string;
  // Min and max steps to run the benchmark test over.
  min: number;
  max: number;
  // The size of the step to take between benchmark runs.
  stepSize: number;
  // A transformation of step to the size passed to the benchmark test.
  stepToSizeTransformation?: (step: number) => number;
  // Option parameters which is given to the benchmark test. (e.g. ops types)
  options?: string[];
  selectedOption?: string;
  benchmarkRuns: BenchmarkRun[];
  params: {[option: string]: {}};
}

export class BenchmarkRun {
  name: string;
  benchmarkTest: BenchmarkTest;

  chartData: ChartData[];
  constructor(name: string, benchmarkTest: BenchmarkTest) {
    this.name = name;
    this.benchmarkTest = benchmarkTest;
    this.chartData = [];
  }

  clearChartData() {
    this.chartData = [];
  }
}

export interface BenchmarkTest {
  run(size: number, opType?: string, params?: {}): Promise<number>;
}
