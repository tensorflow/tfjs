/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export interface BenchmarkTest {
  run(size: number, opType?: string, params?: {}): Promise<number>;
}

export interface BenchmarkLog {
  averageTimeMs: number;
  params: string;
}

export interface HardwareInfo {
  /** `inxi` output. */
  cpuInfo?: string;

  /** Processed `free` output. */
  memInfo?: string;

  /** Processed `nvidia-smi` output. */
  cudaGPUInfo?: string;
}

export type BenchmarkBackend =
     'browser' | 'node-libtensorflow-cpu' | 'node-libtensorflow-cuda'
     | 'node-gles' | 'python-tensorflow-cpu' | 'python-tensorflow-cuda';

/** Information about a benchmark backend environment. */
export interface BenchmarkBackendInfo {
  benchmarkBackend: BenchmarkBackend;
}

/** Metadata specific to the browser environment. */
export interface BrowserBenchmarkBackendInfo extends BenchmarkBackendInfo {  
  userAgent: string;
}

export interface ServerSideBenchmarkBackendInfo extends BenchmarkBackendInfo {
  /**
   * Metadata for the node environment.
   * `uname -a` output.
   */
  systemInfo?: string;

  hardwareInfo?: HardwareInfo;
}

/** Metadata specific to the Node.js environment. */
export interface NodeBenchmarkBackendInfo
    extends ServerSideBenchmarkBackendInfo {
  nodeVersion?: string;

  tfjsNodeVersion?: string;
  tfjsNodeUsesCUDA?: boolean;
  cudaVersion?: string;
}

/** Metadata specific to the Python environment. */
export interface PythonBenchmarkBackendInfo
    extends ServerSideBenchmarkBackendInfo {
  pythonVersion?: string;

  tensorflowVersion?: string;
  kerasVersion?: string;
  tensorflowUsesCUDA?: boolean;
  cudaVersion?: string;
}

/**
 * The log from an individual benchmark task.
 * 
 * This is the generic interface for benchmark tasks. It is applicable to
 * - model-related benchmark tasks, e.g., the predict() call of a certain
 *   model, 
 * - benchmark tasks unrelated to a model, e.g., creation of a tensor with
 *   a given dtype and shape on the CUDA GPU.
 * 
 * Note that the task involved in a TaskLog should be indivisible. For example,
 * a `TaskLog` should be about only the `predict()` call of a certain model
 * with a given batch size. If the model is benchmarked for both its
 * `predict()` call and its `fit()` call, the results should be represented
 * by two `TaskLog` objects. Similarly, if the `predict()` method of a model
 * is benchmarked under different batch sizes or sequence lengths, the results
 * should be represented by different `TaskLog` objects.
 */
export interface TaskLog {
  /** Name of the task. */
  taskName: string;

  /**
   * Number of individual runs that are timed.
   *
   * - For predict() and non-model-based tasks, this is the number of function
   *   calls.
   * - For fit() and fitDatset(), this is the number of epochs of a single call.
   */
  numBenchmarkedRuns: number;

  /**
   * Number of burn-in (i.e., warm-up) runs that take place prior to the
   * benchmarked runs.
   *
   * Follows the same convention as `numBenchmarkedRuns` for counting runs.
   */
  numBurnInRuns: number;

  /** A timestamp (epoch time) for the benchmark task. */
  timestamp: number;

  /** Arithmetic mean of the wall times from the benchmarked runs. */
  averageTimeMs: number;

  /** Median of the wall times from the benchmarked runs.*/
  medianTimeMs?: number;

  /** Minimum of the wall times from the benchmarked runs.*/
  minTimeMs?: number;
}

/** Type of the model-related benchmark task. */
export type ModelTask = 'predict'|'fit'|'fitDataset';

/**
 * TODO(cais):
 */
export interface ModelTaskLog extends TaskLog {
  /** Name of the model. */
  modelName: string;

  /**
   * This overrides the `taskName` field of the base interface to be
   * model-specific.
   */
  taskName: ModelTask;

  /** Batch size used for the model benchmark task. */
  batchSize: number;
}

export type MultiBackendLog = {[benchmarkBackend in BenchmarkBackend]: TaskLog};

export type MultiModelLog = {[modelTask in ModelTask]: MultiBackendLog};

export type BenchmarkHistory = {[datetime: string]: MultiModelLog};
