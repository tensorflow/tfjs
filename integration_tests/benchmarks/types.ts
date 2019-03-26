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

/**
 * =======================================================
 * Old types used by the tjfs-core benchmarks.
 * =======================================================
 */

export interface BenchmarkTest {
  run(size: number, opType?: string, params?: {}): Promise<number>;
}

export interface BenchmarkModelTest extends BenchmarkTest {
  loadModel(): void;
}

export interface BenchmarkLog {
  averageTimeMs: number;
  params: string;
}

/**
 * =======================================================
 * New types to be used by tfjs-layers and tfjs-node
 * benchmarks.
 *
 * We plan to migrate the tfjs-core benchmarks to the new
 * type system as well.
 * =======================================================
 */

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
  numWarmUpRuns: number;

  /** A timestamp (epoch time) for the benchmark task. */
  timestamp: number;

  /**
   * Arithmetic mean of the wall times from the benchmarked runs.
   *
   * This is the primary metric displayed by the dashboard. Use caution
   * when making change to this field.
   */
  averageTimeMs: number;

  /** Median of the wall times from the benchmarked runs. */
  medianTimeMs?: number;

  /** Minimum of the wall times from the benchmarked runs. */
  minTimeMs?: number;

  environment: EnvironmentInfo;
}

/** Type of the model-related benchmark task. */
export type ModelTask = 'predict'|'fit'|'fitDataset';

/** The logs from benchmarking a specific model-related task. */
export interface ModelTaskLog extends TaskLog {
  /** Name of the model. */
  modelName: string;

  /**
   * A description of the model. 
   * 
   * Usually more detailed tha `modelName`.
   */
  modelDescription?: string;

  /**
   * This overrides the `taskName` field of the base interface to be
   * model-specific.
   */
  taskName: ModelTask;

  /** Batch size used for the model benchmark task. */
  batchSize: number;
}

/**
 * Log from benchmarking the same task in different environments.
 *
 * E.g., benchmarking the predict() call of MobileNetV2 on
 * node-libtensorflow-cpu and python-tensorflow-cpu.
 */
export type MultiEnvironmentTaskLog = {
  [environmentType in BenchmarkEnvironmentType]?: TaskLog
};

/**
 * Log from benchmarking a number of tasks, each running on a number of
 * environments.
 *
 * E.g., benchmarking the predict() and fit() call of MobileNetV2,
 * each of which is benchmarked on node-libtensorflow-cpu and
 * python-tensorflow-cpu.
 */
export type TaskGroupLog = {
  [task: string]: MultiEnvironmentTaskLog
};

/**
 * Log from a suite of task groups.
 *
 * The `data` field poitns to a collection of task groups (e.g., a collection
 * of models), each of which involves multiple tasks. Each task may involve
 * multiple environments.
 */
export interface SuiteLog {
  data: {[taskGroupName: string]: TaskGroupLog};

  metadata: BenchmarkMetadata;
}

/** Benchmark logs from multiple days. */
export type BenchmarkHistory = {
  [datetime: string]: TaskGroupLog
};

export type CodeRepository =
    'tfjs'|'tfjs-converter'|'tfjs-core'|'tfjs-data'|'tfjs-layers'|'tfjs-node';

/**
 * The git hash code of the related TensorFlow.js code repositories.
 */
export type CommitHashes = {[repo in CodeRepository]?: string};

/**
 * Metadata for a run of a benchmark suite.
 *
 * See the `SuiteLog` interface below.
 */
export interface BenchmarkMetadata {
  commitHashes: CommitHashes;
  timestamp: number;
}

/**
 * Information about hardware on which benchmarks are run.
 */
export interface HardwareInfo {
  /** `inxi` output. */
  cpuInfo?: string;

  /** Processed `free` output. */
  memInfo?: string;
}

/**
 * Enumerates all environments that TensorFlow.js benchmarks may happen.
 *
 * This type union is meant to be extended in the future.
 */
export type BenchmarkEnvironmentType = 'chrome-linux'|'chrome-mac'|
    'firefox-linux'|'firefox-mac'|'safari-mac'|'ios-11'|
    'node-libtensorflow-cpu'|'node-libtensorflow-cuda'|'node-gles'|
    'python-tensorflow-cpu'|'python-tensorflow-cuda';

/** Information about a benchmark environment. */
export interface EnvironmentInfo {
  type: BenchmarkEnvironmentType;

  /**
   * Metadata for the node environment.
   * `uname -a` output.
   */
  systemInfo?: string;
}

/** Metadata specific to the browser environment. */
export interface BrowserEnvironmentInfo extends EnvironmentInfo {
  userAgent: string;

  webGLVersion?: string;
}

export interface ServerSideEnvironmentInfo extends EnvironmentInfo {
  hardwareInfo?: HardwareInfo;

  /** Processed `nvidia-smi` output. */
  cudaGPUInfo?: string;

  /** Can be extracted from `nvcc --version`. */
  cudaVersion?: string;
}

/** Metadata specific to the Node.js environment. */
export interface NodeEnvironmentInfo extends ServerSideEnvironmentInfo {
  /** `node --version` output. */
  nodeVersion?: string;

  tfjsNodeVersion?: string;
  tfjsNodeUsesCUDA?: boolean;
}

/** Metadata specific to the Python environment. */
export interface PythonEnvironmentInfo extends
    ServerSideEnvironmentInfo {
  pythonVersion?: string;

  tensorflowVersion?: string;
  kerasVersion?: string;
  tensorflowUsesCUDA?: boolean;
}
