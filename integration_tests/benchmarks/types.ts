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
 * Interfaces that correspond to collections in Firestore Collections.
 */

export type VersionSetCollection = {[versionSetId: string]: VersionSet};

export type EnvironmentCollection = {[environmentId: string]: EnvironmentInfo};

/**
 * Used by the dashboard front end to quickly retrieve a list
 * of available benchmark tasks.
 *
 * New documents (rows) will be added to this collection (table) only when a
 * new benchmark task (e.g., a new model, or a new function of an existing
 * model) has been added. This is how a client frontend knows what models and
 * function (and non-model tasks) are available in the database.
 * Therefore, multiple runs of the same task will reuse the same document (row)
 * in this collection (table).
 *
 * Note: this collection has redundant information with `BenchmarkRun` and
 * `BenchmarkRunCollection`. It is essentially an index that speeds up queries
 * for all available tasks. In principle, this collection can be recreated
 * from `BenchmarkRunCollection` if necessary.
 */
export type TaskCollection = {[taskId: string]: Task};

/**
 * A collection containing information about benchmarked models.
 *
 * Their versions and description.
 *
 * This applies to only model-based benchmarks and does not apply to
 * non-model benchmarks.
 */
export type ModelCollection = {[modelId: string]: Model};

/** The collection that stores the actual benchmark results data. */
export type BenchmarkRunCollection = {[benchmarkRunId: string]: BenchmarkRun};

/** Version sets. */

export type CodeRepository =
    'tfjs'|'tfjs-converter'|'tfjs-core'|'tfjs-data'|'tfjs-layers'|'tfjs-node';

export interface VersionSet {
  commitHashes?: {[repo in CodeRepository]?: string};
  versions?: {[repo in CodeRepository]?: string};
}

/** Environments. */

export type BrowserEnvironmentType =
    'chrome-linux' | 'chrome-mac' | 'firefox-linux' | 'firefox-mac' |
    'safari-mac' | 'chrome-windows' | 'firefox-windows' |
    'chrome-ios-11' | 'safari-ios-11';
export type NodeEnvironmentType =
    'node-libtensorflow-cpu' | 'node-libtensorflow-cuda' | 'node-gles';
export type PythonEnvironmentType =
    'python-tensorflow-cpu' | 'python-tensorflow-cuda';
export type ServerSideEnvironmentType =
    NodeEnvironmentType | PythonEnvironmentType;
export type BenchmarkEnvironmentType =
    BrowserEnvironmentType | ServerSideEnvironmentType;

export interface EnvironmentInfo {
  type: BenchmarkEnvironmentType;

  /** `uname -a` output. */
  systemInfo?: string;

  /** `inxi` output. */
  cpuInfo?: string;

  /** Processed `free` output. */
  memInfo?: string;
}

export interface BrowserEnvironmentInfo extends EnvironmentInfo {
  type: BrowserEnvironmentType;

  /** Browser `navigator.userAgent`. */
  userAgent: string;

  webGLVersion?: string;
}

export interface ServerSideEnvironmentInfo extends EnvironmentInfo {
  type: ServerSideEnvironmentType;

  /** Processed output from `nvidia-smi`. */
  cudaGPUInfo?: string;

  /** Output from `nvcc --version`. */
  cudaVersion?: string;
}

export interface NodeEnvironmentInfo extends ServerSideEnvironmentInfo {
  type: NodeEnvironmentType;

  /** `node --version` output. */
  nodeVersion: string;
  tfjsNodeVersion?: string;
  tfjsNodeUsesCUDA?: boolean;
}

export interface PythonEnvironmentInfo extends ServerSideEnvironmentInfo {
  type: PythonEnvironmentType;

  /** Output of `python --version`. */
  pythonVersion?: string;

  tensorflowVersion?: string;
  kerasVersion?: string;
  tensorflowUsesCUDA?: boolean;
}

/**
 * Task types, names and function names under each task.
 *
 * If a task is performed in multiple environments (e.g., in tfjs-node and
 * Python), they should correspond the same `Task` object.
 */
export interface Task {
  taskType: TaskType;

  /**
   * Name of the task.
   *
   * For model-based tasks, this is the name of the model.
   */
  taskName: string;

  /**
   * For model-based tasks, this is the function name, e.g.,
   * predict(), fit(), fitDataset().
   */
  functionName?: string;
}

export interface ModelTask extends Task {
  /** A reference to Model in ModelCollection. */
  modelId: string;

  functionName: ModelFunctionName;
}

/** Information about a benchmarked model. */
export type ModelType =
    'keras-model' | 'tensorflow-model' | 'tfjs-layers-model' |
    'tfjs-graph-model';

export interface Model {
  type: ModelType;

  name: string;

  version: string;

  description: string;
}

/** Benchmark tasks logs. */

// TODO(cais): Add new types in the future, such as low-level tensor
//   operations, etc.
export type TaskType = 'model';

export type ModelFormat = 'LayersModel' | 'GraphModel';

export type ModelFunctionName = 'predict' | 'fit' | 'fitDataset';

export type FunctionName = ModelFunctionName;

/**
 * The results and related data from a benchmark run.
 *
 * A benchmark run is a full set of iterations that assess the performance
 * of a specific task (see `Task` interface) in a given environment,
 * under a given version / commit set. It includes a number of warm-up
 * (a.k.a., burn-in) iterations, followed by a number of benchmarked
 * iterations.
 *
 * See the doc strings below for what an iteration means in the context
 * of Model.predict(), fit(), fitDataset() and non-model-based tasks.
 */
export interface BenchmarkRun {
  versionSetId: string;
  environmentId: string;
  taskId: string;

  taskType: TaskType;

  functionName: string;

  /**
   * Number of burn-in (i.e., warm-up) iterations that take place prior to the
   * benchmarked iterations.
   *
   * Follows the same convention as `numBenchmarkedIterations` for counting
   * iterations.
   */
  numWarmUpIterations: number;

  /**
   * Number of individual iterations that are timed.
   *
   * - For predict() and non-model-based tasks, this is the number of function
   *   calls.
   * - For fit() and fitDatset(), this is the number of epochs of a single call.
   */
  numBenchmarkedIterations: number;

  /** The ending timestamp of the task, in milliseconds since epoch. */
  endingTimestampMs: number;

  /**
   * Raw benchmarked run times in milliseconds.
   *
   * This field is optional because for certain types of benchmarks (e.g.,
   * tf.LayersModel.fit() calls), the individual-iteration times are not
   * available (e.g., cannot obtain epoch-by-epoch times without using
   * a callback; but a callback affects the timing itself.)
   * However, in cases where the individual-iteration times are available
   * (e.g., tf.LayersModel.predict calls), it should be populated.
   *
   * - For predict() and non-model-based tasks, each item of the array is
   *   the wall time of a single predict() call or some other type of function
   *   call.
   * - For fit() and fitDatset(), this field is not populated, due to the
   *   fact that obtaining per-epoch time requires a callback and that may
   *   affect the performance of the fit() / fitDataset() call.
   */
  timesMs?: number[];

  /**
   * Arithmetic mean of `benchmarkedRunTimesMs`.
   *
   * - For predict() and non-model-based tasks, this is the arithmetic mean
   *   of `timeMs`.
   * - For a fit() or fitDataset() task, this is the total time of the function
   *   call divided by the number of training epochs (i.e.,
   *   `numBenchmarkedIterations`).
   */
  averageTimeMs: number;
}

export interface ModelBenchmarkRun extends BenchmarkRun {
  taskType: 'model';

  modelFormat: ModelFormat;

  modelName: string;

  functionName: ModelFunctionName;

  batchSize: number;
}

export interface ModelTrainingBenchmarkRun extends ModelBenchmarkRun {
  loss: string;
  optimizer: string;
}
