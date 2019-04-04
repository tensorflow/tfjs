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
}

export interface BrowserEnvironmentInfo extends EnvironmentInfo {
  type: BrowserEnvironmentType;
  userAgent: string;
  webGLVersion?: string;
}

export interface ServerSideEnvironmentInfo extends EnvironmentInfo {
  type: ServerSideEnvironmentType;
  
  /**
   * Metadata for the node environment.
   * `uname -a` output.
   */
  systemInfo?: string;

  cpuInfo?: string;
  memInfo?: string;
  cudaGPUInfo?: string;
  cudaVersion?: string;
}

export interface NodeEnvironmentInfo extends ServerSideEnvironmentInfo {
  type: NodeEnvironmentType;

  /** `node --version` output. */
  nodeVersion?: string;
  tfjsNodeVersion?: string;
  tfjsNodeUsesCUDA?: boolean;
}

export interface PythonEnvironmentInfo extends ServerSideEnvironmentInfo {
  type: PythonEnvironmentType;
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
 **/
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
  functionName?: ModelFunctionName;
};

/**
 * This corresponds to 
 */
export type TaskCollection = {[taskId: string]: Task};

/** Benchmark tasks logs. */

// TODO(tensorflowjs): Add new types in the future, such as low-level tensor
// operations, etc.
export type TaskType = 'model';

export type ModelFunctionName = 'predict' | 'fit' | 'fitDataset';

export type FunctionName = ModelFunctionName;

export interface TaskLog {
  versionSetId: string;
  environmentId: string;
  taskId: string;

  taskType: TaskType;

  functionName: FunctionName;

  /**
   * Number of burn-in (i.e., warm-up) runs that take place prior to the
   * benchmarked runs.
   *
   * Follows the same convention as `numBenchmarkedRuns` for counting runs.
   */
  numWarmUpRuns: number;

  /**
   * Number of individual runs that are timed.
   *
   * - For predict() and non-model-based tasks, this is the number of function
   *   calls.
   * - For fit() and fitDatset(), this is the number of epochs of a single call.
   */
  numBenchmarkedRuns: number;

  /** The ending timestap of the task, in milliseconds since epoch. */
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
   **/
  timesMs?: number[];
  
  /**
   * Arithemtic mean of `benchmarkedRunTimesMs`.
   * 
   * This redundant field is created to support faster database querying.
   */
  averageTimeMs: number;
}

export interface ModelTaskLog extends TaskLog {
  taskType: 'model';

  modelName: string;

  /**
   * A description of the model. 
   * 
   * Usually more detailed than `modelName`, and may contain information
   * such as the constituent layers and the optimizer used for training.
   */
  modelDescription?: string;

  functionName: ModelFunctionName;

  batchSize: number;
}

export interface ModelTrainingTaskLog extends ModelTaskLog {
  loss: string;
  optimizer: string;
}
