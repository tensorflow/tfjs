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

export interface BenchmarkMetadata {
  environment: 'browser' | 'node';

  /** Metadata for the browser environment. */
  userAgent?: string;

  /**
   * Metadata for the node environment.
   * `uname -a` output.
   */
  systemInfo?: string;

  hardwareInfo?: HardwareInfo;

  pythonVersion?: string;
  nodeVersion?: string;
  tensorflowVersion?: string;

  /** This should be x.y.z-tf for tf.keras. */
  kerasVersion?: string;

  tensorflowUsesGPU?: boolean;
  tfjsNodeVersion?: string;
  tfjsNodeUsesGPU?: boolean;
}

export interface ModelTaskLog {
  /**
   * For fit() and fitDatset(), this is the number of epochs of a single call.
   * For predict(), this is the number of predict() calls.
   */
  numBenchmarksRuns: number;
  numBurnInRuns: number;

  batchSize: number;
  averageTimeMs: number;
  medianTimeMs: number;
  minTimeMs: number;

  timestamp: number;
}

export type ModelBenchmarkTask =
    'predict'|'fit'|'fitDataset'|'pyPredict'|'pyFit'|'pyFitDataset';

/**
 * The benchmark results from a number of models in a given
 * environment, from a certain run of an entire benchmark suite.
 */
export type ModelLog = {[taskName in ModelBenchmarkTask]: ModelTaskLog};

export type MultiModelLog = {[modelName: string]: ModelLog};

/**
 * The same as MultiModelLog, but with metadata and timestamp.
 */
export interface ModelSuiteLog {
  data: MultiModelLog;
  metadata: BenchmarkMetadata;

  // The starting timestamp of the suite.
  timestamp: number;
}

export type ModelBenchmarkHistory = ModelSuiteLog[];
