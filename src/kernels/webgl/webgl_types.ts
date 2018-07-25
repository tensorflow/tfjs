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

// TODO(nsthorat): Move these to the webgl official typings.
export interface WebGL2DisjointQueryTimerExtension {
  TIME_ELAPSED_EXT: number;
  GPU_DISJOINT_EXT: number;
}

export interface WebGL1DisjointQueryTimerExtension {
  TIME_ELAPSED_EXT: number;
  QUERY_RESULT_AVAILABLE_EXT: number;
  GPU_DISJOINT_EXT: number;
  QUERY_RESULT_EXT: number;
  createQueryEXT: () => {};
  beginQueryEXT: (ext: number, query: WebGLQuery) => void;
  endQueryEXT: (ext: number) => void;
  deleteQueryEXT: (query: WebGLQuery) => void;
  isQueryEXT: (query: WebGLQuery) => boolean;
  getQueryObjectEXT:
      (query: WebGLQuery, queryResultAvailableExt: number) => number;
}

export interface WebGLContextAttributes {
  alpha?: boolean;
  antialias?: boolean;
  premultipliedAlpha?: boolean;
  preserveDrawingBuffer?: boolean;
  depth?: boolean;
  stencil?: boolean;
  failIfMajorPerformanceCaveat?: boolean;
}
