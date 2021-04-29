/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

/** Category proto instance used in NLP tasks. */
export declare interface Category {
  score: number;
  className: string;
}

/** Class proto instance used in vision tasks. */
export declare interface Class {
  getClassName(): string;
  hasClassName(): boolean;

  getDisplayName(): string;
  hasDisplayName(): boolean;

  getIndex(): number;
  hasIndex(): boolean;

  getScore(): number;
  hasScore(): boolean;
}

/** Stores the status of WASM features that user's browser supports. */
export declare interface WasmFeatures {
  simd: boolean;
  multiThreading: boolean;
}

/** Base class for task libraries. */
export declare class BaseTaskLibrary {
  /** Cleans up resources when the instance is no longer needed. */
  cleanUp(): void;
}
