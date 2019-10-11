/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {ENV} from '@tensorflow/tfjs-core';

/** Whether we submit commands to the device queue immediately. */
ENV.registerFlag('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', () => true);

/**
 * Whether we forward execution to the CPU backend if tensors are small and
 * reside on the CPU.
 */
ENV.registerFlag('WEBGPU_CPU_FORWARD', () => true);

/**
 * Thread register block size for matmul kernel. If 0, we use the version of
 * matMul without register blocking.
 */
ENV.registerFlag('WEBGPU_MATMUL_WORK_PER_THREAD', () => 4);

/**
 * -1: conv2d_naive
 *  0: conv2d_mm with matmul without register blocking
 * >0: conv2d_mm with matmul_packed with WPT=this
 */
ENV.registerFlag('WEBGPU_CONV2D_WORK_PER_THREAD', () => 2);
