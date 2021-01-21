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

import {env} from '@tensorflow/tfjs-core';

const ENV = env();

/** Whether we submit commands to the device queue immediately. */
ENV.registerFlag('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', () => true);

/**
 * Whether we forward execution to the CPU backend if tensors are small and
 * reside on the CPU.
 */
ENV.registerFlag('WEBGPU_CPU_FORWARD', () => true);

/**
 * Thread register block size for matmul kernel.
 */
ENV.registerFlag('WEBGPU_MATMUL_WORK_PER_THREAD', () => 4);

/**
 * Whether to use conv2d_naive which directly implement the conv2d logic rather
 * than using a matmul to simulate.
 */
ENV.registerFlag('WEBGPU_USE_NAIVE_CONV2D', () => false);

/**
 * Whether we will run im2col as a separate shader for convolution.
 */
ENV.registerFlag('WEBGPU_CONV_SEPARATE_IM2COL_SHADER', () => false);

/**
 * Whether we use low power GPU. Otherwise, a high performance GPU will be
 * requested.
 */
ENV.registerFlag('WEBGPU_USE_LOW_POWER_GPU', () => false);
