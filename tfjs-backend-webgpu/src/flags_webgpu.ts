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

/** The batched dispatching calls size in the device queue. */
ENV.registerFlag('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', () => 15);

/**
 * Whether we forward execution to the CPU backend if tensors are small and
 * reside on the CPU.
 */
ENV.registerFlag('WEBGPU_CPU_FORWARD', () => true);

/**
 * This flag is used to test different types of matmul programs.
 *
 * See MatMulProgramType in webgpu_util.ts for a list of available values.
 */
ENV.registerFlag('WEBGPU_MATMUL_PROGRAM_TYPE', () => -1);

/**
 * Whether to use conv2dTranspose_naive which directly implement the
 * conv2dTranspose logic rather than using a matmul to simulate.
 */
ENV.registerFlag('WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE', () => true);

/**
 * Whether we use low power GPU. Otherwise, a high performance GPU will be
 * requested.
 */
ENV.registerFlag('WEBGPU_USE_LOW_POWER_GPU', () => false);

/**
 * Threshold for input tensor size that determines whether WebGPU backend will
 * delegate computation to CPU.
 *
 * Default value is 1000.
 */
ENV.registerFlag('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD', () => 1000);

/**
 * Whether to use a dummy canvas to make profiling tools like PIX work with
 * TFJS webgpu backend.
 */
ENV.registerFlag('WEBGPU_USE_PROFILE_TOOL', () => false);

/**
 * Whether to use import API.
 */
ENV.registerFlag('WEBGPU_IMPORT_EXTERNAL_TEXTURE', () => true);

/**
 * Whether to use conv2dNaive for debugging.
 */
ENV.registerFlag('WEBGPU_USE_NAIVE_CONV2D_DEBUG', () => false);

/**
 * Threshold to increase dispatched workgroups for matmul. If too few workgroups
 * are dispatched, it means the hardware may be in low occupancy.
 * -1 means it's not set by the user. A default strategy will be applied.
 */
ENV.registerFlag(
    'WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL', () => -1);

/**
 * Whether we will run im2col as a separate shader for convolution.
 */
ENV.registerFlag('WEBGPU_CONV_SEPARATE_IM2COL_SHADER', () => false);

/**
 * A string used to match shader key. If any matches, print the related shader.
 * Seperated by comma. 'all' to print all. 'binary' to print binary(add, mul,
 * etc.). 'unary,conv2d' to print both unary and conv2d.
 */
ENV.registerFlag('WEBGPU_PRINT_SHADER', () => '');

/** Experimental flag, whether enter compile only phase. */
ENV.registerFlag('WEBGPU_ENGINE_COMPILE_ONLY', () => false);
