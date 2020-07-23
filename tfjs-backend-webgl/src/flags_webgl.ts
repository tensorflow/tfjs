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

import {device_util, env} from '@tensorflow/tfjs-core';

import {getMaxTexturesInShader, getWebGLDisjointQueryTimerVersion, getWebGLMaxTextureSize, isCapableOfRenderingToFloatTexture, isDownloadFloatTextureEnabled, isWebGLFenceEnabled, isWebGLVersionEnabled} from './webgl_util';

const ENV = env();

/**
 * This file contains WebGL-specific flag registrations.
 */

/**
 * True if WebGL is supported.
 */
ENV.registerFlag('HAS_WEBGL', () => ENV.getNumber('WEBGL_VERSION') > 0);

/** 0: No WebGL, 1: WebGL 1.0, 2: WebGL 2.0. */
ENV.registerFlag('WEBGL_VERSION', () => {
  if (isWebGLVersionEnabled(2)) {
    return 2;
  } else if (isWebGLVersionEnabled(1)) {
    return 1;
  }
  return 0;
});

/** Whether to check for numerical representation problems. */
ENV.registerFlag('WEBGL_CHECK_NUMERICAL_PROBLEMS', () => false);

ENV.registerFlag(
    'WEBGL_BUFFER_SUPPORTED', () => ENV.get('WEBGL_VERSION') === 2);

/** Whether the WebGL backend will sometimes forward ops to the CPU. */
ENV.registerFlag('WEBGL_CPU_FORWARD', () => true);

/** Whether the WebGL backend will always use f16 textures for rendering. */
ENV.registerFlag('WEBGL_FORCE_F16_TEXTURES', () => false);

/** Whether to turn all packing related flags on. */
ENV.registerFlag('WEBGL_PACK', () => ENV.getBool('HAS_WEBGL'));

/** Whether we will pack the batchnormalization op. */
ENV.registerFlag('WEBGL_PACK_NORMALIZATION', () => ENV.getBool('WEBGL_PACK'));

/** Whether we will pack the clip op. */
ENV.registerFlag('WEBGL_PACK_CLIP', () => ENV.getBool('WEBGL_PACK'));

/** Whether we will pack the depthwise conv op. */
// TODO: https://github.com/tensorflow/tfjs/issues/1679
ENV.registerFlag('WEBGL_PACK_DEPTHWISECONV', () => false);

/** Whether we will pack binary ops. */
ENV.registerFlag(
    'WEBGL_PACK_BINARY_OPERATIONS', () => ENV.getBool('WEBGL_PACK'));

/** Whether we will pack unary ops. */
ENV.registerFlag(
    'WEBGL_PACK_UNARY_OPERATIONS', () => ENV.getBool('WEBGL_PACK'));

/** Whether we will pack array ops. */
ENV.registerFlag(
    'WEBGL_PACK_ARRAY_OPERATIONS', () => ENV.getBool('WEBGL_PACK'));

/** Whether we will pack image ops. */
ENV.registerFlag(
    'WEBGL_PACK_IMAGE_OPERATIONS', () => ENV.getBool('WEBGL_PACK'));

/** Whether we will pack reduce ops. */
ENV.registerFlag('WEBGL_PACK_REDUCE', () => ENV.getBool('WEBGL_PACK'));

/** Whether packed WebGL kernels lazily unpack their outputs. */
ENV.registerFlag('WEBGL_LAZILY_UNPACK', () => ENV.getBool('WEBGL_PACK'));

/** Whether we will use the im2col algorithm to speed up convolutions. */
ENV.registerFlag('WEBGL_CONV_IM2COL', () => ENV.getBool('WEBGL_PACK'));

/** The maximum texture dimension. */
ENV.registerFlag(
    'WEBGL_MAX_TEXTURE_SIZE',
    () => getWebGLMaxTextureSize(ENV.getNumber('WEBGL_VERSION')));

/** The maximum texture dimension. */
ENV.registerFlag(
    'WEBGL_MAX_TEXTURES_IN_SHADER',
    () => getMaxTexturesInShader(ENV.getNumber('WEBGL_VERSION')));

/**
 * The disjoint_query_timer extension version.
 * 0: disabled, 1: EXT_disjoint_timer_query, 2:
 * EXT_disjoint_timer_query_webgl2.
 * In Firefox with WebGL 2.0,
 * EXT_disjoint_timer_query_webgl2 is not available, so we must use the
 * WebGL 1.0 extension.
 */
ENV.registerFlag('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', () => {
  const webGLVersion = ENV.getNumber('WEBGL_VERSION');

  if (webGLVersion === 0) {
    return 0;
  }
  return getWebGLDisjointQueryTimerVersion(webGLVersion);
});

/**
 * Whether the timer object from the disjoint_query_timer extension gives
 * timing information that is reliable.
 */
ENV.registerFlag(
    'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE',
    () => ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0 &&
        !device_util.isMobile());

/**
 * Whether the device is physically capable of rendering to float32 textures.
 */
ENV.registerFlag(
    'WEBGL_RENDER_FLOAT32_CAPABLE',
    () => isCapableOfRenderingToFloatTexture(ENV.getNumber('WEBGL_VERSION')));

/**
 * Whether rendering to float32 textures is enabled. If disabled, renders to
 * float16 textures.
 */
ENV.registerFlag('WEBGL_RENDER_FLOAT32_ENABLED', () => {
  return ENV.getBool('WEBGL_FORCE_F16_TEXTURES') ?
      false :
      ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE');
});

/**
 * Whether downloading float textures is enabled (16 or 32 bit). If disabled,
 * uses IEEE 754 encoding of the float32 values to 4 uint8 when downloading.
 */
ENV.registerFlag(
    'WEBGL_DOWNLOAD_FLOAT_ENABLED',
    () => isDownloadFloatTextureEnabled(ENV.getNumber('WEBGL_VERSION')));

/** Whether the fence API is available. */
ENV.registerFlag(
    'WEBGL_FENCE_API_ENABLED',
    () => isWebGLFenceEnabled(ENV.getNumber('WEBGL_VERSION')));

/**
 * Tensors with size <= than this will be uploaded as uniforms, not textures.
 */
ENV.registerFlag('WEBGL_SIZE_UPLOAD_UNIFORM', () => {
  // Use uniform uploads only when 32bit floats are supported. In
  // 16bit
  // environments there are problems with comparing a 16bit texture value
  // with a 32bit uniform value.
  const useUniforms = ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED');
  return useUniforms ? 4 : 0;
});

/**
 * If the total number of bytes allocated on the GPU is greater than this
 * number, we will aggressively delete textures upon disposal with
 * gl.deleteMatrixTexture, rather than making them available for reuse.
 *
 * Default value -1 indicates that we will never aggressively delete textures.
 */
ENV.registerFlag(
    'WEBGL_DELETE_TEXTURE_THRESHOLD',
    () => {
      return -1;
    },
    threshold => {
      if (threshold < 0 && threshold !== -1) {
        throw new Error(
            `WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never ` +
            `delete) or at least 0, but got ${threshold}.`);
      }
    });
