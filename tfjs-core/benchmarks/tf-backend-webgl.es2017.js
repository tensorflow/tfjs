/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core')) :
  typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core'], factory) :
  (global = global || self, factory(global.tf = global.tf || {}, global.tf));
}(this, (function (exports, tf) { 'use strict';

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
  const contexts = {};
  const WEBGL_ATTRIBUTES = {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      depth: false,
      stencil: false,
      failIfMajorPerformanceCaveat: true
  };
  function setWebGLContext(webGLVersion, gl) {
      contexts[webGLVersion] = gl;
  }
  function getWebGLContext(webGLVersion) {
      if (!(webGLVersion in contexts)) {
          contexts[webGLVersion] = getWebGLRenderingContext(webGLVersion);
      }
      const gl = contexts[webGLVersion];
      if (gl.isContextLost()) {
          delete contexts[webGLVersion];
          return getWebGLContext(webGLVersion);
      }
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.STENCIL_TEST);
      gl.disable(gl.BLEND);
      gl.disable(gl.DITHER);
      gl.disable(gl.POLYGON_OFFSET_FILL);
      gl.disable(gl.SAMPLE_COVERAGE);
      gl.enable(gl.SCISSOR_TEST);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      return contexts[webGLVersion];
  }
  function createCanvas(webGLVersion) {
      if (typeof OffscreenCanvas !== 'undefined' && webGLVersion === 2) {
          return new OffscreenCanvas(300, 150);
      }
      else if (typeof document !== 'undefined') {
          return document.createElement('canvas');
      }
      else {
          throw new Error('Cannot create a canvas in this context');
      }
  }
  function getWebGLRenderingContext(webGLVersion) {
      if (webGLVersion !== 1 && webGLVersion !== 2) {
          throw new Error('Cannot get WebGL rendering context, WebGL is disabled.');
      }
      const canvas = createCanvas(webGLVersion);
      canvas.addEventListener('webglcontextlost', (ev) => {
          ev.preventDefault();
          delete contexts[webGLVersion];
      }, false);
      if (webGLVersion === 1) {
          return (canvas.getContext('webgl', WEBGL_ATTRIBUTES) ||
              canvas.getContext('experimental-webgl', WEBGL_ATTRIBUTES));
      }
      return canvas.getContext('webgl2', WEBGL_ATTRIBUTES);
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  var PackingScheme;
  (function (PackingScheme) {
      /**
       * All values in a single texel are densely packed without any constraints.
       *
       * This is how the shader encodes a tensor with shape = [2, 3, 4]
       * (indices are [batch, row, col]).
       *
       * 000|001   010|011   020|021
       * -------   -------   -------
       * 002|003   012|013   022|023
       *
       * 100|101   110|111   120|121
       * -------   -------   -------
       * 102|103   112|113   122|123
       *
       */
      PackingScheme[PackingScheme["DENSE"] = 0] = "DENSE";
      /**
       * Single texels contain only values from the same batch, and from adjacent
       * rows and columns.
       *
       * This is how the shader encodes a tensor with shape = [2, 3, 5]
       * (indices are [batch, row, col]).
       *
       * 000|001   002|003   004|xxx   020|021   022|023   024|xxx
       * -------   -------   -------   -------   -------   -------
       * 010|011   012|013   014|xxx   xxx|xxx   xxx|xxx   xxx|xxx
       *
       * 100|101   102|103   104|xxx   120|121   122|123   124|xxx
       * -------   -------   -------   -------   -------   -------
       * 110|111   112|113   114|xxx   xxx|xxx   xxx|xxx   xxx|xxx
       *
       */
      PackingScheme[PackingScheme["SHARED_BATCH"] = 1] = "SHARED_BATCH";
  })(PackingScheme || (PackingScheme = {}));
  var TextureUsage;
  (function (TextureUsage) {
      TextureUsage[TextureUsage["RENDER"] = 0] = "RENDER";
      TextureUsage[TextureUsage["UPLOAD"] = 1] = "UPLOAD";
      TextureUsage[TextureUsage["PIXELS"] = 2] = "PIXELS";
      TextureUsage[TextureUsage["DOWNLOAD"] = 3] = "DOWNLOAD";
  })(TextureUsage || (TextureUsage = {}));
  var PhysicalTextureType;
  (function (PhysicalTextureType) {
      PhysicalTextureType[PhysicalTextureType["UNPACKED_FLOAT16"] = 0] = "UNPACKED_FLOAT16";
      PhysicalTextureType[PhysicalTextureType["UNPACKED_FLOAT32"] = 1] = "UNPACKED_FLOAT32";
      PhysicalTextureType[PhysicalTextureType["PACKED_4X1_UNSIGNED_BYTE"] = 2] = "PACKED_4X1_UNSIGNED_BYTE";
      PhysicalTextureType[PhysicalTextureType["PACKED_2X2_FLOAT32"] = 3] = "PACKED_2X2_FLOAT32";
      PhysicalTextureType[PhysicalTextureType["PACKED_2X2_FLOAT16"] = 4] = "PACKED_2X2_FLOAT16";
  })(PhysicalTextureType || (PhysicalTextureType = {}));
  function getUnpackedMatrixTextureShapeWidthHeight(rows, columns) {
      return [columns, rows];
  }
  function getUnpackedArraySizeFromMatrixSize(matrixSize, channelsPerTexture) {
      return matrixSize * channelsPerTexture;
  }
  /**
   * Get shape for densely packed RGBA texture.
   */
  function getDenseTexShape(shape) {
      const size = tf.util.sizeFromShape(shape);
      const texelsNeeded = Math.ceil(size / 4);
      return tf.util.sizeToSquarishShape(texelsNeeded);
  }
  function getPackedMatrixTextureShapeWidthHeight(rows, columns) {
      return [
          Math.max(1, Math.ceil(columns / 2)), Math.max(1, Math.ceil(rows / 2))
      ];
  }
  function getPackedRGBAArraySizeFromMatrixShape(rows, columns) {
      const [w, h] = getPackedMatrixTextureShapeWidthHeight(rows, columns);
      return w * h * 4;
  }
  function getTextureConfig(
  // tslint:disable-next-line:no-any
  gl, textureHalfFloatExtension) {
      // tslint:disable-next-line:no-any
      const glany = gl;
      let internalFormatFloat;
      let internalFormatHalfFloat;
      let internalFormatPackedHalfFloat;
      let internalFormatPackedFloat;
      let textureFormatFloat;
      let downloadTextureFormat;
      let downloadUnpackNumChannels;
      let defaultNumChannels;
      let textureTypeHalfFloat;
      let textureTypeFloat;
      if (tf.env().getNumber('WEBGL_VERSION') === 2) {
          internalFormatFloat = glany.R32F;
          internalFormatHalfFloat = glany.R16F;
          internalFormatPackedHalfFloat = glany.RGBA16F;
          internalFormatPackedFloat = glany.RGBA32F;
          textureFormatFloat = glany.RED;
          downloadUnpackNumChannels = 4;
          defaultNumChannels = 1;
          textureTypeHalfFloat = glany.HALF_FLOAT;
          textureTypeFloat = glany.FLOAT;
      }
      else {
          internalFormatFloat = gl.RGBA;
          internalFormatHalfFloat = gl.RGBA;
          internalFormatPackedHalfFloat = gl.RGBA;
          internalFormatPackedFloat = glany.RGBA;
          textureFormatFloat = gl.RGBA;
          downloadUnpackNumChannels = 4;
          defaultNumChannels = 4;
          textureTypeHalfFloat = textureHalfFloatExtension != null ?
              textureHalfFloatExtension.HALF_FLOAT_OES :
              null;
          textureTypeFloat = gl.FLOAT;
      }
      downloadTextureFormat = gl.RGBA;
      return {
          internalFormatFloat,
          internalFormatHalfFloat,
          internalFormatPackedHalfFloat,
          internalFormatPackedFloat,
          textureFormatFloat,
          downloadTextureFormat,
          downloadUnpackNumChannels,
          defaultNumChannels,
          textureTypeHalfFloat,
          textureTypeFloat
      };
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  function callAndCheck(gl, debugMode, func) {
      const returnValue = func();
      if (debugMode) {
          checkWebGLError(gl);
      }
      return returnValue;
  }
  function checkWebGLError(gl) {
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
          throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
      }
  }
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
  const MIN_FLOAT16 = 5.96e-8;
  const MAX_FLOAT16 = 65504;
  function canBeRepresented(num) {
      if (tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') || num === 0 ||
          (MIN_FLOAT16 < Math.abs(num) && Math.abs(num) < MAX_FLOAT16)) {
          return true;
      }
      return false;
  }
  function getWebGLErrorMessage(gl, status) {
      switch (status) {
          case gl.NO_ERROR:
              return 'NO_ERROR';
          case gl.INVALID_ENUM:
              return 'INVALID_ENUM';
          case gl.INVALID_VALUE:
              return 'INVALID_VALUE';
          case gl.INVALID_OPERATION:
              return 'INVALID_OPERATION';
          case gl.INVALID_FRAMEBUFFER_OPERATION:
              return 'INVALID_FRAMEBUFFER_OPERATION';
          case gl.OUT_OF_MEMORY:
              return 'OUT_OF_MEMORY';
          case gl.CONTEXT_LOST_WEBGL:
              return 'CONTEXT_LOST_WEBGL';
          default:
              return `Unknown error code ${status}`;
      }
  }
  function getExtensionOrThrow(gl, debug, extensionName) {
      return throwIfNull(gl, debug, () => gl.getExtension(extensionName), 'Extension "' + extensionName + '" not supported on this browser.');
  }
  function createVertexShader(gl, debug, vertexShaderSource) {
      const vertexShader = throwIfNull(gl, debug, () => gl.createShader(gl.VERTEX_SHADER), 'Unable to create vertex WebGLShader.');
      callAndCheck(gl, debug, () => gl.shaderSource(vertexShader, vertexShaderSource));
      callAndCheck(gl, debug, () => gl.compileShader(vertexShader));
      if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
          console.log(gl.getShaderInfoLog(vertexShader));
          throw new Error('Failed to compile vertex shader.');
      }
      return vertexShader;
  }
  function createFragmentShader(gl, debug, fragmentShaderSource) {
      const fragmentShader = throwIfNull(gl, debug, () => gl.createShader(gl.FRAGMENT_SHADER), 'Unable to create fragment WebGLShader.');
      callAndCheck(gl, debug, () => gl.shaderSource(fragmentShader, fragmentShaderSource));
      callAndCheck(gl, debug, () => gl.compileShader(fragmentShader));
      if (gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS) === false) {
          logShaderSourceAndInfoLog(fragmentShaderSource, gl.getShaderInfoLog(fragmentShader));
          throw new Error('Failed to compile fragment shader.');
      }
      return fragmentShader;
  }
  const lineNumberRegex = /ERROR: [0-9]+:([0-9]+):/g;
  function logShaderSourceAndInfoLog(shaderSource, shaderInfoLog) {
      const lineNumberRegexResult = lineNumberRegex.exec(shaderInfoLog);
      if (lineNumberRegexResult == null) {
          console.log(`Couldn't parse line number in error: ${shaderInfoLog}`);
          console.log(shaderSource);
          return;
      }
      const lineNumber = +lineNumberRegexResult[1];
      const shaderLines = shaderSource.split('\n');
      const pad = shaderLines.length.toString().length + 2;
      const linesWithLineNumbers = shaderLines.map((line, lineNumber) => tf.util.rightPad((lineNumber + 1).toString(), pad) + line);
      let maxLineLength = 0;
      for (let i = 0; i < linesWithLineNumbers.length; i++) {
          maxLineLength = Math.max(linesWithLineNumbers[i].length, maxLineLength);
      }
      const beforeErrorLines = linesWithLineNumbers.slice(0, lineNumber - 1);
      const errorLine = linesWithLineNumbers.slice(lineNumber - 1, lineNumber);
      const afterErrorLines = linesWithLineNumbers.slice(lineNumber);
      console.log(beforeErrorLines.join('\n'));
      console.log(shaderInfoLog.split('\n')[0]);
      console.log(`%c ${tf.util.rightPad(errorLine[0], maxLineLength)}`, 'border:1px solid red; background-color:#e3d2d2; color:#a61717');
      console.log(afterErrorLines.join('\n'));
  }
  function createProgram(gl, debug) {
      return throwIfNull(gl, debug, () => gl.createProgram(), 'Unable to create WebGLProgram.');
  }
  function linkProgram(gl, debug, program) {
      callAndCheck(gl, debug, () => gl.linkProgram(program));
      if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
          console.log(gl.getProgramInfoLog(program));
          throw new Error('Failed to link vertex and fragment shaders.');
      }
  }
  function validateProgram(gl, debug, program) {
      callAndCheck(gl, debug, () => gl.validateProgram(program));
      if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
          console.log(gl.getProgramInfoLog(program));
          throw new Error('Shader program validation failed.');
      }
  }
  function createStaticVertexBuffer(gl, debug, data) {
      const buffer = throwIfNull(gl, debug, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
      callAndCheck(gl, debug, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
      callAndCheck(gl, debug, () => gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW));
      return buffer;
  }
  function createStaticIndexBuffer(gl, debug, data) {
      const buffer = throwIfNull(gl, debug, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
      callAndCheck(gl, debug, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer));
      callAndCheck(gl, debug, () => gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW));
      return buffer;
  }
  function createTexture(gl, debug) {
      return throwIfNull(gl, debug, () => gl.createTexture(), 'Unable to create WebGLTexture.');
  }
  function validateTextureSize(width, height) {
      const maxTextureSize = tf.env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
      if ((width <= 0) || (height <= 0)) {
          const requested = `[${width}x${height}]`;
          throw new Error('Requested texture size ' + requested + ' is invalid.');
      }
      if ((width > maxTextureSize) || (height > maxTextureSize)) {
          const requested = `[${width}x${height}]`;
          const max = `[${maxTextureSize}x${maxTextureSize}]`;
          throw new Error('Requested texture size ' + requested +
              ' greater than WebGL maximum on this browser / GPU ' + max + '.');
      }
  }
  function createFramebuffer(gl, debug) {
      return throwIfNull(gl, debug, () => gl.createFramebuffer(), 'Unable to create WebGLFramebuffer.');
  }
  function bindVertexBufferToProgramAttribute(gl, debug, program, attribute, buffer, arrayEntriesPerItem, itemStrideInBytes, itemOffsetInBytes) {
      const loc = gl.getAttribLocation(program, attribute);
      if (loc === -1) {
          // The GPU compiler decided to strip out this attribute because it's unused,
          // thus no need to bind.
          return false;
      }
      callAndCheck(gl, debug, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
      callAndCheck(gl, debug, () => gl.vertexAttribPointer(loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes, itemOffsetInBytes));
      callAndCheck(gl, debug, () => gl.enableVertexAttribArray(loc));
      return true;
  }
  function bindTextureUnit(gl, debug, texture, textureUnit) {
      validateTextureUnit(gl, textureUnit);
      callAndCheck(gl, debug, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
      callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, texture));
  }
  function getProgramUniformLocationOrThrow(gl, debug, program, uniformName) {
      return throwIfNull(gl, debug, () => gl.getUniformLocation(program, uniformName), 'uniform "' + uniformName + '" not present in program.');
  }
  function getProgramUniformLocation(gl, program, uniformName) {
      return gl.getUniformLocation(program, uniformName);
  }
  function bindTextureToProgramUniformSampler(gl, debug, program, texture, uniformSamplerLocation, textureUnit) {
      callAndCheck(gl, debug, () => bindTextureUnit(gl, debug, texture, textureUnit));
      callAndCheck(gl, debug, () => gl.uniform1i(uniformSamplerLocation, textureUnit));
  }
  function bindColorTextureToFramebuffer(gl, debug, texture, framebuffer) {
      callAndCheck(gl, debug, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
      callAndCheck(gl, debug, () => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0));
  }
  function unbindColorTextureFromFramebuffer(gl, debug, framebuffer) {
      callAndCheck(gl, debug, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
      callAndCheck(gl, debug, () => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0));
  }
  function validateFramebuffer(gl) {
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
          throw new Error('Error binding framebuffer: ' + getFramebufferErrorMessage(gl, status));
      }
  }
  function getFramebufferErrorMessage(gl, status) {
      switch (status) {
          case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
              return 'FRAMEBUFFER_INCOMPLETE_ATTACHMENT';
          case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
              return 'FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT';
          case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
              return 'FRAMEBUFFER_INCOMPLETE_DIMENSIONS';
          case gl.FRAMEBUFFER_UNSUPPORTED:
              return 'FRAMEBUFFER_UNSUPPORTED';
          default:
              return `unknown error ${status}`;
      }
  }
  function throwIfNull(gl, debug, returnTOrNull, failureMessage) {
      const tOrNull = callAndCheck(gl, debug, () => returnTOrNull());
      if (tOrNull == null) {
          throw new Error(failureMessage);
      }
      return tOrNull;
  }
  function validateTextureUnit(gl, textureUnit) {
      const maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
      const glTextureUnit = textureUnit + gl.TEXTURE0;
      if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
          const textureUnitRange = `[gl.TEXTURE0, gl.TEXTURE${maxTextureUnit}]`;
          throw new Error(`textureUnit must be in ${textureUnitRange}.`);
      }
  }
  function getBatchDim(shape, dimsToSkip = 2) {
      return tf.util.sizeFromShape(shape.slice(0, shape.length - dimsToSkip));
  }
  function getRowsCols(shape) {
      if (shape.length === 0) {
          throw Error('Cannot get rows and columns of an empty shape array.');
      }
      return [
          shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]
      ];
  }
  function getShapeAs3D(shape) {
      let shapeAs3D = [1, 1, 1];
      const isScalar = shape.length === 0 || (shape.length === 1 && shape[0] === 1);
      if (!isScalar) {
          shapeAs3D =
              [getBatchDim(shape), ...getRowsCols(shape)];
      }
      return shapeAs3D;
  }
  function getTextureShapeFromLogicalShape(logShape, isPacked = false) {
      let maxTexSize = tf.env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
      if (isPacked) {
          maxTexSize = maxTexSize * 2;
          // This logic ensures we accurately count the number of packed texels needed
          // to accommodate the tensor. We can only pack values in the same texel if
          // they are from adjacent pairs of rows/cols within the same batch. So if a
          // tensor has 3 rows, we pretend it has 4 rows in order to account for the
          // fact that the texels containing the third row are half empty.
          logShape = logShape.map((d, i) => i >= logShape.length - 2 ?
              tf.util.nearestLargerEven(logShape[i]) :
              logShape[i]);
          // Packed texture height is at least 2 (the channel height of a single
          // texel).
          if (logShape.length === 1) {
              logShape = [2, logShape[0]];
          }
      }
      // If logical shape is 2, we don't squeeze, since we want to match physical.
      if (logShape.length !== 2) {
          const squeezeResult = tf.util.squeezeShape(logShape);
          logShape = squeezeResult.newShape;
      }
      let size = tf.util.sizeFromShape(logShape);
      if (logShape.length <= 1 && size <= maxTexSize) {
          return [1, size];
      }
      else if (logShape.length === 2 && logShape[0] <= maxTexSize &&
          logShape[1] <= maxTexSize) {
          return logShape;
      }
      else if (logShape.length === 3 && logShape[0] * logShape[1] <= maxTexSize &&
          logShape[2] <= maxTexSize) {
          return [logShape[0] * logShape[1], logShape[2]];
      }
      else if (logShape.length === 3 && logShape[0] <= maxTexSize &&
          logShape[1] * logShape[2] <= maxTexSize) {
          return [logShape[0], logShape[1] * logShape[2]];
      }
      else if (logShape.length === 4 &&
          logShape[0] * logShape[1] * logShape[2] <= maxTexSize &&
          logShape[3] <= maxTexSize) {
          return [logShape[0] * logShape[1] * logShape[2], logShape[3]];
      }
      else if (logShape.length === 4 && logShape[0] <= maxTexSize &&
          logShape[1] * logShape[2] * logShape[3] <= maxTexSize) {
          return [logShape[0], logShape[1] * logShape[2] * logShape[3]];
      }
      else {
          if (isPacked) {
              // For packed textures size equals the number of channels required to
              // accommodate the texture data. However in order to squarify such that
              // inner dimensions stay even, we rewrite size to equal the number of
              // texels. Then in the return statement we rehydrate the squarified
              // dimensions to channel units.
              const batchDim = getBatchDim(logShape);
              let rows = 2, cols = 2;
              if (logShape.length) {
                  [rows, cols] = getRowsCols(logShape);
              }
              size = batchDim * (rows / 2) * (cols / 2);
              return tf.util.sizeToSquarishShape(size).map(d => d * 2);
          }
          return tf.util.sizeToSquarishShape(size);
      }
  }
  function isEven(n) {
      return n % 2 === 0;
  }
  /**
   * This determines whether reshaping a packed texture requires rearranging
   * the data within the texture, assuming 2x2 packing.
   */
  function isReshapeFree(shape1, shape2) {
      shape1 = shape1.slice(-2);
      shape2 = shape2.slice(-2);
      if (tf.util.arraysEqual(shape1, shape2)) {
          return true;
      }
      if (!shape1.length || !shape2.length) { // One of the shapes is a scalar.
          return true;
      }
      if (shape1[0] === 0 || shape1[1] === 0 || shape2[0] === 0 ||
          shape2[1] === 0) {
          return true;
      }
      if (shape1.length !== shape2.length) { // One of the shapes is a vector.
          const shape1Cols = shape1.slice(-1)[0];
          const shape2Cols = shape2.slice(-1)[0];
          if (shape1Cols === shape2Cols) {
              return true;
          }
          if (isEven(shape1Cols) && isEven(shape2Cols) &&
              (shape1[0] === 1 || shape2[0] === 1)) {
              return true;
          }
      }
      return shape1[1] === shape2[1] && isEven(shape1[0]) && isEven(shape2[0]);
  }
  // We cache webgl params because the environment gets reset between
  // unit tests and we don't want to constantly query the WebGLContext for
  // MAX_TEXTURE_SIZE.
  let MAX_TEXTURE_SIZE;
  let MAX_TEXTURES_IN_SHADER;
  function getWebGLMaxTextureSize(webGLVersion) {
      if (MAX_TEXTURE_SIZE == null) {
          const gl = getWebGLContext(webGLVersion);
          MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      }
      return MAX_TEXTURE_SIZE;
  }
  function getMaxTexturesInShader(webGLVersion) {
      if (MAX_TEXTURES_IN_SHADER == null) {
          const gl = getWebGLContext(webGLVersion);
          MAX_TEXTURES_IN_SHADER = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
      }
      // We cap at 16 to avoid spurious runtime "memory exhausted" error.
      return Math.min(16, MAX_TEXTURES_IN_SHADER);
  }
  function getWebGLDisjointQueryTimerVersion(webGLVersion) {
      if (webGLVersion === 0) {
          return 0;
      }
      let queryTimerVersion;
      const gl = getWebGLContext(webGLVersion);
      if (hasExtension(gl, 'EXT_disjoint_timer_query_webgl2') &&
          webGLVersion === 2) {
          queryTimerVersion = 2;
      }
      else if (hasExtension(gl, 'EXT_disjoint_timer_query')) {
          queryTimerVersion = 1;
      }
      else {
          queryTimerVersion = 0;
      }
      return queryTimerVersion;
  }
  function hasExtension(gl, extensionName) {
      const ext = gl.getExtension(extensionName);
      return ext != null;
  }
  function isWebGLVersionEnabled(webGLVersion) {
      try {
          const gl = getWebGLContext(webGLVersion);
          if (gl != null) {
              return true;
          }
      }
      catch (e) {
          return false;
      }
      return false;
  }
  function isCapableOfRenderingToFloatTexture(webGLVersion) {
      if (webGLVersion === 0) {
          return false;
      }
      const gl = getWebGLContext(webGLVersion);
      if (webGLVersion === 1) {
          if (!hasExtension(gl, 'OES_texture_float')) {
              return false;
          }
      }
      else {
          if (!hasExtension(gl, 'EXT_color_buffer_float')) {
              return false;
          }
      }
      const isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
      return isFrameBufferComplete;
  }
  /**
   * Check if we can download values from a float/half-float texture.
   *
   * Note that for performance reasons we use binding a texture to a framebuffer
   * as a proxy for ability to download float values later using readPixels. The
   * texture params of this texture will not match those in readPixels exactly
   * but if we are unable to bind some kind of float texture to the frameBuffer
   * then we definitely will not be able to read float values from it.
   */
  function isDownloadFloatTextureEnabled(webGLVersion) {
      if (webGLVersion === 0) {
          return false;
      }
      const gl = getWebGLContext(webGLVersion);
      if (webGLVersion === 1) {
          if (!hasExtension(gl, 'OES_texture_float')) {
              return false;
          }
          if (!hasExtension(gl, 'WEBGL_color_buffer_float')) {
              return false;
          }
      }
      else {
          if (hasExtension(gl, 'EXT_color_buffer_float')) {
              return createFloatTextureAndBindToFramebuffer(gl);
          }
          const COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
          if (hasExtension(gl, COLOR_BUFFER_HALF_FLOAT)) {
              const textureHalfFloatExtension = gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
              return createHalfFloatTextureAndBindToFramebuffer(gl, textureHalfFloatExtension);
          }
          return false;
      }
      const isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
      return isFrameBufferComplete;
  }
  function createFloatTextureAndBindToFramebuffer(gl) {
      const texConfig = getTextureConfig(gl);
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      const width = 1;
      const height = 1;
      gl.texImage2D(gl.TEXTURE_2D, 0, texConfig.internalFormatFloat, width, height, 0, texConfig.textureFormatFloat, texConfig.textureTypeFloat, null);
      const frameBuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(frameBuffer);
      return isFrameBufferComplete;
  }
  function createHalfFloatTextureAndBindToFramebuffer(
  // tslint:disable-next-line:no-any
  gl, textureHalfFloatExtension) {
      const texConfig = getTextureConfig(gl, textureHalfFloatExtension);
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      const width = 1;
      const height = 1;
      gl.texImage2D(gl.TEXTURE_2D, 0, texConfig.internalFormatHalfFloat, width, height, 0, texConfig.textureFormatFloat, texConfig.textureTypeHalfFloat, null);
      const frameBuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(frameBuffer);
      return isFrameBufferComplete;
  }
  function isWebGLFenceEnabled(webGLVersion) {
      if (webGLVersion !== 2) {
          return false;
      }
      const gl = getWebGLContext(webGLVersion);
      // tslint:disable-next-line:no-any
      const isEnabled = gl.fenceSync != null;
      return isEnabled;
  }

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
  const ENV = tf.env();
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
      }
      else if (isWebGLVersionEnabled(1)) {
          return 1;
      }
      return 0;
  });
  ENV.registerFlag('WEBGL_BUFFER_SUPPORTED', () => ENV.get('WEBGL_VERSION') === 2);
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
  ENV.registerFlag('WEBGL_PACK_BINARY_OPERATIONS', () => ENV.getBool('WEBGL_PACK'));
  /** Whether we will pack unary ops. */
  ENV.registerFlag('WEBGL_PACK_UNARY_OPERATIONS', () => ENV.getBool('WEBGL_PACK'));
  /** Whether we will pack array ops. */
  ENV.registerFlag('WEBGL_PACK_ARRAY_OPERATIONS', () => ENV.getBool('WEBGL_PACK'));
  /** Whether we will pack image ops. */
  ENV.registerFlag('WEBGL_PACK_IMAGE_OPERATIONS', () => ENV.getBool('WEBGL_PACK'));
  /** Whether we will pack reduce ops. */
  ENV.registerFlag('WEBGL_PACK_REDUCE', () => ENV.getBool('WEBGL_PACK'));
  /** Whether packed WebGL kernels lazily unpack their outputs. */
  ENV.registerFlag('WEBGL_LAZILY_UNPACK', () => ENV.getBool('WEBGL_PACK'));
  /** Whether we will use the im2col algorithm to speed up convolutions. */
  ENV.registerFlag('WEBGL_CONV_IM2COL', () => ENV.getBool('WEBGL_PACK'));
  /** The maximum texture dimension. */
  ENV.registerFlag('WEBGL_MAX_TEXTURE_SIZE', () => getWebGLMaxTextureSize(ENV.getNumber('WEBGL_VERSION')));
  /** The maximum texture dimension. */
  ENV.registerFlag('WEBGL_MAX_TEXTURES_IN_SHADER', () => getMaxTexturesInShader(ENV.getNumber('WEBGL_VERSION')));
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
  ENV.registerFlag('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', () => ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0 &&
      !tf.device_util.isMobile());
  /**
   * Whether the device is physically capable of rendering to float32 textures.
   */
  ENV.registerFlag('WEBGL_RENDER_FLOAT32_CAPABLE', () => isCapableOfRenderingToFloatTexture(ENV.getNumber('WEBGL_VERSION')));
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
  ENV.registerFlag('WEBGL_DOWNLOAD_FLOAT_ENABLED', () => isDownloadFloatTextureEnabled(ENV.getNumber('WEBGL_VERSION')));
  /** Whether the fence API is available. */
  ENV.registerFlag('WEBGL_FENCE_API_ENABLED', () => isWebGLFenceEnabled(ENV.getNumber('WEBGL_VERSION')));
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
  class AddNProgram {
      constructor(outputShape, shapes) {
          this.outputShape = [];
          this.outputShape = outputShape;
          this.variableNames = shapes.map((_, i) => `T${i}`);
          const snippets = [];
          // Get target elements from every input tensor.
          this.variableNames.forEach(variable => {
              snippets.push(`float v${variable} = get${variable}AtOutCoords();`);
          });
          // Calculate the sum of all elements.
          const operation = this.variableNames
              .map(variable => {
              return `v${variable}`;
          })
              .join(' + ');
          this.userCode = `
      void main() {
        ${snippets.join('\n        ')}

        float result = ${operation};
        setOutput(result);
      }
    `;
      }
  }

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
  class AddNPackedProgram {
      constructor(outputShape, shapes) {
          this.outputShape = [];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = outputShape;
          this.variableNames = shapes.map((_, i) => `T${i}`);
          const snippets = [];
          // Get target elements from every input tensor.
          this.variableNames.forEach(variable => {
              snippets.push(`vec4 v${variable} = get${variable}AtOutCoords();`);
          });
          // Calculate the sum of all elements.
          const operation = this.variableNames
              .map(variable => {
              return `v${variable}`;
          })
              .join(' + ');
          this.userCode = `
      void main() {
        ${snippets.join('\n        ')}

        vec4 result = ${operation};
        setOutput(result);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class ArgMinMaxProgram {
      constructor(reduceInfo, op, firstPass) {
          this.variableNames = ['A'];
          const windowSize = reduceInfo.windowSize;
          const batchSize = reduceInfo.batchSize;
          const inSize = reduceInfo.inSize;
          const outSize = Math.ceil(inSize / windowSize);
          if (!firstPass) {
              this.variableNames.push('bestIndicesA');
          }
          this.outputShape = [batchSize, outSize];
          const compOp = (op === 'max') ? '>' : '<';
          const indexSnippet = firstPass ?
              'inOffset + i;' :
              'round(getBestIndicesA(batch, inOffset + i));';
          this.userCode = `
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${windowSize};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${windowSize}; i++) {
          int inIdx = ${indexSnippet};
          float candidate = getA(batch, inIdx);
          if (candidate ${compOp} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `;
      }
  }

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
  function getVecChannels(name, rank) {
      return ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank).map(d => `${name}.${d}`);
  }
  function getChannels(name, rank) {
      if (rank === 1) {
          return [name];
      }
      return getVecChannels(name, rank);
  }
  function getSourceCoords(rank, dims) {
      if (rank === 1) {
          return 'rc';
      }
      let coords = '';
      for (let i = 0; i < rank; i++) {
          coords += dims[i];
          if (i < rank - 1) {
              coords += ',';
          }
      }
      return coords;
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  function getGlslDifferences() {
      let version;
      let attribute;
      let varyingVs;
      let varyingFs;
      let texture2D;
      let output;
      let defineOutput;
      let defineSpecialNaN;
      let defineSpecialInf;
      let defineRound;
      if (tf.env().getNumber('WEBGL_VERSION') === 2) {
          version = '#version 300 es';
          attribute = 'in';
          varyingVs = 'out';
          varyingFs = 'in';
          texture2D = 'texture';
          output = 'outputColor';
          defineOutput = 'out vec4 outputColor;';
          // Use custom isnan definition to work across differences between
          // implementations on various platforms. While this should happen in ANGLE
          // we still see differences between android and windows (on chrome) when
          // using isnan directly.
          defineSpecialNaN = `
      bool isnan_custom(float val) {
        return (val > 0.0 || val < 0.0) ? false : val != 0.0;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `;
          // In webgl 2 we do not need to specify a custom isinf so there is no
          // need for a special INFINITY constant.
          defineSpecialInf = ``;
          defineRound = `
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `;
      }
      else {
          version = '';
          attribute = 'attribute';
          varyingVs = 'varying';
          varyingFs = 'varying';
          texture2D = 'texture2D';
          output = 'gl_FragColor';
          defineOutput = '';
          // WebGL1 has no built in isnan so we define one here.
          defineSpecialNaN = `
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `;
          defineSpecialInf = `
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `;
          defineRound = `
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `;
      }
      return {
          version,
          attribute,
          varyingVs,
          varyingFs,
          texture2D,
          output,
          defineOutput,
          defineSpecialNaN,
          defineSpecialInf,
          defineRound
      };
  }

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
  /**
   * Produces GLSL code that derives logical coordinates from a flat
   * index. The code performs integer division with each stride and decrements
   * the index until the index equals the final dimension coordinate.
   */
  function getLogicalCoordinatesFromFlatIndex(coords, shape, index = 'index') {
      const strides = tf.util.computeStrides(shape);
      return strides
          .map((stride, i) => {
          const line1 = `int ${coords[i]} = ${index} / ${stride}`;
          const line2 = i === strides.length - 1 ?
              `int ${coords[i + 1]} = ${index} - ${coords[i]} * ${stride}` :
              `index -= ${coords[i]} * ${stride}`;
          return `${line1}; ${line2};`;
      })
          .join('');
  }
  /**
   * Produces GLSL that computes the flat index from 3D coordinates.
   */
  function getFlatIndexFrom3D(shape) {
      const strides = tf.util.computeStrides(shape).map(d => d.toString());
      return `
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${strides[0]} + coords.y * ${strides[1]} + coords.z;
  }
`;
  }
  const ENCODE_FLOAT_SNIPPET = `
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`;

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  const { getBroadcastDims } = tf.backend_util;
  function makeShader(inputsInfo, outputShape, userCode, usesPackedTextures) {
      const prefixSnippets = [];
      inputsInfo.forEach(x => {
          const size = tf.util.sizeFromShape(x.shapeInfo.logicalShape);
          // Snippet when we decided to upload the values as uniform.
          if (x.shapeInfo.isUniform) {
              prefixSnippets.push(`uniform float ${x.name}${size > 1 ? `[${size}]` : ''};`);
          }
          else {
              prefixSnippets.push(`uniform sampler2D ${x.name};`);
              prefixSnippets.push(`uniform int offset${x.name};`);
          }
      });
      const inputPrefixSnippet = prefixSnippets.join('\n');
      const inputSamplingSnippet = inputsInfo
          .map(x => getInputSamplingSnippet(x, outputShape, usesPackedTextures))
          .join('\n');
      const outTexShape = outputShape.texShape;
      const glsl = getGlslDifferences();
      const floatTextureSampleSnippet = getFloatTextureSampleSnippet(glsl);
      let outputSamplingSnippet;
      let floatTextureSetOutputSnippet;
      let shaderPrefix = getShaderPrefix(glsl);
      if (outputShape.isPacked) {
          outputSamplingSnippet =
              getPackedOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
          floatTextureSetOutputSnippet = getFloatTextureSetRGBASnippet(glsl);
      }
      else {
          outputSamplingSnippet =
              getOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
          floatTextureSetOutputSnippet = getFloatTextureSetRSnippet(glsl);
      }
      if (usesPackedTextures) {
          shaderPrefix += SHADER_PACKED_PREFIX;
      }
      const source = [
          shaderPrefix, floatTextureSampleSnippet, floatTextureSetOutputSnippet,
          inputPrefixSnippet, outputSamplingSnippet, inputSamplingSnippet, userCode
      ].join('\n');
      return source;
  }
  function getSamplerFromInInfo(inInfo) {
      const shape = inInfo.shapeInfo.logicalShape;
      switch (shape.length) {
          case 0:
              return getSamplerScalar(inInfo);
          case 1:
              return getSampler1D(inInfo);
          case 2:
              return getSampler2D(inInfo);
          case 3:
              return getSampler3D(inInfo);
          case 4:
              return getSampler4D(inInfo);
          case 5:
              return getSampler5D(inInfo);
          case 6:
              return getSampler6D(inInfo);
          default:
              throw new Error(`${shape.length}-D input sampling` +
                  ` is not yet supported`);
      }
  }
  function getPackedSamplerFromInInfo(inInfo) {
      const shape = inInfo.shapeInfo.logicalShape;
      switch (shape.length) {
          case 0:
              return getPackedSamplerScalar(inInfo);
          case 1:
              return getPackedSampler1D(inInfo);
          case 2:
              return getPackedSampler2D(inInfo);
          case 3:
              return getPackedSampler3D(inInfo);
          default:
              return getPackedSamplerND(inInfo);
      }
  }
  function getInputSamplingSnippet(inInfo, outShapeInfo, usesPackedTextures = false) {
      let res = '';
      if (usesPackedTextures) {
          res += getPackedSamplerFromInInfo(inInfo);
      }
      else {
          res += getSamplerFromInInfo(inInfo);
      }
      const inShape = inInfo.shapeInfo.logicalShape;
      const outShape = outShapeInfo.logicalShape;
      if (inShape.length <= outShape.length) {
          if (usesPackedTextures) {
              res += getPackedSamplerAtOutputCoords(inInfo, outShapeInfo);
          }
          else {
              res += getSamplerAtOutputCoords(inInfo, outShapeInfo);
          }
      }
      return res;
  }
  function getPackedOutputSamplingSnippet(outShape, outTexShape) {
      switch (outShape.length) {
          case 0:
              return getOutputScalarCoords();
          case 1:
              return getOutputPacked1DCoords(outShape, outTexShape);
          case 2:
              return getOutputPacked2DCoords(outShape, outTexShape);
          case 3:
              return getOutputPacked3DCoords(outShape, outTexShape);
          default:
              return getOutputPackedNDCoords(outShape, outTexShape);
      }
  }
  function getOutputSamplingSnippet(outShape, outTexShape) {
      switch (outShape.length) {
          case 0:
              return getOutputScalarCoords();
          case 1:
              return getOutput1DCoords(outShape, outTexShape);
          case 2:
              return getOutput2DCoords(outShape, outTexShape);
          case 3:
              return getOutput3DCoords(outShape, outTexShape);
          case 4:
              return getOutput4DCoords(outShape, outTexShape);
          case 5:
              return getOutput5DCoords(outShape, outTexShape);
          case 6:
              return getOutput6DCoords(outShape, outTexShape);
          default:
              throw new Error(`${outShape.length}-D output sampling is not yet supported`);
      }
  }
  function getFloatTextureSampleSnippet(glsl) {
      return `
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${glsl.texture2D}(textureSampler, uv).r;
    }
  `;
  }
  function getFloatTextureSetRSnippet(glsl) {
      return `
    void setOutput(float val) {
      ${glsl.output} = vec4(val, 0, 0, 0);
    }
  `;
  }
  function getFloatTextureSetRGBASnippet(glsl) {
      return `
    void setOutput(vec4 val) {
      ${glsl.output} = val;
    }
  `;
  }
  function getShaderPrefix(glsl) {
      const SHADER_PREFIX = `${glsl.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${glsl.varyingFs} vec2 resultUV;
    ${glsl.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${glsl.defineSpecialNaN}
    ${glsl.defineSpecialInf}
    ${glsl.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${SAMPLE_1D_SNIPPET}
    ${SAMPLE_2D_SNIPPET}
    ${SAMPLE_3D_SNIPPET}
  `;
      return SHADER_PREFIX;
  }
  const SAMPLE_1D_SNIPPET = `
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;
  const SAMPLE_2D_SNIPPET = `
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;
  const SAMPLE_3D_SNIPPET = `
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;
  const SHADER_PACKED_PREFIX = `
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;
  function getOutputScalarCoords() {
      return `
    int getOutputCoords() {
      return 0;
    }
  `;
  }
  function getOutputPacked1DCoords(shape, texShape) {
      const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
      if (packedTexShape[0] === 1) {
          return `
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${packedTexShape[1]}.0);
      }
    `;
      }
      if (packedTexShape[1] === 1) {
          return `
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${packedTexShape[0]}.0);
      }
    `;
      }
      return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      return 2 * (resTexRC.x * ${packedTexShape[1]} + resTexRC.y);
    }
  `;
  }
  function getOutput1DCoords(shape, texShape) {
      if (texShape[0] === 1) {
          return `
      int getOutputCoords() {
        return int(resultUV.x * ${texShape[1]}.0);
      }
    `;
      }
      if (texShape[1] === 1) {
          return `
      int getOutputCoords() {
        return int(resultUV.y * ${texShape[0]}.0);
      }
    `;
      }
      return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      return resTexRC.x * ${texShape[1]} + resTexRC.y;
    }
  `;
  }
  function getOutputPacked3DCoords(shape, texShape) {
      const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
      const texelsInLogicalRow = Math.ceil(shape[2] / 2);
      const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[1] / 2);
      return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec3(b, r, c);
    }
  `;
  }
  function getOutput3DCoords(shape, texShape) {
      const coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);
      return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec3(r, c, d);
    }
  `;
  }
  function getOutputPackedNDCoords(shape, texShape) {
      const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
      const texelsInLogicalRow = Math.ceil(shape[shape.length - 1] / 2);
      const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[shape.length - 2] / 2);
      let texelsInBatchN = texelsInBatch;
      let batches = ``;
      let coords = 'b, r, c';
      for (let b = 2; b < shape.length - 1; b++) {
          texelsInBatchN *= shape[shape.length - b - 1];
          batches = `
      int b${b} = index / ${texelsInBatchN};
      index -= b${b} * ${texelsInBatchN};
    ` + batches;
          coords = `b${b}, ` + coords;
      }
      return `
    ivec${shape.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      ${batches}

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec${shape.length}(${coords});
    }
  `;
  }
  function getOutput4DCoords(shape, texShape) {
      const coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2'], shape);
      return `
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec4(r, c, d, d2);
    }
  `;
  }
  function getOutput5DCoords(shape, texShape) {
      const coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2', 'd3'], shape);
      return `
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${texShape[0]},
                             ${texShape[1]}));

      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      ${coordsFromIndexSnippet}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `;
  }
  function getOutput6DCoords(shape, texShape) {
      const coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2', 'd3', 'd4'], shape);
      return `
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      ${coordsFromIndexSnippet}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `;
  }
  function getOutputPacked2DCoords(shape, texShape) {
      const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
      if (tf.util.arraysEqual(shape, texShape)) {
          return `
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      }
    `;
      }
      // texels needed to accommodate a logical row
      const texelsInLogicalRow = Math.ceil(shape[1] / 2);
      /**
       * getOutputCoords
       *
       * resTexRC: The rows and columns of the texels. If you move over one
       * texel to the right in the packed texture, you are moving over one column
       * (not two).
       *
       * index: The texel index
       */
      return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));

      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec2(r, c);
    }
  `;
  }
  function getOutput2DCoords(shape, texShape) {
      if (tf.util.arraysEqual(shape, texShape)) {
          return `
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${texShape[0]}, ${texShape[1]}));
      }
    `;
      }
      if (shape[1] === 1) {
          return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `;
      }
      if (shape[0] === 1) {
          return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `;
      }
      return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      int r = index / ${shape[1]};
      int c = index - r * ${shape[1]};
      return ivec2(r, c);
    }
  `;
  }
  function getFlatOffsetUniformName(texName) {
      return `offset${texName}`;
  }
  function getPackedSamplerScalar(inputInfo) {
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const glsl = getGlslDifferences();
      return `
    vec4 ${funcName}() {
      return ${glsl.texture2D}(${texName}, halfCR);
    }
  `;
  }
  function getSamplerScalar(inputInfo) {
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      if (inputInfo.shapeInfo.isUniform) {
          return `float ${funcName}() {return ${texName};}`;
      }
      const [texNumR, texNumC] = inputInfo.shapeInfo.texShape;
      if (texNumR === 1 && texNumC === 1) {
          return `
      float ${funcName}() {
        return sampleTexture(${texName}, halfCR);
      }
    `;
      }
      const [tNumR, tNumC] = inputInfo.shapeInfo.texShape;
      const offset = getFlatOffsetUniformName(texName);
      return `
    float ${funcName}() {
      vec2 uv = uvFromFlat(${tNumR}, ${tNumC}, ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  function getPackedSampler1D(inputInfo) {
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const texShape = inputInfo.shapeInfo.texShape;
      const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
      const glsl = getGlslDifferences();
      return `
    vec4 ${funcName}(int index) {
      vec2 uv = packedUVfrom1D(
        ${packedTexShape[0]}, ${packedTexShape[1]}, index);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
  }
  function getSampler1D(inputInfo) {
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      if (inputInfo.shapeInfo.isUniform) {
          // Uniform arrays will be less than 65505 (no risk of float16 overflow).
          return `
      float ${funcName}(int index) {
        ${getUniformSampler(inputInfo)}
      }
    `;
      }
      const texShape = inputInfo.shapeInfo.texShape;
      const tNumR = texShape[0];
      const tNumC = texShape[1];
      if (tNumC === 1 && tNumR === 1) {
          return `
      float ${funcName}(int index) {
        return sampleTexture(${texName}, halfCR);
      }
    `;
      }
      const offset = getFlatOffsetUniformName(texName);
      if (tNumC === 1) {
          return `
      float ${funcName}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${offset}) + 0.5) / ${tNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
      }
      if (tNumR === 1) {
          return `
      float ${funcName}(int index) {
        vec2 uv = vec2((float(index + ${offset}) + 0.5) / ${tNumC}.0, 0.5);
        return sampleTexture(${texName}, uv);
      }
    `;
      }
      return `
    float ${funcName}(int index) {
      vec2 uv = uvFromFlat(${tNumR}, ${tNumC}, index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  function getPackedSampler2D(inputInfo) {
      const shape = inputInfo.shapeInfo.logicalShape;
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const texShape = inputInfo.shapeInfo.texShape;
      const texNumR = texShape[0];
      const texNumC = texShape[1];
      const glsl = getGlslDifferences();
      if (texShape != null && tf.util.arraysEqual(shape, texShape)) {
          return `
      vec4 ${funcName}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);

        return ${glsl.texture2D}(${texName}, uv);
      }
    `;
      }
      const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
      const valuesPerRow = Math.ceil(shape[1] / 2);
      return `
    vec4 ${funcName}(int row, int col) {
      vec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${packedTexShape[1]}, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
  }
  function getSampler2D(inputInfo) {
      const shape = inputInfo.shapeInfo.logicalShape;
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const texShape = inputInfo.shapeInfo.texShape;
      if (texShape != null && tf.util.arraysEqual(shape, texShape)) {
          const texNumR = texShape[0];
          const texNumC = texShape[1];
          return `
    float ${funcName}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
      }
      const { newShape, keptDims } = tf.util.squeezeShape(shape);
      const squeezedShape = newShape;
      if (squeezedShape.length < shape.length) {
          const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
          const params = ['row', 'col'];
          return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
      }
      if (inputInfo.shapeInfo.isUniform) {
          // Uniform arrays will be less than 65505 (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${shape[1]}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
      }
      const texNumR = texShape[0];
      const texNumC = texShape[1];
      const offset = getFlatOffsetUniformName(texName);
      if (texNumC === 1) {
          // index is used directly as physical (no risk of float16 overflow).
          return `
    float ${funcName}(int row, int col) {
      float index = dot(vec3(row, col, ${offset}), vec3(${shape[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
      }
      if (texNumR === 1) {
          // index is used directly as physical (no risk of float16 overflow).
          return `
    float ${funcName}(int row, int col) {
      float index = dot(vec3(row, col, ${offset}), vec3(${shape[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${texNumC}.0, 0.5);
      return sampleTexture(${texName}, uv);
    }
  `;
      }
      return `
  float ${funcName}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${shape[1]} + col + ${offset};
    vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
    return sampleTexture(${texName}, uv);
  }
`;
  }
  function getPackedSampler3D(inputInfo) {
      const shape = inputInfo.shapeInfo.logicalShape;
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const texShape = inputInfo.shapeInfo.texShape;
      const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
      if (shape[0] === 1) {
          const squeezedShape = shape.slice(1);
          const keptDims = [1, 2];
          const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
          const params = ['b', 'row', 'col'];
          return `
        ${getPackedSamplerFromInInfo(newInputInfo)}
        vec4 ${funcName}(int b, int row, int col) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
      }
      const texNumR = packedTexShape[0];
      const texNumC = packedTexShape[1];
      const valuesPerRow = Math.ceil(shape[2] / 2);
      const texelsInBatch = valuesPerRow * Math.ceil(shape[1] / 2);
      const glsl = getGlslDifferences();
      return `
    vec4 ${funcName}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${valuesPerRow}, b, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
  }
  function getSampler3D(inputInfo) {
      const shape = inputInfo.shapeInfo.logicalShape;
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const stride0 = shape[1] * shape[2];
      const stride1 = shape[2];
      const { newShape, keptDims } = tf.util.squeezeShape(shape);
      const squeezedShape = newShape;
      if (squeezedShape.length < shape.length) {
          const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
          const params = ['row', 'col', 'depth'];
          return `
        ${getSamplerFromInInfo(newInputInfo)}
        float ${funcName}(int row, int col, int depth) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
      }
      if (inputInfo.shapeInfo.isUniform) {
          // Uniform arrays will be less than 65505 (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${stride0}, ${stride1}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
      }
      const texShape = inputInfo.shapeInfo.texShape;
      const texNumR = texShape[0];
      const texNumC = texShape[1];
      const flatOffset = inputInfo.shapeInfo.flatOffset;
      if (texNumC === stride0 && flatOffset == null) {
          // texC is used directly as physical (no risk of float16 overflow).
          return `
        float ${funcName}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${stride1}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${texNumC}.0, ${texNumR}.0);
          return sampleTexture(${texName}, uv);
        }
      `;
      }
      if (texNumC === stride1 && flatOffset == null) {
          // texR is used directly as physical (no risk of float16 overflow).
          return `
    float ${funcName}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${shape[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
      }
      const offset = getFlatOffsetUniformName(texName);
      return `
      float ${funcName}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${stride0} + col * ${stride1} + depth + ${offset};
        vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
        return sampleTexture(${texName}, uv);
      }
  `;
  }
  function getPackedSamplerND(inputInfo) {
      const shape = inputInfo.shapeInfo.logicalShape;
      const rank = shape.length;
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const texShape = inputInfo.shapeInfo.texShape;
      const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
      const texNumR = packedTexShape[0];
      const texNumC = packedTexShape[1];
      const valuesPerRow = Math.ceil(shape[rank - 1] / 2);
      let texelsInBatch = valuesPerRow * Math.ceil(shape[rank - 2] / 2);
      let params = `int b, int row, int col`;
      let index = `b * ${texelsInBatch} + (row / 2) * ${valuesPerRow} + (col / 2)`;
      for (let b = 2; b < rank - 1; b++) {
          params = `int b${b}, ` + params;
          texelsInBatch *= shape[rank - b - 1];
          index = `b${b} * ${texelsInBatch} + ` + index;
      }
      const glsl = getGlslDifferences();
      return `
    vec4 ${funcName}(${params}) {
      int index = ${index};
      int texR = index / ${texNumC};
      int texC = index - texR * ${texNumC};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}, ${texNumR});
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
  }
  function getSampler4D(inputInfo) {
      const shape = inputInfo.shapeInfo.logicalShape;
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const stride2 = shape[3];
      const stride1 = shape[2] * stride2;
      const stride0 = shape[1] * stride1;
      const { newShape, keptDims } = tf.util.squeezeShape(shape);
      if (newShape.length < shape.length) {
          const newInputInfo = squeezeInputInfo(inputInfo, newShape);
          const params = ['row', 'col', 'depth', 'depth2'];
          return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth, int depth2) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
      }
      if (inputInfo.shapeInfo.isUniform) {
          // Uniform arrays will be less than 65505 (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${stride0}, ${stride1}, ${stride2}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
      }
      const flatOffset = inputInfo.shapeInfo.flatOffset;
      const texShape = inputInfo.shapeInfo.texShape;
      const texNumR = texShape[0];
      const texNumC = texShape[1];
      if (texNumC === stride0 && flatOffset == null) {
          // texC is used directly as physical (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${stride1}, ${stride2}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
      }
      if (texNumC === stride2 && flatOffset == null) {
          // texR is used directly as physical (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${shape[1] * shape[2]}, ${shape[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
      }
      const offset = getFlatOffsetUniformName(texName);
      return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} +
          depth * ${stride2} + depth2;
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  function getSampler5D(inputInfo) {
      const shape = inputInfo.shapeInfo.logicalShape;
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const stride3 = shape[4];
      const stride2 = shape[3] * stride3;
      const stride1 = shape[2] * stride2;
      const stride0 = shape[1] * stride1;
      const { newShape, keptDims } = tf.util.squeezeShape(shape);
      if (newShape.length < shape.length) {
          const newInputInfo = squeezeInputInfo(inputInfo, newShape);
          const params = ['row', 'col', 'depth', 'depth2', 'depth3'];
          return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
      }
      if (inputInfo.shapeInfo.isUniform) {
          // Uniform arrays will be less than 65505 (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${stride0}, ${stride1}, ${stride2}, ${stride3})) +
          depth3;
        ${getUniformSampler(inputInfo)}
      }
    `;
      }
      const flatOffset = inputInfo.shapeInfo.flatOffset;
      const texShape = inputInfo.shapeInfo.texShape;
      const texNumR = texShape[0];
      const texNumC = texShape[1];
      if (texNumC === stride0 && flatOffset == null) {
          // texC is used directly as physical (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${stride1}, ${stride2}, ${stride3}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
      }
      if (texNumC === stride3 && flatOffset == null) {
          // texR is used directly as physical (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${shape[1] * shape[2] * shape[3]},
               ${shape[2] * shape[3]}, ${shape[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
      }
      const offset = getFlatOffsetUniformName(texName);
      return `
    float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} + depth * ${stride2} +
          depth2 * ${stride3} + depth3 + ${offset};
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  function getSampler6D(inputInfo) {
      const shape = inputInfo.shapeInfo.logicalShape;
      const texName = inputInfo.name;
      const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
      const { newShape, keptDims } = tf.util.squeezeShape(shape);
      if (newShape.length < shape.length) {
          const newInputInfo = squeezeInputInfo(inputInfo, newShape);
          const params = ['row', 'col', 'depth', 'depth2', 'depth3', 'depth4'];
          return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
      }
      const stride4 = shape[5];
      const stride3 = shape[4] * stride4;
      const stride2 = shape[3] * stride3;
      const stride1 = shape[2] * stride2;
      const stride0 = shape[1] * stride1;
      if (inputInfo.shapeInfo.isUniform) {
          // Uniform arrays will be less than 65505 (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${stride0}, ${stride1}, ${stride2}, ${stride3})) +
          dot(
            vec2(depth3, depth4),
            vec2(${stride4}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
      }
      const flatOffset = inputInfo.shapeInfo.flatOffset;
      const texShape = inputInfo.shapeInfo.texShape;
      const texNumR = texShape[0];
      const texNumC = texShape[1];
      if (texNumC === stride0 && flatOffset == null) {
          // texC is used directly as physical (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${stride1}, ${stride2}, ${stride3}, ${stride4})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
      }
      if (texNumC === stride4 && flatOffset == null) {
          // texR is used directly as physical (no risk of float16 overflow).
          return `
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${shape[1] * shape[2] * shape[3] * shape[4]},
               ${shape[2] * shape[3] * shape[4]},
               ${shape[3] * shape[4]},
               ${shape[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
      }
      const offset = getFlatOffsetUniformName(texName);
      return `
    float ${funcName}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} + depth * ${stride2} +
          depth2 * ${stride3} + depth3 * ${stride4} + depth4 + ${offset};
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  function getUniformSampler(inputInfo) {
      const texName = inputInfo.name;
      const inSize = tf.util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
      if (inSize < 2) {
          return `return ${texName};`;
      }
      return `
    for (int i = 0; i < ${inSize}; i++) {
      if (i == index) {
        return ${texName}[i];
      }
    }
  `;
  }
  function getPackedSamplerAtOutputCoords(inputInfo, outShapeInfo) {
      const texName = inputInfo.name;
      const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
      const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
      const inRank = inputInfo.shapeInfo.logicalShape.length;
      const outRank = outShapeInfo.logicalShape.length;
      const broadcastDims = getBroadcastDims(inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
      const type = getCoordsDataType(outRank);
      const rankDiff = outRank - inRank;
      let coordsSnippet;
      const fields = ['x', 'y', 'z', 'w', 'u', 'v'];
      if (inRank === 0) {
          coordsSnippet = '';
      }
      else if (outRank < 2 && broadcastDims.length >= 1) {
          coordsSnippet = 'coords = 0;';
      }
      else {
          coordsSnippet =
              broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`)
                  .join('\n');
      }
      let unpackedCoordsSnippet = '';
      if (outRank < 2 && inRank > 0) {
          unpackedCoordsSnippet = 'coords';
      }
      else {
          unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
              .map((s, i) => `coords.${fields[i + rankDiff]}`)
              .join(', ');
      }
      let output = `return outputValue;`;
      const inSize = tf.util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
      const isInputScalar = inSize === 1;
      const outSize = tf.util.sizeFromShape(outShapeInfo.logicalShape);
      const isOutputScalar = outSize === 1;
      if (inRank === 1 && !isInputScalar && !isOutputScalar) {
          output = `
      return vec4(outputValue.xy, outputValue.xy);
    `;
      }
      else if (isInputScalar && !isOutputScalar) {
          if (outRank === 1) {
              output = `
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `;
          }
          else {
              output = `
        return vec4(outputValue.x);
      `;
          }
      }
      else if (broadcastDims.length) {
          const rows = inRank - 2;
          const cols = inRank - 1;
          if (broadcastDims.indexOf(rows) > -1 && broadcastDims.indexOf(cols) > -1) {
              output = `return vec4(outputValue.x);`;
          }
          else if (broadcastDims.indexOf(rows) > -1) {
              output = `return vec4(outputValue.x, outputValue.y, ` +
                  `outputValue.x, outputValue.y);`;
          }
          else if (broadcastDims.indexOf(cols) > -1) {
              output = `return vec4(outputValue.xx, outputValue.zz);`;
          }
      }
      return `
    vec4 ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      vec4 outputValue = get${texFuncSnippet}(${unpackedCoordsSnippet});
      ${output}
    }
  `;
  }
  function getSamplerAtOutputCoords(inputInfo, outShapeInfo) {
      const texName = inputInfo.name;
      const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
      const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
      const outTexShape = outShapeInfo.texShape;
      const inTexShape = inputInfo.shapeInfo.texShape;
      const inRank = inputInfo.shapeInfo.logicalShape.length;
      const outRank = outShapeInfo.logicalShape.length;
      if (!inputInfo.shapeInfo.isUniform && inRank === outRank &&
          inputInfo.shapeInfo.flatOffset == null &&
          tf.util.arraysEqual(inTexShape, outTexShape)) {
          return `
      float ${funcName}() {
        return sampleTexture(${texName}, resultUV);
      }
    `;
      }
      const type = getCoordsDataType(outRank);
      const broadcastDims = getBroadcastDims(inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
      const rankDiff = outRank - inRank;
      let coordsSnippet;
      const fields = ['x', 'y', 'z', 'w', 'u', 'v'];
      if (inRank === 0) {
          coordsSnippet = '';
      }
      else if (outRank < 2 && broadcastDims.length >= 1) {
          coordsSnippet = 'coords = 0;';
      }
      else {
          coordsSnippet =
              broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`)
                  .join('\n');
      }
      let unpackedCoordsSnippet = '';
      if (outRank < 2 && inRank > 0) {
          unpackedCoordsSnippet = 'coords';
      }
      else {
          unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
              .map((s, i) => `coords.${fields[i + rankDiff]}`)
              .join(', ');
      }
      return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      return get${texFuncSnippet}(${unpackedCoordsSnippet});
    }
  `;
  }
  function getCoordsDataType(rank) {
      if (rank <= 1) {
          return 'int';
      }
      else if (rank === 2) {
          return 'ivec2';
      }
      else if (rank === 3) {
          return 'ivec3';
      }
      else if (rank === 4) {
          return 'ivec4';
      }
      else if (rank === 5) {
          return 'ivec5';
      }
      else if (rank === 6) {
          return 'ivec6';
      }
      else {
          throw Error(`GPU for rank ${rank} is not yet supported`);
      }
  }
  /** Returns a new input info (a copy) that has a squeezed logical shape. */
  function squeezeInputInfo(inInfo, squeezedShape) {
      // Deep copy.
      const newInputInfo = JSON.parse(JSON.stringify(inInfo));
      newInputInfo.shapeInfo.logicalShape = squeezedShape;
      return newInputInfo;
  }
  function getSqueezedParams(params, keptDims) {
      return keptDims.map(d => params[d]).join(', ');
  }

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
  class ArgMinMaxPackedProgram {
      constructor(shape, windowSize, op, firstPass) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = true;
          tf.util.assert(shape.length > 2, () => `Packed arg${op.charAt(0).toUpperCase() +
            op.slice(1)} supports only inputs with rank above 2.`);
          const inSize = shape[shape.length - 1];
          const outSize = Math.ceil(inSize / windowSize);
          this.outputShape = shape.slice(0, -1);
          if (outSize > 1) {
              this.outputShape.push(outSize);
          }
          if (!firstPass) {
              this.variableNames.push('bestIndicesA');
          }
          const outShape = this.outputShape;
          const rank = outShape.length;
          const dtype = getCoordsDataType(rank);
          const coords = getChannels('coords', rank);
          let sourceLocSetup;
          let sourceRank;
          if (outSize === 1) {
              sourceRank = rank + 1;
              const sourceLocDType = getCoordsDataType(sourceRank);
              sourceLocSetup = `
        ${sourceLocDType} sourceLocR = ${sourceLocDType}(${coords.join()}, 0);
        ++${coords[rank - 1]};
        ${sourceLocDType} sourceLocG = ${sourceLocDType}(${coords.join()}, 0);
        ++${coords[rank - 2]};
        ${sourceLocDType} sourceLocA = ${sourceLocDType}(${coords.join()}, 0);
        --${coords[rank - 1]};
        ${sourceLocDType} sourceLocB = ${sourceLocDType}(${coords.join()}, 0);
        --${coords[rank - 2]};`;
          }
          else {
              sourceRank = rank;
              sourceLocSetup = `
        ${dtype} sourceLocR = coords;
        ++${coords[rank - 1]};
        ${dtype} sourceLocG = coords;
        ++${coords[rank - 2]};
        ${dtype} sourceLocA = coords;
        --${coords[rank - 1]};
        ${dtype} sourceLocB = coords;
        --${coords[rank - 2]};`;
          }
          const channels = ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, sourceRank);
          const inChannel = '.' + channels[sourceRank - 1]; // e.g. ".b" for rank 3.
          const intChannels = channels.map(x => 'int ' + x);
          const srcRCoords = getChannels('sourceLocR', sourceRank - 1).concat('inIdx.r');
          const srcGCoords = getChannels('sourceLocG', sourceRank - 1).concat('inIdx.g');
          const srcBCoords = getChannels('sourceLocB', sourceRank - 1).concat('inIdx.b');
          const srcACoords = getChannels('sourceLocA', sourceRank - 1).concat('inIdx.a');
          const compOp = (op === 'max') ? 'greaterThan' : 'lessThan';
          const fetchCandidateIdx = firstPass ? '' : `
          inIdx = round(vec4(getBestIndicesAChannel(${srcRCoords.join()}),
                             getBestIndicesAChannel(${srcGCoords.join()}),
                             getBestIndicesAChannel(${srcBCoords.join()}),
                             getBestIndicesAChannel(${srcACoords.join()})));`;
          const fetchValue = `vec4(
            getAChannel(${srcRCoords.join()}),
            hasNextCol ? getAChannel(${srcGCoords.join()}) : 0.,
            hasNextRow ? getAChannel(${srcBCoords.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${srcACoords.join()}) : 0.)`;
          const getBestIndicesAChannelSnippet = firstPass ? '' : `
      float getBestIndicesAChannel(${intChannels.join()}) {
        return getChannel(getBestIndicesA(${channels.join()}),
                                          vec2(${channels.slice(-2).join()}));
      }`;
          this.userCode = `
      float getAChannel(${intChannels.join()}) {
        return getChannel(getA(${channels.join()}),
                               vec2(${channels.slice(-2).join()}));
      }
      ${getBestIndicesAChannelSnippet}
      void main() {
        ${dtype} coords = getOutputCoords();
        bool hasNextCol = ${coords[rank - 1]} < ${outShape[rank - 1] - 1};
        bool hasNextRow = ${coords[rank - 2]} < ${outShape[rank - 2] - 1};
        ${sourceLocSetup}
        ivec4 srcIdx = ivec4(sourceLocR${inChannel}, sourceLocG${inChannel},
          sourceLocB${inChannel}, sourceLocA${inChannel}) * ${windowSize};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${fetchValue};

        for (int i = 0; i < ${windowSize}; i++) {
          inIdx = srcIdx;
          ${fetchCandidateIdx}
          vec4 candidate = ${fetchValue};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${compOp}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class AvgPool2DBackpropProgram {
      constructor(convInfo) {
          this.variableNames = ['dy'];
          this.outputShape = convInfo.inShape;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          const avgMultiplier = 1 / (filterHeight * filterWidth);
          this.userCode = `
      const ivec2 pads = ivec2(${padTop}, ${padLeft});
      const float avgMultiplier = float(${avgMultiplier});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${effectiveFilterHeight};
            wR += ${dilationHeight}) {
          float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

          if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${effectiveFilterWidth};
            wC+= ${dilationWidth}) {
            float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

            if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }
  class AvgPool3DBackpropProgram {
      constructor(convInfo) {
          this.variableNames = ['dy'];
          this.outputShape = convInfo.inShape;
          const filterDepth = convInfo.filterDepth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterDepth = convInfo.effectiveFilterDepth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
          const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          const avgMultiplier = 1 / (filterDepth * filterHeight * filterWidth);
          this.userCode = `
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});
      const float avgMultiplier = float(${avgMultiplier});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${effectiveFilterDepth};
            wD += ${dilationDepth}) {
          float dyD = float(dyDCorner + wD) / ${strideDepth}.0;

          if (dyD < 0.0 || dyD >= ${convInfo.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${effectiveFilterHeight};
              wR += ${dilationHeight}) {
            float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

            if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${effectiveFilterWidth};
                wC += ${dilationWidth}) {
              float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

              if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class BatchNormProgram {
      constructor(xShape, meanShape, varianceShape, offsetShape, scaleShape, varianceEpsilon) {
          this.outputShape = [];
          this.variableNames = ['x', 'mean', 'variance'];
          tf.backend_util.assertAndGetBroadcastShape(xShape, meanShape);
          tf.backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
          let offsetSnippet = '0.0';
          if (offsetShape != null) {
              tf.backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
              this.variableNames.push('offset');
              offsetSnippet = 'getOffsetAtOutCoords()';
          }
          let scaleSnippet = '1.0';
          if (scaleShape != null) {
              tf.backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
              this.variableNames.push('scale');
              scaleSnippet = 'getScaleAtOutCoords()';
          }
          this.outputShape = xShape;
          this.userCode = `
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${offsetSnippet};
        float scale = ${scaleSnippet};
        float inv = scale * inversesqrt(variance + float(${varianceEpsilon}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `;
      }
  }

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
  class BatchNormPackedProgram {
      constructor(xShape, meanShape, varianceShape, offsetShape, scaleShape, varianceEpsilon) {
          this.packedInputs = true;
          this.packedOutput = true;
          this.variableNames = ['x', 'mean', 'variance'];
          tf.backend_util.assertAndGetBroadcastShape(xShape, meanShape);
          tf.backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
          let offsetSnippet = 'vec4(0.0)';
          if (offsetShape != null) {
              tf.backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
              this.variableNames.push('offset');
              offsetSnippet = 'getOffsetAtOutCoords()';
          }
          let scaleSnippet = 'vec4(1.0)';
          if (scaleShape != null) {
              tf.backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
              this.variableNames.push('scale');
              scaleSnippet = 'getScaleAtOutCoords()';
          }
          this.outputShape = xShape;
          this.userCode = `
      void main() {
        vec4 offset = ${offsetSnippet};
        vec4 scale = ${scaleSnippet};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${varianceEpsilon}));

        setOutput((x - mean) * inv + offset);
      }
    `;
      }
  }

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
  // (Ar + Ai)(Br + Bi) =
  // ArBr + ArBi + AiBr + AiBi = ArBr - AB + ArBi + AiBr
  // Yr = ArBr - AB
  // Yi = ArBi + AiBr
  const COMPLEX_MULTIPLY = {
      REAL: 'return areal * breal - aimag * bimag;',
      IMAG: 'return areal * bimag + aimag * breal;'
  };
  class BinaryOpComplexProgram {
      constructor(op, aShape, bShape) {
          this.variableNames = ['AReal', 'AImag', 'BReal', 'BImag'];
          this.outputShape = tf.backend_util.assertAndGetBroadcastShape(aShape, bShape);
          this.userCode = `
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${op}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  const CHECK_NAN_SNIPPET = `
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;
  const ADD = 'return a + b;';
  const SUB = 'return a - b;';
  const MUL = 'return a * b;';
  // Without the equality check div produces 0.9999 for a = b, which when
  // floored can cause errors.
  const DIV = `
if (a == b) {
  return 1.0;
};
return a / b;`;
  // We use native integer division to deal with floating point imprecision. Since
  // we implement floor division and glsl implements truncated division, we
  // correct for this by subtracting 1 from result when the result is negative and
  // there is a remainder.
  const INT_DIV = `
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`;
  const POW = `
if(a < 0.0 && floor(b) < b){
  return NAN;
}
if (b == 0.0) {
  return 1.0;
}
return (round(mod(b, 2.0)) != 1) ?
    pow(abs(a), b) : sign(a) * pow(abs(a), b);
`;
  const EQUAL = `return float(a == b);`;
  const NOT_EQUAL = `return float(a != b);`;
  const LESS = `return float(a < b);`;
  const LESS_EQUAL = `return float(a <= b);`;
  const GREATER = `return float(a > b);`;
  const GREATER_EQUAL = `return float(a >= b);`;
  const LOGICAL_AND = `return float(a >= 1.0 && b >= 1.0);`;
  const LOGICAL_OR = `return float(a >= 1.0 || b >= 1.0);`;
  const MAX = CHECK_NAN_SNIPPET + `
  return max(a, b);
`;
  const MIN = CHECK_NAN_SNIPPET + `
  return min(a, b);
`;
  const MOD = `if (b == 0.0) return NAN;
  return mod(a, b);`;
  const ATAN2 = CHECK_NAN_SNIPPET + `
  return atan(a, b);
`;
  const ELU_DER = `return (b >= 1.0) ? a : a * (b + 1.0);`;
  const PRELU = `return (a < 0.) ? b * a : a;`;
  class BinaryOpProgram {
      constructor(op, aShape, bShape) {
          this.variableNames = ['A', 'B'];
          this.outputShape = tf.backend_util.assertAndGetBroadcastShape(aShape, bShape);
          this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  const CHECK_NAN_SNIPPET$1 = `
  result.r = isNaN.r > 0. ? NAN : result.r;
  result.g = isNaN.g > 0. ? NAN : result.g;
  result.b = isNaN.b > 0. ? NAN : result.b;
  result.a = isNaN.a > 0. ? NAN : result.a;
`;
  // We do the same as in ./binaryop_gpu, with vec4 and ivec4.
  // On Linux, the vectorized implementation produces NaNs when a and b are 0.
  const DIV$1 = `
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`;
  const INT_DIV$1 = `
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`;
  const POW$1 = `
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  vec4 isNaN = vec4(lessThan(a, vec4(0.0))) * vec4(lessThan(floor(b), b));
  ` +
      CHECK_NAN_SNIPPET$1 + `
  return result;
`;
  const PRELU$1 = `
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;
  const ELU_DER$1 = `
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`;
  const ATAN2$1 = `
  vec4 result = atan(a, b);
  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
  ` +
      CHECK_NAN_SNIPPET$1 + `
  return result;
`;
  const EQUAL$1 = `
  return vec4(equal(a, b));
`;
  const NOT_EQUAL$1 = `
  return vec4(notEqual(a, b));
`;
  const LESS$1 = `
  return vec4(lessThan(a, b));
`;
  const LESS_EQUAL$1 = `
  return vec4(lessThanEqual(a, b));
`;
  const GREATER$1 = `
  return vec4(greaterThan(a, b));
`;
  const GREATER_EQUAL$1 = `
  return vec4(greaterThanEqual(a, b));
`;
  const LOGICAL_AND$1 = `
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`;
  const LOGICAL_OR$1 = `
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`;
  const MAX$1 = `
  vec4 result = vec4(max(a, b));
  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
  ` +
      CHECK_NAN_SNIPPET$1 + `
  return result;
`;
  const MIN$1 = `
  vec4 result = vec4(min(a, b));
  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
  ` +
      CHECK_NAN_SNIPPET$1 + `
  return result;
`;
  const MOD$1 = `
  vec4 result = mod(a, b);
  vec4 isNaN = vec4(equal(b, vec4(0.0)));
  ` +
      CHECK_NAN_SNIPPET$1 + `
  return result;
`;
  class BinaryOpPackedProgram {
      constructor(op, aShape, bShape, checkOutOfBounds = false) {
          this.variableNames = ['A', 'B'];
          this.supportsBroadcasting = true;
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = tf.backend_util.assertAndGetBroadcastShape(aShape, bShape);
          const rank = this.outputShape.length;
          let checkOutOfBoundsString = '';
          if (checkOutOfBounds) {
              if (rank === 0 || tf.util.sizeFromShape(this.outputShape) === 1) {
                  checkOutOfBoundsString = `
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;
              }
              else {
                  const dtype = getCoordsDataType(rank);
                  checkOutOfBoundsString = `
          ${dtype} coords = getOutputCoords();
        `;
                  if (rank === 1) {
                      checkOutOfBoundsString += `
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;
                  }
                  else {
                      const channels = getChannels('coords', rank);
                      checkOutOfBoundsString += `
            bool nextRowOutOfBounds =
              (${channels[rank - 2]} + 1) >= ${this.outputShape[rank - 2]};
            bool nextColOutOfBounds =
              (${channels[rank - 1]} + 1) >= ${this.outputShape[rank - 1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `;
                  }
              }
          }
          this.userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${op}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${checkOutOfBoundsString}

        setOutput(result);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class ClipProgram {
      constructor(aShape) {
          this.variableNames = ['A'];
          this.outputShape = aShape;
          this.userCode = `
      uniform float minVal;
      uniform float maxVal;

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `;
      }
      getCustomSetupFunc(min, max) {
          return (gpgpu, webGLProgram) => {
              if (this.minLoc == null) {
                  this.minLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'minVal');
                  this.maxLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'maxVal');
              }
              gpgpu.gl.uniform1f(this.minLoc, min);
              gpgpu.gl.uniform1f(this.maxLoc, max);
          };
      }
  }

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
  class ClipPackedProgram {
      constructor(aShape) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = aShape;
          this.userCode = `
      uniform float minVal;
      uniform float maxVal;

      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `;
      }
      getCustomSetupFunc(min, max) {
          return (gpgpu, webGLProgram) => {
              if (this.minLoc == null) {
                  this.minLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'minVal');
                  this.maxLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'maxVal');
              }
              gpgpu.gl.uniform1f(this.minLoc, min);
              gpgpu.gl.uniform1f(this.maxLoc, max);
          };
      }
  }

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
  class ComplexAbsProgram {
      constructor(shape) {
          this.variableNames = ['real', 'imag'];
          this.outputShape = shape;
          this.userCode = `
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class ConcatProgram {
      // Concats 2d tensors along axis=1. See comments in MathBackendWebGL.concat().
      constructor(shapes) {
          this.outputShape = [];
          this.outputShape = tf.backend_util.computeOutShape(shapes, 1 /* axis */);
          this.variableNames = shapes.map((_, i) => `T${i}`);
          const offsets = new Array(shapes.length - 1);
          offsets[0] = shapes[0][1];
          for (let i = 1; i < offsets.length; i++) {
              offsets[i] = offsets[i - 1] + shapes[i][1];
          }
          const snippets = [`if (yC < ${offsets[0]}) setOutput(getT0(yR, yC));`];
          for (let i = 1; i < offsets.length; i++) {
              const shift = offsets[i - 1];
              snippets.push(`else if (yC < ${offsets[i]}) ` +
                  `setOutput(getT${i}(yR, yC-${shift}));`);
          }
          const lastIndex = offsets.length;
          const lastShift = offsets[offsets.length - 1];
          snippets.push(`else setOutput(getT${lastIndex}(yR, yC-${lastShift}));`);
          this.userCode = `
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${snippets.join('\n        ')}
      }
    `;
      }
  }

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
  class ConcatPackedProgram {
      constructor(shapes, axis) {
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = [];
          this.outputShape = tf.backend_util.computeOutShape(shapes, axis);
          const shape = this.outputShape;
          const rank = shape.length;
          const dtype = getCoordsDataType(rank);
          const coords = getChannels('coords', rank);
          const channels = ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank);
          this.variableNames = shapes.map((_, i) => `T${i}`);
          const offsets = new Array(shapes.length - 1);
          offsets[0] = shapes[0][axis];
          for (let i = 1; i < offsets.length; i++) {
              offsets[i] = offsets[i - 1] + shapes[i][axis];
          }
          const channel = channels[axis];
          const lastChannels = channels.slice(-2);
          const allChannels = channels.join();
          let getValueSnippet = `if (${channel} < ${offsets[0]}) {
        return getChannel(
            getT0(${allChannels}), vec2(${lastChannels.join()}));
        }`;
          for (let i = 1; i < offsets.length; i++) {
              const shift = offsets[i - 1];
              // Note: the >= comparison below may seem unnecessary given the check
              // above but is needed to workaround branch execution issues on some
              // devices. It makes all the conditions exclusive without relying on
              // execution order.
              getValueSnippet += `
        if (${channel} < ${offsets[i]}  && ${channel} >= ${offsets[i - 1]}) {
          return getChannel(
            getT${i}(${shiftedChannels(channels, channel, shift)}),
            vec2(${shiftedChannels(lastChannels, channel, shift)}));
        }`;
          }
          const lastIndex = offsets.length;
          const shift = offsets[offsets.length - 1];
          getValueSnippet += `
        return getChannel(
          getT${lastIndex}(${shiftedChannels(channels, channel, shift)}),
          vec2(${shiftedChannels(lastChannels, channel, shift)}));`;
          this.userCode = `
      float getValue(${channels.map(x => 'int ' + x)}) {
        ${getValueSnippet}
      }

      void main() {
        ${dtype} coords = getOutputCoords();
        vec4 result = vec4(getValue(${coords}), 0., 0., 0.);

        ${coords[rank - 1]} = ${coords[rank - 1]} + 1;
        if (${coords[rank - 1]} < ${shape[rank - 1]}) {
          result.g = getValue(${coords});
        }

        ${coords[rank - 2]} = ${coords[rank - 2]} + 1;
        if (${coords[rank - 2]} < ${shape[rank - 2]}) {
          result.a = getValue(${coords});
        }

        ${coords[rank - 1]} = ${coords[rank - 1]} - 1;
        if (${coords[rank - 2]} < ${shape[rank - 2]} &&
            ${coords[rank - 1]} < ${shape[rank - 1]}) {
          result.b = getValue(${coords});
        }
        setOutput(result);
      }
    `;
      }
  }
  /**
   * Return an expression for coordinates into a vector where a given channel
   * will be offset by [shift].
   *
   * @param channels the channels to consider
   * @param channel the channel we want shifted
   * @param shift  the amount to subtract from the channel.
   *
   * @returns a string of the form 'x, y-[shift], z' where any one channel can
   * have the shift applied.
   */
  function shiftedChannels(channels, channel, shift) {
      const channelIdx = channels.indexOf(channel);
      const res = channels.map((c, idx) => {
          if (idx === channelIdx) {
              return `${c} - ${shift}`;
          }
          else {
              return c;
          }
      });
      return res.join();
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class Conv2DDerFilterProgram {
      constructor(convInfo) {
          this.variableNames = ['x', 'dy'];
          this.outputShape = convInfo.filterShape;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          const isChannelsLast = convInfo.dataFormat === 'channelsLast';
          this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${convInfo.batchSize}; b++) {
          for (int yR = 0; yR < ${convInfo.outHeight}; yR++) {
            int xR = wR + yR * ${strideHeight} - ${padTop};

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${convInfo.outWidth}; yC++) {
              int xC = wC + yC * ${strideWidth} - ${padLeft};

              if (xC < 0 || xC >= ${convInfo.inWidth}) {
                continue;
              }

              if (${isChannelsLast}) {
                float dyValue = getDy(b, yR, yC, d2);
                float xValue = getX(b, xR, xC, d1);
                dotProd += (xValue * dyValue);
              } else {
                float dyValue = getDy(b, d2, yR, yC);
                float xValue = getX(b, d1, xR, xC);
                dotProd += (xValue * dyValue);
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }
  class Conv2DDerInputProgram {
      constructor(convInfo) {
          this.variableNames = ['dy', 'W'];
          this.outputShape = convInfo.inShape;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const isChannelsLast = convInfo.dataFormat === 'channelsLast';
          const padTop = filterHeight - 1 - convInfo.padInfo.top;
          const padLeft = filterWidth - 1 - convInfo.padInfo.left;
          const rowDim = isChannelsLast ? 1 : 2;
          const colDim = isChannelsLast ? 2 : 3;
          const channelDim = isChannelsLast ? 3 : 1;
          this.userCode = `
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${channelDim}];

        ivec2 dyCorner = ivec2(coords[${rowDim}], coords[${colDim}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

          if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${filterHeight} - 1 - wR;

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

            if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${filterWidth} - 1 - wC;

            for (int d2 = 0; d2 < ${convInfo.outChannels}; d2++) {

              if (${isChannelsLast}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }
  class Conv3DDerFilterProgram {
      constructor(convInfo) {
          this.variableNames = ['x', 'dy'];
          this.outputShape = convInfo.filterShape;
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const padFront = convInfo.padInfo.front;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          this.userCode = `
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${convInfo.batchSize}; b++) {
          for (int yF = 0; yF < ${convInfo.outDepth}; yF++) {
            int xF = wF + yF * ${strideDepth} - ${padFront};

            if (xF < 0 || xF >= ${convInfo.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${convInfo.outHeight}; yR++) {
              int xR = wR + yR * ${strideHeight} - ${padTop};

              if (xR < 0 || xR >= ${convInfo.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${convInfo.outWidth}; yC++) {
                int xC = wC + yC * ${strideWidth} - ${padLeft};

                if (xC < 0 || xC >= ${convInfo.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }
  class Conv3DDerInputProgram {
      constructor(convInfo) {
          this.variableNames = ['dy', 'W'];
          this.outputShape = convInfo.inShape;
          const filterDepth = convInfo.filterDepth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const padFront = filterDepth - 1 - convInfo.padInfo.front;
          const padTop = filterHeight - 1 - convInfo.padInfo.top;
          const padLeft = filterWidth - 1 - convInfo.padInfo.left;
          this.userCode = `
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${filterDepth}; wF++) {
          float dyF = float(dyFCorner + wF) / ${strideDepth}.0;

          if (dyF < 0.0 || dyF >= ${convInfo.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${filterDepth} - 1 - wF;

          for (int wR = 0; wR < ${filterHeight}; wR++) {
            float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

            if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${filterHeight} - 1 - wR;

            for (int wC = 0; wC < ${filterWidth}; wC++) {
              float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

              if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${filterWidth} - 1 - wC;

              for (int d2 = 0; d2 < ${convInfo.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class DepthwiseConv2DDerFilterProgram {
      constructor(convInfo) {
          this.variableNames = ['x', 'dy'];
          this.outputShape = convInfo.filterShape;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          const channelMul = convInfo.outChannels / convInfo.inChannels;
          this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${channelMul} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${convInfo.batchSize}; b++) {
          for (int yR = 0; yR < ${convInfo.outHeight}; yR++) {
            int xR = wR + yR * ${strideHeight} - ${padTop};

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${convInfo.outWidth}; yC++) {
              int xC = wC + yC * ${strideWidth} - ${padLeft};

              if (xC < 0 || xC >= ${convInfo.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }
  class DepthwiseConv2DDerInputProgram {
      constructor(convInfo) {
          this.variableNames = ['dy', 'W'];
          this.outputShape = convInfo.inShape;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const padTop = filterHeight - 1 - convInfo.padInfo.top;
          const padLeft = filterWidth - 1 - convInfo.padInfo.left;
          const channelMul = convInfo.outChannels / convInfo.inChannels;
          this.userCode = `
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${filterHeight}; wR++) {
          float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

          if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${filterHeight} - 1 - wR;

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

            if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${filterWidth} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${channelMul}; dm++) {
              int d2 = d1 * ${channelMul} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class Conv2DProgram {
      constructor(convInfo, addBias = false, activation = null, hasPreluActivationWeights = false) {
          this.variableNames = ['x', 'W'];
          this.outputShape = convInfo.outShape;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const inputDepthNearestVec4 = Math.floor(convInfo.inChannels / 4) * 4;
          const inputDepthVec4Remainder = convInfo.inChannels % 4;
          const isChannelsLast = convInfo.dataFormat === 'channelsLast';
          const rowDim = isChannelsLast ? 1 : 2;
          const colDim = isChannelsLast ? 2 : 3;
          const channelDim = isChannelsLast ? 3 : 1;
          let activationSnippet = '', applyActivationSnippet = '';
          if (activation) {
              if (hasPreluActivationWeights) {
                  activationSnippet = `float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
              }
              else {
                  activationSnippet = `
          float activation(float x) {
            ${activation}
          }
        `;
              }
              applyActivationSnippet = `result = activation(result);`;
          }
          const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
          if (addBias) {
              this.variableNames.push('bias');
          }
          if (hasPreluActivationWeights) {
              this.variableNames.push('preluActivationWeights');
          }
          this.userCode = `
      ${activationSnippet}

      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${channelDim}];

        ivec2 xRCCorner =
            ivec2(coords[${rowDim}], coords[${colDim}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          int xR = xRCorner + wR * ${dilationHeight};

          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            int xC = xCCorner + wC * ${dilationWidth};

            if (xC < 0 || xC >= ${convInfo.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${inputDepthNearestVec4}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${isChannelsLast}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${inputDepthVec4Remainder === 1}) {

              if (${isChannelsLast}) {
                dotProd +=
                    getX(batch, xR, xC, ${inputDepthNearestVec4}) *
                    getW(wR, wC, ${inputDepthNearestVec4}, d2);
              } else {
                dotProd +=
                    getX(batch, ${inputDepthNearestVec4}, xR, xC) *
                    getW(wR, wC, ${inputDepthNearestVec4}, d2);
              }

            } else if (${inputDepthVec4Remainder === 2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${inputDepthNearestVec4}, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 1, d2)
              );

              if (${isChannelsLast}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${inputDepthNearestVec4}, xR, xC),
                  getX(batch, ${inputDepthNearestVec4} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${inputDepthVec4Remainder === 3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${inputDepthNearestVec4}, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 1, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 2, d2)
              );

              if (${isChannelsLast}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 1),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${inputDepthNearestVec4}, xR, xC),
                  getX(batch, ${inputDepthNearestVec4} + 1, xR, xC),
                  getX(batch, ${inputDepthNearestVec4} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
      }
  }
  class Conv3DProgram {
      constructor(convInfo) {
          this.variableNames = ['x', 'W'];
          this.outputShape = convInfo.outShape;
          const padFront = convInfo.padInfo.front;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const filterDepth = convInfo.filterDepth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const inputDepthNearestVec4 = Math.floor(convInfo.inChannels / 4) * 4;
          const inputDepthVec4Remainder = convInfo.inChannels % 4;
          this.userCode = `
      const ivec3 strides = ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${filterDepth}; wF++) {
          int xF = xFCorner + wF * ${dilationDepth};

          if (xF < 0 || xF >= ${convInfo.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${filterHeight}; wR++) {
            int xR = xRCorner + wR * ${dilationHeight};

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${filterWidth}; wC++) {
              int xC = xCCorner + wC * ${dilationWidth};

              if (xC < 0 || xC >= ${convInfo.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${inputDepthNearestVec4}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${inputDepthVec4Remainder === 1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4}) *
                  getW(wF, wR, wC, ${inputDepthNearestVec4}, d2);
              } else if (${inputDepthVec4Remainder === 2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${inputDepthNearestVec4}, d2),
                  getW(wF, wR, wC, ${inputDepthNearestVec4} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${inputDepthVec4Remainder === 3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4} + 1),
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${inputDepthNearestVec4}, d2),
                  getW(wF, wR, wC, ${inputDepthNearestVec4} + 1, d2),
                  getW(wF, wR, wC, ${inputDepthNearestVec4} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class DepthwiseConv2DProgram {
      constructor(convInfo, addBias = false, activation = null, hasPreluActivation = false) {
          this.variableNames = ['x', 'W'];
          this.outputShape = convInfo.outShape;
          const xNumRows = convInfo.inHeight;
          const xNumCols = convInfo.inWidth;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const channelMul = convInfo.outChannels / convInfo.inChannels;
          let activationSnippet = '', applyActivationSnippet = '';
          if (activation) {
              if (hasPreluActivation) {
                  activationSnippet = `float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
              }
              else {
                  activationSnippet = `
          float activation(float x) {
            ${activation}
          }
        `;
              }
              applyActivationSnippet = `result = activation(result);`;
          }
          const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
          if (addBias) {
              this.variableNames.push('bias');
          }
          if (hasPreluActivation) {
              this.variableNames.push('preluActivationWeights');
          }
          this.userCode = `
      ${activationSnippet}

      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          int xR = xRCorner + wR * ${dilationHeight};

          if (xR < 0 || xR >= ${xNumRows}) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            int xC = xCCorner + wC * ${dilationWidth};

            if (xC < 0 || xC >= ${xNumCols}) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
      }
  }

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
  class DepthwiseConvPacked2DProgram {
      constructor(convInfo, addBias = false, activation = null, hasPreluActivation = false) {
          this.variableNames = ['x', 'W'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = convInfo.outShape;
          const xNumRows = convInfo.inHeight;
          const xNumCols = convInfo.inWidth;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const filterHeight = convInfo.filterHeight;
          const filterWidth = convInfo.filterWidth;
          const texelsAcross = filterWidth;
          let mainLoop = `int xR; int xC; int xCOffset;`;
          for (let r = 0; r < filterHeight; r++) {
              for (let c = 0; c < filterWidth; c++) {
                  mainLoop += `
          vec4 xTexelR${r}C${c * 2} = vec4(0.);
          vec4 wR${r}C${c} = vec4(0.);
          vec4 xR${r}C${c} = vec4(0.);`;
              }
          }
          /**
           * This vectorized implementation works by gathering the values needed for
           * each output channel's dot product into vec4's and then multiplying them
           * all together (this happens in the final double for-loop below). Most of
           * the main loop consists of constructing these vec4's with the minimum
           * number of texture2D calls, which means making use of all four returned
           * values from a texture2D call at once.
           */
          for (let r = 0; r < filterHeight; r++) {
              for (let texelC = 0; texelC < texelsAcross; texelC++) {
                  const c = texelC * 2;
                  mainLoop += `
          xR = xRCorner + ${r * dilationHeight};
          xC = xCCorner + ${c * dilationWidth};
        `;
                  if (strideWidth === 1) {
                      if (c < filterWidth) {
                          // If padding is odd, the outer texels have to be composed.
                          if (padLeft % 2 === 1) {
                              // TODO: Ensure vec4 previous does not result in redundant sample,
                              // and avoid setting xTexelRC's that exceed the boundary in the
                              // first place rather than resetting them to vec4(0)).
                              // To compute xCOffset:
                              // - If padding is odd, we must add 1 to ensure we ask for an
                              // even-numbered row.
                              // - We subtract 2 to access the previous texel.
                              mainLoop += `
                xCOffset = xC + 1;
                if(xR >= 0 && xR < ${xNumRows} && xCOffset >= 0 && xCOffset < ${xNumCols}) {
                  xTexelR${r}C${c} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if(xCOffset + 1 >= ${xNumCols}) {
                    xTexelR${r}C${c}.zw = vec2(0.);
                  }
                } else {
                  xTexelR${r}C${c} = vec4(0.);
                }

                xCOffset = xC + 1 - 2;
                if(xR >= 0 && xR < ${xNumRows} && xCOffset >= 0 && xCOffset < ${xNumCols}) {
                  vec4 previous = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if(xCOffset + 1 >= ${xNumCols}) {
                    previous.zw = vec2(0.);
                  }

                  xR${r}C${c} = vec4(previous.zw, xTexelR${r}C${c}.xy);
                } else {
                  xR${r}C${c} = vec4(0, 0, xTexelR${r}C${c}.xy);
                }
              `;
                          }
                          else {
                              // Padding is even, so xRC corresponds to a single texel.
                              mainLoop += `
                if(xR >= 0 && xR < ${xNumRows} && xC >= 0 && xC < ${xNumCols}) {
                  xTexelR${r}C${c} = getX(batch, xR, xC, d1);
                } else {
                  xTexelR${r}C${c} = vec4(0.);
                }

                xR${r}C${c} = xTexelR${r}C${c};
              `;
                          }
                          if (c + 1 < filterWidth) {
                              // If dilation is even, the second entry should match the first
                              // (either both are composed or both are single samples). But if
                              // dilation is odd, then the second entry should be the opposite
                              // of the first (if the first is composed, the second is a single
                              // sample, and vice versa.)
                              const nextTexelOffset = padLeft % 2 === 0 ?
                                  tf.util.nearestLargerEven(dilationWidth) :
                                  dilationWidth;
                              if ((dilationWidth % 2 === 0 && padLeft % 2 === 1) ||
                                  (dilationWidth % 2 !== 0 && padLeft % 2 !== 1)) {
                                  mainLoop += `
                  xCOffset = xC + ${padLeft % 2} + ${nextTexelOffset};

                  if(xR >= 0 && xR < ${xNumRows} &&
                    xCOffset >= 0 && xCOffset < ${xNumCols}) {
                    xTexelR${r}C${c + 2} = getX(batch, xR, xCOffset, d1);
                  }
                `;
                                  // If dilation > 1 then the xRC's will not be able to share any
                                  // values, so each xRC will require two unique calls to getX.
                                  if (dilationWidth > 1) {
                                      mainLoop += `
                    xCOffset -= 2;
                    if(xR >= 0 && xR < ${xNumRows} &&
                      xCOffset >= 0 && xCOffset < ${xNumCols}) {
                      xTexelR${r}C${c} = getX(batch, xR, xCOffset, d1);
                    } else {
                      xTexelR${r}C${c} = vec4(0.);
                    }
                  `;
                                  }
                                  mainLoop += `
                  xR${r}C${c + 1} = vec4(
                    xTexelR${r}C${c}.zw, xTexelR${r}C${c + 2}.xy);
                `;
                              }
                              else {
                                  mainLoop += `
                  xCOffset = xC + ${nextTexelOffset};

                  if(xR >= 0 && xR < ${xNumRows} &&
                    xCOffset >= 0 && xCOffset < ${xNumCols}) {
                    xTexelR${r}C${c + 2} = getX(batch, xR, xCOffset, d1);
                  }

                  xR${r}C${c + 1} = xTexelR${r}C${c + 2};
                `;
                              }
                          }
                      }
                  }
                  else { // stride > 1
                      if (c < filterWidth) {
                          mainLoop += `
              if(xR >= 0 && xR < ${xNumRows}) {
            `;
                          // Depending on whether padLeft is even or odd, we want either the
                          // xy or zw channels from X texels for xR${r}C${c}. If padLeft is
                          // even, xR${r}C${c + 1} is simply the zw channels of texels we've
                          // already sampled. But if padLeft is odd, xR${r}C{$c + 1}.zw will
                          // need to come from the xy channels of a new texel, hence the `vec4
                          // final` initialized below.
                          if (padLeft % 2 === 1) {
                              mainLoop += `
                xCOffset = xC + 1 - ${strideWidth};
                if(xCOffset >= 0 && xCOffset < ${xNumCols}) {
                  xTexelR${r}C${c} = getX(batch, xR, xCOffset, d1);
                } else {
                  xTexelR${r}C${c} = vec4(0.);
                }

                if(xC + 1 >= 0 && xC + 1 < ${xNumCols}) {
                  xTexelR${r}C${c + 2} = getX(batch, xR, xC + 1, d1);
                } else {
                  xTexelR${r}C${c + 2} = vec4(0.);
                }

                xR${r}C${c} = vec4(
                  xTexelR${r}C${c}.zw, xTexelR${r}C${c + 2}.zw);
              `;
                              if (c + 1 < filterWidth) {
                                  mainLoop += `
                  vec4 final = vec4(0.);
                  xCOffset = xC + 1 + ${strideWidth};
                  if(xCOffset >= 0 && xCOffset < ${xNumCols}) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xR${r}C${c + 1} = vec4(xTexelR${r}C${c + 2}.xy, final.xy);
                `;
                              }
                          }
                          else {
                              mainLoop += `
                if(xC >= 0 && xC < ${xNumCols}) {
                  xTexelR${r}C${c} = getX(batch, xR, xC, d1);
                } else {
                  xTexelR${r}C${c} = vec4(0.);
                }

                xCOffset = xC + ${strideWidth};
                if(xCOffset >= 0 && xCOffset < ${xNumCols}) {
                  xTexelR${r}C${c + 2} = getX(batch, xR, xCOffset, d1);
                } else {
                  xTexelR${r}C${c + 2} = vec4(0.);
                }

                xR${r}C${c} = vec4(
                  xTexelR${r}C${c}.xy, xTexelR${r}C${c + 2}.xy);
              `;
                              if (c + 1 < filterWidth) {
                                  mainLoop += `
                  xR${r}C${c + 1} = vec4(
                    xTexelR${r}C${c}.zw, xTexelR${r}C${c + 2}.zw);
                `;
                              }
                          }
                          mainLoop += `}`;
                      }
                  }
                  if (c < filterWidth) {
                      mainLoop += `
            vec4 wTexelR${r}C${c} = getW(${r}, ${c}, d1, q);
            wR${r}C${c} = vec4(wTexelR${r}C${c}.xz, wTexelR${r}C${c}.xz);
          `;
                      if (c + 1 < filterWidth) {
                          mainLoop += `
              vec4 wTexelR${r}C${c + 1} = getW(${r}, ${c + 1}, d1, q);
              wR${r}C${c + 1} =
                vec4(wTexelR${r}C${c + 1}.xz, wTexelR${r}C${c + 1}.xz);`;
                      }
                  }
              }
          }
          for (let r = 0; r < filterHeight; r++) {
              for (let c = 0; c < filterWidth; c++) {
                  mainLoop += `dotProd += xR${r}C${c} * wR${r}C${c};`;
              }
          }
          let activationSnippet = '', applyActivationSnippet = '';
          if (activation) {
              if (hasPreluActivation) {
                  activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
              }
              else {
                  activationSnippet = `vec4 activation(vec4 x) {
          ${activation}
        }`;
              }
              applyActivationSnippet = `result = activation(result);`;
          }
          const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
          if (addBias) {
              this.variableNames.push('bias');
          }
          if (hasPreluActivation) {
              this.variableNames.push('preluActivationWeights');
          }
          this.userCode = `
      ${activationSnippet}

      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {

        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2;
        int q = 0;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        vec4 dotProd = vec4(0.);

        ${mainLoop}

        vec4 result = dotProd;
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class CropAndResizeProgram {
      constructor(imageShape, boxShape, cropSize, method, extrapolationValue) {
          this.variableNames = ['Image', 'Boxes', 'BoxInd'];
          this.outputShape = [];
          const [batch, imageHeight, imageWidth, depth] = imageShape;
          const [numBoxes,] = boxShape;
          const [cropHeight, cropWidth] = cropSize;
          this.outputShape = [numBoxes, cropHeight, cropWidth, depth];
          const methodId = method === 'bilinear' ? 1 : 0;
          const [inputHeightFloat, inputWidthFloat] = [`${imageHeight - 1}.0`, `${imageWidth - 1}.0`];
          const [heightRatio, heightScale, inY] = cropHeight > 1 ?
              [
                  `${(imageHeight - 1) / (cropHeight - 1)}`,
                  '(y2-y1) * height_ratio',
                  `y1*${inputHeightFloat} + float(y)*(height_scale)`,
              ] :
              [
                  '0.0',
                  '0.0',
                  `0.5 * (y1+y2) * ${inputHeightFloat}`,
              ];
          const [widthRatio, widthScale, inX] = cropWidth > 1 ?
              [
                  `${(imageWidth - 1) / (cropWidth - 1)}`,
                  '(x2-x1) * width_ratio',
                  `x1*${inputWidthFloat} + float(x)*(width_scale)`,
              ] :
              [
                  '0.0',
                  '0.0',
                  `0.5 * (x1+x2) * ${inputWidthFloat}`,
              ];
          // Reference implementation
          // tslint:disable-next-line:max-line-length
          // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op_gpu.cu.cc
          this.userCode = `
      const float height_ratio = float(${heightRatio});
      const float width_ratio = float(${widthRatio});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${batch}) {
          return;
        }

        float height_scale = ${heightScale};
        float width_scale = ${widthScale};

        float in_y = ${inY};
        if( in_y < 0.0 || in_y > ${inputHeightFloat} ) {
          setOutput(float(${extrapolationValue}));
          return;
        }
        float in_x = ${inX};
        if( in_x < 0.0 || in_x > ${inputWidthFloat} ) {
          setOutput(float(${extrapolationValue}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${methodId} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class CumSumProgram {
      constructor(shape, exclusive, reverse) {
          this.variableNames = ['x'];
          this.outputShape = shape;
          const rank = shape.length;
          const finalDim = shape[shape.length - 1];
          const comparator = reverse ? '<' : '>';
          this.userCode = `
      int getIndex(int i) {
        ${reverse ? `return ${finalDim} -i - 1;` : 'return i;'}
      }

      void main() {
        ${getCoordsDataType(rank)} coords = getOutputCoords();
        int end = ${getFinalCoord(rank, 'coords')};
        float val = 0.0;
        for (int i = ${finalDim} - 1; i >= 0; i -= 1) {
          int idx = getIndex(i);
          if (idx ${comparator} end) {
            continue;
          }
          if (idx == end && ${exclusive}) {
            continue;
          }
          ${getFinalCoord(rank, 'coords')} = idx;
          val += getX(${getCoords(rank, 'coords')});
        }
        setOutput(val);
      }
    `;
      }
  }
  function getCoords(rank, name) {
      if (rank === 1) {
          return `${name}`;
      }
      else if (rank === 2) {
          return `${name}.x, ${name}.y`;
      }
      else if (rank === 3) {
          return `${name}.x, ${name}.y, ${name}.z`;
      }
      else if (rank === 4) {
          return `${name}.x, ${name}.y, ${name}.z, ${name}.w`;
      }
      else {
          throw Error(`Cumulative sum for rank ${rank} is not yet supported`);
      }
  }
  function getFinalCoord(rank, name) {
      if (rank === 1) {
          return `${name}`;
      }
      else if (rank === 2) {
          return `${name}.y`;
      }
      else if (rank === 3) {
          return `${name}.z`;
      }
      else if (rank === 4) {
          return `${name}.w`;
      }
      else {
          throw Error(`Cumulative sum for rank ${rank} is not yet supported`);
      }
  }

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
  class DecodeMatrixProgram {
      constructor(outputShape) {
          this.variableNames = ['A'];
          this.packedInputs = false;
          this.packedOutput = true;
          this.outPackingScheme = PackingScheme.DENSE;
          const texShape = getDenseTexShape(outputShape);
          const glsl = getGlslDifferences();
          this.outputShape = outputShape;
          this.userCode = `
      ivec3 outCoordsFromFlatIndex(int index) {
        ${getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], outputShape)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx *
          vec2(${texShape[0]}, ${texShape[1]}));
        int index = 4 * (resTexRC.x * ${texShape[1]} + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${glsl.output} = result;
      }
    `;
      }
  }

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
  class DecodeMatrixPackedProgram {
      constructor(outputShape) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outPackingScheme = PackingScheme.DENSE;
          const texShape = getDenseTexShape(outputShape);
          const glsl = getGlslDifferences();
          this.outputShape = outputShape;
          this.userCode = `
      ivec3 outCoordsFromFlatIndex(int index) {
        ${getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], outputShape)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx *
          vec2(${texShape[0]}, ${texShape[1]}));
        int index = 4 * (resTexRC.x * ${texShape[1]} + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${glsl.output} = result;
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class DepthToSpaceProgram {
      constructor(outputShape, blockSize, dataFormat) {
          this.variableNames = ['x'];
          this.outputShape = [];
          this.outputShape = outputShape;
          this.blockSize = blockSize;
          this.dataFormat = dataFormat;
          this.userCode = `
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${blockSize};
      int offset_h = imod(h, ${blockSize});
      int in_w = w / ${blockSize};
      int offset_w = imod(w, ${blockSize});
      int offset_d = (offset_h * ${blockSize} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `;
      }
      getHeightCoordString() {
          if (this.dataFormat === 'NHWC') {
              return `coords[1]`;
          }
          else {
              return `coords[2]`;
          }
      }
      getWidthCoordString() {
          if (this.dataFormat === 'NHWC') {
              return `coords[2]`;
          }
          else {
              return `coords[3]`;
          }
      }
      getDepthCoordString() {
          if (this.dataFormat === 'NHWC') {
              return `coords[3]`;
          }
          else {
              return `coords[1]`;
          }
      }
      getOutputDepthSize() {
          if (this.dataFormat === 'NHWC') {
              return this.outputShape[3];
          }
          else {
              return this.outputShape[1];
          }
      }
      getInputSamplingString() {
          if (this.dataFormat === 'NHWC') {
              return `getX(b, in_h, in_w, in_d)`;
          }
          else {
              return `getX(b, in_d, in_h, in_w)`;
          }
      }
  }

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
  class DiagProgram {
      constructor(size) {
          this.variableNames = ['X'];
          this.outputShape = [size, size];
          this.userCode = `
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class EncodeFloatProgram {
      constructor(outputShape) {
          this.variableNames = ['A'];
          this.outTexUsage = TextureUsage.DOWNLOAD;
          const glsl = getGlslDifferences();
          this.outputShape = outputShape;
          this.userCode = `
      ${ENCODE_FLOAT_SNIPPET}

      void main() {
        float x = getAAtOutCoords();
        ${glsl.output} = encode_float(x);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class EncodeFloatPackedProgram {
      constructor(outputShape) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = false;
          this.outTexUsage = TextureUsage.DOWNLOAD;
          const glsl = getGlslDifferences();
          this.outputShape = outputShape;
          this.userCode = `
      ${ENCODE_FLOAT_SNIPPET}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${glsl.output} = encode_float(x);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class EncodeMatrixProgram {
      constructor(outputShape, texShape, inputIsUnsignedByte = false) {
          this.variableNames = ['A'];
          const glsl = getGlslDifferences();
          const [height, width] = texShape;
          this.outputShape = outputShape;
          let output = `result`;
          if (inputIsUnsignedByte) {
              output = `floor(result * 255. + 0.5)`;
          }
          this.userCode = `
      ${getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 coords = getOutputCoords();

        int flatIndex = getFlatIndex(coords);
        int offset = imod(flatIndex, 4);

        flatIndex = idiv(flatIndex, 4, 1.);
        
        int r = flatIndex / ${width};
        int c = imod(flatIndex, ${width});
        vec2 uv = (vec2(c, r) + halfCR) / vec2(${width}.0, ${height}.0);
        vec4 values = ${glsl.texture2D}(A, uv);

        float result;

        if(offset == 0) {
          result = values[0];
        } else if(offset == 1) {
          result = values[1];
        } else if(offset == 2) {
          result = values[2];
        } else {
          result = values[3];
        }

        ${glsl.output} = vec4(${output}, 0., 0., 0.);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  /*
  This is how the shader encodes a tensor with shape = [2, 3, 5]
  (indices are [batch, row, col]).

  000|001   002|003   004|xxx   020|021   022|023   024|xxx
  -------   -------   -------   -------   -------   -------
  010|011   012|013   014|xxx   xxx|xxx   xxx|xxx   xxx|xxx

  100|101   102|103   104|xxx   120|121   122|123   124|xxx
  -------   -------   -------   -------   -------   -------
  110|111   112|113   114|xxx   xxx|xxx   xxx|xxx   xxx|xxx

  Single texels contain only values from the same batch, and from adjacent rows
  and columns.
   */
  class EncodeMatrixPackedProgram {
      constructor(outputShape, texShape, inputIsUnsignedByte = false) {
          this.variableNames = ['A'];
          this.packedInputs = false;
          this.packedOutput = true;
          const glsl = getGlslDifferences();
          const [height, width] = texShape;
          this.outputShape = outputShape;
          let mainLoop = '';
          let output = 'result';
          if (inputIsUnsignedByte) {
              output = 'floor(result * 255. + 0.5)';
          }
          for (let row = 0; row <= 1; row++) {
              for (let col = 0; col <= 1; col++) {
                  const channel = row * 2 + col;
                  mainLoop += `
          localCoords = coords;
          if(localCoords[2] + ${col} < ${outputShape[2]}) {
            localCoords[2] += ${col};
            if(localCoords[1] + ${row} < ${outputShape[1]}) {
              localCoords[1] += ${row};

              flatIndex = getFlatIndex(localCoords);
              offset = imod(flatIndex, 4);

              flatIndex = idiv(flatIndex, 4, 1.);

              r = flatIndex / ${width};
              c = imod(flatIndex, ${width});
              uv = (vec2(c, r) + halfCR) / vec2(${width}.0, ${height}.0);
              values = ${glsl.texture2D}(A, uv);

              if(offset == 0) {
                result[${channel}] = values[0];
              } else if(offset == 1) {
                result[${channel}] = values[1];
              } else if(offset == 2) {
                result[${channel}] = values[2];
              } else {
                result[${channel}] = values[3];
              }
            }
          }
        `;
              }
          }
          this.userCode = `
      ${getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 coords = getOutputCoords();

        vec4 result = vec4(0.);
        int flatIndex, r, c, offset;
        ivec3 localCoords;
        vec2 uv;
        vec4 values;

        ${mainLoop}

        ${glsl.output} = ${output};
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  const COMPLEX_FFT = {
      REAL: 'return real * expR - imag * expI;',
      IMAG: 'return real * expI + imag * expR;'
  };
  class FFTProgram {
      constructor(op, inputShape, inverse) {
          this.variableNames = ['real', 'imag'];
          const innerDim = inputShape[1];
          this.outputShape = inputShape;
          const exponentMultiplierSnippet = inverse ? `2.0 * ${Math.PI}` : `-2.0 * ${Math.PI}`;
          const resultDenominator = inverse ? `${innerDim}.0` : '1.0';
          this.userCode = `
      const float exponentMultiplier = ${exponentMultiplierSnippet};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${op}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${innerDim});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${innerDim}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${resultDenominator};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `;
      }
  }

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
  class FillProgram {
      constructor(shape, value) {
          this.outputShape = [];
          this.variableNames = ['x'];
          this.outputShape = shape;
          this.userCode = `
      uniform float value;
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `;
      }
      getCustomSetupFunc(value) {
          return (gpgpu, webGLProgram) => {
              if (this.valueLoc == null) {
                  this.valueLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'value');
              }
              gpgpu.gl.uniform1f(this.valueLoc, value);
          };
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class GatherProgram {
      constructor(aShape, indicesLength, axis) {
          this.variableNames = ['A', 'indices'];
          const outputShape = aShape.slice();
          outputShape[axis] = indicesLength;
          this.outputShape = outputShape;
          this.rank = outputShape.length;
          const dtype = getCoordsDataType(this.rank);
          const sourceCoords = getSourceCoords$1(aShape, axis);
          this.userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        setOutput(getA(${sourceCoords}));
      }
    `;
      }
  }
  function getSourceCoords$1(aShape, axis) {
      const rank = aShape.length;
      if (rank > 4) {
          throw Error(`Gather for rank ${rank} is not yet supported`);
      }
      if (rank === 1) {
          return `int(getIndices(resRC))`;
      }
      const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
      const sourceCoords = [];
      for (let i = 0; i < aShape.length; i++) {
          if (i === axis) {
              sourceCoords.push(`int(getIndices(${currentCoords[i]}))`);
          }
          else {
              sourceCoords.push(`${currentCoords[i]}`);
          }
      }
      return sourceCoords.join();
  }

  class GatherNDProgram {
      constructor(sliceDim, strides, shape) {
          this.sliceDim = sliceDim;
          this.strides = strides;
          this.variableNames = ['x', 'indices'];
          this.outputShape = shape;
          const stridesType = getCoordsDataType(strides.length);
          const dtype = getCoordsDataType(shape.length);
          const strideString = this.sliceDim > 1 ? 'strides[j]' : 'strides';
          this.userCode = `
        ${stridesType} strides = ${stridesType}(${this.strides});
         void main() {
          ${dtype} coords = getOutputCoords();
          int flattenIndex = 0;
          for (int j = 0; j < ${this.sliceDim}; j++) {
            int index = round(getIndices(coords[0], j));
            flattenIndex += index * ${strideString};
          }
          setOutput(getX(flattenIndex, coords[1]));
        }
      `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  function createVertexShader$1(gl, debug) {
      const glsl = getGlslDifferences();
      const vertexShaderSource = `${glsl.version}
    precision highp float;
    ${glsl.attribute} vec3 clipSpacePos;
    ${glsl.attribute} vec2 uv;
    ${glsl.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`;
      return createVertexShader(gl, debug, vertexShaderSource);
  }
  function createVertexBuffer(gl, debug) {
      // [x y z u v] * [upper-left, lower-left, upper-right, lower-right]
      const vertexArray = new Float32Array([-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
      return createStaticVertexBuffer(gl, debug, vertexArray);
  }
  function createIndexBuffer(gl, debug) {
      // OpenGL (and WebGL) have "CCW == front" winding
      const triangleVertexIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);
      return createStaticIndexBuffer(gl, debug, triangleVertexIndices);
  }
  function createAndConfigureTexture(gl, debug, width, height, internalFormat, textureFormat, textureType) {
      validateTextureSize(width, height);
      const texture = createTexture(gl, debug);
      const tex2d = gl.TEXTURE_2D;
      callAndCheck(gl, debug, () => gl.bindTexture(tex2d, texture));
      callAndCheck(gl, debug, () => gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE));
      callAndCheck(gl, debug, () => gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE));
      callAndCheck(gl, debug, () => gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST));
      callAndCheck(gl, debug, () => gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST));
      callAndCheck(gl, debug, () => gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, textureFormat, textureType, null));
      callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, null));
      return texture;
  }
  function createFloat32MatrixTexture(gl, debug, rows, columns, textureConfig) {
      const [width, height] = getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
      return createAndConfigureTexture(gl, debug, width, height, textureConfig.internalFormatFloat, textureConfig.textureFormatFloat, gl.FLOAT);
  }
  function createFloat16MatrixTexture(gl, debug, rows, columns, textureConfig) {
      const [width, height] = getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
      return createAndConfigureTexture(gl, debug, width, height, textureConfig.internalFormatHalfFloat, textureConfig.textureFormatFloat, textureConfig.textureTypeHalfFloat);
  }
  function createUnsignedBytesMatrixTexture(gl, debug, rows, columns, textureConfig) {
      const [width, height] = getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
      return createAndConfigureTexture(gl, debug, width, height, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE);
  }
  function createPackedMatrixTexture(gl, debug, rows, columns, textureConfig) {
      const [width, height] = getPackedMatrixTextureShapeWidthHeight(rows, columns);
      return createAndConfigureTexture(gl, debug, width, height, textureConfig.internalFormatPackedFloat, gl.RGBA, gl.FLOAT);
  }
  function createFloat16PackedMatrixTexture(gl, debug, rows, columns, textureConfig) {
      const [width, height] = getPackedMatrixTextureShapeWidthHeight(rows, columns);
      return createAndConfigureTexture(gl, debug, width, height, textureConfig.internalFormatPackedHalfFloat, gl.RGBA, textureConfig.textureTypeHalfFloat);
  }
  function bindVertexProgramAttributeStreams(gl, debug, program, vertexBuffer) {
      const posOffset = 0; // x is the first buffer element
      const uvOffset = 3 * 4; // uv comes after [x y z]
      const stride = (3 * 4) + (2 * 4); // xyz + uv, each entry is 4-byte float.
      callAndCheck(gl, debug, () => gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer));
      const success = bindVertexBufferToProgramAttribute(gl, debug, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);
      return success &&
          bindVertexBufferToProgramAttribute(gl, debug, program, 'uv', vertexBuffer, 2, stride, uvOffset);
  }
  function uploadDenseMatrixToTexture(gl, debug, texture, width, height, data, textureConfig) {
      callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, texture));
      let dataForUpload, texelDataType, internalFormat;
      if (data instanceof Uint8Array) {
          dataForUpload = new Uint8Array(width * height * 4);
          texelDataType = gl.UNSIGNED_BYTE;
          internalFormat = gl.RGBA;
      }
      else {
          dataForUpload = new Float32Array(width * height * 4);
          texelDataType = gl.FLOAT;
          internalFormat = textureConfig.internalFormatPackedFloat;
      }
      dataForUpload.set(data);
      callAndCheck(gl, debug, () => gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, gl.RGBA, texelDataType, dataForUpload));
      callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, null));
  }
  function uploadPixelDataToTexture(gl, debug, texture, pixels) {
      callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, texture));
      if (pixels.data instanceof Uint8Array) {
          callAndCheck(gl, debug, () => gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, pixels.width, pixels.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels.data));
      }
      else {
          callAndCheck(gl, debug, () => gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, pixels));
      }
      callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, null));
  }
  function createBufferFromOutputTexture(gl2, debug, rows, columns, textureConfig) {
      // Create and bind the buffer.
      const buffer = gl2.createBuffer();
      callAndCheck(gl2, debug, () => gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer));
      // Initialize the buffer to the size of the texture in bytes.
      const bytesPerFloat = 4;
      const valuesPerTexel = 4;
      const bufferSizeBytes = bytesPerFloat * valuesPerTexel * rows * columns;
      callAndCheck(gl2, debug, () => gl2.bufferData(gl2.PIXEL_PACK_BUFFER, bufferSizeBytes, gl2.STREAM_READ));
      // Enqueue a command on the GPU command queue to copy of texture into the
      // buffer.
      callAndCheck(gl2, debug, () => gl2.readPixels(0, 0, columns, rows, gl2.RGBA, gl2.FLOAT, 0));
      callAndCheck(gl2, debug, () => gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null));
      return buffer;
  }
  function downloadFloat32MatrixFromBuffer(gl, buffer, size) {
      const gl2 = gl;
      const downloadTarget = new Float32Array(size);
      gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
      gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, downloadTarget);
      gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);
      return downloadTarget;
  }
  function downloadByteEncodedFloatMatrixFromOutputTexture(gl, debug, rows, columns, textureConfig) {
      const [w, h] = getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
      const numChannels = 4;
      const downloadTarget = new Uint8Array(getUnpackedArraySizeFromMatrixSize(rows * columns, numChannels));
      callAndCheck(gl, debug, () => gl.readPixels(0, 0, w, h, textureConfig.downloadTextureFormat, gl.UNSIGNED_BYTE, downloadTarget));
      // By wrapping the buffer in a Float32Array, we use native browser IEEE 754
      // decoding of the 4 bytes that back each 32 bit float.
      return new Float32Array(downloadTarget.buffer);
  }
  function downloadPackedMatrixFromBuffer(gl, buffer, batch, rows, cols, physicalRows, physicalCols, textureConfig) {
      const gl2 = gl;
      const downloadTarget = new Float32Array(getPackedRGBAArraySizeFromMatrixShape(physicalRows, physicalCols));
      gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
      gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, downloadTarget);
      gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);
      return downloadTarget;
  }
  function downloadMatrixFromPackedOutputTexture(gl, debug, physicalRows, physicalCols) {
      const packedRGBA = new Float32Array(physicalRows * physicalCols * 4);
      callAndCheck(gl, debug, () => gl.readPixels(0, 0, physicalCols, physicalRows, gl.RGBA, gl.FLOAT, packedRGBA));
      return packedRGBA;
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class GPGPUContext {
      constructor(gl) {
          this.outputTexture = null;
          this.program = null;
          this.disposed = false;
          this.vertexAttrsAreBound = false;
          this.itemsToPoll = [];
          const glVersion = tf.env().getNumber('WEBGL_VERSION');
          if (gl != null) {
              this.gl = gl;
              setWebGLContext(glVersion, gl);
          }
          else {
              this.gl = getWebGLContext(glVersion);
          }
          // WebGL 2.0 enables texture floats without an extension.
          let COLOR_BUFFER_FLOAT = 'WEBGL_color_buffer_float';
          const COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
          if (tf.env().getNumber('WEBGL_VERSION') === 1) {
              const TEXTURE_FLOAT = 'OES_texture_float';
              const TEXTURE_HALF_FLOAT = 'OES_texture_half_float';
              this.textureFloatExtension =
                  getExtensionOrThrow(this.gl, this.debug, TEXTURE_FLOAT);
              if (hasExtension(this.gl, TEXTURE_HALF_FLOAT)) {
                  this.textureHalfFloatExtension = getExtensionOrThrow(this.gl, this.debug, TEXTURE_HALF_FLOAT);
              }
              else if (tf.env().get('WEBGL_FORCE_F16_TEXTURES')) {
                  throw new Error('GL context does not support half float textures, yet the ' +
                      'environment flag WEBGL_FORCE_F16_TEXTURES is set to true.');
              }
              this.colorBufferFloatExtension = this.gl.getExtension(COLOR_BUFFER_FLOAT);
              if (hasExtension(this.gl, COLOR_BUFFER_HALF_FLOAT)) {
                  this.colorBufferHalfFloatExtension = getExtensionOrThrow(this.gl, this.debug, COLOR_BUFFER_HALF_FLOAT);
              }
              else if (tf.env().get('WEBGL_FORCE_F16_TEXTURES')) {
                  throw new Error('GL context does not support color renderable half floats, yet ' +
                      'the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.');
              }
          }
          else {
              COLOR_BUFFER_FLOAT = 'EXT_color_buffer_float';
              if (hasExtension(this.gl, COLOR_BUFFER_FLOAT)) {
                  this.colorBufferFloatExtension =
                      this.gl.getExtension(COLOR_BUFFER_FLOAT);
              }
              else if (hasExtension(this.gl, COLOR_BUFFER_HALF_FLOAT)) {
                  this.colorBufferHalfFloatExtension =
                      this.gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
              }
              else {
                  throw new Error('GL context does not support color renderable floats');
              }
          }
          this.vertexBuffer = createVertexBuffer(this.gl, this.debug);
          this.indexBuffer = createIndexBuffer(this.gl, this.debug);
          this.framebuffer = createFramebuffer(this.gl, this.debug);
          this.textureConfig =
              getTextureConfig(this.gl, this.textureHalfFloatExtension);
      }
      get debug() {
          return tf.env().getBool('DEBUG');
      }
      dispose() {
          if (this.disposed) {
              return;
          }
          if (this.program != null) {
              console.warn('Disposing a GPGPUContext that still has a bound WebGLProgram.' +
                  ' This is probably a resource leak, delete the program with ' +
                  'GPGPUContext.deleteProgram before disposing.');
          }
          if (this.outputTexture != null) {
              console.warn('Disposing a GPGPUContext that still has a bound output matrix ' +
                  'texture.  This is probably a resource leak, delete the output ' +
                  'matrix texture with GPGPUContext.deleteMatrixTexture before ' +
                  'disposing.');
          }
          const gl = this.gl;
          callAndCheck(gl, this.debug, () => gl.finish());
          callAndCheck(gl, this.debug, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
          callAndCheck(gl, this.debug, () => gl.deleteFramebuffer(this.framebuffer));
          callAndCheck(gl, this.debug, () => gl.bindBuffer(gl.ARRAY_BUFFER, null));
          callAndCheck(gl, this.debug, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null));
          callAndCheck(gl, this.debug, () => gl.deleteBuffer(this.indexBuffer));
          this.disposed = true;
      }
      createFloat32MatrixTexture(rows, columns) {
          this.throwIfDisposed();
          return createFloat32MatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
      }
      createFloat16MatrixTexture(rows, columns) {
          this.throwIfDisposed();
          return createFloat16MatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
      }
      createUnsignedBytesMatrixTexture(rows, columns) {
          this.throwIfDisposed();
          return createUnsignedBytesMatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
      }
      uploadPixelDataToTexture(texture, pixels) {
          this.throwIfDisposed();
          uploadPixelDataToTexture(this.gl, this.debug, texture, pixels);
      }
      uploadDenseMatrixToTexture(texture, width, height, data) {
          this.throwIfDisposed();
          uploadDenseMatrixToTexture(this.gl, this.debug, texture, width, height, data, this.textureConfig);
      }
      createFloat16PackedMatrixTexture(rows, columns) {
          this.throwIfDisposed();
          return createFloat16PackedMatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
      }
      createPackedMatrixTexture(rows, columns) {
          this.throwIfDisposed();
          return createPackedMatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
      }
      deleteMatrixTexture(texture) {
          this.throwIfDisposed();
          if (this.outputTexture === texture) {
              unbindColorTextureFromFramebuffer(this.gl, this.debug, this.framebuffer);
              this.outputTexture = null;
          }
          callAndCheck(this.gl, this.debug, () => this.gl.deleteTexture(texture));
      }
      downloadByteEncodedFloatMatrixFromOutputTexture(texture, rows, columns) {
          return this.downloadMatrixDriver(texture, () => downloadByteEncodedFloatMatrixFromOutputTexture(this.gl, this.debug, rows, columns, this.textureConfig));
      }
      downloadPackedMatrixFromBuffer(buffer, batch, rows, columns, physicalRows, physicalCols) {
          return downloadPackedMatrixFromBuffer(this.gl, buffer, batch, rows, columns, physicalRows, physicalCols, this.textureConfig);
      }
      downloadFloat32MatrixFromBuffer(buffer, size) {
          return downloadFloat32MatrixFromBuffer(this.gl, buffer, size);
      }
      createBufferFromTexture(texture, rows, columns) {
          this.bindTextureToFrameBuffer(texture);
          const result = createBufferFromOutputTexture(this.gl, this.debug, rows, columns, this.textureConfig);
          this.unbindTextureToFrameBuffer();
          return result;
      }
      createAndWaitForFence() {
          const fenceContext = this.createFence(this.gl);
          return this.pollFence(fenceContext);
      }
      createFence(gl) {
          let query;
          let isFencePassed;
          if (tf.env().getBool('WEBGL_FENCE_API_ENABLED')) {
              const gl2 = gl;
              const sync = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
              gl.flush();
              isFencePassed = () => {
                  const status = gl2.clientWaitSync(sync, 0, 0);
                  return status === gl2.ALREADY_SIGNALED ||
                      status === gl2.CONDITION_SATISFIED;
              };
              query = sync;
          }
          else if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
              query = this.beginQuery();
              this.endQuery();
              isFencePassed = () => this.isQueryAvailable(query, tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'));
          }
          else {
              // If we have no way to fence, return true immediately. This will fire in
              // WebGL 1.0 when there is no disjoint query timer. In this case, because
              // the fence passes immediately, we'll immediately ask for a download of
              // the texture, which will cause the UI thread to hang.
              isFencePassed = () => true;
          }
          return { query, isFencePassed };
      }
      downloadMatrixFromPackedTexture(texture, physicalRows, physicalCols) {
          return this.downloadMatrixDriver(texture, () => downloadMatrixFromPackedOutputTexture(this.gl, this.debug, physicalRows, physicalCols));
      }
      createProgram(fragmentShaderSource) {
          this.throwIfDisposed();
          const gl = this.gl;
          const fragmentShader = createFragmentShader(gl, this.debug, fragmentShaderSource);
          const vertexShader = createVertexShader$1(gl, this.debug);
          const program = createProgram(gl, this.debug);
          callAndCheck(gl, this.debug, () => gl.attachShader(program, vertexShader));
          callAndCheck(gl, this.debug, () => gl.attachShader(program, fragmentShader));
          linkProgram(gl, this.debug, program);
          if (this.debug) {
              validateProgram(gl, this.debug, program);
          }
          if (!this.vertexAttrsAreBound) {
              this.setProgram(program);
              this.vertexAttrsAreBound = bindVertexProgramAttributeStreams(gl, this.debug, this.program, this.vertexBuffer);
          }
          return program;
      }
      deleteProgram(program) {
          this.throwIfDisposed();
          if (program === this.program) {
              this.program = null;
          }
          if (program != null) {
              callAndCheck(this.gl, this.debug, () => this.gl.deleteProgram(program));
          }
      }
      setProgram(program) {
          this.throwIfDisposed();
          this.program = program;
          if ((this.program != null) && this.debug) {
              validateProgram(this.gl, this.debug, this.program);
          }
          callAndCheck(this.gl, this.debug, () => this.gl.useProgram(program));
      }
      getUniformLocation(program, uniformName, shouldThrow = true) {
          this.throwIfDisposed();
          if (shouldThrow) {
              return getProgramUniformLocationOrThrow(this.gl, this.debug, program, uniformName);
          }
          else {
              return getProgramUniformLocation(this.gl, program, uniformName);
          }
      }
      getAttributeLocation(program, attribute) {
          this.throwIfDisposed();
          return callAndCheck(this.gl, this.debug, () => this.gl.getAttribLocation(program, attribute));
      }
      getUniformLocationNoThrow(program, uniformName) {
          this.throwIfDisposed();
          return this.gl.getUniformLocation(program, uniformName);
      }
      setInputMatrixTexture(inputMatrixTexture, uniformLocation, textureUnit) {
          this.throwIfDisposed();
          this.throwIfNoProgram();
          bindTextureToProgramUniformSampler(this.gl, this.debug, this.program, inputMatrixTexture, uniformLocation, textureUnit);
      }
      setOutputMatrixTexture(outputMatrixTexture, rows, columns) {
          this.setOutputMatrixTextureDriver(outputMatrixTexture, columns, rows);
      }
      setOutputPackedMatrixTexture(outputPackedMatrixTexture, rows, columns) {
          this.throwIfDisposed();
          const [width, height] = getPackedMatrixTextureShapeWidthHeight(rows, columns);
          this.setOutputMatrixTextureDriver(outputPackedMatrixTexture, width, height);
      }
      setOutputMatrixWriteRegion(startRow, numRows, startColumn, numColumns) {
          this.setOutputMatrixWriteRegionDriver(startColumn, startRow, numColumns, numRows);
      }
      setOutputPackedMatrixWriteRegion(startRow, numRows, startColumn, numColumns) {
          throw new Error('setOutputPackedMatrixWriteRegion not implemented.');
      }
      debugValidate() {
          if (this.program != null) {
              validateProgram(this.gl, this.debug, this.program);
          }
          validateFramebuffer(this.gl);
      }
      executeProgram() {
          this.throwIfDisposed();
          this.throwIfNoProgram();
          const gl = this.gl;
          if (this.debug) {
              this.debugValidate();
          }
          callAndCheck(gl, this.debug, () => gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0));
      }
      blockUntilAllProgramsCompleted() {
          this.throwIfDisposed();
          callAndCheck(this.gl, this.debug, () => this.gl.finish());
      }
      getQueryTimerExtension() {
          if (this.disjointQueryTimerExtension == null) {
              this.disjointQueryTimerExtension =
                  getExtensionOrThrow(this.gl, this.debug, tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2 ?
                      'EXT_disjoint_timer_query_webgl2' :
                      'EXT_disjoint_timer_query');
          }
          return this.disjointQueryTimerExtension;
      }
      getQueryTimerExtensionWebGL2() {
          return this.getQueryTimerExtension();
      }
      getQueryTimerExtensionWebGL1() {
          return this.getQueryTimerExtension();
      }
      beginQuery() {
          if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
              const gl2 = this.gl;
              const ext = this.getQueryTimerExtensionWebGL2();
              const query = gl2.createQuery();
              gl2.beginQuery(ext.TIME_ELAPSED_EXT, query);
              return query;
          }
          const ext = this.getQueryTimerExtensionWebGL1();
          const query = ext.createQueryEXT();
          ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query);
          return query;
      }
      endQuery() {
          if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
              const gl2 = this.gl;
              const ext = this.getQueryTimerExtensionWebGL2();
              gl2.endQuery(ext.TIME_ELAPSED_EXT);
              return;
          }
          const ext = this.getQueryTimerExtensionWebGL1();
          ext.endQueryEXT(ext.TIME_ELAPSED_EXT);
      }
      async waitForQueryAndGetTime(query) {
          await tf.util.repeatedTry(() => this.disposed || // while testing contexts are created / disposed
              // in rapid succession, so without this check we
              // may poll for the query timer indefinitely
              this.isQueryAvailable(query, tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION')));
          return this.getQueryTime(query, tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'));
      }
      getQueryTime(query, queryTimerVersion) {
          if (queryTimerVersion === 0) {
              return null;
          }
          if (queryTimerVersion === 2) {
              const gl2 = this.gl;
              const timeElapsedNanos = gl2.getQueryParameter(query, gl2.QUERY_RESULT);
              // Return milliseconds.
              return timeElapsedNanos / 1000000;
          }
          else {
              const ext = this.getQueryTimerExtensionWebGL1();
              const timeElapsedNanos = ext.getQueryObjectEXT(query, ext.QUERY_RESULT_EXT);
              // Return milliseconds.
              return timeElapsedNanos / 1000000;
          }
      }
      isQueryAvailable(query, queryTimerVersion) {
          if (queryTimerVersion === 0) {
              return true;
          }
          if (queryTimerVersion === 2) {
              const gl2 = this.gl;
              const ext = this.getQueryTimerExtensionWebGL2();
              const available = gl2.getQueryParameter(query, gl2.QUERY_RESULT_AVAILABLE);
              if (this.disjoint == null) {
                  this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
              }
              return available && !this.disjoint;
          }
          else {
              const ext = this.getQueryTimerExtensionWebGL1();
              const available = ext.getQueryObjectEXT(query, ext.QUERY_RESULT_AVAILABLE_EXT);
              if (this.disjoint == null) {
                  this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
              }
              return available && !this.disjoint;
          }
      }
      pollFence(fenceContext) {
          return new Promise(resolve => {
              this.addItemToPoll(() => fenceContext.isFencePassed(), () => resolve());
          });
      }
      pollItems() {
          // Find the last query that has finished.
          const index = linearSearchLastTrue(this.itemsToPoll.map(x => x.isDoneFn));
          for (let i = 0; i <= index; ++i) {
              const { resolveFn } = this.itemsToPoll[i];
              resolveFn();
          }
          this.itemsToPoll = this.itemsToPoll.slice(index + 1);
      }
      addItemToPoll(isDoneFn, resolveFn) {
          this.itemsToPoll.push({ isDoneFn, resolveFn });
          if (this.itemsToPoll.length > 1) {
              // We already have a running loop that polls.
              return;
          }
          // Start a new loop that polls.
          tf.util.repeatedTry(() => {
              this.pollItems();
              // End the loop if no more items to poll.
              return this.itemsToPoll.length === 0;
          });
      }
      bindTextureToFrameBuffer(texture) {
          this.throwIfDisposed();
          bindColorTextureToFramebuffer(this.gl, this.debug, texture, this.framebuffer);
          if (this.debug) {
              validateFramebuffer(this.gl);
          }
      }
      unbindTextureToFrameBuffer() {
          if (this.outputTexture != null) {
              bindColorTextureToFramebuffer(this.gl, this.debug, this.outputTexture, this.framebuffer);
              if (this.debug) {
                  validateFramebuffer(this.gl);
              }
          }
          else {
              unbindColorTextureFromFramebuffer(this.gl, this.debug, this.framebuffer);
          }
      }
      downloadMatrixDriver(texture, downloadAndDecode) {
          this.bindTextureToFrameBuffer(texture);
          const result = downloadAndDecode();
          this.unbindTextureToFrameBuffer();
          return result;
      }
      setOutputMatrixTextureDriver(outputMatrixTextureMaybePacked, width, height) {
          this.throwIfDisposed();
          const gl = this.gl;
          bindColorTextureToFramebuffer(gl, this.debug, outputMatrixTextureMaybePacked, this.framebuffer);
          if (this.debug) {
              validateFramebuffer(gl);
          }
          this.outputTexture = outputMatrixTextureMaybePacked;
          callAndCheck(gl, this.debug, () => gl.viewport(0, 0, width, height));
          callAndCheck(gl, this.debug, () => gl.scissor(0, 0, width, height));
      }
      setOutputMatrixWriteRegionDriver(x, y, width, height) {
          this.throwIfDisposed();
          callAndCheck(this.gl, this.debug, () => this.gl.scissor(x, y, width, height));
      }
      throwIfDisposed() {
          if (this.disposed) {
              throw new Error('Attempted to use disposed GPGPUContext.');
          }
      }
      throwIfNoProgram() {
          if (this.program == null) {
              throw new Error('No GPU program is currently set.');
          }
      }
  }
  /**
   * Finds the index of the last true element using linear search.
   * Note: We can't do binary search because Chrome expects us to explicitly
   * test all fences before download:
   * https://github.com/tensorflow/tfjs/issues/1145
   */
  function linearSearchLastTrue(arr) {
      let i = 0;
      for (; i < arr.length; ++i) {
          const isDone = arr[i]();
          if (!isDone) {
              break;
          }
      }
      return i - 1;
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  function compileProgram(gpgpu, program, inputs, output) {
      const userCode = program.userCode;
      const inputInfos = inputs.map((input, i) => {
          const shapeInfo = {
              logicalShape: input.shape,
              texShape: input.isUniform ? null : input.texData.texShape,
              isUniform: input.isUniform,
              isPacked: input.isUniform ? false : input.texData.isPacked,
              flatOffset: null
          };
          if (input.texData != null && input.texData.slice != null &&
              input.texData.slice.flatOffset > 0) {
              shapeInfo.flatOffset = input.texData.slice.flatOffset;
          }
          return { name: program.variableNames[i], shapeInfo };
      });
      const inShapeInfos = inputInfos.map(x => x.shapeInfo);
      const outShapeInfo = {
          logicalShape: output.shape,
          texShape: output.texData.texShape,
          isUniform: false,
          isPacked: output.texData.isPacked,
          flatOffset: null
      };
      const source = makeShader(inputInfos, outShapeInfo, userCode, program.packedInputs);
      const webGLProgram = gpgpu.createProgram(source);
      // Add special uniforms (NAN, INFINITY)
      let infLoc = null;
      const nanLoc = gpgpu.getUniformLocation(webGLProgram, 'NAN', false);
      if (tf.env().getNumber('WEBGL_VERSION') === 1) {
          infLoc = gpgpu.getUniformLocation(webGLProgram, 'INFINITY', false);
      }
      // Add user-defined uniforms
      const uniformLocations = {};
      for (let i = 0; i < program.variableNames.length; i++) {
          const varName = program.variableNames[i];
          const shouldThrow = false;
          uniformLocations[varName] =
              gpgpu.getUniformLocation(webGLProgram, varName, shouldThrow);
          uniformLocations[`offset${varName}`] =
              gpgpu.getUniformLocation(webGLProgram, `offset${varName}`, shouldThrow);
      }
      return {
          program,
          source,
          webGLProgram,
          uniformLocations,
          inShapeInfos,
          outShapeInfo,
          infLoc,
          nanLoc,
      };
  }
  function validateBinaryAndProgram(shapeInfos, inputs) {
      if (shapeInfos.length !== inputs.length) {
          throw Error(`Binary was compiled with ${shapeInfos.length} inputs, but ` +
              `was executed with ${inputs.length} inputs`);
      }
      shapeInfos.forEach((s, i) => {
          const shapeA = s.logicalShape;
          const input = inputs[i];
          const shapeB = input.shape;
          if (!tf.util.arraysEqual(shapeA, shapeB)) {
              throw Error(`Binary was compiled with different shapes than ` +
                  `the current args. Shapes ${shapeA} and ${shapeB} must match`);
          }
          // The input is uploaded as uniform.
          if (s.isUniform && input.isUniform) {
              return;
          }
          const texShapeA = s.texShape;
          const texShapeB = input.isUniform ? null : input.texData.texShape;
          if (!tf.util.arraysEqual(texShapeA, texShapeB)) {
              throw Error(`Binary was compiled with different texture shapes than the` +
                  ` current args. Shape ${texShapeA} and ${texShapeB} must match`);
          }
      });
  }
  function runProgram(gpgpu, binary, inputs, output, customSetup) {
      validateBinaryAndProgram(binary.inShapeInfos, inputs);
      validateBinaryAndProgram([binary.outShapeInfo], [output]);
      const outTex = output.texData.texture;
      const outTexShape = output.texData.texShape;
      if (output.texData.isPacked) {
          gpgpu.setOutputPackedMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
      }
      else {
          gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
      }
      gpgpu.setProgram(binary.webGLProgram);
      // Set special uniforms (NAN, INFINITY)
      if (tf.env().getNumber('WEBGL_VERSION') === 1) {
          if (binary.infLoc !== null) {
              gpgpu.gl.uniform1f(binary.infLoc, Infinity);
          }
      }
      if (binary.nanLoc !== null) {
          gpgpu.gl.uniform1f(binary.nanLoc, NaN);
      }
      // Set user-defined inputs
      inputs.forEach((input, i) => {
          const varName = binary.program.variableNames[i];
          const varLoc = binary.uniformLocations[varName];
          const varOffsetLoc = binary.uniformLocations[`offset${varName}`];
          if (varLoc == null) {
              // The compiler inferred that this variable is not used in this shader.
              return;
          }
          if (input.isUniform) {
              // Upload the values of the tensor as uniform.
              if (tf.util.sizeFromShape(input.shape) < 2) {
                  gpgpu.gl.uniform1f(varLoc, input.uniformValues[0]);
              }
              else {
                  let vals = input.uniformValues;
                  if (!(vals instanceof Float32Array)) {
                      vals = new Float32Array(vals);
                  }
                  gpgpu.gl.uniform1fv(varLoc, vals);
              }
              return;
          }
          // If the input was sliced, upload the flat offset index.
          if (input.texData.slice != null && varOffsetLoc != null) {
              gpgpu.gl.uniform1i(varOffsetLoc, input.texData.slice.flatOffset);
          }
          gpgpu.setInputMatrixTexture(input.texData.texture, varLoc, i);
      });
      if (customSetup != null) {
          customSetup(gpgpu, binary.webGLProgram);
      }
      gpgpu.executeProgram();
  }
  function makeShaderKey(program, inputs, output) {
      let keyInputs = '';
      inputs.concat(output).forEach(x => {
          const hasOffset = x.texData != null && x.texData.slice != null &&
              x.texData.slice.flatOffset > 0;
          const texShape = x.isUniform ? 'uniform' : x.texData.texShape;
          keyInputs += `${x.shape}_${texShape}_${hasOffset}`;
      });
      const keyUserCode = program.userCode;
      let key = program.constructor.name;
      // Fast string concat. See https://jsperf.com/string-concatenation/14.
      key += '_' + keyInputs + '_' + keyUserCode;
      return key;
  }

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
  class Im2ColPackedProgram {
      constructor(outputShape, inputShape, convInfo) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = outputShape;
          const { filterWidth, inChannels, strideWidth, strideHeight, padInfo, outWidth, dilationWidth, dilationHeight, dataFormat } = convInfo;
          const { left, top } = padInfo;
          const itemsPerBlockRow = inChannels * filterWidth;
          const glsl = getGlslDifferences();
          const isChannelsLast = dataFormat === 'channelsLast';
          const rowDim = isChannelsLast ? 0 : 1;
          const colDim = isChannelsLast ? 1 : 2;
          let unrolled = ``;
          for (let row = 0; row <= 1; row++) {
              for (let col = 0; col <= 1; col++) {
                  unrolled += `
          blockIndex = rc.y + ${col};
          pos = rc.x + ${row};

          if(blockIndex < ${outputShape[1]} && pos < ${outputShape[0]}) {
            offsetY = int(blockIndex / (${outWidth})) * ${strideHeight} - ${top};
            d0 = offsetY + ${dilationHeight} * (pos / ${itemsPerBlockRow});

            if(d0 < ${inputShape[rowDim]} && d0 >= 0) {

              offsetX = int(mod(float(blockIndex), ${outWidth}.) * ${strideWidth}. - ${left}.);
              d1 = offsetX + ${dilationWidth} * (int(mod(float(pos), ${itemsPerBlockRow}.) / ${inChannels}.));

              if(d1 < ${inputShape[colDim]} && d1 >= 0) {

                ch = int(mod(float(pos), ${inChannels}.));

                if (${isChannelsLast}) {
                  innerDims = vec2(d1, ch);
                  result[${row * 2 + col}] = getChannel(
                    getA(d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${row * 2 + col}] = getChannel(
                    getA(ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;
              }
          }
          this.userCode = `
      void main() {
        ivec2 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${unrolled}

        ${glsl.output} = result;
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class LRNProgram {
      constructor(xShape, radius, bias, alpha, beta) {
          this.variableNames = ['x'];
          this.outputShape = [];
          const rad = radius;
          const maxD = xShape[3] - 1;
          this.outputShape = xShape;
          // optimize pow(bias + alpha * sum, -beta)
          // src: https://github.com/tensorflow/tensorflow/..
          // blob/26033a1644a9c4a5fbe3170ab2e864b6a4ccd4ca/..
          // tensorflow/core/kernels/mkl_lrn_op.cc#L320
          let powOperator;
          const basis = `float(${bias}) + float(${alpha}) * sum`;
          if (beta === 0.5) {
              powOperator = `inversesqrt(${basis})`;
          }
          else if (beta === 1.0) {
              powOperator = `1.0/(${basis})`;
          }
          else {
              powOperator = `exp(log(${basis}) * float(-${beta}));`;
          }
          this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${rad}; j <= ${rad}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${maxD}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${powOperator};
        setOutput(val);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class LRNGradProgram {
      constructor(inputShape, depthRadius, bias, alpha, beta) {
          this.variableNames = ['inputImage', 'outputImage', 'dy'];
          this.outputShape = [];
          this.outputShape = inputShape;
          this.depth = inputShape[3];
          this.depthRadius = depthRadius;
          this.bias = bias;
          this.alpha = alpha;
          this.beta = beta;
          this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${depthRadius})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${depthRadius} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${alpha}) * norm + float(${bias});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${alpha})
                * float(${beta})
                * getInputImage(b ,r ,c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${beta});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2019 Google LLC All Rights Reserved.
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
  class LRNPackedProgram {
      constructor(xShape, radius, bias, alpha, beta) {
          this.variableNames = ['x'];
          this.outputShape = [];
          this.packedInputs = true;
          this.packedOutput = true;
          const rad = radius;
          const maxD = xShape[3] - 1;
          this.outputShape = xShape;
          // optimize pow(bias + alpha * sum, -beta)
          // src: https://github.com/tensorflow/tensorflow/..
          // blob/26033a1644a9c4a5fbe3170ab2e864b6a4ccd4ca/..
          // tensorflow/core/kernels/mkl_lrn_op.cc#L320
          let powOperator;
          const basis = `float(${bias}) + float(${alpha}) * sum`;
          if (beta === 0.5) {
              powOperator = `inversesqrt(${basis})`;
          }
          else if (beta === 1.0) {
              powOperator = `1.0/(${basis})`;
          }
          else {
              powOperator = `exp(log(${basis}) * float(-${beta}));`;
          }
          this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${rad};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${rad}; j <= ${rad}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${maxD}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${powOperator};
        setOutput(result);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class MaxPool2DBackpropProgram {
      constructor(convInfo) {
          this.variableNames = ['dy', 'maxPos'];
          this.outputShape = convInfo.inShape;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationHeight = convInfo.dilationHeight;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          const lastIndex = effectiveFilterHeight * effectiveFilterWidth - 1;
          this.userCode = `
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${effectiveFilterHeight};
          wR += ${dilationHeight}) {
          float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

          if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${effectiveFilterWidth}; wC++) {
            float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

            if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${lastIndex} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${effectiveFilterWidth} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }
  class MaxPool3DBackpropProgram {
      constructor(convInfo) {
          this.variableNames = ['dy', 'maxPos'];
          this.outputShape = convInfo.inShape;
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterDepth = convInfo.effectiveFilterDepth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
          const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
          const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
          const lastIndex = effectiveFilterDepth * effectiveFilterHeight * effectiveFilterWidth - 1;
          this.userCode = `
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${effectiveFilterDepth};
           wD += ${dilationDepth}) {
          float dyD = float(dyDCorner + wD) / ${strideDepth}.0;

          if (dyD < 0.0 || dyD >= ${convInfo.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${effectiveFilterHeight};
              wR += ${dilationHeight}) {
            float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

            if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${effectiveFilterWidth};
                wC += ${dilationWidth}) {
              float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

              if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${lastIndex} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${effectiveFilterHeight} * ${effectiveFilterWidth} +
                  wR * ${effectiveFilterWidth} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class MatMulPackedProgram {
      constructor(aShape, outputShape, transposeA = false, transposeB = false, addBias = false, activation = null, hasPreluActivation = false) {
          this.variableNames = ['matrixA', 'matrixB'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = outputShape;
          const sharedDim = transposeA ? aShape[1] : aShape[2];
          const sharedDimensionPacked = Math.ceil(sharedDim / 2);
          const aSample = transposeA ? 'i * 2, rc.y' : 'rc.y, i * 2';
          const bSample = transposeB ? 'rc.z, i * 2' : 'i * 2, rc.z';
          const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
          const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];
          let activationSnippet = '', applyActivationSnippet = '';
          if (activation) {
              if (hasPreluActivation) {
                  activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
              }
              else {
                  activationSnippet = `vec4 activation(vec4 x) {
          ${activation}
        }`;
              }
              applyActivationSnippet = `result = activation(result);`;
          }
          const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
          if (addBias) {
              this.variableNames.push('bias');
          }
          if (hasPreluActivation) {
              this.variableNames.push('preluActivationWeights');
          }
          this.userCode = `
      ${activationSnippet}

      const float sharedDimension = ${sharedDimensionPacked}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        for (int i = 0; i < ${sharedDimensionPacked}; i++) {
          vec4 a = getMatrixA(rc.x, ${aSample});
          vec4 b = getMatrixB(rc.x, ${bSample});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${aSwizzle[0]} * ${bSwizzle[0]});
          result += (${aSwizzle[1]} * ${bSwizzle[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${addBiasSnippet}

        ${applyActivationSnippet}

        setOutput(result);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class MultinomialProgram {
      constructor(batchSize, numOutcomes, numSamples) {
          this.variableNames = ['probs'];
          this.outputShape = [batchSize, numSamples];
          this.userCode = `
      uniform float seed;

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${numOutcomes - 1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${numOutcomes - 1}));
      }
    `;
      }
      getCustomSetupFunc(seed) {
          return (gpgpu, webGLProgram) => {
              if (this.seedLoc == null) {
                  this.seedLoc = gpgpu.getUniformLocation(webGLProgram, 'seed');
              }
              gpgpu.gl.uniform1f(this.seedLoc, seed);
          };
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class OneHotProgram {
      constructor(numIndices, depth, onValue, offValue) {
          this.variableNames = ['indices'];
          this.outputShape = [numIndices, depth];
          this.userCode = `
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${offValue}), float(${onValue}),
                      float(index == coords.y)));
      }
    `;
      }
  }

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
  class PackProgram {
      constructor(outputShape) {
          this.variableNames = ['A'];
          this.packedInputs = false;
          this.packedOutput = true;
          // Only input / output 3D tensors.
          this.outputShape = outputShape;
          const rank = outputShape.length;
          if (rank === 0) {
              this.userCode = `
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;
          }
          else {
              const channels = getChannels('rc', rank);
              const dtype = getCoordsDataType(rank);
              const outOfBoundsCondition = getOutOfBoundsCondition(rank, outputShape, channels);
              const setup = getSetup(rank, outputShape[outputShape.length - 1], outputShape[outputShape.length - 2], channels);
              const output = getOutput(outputShape, channels);
              this.userCode = `
        void main() {
          ${dtype} rc = getOutputCoords();

          if(${outOfBoundsCondition}) {
            setOutput(vec4(0));
          } else {
            ${setup}

            setOutput(vec4(${output}));
          }
        }
      `;
          }
      }
  }
  function getSourceCoordsArr(rank, dims) {
      const coords = [];
      for (let row = 0; row <= 1; row++) {
          for (let col = 0; col <= 1; col++) {
              let coord = `${row === 0 ? 'r' : 'rp1'}, ${col === 0 ? 'c' : 'cp1'}`;
              for (let d = 2; d < rank; d++) {
                  coord = `${dims[dims.length - 1 - d]},` + coord;
              }
              coords.push(coord);
          }
      }
      return coords;
  }
  function getOutOfBoundsCondition(rank, shape, dims) {
      if (rank === 1) {
          return `rc > ${shape[0]}`;
      }
      let cond = '';
      for (let i = rank - 2; i < rank; i++) {
          cond += `${dims[i]} >= ${shape[i]}`;
          if (i < rank - 1) {
              cond += '||';
          }
      }
      return cond;
  }
  function getSetup(rank, cols, rows, dims) {
      if (rank === 1) {
          return '';
      }
      const innerDims = dims.slice(-2);
      return `
    int r = ${innerDims[0]};
    int c = ${innerDims[1]};
    int rp1 = r + 1;
    int cp1 = c + 1;

    bool cEdge = cp1 >= ${cols};
    bool rEdge = rp1 >= ${rows};
  `;
  }
  function getOutput(shape, dims) {
      const rank = shape.length;
      const sourceCoords = getSourceCoordsArr(rank, dims);
      if (rank === 1) {
          return `getA(rc),
            rc + 1 >= ${shape[0]} ? 0. : getA(rc + 1),
            0, 0`;
      }
      return `getA(${sourceCoords[0]}),
          cEdge ? 0. : getA(${sourceCoords[1]}),
          rEdge ? 0. : getA(${sourceCoords[2]}),
          rEdge || cEdge ? 0. : getA(${sourceCoords[3]})`;
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class PadProgram {
      constructor(xShape, paddings, constantValue) {
          this.variableNames = ['x'];
          this.outputShape = paddings.map((p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
          const rank = xShape.length;
          const type = getCoordsDataType(rank);
          const start = paddings.map(p => p[0]).join(',');
          const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
          const unpackedCoords = ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank);
          if (rank === 1) {
              this.userCode = `
        int start = ${start};
        int end = ${end};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(float(${constantValue}));
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;
              return;
          }
          this.userCode = `
      ${type} start = ${type}(${start});
      ${type} end = ${type}(${end});

      void main() {
        ${type} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(float(${constantValue}));
        } else {
          ${type} coords = outC - start;
          setOutput(getX(${unpackedCoords}));
        }
      }
    `;
      }
  }

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
  class PadPackedProgram {
      constructor(xShape, paddings, constantValue) {
          this.variableNames = ['x'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = paddings.map((p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
          const rank = xShape.length;
          const dtype = getCoordsDataType(rank);
          const start = paddings.map(p => p[0]).join(',');
          const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
          const coords = getChannels('rc', rank);
          const source = getChannels('source', rank);
          const cLimit = `${coords[rank - 1]} < ${this.outputShape[rank - 1]}`;
          const innerDims = rank === 1 ? 'source' : `vec2(${source.slice(-2).join()})`;
          const componentSetup = [
              `${dtype} rc = outputLoc;`, `${coords[rank - 1]} += 1;
       if(${cLimit}) {
      `,
              rank === 1 ? '' : `}
       rc = outputLoc;
       ${coords[rank - 2]} += 1;
       if(${coords[rank - 2]} < ${this.outputShape[rank - 2]}) {`,
              rank === 1 ? '' : `  ${coords[rank - 1]} += 1;
         if(${cLimit}) {`
          ];
          const paddingArea = rank === 1 ?
              'rc < start || rc >= end' :
              'any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))';
          let mainLoop = '';
          for (let i = 0, j = rank === 1 ? 2 : 4; i < j; i++) {
              mainLoop += `
        ${componentSetup[i]}
        if (${paddingArea}) {
          result[${i}] = float(${constantValue});
        } else {
          ${dtype} source = rc - start;
          result[${i}] = getChannel(getX(${source.join()}), ${innerDims});
        }
      `;
          }
          mainLoop += (rank === 1 ? `} ` : `}}`);
          this.userCode = `
      const ${dtype} start = ${dtype}(${start});
      const ${dtype} end = ${dtype}(${end});

      void main() {
        ${dtype} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${mainLoop}
        setOutput(result);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class Pool2DProgram {
      constructor(convInfo, poolType, computePositions, flattenPositions = false, includeBatchInIndex = false) {
          this.variableNames = ['x'];
          if (poolType === 'avg' && computePositions) {
              throw new Error('Cannot compute positions for average pool.');
          }
          const filterWidth = convInfo.filterWidth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          this.outputShape = convInfo.outShape;
          const isAvgPool = poolType === 'avg';
          const batchFlattenPositionStr = `((batch  * ${convInfo.inHeight} + xR) * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + d`;
          const flattenPositionStr = `(xR * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + d`;
          let initializationValue = '0.0';
          if (!isAvgPool) {
              // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
              initializationValue = '-1.0 / 1e-20';
          }
          if (computePositions) {
              const compareOp = '>=';
              this.userCode = `
        const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
        const ivec2 pads = ivec2(${padTop}, ${padLeft});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${effectiveFilterHeight};
              wR += ${dilationHeight}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${effectiveFilterWidth};
                wC += ${dilationWidth}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${convInfo.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value ${compareOp} currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${flattenPositions ? (includeBatchInIndex ? batchFlattenPositionStr :
                flattenPositionStr) :
                `wR * ${effectiveFilterWidth} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;
              return;
          }
          const compareOp = 'max';
          let returnValue = `${poolType}(${poolType}(${poolType}(` +
              'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
          if (poolType === 'avg') {
              returnValue = `avgValue / count`;
          }
          const filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
          const filterWidthVec4Remainder = filterWidth % 4;
          const updateSnippet = `
      if (${isAvgPool}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;
          this.userCode = `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});
      const float initializationValue = ${initializationValue};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${convInfo.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${initializationValue});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${effectiveFilterHeight};
            wR += ${dilationHeight}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidthNearestVec4}; wC += 4) {
            int xC = xCCorner + wC * ${dilationWidth};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              getValue(batch, xR, xC + 2 * ${dilationWidth}, d),
              getValue(batch, xR, xC + 3 * ${dilationWidth}, d)
            );

            ${updateSnippet}
          }

          int xC = xCCorner + ${filterWidthNearestVec4};
          if (${filterWidthVec4Remainder === 1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${updateSnippet}
          } else if (${filterWidthVec4Remainder === 2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              initializationValue,
              initializationValue
            );

            ${updateSnippet}
          } else if (${filterWidthVec4Remainder === 3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              getValue(batch, xR, xC + 2 * ${dilationWidth}, d),
              initializationValue
            );

            ${updateSnippet}
          }
        }
        setOutput(${returnValue});
      }
    `;
      }
  }
  class Pool3DProgram {
      constructor(convInfo, poolType, computePositions, flattenPositions = false, includeBatchInIndex = false) {
          this.variableNames = ['x'];
          if (poolType === 'avg' && computePositions) {
              throw new Error('Cannot compute positions for average pool.');
          }
          const filterWidth = convInfo.filterWidth;
          const strideDepth = convInfo.strideDepth;
          const strideHeight = convInfo.strideHeight;
          const strideWidth = convInfo.strideWidth;
          const dilationDepth = convInfo.dilationDepth;
          const dilationHeight = convInfo.dilationHeight;
          const dilationWidth = convInfo.dilationWidth;
          const effectiveFilterDepth = convInfo.effectiveFilterDepth;
          const effectiveFilterHeight = convInfo.effectiveFilterHeight;
          const effectiveFilterWidth = convInfo.effectiveFilterWidth;
          const padFront = convInfo.padInfo.front;
          const padTop = convInfo.padInfo.top;
          const padLeft = convInfo.padInfo.left;
          this.outputShape = convInfo.outShape;
          const isAvgPool = poolType === 'avg';
          let initializationValue = '0.0';
          if (!isAvgPool) {
              // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
              initializationValue = '-1.0 / 1e-20';
          }
          if (computePositions) {
              const compareOp = '>=';
              this.userCode = `
        const ivec3 strides =
            ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${effectiveFilterDepth};
              wD += ${dilationDepth}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${convInfo.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${effectiveFilterHeight};
                wR += ${dilationHeight}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${convInfo.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${effectiveFilterWidth};
                  wC += ${dilationWidth}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${convInfo.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value ${compareOp} currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${flattenPositions ?
                (includeBatchInIndex ?
                    `(((batch * ${convInfo.inDepth} + xD) * ${convInfo.inHeight} + xR) * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + ch` :
                    `((xD * ${convInfo.inHeight} + xR) * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + ch`) :
                `wD * ${effectiveFilterHeight} * ${effectiveFilterWidth} +
                      wR * ${effectiveFilterWidth} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;
              return;
          }
          const compareOp = 'max';
          let returnValue = `${poolType}(${poolType}(${poolType}(` +
              'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
          if (poolType === 'avg') {
              returnValue = `avgValue / count`;
          }
          const filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
          const filterWidthVec4Remainder = filterWidth % 4;
          const updateSnippet = `
      if (${isAvgPool}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;
          this.userCode = `
      const ivec3 strides =
        ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});
      const float initializationValue = ${initializationValue};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${convInfo.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${initializationValue});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${effectiveFilterDepth};
            wD += ${dilationDepth}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${convInfo.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${effectiveFilterHeight};
            wR += ${dilationHeight}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${filterWidthNearestVec4}; wC += 4) {
              int xC = xCCorner + wC * ${dilationWidth};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 2 * ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 3 * ${dilationWidth}, ch)
              );

              ${updateSnippet}
            }

            int xC = xCCorner + ${filterWidthNearestVec4};
            if (${filterWidthVec4Remainder === 1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${updateSnippet}
            } else if (${filterWidthVec4Remainder === 2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                initializationValue,
                initializationValue
              );

              ${updateSnippet}
            } else if (${filterWidthVec4Remainder === 3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 2 * ${dilationWidth}, ch),
                initializationValue
              );

              ${updateSnippet}
            }
          }
          setOutput(${returnValue});
        }
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class ReduceProgram {
      constructor(reduceInfo, reduceType) {
          this.variableNames = ['x'];
          const windowSize = reduceInfo.windowSize;
          const batchSize = reduceInfo.batchSize;
          const inSize = reduceInfo.inSize;
          const outSize = Math.ceil(inSize / windowSize);
          this.outputShape = [batchSize, outSize];
          let initializationValue = '0.0';
          let compareOp = ``;
          if (reduceType === 'prod') {
              initializationValue = '1.0';
          }
          else if (reduceType === 'min') {
              // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
              initializationValue = '1.0 / 1e-20';
              compareOp = `min`;
          }
          else if (reduceType === 'max') {
              // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
              initializationValue = '-1.0 / 1e-20';
              compareOp = `max`;
          }
          let returnValue = `${reduceType}(${reduceType}(${reduceType}(` +
              'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
          if (reduceType === 'sum') {
              returnValue = `sumValue`;
          }
          else if (reduceType === 'prod') {
              returnValue = `prodValue`;
          }
          else if (reduceType === 'all') {
              returnValue = `allValue`;
          }
          else if (reduceType === 'any') {
              returnValue = `anyValue`;
          }
          const windowSizeNearestVec4 = Math.floor(windowSize / 4) * 4;
          const windowSizeVec4Remainder = windowSize % 4;
          let updateSnippet = `
      if (${reduceType === 'sum'}) {
        sumValue += dot(values, ones);
      } else if (${reduceType === 'prod'}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;
          let vecType = `vec4`;
          if (reduceType === 'all') {
              initializationValue = '1.0';
              updateSnippet = `
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `;
              vecType = `bvec4`;
          }
          else if (reduceType === 'any') {
              initializationValue = '0.0';
              updateSnippet = `
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `;
              vecType = `bvec4`;
          }
          let checkOutOfBounds = '';
          if (inSize % windowSize > 0) {
              checkOutOfBounds = `
        if (inIdx < 0 || inIdx >= ${inSize}) {
          return initializationValue;
        }
      `;
          }
          this.userCode = `
      const float initializationValue = ${initializationValue};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${checkOutOfBounds}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${windowSize};

        vec4 minMaxValue = vec4(${initializationValue});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${windowSizeNearestVec4}; i += 4) {
          int inIdx = inOffset + i;
          ${vecType} values = ${vecType}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${updateSnippet}
        }

        int inIdx = inOffset + ${windowSizeNearestVec4};
        if (${windowSizeVec4Remainder === 1}) {
          ${vecType} values = ${vecType}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${updateSnippet}
        } else if (${windowSizeVec4Remainder === 2}) {
          ${vecType} values = ${vecType}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${updateSnippet}
        } else if (${windowSizeVec4Remainder === 3}) {
          ${vecType} values = ${vecType}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${updateSnippet}
        }
        setOutput(${returnValue});
      }
    `;
      }
  }

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
  class ReshapePackedProgram {
      constructor(outputShape, inputShape) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = outputShape;
          let mainLoop = ``;
          for (let i = 0; i < 4; i++) {
              let thisRC = `thisRC = rc;`;
              if (i % 2 === 1) {
                  thisRC += `thisRC.z += 1;`;
              }
              if (i > 1) {
                  thisRC += `thisRC.y += 1;`;
              }
              mainLoop += `
        ${thisRC}
        ${i > 0 ? `if(thisRC.y < rows && thisRC.z < cols){` : ''}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${i}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${i > 0 ? '}' : ''}
      `;
          }
          this.userCode = `
      ${getReshapedInputCoords(inputShape)}
      ${getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${outputShape[1]};
        int cols = ${outputShape[2]};

        ${mainLoop}

        setOutput(result);
      }
    `;
      }
  }
  function getReshapedInputCoords(shape) {
      const coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);
      return `
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${coordsFromIndexSnippet}
      return ivec3(r, c, d);
    }
  `;
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class ResizeBilinearBackpropProgram {
      constructor(dy, x, alignCorners) {
          this.variableNames = ['dy'];
          this.outputShape = [];
          this.outputShape = x.shape;
          const [, xHeight, xWidth,] = x.shape;
          const [, yHeight, yWidth] = dy.shape;
          // In the backwards pass, we want to find the pixels that were generated for
          // each pixel in the input image the forward pass and add the corresponding
          // coefficient from dy to the gradient (with some interpolation).
          const effectiveXSize = [
              (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
              (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
          ];
          const effectiveYSize = [
              (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
              (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
          ];
          const heightScale = effectiveXSize[0] / effectiveYSize[0];
          const widthScale = effectiveXSize[1] / effectiveYSize[1];
          const invHeightScale = 1 / heightScale;
          const invWidthScale = 1 / widthScale;
          // This defines the size of the window of values around a particular
          // index in dy that we want to search for contributions to dx.
          const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
          const winWidth = (Math.ceil(invWidthScale) * 2) + 2;
          this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${heightScale});
        const float widthScale = float(${widthScale});

        const float invHeightScale = float(${invHeightScale});
        const float invWidthScale = float(${invWidthScale});

        const int winHeight = int(${winHeight});
        const int winWidth = int(${winWidth});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${yHeight}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${yWidth}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${xHeight - 1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${xWidth - 1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class ResizeBilinearProgram {
      constructor(inputShape, newHeight, newWidth, alignCorners) {
          this.variableNames = ['A'];
          this.outputShape = [];
          const [batch, oldHeight, oldWidth, depth] = inputShape;
          this.outputShape = [batch, newHeight, newWidth, depth];
          const effectiveInSize = [
              (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
              (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
          ];
          const effectiveOutSize = [
              (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
              (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
          ];
          this.userCode = `
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${effectiveInSize[0] / effectiveOutSize[0]},
          ${effectiveInSize[1] / effectiveOutSize[1]});
      const vec2 inputShapeRC = vec2(${oldHeight}.0, ${oldWidth}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = vec2(yRC) * effectiveInputOverOutputRatioRC;

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `;
      }
  }

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
  class ResizeBilinearPackedProgram {
      constructor(inputShape, newHeight, newWidth, alignCorners) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = [];
          const [batch, oldHeight, oldWidth, depth] = inputShape;
          this.outputShape = [batch, newHeight, newWidth, depth];
          const effectiveInSize = [
              (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
              (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
          ];
          const effectiveOutSize = [
              (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
              (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
          ];
          this.userCode = `
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${effectiveInSize[0] / effectiveOutSize[0]},
          ${effectiveInSize[1] / effectiveOutSize[1]},
          ${effectiveInSize[1] / effectiveOutSize[1]});
      const vec3 inputShapeRC = vec3(${oldHeight}.0, ${oldWidth}.0,
                                     ${oldWidth}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = vec3(yRC) * effectiveInputOverOutputRatioRC;

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(sourceFracIndexRC);
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${depth - 1};
        bool hasNextRow = coords.z < ${newWidth - 1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google LLC All Rights Reserved.
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
  class ResizeNearestNeigborBackpropProgram {
      constructor(dy, x, alignCorners) {
          this.variableNames = ['dy'];
          this.outputShape = [];
          this.outputShape = x.shape;
          const [, xHeight, xWidth,] = x.shape;
          const [, yHeight, yWidth] = dy.shape;
          // In the backwards pass, we want to find the pixels that were generated for
          // each pixel in the input image the forward pass and add the corresponding
          // coefficient from dy to the gradient (with some interpolation).
          const effectiveXSize = [
              (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
              (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
          ];
          const effectiveYSize = [
              (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
              (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
          ];
          const heightScale = effectiveXSize[0] / effectiveYSize[0];
          const widthScale = effectiveXSize[1] / effectiveYSize[1];
          const invHeightScale = 1 / heightScale;
          const invWidthScale = 1 / widthScale;
          // This defines the size of the window of values around a particular
          // index in dy that we want to search for contributions to dx.
          const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
          const winWidth = (Math.ceil(invWidthScale) * 2) + 2;
          this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${heightScale});
        const float widthScale = float(${widthScale});

        const float invHeightScale = float(${invHeightScale});
        const float invWidthScale = float(${invWidthScale});

        const int winHeight = int(${winHeight});
        const int winWidth = int(${winWidth});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${yHeight}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${yWidth}) {
              continue;
            }

            float sourceFracRow =
              float(${effectiveXSize[0]}) *
                (float(dyR) / float(${effectiveYSize[0]}));

            float sourceFracCol =
                float(${effectiveXSize[1]}) *
                  (float(dyC) / float(${effectiveYSize[1]}));

            int sourceNearestRow = int(min(
                float(int(${xHeight}) - 1),
                ${alignCorners} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${xWidth}) - 1),
                ${alignCorners} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class ResizeNearestNeighborProgram {
      constructor(inputShape, newHeight, newWidth, alignCorners) {
          this.variableNames = ['A'];
          this.outputShape = [];
          const [batch, oldHeight, oldWidth, depth] = inputShape;
          this.outputShape = [batch, newHeight, newWidth, depth];
          const effectiveInSize = [
              (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
              (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
          ];
          const effectiveOutSize = [
              (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
              (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
          ];
          // When align corners is false, we rounds the value with floor.
          const roundBase = alignCorners ? '0.5' : '0.0';
          this.userCode = `
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${effectiveInSize[0] / effectiveOutSize[0]},
          ${effectiveInSize[1] / effectiveOutSize[1]});
      const vec2 inputShapeRC = vec2(${oldHeight}.0, ${oldWidth}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = vec2(yRC) * effectiveInputOverOutputRatioRC;

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${roundBase})));

        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class ReverseProgram {
      constructor(xShape, axis) {
          this.variableNames = ['x'];
          const rank = xShape.length;
          if (rank > 4) {
              throw new Error(`WebGL backend: Reverse of rank-${rank} tensor is not yet supported`);
          }
          this.outputShape = xShape;
          if (rank === 1) {
              this.userCode = `
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${xShape[0]} - coord - 1));
        }
      `;
              return;
          }
          const getInCoord = (i) => {
              if (axis.indexOf(i) !== -1 && xShape[i] !== 1) {
                  return `${xShape[i]} - coords[${i}] - 1`;
              }
              return `coords[${i}]`;
          };
          const inCoords = xShape.map((_, i) => getInCoord(i)).join(',');
          const type = getCoordsDataType(rank);
          this.userCode = `
      void main() {
        ${type} coords = getOutputCoords();
        setOutput(getX(${inCoords}));
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2019 Google LLC All Rights Reserved.
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
  class ReversePackedProgram {
      constructor(xShape, axis) {
          this.variableNames = ['x'];
          this.packedInputs = true;
          this.packedOutput = true;
          const rank = xShape.length;
          if (rank > 4) {
              throw new Error(`WebGL backend: Reverse of rank-${rank} tensor is not yet supported`);
          }
          this.outputShape = xShape;
          const channels = getChannels('rc', rank);
          const nextColumn = `${channels[rank - 1]} + 1 < ${this.outputShape[rank - 1]}`;
          const nextRow = `${channels[rank - 2]} + 1 < ${this.outputShape[rank - 2]}`;
          const type = getCoordsDataType(rank);
          if (rank === 1) {
              this.userCode = `
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${xShape[0]} - rc - 1),
            ${xShape[0]} - rc - 1);
          if(${nextColumn}){
              result.g = getChannel(getX(${xShape[0]} - (rc  + 1) - 1),
                ${xShape[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `;
          }
          else {
              this.userCode = `
        void main() {
          ${type} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${getR(channels.slice())};
          if(${nextColumn}){
            result.g = ${getG(channels.slice())};
          }
          if(${nextRow}) {
            result.b = ${getB(channels.slice())};
            if(${nextColumn}) {
              result.a = ${getA(channels.slice())};
            }
          }
          setOutput(result);
        }
    `;
          }
          function getR(channels) {
              return getChannel(channels);
          }
          function getG(channels) {
              channels[rank - 1] = '(' + channels[rank - 1] + ` + 1)`;
              return getChannel(channels);
          }
          function getB(channels) {
              channels[rank - 2] = '(' + channels[rank - 2] + ` + 1)`;
              return getChannel(channels);
          }
          function getA(channels) {
              channels[rank - 1] = '(' + channels[rank - 1] + ` + 1)`;
              channels[rank - 2] = '(' + channels[rank - 2] + ` + 1)`;
              return getChannel(channels);
          }
          function getChannel(channels) {
              const inCoordsArray = xShape.map((_, i) => getInCoord(i, channels));
              const inCoords = inCoordsArray.join(',');
              const innerDims = inCoordsArray.slice(-2).join(',');
              return `getChannel(getX(${inCoords}), vec2(${innerDims}))`;
          }
          function getInCoord(i, channels1) {
              if (axis.indexOf(i) !== -1 && xShape[i] !== 1) {
                  return `${xShape[i]} - ${channels1[i]} - 1`;
              }
              else {
                  return `${channels1[i]}`;
              }
          }
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class ScatterProgram {
      constructor(updateSize, sliceDim, indicesRank, updatesRank, strides, shape, summingDupeIndex = true) {
          this.variableNames = ['updates', 'indices', 'defaultValue'];
          this.outputShape = shape;
          const stridesType = getCoordsDataType(strides.length);
          const dtype = getCoordsDataType(shape.length);
          let indicesString = '';
          if (indicesRank === 1) {
              indicesString = 'i';
          }
          else if (indicesRank === 2) {
              indicesString = 'i, j';
          }
          const indicesSnippet = `getIndices(${indicesString})`;
          let updatesString = '';
          if (updatesRank === 1) {
              updatesString = 'i';
          }
          else if (updatesRank === 2) {
              updatesString = 'i, coords[1]';
          }
          const updatesSnippet = `getUpdates(${updatesString})`;
          const strideString = sliceDim > 1 ? 'strides[j]' : 'strides';
          this.userCode = `
        ${stridesType} strides = ${stridesType}(${strides});

        void main() {
          ${dtype} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${updateSize}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${sliceDim}; j++) {
              int index = round(${indicesSnippet});
              flattenedIndex += index * ${strideString};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${updatesSnippet};
              found = true;
            }
          }
          setOutput(mix(getDefaultValue(), sum, float(found)));
        }
      `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class SegmentOpProgram {
      constructor(segOpInfo, segOpType) {
          this.variableNames = ['x', 'segmentIds'];
          const windowSize = segOpInfo.windowSize;
          const batchSize = segOpInfo.batchSize;
          const inSize = segOpInfo.inSize;
          const numSegments = segOpInfo.numSegments;
          const outSize = numSegments * Math.ceil(inSize / windowSize);
          this.outputShape = [batchSize, outSize];
          const initializationValue = '0.0';
          const returnValue = `sumValue`;
          const windowSizeNearestVec4 = Math.floor(windowSize / 4) * 4;
          const windowSizeVec4Remainder = windowSize % 4;
          const updateSnippet = `
        sumValue += dot(values, segFilter);
    `;
          let checkValueOutOfBounds = '';
          if (inSize % windowSize > 0) {
              checkValueOutOfBounds = `
        if (inIdx < 0 || inIdx >= ${inSize}) {
          return initializationValue;
        }
      `;
          }
          let checkSegmentIdOutOfBounds = '';
          if (inSize % windowSize > 0) {
              checkSegmentIdOutOfBounds = `
        if (inIdx < 0 || inIdx >= ${inSize}) {
          return -1.0;
        }
      `;
          }
          this.userCode = `
      const float initializationValue = ${initializationValue};

      float getValue(int batch, int inIdx) {
        ${checkValueOutOfBounds}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${checkSegmentIdOutOfBounds}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${numSegments})) * float(${windowSize}));
        int currentSeg = int(mod(float(outIdx), float(${numSegments})));

        float sumValue = 0.0;

        for (int i = 0; i < ${windowSizeNearestVec4}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${updateSnippet}
        }

        int inIdx = inOffset + ${windowSizeNearestVec4};
        if (${windowSizeVec4Remainder === 1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${updateSnippet}
        } else if (${windowSizeVec4Remainder === 2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${updateSnippet}
        } else if (${windowSizeVec4Remainder === 3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${updateSnippet}
        }
        setOutput(${returnValue});
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class SelectProgram {
      constructor(cRank, shape, rank) {
          this.variableNames = ['c', 'a', 'b'];
          this.outputShape = shape;
          let cCoords;
          let abCoords;
          if (rank > 4) {
              throw Error(`Where for rank ${rank} is not yet supported`);
          }
          if (rank === 1) {
              abCoords = `resRC`;
              cCoords = `resRC`;
          }
          else {
              const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
              const cCoordVars = [];
              const abCoordVars = [];
              for (let i = 0; i < shape.length; i++) {
                  abCoordVars.push(`${currentCoords[i]}`);
                  if (i < cRank) {
                      cCoordVars.push(`${currentCoords[i]}`);
                  }
              }
              cCoords = cCoordVars.join();
              abCoords = abCoordVars.join();
          }
          const dtype = getCoordsDataType(rank);
          this.userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        float cVal = getC(${cCoords});
        if (cVal >= 1.0) {
          setOutput(getA(${abCoords}));
        } else {
          setOutput(getB(${abCoords}));
        }
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class SliceProgram {
      constructor(destSize) {
          this.variableNames = ['source'];
          this.outputShape = destSize;
          this.rank = destSize.length;
          const dtype = getCoordsDataType(this.rank);
          const uniformPart = `uniform int start[${this.rank}];`;
          const sourceCoords = getCoords$1(this.rank);
          let body;
          const coordSum = destSize.map((_, i) => {
              return `sourceLoc.${coords[i]} = start[${i}] + coords.${coords[i]};`;
          });
          body = `
        ${dtype} sourceLoc;
        ${dtype} coords = getOutputCoords();
        ${coordSum.join('\n')}
      `;
          this.userCode = `
      ${uniformPart}
      void main() {
        ${body}
        setOutput(getSource(${sourceCoords}));
      }
    `;
      }
      getCustomSetupFunc(start) {
          if (start.length !== this.rank) {
              throw Error(`The rank (${this.rank}) of the program must match the ` +
                  `length of start (${start.length})`);
          }
          return (gpgpu, webGLProgram) => {
              if (this.startLoc == null) {
                  this.startLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'start');
                  if (this.startLoc == null) {
                      // This means the compiler has optimized and realized it doesn't need
                      // the uniform.
                      return;
                  }
              }
              gpgpu.gl.uniform1iv(this.startLoc, start);
          };
      }
  }
  const coords = ['x', 'y', 'z', 'w', 'u', 'v'];
  function getCoords$1(rank) {
      if (rank === 1) {
          return 'sourceLoc';
      }
      else if (rank <= 6) {
          return coords.slice(0, rank).map(x => 'sourceLoc.' + x).join(',');
      }
      else {
          throw Error(`Slicing for rank ${rank} is not yet supported`);
      }
  }

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
  class SlicePackedProgram {
      constructor(destSize) {
          this.variableNames = ['source'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = destSize;
          this.rank = destSize.length;
          const dtype = getCoordsDataType(this.rank);
          const coords = getChannels('coords', this.rank);
          const sourceLoc = getChannels('sourceLoc', this.rank);
          const innerDims = this.rank === 1 ? 'sourceLoc' : `vec2(${sourceLoc.slice(-2).join()})`;
          const getChannel = `getChannel(getSource(${sourceLoc.join()}), ${innerDims})`;
          const upperRow = `
      result.x = ${getChannel};
      if (++${coords[this.rank - 1]} < ${destSize[this.rank - 1]}) {
        ++${sourceLoc[this.rank - 1]};
        result.y = ${getChannel};
        --${sourceLoc[this.rank - 1]};
      }
    `;
          const lowerRow = this.rank === 1 ? '' : `
      --${coords[this.rank - 1]};
      if (++${coords[this.rank - 2]} < ${destSize[this.rank - 2]}) {
        ++${sourceLoc[this.rank - 2]};
        result.z = ${getChannel};
        if (++${coords[this.rank - 1]} < ${destSize[this.rank - 1]}) {
          ++${sourceLoc[this.rank - 1]};
          result.w = ${getChannel};
        }
      }
    `;
          const sourceLocSetup = this.rank <= 4 ?
              `sourceLoc = coords +
            ${dtype}(${destSize.map((_, i) => `start[${i}]`).join()});` :
              destSize.map((_, i) => `${sourceLoc[i]} = ${coords[i]} + start[${i}];`)
                  .join('\n');
          this.userCode = `
      uniform int start[${this.rank}];
      void main() {
        ${dtype} coords = getOutputCoords();
        ${dtype} sourceLoc;
        ${sourceLocSetup}
        vec4 result = vec4(0.);
        ${upperRow}
        ${lowerRow}
        setOutput(result);
      }
    `;
      }
      getCustomSetupFunc(start) {
          if (start.length !== this.rank) {
              throw Error(`The rank (${this.rank}) of the program must match the ` +
                  `length of start (${start.length})`);
          }
          return (gpgpu, webGLProgram) => {
              if (this.startLoc == null) {
                  this.startLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'start');
                  if (this.startLoc == null) {
                      // This means the compiler has optimized and realized it doesn't need
                      // the uniform.
                      return;
                  }
              }
              gpgpu.gl.uniform1iv(this.startLoc, start);
          };
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class StridedSliceProgram {
      constructor(begin, strides, size) {
          this.variableNames = ['x'];
          this.outputShape = size;
          const rank = size.length;
          const inputDtype = getCoordsDataType(size.length);
          const dtype = getCoordsDataType(size.length);
          let newCoords = '';
          if (rank === 1) {
              newCoords = 'coords * strides + begin';
          }
          else {
              let outputAxis = 0;
              newCoords =
                  size.map((_, i) => {
                      outputAxis++;
                      return size.length === 1 ?
                          `coords * strides[${i}] + begin[${i}]` :
                          `coords[${outputAxis - 1}] * strides[${i}] + begin[${i}]`;
                  })
                      .join(',');
          }
          this.userCode = `
      ${inputDtype} begin = ${inputDtype}(${begin});
      ${inputDtype} strides = ${inputDtype}(${strides});

      void main() {
        ${dtype} coords = getOutputCoords();
        setOutput(getX(${newCoords}));
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class TextureManager {
      constructor(gpgpu) {
          this.gpgpu = gpgpu;
          this.numUsedTextures = 0;
          this.numFreeTextures = 0;
          this.freeTextures = {};
          this.logEnabled = false;
          this.usedTextures = {};
      }
      acquireTexture(shapeRC, usage, isPacked) {
          const physicalTexType = getPhysicalFromLogicalTextureType(usage, isPacked);
          const shapeKey = getKeyFromTextureShape(shapeRC, physicalTexType, isPacked);
          if (!(shapeKey in this.freeTextures)) {
              this.freeTextures[shapeKey] = [];
          }
          if (!(shapeKey in this.usedTextures)) {
              this.usedTextures[shapeKey] = [];
          }
          if (this.freeTextures[shapeKey].length > 0) {
              this.numFreeTextures--;
              this.numUsedTextures++;
              this.log();
              const newTexture = this.freeTextures[shapeKey].shift();
              this.usedTextures[shapeKey].push(newTexture);
              return newTexture;
          }
          this.numUsedTextures++;
          this.log();
          let newTexture;
          if (physicalTexType === PhysicalTextureType.PACKED_2X2_FLOAT32) {
              newTexture = this.gpgpu.createPackedMatrixTexture(shapeRC[0], shapeRC[1]);
          }
          else if (physicalTexType === PhysicalTextureType.PACKED_2X2_FLOAT16) {
              newTexture =
                  this.gpgpu.createFloat16PackedMatrixTexture(shapeRC[0], shapeRC[1]);
          }
          else if (physicalTexType === PhysicalTextureType.UNPACKED_FLOAT32) {
              newTexture =
                  this.gpgpu.createFloat32MatrixTexture(shapeRC[0], shapeRC[1]);
          }
          else if (physicalTexType === PhysicalTextureType.UNPACKED_FLOAT16) {
              newTexture =
                  this.gpgpu.createFloat16MatrixTexture(shapeRC[0], shapeRC[1]);
          }
          else if (physicalTexType === PhysicalTextureType.PACKED_4X1_UNSIGNED_BYTE) {
              newTexture =
                  this.gpgpu.createUnsignedBytesMatrixTexture(shapeRC[0], shapeRC[1]);
          }
          this.usedTextures[shapeKey].push(newTexture);
          return newTexture;
      }
      releaseTexture(texture, shape, logicalTexType, isPacked) {
          if (this.freeTextures == null) {
              // Already disposed.
              return;
          }
          const physicalTexType = getPhysicalFromLogicalTextureType(logicalTexType, isPacked);
          const shapeKey = getKeyFromTextureShape(shape, physicalTexType, isPacked);
          if (!(shapeKey in this.freeTextures)) {
              this.freeTextures[shapeKey] = [];
          }
          this.freeTextures[shapeKey].push(texture);
          this.numFreeTextures++;
          this.numUsedTextures--;
          const texList = this.usedTextures[shapeKey];
          const texIndex = texList.indexOf(texture);
          if (texIndex < 0) {
              throw new Error('Cannot release a texture that was never provided by this ' +
                  'texture manager');
          }
          texList.splice(texIndex, 1);
          this.log();
      }
      log() {
          if (!this.logEnabled) {
              return;
          }
          const total = this.numFreeTextures + this.numUsedTextures;
          console.log('Free/Used', `${this.numFreeTextures} / ${this.numUsedTextures}`, `(${total})`);
      }
      getNumUsedTextures() {
          return this.numUsedTextures;
      }
      getNumFreeTextures() {
          return this.numFreeTextures;
      }
      dispose() {
          if (this.freeTextures == null) {
              // Already disposed.
              return;
          }
          for (const texShape in this.freeTextures) {
              this.freeTextures[texShape].forEach(tex => {
                  this.gpgpu.deleteMatrixTexture(tex);
              });
          }
          for (const texShape in this.usedTextures) {
              this.usedTextures[texShape].forEach(tex => {
                  this.gpgpu.deleteMatrixTexture(tex);
              });
          }
          this.freeTextures = null;
          this.usedTextures = null;
          this.numUsedTextures = 0;
          this.numFreeTextures = 0;
      }
  }
  function getPhysicalTextureForRendering(isPacked) {
      if (tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED')) {
          if (isPacked) {
              return PhysicalTextureType.PACKED_2X2_FLOAT32;
          }
          return PhysicalTextureType.UNPACKED_FLOAT32;
      }
      if (isPacked) {
          return PhysicalTextureType.PACKED_2X2_FLOAT16;
      }
      return PhysicalTextureType.UNPACKED_FLOAT16;
  }
  function getPhysicalFromLogicalTextureType(logicalTexType, isPacked) {
      if (logicalTexType === TextureUsage.UPLOAD) {
          return PhysicalTextureType.PACKED_2X2_FLOAT32;
      }
      else if (logicalTexType === TextureUsage.RENDER || logicalTexType == null) {
          return getPhysicalTextureForRendering(isPacked);
      }
      else if (logicalTexType === TextureUsage.DOWNLOAD ||
          logicalTexType === TextureUsage.PIXELS) {
          return PhysicalTextureType.PACKED_4X1_UNSIGNED_BYTE;
      }
      throw new Error(`Unknown logical texture type ${logicalTexType}`);
  }
  function getKeyFromTextureShape(shapeRowsCol, physicalTexType, isPacked) {
      return `${shapeRowsCol[0]}_${shapeRowsCol[1]}_${physicalTexType}_${isPacked}`;
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class TileProgram {
      constructor(aShape, reps) {
          this.variableNames = ['A'];
          const outputShape = new Array(aShape.length);
          for (let i = 0; i < outputShape.length; i++) {
              outputShape[i] = aShape[i] * reps[i];
          }
          this.outputShape = outputShape;
          this.rank = outputShape.length;
          const dtype = getCoordsDataType(this.rank);
          const sourceCoords = getSourceCoords$2(aShape);
          this.userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        setOutput(getA(${sourceCoords}));
      }
    `;
      }
  }
  function getSourceCoords$2(aShape) {
      const rank = aShape.length;
      if (rank > 5) {
          throw Error(`Tile for rank ${rank} is not yet supported`);
      }
      if (rank === 1) {
          return `imod(resRC, ${aShape[0]})`;
      }
      const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w', 'resRC.u'];
      const sourceCoords = [];
      for (let i = 0; i < aShape.length; i++) {
          sourceCoords.push(`imod(${currentCoords[i]}, ${aShape[i]})`);
      }
      return sourceCoords.join();
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class UnaryOpProgram {
      constructor(aShape, opSnippet) {
          this.variableNames = ['A'];
          this.outputShape = aShape;
          this.userCode = `
      float unaryOperation(float x) {
        ${opSnippet}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `;
      }
  }
  const CHECK_NAN_SNIPPET$2 = `if (isnan(x)) return x;`;
  const LINEAR = `return x;`;
  const ABS = `return abs(x);`;
  const RELU = CHECK_NAN_SNIPPET$2 + `
  return (x < 0.0) ? 0.0 : x;
`;
  const RELU6 = CHECK_NAN_SNIPPET$2 + `
  return (x < 0.0) ? 0.0 : min(6.0, x);
`;
  const ELU = `return (x >= 0.0) ? x : (exp(x) - 1.0);`;
  const SELU = `
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${tf.backend_util.SELU_SCALEALPHA};
  float scale = ${tf.backend_util.SELU_SCALE};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`;
  function STEP(alpha = 0.0) {
      return CHECK_NAN_SNIPPET$2 + `
    return x > 0.0 ? 1.0 : float(${alpha});
  `;
  }
  const NEG = `return -x;`;
  const CEIL = `return ceil(x);`;
  const FLOOR = `return floor(x);`;
  const SIGN = `
  if (isnan(x)) { return 0.0; }
  return sign(x);
`;
  const IS_NAN = `return float(isnan(x));`;
  const IS_INF = `return float(isinf(x));`;
  const IS_FINITE = `return float(!isnan(x) && !isinf(x));`;
  const ROUND = `
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`;
  const EXP = `return exp(x);`;
  const EXPM1 = `return exp(x) - 1.0;`;
  const LOG = `if (x < 0.0) return NAN;
  return log(x);`;
  const LOG1P = `return log(1.0 + x);`;
  const SQRT = `return sqrt(x);`;
  const RSQRT = `return inversesqrt(x);`;
  const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * x));`;
  /**
   * mirrors the implementation of tf.nn.softplus: https://goo.gl/vkcvwX
   *
   * epsilon is the difference between 1.0 and the next representable
   * float. For a single precision 32 bit float this should be 2^-23, see:
   * https://math.byu.edu/~schow/work/IEEEFloatingPoint.htm
   *
   * too_large = (x > -threshold) is value above which exp(x) may overflow
   * but softplus(x) == x is within machine epsilon
   *
   * too_small = (x < threshold) is value below which exp(x) may underflow,
   * but softplus(x) == exp(x) is within machine epsilon.
   */
  const SOFTPLUS = `
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`;
  const SIN = CHECK_NAN_SNIPPET$2 + `
  return sin(x);
`;
  const COS = CHECK_NAN_SNIPPET$2 + `
  return cos(x);
`;
  const TAN = `return tan(x);`;
  const ASIN = CHECK_NAN_SNIPPET$2 + `
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`;
  const ACOS = CHECK_NAN_SNIPPET$2 + `
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`;
  const ATAN = CHECK_NAN_SNIPPET$2 + `
  return atan(x);
`;
  const SINH = `
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`;
  const COSH = `
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`;
  const TANH = `
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`;
  const ASINH = CHECK_NAN_SNIPPET$2 + `return log(x + sqrt(x * x + 1.0));`;
  const ACOSH = CHECK_NAN_SNIPPET$2 + `
  if (x < 1.0) return NAN;
  return log(x + sqrt(x * x - 1.0));`;
  const ATANH = CHECK_NAN_SNIPPET$2 + `
  if ((x < -1.0) || (x > 1.0)) return NAN;
  return (log(1.0 + x) - log(1.0 - x)) / 2.0;`;
  const ERF = `
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${tf.backend_util.ERF_P};
  float a1 = ${tf.backend_util.ERF_A1};
  float a2 = ${tf.backend_util.ERF_A2};
  float a3 = ${tf.backend_util.ERF_A3};
  float a4 = ${tf.backend_util.ERF_A4};
  float a5 = ${tf.backend_util.ERF_A5};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`;
  const SQUARE = `return x * x;`;
  const RECIPROCAL = `return 1.0 / x;`;
  const LOGICAL_NOT = `return float(!(x >= 1.0));`;
  const TO_INT = `return float(int(x));`;
  const CLONE = 'return x;';

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
  const LINEAR$1 = `return x;`;
  const LOG$1 = `
  vec4 result = log(x);
  vec4 isNaN = vec4(lessThan(x, vec4(0.0)));
  result.r = isNaN.r == 1.0 ? NAN : result.r;
  result.g = isNaN.g == 1.0 ? NAN : result.g;
  result.b = isNaN.b == 1.0 ? NAN : result.b;
  result.a = isNaN.a == 1.0 ? NAN : result.a;

  return result;
`;
  const RELU$1 = `
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;
  const RELU6$1 = `
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;
  const ELU$1 = `
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`;
  class UnaryOpPackedProgram {
      constructor(aShape, opSnippet) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = true;
          this.outputShape = aShape;
          this.userCode = `
      vec4 unaryOperation(vec4 x) {
        ${opSnippet}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `;
      }
  }

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
  class UnpackProgram {
      constructor(outputShape) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = false;
          this.outputShape = outputShape;
          const rank = outputShape.length;
          const channels = getChannels('rc', rank);
          const dtype = getCoordsDataType(rank);
          const sourceCoords = getSourceCoords(rank, channels);
          const innerDims = channels.slice(-2);
          const coords = rank <= 1 ? 'rc' : `vec2(${innerDims.join(',')})`;
          this.userCode = `
      void main() {
        ${dtype} rc = getOutputCoords();
        vec4 packedInput = getA(${sourceCoords});

        setOutput(getChannel(packedInput, ${coords}));
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  const { segment_util } = tf.backend_util;
  const nonMaxSuppressionV3 = tf.kernel_impls.nonMaxSuppressionV3;
  const split = tf.kernel_impls.split;
  const tile = tf.kernel_impls.tile;
  const topkImpl = tf.kernel_impls.topkImpl;
  const whereImpl = tf.kernel_impls.whereImpl;
  const EPSILON_FLOAT32 = 1e-7;
  const EPSILON_FLOAT16 = 1e-4;
  const binaryCaches = {};
  function getBinaryCache(webGLVersion) {
      if (webGLVersion in binaryCaches) {
          return binaryCaches[webGLVersion];
      }
      binaryCaches[webGLVersion] = {};
      return binaryCaches[webGLVersion];
  }
  function mapActivationToShaderProgram(activation, packed = false) {
      if (activation === 'linear') {
          if (packed) {
              return LINEAR$1;
          }
          return LINEAR;
      }
      else if (activation === 'relu') {
          if (packed) {
              return RELU$1;
          }
          return RELU;
      }
      else if (activation === 'elu') {
          if (packed) {
              return ELU$1;
          }
          return ELU;
      }
      else if (activation === 'relu6') {
          if (packed) {
              return RELU6$1;
          }
          return RELU6;
      }
      else if (activation === 'prelu') {
          if (packed) {
              return PRELU$1;
          }
          return PRELU;
      }
      throw new Error(`Activation ${activation} has not been implemented for the WebGL backend.`);
  }
  // Empirically determined constant used to determine size threshold for handing
  // off execution to the CPU.
  const CPU_HANDOFF_SIZE_THRESHOLD = 128;
  // Empirically determined constant used to decide the number of MB on GPU
  // before we warn about high memory use. The MB are this constant * screen area
  // * dpi / 1024 / 1024.
  const BEFORE_PAGING_CONSTANT = 600;
  function numMBBeforeWarning() {
      if (tf.env().global.screen == null) {
          return 1024; // 1 GB.
      }
      return (tf.env().global.screen.height * tf.env().global.screen.width *
          window.devicePixelRatio) *
          BEFORE_PAGING_CONSTANT / 1024 / 1024;
  }
  // Empirically determined minimal shared dimension in matmul before we forward
  // to a.mul(b).sum() in order to take advantage of GPU parallelism. See
  // https://github.com/tensorflow/tfjs-core/pull/1379 for benchmarks.
  const MATMUL_SHARED_DIM_THRESHOLD = 1000;
  class MathBackendWebGL extends tf.KernelBackend {
      constructor(gpgpu) {
          super();
          // Maps data ids that have a pending read operation, to list of subscribers.
          this.pendingRead = new WeakMap();
          // List of data ids that are scheduled for disposal, but are waiting on a
          // pending read operation.
          this.pendingDisposal = new WeakSet();
          // Used to count the number of 'shallow' sliced tensors that point to the
          // same data id.
          this.dataRefCount = new WeakMap();
          this.numBytesInGPU = 0;
          // Accumulated time spent (including blocking) in uploading data to webgl.
          this.uploadWaitMs = 0;
          // Accumulated time spent (including blocking in downloading data from webgl.
          this.downloadWaitMs = 0;
          this.warnedAboutMemory = false;
          this.pendingDeletes = 0;
          this.disposed = false;
          if (!tf.env().getBool('HAS_WEBGL')) {
              throw new Error('WebGL is not supported on this device');
          }
          if (gpgpu == null) {
              const gl = getWebGLContext(tf.env().getNumber('WEBGL_VERSION'));
              this.binaryCache = getBinaryCache(tf.env().getNumber('WEBGL_VERSION'));
              this.gpgpu = new GPGPUContext(gl);
              this.canvas = gl.canvas;
              this.gpgpuCreatedLocally = true;
          }
          else {
              this.gpgpu = gpgpu;
              this.binaryCache = {};
              this.gpgpuCreatedLocally = false;
              this.canvas = gpgpu.gl.canvas;
          }
          this.textureManager = new TextureManager(this.gpgpu);
          this.numMBBeforeWarning = numMBBeforeWarning();
          this.texData = new tf.DataStorage(this, tf.engine());
      }
      numDataIds() {
          return this.texData.numDataIds() +
              (this.cpuBackend ? this.cpuBackend.numDataIds() : 0) -
              this.pendingDeletes;
      }
      write(values, shape, dtype) {
          if (tf.env().getBool('DEBUG')) {
              this.checkNumericalProblems(values);
          }
          if (dtype === 'complex64' && values != null) {
              throw new Error(`Cannot write to a complex64 dtype. ` +
                  `Please use tf.complex(real, imag).`);
          }
          const dataId = {};
          this.texData.set(dataId, { shape, dtype, values, usage: TextureUsage.UPLOAD });
          return dataId;
      }
      move(dataId, values, shape, dtype) {
          if (tf.env().getBool('DEBUG')) {
              this.checkNumericalProblems(values);
          }
          if (dtype === 'complex64') {
              throw new Error(`Cannot write to a complex64 dtype. ` +
                  `Please use tf.complex(real, imag).`);
          }
          this.texData.set(dataId, { shape, dtype, values, usage: TextureUsage.UPLOAD });
      }
      readSync(dataId) {
          const texData = this.texData.get(dataId);
          const { values, dtype, complexTensors, slice, shape, isPacked } = texData;
          if (slice != null) {
              let program;
              if (isPacked) {
                  program = new UnaryOpPackedProgram(shape, CLONE);
              }
              else {
                  program = new UnaryOpProgram(shape, CLONE);
              }
              const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
              const data = this.readSync(res.dataId);
              this.disposeData(res.dataId);
              return data;
          }
          if (values != null) {
              return this.convertAndCacheOnCPU(dataId);
          }
          if (dtype === 'string') {
              return values;
          }
          const shouldTimeProgram = this.activeTimers != null;
          let start;
          if (shouldTimeProgram) {
              start = tf.util.now();
          }
          let result;
          if (dtype === 'complex64') {
              const realValues = complexTensors.real.dataSync();
              const imagValues = complexTensors.imag.dataSync();
              result = tf.backend_util.mergeRealAndImagArrays(realValues, imagValues);
          }
          else {
              result = this.getValuesFromTexture(dataId);
          }
          if (shouldTimeProgram) {
              this.downloadWaitMs += tf.util.now() - start;
          }
          return this.convertAndCacheOnCPU(dataId, result);
      }
      async read(dataId) {
          if (this.pendingRead.has(dataId)) {
              const subscribers = this.pendingRead.get(dataId);
              return new Promise(resolve => subscribers.push(resolve));
          }
          const texData = this.texData.get(dataId);
          const { values, shape, slice, dtype, complexTensors, isPacked } = texData;
          if (slice != null) {
              let program;
              if (isPacked) {
                  program = new UnaryOpPackedProgram(shape, CLONE);
              }
              else {
                  program = new UnaryOpProgram(shape, CLONE);
              }
              const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
              const data = this.read(res.dataId);
              this.disposeData(res.dataId);
              return data;
          }
          if (values != null) {
              return this.convertAndCacheOnCPU(dataId);
          }
          if (!tf.env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
              tf.env().getNumber('WEBGL_VERSION') === 2) {
              throw new Error(`tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and ` +
                  `WEBGL_VERSION=2 not yet supported.`);
          }
          let buffer = null;
          let tmpDownloadTarget;
          if (dtype !== 'complex64' && tf.env().get('WEBGL_BUFFER_SUPPORTED')) {
              // Possibly copy the texture into a buffer before inserting a fence.
              tmpDownloadTarget = this.decode(dataId);
              const tmpData = this.texData.get(tmpDownloadTarget.dataId);
              buffer = this.gpgpu.createBufferFromTexture(tmpData.texture, ...getDenseTexShape(shape));
          }
          this.pendingRead.set(dataId, []);
          if (dtype !== 'complex64') {
              // Create a fence and wait for it to resolve.
              await this.gpgpu.createAndWaitForFence();
          }
          // Download the values from the GPU.
          let vals;
          if (dtype === 'complex64') {
              const ps = await Promise.all([complexTensors.real.data(), complexTensors.imag.data()]);
              const realValues = ps[0];
              const imagValues = ps[1];
              vals = tf.backend_util.mergeRealAndImagArrays(realValues, imagValues);
          }
          else if (buffer == null) {
              vals = this.getValuesFromTexture(dataId);
          }
          else {
              const size = tf.util.sizeFromShape(shape);
              vals = this.gpgpu.downloadFloat32MatrixFromBuffer(buffer, size);
          }
          if (tmpDownloadTarget != null) {
              this.disposeData(tmpDownloadTarget.dataId);
          }
          const dTypeVals = this.convertAndCacheOnCPU(dataId, vals);
          const subscribers = this.pendingRead.get(dataId);
          this.pendingRead.delete(dataId);
          // Notify all pending reads.
          subscribers.forEach(resolve => resolve(dTypeVals));
          if (this.pendingDisposal.has(dataId)) {
              this.pendingDisposal.delete(dataId);
              this.disposeData(dataId);
              this.pendingDeletes--;
          }
          return dTypeVals;
      }
      checkNumericalProblems(values) {
          if (values == null) {
              return;
          }
          for (let i = 0; i < values.length; i++) {
              const num = values[i];
              if (!canBeRepresented(num)) {
                  if (tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
                      throw Error(`The value ${num} cannot be represented with your ` +
                          `current settings. Consider enabling float32 rendering: ` +
                          `'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`);
                  }
                  throw Error(`The value ${num} cannot be represented on this device.`);
              }
          }
      }
      getValuesFromTexture(dataId) {
          const { shape, dtype, isPacked } = this.texData.get(dataId);
          const size = tf.util.sizeFromShape(shape);
          if (tf.env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
              const tmpTarget = this.decode(dataId);
              const tmpData = this.texData.get(tmpTarget.dataId);
              const vals = this.gpgpu
                  .downloadMatrixFromPackedTexture(tmpData.texture, ...getDenseTexShape(shape))
                  .subarray(0, size);
              this.disposeData(tmpTarget.dataId);
              return vals;
          }
          const shouldUsePackedProgram = tf.env().getBool('WEBGL_PACK') && isPacked === true;
          const outputShape = shouldUsePackedProgram ? getShapeAs3D(shape) : shape;
          const program = shouldUsePackedProgram ?
              new EncodeFloatPackedProgram(outputShape) :
              new EncodeFloatProgram(outputShape);
          const output = this.runWebGLProgram(program, [{ shape: outputShape, dtype, dataId }], 'float32');
          const tmpData = this.texData.get(output.dataId);
          const vals = this.gpgpu
              .downloadByteEncodedFloatMatrixFromOutputTexture(tmpData.texture, tmpData.texShape[0], tmpData.texShape[1])
              .subarray(0, size);
          this.disposeData(output.dataId);
          return vals;
      }
      async time(f) {
          const oldActiveTimers = this.activeTimers;
          const newActiveTimers = [];
          let outerMostTime = false;
          if (this.programTimersStack == null) {
              this.programTimersStack = newActiveTimers;
              outerMostTime = true;
          }
          else {
              this.activeTimers.push(newActiveTimers);
          }
          this.activeTimers = newActiveTimers;
          f();
          // needing to split these up because util.flatten only accepts certain types
          const flattenedActiveTimerQueries = tf.util.flatten(this.activeTimers.map((d) => d.query))
              .filter(d => d != null);
          const flattenedActiveTimerNames = tf.util.flatten(this.activeTimers.map((d) => d.name))
              .filter(d => d != null);
          this.activeTimers = oldActiveTimers;
          if (outerMostTime) {
              this.programTimersStack = null;
          }
          const res = {
              uploadWaitMs: this.uploadWaitMs,
              downloadWaitMs: this.downloadWaitMs,
              kernelMs: null,
              wallMs: null // will be filled by the engine
          };
          if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
              const kernelMs = await Promise.all(flattenedActiveTimerQueries);
              res['kernelMs'] = tf.util.sum(kernelMs);
              res['getExtraProfileInfo'] = () => kernelMs.map((d, i) => ({ name: flattenedActiveTimerNames[i], ms: d }))
                  .map(d => `${d.name}: ${d.ms}`)
                  .join(', ');
          }
          else {
              res['kernelMs'] = {
                  error: 'WebGL query timers are not supported in this environment.'
              };
          }
          this.uploadWaitMs = 0;
          this.downloadWaitMs = 0;
          return res;
      }
      memory() {
          return { unreliable: false, numBytesInGPU: this.numBytesInGPU };
      }
      startTimer() {
          if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
              return this.gpgpu.beginQuery();
          }
          return { startMs: tf.util.now(), endMs: null };
      }
      endTimer(query) {
          if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
              this.gpgpu.endQuery();
              return query;
          }
          query.endMs = tf.util.now();
          return query;
      }
      async getQueryTime(query) {
          if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
              return this.gpgpu.waitForQueryAndGetTime(query);
          }
          const timerQuery = query;
          return timerQuery.endMs - timerQuery.startMs;
      }
      disposeData(dataId) {
          if (this.pendingDisposal.has(dataId)) {
              return;
          }
          if (this.pendingRead.has(dataId)) {
              this.pendingDisposal.add(dataId);
              this.pendingDeletes++;
              return;
          }
          // No-op if already disposed.
          if (!this.texData.has(dataId)) {
              return;
          }
          this.releaseGPUData(dataId);
          const { complexTensors } = this.texData.get(dataId);
          if (complexTensors != null) {
              complexTensors.real.dispose();
              complexTensors.imag.dispose();
          }
          this.texData.delete(dataId);
      }
      releaseGPUData(dataId) {
          const { texture, dtype, texShape, usage, isPacked, slice } = this.texData.get(dataId);
          const key = slice && slice.origDataId || dataId;
          const refCount = this.dataRefCount.get(key);
          if (refCount > 1) {
              this.dataRefCount.set(key, refCount - 1);
          }
          else {
              this.dataRefCount.delete(key);
              if (texture != null) {
                  this.numBytesInGPU -= this.computeBytes(texShape, dtype);
                  this.textureManager.releaseTexture(texture, texShape, usage, isPacked);
              }
          }
          const texData = this.texData.get(dataId);
          texData.texture = null;
          texData.texShape = null;
          texData.isPacked = false;
          texData.slice = null;
      }
      getTexture(dataId) {
          this.uploadToGPU(dataId);
          return this.texData.get(dataId).texture;
      }
      /**
       * Returns internal information for the specific data bucket. Used in unit
       * tests.
       */
      getDataInfo(dataId) {
          return this.texData.get(dataId);
      }
      getCPUBackend() {
          if (!tf.env().getBool('WEBGL_CPU_FORWARD')) {
              return null;
          }
          if (this.cpuBackend == null) {
              this.cpuBackend = tf.engine().findBackend('cpu');
          }
          return this.cpuBackend;
      }
      /*
      Tests whether all the inputs to an op are small and on the CPU. This heuristic
      determines when it would be faster to execute a kernel on the CPU. WebGL
      kernels opt into running this check and forwarding when appropriate.
      TODO(https://github.com/tensorflow/tfjs/issues/872): Develop a more
      sustainable strategy for optimizing backend execution of ops.
       */
      shouldExecuteOnCPU(inputs, sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD) {
          return this.getCPUBackend() != null &&
              inputs.every(input => this.texData.get(input.dataId).texture == null &&
                  tf.util.sizeFromShape(input.shape) < sizeThreshold);
      }
      getGPGPUContext() {
          return this.gpgpu;
      }
      complex(real, imag) {
          const result = this.makeOutput(real.shape, 'complex64');
          const resultData = this.texData.get(result.dataId);
          // The backend owns the reference to the underlying real and imaginary
          // clones. These will explicitly get disposed when the complex tensor is
          // disposed.
          resultData.complexTensors = {
              real: tf.engine().keep(real.clone()),
              imag: tf.engine().keep(imag.clone())
          };
          return result;
      }
      real(input) {
          const resultData = this.texData.get(input.dataId);
          return resultData.complexTensors.real.clone();
      }
      imag(input) {
          const resultData = this.texData.get(input.dataId);
          return resultData.complexTensors.imag.clone();
      }
      slice(x, begin, size) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.slice(x, begin, size);
          }
          // Short-circuit computation if the slice is zero-sized.
          if (tf.util.sizeFromShape(size) === 0) {
              return tf.tensor([], size, x.dtype);
          }
          const { isPacked } = this.texData.get(x.dataId);
          const isContinous = tf.slice_util.isSliceContinous(x.shape, begin, size);
          if (isPacked || !isContinous) {
              const program = tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
                  new SlicePackedProgram(size) :
                  new SliceProgram(size);
              const customSetup = program.getCustomSetupFunc(begin);
              return this.compileAndRun(program, [x], null, customSetup);
          }
          this.uploadToGPU(x.dataId);
          return this.shallowSlice(x, begin, size);
      }
      shallowSlice(x, begin, size) {
          const xTexData = this.texData.get(x.dataId);
          const t = this.makeOutput(size, x.dtype);
          const newTexData = this.texData.get(t.dataId);
          // Copy texture data from the original tensor.
          Object.assign(newTexData, xTexData);
          newTexData.shape = size;
          newTexData.dtype = x.dtype;
          let flatOffset = tf.slice_util.computeFlatOffset(begin, x.strides);
          if (xTexData.slice) {
              // We are slicing an already sliced tensor, so we have to accumulate
              // the offset.
              flatOffset += xTexData.slice.flatOffset;
          }
          newTexData.slice = {
              flatOffset,
              // Point to the original dataId, which is used to do ref counting.
              origDataId: xTexData.slice && xTexData.slice.origDataId || x.dataId
          };
          // Increase the ref count for that data bucket.
          const refCount = this.dataRefCount.get(newTexData.slice.origDataId) || 1;
          this.dataRefCount.set(newTexData.slice.origDataId, refCount + 1);
          return t;
      }
      stridedSlice(x, begin, end, strides) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.stridedSlice(x, begin, end, strides);
          }
          const outShape = tf.slice_util.computeOutShape(begin, end, strides);
          if (outShape.some(axis => axis === 0)) {
              return tf.tensor([], outShape);
          }
          const program = new StridedSliceProgram(begin, strides, outShape);
          return this.compileAndRun(program, [x]);
      }
      reverse(x, axis) {
          const program = tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
              new ReversePackedProgram(x.shape, axis) :
              new ReverseProgram(x.shape, axis);
          return this.compileAndRun(program, [x]);
      }
      concat(tensors, axis) {
          if (tensors[0].dtype === 'complex64') {
              const reals = tensors.map((t) => tf.real(t));
              const imags = tensors.map((t) => tf.imag(t));
              return tf.complex(this.concat(reals, axis), this.concat(imags, axis));
          }
          if (this.shouldExecuteOnCPU(tensors)) {
              return this.cpuBackend.concat(tensors, axis);
          }
          if (tensors.length === 1) {
              return tensors[0];
          }
          if (tensors.length > tf.env().getNumber('WEBGL_MAX_TEXTURES_IN_SHADER')) {
              const midIndex = Math.floor(tensors.length / 2);
              const leftSide = this.concat(tensors.slice(0, midIndex), axis);
              const rightSide = this.concat(tensors.slice(midIndex), axis);
              return this.concat([leftSide, rightSide], axis);
          }
          if (tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') && tensors[0].rank > 1) {
              const program = new ConcatPackedProgram(tensors.map(t => t.shape), axis);
              return this.compileAndRun(program, tensors);
          }
          // Any concat of n-dimensional tensors across any axis can be reduced to
          // a concatenation of two-dimensional tensors across the axis 1 by first
          // partitioning the axes of the original tensors into those less than the
          // axis to be concatenated and the rest. Then reshape the tensors
          // into a two-dimensional tensor by collapsing these two sets of axes and
          // concatenate the resulting matrices across the axis 1, finally reshaping
          // the result to have the proper shape.
          const outShape = tf.backend_util.computeOutShape(tensors.map(t => t.shape), axis);
          const tensors2D = tensors.map(t => t.as2D(-1, tf.util.sizeFromShape(t.shape.slice(axis))));
          const program = new ConcatProgram(tensors2D.map(t => t.shape));
          const res = this.compileAndRun(program, tensors2D);
          return res.reshape(outShape);
      }
      neg(x) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.neg(x);
          }
          if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
              return this.packedUnaryOp(x, NEG, x.dtype);
          }
          const program = new UnaryOpProgram(x.shape, NEG);
          return this.compileAndRun(program, [x]);
      }
      batchMatMul(a, b, transposeA, transposeB) {
          const outerShapeA = transposeA ? a.shape[2] : a.shape[1];
          const outerShapeB = transposeB ? b.shape[1] : b.shape[2];
          const sharedDim = transposeA ? a.shape[1] : a.shape[2];
          const [batch, ,] = a.shape;
          // Since the matrices are vectors, it is faster to call mul().sum()
          // because sum() is O(sqrt(N)) due to divide-and-conquer.
          if ((outerShapeA === 1 || outerShapeB === 1) &&
              sharedDim > MATMUL_SHARED_DIM_THRESHOLD) {
              if (transposeA) {
                  a = tf.transpose(a, [0, 2, 1]);
              }
              if (transposeB) {
                  b = tf.transpose(b, [0, 2, 1]);
              }
              const a3D = outerShapeB === 1 ? a : a.as3D(batch, sharedDim, 1);
              const axis = outerShapeB === 1 ? 2 : 1;
              const b3D = outerShapeB === 1 ? b.as3D(batch, 1, sharedDim) : b;
              return this.multiply(a3D, b3D).sum(axis, true /* keepDims */);
          }
          const dtype = tf.upcastType(a.dtype, b.dtype);
          const program = new MatMulPackedProgram(a.shape, [batch, outerShapeA, outerShapeB], transposeA, transposeB);
          return this.compileAndRun(program, [a, b], dtype);
      }
      fusedBatchMatMul({ a, b, transposeA, transposeB, bias, activation, preluActivationWeights }) {
          const outerShapeA = transposeA ? a.shape[2] : a.shape[1];
          const outerShapeB = transposeB ? b.shape[1] : b.shape[2];
          const [batch, ,] = a.shape;
          const dtype = tf.upcastType(a.dtype, b.dtype);
          const hasBias = bias != null;
          const hasPreluActivationWeights = preluActivationWeights != null;
          const fusedActivation = activation ? mapActivationToShaderProgram(activation, true) : null;
          const program = new MatMulPackedProgram(a.shape, [batch, outerShapeA, outerShapeB], transposeA, transposeB, hasBias, fusedActivation, hasPreluActivationWeights);
          const inputs = [a, b];
          if (bias) {
              inputs.push(bias);
          }
          if (preluActivationWeights) {
              inputs.push(preluActivationWeights);
          }
          return this.compileAndRun(program, inputs, dtype);
      }
      multiply(a, b) {
          if (a.dtype === 'complex64') {
              const aData = this.texData.get(a.dataId);
              const bData = this.texData.get(b.dataId);
              const realProgram = new BinaryOpComplexProgram(COMPLEX_MULTIPLY.REAL, a.shape, b.shape);
              const imagProgram = new BinaryOpComplexProgram(COMPLEX_MULTIPLY.IMAG, a.shape, b.shape);
              const inputs = [
                  this.makeComplexComponentTensorInfo(a, aData.complexTensors.real),
                  this.makeComplexComponentTensorInfo(a, aData.complexTensors.imag),
                  this.makeComplexComponentTensorInfo(b, bData.complexTensors.real),
                  this.makeComplexComponentTensorInfo(b, bData.complexTensors.imag)
              ];
              const real = this.compileAndRun(realProgram, inputs);
              const imag = this.compileAndRun(imagProgram, inputs);
              const complex = this.complex(real, imag);
              real.dispose();
              imag.dispose();
              return complex;
          }
          if (this.shouldExecuteOnCPU([a, b])) {
              return this.cpuBackend.multiply(a, b);
          }
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, MUL, a.dtype);
          }
          const program = new BinaryOpProgram(MUL, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], a.dtype);
      }
      batchNormalization(x, mean, variance, varianceEpsilon, scale, offset) {
          const inputs = [x, mean, variance];
          let offsetShape = null;
          if (offset != null) {
              offsetShape = offset.shape;
              inputs.push(offset);
          }
          let scaleShape = null;
          if (scale != null) {
              scaleShape = scale.shape;
              inputs.push(scale);
          }
          if (tf.env().getBool('WEBGL_PACK_NORMALIZATION')) {
              const batchNormPackedProgram = new BatchNormPackedProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape, varianceEpsilon);
              return this.compileAndRun(batchNormPackedProgram, inputs);
          }
          const batchNormProgram = new BatchNormProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape, varianceEpsilon);
          return this.compileAndRun(batchNormProgram, inputs);
      }
      localResponseNormalization4D(x, radius, bias, alpha, beta) {
          const program = tf.env().getBool('WEBGL_PACK_NORMALIZATION') ?
              new LRNPackedProgram(x.shape, radius, bias, alpha, beta) :
              new LRNProgram(x.shape, radius, bias, alpha, beta);
          return this.compileAndRun(program, [x]);
      }
      LRNGrad(dy, inputImage, outputImage, depthRadius, bias, alpha, beta) {
          const program = new LRNGradProgram(inputImage.shape, depthRadius, bias, alpha, beta);
          return this.compileAndRun(program, [inputImage, outputImage, dy]);
      }
      tile(x, reps) {
          if (x.dtype === 'string') {
              const data = this.readSync(x.dataId);
              const decodedData = data.map(d => tf.util.decodeString(d));
              const buf = tf.buffer(x.shape, x.dtype, decodedData);
              return tile(buf, reps);
          }
          const program = new TileProgram(x.shape, reps);
          return this.compileAndRun(program, [x]);
      }
      pad(x, paddings, constantValue) {
          const program = tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
              new PadPackedProgram(x.shape, paddings, constantValue) :
              new PadProgram(x.shape, paddings, constantValue);
          return this.compileAndRun(program, [x]);
      }
      gather(x, indices, axis) {
          if (this.shouldExecuteOnCPU([x, indices])) {
              return this.cpuBackend.gather(x, indices, axis);
          }
          const program = new GatherProgram(x.shape, indices.size, axis);
          return this.compileAndRun(program, [x, indices]);
      }
      batchToSpaceND(x, blockShape, crops) {
          tf.util.assert(x.rank <= 4, () => 'batchToSpaceND for rank > 4 with a WebGL backend not ' +
              'implemented yet');
          const prod = blockShape.reduce((a, b) => a * b);
          const reshaped = tf.backend_util.getReshaped(x.shape, blockShape, prod);
          const permuted = tf.backend_util.getPermuted(reshaped.length, blockShape.length);
          const reshapedPermuted = tf.backend_util.getReshapedPermuted(x.shape, blockShape, prod);
          const sliceBeginCoords = tf.backend_util.getSliceBeginCoords(crops, blockShape.length);
          const sliceSize = tf.backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);
          return tf.transpose(x.reshape(reshaped), permuted)
              .reshape(reshapedPermuted)
              .slice(sliceBeginCoords, sliceSize);
      }
      spaceToBatchND(x, blockShape, paddings) {
          tf.util.assert(x.rank <= 4, () => 'spaceToBatchND for rank > 4 with a WebGL backend not ' +
              'implemented yet');
          const prod = blockShape.reduce((a, b) => a * b);
          const completePaddings = [[0, 0]];
          completePaddings.push(...paddings);
          for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
              completePaddings.push([0, 0]);
          }
          const paddedX = x.pad(completePaddings);
          const reshapedPaddedShape = tf.backend_util.getReshaped(paddedX.shape, blockShape, prod, false);
          const permutedReshapedPaddedPermutation = tf.backend_util.getPermuted(reshapedPaddedShape.length, blockShape.length, false);
          const flattenShape = tf.backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);
          return tf.transpose(paddedX.reshape(reshapedPaddedShape), permutedReshapedPaddedPermutation)
              .reshape(flattenShape);
      }
      reduce(x, reduceType, dtype) {
          const batchSize = x.shape[0];
          const inSize = x.shape[1];
          const windowSize = tf.backend_util.computeOptimalWindowSize(inSize);
          const reduceInfo = { windowSize, inSize, batchSize };
          const program = new ReduceProgram(reduceInfo, reduceType);
          const output = this.compileAndRun(program, [x], dtype);
          // No need to run another GPGPU program.
          if (output.shape[1] === 1) {
              return output;
          }
          return this.reduce(output, reduceType, dtype);
      }
      argReduce(x, reduceType, bestIndicesA = null) {
          let batchSize = x.shape[0];
          let inSize = x.shape[1];
          if (bestIndicesA != null) {
              batchSize = bestIndicesA.shape[0];
              inSize = bestIndicesA.shape[1];
          }
          const windowSize = tf.backend_util.computeOptimalWindowSize(inSize);
          const reduceInfo = { windowSize, inSize, batchSize };
          const program = new ArgMinMaxProgram(reduceInfo, reduceType, bestIndicesA == null);
          const inputs = [x];
          if (bestIndicesA != null) {
              inputs.push(bestIndicesA);
          }
          const output = this.compileAndRun(program, inputs, 'int32');
          // No need to run another GPGPU program.
          if (output.shape[1] === 1) {
              return output;
          }
          return this.argReduce(x, reduceType, output);
      }
      argReducePacked(x, reduceType, bestIndicesA = null) {
          const inShape = bestIndicesA != null ? bestIndicesA.shape : x.shape;
          const inSize = inShape[inShape.length - 1];
          const windowSize = tf.backend_util.computeOptimalWindowSize(inSize);
          const program = new ArgMinMaxPackedProgram(inShape, windowSize, reduceType, bestIndicesA == null);
          const inputs = bestIndicesA == null ? [x] : [x, bestIndicesA];
          const output = this.compileAndRun(program, inputs, 'int32');
          if (output.rank === x.rank) {
              return this.argReducePacked(x, reduceType, output);
          }
          return output;
      }
      sum(x, axes) {
          tf.backend_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const inSize = tf.util.sizeFromShape(reduceShape);
          const a2D = x.as2D(-1, inSize);
          const outputDType = tf.sumOutType(x.dtype);
          return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
      }
      prod(x, axes) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.prod(x, axes);
          }
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const inSize = tf.util.sizeFromShape(reduceShape);
          const a2D = x.as2D(-1, inSize);
          const outputDType = tf.sumOutType(x.dtype);
          return this.reduce(a2D, 'prod', outputDType).reshape(outShape);
      }
      unsortedSegmentSum(x, segmentIds, numSegments) {
          let axis = 0;
          const permutation = tf.backend_util.getAxesPermutation([axis], x.rank);
          let permutedX = x;
          if (permutation != null) {
              permutedX = tf.transpose(x, permutation);
              axis = tf.backend_util.getInnerMostAxes(1, x.rank)[0];
          }
          const outShape = segment_util.computeOutShape(permutedX.shape, axis, numSegments);
          const inSize = tf.util.sizeFromShape([permutedX.shape[axis]]);
          const a2D = permutedX.as2D(-1, inSize);
          const outputDType = tf.sumOutType(x.dtype);
          let result = this.segOpCompute(a2D, 'unsortedSegmentSum', segmentIds, outputDType, numSegments)
              .reshape(outShape);
          if (permutation != null) {
              result =
                  tf.transpose(result, tf.backend_util.getUndoAxesPermutation(permutation));
          }
          return result;
      }
      segOpCompute(x, segOpType, segmentIds, dtype, numSegments) {
          const batchSize = x.shape[0];
          const inSize = x.shape[1];
          const windowSize = segment_util.segOpComputeOptimalWindowSize(inSize, numSegments);
          const segOpInfo = { windowSize, inSize, batchSize, numSegments };
          const program = new SegmentOpProgram(segOpInfo, segOpType);
          const output = this.compileAndRun(program, [x, segmentIds], dtype);
          // No need to run another GPGPU program.
          if (output.shape[1] === numSegments) {
              return output;
          }
          segmentIds = tf.range(0, numSegments).tile([inSize / windowSize]);
          return this.segOpCompute(output, segOpType, segmentIds, dtype, numSegments);
      }
      argMinMaxReduce(x, axis, reduceType) {
          const axes = [axis];
          tf.backend_util.assertAxesAreInnerMostDims('arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes, x.rank);
          if (!tf.env().getBool('WEBGL_PACK_REDUCE') || x.rank <= 2) {
              const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
              const inSize = tf.util.sizeFromShape(reduceShape);
              const a2D = x.as2D(-1, inSize);
              return this.argReduce(a2D, reduceType).reshape(outShape);
          }
          return this.argReducePacked(x, reduceType);
      }
      argMin(x, axis) {
          return this.argMinMaxReduce(x, axis, 'min');
      }
      argMax(x, axis) {
          return this.argMinMaxReduce(x, axis, 'max');
      }
      cumsum(x, axis, exclusive, reverse) {
          if (axis !== x.rank - 1) {
              throw new Error(`WebGL cumsum shader expects an inner-most axis=${x.rank - 1} ` +
                  `but got axis=${axis}`);
          }
          const program = new CumSumProgram(x.shape, exclusive, reverse);
          return this.compileAndRun(program, [x]);
      }
      equal(a, b) {
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, EQUAL$1, 'bool');
          }
          const program = new BinaryOpProgram(EQUAL, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], 'bool');
      }
      notEqual(a, b) {
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, NOT_EQUAL$1, 'bool');
          }
          const program = new BinaryOpProgram(NOT_EQUAL, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], 'bool');
      }
      less(a, b) {
          if (this.shouldExecuteOnCPU([a, b])) {
              return this.cpuBackend.less(a, b);
          }
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, LESS$1, 'bool');
          }
          const program = new BinaryOpProgram(LESS, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], 'bool');
      }
      lessEqual(a, b) {
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, LESS_EQUAL$1, 'bool');
          }
          const program = new BinaryOpProgram(LESS_EQUAL, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], 'bool');
      }
      greater(a, b) {
          if (this.shouldExecuteOnCPU([a, b])) {
              return this.cpuBackend.greater(a, b);
          }
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, GREATER$1, 'bool');
          }
          const program = new BinaryOpProgram(GREATER, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], 'bool');
      }
      greaterEqual(a, b) {
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, GREATER_EQUAL$1, 'bool');
          }
          const program = new BinaryOpProgram(GREATER_EQUAL, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], 'bool');
      }
      logicalNot(x) {
          const program = new UnaryOpProgram(x.shape, LOGICAL_NOT);
          return this.compileAndRun(program, [x]);
      }
      logicalAnd(a, b) {
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, LOGICAL_AND$1, 'bool');
          }
          const program = new BinaryOpProgram(LOGICAL_AND, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], 'bool');
      }
      logicalOr(a, b) {
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, LOGICAL_OR$1, 'bool');
          }
          const program = new BinaryOpProgram(LOGICAL_OR, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], 'bool');
      }
      select(condition, a, b) {
          const program = new SelectProgram(condition.rank, a.shape, a.rank);
          return this.compileAndRun(program, [condition, a, b], tf.upcastType(a.dtype, b.dtype));
      }
      where(condition) {
          tf.backend_util.warn('tf.where() in webgl locks the UI thread. ' +
              'Call tf.whereAsync() instead');
          const condVals = condition.dataSync();
          return whereImpl(condition.shape, condVals);
      }
      topk(x, k, sorted) {
          const xVals = x.dataSync();
          return topkImpl(xVals, x.shape, x.dtype, k, sorted);
      }
      min(x, axes) {
          tf.backend_util.assertAxesAreInnerMostDims('min', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const inSize = tf.util.sizeFromShape(reduceShape);
          const a2D = x.as2D(-1, inSize);
          return this.reduce(a2D, 'min', a2D.dtype).reshape(outShape);
      }
      minimum(a, b) {
          if (this.shouldExecuteOnCPU([a, b])) {
              return this.cpuBackend.minimum(a, b);
          }
          const program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
              new BinaryOpPackedProgram(MIN$1, a.shape, b.shape) :
              new BinaryOpProgram(MIN, a.shape, b.shape);
          return this.compileAndRun(program, [a, b]);
      }
      mod(a, b) {
          const program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
              new BinaryOpPackedProgram(MOD$1, a.shape, b.shape) :
              new BinaryOpProgram(MOD, a.shape, b.shape);
          return this.compileAndRun(program, [a, b]);
      }
      max(x, axes) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.max(x, axes);
          }
          tf.backend_util.assertAxesAreInnerMostDims('max', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const inSize = tf.util.sizeFromShape(reduceShape);
          const a2D = x.as2D(-1, inSize);
          return this.reduce(a2D, 'max', a2D.dtype).reshape(outShape);
      }
      maximum(a, b) {
          if (this.shouldExecuteOnCPU([a, b])) {
              return this.cpuBackend.maximum(a, b);
          }
          const program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
              new BinaryOpPackedProgram(MAX$1, a.shape, b.shape) :
              new BinaryOpProgram(MAX, a.shape, b.shape);
          return this.compileAndRun(program, [a, b]);
      }
      all(x, axes) {
          tf.backend_util.assertAxesAreInnerMostDims('all', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const inSize = tf.util.sizeFromShape(reduceShape);
          const a2D = x.as2D(-1, inSize);
          return this.reduce(a2D, 'all', a2D.dtype).reshape(outShape);
      }
      any(x, axes) {
          tf.backend_util.assertAxesAreInnerMostDims('any', axes, x.rank);
          const [outShape, reduceShape] = tf.backend_util.computeOutAndReduceShapes(x.shape, axes);
          const inSize = tf.util.sizeFromShape(reduceShape);
          const a2D = x.as2D(-1, inSize);
          return this.reduce(a2D, 'any', a2D.dtype).reshape(outShape);
      }
      floorDiv(a, b) {
          const op = INT_DIV;
          const outputDtype = 'int32';
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, INT_DIV$1, outputDtype);
          }
          const program = new BinaryOpProgram(op, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], outputDtype);
      }
      add(a, b) {
          if (a.dtype === 'complex64' && b.dtype === 'complex64') {
              return this.complexSeparableBinaryOp(a, b, ADD);
          }
          if (this.shouldExecuteOnCPU([a, b])) {
              return this.cpuBackend.add(a, b);
          }
          const dtype = tf.upcastType(a.dtype, b.dtype);
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, ADD, dtype);
          }
          const program = new BinaryOpProgram(ADD, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], dtype);
      }
      packedUnaryOp(x, op, dtype) {
          const program = new UnaryOpPackedProgram(x.shape, op);
          return this.compileAndRun(program, [x], dtype);
      }
      packedBinaryOp(a, b, op, dtype, checkOutOfBounds = false) {
          const program = new BinaryOpPackedProgram(op, a.shape, b.shape, checkOutOfBounds);
          return this.compileAndRun(program, [a, b], dtype);
      }
      /**
       * Computes a complex binary operation that can be decomposed into a simple
       * binary operation on both the real and imagary parts.
       */
      complexSeparableBinaryOp(a, b, op) {
          const aData = this.texData.get(a.dataId);
          const bData = this.texData.get(b.dataId);
          const [real, imag] = [
              [aData.complexTensors.real, bData.complexTensors.real],
              [aData.complexTensors.imag, bData.complexTensors.imag]
          ].map(complexParts => {
              const [aPart, bPart] = complexParts;
              const aHandle = this.makeComplexComponentTensorInfo(a, aPart);
              const bHandle = this.makeComplexComponentTensorInfo(b, bPart);
              const program = new BinaryOpProgram(op, a.shape, b.shape);
              return this.compileAndRun(program, [aHandle, bHandle], tf.upcastType(aPart.dtype, bPart.dtype));
          });
          const complex = this.complex(real, imag);
          real.dispose();
          imag.dispose();
          return complex;
      }
      // Returns a TensorInfo with the complex shape and the dataId of the
      // underlying part. We need to do this because a reshaped complex tensor is
      // not reflected in its parts.
      makeComplexComponentTensorInfo(complexTensor, complexPart) {
          return {
              dataId: complexPart.dataId,
              dtype: complexPart.dtype,
              shape: complexTensor.shape
          };
      }
      addN(tensors) {
          if (tensors.length === 1) {
              return tensors[0];
          }
          // Limit the number of uploaded textures for optimization.
          if (tensors.length > tf.env().get('WEBGL_MAX_TEXTURES_IN_SHADER')) {
              const midIndex = Math.floor(tensors.length / 2);
              const leftSide = this.addN(tensors.slice(0, midIndex));
              const rightSide = this.addN(tensors.slice(midIndex));
              return this.addN([leftSide, rightSide]);
          }
          const dtype = tensors.map(t => t.dtype).reduce((d1, d2) => tf.upcastType(d1, d2));
          const shapes = tensors.map(t => t.shape);
          // We can make sure shapes are identical in op level.
          const usePackedOp = tf.env().getBool('WEBGL_PACK');
          const program = usePackedOp ?
              new AddNPackedProgram(tensors[0].shape, shapes) :
              new AddNProgram(tensors[0].shape, shapes);
          return this.compileAndRun(program, tensors, dtype);
      }
      subtract(a, b) {
          if (a.dtype === 'complex64' && b.dtype === 'complex64') {
              return this.complexSeparableBinaryOp(a, b, SUB);
          }
          if (this.shouldExecuteOnCPU([a, b])) {
              return this.cpuBackend.subtract(a, b);
          }
          const dtype = tf.upcastType(a.dtype, b.dtype);
          if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
              return this.packedBinaryOp(a, b, SUB, a.dtype);
          }
          const program = new BinaryOpProgram(SUB, a.shape, b.shape);
          return this.compileAndRun(program, [a, b], dtype);
      }
      pow(a, b) {
          const usePackedOp = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS');
          const program = usePackedOp ?
              new BinaryOpPackedProgram(POW$1, a.shape, b.shape) :
              new BinaryOpProgram(POW, a.shape, b.shape);
          const dtype = tf.upcastType(a.dtype, b.dtype);
          return this.compileAndRun(program, [a, b], dtype);
      }
      ceil(x) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.ceil(x);
          }
          if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
              return this.packedUnaryOp(x, CEIL, x.dtype);
          }
          const program = new UnaryOpProgram(x.shape, CEIL);
          return this.compileAndRun(program, [x]);
      }
      floor(x) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.floor(x);
          }
          if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
              return this.packedUnaryOp(x, FLOOR, x.dtype);
          }
          const program = new UnaryOpProgram(x.shape, FLOOR);
          return this.compileAndRun(program, [x]);
      }
      sign(x) {
          const program = new UnaryOpProgram(x.shape, SIGN);
          return this.compileAndRun(program, [x]);
      }
      isNaN(x) {
          const program = new UnaryOpProgram(x.shape, IS_NAN);
          return this.compileAndRun(program, [x], 'bool');
      }
      isInf(x) {
          const program = new UnaryOpProgram(x.shape, IS_INF);
          return this.compileAndRun(program, [x], 'bool');
      }
      isFinite(x) {
          const program = new UnaryOpProgram(x.shape, IS_FINITE);
          return this.compileAndRun(program, [x], 'bool');
      }
      round(x) {
          const program = new UnaryOpProgram(x.shape, ROUND);
          return this.compileAndRun(program, [x]);
      }
      exp(x) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.exp(x);
          }
          if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
              return this.packedUnaryOp(x, EXP, x.dtype);
          }
          const program = new UnaryOpProgram(x.shape, EXP);
          return this.compileAndRun(program, [x]);
      }
      expm1(x) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.expm1(x);
          }
          if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
              return this.packedUnaryOp(x, EXPM1, x.dtype);
          }
          const program = new UnaryOpProgram(x.shape, EXPM1);
          return this.compileAndRun(program, [x]);
      }
      softmax(logits, dim) {
          const axes = tf.util.parseAxisParam([dim], logits.shape);
          const maxLogit = this.max(logits, axes);
          const expandedShape = tf.backend_util.expandShapeToKeepDim(maxLogit.shape, axes);
          const a = this.subtract(logits, maxLogit.reshape(expandedShape));
          const b = this.exp(a);
          const sumExp = this.sum(b, axes).reshape(expandedShape);
          // TODO(annxingyuan): Call divImpl rather than op as part of softmax kernel
          // modularization.
          return tf.div(b, sumExp);
      }
      log(x) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.log(x);
          }
          if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
              return this.packedUnaryOp(x, LOG$1, x.dtype);
          }
          const program = new UnaryOpProgram(x.shape, LOG);
          return this.compileAndRun(program, [x]);
      }
      log1p(x) {
          const program = new UnaryOpProgram(x.shape, LOG1P);
          return this.compileAndRun(program, [x]);
      }
      sqrt(x) {
          const program = new UnaryOpProgram(x.shape, SQRT);
          return this.compileAndRun(program, [x]);
      }
      rsqrt(x) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.rsqrt(x);
          }
          const program = new UnaryOpProgram(x.shape, RSQRT);
          return this.compileAndRun(program, [x]);
      }
      reciprocal(x) {
          const program = new UnaryOpProgram(x.shape, RECIPROCAL);
          return this.compileAndRun(program, [x]);
      }
      relu(x) {
          let program;
          if (tf.env().getBool('WEBGL_PACK')) {
              program = new UnaryOpPackedProgram(x.shape, RELU$1);
          }
          else {
              program = new UnaryOpProgram(x.shape, RELU);
          }
          return this.compileAndRun(program, [x]);
      }
      relu6(x) {
          let program;
          if (tf.env().getBool('WEBGL_PACK')) {
              program = new UnaryOpPackedProgram(x.shape, RELU6$1);
          }
          else {
              program = new UnaryOpProgram(x.shape, RELU6);
          }
          return this.compileAndRun(program, [x]);
      }
      prelu(x, alpha) {
          const program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
              new BinaryOpPackedProgram(PRELU$1, x.shape, alpha.shape) :
              new BinaryOpProgram(PRELU, x.shape, alpha.shape);
          return this.compileAndRun(program, [x, alpha]);
      }
      elu(x) {
          if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
              return this.packedUnaryOp(x, ELU$1, x.dtype);
          }
          const program = new UnaryOpProgram(x.shape, ELU);
          return this.compileAndRun(program, [x]);
      }
      eluDer(dy, y) {
          const program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
              new BinaryOpPackedProgram(ELU_DER$1, dy.shape, y.shape) :
              new BinaryOpProgram(ELU_DER, dy.shape, y.shape);
          return this.compileAndRun(program, [dy, y]);
      }
      selu(x) {
          const program = new UnaryOpProgram(x.shape, SELU);
          return this.compileAndRun(program, [x]);
      }
      int(x) {
          const program = new UnaryOpProgram(x.shape, TO_INT);
          return this.compileAndRun(program, [x], 'int32');
      }
      clip(x, min, max) {
          let program;
          if (tf.env().getBool('WEBGL_PACK_CLIP')) {
              program = new ClipPackedProgram(x.shape);
          }
          else {
              program = new ClipProgram(x.shape);
          }
          const customSetup = program.getCustomSetupFunc(min, max);
          return this.compileAndRun(program, [x], null, customSetup);
      }
      abs(x) {
          if (this.shouldExecuteOnCPU([x])) {
              return this.cpuBackend.abs(x);
          }
          if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
              return this.packedUnaryOp(x, ABS, x.dtype);
          }
          const program = new UnaryOpProgram(x.shape, ABS);
          return this.compileAndRun(program, [x]);
      }
      complexAbs(x) {
          const xData = this.texData.get(x.dataId);
          const program = new ComplexAbsProgram(x.shape);
          const inputs = [
              this.makeComplexComponentTensorInfo(x, xData.complexTensors.real),
              this.makeComplexComponentTensorInfo(x, xData.complexTensors.imag),
          ];
          return this.compileAndRun(program, inputs);
      }
      sigmoid(x) {
          const program = new UnaryOpProgram(x.shape, SIGMOID);
          return this.compileAndRun(program, [x]);
      }
      softplus(x) {
          const program = new UnaryOpProgram(x.shape, SOFTPLUS);
          return this.compileAndRun(program, [x]);
      }
      sin(x) {
          const program = new UnaryOpProgram(x.shape, SIN);
          return this.compileAndRun(program, [x]);
      }
      cos(x) {
          const program = new UnaryOpProgram(x.shape, COS);
          return this.compileAndRun(program, [x]);
      }
      tan(x) {
          const program = new UnaryOpProgram(x.shape, TAN);
          return this.compileAndRun(program, [x]);
      }
      asin(x) {
          const program = new UnaryOpProgram(x.shape, ASIN);
          return this.compileAndRun(program, [x]);
      }
      acos(x) {
          const program = new UnaryOpProgram(x.shape, ACOS);
          return this.compileAndRun(program, [x]);
      }
      atan(x) {
          const program = new UnaryOpProgram(x.shape, ATAN);
          return this.compileAndRun(program, [x]);
      }
      atan2(a, b) {
          const program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
              new BinaryOpPackedProgram(ATAN2$1, a.shape, b.shape) :
              new BinaryOpProgram(ATAN2, a.shape, b.shape);
          return this.compileAndRun(program, [a, b]);
      }
      sinh(x) {
          const program = new UnaryOpProgram(x.shape, SINH);
          return this.compileAndRun(program, [x]);
      }
      cosh(x) {
          const program = new UnaryOpProgram(x.shape, COSH);
          return this.compileAndRun(program, [x]);
      }
      tanh(x) {
          const program = new UnaryOpProgram(x.shape, TANH);
          return this.compileAndRun(program, [x]);
      }
      asinh(x) {
          const program = new UnaryOpProgram(x.shape, ASINH);
          return this.compileAndRun(program, [x]);
      }
      acosh(x) {
          const program = new UnaryOpProgram(x.shape, ACOSH);
          return this.compileAndRun(program, [x]);
      }
      atanh(x) {
          const program = new UnaryOpProgram(x.shape, ATANH);
          return this.compileAndRun(program, [x]);
      }
      erf(x) {
          const program = new UnaryOpProgram(x.shape, ERF);
          return this.compileAndRun(program, [x]);
      }
      step(x, alpha) {
          const program = new UnaryOpProgram(x.shape, STEP(alpha));
          return this.compileAndRun(program, [x]);
      }
      conv2dByMatMul(x, filter, convInfo, bias, activation, preluActivationWeights) {
          // Reshapes conv2D input to 2D tensors, uses matMul and then reshape the
          // result from 2D to 4D.
          const xShape = x.shape;
          const xTexData = this.texData.get(x.dataId);
          const sharedMatMulDim = convInfo.inChannels;
          const outerShapeX = xShape[0] * xShape[1] * xShape[2];
          const outerShapeFilter = convInfo.outChannels;
          const isChannelsLast = convInfo.dataFormat === 'channelsLast';
          const transposeA = false;
          const transposeB = false;
          // TODO: Once reduction ops are packed, batchMatMul will always be packed
          // and we can remove this condition.
          const batchMatMulWillBeUnpacked = (outerShapeX === 1 || outerShapeFilter === 1) &&
              sharedMatMulDim > MATMUL_SHARED_DIM_THRESHOLD;
          const reshapeWillBeExpensive = xShape[2] % 2 !== 0 && !!xTexData.isPacked;
          if (batchMatMulWillBeUnpacked || !tf.env().getBool('WEBGL_LAZILY_UNPACK') ||
              !tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ||
              !reshapeWillBeExpensive) {
              const targetShape = isChannelsLast ? xShape[0] * xShape[1] * xShape[2] :
                  xShape[0] * xShape[2] * xShape[3];
              const xReshaped = this.reshape(x, [1, targetShape, convInfo.inChannels]);
              const filterReshaped = this.reshape(filter, [1, convInfo.inChannels, convInfo.outChannels]);
              return this.reshape(this.fusedBatchMatMul({
                  a: xReshaped,
                  b: filterReshaped,
                  transposeA,
                  transposeB,
                  bias,
                  activation,
                  preluActivationWeights
              }), convInfo.outShape);
          }
          // Following optimization is specific to packed |x| with odd row count
          // (For example, in channelLast mode, 'row count' refers to x.shape[2]):
          // we avoid expensive packed 2x2 reshape by padding row count to next,
          // even number. When x.shape[2] is odd, the result of packed batchMatMul is
          // the same (has the same texture layout and and values in the texture) as
          // it is for even x.shape[2] + 1. We make the odd-rows tensor to look like
          // even-rows tensor before the operation and, after the batchMatMul,
          // fix the even-rows result to have odd number of rows.
          const targetShape = isChannelsLast ?
              xShape[0] * xShape[1] * (xShape[2] + 1) :
              xShape[0] * xShape[2] * (xShape[3] + 1);
          const xReshaped = {
              dataId: x.dataId,
              shape: [1, targetShape, convInfo.inChannels],
              dtype: x.dtype
          };
          // xTexData.shape gets referenced from GPGPUBinary.inShapeInfos.
          // Decrementing row count, after batchMatMul->...->compileProgram leads to
          // invalid row count within the reference in GPGPUBinary.inShapeInfos.
          // Alternative fix would be to provide a copy to GPGPUBinary.inShapeInfos
          // in compileProgram method, but that would affect compilation of all
          // programs - instead, provide a copy here, with even row count, before
          // calling batchMatMul->...->compileProgram and after that, the original
          // xTexData.shape is restored.
          const originalXTexDataShape = xTexData.shape;
          xTexData.shape = xTexData.shape.slice();
          xTexData.shape[xTexData.shape.length - 2]++;
          tf.util.assert(isReshapeFree(xTexData.shape, xReshaped.shape), () => `packed reshape ${xTexData.shape} to ${xReshaped.shape} isn't free`);
          const filterReshaped = this.reshape(filter, [1, convInfo.inChannels, convInfo.outChannels]);
          const pointwiseConv = this.fusedBatchMatMul({
              a: xReshaped,
              b: filterReshaped,
              transposeA,
              transposeB,
              bias,
              activation,
              preluActivationWeights
          });
          const pointwiseConvTexData = this.texData.get(pointwiseConv.dataId);
          tf.util.assert(pointwiseConvTexData.isPacked, () => 'batchMatMul result is expected to be packed');
          // Restore the input shape to original.
          xTexData.shape = originalXTexDataShape;
          // Set the output shape - there is no need for expensive reshape as data
          // layout is already correct.
          pointwiseConvTexData.shape = convInfo.outShape;
          return tf.engine().makeTensorFromDataId(pointwiseConv.dataId, convInfo.outShape, pointwiseConv.dtype);
      }
      conv2dWithIm2Row(x, filter, convInfo, bias, activation, preluActivationWeights) {
          // Rearranges conv2d input so each block to be convolved over forms the
          // column of a new matrix with shape [filterWidth * filterHeight *
          // inChannels, outHeight * outWidth]. The filter is also rearranged so each
          // output channel forms a row of a new matrix with shape [outChannels,
          // filterWidth * filterHeight * inChannels]. The convolution is then
          // computed by multiplying these matrices and reshaping the result.
          const { filterWidth, filterHeight, inChannels, outWidth, outHeight, dataFormat } = convInfo;
          const isChannelsLast = dataFormat === 'channelsLast';
          const sharedDim = filterWidth * filterHeight * inChannels;
          const numCols = outHeight * outWidth;
          const x2ColShape = [sharedDim, numCols];
          const transposeA = true;
          const transposeB = false;
          const xSqueezed = x.squeeze([0]);
          const w2Row = filter.reshape([1, sharedDim, -1]);
          const im2ColProgram = new Im2ColPackedProgram(x2ColShape, xSqueezed.shape, convInfo);
          const im2Col = this.compileAndRun(im2ColProgram, [xSqueezed]).reshape([
              1, x2ColShape[0], x2ColShape[1]
          ]);
          const hasBias = bias != null;
          const hasPreluActivationWeights = preluActivationWeights != null;
          const fusedActivation = activation ? mapActivationToShaderProgram(activation, true) : null;
          const matmulProgram = new MatMulPackedProgram(im2Col.shape, [1, numCols, convInfo.outChannels], transposeA, transposeB, hasBias, fusedActivation, hasPreluActivationWeights);
          const inputs = [im2Col, w2Row];
          if (bias) {
              inputs.push(bias);
          }
          if (hasPreluActivationWeights) {
              inputs.push(preluActivationWeights);
          }
          const product = this.compileAndRun(matmulProgram, inputs);
          if (isChannelsLast) {
              return product.reshape([1, outHeight, outWidth, convInfo.outChannels]);
          }
          else {
              return product.reshape([1, convInfo.outChannels, outHeight, outWidth]);
          }
      }
      fusedConv2d({ input, filter, convInfo, bias, activation, preluActivationWeights }) {
          if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
              convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
              convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
              (convInfo.padInfo.type === 'SAME' ||
                  convInfo.padInfo.type === 'VALID')) {
              return this.conv2dByMatMul(input, filter, convInfo, bias, activation, preluActivationWeights);
          }
          if (tf.env().getBool('WEBGL_CONV_IM2COL') && input.shape[0] === 1) {
              return this.conv2dWithIm2Row(input, filter, convInfo, bias, activation, preluActivationWeights);
          }
          const hasBias = bias != null;
          const hasPreluActivationWeights = preluActivationWeights != null;
          const fusedActivation = activation ? mapActivationToShaderProgram(activation, false) : null;
          const program = new Conv2DProgram(convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
          const inputs = [input, filter];
          if (bias) {
              inputs.push(bias);
          }
          if (preluActivationWeights) {
              inputs.push(preluActivationWeights);
          }
          return this.compileAndRun(program, inputs);
      }
      conv2d(x, filter, convInfo) {
          if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
              convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
              convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
              (convInfo.padInfo.type === 'SAME' ||
                  convInfo.padInfo.type === 'VALID')) {
              return this.conv2dByMatMul(x, filter, convInfo);
          }
          if (tf.env().getBool('WEBGL_CONV_IM2COL') && x.shape[0] === 1) {
              return this.conv2dWithIm2Row(x, filter, convInfo);
          }
          const program = new Conv2DProgram(convInfo);
          return this.compileAndRun(program, [x, filter]);
      }
      conv2dDerInput(dy, filter, convInfo) {
          const program = new Conv2DDerInputProgram(convInfo);
          return this.compileAndRun(program, [dy, filter]);
      }
      conv2dDerFilter(x, dy, convInfo) {
          const program = new Conv2DDerFilterProgram(convInfo);
          return this.compileAndRun(program, [x, dy]);
      }
      fusedDepthwiseConv2D({ input, filter, convInfo, bias, activation, preluActivationWeights }) {
          const shouldPackDepthwiseConv = tf.env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
              convInfo.strideWidth <= 2 &&
              convInfo.outChannels / convInfo.inChannels === 1;
          const fusedActivation = activation ?
              mapActivationToShaderProgram(activation, shouldPackDepthwiseConv) :
              null;
          const inputs = [input, filter];
          const hasBias = bias != null;
          const hasPreluActivationWeights = preluActivationWeights != null;
          if (hasBias) {
              inputs.push(bias);
          }
          if (hasPreluActivationWeights) {
              inputs.push(preluActivationWeights);
          }
          let program;
          if (shouldPackDepthwiseConv) {
              program = new DepthwiseConvPacked2DProgram(convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
              return this.compileAndRun(program, inputs);
          }
          program = new DepthwiseConv2DProgram(convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
          return this.compileAndRun(program, inputs);
      }
      depthwiseConv2D(x, filter, convInfo) {
          let program;
          if (tf.env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
              convInfo.strideWidth <= 2 &&
              convInfo.outChannels / convInfo.inChannels === 1) {
              program = new DepthwiseConvPacked2DProgram(convInfo);
              return this.compileAndRun(program, [x, filter]);
          }
          program = new DepthwiseConv2DProgram(convInfo);
          return this.compileAndRun(program, [x, filter]);
      }
      depthwiseConv2DDerInput(dy, filter, convInfo) {
          const program = new DepthwiseConv2DDerInputProgram(convInfo);
          return this.compileAndRun(program, [dy, filter]);
      }
      depthwiseConv2DDerFilter(x, dy, convInfo) {
          const program = new DepthwiseConv2DDerFilterProgram(convInfo);
          return this.compileAndRun(program, [x, dy]);
      }
      conv3d(x, filter, convInfo) {
          const program = new Conv3DProgram(convInfo);
          return this.compileAndRun(program, [x, filter]);
      }
      conv3dDerInput(dy, filter, convInfo) {
          const program = new Conv3DDerInputProgram(convInfo);
          return this.compileAndRun(program, [dy, filter]);
      }
      conv3dDerFilter(x, dy, convInfo) {
          const program = new Conv3DDerFilterProgram(convInfo);
          return this.compileAndRun(program, [x, dy]);
      }
      maxPool(x, convInfo) {
          const program = new Pool2DProgram(convInfo, 'max', false);
          return this.compileAndRun(program, [x]);
      }
      avgPool(x, convInfo) {
          const program = new Pool2DProgram(convInfo, 'avg', false);
          return this.compileAndRun(program, [x], 'float32');
      }
      maxPoolBackprop(dy, x, y, convInfo) {
          const getPositions = true;
          const maxPoolPositionsProgram = new Pool2DProgram(convInfo, 'max', getPositions);
          const maxPoolPositions = this.compileAndRun(maxPoolPositionsProgram, [x]);
          const maxPoolBackPropProgram = new MaxPool2DBackpropProgram(convInfo);
          const result = this.compileAndRun(maxPoolBackPropProgram, [dy, maxPoolPositions], x.dtype);
          maxPoolPositions.dispose();
          return result;
      }
      avgPoolBackprop(dy, x, convInfo) {
          const avgPoolBackpropProgram = new AvgPool2DBackpropProgram(convInfo);
          return this.compileAndRun(avgPoolBackpropProgram, [dy], x.dtype);
      }
      cast(x, dtype) {
          return tf.backend_util.castTensor(x, dtype, this);
      }
      unstack(x, axis) {
          const num = x.shape[axis];
          const outShape = new Array(x.rank - 1);
          let outIndex = 0;
          for (let i = 0; i < x.rank; i++) {
              if (i !== axis) {
                  outShape[outIndex++] = x.shape[i];
              }
          }
          const begin = new Array(x.rank).fill(0);
          const size = x.shape.slice();
          size[axis] = 1;
          const res = new Array(num);
          for (let i = 0; i < res.length; i++) {
              begin[axis] = i;
              res[i] = this.slice(x, begin, size).reshape(outShape);
          }
          return res;
      }
      avgPool3d(x, convInfo) {
          const program = new Pool3DProgram(convInfo, 'avg', false);
          return this.compileAndRun(program, [x], 'float32');
      }
      avgPool3dBackprop(dy, x, convInfo) {
          const avgPool3dBackpropProgram = new AvgPool3DBackpropProgram(convInfo);
          return this.compileAndRun(avgPool3dBackpropProgram, [dy], x.dtype);
      }
      maxPool3d(x, convInfo) {
          const program = new Pool3DProgram(convInfo, 'max', false);
          return this.compileAndRun(program, [x], 'float32');
      }
      maxPool3dBackprop(dy, x, y, convInfo) {
          const getPositions = true;
          const maxPool3dPositionsProgram = new Pool3DProgram(convInfo, 'max', getPositions);
          const maxPool3dPositions = this.compileAndRun(maxPool3dPositionsProgram, [x]);
          const maxPool3dBackPropProgram = new MaxPool3DBackpropProgram(convInfo);
          const result = this.compileAndRun(maxPool3dBackPropProgram, [dy, maxPool3dPositions], x.dtype);
          maxPool3dPositions.dispose();
          return result;
      }
      reshape(x, shape) {
          const texData = this.texData.get(x.dataId);
          if (texData.isPacked && !isReshapeFree(x.shape, shape) &&
              !(texData.texture !== null &&
                  isReshapeFree(texData.shape, shape))) {
              const info = this.packedReshape(x, shape);
              return tf.engine().makeTensorFromDataId(info.dataId, info.shape, info.dtype);
          }
          return tf.backend_util.reshapeTensor(x, shape);
      }
      resizeBilinear(x, newHeight, newWidth, alignCorners) {
          const program = tf.env().getBool('WEBGL_PACK_IMAGE_OPERATIONS') ?
              new ResizeBilinearPackedProgram(x.shape, newHeight, newWidth, alignCorners) :
              new ResizeBilinearProgram(x.shape, newHeight, newWidth, alignCorners);
          return this.compileAndRun(program, [x], 'float32');
      }
      resizeBilinearBackprop(dy, x, alignCorners) {
          const program = new ResizeBilinearBackpropProgram(dy, x, alignCorners);
          return this.compileAndRun(program, [dy]);
      }
      resizeNearestNeighbor(x, newHeight, newWidth, alignCorners) {
          const program = new ResizeNearestNeighborProgram(x.shape, newHeight, newWidth, alignCorners);
          return this.compileAndRun(program, [x]);
      }
      resizeNearestNeighborBackprop(dy, x, alignCorners) {
          const program = new ResizeNearestNeigborBackpropProgram(dy, x, alignCorners);
          return this.compileAndRun(program, [dy]);
      }
      multinomial(logits, normalized, numSamples, seed) {
          const probs = normalized ? logits : tf.softmax(logits);
          const batchSize = probs.shape[0];
          const numOutcomes = probs.shape[1];
          const program = new MultinomialProgram(batchSize, numOutcomes, numSamples);
          const customSetup = program.getCustomSetupFunc(seed);
          return this.compileAndRun(program, [probs], 'int32', customSetup);
      }
      oneHot(indices, depth, onValue, offValue) {
          const program = new OneHotProgram(indices.size, depth, onValue, offValue);
          return this.compileAndRun(program, [indices]);
      }
      diag(x) {
          const program = new DiagProgram(x.size);
          return this.compileAndRun(program, [x]);
      }
      nonMaxSuppression(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
          tf.backend_util.warn('tf.nonMaxSuppression() in webgl locks the UI thread. ' +
              'Call tf.nonMaxSuppressionAsync() instead');
          const boxesVals = boxes.dataSync();
          const scoresVals = scores.dataSync();
          return nonMaxSuppressionV3(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
      }
      cropAndResize(image, boxes, boxIndex, cropSize, method, extrapolationValue) {
          const program = new CropAndResizeProgram(image.shape, boxes.shape, cropSize, method, extrapolationValue);
          return this.compileAndRun(program, [image, boxes, boxIndex], 'float32');
      }
      depthToSpace(x, blockSize, dataFormat) {
          tf.util.assert(blockSize > 1, () => `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);
          const batchSize = x.shape[0];
          const inputHeight = (dataFormat === 'NHWC') ? x.shape[1] : x.shape[2];
          const inputWidth = (dataFormat === 'NHWC') ? x.shape[2] : x.shape[3];
          const inputDepth = (dataFormat === 'NHWC') ? x.shape[3] : x.shape[1];
          const outputHeight = inputHeight * blockSize;
          const outputWidth = inputWidth * blockSize;
          const outputDepth = inputDepth / (blockSize * blockSize);
          const outputShape = (dataFormat === 'NHWC') ?
              [batchSize, outputHeight, outputWidth, outputDepth] :
              [batchSize, outputDepth, outputHeight, outputWidth];
          const program = new DepthToSpaceProgram(outputShape, blockSize, dataFormat);
          return this.compileAndRun(program, [x]);
      }
      split(x, sizeSplits, axis) {
          return split(x, sizeSplits, axis);
      }
      scatterND(indices, updates, shape) {
          const { sliceRank, numUpdates, sliceSize, strides, outputSize } = tf.backend_util.calculateShapes(updates, indices, shape);
          const flattenShape = [outputSize / sliceSize, sliceSize];
          const flattenIndices = indices.reshape([numUpdates, sliceRank]);
          const flattenX = updates.reshape([numUpdates, sliceSize]);
          if (outputSize === 0) {
              return tf.backend_util.reshapeTensor(tf.tensor([]), shape);
          }
          const defaultValue = tf.scalar(0);
          const program = new ScatterProgram(numUpdates, sliceRank, flattenIndices.rank, flattenX.rank, strides, flattenShape);
          const res = this.compileAndRun(program, [flattenX, flattenIndices, defaultValue]);
          return res.reshape(shape);
      }
      sparseToDense(sparseIndices, sparseValues, outputShape, defaultValue) {
          const { sliceRank, numUpdates, strides, outputSize } = tf.backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);
          const sumDupeIndices = false;
          const program = new ScatterProgram(numUpdates, sliceRank, sparseIndices.rank, sparseValues.rank, strides, [outputSize, 1], sumDupeIndices);
          const res = this.compileAndRun(program, [sparseValues, sparseIndices, defaultValue]);
          return res.reshape(outputShape);
      }
      fft(x) {
          const inverse = false;
          return this.fftImpl(x, inverse);
      }
      ifft(x) {
          const inverse = true;
          return this.fftImpl(x, inverse);
      }
      fftImpl(x, inverse) {
          const xData = this.texData.get(x.dataId);
          const realProgram = new FFTProgram(COMPLEX_FFT.REAL, x.shape, inverse);
          const imagProgram = new FFTProgram(COMPLEX_FFT.IMAG, x.shape, inverse);
          const inputs = [
              this.makeComplexComponentTensorInfo(x, xData.complexTensors.real),
              this.makeComplexComponentTensorInfo(x, xData.complexTensors.imag),
          ];
          const real = this.compileAndRun(realProgram, inputs);
          const imag = this.compileAndRun(imagProgram, inputs);
          const complex = this.complex(real, imag).as2D(x.shape[0], x.shape[1]);
          real.dispose();
          imag.dispose();
          return complex;
      }
      gatherND(x, indices) {
          const indicesShape = indices.shape;
          const sliceRank = indicesShape[indicesShape.length - 1];
          const [resultShape, numSlices, sliceSize, strides] = tf.backend_util.prepareAndValidate(x, indices);
          const flattenIndices = indices.reshape([numSlices, sliceRank]);
          const flattenX = x.reshape([x.size / sliceSize, sliceSize]);
          const program = new GatherNDProgram(sliceRank, strides, [numSlices, sliceSize]);
          const res = this.compileAndRun(program, [flattenX, flattenIndices]);
          return res.reshape(resultShape);
      }
      fill(shape, value, dtype) {
          dtype = dtype || tf.util.inferDtype(value);
          if (dtype === 'string') {
              // String type should be handled in CPU memory.
              const values = tf.util.getArrayFromDType(dtype, tf.util.sizeFromShape(shape));
              values.fill(value);
              return tf.engine().makeTensor(values, shape, dtype, this);
          }
          else {
              const program = new FillProgram(shape, value);
              const customSetup = program.getCustomSetupFunc(value);
              return this.compileAndRun(program, [], dtype, customSetup);
          }
      }
      onesLike(x) {
          if (x.dtype === 'string') {
              throw new Error('onesLike is not supported under string dtype');
          }
          else {
              // TODO(cais, smilkov): Add WebGL shader for onesLike:
              //   https://github.com/tensorflow/tfjs/issues/1293
              return this.fill(x.shape, 1, x.dtype);
          }
      }
      zerosLike(x) {
          return this.fill(x.shape, x.dtype === 'string' ? '' : 0, x.dtype);
      }
      linspace(start, stop, num) {
          // TODO: Use CPU implementation due to the precision problem in Safari.
          return tf.backend_util.linspaceImpl(start, stop, num);
      }
      makeTensorInfo(shape, dtype) {
          const dataId = this.write(null /* values */, shape, dtype);
          this.texData.get(dataId).usage = null;
          return { dataId, shape, dtype };
      }
      makeOutput(shape, dtype) {
          const { dataId } = this.makeTensorInfo(shape, dtype);
          return tf.engine().makeTensorFromDataId(dataId, shape, dtype, this);
      }
      unpackTensor(input) {
          const program = new UnpackProgram(input.shape);
          return this.runWebGLProgram(program, [input], input.dtype);
      }
      packTensor(input) {
          const program = new PackProgram(input.shape);
          const preventEagerUnpackingOutput = true;
          return this.runWebGLProgram(program, [input], input.dtype, null /* customSetup */, preventEagerUnpackingOutput);
      }
      packedReshape(input, afterShape) {
          const input3DShape = [
              getBatchDim(input.shape),
              ...getRowsCols(input.shape)
          ];
          const input3D = {
              dtype: input.dtype,
              shape: input3DShape,
              dataId: input.dataId
          };
          const afterShapeAs3D = [
              getBatchDim(afterShape), ...getRowsCols(afterShape)
          ];
          const program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
          const preventEagerUnpackingOfOutput = true;
          const output = this.runWebGLProgram(program, [input3D], input.dtype, null /* customSetup */, preventEagerUnpackingOfOutput);
          return { dataId: output.dataId, shape: afterShape, dtype: output.dtype };
      }
      decode(dataId) {
          const texData = this.texData.get(dataId);
          const { isPacked, shape, dtype } = texData;
          const shapeAs3D = getShapeAs3D(shape);
          let program;
          if (isPacked) {
              program = new DecodeMatrixPackedProgram(shapeAs3D);
          }
          else {
              program = new DecodeMatrixProgram(shapeAs3D);
          }
          const preventEagerUnpackingOfOutput = true;
          const out = this.runWebGLProgram(program, [{ shape: shapeAs3D, dtype, dataId }], dtype, null /* customSetup */, preventEagerUnpackingOfOutput);
          return { dtype, shape, dataId: out.dataId };
      }
      runWebGLProgram(program, inputs, outputDtype, customSetup, preventEagerUnpackingOfOutput = false) {
          const output = this.makeTensorInfo(program.outputShape, outputDtype);
          const outData = this.texData.get(output.dataId);
          if (program.packedOutput) {
              outData.isPacked = true;
          }
          if (program.outPackingScheme === PackingScheme.DENSE) {
              const texelShape = getDenseTexShape(program.outputShape);
              // For a densely packed output, we explicitly set texShape
              // so it doesn't get assigned later according to our typical packing
              // scheme wherein a single texel can only contain values from adjacent
              // rows/cols.
              outData.texShape = texelShape.map(d => d * 2);
          }
          if (program.outTexUsage != null) {
              outData.usage = program.outTexUsage;
          }
          if (tf.util.sizeFromShape(output.shape) === 0) {
              // Short-circuit the computation since the result is empty (has 0 in its
              // shape).
              outData.values =
                  tf.util.getTypedArrayFromDType(output.dtype, 0);
              return output;
          }
          const dataToDispose = [];
          const inputsData = inputs.map(input => {
              if (input.dtype === 'complex64') {
                  throw new Error(`GPGPUProgram does not support complex64 input. For complex64 ` +
                      `dtypes, please separate the program into real and imaginary ` +
                      `parts.`);
              }
              let texData = this.texData.get(input.dataId);
              if (texData.texture == null) {
                  if (!program.packedInputs &&
                      tf.util.sizeFromShape(input.shape) <=
                          tf.env().getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')) {
                      // Upload small tensors that live on the CPU as uniforms, not as
                      // textures. Do this only when the environment supports 32bit floats
                      // due to problems when comparing 16bit floats with 32bit floats.
                      // TODO(https://github.com/tensorflow/tfjs/issues/821): Make it
                      // possible for packed shaders to sample from uniforms.
                      return {
                          shape: input.shape,
                          texData: null,
                          isUniform: true,
                          uniformValues: texData.values
                      };
                  }
                  // This ensures that if a packed program's inputs have not yet been
                  // uploaded to the GPU, they get uploaded as packed right off the bat.
                  if (program.packedInputs) {
                      texData.isPacked = true;
                      texData.shape = input.shape;
                  }
              }
              else if (!!texData.isPacked !== !!program.packedInputs) {
                  input = texData.isPacked ? this.unpackTensor(input) :
                      this.packTensor(input);
                  dataToDispose.push(input);
                  texData = this.texData.get(input.dataId);
              }
              else if (texData.isPacked &&
                  !isReshapeFree(texData.shape, input.shape)) {
                  // This is a special case where a texture exists for a tensor
                  // but the shapes are incompatible (due to packing constraints) because
                  // the tensor did not have a chance to go through the packed reshape
                  // shader. This only happens when we reshape the *same* tensor to form
                  // *distinct* inputs to an op, e.g. dotting a vector with itself. This
                  // case will disappear once packed uploading is the default.
                  const savedInput = input;
                  const targetShape = input.shape;
                  input.shape = texData.shape;
                  input = this.packedReshape(input, targetShape);
                  dataToDispose.push(input);
                  texData = this.texData.get(input.dataId);
                  savedInput.shape = targetShape;
              }
              this.uploadToGPU(input.dataId);
              return { shape: input.shape, texData, isUniform: false };
          });
          this.uploadToGPU(output.dataId);
          const outputData = { shape: output.shape, texData: outData, isUniform: false };
          const key = makeShaderKey(program, inputsData, outputData);
          const binary = this.getAndSaveBinary(key, () => {
              return compileProgram(this.gpgpu, program, inputsData, outputData);
          });
          const shouldTimeProgram = this.activeTimers != null;
          let query;
          if (shouldTimeProgram) {
              query = this.startTimer();
          }
          runProgram(this.gpgpu, binary, inputsData, outputData, customSetup);
          dataToDispose.forEach(info => this.disposeData(info.dataId));
          if (shouldTimeProgram) {
              query = this.endTimer(query);
              this.activeTimers.push({ name: program.constructor.name, query: this.getQueryTime(query) });
          }
          if (!tf.env().getBool('WEBGL_LAZILY_UNPACK') && outData.isPacked &&
              preventEagerUnpackingOfOutput === false) {
              const unpacked = this.unpackTensor(output);
              this.disposeData(output.dataId);
              return unpacked;
          }
          return output;
      }
      compileAndRun(program, inputs, outputDtype, customSetup, preventEagerUnpackingOfOutput = false) {
          outputDtype = outputDtype || inputs[0].dtype;
          const outInfo = this.runWebGLProgram(program, inputs, outputDtype, customSetup, preventEagerUnpackingOfOutput);
          return tf.engine().makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
      }
      getAndSaveBinary(key, getBinary) {
          if (!(key in this.binaryCache)) {
              this.binaryCache[key] = getBinary();
          }
          return this.binaryCache[key];
      }
      getTextureManager() {
          return this.textureManager;
      }
      dispose() {
          if (this.disposed) {
              return;
          }
          // Avoid disposing the compiled webgl programs during unit testing because
          // it slows down test execution.
          if (!tf.env().getBool('IS_TEST')) {
              const allKeys = Object.keys(this.binaryCache);
              allKeys.forEach(key => {
                  this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
                  delete this.binaryCache[key];
              });
          }
          this.textureManager.dispose();
          if (this.canvas != null &&
              (typeof (HTMLCanvasElement) !== 'undefined' &&
                  this.canvas instanceof HTMLCanvasElement)) {
              this.canvas.remove();
          }
          else {
              this.canvas = null;
          }
          if (this.gpgpuCreatedLocally) {
              this.gpgpu.program = null;
              this.gpgpu.dispose();
          }
          this.disposed = true;
      }
      floatPrecision() {
          if (this.floatPrecisionValue == null) {
              this.floatPrecisionValue = tf.tidy(() => {
                  if (!tf.env().get('WEBGL_RENDER_FLOAT32_ENABLED')) {
                      // Momentarily switching DEBUG flag to false so we don't throw an
                      // error trying to upload a small value.
                      const debugFlag = tf.env().getBool('DEBUG');
                      tf.env().set('DEBUG', false);
                      const underflowCheckValue = this.abs(tf.scalar(1e-8)).dataSync()[0];
                      tf.env().set('DEBUG', debugFlag);
                      if (underflowCheckValue > 0) {
                          return 32;
                      }
                  }
                  return 16;
              });
          }
          return this.floatPrecisionValue;
      }
      /** Returns the smallest representable number.  */
      epsilon() {
          return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
      }
      uploadToGPU(dataId) {
          const texData = this.texData.get(dataId);
          const { shape, dtype, values, texture, usage, isPacked } = texData;
          if (texture != null) {
              // Array is already on GPU. No-op.
              return;
          }
          const shouldTimeProgram = this.activeTimers != null;
          let start;
          if (shouldTimeProgram) {
              start = tf.util.now();
          }
          let texShape = texData.texShape;
          if (texShape == null) {
              texShape = getTextureShapeFromLogicalShape(shape, isPacked);
              texData.texShape = texShape;
          }
          if (values != null) {
              const shapeAs3D = getShapeAs3D(shape);
              let program;
              let width = texShape[1], height = texShape[0];
              const isByteArray = values instanceof Uint8Array;
              if (isPacked) {
                  [width, height] = getPackedMatrixTextureShapeWidthHeight(texShape[0], texShape[1]);
                  program = new EncodeMatrixPackedProgram(shapeAs3D, [height, width], isByteArray);
              }
              else {
                  program =
                      new EncodeMatrixProgram(shapeAs3D, [height, width], isByteArray);
              }
              const tempDenseInputHandle = this.makeTensorInfo([height, width], dtype);
              if (isByteArray) {
                  this.texData.get(tempDenseInputHandle.dataId).usage =
                      TextureUsage.PIXELS;
              }
              else {
                  this.texData.get(tempDenseInputHandle.dataId).usage =
                      TextureUsage.UPLOAD;
              }
              this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(tempDenseInputHandle.dataId), width, height, values);
              // We want the output to remain packed regardless of the value of
              // WEBGL_PACK.
              const preventEagerUnpacking = true;
              const encodedOutputTarget = this.runWebGLProgram(program, [tempDenseInputHandle], dtype, null, preventEagerUnpacking);
              // Have the original texture assume the identity of the encoded output.
              const outputTexData = this.texData.get(encodedOutputTarget.dataId);
              texData.texture = outputTexData.texture;
              texData.texShape = outputTexData.texShape;
              texData.isPacked = outputTexData.isPacked;
              texData.usage = outputTexData.usage;
              this.disposeData(tempDenseInputHandle.dataId);
              this.texData.delete(encodedOutputTarget.dataId);
              // Once uploaded, don't store the values on cpu.
              texData.values = null;
              if (shouldTimeProgram) {
                  this.uploadWaitMs += tf.util.now() - start;
              }
          }
          else {
              const newTexture = this.acquireTexture(texShape, usage, dtype, isPacked);
              texData.texture = newTexture;
          }
      }
      convertAndCacheOnCPU(dataId, float32Values) {
          const texData = this.texData.get(dataId);
          const { dtype } = texData;
          this.releaseGPUData(dataId);
          if (float32Values != null) {
              texData.values = float32ToTypedArray(float32Values, dtype);
          }
          return texData.values;
      }
      acquireTexture(texShape, texType, dtype, isPacked) {
          this.numBytesInGPU += this.computeBytes(texShape, dtype);
          if (!this.warnedAboutMemory &&
              this.numBytesInGPU > this.numMBBeforeWarning * 1024 * 1024) {
              const mb = (this.numBytesInGPU / 1024 / 1024).toFixed(2);
              this.warnedAboutMemory = true;
              console.warn(`High memory usage in GPU: ${mb} MB, ` +
                  `most likely due to a memory leak`);
          }
          return this.textureManager.acquireTexture(texShape, texType, isPacked);
      }
      computeBytes(shape, dtype) {
          return shape[0] * shape[1] * tf.util.bytesPerElement(dtype);
      }
  }
  function float32ToTypedArray(a, dtype) {
      if (dtype === 'float32' || dtype === 'complex64') {
          return a;
      }
      else if (dtype === 'int32' || dtype === 'bool') {
          const result = (dtype === 'int32') ? new Int32Array(a.length) :
              new Uint8Array(a.length);
          for (let i = 0; i < result.length; ++i) {
              result[i] = Math.round(a[i]);
          }
          return result;
      }
      else {
          throw new Error(`Unknown dtype ${dtype}`);
      }
  }

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  function divImpl(a, b, backend) {
      let program = new BinaryOpProgram(DIV, a.shape, b.shape);
      if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
          program = new BinaryOpPackedProgram(DIV$1, a.shape, b.shape, true);
      }
      const output = backend.runWebGLProgram(program, [a, b], 'float32');
      return output;
  }

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  const divConfig = {
      kernelName: tf.Div,
      backendName: 'webgl',
      kernelFunc: ({ inputs, backend }) => {
          const { a, b } = inputs;
          const webglBackend = backend;
          return divImpl(a, b, webglBackend);
      }
  };

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class FromPixelsProgram {
      constructor(outputShape) {
          this.variableNames = ['A'];
          const glsl = getGlslDifferences();
          const [height, width,] = outputShape;
          this.outputShape = outputShape;
          this.userCode = `
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${width}.0, ${height}.0);

        vec4 values = ${glsl.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `;
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  class FromPixelsPackedProgram {
      constructor(outputShape) {
          this.variableNames = ['A'];
          this.packedInputs = false;
          this.packedOutput = true;
          const glsl = getGlslDifferences();
          const [height, width,] = outputShape;
          this.outputShape = outputShape;
          this.userCode = `
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${width}.0, ${height}.0);
            vec4 values = ${glsl.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${glsl.output} = result;
      }
    `;
      }
  }

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
  const fromPixelsConfig = {
      kernelName: tf.FromPixels,
      backendName: 'webgl',
      kernelFunc: fromPixels,
  };
  let fromPixels2DContext;
  function fromPixels(args) {
      const { inputs, backend, attrs } = args;
      let { pixels } = inputs;
      const { numChannels } = attrs;
      const isVideo = typeof (HTMLVideoElement) !== 'undefined' &&
          pixels instanceof HTMLVideoElement;
      const isImage = typeof (HTMLImageElement) !== 'undefined' &&
          pixels instanceof HTMLImageElement;
      const [width, height] = isVideo ?
          [
              pixels.videoWidth,
              pixels.videoHeight
          ] :
          [pixels.width, pixels.height];
      const texShape = [height, width];
      const outShape = [height, width, numChannels];
      if (isImage || isVideo) {
          if (fromPixels2DContext == null) {
              fromPixels2DContext = document.createElement('canvas').getContext('2d');
          }
          fromPixels2DContext.canvas.width = width;
          fromPixels2DContext.canvas.height = height;
          fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
          pixels = fromPixels2DContext.canvas;
      }
      const tempPixelHandle = backend.makeTensorInfo(texShape, 'int32');
      // This is a byte texture with pixels.
      backend.texData.get(tempPixelHandle.dataId).usage = TextureUsage.PIXELS;
      backend.gpgpu.uploadPixelDataToTexture(backend.getTexture(tempPixelHandle.dataId), pixels);
      const program = tf.env().getBool('WEBGL_PACK') ?
          new FromPixelsPackedProgram(outShape) :
          new FromPixelsProgram(outShape);
      const res = backend.runWebGLProgram(program, [tempPixelHandle], 'int32');
      backend.disposeData(tempPixelHandle.dataId);
      return res;
  }

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  function maxPoolWithArgmaxImpl(x, includeBatchInIndex, convInfo, backend) {
      let program = new Pool2DProgram(convInfo, 'max', false);
      const poolOutput = backend.runWebGLProgram(program, [x], 'float32');
      program = new Pool2DProgram(convInfo, 'max', true, true, includeBatchInIndex);
      const indexOutput = backend.runWebGLProgram(program, [x], 'float32');
      return [poolOutput, indexOutput];
  }

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  const maxPoolWithArgmaxConfig = {
      kernelName: tf.MaxPoolWithArgmax,
      backendName: 'webgl',
      kernelFunc: ({ inputs, attrs, backend }) => {
          const { x } = inputs;
          const { filterSize, strides, pad, includeBatchInIndex } = attrs;
          const webglBackend = backend;
          tf.util.assert(x.shape.length === 4, () => `Error in maxPool: input must be rank 4 but got rank ${x.shape.length}.`);
          const dilations = [1, 1];
          tf.util.assert(tf.backend_util.eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in maxPool: Either strides or dilations must be 1. ' +
              `Got strides ${strides} and dilations '${dilations}'`);
          const convInfo = tf.backend_util.computePool2DInfo(x.shape, filterSize, strides, dilations, pad);
          const [result, indexes] = maxPoolWithArgmaxImpl(x, includeBatchInIndex, convInfo, webglBackend);
          return [result, indexes];
      }
  };

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
  const nonMaxSuppressionV5 = tf.kernel_impls.nonMaxSuppressionV5;
  const nonMaxSuppressionV5Config = {
      kernelName: tf.NonMaxSuppressionV5,
      backendName: 'webgl',
      kernelFunc: ({ inputs, backend, attrs }) => {
          tf.backend_util.warn('tf.nonMaxSuppression() in webgl locks the UI thread. ' +
              'Call tf.nonMaxSuppressionAsync() instead');
          const { boxes, scores } = inputs;
          const { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma } = attrs;
          const gpuBackend = backend;
          const boxesVals = gpuBackend.readSync(boxes.dataId);
          const scoresVals = gpuBackend.readSync(scores.dataId);
          const maxOutputSizeVal = maxOutputSize;
          const iouThresholdVal = iouThreshold;
          const scoreThresholdVal = scoreThreshold;
          const softNmsSigmaVal = softNmsSigma;
          const { selectedIndices, selectedScores } = nonMaxSuppressionV5(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal, softNmsSigmaVal);
          return [selectedIndices, selectedScores];
      }
  };

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
  const squareConfig = {
      kernelName: tf.Square,
      backendName: 'webgl',
      kernelFunc: ({ inputs, backend }) => {
          const { x } = inputs;
          const webglBackend = backend;
          const program = new UnaryOpProgram(x.shape, SQUARE);
          return webglBackend.runWebGLProgram(program, [x], x.dtype);
      }
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  const squaredDifferenceConfig = {
      kernelName: tf.SquaredDifference,
      backendName: 'webgl',
      kernelFunc: ({ inputs, backend }) => {
          const { a, b } = inputs;
          const SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';
          const webGLBackend = backend;
          const program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
              new BinaryOpPackedProgram(SQUARED_DIFFERENCE, a.shape, b.shape) :
              new BinaryOpProgram(SQUARED_DIFFERENCE, a.shape, b.shape);
          return webGLBackend.compileAndRun(program, [a, b]);
      }
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  function transposeImpl(xVals, xShape, dtype, perm, newShape) {
      const xRank = xShape.length;
      const xSize = tf.util.sizeFromShape(xShape);
      const xStrides = tf.util.computeStrides(xShape);
      const newStrides = tf.util.computeStrides(newShape);
      const result = tf.util.getTypedArrayFromDType(dtype, tf.util.sizeFromShape(newShape));
      for (let i = 0; i < xSize; ++i) {
          const loc = tf.util.indexToLoc(i, xRank, xStrides);
          // Permute location.
          const newLoc = new Array(loc.length);
          for (let i = 0; i < newLoc.length; i++) {
              newLoc[i] = loc[perm[i]];
          }
          const newIndex = tf.util.locToIndex(newLoc, xRank, newStrides);
          result[newIndex] = xVals[i];
      }
      return result;
  }

  /**
   * @license
   * Copyright 2017 Google Inc. All Rights Reserved.
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
  class TransposeProgram {
      constructor(aShape, newDim) {
          this.variableNames = ['A'];
          const outputShape = new Array(aShape.length);
          for (let i = 0; i < outputShape.length; i++) {
              outputShape[i] = aShape[newDim[i]];
          }
          this.outputShape = outputShape;
          this.rank = outputShape.length;
          const dtype = getCoordsDataType(this.rank);
          const switched = getSwitchedCoords(newDim);
          this.userCode = `
    void main() {
      ${dtype} resRC = getOutputCoords();
      setOutput(getA(${switched}));
    }
    `;
      }
  }
  function getSwitchedCoords(newDim) {
      const rank = newDim.length;
      if (rank > 6) {
          throw Error(`Transpose for rank ${rank} is not yet supported`);
      }
      const originalOrder = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w', 'resRC.u', 'resRC.v'];
      const switchedCoords = new Array(rank);
      for (let i = 0; i < newDim.length; i++) {
          switchedCoords[newDim[i]] = originalOrder[i];
      }
      return switchedCoords.join();
  }

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
  class TransposePackedProgram {
      constructor(aShape, newDim) {
          this.variableNames = ['A'];
          this.packedInputs = true;
          this.packedOutput = true;
          const outputShape = new Array(aShape.length);
          for (let i = 0; i < outputShape.length; i++) {
              outputShape[i] = aShape[newDim[i]];
          }
          this.outputShape = outputShape;
          this.rank = outputShape.length;
          if (this.rank > 6) {
              throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);
          }
          const dtype = getCoordsDataType(this.rank);
          const outputOrder = getVecChannels('rc', this.rank);
          const switchedOrder = new Array(this.rank);
          for (let i = 0; i < newDim.length; i++) {
              switchedOrder[newDim[i]] = outputOrder[i];
          }
          const innerDims = `vec2(${switchedOrder.slice(-2).join()})`;
          const nextColumn = `++${outputOrder[this.rank - 1]} < ${outputShape[this.rank - 1]}`;
          const getc = `getChannel(getA(${switchedOrder.join()}), ${innerDims})`;
          this.userCode = `
    void main() {
      ${dtype} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${getc};
      if(${nextColumn}) {
        result[1] = ${getc};
      }
      --${outputOrder[this.rank - 1]};
      if(++${outputOrder[this.rank - 2]} < ${outputShape[this.rank - 2]}) {
        result[2] = ${getc};
        if(${nextColumn}) {
          result[3] = ${getc};
        }
      }
      setOutput(result);
    }
    `;
      }
  }

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  function transposeImpl$1(x, perm, backend) {
      const program = tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
          new TransposePackedProgram(x.shape, perm) :
          new TransposeProgram(x.shape, perm);
      return backend.runWebGLProgram(program, [x], x.dtype);
  }

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  const transposeConfig = {
      kernelName: tf.Transpose,
      backendName: 'webgl',
      kernelFunc: ({ inputs, attrs, backend }) => {
          const { x } = inputs;
          const { perm } = attrs;
          const webglBackend = backend;
          const xRank = x.shape.length;
          const newShape = new Array(xRank);
          for (let i = 0; i < newShape.length; i++) {
              newShape[i] = x.shape[perm[i]];
          }
          let out;
          if (webglBackend.shouldExecuteOnCPU([x])) {
              const xTexData = webglBackend.texData.get(x.dataId);
              const values = xTexData.values;
              const outValues = transposeImpl(values, x.shape, x.dtype, perm, newShape);
              out = webglBackend.makeTensorInfo(newShape, x.dtype);
              const outData = webglBackend.texData.get(out.dataId);
              outData.values = outValues;
          }
          else {
              out = transposeImpl$1(x, perm, webglBackend);
          }
          return out;
      }
  };

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  // List all kernel configs here
  const kernelConfigs = [
      fromPixelsConfig, divConfig, nonMaxSuppressionV5Config, squareConfig,
      squaredDifferenceConfig, transposeConfig, maxPoolWithArgmaxConfig
  ];
  for (const kernelConfig of kernelConfigs) {
      tf.registerKernel(kernelConfig);
  }

  /**
   * @license
   * Copyright 2020 Google LLC. All Rights Reserved.
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
  if (tf.device_util.isBrowser()) {
      tf.registerBackend('webgl', () => new MathBackendWebGL(), 2 /* priority */);
  }

  exports.MathBackendWebGL = MathBackendWebGL;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-backend-webgl.es2017.js.map
