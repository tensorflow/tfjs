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
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core'), require('seedrandom')) :
    typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core', 'seedrandom'], factory) :
    (global = global || self, factory(global.tf = global.tf || {}, global.tf, global.seedrandom));
}(this, (function (exports, tf, seedrandom) { 'use strict';

    /*! *****************************************************************************
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the Apache License, Version 2.0 (the "License"); you may not use
    this file except in compliance with the License. You may obtain a copy of the
    License at http://www.apache.org/licenses/LICENSE-2.0

    THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
    WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
    MERCHANTABLITY OR NON-INFRINGEMENT.

    See the Apache Version 2.0 License for specific language governing permissions
    and limitations under the License.
    ***************************************************************************** */
    /* global Reflect, Promise */

    var extendStatics = function(d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };

    function __extends(d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    }

    function __awaiter(thisArg, _arguments, P, generator) {
        function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    }

    function __generator(thisArg, body) {
        var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
        return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
        function verb(n) { return function (v) { return step([n, v]); }; }
        function step(op) {
            if (f) throw new TypeError("Generator is already executing.");
            while (_) try {
                if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
                if (y = 0, t) op = [op[0] & 2, t.value];
                switch (op[0]) {
                    case 0: case 1: t = op; break;
                    case 4: _.label++; return { value: op[1], done: false };
                    case 5: _.label++; y = op[1]; op = [0]; continue;
                    case 7: op = _.ops.pop(); _.trys.pop(); continue;
                    default:
                        if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                        if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                        if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                        if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                        if (t[2]) _.ops.pop();
                        _.trys.pop(); continue;
                }
                op = body.call(thisArg, _);
            } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
            if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
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
    var contexts = {};
    var WEBGL_ATTRIBUTES = {
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
        var gl = contexts[webGLVersion];
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
        var canvas = createCanvas(webGLVersion);
        canvas.addEventListener('webglcontextlost', function (ev) {
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
     * Copyright 2017 Google LLC. All Rights Reserved.
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
        var size = tf.util.sizeFromShape(shape);
        var texelsNeeded = Math.ceil(size / 4);
        return tf.util.sizeToSquarishShape(texelsNeeded);
    }
    function getPackedMatrixTextureShapeWidthHeight(rows, columns) {
        return [
            Math.max(1, Math.ceil(columns / 2)), Math.max(1, Math.ceil(rows / 2))
        ];
    }
    function getPackedRGBAArraySizeFromMatrixShape(rows, columns) {
        var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
        return w * h * 4;
    }
    function getTextureConfig(
    // tslint:disable-next-line:no-any
    gl, textureHalfFloatExtension) {
        // tslint:disable-next-line:no-any
        var glany = gl;
        var internalFormatFloat;
        var internalFormatHalfFloat;
        var internalFormatPackedHalfFloat;
        var internalFormatPackedFloat;
        var textureFormatFloat;
        var downloadTextureFormat;
        var downloadUnpackNumChannels;
        var defaultNumChannels;
        var textureTypeHalfFloat;
        var textureTypeFloat;
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
            internalFormatFloat: internalFormatFloat,
            internalFormatHalfFloat: internalFormatHalfFloat,
            internalFormatPackedHalfFloat: internalFormatPackedHalfFloat,
            internalFormatPackedFloat: internalFormatPackedFloat,
            textureFormatFloat: textureFormatFloat,
            downloadTextureFormat: downloadTextureFormat,
            downloadUnpackNumChannels: downloadUnpackNumChannels,
            defaultNumChannels: defaultNumChannels,
            textureTypeHalfFloat: textureTypeHalfFloat,
            textureTypeFloat: textureTypeFloat
        };
    }

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
        var returnValue = func();
        if (debugMode) {
            checkWebGLError(gl);
        }
        return returnValue;
    }
    function checkWebGLError(gl) {
        var error = gl.getError();
        if (error !== gl.NO_ERROR) {
            throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
        }
    }
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    var MIN_FLOAT16 = 5.96e-8;
    var MAX_FLOAT16 = 65504;
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
                return "Unknown error code " + status;
        }
    }
    function getExtensionOrThrow(gl, debug, extensionName) {
        return throwIfNull(gl, debug, function () { return gl.getExtension(extensionName); }, 'Extension "' + extensionName + '" not supported on this browser.');
    }
    function createVertexShader(gl, debug, vertexShaderSource) {
        var vertexShader = throwIfNull(gl, debug, function () { return gl.createShader(gl.VERTEX_SHADER); }, 'Unable to create vertex WebGLShader.');
        callAndCheck(gl, debug, function () { return gl.shaderSource(vertexShader, vertexShaderSource); });
        callAndCheck(gl, debug, function () { return gl.compileShader(vertexShader); });
        if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
            console.log(gl.getShaderInfoLog(vertexShader));
            throw new Error('Failed to compile vertex shader.');
        }
        return vertexShader;
    }
    function createFragmentShader(gl, debug, fragmentShaderSource) {
        var fragmentShader = throwIfNull(gl, debug, function () { return gl.createShader(gl.FRAGMENT_SHADER); }, 'Unable to create fragment WebGLShader.');
        callAndCheck(gl, debug, function () { return gl.shaderSource(fragmentShader, fragmentShaderSource); });
        callAndCheck(gl, debug, function () { return gl.compileShader(fragmentShader); });
        if (gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS) === false) {
            logShaderSourceAndInfoLog(fragmentShaderSource, gl.getShaderInfoLog(fragmentShader));
            throw new Error('Failed to compile fragment shader.');
        }
        return fragmentShader;
    }
    var lineNumberRegex = /ERROR: [0-9]+:([0-9]+):/g;
    function logShaderSourceAndInfoLog(shaderSource, shaderInfoLog) {
        var lineNumberRegexResult = lineNumberRegex.exec(shaderInfoLog);
        if (lineNumberRegexResult == null) {
            console.log("Couldn't parse line number in error: " + shaderInfoLog);
            console.log(shaderSource);
            return;
        }
        var lineNumber = +lineNumberRegexResult[1];
        var shaderLines = shaderSource.split('\n');
        var pad = shaderLines.length.toString().length + 2;
        var linesWithLineNumbers = shaderLines.map(function (line, lineNumber) {
            return tf.util.rightPad((lineNumber + 1).toString(), pad) + line;
        });
        var maxLineLength = 0;
        for (var i = 0; i < linesWithLineNumbers.length; i++) {
            maxLineLength = Math.max(linesWithLineNumbers[i].length, maxLineLength);
        }
        var beforeErrorLines = linesWithLineNumbers.slice(0, lineNumber - 1);
        var errorLine = linesWithLineNumbers.slice(lineNumber - 1, lineNumber);
        var afterErrorLines = linesWithLineNumbers.slice(lineNumber);
        console.log(beforeErrorLines.join('\n'));
        console.log(shaderInfoLog.split('\n')[0]);
        console.log("%c " + tf.util.rightPad(errorLine[0], maxLineLength), 'border:1px solid red; background-color:#e3d2d2; color:#a61717');
        console.log(afterErrorLines.join('\n'));
    }
    function createProgram(gl, debug) {
        return throwIfNull(gl, debug, function () { return gl.createProgram(); }, 'Unable to create WebGLProgram.');
    }
    function linkProgram(gl, debug, program) {
        callAndCheck(gl, debug, function () { return gl.linkProgram(program); });
        if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
            console.log(gl.getProgramInfoLog(program));
            throw new Error('Failed to link vertex and fragment shaders.');
        }
    }
    function validateProgram(gl, debug, program) {
        callAndCheck(gl, debug, function () { return gl.validateProgram(program); });
        if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
            console.log(gl.getProgramInfoLog(program));
            throw new Error('Shader program validation failed.');
        }
    }
    function createStaticVertexBuffer(gl, debug, data) {
        var buffer = throwIfNull(gl, debug, function () { return gl.createBuffer(); }, 'Unable to create WebGLBuffer');
        callAndCheck(gl, debug, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, buffer); });
        callAndCheck(gl, debug, function () { return gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); });
        return buffer;
    }
    function createStaticIndexBuffer(gl, debug, data) {
        var buffer = throwIfNull(gl, debug, function () { return gl.createBuffer(); }, 'Unable to create WebGLBuffer');
        callAndCheck(gl, debug, function () { return gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer); });
        callAndCheck(gl, debug, function () { return gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW); });
        return buffer;
    }
    function getNumChannels() {
        if (tf.env().getNumber('WEBGL_VERSION') === 2) {
            return 1;
        }
        return 4;
    }
    function createTexture(gl, debug) {
        return throwIfNull(gl, debug, function () { return gl.createTexture(); }, 'Unable to create WebGLTexture.');
    }
    function validateTextureSize(width, height) {
        var maxTextureSize = tf.env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
        if ((width <= 0) || (height <= 0)) {
            var requested = "[" + width + "x" + height + "]";
            throw new Error('Requested texture size ' + requested + ' is invalid.');
        }
        if ((width > maxTextureSize) || (height > maxTextureSize)) {
            var requested = "[" + width + "x" + height + "]";
            var max = "[" + maxTextureSize + "x" + maxTextureSize + "]";
            throw new Error('Requested texture size ' + requested +
                ' greater than WebGL maximum on this browser / GPU ' + max + '.');
        }
    }
    function createFramebuffer(gl, debug) {
        return throwIfNull(gl, debug, function () { return gl.createFramebuffer(); }, 'Unable to create WebGLFramebuffer.');
    }
    function bindVertexBufferToProgramAttribute(gl, debug, program, attribute, buffer, arrayEntriesPerItem, itemStrideInBytes, itemOffsetInBytes) {
        var loc = gl.getAttribLocation(program, attribute);
        if (loc === -1) {
            // The GPU compiler decided to strip out this attribute because it's unused,
            // thus no need to bind.
            return false;
        }
        callAndCheck(gl, debug, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, buffer); });
        callAndCheck(gl, debug, function () { return gl.vertexAttribPointer(loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes, itemOffsetInBytes); });
        callAndCheck(gl, debug, function () { return gl.enableVertexAttribArray(loc); });
        return true;
    }
    function bindTextureUnit(gl, debug, texture, textureUnit) {
        validateTextureUnit(gl, textureUnit);
        callAndCheck(gl, debug, function () { return gl.activeTexture(gl.TEXTURE0 + textureUnit); });
        callAndCheck(gl, debug, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    }
    function unbindTextureUnit(gl, debug, textureUnit) {
        validateTextureUnit(gl, textureUnit);
        callAndCheck(gl, debug, function () { return gl.activeTexture(gl.TEXTURE0 + textureUnit); });
        callAndCheck(gl, debug, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    }
    function getProgramUniformLocationOrThrow(gl, debug, program, uniformName) {
        return throwIfNull(gl, debug, function () { return gl.getUniformLocation(program, uniformName); }, 'uniform "' + uniformName + '" not present in program.');
    }
    function getProgramUniformLocation(gl, program, uniformName) {
        return gl.getUniformLocation(program, uniformName);
    }
    function bindTextureToProgramUniformSampler(gl, debug, program, texture, uniformSamplerLocation, textureUnit) {
        callAndCheck(gl, debug, function () { return bindTextureUnit(gl, debug, texture, textureUnit); });
        callAndCheck(gl, debug, function () { return gl.uniform1i(uniformSamplerLocation, textureUnit); });
    }
    function bindCanvasToFramebuffer(gl, debug) {
        callAndCheck(gl, debug, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, null); });
        callAndCheck(gl, debug, function () { return gl.viewport(0, 0, gl.canvas.width, gl.canvas.height); });
        callAndCheck(gl, debug, function () { return gl.scissor(0, 0, gl.canvas.width, gl.canvas.height); });
    }
    function bindColorTextureToFramebuffer(gl, debug, texture, framebuffer) {
        callAndCheck(gl, debug, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); });
        callAndCheck(gl, debug, function () { return gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); });
    }
    function unbindColorTextureFromFramebuffer(gl, debug, framebuffer) {
        callAndCheck(gl, debug, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); });
        callAndCheck(gl, debug, function () { return gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0); });
    }
    function validateFramebuffer(gl) {
        var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
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
                return "unknown error " + status;
        }
    }
    function throwIfNull(gl, debug, returnTOrNull, failureMessage) {
        var tOrNull = callAndCheck(gl, debug, function () { return returnTOrNull(); });
        if (tOrNull == null) {
            throw new Error(failureMessage);
        }
        return tOrNull;
    }
    function validateTextureUnit(gl, textureUnit) {
        var maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
        var glTextureUnit = textureUnit + gl.TEXTURE0;
        if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
            var textureUnitRange = "[gl.TEXTURE0, gl.TEXTURE" + maxTextureUnit + "]";
            throw new Error("textureUnit must be in " + textureUnitRange + ".");
        }
    }
    function getBatchDim(shape, dimsToSkip) {
        if (dimsToSkip === void 0) { dimsToSkip = 2; }
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
        var shapeAs3D = [1, 1, 1];
        var isScalar = shape.length === 0 || (shape.length === 1 && shape[0] === 1);
        if (!isScalar) {
            shapeAs3D =
                [getBatchDim(shape)].concat(getRowsCols(shape));
        }
        return shapeAs3D;
    }
    function getTextureShapeFromLogicalShape(logShape, isPacked) {
        var _a;
        if (isPacked === void 0) { isPacked = false; }
        var maxTexSize = tf.env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
        if (isPacked) {
            maxTexSize = maxTexSize * 2;
            // This logic ensures we accurately count the number of packed texels needed
            // to accommodate the tensor. We can only pack values in the same texel if
            // they are from adjacent pairs of rows/cols within the same batch. So if a
            // tensor has 3 rows, we pretend it has 4 rows in order to account for the
            // fact that the texels containing the third row are half empty.
            logShape = logShape.map(function (d, i) { return i >= logShape.length - 2 ?
                tf.util.nearestLargerEven(logShape[i]) :
                logShape[i]; });
            // Packed texture height is at least 2 (the channel height of a single
            // texel).
            if (logShape.length === 1) {
                logShape = [2, logShape[0]];
            }
        }
        // If logical shape is 2, we don't squeeze, since we want to match physical.
        if (logShape.length !== 2) {
            var squeezeResult = tf.util.squeezeShape(logShape);
            logShape = squeezeResult.newShape;
        }
        var size = tf.util.sizeFromShape(logShape);
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
                var batchDim = getBatchDim(logShape);
                var rows = 2, cols = 2;
                if (logShape.length) {
                    _a = getRowsCols(logShape), rows = _a[0], cols = _a[1];
                }
                size = batchDim * (rows / 2) * (cols / 2);
                return tf.util.sizeToSquarishShape(size).map(function (d) { return d * 2; });
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
            var shape1Cols = shape1.slice(-1)[0];
            var shape2Cols = shape2.slice(-1)[0];
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
    var MAX_TEXTURE_SIZE;
    var MAX_TEXTURES_IN_SHADER;
    function getWebGLMaxTextureSize(webGLVersion) {
        if (MAX_TEXTURE_SIZE == null) {
            var gl = getWebGLContext(webGLVersion);
            MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
        }
        return MAX_TEXTURE_SIZE;
    }
    function resetMaxTextureSize() {
        MAX_TEXTURE_SIZE = null;
    }
    function resetMaxTexturesInShader() {
        MAX_TEXTURES_IN_SHADER = null;
    }
    function getMaxTexturesInShader(webGLVersion) {
        if (MAX_TEXTURES_IN_SHADER == null) {
            var gl = getWebGLContext(webGLVersion);
            MAX_TEXTURES_IN_SHADER = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
        }
        // We cap at 16 to avoid spurious runtime "memory exhausted" error.
        return Math.min(16, MAX_TEXTURES_IN_SHADER);
    }
    function getWebGLDisjointQueryTimerVersion(webGLVersion) {
        if (webGLVersion === 0) {
            return 0;
        }
        var queryTimerVersion;
        var gl = getWebGLContext(webGLVersion);
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
        var ext = gl.getExtension(extensionName);
        return ext != null;
    }
    function isWebGLVersionEnabled(webGLVersion) {
        try {
            var gl = getWebGLContext(webGLVersion);
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
        var gl = getWebGLContext(webGLVersion);
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
        var isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
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
        var gl = getWebGLContext(webGLVersion);
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
            var COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
            if (hasExtension(gl, COLOR_BUFFER_HALF_FLOAT)) {
                var textureHalfFloatExtension = gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
                return createHalfFloatTextureAndBindToFramebuffer(gl, textureHalfFloatExtension);
            }
            return false;
        }
        var isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
        return isFrameBufferComplete;
    }
    function createFloatTextureAndBindToFramebuffer(gl) {
        var texConfig = getTextureConfig(gl);
        var texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        var width = 1;
        var height = 1;
        gl.texImage2D(gl.TEXTURE_2D, 0, texConfig.internalFormatFloat, width, height, 0, texConfig.textureFormatFloat, texConfig.textureTypeFloat, null);
        var frameBuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        var isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteTexture(texture);
        gl.deleteFramebuffer(frameBuffer);
        return isFrameBufferComplete;
    }
    function createHalfFloatTextureAndBindToFramebuffer(
    // tslint:disable-next-line:no-any
    gl, textureHalfFloatExtension) {
        var texConfig = getTextureConfig(gl, textureHalfFloatExtension);
        var texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        var width = 1;
        var height = 1;
        gl.texImage2D(gl.TEXTURE_2D, 0, texConfig.internalFormatHalfFloat, width, height, 0, texConfig.textureFormatFloat, texConfig.textureTypeHalfFloat, null);
        var frameBuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        var isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
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
        var gl = getWebGLContext(webGLVersion);
        // tslint:disable-next-line:no-any
        var isEnabled = gl.fenceSync != null;
        return isEnabled;
    }

    var webgl_util = {
        __proto__: null,
        callAndCheck: callAndCheck,
        canBeRepresented: canBeRepresented,
        getWebGLErrorMessage: getWebGLErrorMessage,
        getExtensionOrThrow: getExtensionOrThrow,
        createVertexShader: createVertexShader,
        createFragmentShader: createFragmentShader,
        createProgram: createProgram,
        linkProgram: linkProgram,
        validateProgram: validateProgram,
        createStaticVertexBuffer: createStaticVertexBuffer,
        createStaticIndexBuffer: createStaticIndexBuffer,
        getNumChannels: getNumChannels,
        createTexture: createTexture,
        validateTextureSize: validateTextureSize,
        createFramebuffer: createFramebuffer,
        bindVertexBufferToProgramAttribute: bindVertexBufferToProgramAttribute,
        bindTextureUnit: bindTextureUnit,
        unbindTextureUnit: unbindTextureUnit,
        getProgramUniformLocationOrThrow: getProgramUniformLocationOrThrow,
        getProgramUniformLocation: getProgramUniformLocation,
        bindTextureToProgramUniformSampler: bindTextureToProgramUniformSampler,
        bindCanvasToFramebuffer: bindCanvasToFramebuffer,
        bindColorTextureToFramebuffer: bindColorTextureToFramebuffer,
        unbindColorTextureFromFramebuffer: unbindColorTextureFromFramebuffer,
        validateFramebuffer: validateFramebuffer,
        getFramebufferErrorMessage: getFramebufferErrorMessage,
        getBatchDim: getBatchDim,
        getRowsCols: getRowsCols,
        getShapeAs3D: getShapeAs3D,
        getTextureShapeFromLogicalShape: getTextureShapeFromLogicalShape,
        isReshapeFree: isReshapeFree,
        getWebGLMaxTextureSize: getWebGLMaxTextureSize,
        resetMaxTextureSize: resetMaxTextureSize,
        resetMaxTexturesInShader: resetMaxTexturesInShader,
        getMaxTexturesInShader: getMaxTexturesInShader,
        getWebGLDisjointQueryTimerVersion: getWebGLDisjointQueryTimerVersion,
        hasExtension: hasExtension,
        isWebGLVersionEnabled: isWebGLVersionEnabled,
        isCapableOfRenderingToFloatTexture: isCapableOfRenderingToFloatTexture,
        isDownloadFloatTextureEnabled: isDownloadFloatTextureEnabled,
        isWebGLFenceEnabled: isWebGLFenceEnabled
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
    var ENV = tf.env();
    /**
     * This file contains WebGL-specific flag registrations.
     */
    /**
     * True if WebGL is supported.
     */
    ENV.registerFlag('HAS_WEBGL', function () { return ENV.getNumber('WEBGL_VERSION') > 0; });
    /** 0: No WebGL, 1: WebGL 1.0, 2: WebGL 2.0. */
    ENV.registerFlag('WEBGL_VERSION', function () {
        if (isWebGLVersionEnabled(2)) {
            return 2;
        }
        else if (isWebGLVersionEnabled(1)) {
            return 1;
        }
        return 0;
    });
    ENV.registerFlag('WEBGL_BUFFER_SUPPORTED', function () { return ENV.get('WEBGL_VERSION') === 2; });
    /** Whether the WebGL backend will sometimes forward ops to the CPU. */
    ENV.registerFlag('WEBGL_CPU_FORWARD', function () { return true; });
    /** Whether the WebGL backend will always use f16 textures for rendering. */
    ENV.registerFlag('WEBGL_FORCE_F16_TEXTURES', function () { return false; });
    /** Whether to turn all packing related flags on. */
    ENV.registerFlag('WEBGL_PACK', function () { return ENV.getBool('HAS_WEBGL'); });
    /** Whether we will pack the batchnormalization op. */
    ENV.registerFlag('WEBGL_PACK_NORMALIZATION', function () { return ENV.getBool('WEBGL_PACK'); });
    /** Whether we will pack the clip op. */
    ENV.registerFlag('WEBGL_PACK_CLIP', function () { return ENV.getBool('WEBGL_PACK'); });
    /** Whether we will pack the depthwise conv op. */
    // TODO: https://github.com/tensorflow/tfjs/issues/1679
    ENV.registerFlag('WEBGL_PACK_DEPTHWISECONV', function () { return false; });
    /** Whether we will pack binary ops. */
    ENV.registerFlag('WEBGL_PACK_BINARY_OPERATIONS', function () { return ENV.getBool('WEBGL_PACK'); });
    /** Whether we will pack unary ops. */
    ENV.registerFlag('WEBGL_PACK_UNARY_OPERATIONS', function () { return ENV.getBool('WEBGL_PACK'); });
    /** Whether we will pack array ops. */
    ENV.registerFlag('WEBGL_PACK_ARRAY_OPERATIONS', function () { return ENV.getBool('WEBGL_PACK'); });
    /** Whether we will pack image ops. */
    ENV.registerFlag('WEBGL_PACK_IMAGE_OPERATIONS', function () { return ENV.getBool('WEBGL_PACK'); });
    /** Whether we will pack reduce ops. */
    ENV.registerFlag('WEBGL_PACK_REDUCE', function () { return ENV.getBool('WEBGL_PACK'); });
    /** Whether packed WebGL kernels lazily unpack their outputs. */
    ENV.registerFlag('WEBGL_LAZILY_UNPACK', function () { return ENV.getBool('WEBGL_PACK'); });
    /** Whether we will use the im2col algorithm to speed up convolutions. */
    ENV.registerFlag('WEBGL_CONV_IM2COL', function () { return ENV.getBool('WEBGL_PACK'); });
    /** The maximum texture dimension. */
    ENV.registerFlag('WEBGL_MAX_TEXTURE_SIZE', function () { return getWebGLMaxTextureSize(ENV.getNumber('WEBGL_VERSION')); });
    /** The maximum texture dimension. */
    ENV.registerFlag('WEBGL_MAX_TEXTURES_IN_SHADER', function () { return getMaxTexturesInShader(ENV.getNumber('WEBGL_VERSION')); });
    /**
     * The disjoint_query_timer extension version.
     * 0: disabled, 1: EXT_disjoint_timer_query, 2:
     * EXT_disjoint_timer_query_webgl2.
     * In Firefox with WebGL 2.0,
     * EXT_disjoint_timer_query_webgl2 is not available, so we must use the
     * WebGL 1.0 extension.
     */
    ENV.registerFlag('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', function () {
        var webGLVersion = ENV.getNumber('WEBGL_VERSION');
        if (webGLVersion === 0) {
            return 0;
        }
        return getWebGLDisjointQueryTimerVersion(webGLVersion);
    });
    /**
     * Whether the timer object from the disjoint_query_timer extension gives
     * timing information that is reliable.
     */
    ENV.registerFlag('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', function () { return ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0 &&
        !tf.device_util.isMobile(); });
    /**
     * Whether the device is physically capable of rendering to float32 textures.
     */
    ENV.registerFlag('WEBGL_RENDER_FLOAT32_CAPABLE', function () { return isCapableOfRenderingToFloatTexture(ENV.getNumber('WEBGL_VERSION')); });
    /**
     * Whether rendering to float32 textures is enabled. If disabled, renders to
     * float16 textures.
     */
    ENV.registerFlag('WEBGL_RENDER_FLOAT32_ENABLED', function () {
        return ENV.getBool('WEBGL_FORCE_F16_TEXTURES') ?
            false :
            ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE');
    });
    /**
     * Whether downloading float textures is enabled (16 or 32 bit). If disabled,
     * uses IEEE 754 encoding of the float32 values to 4 uint8 when downloading.
     */
    ENV.registerFlag('WEBGL_DOWNLOAD_FLOAT_ENABLED', function () { return isDownloadFloatTextureEnabled(ENV.getNumber('WEBGL_VERSION')); });
    /** Whether the fence API is available. */
    ENV.registerFlag('WEBGL_FENCE_API_ENABLED', function () { return isWebGLFenceEnabled(ENV.getNumber('WEBGL_VERSION')); });
    /**
     * Tensors with size <= than this will be uploaded as uniforms, not textures.
     */
    ENV.registerFlag('WEBGL_SIZE_UPLOAD_UNIFORM', function () {
        // Use uniform uploads only when 32bit floats are supported. In
        // 16bit
        // environments there are problems with comparing a 16bit texture value
        // with a 32bit uniform value.
        var useUniforms = ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED');
        return useUniforms ? 4 : 0;
    });

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
    var AddNProgram = /** @class */ (function () {
        function AddNProgram(outputShape, shapes) {
            this.outputShape = [];
            this.outputShape = outputShape;
            this.variableNames = shapes.map(function (_, i) { return "T" + i; });
            var snippets = [];
            // Get target elements from every input tensor.
            this.variableNames.forEach(function (variable) {
                snippets.push("float v" + variable + " = get" + variable + "AtOutCoords();");
            });
            // Calculate the sum of all elements.
            var operation = this.variableNames
                .map(function (variable) {
                return "v" + variable;
            })
                .join(' + ');
            this.userCode = "\n      void main() {\n        " + snippets.join('\n        ') + "\n\n        float result = " + operation + ";\n        setOutput(result);\n      }\n    ";
        }
        return AddNProgram;
    }());

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
    var AddNPackedProgram = /** @class */ (function () {
        function AddNPackedProgram(outputShape, shapes) {
            this.outputShape = [];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = outputShape;
            this.variableNames = shapes.map(function (_, i) { return "T" + i; });
            var snippets = [];
            // Get target elements from every input tensor.
            this.variableNames.forEach(function (variable) {
                snippets.push("vec4 v" + variable + " = get" + variable + "AtOutCoords();");
            });
            // Calculate the sum of all elements.
            var operation = this.variableNames
                .map(function (variable) {
                return "v" + variable;
            })
                .join(' + ');
            this.userCode = "\n      void main() {\n        " + snippets.join('\n        ') + "\n\n        vec4 result = " + operation + ";\n        setOutput(result);\n      }\n    ";
        }
        return AddNPackedProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var ArgMinMaxProgram = /** @class */ (function () {
        function ArgMinMaxProgram(reduceInfo, op, firstPass) {
            this.variableNames = ['A'];
            var windowSize = reduceInfo.windowSize;
            var batchSize = reduceInfo.batchSize;
            var inSize = reduceInfo.inSize;
            var outSize = Math.ceil(inSize / windowSize);
            if (!firstPass) {
                this.variableNames.push('bestIndicesA');
            }
            this.outputShape = [batchSize, outSize];
            var compOp = (op === 'max') ? '>' : '<';
            var indexSnippet = firstPass ?
                'inOffset + i;' :
                'round(getBestIndicesA(batch, inOffset + i));';
            this.userCode = "\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int batch = coords[0];\n        int outIdx = coords[1];\n        int inOffset = outIdx * " + windowSize + ";\n\n        int bestIndex = inOffset;\n        float bestValue = getA(batch, bestIndex);\n\n        for (int i = 0; i < " + windowSize + "; i++) {\n          int inIdx = " + indexSnippet + ";\n          float candidate = getA(batch, inIdx);\n          if (candidate " + compOp + " bestValue) {\n            bestValue = candidate;\n            bestIndex = inIdx;\n          }\n        }\n        setOutput(float(bestIndex));\n      }\n    ";
        }
        return ArgMinMaxProgram;
    }());

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
        return ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank).map(function (d) { return name + "." + d; });
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
        var coords = '';
        for (var i = 0; i < rank; i++) {
            coords += dims[i];
            if (i < rank - 1) {
                coords += ',';
            }
        }
        return coords;
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
    function getGlslDifferences() {
        var version;
        var attribute;
        var varyingVs;
        var varyingFs;
        var texture2D;
        var output;
        var defineOutput;
        var defineSpecialNaN;
        var defineSpecialInf;
        var defineRound;
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
            defineSpecialNaN = "\n      bool isnan_custom(float val) {\n        return (val > 0.0 || val < 0.0) ? false : val != 0.0;\n      }\n\n      bvec4 isnan_custom(vec4 val) {\n        return bvec4(isnan_custom(val.x),\n          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));\n      }\n\n      #define isnan(value) isnan_custom(value)\n    ";
            // In webgl 2 we do not need to specify a custom isinf so there is no
            // need for a special INFINITY constant.
            defineSpecialInf = "";
            defineRound = "\n      #define round(value) newRound(value)\n      int newRound(float value) {\n        return int(floor(value + 0.5));\n      }\n\n      ivec4 newRound(vec4 value) {\n        return ivec4(floor(value + vec4(0.5)));\n      }\n    ";
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
            defineSpecialNaN = "\n      #define isnan(value) isnan_custom(value)\n      bool isnan_custom(float val) {\n        return (val > 0. || val < 1. || val == 0.) ? false : true;\n      }\n      bvec4 isnan_custom(vec4 val) {\n        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));\n      }\n    ";
            defineSpecialInf = "\n      uniform float INFINITY;\n\n      bool isinf(float val) {\n        return abs(val) == INFINITY;\n      }\n      bvec4 isinf(vec4 val) {\n        return equal(abs(val), vec4(INFINITY));\n      }\n    ";
            defineRound = "\n      int round(float value) {\n        return int(floor(value + 0.5));\n      }\n\n      ivec4 round(vec4 value) {\n        return ivec4(floor(value + vec4(0.5)));\n      }\n    ";
        }
        return {
            version: version,
            attribute: attribute,
            varyingVs: varyingVs,
            varyingFs: varyingFs,
            texture2D: texture2D,
            output: output,
            defineOutput: defineOutput,
            defineSpecialNaN: defineSpecialNaN,
            defineSpecialInf: defineSpecialInf,
            defineRound: defineRound
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
    function getLogicalCoordinatesFromFlatIndex(coords, shape, index) {
        if (index === void 0) { index = 'index'; }
        var strides = tf.util.computeStrides(shape);
        return strides
            .map(function (stride, i) {
            var line1 = "int " + coords[i] + " = " + index + " / " + stride;
            var line2 = i === strides.length - 1 ?
                "int " + coords[i + 1] + " = " + index + " - " + coords[i] + " * " + stride :
                "index -= " + coords[i] + " * " + stride;
            return line1 + "; " + line2 + ";";
        })
            .join('');
    }
    /**
     * Produces GLSL that computes the flat index from 3D coordinates.
     */
    function getFlatIndexFrom3D(shape) {
        var strides = tf.util.computeStrides(shape).map(function (d) { return d.toString(); });
        return "\n  int getFlatIndex(ivec3 coords) {\n    return coords.x * " + strides[0] + " + coords.y * " + strides[1] + " + coords.z;\n  }\n";
    }
    var ENCODE_FLOAT_SNIPPET = "\n  const float FLOAT_MAX = 1.70141184e38;\n  const float FLOAT_MIN = 1.17549435e-38;\n\n  lowp vec4 encode_float(highp float v) {\n    if (isnan(v)) {\n      return vec4(255, 255, 255, 255);\n    }\n\n    highp float av = abs(v);\n\n    if(av < FLOAT_MIN) {\n      return vec4(0.0, 0.0, 0.0, 0.0);\n    } else if(v > FLOAT_MAX) {\n      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;\n    } else if(v < -FLOAT_MAX) {\n      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;\n    }\n\n    highp vec4 c = vec4(0,0,0,0);\n\n    highp float e = floor(log2(av));\n    highp float m = exp2(fract(log2(av))) - 1.0;\n\n    c[2] = floor(128.0 * m);\n    m -= c[2] / 128.0;\n    c[1] = floor(32768.0 * m);\n    m -= c[1] / 32768.0;\n    c[0] = floor(8388608.0 * m);\n\n    highp float ebias = e + 127.0;\n    c[3] = floor(ebias / 2.0);\n    ebias -= c[3] * 2.0;\n    c[2] += floor(ebias) * 128.0;\n\n    c[3] += 128.0 * step(0.0, -v);\n\n    return c / 255.0;\n  }\n";

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var getBroadcastDims = tf.backend_util.getBroadcastDims;
    function makeShader(inputsInfo, outputShape, userCode, usesPackedTextures) {
        var prefixSnippets = [];
        inputsInfo.forEach(function (x) {
            var size = tf.util.sizeFromShape(x.shapeInfo.logicalShape);
            // Snippet when we decided to upload the values as uniform.
            if (x.shapeInfo.isUniform) {
                prefixSnippets.push("uniform float " + x.name + (size > 1 ? "[" + size + "]" : '') + ";");
            }
            else {
                prefixSnippets.push("uniform sampler2D " + x.name + ";");
                prefixSnippets.push("uniform int offset" + x.name + ";");
            }
        });
        var inputPrefixSnippet = prefixSnippets.join('\n');
        var inputSamplingSnippet = inputsInfo
            .map(function (x) { return getInputSamplingSnippet(x, outputShape, usesPackedTextures); })
            .join('\n');
        var outTexShape = outputShape.texShape;
        var glsl = getGlslDifferences();
        var floatTextureSampleSnippet = getFloatTextureSampleSnippet(glsl);
        var outputSamplingSnippet;
        var floatTextureSetOutputSnippet;
        var shaderPrefix = getShaderPrefix(glsl);
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
        var source = [
            shaderPrefix, floatTextureSampleSnippet, floatTextureSetOutputSnippet,
            inputPrefixSnippet, outputSamplingSnippet, inputSamplingSnippet, userCode
        ].join('\n');
        return source;
    }
    function getSamplerFromInInfo(inInfo) {
        var shape = inInfo.shapeInfo.logicalShape;
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
                throw new Error(shape.length + "-D input sampling" +
                    " is not yet supported");
        }
    }
    function getPackedSamplerFromInInfo(inInfo) {
        var shape = inInfo.shapeInfo.logicalShape;
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
    function getInputSamplingSnippet(inInfo, outShapeInfo, usesPackedTextures) {
        if (usesPackedTextures === void 0) { usesPackedTextures = false; }
        var res = '';
        if (usesPackedTextures) {
            res += getPackedSamplerFromInInfo(inInfo);
        }
        else {
            res += getSamplerFromInInfo(inInfo);
        }
        var inShape = inInfo.shapeInfo.logicalShape;
        var outShape = outShapeInfo.logicalShape;
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
                throw new Error(outShape.length + "-D output sampling is not yet supported");
        }
    }
    function getFloatTextureSampleSnippet(glsl) {
        return "\n    float sampleTexture(sampler2D textureSampler, vec2 uv) {\n      return " + glsl.texture2D + "(textureSampler, uv).r;\n    }\n  ";
    }
    function getFloatTextureSetRSnippet(glsl) {
        return "\n    void setOutput(float val) {\n      " + glsl.output + " = vec4(val, 0, 0, 0);\n    }\n  ";
    }
    function getFloatTextureSetRGBASnippet(glsl) {
        return "\n    void setOutput(vec4 val) {\n      " + glsl.output + " = val;\n    }\n  ";
    }
    function getShaderPrefix(glsl) {
        var SHADER_PREFIX = glsl.version + "\n    precision highp float;\n    precision highp int;\n    precision highp sampler2D;\n    " + glsl.varyingFs + " vec2 resultUV;\n    " + glsl.defineOutput + "\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    struct ivec5\n    {\n      int x;\n      int y;\n      int z;\n      int w;\n      int u;\n    };\n\n    struct ivec6\n    {\n      int x;\n      int y;\n      int z;\n      int w;\n      int u;\n      int v;\n    };\n\n    uniform float NAN;\n    " + glsl.defineSpecialNaN + "\n    " + glsl.defineSpecialInf + "\n    " + glsl.defineRound + "\n\n    int imod(int x, int y) {\n      return x - y * (x / y);\n    }\n\n    int idiv(int a, int b, float sign) {\n      int res = a / b;\n      int mod = imod(a, b);\n      if (sign < 0. && mod != 0) {\n        res -= 1;\n      }\n      return res;\n    }\n\n    //Based on the work of Dave Hoskins\n    //https://www.shadertoy.com/view/4djSRW\n    #define HASHSCALE1 443.8975\n    float random(float seed){\n      vec2 p = resultUV * seed;\n      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);\n      p3 += dot(p3, p3.yzx + 19.19);\n      return fract((p3.x + p3.y) * p3.z);\n    }\n\n    " + SAMPLE_1D_SNIPPET + "\n    " + SAMPLE_2D_SNIPPET + "\n    " + SAMPLE_3D_SNIPPET + "\n  ";
        return SHADER_PREFIX;
    }
    var SAMPLE_1D_SNIPPET = "\nvec2 uvFromFlat(int texNumR, int texNumC, int index) {\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\nvec2 packedUVfrom1D(int texNumR, int texNumC, int index) {\n  int texelIndex = index / 2;\n  int texR = texelIndex / texNumC;\n  int texC = texelIndex - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
    var SAMPLE_2D_SNIPPET = "\nvec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,\n  int texNumC, int row, int col) {\n  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);\n  int texR = texelIndex / texNumC;\n  int texC = texelIndex - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
    var SAMPLE_3D_SNIPPET = "\nvec2 packedUVfrom3D(int texNumR, int texNumC,\n    int texelsInBatch, int texelsInLogicalRow, int b,\n    int row, int col) {\n  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
    var SHADER_PACKED_PREFIX = "\n  float getChannel(vec4 frag, vec2 innerDims) {\n    vec2 modCoord = mod(innerDims, 2.);\n    return modCoord.x == 0. ?\n      (modCoord.y == 0. ? frag.r : frag.g) :\n      (modCoord.y == 0. ? frag.b : frag.a);\n  }\n  float getChannel(vec4 frag, int dim) {\n    float modCoord = mod(float(dim), 2.);\n    return modCoord == 0. ? frag.r : frag.g;\n  }\n";
    function getOutputScalarCoords() {
        return "\n    int getOutputCoords() {\n      return 0;\n    }\n  ";
    }
    function getOutputPacked1DCoords(shape, texShape) {
        var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
        if (packedTexShape[0] === 1) {
            return "\n      int getOutputCoords() {\n        return 2 * int(resultUV.x * " + packedTexShape[1] + ".0);\n      }\n    ";
        }
        if (packedTexShape[1] === 1) {
            return "\n      int getOutputCoords() {\n        return 2 * int(resultUV.y * " + packedTexShape[0] + ".0);\n      }\n    ";
        }
        return "\n    int getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n      return 2 * (resTexRC.x * " + packedTexShape[1] + " + resTexRC.y);\n    }\n  ";
    }
    function getOutput1DCoords(shape, texShape) {
        if (texShape[0] === 1) {
            return "\n      int getOutputCoords() {\n        return int(resultUV.x * " + texShape[1] + ".0);\n      }\n    ";
        }
        if (texShape[1] === 1) {
            return "\n      int getOutputCoords() {\n        return int(resultUV.y * " + texShape[0] + ".0);\n      }\n    ";
        }
        return "\n    int getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + texShape[0] + ", " + texShape[1] + "));\n      return resTexRC.x * " + texShape[1] + " + resTexRC.y;\n    }\n  ";
    }
    function getOutputPacked3DCoords(shape, texShape) {
        var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
        var texelsInLogicalRow = Math.ceil(shape[2] / 2);
        var texelsInBatch = texelsInLogicalRow * Math.ceil(shape[1] / 2);
        return "\n    ivec3 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n      int index = resTexRC.x * " + packedTexShape[1] + " + resTexRC.y;\n\n      int b = index / " + texelsInBatch + ";\n      index -= b * " + texelsInBatch + ";\n\n      int r = 2 * (index / " + texelsInLogicalRow + ");\n      int c = imod(index, " + texelsInLogicalRow + ") * 2;\n\n      return ivec3(b, r, c);\n    }\n  ";
    }
    function getOutput3DCoords(shape, texShape) {
        var coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);
        return "\n    ivec3 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + texShape[0] + ", " + texShape[1] + "));\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n      " + coordsFromIndexSnippet + "\n      return ivec3(r, c, d);\n    }\n  ";
    }
    function getOutputPackedNDCoords(shape, texShape) {
        var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
        var texelsInLogicalRow = Math.ceil(shape[shape.length - 1] / 2);
        var texelsInBatch = texelsInLogicalRow * Math.ceil(shape[shape.length - 2] / 2);
        var texelsInBatchN = texelsInBatch;
        var batches = "";
        var coords = 'b, r, c';
        for (var b = 2; b < shape.length - 1; b++) {
            texelsInBatchN *= shape[shape.length - b - 1];
            batches = "\n      int b" + b + " = index / " + texelsInBatchN + ";\n      index -= b" + b + " * " + texelsInBatchN + ";\n    " + batches;
            coords = "b" + b + ", " + coords;
        }
        return "\n    ivec" + shape.length + " getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n      int index = resTexRC.x * " + packedTexShape[1] + " + resTexRC.y;\n\n      " + batches + "\n\n      int b = index / " + texelsInBatch + ";\n      index -= b * " + texelsInBatch + ";\n\n      int r = 2 * (index / " + texelsInLogicalRow + ");\n      int c = imod(index, " + texelsInLogicalRow + ") * 2;\n\n      return ivec" + shape.length + "(" + coords + ");\n    }\n  ";
    }
    function getOutput4DCoords(shape, texShape) {
        var coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2'], shape);
        return "\n    ivec4 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n        vec2(" + texShape[0] + ", " + texShape[1] + "));\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n      " + coordsFromIndexSnippet + "\n      return ivec4(r, c, d, d2);\n    }\n  ";
    }
    function getOutput5DCoords(shape, texShape) {
        var coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2', 'd3'], shape);
        return "\n    ivec5 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx * vec2(" + texShape[0] + ",\n                             " + texShape[1] + "));\n\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n\n      " + coordsFromIndexSnippet + "\n\n      ivec5 outShape = ivec5(r, c, d, d2, d3);\n      return outShape;\n    }\n  ";
    }
    function getOutput6DCoords(shape, texShape) {
        var coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2', 'd3', 'd4'], shape);
        return "\n    ivec6 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n        vec2(" + texShape[0] + ", " + texShape[1] + "));\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n\n      " + coordsFromIndexSnippet + "\n\n      ivec6 result = ivec6(r, c, d, d2, d3, d4);\n      return result;\n    }\n  ";
    }
    function getOutputPacked2DCoords(shape, texShape) {
        var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
        if (tf.util.arraysEqual(shape, texShape)) {
            return "\n      ivec2 getOutputCoords() {\n        return 2 * ivec2(resultUV.yx * vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n      }\n    ";
        }
        // texels needed to accommodate a logical row
        var texelsInLogicalRow = Math.ceil(shape[1] / 2);
        /**
         * getOutputCoords
         *
         * resTexRC: The rows and columns of the texels. If you move over one
         * texel to the right in the packed texture, you are moving over one column
         * (not two).
         *
         * index: The texel index
         */
        return "\n    ivec2 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n\n      int index = resTexRC.x * " + packedTexShape[1] + " + resTexRC.y;\n      int r = 2 * (index / " + texelsInLogicalRow + ");\n      int c = imod(index, " + texelsInLogicalRow + ") * 2;\n\n      return ivec2(r, c);\n    }\n  ";
    }
    function getOutput2DCoords(shape, texShape) {
        if (tf.util.arraysEqual(shape, texShape)) {
            return "\n      ivec2 getOutputCoords() {\n        return ivec2(resultUV.yx * vec2(" + texShape[0] + ", " + texShape[1] + "));\n      }\n    ";
        }
        if (shape[1] === 1) {
            return "\n      ivec2 getOutputCoords() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n                               vec2(" + texShape[0] + ", " + texShape[1] + "));\n        int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n        return ivec2(index, 0);\n      }\n    ";
        }
        if (shape[0] === 1) {
            return "\n      ivec2 getOutputCoords() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n                               vec2(" + texShape[0] + ", " + texShape[1] + "));\n        int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n        return ivec2(0, index);\n      }\n    ";
        }
        return "\n    ivec2 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + texShape[0] + ", " + texShape[1] + "));\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n      int r = index / " + shape[1] + ";\n      int c = index - r * " + shape[1] + ";\n      return ivec2(r, c);\n    }\n  ";
    }
    function getFlatOffsetUniformName(texName) {
        return "offset" + texName;
    }
    function getPackedSamplerScalar(inputInfo) {
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var glsl = getGlslDifferences();
        return "\n    vec4 " + funcName + "() {\n      return " + glsl.texture2D + "(" + texName + ", halfCR);\n    }\n  ";
    }
    function getSamplerScalar(inputInfo) {
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        if (inputInfo.shapeInfo.isUniform) {
            return "float " + funcName + "() {return " + texName + ";}";
        }
        var _a = inputInfo.shapeInfo.texShape, texNumR = _a[0], texNumC = _a[1];
        if (texNumR === 1 && texNumC === 1) {
            return "\n      float " + funcName + "() {\n        return sampleTexture(" + texName + ", halfCR);\n      }\n    ";
        }
        var _b = inputInfo.shapeInfo.texShape, tNumR = _b[0], tNumC = _b[1];
        var offset = getFlatOffsetUniformName(texName);
        return "\n    float " + funcName + "() {\n      vec2 uv = uvFromFlat(" + tNumR + ", " + tNumC + ", " + offset + ");\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    function getPackedSampler1D(inputInfo) {
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var texShape = inputInfo.shapeInfo.texShape;
        var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
        var glsl = getGlslDifferences();
        return "\n    vec4 " + funcName + "(int index) {\n      vec2 uv = packedUVfrom1D(\n        " + packedTexShape[0] + ", " + packedTexShape[1] + ", index);\n      return " + glsl.texture2D + "(" + texName + ", uv);\n    }\n  ";
    }
    function getSampler1D(inputInfo) {
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        if (inputInfo.shapeInfo.isUniform) {
            // Uniform arrays will be less than 65505 (no risk of float16 overflow).
            return "\n      float " + funcName + "(int index) {\n        " + getUniformSampler(inputInfo) + "\n      }\n    ";
        }
        var texShape = inputInfo.shapeInfo.texShape;
        var tNumR = texShape[0];
        var tNumC = texShape[1];
        if (tNumC === 1 && tNumR === 1) {
            return "\n      float " + funcName + "(int index) {\n        return sampleTexture(" + texName + ", halfCR);\n      }\n    ";
        }
        var offset = getFlatOffsetUniformName(texName);
        if (tNumC === 1) {
            return "\n      float " + funcName + "(int index) {\n        vec2 uv = vec2(0.5, (float(index + " + offset + ") + 0.5) / " + tNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
        }
        if (tNumR === 1) {
            return "\n      float " + funcName + "(int index) {\n        vec2 uv = vec2((float(index + " + offset + ") + 0.5) / " + tNumC + ".0, 0.5);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
        }
        return "\n    float " + funcName + "(int index) {\n      vec2 uv = uvFromFlat(" + tNumR + ", " + tNumC + ", index + " + offset + ");\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    function getPackedSampler2D(inputInfo) {
        var shape = inputInfo.shapeInfo.logicalShape;
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var texShape = inputInfo.shapeInfo.texShape;
        var texNumR = texShape[0];
        var texNumC = texShape[1];
        var glsl = getGlslDifferences();
        if (texShape != null && tf.util.arraysEqual(shape, texShape)) {
            return "\n      vec4 " + funcName + "(int row, int col) {\n        vec2 uv = (vec2(col, row) + halfCR) / vec2(" + texNumC + ".0, " + texNumR + ".0);\n\n        return " + glsl.texture2D + "(" + texName + ", uv);\n      }\n    ";
        }
        var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
        var valuesPerRow = Math.ceil(shape[1] / 2);
        return "\n    vec4 " + funcName + "(int row, int col) {\n      vec2 uv = packedUVfrom2D(" + valuesPerRow + ", " + packedTexShape[0] + ", " + packedTexShape[1] + ", row, col);\n      return " + glsl.texture2D + "(" + texName + ", uv);\n    }\n  ";
    }
    function getSampler2D(inputInfo) {
        var shape = inputInfo.shapeInfo.logicalShape;
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var texShape = inputInfo.shapeInfo.texShape;
        if (texShape != null && tf.util.arraysEqual(shape, texShape)) {
            var texNumR_1 = texShape[0];
            var texNumC_1 = texShape[1];
            return "\n    float " + funcName + "(int row, int col) {\n      vec2 uv = (vec2(col, row) + halfCR) / vec2(" + texNumC_1 + ".0, " + texNumR_1 + ".0);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
        }
        var _a = tf.util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
        var squeezedShape = newShape;
        if (squeezedShape.length < shape.length) {
            var newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
            var params = ['row', 'col'];
            return "\n      " + getSamplerFromInInfo(newInputInfo) + "\n      float " + funcName + "(int row, int col) {\n        return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n      }\n    ";
        }
        if (inputInfo.shapeInfo.isUniform) {
            // Uniform arrays will be less than 65505 (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col) {\n        int index = round(dot(vec2(row, col), vec2(" + shape[1] + ", 1)));\n        " + getUniformSampler(inputInfo) + "\n      }\n    ";
        }
        var texNumR = texShape[0];
        var texNumC = texShape[1];
        var offset = getFlatOffsetUniformName(texName);
        if (texNumC === 1) {
            // index is used directly as physical (no risk of float16 overflow).
            return "\n    float " + funcName + "(int row, int col) {\n      float index = dot(vec3(row, col, " + offset + "), vec3(" + shape[1] + ", 1, 1));\n      vec2 uv = vec2(0.5, (index + 0.5) / " + texNumR + ".0);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
        }
        if (texNumR === 1) {
            // index is used directly as physical (no risk of float16 overflow).
            return "\n    float " + funcName + "(int row, int col) {\n      float index = dot(vec3(row, col, " + offset + "), vec3(" + shape[1] + ", 1, 1));\n      vec2 uv = vec2((index + 0.5) / " + texNumC + ".0, 0.5);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
        }
        return "\n  float " + funcName + "(int row, int col) {\n    // Explicitly use integer operations as dot() only works on floats.\n    int index = row * " + shape[1] + " + col + " + offset + ";\n    vec2 uv = uvFromFlat(" + texNumR + ", " + texNumC + ", index);\n    return sampleTexture(" + texName + ", uv);\n  }\n";
    }
    function getPackedSampler3D(inputInfo) {
        var shape = inputInfo.shapeInfo.logicalShape;
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var texShape = inputInfo.shapeInfo.texShape;
        var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
        if (shape[0] === 1) {
            var squeezedShape = shape.slice(1);
            var keptDims = [1, 2];
            var newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
            var params = ['b', 'row', 'col'];
            return "\n        " + getPackedSamplerFromInInfo(newInputInfo) + "\n        vec4 " + funcName + "(int b, int row, int col) {\n          return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n        }\n      ";
        }
        var texNumR = packedTexShape[0];
        var texNumC = packedTexShape[1];
        var valuesPerRow = Math.ceil(shape[2] / 2);
        var texelsInBatch = valuesPerRow * Math.ceil(shape[1] / 2);
        var glsl = getGlslDifferences();
        return "\n    vec4 " + funcName + "(int b, int row, int col) {\n      vec2 uv = packedUVfrom3D(\n        " + texNumR + ", " + texNumC + ", " + texelsInBatch + ", " + valuesPerRow + ", b, row, col);\n      return " + glsl.texture2D + "(" + texName + ", uv);\n    }\n  ";
    }
    function getSampler3D(inputInfo) {
        var shape = inputInfo.shapeInfo.logicalShape;
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var stride0 = shape[1] * shape[2];
        var stride1 = shape[2];
        var _a = tf.util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
        var squeezedShape = newShape;
        if (squeezedShape.length < shape.length) {
            var newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
            var params = ['row', 'col', 'depth'];
            return "\n        " + getSamplerFromInInfo(newInputInfo) + "\n        float " + funcName + "(int row, int col, int depth) {\n          return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n        }\n      ";
        }
        if (inputInfo.shapeInfo.isUniform) {
            // Uniform arrays will be less than 65505 (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth) {\n        int index = round(dot(vec3(row, col, depth),\n                          vec3(" + stride0 + ", " + stride1 + ", 1)));\n        " + getUniformSampler(inputInfo) + "\n      }\n    ";
        }
        var texShape = inputInfo.shapeInfo.texShape;
        var texNumR = texShape[0];
        var texNumC = texShape[1];
        var flatOffset = inputInfo.shapeInfo.flatOffset;
        if (texNumC === stride0 && flatOffset == null) {
            // texC is used directly as physical (no risk of float16 overflow).
            return "\n        float " + funcName + "(int row, int col, int depth) {\n          float texR = float(row);\n          float texC = dot(vec2(col, depth), vec2(" + stride1 + ", 1));\n          vec2 uv = (vec2(texC, texR) + halfCR) /\n                     vec2(" + texNumC + ".0, " + texNumR + ".0);\n          return sampleTexture(" + texName + ", uv);\n        }\n      ";
        }
        if (texNumC === stride1 && flatOffset == null) {
            // texR is used directly as physical (no risk of float16 overflow).
            return "\n    float " + funcName + "(int row, int col, int depth) {\n      float texR = dot(vec2(row, col), vec2(" + shape[1] + ", 1));\n      float texC = float(depth);\n      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + texNumC + ".0, " + texNumR + ".0);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
        }
        var offset = getFlatOffsetUniformName(texName);
        return "\n      float " + funcName + "(int row, int col, int depth) {\n        // Explicitly use integer operations as dot() only works on floats.\n        int index = row * " + stride0 + " + col * " + stride1 + " + depth + " + offset + ";\n        vec2 uv = uvFromFlat(" + texNumR + ", " + texNumC + ", index);\n        return sampleTexture(" + texName + ", uv);\n      }\n  ";
    }
    function getPackedSamplerND(inputInfo) {
        var shape = inputInfo.shapeInfo.logicalShape;
        var rank = shape.length;
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var texShape = inputInfo.shapeInfo.texShape;
        var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
        var texNumR = packedTexShape[0];
        var texNumC = packedTexShape[1];
        var valuesPerRow = Math.ceil(shape[rank - 1] / 2);
        var texelsInBatch = valuesPerRow * Math.ceil(shape[rank - 2] / 2);
        var params = "int b, int row, int col";
        var index = "b * " + texelsInBatch + " + (row / 2) * " + valuesPerRow + " + (col / 2)";
        for (var b = 2; b < rank - 1; b++) {
            params = "int b" + b + ", " + params;
            texelsInBatch *= shape[rank - b - 1];
            index = "b" + b + " * " + texelsInBatch + " + " + index;
        }
        var glsl = getGlslDifferences();
        return "\n    vec4 " + funcName + "(" + params + ") {\n      int index = " + index + ";\n      int texR = index / " + texNumC + ";\n      int texC = index - texR * " + texNumC + ";\n      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + texNumC + ", " + texNumR + ");\n      return " + glsl.texture2D + "(" + texName + ", uv);\n    }\n  ";
    }
    function getSampler4D(inputInfo) {
        var shape = inputInfo.shapeInfo.logicalShape;
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var stride2 = shape[3];
        var stride1 = shape[2] * stride2;
        var stride0 = shape[1] * stride1;
        var _a = tf.util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
        if (newShape.length < shape.length) {
            var newInputInfo = squeezeInputInfo(inputInfo, newShape);
            var params = ['row', 'col', 'depth', 'depth2'];
            return "\n      " + getSamplerFromInInfo(newInputInfo) + "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n      }\n    ";
        }
        if (inputInfo.shapeInfo.isUniform) {
            // Uniform arrays will be less than 65505 (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        int index = round(dot(vec4(row, col, depth, depth2),\n                          vec4(" + stride0 + ", " + stride1 + ", " + stride2 + ", 1)));\n        " + getUniformSampler(inputInfo) + "\n      }\n    ";
        }
        var flatOffset = inputInfo.shapeInfo.flatOffset;
        var texShape = inputInfo.shapeInfo.texShape;
        var texNumR = texShape[0];
        var texNumC = texShape[1];
        if (texNumC === stride0 && flatOffset == null) {
            // texC is used directly as physical (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        float texR = float(row);\n        float texC =\n            dot(vec3(col, depth, depth2),\n                vec3(" + stride1 + ", " + stride2 + ", 1));\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
        }
        if (texNumC === stride2 && flatOffset == null) {
            // texR is used directly as physical (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        float texR = dot(vec3(row, col, depth),\n                         vec3(" + shape[1] * shape[2] + ", " + shape[2] + ", 1));\n        float texC = float(depth2);\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
        }
        var offset = getFlatOffsetUniformName(texName);
        return "\n    float " + funcName + "(int row, int col, int depth, int depth2) {\n      // Explicitly use integer operations as dot() only works on floats.\n      int index = row * " + stride0 + " + col * " + stride1 + " +\n          depth * " + stride2 + " + depth2;\n      vec2 uv = uvFromFlat(" + texNumR + ", " + texNumC + ", index + " + offset + ");\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    function getSampler5D(inputInfo) {
        var shape = inputInfo.shapeInfo.logicalShape;
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var stride3 = shape[4];
        var stride2 = shape[3] * stride3;
        var stride1 = shape[2] * stride2;
        var stride0 = shape[1] * stride1;
        var _a = tf.util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
        if (newShape.length < shape.length) {
            var newInputInfo = squeezeInputInfo(inputInfo, newShape);
            var params = ['row', 'col', 'depth', 'depth2', 'depth3'];
            return "\n      " + getSamplerFromInInfo(newInputInfo) + "\n      float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n        return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n      }\n    ";
        }
        if (inputInfo.shapeInfo.isUniform) {
            // Uniform arrays will be less than 65505 (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n        float index = dot(\n          vec4(row, col, depth, depth2),\n          vec4(" + stride0 + ", " + stride1 + ", " + stride2 + ", " + stride3 + ")) +\n          depth3;\n        " + getUniformSampler(inputInfo) + "\n      }\n    ";
        }
        var flatOffset = inputInfo.shapeInfo.flatOffset;
        var texShape = inputInfo.shapeInfo.texShape;
        var texNumR = texShape[0];
        var texNumC = texShape[1];
        if (texNumC === stride0 && flatOffset == null) {
            // texC is used directly as physical (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n        int texR = row;\n        float texC = dot(vec4(col, depth, depth2, depth3),\n                         vec4(" + stride1 + ", " + stride2 + ", " + stride3 + ", 1));\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
        }
        if (texNumC === stride3 && flatOffset == null) {
            // texR is used directly as physical (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n        float texR = dot(\n          vec4(row, col, depth, depth2),\n          vec4(" + shape[1] * shape[2] * shape[3] + ",\n               " + shape[2] * shape[3] + ", " + shape[3] + ", 1));\n        int texC = depth3;\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
        }
        var offset = getFlatOffsetUniformName(texName);
        return "\n    float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n      // Explicitly use integer operations as dot() only works on floats.\n      int index = row * " + stride0 + " + col * " + stride1 + " + depth * " + stride2 + " +\n          depth2 * " + stride3 + " + depth3 + " + offset + ";\n      vec2 uv = uvFromFlat(" + texNumR + ", " + texNumC + ", index);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    function getSampler6D(inputInfo) {
        var shape = inputInfo.shapeInfo.logicalShape;
        var texName = inputInfo.name;
        var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
        var _a = tf.util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
        if (newShape.length < shape.length) {
            var newInputInfo = squeezeInputInfo(inputInfo, newShape);
            var params = ['row', 'col', 'depth', 'depth2', 'depth3', 'depth4'];
            return "\n      " + getSamplerFromInInfo(newInputInfo) + "\n      float " + funcName + "(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n      }\n    ";
        }
        var stride4 = shape[5];
        var stride3 = shape[4] * stride4;
        var stride2 = shape[3] * stride3;
        var stride1 = shape[2] * stride2;
        var stride0 = shape[1] * stride1;
        if (inputInfo.shapeInfo.isUniform) {
            // Uniform arrays will be less than 65505 (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth,\n                  int depth2, int depth3, int depth4) {\n        int index = round(dot(\n          vec4(row, col, depth, depth2),\n          vec4(" + stride0 + ", " + stride1 + ", " + stride2 + ", " + stride3 + ")) +\n          dot(\n            vec2(depth3, depth4),\n            vec2(" + stride4 + ", 1)));\n        " + getUniformSampler(inputInfo) + "\n      }\n    ";
        }
        var flatOffset = inputInfo.shapeInfo.flatOffset;
        var texShape = inputInfo.shapeInfo.texShape;
        var texNumR = texShape[0];
        var texNumC = texShape[1];
        if (texNumC === stride0 && flatOffset == null) {
            // texC is used directly as physical (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        int texR = row;\n        float texC = dot(vec4(col, depth, depth2, depth3),\n          vec4(" + stride1 + ", " + stride2 + ", " + stride3 + ", " + stride4 + ")) +\n               float(depth4);\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
        }
        if (texNumC === stride4 && flatOffset == null) {
            // texR is used directly as physical (no risk of float16 overflow).
            return "\n      float " + funcName + "(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        float texR = dot(vec4(row, col, depth, depth2),\n          vec4(" + shape[1] * shape[2] * shape[3] * shape[4] + ",\n               " + shape[2] * shape[3] * shape[4] + ",\n               " + shape[3] * shape[4] + ",\n               " + shape[4] + ")) + float(depth3);\n        int texC = depth4;\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
        }
        var offset = getFlatOffsetUniformName(texName);
        return "\n    float " + funcName + "(int row, int col, int depth,\n                  int depth2, int depth3, int depth4) {\n      // Explicitly use integer operations as dot() only works on floats.\n      int index = row * " + stride0 + " + col * " + stride1 + " + depth * " + stride2 + " +\n          depth2 * " + stride3 + " + depth3 * " + stride4 + " + depth4 + " + offset + ";\n      vec2 uv = uvFromFlat(" + texNumR + ", " + texNumC + ", index);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    function getUniformSampler(inputInfo) {
        var texName = inputInfo.name;
        var inSize = tf.util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
        if (inSize < 2) {
            return "return " + texName + ";";
        }
        return "\n    for (int i = 0; i < " + inSize + "; i++) {\n      if (i == index) {\n        return " + texName + "[i];\n      }\n    }\n  ";
    }
    function getPackedSamplerAtOutputCoords(inputInfo, outShapeInfo) {
        var texName = inputInfo.name;
        var texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
        var funcName = 'get' + texFuncSnippet + 'AtOutCoords';
        var inRank = inputInfo.shapeInfo.logicalShape.length;
        var outRank = outShapeInfo.logicalShape.length;
        var broadcastDims = getBroadcastDims(inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
        var type = getCoordsDataType(outRank);
        var rankDiff = outRank - inRank;
        var coordsSnippet;
        var fields = ['x', 'y', 'z', 'w', 'u', 'v'];
        if (inRank === 0) {
            coordsSnippet = '';
        }
        else if (outRank < 2 && broadcastDims.length >= 1) {
            coordsSnippet = 'coords = 0;';
        }
        else {
            coordsSnippet =
                broadcastDims.map(function (d) { return "coords." + fields[d + rankDiff] + " = 0;"; })
                    .join('\n');
        }
        var unpackedCoordsSnippet = '';
        if (outRank < 2 && inRank > 0) {
            unpackedCoordsSnippet = 'coords';
        }
        else {
            unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
                .map(function (s, i) { return "coords." + fields[i + rankDiff]; })
                .join(', ');
        }
        var output = "return outputValue;";
        var inSize = tf.util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
        var isInputScalar = inSize === 1;
        var outSize = tf.util.sizeFromShape(outShapeInfo.logicalShape);
        var isOutputScalar = outSize === 1;
        if (inRank === 1 && !isInputScalar && !isOutputScalar) {
            output = "\n      return vec4(outputValue.xy, outputValue.xy);\n    ";
        }
        else if (isInputScalar && !isOutputScalar) {
            if (outRank === 1) {
                output = "\n        return vec4(outputValue.x, outputValue.x, 0., 0.);\n      ";
            }
            else {
                output = "\n        return vec4(outputValue.x);\n      ";
            }
        }
        else if (broadcastDims.length) {
            var rows = inRank - 2;
            var cols = inRank - 1;
            if (broadcastDims.indexOf(rows) > -1 && broadcastDims.indexOf(cols) > -1) {
                output = "return vec4(outputValue.x);";
            }
            else if (broadcastDims.indexOf(rows) > -1) {
                output = "return vec4(outputValue.x, outputValue.y, " +
                    "outputValue.x, outputValue.y);";
            }
            else if (broadcastDims.indexOf(cols) > -1) {
                output = "return vec4(outputValue.xx, outputValue.zz);";
            }
        }
        return "\n    vec4 " + funcName + "() {\n      " + type + " coords = getOutputCoords();\n      " + coordsSnippet + "\n      vec4 outputValue = get" + texFuncSnippet + "(" + unpackedCoordsSnippet + ");\n      " + output + "\n    }\n  ";
    }
    function getSamplerAtOutputCoords(inputInfo, outShapeInfo) {
        var texName = inputInfo.name;
        var texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
        var funcName = 'get' + texFuncSnippet + 'AtOutCoords';
        var outTexShape = outShapeInfo.texShape;
        var inTexShape = inputInfo.shapeInfo.texShape;
        var inRank = inputInfo.shapeInfo.logicalShape.length;
        var outRank = outShapeInfo.logicalShape.length;
        if (!inputInfo.shapeInfo.isUniform && inRank === outRank &&
            inputInfo.shapeInfo.flatOffset == null &&
            tf.util.arraysEqual(inTexShape, outTexShape)) {
            return "\n      float " + funcName + "() {\n        return sampleTexture(" + texName + ", resultUV);\n      }\n    ";
        }
        var type = getCoordsDataType(outRank);
        var broadcastDims = getBroadcastDims(inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
        var rankDiff = outRank - inRank;
        var coordsSnippet;
        var fields = ['x', 'y', 'z', 'w', 'u', 'v'];
        if (inRank === 0) {
            coordsSnippet = '';
        }
        else if (outRank < 2 && broadcastDims.length >= 1) {
            coordsSnippet = 'coords = 0;';
        }
        else {
            coordsSnippet =
                broadcastDims.map(function (d) { return "coords." + fields[d + rankDiff] + " = 0;"; })
                    .join('\n');
        }
        var unpackedCoordsSnippet = '';
        if (outRank < 2 && inRank > 0) {
            unpackedCoordsSnippet = 'coords';
        }
        else {
            unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
                .map(function (s, i) { return "coords." + fields[i + rankDiff]; })
                .join(', ');
        }
        return "\n    float " + funcName + "() {\n      " + type + " coords = getOutputCoords();\n      " + coordsSnippet + "\n      return get" + texFuncSnippet + "(" + unpackedCoordsSnippet + ");\n    }\n  ";
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
            throw Error("GPU for rank " + rank + " is not yet supported");
        }
    }
    /** Returns a new input info (a copy) that has a squeezed logical shape. */
    function squeezeInputInfo(inInfo, squeezedShape) {
        // Deep copy.
        var newInputInfo = JSON.parse(JSON.stringify(inInfo));
        newInputInfo.shapeInfo.logicalShape = squeezedShape;
        return newInputInfo;
    }
    function getSqueezedParams(params, keptDims) {
        return keptDims.map(function (d) { return params[d]; }).join(', ');
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
    var ArgMinMaxPackedProgram = /** @class */ (function () {
        function ArgMinMaxPackedProgram(shape, windowSize, op, firstPass) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = true;
            tf.util.assert(shape.length > 2, function () { return "Packed arg" + (op.charAt(0).toUpperCase() +
                op.slice(1)) + " supports only inputs with rank above 2."; });
            var inSize = shape[shape.length - 1];
            var outSize = Math.ceil(inSize / windowSize);
            this.outputShape = shape.slice(0, -1);
            if (outSize > 1) {
                this.outputShape.push(outSize);
            }
            if (!firstPass) {
                this.variableNames.push('bestIndicesA');
            }
            var outShape = this.outputShape;
            var rank = outShape.length;
            var dtype = getCoordsDataType(rank);
            var coords = getChannels('coords', rank);
            var sourceLocSetup;
            var sourceRank;
            if (outSize === 1) {
                sourceRank = rank + 1;
                var sourceLocDType = getCoordsDataType(sourceRank);
                sourceLocSetup = "\n        " + sourceLocDType + " sourceLocR = " + sourceLocDType + "(" + coords.join() + ", 0);\n        ++" + coords[rank - 1] + ";\n        " + sourceLocDType + " sourceLocG = " + sourceLocDType + "(" + coords.join() + ", 0);\n        ++" + coords[rank - 2] + ";\n        " + sourceLocDType + " sourceLocA = " + sourceLocDType + "(" + coords.join() + ", 0);\n        --" + coords[rank - 1] + ";\n        " + sourceLocDType + " sourceLocB = " + sourceLocDType + "(" + coords.join() + ", 0);\n        --" + coords[rank - 2] + ";";
            }
            else {
                sourceRank = rank;
                sourceLocSetup = "\n        " + dtype + " sourceLocR = coords;\n        ++" + coords[rank - 1] + ";\n        " + dtype + " sourceLocG = coords;\n        ++" + coords[rank - 2] + ";\n        " + dtype + " sourceLocA = coords;\n        --" + coords[rank - 1] + ";\n        " + dtype + " sourceLocB = coords;\n        --" + coords[rank - 2] + ";";
            }
            var channels = ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, sourceRank);
            var inChannel = '.' + channels[sourceRank - 1]; // e.g. ".b" for rank 3.
            var intChannels = channels.map(function (x) { return 'int ' + x; });
            var srcRCoords = getChannels('sourceLocR', sourceRank - 1).concat('inIdx.r');
            var srcGCoords = getChannels('sourceLocG', sourceRank - 1).concat('inIdx.g');
            var srcBCoords = getChannels('sourceLocB', sourceRank - 1).concat('inIdx.b');
            var srcACoords = getChannels('sourceLocA', sourceRank - 1).concat('inIdx.a');
            var compOp = (op === 'max') ? 'greaterThan' : 'lessThan';
            var fetchCandidateIdx = firstPass ? '' : "\n          inIdx = round(vec4(getBestIndicesAChannel(" + srcRCoords.join() + "),\n                             getBestIndicesAChannel(" + srcGCoords.join() + "),\n                             getBestIndicesAChannel(" + srcBCoords.join() + "),\n                             getBestIndicesAChannel(" + srcACoords.join() + ")));";
            var fetchValue = "vec4(\n            getAChannel(" + srcRCoords.join() + "),\n            hasNextCol ? getAChannel(" + srcGCoords.join() + ") : 0.,\n            hasNextRow ? getAChannel(" + srcBCoords.join() + ") : 0.,\n            hasNextRow && hasNextCol ? getAChannel(" + srcACoords.join() + ") : 0.)";
            var getBestIndicesAChannelSnippet = firstPass ? '' : "\n      float getBestIndicesAChannel(" + intChannels.join() + ") {\n        return getChannel(getBestIndicesA(" + channels.join() + "),\n                                          vec2(" + channels.slice(-2).join() + "));\n      }";
            this.userCode = "\n      float getAChannel(" + intChannels.join() + ") {\n        return getChannel(getA(" + channels.join() + "),\n                               vec2(" + channels.slice(-2).join() + "));\n      }\n      " + getBestIndicesAChannelSnippet + "\n      void main() {\n        " + dtype + " coords = getOutputCoords();\n        bool hasNextCol = " + coords[rank - 1] + " < " + (outShape[rank - 1] - 1) + ";\n        bool hasNextRow = " + coords[rank - 2] + " < " + (outShape[rank - 2] - 1) + ";\n        " + sourceLocSetup + "\n        ivec4 srcIdx = ivec4(sourceLocR" + inChannel + ", sourceLocG" + inChannel + ",\n          sourceLocB" + inChannel + ", sourceLocA" + inChannel + ") * " + windowSize + ";\n        ivec4 inIdx = srcIdx;\n        vec4 bestIndex = vec4(inIdx);\n        vec4 bestValue = " + fetchValue + ";\n\n        for (int i = 0; i < " + windowSize + "; i++) {\n          inIdx = srcIdx;\n          " + fetchCandidateIdx + "\n          vec4 candidate = " + fetchValue + ";\n          bvec4 nan = isnan(candidate);\n          bvec4 replace = bvec4(\n            vec4(" + compOp + "(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));\n\n          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,\n                           replace.y  ? candidate.y : bestValue.y,\n                           replace.z  ? candidate.z : bestValue.z,\n                           replace.w  ? candidate.w : bestValue.w);\n          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));\n          srcIdx++;\n        }\n        setOutput(bestIndex);\n      }\n    ";
        }
        return ArgMinMaxPackedProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var AvgPool2DBackpropProgram = /** @class */ (function () {
        function AvgPool2DBackpropProgram(convInfo) {
            this.variableNames = ['dy'];
            this.outputShape = convInfo.inShape;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var effectiveFilterHeight = convInfo.effectiveFilterHeight;
            var effectiveFilterWidth = convInfo.effectiveFilterWidth;
            var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
            var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
            var avgMultiplier = 1 / (filterHeight * filterWidth);
            this.userCode = "\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n      const float avgMultiplier = float(" + avgMultiplier + ");\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n\n        ivec2 dyRCCorner = coords.yz - pads;\n        int dyRCorner = dyRCCorner.x;\n        int dyCCorner = dyRCCorner.y;\n\n        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < " + effectiveFilterHeight + ";\n            wR += " + dilationHeight + ") {\n          float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n          if (dyR < 0.0 || dyR >= " + convInfo.outHeight + ".0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          for (int wC = 0; wC < " + effectiveFilterWidth + ";\n            wC+= " + dilationWidth + ") {\n            float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n            if (dyC < 0.0 || dyC >= " + convInfo.outWidth + ".0 ||\n                fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            float dyValue = getDy(b, idyR, idyC, d);\n\n            dotProd += dyValue * avgMultiplier;\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return AvgPool2DBackpropProgram;
    }());
    var AvgPool3DBackpropProgram = /** @class */ (function () {
        function AvgPool3DBackpropProgram(convInfo) {
            this.variableNames = ['dy'];
            this.outputShape = convInfo.inShape;
            var filterDepth = convInfo.filterDepth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var strideDepth = convInfo.strideDepth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationDepth = convInfo.dilationDepth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var effectiveFilterDepth = convInfo.effectiveFilterDepth;
            var effectiveFilterHeight = convInfo.effectiveFilterHeight;
            var effectiveFilterWidth = convInfo.effectiveFilterWidth;
            var padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
            var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
            var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
            var avgMultiplier = 1 / (filterDepth * filterHeight * filterWidth);
            this.userCode = "\n      const ivec3 pads = ivec3(" + padFront + ", " + padTop + ", " + padLeft + ");\n      const float avgMultiplier = float(" + avgMultiplier + ");\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int ch = coords.u;\n\n        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;\n        int dyDCorner = dyCorner.x;\n        int dyRCorner = dyCorner.y;\n        int dyCCorner = dyCorner.z;\n\n        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get\n        // dx(xD, xR, xC, ch).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n\n        for (int wD = 0; wD < " + effectiveFilterDepth + ";\n            wD += " + dilationDepth + ") {\n          float dyD = float(dyDCorner + wD) / " + strideDepth + ".0;\n\n          if (dyD < 0.0 || dyD >= " + convInfo.outDepth + ".0 || fract(dyD) > 0.0) {\n            continue;\n          }\n          int idyD = int(dyD);\n\n          for (int wR = 0; wR < " + effectiveFilterHeight + ";\n              wR += " + dilationHeight + ") {\n            float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n            if (dyR < 0.0 || dyR >= " + convInfo.outHeight + ".0 ||\n                fract(dyR) > 0.0) {\n              continue;\n            }\n            int idyR = int(dyR);\n\n            for (int wC = 0; wC < " + effectiveFilterWidth + ";\n                wC += " + dilationWidth + ") {\n              float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n              if (dyC < 0.0 || dyC >= " + convInfo.outWidth + ".0 ||\n                  fract(dyC) > 0.0) {\n                continue;\n              }\n              int idyC = int(dyC);\n\n              float dyValue = getDy(batch, idyD, idyR, idyC, ch);\n\n              dotProd += dyValue * avgMultiplier;\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return AvgPool3DBackpropProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var BatchNormProgram = /** @class */ (function () {
        function BatchNormProgram(xShape, meanShape, varianceShape, offsetShape, scaleShape, varianceEpsilon) {
            this.outputShape = [];
            this.variableNames = ['x', 'mean', 'variance'];
            tf.backend_util.assertAndGetBroadcastShape(xShape, meanShape);
            tf.backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
            var offsetSnippet = '0.0';
            if (offsetShape != null) {
                tf.backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
                this.variableNames.push('offset');
                offsetSnippet = 'getOffsetAtOutCoords()';
            }
            var scaleSnippet = '1.0';
            if (scaleShape != null) {
                tf.backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
                this.variableNames.push('scale');
                scaleSnippet = 'getScaleAtOutCoords()';
            }
            this.outputShape = xShape;
            this.userCode = "\n      void main() {\n        float x = getXAtOutCoords();\n        float mean = getMeanAtOutCoords();\n        float variance = getVarianceAtOutCoords();\n        float offset = " + offsetSnippet + ";\n        float scale = " + scaleSnippet + ";\n        float inv = scale * inversesqrt(variance + float(" + varianceEpsilon + "));\n        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));\n      }\n    ";
        }
        return BatchNormProgram;
    }());

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
    var BatchNormPackedProgram = /** @class */ (function () {
        function BatchNormPackedProgram(xShape, meanShape, varianceShape, offsetShape, scaleShape, varianceEpsilon) {
            this.packedInputs = true;
            this.packedOutput = true;
            this.variableNames = ['x', 'mean', 'variance'];
            tf.backend_util.assertAndGetBroadcastShape(xShape, meanShape);
            tf.backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
            var offsetSnippet = 'vec4(0.0)';
            if (offsetShape != null) {
                tf.backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
                this.variableNames.push('offset');
                offsetSnippet = 'getOffsetAtOutCoords()';
            }
            var scaleSnippet = 'vec4(1.0)';
            if (scaleShape != null) {
                tf.backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
                this.variableNames.push('scale');
                scaleSnippet = 'getScaleAtOutCoords()';
            }
            this.outputShape = xShape;
            this.userCode = "\n      void main() {\n        vec4 offset = " + offsetSnippet + ";\n        vec4 scale = " + scaleSnippet + ";\n\n        vec4 x = getXAtOutCoords();\n        vec4 mean = getMeanAtOutCoords();\n        vec4 variance = getVarianceAtOutCoords();\n\n        vec4 inv = scale * inversesqrt(variance + vec4(" + varianceEpsilon + "));\n\n        setOutput((x - mean) * inv + offset);\n      }\n    ";
        }
        return BatchNormPackedProgram;
    }());

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
    var COMPLEX_MULTIPLY = {
        REAL: 'return areal * breal - aimag * bimag;',
        IMAG: 'return areal * bimag + aimag * breal;'
    };
    var BinaryOpComplexProgram = /** @class */ (function () {
        function BinaryOpComplexProgram(op, aShape, bShape) {
            this.variableNames = ['AReal', 'AImag', 'BReal', 'BImag'];
            this.outputShape = tf.backend_util.assertAndGetBroadcastShape(aShape, bShape);
            this.userCode = "\n      float binaryOpComplex(\n          float areal, float aimag, float breal, float bimag) {\n        " + op + "\n      }\n\n      void main() {\n        float areal = getARealAtOutCoords();\n        float aimag = getAImagAtOutCoords();\n        float breal = getBRealAtOutCoords();\n        float bimag = getBImagAtOutCoords();\n        setOutput(binaryOpComplex(areal, aimag, breal, bimag));\n      }\n    ";
        }
        return BinaryOpComplexProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var CHECK_NAN_SNIPPET = "\n  if (isnan(a)) return a;\n  if (isnan(b)) return b;\n";
    var ADD = 'return a + b;';
    var SUB = 'return a - b;';
    var MUL = 'return a * b;';
    // Without the equality check div produces 0.9999 for a = b, which when
    // floored can cause errors.
    var DIV = "\nif (a == b) {\n  return 1.0;\n};\nreturn a / b;";
    // We use native integer division to deal with floating point imprecision. Since
    // we implement floor division and glsl implements truncated division, we
    // correct for this by subtracting 1 from result when the result is negative and
    // there is a remainder.
    var INT_DIV = "\n  float s = sign(a) * sign(b);\n  int ia = round(a);\n  int ib = round(b);\n  if (ib != 0) {\n    // Windows (D3D) wants guaranteed non-zero int division at compile-time.\n    return float(idiv(ia, ib, s));\n  } else {\n    return NAN;\n  }\n";
    var POW = "\nif(a < 0.0 && floor(b) < b){\n  return NAN;\n}\nif (b == 0.0) {\n  return 1.0;\n}\nreturn (round(mod(b, 2.0)) != 1) ?\n    pow(abs(a), b) : sign(a) * pow(abs(a), b);\n";
    var EQUAL = "return float(a == b);";
    var NOT_EQUAL = "return float(a != b);";
    var LESS = "return float(a < b);";
    var LESS_EQUAL = "return float(a <= b);";
    var GREATER = "return float(a > b);";
    var GREATER_EQUAL = "return float(a >= b);";
    var LOGICAL_AND = "return float(a >= 1.0 && b >= 1.0);";
    var LOGICAL_OR = "return float(a >= 1.0 || b >= 1.0);";
    var MAX = CHECK_NAN_SNIPPET + "\n  return max(a, b);\n";
    var MIN = CHECK_NAN_SNIPPET + "\n  return min(a, b);\n";
    var MOD = "if (b == 0.0) return NAN;\n  return mod(a, b);";
    var ATAN2 = CHECK_NAN_SNIPPET + "\n  return atan(a, b);\n";
    var ELU_DER = "return (b >= 1.0) ? a : a * (b + 1.0);";
    var PRELU = "return (a < 0.) ? b * a : a;";
    var BinaryOpProgram = /** @class */ (function () {
        function BinaryOpProgram(op, aShape, bShape) {
            this.variableNames = ['A', 'B'];
            this.outputShape = tf.backend_util.assertAndGetBroadcastShape(aShape, bShape);
            this.userCode = "\n      float binaryOperation(float a, float b) {\n        " + op + "\n      }\n\n      void main() {\n        float a = getAAtOutCoords();\n        float b = getBAtOutCoords();\n        setOutput(binaryOperation(a, b));\n      }\n    ";
        }
        return BinaryOpProgram;
    }());

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
    var CHECK_NAN_SNIPPET$1 = "\n  result.r = isNaN.r > 0. ? NAN : result.r;\n  result.g = isNaN.g > 0. ? NAN : result.g;\n  result.b = isNaN.b > 0. ? NAN : result.b;\n  result.a = isNaN.a > 0. ? NAN : result.a;\n";
    // We do the same as in ./binaryop_gpu, with vec4 and ivec4.
    // On Linux, the vectorized implementation produces NaNs when a and b are 0.
    var DIV$1 = "\n  // vec4 one = vec4(equal(a, b));\n  // return one + (vec4(1.0) - one) * a / b;\n  vec4 result = a / b;\n  if(a.x == b.x) {\n    result.x = 1.;\n  }\n  if(a.y == b.y) {\n    result.y = 1.;\n  }\n  if(a.z == b.z) {\n    result.z = 1.;\n  }\n  if(a.w == b.w) {\n    result.w = 1.;\n  }\n\n  return result;\n";
    var INT_DIV$1 = "\n  ivec4 ia = round(a);\n  ivec4 ib = round(b);\n  bvec4 cond = notEqual(ib, ivec4(0));\n  ivec4 result = ivec4(0);\n  vec4 s = sign(a) * sign(b);\n\n  // Windows (D3D) wants guaranteed non-zero int division at compile-time.\n  if (cond[0]) {\n    result[0] = idiv(ia[0], ib[0], s[0]);\n  }\n  if (cond[1]) {\n    result[1] = idiv(ia[1], ib[1], s[1]);\n  }\n  if (cond[2]) {\n    result[2] = idiv(ia[2], ib[2], s[2]);\n  }\n  if (cond[3]) {\n    result[3] = idiv(ia[3], ib[3], s[3]);\n  }\n  return vec4(result);\n";
    var POW$1 = "\n  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.\n  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));\n  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);\n  vec4 result = multiplier * pow(abs(a), b);\n\n  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS\n  bvec4 isExpZero = equal(b, vec4(0.0));\n  result.r = isExpZero.r ? 1.0 : result.r;\n  result.g = isExpZero.g ? 1.0 : result.g;\n  result.b = isExpZero.b ? 1.0 : result.b;\n  result.a = isExpZero.a ? 1.0 : result.a;\n\n  vec4 isNaN = vec4(lessThan(a, vec4(0.0))) * vec4(lessThan(floor(b), b));\n  " +
        CHECK_NAN_SNIPPET$1 + "\n  return result;\n";
    var PRELU$1 = "\n  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));\n  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);\n";
    var ELU_DER$1 = "\n  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));\n  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));\n";
    var ATAN2$1 = "\n  vec4 result = atan(a, b);\n  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));\n  " +
        CHECK_NAN_SNIPPET$1 + "\n  return result;\n";
    var EQUAL$1 = "\n  return vec4(equal(a, b));\n";
    var NOT_EQUAL$1 = "\n  return vec4(notEqual(a, b));\n";
    var LESS$1 = "\n  return vec4(lessThan(a, b));\n";
    var LESS_EQUAL$1 = "\n  return vec4(lessThanEqual(a, b));\n";
    var GREATER$1 = "\n  return vec4(greaterThan(a, b));\n";
    var GREATER_EQUAL$1 = "\n  return vec4(greaterThanEqual(a, b));\n";
    var LOGICAL_AND$1 = "\n  return vec4(\n    vec4(greaterThanEqual(a, vec4(1.0))) *\n    vec4(greaterThanEqual(b, vec4(1.0))));\n";
    var LOGICAL_OR$1 = "\n  return min(\n    vec4(greaterThanEqual(a, vec4(1.0))) +\n    vec4(greaterThanEqual(b, vec4(1.0))),\n    vec4(1.0));\n";
    var MAX$1 = "\n  vec4 result = vec4(max(a, b));\n  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));\n  " +
        CHECK_NAN_SNIPPET$1 + "\n  return result;\n";
    var MIN$1 = "\n  vec4 result = vec4(min(a, b));\n  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));\n  " +
        CHECK_NAN_SNIPPET$1 + "\n  return result;\n";
    var MOD$1 = "\n  vec4 result = mod(a, b);\n  vec4 isNaN = vec4(equal(b, vec4(0.0)));\n  " +
        CHECK_NAN_SNIPPET$1 + "\n  return result;\n";
    var BinaryOpPackedProgram = /** @class */ (function () {
        function BinaryOpPackedProgram(op, aShape, bShape, checkOutOfBounds) {
            if (checkOutOfBounds === void 0) { checkOutOfBounds = false; }
            this.variableNames = ['A', 'B'];
            this.supportsBroadcasting = true;
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = tf.backend_util.assertAndGetBroadcastShape(aShape, bShape);
            var rank = this.outputShape.length;
            var checkOutOfBoundsString = '';
            if (checkOutOfBounds) {
                if (rank === 0 || tf.util.sizeFromShape(this.outputShape) === 1) {
                    checkOutOfBoundsString = "\n          result.y = 0.;\n          result.z = 0.;\n          result.w = 0.;\n        ";
                }
                else {
                    var dtype = getCoordsDataType(rank);
                    checkOutOfBoundsString = "\n          " + dtype + " coords = getOutputCoords();\n        ";
                    if (rank === 1) {
                        checkOutOfBoundsString += "\n            result.y = (coords + 1) >= " + this.outputShape[0] + " ? 0. : result.y;\n            result.z = 0.;\n            result.w = 0.;\n          ";
                    }
                    else {
                        var channels = getChannels('coords', rank);
                        checkOutOfBoundsString += "\n            bool nextRowOutOfBounds =\n              (" + channels[rank - 2] + " + 1) >= " + this.outputShape[rank - 2] + ";\n            bool nextColOutOfBounds =\n              (" + channels[rank - 1] + " + 1) >= " + this.outputShape[rank - 1] + ";\n            result.y = nextColOutOfBounds ? 0. : result.y;\n            result.z = nextRowOutOfBounds ? 0. : result.z;\n            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;\n          ";
                    }
                }
            }
            this.userCode = "\n      vec4 binaryOperation(vec4 a, vec4 b) {\n        " + op + "\n      }\n\n      void main() {\n        vec4 a = getAAtOutCoords();\n        vec4 b = getBAtOutCoords();\n\n        vec4 result = binaryOperation(a, b);\n        " + checkOutOfBoundsString + "\n\n        setOutput(result);\n      }\n    ";
        }
        return BinaryOpPackedProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var ClipProgram = /** @class */ (function () {
        function ClipProgram(aShape) {
            this.variableNames = ['A'];
            this.outputShape = aShape;
            this.userCode = "\n      uniform float minVal;\n      uniform float maxVal;\n\n      void main() {\n        float value = getAAtOutCoords();\n        if (isnan(value)) {\n          setOutput(value);\n          return;\n        }\n\n        setOutput(clamp(value, minVal, maxVal));\n      }\n    ";
        }
        ClipProgram.prototype.getCustomSetupFunc = function (min, max) {
            var _this = this;
            return function (gpgpu, webGLProgram) {
                if (_this.minLoc == null) {
                    _this.minLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'minVal');
                    _this.maxLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'maxVal');
                }
                gpgpu.gl.uniform1f(_this.minLoc, min);
                gpgpu.gl.uniform1f(_this.maxLoc, max);
            };
        };
        return ClipProgram;
    }());

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
    var ClipPackedProgram = /** @class */ (function () {
        function ClipPackedProgram(aShape) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = aShape;
            this.userCode = "\n      uniform float minVal;\n      uniform float maxVal;\n\n      void main() {\n        vec4 value = getAAtOutCoords();\n\n        if (any(isnan(value))) {\n          setOutput(value);\n          return;\n        }\n\n        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));\n      }\n    ";
        }
        ClipPackedProgram.prototype.getCustomSetupFunc = function (min, max) {
            var _this = this;
            return function (gpgpu, webGLProgram) {
                if (_this.minLoc == null) {
                    _this.minLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'minVal');
                    _this.maxLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'maxVal');
                }
                gpgpu.gl.uniform1f(_this.minLoc, min);
                gpgpu.gl.uniform1f(_this.maxLoc, max);
            };
        };
        return ClipPackedProgram;
    }());

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
    var ComplexAbsProgram = /** @class */ (function () {
        function ComplexAbsProgram(shape) {
            this.variableNames = ['real', 'imag'];
            this.outputShape = shape;
            this.userCode = "\n      void main() {\n        float re = abs(getRealAtOutCoords());\n        float im = abs(getImagAtOutCoords());\n        float mx = max(re, im);\n\n        // sadly the length function in glsl is not underflow-safe\n        // (at least not on Intel GPUs). So the safe solution is\n        // to ensure underflow-safety in all cases.\n        setOutput(\n          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))\n        );\n      }\n    ";
        }
        return ComplexAbsProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var ConcatProgram = /** @class */ (function () {
        // Concats 2d tensors along axis=1. See comments in MathBackendWebGL.concat().
        function ConcatProgram(shapes) {
            this.outputShape = [];
            this.outputShape = tf.backend_util.computeOutShape(shapes, 1 /* axis */);
            this.variableNames = shapes.map(function (_, i) { return "T" + i; });
            var offsets = new Array(shapes.length - 1);
            offsets[0] = shapes[0][1];
            for (var i = 1; i < offsets.length; i++) {
                offsets[i] = offsets[i - 1] + shapes[i][1];
            }
            var snippets = ["if (yC < " + offsets[0] + ") setOutput(getT0(yR, yC));"];
            for (var i = 1; i < offsets.length; i++) {
                var shift = offsets[i - 1];
                snippets.push("else if (yC < " + offsets[i] + ") " +
                    ("setOutput(getT" + i + "(yR, yC-" + shift + "));"));
            }
            var lastIndex = offsets.length;
            var lastShift = offsets[offsets.length - 1];
            snippets.push("else setOutput(getT" + lastIndex + "(yR, yC-" + lastShift + "));");
            this.userCode = "\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int yR = coords.x;\n        int yC = coords.y;\n\n        " + snippets.join('\n        ') + "\n      }\n    ";
        }
        return ConcatProgram;
    }());

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
    var ConcatPackedProgram = /** @class */ (function () {
        function ConcatPackedProgram(shapes, axis) {
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = [];
            this.outputShape = tf.backend_util.computeOutShape(shapes, axis);
            var shape = this.outputShape;
            var rank = shape.length;
            var dtype = getCoordsDataType(rank);
            var coords = getChannels('coords', rank);
            var channels = ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank);
            this.variableNames = shapes.map(function (_, i) { return "T" + i; });
            var offsets = new Array(shapes.length - 1);
            offsets[0] = shapes[0][axis];
            for (var i = 1; i < offsets.length; i++) {
                offsets[i] = offsets[i - 1] + shapes[i][axis];
            }
            var channel = channels[axis];
            var lastChannels = channels.slice(-2);
            var allChannels = channels.join();
            var getValueSnippet = "if (" + channel + " < " + offsets[0] + ") {\n        return getChannel(\n            getT0(" + allChannels + "), vec2(" + lastChannels.join() + "));\n        }";
            for (var i = 1; i < offsets.length; i++) {
                var shift_1 = offsets[i - 1];
                // Note: the >= comparison below may seem unnecessary given the check
                // above but is needed to workaround branch execution issues on some
                // devices. It makes all the conditions exclusive without relying on
                // execution order.
                getValueSnippet += "\n        if (" + channel + " < " + offsets[i] + "  && " + channel + " >= " + offsets[i - 1] + ") {\n          return getChannel(\n            getT" + i + "(" + shiftedChannels(channels, channel, shift_1) + "),\n            vec2(" + shiftedChannels(lastChannels, channel, shift_1) + "));\n        }";
            }
            var lastIndex = offsets.length;
            var shift = offsets[offsets.length - 1];
            getValueSnippet += "\n        return getChannel(\n          getT" + lastIndex + "(" + shiftedChannels(channels, channel, shift) + "),\n          vec2(" + shiftedChannels(lastChannels, channel, shift) + "));";
            this.userCode = "\n      float getValue(" + channels.map(function (x) { return 'int ' + x; }) + ") {\n        " + getValueSnippet + "\n      }\n\n      void main() {\n        " + dtype + " coords = getOutputCoords();\n        vec4 result = vec4(getValue(" + coords + "), 0., 0., 0.);\n\n        " + coords[rank - 1] + " = " + coords[rank - 1] + " + 1;\n        if (" + coords[rank - 1] + " < " + shape[rank - 1] + ") {\n          result.g = getValue(" + coords + ");\n        }\n\n        " + coords[rank - 2] + " = " + coords[rank - 2] + " + 1;\n        if (" + coords[rank - 2] + " < " + shape[rank - 2] + ") {\n          result.a = getValue(" + coords + ");\n        }\n\n        " + coords[rank - 1] + " = " + coords[rank - 1] + " - 1;\n        if (" + coords[rank - 2] + " < " + shape[rank - 2] + " &&\n            " + coords[rank - 1] + " < " + shape[rank - 1] + ") {\n          result.b = getValue(" + coords + ");\n        }\n        setOutput(result);\n      }\n    ";
        }
        return ConcatPackedProgram;
    }());
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
        var channelIdx = channels.indexOf(channel);
        var res = channels.map(function (c, idx) {
            if (idx === channelIdx) {
                return c + " - " + shift;
            }
            else {
                return c;
            }
        });
        return res.join();
    }

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var Conv2DDerFilterProgram = /** @class */ (function () {
        function Conv2DDerFilterProgram(convInfo) {
            this.variableNames = ['x', 'dy'];
            this.outputShape = convInfo.filterShape;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            var isChannelsLast = convInfo.dataFormat === 'channelsLast';
            this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int wR = coords.x;\n        int wC = coords.y;\n        int d1 = coords.z;\n        int d2 = coords.w;\n\n        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n\n        for (int b = 0; b < " + convInfo.batchSize + "; b++) {\n          for (int yR = 0; yR < " + convInfo.outHeight + "; yR++) {\n            int xR = wR + yR * " + strideHeight + " - " + padTop + ";\n\n            if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n              continue;\n            }\n\n            for (int yC = 0; yC < " + convInfo.outWidth + "; yC++) {\n              int xC = wC + yC * " + strideWidth + " - " + padLeft + ";\n\n              if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n                continue;\n              }\n\n              if (" + isChannelsLast + ") {\n                float dyValue = getDy(b, yR, yC, d2);\n                float xValue = getX(b, xR, xC, d1);\n                dotProd += (xValue * dyValue);\n              } else {\n                float dyValue = getDy(b, d2, yR, yC);\n                float xValue = getX(b, d1, xR, xC);\n                dotProd += (xValue * dyValue);\n              }\n\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return Conv2DDerFilterProgram;
    }());
    var Conv2DDerInputProgram = /** @class */ (function () {
        function Conv2DDerInputProgram(convInfo) {
            this.variableNames = ['dy', 'W'];
            this.outputShape = convInfo.inShape;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var isChannelsLast = convInfo.dataFormat === 'channelsLast';
            var padTop = filterHeight - 1 - convInfo.padInfo.top;
            var padLeft = filterWidth - 1 - convInfo.padInfo.left;
            var rowDim = isChannelsLast ? 1 : 2;
            var colDim = isChannelsLast ? 2 : 3;
            var channelDim = isChannelsLast ? 3 : 1;
            this.userCode = "\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords[0];\n        int d1 = coords[" + channelDim + "];\n\n        ivec2 dyCorner = ivec2(coords[" + rowDim + "], coords[" + colDim + "]) - pads;\n        int dyRCorner = dyCorner.x;\n        int dyCCorner = dyCorner.y;\n\n        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < " + filterHeight + "; wR++) {\n          float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n          if (dyR < 0.0 || dyR >= " + convInfo.outHeight + ".0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          int wRPerm = " + filterHeight + " - 1 - wR;\n\n          for (int wC = 0; wC < " + filterWidth + "; wC++) {\n            float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n            if (dyC < 0.0 || dyC >= " + convInfo.outWidth + ".0 ||\n                fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            int wCPerm = " + filterWidth + " - 1 - wC;\n\n            for (int d2 = 0; d2 < " + convInfo.outChannels + "; d2++) {\n\n              if (" + isChannelsLast + ") {\n                float xValue = getDy(batch, idyR, idyC, d2);\n                float wValue = getW(wRPerm, wCPerm, d1, d2);\n                dotProd += xValue * wValue;\n              } else {\n                float xValue = getDy(batch, d2, idyR, idyC);\n                float wValue = getW(wRPerm, wCPerm, d1, d2);\n                dotProd += xValue * wValue;\n              }\n\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return Conv2DDerInputProgram;
    }());
    var Conv3DDerFilterProgram = /** @class */ (function () {
        function Conv3DDerFilterProgram(convInfo) {
            this.variableNames = ['x', 'dy'];
            this.outputShape = convInfo.filterShape;
            var strideDepth = convInfo.strideDepth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var padFront = convInfo.padInfo.front;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            this.userCode = "\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int wF = coords.x;\n        int wR = coords.y;\n        int wC = coords.z;\n        int d1 = coords.w;\n        int d2 = coords.u;\n\n        float dotProd = 0.0;\n\n        for (int b = 0; b < " + convInfo.batchSize + "; b++) {\n          for (int yF = 0; yF < " + convInfo.outDepth + "; yF++) {\n            int xF = wF + yF * " + strideDepth + " - " + padFront + ";\n\n            if (xF < 0 || xF >= " + convInfo.inDepth + ") {\n              continue;\n            }\n\n            for (int yR = 0; yR < " + convInfo.outHeight + "; yR++) {\n              int xR = wR + yR * " + strideHeight + " - " + padTop + ";\n\n              if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n                continue;\n              }\n\n              for (int yC = 0; yC < " + convInfo.outWidth + "; yC++) {\n                int xC = wC + yC * " + strideWidth + " - " + padLeft + ";\n\n                if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n                  continue;\n                }\n\n                float dyValue = getDy(b, yF, yR, yC, d2);\n                float xValue = getX(b, xF, xR, xC, d1);\n                dotProd += (xValue * dyValue);\n              }\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return Conv3DDerFilterProgram;
    }());
    var Conv3DDerInputProgram = /** @class */ (function () {
        function Conv3DDerInputProgram(convInfo) {
            this.variableNames = ['dy', 'W'];
            this.outputShape = convInfo.inShape;
            var filterDepth = convInfo.filterDepth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var strideDepth = convInfo.strideDepth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var padFront = filterDepth - 1 - convInfo.padInfo.front;
            var padTop = filterHeight - 1 - convInfo.padInfo.top;
            var padLeft = filterWidth - 1 - convInfo.padInfo.left;
            this.userCode = "\n      const ivec3 pads = ivec3(" + padFront + ", " + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int d1 = coords.u;\n\n\n        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;\n        int dyFCorner = dyCorner.x;\n        int dyRCorner = dyCorner.y;\n        int dyCCorner = dyCorner.z;\n\n        float dotProd = 0.0;\n        for (int wF = 0; wF < " + filterDepth + "; wF++) {\n          float dyF = float(dyFCorner + wF) / " + strideDepth + ".0;\n\n          if (dyF < 0.0 || dyF >= " + convInfo.outDepth + ".0 || fract(dyF) > 0.0) {\n            continue;\n          }\n          int idyF = int(dyF);\n\n          int wFPerm = " + filterDepth + " - 1 - wF;\n\n          for (int wR = 0; wR < " + filterHeight + "; wR++) {\n            float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n            if (dyR < 0.0 || dyR >= " + convInfo.outHeight + ".0 ||\n              fract(dyR) > 0.0) {\n              continue;\n            }\n            int idyR = int(dyR);\n\n            int wRPerm = " + filterHeight + " - 1 - wR;\n\n            for (int wC = 0; wC < " + filterWidth + "; wC++) {\n              float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n              if (dyC < 0.0 || dyC >= " + convInfo.outWidth + ".0 ||\n                  fract(dyC) > 0.0) {\n                continue;\n              }\n              int idyC = int(dyC);\n\n              int wCPerm = " + filterWidth + " - 1 - wC;\n\n              for (int d2 = 0; d2 < " + convInfo.outChannels + "; d2++) {\n                float xValue = getDy(batch, idyF, idyR, idyC, d2);\n                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);\n                dotProd += xValue * wValue;\n              }\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return Conv3DDerInputProgram;
    }());

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
    var DepthwiseConv2DDerFilterProgram = /** @class */ (function () {
        function DepthwiseConv2DDerFilterProgram(convInfo) {
            this.variableNames = ['x', 'dy'];
            this.outputShape = convInfo.filterShape;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            var channelMul = convInfo.outChannels / convInfo.inChannels;
            this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int wR = coords.x;\n        int wC = coords.y;\n        int d1 = coords.z;\n        int dm = coords.w;\n        int d2 = d1 * " + channelMul + " + dm;\n\n        float dotProd = 0.0;\n\n        // TO DO: Vec4 over the batch size\n        for (int b = 0; b < " + convInfo.batchSize + "; b++) {\n          for (int yR = 0; yR < " + convInfo.outHeight + "; yR++) {\n            int xR = wR + yR * " + strideHeight + " - " + padTop + ";\n\n            if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n              continue;\n            }\n\n            for (int yC = 0; yC < " + convInfo.outWidth + "; yC++) {\n              int xC = wC + yC * " + strideWidth + " - " + padLeft + ";\n\n              if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n                continue;\n              }\n\n              float dyValue = getDy(b, yR, yC, d2);\n              float xValue = getX(b, xR, xC, d1);\n              dotProd += (xValue * dyValue);\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return DepthwiseConv2DDerFilterProgram;
    }());
    var DepthwiseConv2DDerInputProgram = /** @class */ (function () {
        function DepthwiseConv2DDerInputProgram(convInfo) {
            this.variableNames = ['dy', 'W'];
            this.outputShape = convInfo.inShape;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var padTop = filterHeight - 1 - convInfo.padInfo.top;
            var padLeft = filterWidth - 1 - convInfo.padInfo.left;
            var channelMul = convInfo.outChannels / convInfo.inChannels;
            this.userCode = "\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords[0];\n        int d1 = coords[3];\n        ivec2 dyCorner = coords.yz - pads;\n        int dyRCorner = dyCorner.x;\n        int dyCCorner = dyCorner.y;\n\n        float dotProd = 0.0;\n\n        for (int wR = 0; wR < " + filterHeight + "; wR++) {\n          float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n          if (dyR < 0.0 || dyR >= " + convInfo.outHeight + ".0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          int wRPerm = " + filterHeight + " - 1 - wR;\n\n          for (int wC = 0; wC < " + filterWidth + "; wC++) {\n            float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n            if (dyC < 0.0 || dyC >= " + convInfo.outWidth + ".0 ||\n                fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            int wCPerm = " + filterWidth + " - 1 - wC;\n\n            // TO DO: Vec4 over the channelMul\n            for (int dm = 0; dm < " + channelMul + "; dm++) {\n              int d2 = d1 * " + channelMul + " + dm;\n              float xValue = getDy(batch, idyR, idyC, d2);\n              float wValue = getW(wRPerm, wCPerm, d1, dm);\n              dotProd += xValue * wValue;\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return DepthwiseConv2DDerInputProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var Conv2DProgram = /** @class */ (function () {
        function Conv2DProgram(convInfo, addBias, activation, hasPreluActivationWeights) {
            if (addBias === void 0) { addBias = false; }
            if (activation === void 0) { activation = null; }
            if (hasPreluActivationWeights === void 0) { hasPreluActivationWeights = false; }
            this.variableNames = ['x', 'W'];
            this.outputShape = convInfo.outShape;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var inputDepthNearestVec4 = Math.floor(convInfo.inChannels / 4) * 4;
            var inputDepthVec4Remainder = convInfo.inChannels % 4;
            var isChannelsLast = convInfo.dataFormat === 'channelsLast';
            var rowDim = isChannelsLast ? 1 : 2;
            var colDim = isChannelsLast ? 2 : 3;
            var channelDim = isChannelsLast ? 3 : 1;
            var activationSnippet = '', applyActivationSnippet = '';
            if (activation) {
                if (hasPreluActivationWeights) {
                    activationSnippet = "float activation(float a) {\n          float b = getPreluActivationWeightsAtOutCoords();\n          " + activation + "\n        }";
                }
                else {
                    activationSnippet = "\n          float activation(float x) {\n            " + activation + "\n          }\n        ";
                }
                applyActivationSnippet = "result = activation(result);";
            }
            var addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
            if (addBias) {
                this.variableNames.push('bias');
            }
            if (hasPreluActivationWeights) {
                this.variableNames.push('preluActivationWeights');
            }
            this.userCode = "\n      " + activationSnippet + "\n\n      const ivec2 strides = ivec2(" + strideHeight + ", " + strideWidth + ");\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords[0];\n        int d2 = coords[" + channelDim + "];\n\n        ivec2 xRCCorner =\n            ivec2(coords[" + rowDim + "], coords[" + colDim + "]) * strides - pads;\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < " + filterHeight + "; wR++) {\n          int xR = xRCorner + wR * " + dilationHeight + ";\n\n          if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n            continue;\n          }\n\n          for (int wC = 0; wC < " + filterWidth + "; wC++) {\n            int xC = xCCorner + wC * " + dilationWidth + ";\n\n            if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n              continue;\n            }\n\n            for (int d1 = 0; d1 < " + inputDepthNearestVec4 + "; d1 += 4) {\n              vec4 wValues = vec4(\n                getW(wR, wC, d1, d2),\n                getW(wR, wC, d1 + 1, d2),\n                getW(wR, wC, d1 + 2, d2),\n                getW(wR, wC, d1 + 3, d2)\n              );\n\n              if (" + isChannelsLast + ") {\n                vec4 xValues = vec4(\n                  getX(batch, xR, xC, d1),\n                  getX(batch, xR, xC, d1 + 1),\n                  getX(batch, xR, xC, d1 + 2),\n                  getX(batch, xR, xC, d1 + 3)\n                );\n                dotProd += dot(xValues, wValues);\n              } else {\n                vec4 xValues = vec4(\n                  getX(batch, d1, xR, xC),\n                  getX(batch, d1 + 1, xR, xC),\n                  getX(batch, d1 + 2, xR, xC),\n                  getX(batch, d1 + 3, xR, xC)\n                );\n                dotProd += dot(xValues, wValues);\n              }\n            }\n\n            if (" + (inputDepthVec4Remainder === 1) + ") {\n\n              if (" + isChannelsLast + ") {\n                dotProd +=\n                    getX(batch, xR, xC, " + inputDepthNearestVec4 + ") *\n                    getW(wR, wC, " + inputDepthNearestVec4 + ", d2);\n              } else {\n                dotProd +=\n                    getX(batch, " + inputDepthNearestVec4 + ", xR, xC) *\n                    getW(wR, wC, " + inputDepthNearestVec4 + ", d2);\n              }\n\n            } else if (" + (inputDepthVec4Remainder === 2) + ") {\n              vec2 wValues = vec2(\n                getW(wR, wC, " + inputDepthNearestVec4 + ", d2),\n                getW(wR, wC, " + inputDepthNearestVec4 + " + 1, d2)\n              );\n\n              if (" + isChannelsLast + ") {\n                vec2 xValues = vec2(\n                  getX(batch, xR, xC, " + inputDepthNearestVec4 + "),\n                  getX(batch, xR, xC, " + inputDepthNearestVec4 + " + 1)\n                );\n                dotProd += dot(xValues, wValues);\n              } else {\n                vec2 xValues = vec2(\n                  getX(batch, " + inputDepthNearestVec4 + ", xR, xC),\n                  getX(batch, " + inputDepthNearestVec4 + " + 1, xR, xC)\n                );\n                dotProd += dot(xValues, wValues);\n              }\n\n            } else if (" + (inputDepthVec4Remainder === 3) + ") {\n              vec3 wValues = vec3(\n                getW(wR, wC, " + inputDepthNearestVec4 + ", d2),\n                getW(wR, wC, " + inputDepthNearestVec4 + " + 1, d2),\n                getW(wR, wC, " + inputDepthNearestVec4 + " + 2, d2)\n              );\n\n              if (" + isChannelsLast + ") {\n                vec3 xValues = vec3(\n                  getX(batch, xR, xC, " + inputDepthNearestVec4 + "),\n                  getX(batch, xR, xC, " + inputDepthNearestVec4 + " + 1),\n                  getX(batch, xR, xC, " + inputDepthNearestVec4 + " + 2)\n                );\n                dotProd += dot(xValues, wValues);\n              } else {\n                vec3 xValues = vec3(\n                  getX(batch, " + inputDepthNearestVec4 + ", xR, xC),\n                  getX(batch, " + inputDepthNearestVec4 + " + 1, xR, xC),\n                  getX(batch, " + inputDepthNearestVec4 + " + 2, xR, xC)\n                );\n                dotProd += dot(xValues, wValues);\n              }\n\n            }\n          }\n        }\n\n        float result = dotProd;\n        " + addBiasSnippet + "\n        " + applyActivationSnippet + "\n        setOutput(result);\n      }\n    ";
        }
        return Conv2DProgram;
    }());
    var Conv3DProgram = /** @class */ (function () {
        function Conv3DProgram(convInfo) {
            this.variableNames = ['x', 'W'];
            this.outputShape = convInfo.outShape;
            var padFront = convInfo.padInfo.front;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            var strideDepth = convInfo.strideDepth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationDepth = convInfo.dilationDepth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var filterDepth = convInfo.filterDepth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var inputDepthNearestVec4 = Math.floor(convInfo.inChannels / 4) * 4;
            var inputDepthVec4Remainder = convInfo.inChannels % 4;
            this.userCode = "\n      const ivec3 strides = ivec3(" + strideDepth + ", " + strideHeight + ", " + strideWidth + ");\n      const ivec3 pads = ivec3(" + padFront + ", " + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int d2 = coords.u;\n\n        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;\n        int xFCorner = xFRCCorner.x;\n        int xRCorner = xFRCCorner.y;\n        int xCCorner = xFRCCorner.z;\n\n        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get\n        // y(yF, yR, yC, d2). ? = to be determined. : = across all\n        // values in that axis.\n        float dotProd = 0.0;\n        for (int wF = 0; wF < " + filterDepth + "; wF++) {\n          int xF = xFCorner + wF * " + dilationDepth + ";\n\n          if (xF < 0 || xF >= " + convInfo.inDepth + ") {\n            continue;\n          }\n\n          for (int wR = 0; wR < " + filterHeight + "; wR++) {\n            int xR = xRCorner + wR * " + dilationHeight + ";\n\n            if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n              continue;\n            }\n\n            for (int wC = 0; wC < " + filterWidth + "; wC++) {\n              int xC = xCCorner + wC * " + dilationWidth + ";\n\n              if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n                continue;\n              }\n\n              for (int d1 = 0; d1 < " + inputDepthNearestVec4 + "; d1 += 4) {\n                vec4 xValues = vec4(\n                  getX(batch, xF, xR, xC, d1),\n                  getX(batch, xF, xR, xC, d1 + 1),\n                  getX(batch, xF, xR, xC, d1 + 2),\n                  getX(batch, xF, xR, xC, d1 + 3)\n                );\n                vec4 wValues = vec4(\n                  getW(wF, wR, wC, d1, d2),\n                  getW(wF, wR, wC, d1 + 1, d2),\n                  getW(wF, wR, wC, d1 + 2, d2),\n                  getW(wF, wR, wC, d1 + 3, d2)\n                );\n\n                dotProd += dot(xValues, wValues);\n              }\n\n              if (" + (inputDepthVec4Remainder === 1) + ") {\n                dotProd +=\n                  getX(batch, xF, xR, xC, " + inputDepthNearestVec4 + ") *\n                  getW(wF, wR, wC, " + inputDepthNearestVec4 + ", d2);\n              } else if (" + (inputDepthVec4Remainder === 2) + ") {\n                vec2 xValues = vec2(\n                  getX(batch, xF, xR, xC, " + inputDepthNearestVec4 + "),\n                  getX(batch, xF, xR, xC, " + inputDepthNearestVec4 + " + 1)\n                );\n                vec2 wValues = vec2(\n                  getW(wF, wR, wC, " + inputDepthNearestVec4 + ", d2),\n                  getW(wF, wR, wC, " + inputDepthNearestVec4 + " + 1, d2)\n                );\n                dotProd += dot(xValues, wValues);\n              } else if (" + (inputDepthVec4Remainder === 3) + ") {\n                vec3 xValues = vec3(\n                  getX(batch, xF, xR, xC, " + inputDepthNearestVec4 + "),\n                  getX(batch, xF, xR, xC, " + inputDepthNearestVec4 + " + 1),\n                  getX(batch, xF, xR, xC, " + inputDepthNearestVec4 + " + 2)\n                );\n                vec3 wValues = vec3(\n                  getW(wF, wR, wC, " + inputDepthNearestVec4 + ", d2),\n                  getW(wF, wR, wC, " + inputDepthNearestVec4 + " + 1, d2),\n                  getW(wF, wR, wC, " + inputDepthNearestVec4 + " + 2, d2)\n                );\n                dotProd += dot(xValues, wValues);\n              }\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return Conv3DProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var DepthwiseConv2DProgram = /** @class */ (function () {
        function DepthwiseConv2DProgram(convInfo, addBias, activation, hasPreluActivation) {
            if (addBias === void 0) { addBias = false; }
            if (activation === void 0) { activation = null; }
            if (hasPreluActivation === void 0) { hasPreluActivation = false; }
            this.variableNames = ['x', 'W'];
            this.outputShape = convInfo.outShape;
            var xNumRows = convInfo.inHeight;
            var xNumCols = convInfo.inWidth;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var channelMul = convInfo.outChannels / convInfo.inChannels;
            var activationSnippet = '', applyActivationSnippet = '';
            if (activation) {
                if (hasPreluActivation) {
                    activationSnippet = "float activation(float a) {\n          float b = getPreluActivationWeightsAtOutCoords();\n          " + activation + "\n        }";
                }
                else {
                    activationSnippet = "\n          float activation(float x) {\n            " + activation + "\n          }\n        ";
                }
                applyActivationSnippet = "result = activation(result);";
            }
            var addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
            if (addBias) {
                this.variableNames.push('bias');
            }
            if (hasPreluActivation) {
                this.variableNames.push('preluActivationWeights');
            }
            this.userCode = "\n      " + activationSnippet + "\n\n      const ivec2 strides = ivec2(" + strideHeight + ", " + strideWidth + ");\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords.x;\n        ivec2 xRCCorner = coords.yz * strides - pads;\n        int d2 = coords.w;\n        int d1 = d2 / " + channelMul + ";\n        int q = d2 - d1 * " + channelMul + ";\n\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.\n        for (int wR = 0; wR < " + filterHeight + "; wR++) {\n          int xR = xRCorner + wR * " + dilationHeight + ";\n\n          if (xR < 0 || xR >= " + xNumRows + ") {\n            continue;\n          }\n\n          for (int wC = 0; wC < " + filterWidth + "; wC++) {\n            int xC = xCCorner + wC * " + dilationWidth + ";\n\n            if (xC < 0 || xC >= " + xNumCols + ") {\n              continue;\n            }\n\n            float xVal = getX(batch, xR, xC, d1);\n            float wVal = getW(wR, wC, d1, q);\n            dotProd += xVal * wVal;\n          }\n        }\n\n        float result = dotProd;\n        " + addBiasSnippet + "\n        " + applyActivationSnippet + "\n        setOutput(result);\n      }\n    ";
        }
        return DepthwiseConv2DProgram;
    }());

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
    var DepthwiseConvPacked2DProgram = /** @class */ (function () {
        function DepthwiseConvPacked2DProgram(convInfo, addBias, activation, hasPreluActivation) {
            if (addBias === void 0) { addBias = false; }
            if (activation === void 0) { activation = null; }
            if (hasPreluActivation === void 0) { hasPreluActivation = false; }
            this.variableNames = ['x', 'W'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = convInfo.outShape;
            var xNumRows = convInfo.inHeight;
            var xNumCols = convInfo.inWidth;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var texelsAcross = filterWidth;
            var mainLoop = "int xR; int xC; int xCOffset;";
            for (var r = 0; r < filterHeight; r++) {
                for (var c = 0; c < filterWidth; c++) {
                    mainLoop += "\n          vec4 xTexelR" + r + "C" + c * 2 + " = vec4(0.);\n          vec4 wR" + r + "C" + c + " = vec4(0.);\n          vec4 xR" + r + "C" + c + " = vec4(0.);";
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
            for (var r = 0; r < filterHeight; r++) {
                for (var texelC = 0; texelC < texelsAcross; texelC++) {
                    var c = texelC * 2;
                    mainLoop += "\n          xR = xRCorner + " + r * dilationHeight + ";\n          xC = xCCorner + " + c * dilationWidth + ";\n        ";
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
                                mainLoop += "\n                xCOffset = xC + 1;\n                if(xR >= 0 && xR < " + xNumRows + " && xCOffset >= 0 && xCOffset < " + xNumCols + ") {\n                  xTexelR" + r + "C" + c + " = getX(batch, xR, xCOffset, d1);\n\n                  // Need to manually clear unused channels in case\n                  // we're reading from recycled texture.\n                  if(xCOffset + 1 >= " + xNumCols + ") {\n                    xTexelR" + r + "C" + c + ".zw = vec2(0.);\n                  }\n                } else {\n                  xTexelR" + r + "C" + c + " = vec4(0.);\n                }\n\n                xCOffset = xC + 1 - 2;\n                if(xR >= 0 && xR < " + xNumRows + " && xCOffset >= 0 && xCOffset < " + xNumCols + ") {\n                  vec4 previous = getX(batch, xR, xCOffset, d1);\n\n                  // Need to manually clear unused channels in case\n                  // we're reading from recycled texture.\n                  if(xCOffset + 1 >= " + xNumCols + ") {\n                    previous.zw = vec2(0.);\n                  }\n\n                  xR" + r + "C" + c + " = vec4(previous.zw, xTexelR" + r + "C" + c + ".xy);\n                } else {\n                  xR" + r + "C" + c + " = vec4(0, 0, xTexelR" + r + "C" + c + ".xy);\n                }\n              ";
                            }
                            else {
                                // Padding is even, so xRC corresponds to a single texel.
                                mainLoop += "\n                if(xR >= 0 && xR < " + xNumRows + " && xC >= 0 && xC < " + xNumCols + ") {\n                  xTexelR" + r + "C" + c + " = getX(batch, xR, xC, d1);\n                } else {\n                  xTexelR" + r + "C" + c + " = vec4(0.);\n                }\n\n                xR" + r + "C" + c + " = xTexelR" + r + "C" + c + ";\n              ";
                            }
                            if (c + 1 < filterWidth) {
                                // If dilation is even, the second entry should match the first
                                // (either both are composed or both are single samples). But if
                                // dilation is odd, then the second entry should be the opposite
                                // of the first (if the first is composed, the second is a single
                                // sample, and vice versa.)
                                var nextTexelOffset = padLeft % 2 === 0 ?
                                    tf.util.nearestLargerEven(dilationWidth) :
                                    dilationWidth;
                                if ((dilationWidth % 2 === 0 && padLeft % 2 === 1) ||
                                    (dilationWidth % 2 !== 0 && padLeft % 2 !== 1)) {
                                    mainLoop += "\n                  xCOffset = xC + " + padLeft % 2 + " + " + nextTexelOffset + ";\n\n                  if(xR >= 0 && xR < " + xNumRows + " &&\n                    xCOffset >= 0 && xCOffset < " + xNumCols + ") {\n                    xTexelR" + r + "C" + (c + 2) + " = getX(batch, xR, xCOffset, d1);\n                  }\n                ";
                                    // If dilation > 1 then the xRC's will not be able to share any
                                    // values, so each xRC will require two unique calls to getX.
                                    if (dilationWidth > 1) {
                                        mainLoop += "\n                    xCOffset -= 2;\n                    if(xR >= 0 && xR < " + xNumRows + " &&\n                      xCOffset >= 0 && xCOffset < " + xNumCols + ") {\n                      xTexelR" + r + "C" + c + " = getX(batch, xR, xCOffset, d1);\n                    } else {\n                      xTexelR" + r + "C" + c + " = vec4(0.);\n                    }\n                  ";
                                    }
                                    mainLoop += "\n                  xR" + r + "C" + (c + 1) + " = vec4(\n                    xTexelR" + r + "C" + c + ".zw, xTexelR" + r + "C" + (c + 2) + ".xy);\n                ";
                                }
                                else {
                                    mainLoop += "\n                  xCOffset = xC + " + nextTexelOffset + ";\n\n                  if(xR >= 0 && xR < " + xNumRows + " &&\n                    xCOffset >= 0 && xCOffset < " + xNumCols + ") {\n                    xTexelR" + r + "C" + (c + 2) + " = getX(batch, xR, xCOffset, d1);\n                  }\n\n                  xR" + r + "C" + (c + 1) + " = xTexelR" + r + "C" + (c + 2) + ";\n                ";
                                }
                            }
                        }
                    }
                    else { // stride > 1
                        if (c < filterWidth) {
                            mainLoop += "\n              if(xR >= 0 && xR < " + xNumRows + ") {\n            ";
                            // Depending on whether padLeft is even or odd, we want either the
                            // xy or zw channels from X texels for xR${r}C${c}. If padLeft is
                            // even, xR${r}C${c + 1} is simply the zw channels of texels we've
                            // already sampled. But if padLeft is odd, xR${r}C{$c + 1}.zw will
                            // need to come from the xy channels of a new texel, hence the `vec4
                            // final` initialized below.
                            if (padLeft % 2 === 1) {
                                mainLoop += "\n                xCOffset = xC + 1 - " + strideWidth + ";\n                if(xCOffset >= 0 && xCOffset < " + xNumCols + ") {\n                  xTexelR" + r + "C" + c + " = getX(batch, xR, xCOffset, d1);\n                } else {\n                  xTexelR" + r + "C" + c + " = vec4(0.);\n                }\n\n                if(xC + 1 >= 0 && xC + 1 < " + xNumCols + ") {\n                  xTexelR" + r + "C" + (c + 2) + " = getX(batch, xR, xC + 1, d1);\n                } else {\n                  xTexelR" + r + "C" + (c + 2) + " = vec4(0.);\n                }\n\n                xR" + r + "C" + c + " = vec4(\n                  xTexelR" + r + "C" + c + ".zw, xTexelR" + r + "C" + (c + 2) + ".zw);\n              ";
                                if (c + 1 < filterWidth) {
                                    mainLoop += "\n                  vec4 final = vec4(0.);\n                  xCOffset = xC + 1 + " + strideWidth + ";\n                  if(xCOffset >= 0 && xCOffset < " + xNumCols + ") {\n                    final = getX(batch, xR, xCOffset, d1);\n                  }\n                  xR" + r + "C" + (c + 1) + " = vec4(xTexelR" + r + "C" + (c + 2) + ".xy, final.xy);\n                ";
                                }
                            }
                            else {
                                mainLoop += "\n                if(xC >= 0 && xC < " + xNumCols + ") {\n                  xTexelR" + r + "C" + c + " = getX(batch, xR, xC, d1);\n                } else {\n                  xTexelR" + r + "C" + c + " = vec4(0.);\n                }\n\n                xCOffset = xC + " + strideWidth + ";\n                if(xCOffset >= 0 && xCOffset < " + xNumCols + ") {\n                  xTexelR" + r + "C" + (c + 2) + " = getX(batch, xR, xCOffset, d1);\n                } else {\n                  xTexelR" + r + "C" + (c + 2) + " = vec4(0.);\n                }\n\n                xR" + r + "C" + c + " = vec4(\n                  xTexelR" + r + "C" + c + ".xy, xTexelR" + r + "C" + (c + 2) + ".xy);\n              ";
                                if (c + 1 < filterWidth) {
                                    mainLoop += "\n                  xR" + r + "C" + (c + 1) + " = vec4(\n                    xTexelR" + r + "C" + c + ".zw, xTexelR" + r + "C" + (c + 2) + ".zw);\n                ";
                                }
                            }
                            mainLoop += "}";
                        }
                    }
                    if (c < filterWidth) {
                        mainLoop += "\n            vec4 wTexelR" + r + "C" + c + " = getW(" + r + ", " + c + ", d1, q);\n            wR" + r + "C" + c + " = vec4(wTexelR" + r + "C" + c + ".xz, wTexelR" + r + "C" + c + ".xz);\n          ";
                        if (c + 1 < filterWidth) {
                            mainLoop += "\n              vec4 wTexelR" + r + "C" + (c + 1) + " = getW(" + r + ", " + (c + 1) + ", d1, q);\n              wR" + r + "C" + (c + 1) + " =\n                vec4(wTexelR" + r + "C" + (c + 1) + ".xz, wTexelR" + r + "C" + (c + 1) + ".xz);";
                        }
                    }
                }
            }
            for (var r = 0; r < filterHeight; r++) {
                for (var c = 0; c < filterWidth; c++) {
                    mainLoop += "dotProd += xR" + r + "C" + c + " * wR" + r + "C" + c + ";";
                }
            }
            var activationSnippet = '', applyActivationSnippet = '';
            if (activation) {
                if (hasPreluActivation) {
                    activationSnippet = "vec4 activation(vec4 a) {\n          vec4 b = getPreluActivationWeightsAtOutCoords();\n          " + activation + "\n        }";
                }
                else {
                    activationSnippet = "vec4 activation(vec4 x) {\n          " + activation + "\n        }";
                }
                applyActivationSnippet = "result = activation(result);";
            }
            var addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
            if (addBias) {
                this.variableNames.push('bias');
            }
            if (hasPreluActivation) {
                this.variableNames.push('preluActivationWeights');
            }
            this.userCode = "\n      " + activationSnippet + "\n\n      const ivec2 strides = ivec2(" + strideHeight + ", " + strideWidth + ");\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n\n        ivec4 coords = getOutputCoords();\n        int batch = coords.x;\n        ivec2 xRCCorner = coords.yz * strides - pads;\n        int d2 = coords.w;\n        int d1 = d2;\n        int q = 0;\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        vec4 dotProd = vec4(0.);\n\n        " + mainLoop + "\n\n        vec4 result = dotProd;\n        " + addBiasSnippet + "\n        " + applyActivationSnippet + "\n        setOutput(result);\n      }\n    ";
        }
        return DepthwiseConvPacked2DProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var CropAndResizeProgram = /** @class */ (function () {
        function CropAndResizeProgram(imageShape, boxShape, cropSize, method, extrapolationValue) {
            this.variableNames = ['Image', 'Boxes', 'BoxInd'];
            this.outputShape = [];
            var batch = imageShape[0], imageHeight = imageShape[1], imageWidth = imageShape[2], depth = imageShape[3];
            var numBoxes = boxShape[0];
            var cropHeight = cropSize[0], cropWidth = cropSize[1];
            this.outputShape = [numBoxes, cropHeight, cropWidth, depth];
            var methodId = method === 'bilinear' ? 1 : 0;
            var _a = [imageHeight - 1 + ".0", imageWidth - 1 + ".0"], inputHeightFloat = _a[0], inputWidthFloat = _a[1];
            var _b = cropHeight > 1 ?
                [
                    "" + (imageHeight - 1) / (cropHeight - 1),
                    '(y2-y1) * height_ratio',
                    "y1*" + inputHeightFloat + " + float(y)*(height_scale)",
                ] :
                [
                    '0.0',
                    '0.0',
                    "0.5 * (y1+y2) * " + inputHeightFloat,
                ], heightRatio = _b[0], heightScale = _b[1], inY = _b[2];
            var _c = cropWidth > 1 ?
                [
                    "" + (imageWidth - 1) / (cropWidth - 1),
                    '(x2-x1) * width_ratio',
                    "x1*" + inputWidthFloat + " + float(x)*(width_scale)",
                ] :
                [
                    '0.0',
                    '0.0',
                    "0.5 * (x1+x2) * " + inputWidthFloat,
                ], widthRatio = _c[0], widthScale = _c[1], inX = _c[2];
            // Reference implementation
            // tslint:disable-next-line:max-line-length
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op_gpu.cu.cc
            this.userCode = "\n      const float height_ratio = float(" + heightRatio + ");\n      const float width_ratio = float(" + widthRatio + ");\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int y = coords[1];\n        int x = coords[2];\n        int d = coords[3];\n\n        // get box vals\n        float y1 = getBoxes(b,0);\n        float x1 = getBoxes(b,1);\n        float y2 = getBoxes(b,2);\n        float x2 = getBoxes(b,3);\n\n        // get image in batch index\n        int bInd = round(getBoxInd(b));\n        if(bInd < 0 || bInd >= " + batch + ") {\n          return;\n        }\n\n        float height_scale = " + heightScale + ";\n        float width_scale = " + widthScale + ";\n\n        float in_y = " + inY + ";\n        if( in_y < 0.0 || in_y > " + inputHeightFloat + " ) {\n          setOutput(float(" + extrapolationValue + "));\n          return;\n        }\n        float in_x = " + inX + ";\n        if( in_x < 0.0 || in_x > " + inputWidthFloat + " ) {\n          setOutput(float(" + extrapolationValue + "));\n          return;\n        }\n\n        vec2 sourceFracIndexCR = vec2(in_x,in_y);\n        if(" + methodId + " == 1) {\n          // Compute the four integer indices.\n          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);\n          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));\n\n          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);\n          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);\n          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);\n          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);\n\n          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);\n\n          float top = topLeft + (topRight - topLeft) * fracCR.x;\n          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;\n          float newValue = top + (bottom - top) * fracCR.y;\n          setOutput(newValue);\n        } else {\n          // Compute the coordinators of nearest neighbor point.\n          ivec2 sourceNearestCR = ivec2(floor(\n            sourceFracIndexCR + vec2(0.5,0.5)));\n          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);\n          setOutput(newValue);\n        }\n      }\n    ";
        }
        return CropAndResizeProgram;
    }());

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
    var CumSumProgram = /** @class */ (function () {
        function CumSumProgram(shape, exclusive, reverse, index) {
            this.variableNames = ['x'];
            this.outputShape = shape;
            var rank = shape.length;
            var val = exclusive ? '0.0' : "getX(" + getCoords(rank, 'coords') + ")";
            var length = shape[shape.length - 1];
            var condition = '';
            var idxString = '';
            // When exclusive is set, the cumsum op becomes roll op that copies the
            // value from the previous index based on the direction specified by the
            // reverse flag.
            if (exclusive) {
                condition = reverse ? "end != " + (length - 1) : 'end != 0';
                idxString = reverse ? 'end + 1' : 'end - 1';
            }
            else {
                condition = reverse ? "end + pow2 < " + length : 'end >= pow2';
                idxString = (reverse ? 'end + pow2' : 'end - pow2');
            }
            this.userCode = "\n      void main() {\n        " + getCoordsDataType(rank) + " coords = getOutputCoords();\n        int end = " + getFinalCoord(rank, 'coords') + ";\n        float val = " + val + ";\n        int pow2 = int(pow(2.0, float(" + index + ")));\n        if (" + condition + ") {\n          int idx = " + idxString + ";\n          " + getFinalCoord(rank, 'coords') + " = idx;\n          val += getX(" + getCoords(rank, 'coords') + ");\n        }\n        setOutput(val);\n      }\n    ";
        }
        return CumSumProgram;
    }());
    function getCoords(rank, name) {
        if (rank === 1) {
            return "" + name;
        }
        else if (rank === 2) {
            return name + ".x, " + name + ".y";
        }
        else if (rank === 3) {
            return name + ".x, " + name + ".y, " + name + ".z";
        }
        else if (rank === 4) {
            return name + ".x, " + name + ".y, " + name + ".z, " + name + ".w";
        }
        else {
            throw Error("Cumulative sum for rank " + rank + " is not yet supported");
        }
    }
    function getFinalCoord(rank, name) {
        if (rank === 1) {
            return "" + name;
        }
        else if (rank === 2) {
            return name + ".y";
        }
        else if (rank === 3) {
            return name + ".z";
        }
        else if (rank === 4) {
            return name + ".w";
        }
        else {
            throw Error("Cumulative sum for rank " + rank + " is not yet supported");
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
    var DecodeMatrixProgram = /** @class */ (function () {
        function DecodeMatrixProgram(outputShape) {
            this.variableNames = ['A'];
            this.packedInputs = false;
            this.packedOutput = true;
            this.outPackingScheme = PackingScheme.DENSE;
            var texShape = getDenseTexShape(outputShape);
            var glsl = getGlslDifferences();
            this.outputShape = outputShape;
            this.userCode = "\n      ivec3 outCoordsFromFlatIndex(int index) {\n        " + getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], outputShape) + "\n        return ivec3(r, c, d);\n      }\n\n      void main() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n          vec2(" + texShape[0] + ", " + texShape[1] + "));\n        int index = 4 * (resTexRC.x * " + texShape[1] + " + resTexRC.y);\n\n        vec4 result = vec4(0.);\n\n        for (int i=0; i<4; i++) {\n          int flatIndex = index + i;\n          ivec3 rc = outCoordsFromFlatIndex(flatIndex);\n          result[i] = getA(rc.x, rc.y, rc.z);\n        }\n\n        " + glsl.output + " = result;\n      }\n    ";
        }
        return DecodeMatrixProgram;
    }());

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
    var DecodeMatrixPackedProgram = /** @class */ (function () {
        function DecodeMatrixPackedProgram(outputShape) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outPackingScheme = PackingScheme.DENSE;
            var texShape = getDenseTexShape(outputShape);
            var glsl = getGlslDifferences();
            this.outputShape = outputShape;
            this.userCode = "\n      ivec3 outCoordsFromFlatIndex(int index) {\n        " + getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], outputShape) + "\n        return ivec3(r, c, d);\n      }\n\n      void main() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n          vec2(" + texShape[0] + ", " + texShape[1] + "));\n        int index = 4 * (resTexRC.x * " + texShape[1] + " + resTexRC.y);\n\n        vec4 result = vec4(0.);\n\n        for (int i=0; i<4; i++) {\n          int flatIndex = index + i;\n          ivec3 rc = outCoordsFromFlatIndex(flatIndex);\n          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));\n        }\n\n        " + glsl.output + " = result;\n      }\n    ";
        }
        return DecodeMatrixPackedProgram;
    }());

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
    var DepthToSpaceProgram = /** @class */ (function () {
        function DepthToSpaceProgram(outputShape, blockSize, dataFormat) {
            this.variableNames = ['x'];
            this.outputShape = [];
            this.outputShape = outputShape;
            this.blockSize = blockSize;
            this.dataFormat = dataFormat;
            this.userCode = "\n    void main() {\n      ivec4 coords = getOutputCoords();\n      int b = coords[0];\n      int h = " + this.getHeightCoordString() + ";\n      int w = " + this.getWidthCoordString() + ";\n      int d = " + this.getDepthCoordString() + ";\n\n      int in_h = h / " + blockSize + ";\n      int offset_h = imod(h, " + blockSize + ");\n      int in_w = w / " + blockSize + ";\n      int offset_w = imod(w, " + blockSize + ");\n      int offset_d = (offset_h * " + blockSize + " + offset_w) *\n        " + this.getOutputDepthSize() + ";\n      int in_d = d + offset_d;\n\n      float result = " + this.getInputSamplingString() + ";\n      setOutput(result);\n    }\n  ";
        }
        DepthToSpaceProgram.prototype.getHeightCoordString = function () {
            if (this.dataFormat === 'NHWC') {
                return "coords[1]";
            }
            else {
                return "coords[2]";
            }
        };
        DepthToSpaceProgram.prototype.getWidthCoordString = function () {
            if (this.dataFormat === 'NHWC') {
                return "coords[2]";
            }
            else {
                return "coords[3]";
            }
        };
        DepthToSpaceProgram.prototype.getDepthCoordString = function () {
            if (this.dataFormat === 'NHWC') {
                return "coords[3]";
            }
            else {
                return "coords[1]";
            }
        };
        DepthToSpaceProgram.prototype.getOutputDepthSize = function () {
            if (this.dataFormat === 'NHWC') {
                return this.outputShape[3];
            }
            else {
                return this.outputShape[1];
            }
        };
        DepthToSpaceProgram.prototype.getInputSamplingString = function () {
            if (this.dataFormat === 'NHWC') {
                return "getX(b, in_h, in_w, in_d)";
            }
            else {
                return "getX(b, in_d, in_h, in_w)";
            }
        };
        return DepthToSpaceProgram;
    }());

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
    var DiagProgram = /** @class */ (function () {
        function DiagProgram(size) {
            this.variableNames = ['X'];
            this.outputShape = [size, size];
            this.userCode = "\n      void main() {\n          ivec2 coords = getOutputCoords();\n          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;\n          setOutput(val);\n      }\n    ";
        }
        return DiagProgram;
    }());

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
    var EncodeFloatProgram = /** @class */ (function () {
        function EncodeFloatProgram(outputShape) {
            this.variableNames = ['A'];
            this.outTexUsage = TextureUsage.DOWNLOAD;
            var glsl = getGlslDifferences();
            this.outputShape = outputShape;
            this.userCode = "\n      " + ENCODE_FLOAT_SNIPPET + "\n\n      void main() {\n        float x = getAAtOutCoords();\n        " + glsl.output + " = encode_float(x);\n      }\n    ";
        }
        return EncodeFloatProgram;
    }());

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
    var EncodeFloatPackedProgram = /** @class */ (function () {
        function EncodeFloatPackedProgram(outputShape) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = false;
            this.outTexUsage = TextureUsage.DOWNLOAD;
            var glsl = getGlslDifferences();
            this.outputShape = outputShape;
            this.userCode = "\n      " + ENCODE_FLOAT_SNIPPET + "\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));\n        " + glsl.output + " = encode_float(x);\n      }\n    ";
        }
        return EncodeFloatPackedProgram;
    }());

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
    var EncodeMatrixProgram = /** @class */ (function () {
        function EncodeMatrixProgram(outputShape, texShape, inputIsUnsignedByte) {
            if (inputIsUnsignedByte === void 0) { inputIsUnsignedByte = false; }
            this.variableNames = ['A'];
            var glsl = getGlslDifferences();
            var height = texShape[0], width = texShape[1];
            this.outputShape = outputShape;
            var output = "result";
            if (inputIsUnsignedByte) {
                output = "floor(result * 255. + 0.5)";
            }
            this.userCode = "\n      " + getFlatIndexFrom3D(outputShape) + "\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n\n        int flatIndex = getFlatIndex(coords);\n        int offset = imod(flatIndex, 4);\n\n        flatIndex = idiv(flatIndex, 4, 1.);\n\n        int r = flatIndex / " + width + ";\n        int c = imod(flatIndex, " + width + ");\n        vec2 uv = (vec2(c, r) + halfCR) / vec2(" + width + ".0, " + height + ".0);\n        vec4 values = " + glsl.texture2D + "(A, uv);\n\n        float result;\n\n        if(offset == 0) {\n          result = values[0];\n        } else if(offset == 1) {\n          result = values[1];\n        } else if(offset == 2) {\n          result = values[2];\n        } else {\n          result = values[3];\n        }\n\n        " + glsl.output + " = vec4(" + output + ", 0., 0., 0.);\n      }\n    ";
        }
        return EncodeMatrixProgram;
    }());

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
    var EncodeMatrixPackedProgram = /** @class */ (function () {
        function EncodeMatrixPackedProgram(outputShape, texShape, inputIsUnsignedByte) {
            if (inputIsUnsignedByte === void 0) { inputIsUnsignedByte = false; }
            this.variableNames = ['A'];
            this.packedInputs = false;
            this.packedOutput = true;
            var glsl = getGlslDifferences();
            var height = texShape[0], width = texShape[1];
            this.outputShape = outputShape;
            var mainLoop = '';
            var output = 'result';
            if (inputIsUnsignedByte) {
                output = 'floor(result * 255. + 0.5)';
            }
            for (var row = 0; row <= 1; row++) {
                for (var col = 0; col <= 1; col++) {
                    var channel = row * 2 + col;
                    mainLoop += "\n          localCoords = coords;\n          if(localCoords[2] + " + col + " < " + outputShape[2] + ") {\n            localCoords[2] += " + col + ";\n            if(localCoords[1] + " + row + " < " + outputShape[1] + ") {\n              localCoords[1] += " + row + ";\n\n              flatIndex = getFlatIndex(localCoords);\n              offset = imod(flatIndex, 4);\n\n              flatIndex = idiv(flatIndex, 4, 1.);\n\n              r = flatIndex / " + width + ";\n              c = imod(flatIndex, " + width + ");\n              uv = (vec2(c, r) + halfCR) / vec2(" + width + ".0, " + height + ".0);\n              values = " + glsl.texture2D + "(A, uv);\n\n              if(offset == 0) {\n                result[" + channel + "] = values[0];\n              } else if(offset == 1) {\n                result[" + channel + "] = values[1];\n              } else if(offset == 2) {\n                result[" + channel + "] = values[2];\n              } else {\n                result[" + channel + "] = values[3];\n              }\n            }\n          }\n        ";
                }
            }
            this.userCode = "\n      " + getFlatIndexFrom3D(outputShape) + "\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n\n        vec4 result = vec4(0.);\n        int flatIndex, r, c, offset;\n        ivec3 localCoords;\n        vec2 uv;\n        vec4 values;\n\n        " + mainLoop + "\n\n        " + glsl.output + " = " + output + ";\n      }\n    ";
        }
        return EncodeMatrixPackedProgram;
    }());

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
    var COMPLEX_FFT = {
        REAL: 'return real * expR - imag * expI;',
        IMAG: 'return real * expI + imag * expR;'
    };
    var FFTProgram = /** @class */ (function () {
        function FFTProgram(op, inputShape, inverse) {
            this.variableNames = ['real', 'imag'];
            var innerDim = inputShape[1];
            this.outputShape = inputShape;
            var exponentMultiplierSnippet = inverse ? "2.0 * " + Math.PI : "-2.0 * " + Math.PI;
            var resultDenominator = inverse ? innerDim + ".0" : '1.0';
            this.userCode = "\n      const float exponentMultiplier = " + exponentMultiplierSnippet + ";\n\n      float unaryOpComplex(float real, float expR, float imag, float expI) {\n        " + op + "\n      }\n\n      float mulMatDFT(int batch, int index) {\n        float indexRatio = float(index) / float(" + innerDim + ");\n        float exponentMultiplierTimesIndexRatio =\n            exponentMultiplier * indexRatio;\n\n        float result = 0.0;\n\n        for (int i = 0; i < " + innerDim + "; i++) {\n          // x = (-2|2 * PI / N) * index * i;\n          float x = exponentMultiplierTimesIndexRatio * float(i);\n          float expR = cos(x);\n          float expI = sin(x);\n          float real = getReal(batch, i);\n          float imag = getImag(batch, i);\n\n          result +=\n              unaryOpComplex(real, expR, imag, expI) / " + resultDenominator + ";\n        }\n\n        return result;\n      }\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        setOutput(mulMatDFT(coords[0], coords[1]));\n      }\n    ";
        }
        return FFTProgram;
    }());

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
    var FillProgram = /** @class */ (function () {
        function FillProgram(shape, value) {
            this.outputShape = [];
            this.variableNames = ['x'];
            this.outputShape = shape;
            this.userCode = "\n      uniform float value;\n      void main() {\n        // Input can be obtained from uniform value.\n        setOutput(value);\n      }\n    ";
        }
        FillProgram.prototype.getCustomSetupFunc = function (value) {
            var _this = this;
            return function (gpgpu, webGLProgram) {
                if (_this.valueLoc == null) {
                    _this.valueLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'value');
                }
                gpgpu.gl.uniform1f(_this.valueLoc, value);
            };
        };
        return FillProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var GatherProgram = /** @class */ (function () {
        function GatherProgram(aShape, indicesLength, axis) {
            this.variableNames = ['A', 'indices'];
            var outputShape = aShape.slice();
            outputShape[axis] = indicesLength;
            this.outputShape = outputShape;
            this.rank = outputShape.length;
            var dtype = getCoordsDataType(this.rank);
            var sourceCoords = getSourceCoords$1(aShape, axis);
            this.userCode = "\n      void main() {\n        " + dtype + " resRC = getOutputCoords();\n        setOutput(getA(" + sourceCoords + "));\n      }\n    ";
        }
        return GatherProgram;
    }());
    function getSourceCoords$1(aShape, axis) {
        var rank = aShape.length;
        if (rank > 4) {
            throw Error("Gather for rank " + rank + " is not yet supported");
        }
        if (rank === 1) {
            return "int(getIndices(resRC))";
        }
        var currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
        var sourceCoords = [];
        for (var i = 0; i < aShape.length; i++) {
            if (i === axis) {
                sourceCoords.push("int(getIndices(" + currentCoords[i] + "))");
            }
            else {
                sourceCoords.push("" + currentCoords[i]);
            }
        }
        return sourceCoords.join();
    }

    var GatherNDProgram = /** @class */ (function () {
        function GatherNDProgram(sliceDim, strides, shape) {
            this.sliceDim = sliceDim;
            this.strides = strides;
            this.variableNames = ['x', 'indices'];
            this.outputShape = shape;
            var stridesType = getCoordsDataType(strides.length);
            var dtype = getCoordsDataType(shape.length);
            var strideString = this.sliceDim > 1 ? 'strides[j]' : 'strides';
            this.userCode = "\n        " + stridesType + " strides = " + stridesType + "(" + this.strides + ");\n         void main() {\n          " + dtype + " coords = getOutputCoords();\n          int flattenIndex = 0;\n          for (int j = 0; j < " + this.sliceDim + "; j++) {\n            int index = round(getIndices(coords[0], j));\n            flattenIndex += index * " + strideString + ";\n          }\n          setOutput(getX(flattenIndex, coords[1]));\n        }\n      ";
        }
        return GatherNDProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
        var glsl = getGlslDifferences();
        var vertexShaderSource = glsl.version + "\n    precision highp float;\n    " + glsl.attribute + " vec3 clipSpacePos;\n    " + glsl.attribute + " vec2 uv;\n    " + glsl.varyingVs + " vec2 resultUV;\n\n    void main() {\n      gl_Position = vec4(clipSpacePos, 1);\n      resultUV = uv;\n    }";
        return createVertexShader(gl, debug, vertexShaderSource);
    }
    function createVertexBuffer(gl, debug) {
        // [x y z u v] * [upper-left, lower-left, upper-right, lower-right]
        var vertexArray = new Float32Array([-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
        return createStaticVertexBuffer(gl, debug, vertexArray);
    }
    function createIndexBuffer(gl, debug) {
        // OpenGL (and WebGL) have "CCW == front" winding
        var triangleVertexIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);
        return createStaticIndexBuffer(gl, debug, triangleVertexIndices);
    }
    function createAndConfigureTexture(gl, debug, width, height, internalFormat, textureFormat, textureType) {
        validateTextureSize(width, height);
        var texture = createTexture(gl, debug);
        var tex2d = gl.TEXTURE_2D;
        callAndCheck(gl, debug, function () { return gl.bindTexture(tex2d, texture); });
        callAndCheck(gl, debug, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
        callAndCheck(gl, debug, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
        callAndCheck(gl, debug, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
        callAndCheck(gl, debug, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
        callAndCheck(gl, debug, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, textureFormat, textureType, null); });
        callAndCheck(gl, debug, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
        return texture;
    }
    function createFloat32MatrixTexture(gl, debug, rows, columns, textureConfig) {
        var _a = getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
        return createAndConfigureTexture(gl, debug, width, height, textureConfig.internalFormatFloat, textureConfig.textureFormatFloat, gl.FLOAT);
    }
    function createFloat16MatrixTexture(gl, debug, rows, columns, textureConfig) {
        var _a = getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
        return createAndConfigureTexture(gl, debug, width, height, textureConfig.internalFormatHalfFloat, textureConfig.textureFormatFloat, textureConfig.textureTypeHalfFloat);
    }
    function createUnsignedBytesMatrixTexture(gl, debug, rows, columns, textureConfig) {
        var _a = getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
        return createAndConfigureTexture(gl, debug, width, height, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE);
    }
    function createPackedMatrixTexture(gl, debug, rows, columns, textureConfig) {
        var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
        return createAndConfigureTexture(gl, debug, width, height, textureConfig.internalFormatPackedFloat, gl.RGBA, gl.FLOAT);
    }
    function createFloat16PackedMatrixTexture(gl, debug, rows, columns, textureConfig) {
        var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
        return createAndConfigureTexture(gl, debug, width, height, textureConfig.internalFormatPackedHalfFloat, gl.RGBA, textureConfig.textureTypeHalfFloat);
    }
    function bindVertexProgramAttributeStreams(gl, debug, program, vertexBuffer) {
        var posOffset = 0; // x is the first buffer element
        var uvOffset = 3 * 4; // uv comes after [x y z]
        var stride = (3 * 4) + (2 * 4); // xyz + uv, each entry is 4-byte float.
        callAndCheck(gl, debug, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer); });
        var success = bindVertexBufferToProgramAttribute(gl, debug, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);
        return success &&
            bindVertexBufferToProgramAttribute(gl, debug, program, 'uv', vertexBuffer, 2, stride, uvOffset);
    }
    function uploadDenseMatrixToTexture(gl, debug, texture, width, height, data, textureConfig) {
        callAndCheck(gl, debug, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
        var dataForUpload, texelDataType, internalFormat;
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
        callAndCheck(gl, debug, function () { return gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, gl.RGBA, texelDataType, dataForUpload); });
        callAndCheck(gl, debug, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    }
    function uploadPixelDataToTexture(gl, debug, texture, pixels) {
        callAndCheck(gl, debug, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
        if (pixels.data instanceof Uint8Array) {
            callAndCheck(gl, debug, function () { return gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, pixels.width, pixels.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels.data); });
        }
        else {
            callAndCheck(gl, debug, function () { return gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, pixels); });
        }
        callAndCheck(gl, debug, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    }
    function createBufferFromOutputTexture(gl2, debug, rows, columns, textureConfig) {
        // Create and bind the buffer.
        var buffer = gl2.createBuffer();
        callAndCheck(gl2, debug, function () { return gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer); });
        // Initialize the buffer to the size of the texture in bytes.
        var bytesPerFloat = 4;
        var valuesPerTexel = 4;
        var bufferSizeBytes = bytesPerFloat * valuesPerTexel * rows * columns;
        callAndCheck(gl2, debug, function () { return gl2.bufferData(gl2.PIXEL_PACK_BUFFER, bufferSizeBytes, gl2.STREAM_READ); });
        // Enqueue a command on the GPU command queue to copy of texture into the
        // buffer.
        callAndCheck(gl2, debug, function () { return gl2.readPixels(0, 0, columns, rows, gl2.RGBA, gl2.FLOAT, 0); });
        callAndCheck(gl2, debug, function () { return gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null); });
        return buffer;
    }
    function downloadFloat32MatrixFromBuffer(gl, buffer, size) {
        var gl2 = gl;
        var downloadTarget = new Float32Array(size);
        gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
        gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, downloadTarget);
        gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);
        return downloadTarget;
    }
    function downloadByteEncodedFloatMatrixFromOutputTexture(gl, debug, rows, columns, textureConfig) {
        var _a = getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
        var numChannels = 4;
        var downloadTarget = new Uint8Array(getUnpackedArraySizeFromMatrixSize(rows * columns, numChannels));
        callAndCheck(gl, debug, function () { return gl.readPixels(0, 0, w, h, textureConfig.downloadTextureFormat, gl.UNSIGNED_BYTE, downloadTarget); });
        // By wrapping the buffer in a Float32Array, we use native browser IEEE 754
        // decoding of the 4 bytes that back each 32 bit float.
        return new Float32Array(downloadTarget.buffer);
    }
    function downloadPackedMatrixFromBuffer(gl, buffer, batch, rows, cols, physicalRows, physicalCols, textureConfig) {
        var gl2 = gl;
        var downloadTarget = new Float32Array(getPackedRGBAArraySizeFromMatrixShape(physicalRows, physicalCols));
        gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
        gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, downloadTarget);
        gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);
        return downloadTarget;
    }
    function downloadMatrixFromPackedOutputTexture(gl, debug, physicalRows, physicalCols) {
        var packedRGBA = new Float32Array(physicalRows * physicalCols * 4);
        callAndCheck(gl, debug, function () { return gl.readPixels(0, 0, physicalCols, physicalRows, gl.RGBA, gl.FLOAT, packedRGBA); });
        return packedRGBA;
    }

    var gpgpu_util = {
        __proto__: null,
        createVertexShader: createVertexShader$1,
        createVertexBuffer: createVertexBuffer,
        createIndexBuffer: createIndexBuffer,
        createFloat32MatrixTexture: createFloat32MatrixTexture,
        createFloat16MatrixTexture: createFloat16MatrixTexture,
        createUnsignedBytesMatrixTexture: createUnsignedBytesMatrixTexture,
        createPackedMatrixTexture: createPackedMatrixTexture,
        createFloat16PackedMatrixTexture: createFloat16PackedMatrixTexture,
        bindVertexProgramAttributeStreams: bindVertexProgramAttributeStreams,
        uploadDenseMatrixToTexture: uploadDenseMatrixToTexture,
        uploadPixelDataToTexture: uploadPixelDataToTexture,
        createBufferFromOutputTexture: createBufferFromOutputTexture,
        downloadFloat32MatrixFromBuffer: downloadFloat32MatrixFromBuffer,
        downloadByteEncodedFloatMatrixFromOutputTexture: downloadByteEncodedFloatMatrixFromOutputTexture,
        downloadPackedMatrixFromBuffer: downloadPackedMatrixFromBuffer,
        downloadMatrixFromPackedOutputTexture: downloadMatrixFromPackedOutputTexture
    };

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var GPGPUContext = /** @class */ (function () {
        function GPGPUContext(gl) {
            this.outputTexture = null;
            this.program = null;
            this.disposed = false;
            this.vertexAttrsAreBound = false;
            this.itemsToPoll = [];
            var glVersion = tf.env().getNumber('WEBGL_VERSION');
            if (gl != null) {
                this.gl = gl;
                setWebGLContext(glVersion, gl);
            }
            else {
                this.gl = getWebGLContext(glVersion);
            }
            // WebGL 2.0 enables texture floats without an extension.
            var COLOR_BUFFER_FLOAT = 'WEBGL_color_buffer_float';
            var COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
            if (tf.env().getNumber('WEBGL_VERSION') === 1) {
                var TEXTURE_FLOAT = 'OES_texture_float';
                var TEXTURE_HALF_FLOAT = 'OES_texture_half_float';
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
        Object.defineProperty(GPGPUContext.prototype, "debug", {
            get: function () {
                return tf.env().getBool('DEBUG');
            },
            enumerable: true,
            configurable: true
        });
        GPGPUContext.prototype.dispose = function () {
            var _this = this;
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
            var gl = this.gl;
            callAndCheck(gl, this.debug, function () { return gl.finish(); });
            callAndCheck(gl, this.debug, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, null); });
            callAndCheck(gl, this.debug, function () { return gl.deleteFramebuffer(_this.framebuffer); });
            callAndCheck(gl, this.debug, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, null); });
            callAndCheck(gl, this.debug, function () { return gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null); });
            callAndCheck(gl, this.debug, function () { return gl.deleteBuffer(_this.indexBuffer); });
            this.disposed = true;
        };
        GPGPUContext.prototype.createFloat32MatrixTexture = function (rows, columns) {
            this.throwIfDisposed();
            return createFloat32MatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
        };
        GPGPUContext.prototype.createFloat16MatrixTexture = function (rows, columns) {
            this.throwIfDisposed();
            return createFloat16MatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
        };
        GPGPUContext.prototype.createUnsignedBytesMatrixTexture = function (rows, columns) {
            this.throwIfDisposed();
            return createUnsignedBytesMatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
        };
        GPGPUContext.prototype.uploadPixelDataToTexture = function (texture, pixels) {
            this.throwIfDisposed();
            uploadPixelDataToTexture(this.gl, this.debug, texture, pixels);
        };
        GPGPUContext.prototype.uploadDenseMatrixToTexture = function (texture, width, height, data) {
            this.throwIfDisposed();
            uploadDenseMatrixToTexture(this.gl, this.debug, texture, width, height, data, this.textureConfig);
        };
        GPGPUContext.prototype.createFloat16PackedMatrixTexture = function (rows, columns) {
            this.throwIfDisposed();
            return createFloat16PackedMatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
        };
        GPGPUContext.prototype.createPackedMatrixTexture = function (rows, columns) {
            this.throwIfDisposed();
            return createPackedMatrixTexture(this.gl, this.debug, rows, columns, this.textureConfig);
        };
        GPGPUContext.prototype.deleteMatrixTexture = function (texture) {
            var _this = this;
            this.throwIfDisposed();
            if (this.outputTexture === texture) {
                unbindColorTextureFromFramebuffer(this.gl, this.debug, this.framebuffer);
                this.outputTexture = null;
            }
            callAndCheck(this.gl, this.debug, function () { return _this.gl.deleteTexture(texture); });
        };
        GPGPUContext.prototype.downloadByteEncodedFloatMatrixFromOutputTexture = function (texture, rows, columns) {
            var _this = this;
            return this.downloadMatrixDriver(texture, function () { return downloadByteEncodedFloatMatrixFromOutputTexture(_this.gl, _this.debug, rows, columns, _this.textureConfig); });
        };
        GPGPUContext.prototype.downloadPackedMatrixFromBuffer = function (buffer, batch, rows, columns, physicalRows, physicalCols) {
            return downloadPackedMatrixFromBuffer(this.gl, buffer, batch, rows, columns, physicalRows, physicalCols, this.textureConfig);
        };
        GPGPUContext.prototype.downloadFloat32MatrixFromBuffer = function (buffer, size) {
            return downloadFloat32MatrixFromBuffer(this.gl, buffer, size);
        };
        GPGPUContext.prototype.createBufferFromTexture = function (texture, rows, columns) {
            this.bindTextureToFrameBuffer(texture);
            var result = createBufferFromOutputTexture(this.gl, this.debug, rows, columns, this.textureConfig);
            this.unbindTextureToFrameBuffer();
            return result;
        };
        GPGPUContext.prototype.createAndWaitForFence = function () {
            var fenceContext = this.createFence(this.gl);
            return this.pollFence(fenceContext);
        };
        GPGPUContext.prototype.createFence = function (gl) {
            var _this = this;
            var query;
            var isFencePassed;
            if (tf.env().getBool('WEBGL_FENCE_API_ENABLED')) {
                var gl2_1 = gl;
                var sync_1 = gl2_1.fenceSync(gl2_1.SYNC_GPU_COMMANDS_COMPLETE, 0);
                gl.flush();
                isFencePassed = function () {
                    var status = gl2_1.clientWaitSync(sync_1, 0, 0);
                    return status === gl2_1.ALREADY_SIGNALED ||
                        status === gl2_1.CONDITION_SATISFIED;
                };
                query = sync_1;
            }
            else if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
                query = this.beginQuery();
                this.endQuery();
                isFencePassed = function () { return _this.isQueryAvailable(query, tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION')); };
            }
            else {
                // If we have no way to fence, return true immediately. This will fire in
                // WebGL 1.0 when there is no disjoint query timer. In this case, because
                // the fence passes immediately, we'll immediately ask for a download of
                // the texture, which will cause the UI thread to hang.
                isFencePassed = function () { return true; };
            }
            return { query: query, isFencePassed: isFencePassed };
        };
        GPGPUContext.prototype.downloadMatrixFromPackedTexture = function (texture, physicalRows, physicalCols) {
            var _this = this;
            return this.downloadMatrixDriver(texture, function () { return downloadMatrixFromPackedOutputTexture(_this.gl, _this.debug, physicalRows, physicalCols); });
        };
        GPGPUContext.prototype.createProgram = function (fragmentShaderSource) {
            this.throwIfDisposed();
            var gl = this.gl;
            var fragmentShader = createFragmentShader(gl, this.debug, fragmentShaderSource);
            var vertexShader = createVertexShader$1(gl, this.debug);
            var program = createProgram(gl, this.debug);
            callAndCheck(gl, this.debug, function () { return gl.attachShader(program, vertexShader); });
            callAndCheck(gl, this.debug, function () { return gl.attachShader(program, fragmentShader); });
            linkProgram(gl, this.debug, program);
            if (this.debug) {
                validateProgram(gl, this.debug, program);
            }
            if (!this.vertexAttrsAreBound) {
                this.setProgram(program);
                this.vertexAttrsAreBound = bindVertexProgramAttributeStreams(gl, this.debug, this.program, this.vertexBuffer);
            }
            return program;
        };
        GPGPUContext.prototype.deleteProgram = function (program) {
            var _this = this;
            this.throwIfDisposed();
            if (program === this.program) {
                this.program = null;
            }
            if (program != null) {
                callAndCheck(this.gl, this.debug, function () { return _this.gl.deleteProgram(program); });
            }
        };
        GPGPUContext.prototype.setProgram = function (program) {
            var _this = this;
            this.throwIfDisposed();
            this.program = program;
            if ((this.program != null) && this.debug) {
                validateProgram(this.gl, this.debug, this.program);
            }
            callAndCheck(this.gl, this.debug, function () { return _this.gl.useProgram(program); });
        };
        GPGPUContext.prototype.getUniformLocation = function (program, uniformName, shouldThrow) {
            if (shouldThrow === void 0) { shouldThrow = true; }
            this.throwIfDisposed();
            if (shouldThrow) {
                return getProgramUniformLocationOrThrow(this.gl, this.debug, program, uniformName);
            }
            else {
                return getProgramUniformLocation(this.gl, program, uniformName);
            }
        };
        GPGPUContext.prototype.getAttributeLocation = function (program, attribute) {
            var _this = this;
            this.throwIfDisposed();
            return callAndCheck(this.gl, this.debug, function () { return _this.gl.getAttribLocation(program, attribute); });
        };
        GPGPUContext.prototype.getUniformLocationNoThrow = function (program, uniformName) {
            this.throwIfDisposed();
            return this.gl.getUniformLocation(program, uniformName);
        };
        GPGPUContext.prototype.setInputMatrixTexture = function (inputMatrixTexture, uniformLocation, textureUnit) {
            this.throwIfDisposed();
            this.throwIfNoProgram();
            bindTextureToProgramUniformSampler(this.gl, this.debug, this.program, inputMatrixTexture, uniformLocation, textureUnit);
        };
        GPGPUContext.prototype.setOutputMatrixTexture = function (outputMatrixTexture, rows, columns) {
            this.setOutputMatrixTextureDriver(outputMatrixTexture, columns, rows);
        };
        GPGPUContext.prototype.setOutputPackedMatrixTexture = function (outputPackedMatrixTexture, rows, columns) {
            this.throwIfDisposed();
            var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
            this.setOutputMatrixTextureDriver(outputPackedMatrixTexture, width, height);
        };
        GPGPUContext.prototype.setOutputMatrixWriteRegion = function (startRow, numRows, startColumn, numColumns) {
            this.setOutputMatrixWriteRegionDriver(startColumn, startRow, numColumns, numRows);
        };
        GPGPUContext.prototype.setOutputPackedMatrixWriteRegion = function (startRow, numRows, startColumn, numColumns) {
            throw new Error('setOutputPackedMatrixWriteRegion not implemented.');
        };
        GPGPUContext.prototype.debugValidate = function () {
            if (this.program != null) {
                validateProgram(this.gl, this.debug, this.program);
            }
            validateFramebuffer(this.gl);
        };
        GPGPUContext.prototype.executeProgram = function () {
            this.throwIfDisposed();
            this.throwIfNoProgram();
            var gl = this.gl;
            if (this.debug) {
                this.debugValidate();
            }
            callAndCheck(gl, this.debug, function () { return gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0); });
        };
        GPGPUContext.prototype.blockUntilAllProgramsCompleted = function () {
            var _this = this;
            this.throwIfDisposed();
            callAndCheck(this.gl, this.debug, function () { return _this.gl.finish(); });
        };
        GPGPUContext.prototype.getQueryTimerExtension = function () {
            if (this.disjointQueryTimerExtension == null) {
                this.disjointQueryTimerExtension =
                    getExtensionOrThrow(this.gl, this.debug, tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2 ?
                        'EXT_disjoint_timer_query_webgl2' :
                        'EXT_disjoint_timer_query');
            }
            return this.disjointQueryTimerExtension;
        };
        GPGPUContext.prototype.getQueryTimerExtensionWebGL2 = function () {
            return this.getQueryTimerExtension();
        };
        GPGPUContext.prototype.getQueryTimerExtensionWebGL1 = function () {
            return this.getQueryTimerExtension();
        };
        GPGPUContext.prototype.beginQuery = function () {
            if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
                var gl2 = this.gl;
                var ext_1 = this.getQueryTimerExtensionWebGL2();
                var query_1 = gl2.createQuery();
                gl2.beginQuery(ext_1.TIME_ELAPSED_EXT, query_1);
                return query_1;
            }
            var ext = this.getQueryTimerExtensionWebGL1();
            var query = ext.createQueryEXT();
            ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query);
            return query;
        };
        GPGPUContext.prototype.endQuery = function () {
            if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
                var gl2 = this.gl;
                var ext_2 = this.getQueryTimerExtensionWebGL2();
                gl2.endQuery(ext_2.TIME_ELAPSED_EXT);
                return;
            }
            var ext = this.getQueryTimerExtensionWebGL1();
            ext.endQueryEXT(ext.TIME_ELAPSED_EXT);
        };
        GPGPUContext.prototype.waitForQueryAndGetTime = function (query) {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, tf.util.repeatedTry(function () { return _this.disposed || // while testing contexts are created / disposed
                                // in rapid succession, so without this check we
                                // may poll for the query timer indefinitely
                                _this.isQueryAvailable(query, tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION')); })];
                        case 1:
                            _a.sent();
                            return [2 /*return*/, this.getQueryTime(query, tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'))];
                    }
                });
            });
        };
        GPGPUContext.prototype.getQueryTime = function (query, queryTimerVersion) {
            if (queryTimerVersion === 0) {
                return null;
            }
            if (queryTimerVersion === 2) {
                var gl2 = this.gl;
                var timeElapsedNanos = gl2.getQueryParameter(query, gl2.QUERY_RESULT);
                // Return milliseconds.
                return timeElapsedNanos / 1000000;
            }
            else {
                var ext = this.getQueryTimerExtensionWebGL1();
                var timeElapsedNanos = ext.getQueryObjectEXT(query, ext.QUERY_RESULT_EXT);
                // Return milliseconds.
                return timeElapsedNanos / 1000000;
            }
        };
        GPGPUContext.prototype.isQueryAvailable = function (query, queryTimerVersion) {
            if (queryTimerVersion === 0) {
                return true;
            }
            if (queryTimerVersion === 2) {
                var gl2 = this.gl;
                var ext = this.getQueryTimerExtensionWebGL2();
                var available = gl2.getQueryParameter(query, gl2.QUERY_RESULT_AVAILABLE);
                if (this.disjoint == null) {
                    this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
                }
                return available && !this.disjoint;
            }
            else {
                var ext = this.getQueryTimerExtensionWebGL1();
                var available = ext.getQueryObjectEXT(query, ext.QUERY_RESULT_AVAILABLE_EXT);
                if (this.disjoint == null) {
                    this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
                }
                return available && !this.disjoint;
            }
        };
        GPGPUContext.prototype.pollFence = function (fenceContext) {
            var _this = this;
            return new Promise(function (resolve) {
                _this.addItemToPoll(function () { return fenceContext.isFencePassed(); }, function () { return resolve(); });
            });
        };
        GPGPUContext.prototype.pollItems = function () {
            // Find the last query that has finished.
            var index = linearSearchLastTrue(this.itemsToPoll.map(function (x) { return x.isDoneFn; }));
            for (var i = 0; i <= index; ++i) {
                var resolveFn = this.itemsToPoll[i].resolveFn;
                resolveFn();
            }
            this.itemsToPoll = this.itemsToPoll.slice(index + 1);
        };
        GPGPUContext.prototype.addItemToPoll = function (isDoneFn, resolveFn) {
            var _this = this;
            this.itemsToPoll.push({ isDoneFn: isDoneFn, resolveFn: resolveFn });
            if (this.itemsToPoll.length > 1) {
                // We already have a running loop that polls.
                return;
            }
            // Start a new loop that polls.
            tf.util.repeatedTry(function () {
                _this.pollItems();
                // End the loop if no more items to poll.
                return _this.itemsToPoll.length === 0;
            });
        };
        GPGPUContext.prototype.bindTextureToFrameBuffer = function (texture) {
            this.throwIfDisposed();
            bindColorTextureToFramebuffer(this.gl, this.debug, texture, this.framebuffer);
            if (this.debug) {
                validateFramebuffer(this.gl);
            }
        };
        GPGPUContext.prototype.unbindTextureToFrameBuffer = function () {
            if (this.outputTexture != null) {
                bindColorTextureToFramebuffer(this.gl, this.debug, this.outputTexture, this.framebuffer);
                if (this.debug) {
                    validateFramebuffer(this.gl);
                }
            }
            else {
                unbindColorTextureFromFramebuffer(this.gl, this.debug, this.framebuffer);
            }
        };
        GPGPUContext.prototype.downloadMatrixDriver = function (texture, downloadAndDecode) {
            this.bindTextureToFrameBuffer(texture);
            var result = downloadAndDecode();
            this.unbindTextureToFrameBuffer();
            return result;
        };
        GPGPUContext.prototype.setOutputMatrixTextureDriver = function (outputMatrixTextureMaybePacked, width, height) {
            this.throwIfDisposed();
            var gl = this.gl;
            bindColorTextureToFramebuffer(gl, this.debug, outputMatrixTextureMaybePacked, this.framebuffer);
            if (this.debug) {
                validateFramebuffer(gl);
            }
            this.outputTexture = outputMatrixTextureMaybePacked;
            callAndCheck(gl, this.debug, function () { return gl.viewport(0, 0, width, height); });
            callAndCheck(gl, this.debug, function () { return gl.scissor(0, 0, width, height); });
        };
        GPGPUContext.prototype.setOutputMatrixWriteRegionDriver = function (x, y, width, height) {
            var _this = this;
            this.throwIfDisposed();
            callAndCheck(this.gl, this.debug, function () { return _this.gl.scissor(x, y, width, height); });
        };
        GPGPUContext.prototype.throwIfDisposed = function () {
            if (this.disposed) {
                throw new Error('Attempted to use disposed GPGPUContext.');
            }
        };
        GPGPUContext.prototype.throwIfNoProgram = function () {
            if (this.program == null) {
                throw new Error('No GPU program is currently set.');
            }
        };
        return GPGPUContext;
    }());
    /**
     * Finds the index of the last true element using linear search.
     * Note: We can't do binary search because Chrome expects us to explicitly
     * test all fences before download:
     * https://github.com/tensorflow/tfjs/issues/1145
     */
    function linearSearchLastTrue(arr) {
        var i = 0;
        for (; i < arr.length; ++i) {
            var isDone = arr[i]();
            if (!isDone) {
                break;
            }
        }
        return i - 1;
    }

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
        var userCode = program.userCode;
        var inputInfos = inputs.map(function (input, i) {
            var shapeInfo = {
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
            return { name: program.variableNames[i], shapeInfo: shapeInfo };
        });
        var inShapeInfos = inputInfos.map(function (x) { return x.shapeInfo; });
        var outShapeInfo = {
            logicalShape: output.shape,
            texShape: output.texData.texShape,
            isUniform: false,
            isPacked: output.texData.isPacked,
            flatOffset: null
        };
        var source = makeShader(inputInfos, outShapeInfo, userCode, program.packedInputs);
        var webGLProgram = gpgpu.createProgram(source);
        // Add special uniforms (NAN, INFINITY)
        var infLoc = null;
        var nanLoc = gpgpu.getUniformLocation(webGLProgram, 'NAN', false);
        if (tf.env().getNumber('WEBGL_VERSION') === 1) {
            infLoc = gpgpu.getUniformLocation(webGLProgram, 'INFINITY', false);
        }
        // Add user-defined uniforms
        var uniformLocations = {};
        for (var i = 0; i < program.variableNames.length; i++) {
            var varName = program.variableNames[i];
            var shouldThrow = false;
            uniformLocations[varName] =
                gpgpu.getUniformLocation(webGLProgram, varName, shouldThrow);
            uniformLocations["offset" + varName] =
                gpgpu.getUniformLocation(webGLProgram, "offset" + varName, shouldThrow);
        }
        return {
            program: program,
            source: source,
            webGLProgram: webGLProgram,
            uniformLocations: uniformLocations,
            inShapeInfos: inShapeInfos,
            outShapeInfo: outShapeInfo,
            infLoc: infLoc,
            nanLoc: nanLoc,
        };
    }
    function validateBinaryAndProgram(shapeInfos, inputs) {
        if (shapeInfos.length !== inputs.length) {
            throw Error("Binary was compiled with " + shapeInfos.length + " inputs, but " +
                ("was executed with " + inputs.length + " inputs"));
        }
        shapeInfos.forEach(function (s, i) {
            var shapeA = s.logicalShape;
            var input = inputs[i];
            var shapeB = input.shape;
            if (!tf.util.arraysEqual(shapeA, shapeB)) {
                throw Error("Binary was compiled with different shapes than " +
                    ("the current args. Shapes " + shapeA + " and " + shapeB + " must match"));
            }
            // The input is uploaded as uniform.
            if (s.isUniform && input.isUniform) {
                return;
            }
            var texShapeA = s.texShape;
            var texShapeB = input.isUniform ? null : input.texData.texShape;
            if (!tf.util.arraysEqual(texShapeA, texShapeB)) {
                throw Error("Binary was compiled with different texture shapes than the" +
                    (" current args. Shape " + texShapeA + " and " + texShapeB + " must match"));
            }
        });
    }
    function runProgram(gpgpu, binary, inputs, output, customSetup) {
        validateBinaryAndProgram(binary.inShapeInfos, inputs);
        validateBinaryAndProgram([binary.outShapeInfo], [output]);
        var outTex = output.texData.texture;
        var outTexShape = output.texData.texShape;
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
        inputs.forEach(function (input, i) {
            var varName = binary.program.variableNames[i];
            var varLoc = binary.uniformLocations[varName];
            var varOffsetLoc = binary.uniformLocations["offset" + varName];
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
                    var vals = input.uniformValues;
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
        var keyInputs = '';
        inputs.concat(output).forEach(function (x) {
            var hasOffset = x.texData != null && x.texData.slice != null &&
                x.texData.slice.flatOffset > 0;
            var texShape = x.isUniform ? 'uniform' : x.texData.texShape;
            keyInputs += x.shape + "_" + texShape + "_" + hasOffset;
        });
        var keyUserCode = program.userCode;
        var key = program.constructor.name;
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
    var Im2ColPackedProgram = /** @class */ (function () {
        function Im2ColPackedProgram(outputShape, inputShape, convInfo) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = outputShape;
            var filterWidth = convInfo.filterWidth, inChannels = convInfo.inChannels, strideWidth = convInfo.strideWidth, strideHeight = convInfo.strideHeight, padInfo = convInfo.padInfo, outWidth = convInfo.outWidth, dilationWidth = convInfo.dilationWidth, dilationHeight = convInfo.dilationHeight, dataFormat = convInfo.dataFormat;
            var left = padInfo.left, top = padInfo.top;
            var itemsPerBlockRow = inChannels * filterWidth;
            var glsl = getGlslDifferences();
            var isChannelsLast = dataFormat === 'channelsLast';
            var rowDim = isChannelsLast ? 0 : 1;
            var colDim = isChannelsLast ? 1 : 2;
            var unrolled = "";
            for (var row = 0; row <= 1; row++) {
                for (var col = 0; col <= 1; col++) {
                    unrolled += "\n          blockIndex = rc.y + " + col + ";\n          pos = rc.x + " + row + ";\n\n          if(blockIndex < " + outputShape[1] + " && pos < " + outputShape[0] + ") {\n            offsetY = int(blockIndex / (" + outWidth + ")) * " + strideHeight + " - " + top + ";\n            d0 = offsetY + " + dilationHeight + " * (pos / " + itemsPerBlockRow + ");\n\n            if(d0 < " + inputShape[rowDim] + " && d0 >= 0) {\n\n              offsetX = int(mod(float(blockIndex), " + outWidth + ".) * " + strideWidth + ". - " + left + ".);\n              d1 = offsetX + " + dilationWidth + " * (int(mod(float(pos), " + itemsPerBlockRow + ".) / " + inChannels + ".));\n\n              if(d1 < " + inputShape[colDim] + " && d1 >= 0) {\n\n                ch = int(mod(float(pos), " + inChannels + ".));\n\n                if (" + isChannelsLast + ") {\n                  innerDims = vec2(d1, ch);\n                  result[" + (row * 2 + col) + "] = getChannel(\n                    getA(d0, int(innerDims.x),\n                    int(innerDims.y)), innerDims);\n                } else {\n                  innerDims = vec2(d0, d1);\n                  result[" + (row * 2 + col) + "] = getChannel(\n                    getA(ch, int(innerDims.x),\n                    int(innerDims.y)), innerDims);\n                }\n              }\n            }\n          }\n        ";
                }
            }
            this.userCode = "\n      void main() {\n        ivec2 rc = getOutputCoords();\n\n        vec4 result = vec4(0);\n\n        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;\n        vec2 innerDims;\n\n        " + unrolled + "\n\n        " + glsl.output + " = result;\n      }\n    ";
        }
        return Im2ColPackedProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var LRNProgram = /** @class */ (function () {
        function LRNProgram(xShape, radius, bias, alpha, beta) {
            this.variableNames = ['x'];
            this.outputShape = [];
            var rad = radius;
            var maxD = xShape[3] - 1;
            this.outputShape = xShape;
            // optimize pow(bias + alpha * sum, -beta)
            // src: https://github.com/tensorflow/tensorflow/..
            // blob/26033a1644a9c4a5fbe3170ab2e864b6a4ccd4ca/..
            // tensorflow/core/kernels/mkl_lrn_op.cc#L320
            var powOperator;
            var basis = "float(" + bias + ") + float(" + alpha + ") * sum";
            if (beta === 0.5) {
                powOperator = "inversesqrt(" + basis + ")";
            }
            else if (beta === 1.0) {
                powOperator = "1.0/(" + basis + ")";
            }
            else {
                powOperator = "exp(log(" + basis + ") * float(-" + beta + "));";
            }
            this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int r = coords[1];\n        int c = coords[2];\n        int d = coords[3];\n        float x = getX(b, r, c, d);\n        float sum = 0.0;\n        for (int j = -" + rad + "; j <= " + rad + "; j++) {\n          int idx = d + j;\n          if (idx >= 0 && idx <=  " + maxD + ") {\n            float z = getX(b, r, c, idx);\n            sum += z * z;\n          }\n        }\n        float val = x * " + powOperator + ";\n        setOutput(val);\n      }\n    ";
        }
        return LRNProgram;
    }());

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
    var LRNGradProgram = /** @class */ (function () {
        function LRNGradProgram(inputShape, depthRadius, bias, alpha, beta) {
            this.variableNames = ['inputImage', 'outputImage', 'dy'];
            this.outputShape = [];
            this.outputShape = inputShape;
            this.depth = inputShape[3];
            this.depthRadius = depthRadius;
            this.bias = bias;
            this.alpha = alpha;
            this.beta = beta;
            this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int r = coords[1];\n        int c = coords[2];\n\n        float result = 0.0;\n        for (int d = 0; d < " + this.depth + "; ++d) {\n          int depthBegin = int(max(0.0, float(d - " + depthRadius + ")));\n          int depthEnd = int(min(float(" + this.depth + "),\n              float(d + " + depthRadius + " + 1)));\n\n          const int MIN_DEPTH_BEGIN = 0;\n          const int MAX_DEPTH_END = " + this.depth + ";\n\n          float norm = 0.0;\n          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {\n            if (k < depthBegin){\n              continue;\n            }\n            else if (k >= depthBegin && k < depthEnd) {\n              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);\n            }\n            else {\n              break;\n            }\n          }\n\n          norm = float(" + alpha + ") * norm + float(" + bias + ");\n\n          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){\n            if (k < depthBegin){\n              continue;\n            }\n            else if (k >= depthBegin && k < depthEnd){\n              float dyi = -2.0 * float(" + alpha + ")\n                * float(" + beta + ")\n                * getInputImage(b ,r ,c, k) * getOutputImage(b, r, c, d)\n                / norm;\n              if (k == d) {\n                dyi += pow(norm, -1.0 * " + beta + ");\n              }\n              if (k == coords[3]) {\n                dyi *= getDy(b, r, c, d);\n                result += dyi;\n              }\n            }\n            else {\n              break;\n            }\n          }\n      }\n      setOutput(result);\n      }\n    ";
        }
        return LRNGradProgram;
    }());

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
    var LRNPackedProgram = /** @class */ (function () {
        function LRNPackedProgram(xShape, radius, bias, alpha, beta) {
            this.variableNames = ['x'];
            this.outputShape = [];
            this.packedInputs = true;
            this.packedOutput = true;
            var rad = radius;
            var maxD = xShape[3] - 1;
            this.outputShape = xShape;
            // optimize pow(bias + alpha * sum, -beta)
            // src: https://github.com/tensorflow/tensorflow/..
            // blob/26033a1644a9c4a5fbe3170ab2e864b6a4ccd4ca/..
            // tensorflow/core/kernels/mkl_lrn_op.cc#L320
            var powOperator;
            var basis = "float(" + bias + ") + float(" + alpha + ") * sum";
            if (beta === 0.5) {
                powOperator = "inversesqrt(" + basis + ")";
            }
            else if (beta === 1.0) {
                powOperator = "1.0/(" + basis + ")";
            }
            else {
                powOperator = "exp(log(" + basis + ") * float(-" + beta + "));";
            }
            this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords.x;\n        int r = coords.y;\n        int c = coords.z;\n        int d = coords.w;\n\n        bool hasNextCol = d < " + this.outputShape[3] + ";\n        bool hasNextRow = c < " + this.outputShape[2] + ";\n\n        vec4 sum = vec4(0.);\n        vec4 xFragAtOutputCoords = getX(b, r, c, d);\n\n        vec4 xAtOutputCoords = vec4(\n          getChannel(xFragAtOutputCoords, vec2(c, d)),\n          hasNextCol ?\n            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,\n          hasNextRow ?\n            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0\n        );\n\n        int firstChannel = d - " + rad + ";\n        vec2 cache = vec2(0.);\n        if(firstChannel >= 0){\n          vec4 firstChannelFrag = getX(b, r, c, firstChannel);\n          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));\n            if(hasNextRow){\n              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));\n            }\n        }\n\n        ivec2 depth = ivec2(d, d + 1);\n        for (int j = - " + rad + "; j <= " + rad + "; j++) {\n          ivec2 idx = depth + j;\n          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));\n          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(" + maxD + "));\n\n          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;\n          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;\n\n          if(depthInRange || depthPlusOneInRange){\n            vec4 z = vec4(0.);\n            vec4 xFragAtCurrentDepth;\n            z.xz = cache.xy;\n            if(depthPlusOneInRange && hasNextCol){\n              xFragAtCurrentDepth = idx.y != d ?\n                getX(b, r, c, idx.y) : xFragAtOutputCoords;\n              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));\n              if(hasNextRow){\n                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));\n              }\n            }\n            cache.xy = z.yw;\n            sum += z * z;\n          }\n        }\n        vec4 result = xAtOutputCoords * " + powOperator + ";\n        setOutput(result);\n      }\n    ";
        }
        return LRNPackedProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var MaxPool2DBackpropProgram = /** @class */ (function () {
        function MaxPool2DBackpropProgram(convInfo) {
            this.variableNames = ['dy', 'maxPos'];
            this.outputShape = convInfo.inShape;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationHeight = convInfo.dilationHeight;
            var effectiveFilterHeight = convInfo.effectiveFilterHeight;
            var effectiveFilterWidth = convInfo.effectiveFilterWidth;
            var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
            var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
            var lastIndex = effectiveFilterHeight * effectiveFilterWidth - 1;
            this.userCode = "\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n\n        ivec2 dyRCCorner = coords.yz - pads;\n        int dyRCorner = dyRCCorner.x;\n        int dyCCorner = dyRCCorner.y;\n\n        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < " + effectiveFilterHeight + ";\n          wR += " + dilationHeight + ") {\n          float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n          if (dyR < 0.0 || dyR >= " + convInfo.outHeight + ".0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          for (int wC = 0; wC < " + effectiveFilterWidth + "; wC++) {\n            float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n            if (dyC < 0.0 || dyC >= " + convInfo.outWidth + ".0 ||\n                fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            float dyValue = getDy(b, idyR, idyC, d);\n            int maxPosValue = " + lastIndex + " - int(getMaxPos(b, idyR, idyC, d));\n\n            // Get the current value, check it against the value from the\n            // position matrix.\n            int curPosValue = wR * " + effectiveFilterWidth + " + wC;\n            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);\n\n            dotProd += dyValue * mask;\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return MaxPool2DBackpropProgram;
    }());
    var MaxPool3DBackpropProgram = /** @class */ (function () {
        function MaxPool3DBackpropProgram(convInfo) {
            this.variableNames = ['dy', 'maxPos'];
            this.outputShape = convInfo.inShape;
            var strideDepth = convInfo.strideDepth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationDepth = convInfo.dilationDepth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var effectiveFilterDepth = convInfo.effectiveFilterDepth;
            var effectiveFilterHeight = convInfo.effectiveFilterHeight;
            var effectiveFilterWidth = convInfo.effectiveFilterWidth;
            var padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
            var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
            var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
            var lastIndex = effectiveFilterDepth * effectiveFilterHeight * effectiveFilterWidth - 1;
            this.userCode = "\n      const ivec3 pads = ivec3(" + padFront + ", " + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int ch = coords.u;\n\n        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;\n        int dyDCorner = dyCorner.x;\n        int dyRCorner = dyCorner.y;\n        int dyCCorner = dyCorner.z;\n\n        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get\n        // dx(xD, xR, xC, ch).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n\n        for (int wD = 0; wD < " + effectiveFilterDepth + ";\n           wD += " + dilationDepth + ") {\n          float dyD = float(dyDCorner + wD) / " + strideDepth + ".0;\n\n          if (dyD < 0.0 || dyD >= " + convInfo.outDepth + ".0 || fract(dyD) > 0.0) {\n            continue;\n          }\n          int idyD = int(dyD);\n\n          for (int wR = 0; wR < " + effectiveFilterHeight + ";\n              wR += " + dilationHeight + ") {\n            float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n            if (dyR < 0.0 || dyR >= " + convInfo.outHeight + ".0 ||\n                fract(dyR) > 0.0) {\n              continue;\n            }\n            int idyR = int(dyR);\n\n            for (int wC = 0; wC < " + effectiveFilterWidth + ";\n                wC += " + dilationWidth + ") {\n              float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n              if (dyC < 0.0 || dyC >= " + convInfo.outWidth + ".0 ||\n                  fract(dyC) > 0.0) {\n                continue;\n              }\n              int idyC = int(dyC);\n\n              float dyValue = getDy(batch, idyD, idyR, idyC, ch);\n              int maxPosValue = " + lastIndex + " -\n                  int(getMaxPos(batch, idyD, idyR, idyC, ch));\n\n              // Get the current value, check it against the value from the\n              // position matrix.\n              int curPosValue =\n                  wD * " + effectiveFilterHeight + " * " + effectiveFilterWidth + " +\n                  wR * " + effectiveFilterWidth + " + wC;\n              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);\n\n              dotProd += dyValue * mask;\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
        }
        return MaxPool3DBackpropProgram;
    }());

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
    var MatMulPackedProgram = /** @class */ (function () {
        function MatMulPackedProgram(aShape, outputShape, transposeA, transposeB, addBias, activation, hasPreluActivation) {
            if (transposeA === void 0) { transposeA = false; }
            if (transposeB === void 0) { transposeB = false; }
            if (addBias === void 0) { addBias = false; }
            if (activation === void 0) { activation = null; }
            if (hasPreluActivation === void 0) { hasPreluActivation = false; }
            this.variableNames = ['matrixA', 'matrixB'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = outputShape;
            var sharedDim = transposeA ? aShape[1] : aShape[2];
            var sharedDimensionPacked = Math.ceil(sharedDim / 2);
            var aSample = transposeA ? 'i * 2, rc.y' : 'rc.y, i * 2';
            var bSample = transposeB ? 'rc.z, i * 2' : 'i * 2, rc.z';
            var aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
            var bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];
            var activationSnippet = '', applyActivationSnippet = '';
            if (activation) {
                if (hasPreluActivation) {
                    activationSnippet = "vec4 activation(vec4 a) {\n          vec4 b = getPreluActivationWeightsAtOutCoords();\n          " + activation + "\n        }";
                }
                else {
                    activationSnippet = "vec4 activation(vec4 x) {\n          " + activation + "\n        }";
                }
                applyActivationSnippet = "result = activation(result);";
            }
            var addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
            if (addBias) {
                this.variableNames.push('bias');
            }
            if (hasPreluActivation) {
                this.variableNames.push('preluActivationWeights');
            }
            this.userCode = "\n      " + activationSnippet + "\n\n      const float sharedDimension = " + sharedDimensionPacked + ".0;\n\n      vec4 dot2x2ARowBCol(ivec3 rc) {\n        vec4 result = vec4(0);\n        for (int i = 0; i < " + sharedDimensionPacked + "; i++) {\n          vec4 a = getMatrixA(rc.x, " + aSample + ");\n          vec4 b = getMatrixB(rc.x, " + bSample + ");\n\n          // These swizzled products need to be separately added.\n          // See: https://github.com/tensorflow/tfjs/issues/1735\n          result += (" + aSwizzle[0] + " * " + bSwizzle[0] + ");\n          result += (" + aSwizzle[1] + " * " + bSwizzle[1] + ");\n        }\n        return result;\n      }\n\n      void main() {\n        ivec3 rc = getOutputCoords();\n        vec4 result = dot2x2ARowBCol(rc);\n\n        " + addBiasSnippet + "\n\n        " + applyActivationSnippet + "\n\n        setOutput(result);\n      }\n    ";
        }
        return MatMulPackedProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var MultinomialProgram = /** @class */ (function () {
        function MultinomialProgram(batchSize, numOutcomes, numSamples) {
            this.variableNames = ['probs'];
            this.outputShape = [batchSize, numSamples];
            this.userCode = "\n      uniform float seed;\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int batch = coords[0];\n\n        float r = random(seed);\n        float cdf = 0.0;\n\n        for (int i = 0; i < " + (numOutcomes - 1) + "; i++) {\n          cdf += getProbs(batch, i);\n\n          if (r < cdf) {\n            setOutput(float(i));\n            return;\n          }\n        }\n\n        // If no other event happened, last event happened.\n        setOutput(float(" + (numOutcomes - 1) + "));\n      }\n    ";
        }
        MultinomialProgram.prototype.getCustomSetupFunc = function (seed) {
            var _this = this;
            return function (gpgpu, webGLProgram) {
                if (_this.seedLoc == null) {
                    _this.seedLoc = gpgpu.getUniformLocation(webGLProgram, 'seed');
                }
                gpgpu.gl.uniform1f(_this.seedLoc, seed);
            };
        };
        return MultinomialProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var OneHotProgram = /** @class */ (function () {
        function OneHotProgram(numIndices, depth, onValue, offValue) {
            this.variableNames = ['indices'];
            this.outputShape = [numIndices, depth];
            this.userCode = "\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int index = round(getIndices(coords.x));\n        setOutput(mix(float(" + offValue + "), float(" + onValue + "),\n                      float(index == coords.y)));\n      }\n    ";
        }
        return OneHotProgram;
    }());

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
    var PackProgram = /** @class */ (function () {
        function PackProgram(outputShape) {
            this.variableNames = ['A'];
            this.packedInputs = false;
            this.packedOutput = true;
            // Only input / output 3D tensors.
            this.outputShape = outputShape;
            var rank = outputShape.length;
            if (rank === 0) {
                this.userCode = "\n        void main() {\n          setOutput(vec4(getA(), 0., 0., 0.));\n        }\n      ";
            }
            else {
                var channels = getChannels('rc', rank);
                var dtype = getCoordsDataType(rank);
                var outOfBoundsCondition = getOutOfBoundsCondition(rank, outputShape, channels);
                var setup = getSetup(rank, outputShape[outputShape.length - 1], outputShape[outputShape.length - 2], channels);
                var output = getOutput(outputShape, channels);
                this.userCode = "\n        void main() {\n          " + dtype + " rc = getOutputCoords();\n\n          if(" + outOfBoundsCondition + ") {\n            setOutput(vec4(0));\n          } else {\n            " + setup + "\n\n            setOutput(vec4(" + output + "));\n          }\n        }\n      ";
            }
        }
        return PackProgram;
    }());
    function getSourceCoordsArr(rank, dims) {
        var coords = [];
        for (var row = 0; row <= 1; row++) {
            for (var col = 0; col <= 1; col++) {
                var coord = (row === 0 ? 'r' : 'rp1') + ", " + (col === 0 ? 'c' : 'cp1');
                for (var d = 2; d < rank; d++) {
                    coord = dims[dims.length - 1 - d] + "," + coord;
                }
                coords.push(coord);
            }
        }
        return coords;
    }
    function getOutOfBoundsCondition(rank, shape, dims) {
        if (rank === 1) {
            return "rc > " + shape[0];
        }
        var cond = '';
        for (var i = rank - 2; i < rank; i++) {
            cond += dims[i] + " >= " + shape[i];
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
        var innerDims = dims.slice(-2);
        return "\n    int r = " + innerDims[0] + ";\n    int c = " + innerDims[1] + ";\n    int rp1 = r + 1;\n    int cp1 = c + 1;\n\n    bool cEdge = cp1 >= " + cols + ";\n    bool rEdge = rp1 >= " + rows + ";\n  ";
    }
    function getOutput(shape, dims) {
        var rank = shape.length;
        var sourceCoords = getSourceCoordsArr(rank, dims);
        if (rank === 1) {
            return "getA(rc),\n            rc + 1 >= " + shape[0] + " ? 0. : getA(rc + 1),\n            0, 0";
        }
        return "getA(" + sourceCoords[0] + "),\n          cEdge ? 0. : getA(" + sourceCoords[1] + "),\n          rEdge ? 0. : getA(" + sourceCoords[2] + "),\n          rEdge || cEdge ? 0. : getA(" + sourceCoords[3] + ")";
    }

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var PadProgram = /** @class */ (function () {
        function PadProgram(xShape, paddings, constantValue) {
            this.variableNames = ['x'];
            this.outputShape = paddings.map(function (p, i) { return p[0] /* beforePad */ + xShape[i] + p[1]; } /* afterPad */);
            var rank = xShape.length;
            var type = getCoordsDataType(rank);
            var start = paddings.map(function (p) { return p[0]; }).join(',');
            var end = paddings.map(function (p, i) { return p[0] + xShape[i]; }).join(',');
            var unpackedCoords = ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank);
            if (rank === 1) {
                this.userCode = "\n        int start = " + start + ";\n        int end = " + end + ";\n\n        void main() {\n          int outC = getOutputCoords();\n          if (outC < start || outC >= end) {\n            setOutput(float(" + constantValue + "));\n          } else {\n            setOutput(getX(outC - start));\n          }\n        }\n      ";
                return;
            }
            this.userCode = "\n      " + type + " start = " + type + "(" + start + ");\n      " + type + " end = " + type + "(" + end + ");\n\n      void main() {\n        " + type + " outC = getOutputCoords();\n        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {\n          setOutput(float(" + constantValue + "));\n        } else {\n          " + type + " coords = outC - start;\n          setOutput(getX(" + unpackedCoords + "));\n        }\n      }\n    ";
        }
        return PadProgram;
    }());

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
    var PadPackedProgram = /** @class */ (function () {
        function PadPackedProgram(xShape, paddings, constantValue) {
            this.variableNames = ['x'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = paddings.map(function (p, i) { return p[0] /* beforePad */ + xShape[i] + p[1]; } /* afterPad */);
            var rank = xShape.length;
            var dtype = getCoordsDataType(rank);
            var start = paddings.map(function (p) { return p[0]; }).join(',');
            var end = paddings.map(function (p, i) { return p[0] + xShape[i]; }).join(',');
            var coords = getChannels('rc', rank);
            var source = getChannels('source', rank);
            var cLimit = coords[rank - 1] + " < " + this.outputShape[rank - 1];
            var innerDims = rank === 1 ? 'source' : "vec2(" + source.slice(-2).join() + ")";
            var componentSetup = [
                dtype + " rc = outputLoc;", coords[rank - 1] + " += 1;\n       if(" + cLimit + ") {\n      ",
                rank === 1 ? '' : "}\n       rc = outputLoc;\n       " + coords[rank - 2] + " += 1;\n       if(" + coords[rank - 2] + " < " + this.outputShape[rank - 2] + ") {",
                rank === 1 ? '' : "  " + coords[rank - 1] + " += 1;\n         if(" + cLimit + ") {"
            ];
            var paddingArea = rank === 1 ?
                'rc < start || rc >= end' :
                'any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))';
            var mainLoop = '';
            for (var i = 0, j = rank === 1 ? 2 : 4; i < j; i++) {
                mainLoop += "\n        " + componentSetup[i] + "\n        if (" + paddingArea + ") {\n          result[" + i + "] = float(" + constantValue + ");\n        } else {\n          " + dtype + " source = rc - start;\n          result[" + i + "] = getChannel(getX(" + source.join() + "), " + innerDims + ");\n        }\n      ";
            }
            mainLoop += (rank === 1 ? "} " : "}}");
            this.userCode = "\n      const " + dtype + " start = " + dtype + "(" + start + ");\n      const " + dtype + " end = " + dtype + "(" + end + ");\n\n      void main() {\n        " + dtype + " outputLoc = getOutputCoords();\n        vec4 result = vec4(0.);\n        " + mainLoop + "\n        setOutput(result);\n      }\n    ";
        }
        return PadPackedProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var Pool2DProgram = /** @class */ (function () {
        function Pool2DProgram(convInfo, poolType, computePositions, flattenPositions, includeBatchInIndex) {
            if (flattenPositions === void 0) { flattenPositions = false; }
            if (includeBatchInIndex === void 0) { includeBatchInIndex = false; }
            this.variableNames = ['x'];
            if (poolType === 'avg' && computePositions) {
                throw new Error('Cannot compute positions for average pool.');
            }
            var filterWidth = convInfo.filterWidth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var effectiveFilterHeight = convInfo.effectiveFilterHeight;
            var effectiveFilterWidth = convInfo.effectiveFilterWidth;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            this.outputShape = convInfo.outShape;
            var isAvgPool = poolType === 'avg';
            var batchFlattenPositionStr = "((batch  * " + convInfo.inHeight + " + xR) * " + convInfo.inWidth + " + xC) * " + convInfo.inChannels + " + d";
            var flattenPositionStr = "(xR * " + convInfo.inWidth + " + xC) * " + convInfo.inChannels + " + d";
            var initializationValue = '0.0';
            if (!isAvgPool) {
                // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
                initializationValue = '-1.0 / 1e-20';
            }
            if (computePositions) {
                var compareOp_1 = '>=';
                this.userCode = "\n        const ivec2 strides = ivec2(" + strideHeight + ", " + strideWidth + ");\n        const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n        void main() {\n          ivec4 coords = getOutputCoords();\n          int batch = coords[0];\n          int d = coords[3];\n\n          ivec2 xRCCorner = coords.yz * strides - pads;\n          int xRCorner = xRCCorner.x;\n          int xCCorner = xRCCorner.y;\n\n          // max/min x(?, ?, d) to get y(yR, yC, d).\n          // ? = to be determined\n          float minMaxValue = 0.0;\n          float minMaxValueFound = 0.0;\n          int minMaxPosition = 0;\n          float avgValue = 0.0;\n\n          for (int wR = 0; wR < " + effectiveFilterHeight + ";\n              wR += " + dilationHeight + ") {\n            int xR = xRCorner + wR;\n\n            if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n              continue;\n            }\n\n            for (int wC = 0; wC < " + effectiveFilterWidth + ";\n                wC += " + dilationWidth + ") {\n              int xC = xCCorner + wC;\n\n              if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n                continue;\n              }\n\n              float value = getX(batch, xR, xC, d);\n\n              // If a min / max value has already been found, use it. If not,\n              // use the current value.\n              float currMinMaxValue = mix(\n                  value, minMaxValue, minMaxValueFound);\n              if (value " + compareOp_1 + " currMinMaxValue) {\n                minMaxValue = value;\n                minMaxValueFound = 1.0;\n                minMaxPosition = " + (flattenPositions ? (includeBatchInIndex ? batchFlattenPositionStr :
                    flattenPositionStr) :
                    "wR * " + effectiveFilterWidth + " + wC") + ";\n              }\n            }\n          }\n          setOutput(float(minMaxPosition));\n        }\n      ";
                return;
            }
            var compareOp = 'max';
            var returnValue = poolType + "(" + poolType + "(" + poolType + "(" +
                'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
            if (poolType === 'avg') {
                returnValue = "avgValue / count";
            }
            var filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
            var filterWidthVec4Remainder = filterWidth % 4;
            var updateSnippet = "\n      if (" + isAvgPool + ") {\n        avgValue += dot(values, ones);\n      } else {\n        minMaxValue = " + compareOp + "(values, minMaxValue);\n      }\n    ";
            this.userCode = "\n      const ivec2 strides = ivec2(" + strideHeight + ", " + strideWidth + ");\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n      const float initializationValue = " + initializationValue + ";\n      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);\n\n      float count = 0.0;\n\n      float getValue(int batch, int xR, int xC, int d) {\n        if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n          return initializationValue;\n        }\n        count += 1.0;\n        return getX(batch, xR, xC, d);\n      }\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords[0];\n        int d = coords[3];\n\n        ivec2 xRCCorner = coords.yz * strides - pads;\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        // max/min x(?, ?, d) to get y(yR, yC, d).\n        // ? = to be determined\n        vec4 minMaxValue = vec4(" + initializationValue + ");\n        float avgValue = 0.0;\n        count = 0.0;\n\n        for (int wR = 0; wR < " + effectiveFilterHeight + ";\n            wR += " + dilationHeight + ") {\n          int xR = xRCorner + wR;\n\n          if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n            continue;\n          }\n\n          for (int wC = 0; wC < " + filterWidthNearestVec4 + "; wC += 4) {\n            int xC = xCCorner + wC * " + dilationWidth + ";\n\n            vec4 values = vec4(\n              getValue(batch, xR, xC, d),\n              getValue(batch, xR, xC + " + dilationWidth + ", d),\n              getValue(batch, xR, xC + 2 * " + dilationWidth + ", d),\n              getValue(batch, xR, xC + 3 * " + dilationWidth + ", d)\n            );\n\n            " + updateSnippet + "\n          }\n\n          int xC = xCCorner + " + filterWidthNearestVec4 + ";\n          if (" + (filterWidthVec4Remainder === 1) + ") {\n            vec4 values = vec4(\n              getValue(batch, xR, xC, d),\n              initializationValue,\n              initializationValue,\n              initializationValue\n            );\n\n            " + updateSnippet + "\n          } else if (" + (filterWidthVec4Remainder === 2) + ") {\n            vec4 values = vec4(\n              getValue(batch, xR, xC, d),\n              getValue(batch, xR, xC + " + dilationWidth + ", d),\n              initializationValue,\n              initializationValue\n            );\n\n            " + updateSnippet + "\n          } else if (" + (filterWidthVec4Remainder === 3) + ") {\n            vec4 values = vec4(\n              getValue(batch, xR, xC, d),\n              getValue(batch, xR, xC + " + dilationWidth + ", d),\n              getValue(batch, xR, xC + 2 * " + dilationWidth + ", d),\n              initializationValue\n            );\n\n            " + updateSnippet + "\n          }\n        }\n        setOutput(" + returnValue + ");\n      }\n    ";
        }
        return Pool2DProgram;
    }());
    var Pool3DProgram = /** @class */ (function () {
        function Pool3DProgram(convInfo, poolType, computePositions, flattenPositions, includeBatchInIndex) {
            if (flattenPositions === void 0) { flattenPositions = false; }
            if (includeBatchInIndex === void 0) { includeBatchInIndex = false; }
            this.variableNames = ['x'];
            if (poolType === 'avg' && computePositions) {
                throw new Error('Cannot compute positions for average pool.');
            }
            var filterWidth = convInfo.filterWidth;
            var strideDepth = convInfo.strideDepth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var dilationDepth = convInfo.dilationDepth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var effectiveFilterDepth = convInfo.effectiveFilterDepth;
            var effectiveFilterHeight = convInfo.effectiveFilterHeight;
            var effectiveFilterWidth = convInfo.effectiveFilterWidth;
            var padFront = convInfo.padInfo.front;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;
            this.outputShape = convInfo.outShape;
            var isAvgPool = poolType === 'avg';
            var initializationValue = '0.0';
            if (!isAvgPool) {
                // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
                initializationValue = '-1.0 / 1e-20';
            }
            if (computePositions) {
                var compareOp_2 = '>=';
                this.userCode = "\n        const ivec3 strides =\n            ivec3(" + strideDepth + ", " + strideHeight + ", " + strideWidth + ");\n        const ivec3 pads = ivec3(" + padFront + ", " + padTop + ", " + padLeft + ");\n\n        void main() {\n          ivec5 coords = getOutputCoords();\n          int batch = coords.x;\n          int ch = coords.u;\n\n          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;\n          int xDCorner = xCorner.x;\n          int xRCorner = xCorner.y;\n          int xCCorner = xCorner.z;\n\n          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).\n          // ? = to be determined\n          float minMaxValue = 0.0;\n          float minMaxValueFound = 0.0;\n          int minMaxPosition = 0;\n\n          for (int wD = 0; wD < " + effectiveFilterDepth + ";\n              wD += " + dilationDepth + ") {\n            int xD = xDCorner + wD;\n\n            if (xD < 0 || xD >= " + convInfo.inDepth + ") {\n              continue;\n            }\n\n            for (int wR = 0; wR < " + effectiveFilterHeight + ";\n                wR += " + dilationHeight + ") {\n              int xR = xRCorner + wR;\n\n              if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n                continue;\n              }\n\n              for (int wC = 0; wC < " + effectiveFilterWidth + ";\n                  wC += " + dilationWidth + ") {\n                int xC = xCCorner + wC;\n\n                if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n                  continue;\n                }\n\n                float value = getX(batch, xD, xR, xC, ch);\n\n                // If a min / max value has already been found, use it. If not,\n                // use the current value.\n                float currMinMaxValue = mix(\n                    value, minMaxValue, minMaxValueFound);\n                if (value " + compareOp_2 + " currMinMaxValue) {\n                  minMaxValue = value;\n                  minMaxValueFound = 1.0;\n                  minMaxPosition = " + (flattenPositions ?
                    (includeBatchInIndex ?
                        "(((batch * " + convInfo.inDepth + " + xD) * " + convInfo.inHeight + " + xR) * " + convInfo.inWidth + " + xC) * " + convInfo.inChannels + " + ch" :
                        "((xD * " + convInfo.inHeight + " + xR) * " + convInfo.inWidth + " + xC) * " + convInfo.inChannels + " + ch") :
                    "wD * " + effectiveFilterHeight + " * " + effectiveFilterWidth + " +\n                      wR * " + effectiveFilterWidth + " + wC") + ";\n                }\n              }\n            }\n          }\n          setOutput(float(minMaxPosition));\n        }\n      ";
                return;
            }
            var compareOp = 'max';
            var returnValue = poolType + "(" + poolType + "(" + poolType + "(" +
                'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
            if (poolType === 'avg') {
                returnValue = "avgValue / count";
            }
            var filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
            var filterWidthVec4Remainder = filterWidth % 4;
            var updateSnippet = "\n      if (" + isAvgPool + ") {\n        avgValue += dot(values, ones);\n      } else {\n        minMaxValue = " + compareOp + "(values, minMaxValue);\n      }\n    ";
            this.userCode = "\n      const ivec3 strides =\n        ivec3(" + strideDepth + ", " + strideHeight + ", " + strideWidth + ");\n      const ivec3 pads = ivec3(" + padFront + ", " + padTop + ", " + padLeft + ");\n      const float initializationValue = " + initializationValue + ";\n      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);\n\n      float count = 0.0;\n\n      float getValue(int batch, int xD, int xR, int xC, int ch) {\n        if (xC < 0 || xC >= " + convInfo.inWidth + ") {\n          return initializationValue;\n        }\n        count += 1.0;\n        return getX(batch, xD, xR, xC, ch);\n      }\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int ch = coords.u;\n\n        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;\n        int xDCorner = xCorner.x;\n        int xRCorner = xCorner.y;\n        int xCCorner = xCorner.z;\n\n        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).\n        // ? = to be determined\n        vec4 minMaxValue = vec4(" + initializationValue + ");\n        float avgValue = 0.0;\n        count = 0.0;\n\n        for (int wD = 0; wD < " + effectiveFilterDepth + ";\n            wD += " + dilationDepth + ") {\n          int xD = xDCorner + wD;\n\n          if (xD < 0 || xD >= " + convInfo.inDepth + ") {\n            continue;\n          }\n\n          for (int wR = 0; wR < " + effectiveFilterHeight + ";\n            wR += " + dilationHeight + ") {\n            int xR = xRCorner + wR;\n\n            if (xR < 0 || xR >= " + convInfo.inHeight + ") {\n              continue;\n            }\n\n            for (int wC = 0; wC < " + filterWidthNearestVec4 + "; wC += 4) {\n              int xC = xCCorner + wC * " + dilationWidth + ";\n\n              vec4 values = vec4(\n                getValue(batch, xD, xR, xC, ch),\n                getValue(batch, xD, xR, xC + " + dilationWidth + ", ch),\n                getValue(batch, xD, xR, xC + 2 * " + dilationWidth + ", ch),\n                getValue(batch, xD, xR, xC + 3 * " + dilationWidth + ", ch)\n              );\n\n              " + updateSnippet + "\n            }\n\n            int xC = xCCorner + " + filterWidthNearestVec4 + ";\n            if (" + (filterWidthVec4Remainder === 1) + ") {\n              vec4 values = vec4(\n                getValue(batch, xD, xR, xC, ch),\n                initializationValue,\n                initializationValue,\n                initializationValue\n              );\n\n              " + updateSnippet + "\n            } else if (" + (filterWidthVec4Remainder === 2) + ") {\n              vec4 values = vec4(\n                getValue(batch, xD, xR, xC, ch),\n                getValue(batch, xD, xR, xC + " + dilationWidth + ", ch),\n                initializationValue,\n                initializationValue\n              );\n\n              " + updateSnippet + "\n            } else if (" + (filterWidthVec4Remainder === 3) + ") {\n              vec4 values = vec4(\n                getValue(batch, xD, xR, xC, ch),\n                getValue(batch, xD, xR, xC + " + dilationWidth + ", ch),\n                getValue(batch, xD, xR, xC + 2 * " + dilationWidth + ", ch),\n                initializationValue\n              );\n\n              " + updateSnippet + "\n            }\n          }\n          setOutput(" + returnValue + ");\n        }\n      }\n    ";
        }
        return Pool3DProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var ReduceProgram = /** @class */ (function () {
        function ReduceProgram(reduceInfo, reduceType) {
            this.variableNames = ['x'];
            var windowSize = reduceInfo.windowSize;
            var batchSize = reduceInfo.batchSize;
            var inSize = reduceInfo.inSize;
            var outSize = Math.ceil(inSize / windowSize);
            this.outputShape = [batchSize, outSize];
            var initializationValue = '0.0';
            var compareOp = "";
            if (reduceType === 'prod') {
                initializationValue = '1.0';
            }
            else if (reduceType === 'min') {
                // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
                initializationValue = '1.0 / 1e-20';
                compareOp = "min";
            }
            else if (reduceType === 'max') {
                // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
                initializationValue = '-1.0 / 1e-20';
                compareOp = "max";
            }
            var returnValue = reduceType + "(" + reduceType + "(" + reduceType + "(" +
                'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
            if (reduceType === 'sum') {
                returnValue = "sumValue";
            }
            else if (reduceType === 'prod') {
                returnValue = "prodValue";
            }
            else if (reduceType === 'all') {
                returnValue = "allValue";
            }
            else if (reduceType === 'any') {
                returnValue = "anyValue";
            }
            var windowSizeNearestVec4 = Math.floor(windowSize / 4) * 4;
            var windowSizeVec4Remainder = windowSize % 4;
            var updateSnippet = "\n      if (" + (reduceType === 'sum') + ") {\n        sumValue += dot(values, ones);\n      } else if (" + (reduceType === 'prod') + ") {\n        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);\n        prodValue *= tmp[0] * tmp[1];\n      } else {\n        minMaxValue = " + compareOp + "(values, minMaxValue);\n      }\n    ";
            var vecType = "vec4";
            if (reduceType === 'all') {
                initializationValue = '1.0';
                updateSnippet = "\n        bool reducedAllValue = all(values);\n        float floatedReducedAllValue = float(reducedAllValue);\n        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);\n      ";
                vecType = "bvec4";
            }
            else if (reduceType === 'any') {
                initializationValue = '0.0';
                updateSnippet = "\n        bool reducedAnyValue = any(values);\n        float floatedReducedAnyValue = float(reducedAnyValue);\n        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);\n      ";
                vecType = "bvec4";
            }
            var checkOutOfBounds = '';
            if (inSize % windowSize > 0) {
                checkOutOfBounds = "\n        if (inIdx < 0 || inIdx >= " + inSize + ") {\n          return initializationValue;\n        }\n      ";
            }
            this.userCode = "\n      const float initializationValue = " + initializationValue + ";\n      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);\n\n      float getValue(int batch, int inIdx) {\n        " + checkOutOfBounds + "\n        return getX(batch, inIdx);\n      }\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int batch = coords[0];\n        int outIdx = coords[1];\n        int inOffset = outIdx * " + windowSize + ";\n\n        vec4 minMaxValue = vec4(" + initializationValue + ");\n        float prodValue = 1.0;\n        float sumValue = 0.0;\n        float allValue = 1.0;\n        float anyValue = 0.0;\n\n        for (int i = 0; i < " + windowSizeNearestVec4 + "; i += 4) {\n          int inIdx = inOffset + i;\n          " + vecType + " values = " + vecType + "(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            getValue(batch, inIdx + 2),\n            getValue(batch, inIdx + 3)\n          );\n\n          " + updateSnippet + "\n        }\n\n        int inIdx = inOffset + " + windowSizeNearestVec4 + ";\n        if (" + (windowSizeVec4Remainder === 1) + ") {\n          " + vecType + " values = " + vecType + "(\n            getValue(batch, inIdx),\n            initializationValue,\n            initializationValue,\n            initializationValue\n          );\n\n          " + updateSnippet + "\n        } else if (" + (windowSizeVec4Remainder === 2) + ") {\n          " + vecType + " values = " + vecType + "(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            initializationValue,\n            initializationValue\n          );\n\n          " + updateSnippet + "\n        } else if (" + (windowSizeVec4Remainder === 3) + ") {\n          " + vecType + " values = " + vecType + "(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            getValue(batch, inIdx + 2),\n            initializationValue\n          );\n\n          " + updateSnippet + "\n        }\n        setOutput(" + returnValue + ");\n      }\n    ";
        }
        return ReduceProgram;
    }());

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
    var ReshapePackedProgram = /** @class */ (function () {
        function ReshapePackedProgram(outputShape, inputShape) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = outputShape;
            var mainLoop = "";
            for (var i = 0; i < 4; i++) {
                var thisRC = "thisRC = rc;";
                if (i % 2 === 1) {
                    thisRC += "thisRC.z += 1;";
                }
                if (i > 1) {
                    thisRC += "thisRC.y += 1;";
                }
                mainLoop += "\n        " + thisRC + "\n        " + (i > 0 ? "if(thisRC.y < rows && thisRC.z < cols){" : '') + "\n          int flatIndex = getFlatIndex(thisRC);\n\n          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);\n          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));\n\n          result[" + i + "] =\n            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);\n        " + (i > 0 ? '}' : '') + "\n      ";
            }
            this.userCode = "\n      " + getReshapedInputCoords(inputShape) + "\n      " + getFlatIndexFrom3D(outputShape) + "\n\n      void main() {\n        ivec3 rc = getOutputCoords();\n\n        vec4 result = vec4(0.);\n\n        ivec3 thisRC;\n        int rows = " + outputShape[1] + ";\n        int cols = " + outputShape[2] + ";\n\n        " + mainLoop + "\n\n        setOutput(result);\n      }\n    ";
        }
        return ReshapePackedProgram;
    }());
    function getReshapedInputCoords(shape) {
        var coordsFromIndexSnippet = getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);
        return "\n    ivec3 inputCoordsFromReshapedOutCoords(int index) {\n      " + coordsFromIndexSnippet + "\n      return ivec3(r, c, d);\n    }\n  ";
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
    var ResizeBilinearBackpropProgram = /** @class */ (function () {
        function ResizeBilinearBackpropProgram(dy, x, alignCorners) {
            this.variableNames = ['dy'];
            this.outputShape = [];
            this.outputShape = x.shape;
            var _a = x.shape, xHeight = _a[1], xWidth = _a[2];
            var _b = dy.shape, yHeight = _b[1], yWidth = _b[2];
            // In the backwards pass, we want to find the pixels that were generated for
            // each pixel in the input image the forward pass and add the corresponding
            // coefficient from dy to the gradient (with some interpolation).
            var effectiveXSize = [
                (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
                (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
            ];
            var effectiveYSize = [
                (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
                (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
            ];
            var heightScale = effectiveXSize[0] / effectiveYSize[0];
            var widthScale = effectiveXSize[1] / effectiveYSize[1];
            var invHeightScale = 1 / heightScale;
            var invWidthScale = 1 / widthScale;
            // This defines the size of the window of values around a particular
            // index in dy that we want to search for contributions to dx.
            var winHeight = (Math.ceil(invHeightScale) * 2) + 2;
            var winWidth = (Math.ceil(invWidthScale) * 2) + 2;
            this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        int r = coords[1];\n        int c = coords[2];\n\n        float accumulator = 0.0;\n\n        const float heightScale = float(" + heightScale + ");\n        const float widthScale = float(" + widthScale + ");\n\n        const float invHeightScale = float(" + invHeightScale + ");\n        const float invWidthScale = float(" + invWidthScale + ");\n\n        const int winHeight = int(" + winHeight + ");\n        const int winWidth = int(" + winWidth + ");\n\n        // Compute bounds for where in dy we will look\n        float startRLerp = floor(float(r) * invHeightScale);\n        int startDyR = int(startRLerp - float(winHeight / 2));\n\n        float startCLerp = floor(float(c) * invWidthScale);\n        int startDyC = int(startCLerp - float(winWidth / 2));\n\n        // Loop over dy\n        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {\n          int dyR = dyROffset + startDyR;\n\n          // Guard against the window exceeding the bounds of dy\n          if (dyR < 0 || dyR >= " + yHeight + ") {\n            continue;\n          }\n\n          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {\n            int dyC = dyCOffset + startDyC;\n\n            // Guard against the window exceeding the bounds of dy\n            if (dyC < 0 || dyC >= " + yWidth + ") {\n              continue;\n            }\n\n            float dxR = float(dyR) * heightScale;\n            int topDxRIndex = int(floor(dxR));\n            int bottomDxRIndex = int(min(ceil(dxR), " + (xHeight - 1) + ".0));\n            float dxRLerp = dxR - float(topDxRIndex);\n            float inverseDxRLerp = 1.0 - dxRLerp;\n\n            float dxC = float(dyC) * widthScale;\n            int leftDxCIndex = int(floor(dxC));\n            int rightDxCIndex = int(min(ceil(dxC), " + (xWidth - 1) + ".0));\n            float dxCLerp = dxC - float(leftDxCIndex);\n            float inverseDxCLerp = 1.0 - dxCLerp;\n\n            if (r == topDxRIndex && c == leftDxCIndex) {\n              // topLeft\n              accumulator +=\n                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;\n            }\n\n            if (r == topDxRIndex && c == rightDxCIndex) {\n              // topRight\n              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;\n            }\n\n            if (r == bottomDxRIndex && c == leftDxCIndex) {\n              // bottomLeft\n              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;\n            }\n\n            if (r == bottomDxRIndex && c == rightDxCIndex) {\n              // bottomRight\n              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;\n            }\n          }\n        }\n        // End loop over dy\n\n        setOutput(accumulator);\n      }\n    ";
        }
        return ResizeBilinearBackpropProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var ResizeBilinearProgram = /** @class */ (function () {
        function ResizeBilinearProgram(inputShape, newHeight, newWidth, alignCorners) {
            this.variableNames = ['A'];
            this.outputShape = [];
            var batch = inputShape[0], oldHeight = inputShape[1], oldWidth = inputShape[2], depth = inputShape[3];
            this.outputShape = [batch, newHeight, newWidth, depth];
            var effectiveInSize = [
                (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
                (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
            ];
            var effectiveOutSize = [
                (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
                (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
            ];
            this.userCode = "\n      const vec2 effectiveInputOverOutputRatioRC = vec2(\n          " + effectiveInSize[0] / effectiveOutSize[0] + ",\n          " + effectiveInSize[1] / effectiveOutSize[1] + ");\n      const vec2 inputShapeRC = vec2(" + oldHeight + ".0, " + oldWidth + ".0);\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        ivec2 yRC = coords.yz;\n\n        // Fractional source index.\n        vec2 sourceFracIndexRC = vec2(yRC) * effectiveInputOverOutputRatioRC;\n\n        // Compute the four integer indices.\n        ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);\n        ivec2 sourceCeilRC = ivec2(\n          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));\n\n        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);\n        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);\n        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);\n        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);\n\n        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);\n\n        float top = topLeft + (topRight - topLeft) * fracRC.y;\n        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;\n        float newValue = top + (bottom - top) * fracRC.x;\n\n        setOutput(newValue);\n      }\n    ";
        }
        return ResizeBilinearProgram;
    }());

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
    var ResizeBilinearPackedProgram = /** @class */ (function () {
        function ResizeBilinearPackedProgram(inputShape, newHeight, newWidth, alignCorners) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = [];
            var batch = inputShape[0], oldHeight = inputShape[1], oldWidth = inputShape[2], depth = inputShape[3];
            this.outputShape = [batch, newHeight, newWidth, depth];
            var effectiveInSize = [
                (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
                (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
            ];
            var effectiveOutSize = [
                (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
                (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
            ];
            this.userCode = "\n      const vec3 effectiveInputOverOutputRatioRC = vec3(\n          " + effectiveInSize[0] / effectiveOutSize[0] + ",\n          " + effectiveInSize[1] / effectiveOutSize[1] + ",\n          " + effectiveInSize[1] / effectiveOutSize[1] + ");\n      const vec3 inputShapeRC = vec3(" + oldHeight + ".0, " + oldWidth + ".0,\n                                     " + oldWidth + ".0);\n\n      float getAValue(int b, int r, int c, int d) {\n        return getChannel(getA(b, r, c, d), vec2(c, d));\n      }\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        // Calculate values for next column in yRC.z.\n        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);\n\n        // Fractional source index.\n        vec3 sourceFracIndexRC = vec3(yRC) * effectiveInputOverOutputRatioRC;\n\n        // Compute the four integer indices.\n        ivec3 sourceFloorRC = ivec3(sourceFracIndexRC);\n        ivec3 sourceCeilRC = ivec3(\n          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));\n\n        // Should we calculate next column and row elements in 2x2 packed cell.\n        bool hasNextCol = d < " + (depth - 1) + ";\n        bool hasNextRow = coords.z < " + (newWidth - 1) + ";\n\n        // In parallel, construct four corners for all four components in\n        // packed 2x2 cell.\n        vec4 topLeft = vec4(\n          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),\n          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)\n                     : 0.0,\n          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)\n                     : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);\n\n        vec4 bottomLeft = vec4(\n          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),\n          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)\n                     : 0.0,\n          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)\n                     : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);\n\n        vec4 topRight = vec4(\n          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),\n          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)\n                     : 0.0,\n          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)\n                     : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);\n\n        vec4 bottomRight = vec4(\n          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),\n          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)\n                     : 0.0,\n          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)\n                     : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);\n\n        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);\n\n        vec4 top = mix(topLeft, topRight, fracRC.yyzz);\n        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);\n        vec4 newValue = mix(top, bottom, fracRC.x);\n\n        setOutput(newValue);\n      }\n    ";
        }
        return ResizeBilinearPackedProgram;
    }());

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
    var ResizeNearestNeigborBackpropProgram = /** @class */ (function () {
        function ResizeNearestNeigborBackpropProgram(dy, x, alignCorners) {
            this.variableNames = ['dy'];
            this.outputShape = [];
            this.outputShape = x.shape;
            var _a = x.shape, xHeight = _a[1], xWidth = _a[2];
            var _b = dy.shape, yHeight = _b[1], yWidth = _b[2];
            // In the backwards pass, we want to find the pixels that were generated for
            // each pixel in the input image the forward pass and add the corresponding
            // coefficient from dy to the gradient (with some interpolation).
            var effectiveXSize = [
                (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
                (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
            ];
            var effectiveYSize = [
                (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
                (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
            ];
            var heightScale = effectiveXSize[0] / effectiveYSize[0];
            var widthScale = effectiveXSize[1] / effectiveYSize[1];
            var invHeightScale = 1 / heightScale;
            var invWidthScale = 1 / widthScale;
            // This defines the size of the window of values around a particular
            // index in dy that we want to search for contributions to dx.
            var winHeight = (Math.ceil(invHeightScale) * 2) + 2;
            var winWidth = (Math.ceil(invWidthScale) * 2) + 2;
            this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        int r = coords[1];\n        int c = coords[2];\n\n        float accumulator = 0.0;\n\n        const float heightScale = float(" + heightScale + ");\n        const float widthScale = float(" + widthScale + ");\n\n        const float invHeightScale = float(" + invHeightScale + ");\n        const float invWidthScale = float(" + invWidthScale + ");\n\n        const int winHeight = int(" + winHeight + ");\n        const int winWidth = int(" + winWidth + ");\n\n        // Compute bounds for where in dy we will look\n        float startRLerp = floor(float(r) * invHeightScale);\n        int startDyR = int(floor(startRLerp - float(winHeight / 2)));\n\n        float startCLerp = floor(float(c) * invWidthScale);\n        int startDyC = int(floor(startCLerp - float(winWidth / 2)));\n\n        // Loop over dy\n        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {\n          int dyR = dyROffset + startDyR;\n\n          // Guard against the window exceeding the bounds of dy\n          if (dyR < 0 || dyR >= " + yHeight + ") {\n            continue;\n          }\n\n          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {\n            int dyC = dyCOffset + startDyC;\n\n            // Guard against the window exceeding the bounds of dy\n            if (dyC < 0 || dyC >= " + yWidth + ") {\n              continue;\n            }\n\n            float sourceFracRow =\n              float(" + effectiveXSize[0] + ") *\n                (float(dyR) / float(" + effectiveYSize[0] + "));\n\n            float sourceFracCol =\n                float(" + effectiveXSize[1] + ") *\n                  (float(dyC) / float(" + effectiveYSize[1] + "));\n\n            int sourceNearestRow = int(min(\n                float(int(" + xHeight + ") - 1),\n                " + alignCorners + " ? float(round(sourceFracRow)) :\n                                  float(floor(sourceFracRow))));\n\n            int sourceNearestCol = int(min(\n                float(int(" + xWidth + ") - 1),\n                " + alignCorners + " ? float(round(sourceFracCol)) :\n                                  float(floor(sourceFracCol))));\n\n            if (r == sourceNearestRow && c == sourceNearestCol) {\n              accumulator += getDy(b, dyR, dyC, d);\n            }\n          }\n        }\n        // End loop over dy\n\n        setOutput(accumulator);\n      }\n    ";
        }
        return ResizeNearestNeigborBackpropProgram;
    }());

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
    var ResizeNearestNeighborProgram = /** @class */ (function () {
        function ResizeNearestNeighborProgram(inputShape, newHeight, newWidth, alignCorners) {
            this.variableNames = ['A'];
            this.outputShape = [];
            var batch = inputShape[0], oldHeight = inputShape[1], oldWidth = inputShape[2], depth = inputShape[3];
            this.outputShape = [batch, newHeight, newWidth, depth];
            var effectiveInSize = [
                (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
                (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
            ];
            var effectiveOutSize = [
                (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
                (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
            ];
            // When align corners is false, we rounds the value with floor.
            var roundBase = alignCorners ? '0.5' : '0.0';
            this.userCode = "\n      const vec2 effectiveInputOverOutputRatioRC = vec2(\n          " + effectiveInSize[0] / effectiveOutSize[0] + ",\n          " + effectiveInSize[1] / effectiveOutSize[1] + ");\n      const vec2 inputShapeRC = vec2(" + oldHeight + ".0, " + oldWidth + ".0);\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        ivec2 yRC = coords.yz;\n\n        // Fractional source index.\n        vec2 sourceFracIndexRC = vec2(yRC) * effectiveInputOverOutputRatioRC;\n\n        // Compute the coordinators of nearest neighbor point.\n        ivec2 sourceNearestRC = ivec2(\n          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + " + roundBase + ")));\n\n        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);\n\n        setOutput(newValue);\n      }\n    ";
        }
        return ResizeNearestNeighborProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var ReverseProgram = /** @class */ (function () {
        function ReverseProgram(xShape, axis) {
            this.variableNames = ['x'];
            var rank = xShape.length;
            if (rank > 4) {
                throw new Error("WebGL backend: Reverse of rank-" + rank + " tensor is not yet supported");
            }
            this.outputShape = xShape;
            if (rank === 1) {
                this.userCode = "\n        void main() {\n          int coord = getOutputCoords();\n          setOutput(getX(" + xShape[0] + " - coord - 1));\n        }\n      ";
                return;
            }
            var getInCoord = function (i) {
                if (axis.indexOf(i) !== -1 && xShape[i] !== 1) {
                    return xShape[i] + " - coords[" + i + "] - 1";
                }
                return "coords[" + i + "]";
            };
            var inCoords = xShape.map(function (_, i) { return getInCoord(i); }).join(',');
            var type = getCoordsDataType(rank);
            this.userCode = "\n      void main() {\n        " + type + " coords = getOutputCoords();\n        setOutput(getX(" + inCoords + "));\n      }\n    ";
        }
        return ReverseProgram;
    }());

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
    var ReversePackedProgram = /** @class */ (function () {
        function ReversePackedProgram(xShape, axis) {
            this.variableNames = ['x'];
            this.packedInputs = true;
            this.packedOutput = true;
            var rank = xShape.length;
            if (rank > 4) {
                throw new Error("WebGL backend: Reverse of rank-" + rank + " tensor is not yet supported");
            }
            this.outputShape = xShape;
            var channels = getChannels('rc', rank);
            var nextColumn = channels[rank - 1] + " + 1 < " + this.outputShape[rank - 1];
            var nextRow = channels[rank - 2] + " + 1 < " + this.outputShape[rank - 2];
            var type = getCoordsDataType(rank);
            if (rank === 1) {
                this.userCode = "\n        void main(){\n          int rc = getOutputCoords();\n          vec4 result = vec4(0.);\n          result.r = getChannel(getX(" + xShape[0] + " - rc - 1),\n            " + xShape[0] + " - rc - 1);\n          if(" + nextColumn + "){\n              result.g = getChannel(getX(" + xShape[0] + " - (rc  + 1) - 1),\n                " + xShape[0] + " - (rc  + 1) - 1);\n          }\n          setOutput(result);\n        }\n      ";
            }
            else {
                this.userCode = "\n        void main() {\n          " + type + " rc = getOutputCoords();\n          vec4 result = vec4(0.);\n          result.r = " + getR(channels.slice()) + ";\n          if(" + nextColumn + "){\n            result.g = " + getG(channels.slice()) + ";\n          }\n          if(" + nextRow + ") {\n            result.b = " + getB(channels.slice()) + ";\n            if(" + nextColumn + ") {\n              result.a = " + getA(channels.slice()) + ";\n            }\n          }\n          setOutput(result);\n        }\n    ";
            }
            function getR(channels) {
                return getChannel(channels);
            }
            function getG(channels) {
                channels[rank - 1] = '(' + channels[rank - 1] + " + 1)";
                return getChannel(channels);
            }
            function getB(channels) {
                channels[rank - 2] = '(' + channels[rank - 2] + " + 1)";
                return getChannel(channels);
            }
            function getA(channels) {
                channels[rank - 1] = '(' + channels[rank - 1] + " + 1)";
                channels[rank - 2] = '(' + channels[rank - 2] + " + 1)";
                return getChannel(channels);
            }
            function getChannel(channels) {
                var inCoordsArray = xShape.map(function (_, i) { return getInCoord(i, channels); });
                var inCoords = inCoordsArray.join(',');
                var innerDims = inCoordsArray.slice(-2).join(',');
                return "getChannel(getX(" + inCoords + "), vec2(" + innerDims + "))";
            }
            function getInCoord(i, channels1) {
                if (axis.indexOf(i) !== -1 && xShape[i] !== 1) {
                    return xShape[i] + " - " + channels1[i] + " - 1";
                }
                else {
                    return "" + channels1[i];
                }
            }
        }
        return ReversePackedProgram;
    }());

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
    var ScatterProgram = /** @class */ (function () {
        function ScatterProgram(updateSize, sliceDim, indicesRank, updatesRank, strides, shape, summingDupeIndex) {
            this.variableNames = ['updates', 'indices', 'defaultValue'];
            this.outputShape = shape;
            var stridesType = getCoordsDataType(strides.length);
            var dtype = getCoordsDataType(shape.length);
            var indicesString = '';
            if (indicesRank === 1) {
                indicesString = 'i';
            }
            else if (indicesRank === 2) {
                indicesString = 'i, j';
            }
            var indicesSnippet = "getIndices(" + indicesString + ")";
            var updatesString = '';
            if (updatesRank === 1) {
                updatesString = 'i';
            }
            else if (updatesRank === 2) {
                updatesString = 'i, coords[1]';
            }
            var updatesSnippet = "getUpdates(" + updatesString + ")";
            var strideString = sliceDim > 1 ? 'strides[j]' : 'strides';
            this.userCode = "\n        " + stridesType + " strides = " + stridesType + "(" + strides + ");\n\n        void main() {\n          " + dtype + " coords = getOutputCoords();\n          float sum = 0.0;\n          bool found = false;\n          for (int i = 0; i < " + updateSize + "; i++) {\n            int flattenedIndex = 0;\n            for (int j = 0; j < " + sliceDim + "; j++) {\n              int index = round(" + indicesSnippet + ");\n              flattenedIndex += index * " + strideString + ";\n            }\n            if (flattenedIndex == coords[0]) {\n              sum += " + updatesSnippet + ";\n              found = true;\n            }\n          }\n          setOutput(mix(getDefaultValue(), sum, float(found)));\n        }\n      ";
        }
        return ScatterProgram;
    }());

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
    var SegmentOpProgram = /** @class */ (function () {
        function SegmentOpProgram(segOpInfo, segOpType) {
            this.variableNames = ['x', 'segmentIds'];
            var windowSize = segOpInfo.windowSize;
            var batchSize = segOpInfo.batchSize;
            var inSize = segOpInfo.inSize;
            var numSegments = segOpInfo.numSegments;
            var outSize = numSegments * Math.ceil(inSize / windowSize);
            this.outputShape = [batchSize, outSize];
            var initializationValue = '0.0';
            var returnValue = "sumValue";
            var windowSizeNearestVec4 = Math.floor(windowSize / 4) * 4;
            var windowSizeVec4Remainder = windowSize % 4;
            var updateSnippet = "\n        sumValue += dot(values, segFilter);\n    ";
            var checkValueOutOfBounds = '';
            if (inSize % windowSize > 0) {
                checkValueOutOfBounds = "\n        if (inIdx < 0 || inIdx >= " + inSize + ") {\n          return initializationValue;\n        }\n      ";
            }
            var checkSegmentIdOutOfBounds = '';
            if (inSize % windowSize > 0) {
                checkSegmentIdOutOfBounds = "\n        if (inIdx < 0 || inIdx >= " + inSize + ") {\n          return -1.0;\n        }\n      ";
            }
            this.userCode = "\n      const float initializationValue = " + initializationValue + ";\n\n      float getValue(int batch, int inIdx) {\n        " + checkValueOutOfBounds + "\n        return getX(batch, inIdx);\n      }\n\n      float getSegmentIdAtIndex(int inIdx) {\n        " + checkSegmentIdOutOfBounds + "\n        return getSegmentIds(inIdx);\n      }\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int batch = coords[0];\n        int outIdx = coords[1];\n        int inOffset = int(floor(float(outIdx) / float(\n          " + numSegments + ")) * float(" + windowSize + "));\n        int currentSeg = int(mod(float(outIdx), float(" + numSegments + ")));\n\n        float sumValue = 0.0;\n\n        for (int i = 0; i < " + windowSizeNearestVec4 + "; i += 4) {\n          int inIdx = inOffset + i;\n          vec4 values = vec4(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            getValue(batch, inIdx + 2),\n            getValue(batch, inIdx + 3)\n          );\n\n          vec4 segFilter = vec4(\n            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0\n          );\n\n          " + updateSnippet + "\n        }\n\n        int inIdx = inOffset + " + windowSizeNearestVec4 + ";\n        if (" + (windowSizeVec4Remainder === 1) + ") {\n          vec4 values = vec4(\n            getValue(batch, inIdx),\n            initializationValue,\n            initializationValue,\n            initializationValue\n          );\n\n          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));\n\n          vec4 segFilter = vec4(\n            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,\n            0,\n            0,\n            0\n          );\n\n          " + updateSnippet + "\n        } else if (" + (windowSizeVec4Remainder === 2) + ") {\n          vec4 values = vec4(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            initializationValue,\n            initializationValue\n          );\n\n          vec4 segFilter = vec4(\n            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,\n              0,\n              0\n          );\n\n          " + updateSnippet + "\n        } else if (" + (windowSizeVec4Remainder === 3) + ") {\n          vec4 values = vec4(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            getValue(batch, inIdx + 2),\n            initializationValue\n          );\n\n          vec4 segFilter = vec4(\n            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,\n            0\n          );\n\n          " + updateSnippet + "\n        }\n        setOutput(" + returnValue + ");\n      }\n    ";
        }
        return SegmentOpProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var SelectProgram = /** @class */ (function () {
        function SelectProgram(cRank, shape, rank) {
            this.variableNames = ['c', 'a', 'b'];
            this.outputShape = shape;
            var cCoords;
            var abCoords;
            if (rank > 4) {
                throw Error("Where for rank " + rank + " is not yet supported");
            }
            if (rank === 1) {
                abCoords = "resRC";
                cCoords = "resRC";
            }
            else {
                var currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
                var cCoordVars = [];
                var abCoordVars = [];
                for (var i = 0; i < shape.length; i++) {
                    abCoordVars.push("" + currentCoords[i]);
                    if (i < cRank) {
                        cCoordVars.push("" + currentCoords[i]);
                    }
                }
                cCoords = cCoordVars.join();
                abCoords = abCoordVars.join();
            }
            var dtype = getCoordsDataType(rank);
            this.userCode = "\n      void main() {\n        " + dtype + " resRC = getOutputCoords();\n        float cVal = getC(" + cCoords + ");\n        if (cVal >= 1.0) {\n          setOutput(getA(" + abCoords + "));\n        } else {\n          setOutput(getB(" + abCoords + "));\n        }\n      }\n    ";
        }
        return SelectProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var SliceProgram = /** @class */ (function () {
        function SliceProgram(destSize) {
            this.variableNames = ['source'];
            this.outputShape = destSize;
            this.rank = destSize.length;
            var dtype = getCoordsDataType(this.rank);
            var uniformPart = "uniform int start[" + this.rank + "];";
            var sourceCoords = getCoords$1(this.rank);
            var body;
            var coordSum = destSize.map(function (_, i) {
                return "sourceLoc." + coords[i] + " = start[" + i + "] + coords." + coords[i] + ";";
            });
            body = "\n        " + dtype + " sourceLoc;\n        " + dtype + " coords = getOutputCoords();\n        " + coordSum.join('\n') + "\n      ";
            this.userCode = "\n      " + uniformPart + "\n      void main() {\n        " + body + "\n        setOutput(getSource(" + sourceCoords + "));\n      }\n    ";
        }
        SliceProgram.prototype.getCustomSetupFunc = function (start) {
            var _this = this;
            if (start.length !== this.rank) {
                throw Error("The rank (" + this.rank + ") of the program must match the " +
                    ("length of start (" + start.length + ")"));
            }
            return function (gpgpu, webGLProgram) {
                if (_this.startLoc == null) {
                    _this.startLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'start');
                    if (_this.startLoc == null) {
                        // This means the compiler has optimized and realized it doesn't need
                        // the uniform.
                        return;
                    }
                }
                gpgpu.gl.uniform1iv(_this.startLoc, start);
            };
        };
        return SliceProgram;
    }());
    var coords = ['x', 'y', 'z', 'w', 'u', 'v'];
    function getCoords$1(rank) {
        if (rank === 1) {
            return 'sourceLoc';
        }
        else if (rank <= 6) {
            return coords.slice(0, rank).map(function (x) { return 'sourceLoc.' + x; }).join(',');
        }
        else {
            throw Error("Slicing for rank " + rank + " is not yet supported");
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
    var SlicePackedProgram = /** @class */ (function () {
        function SlicePackedProgram(destSize) {
            this.variableNames = ['source'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = destSize;
            this.rank = destSize.length;
            var dtype = getCoordsDataType(this.rank);
            var coords = getChannels('coords', this.rank);
            var sourceLoc = getChannels('sourceLoc', this.rank);
            var innerDims = this.rank === 1 ? 'sourceLoc' : "vec2(" + sourceLoc.slice(-2).join() + ")";
            var getChannel = "getChannel(getSource(" + sourceLoc.join() + "), " + innerDims + ")";
            var upperRow = "\n      result.x = " + getChannel + ";\n      if (++" + coords[this.rank - 1] + " < " + destSize[this.rank - 1] + ") {\n        ++" + sourceLoc[this.rank - 1] + ";\n        result.y = " + getChannel + ";\n        --" + sourceLoc[this.rank - 1] + ";\n      }\n    ";
            var lowerRow = this.rank === 1 ? '' : "\n      --" + coords[this.rank - 1] + ";\n      if (++" + coords[this.rank - 2] + " < " + destSize[this.rank - 2] + ") {\n        ++" + sourceLoc[this.rank - 2] + ";\n        result.z = " + getChannel + ";\n        if (++" + coords[this.rank - 1] + " < " + destSize[this.rank - 1] + ") {\n          ++" + sourceLoc[this.rank - 1] + ";\n          result.w = " + getChannel + ";\n        }\n      }\n    ";
            var sourceLocSetup = this.rank <= 4 ?
                "sourceLoc = coords +\n            " + dtype + "(" + destSize.map(function (_, i) { return "start[" + i + "]"; }).join() + ");" :
                destSize.map(function (_, i) { return sourceLoc[i] + " = " + coords[i] + " + start[" + i + "];"; })
                    .join('\n');
            this.userCode = "\n      uniform int start[" + this.rank + "];\n      void main() {\n        " + dtype + " coords = getOutputCoords();\n        " + dtype + " sourceLoc;\n        " + sourceLocSetup + "\n        vec4 result = vec4(0.);\n        " + upperRow + "\n        " + lowerRow + "\n        setOutput(result);\n      }\n    ";
        }
        SlicePackedProgram.prototype.getCustomSetupFunc = function (start) {
            var _this = this;
            if (start.length !== this.rank) {
                throw Error("The rank (" + this.rank + ") of the program must match the " +
                    ("length of start (" + start.length + ")"));
            }
            return function (gpgpu, webGLProgram) {
                if (_this.startLoc == null) {
                    _this.startLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'start');
                    if (_this.startLoc == null) {
                        // This means the compiler has optimized and realized it doesn't need
                        // the uniform.
                        return;
                    }
                }
                gpgpu.gl.uniform1iv(_this.startLoc, start);
            };
        };
        return SlicePackedProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var StridedSliceProgram = /** @class */ (function () {
        function StridedSliceProgram(begin, strides, size) {
            this.variableNames = ['x'];
            this.outputShape = size;
            var rank = size.length;
            var inputDtype = getCoordsDataType(size.length);
            var dtype = getCoordsDataType(size.length);
            var newCoords = '';
            if (rank === 1) {
                newCoords = 'coords * strides + begin';
            }
            else {
                var outputAxis_1 = 0;
                newCoords =
                    size.map(function (_, i) {
                        outputAxis_1++;
                        return size.length === 1 ?
                            "coords * strides[" + i + "] + begin[" + i + "]" :
                            "coords[" + (outputAxis_1 - 1) + "] * strides[" + i + "] + begin[" + i + "]";
                    })
                        .join(',');
            }
            this.userCode = "\n      " + inputDtype + " begin = " + inputDtype + "(" + begin + ");\n      " + inputDtype + " strides = " + inputDtype + "(" + strides + ");\n\n      void main() {\n        " + dtype + " coords = getOutputCoords();\n        setOutput(getX(" + newCoords + "));\n      }\n    ";
        }
        return StridedSliceProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var TextureManager = /** @class */ (function () {
        function TextureManager(gpgpu) {
            this.gpgpu = gpgpu;
            this.numUsedTextures = 0;
            this.numFreeTextures = 0;
            this.freeTextures = {};
            this.logEnabled = false;
            this.usedTextures = {};
        }
        TextureManager.prototype.acquireTexture = function (shapeRC, usage, isPacked) {
            var physicalTexType = getPhysicalFromLogicalTextureType(usage, isPacked);
            var shapeKey = getKeyFromTextureShape(shapeRC, physicalTexType, isPacked);
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
                var newTexture_1 = this.freeTextures[shapeKey].shift();
                this.usedTextures[shapeKey].push(newTexture_1);
                return newTexture_1;
            }
            this.numUsedTextures++;
            this.log();
            var newTexture;
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
        };
        TextureManager.prototype.releaseTexture = function (texture, shape, logicalTexType, isPacked) {
            if (this.freeTextures == null) {
                // Already disposed.
                return;
            }
            var physicalTexType = getPhysicalFromLogicalTextureType(logicalTexType, isPacked);
            var shapeKey = getKeyFromTextureShape(shape, physicalTexType, isPacked);
            if (!(shapeKey in this.freeTextures)) {
                this.freeTextures[shapeKey] = [];
            }
            this.freeTextures[shapeKey].push(texture);
            this.numFreeTextures++;
            this.numUsedTextures--;
            var texList = this.usedTextures[shapeKey];
            var texIndex = texList.indexOf(texture);
            if (texIndex < 0) {
                throw new Error('Cannot release a texture that was never provided by this ' +
                    'texture manager');
            }
            texList.splice(texIndex, 1);
            this.log();
        };
        TextureManager.prototype.log = function () {
            if (!this.logEnabled) {
                return;
            }
            var total = this.numFreeTextures + this.numUsedTextures;
            console.log('Free/Used', this.numFreeTextures + " / " + this.numUsedTextures, "(" + total + ")");
        };
        TextureManager.prototype.getNumUsedTextures = function () {
            return this.numUsedTextures;
        };
        TextureManager.prototype.getNumFreeTextures = function () {
            return this.numFreeTextures;
        };
        TextureManager.prototype.dispose = function () {
            var _this = this;
            if (this.freeTextures == null) {
                // Already disposed.
                return;
            }
            for (var texShape in this.freeTextures) {
                this.freeTextures[texShape].forEach(function (tex) {
                    _this.gpgpu.deleteMatrixTexture(tex);
                });
            }
            for (var texShape in this.usedTextures) {
                this.usedTextures[texShape].forEach(function (tex) {
                    _this.gpgpu.deleteMatrixTexture(tex);
                });
            }
            this.freeTextures = null;
            this.usedTextures = null;
            this.numUsedTextures = 0;
            this.numFreeTextures = 0;
        };
        return TextureManager;
    }());
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
        throw new Error("Unknown logical texture type " + logicalTexType);
    }
    function getKeyFromTextureShape(shapeRowsCol, physicalTexType, isPacked) {
        return shapeRowsCol[0] + "_" + shapeRowsCol[1] + "_" + physicalTexType + "_" + isPacked;
    }

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var TileProgram = /** @class */ (function () {
        function TileProgram(aShape, reps) {
            this.variableNames = ['A'];
            var outputShape = new Array(aShape.length);
            for (var i = 0; i < outputShape.length; i++) {
                outputShape[i] = aShape[i] * reps[i];
            }
            this.outputShape = outputShape;
            this.rank = outputShape.length;
            var dtype = getCoordsDataType(this.rank);
            var sourceCoords = getSourceCoords$2(aShape);
            this.userCode = "\n      void main() {\n        " + dtype + " resRC = getOutputCoords();\n        setOutput(getA(" + sourceCoords + "));\n      }\n    ";
        }
        return TileProgram;
    }());
    function getSourceCoords$2(aShape) {
        var rank = aShape.length;
        if (rank > 5) {
            throw Error("Tile for rank " + rank + " is not yet supported");
        }
        if (rank === 1) {
            return "imod(resRC, " + aShape[0] + ")";
        }
        var currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w', 'resRC.u'];
        var sourceCoords = [];
        for (var i = 0; i < aShape.length; i++) {
            sourceCoords.push("imod(" + currentCoords[i] + ", " + aShape[i] + ")");
        }
        return sourceCoords.join();
    }

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var UnaryOpProgram = /** @class */ (function () {
        function UnaryOpProgram(aShape, opSnippet) {
            this.variableNames = ['A'];
            this.outputShape = aShape;
            this.userCode = "\n      float unaryOperation(float x) {\n        " + opSnippet + "\n      }\n\n      void main() {\n        float x = getAAtOutCoords();\n        float y = unaryOperation(x);\n\n        setOutput(y);\n      }\n    ";
        }
        return UnaryOpProgram;
    }());
    var CHECK_NAN_SNIPPET$2 = "if (isnan(x)) return x;";
    var LINEAR = "return x;";
    var ABS = "return abs(x);";
    var RELU = CHECK_NAN_SNIPPET$2 + "\n  return (x < 0.0) ? 0.0 : x;\n";
    var RELU6 = CHECK_NAN_SNIPPET$2 + "\n  return (x < 0.0) ? 0.0 : min(6.0, x);\n";
    var ELU = "return (x >= 0.0) ? x : (exp(x) - 1.0);";
    var SELU = "\n  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.\n  // see: https://arxiv.org/abs/1706.02515\n  float scaleAlpha = " + tf.backend_util.SELU_SCALEALPHA + ";\n  float scale = " + tf.backend_util.SELU_SCALE + ";\n  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);\n";
    function STEP(alpha) {
        if (alpha === void 0) { alpha = 0.0; }
        return CHECK_NAN_SNIPPET$2 + ("\n    return x > 0.0 ? 1.0 : float(" + alpha + ");\n  ");
    }
    var NEG = "return -x;";
    var CEIL = "return ceil(x);";
    var FLOOR = "return floor(x);";
    var SIGN = "\n  if (isnan(x)) { return 0.0; }\n  return sign(x);\n";
    var IS_NAN = "return float(isnan(x));";
    var IS_INF = "return float(isinf(x));";
    var IS_FINITE = "return float(!isnan(x) && !isinf(x));";
    var ROUND = "\n  // OpenGL ES does not support round function.\n  // The algorithm is based on banker's rounding.\n  float base = floor(x);\n  if ((x - base) < 0.5) {\n    return floor(x);\n  } else if ((x - base) > 0.5) {\n    return ceil(x);\n  } else {\n    if (mod(base, 2.0) == 0.0) {\n      return base;\n    } else {\n      return base + 1.0;\n    }\n  }\n";
    var EXP = "return exp(x);";
    var EXPM1 = "return exp(x) - 1.0;";
    var LOG = "if (x < 0.0) return NAN;\n  return log(x);";
    var LOG1P = "return log(1.0 + x);";
    var SQRT = "return sqrt(x);";
    var RSQRT = "return inversesqrt(x);";
    var SIGMOID = "return 1.0 / (1.0 + exp(-1.0 * x));";
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
    var SOFTPLUS = "\n  float epsilon = 1.1920928955078125e-7;\n  float threshold = log(epsilon) + 2.0;\n\n  bool too_large = x > -threshold;\n  bool too_small = x < threshold;\n\n  float result;\n  float exp_x = exp(x);\n\n  if (too_large){\n    result = x;\n  }\n  else if (too_small){\n    result = exp_x;\n  }\n  else{\n    result = log(exp_x + 1.0);\n  }\n  return result;\n";
    var SIN = CHECK_NAN_SNIPPET$2 + "\n  return sin(x);\n";
    var COS = CHECK_NAN_SNIPPET$2 + "\n  return cos(x);\n";
    var TAN = "return tan(x);";
    var ASIN = CHECK_NAN_SNIPPET$2 + "\n  if (abs(x) > 1.) {\n    return NAN;\n  }\n  return asin(x);\n";
    var ACOS = CHECK_NAN_SNIPPET$2 + "\n  if (abs(x) > 1.) {\n    return NAN;\n  }\n  return acos(x);\n";
    var ATAN = CHECK_NAN_SNIPPET$2 + "\n  return atan(x);\n";
    var SINH = "\n  float e2x = exp(x);\n  return (e2x - 1.0 / e2x) / 2.0;\n";
    var COSH = "\n  float e2x = exp(-x);\n  return (e2x + 1.0 / e2x) / 2.0;\n";
    var TANH = "\n  float e2x = exp(-2.0 * abs(x));\n  return sign(x) * (1.0 - e2x) / (1.0 + e2x);\n";
    var ASINH = CHECK_NAN_SNIPPET$2 + "return log(x + sqrt(x * x + 1.0));";
    var ACOSH = CHECK_NAN_SNIPPET$2 + "\n  if (x < 1.0) return NAN;\n  return log(x + sqrt(x * x - 1.0));";
    var ATANH = CHECK_NAN_SNIPPET$2 + "\n  if ((x < -1.0) || (x > 1.0)) return NAN;\n  return (log(1.0 + x) - log(1.0 - x)) / 2.0;";
    var ERF = "\n  // Error function is calculated approximately with elementary function.\n  // See \"Handbook of Mathematical Functions with Formulas,\n  // Graphs, and Mathematical Tables\", Abramowitz and Stegun.\n  float p = " + tf.backend_util.ERF_P + ";\n  float a1 = " + tf.backend_util.ERF_A1 + ";\n  float a2 = " + tf.backend_util.ERF_A2 + ";\n  float a3 = " + tf.backend_util.ERF_A3 + ";\n  float a4 = " + tf.backend_util.ERF_A4 + ";\n  float a5 = " + tf.backend_util.ERF_A5 + ";\n\n  float sign = sign(x);\n  x = abs(x);\n  float t = 1.0 / (1.0 + p * x);\n  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));\n";
    var SQUARE = "return x * x;";
    var RECIPROCAL = "return 1.0 / x;";
    var LOGICAL_NOT = "return float(!(x >= 1.0));";
    var TO_INT = "return float(int(x));";
    var CLONE = 'return x;';

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
    var LINEAR$1 = "return x;";
    var LOG$1 = "\n  vec4 result = log(x);\n  vec4 isNaN = vec4(lessThan(x, vec4(0.0)));\n  result.r = isNaN.r == 1.0 ? NAN : result.r;\n  result.g = isNaN.g == 1.0 ? NAN : result.g;\n  result.b = isNaN.b == 1.0 ? NAN : result.b;\n  result.a = isNaN.a == 1.0 ? NAN : result.a;\n\n  return result;\n";
    var RELU$1 = "\n  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));\n  bvec4 isNaN = isnan(x);\n\n  result.r = isNaN.r ? x.r : result.r;\n  result.g = isNaN.g ? x.g : result.g;\n  result.b = isNaN.b ? x.b : result.b;\n  result.a = isNaN.a ? x.a : result.a;\n\n  return result;\n";
    var RELU6$1 = "\n  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));\n  bvec4 isNaN = isnan(x);\n\n  result.r = isNaN.r ? x.r : result.r;\n  result.g = isNaN.g ? x.g : result.g;\n  result.b = isNaN.b ? x.b : result.b;\n  result.a = isNaN.a ? x.a : result.a;\n\n  return result;\n";
    var ELU$1 = "\n  vec4 result;\n\n  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);\n  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);\n  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);\n  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);\n\n  return result;\n";
    var UnaryOpPackedProgram = /** @class */ (function () {
        function UnaryOpPackedProgram(aShape, opSnippet) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = true;
            this.outputShape = aShape;
            this.userCode = "\n      vec4 unaryOperation(vec4 x) {\n        " + opSnippet + "\n      }\n\n      void main() {\n        vec4 x = getAAtOutCoords();\n        vec4 y = unaryOperation(x);\n\n        setOutput(y);\n      }\n    ";
        }
        return UnaryOpPackedProgram;
    }());

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
    var UnpackProgram = /** @class */ (function () {
        function UnpackProgram(outputShape) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = false;
            this.outputShape = outputShape;
            var rank = outputShape.length;
            var channels = getChannels('rc', rank);
            var dtype = getCoordsDataType(rank);
            var sourceCoords = getSourceCoords(rank, channels);
            var innerDims = channels.slice(-2);
            var coords = rank <= 1 ? 'rc' : "vec2(" + innerDims.join(',') + ")";
            this.userCode = "\n      void main() {\n        " + dtype + " rc = getOutputCoords();\n        vec4 packedInput = getA(" + sourceCoords + ");\n\n        setOutput(getChannel(packedInput, " + coords + "));\n      }\n    ";
        }
        return UnpackProgram;
    }());

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var segment_util = tf.backend_util.segment_util;
    var split = tf.kernel_impls.split;
    var tile = tf.kernel_impls.tile;
    var topkImpl = tf.kernel_impls.topkImpl;
    var whereImpl = tf.kernel_impls.whereImpl;
    var EPSILON_FLOAT32 = 1e-7;
    var EPSILON_FLOAT16 = 1e-4;
    var binaryCaches = {};
    function getBinaryCache(webGLVersion) {
        if (webGLVersion in binaryCaches) {
            return binaryCaches[webGLVersion];
        }
        binaryCaches[webGLVersion] = {};
        return binaryCaches[webGLVersion];
    }
    function mapActivationToShaderProgram(activation, packed) {
        if (packed === void 0) { packed = false; }
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
        throw new Error("Activation " + activation + " has not been implemented for the WebGL backend.");
    }
    // Empirically determined constant used to determine size threshold for handing
    // off execution to the CPU.
    var CPU_HANDOFF_SIZE_THRESHOLD = 128;
    // Empirically determined constant used to decide the number of MB on GPU
    // before we warn about high memory use. The MB are this constant * screen area
    // * dpi / 1024 / 1024.
    var BEFORE_PAGING_CONSTANT = 600;
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
    var MATMUL_SHARED_DIM_THRESHOLD = 1000;
    var MathBackendWebGL = /** @class */ (function (_super) {
        __extends(MathBackendWebGL, _super);
        function MathBackendWebGL(gpgpu) {
            var _this = _super.call(this) || this;
            // Maps data ids that have a pending read operation, to list of subscribers.
            _this.pendingRead = new WeakMap();
            // List of data ids that are scheduled for disposal, but are waiting on a
            // pending read operation.
            _this.pendingDisposal = new WeakSet();
            // Used to count the number of 'shallow' sliced tensors that point to the
            // same data id.
            _this.dataRefCount = new WeakMap();
            _this.numBytesInGPU = 0;
            // Accumulated time spent (including blocking) in uploading data to webgl.
            _this.uploadWaitMs = 0;
            // Accumulated time spent (including blocking in downloading data from webgl.
            _this.downloadWaitMs = 0;
            _this.warnedAboutMemory = false;
            _this.warnedAboutCPUBackend = false;
            _this.pendingDeletes = 0;
            _this.disposed = false;
            if (!tf.env().getBool('HAS_WEBGL')) {
                throw new Error('WebGL is not supported on this device');
            }
            if (gpgpu == null) {
                var gl = getWebGLContext(tf.env().getNumber('WEBGL_VERSION'));
                _this.binaryCache = getBinaryCache(tf.env().getNumber('WEBGL_VERSION'));
                _this.gpgpu = new GPGPUContext(gl);
                _this.canvas = gl.canvas;
                _this.gpgpuCreatedLocally = true;
            }
            else {
                _this.gpgpu = gpgpu;
                _this.binaryCache = {};
                _this.gpgpuCreatedLocally = false;
                _this.canvas = gpgpu.gl.canvas;
            }
            _this.textureManager = new TextureManager(_this.gpgpu);
            _this.numMBBeforeWarning = numMBBeforeWarning();
            _this.texData = new tf.DataStorage(_this, tf.engine());
            return _this;
        }
        MathBackendWebGL.prototype.numDataIds = function () {
            return this.texData.numDataIds() +
                (this.cpuBackend ? this.cpuBackend.numDataIds() : 0) -
                this.pendingDeletes;
        };
        MathBackendWebGL.prototype.write = function (values, shape, dtype) {
            if (tf.env().getBool('DEBUG')) {
                this.checkNumericalProblems(values);
            }
            if (dtype === 'complex64' && values != null) {
                throw new Error("Cannot write to a complex64 dtype. " +
                    "Please use tf.complex(real, imag).");
            }
            var dataId = {};
            this.texData.set(dataId, { shape: shape, dtype: dtype, values: values, usage: TextureUsage.UPLOAD });
            return dataId;
        };
        MathBackendWebGL.prototype.move = function (dataId, values, shape, dtype) {
            if (tf.env().getBool('DEBUG')) {
                this.checkNumericalProblems(values);
            }
            if (dtype === 'complex64') {
                throw new Error("Cannot write to a complex64 dtype. " +
                    "Please use tf.complex(real, imag).");
            }
            this.texData.set(dataId, { shape: shape, dtype: dtype, values: values, usage: TextureUsage.UPLOAD });
        };
        MathBackendWebGL.prototype.readSync = function (dataId) {
            var texData = this.texData.get(dataId);
            var values = texData.values, dtype = texData.dtype, complexTensors = texData.complexTensors, slice = texData.slice, shape = texData.shape, isPacked = texData.isPacked;
            if (slice != null) {
                var program = void 0;
                if (isPacked) {
                    program = new UnaryOpPackedProgram(shape, CLONE);
                }
                else {
                    program = new UnaryOpProgram(shape, CLONE);
                }
                var res = this.runWebGLProgram(program, [{ dataId: dataId, shape: shape, dtype: dtype }], dtype);
                var data = this.readSync(res.dataId);
                this.disposeData(res.dataId);
                return data;
            }
            if (values != null) {
                return this.convertAndCacheOnCPU(dataId);
            }
            if (dtype === 'string') {
                return values;
            }
            var shouldTimeProgram = this.activeTimers != null;
            var start;
            if (shouldTimeProgram) {
                start = tf.util.now();
            }
            var result;
            if (dtype === 'complex64') {
                var realValues = complexTensors.real.dataSync();
                var imagValues = complexTensors.imag.dataSync();
                result = tf.backend_util.mergeRealAndImagArrays(realValues, imagValues);
            }
            else {
                result = this.getValuesFromTexture(dataId);
            }
            if (shouldTimeProgram) {
                this.downloadWaitMs += tf.util.now() - start;
            }
            return this.convertAndCacheOnCPU(dataId, result);
        };
        MathBackendWebGL.prototype.read = function (dataId) {
            return __awaiter(this, void 0, void 0, function () {
                var subscribers_1, texData, values, shape, slice, dtype, complexTensors, isPacked, program, res, data, buffer, tmpDownloadTarget, tmpData, vals, ps, realValues, imagValues, size, dTypeVals, subscribers;
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            if (this.pendingRead.has(dataId)) {
                                subscribers_1 = this.pendingRead.get(dataId);
                                return [2 /*return*/, new Promise(function (resolve) { return subscribers_1.push(resolve); })];
                            }
                            texData = this.texData.get(dataId);
                            values = texData.values, shape = texData.shape, slice = texData.slice, dtype = texData.dtype, complexTensors = texData.complexTensors, isPacked = texData.isPacked;
                            if (slice != null) {
                                program = void 0;
                                if (isPacked) {
                                    program = new UnaryOpPackedProgram(shape, CLONE);
                                }
                                else {
                                    program = new UnaryOpProgram(shape, CLONE);
                                }
                                res = this.runWebGLProgram(program, [{ dataId: dataId, shape: shape, dtype: dtype }], dtype);
                                data = this.read(res.dataId);
                                this.disposeData(res.dataId);
                                return [2 /*return*/, data];
                            }
                            if (values != null) {
                                return [2 /*return*/, this.convertAndCacheOnCPU(dataId)];
                            }
                            if (!tf.env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
                                tf.env().getNumber('WEBGL_VERSION') === 2) {
                                throw new Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and " +
                                    "WEBGL_VERSION=2 not yet supported.");
                            }
                            buffer = null;
                            if (dtype !== 'complex64' && tf.env().get('WEBGL_BUFFER_SUPPORTED')) {
                                // Possibly copy the texture into a buffer before inserting a fence.
                                tmpDownloadTarget = this.decode(dataId);
                                tmpData = this.texData.get(tmpDownloadTarget.dataId);
                                buffer = (_a = this.gpgpu).createBufferFromTexture.apply(_a, [tmpData.texture].concat(getDenseTexShape(shape)));
                            }
                            this.pendingRead.set(dataId, []);
                            if (!(dtype !== 'complex64')) return [3 /*break*/, 2];
                            // Create a fence and wait for it to resolve.
                            return [4 /*yield*/, this.gpgpu.createAndWaitForFence()];
                        case 1:
                            // Create a fence and wait for it to resolve.
                            _b.sent();
                            _b.label = 2;
                        case 2:
                            if (!(dtype === 'complex64')) return [3 /*break*/, 4];
                            return [4 /*yield*/, Promise.all([complexTensors.real.data(), complexTensors.imag.data()])];
                        case 3:
                            ps = _b.sent();
                            realValues = ps[0];
                            imagValues = ps[1];
                            vals = tf.backend_util.mergeRealAndImagArrays(realValues, imagValues);
                            return [3 /*break*/, 5];
                        case 4:
                            if (buffer == null) {
                                vals = this.getValuesFromTexture(dataId);
                            }
                            else {
                                size = tf.util.sizeFromShape(shape);
                                vals = this.gpgpu.downloadFloat32MatrixFromBuffer(buffer, size);
                            }
                            _b.label = 5;
                        case 5:
                            if (tmpDownloadTarget != null) {
                                this.disposeData(tmpDownloadTarget.dataId);
                            }
                            dTypeVals = this.convertAndCacheOnCPU(dataId, vals);
                            subscribers = this.pendingRead.get(dataId);
                            this.pendingRead.delete(dataId);
                            // Notify all pending reads.
                            subscribers.forEach(function (resolve) { return resolve(dTypeVals); });
                            if (this.pendingDisposal.has(dataId)) {
                                this.pendingDisposal.delete(dataId);
                                this.disposeData(dataId);
                                this.pendingDeletes--;
                            }
                            return [2 /*return*/, dTypeVals];
                    }
                });
            });
        };
        MathBackendWebGL.prototype.checkNumericalProblems = function (values) {
            if (values == null) {
                return;
            }
            for (var i = 0; i < values.length; i++) {
                var num = values[i];
                if (!canBeRepresented(num)) {
                    if (tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
                        throw Error("The value " + num + " cannot be represented with your " +
                            "current settings. Consider enabling float32 rendering: " +
                            "'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'");
                    }
                    throw Error("The value " + num + " cannot be represented on this device.");
                }
            }
        };
        MathBackendWebGL.prototype.getValuesFromTexture = function (dataId) {
            var _a;
            var _b = this.texData.get(dataId), shape = _b.shape, dtype = _b.dtype, isPacked = _b.isPacked;
            var size = tf.util.sizeFromShape(shape);
            if (tf.env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
                var tmpTarget = this.decode(dataId);
                var tmpData_1 = this.texData.get(tmpTarget.dataId);
                var vals_1 = (_a = this.gpgpu).downloadMatrixFromPackedTexture.apply(_a, [tmpData_1.texture].concat(getDenseTexShape(shape))).subarray(0, size);
                this.disposeData(tmpTarget.dataId);
                return vals_1;
            }
            var shouldUsePackedProgram = tf.env().getBool('WEBGL_PACK') && isPacked === true;
            var outputShape = shouldUsePackedProgram ? getShapeAs3D(shape) : shape;
            var program = shouldUsePackedProgram ?
                new EncodeFloatPackedProgram(outputShape) :
                new EncodeFloatProgram(outputShape);
            var output = this.runWebGLProgram(program, [{ shape: outputShape, dtype: dtype, dataId: dataId }], 'float32');
            var tmpData = this.texData.get(output.dataId);
            var vals = this.gpgpu
                .downloadByteEncodedFloatMatrixFromOutputTexture(tmpData.texture, tmpData.texShape[0], tmpData.texShape[1])
                .subarray(0, size);
            this.disposeData(output.dataId);
            return vals;
        };
        MathBackendWebGL.prototype.time = function (f) {
            return __awaiter(this, void 0, void 0, function () {
                var oldActiveTimers, newActiveTimers, outerMostTime, flattenedActiveTimerQueries, flattenedActiveTimerNames, res, kernelMs_1;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            oldActiveTimers = this.activeTimers;
                            newActiveTimers = [];
                            outerMostTime = false;
                            if (this.programTimersStack == null) {
                                this.programTimersStack = newActiveTimers;
                                outerMostTime = true;
                            }
                            else {
                                this.activeTimers.push(newActiveTimers);
                            }
                            this.activeTimers = newActiveTimers;
                            f();
                            flattenedActiveTimerQueries = tf.util.flatten(this.activeTimers.map(function (d) { return d.query; }))
                                .filter(function (d) { return d != null; });
                            flattenedActiveTimerNames = tf.util.flatten(this.activeTimers.map(function (d) { return d.name; }))
                                .filter(function (d) { return d != null; });
                            this.activeTimers = oldActiveTimers;
                            if (outerMostTime) {
                                this.programTimersStack = null;
                            }
                            res = {
                                uploadWaitMs: this.uploadWaitMs,
                                downloadWaitMs: this.downloadWaitMs,
                                kernelMs: null,
                                wallMs: null // will be filled by the engine
                            };
                            if (!(tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0)) return [3 /*break*/, 2];
                            return [4 /*yield*/, Promise.all(flattenedActiveTimerQueries)];
                        case 1:
                            kernelMs_1 = _a.sent();
                            res['kernelMs'] = tf.util.sum(kernelMs_1);
                            res['getExtraProfileInfo'] = function () {
                                return kernelMs_1.map(function (d, i) { return ({ name: flattenedActiveTimerNames[i], ms: d }); })
                                    .map(function (d) { return d.name + ": " + d.ms; })
                                    .join(', ');
                            };
                            return [3 /*break*/, 3];
                        case 2:
                            res['kernelMs'] = {
                                error: 'WebGL query timers are not supported in this environment.'
                            };
                            _a.label = 3;
                        case 3:
                            this.uploadWaitMs = 0;
                            this.downloadWaitMs = 0;
                            return [2 /*return*/, res];
                    }
                });
            });
        };
        MathBackendWebGL.prototype.memory = function () {
            return { unreliable: false, numBytesInGPU: this.numBytesInGPU };
        };
        MathBackendWebGL.prototype.startTimer = function () {
            if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
                return this.gpgpu.beginQuery();
            }
            return { startMs: tf.util.now(), endMs: null };
        };
        MathBackendWebGL.prototype.endTimer = function (query) {
            if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
                this.gpgpu.endQuery();
                return query;
            }
            query.endMs = tf.util.now();
            return query;
        };
        MathBackendWebGL.prototype.getQueryTime = function (query) {
            return __awaiter(this, void 0, void 0, function () {
                var timerQuery;
                return __generator(this, function (_a) {
                    if (tf.env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
                        return [2 /*return*/, this.gpgpu.waitForQueryAndGetTime(query)];
                    }
                    timerQuery = query;
                    return [2 /*return*/, timerQuery.endMs - timerQuery.startMs];
                });
            });
        };
        MathBackendWebGL.prototype.disposeData = function (dataId) {
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
            var complexTensors = this.texData.get(dataId).complexTensors;
            if (complexTensors != null) {
                complexTensors.real.dispose();
                complexTensors.imag.dispose();
            }
            this.texData.delete(dataId);
        };
        MathBackendWebGL.prototype.releaseGPUData = function (dataId) {
            var _a = this.texData.get(dataId), texture = _a.texture, dtype = _a.dtype, texShape = _a.texShape, usage = _a.usage, isPacked = _a.isPacked, slice = _a.slice;
            var key = slice && slice.origDataId || dataId;
            var refCount = this.dataRefCount.get(key);
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
            var texData = this.texData.get(dataId);
            texData.texture = null;
            texData.texShape = null;
            texData.isPacked = false;
            texData.slice = null;
        };
        MathBackendWebGL.prototype.getTexture = function (dataId) {
            this.uploadToGPU(dataId);
            return this.texData.get(dataId).texture;
        };
        /**
         * Returns internal information for the specific data bucket. Used in unit
         * tests.
         */
        MathBackendWebGL.prototype.getDataInfo = function (dataId) {
            return this.texData.get(dataId);
        };
        MathBackendWebGL.prototype.getCPUBackend = function () {
            if (!tf.env().getBool('WEBGL_CPU_FORWARD')) {
                return null;
            }
            if (this.cpuBackend == null) {
                this.cpuBackend = tf.engine().findBackend('cpu');
            }
            return this.cpuBackend;
        };
        /*
        Tests whether all the inputs to an op are small and on the CPU. This heuristic
        determines when it would be faster to execute a kernel on the CPU. WebGL
        kernels opt into running this check and forwarding when appropriate.
        TODO(https://github.com/tensorflow/tfjs/issues/872): Develop a more
        sustainable strategy for optimizing backend execution of ops.
         */
        MathBackendWebGL.prototype.shouldExecuteOnCPU = function (inputs, sizeThreshold) {
            var _this = this;
            if (sizeThreshold === void 0) { sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD; }
            var cpuBackend = this.getCPUBackend();
            if (!this.warnedAboutCPUBackend && cpuBackend == null &&
                !tf.env().getBool('IS_TEST')) {
                console.warn('Your application contains ops that are small enough to be ' +
                    'executed on the CPU backend, however the CPU backend cannot ' +
                    'be found. Consider importing the CPU backend ' +
                    '(@tensorflow/tfjs-backend-cpu) for better performance.');
                this.warnedAboutCPUBackend = true;
            }
            return cpuBackend != null &&
                inputs.every(function (input) { return _this.texData.get(input.dataId).texture == null &&
                    tf.util.sizeFromShape(input.shape) < sizeThreshold; });
        };
        MathBackendWebGL.prototype.getGPGPUContext = function () {
            return this.gpgpu;
        };
        MathBackendWebGL.prototype.complex = function (real, imag) {
            var result = this.makeOutput(real.shape, 'complex64');
            var resultData = this.texData.get(result.dataId);
            // The backend owns the reference to the underlying real and imaginary
            // clones. These will explicitly get disposed when the complex tensor is
            // disposed.
            resultData.complexTensors = {
                real: tf.engine().keep(real.clone()),
                imag: tf.engine().keep(imag.clone())
            };
            return result;
        };
        MathBackendWebGL.prototype.real = function (input) {
            var resultData = this.texData.get(input.dataId);
            return resultData.complexTensors.real.clone();
        };
        MathBackendWebGL.prototype.imag = function (input) {
            var resultData = this.texData.get(input.dataId);
            return resultData.complexTensors.imag.clone();
        };
        MathBackendWebGL.prototype.slice = function (x, begin, size) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.slice(x, begin, size);
            }
            // Short-circuit computation if the slice is zero-sized.
            if (tf.util.sizeFromShape(size) === 0) {
                return tf.tensor([], size, x.dtype);
            }
            var isPacked = this.texData.get(x.dataId).isPacked;
            var isContinous = tf.slice_util.isSliceContinous(x.shape, begin, size);
            if (isPacked || !isContinous) {
                var program = tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
                    new SlicePackedProgram(size) :
                    new SliceProgram(size);
                var customSetup = program.getCustomSetupFunc(begin);
                return this.compileAndRun(program, [x], null, customSetup);
            }
            this.uploadToGPU(x.dataId);
            return this.shallowSlice(x, begin, size);
        };
        MathBackendWebGL.prototype.shallowSlice = function (x, begin, size) {
            var xTexData = this.texData.get(x.dataId);
            var t = this.makeOutput(size, x.dtype);
            var newTexData = this.texData.get(t.dataId);
            // Copy texture data from the original tensor.
            Object.assign(newTexData, xTexData);
            newTexData.shape = size;
            newTexData.dtype = x.dtype;
            var flatOffset = tf.slice_util.computeFlatOffset(begin, x.strides);
            if (xTexData.slice) {
                // We are slicing an already sliced tensor, so we have to accumulate
                // the offset.
                flatOffset += xTexData.slice.flatOffset;
            }
            newTexData.slice = {
                flatOffset: flatOffset,
                // Point to the original dataId, which is used to do ref counting.
                origDataId: xTexData.slice && xTexData.slice.origDataId || x.dataId
            };
            // Increase the ref count for that data bucket.
            var refCount = this.dataRefCount.get(newTexData.slice.origDataId) || 1;
            this.dataRefCount.set(newTexData.slice.origDataId, refCount + 1);
            return t;
        };
        MathBackendWebGL.prototype.stridedSlice = function (x, begin, end, strides) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.stridedSlice(x, begin, end, strides);
            }
            var outShape = tf.slice_util.computeOutShape(begin, end, strides);
            if (outShape.some(function (axis) { return axis === 0; })) {
                return tf.tensor([], outShape);
            }
            var program = new StridedSliceProgram(begin, strides, outShape);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.reverse = function (x, axis) {
            var program = tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
                new ReversePackedProgram(x.shape, axis) :
                new ReverseProgram(x.shape, axis);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.concat = function (tensors, axis) {
            if (tensors[0].dtype === 'complex64') {
                var reals = tensors.map(function (t) { return tf.real(t); });
                var imags = tensors.map(function (t) { return tf.imag(t); });
                return tf.complex(this.concat(reals, axis), this.concat(imags, axis));
            }
            if (this.shouldExecuteOnCPU(tensors)) {
                return this.cpuBackend.concat(tensors, axis);
            }
            if (tensors.length === 1) {
                return tensors[0];
            }
            if (tensors.length > tf.env().getNumber('WEBGL_MAX_TEXTURES_IN_SHADER')) {
                var midIndex = Math.floor(tensors.length / 2);
                var leftSide = this.concat(tensors.slice(0, midIndex), axis);
                var rightSide = this.concat(tensors.slice(midIndex), axis);
                return this.concat([leftSide, rightSide], axis);
            }
            if (tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') && tensors[0].rank > 1) {
                var program_1 = new ConcatPackedProgram(tensors.map(function (t) { return t.shape; }), axis);
                return this.compileAndRun(program_1, tensors);
            }
            // Any concat of n-dimensional tensors across any axis can be reduced to
            // a concatenation of two-dimensional tensors across the axis 1 by first
            // partitioning the axes of the original tensors into those less than the
            // axis to be concatenated and the rest. Then reshape the tensors
            // into a two-dimensional tensor by collapsing these two sets of axes and
            // concatenate the resulting matrices across the axis 1, finally reshaping
            // the result to have the proper shape.
            var outShape = tf.backend_util.computeOutShape(tensors.map(function (t) { return t.shape; }), axis);
            var tensors2D = tensors.map(function (t) { return t.as2D(-1, tf.util.sizeFromShape(t.shape.slice(axis))); });
            var program = new ConcatProgram(tensors2D.map(function (t) { return t.shape; }));
            var res = this.compileAndRun(program, tensors2D);
            return res.reshape(outShape);
        };
        MathBackendWebGL.prototype.neg = function (x) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.neg(x);
            }
            if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
                return this.packedUnaryOp(x, NEG, x.dtype);
            }
            var program = new UnaryOpProgram(x.shape, NEG);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.batchMatMul = function (a, b, transposeA, transposeB) {
            var outerShapeA = transposeA ? a.shape[2] : a.shape[1];
            var outerShapeB = transposeB ? b.shape[1] : b.shape[2];
            var sharedDim = transposeA ? a.shape[1] : a.shape[2];
            var _a = a.shape, batch = _a[0];
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
                var a3D = outerShapeB === 1 ? a : a.as3D(batch, sharedDim, 1);
                var axis = outerShapeB === 1 ? 2 : 1;
                var b3D = outerShapeB === 1 ? b.as3D(batch, 1, sharedDim) : b;
                return this.multiply(a3D, b3D).sum(axis, true /* keepDims */);
            }
            var dtype = tf.upcastType(a.dtype, b.dtype);
            var program = new MatMulPackedProgram(a.shape, [batch, outerShapeA, outerShapeB], transposeA, transposeB);
            return this.compileAndRun(program, [a, b], dtype);
        };
        MathBackendWebGL.prototype.fusedBatchMatMul = function (_a) {
            var a = _a.a, b = _a.b, transposeA = _a.transposeA, transposeB = _a.transposeB, bias = _a.bias, activation = _a.activation, preluActivationWeights = _a.preluActivationWeights;
            var outerShapeA = transposeA ? a.shape[2] : a.shape[1];
            var outerShapeB = transposeB ? b.shape[1] : b.shape[2];
            var _b = a.shape, batch = _b[0];
            var dtype = tf.upcastType(a.dtype, b.dtype);
            var hasBias = bias != null;
            var hasPreluActivationWeights = preluActivationWeights != null;
            var fusedActivation = activation ? mapActivationToShaderProgram(activation, true) : null;
            var program = new MatMulPackedProgram(a.shape, [batch, outerShapeA, outerShapeB], transposeA, transposeB, hasBias, fusedActivation, hasPreluActivationWeights);
            var inputs = [a, b];
            if (bias) {
                inputs.push(bias);
            }
            if (preluActivationWeights) {
                inputs.push(preluActivationWeights);
            }
            return this.compileAndRun(program, inputs, dtype);
        };
        MathBackendWebGL.prototype.multiply = function (a, b) {
            if (a.dtype === 'complex64') {
                var aData = this.texData.get(a.dataId);
                var bData = this.texData.get(b.dataId);
                var realProgram = new BinaryOpComplexProgram(COMPLEX_MULTIPLY.REAL, a.shape, b.shape);
                var imagProgram = new BinaryOpComplexProgram(COMPLEX_MULTIPLY.IMAG, a.shape, b.shape);
                var inputs = [
                    this.makeComplexComponentTensorInfo(a, aData.complexTensors.real),
                    this.makeComplexComponentTensorInfo(a, aData.complexTensors.imag),
                    this.makeComplexComponentTensorInfo(b, bData.complexTensors.real),
                    this.makeComplexComponentTensorInfo(b, bData.complexTensors.imag)
                ];
                var real_1 = this.compileAndRun(realProgram, inputs);
                var imag_1 = this.compileAndRun(imagProgram, inputs);
                var complex_1 = this.complex(real_1, imag_1);
                real_1.dispose();
                imag_1.dispose();
                return complex_1;
            }
            if (this.shouldExecuteOnCPU([a, b])) {
                return this.cpuBackend.multiply(a, b);
            }
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, MUL, a.dtype);
            }
            var program = new BinaryOpProgram(MUL, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], a.dtype);
        };
        MathBackendWebGL.prototype.batchNorm = function (x, mean, variance, offset, scale, varianceEpsilon) {
            var inputs = [x, mean, variance];
            var offsetShape = null;
            if (offset != null) {
                offsetShape = offset.shape;
                inputs.push(offset);
            }
            var scaleShape = null;
            if (scale != null) {
                scaleShape = scale.shape;
                inputs.push(scale);
            }
            if (tf.env().getBool('WEBGL_PACK_NORMALIZATION')) {
                var batchNormPackedProgram = new BatchNormPackedProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape, varianceEpsilon);
                return this.compileAndRun(batchNormPackedProgram, inputs);
            }
            var batchNormProgram = new BatchNormProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape, varianceEpsilon);
            return this.compileAndRun(batchNormProgram, inputs);
        };
        MathBackendWebGL.prototype.localResponseNormalization4D = function (x, radius, bias, alpha, beta) {
            var program = tf.env().getBool('WEBGL_PACK_NORMALIZATION') ?
                new LRNPackedProgram(x.shape, radius, bias, alpha, beta) :
                new LRNProgram(x.shape, radius, bias, alpha, beta);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.LRNGrad = function (dy, inputImage, outputImage, depthRadius, bias, alpha, beta) {
            var program = new LRNGradProgram(inputImage.shape, depthRadius, bias, alpha, beta);
            return this.compileAndRun(program, [inputImage, outputImage, dy]);
        };
        MathBackendWebGL.prototype.tile = function (x, reps) {
            if (x.dtype === 'string') {
                var data = this.readSync(x.dataId);
                var decodedData = data.map(function (d) { return tf.util.decodeString(d); });
                var buf = tf.buffer(x.shape, x.dtype, decodedData);
                return tile(buf, reps);
            }
            var program = new TileProgram(x.shape, reps);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.pad = function (x, paddings, constantValue) {
            var program = tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
                new PadPackedProgram(x.shape, paddings, constantValue) :
                new PadProgram(x.shape, paddings, constantValue);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.gather = function (x, indices, axis) {
            if (this.shouldExecuteOnCPU([x, indices])) {
                return this.cpuBackend.gather(x, indices, axis);
            }
            var program = new GatherProgram(x.shape, indices.size, axis);
            return this.compileAndRun(program, [x, indices]);
        };
        MathBackendWebGL.prototype.batchToSpaceND = function (x, blockShape, crops) {
            tf.util.assert(x.rank <= 4, function () { return 'batchToSpaceND for rank > 4 with a WebGL backend not ' +
                'implemented yet'; });
            var prod = blockShape.reduce(function (a, b) { return a * b; });
            var reshaped = tf.backend_util.getReshaped(x.shape, blockShape, prod);
            var permuted = tf.backend_util.getPermuted(reshaped.length, blockShape.length);
            var reshapedPermuted = tf.backend_util.getReshapedPermuted(x.shape, blockShape, prod);
            var sliceBeginCoords = tf.backend_util.getSliceBeginCoords(crops, blockShape.length);
            var sliceSize = tf.backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);
            return tf.transpose(x.reshape(reshaped), permuted)
                .reshape(reshapedPermuted)
                .slice(sliceBeginCoords, sliceSize);
        };
        MathBackendWebGL.prototype.spaceToBatchND = function (x, blockShape, paddings) {
            tf.util.assert(x.rank <= 4, function () { return 'spaceToBatchND for rank > 4 with a WebGL backend not ' +
                'implemented yet'; });
            var prod = blockShape.reduce(function (a, b) { return a * b; });
            var completePaddings = [[0, 0]];
            completePaddings.push.apply(completePaddings, paddings);
            for (var i = 1 + blockShape.length; i < x.shape.length; ++i) {
                completePaddings.push([0, 0]);
            }
            var paddedX = x.pad(completePaddings);
            var reshapedPaddedShape = tf.backend_util.getReshaped(paddedX.shape, blockShape, prod, false);
            var permutedReshapedPaddedPermutation = tf.backend_util.getPermuted(reshapedPaddedShape.length, blockShape.length, false);
            var flattenShape = tf.backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);
            return tf.transpose(paddedX.reshape(reshapedPaddedShape), permutedReshapedPaddedPermutation)
                .reshape(flattenShape);
        };
        MathBackendWebGL.prototype.reduce = function (x, reduceType, dtype) {
            var batchSize = x.shape[0];
            var inSize = x.shape[1];
            var windowSize = tf.backend_util.computeOptimalWindowSize(inSize);
            var reduceInfo = { windowSize: windowSize, inSize: inSize, batchSize: batchSize };
            var program = new ReduceProgram(reduceInfo, reduceType);
            var output = this.compileAndRun(program, [x], dtype);
            // No need to run another GPGPU program.
            if (output.shape[1] === 1) {
                return output;
            }
            return this.reduce(output, reduceType, dtype);
        };
        MathBackendWebGL.prototype.argReduce = function (x, reduceType, bestIndicesA) {
            if (bestIndicesA === void 0) { bestIndicesA = null; }
            var batchSize = x.shape[0];
            var inSize = x.shape[1];
            if (bestIndicesA != null) {
                batchSize = bestIndicesA.shape[0];
                inSize = bestIndicesA.shape[1];
            }
            var windowSize = tf.backend_util.computeOptimalWindowSize(inSize);
            var reduceInfo = { windowSize: windowSize, inSize: inSize, batchSize: batchSize };
            var program = new ArgMinMaxProgram(reduceInfo, reduceType, bestIndicesA == null);
            var inputs = [x];
            if (bestIndicesA != null) {
                inputs.push(bestIndicesA);
            }
            var output = this.compileAndRun(program, inputs, 'int32');
            // No need to run another GPGPU program.
            if (output.shape[1] === 1) {
                return output;
            }
            return this.argReduce(x, reduceType, output);
        };
        MathBackendWebGL.prototype.argReducePacked = function (x, reduceType, bestIndicesA) {
            if (bestIndicesA === void 0) { bestIndicesA = null; }
            var inShape = bestIndicesA != null ? bestIndicesA.shape : x.shape;
            var inSize = inShape[inShape.length - 1];
            var windowSize = tf.backend_util.computeOptimalWindowSize(inSize);
            var program = new ArgMinMaxPackedProgram(inShape, windowSize, reduceType, bestIndicesA == null);
            var inputs = bestIndicesA == null ? [x] : [x, bestIndicesA];
            var output = this.compileAndRun(program, inputs, 'int32');
            if (output.rank === x.rank) {
                return this.argReducePacked(x, reduceType, output);
            }
            return output;
        };
        MathBackendWebGL.prototype.sum = function (x, axes) {
            tf.backend_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
            var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
            var inSize = tf.util.sizeFromShape(reduceShape);
            var a2D = x.as2D(-1, inSize);
            var outputDType = tf.sumOutType(x.dtype);
            return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
        };
        MathBackendWebGL.prototype.prod = function (x, axes) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.prod(x, axes);
            }
            var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
            var inSize = tf.util.sizeFromShape(reduceShape);
            var a2D = x.as2D(-1, inSize);
            var outputDType = tf.sumOutType(x.dtype);
            return this.reduce(a2D, 'prod', outputDType).reshape(outShape);
        };
        MathBackendWebGL.prototype.unsortedSegmentSum = function (x, segmentIds, numSegments) {
            var axis = 0;
            var permutation = tf.backend_util.getAxesPermutation([axis], x.rank);
            var permutedX = x;
            if (permutation != null) {
                permutedX = tf.transpose(x, permutation);
                axis = tf.backend_util.getInnerMostAxes(1, x.rank)[0];
            }
            var outShape = segment_util.computeOutShape(permutedX.shape, axis, numSegments);
            var inSize = tf.util.sizeFromShape([permutedX.shape[axis]]);
            var a2D = permutedX.as2D(-1, inSize);
            var outputDType = tf.sumOutType(x.dtype);
            var result = this.segOpCompute(a2D, 'unsortedSegmentSum', segmentIds, outputDType, numSegments)
                .reshape(outShape);
            if (permutation != null) {
                result =
                    tf.transpose(result, tf.backend_util.getUndoAxesPermutation(permutation));
            }
            return result;
        };
        MathBackendWebGL.prototype.segOpCompute = function (x, segOpType, segmentIds, dtype, numSegments) {
            var batchSize = x.shape[0];
            var inSize = x.shape[1];
            var windowSize = segment_util.segOpComputeOptimalWindowSize(inSize, numSegments);
            var segOpInfo = { windowSize: windowSize, inSize: inSize, batchSize: batchSize, numSegments: numSegments };
            var program = new SegmentOpProgram(segOpInfo, segOpType);
            var output = this.compileAndRun(program, [x, segmentIds], dtype);
            // No need to run another GPGPU program.
            if (output.shape[1] === numSegments) {
                return output;
            }
            segmentIds = tf.range(0, numSegments).tile([inSize / windowSize]);
            return this.segOpCompute(output, segOpType, segmentIds, dtype, numSegments);
        };
        MathBackendWebGL.prototype.argMinMaxReduce = function (x, axis, reduceType) {
            var axes = [axis];
            tf.backend_util.assertAxesAreInnerMostDims('arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes, x.rank);
            if (!tf.env().getBool('WEBGL_PACK_REDUCE') || x.rank <= 2) {
                var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
                var inSize = tf.util.sizeFromShape(reduceShape);
                var a2D = x.as2D(-1, inSize);
                return this.argReduce(a2D, reduceType).reshape(outShape);
            }
            return this.argReducePacked(x, reduceType);
        };
        MathBackendWebGL.prototype.argMin = function (x, axis) {
            return this.argMinMaxReduce(x, axis, 'min');
        };
        MathBackendWebGL.prototype.argMax = function (x, axis) {
            return this.argMinMaxReduce(x, axis, 'max');
        };
        MathBackendWebGL.prototype.cumsum = function (x, axis, exclusive, reverse) {
            if (axis !== x.rank - 1) {
                throw new Error("WebGL cumsum shader expects an inner-most axis=" + (x.rank - 1) + " " +
                    ("but got axis=" + axis));
            }
            var size = x.shape[x.shape.length - 1];
            var result = x;
            // Use cumsum parallel algorithm, ref:
            // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
            for (var i = 0; i <= Math.ceil(Math.log2(size)) - 1; i++) {
                var program = new CumSumProgram(x.shape, false, reverse, i);
                result = this.compileAndRun(program, [result]);
            }
            // For exclusive cumsum, shift the end result in the direction of sum and
            // add 0 to the front index.
            if (exclusive) {
                var program = new CumSumProgram(x.shape, exclusive, reverse, 0);
                result = this.compileAndRun(program, [result]);
            }
            return result;
        };
        MathBackendWebGL.prototype.equal = function (a, b) {
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, EQUAL$1, 'bool');
            }
            var program = new BinaryOpProgram(EQUAL, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], 'bool');
        };
        MathBackendWebGL.prototype.notEqual = function (a, b) {
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, NOT_EQUAL$1, 'bool');
            }
            var program = new BinaryOpProgram(NOT_EQUAL, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], 'bool');
        };
        MathBackendWebGL.prototype.less = function (a, b) {
            if (this.shouldExecuteOnCPU([a, b])) {
                return this.cpuBackend.less(a, b);
            }
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, LESS$1, 'bool');
            }
            var program = new BinaryOpProgram(LESS, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], 'bool');
        };
        MathBackendWebGL.prototype.lessEqual = function (a, b) {
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, LESS_EQUAL$1, 'bool');
            }
            var program = new BinaryOpProgram(LESS_EQUAL, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], 'bool');
        };
        MathBackendWebGL.prototype.greater = function (a, b) {
            if (this.shouldExecuteOnCPU([a, b])) {
                return this.cpuBackend.greater(a, b);
            }
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, GREATER$1, 'bool');
            }
            var program = new BinaryOpProgram(GREATER, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], 'bool');
        };
        MathBackendWebGL.prototype.greaterEqual = function (a, b) {
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, GREATER_EQUAL$1, 'bool');
            }
            var program = new BinaryOpProgram(GREATER_EQUAL, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], 'bool');
        };
        MathBackendWebGL.prototype.logicalNot = function (x) {
            var program = new UnaryOpProgram(x.shape, LOGICAL_NOT);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.logicalAnd = function (a, b) {
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, LOGICAL_AND$1, 'bool');
            }
            var program = new BinaryOpProgram(LOGICAL_AND, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], 'bool');
        };
        MathBackendWebGL.prototype.logicalOr = function (a, b) {
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, LOGICAL_OR$1, 'bool');
            }
            var program = new BinaryOpProgram(LOGICAL_OR, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], 'bool');
        };
        MathBackendWebGL.prototype.select = function (condition, a, b) {
            var program = new SelectProgram(condition.rank, a.shape, a.rank);
            return this.compileAndRun(program, [condition, a, b], tf.upcastType(a.dtype, b.dtype));
        };
        MathBackendWebGL.prototype.where = function (condition) {
            tf.backend_util.warn('tf.where() in webgl locks the UI thread. ' +
                'Call tf.whereAsync() instead');
            var condVals = condition.dataSync();
            return whereImpl(condition.shape, condVals);
        };
        MathBackendWebGL.prototype.topk = function (x, k, sorted) {
            var xVals = x.dataSync();
            return topkImpl(xVals, x.shape, x.dtype, k, sorted);
        };
        MathBackendWebGL.prototype.min = function (x, axes) {
            tf.backend_util.assertAxesAreInnerMostDims('min', axes, x.rank);
            var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
            var inSize = tf.util.sizeFromShape(reduceShape);
            var a2D = x.as2D(-1, inSize);
            return this.reduce(a2D, 'min', a2D.dtype).reshape(outShape);
        };
        MathBackendWebGL.prototype.minimum = function (a, b) {
            if (this.shouldExecuteOnCPU([a, b])) {
                return this.cpuBackend.minimum(a, b);
            }
            var program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
                new BinaryOpPackedProgram(MIN$1, a.shape, b.shape) :
                new BinaryOpProgram(MIN, a.shape, b.shape);
            return this.compileAndRun(program, [a, b]);
        };
        MathBackendWebGL.prototype.mod = function (a, b) {
            var program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
                new BinaryOpPackedProgram(MOD$1, a.shape, b.shape) :
                new BinaryOpProgram(MOD, a.shape, b.shape);
            return this.compileAndRun(program, [a, b]);
        };
        MathBackendWebGL.prototype.maximum = function (a, b) {
            if (this.shouldExecuteOnCPU([a, b])) {
                return this.cpuBackend.maximum(a, b);
            }
            var program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
                new BinaryOpPackedProgram(MAX$1, a.shape, b.shape) :
                new BinaryOpProgram(MAX, a.shape, b.shape);
            return this.compileAndRun(program, [a, b]);
        };
        MathBackendWebGL.prototype.all = function (x, axes) {
            tf.backend_util.assertAxesAreInnerMostDims('all', axes, x.rank);
            var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
            var inSize = tf.util.sizeFromShape(reduceShape);
            var a2D = x.as2D(-1, inSize);
            return this.reduce(a2D, 'all', a2D.dtype).reshape(outShape);
        };
        MathBackendWebGL.prototype.any = function (x, axes) {
            tf.backend_util.assertAxesAreInnerMostDims('any', axes, x.rank);
            var _a = tf.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
            var inSize = tf.util.sizeFromShape(reduceShape);
            var a2D = x.as2D(-1, inSize);
            return this.reduce(a2D, 'any', a2D.dtype).reshape(outShape);
        };
        MathBackendWebGL.prototype.floorDiv = function (a, b) {
            var op = INT_DIV;
            var outputDtype = 'int32';
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, INT_DIV$1, outputDtype);
            }
            var program = new BinaryOpProgram(op, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], outputDtype);
        };
        MathBackendWebGL.prototype.add = function (a, b) {
            if (a.dtype === 'complex64' && b.dtype === 'complex64') {
                return this.complexSeparableBinaryOp(a, b, ADD);
            }
            if (this.shouldExecuteOnCPU([a, b])) {
                return this.cpuBackend.add(a, b);
            }
            var dtype = tf.upcastType(a.dtype, b.dtype);
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, ADD, dtype);
            }
            var program = new BinaryOpProgram(ADD, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], dtype);
        };
        MathBackendWebGL.prototype.packedUnaryOp = function (x, op, dtype) {
            var program = new UnaryOpPackedProgram(x.shape, op);
            return this.compileAndRun(program, [x], dtype);
        };
        MathBackendWebGL.prototype.packedBinaryOp = function (a, b, op, dtype, checkOutOfBounds) {
            if (checkOutOfBounds === void 0) { checkOutOfBounds = false; }
            var program = new BinaryOpPackedProgram(op, a.shape, b.shape, checkOutOfBounds);
            return this.compileAndRun(program, [a, b], dtype);
        };
        /**
         * Computes a complex binary operation that can be decomposed into a simple
         * binary operation on both the real and imagary parts.
         */
        MathBackendWebGL.prototype.complexSeparableBinaryOp = function (a, b, op) {
            var _this = this;
            var aData = this.texData.get(a.dataId);
            var bData = this.texData.get(b.dataId);
            var _a = [
                [aData.complexTensors.real, bData.complexTensors.real],
                [aData.complexTensors.imag, bData.complexTensors.imag]
            ].map(function (complexParts) {
                var aPart = complexParts[0], bPart = complexParts[1];
                var aHandle = _this.makeComplexComponentTensorInfo(a, aPart);
                var bHandle = _this.makeComplexComponentTensorInfo(b, bPart);
                var program = new BinaryOpProgram(op, a.shape, b.shape);
                return _this.compileAndRun(program, [aHandle, bHandle], tf.upcastType(aPart.dtype, bPart.dtype));
            }), real = _a[0], imag = _a[1];
            var complex = this.complex(real, imag);
            real.dispose();
            imag.dispose();
            return complex;
        };
        // Returns a TensorInfo with the complex shape and the dataId of the
        // underlying part. We need to do this because a reshaped complex tensor is
        // not reflected in its parts.
        MathBackendWebGL.prototype.makeComplexComponentTensorInfo = function (complexTensor, complexPart) {
            return {
                dataId: complexPart.dataId,
                dtype: complexPart.dtype,
                shape: complexTensor.shape
            };
        };
        MathBackendWebGL.prototype.addN = function (tensors) {
            if (tensors.length === 1) {
                return tensors[0];
            }
            // Limit the number of uploaded textures for optimization.
            if (tensors.length > tf.env().get('WEBGL_MAX_TEXTURES_IN_SHADER')) {
                var midIndex = Math.floor(tensors.length / 2);
                var leftSide = this.addN(tensors.slice(0, midIndex));
                var rightSide = this.addN(tensors.slice(midIndex));
                return this.addN([leftSide, rightSide]);
            }
            var dtype = tensors.map(function (t) { return t.dtype; }).reduce(function (d1, d2) { return tf.upcastType(d1, d2); });
            var shapes = tensors.map(function (t) { return t.shape; });
            // We can make sure shapes are identical in op level.
            var usePackedOp = tf.env().getBool('WEBGL_PACK');
            var program = usePackedOp ?
                new AddNPackedProgram(tensors[0].shape, shapes) :
                new AddNProgram(tensors[0].shape, shapes);
            return this.compileAndRun(program, tensors, dtype);
        };
        MathBackendWebGL.prototype.subtract = function (a, b) {
            if (a.dtype === 'complex64' && b.dtype === 'complex64') {
                return this.complexSeparableBinaryOp(a, b, SUB);
            }
            if (this.shouldExecuteOnCPU([a, b])) {
                return this.cpuBackend.subtract(a, b);
            }
            var dtype = tf.upcastType(a.dtype, b.dtype);
            if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
                return this.packedBinaryOp(a, b, SUB, a.dtype);
            }
            var program = new BinaryOpProgram(SUB, a.shape, b.shape);
            return this.compileAndRun(program, [a, b], dtype);
        };
        MathBackendWebGL.prototype.pow = function (a, b) {
            var usePackedOp = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS');
            var program = usePackedOp ?
                new BinaryOpPackedProgram(POW$1, a.shape, b.shape) :
                new BinaryOpProgram(POW, a.shape, b.shape);
            var dtype = tf.upcastType(a.dtype, b.dtype);
            return this.compileAndRun(program, [a, b], dtype);
        };
        MathBackendWebGL.prototype.ceil = function (x) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.ceil(x);
            }
            if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
                return this.packedUnaryOp(x, CEIL, x.dtype);
            }
            var program = new UnaryOpProgram(x.shape, CEIL);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.floor = function (x) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.floor(x);
            }
            if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
                return this.packedUnaryOp(x, FLOOR, x.dtype);
            }
            var program = new UnaryOpProgram(x.shape, FLOOR);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.sign = function (x) {
            var program = new UnaryOpProgram(x.shape, SIGN);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.isNaN = function (x) {
            var program = new UnaryOpProgram(x.shape, IS_NAN);
            return this.compileAndRun(program, [x], 'bool');
        };
        MathBackendWebGL.prototype.isInf = function (x) {
            var program = new UnaryOpProgram(x.shape, IS_INF);
            return this.compileAndRun(program, [x], 'bool');
        };
        MathBackendWebGL.prototype.isFinite = function (x) {
            var program = new UnaryOpProgram(x.shape, IS_FINITE);
            return this.compileAndRun(program, [x], 'bool');
        };
        MathBackendWebGL.prototype.round = function (x) {
            var program = new UnaryOpProgram(x.shape, ROUND);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.exp = function (x) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.exp(x);
            }
            if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
                return this.packedUnaryOp(x, EXP, x.dtype);
            }
            var program = new UnaryOpProgram(x.shape, EXP);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.expm1 = function (x) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.expm1(x);
            }
            if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
                return this.packedUnaryOp(x, EXPM1, x.dtype);
            }
            var program = new UnaryOpProgram(x.shape, EXPM1);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.softmax = function (logits, dim) {
            var axes = tf.util.parseAxisParam([dim], logits.shape);
            // TODO(annxingyuan): Call maxImpl rather than op as part of softmax kernel
            // modularization.
            var maxLogit = tf.max(logits, axes);
            var expandedShape = tf.backend_util.expandShapeToKeepDim(maxLogit.shape, axes);
            var a = this.subtract(logits, maxLogit.reshape(expandedShape));
            var b = this.exp(a);
            var sumExp = this.sum(b, axes).reshape(expandedShape);
            // TODO(annxingyuan): Call divImpl rather than op as part of softmax kernel
            // modularization.
            return tf.div(b, sumExp);
        };
        MathBackendWebGL.prototype.log = function (x) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.log(x);
            }
            if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
                return this.packedUnaryOp(x, LOG$1, x.dtype);
            }
            var program = new UnaryOpProgram(x.shape, LOG);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.log1p = function (x) {
            var program = new UnaryOpProgram(x.shape, LOG1P);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.sqrt = function (x) {
            var program = new UnaryOpProgram(x.shape, SQRT);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.rsqrt = function (x) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.rsqrt(x);
            }
            var program = new UnaryOpProgram(x.shape, RSQRT);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.reciprocal = function (x) {
            var program = new UnaryOpProgram(x.shape, RECIPROCAL);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.relu = function (x) {
            var program;
            if (tf.env().getBool('WEBGL_PACK')) {
                program = new UnaryOpPackedProgram(x.shape, RELU$1);
            }
            else {
                program = new UnaryOpProgram(x.shape, RELU);
            }
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.relu6 = function (x) {
            var program;
            if (tf.env().getBool('WEBGL_PACK')) {
                program = new UnaryOpPackedProgram(x.shape, RELU6$1);
            }
            else {
                program = new UnaryOpProgram(x.shape, RELU6);
            }
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.prelu = function (x, alpha) {
            var program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
                new BinaryOpPackedProgram(PRELU$1, x.shape, alpha.shape) :
                new BinaryOpProgram(PRELU, x.shape, alpha.shape);
            return this.compileAndRun(program, [x, alpha]);
        };
        MathBackendWebGL.prototype.elu = function (x) {
            if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
                return this.packedUnaryOp(x, ELU$1, x.dtype);
            }
            var program = new UnaryOpProgram(x.shape, ELU);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.eluDer = function (dy, y) {
            var program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
                new BinaryOpPackedProgram(ELU_DER$1, dy.shape, y.shape) :
                new BinaryOpProgram(ELU_DER, dy.shape, y.shape);
            return this.compileAndRun(program, [dy, y]);
        };
        MathBackendWebGL.prototype.selu = function (x) {
            var program = new UnaryOpProgram(x.shape, SELU);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.int = function (x) {
            var program = new UnaryOpProgram(x.shape, TO_INT);
            return this.compileAndRun(program, [x], 'int32');
        };
        MathBackendWebGL.prototype.clip = function (x, min, max) {
            var program;
            if (tf.env().getBool('WEBGL_PACK_CLIP')) {
                program = new ClipPackedProgram(x.shape);
            }
            else {
                program = new ClipProgram(x.shape);
            }
            var customSetup = program.getCustomSetupFunc(min, max);
            return this.compileAndRun(program, [x], null, customSetup);
        };
        MathBackendWebGL.prototype.abs = function (x) {
            if (this.shouldExecuteOnCPU([x])) {
                return this.cpuBackend.abs(x);
            }
            if (tf.env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
                return this.packedUnaryOp(x, ABS, x.dtype);
            }
            var program = new UnaryOpProgram(x.shape, ABS);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.complexAbs = function (x) {
            var xData = this.texData.get(x.dataId);
            var program = new ComplexAbsProgram(x.shape);
            var inputs = [
                this.makeComplexComponentTensorInfo(x, xData.complexTensors.real),
                this.makeComplexComponentTensorInfo(x, xData.complexTensors.imag),
            ];
            return this.compileAndRun(program, inputs);
        };
        MathBackendWebGL.prototype.sigmoid = function (x) {
            var program = new UnaryOpProgram(x.shape, SIGMOID);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.softplus = function (x) {
            var program = new UnaryOpProgram(x.shape, SOFTPLUS);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.sin = function (x) {
            var program = new UnaryOpProgram(x.shape, SIN);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.cos = function (x) {
            var program = new UnaryOpProgram(x.shape, COS);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.tan = function (x) {
            var program = new UnaryOpProgram(x.shape, TAN);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.asin = function (x) {
            var program = new UnaryOpProgram(x.shape, ASIN);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.acos = function (x) {
            var program = new UnaryOpProgram(x.shape, ACOS);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.atan = function (x) {
            var program = new UnaryOpProgram(x.shape, ATAN);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.atan2 = function (a, b) {
            var program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
                new BinaryOpPackedProgram(ATAN2$1, a.shape, b.shape) :
                new BinaryOpProgram(ATAN2, a.shape, b.shape);
            return this.compileAndRun(program, [a, b]);
        };
        MathBackendWebGL.prototype.sinh = function (x) {
            var program = new UnaryOpProgram(x.shape, SINH);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.cosh = function (x) {
            var program = new UnaryOpProgram(x.shape, COSH);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.tanh = function (x) {
            var program = new UnaryOpProgram(x.shape, TANH);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.asinh = function (x) {
            var program = new UnaryOpProgram(x.shape, ASINH);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.acosh = function (x) {
            var program = new UnaryOpProgram(x.shape, ACOSH);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.atanh = function (x) {
            var program = new UnaryOpProgram(x.shape, ATANH);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.erf = function (x) {
            var program = new UnaryOpProgram(x.shape, ERF);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.step = function (x, alpha) {
            var program = new UnaryOpProgram(x.shape, STEP(alpha));
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.conv2dByMatMul = function (x, filter, convInfo, bias, activation, preluActivationWeights) {
            // Reshapes conv2D input to 2D tensors, uses matMul and then reshape the
            // result from 2D to 4D.
            var xShape = x.shape;
            var xTexData = this.texData.get(x.dataId);
            var sharedMatMulDim = convInfo.inChannels;
            var outerShapeX = xShape[0] * xShape[1] * xShape[2];
            var outerShapeFilter = convInfo.outChannels;
            var isChannelsLast = convInfo.dataFormat === 'channelsLast';
            var transposeA = false;
            var transposeB = false;
            // TODO: Once reduction ops are packed, batchMatMul will always be packed
            // and we can remove this condition.
            var batchMatMulWillBeUnpacked = (outerShapeX === 1 || outerShapeFilter === 1) &&
                sharedMatMulDim > MATMUL_SHARED_DIM_THRESHOLD;
            var reshapeWillBeExpensive = xShape[2] % 2 !== 0 && !!xTexData.isPacked;
            if (batchMatMulWillBeUnpacked || !tf.env().getBool('WEBGL_LAZILY_UNPACK') ||
                !tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ||
                !reshapeWillBeExpensive) {
                var targetShape_1 = isChannelsLast ? xShape[0] * xShape[1] * xShape[2] :
                    xShape[0] * xShape[2] * xShape[3];
                var xReshaped_1 = this.reshape(x, [1, targetShape_1, convInfo.inChannels]);
                var filterReshaped_1 = this.reshape(filter, [1, convInfo.inChannels, convInfo.outChannels]);
                return this.reshape(this.fusedBatchMatMul({
                    a: xReshaped_1,
                    b: filterReshaped_1,
                    transposeA: transposeA,
                    transposeB: transposeB,
                    bias: bias,
                    activation: activation,
                    preluActivationWeights: preluActivationWeights
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
            var targetShape = isChannelsLast ?
                xShape[0] * xShape[1] * (xShape[2] + 1) :
                xShape[0] * xShape[2] * (xShape[3] + 1);
            var xReshaped = {
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
            var originalXTexDataShape = xTexData.shape;
            xTexData.shape = xTexData.shape.slice();
            xTexData.shape[xTexData.shape.length - 2]++;
            tf.util.assert(isReshapeFree(xTexData.shape, xReshaped.shape), function () { return "packed reshape " + xTexData.shape + " to " + xReshaped.shape + " isn't free"; });
            var filterReshaped = this.reshape(filter, [1, convInfo.inChannels, convInfo.outChannels]);
            var pointwiseConv = this.fusedBatchMatMul({
                a: xReshaped,
                b: filterReshaped,
                transposeA: transposeA,
                transposeB: transposeB,
                bias: bias,
                activation: activation,
                preluActivationWeights: preluActivationWeights
            });
            var pointwiseConvTexData = this.texData.get(pointwiseConv.dataId);
            tf.util.assert(pointwiseConvTexData.isPacked, function () { return 'batchMatMul result is expected to be packed'; });
            // Restore the input shape to original.
            xTexData.shape = originalXTexDataShape;
            // Set the output shape - there is no need for expensive reshape as data
            // layout is already correct.
            pointwiseConvTexData.shape = convInfo.outShape;
            return tf.engine().makeTensorFromDataId(pointwiseConv.dataId, convInfo.outShape, pointwiseConv.dtype);
        };
        MathBackendWebGL.prototype.conv2dWithIm2Row = function (x, filter, convInfo, bias, activation, preluActivationWeights) {
            // Rearranges conv2d input so each block to be convolved over forms the
            // column of a new matrix with shape [filterWidth * filterHeight *
            // inChannels, outHeight * outWidth]. The filter is also rearranged so each
            // output channel forms a row of a new matrix with shape [outChannels,
            // filterWidth * filterHeight * inChannels]. The convolution is then
            // computed by multiplying these matrices and reshaping the result.
            var filterWidth = convInfo.filterWidth, filterHeight = convInfo.filterHeight, inChannels = convInfo.inChannels, outWidth = convInfo.outWidth, outHeight = convInfo.outHeight, dataFormat = convInfo.dataFormat;
            var isChannelsLast = dataFormat === 'channelsLast';
            var sharedDim = filterWidth * filterHeight * inChannels;
            var numCols = outHeight * outWidth;
            var x2ColShape = [sharedDim, numCols];
            var transposeA = true;
            var transposeB = false;
            var xSqueezed = x.squeeze([0]);
            var w2Row = filter.reshape([1, sharedDim, -1]);
            var im2ColProgram = new Im2ColPackedProgram(x2ColShape, xSqueezed.shape, convInfo);
            var im2Col = this.compileAndRun(im2ColProgram, [xSqueezed]).reshape([
                1, x2ColShape[0], x2ColShape[1]
            ]);
            var hasBias = bias != null;
            var hasPreluActivationWeights = preluActivationWeights != null;
            var fusedActivation = activation ? mapActivationToShaderProgram(activation, true) : null;
            var matmulProgram = new MatMulPackedProgram(im2Col.shape, [1, numCols, convInfo.outChannels], transposeA, transposeB, hasBias, fusedActivation, hasPreluActivationWeights);
            var inputs = [im2Col, w2Row];
            if (bias) {
                inputs.push(bias);
            }
            if (hasPreluActivationWeights) {
                inputs.push(preluActivationWeights);
            }
            var product = this.compileAndRun(matmulProgram, inputs);
            if (isChannelsLast) {
                return product.reshape([1, outHeight, outWidth, convInfo.outChannels]);
            }
            else {
                return product.reshape([1, convInfo.outChannels, outHeight, outWidth]);
            }
        };
        MathBackendWebGL.prototype.fusedConv2d = function (_a) {
            var input = _a.input, filter = _a.filter, convInfo = _a.convInfo, bias = _a.bias, activation = _a.activation, preluActivationWeights = _a.preluActivationWeights;
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
            var hasBias = bias != null;
            var hasPreluActivationWeights = preluActivationWeights != null;
            var fusedActivation = activation ? mapActivationToShaderProgram(activation, false) : null;
            var program = new Conv2DProgram(convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
            var inputs = [input, filter];
            if (bias) {
                inputs.push(bias);
            }
            if (preluActivationWeights) {
                inputs.push(preluActivationWeights);
            }
            return this.compileAndRun(program, inputs);
        };
        MathBackendWebGL.prototype.conv2d = function (x, filter, convInfo) {
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
            var program = new Conv2DProgram(convInfo);
            return this.compileAndRun(program, [x, filter]);
        };
        MathBackendWebGL.prototype.conv2dDerInput = function (dy, filter, convInfo) {
            var program = new Conv2DDerInputProgram(convInfo);
            return this.compileAndRun(program, [dy, filter]);
        };
        MathBackendWebGL.prototype.conv2dDerFilter = function (x, dy, convInfo) {
            var program = new Conv2DDerFilterProgram(convInfo);
            return this.compileAndRun(program, [x, dy]);
        };
        MathBackendWebGL.prototype.fusedDepthwiseConv2D = function (_a) {
            var input = _a.input, filter = _a.filter, convInfo = _a.convInfo, bias = _a.bias, activation = _a.activation, preluActivationWeights = _a.preluActivationWeights;
            var shouldPackDepthwiseConv = tf.env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
                convInfo.strideWidth <= 2 &&
                convInfo.outChannels / convInfo.inChannels === 1;
            var fusedActivation = activation ?
                mapActivationToShaderProgram(activation, shouldPackDepthwiseConv) :
                null;
            var inputs = [input, filter];
            var hasBias = bias != null;
            var hasPreluActivationWeights = preluActivationWeights != null;
            if (hasBias) {
                inputs.push(bias);
            }
            if (hasPreluActivationWeights) {
                inputs.push(preluActivationWeights);
            }
            var program;
            if (shouldPackDepthwiseConv) {
                program = new DepthwiseConvPacked2DProgram(convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
                return this.compileAndRun(program, inputs);
            }
            program = new DepthwiseConv2DProgram(convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
            return this.compileAndRun(program, inputs);
        };
        MathBackendWebGL.prototype.depthwiseConv2D = function (x, filter, convInfo) {
            var program;
            if (tf.env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
                convInfo.strideWidth <= 2 &&
                convInfo.outChannels / convInfo.inChannels === 1) {
                program = new DepthwiseConvPacked2DProgram(convInfo);
                return this.compileAndRun(program, [x, filter]);
            }
            program = new DepthwiseConv2DProgram(convInfo);
            return this.compileAndRun(program, [x, filter]);
        };
        MathBackendWebGL.prototype.depthwiseConv2DDerInput = function (dy, filter, convInfo) {
            var program = new DepthwiseConv2DDerInputProgram(convInfo);
            return this.compileAndRun(program, [dy, filter]);
        };
        MathBackendWebGL.prototype.depthwiseConv2DDerFilter = function (x, dy, convInfo) {
            var program = new DepthwiseConv2DDerFilterProgram(convInfo);
            return this.compileAndRun(program, [x, dy]);
        };
        MathBackendWebGL.prototype.conv3d = function (x, filter, convInfo) {
            var program = new Conv3DProgram(convInfo);
            return this.compileAndRun(program, [x, filter]);
        };
        MathBackendWebGL.prototype.conv3dDerInput = function (dy, filter, convInfo) {
            var program = new Conv3DDerInputProgram(convInfo);
            return this.compileAndRun(program, [dy, filter]);
        };
        MathBackendWebGL.prototype.conv3dDerFilter = function (x, dy, convInfo) {
            var program = new Conv3DDerFilterProgram(convInfo);
            return this.compileAndRun(program, [x, dy]);
        };
        MathBackendWebGL.prototype.maxPool = function (x, convInfo) {
            var program = new Pool2DProgram(convInfo, 'max', false);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.avgPool = function (x, convInfo) {
            var program = new Pool2DProgram(convInfo, 'avg', false);
            return this.compileAndRun(program, [x], 'float32');
        };
        MathBackendWebGL.prototype.maxPoolBackprop = function (dy, x, y, convInfo) {
            var getPositions = true;
            var maxPoolPositionsProgram = new Pool2DProgram(convInfo, 'max', getPositions);
            var maxPoolPositions = this.compileAndRun(maxPoolPositionsProgram, [x]);
            var maxPoolBackPropProgram = new MaxPool2DBackpropProgram(convInfo);
            var result = this.compileAndRun(maxPoolBackPropProgram, [dy, maxPoolPositions], x.dtype);
            maxPoolPositions.dispose();
            return result;
        };
        MathBackendWebGL.prototype.avgPoolBackprop = function (dy, x, convInfo) {
            var avgPoolBackpropProgram = new AvgPool2DBackpropProgram(convInfo);
            return this.compileAndRun(avgPoolBackpropProgram, [dy], x.dtype);
        };
        MathBackendWebGL.prototype.cast = function (x, dtype) {
            return tf.backend_util.castTensor(x, dtype, this);
        };
        MathBackendWebGL.prototype.unstack = function (x, axis) {
            var num = x.shape[axis];
            var outShape = new Array(x.rank - 1);
            var outIndex = 0;
            for (var i = 0; i < x.rank; i++) {
                if (i !== axis) {
                    outShape[outIndex++] = x.shape[i];
                }
            }
            var begin = new Array(x.rank).fill(0);
            var size = x.shape.slice();
            size[axis] = 1;
            var res = new Array(num);
            for (var i = 0; i < res.length; i++) {
                begin[axis] = i;
                res[i] = this.slice(x, begin, size).reshape(outShape);
            }
            return res;
        };
        MathBackendWebGL.prototype.avgPool3d = function (x, convInfo) {
            var program = new Pool3DProgram(convInfo, 'avg', false);
            return this.compileAndRun(program, [x], 'float32');
        };
        MathBackendWebGL.prototype.avgPool3dBackprop = function (dy, x, convInfo) {
            var avgPool3dBackpropProgram = new AvgPool3DBackpropProgram(convInfo);
            return this.compileAndRun(avgPool3dBackpropProgram, [dy], x.dtype);
        };
        MathBackendWebGL.prototype.maxPool3d = function (x, convInfo) {
            var program = new Pool3DProgram(convInfo, 'max', false);
            return this.compileAndRun(program, [x], 'float32');
        };
        MathBackendWebGL.prototype.maxPool3dBackprop = function (dy, x, y, convInfo) {
            var getPositions = true;
            var maxPool3dPositionsProgram = new Pool3DProgram(convInfo, 'max', getPositions);
            var maxPool3dPositions = this.compileAndRun(maxPool3dPositionsProgram, [x]);
            var maxPool3dBackPropProgram = new MaxPool3DBackpropProgram(convInfo);
            var result = this.compileAndRun(maxPool3dBackPropProgram, [dy, maxPool3dPositions], x.dtype);
            maxPool3dPositions.dispose();
            return result;
        };
        MathBackendWebGL.prototype.reshape = function (x, shape) {
            var texData = this.texData.get(x.dataId);
            if (texData.isPacked && !isReshapeFree(x.shape, shape) &&
                !(texData.texture !== null &&
                    isReshapeFree(texData.shape, shape))) {
                var info = this.packedReshape(x, shape);
                return tf.engine().makeTensorFromDataId(info.dataId, info.shape, info.dtype);
            }
            return tf.backend_util.reshapeTensor(x, shape);
        };
        MathBackendWebGL.prototype.resizeBilinear = function (x, newHeight, newWidth, alignCorners) {
            var program = tf.env().getBool('WEBGL_PACK_IMAGE_OPERATIONS') ?
                new ResizeBilinearPackedProgram(x.shape, newHeight, newWidth, alignCorners) :
                new ResizeBilinearProgram(x.shape, newHeight, newWidth, alignCorners);
            return this.compileAndRun(program, [x], 'float32');
        };
        MathBackendWebGL.prototype.resizeBilinearBackprop = function (dy, x, alignCorners) {
            var program = new ResizeBilinearBackpropProgram(dy, x, alignCorners);
            return this.compileAndRun(program, [dy]);
        };
        MathBackendWebGL.prototype.resizeNearestNeighbor = function (x, newHeight, newWidth, alignCorners) {
            var program = new ResizeNearestNeighborProgram(x.shape, newHeight, newWidth, alignCorners);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.resizeNearestNeighborBackprop = function (dy, x, alignCorners) {
            var program = new ResizeNearestNeigborBackpropProgram(dy, x, alignCorners);
            return this.compileAndRun(program, [dy]);
        };
        MathBackendWebGL.prototype.multinomial = function (logits, normalized, numSamples, seed) {
            var probs = normalized ? logits : tf.softmax(logits);
            var batchSize = probs.shape[0];
            var numOutcomes = probs.shape[1];
            var program = new MultinomialProgram(batchSize, numOutcomes, numSamples);
            var customSetup = program.getCustomSetupFunc(seed);
            return this.compileAndRun(program, [probs], 'int32', customSetup);
        };
        MathBackendWebGL.prototype.oneHot = function (indices, depth, onValue, offValue) {
            var program = new OneHotProgram(indices.size, depth, onValue, offValue);
            return this.compileAndRun(program, [indices]);
        };
        MathBackendWebGL.prototype.diag = function (x) {
            var program = new DiagProgram(x.size);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.cropAndResize = function (image, boxes, boxIndex, cropSize, method, extrapolationValue) {
            var program = new CropAndResizeProgram(image.shape, boxes.shape, cropSize, method, extrapolationValue);
            return this.compileAndRun(program, [image, boxes, boxIndex], 'float32');
        };
        MathBackendWebGL.prototype.depthToSpace = function (x, blockSize, dataFormat) {
            tf.util.assert(blockSize > 1, function () {
                return "blockSize should be > 1 for depthToSpace, but was: " + blockSize;
            });
            var batchSize = x.shape[0];
            var inputHeight = (dataFormat === 'NHWC') ? x.shape[1] : x.shape[2];
            var inputWidth = (dataFormat === 'NHWC') ? x.shape[2] : x.shape[3];
            var inputDepth = (dataFormat === 'NHWC') ? x.shape[3] : x.shape[1];
            var outputHeight = inputHeight * blockSize;
            var outputWidth = inputWidth * blockSize;
            var outputDepth = inputDepth / (blockSize * blockSize);
            var outputShape = (dataFormat === 'NHWC') ?
                [batchSize, outputHeight, outputWidth, outputDepth] :
                [batchSize, outputDepth, outputHeight, outputWidth];
            var program = new DepthToSpaceProgram(outputShape, blockSize, dataFormat);
            return this.compileAndRun(program, [x]);
        };
        MathBackendWebGL.prototype.split = function (x, sizeSplits, axis) {
            return split(x, sizeSplits, axis);
        };
        MathBackendWebGL.prototype.scatterND = function (indices, updates, shape) {
            var _a = tf.backend_util.calculateShapes(updates, indices, shape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, sliceSize = _a.sliceSize, strides = _a.strides, outputSize = _a.outputSize;
            var flattenShape = [outputSize / sliceSize, sliceSize];
            var flattenIndices = indices.reshape([numUpdates, sliceRank]);
            var flattenX = updates.reshape([numUpdates, sliceSize]);
            if (outputSize === 0) {
                return tf.backend_util.reshapeTensor(tf.tensor([]), shape);
            }
            var defaultValue = tf.scalar(0);
            var program = new ScatterProgram(numUpdates, sliceRank, flattenIndices.rank, flattenX.rank, strides, flattenShape);
            var res = this.compileAndRun(program, [flattenX, flattenIndices, defaultValue]);
            return res.reshape(shape);
        };
        MathBackendWebGL.prototype.sparseToDense = function (sparseIndices, sparseValues, outputShape, defaultValue) {
            var _a = tf.backend_util.calculateShapes(sparseValues, sparseIndices, outputShape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, strides = _a.strides, outputSize = _a.outputSize;
            var sumDupeIndices = false;
            var program = new ScatterProgram(numUpdates, sliceRank, sparseIndices.rank, sparseValues.rank, strides, [outputSize, 1], sumDupeIndices);
            var res = this.compileAndRun(program, [sparseValues, sparseIndices, defaultValue]);
            return res.reshape(outputShape);
        };
        MathBackendWebGL.prototype.fft = function (x) {
            var inverse = false;
            return this.fftImpl(x, inverse);
        };
        MathBackendWebGL.prototype.ifft = function (x) {
            var inverse = true;
            return this.fftImpl(x, inverse);
        };
        MathBackendWebGL.prototype.fftImpl = function (x, inverse) {
            var xData = this.texData.get(x.dataId);
            var realProgram = new FFTProgram(COMPLEX_FFT.REAL, x.shape, inverse);
            var imagProgram = new FFTProgram(COMPLEX_FFT.IMAG, x.shape, inverse);
            var inputs = [
                this.makeComplexComponentTensorInfo(x, xData.complexTensors.real),
                this.makeComplexComponentTensorInfo(x, xData.complexTensors.imag),
            ];
            var real = this.compileAndRun(realProgram, inputs);
            var imag = this.compileAndRun(imagProgram, inputs);
            var complex = this.complex(real, imag).as2D(x.shape[0], x.shape[1]);
            real.dispose();
            imag.dispose();
            return complex;
        };
        MathBackendWebGL.prototype.gatherND = function (x, indices) {
            var indicesShape = indices.shape;
            var sliceRank = indicesShape[indicesShape.length - 1];
            var _a = tf.backend_util.prepareAndValidate(x, indices), resultShape = _a[0], numSlices = _a[1], sliceSize = _a[2], strides = _a[3];
            var flattenIndices = indices.reshape([numSlices, sliceRank]);
            var flattenX = x.reshape([x.size / sliceSize, sliceSize]);
            var program = new GatherNDProgram(sliceRank, strides, [numSlices, sliceSize]);
            var res = this.compileAndRun(program, [flattenX, flattenIndices]);
            return res.reshape(resultShape);
        };
        MathBackendWebGL.prototype.fill = function (shape, value, dtype) {
            dtype = dtype || tf.util.inferDtype(value);
            if (dtype === 'string') {
                // String type should be handled in CPU memory.
                var values = tf.util.getArrayFromDType(dtype, tf.util.sizeFromShape(shape));
                values.fill(value);
                return tf.engine().makeTensor(values, shape, dtype, this);
            }
            else {
                var program = new FillProgram(shape, value);
                var customSetup = program.getCustomSetupFunc(value);
                return this.compileAndRun(program, [], dtype, customSetup);
            }
        };
        MathBackendWebGL.prototype.onesLike = function (x) {
            if (x.dtype === 'string') {
                throw new Error('onesLike is not supported under string dtype');
            }
            else {
                // TODO(cais, smilkov): Add WebGL shader for onesLike:
                //   https://github.com/tensorflow/tfjs/issues/1293
                return this.fill(x.shape, 1, x.dtype);
            }
        };
        MathBackendWebGL.prototype.zerosLike = function (x) {
            return this.fill(x.shape, x.dtype === 'string' ? '' : 0, x.dtype);
        };
        MathBackendWebGL.prototype.linspace = function (start, stop, num) {
            // TODO: Use CPU implementation due to the precision problem in Safari.
            return tf.backend_util.linspaceImpl(start, stop, num);
        };
        MathBackendWebGL.prototype.makeTensorInfo = function (shape, dtype) {
            var dataId = this.write(null /* values */, shape, dtype);
            this.texData.get(dataId).usage = null;
            return { dataId: dataId, shape: shape, dtype: dtype };
        };
        MathBackendWebGL.prototype.makeOutput = function (shape, dtype) {
            var dataId = this.makeTensorInfo(shape, dtype).dataId;
            return tf.engine().makeTensorFromDataId(dataId, shape, dtype, this);
        };
        MathBackendWebGL.prototype.unpackTensor = function (input) {
            var program = new UnpackProgram(input.shape);
            return this.runWebGLProgram(program, [input], input.dtype);
        };
        MathBackendWebGL.prototype.packTensor = function (input) {
            var program = new PackProgram(input.shape);
            var preventEagerUnpackingOutput = true;
            return this.runWebGLProgram(program, [input], input.dtype, null /* customSetup */, preventEagerUnpackingOutput);
        };
        MathBackendWebGL.prototype.packedReshape = function (input, afterShape) {
            var input3DShape = [
                getBatchDim(input.shape)
            ].concat(getRowsCols(input.shape));
            var input3D = {
                dtype: input.dtype,
                shape: input3DShape,
                dataId: input.dataId
            };
            var afterShapeAs3D = [
                getBatchDim(afterShape)
            ].concat(getRowsCols(afterShape));
            var program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
            var preventEagerUnpackingOfOutput = true;
            var output = this.runWebGLProgram(program, [input3D], input.dtype, null /* customSetup */, preventEagerUnpackingOfOutput);
            return { dataId: output.dataId, shape: afterShape, dtype: output.dtype };
        };
        MathBackendWebGL.prototype.decode = function (dataId) {
            var texData = this.texData.get(dataId);
            var isPacked = texData.isPacked, shape = texData.shape, dtype = texData.dtype;
            var shapeAs3D = getShapeAs3D(shape);
            var program;
            if (isPacked) {
                program = new DecodeMatrixPackedProgram(shapeAs3D);
            }
            else {
                program = new DecodeMatrixProgram(shapeAs3D);
            }
            var preventEagerUnpackingOfOutput = true;
            var out = this.runWebGLProgram(program, [{ shape: shapeAs3D, dtype: dtype, dataId: dataId }], dtype, null /* customSetup */, preventEagerUnpackingOfOutput);
            return { dtype: dtype, shape: shape, dataId: out.dataId };
        };
        MathBackendWebGL.prototype.runWebGLProgram = function (program, inputs, outputDtype, customSetup, preventEagerUnpackingOfOutput) {
            var _this = this;
            if (preventEagerUnpackingOfOutput === void 0) { preventEagerUnpackingOfOutput = false; }
            var output = this.makeTensorInfo(program.outputShape, outputDtype);
            var outData = this.texData.get(output.dataId);
            if (program.packedOutput) {
                outData.isPacked = true;
            }
            if (program.outPackingScheme === PackingScheme.DENSE) {
                var texelShape = getDenseTexShape(program.outputShape);
                // For a densely packed output, we explicitly set texShape
                // so it doesn't get assigned later according to our typical packing
                // scheme wherein a single texel can only contain values from adjacent
                // rows/cols.
                outData.texShape = texelShape.map(function (d) { return d * 2; });
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
            var dataToDispose = [];
            var inputsData = inputs.map(function (input) {
                if (input.dtype === 'complex64') {
                    throw new Error("GPGPUProgram does not support complex64 input. For complex64 " +
                        "dtypes, please separate the program into real and imaginary " +
                        "parts.");
                }
                var texData = _this.texData.get(input.dataId);
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
                    input = texData.isPacked ? _this.unpackTensor(input) :
                        _this.packTensor(input);
                    dataToDispose.push(input);
                    texData = _this.texData.get(input.dataId);
                }
                else if (texData.isPacked &&
                    !isReshapeFree(texData.shape, input.shape)) {
                    // This is a special case where a texture exists for a tensor
                    // but the shapes are incompatible (due to packing constraints) because
                    // the tensor did not have a chance to go through the packed reshape
                    // shader. This only happens when we reshape the *same* tensor to form
                    // *distinct* inputs to an op, e.g. dotting a vector with itself. This
                    // case will disappear once packed uploading is the default.
                    var savedInput = input;
                    var targetShape = input.shape;
                    input.shape = texData.shape;
                    input = _this.packedReshape(input, targetShape);
                    dataToDispose.push(input);
                    texData = _this.texData.get(input.dataId);
                    savedInput.shape = targetShape;
                }
                _this.uploadToGPU(input.dataId);
                return { shape: input.shape, texData: texData, isUniform: false };
            });
            this.uploadToGPU(output.dataId);
            var outputData = { shape: output.shape, texData: outData, isUniform: false };
            var key = makeShaderKey(program, inputsData, outputData);
            var binary = this.getAndSaveBinary(key, function () {
                return compileProgram(_this.gpgpu, program, inputsData, outputData);
            });
            var shouldTimeProgram = this.activeTimers != null;
            var query;
            if (shouldTimeProgram) {
                query = this.startTimer();
            }
            runProgram(this.gpgpu, binary, inputsData, outputData, customSetup);
            dataToDispose.forEach(function (info) { return _this.disposeData(info.dataId); });
            if (shouldTimeProgram) {
                query = this.endTimer(query);
                this.activeTimers.push({ name: program.constructor.name, query: this.getQueryTime(query) });
            }
            if (!tf.env().getBool('WEBGL_LAZILY_UNPACK') && outData.isPacked &&
                preventEagerUnpackingOfOutput === false) {
                var unpacked = this.unpackTensor(output);
                this.disposeData(output.dataId);
                return unpacked;
            }
            return output;
        };
        MathBackendWebGL.prototype.compileAndRun = function (program, inputs, outputDtype, customSetup, preventEagerUnpackingOfOutput) {
            if (preventEagerUnpackingOfOutput === void 0) { preventEagerUnpackingOfOutput = false; }
            outputDtype = outputDtype || inputs[0].dtype;
            var outInfo = this.runWebGLProgram(program, inputs, outputDtype, customSetup, preventEagerUnpackingOfOutput);
            return tf.engine().makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
        };
        MathBackendWebGL.prototype.getAndSaveBinary = function (key, getBinary) {
            if (!(key in this.binaryCache)) {
                this.binaryCache[key] = getBinary();
            }
            return this.binaryCache[key];
        };
        MathBackendWebGL.prototype.getTextureManager = function () {
            return this.textureManager;
        };
        MathBackendWebGL.prototype.dispose = function () {
            var _this = this;
            if (this.disposed) {
                return;
            }
            // Avoid disposing the compiled webgl programs during unit testing because
            // it slows down test execution.
            if (!tf.env().getBool('IS_TEST')) {
                var allKeys = Object.keys(this.binaryCache);
                allKeys.forEach(function (key) {
                    _this.gpgpu.deleteProgram(_this.binaryCache[key].webGLProgram);
                    delete _this.binaryCache[key];
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
        };
        MathBackendWebGL.prototype.floatPrecision = function () {
            var _this = this;
            if (this.floatPrecisionValue == null) {
                this.floatPrecisionValue = tf.tidy(function () {
                    if (!tf.env().get('WEBGL_RENDER_FLOAT32_ENABLED')) {
                        // Momentarily switching DEBUG flag to false so we don't throw an
                        // error trying to upload a small value.
                        var debugFlag = tf.env().getBool('DEBUG');
                        tf.env().set('DEBUG', false);
                        var underflowCheckValue = _this.abs(tf.scalar(1e-8)).dataSync()[0];
                        tf.env().set('DEBUG', debugFlag);
                        if (underflowCheckValue > 0) {
                            return 32;
                        }
                    }
                    return 16;
                });
            }
            return this.floatPrecisionValue;
        };
        /** Returns the smallest representable number.  */
        MathBackendWebGL.prototype.epsilon = function () {
            return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
        };
        MathBackendWebGL.prototype.uploadToGPU = function (dataId) {
            var _a;
            var texData = this.texData.get(dataId);
            var shape = texData.shape, dtype = texData.dtype, values = texData.values, texture = texData.texture, usage = texData.usage, isPacked = texData.isPacked;
            if (texture != null) {
                // Array is already on GPU. No-op.
                return;
            }
            var shouldTimeProgram = this.activeTimers != null;
            var start;
            if (shouldTimeProgram) {
                start = tf.util.now();
            }
            var texShape = texData.texShape;
            if (texShape == null) {
                texShape = getTextureShapeFromLogicalShape(shape, isPacked);
                texData.texShape = texShape;
            }
            if (values != null) {
                var shapeAs3D = getShapeAs3D(shape);
                var program = void 0;
                var width = texShape[1], height = texShape[0];
                var isByteArray = values instanceof Uint8Array;
                if (isPacked) {
                    _a = getPackedMatrixTextureShapeWidthHeight(texShape[0], texShape[1]), width = _a[0], height = _a[1];
                    program = new EncodeMatrixPackedProgram(shapeAs3D, [height, width], isByteArray);
                }
                else {
                    program =
                        new EncodeMatrixProgram(shapeAs3D, [height, width], isByteArray);
                }
                var tempDenseInputHandle = this.makeTensorInfo([height, width], dtype);
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
                var preventEagerUnpacking = true;
                var encodedOutputTarget = this.runWebGLProgram(program, [tempDenseInputHandle], dtype, null, preventEagerUnpacking);
                // Have the original texture assume the identity of the encoded output.
                var outputTexData = this.texData.get(encodedOutputTarget.dataId);
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
                var newTexture = this.acquireTexture(texShape, usage, dtype, isPacked);
                texData.texture = newTexture;
            }
        };
        MathBackendWebGL.prototype.convertAndCacheOnCPU = function (dataId, float32Values) {
            var texData = this.texData.get(dataId);
            var dtype = texData.dtype;
            this.releaseGPUData(dataId);
            if (float32Values != null) {
                texData.values = float32ToTypedArray(float32Values, dtype);
            }
            return texData.values;
        };
        MathBackendWebGL.prototype.acquireTexture = function (texShape, texType, dtype, isPacked) {
            this.numBytesInGPU += this.computeBytes(texShape, dtype);
            if (!this.warnedAboutMemory &&
                this.numBytesInGPU > this.numMBBeforeWarning * 1024 * 1024) {
                var mb = (this.numBytesInGPU / 1024 / 1024).toFixed(2);
                this.warnedAboutMemory = true;
                console.warn("High memory usage in GPU: " + mb + " MB, " +
                    "most likely due to a memory leak");
            }
            return this.textureManager.acquireTexture(texShape, texType, isPacked);
        };
        MathBackendWebGL.prototype.computeBytes = function (shape, dtype) {
            return shape[0] * shape[1] * tf.util.bytesPerElement(dtype);
        };
        return MathBackendWebGL;
    }(tf.KernelBackend));
    function float32ToTypedArray(a, dtype) {
        if (dtype === 'float32' || dtype === 'complex64') {
            return a;
        }
        else if (dtype === 'int32' || dtype === 'bool') {
            var result = (dtype === 'int32') ? new Int32Array(a.length) :
                new Uint8Array(a.length);
            for (var i = 0; i < result.length; ++i) {
                result[i] = Math.round(a[i]);
            }
            return result;
        }
        else {
            throw new Error("Unknown dtype " + dtype);
        }
    }

    /** @license See the LICENSE file. */
    // This code is auto-generated, do not modify this file!
    var version = '0.0.0';

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
        var program = new BinaryOpProgram(DIV, a.shape, b.shape);
        if (tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
            program = new BinaryOpPackedProgram(DIV$1, a.shape, b.shape, true);
        }
        var output = backend.runWebGLProgram(program, [a, b], 'float32');
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
    var divConfig = {
        kernelName: tf.Div,
        backendName: 'webgl',
        kernelFunc: function (_a) {
            var inputs = _a.inputs, backend = _a.backend;
            var _b = inputs, a = _b.a, b = _b.b;
            var webglBackend = backend;
            return divImpl(a, b, webglBackend);
        }
    };

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
    var FromPixelsProgram = /** @class */ (function () {
        function FromPixelsProgram(outputShape) {
            this.variableNames = ['A'];
            var glsl = getGlslDifferences();
            var height = outputShape[0], width = outputShape[1];
            this.outputShape = outputShape;
            this.userCode = "\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int texR = coords[0];\n        int texC = coords[1];\n        int depth = coords[2];\n        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + width + ".0, " + height + ".0);\n\n        vec4 values = " + glsl.texture2D + "(A, uv);\n        float value;\n        if (depth == 0) {\n          value = values.r;\n        } else if (depth == 1) {\n          value = values.g;\n        } else if (depth == 2) {\n          value = values.b;\n        } else if (depth == 3) {\n          value = values.a;\n        }\n\n        setOutput(floor(value * 255.0 + 0.5));\n      }\n    ";
        }
        return FromPixelsProgram;
    }());

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
    var FromPixelsPackedProgram = /** @class */ (function () {
        function FromPixelsPackedProgram(outputShape) {
            this.variableNames = ['A'];
            this.packedInputs = false;
            this.packedOutput = true;
            var glsl = getGlslDifferences();
            var height = outputShape[0], width = outputShape[1];
            this.outputShape = outputShape;
            this.userCode = "\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int texR = coords[0];\n        int texC = coords[1];\n        int depth = coords[2];\n\n        vec4 result = vec4(0.);\n\n        for(int row=0; row<=1; row++) {\n          for(int col=0; col<=1; col++) {\n            texC = coords[1] + row;\n            depth = coords[2] + col;\n\n            vec2 uv = (vec2(texC, texR) + halfCR) /\n                       vec2(" + width + ".0, " + height + ".0);\n            vec4 values = " + glsl.texture2D + "(A, uv);\n            float value;\n            if (depth == 0) {\n              value = values.r;\n            } else if (depth == 1) {\n              value = values.g;\n            } else if (depth == 2) {\n              value = values.b;\n            } else if (depth == 3) {\n              value = values.a;\n            }\n\n            result[row * 2 + col] = floor(value * 255.0 + 0.5);\n          }\n        }\n\n        " + glsl.output + " = result;\n      }\n    ";
        }
        return FromPixelsPackedProgram;
    }());

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
    var fromPixelsConfig = {
        kernelName: tf.FromPixels,
        backendName: 'webgl',
        kernelFunc: fromPixels,
    };
    var fromPixels2DContext;
    function fromPixels(args) {
        var inputs = args.inputs, backend = args.backend, attrs = args.attrs;
        var pixels = inputs.pixels;
        var numChannels = attrs.numChannels;
        var isVideo = typeof (HTMLVideoElement) !== 'undefined' &&
            pixels instanceof HTMLVideoElement;
        var isImage = typeof (HTMLImageElement) !== 'undefined' &&
            pixels instanceof HTMLImageElement;
        var _a = isVideo ?
            [
                pixels.videoWidth,
                pixels.videoHeight
            ] :
            [pixels.width, pixels.height], width = _a[0], height = _a[1];
        var texShape = [height, width];
        var outShape = [height, width, numChannels];
        if (isImage || isVideo) {
            if (fromPixels2DContext == null) {
                fromPixels2DContext = document.createElement('canvas').getContext('2d');
            }
            fromPixels2DContext.canvas.width = width;
            fromPixels2DContext.canvas.height = height;
            fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
            pixels = fromPixels2DContext.canvas;
        }
        var tempPixelHandle = backend.makeTensorInfo(texShape, 'int32');
        // This is a byte texture with pixels.
        backend.texData.get(tempPixelHandle.dataId).usage = TextureUsage.PIXELS;
        backend.gpgpu.uploadPixelDataToTexture(backend.getTexture(tempPixelHandle.dataId), pixels);
        var program = tf.env().getBool('WEBGL_PACK') ?
            new FromPixelsPackedProgram(outShape) :
            new FromPixelsProgram(outShape);
        var res = backend.runWebGLProgram(program, [tempPixelHandle], 'int32');
        backend.disposeData(tempPixelHandle.dataId);
        return res;
    }

    /**
     * @license
     * Copyright 2020 Google Inc. All Rights Reserved.
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
    function maxImpl(aVals, reduceSize, outShape, dtype) {
        const vals = tf.util.getTypedArrayFromDType(dtype, tf.util.sizeFromShape(outShape));
        for (let i = 0; i < vals.length; ++i) {
            const offset = i * reduceSize;
            let max = aVals[offset];
            for (let j = 0; j < reduceSize; ++j) {
                const value = aVals[offset + j];
                if (value > max) {
                    max = value;
                }
            }
            vals[i] = max;
        }
        return vals;
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
    const nonMaxSuppressionV3Impl = tf.kernel_impls.nonMaxSuppressionV3Impl;
    const split$1 = tf.kernel_impls.split;
    const tile$1 = tf.kernel_impls.tile;
    const topkImpl$1 = tf.kernel_impls.topkImpl;
    const whereImpl$1 = tf.kernel_impls.whereImpl;

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
    var maxImplCPU = maxImpl, transposeImplCPU = transposeImpl;

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
    function reduce(x, dtype, reductionType, backend) {
        var _a = x.shape, batchSize = _a[0], inSize = _a[1];
        var windowSize = tf.backend_util.computeOptimalWindowSize(inSize);
        var reduceInfo = { windowSize: windowSize, inSize: inSize, batchSize: batchSize };
        var program = new ReduceProgram(reduceInfo, reductionType);
        var output = backend.runWebGLProgram(program, [x], dtype);
        if (output.shape[1] === 1) {
            return output;
        }
        return reduce(output, dtype, reductionType, backend);
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
    function packedReshape(input, afterShape, backend) {
        var input3DShape = [getBatchDim(input.shape)].concat(getRowsCols(input.shape));
        var input3D = {
            dtype: input.dtype,
            shape: input3DShape,
            dataId: input.dataId
        };
        var afterShapeAs3D = [getBatchDim(afterShape)].concat(getRowsCols(afterShape));
        var program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
        var preventEagerUnpackingOfOutput = true;
        var output = backend.runWebGLProgram(program, [input3D], input.dtype, null /* customSetup */, preventEagerUnpackingOfOutput);
        return { dataId: output.dataId, shape: afterShape, dtype: output.dtype };
    }
    function reshape(x, afterShape, backend) {
        var xTexData = backend.texData.get(x.dataId);
        if (xTexData.isPacked && !isReshapeFree(x.shape, afterShape) &&
            !(xTexData.texture !== null &&
                isReshapeFree(xTexData.shape, afterShape))) {
            return packedReshape(x, afterShape, backend);
        }
        return { dataId: x.dataId, shape: afterShape, dtype: x.dtype };
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
    function maxImpl$1(x, reduceShape, outShape, backend) {
        var inSize = tf.util.sizeFromShape(reduceShape);
        var xSize = tf.util.sizeFromShape(x.shape);
        var batchSize = xSize / inSize;
        var reshapedInput = reshape(x, [batchSize, inSize], backend);
        var reduced = reduce(reshapedInput, x.dtype, 'max', backend);
        if (reshapedInput.dataId !== x.dataId) {
            // dispose the output of the packed reshape.
            backend.disposeData(reshapedInput.dataId);
        }
        return reshape(reduced, outShape, backend);
    }

    /**
     * @license
     * Copyright 2017 Google LLC. All Rights Reserved.
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
    var TransposeProgram = /** @class */ (function () {
        function TransposeProgram(aShape, newDim) {
            this.variableNames = ['A'];
            var outputShape = new Array(aShape.length);
            for (var i = 0; i < outputShape.length; i++) {
                outputShape[i] = aShape[newDim[i]];
            }
            this.outputShape = outputShape;
            this.rank = outputShape.length;
            var dtype = getCoordsDataType(this.rank);
            var switched = getSwitchedCoords(newDim);
            this.userCode = "\n    void main() {\n      " + dtype + " resRC = getOutputCoords();\n      setOutput(getA(" + switched + "));\n    }\n    ";
        }
        return TransposeProgram;
    }());
    function getSwitchedCoords(newDim) {
        var rank = newDim.length;
        if (rank > 6) {
            throw Error("Transpose for rank " + rank + " is not yet supported");
        }
        var originalOrder = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w', 'resRC.u', 'resRC.v'];
        var switchedCoords = new Array(rank);
        for (var i = 0; i < newDim.length; i++) {
            switchedCoords[newDim[i]] = originalOrder[i];
        }
        return switchedCoords.join();
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
    var TransposePackedProgram = /** @class */ (function () {
        function TransposePackedProgram(aShape, newDim) {
            this.variableNames = ['A'];
            this.packedInputs = true;
            this.packedOutput = true;
            var outputShape = new Array(aShape.length);
            for (var i = 0; i < outputShape.length; i++) {
                outputShape[i] = aShape[newDim[i]];
            }
            this.outputShape = outputShape;
            this.rank = outputShape.length;
            if (this.rank > 6) {
                throw Error("Packed transpose for rank " + this.rank + " is not yet supported.");
            }
            var dtype = getCoordsDataType(this.rank);
            var outputOrder = getVecChannels('rc', this.rank);
            var switchedOrder = new Array(this.rank);
            for (var i = 0; i < newDim.length; i++) {
                switchedOrder[newDim[i]] = outputOrder[i];
            }
            var innerDims = "vec2(" + switchedOrder.slice(-2).join() + ")";
            var nextColumn = "++" + outputOrder[this.rank - 1] + " < " + outputShape[this.rank - 1];
            var getc = "getChannel(getA(" + switchedOrder.join() + "), " + innerDims + ")";
            this.userCode = "\n    void main() {\n      " + dtype + " rc = getOutputCoords();\n      vec4 result = vec4(0.);\n      result[0] = " + getc + ";\n      if(" + nextColumn + ") {\n        result[1] = " + getc + ";\n      }\n      --" + outputOrder[this.rank - 1] + ";\n      if(++" + outputOrder[this.rank - 2] + " < " + outputShape[this.rank - 2] + ") {\n        result[2] = " + getc + ";\n        if(" + nextColumn + ") {\n          result[3] = " + getc + ";\n        }\n      }\n      setOutput(result);\n    }\n    ";
        }
        return TransposePackedProgram;
    }());

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
        var program = tf.env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
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
    var maxConfig = {
        kernelName: tf.Max,
        backendName: 'webgl',
        kernelFunc: function (_a) {
            var inputs = _a.inputs, attrs = _a.attrs, backend = _a.backend;
            var x = inputs.x;
            var reductionIndices = attrs.reductionIndices;
            var webglBackend = backend;
            var xRank = x.shape.length;
            var origAxes = tf.util.parseAxisParam(reductionIndices, x.shape);
            var axes = origAxes;
            var permutedAxes = tf.backend_util.getAxesPermutation(axes, xRank);
            var maxInputIsTransposed = permutedAxes != null;
            var shouldExecuteOnCPU = webglBackend.shouldExecuteOnCPU([x]);
            var maxInput = x;
            if (maxInputIsTransposed) {
                if (shouldExecuteOnCPU) {
                    var xTexData = webglBackend.texData.get(maxInput.dataId);
                    var values = xTexData.values;
                    var newShape = new Array(xRank);
                    for (var i = 0; i < newShape.length; i++) {
                        newShape[i] = x.shape[permutedAxes[i]];
                    }
                    var maxInputValues = transposeImplCPU(values, x.shape, x.dtype, permutedAxes, newShape);
                    maxInput = webglBackend.makeTensorInfo(newShape, x.dtype);
                    var maxInputData = webglBackend.texData.get(maxInput.dataId);
                    maxInputData.values = maxInputValues;
                }
                else {
                    maxInput = transposeImpl$1(x, permutedAxes, webglBackend);
                }
                axes = tf.backend_util.getInnerMostAxes(axes.length, xRank);
            }
            tf.backend_util.assertAxesAreInnerMostDims('max', axes, xRank);
            var _b = tf.backend_util.computeOutAndReduceShapes(maxInput.shape, axes), maxOutShape = _b[0], reduceShape = _b[1];
            var out;
            if (shouldExecuteOnCPU) {
                var xTexData = webglBackend.texData.get(maxInput.dataId);
                var values = xTexData.values;
                var outValues = maxImplCPU(values, tf.util.sizeFromShape(reduceShape), maxOutShape, x.dtype);
                out = webglBackend.makeTensorInfo(maxOutShape, x.dtype);
                var outData = webglBackend.texData.get(out.dataId);
                outData.values = outValues;
            }
            else {
                out = maxImpl$1(maxInput, reduceShape, maxOutShape, webglBackend);
            }
            if (maxInputIsTransposed) {
                webglBackend.disposeData(maxInput.dataId);
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
    function maxPoolWithArgmaxImpl(x, includeBatchInIndex, convInfo, backend) {
        var program = new Pool2DProgram(convInfo, 'max', false);
        var poolOutput = backend.runWebGLProgram(program, [x], 'float32');
        program = new Pool2DProgram(convInfo, 'max', true, true, includeBatchInIndex);
        var indexOutput = backend.runWebGLProgram(program, [x], 'float32');
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
    var maxPoolWithArgmaxConfig = {
        kernelName: tf.MaxPoolWithArgmax,
        backendName: 'webgl',
        kernelFunc: function (_a) {
            var inputs = _a.inputs, attrs = _a.attrs, backend = _a.backend;
            var x = inputs.x;
            var _b = attrs, filterSize = _b.filterSize, strides = _b.strides, pad = _b.pad, includeBatchInIndex = _b.includeBatchInIndex;
            var webglBackend = backend;
            tf.util.assert(x.shape.length === 4, function () { return "Error in maxPool: input must be rank 4 but got rank " + x.shape.length + "."; });
            var dilations = [1, 1];
            tf.util.assert(tf.backend_util.eitherStridesOrDilationsAreOne(strides, dilations), function () { return 'Error in maxPool: Either strides or dilations must be 1. ' +
                ("Got strides " + strides + " and dilations '" + dilations + "'"); });
            var convInfo = tf.backend_util.computePool2DInfo(x.shape, filterSize, strides, dilations, pad);
            var _c = maxPoolWithArgmaxImpl(x, includeBatchInIndex, convInfo, webglBackend), result = _c[0], indexes = _c[1];
            return [result, indexes];
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
    var nonMaxSuppressionV3Config = {
        kernelName: tf.NonMaxSuppressionV3,
        backendName: 'webgl',
        kernelFunc: function (_a) {
            var inputs = _a.inputs, backend = _a.backend, attrs = _a.attrs;
            tf.backend_util.warn('tf.nonMaxSuppression() in webgl locks the UI thread. ' +
                'Call tf.nonMaxSuppressionAsync() instead');
            var _b = inputs, boxes = _b.boxes, scores = _b.scores;
            var _c = attrs, maxOutputSize = _c.maxOutputSize, iouThreshold = _c.iouThreshold, scoreThreshold = _c.scoreThreshold;
            var gpuBackend = backend;
            var boxesVals = gpuBackend.readSync(boxes.dataId);
            var scoresVals = gpuBackend.readSync(scores.dataId);
            var maxOutputSizeVal = maxOutputSize;
            var iouThresholdVal = iouThreshold;
            var scoreThresholdVal = scoreThreshold;
            return tf.kernel_impls.nonMaxSuppressionV3Impl(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal);
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
    var nonMaxSuppressionV5Impl = tf.kernel_impls.nonMaxSuppressionV5Impl;
    var nonMaxSuppressionV5Config = {
        kernelName: tf.NonMaxSuppressionV5,
        backendName: 'webgl',
        kernelFunc: function (_a) {
            var inputs = _a.inputs, backend = _a.backend, attrs = _a.attrs;
            tf.backend_util.warn('tf.nonMaxSuppression() in webgl locks the UI thread. ' +
                'Call tf.nonMaxSuppressionAsync() instead');
            var _b = inputs, boxes = _b.boxes, scores = _b.scores;
            var _c = attrs, maxOutputSize = _c.maxOutputSize, iouThreshold = _c.iouThreshold, scoreThreshold = _c.scoreThreshold, softNmsSigma = _c.softNmsSigma;
            var gpuBackend = backend;
            var boxesVals = gpuBackend.readSync(boxes.dataId);
            var scoresVals = gpuBackend.readSync(scores.dataId);
            var maxOutputSizeVal = maxOutputSize;
            var iouThresholdVal = iouThreshold;
            var scoreThresholdVal = scoreThreshold;
            var softNmsSigmaVal = softNmsSigma;
            var _d = nonMaxSuppressionV5Impl(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal, softNmsSigmaVal), selectedIndices = _d.selectedIndices, selectedScores = _d.selectedScores;
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
    var squareConfig = {
        kernelName: tf.Square,
        backendName: 'webgl',
        kernelFunc: function (_a) {
            var inputs = _a.inputs, backend = _a.backend;
            var x = inputs.x;
            var webglBackend = backend;
            var program = new UnaryOpProgram(x.shape, SQUARE);
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
    var squaredDifferenceConfig = {
        kernelName: tf.SquaredDifference,
        backendName: 'webgl',
        kernelFunc: function (_a) {
            var inputs = _a.inputs, backend = _a.backend;
            var _b = inputs, a = _b.a, b = _b.b;
            var SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';
            var webGLBackend = backend;
            var program = tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
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
    var transposeConfig = {
        kernelName: tf.Transpose,
        backendName: 'webgl',
        kernelFunc: function (_a) {
            var inputs = _a.inputs, attrs = _a.attrs, backend = _a.backend;
            var x = inputs.x;
            var perm = attrs.perm;
            var webglBackend = backend;
            var xRank = x.shape.length;
            var newShape = new Array(xRank);
            for (var i = 0; i < newShape.length; i++) {
                newShape[i] = x.shape[perm[i]];
            }
            var out;
            if (webglBackend.shouldExecuteOnCPU([x])) {
                var xTexData = webglBackend.texData.get(x.dataId);
                var values = xTexData.values;
                var outValues = transposeImplCPU(values, x.shape, x.dtype, perm, newShape);
                out = webglBackend.makeTensorInfo(newShape, x.dtype);
                var outData = webglBackend.texData.get(out.dataId);
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
    var kernelConfigs = [
        maxConfig, fromPixelsConfig, divConfig, nonMaxSuppressionV3Config,
        nonMaxSuppressionV5Config, squareConfig, squaredDifferenceConfig,
        transposeConfig, maxPoolWithArgmaxConfig
    ];
    for (var _i = 0, kernelConfigs_1 = kernelConfigs; _i < kernelConfigs_1.length; _i++) {
        var kernelConfig = kernelConfigs_1[_i];
        tf.registerKernel(kernelConfig);
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
    /**
     * Enforce use of half precision textures if available on the platform.
     */
    /** @doc {heading: 'Environment', namespace: 'webgl'} */
    function forceHalfFloat() {
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
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
        tf.registerBackend('webgl', function () { return new MathBackendWebGL(); }, 2 /* priority */);
    }
    var webgl = { forceHalfFloat: forceHalfFloat };

    exports.GPGPUContext = GPGPUContext;
    exports.MathBackendWebGL = MathBackendWebGL;
    exports.forceHalfFloat = forceHalfFloat;
    exports.gpgpu_util = gpgpu_util;
    exports.setWebGLContext = setWebGLContext;
    exports.version_webgl = version;
    exports.webgl = webgl;
    exports.webgl_util = webgl_util;

    Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-backend-webgl.js.map
