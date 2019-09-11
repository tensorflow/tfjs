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

import {describeWithFlags} from '../../jasmine_util';
import {WEBGL_ENVS} from './backend_webgl_test_registry';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_util from './gpgpu_util';
import * as tex_util from './tex_util';
import {getActiveContext} from './webgl_context_manager';

describeWithFlags('gpgpu_util createWebGLContext', WEBGL_ENVS, () => {
  let gpgpu: GPGPUContext;
  let gl: WebGLRenderingContext;

  beforeEach(() => {
    gpgpu = new GPGPUContext();
    gl = getActiveContext();
  });

  afterEach(() => {
    gpgpu.dispose();
    gl = null;
  });

  it('disables DEPTH_TEST and STENCIL_TEST', () => {
    expect(gl.getParameter(gl.DEPTH_TEST)).toEqual(false);
    expect(gl.getParameter(gl.STENCIL_TEST)).toEqual(false);
  });

  it('disables BLEND', () => {
    expect(gl.getParameter(gl.BLEND)).toEqual(false);
  });

  it('disables DITHER, POLYGON_OFFSET_FILL', () => {
    expect(gl.getParameter(gl.DITHER)).toEqual(false);
    expect(gl.getParameter(gl.POLYGON_OFFSET_FILL)).toEqual(false);
  });

  it('enables CULL_FACE with BACK', () => {
    expect(gl.getParameter(gl.CULL_FACE)).toEqual(true);
    expect(gl.getParameter(gl.CULL_FACE_MODE)).toEqual(gl.BACK);
  });

  it('enables SCISSOR_TEST', () => {
    expect(gl.getParameter(gl.SCISSOR_TEST)).toEqual(true);
  });
});

describeWithFlags('gpgpu_util createFloat32MatrixTexture', WEBGL_ENVS, () => {
  let gl: WebGLRenderingContext;
  beforeEach(() => {
    gl = getActiveContext();
  });

  afterEach(() => {
    gl = null;
  });

  it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = tex_util.getTextureConfig(gl);
    const debug = false;
    const tex =
        gpgpu_util.createFloat32MatrixTexture(gl, debug, 32, 32, textureConfig);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    expect(gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S))
        .toEqual(gl.CLAMP_TO_EDGE);
    expect(gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T))
        .toEqual(gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });

  it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = tex_util.getTextureConfig(gl);
    const debug = false;
    const tex = gpgpu_util.createFloat32MatrixTexture(
        gl, debug, 32, 32, textureConfig);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    expect(gl.getTexParameter(
               gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER))
        .toEqual(gl.NEAREST);
    expect(gl.getTexParameter(
               gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER))
        .toEqual(gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });
});

describeWithFlags('gpgpu_util createPackedMatrixTexture', WEBGL_ENVS, () => {
  let gl: WebGLRenderingContext;
  beforeEach(() => {
    gl = getActiveContext();
  });

  afterEach(() => {
    gl = null;
  });

  it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = tex_util.getTextureConfig(gl);
    const debug = false;
    const tex = gpgpu_util.createPackedMatrixTexture(
        gl, debug, 32, 32, textureConfig);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    expect(
        gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S))
        .toEqual(gl.CLAMP_TO_EDGE);
    expect(
        gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T))
        .toEqual(gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });

  it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = tex_util.getTextureConfig(gl);
    const debug = false;
    const tex = gpgpu_util.createPackedMatrixTexture(
        gl, debug, 32, 32, textureConfig);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    expect(gl.getTexParameter(
               gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER))
        .toEqual(gl.NEAREST);
    expect(gl.getTexParameter(
               gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER))
        .toEqual(gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });
});
