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

describeWithFlags('gpgpu_util createWebGLContext', WEBGL_ENVS, () => {
  let gpgpu: GPGPUContext;

  beforeEach(() => {
    gpgpu = new GPGPUContext();
  });

  afterEach(() => {
    gpgpu.dispose();
  });

  it('disables DEPTH_TEST and STENCIL_TEST', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.DEPTH_TEST)).toEqual(false);
    expect(gpgpu.gl.getParameter(gpgpu.gl.STENCIL_TEST)).toEqual(false);
  });

  it('disables BLEND', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.BLEND)).toEqual(false);
  });

  it('disables DITHER, POLYGON_OFFSET_FILL', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.DITHER)).toEqual(false);
    expect(gpgpu.gl.getParameter(gpgpu.gl.POLYGON_OFFSET_FILL)).toEqual(false);
  });

  it('enables CULL_FACE with BACK', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.CULL_FACE)).toEqual(true);
    expect(gpgpu.gl.getParameter(gpgpu.gl.CULL_FACE_MODE))
        .toEqual(gpgpu.gl.BACK);
  });

  it('enables SCISSOR_TEST', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.SCISSOR_TEST)).toEqual(true);
  });
});

describeWithFlags('gpgpu_util createFloat32MatrixTexture', WEBGL_ENVS, () => {
  it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = tex_util.getTextureConfig(gpgpu.gl);
    const debug = false;
    const tex = gpgpu_util.createFloat32MatrixTexture(
        gpgpu.gl, debug, 32, 32, textureConfig);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
    expect(
        gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_S))
        .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
    expect(
        gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_T))
        .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });

  it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = tex_util.getTextureConfig(gpgpu.gl);
    const debug = false;
    const tex = gpgpu_util.createFloat32MatrixTexture(
        gpgpu.gl, debug, 32, 32, textureConfig);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
    expect(gpgpu.gl.getTexParameter(
               gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MIN_FILTER))
        .toEqual(gpgpu.gl.NEAREST);
    expect(gpgpu.gl.getTexParameter(
               gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MAG_FILTER))
        .toEqual(gpgpu.gl.NEAREST);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });
});

describeWithFlags('gpgpu_util createPackedMatrixTexture', WEBGL_ENVS, () => {
  it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = tex_util.getTextureConfig(gpgpu.gl);
    const debug = false;
    const tex = gpgpu_util.createPackedMatrixTexture(
        gpgpu.gl, debug, 32, 32, textureConfig);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
    expect(
        gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_S))
        .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
    expect(
        gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_T))
        .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });

  it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = tex_util.getTextureConfig(gpgpu.gl);
    const debug = false;
    const tex = gpgpu_util.createPackedMatrixTexture(
        gpgpu.gl, debug, 32, 32, textureConfig);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
    expect(gpgpu.gl.getTexParameter(
               gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MIN_FILTER))
        .toEqual(gpgpu.gl.NEAREST);
    expect(gpgpu.gl.getTexParameter(
               gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MAG_FILTER))
        .toEqual(gpgpu.gl.NEAREST);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });
});
