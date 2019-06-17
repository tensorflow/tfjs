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

import {ENV} from '../../environment';
import {BROWSER_ENVS, describeWithFlags} from '../../jasmine_util';

import {getWebGLContext} from './canvas_util';

describeWithFlags('canvas_util', BROWSER_ENVS, () => {
  it('Returns a valid canvas', () => {
    const canvas = getWebGLContext(ENV.getNumber('WEBGL_VERSION')).canvas as (
                       HTMLCanvasElement | OffscreenCanvas);
    expect(
        (canvas instanceof HTMLCanvasElement) ||
        (canvas instanceof OffscreenCanvas))
        .toBe(true);
  });

  it('Returns a valid gl context', () => {
    const gl = getWebGLContext(ENV.getNumber('WEBGL_VERSION'));
    expect(gl.isContextLost()).toBe(false);
  });
});

describeWithFlags('canvas_util webgl2', {flags: {WEBGL_VERSION: 2}}, () => {
  it('is ok when the user requests webgl 1 canvas', () => {
    const canvas = getWebGLContext(1).canvas;
    expect((canvas instanceof HTMLCanvasElement)).toBe(true);
  });
});
