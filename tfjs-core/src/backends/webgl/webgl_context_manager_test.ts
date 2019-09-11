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

import {BROWSER_ENVS, describeWithFlags} from '../../jasmine_util';

import {getActiveContext, getContextByVersion, setContextCleanup, setContextFactory} from './webgl_context_manager';

describeWithFlags('webgl_context_manager', BROWSER_ENVS, () => {
  it('returns the active context for browser WebGL', () => {
    const canvas = getActiveContext();
    expect(
        (canvas instanceof WebGL2RenderingContext) ||
        (canvas instanceof WebGLRenderingContext))
        .toBe(true);
  });
});

describeWithFlags(
    'webgl_context_manager webgl2', {flags: {WEBGL_VERSION: 2}}, () => {
      it('returns webgl1 canvas under webgl2', () => {
        const canvas = getContextByVersion(1);
        expect(canvas instanceof WebGLRenderingContext).toBe(true);
      });
    });

describe('webgl_context_manager create/cleanup', () => {
  afterAll(() => {
    // Reset global context creation and cleanup:
    setContextCleanup(null);
    setContextFactory(null);
  });

  it('should call factory method to create WebGLRenderingContext', () => {
    let created = false;
    let cleanedup = false;
    let contextLost = false;
    const contextFake = {
      disable: (cap: number) => {},
      enable: (cap: number) => {},
      cullFace: (cap: number) => {},
      isContextLost: () => {
        return contextLost;
      }
    } as WebGLRenderingContext;

    setContextFactory((version: number) => {
      created = true;
      return contextFake;
    });

    // Request context version '10' to bypass any cached system WebGL versions:
    const context = getContextByVersion(10);
    expect(created).toBe(true);
    expect(context).toBe(contextFake);

    // Mark fake context as disposed so it will be cleanedup on next context
    // creation request:
    setContextCleanup((context: WebGLRenderingContext) => {
      expect(context).toBe(contextFake);
      cleanedup = true;

      // Set context lost back to false to prevent an endless loop:
      contextLost = false;
    });

    contextLost = true;
    getContextByVersion(10);

    expect(cleanedup).toBe(true);
  });
});
