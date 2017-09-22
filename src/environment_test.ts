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
import * as device_util from './device_util';
import {Environment} from './environment';

describe('disjoint query timer', () => {
  it('mobile', () => {
    spyOn(device_util, 'isMobile').and.returnValue(true);

    const env = new Environment();

    expect(env.get('WEBGL_DISJOINT_QUERY_TIMER')).toBe(false);
  });

  it('not mobile', () => {
    spyOn(device_util, 'isMobile').and.returnValue(false);

    const env = new Environment();

    expect(env.get('WEBGL_DISJOINT_QUERY_TIMER')).toBe(true);
  });
});

describe('WebGL version', () => {
  it('webgl 1', () => {
    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string) => {
        if (context === 'webgl') {
          return {
            getExtension: (a: string) => {
              return {loseContext: () => {}};
            }
          };
        }
        return null;
      }
    });

    const env = new Environment();
    expect(env.get('WEBGL_VERSION')).toBe(1);
  });

  it('webgl 2', () => {
    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string) => {
        if (context === 'webgl2') {
          return {
            getExtension: (a: string) => {
              return {loseContext: () => {}};
            }
          };
        }
        return null;
      }
    });

    const env = new Environment();
    expect(env.get('WEBGL_VERSION')).toBe(2);
  });

  it('no webgl', () => {
    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string): WebGLRenderingContext => null
    });

    const env = new Environment();
    expect(env.get('WEBGL_VERSION')).toBe(0);
  });
});
