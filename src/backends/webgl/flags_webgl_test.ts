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

import * as device_util from '../../device_util';
import {ENV} from '../../environment';
import {webgl_util} from '../../webgl';

import * as canvas_util from './canvas_util';

describe('HAS_WEBGL', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('false when version is 0', () => {
    ENV.set('WEBGL_VERSION', 0);
    expect(ENV.getBool('HAS_WEBGL')).toBe(false);
  });

  it('true when version is 1', () => {
    ENV.set('WEBGL_VERSION', 1);
    expect(ENV.getBool('HAS_WEBGL')).toBe(true);
  });

  it('true when version is 2', () => {
    ENV.set('WEBGL_VERSION', 2);
    expect(ENV.getBool('HAS_WEBGL')).toBe(true);
  });
});

describe('WEBGL_PACK', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when HAS_WEBGL is true', () => {
    ENV.set('HAS_WEBGL', true);
    expect(ENV.getBool('WEBGL_PACK')).toBe(true);
  });

  it('false when HAS_WEBGL is false', () => {
    ENV.set('HAS_WEBGL', false);
    expect(ENV.getBool('WEBGL_PACK')).toBe(false);
  });
});

describe('WEBGL_PACK_NORMALIZATION', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when WEBGL_PACK is true', () => {
    ENV.set('WEBGL_PACK', true);
    expect(ENV.getBool('WEBGL_PACK_NORMALIZATION')).toBe(true);
  });

  it('false when WEBGL_PACK is false', () => {
    ENV.set('WEBGL_PACK', false);
    expect(ENV.getBool('WEBGL_PACK_NORMALIZATION')).toBe(false);
  });
});

describe('WEBGL_PACK_CLIP', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when WEBGL_PACK is true', () => {
    ENV.set('WEBGL_PACK', true);
    expect(ENV.getBool('WEBGL_PACK_CLIP')).toBe(true);
  });

  it('false when WEBGL_PACK is false', () => {
    ENV.set('WEBGL_PACK', false);
    expect(ENV.getBool('WEBGL_PACK_CLIP')).toBe(false);
  });
});

// TODO: https://github.com/tensorflow/tfjs/issues/1679
// describe('WEBGL_PACK_DEPTHWISECONV', () => {
//   beforeEach(() => ENV.reset());
//   afterAll(() => ENV.reset());

//   it('true when WEBGL_PACK is true', () => {
//     ENV.set('WEBGL_PACK', true);
//     expect(ENV.getBool('WEBGL_PACK_DEPTHWISECONV')).toBe(true);
//   });

//   it('false when WEBGL_PACK is false', () => {
//     ENV.set('WEBGL_PACK', false);
//     expect(ENV.getBool('WEBGL_PACK_DEPTHWISECONV')).toBe(false);
//   });
// });

describe('WEBGL_PACK_BINARY_OPERATIONS', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when WEBGL_PACK is true', () => {
    ENV.set('WEBGL_PACK', true);
    expect(ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')).toBe(true);
  });

  it('false when WEBGL_PACK is false', () => {
    ENV.set('WEBGL_PACK', false);
    expect(ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')).toBe(false);
  });
});

describe('WEBGL_PACK_ARRAY_OPERATIONS', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when WEBGL_PACK is true', () => {
    ENV.set('WEBGL_PACK', true);
    expect(ENV.getBool('WEBGL_PACK_ARRAY_OPERATIONS')).toBe(true);
  });

  it('false when WEBGL_PACK is false', () => {
    ENV.set('WEBGL_PACK', false);
    expect(ENV.getBool('WEBGL_PACK_ARRAY_OPERATIONS')).toBe(false);
  });
});

describe('WEBGL_PACK_IMAGE_OPERATIONS', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when WEBGL_PACK is true', () => {
    ENV.set('WEBGL_PACK', true);
    expect(ENV.getBool('WEBGL_PACK_IMAGE_OPERATIONS')).toBe(true);
  });

  it('false when WEBGL_PACK is false', () => {
    ENV.set('WEBGL_PACK', false);
    expect(ENV.getBool('WEBGL_PACK_IMAGE_OPERATIONS')).toBe(false);
  });
});

describe('WEBGL_PACK_REDUCE', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when WEBGL_PACK is true', () => {
    ENV.set('WEBGL_PACK', true);
    expect(ENV.getBool('WEBGL_PACK_REDUCE')).toBe(true);
  });

  it('false when WEBGL_PACK is false', () => {
    ENV.set('WEBGL_PACK', false);
    expect(ENV.getBool('WEBGL_PACK_REDUCE')).toBe(false);
  });
});

describe('WEBGL_LAZILY_UNPACK', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when WEBGL_PACK is true', () => {
    ENV.set('WEBGL_PACK', true);
    expect(ENV.getBool('WEBGL_LAZILY_UNPACK')).toBe(true);
  });

  it('false when WEBGL_PACK is false', () => {
    ENV.set('WEBGL_PACK', false);
    expect(ENV.getBool('WEBGL_LAZILY_UNPACK')).toBe(false);
  });
});

describe('WEBGL_CONV_IM2COL', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('true when WEBGL_PACK is true', () => {
    ENV.set('WEBGL_PACK', true);
    expect(ENV.getBool('WEBGL_CONV_IM2COL')).toBe(true);
  });

  it('false when WEBGL_PACK is false', () => {
    ENV.set('WEBGL_PACK', false);
    expect(ENV.getBool('WEBGL_CONV_IM2COL')).toBe(false);
  });
});

describe('WEBGL_MAX_TEXTURE_SIZE', () => {
  beforeEach(() => {
    ENV.reset();
    webgl_util.resetMaxTextureSize();

    spyOn(canvas_util, 'getWebGLContext').and.returnValue({
      MAX_TEXTURE_SIZE: 101,
      getParameter: (param: number) => {
        if (param === 101) {
          return 50;
        }
        throw new Error(`Got undefined param ${param}.`);
      }
    });
  });
  afterAll(() => {
    ENV.reset();
    webgl_util.resetMaxTextureSize();
  });

  it('is a function of gl.getParameter(MAX_TEXTURE_SIZE)', () => {
    expect(ENV.getNumber('WEBGL_MAX_TEXTURE_SIZE')).toBe(50);
  });
});

describe('WEBGL_MAX_TEXTURES_IN_SHADER', () => {
  let maxTextures: number;
  beforeEach(() => {
    ENV.reset();
    webgl_util.resetMaxTexturesInShader();

    spyOn(canvas_util, 'getWebGLContext').and.callFake(() => {
      return {
        MAX_TEXTURE_IMAGE_UNITS: 101,
        getParameter: (param: number) => {
          if (param === 101) {
            return maxTextures;
          }
          throw new Error(`Got undefined param ${param}.`);
        }
      };
    });
  });
  afterAll(() => {
    ENV.reset();
    webgl_util.resetMaxTexturesInShader();
  });

  it('is a function of gl.getParameter(MAX_TEXTURE_IMAGE_UNITS)', () => {
    maxTextures = 10;
    expect(ENV.getNumber('WEBGL_MAX_TEXTURES_IN_SHADER')).toBe(10);
  });

  it('is capped at 16', () => {
    maxTextures = 20;
    expect(ENV.getNumber('WEBGL_MAX_TEXTURES_IN_SHADER')).toBe(16);
  });
});

describe('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('disjoint query timer disabled', () => {
    ENV.set('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', 0);

    expect(ENV.getBool('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
        .toBe(false);
  });

  it('disjoint query timer enabled, mobile', () => {
    ENV.set('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', 1);
    spyOn(device_util, 'isMobile').and.returnValue(true);

    expect(ENV.getBool('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
        .toBe(false);
  });

  it('disjoint query timer enabled, not mobile', () => {
    ENV.set('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', 1);

    spyOn(device_util, 'isMobile').and.returnValue(false);

    expect(ENV.getBool('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
        .toBe(true);
  });
});

describe('WEBGL_SIZE_UPLOAD_UNIFORM', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('is 0 when there is no float32 bit support', () => {
    ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', false);
    expect(ENV.getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')).toBe(0);
  });

  it('is > 0 when there is float32 bit support', () => {
    ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', true);
    expect(ENV.getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')).toBeGreaterThan(0);
  });
});
