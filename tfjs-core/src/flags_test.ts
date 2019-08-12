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

import * as device_util from './device_util';
import {ENV} from './environment';

describe('DEBUG', () => {
  beforeEach(() => {
    ENV.reset();
    spyOn(console, 'warn').and.callFake((msg: string) => {});
  });
  afterAll(() => ENV.reset());

  it('disabled by default', () => {
    expect(ENV.getBool('DEBUG')).toBe(false);
  });

  it('warns when enabled', () => {
    const consoleWarnSpy = console.warn as jasmine.Spy;
    ENV.set('DEBUG', true);
    expect(consoleWarnSpy.calls.count()).toBe(1);
    expect((consoleWarnSpy.calls.first().args[0] as string)
               .startsWith('Debugging mode is ON. '))
        .toBe(true);

    expect(ENV.getBool('DEBUG')).toBe(true);
    expect(consoleWarnSpy.calls.count()).toBe(1);
  });
});

describe('IS_BROWSER', () => {
  let isBrowser: boolean;
  beforeEach(() => {
    ENV.reset();
    spyOn(device_util, 'isBrowser').and.callFake(() => isBrowser);
  });
  afterAll(() => ENV.reset());

  it('isBrowser: true', () => {
    isBrowser = true;
    expect(ENV.getBool('IS_BROWSER')).toBe(true);
  });

  it('isBrowser: false', () => {
    isBrowser = false;
    expect(ENV.getBool('IS_BROWSER')).toBe(false);
  });
});

describe('PROD', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('disabled by default', () => {
    expect(ENV.getBool('PROD')).toBe(false);
  });
});

describe('TENSORLIKE_CHECK_SHAPE_CONSISTENCY', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('disabled when debug is disabled', () => {
    ENV.set('DEBUG', false);
    expect(ENV.getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')).toBe(false);
  });

  it('enabled when debug is enabled', () => {
    ENV.set('DEBUG', true);
    expect(ENV.getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')).toBe(true);
  });
});

describe('DEPRECATION_WARNINGS_ENABLED', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('enabled by default', () => {
    expect(ENV.getBool('DEPRECATION_WARNINGS_ENABLED')).toBe(true);
  });
});

describe('IS_TEST', () => {
  beforeEach(() => ENV.reset());
  afterAll(() => ENV.reset());

  it('disabled by default', () => {
    expect(ENV.getBool('IS_TEST')).toBe(false);
  });
});
