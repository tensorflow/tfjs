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
import * as tf from './index';

describe('DEBUG', () => {
  beforeEach(() => {
    tf.environment().reset();
    spyOn(console, 'warn').and.callFake((msg: string) => {});
  });
  afterAll(() => tf.environment().reset());

  it('disabled by default', () => {
    expect(tf.environment().getBool('DEBUG')).toBe(false);
  });

  it('warns when enabled', () => {
    const consoleWarnSpy = console.warn as jasmine.Spy;
    tf.environment().set('DEBUG', true);
    expect(consoleWarnSpy.calls.count()).toBe(1);
    expect((consoleWarnSpy.calls.first().args[0] as string)
               .startsWith('Debugging mode is ON. '))
        .toBe(true);

    expect(tf.environment().getBool('DEBUG')).toBe(true);
    expect(consoleWarnSpy.calls.count()).toBe(1);
  });
});

describe('IS_BROWSER', () => {
  let isBrowser: boolean;
  beforeEach(() => {
    tf.environment().reset();
    spyOn(device_util, 'isBrowser').and.callFake(() => isBrowser);
  });
  afterAll(() => tf.environment().reset());

  it('isBrowser: true', () => {
    isBrowser = true;
    expect(tf.environment().getBool('IS_BROWSER')).toBe(true);
  });

  it('isBrowser: false', () => {
    isBrowser = false;
    expect(tf.environment().getBool('IS_BROWSER')).toBe(false);
  });
});

describe('PROD', () => {
  beforeEach(() => tf.environment().reset());
  afterAll(() => tf.environment().reset());

  it('disabled by default', () => {
    expect(tf.environment().getBool('PROD')).toBe(false);
  });
});

describe('TENSORLIKE_CHECK_SHAPE_CONSISTENCY', () => {
  beforeEach(() => tf.environment().reset());
  afterAll(() => tf.environment().reset());

  it('disabled when debug is disabled', () => {
    tf.environment().set('DEBUG', false);
    expect(tf.environment().getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY'))
        .toBe(false);
  });

  it('enabled when debug is enabled', () => {
    tf.environment().set('DEBUG', true);
    expect(tf.environment().getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY'))
        .toBe(true);
  });
});

describe('DEPRECATION_WARNINGS_ENABLED', () => {
  beforeEach(() => tf.environment().reset());
  afterAll(() => tf.environment().reset());

  it('enabled by default', () => {
    expect(tf.environment().getBool('DEPRECATION_WARNINGS_ENABLED')).toBe(true);
  });
});

describe('IS_TEST', () => {
  beforeEach(() => tf.environment().reset());
  afterAll(() => tf.environment().reset());

  it('disabled by default', () => {
    expect(tf.environment().getBool('IS_TEST')).toBe(false);
  });
});
