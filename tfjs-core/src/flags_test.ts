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

import * as device_util from './device_util';
import * as tf from './index';

describe('DEBUG', () => {
  beforeEach(() => {
    tf.env().reset();
    spyOn(console, 'warn').and.callFake((msg: string) => {});
  });
  afterAll(() => tf.env().reset());

  it('disabled by default', () => {
    expect(tf.env().getBool('DEBUG')).toBe(false);
  });

  it('warns when enabled', () => {
    const consoleWarnSpy = console.warn as jasmine.Spy;
    tf.env().set('DEBUG', true);
    expect(consoleWarnSpy.calls.count()).toBe(1);
    expect((consoleWarnSpy.calls.first().args[0] as string)
               .startsWith('Debugging mode is ON. '))
        .toBe(true);

    expect(tf.env().getBool('DEBUG')).toBe(true);
    expect(consoleWarnSpy.calls.count()).toBe(1);
  });
});

// TODO (yassogba) figure out why this spy is not working / fix this test.
describe('IS_BROWSER', () => {
  let isBrowser: boolean;
  beforeEach(() => {
    tf.env().reset();
    spyOn(device_util, 'isBrowser').and.callFake(() => isBrowser);
  });
  afterAll(() => tf.env().reset());

  // tslint:disable-next-line: ban
  xit('isBrowser: true', () => {
    isBrowser = true;
    expect(tf.env().getBool('IS_BROWSER')).toBe(true);
  });

  // tslint:disable-next-line: ban
  xit('isBrowser: false', () => {
    isBrowser = false;
    expect(tf.env().getBool('IS_BROWSER')).toBe(false);
  });
});

describe('PROD', () => {
  beforeEach(() => tf.env().reset());
  afterAll(() => tf.env().reset());

  it('disabled by default', () => {
    expect(tf.env().getBool('PROD')).toBe(false);
  });
});

describe('TENSORLIKE_CHECK_SHAPE_CONSISTENCY', () => {
  beforeEach(() => tf.env().reset());
  afterAll(() => tf.env().reset());

  it('disabled when debug is disabled', () => {
    tf.env().set('DEBUG', false);
    expect(tf.env().getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')).toBe(false);
  });

  it('enabled when debug is enabled', () => {
    // Silence debug warnings.
    spyOn(console, 'warn');
    tf.enableDebugMode();
    expect(tf.env().getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')).toBe(true);
  });
});

describe('DEPRECATION_WARNINGS_ENABLED', () => {
  beforeEach(() => tf.env().reset());
  afterAll(() => tf.env().reset());

  it('enabled by default', () => {
    expect(tf.env().getBool('DEPRECATION_WARNINGS_ENABLED')).toBe(true);
  });
});

describe('IS_TEST', () => {
  beforeEach(() => tf.env().reset());
  afterAll(() => tf.env().reset());

  it('disabled by default', () => {
    expect(tf.env().getBool('IS_TEST')).toBe(false);
  });
});

describe('async flags test', () => {
  const asyncFlagName = 'ASYNC_FLAG';
  beforeEach(() => tf.env().registerFlag(asyncFlagName, async () => true));

  afterEach(() => tf.env().reset());

  it('evaluating async flag works', async () => {
    const flagVal = await tf.env().getAsync(asyncFlagName);
    expect(flagVal).toBe(true);
  });

  it('evaluating async flag synchronously fails', async () => {
    expect(() => tf.env().get(asyncFlagName)).toThrow();
  });
});
