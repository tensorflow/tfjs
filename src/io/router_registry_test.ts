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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {BROWSER_ENVS} from '../test_util';
import {BrowserIndexedDB, browserIndexedDB} from './indexed_db';
import {BrowserLocalStorage, browserLocalStorage} from './local_storage';
import {IORouterRegistry} from './router_registry';

describeWithFlags('IORouterRegistry', BROWSER_ENVS, () => {
  const localStorageRouter = (url: string) => {
    const scheme = 'localstorage://';
    if (url.startsWith(scheme)) {
      return browserLocalStorage(url.slice(scheme.length));
    } else {
      return null;
    }
  };

  const indexedDBRouter = (url: string) => {
    const scheme = 'indexeddb://';
    if (url.startsWith(scheme)) {
      return browserIndexedDB(url.slice(scheme.length));
    } else {
      return null;
    }
  };

  let tempRegistryInstance: IORouterRegistry = null;
  beforeEach(() => {
    // Force reset registry for testing.
    // tslint:disable:no-any
    tempRegistryInstance = (IORouterRegistry as any).instance;
    (IORouterRegistry as any).instance = null;
    // tslint:enable:no-any
  });

  afterEach(() => {
    // tslint:disable-next-line:no-any
    (IORouterRegistry as any).instance = tempRegistryInstance;
  });

  it('getSaveHandler succeeds', () => {
    IORouterRegistry.registerSaveRouter(localStorageRouter);
    IORouterRegistry.registerSaveRouter(indexedDBRouter);

    const out1 = tf.io.getSaveHandlers('localstorage://foo-model');
    expect(out1.length).toEqual(1);
    expect(out1[0] instanceof BrowserLocalStorage).toEqual(true);
    const out2 = tf.io.getSaveHandlers('indexeddb://foo-model');
    expect(out2.length).toEqual(1);
    expect(out2[0] instanceof BrowserIndexedDB).toEqual(true);
  });

  it('getLoadHandler succeeds', () => {
    IORouterRegistry.registerLoadRouter(localStorageRouter);
    IORouterRegistry.registerLoadRouter(indexedDBRouter);

    const out1 = tf.io.getLoadHandlers('localstorage://foo-model');
    expect(out1.length).toEqual(1);
    expect(out1[0] instanceof BrowserLocalStorage).toEqual(true);
    const out2 = tf.io.getLoadHandlers('indexeddb://foo-model');
    expect(out2.length).toEqual(1);
    expect(out2[0] instanceof BrowserIndexedDB).toEqual(true);
  });

  it('getSaveHandler fails', () => {
    IORouterRegistry.registerSaveRouter(localStorageRouter);

    expect(tf.io.getSaveHandlers('invalidscheme://foo-model')).toEqual([]);
    // Check there is no crosstalk between save and load handlers.
    expect(tf.io.getLoadHandlers('localstorage://foo-model')).toEqual([]);
  });
});
