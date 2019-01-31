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

import {IOHandler} from './types';

export type IORouter = (url: string|string[], onProgress?: Function) =>
    IOHandler;

export class IORouterRegistry {
  // Singleton instance.
  private static instance: IORouterRegistry;

  private saveRouters: IORouter[];
  private loadRouters: IORouter[];

  private constructor() {
    this.saveRouters = [];
    this.loadRouters = [];
  }

  private static getInstance(): IORouterRegistry {
    if (IORouterRegistry.instance == null) {
      IORouterRegistry.instance = new IORouterRegistry();
    }
    return IORouterRegistry.instance;
  }

  /**
   * Register a save-handler router.
   *
   * @param saveRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `save` method defined or `null`.
   */
  static registerSaveRouter(saveRouter: IORouter) {
    IORouterRegistry.getInstance().saveRouters.push(saveRouter);
  }

  /**
   * Register a load-handler router.
   *
   * @param loadRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `load` method defined or `null`.
   */
  static registerLoadRouter(loadRouter: IORouter) {
    IORouterRegistry.getInstance().loadRouters.push(loadRouter);
  }

  /**
   * Look up IOHandler for saving, given a URL-like string.
   *
   * @param url
   * @returns If only one match is found, an instance of IOHandler with the
   * `save` method defined. If no match is found, `null`.
   * @throws Error, if more than one match is found.
   */
  static getSaveHandlers(url: string|string[]): IOHandler[] {
    return IORouterRegistry.getHandlers(url, 'save');
  }

  /**
   * Look up IOHandler for loading, given a URL-like string.
   *
   * @param url
   * @param onProgress Optional, progress callback function, fired periodically
   *   before the load is completed.
   * @returns All valid handlers for `url`, given the currently registered
   *   handler routers.
   */
  static getLoadHandlers(url: string|string[], onProgress?: Function):
      IOHandler[] {
    return IORouterRegistry.getHandlers(url, 'load', onProgress);
  }

  private static getHandlers(
      url: string|string[], handlerType: 'save'|'load',
      onProgress?: Function): IOHandler[] {
    const validHandlers: IOHandler[] = [];
    const routers = handlerType === 'load' ?
        IORouterRegistry.getInstance().loadRouters :
        IORouterRegistry.getInstance().saveRouters;
    routers.forEach(router => {
      const handler = router(url, onProgress);
      if (handler !== null) {
        validHandlers.push(handler);
      }
    });
    return validHandlers;
  }
}

export const registerSaveRouter = (loudRouter: IORouter) =>
    IORouterRegistry.registerSaveRouter(loudRouter);
export const registerLoadRouter = (loudRouter: IORouter) =>
    IORouterRegistry.registerLoadRouter(loudRouter);
export const getSaveHandlers = (url: string|string[]) =>
    IORouterRegistry.getSaveHandlers(url);
export const getLoadHandlers = (url: string|string[], onProgress?: Function) =>
    IORouterRegistry.getLoadHandlers(url);
