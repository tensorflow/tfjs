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

import {Drawable, isSurface, isSurfaceInfo} from '../types';
import {visor} from '../visor';

export function getDrawArea(drawable: Drawable): HTMLElement {
  if (drawable instanceof HTMLElement) {
    return drawable;
  } else if (isSurface(drawable)) {
    return drawable.drawArea;
  } else if (isSurfaceInfo(drawable)) {
    const surface = visor().surface(
        {name: drawable.name, tab: drawable.tab, styles: drawable.styles});
    return surface.drawArea;
  } else {
    throw new Error('Not a drawable');
  }
}

export function shallowEquals(
    // tslint:disable-next-line:no-any
    a: {[key: string]: any}, b: {[key: string]: any}) {
  const aProps = Object.getOwnPropertyNames(a);
  const bProps = Object.getOwnPropertyNames(b);

  if (aProps.length !== bProps.length) {
    return false;
  }

  for (let i = 0; i < aProps.length; i++) {
    const prop = aProps[i];
    if (a[prop] !== b[prop]) {
      return false;
    }
  }

  return true;
}

export async function nextFrame() {
  await new Promise(r => requestAnimationFrame(r));
}
