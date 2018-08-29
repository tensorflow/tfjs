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

import {css} from 'glamor';
import {tachyons as tac} from 'glamor-tachyons';

import {getDrawArea} from '../render/render_utils';
import {Drawable} from '../types';

const DEFAULT_SUBSURFACE_OPTS = {
  prepend: false,
};

/**
 * Utility function to create/retrieve divs within an HTMLElement|Surface
 */
export function subSurface(parent: Drawable, name: string, opts: Options = {}) {
  const container = getDrawArea(parent);
  const style = css({...tac('mv2')});
  const finalOpts = Object.assign({}, DEFAULT_SUBSURFACE_OPTS, opts);

  let sub: HTMLElement|null = container.querySelector(`div[data-name=${name}]`);
  if (!sub) {
    sub = document.createElement('div');
    sub.setAttribute('class', `${style}`);
    sub.dataset.name = name;

    if (finalOpts.prepend) {
      container.insertBefore(sub, container.firstChild);
    } else {
      container.appendChild(sub);
    }
  }
  return sub;
}

interface Options {
  prepend?: boolean;
}
