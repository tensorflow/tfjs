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
