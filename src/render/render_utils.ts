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
