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

import {terser} from 'rollup-plugin-terser';

/**
 * Returns a standardized list of browser package configuration options
 * that we want to use in all our rollup files and ship to NPM.
 *
 * @param {string} fileName
 * @param {string} preamble
 * @param {boolean} visualize - produce bundle visualizations for certain
 *     bundles
 * @param {boolean} ci is this a CI build
 * @param {object} terserExtraOptions is any extra options passed to terser
 */
export function getBrowserBundleConfigOptions(
    config, name, fileName, preamble, visualize, ci, terserExtraOptions = {}) {
  const bundles = [];

  const terserPlugin =
      terser({output: {preamble, comments: false}, ...terserExtraOptions});
  const extend = true;
  const umdFormat = 'umd';
  const fesmFormat = 'es';

  // UMD ES5 minified
  bundles.push(config({
    plugins: [terserPlugin],
    output: {
      format: umdFormat,
      name,
      extend,
      file: `dist/${fileName}.min.js`,
      freeze: false
    },
    tsCompilerOptions: {target: 'es5'},
    visualize
  }));

  if (ci) {
    // In CI we do not build all the possible bundles.
    return bundles;
  }

  // UMD ES5 unminified
  bundles.push(config({
    output: {
      format: umdFormat,
      name,
      extend,
      file: `dist/${fileName}.js`,
      freeze: false
    },
    tsCompilerOptions: {target: 'es5'}
  }));

  // UMD ES2017
  bundles.push(config({
    output:
        {format: umdFormat, name, extend, file: `dist/${fileName}.es2017.js`},
    tsCompilerOptions: {target: 'es2017'}
  }));

  // UMD ES2017 minified
  bundles.push(config({
    plugins: [terserPlugin],
    output: {
      format: umdFormat,
      name,
      extend,
      file: `dist/${fileName}.es2017.min.js`
    },
    tsCompilerOptions: {target: 'es2017'},
    visualize
  }));

  // FESM ES2017
  bundles.push(config({
    output:
        {format: fesmFormat, name, extend, file: `dist/${fileName}.fesm.js`},
    tsCompilerOptions: {target: 'es2017'}
  }));

  // FESM ES2017 minified
  bundles.push(config({
    plugins: [terserPlugin],
    output: {
      format: fesmFormat,
      name,
      extend,
      file: `dist/${fileName}.fesm.min.js`
    },
    tsCompilerOptions: {target: 'es2017'},
    visualize
  }));


  return bundles;
}
