/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {TAGS} from './constants';

// tslint:disable-next-line:no-any
declare let __karma__: any;
if (typeof __karma__ !== 'undefined') {
  const args = __karma__.config.args || [];

  let tags;

  args.forEach((arg: string, i: number) => {
    if (arg === '--tags') {
      tags = parseTags(args[i + 1]);
    }
  });

  setupTestFilters(tags);
}

/**
 * Given a string separated with comma, validate and return tags as an array.
 */
function parseTags(tagsInput: string): string[] {
  if (!tagsInput || tagsInput === '') {
    throw new Error(
        '--tags did not have any value. Please specify tags separated ' +
        'by comma.');
  }

  const tags = tagsInput.split(',');

  const $tags = [];

  for (let i = 0; i < tags.length; i++) {
    const tag = tags[i].trim();

    if (!TAGS.includes(tag)) {
      throw new Error(`Tag ${tag} is not supported. Supported tags: ${TAGS}`);
    }
    $tags.push(tag);
  }

  return $tags;
}

/**
 * Run Jasmine tests only for allowlisted tags.
 */
function setupTestFilters(tags: string[] = []) {
  const env = jasmine.getEnv();

  // Account for --grep flag passed to karma by saving the existing specFilter.
  const grepFilter = env.specFilter;

  // tslint:disable-next-line: no-any
  env.specFilter = (spec: any) => {
    // Filter out tests if the --grep flag is passed.
    if (!grepFilter(spec)) {
      return false;
    }

    const name = spec.getFullName();

    // Only include a test if it belongs to one of the specified tags.
    for (let i = 0; i < tags.length; i++) {
      const tag = tags[i];
      if (name.includes(tag)) {
        return true;
      }
    }

    // Otherwise ignore the test.
    return false;
  };
}
