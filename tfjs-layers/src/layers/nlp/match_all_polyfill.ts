/**
 * @license
 * Copyright 2023 Google LLC.
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

// TODO(mattSoulanille): Replace this with automatic polyfilling using core-js.
export function *matchAll(str: string, regexp: RegExp): IterableIterator<RegExpMatchArray> {
  // Remove the global flag since str.match does not work with it.
  const flags = regexp.flags.replace(/g/g, '');
  regexp = new RegExp(regexp, flags);

  let match = str.match(regexp);
  let offset = 0;
  let restOfStr = str;
  while (match != null) {
    if (match.index == null) {
      console.error(match);
      throw new Error(`Matched string '${match[0]}' has no index`);
    }

    // Remove up to and including the first match from the input string
    // so the next loop can find the next match.
    const matchEnd = match.index + match[0].length;
    restOfStr = restOfStr.slice(matchEnd);

    // Adjust the match to look like a result from matchAll.
    match.index += offset;
    match.input = str;

    offset += matchEnd;
    yield match;
    match = restOfStr.match(regexp);
  }
}
