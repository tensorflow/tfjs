/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

// This index.ts file is used by Bazel as an entrypoint to this package instead
// of the index.ts in '../src'. The reason for this is that we want this package
// to be importable by the same name as it has in npm. Normally, this is handled
// by the 'main' entry in the package.json file, but ts_library has no concept
// of package.json files. Instead, we set 'module_name' (and 'package_name') to
// this package's name on npm to achieve the same effect.

// The reason we need a separate index.ts file to do this is because of how
// rules_nodejs's linker works. If we didn't use a separate index.ts and
// instead set the 'module_name' of '../src/index.ts', then the linker's output
// (viewable in the root node_modules directory after yarn build) would look
// like this:
//
// @tensorflow/this-package-name/
//   index.js
//   foo.js
//   bar.js
//   dist -> . (symbolic link to current directory)
//
// This causes bundling issues for packages that import the main 'index.ts' file
// and additional files from 'dist/' because rollup has been instructed to treat
// symlinks as real files (a requirement to make it run in Bazel), so you can
// easily end up with multiple copies of 'foo.js' if it is imported manually
// from '@tensorflow/this-package-name/dist/foo' and if the 'index.ts' file also
// imports it.
//
// There isn't really anything the linker can do to avoid this structure, since
// since 'index.js' must be importable as @tensorflow/this-package-name and the
// linker can't rewrite the import statements in index.js.
//
// So why the extra directory? Why not put the index.ts file in '../'? Again,
// this is related to how the linker works. The linker symlinks the bazel output
// directory containing the ts_library's output to a directory in node_modules.
// This output directory contains the outputs of other rules as well, including
// the rules that copy files to the 'dist' directory for publishing. When one
// of these other rules that writes to 'dist' is run, it overwrites the 'dist'
// symlink created by the linker.
//
// Putting the entrypoint in its own directory allows the linker to link
// ts_library targets with module_name (and package_name) set to
// '@tensorflow/this-package-name/dist' without having the 'dist' symlink
// overwritten. With this approach, the output struture looks like this:
//
// @tensorflow/this-package-name/
//   index.js (from this directory's index.ts)
//   dist/
//     index.js (from ../src/index.ts)
//     foo.js
//     bar.js
//

export * from '@tensorflow/tfjs-backend-cpu/dist/index';
