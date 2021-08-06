# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

load("//tools:copy_to_dist.bzl", "copy_to_dist")

"""Extracts the file outputs of ts_library"""

def ts_library_outputs(name, srcs):
    # ES Module es2017 compilation results. This is configured in
    # tools/defaults.bzl. Outputs '.mjs' files.
    es6 = name + "_es6_sources"
    native.filegroup(
        name = es6,
        srcs = srcs,
        output_group = "es6_sources",
    )

    # Commonjs es2017 compilation results. Outputs '.js' files
    es5 = name + "_es5_sources"
    native.filegroup(
        name = es5,
        srcs = srcs,
        output_group = "es5_sources",
    )

    # Declaration (.d.ts) files
    declaration = name + "_declaration"
    native.filegroup(
        name = declaration,
        srcs = srcs,
    )

    # Combination of the above
    native.filegroup(
        name = name,
        srcs = [es6, es5, declaration],
    )

def copy_ts_library_to_output(name, srcs, root="src"):
    # Declaration (.d.ts) files
    declaration = name + "_declaration"
    native.filegroup(
        name = declaration,
        srcs = srcs,
    )

    # ES Module es2017 compilation results. This is configured in
    # tools/defaults.bzl. Outputs '.mjs' files.
    esm = name + "_esm_sources"
    native.filegroup(
        name = esm,
        srcs = srcs,
        output_group = "es6_sources",
    )

    copy_esm = name + "_copy_esm"
    copy_to_dist(
        name = copy_esm,
        srcs = [esm],
        root = root,
        dest_dir = "esm",
        extension = "js", # Rewrite '.mjs' extension to '.js'
    )

    copy_esm_declaration = name + "_copy_esm_declaration"
    copy_to_dist(
        name = copy_esm_declaration,
        srcs = [declaration],
        root = root,
        dest_dir = "esm",
    )

    # Commonjs es2017 compilation results. Outputs '.js' files
    cjs = name + "_cjs_sources"
    native.filegroup(
        name = cjs,
        srcs = srcs,
        output_group = "es5_sources",
    )

    copy_cjs = name + "_copy_cjs"
    copy_to_dist(
        name = copy_cjs,
        srcs = [cjs, declaration],
        root = root,
    )

    native.filegroup(
        name = name,
        srcs = [copy_esm, copy_esm_declaration, copy_cjs],
    )
