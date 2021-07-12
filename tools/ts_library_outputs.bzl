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
