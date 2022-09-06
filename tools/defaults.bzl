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

"""Re-export of some bazel rules with repository-wide defaults."""

load("@npm//@bazel/concatjs:index.bzl", _ts_library = "ts_library")
load("@npm//@bazel/esbuild:index.bzl", _esbuild = "esbuild")

def ts_library(**kwargs):
    # Use the ts_library tsconfig by default. The ts_library tsconfig does not
    # have "incremental = true" because ts_library does not support incremental
    # builds at the level of the typescript compiler (it supports incrementally
    # rebuilding at the level of Bazel build targets).
    tsconfig = kwargs.pop("tsconfig", "@//:tsconfig_ts_library")

    # Use es2017 for esModule (prodmode) outputs by default. This is transpiled
    # down to es5 in tfjs_bundle.bzl. Use es5 for commonjs (devmode) outputs by
    # default since they are directly used by nodejs.
    # The devmode_target corresponds to es5_sources and the prodmode_target
    # corresponds to es6_sources.
    # https://github.com/bazelbuild/rules_nodejs/issues/2094
    devmode_target = kwargs.pop("devmode_target", "es5")
    prodmode_target = kwargs.pop("prodmode_target", "es2017")

    # By default, set package_name based on module_name to mimic pre-4.0 behavior
    # of automatically linking the ts_library.
    maybe_module_name = kwargs["module_name"] if "module_name" in kwargs else None
    package_name = kwargs.pop("package_name", maybe_module_name)

    _ts_library(
        tsconfig = tsconfig,
        devmode_target = devmode_target,
        prodmode_target = prodmode_target,
        package_name = package_name,
        **kwargs
    )

def esbuild(**kwargs):
    # Make sure esbuild always resolve the module (.mjs) files before .js files.
    args = kwargs.pop("args", {})
    args["resolveExtensions"] = [".mjs", ".js"]

    _esbuild(
        args = args,
        **kwargs
    )
