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

workspace(
    name = "tfjs",
    managed_directories = {
        "@npm": ["node_modules"],
    },
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "build_bazel_rules_nodejs",
    sha256 = "dcc55f810142b6cf46a44d0180a5a7fb923c04a5061e2e8d8eb05ccccc60864b",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/5.8.0/rules_nodejs-5.8.0.tar.gz"],
)

# Install rules_nodejs dependencies.
load("@build_bazel_rules_nodejs//:repositories.bzl", "build_bazel_rules_nodejs_dependencies")

build_bazel_rules_nodejs_dependencies()

load("@rules_nodejs//nodejs:repositories.bzl", "nodejs_register_toolchains")

nodejs_register_toolchains(
    name = "nodejs",
    node_version = "18.7.0",
)

# Install the yarn tool
load("@rules_nodejs//nodejs:yarn_repositories.bzl", "yarn_repositories")

yarn_repositories(
    name = "yarn",
    node_repository = "nodejs",
)

# Install yarn packages
load("@build_bazel_rules_nodejs//:index.bzl", "yarn_install")

yarn_install(
    name = "npm",
    exports_directories_only = False,  # Required for ts_library
    package_json = "//:package.json",
    package_path = "/",
    symlink_node_modules = True,
    yarn = "@yarn//:bin/yarn",
    yarn_lock = "//:yarn.lock",
)

# Fetch transitive Bazel dependencies of karma_web_test
http_archive(
    name = "io_bazel_rules_webtesting",
    sha256 = "9bb461d5ef08e850025480bab185fd269242d4e533bca75bfb748001ceb343c3",
    urls = ["https://github.com/bazelbuild/rules_webtesting/releases/download/0.3.3/rules_webtesting.tar.gz"],
)

# Set up web testing, choose browsers we can test on
load("@io_bazel_rules_webtesting//web:repositories.bzl", "web_test_repositories")

web_test_repositories()

load("@io_bazel_rules_webtesting//web/versioned:browsers-0.3.2.bzl", "browser_repositories")

browser_repositories(
    chromium = True,
)

# Esbuild toolchain
load("@build_bazel_rules_nodejs//toolchains/esbuild:esbuild_repositories.bzl", "esbuild_repositories")

esbuild_repositories(npm_repository = "npm")

# Emscripten toolchain
http_archive(
    name = "emsdk",
    # TODO: Remove repo_mapping when emsdk updates to rules_nodejs 5
    repo_mapping = {"@nodejs": "@nodejs_host"},
    strip_prefix = "emsdk-3.1.26/bazel",
    urls = ["https://github.com/emscripten-core/emsdk/archive/refs/tags/3.1.26.tar.gz"],
)

load("@emsdk//:deps.bzl", emsdk_deps = "deps")

emsdk_deps()

load("@emsdk//:emscripten_deps.bzl", emsdk_emscripten_deps = "emscripten_deps")

emsdk_emscripten_deps()

load("@emsdk//:toolchains.bzl", "register_emscripten_toolchains")

register_emscripten_toolchains()

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# xnnpack used for fast vectorized wasm operations
git_repository(
    name = "xnnpack",
    commit = "bad81856dce9fea265866c87ea524afbbdd69012",
    remote = "https://github.com/google/XNNPACK.git",
)

# The libraries below are transitive dependencies of XNNPACK that we need to
# explicitly enumerate here. See https://docs.bazel.build/versions/master/external.html#transitive-dependencies

# FP16 library, used for half-precision conversions
http_archive(
    name = "FP16",
    build_file = "@xnnpack//third_party:FP16.BUILD",
    sha256 = "e66e65515fa09927b348d3d584c68be4215cfe664100d01c9dbc7655a5716d70",
    strip_prefix = "FP16-0a92994d729ff76a58f692d3028ca1b64b145d91",
    urls = [
        "https://github.com/Maratyszcza/FP16/archive/0a92994d729ff76a58f692d3028ca1b64b145d91.zip",
    ],
)

# FXdiv library, used for repeated integer division by the same factor
http_archive(
    name = "FXdiv",
    sha256 = "ab7dfb08829bee33dca38405d647868fb214ac685e379ec7ef2bebcd234cd44d",
    strip_prefix = "FXdiv-b408327ac2a15ec3e43352421954f5b1967701d1",
    urls = [
        "https://github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip",
    ],
)

# pthreadpool library, used for parallelization
http_archive(
    name = "pthreadpool",
    sha256 = "e6370550a1abf1503daf3c2c196e0a1c2b253440c39e1a57740ff49af2d8bedf",
    strip_prefix = "pthreadpool-43edadc654d6283b4b6e45ba09a853181ae8e850",
    urls = [
        "https://github.com/Maratyszcza/pthreadpool/archive/43edadc654d6283b4b6e45ba09a853181ae8e850.zip",
    ],
)

# cpuinfo library, used for detecting processor characteristics
http_archive(
    name = "cpuinfo",
    sha256 = "ba668f9f8ea5b4890309b7db1ed2e152aaaf98af6f9a8a63dbe1b75c04e52cb9",
    strip_prefix = "cpuinfo-3dc310302210c1891ffcfb12ae67b11a3ad3a150",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/3dc310302210c1891ffcfb12ae67b11a3ad3a150.zip",
    ],
)

# Google Test framework, used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    sha256 = "5cb522f1427558c6df572d6d0e1bf0fd076428633d080e88ad5312be0b6a8859",
    strip_prefix = "googletest-e23cdb78e9fef1f69a9ef917f447add5638daf2a",
    urls = ["https://github.com/google/googletest/archive/e23cdb78e9fef1f69a9ef917f447add5638daf2a.zip"],
)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-main",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/main.zip"],
)

http_archive(
    name = "rules_python",
    sha256 = "29a801171f7ca190c543406f9894abf2d483c206e14d6acbd695623662320097",
    strip_prefix = "rules_python-0.18.1",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.18.1/rules_python-0.18.1.tar.gz",
)

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

# TODO(mattSoulanille): Change the docker so it doesn't run as root?
# https://github.com/bazelbuild/rules_python/pull/713
# https://github.com/GoogleCloudPlatform/cloud-builders/issues/641
python_register_toolchains(
    name = "python3_8",
    ignore_root_user_error = True,
    # Available versions are listed in @rules_python//python:versions.bzl.
    python_version = "3.8",
)

load("@python3_8//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "tensorflowjs_deps",
    python_interpreter_target = interpreter,
    requirements_lock = "@//tfjs-converter/python:requirements_lock.txt",
)

load("@tensorflowjs_deps//:requirements.bzl", install_tfjs_deps = "install_deps")

install_tfjs_deps()

pip_parse(
    name = "tensorflowjs_dev_deps",
    python_interpreter_target = interpreter,
    requirements_lock = "@//tfjs-converter/python:requirements-dev_lock.txt",
)

load("@tensorflowjs_dev_deps//:requirements.bzl", install_tfjs_dev_deps = "install_deps")

install_tfjs_dev_deps()

load("//tfjs-tflite:tflite_repositories.bzl", "tflite_repositories")

tflite_repositories()

load("//tfjs-tfdf:tfdf_repositories.bzl", "tfdf_repositories")

tfdf_repositories()
