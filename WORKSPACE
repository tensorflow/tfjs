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
    sha256 = "94070eff79305be05b7699207fbac5d2608054dd53e6109f7d00d923919ff45a",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/5.8.2/rules_nodejs-5.8.2.tar.gz"],
)

# Install rules_nodejs dependencies.
load("@build_bazel_rules_nodejs//:repositories.bzl", "build_bazel_rules_nodejs_dependencies")

build_bazel_rules_nodejs_dependencies()

load("@rules_nodejs//nodejs:repositories.bzl", "nodejs_register_toolchains")

nodejs_register_toolchains(
    name = "nodejs",
    node_version = "18.13.0",
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
    sha256 = "b8270749b99d8d14922d1831b93781a5560fba6f7bce65cd477fc1b6aa262535",
    strip_prefix = "emsdk-3.1.28/bazel",
    urls = ["https://github.com/emscripten-core/emsdk/archive/refs/tags/3.1.28.tar.gz"],
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
    commit = "5e8033a72a8d0f1c2b1f06e29137cc697c6b661d",
    remote = "https://github.com/google/XNNPACK.git",
    shallow_since = "1643627844 -0800",
)

# clog library, used for logging
http_archive(
    name = "clog",
    build_file = "@xnnpack//third_party:clog.BUILD",
    sha256 = "3f2dc1970f397a0e59db72f9fca6ff144b216895c1d606f6c94a507c1e53a025",
    strip_prefix = "cpuinfo-d5e37adf1406cf899d7d9ec1d317c47506ccb970",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/d5e37adf1406cf899d7d9ec1d317c47506ccb970.tar.gz",
    ],
)

git_repository(
    name = "com_google_googletest",
    commit = "cd17fa2abda2a2e4111cdabd62a87aea16835014",
    remote = "https://github.com/google/googletest.git",
    shallow_since = "1570558426 -0400",
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

TENSORFLOW_COMMIT = "5d37bd0350f0144632629c1aa2ebaef6ca76300b"

TENSORFLOW_SHA256 = "aefecbc982586a731f179c730cff5307c5906b8bebb7474da92ad974ae228a8d"

http_archive(
    name = "org_tensorflow",
    sha256 = TENSORFLOW_SHA256,
    strip_prefix = "tensorflow-" + TENSORFLOW_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/" + TENSORFLOW_COMMIT +
        ".tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

http_archive(
    name = "darts_clone",
    build_file = "@org_tensorflow_text//third_party/darts_clone:BUILD.bzl",
    sha256 = "c97f55d05c98da6fcaf7f9ecc6a6dc6bc5b18b8564465f77abff8879d446491c",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    urls = [
        "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.zip",
    ],
)

http_archive(
    name = "com_google_sentencepiece",
    build_file = "@org_tensorflow_text//third_party/sentencepiece:BUILD",
    patch_args = ["-p1"],
    patches = ["//third_party/sentencepiece:sp.patch"],
    sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
    strip_prefix = "sentencepiece-0.1.96",
    urls = [
        "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip",
    ],
)

http_archive(
    name = "org_tensorflow_text",
    sha256 = "29ead7ffc398266c5d3e6211c11d2d49632cf1412d0f94bafbc8b76a2da5e644",
    strip_prefix = "text-2.12.1",
    urls = [
        "https://github.com/tensorflow/text/archive/refs/tags/v2.12.1.tar.gz",
    ],
)
