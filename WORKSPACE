load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "build_bazel_rules_nodejs",
    sha256 = "fcc6dccb39ca88d481224536eb8f9fa754619676c6163f87aa6af94059b02b12",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/3.2.0/rules_nodejs-3.2.0.tar.gz"],
)

load("@build_bazel_rules_nodejs//:index.bzl", "yarn_install")
yarn_install(
    name = "npm",
    package_json = "//:package.json",
    yarn_lock = "//:yarn.lock",
)

# Emscripten toolchain
http_archive(
    name = "emsdk",
    strip_prefix = "emsdk-c1589b55641787d55d53e883852035beea9aec3f/bazel",
    url = "https://github.com/emscripten-core/emsdk/archive/c1589b55641787d55d53e883852035beea9aec3f.tar.gz",
    sha256 = "7a58a9996b113d3e0675df30b5f17e28aa47de2e684a844f05394fe2f6f12e8e",
)

load("@emsdk//:deps.bzl", emsdk_deps = "deps")
emsdk_deps()

load("@emsdk//:emscripten_deps.bzl", emsdk_emscripten_deps = "emscripten_deps")
emsdk_emscripten_deps()


load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
# xnnpack used for fast vectorized wasm operations
git_repository(
    name = "xnnpack",
    commit = "3bfbdaf00211b313b143af39279bb6bf1f7effc0",
    remote = "https://github.com/google/XNNPACK.git",
    shallow_since = "1617056836 -0700",
)

# The libraries below are transitive dependencies of XNNPACK that we need to
# explicitly enumerate here. See https://docs.bazel.build/versions/master/external.html#transitive-dependencies

# FP16 library, used for half-precision conversions
http_archive(
    name = "FP16",
    build_file = "@xnnpack//third_party:FP16.BUILD",
    sha256 = "0d56bb92f649ec294dbccb13e04865e3c82933b6f6735d1d7145de45da700156",
    strip_prefix = "FP16-3c54eacb74f6f5e39077300c5564156c424d77ba",
    urls = [
        "https://github.com/Maratyszcza/FP16/archive/3c54eacb74f6f5e39077300c5564156c424d77ba.zip",
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
    sha256 = "8461f6540ae9f777ce20d1c0d1d249e5e61c438744fb390c0c6f91940aa69ea3",
    strip_prefix = "pthreadpool-545ebe9f225aec6dca49109516fac02e973a3de2",
    urls = [
        "https://github.com/Maratyszcza/pthreadpool/archive/545ebe9f225aec6dca49109516fac02e973a3de2.zip",
    ],
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

# cpuinfo library, used for detecting processor characteristics
http_archive(
    name = "cpuinfo",
    build_file = "@xnnpack//third_party:cpuinfo.BUILD",
    patches = ["@xnnpack//third_party:cpuinfo.patch"],
    sha256 = "a7f9a188148a1660149878f737f42783e72f33a4f842f3e362fee2c981613e53",
    strip_prefix = "cpuinfo-ed8b86a253800bafdb7b25c5c399f91bff9cb1f3",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/ed8b86a253800bafdb7b25c5c399f91bff9cb1f3.zip",
    ],
)

# psimd library, used for fallback 128-bit SIMD micro-kernels
http_archive(
    name = "psimd",
    build_file = "@xnnpack//third_party:psimd.BUILD",
    sha256 = "dc615342bcbe51ca885323e51b68b90ed9bb9fa7df0f4419dbfa0297d5e837b7",
    strip_prefix = "psimd-072586a71b55b7f8c584153d223e95687148a900",
    urls = [
        "https://github.com/Maratyszcza/psimd/archive/072586a71b55b7f8c584153d223e95687148a900.zip",
    ],
)

git_repository(
    name = "com_google_googletest",
    commit = "cd17fa2abda2a2e4111cdabd62a87aea16835014",
    remote = "https://github.com/google/googletest.git",
)

http_archive(
    name = "rules_cc",
    sha256 = "90d5a66950b492cbf86201cdc49c4b59796a85a4eb9fd63c07afe5f7132ea623",
    strip_prefix = "rules_cc-8346df34b6593b051403b8e429db15c7f4ead937",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/8346df34b6593b051403b8e429db15c7f4ead937.zip",
    ],
)
