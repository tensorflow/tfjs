# Description:
#   Portable pthread-based thread pool for C and C++

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "pthreadpool_impl",
    srcs = select({
        "//third_party/emscripten:pthreads_off": ["src/threadpool-shim.c"],
        "//conditions:default": ["src/threadpool-pthreads.c"],
    }) + [
        "src/threadpool-legacy.c",
    ],
    copts = [
        "-std=gnu11",
        "-Wno-deprecated-declarations",
        "-Ithird_party/cpuinfo/include",
        "-Ithird_party/cpuinfo/src",
    ] + select({
        ":optimized_build": ["-O2"],
        "//conditions:default": [],
    }) + select({
        ":linux_aarch64": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":chromiumos_arm64": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":chromiumos_armv7": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":android_arm64": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":android_armv7": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        "//conditions:default": ["-DPTHREADPOOL_USE_CPUINFO=0"],
    }) + select({
        ":pthreadpool_sync_primitive_explicit_condvar": [
            "-DPTHREADPOOL_USE_FUTEX=0",
        ],
        ":pthreadpool_sync_primitive_explicit_futex": [
            "-DPTHREADPOOL_USE_FUTEX=1",
        ],
        "//conditions:default": [],
    }),
    includes = [
        "include",
    ],
    linkopts = select({
        "//third_party/emscripten:pthreads_on": [
            "-s ALLOW_BLOCKING_ON_MAIN_THREAD=1",
            "-s PTHREAD_POOL_SIZE=8",
        ],
        "//conditions:default": [],
    }),
    strip_include_prefix = "include",
    textual_hdrs = [
        "include/pthreadpool.h",
        "src/threadpool-atomics.h",
        "src/threadpool-utils.h",
    ],
    deps = [
        "//third_party/FXdiv",
    ] + select({
        ":linux_aarch64": ["//third_party/cpuinfo"],
        ":chromiumos_arm64": ["//third_party/cpuinfo"],
        ":chromiumos_armv7": ["//third_party/cpuinfo"],
        ":android_arm64": ["//third_party/cpuinfo"],
        ":android_armv7": ["//third_party/cpuinfo"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "pthreadpool",
    hdrs = [
        "include/pthreadpool.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":pthreadpool_impl",
    ],
)

################################## Unit tests ##################################

EMSCRIPTEN_TEST_LINKOPTS = [
    "-s ASSERTIONS=2",
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
    "-s DEMANGLE_SUPPORT=1",
    "-s EXIT_RUNTIME=1",
    "-s ALLOW_MEMORY_GROWTH=0",
    "-s TOTAL_MEMORY=67108864",  # 64M
]

cc_test(
    name = "pthreadpool_test",
    srcs = ["test/pthreadpool.cc"],
    linkopts = select({
        ":emscripten": EMSCRIPTEN_TEST_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pthreadpool",
        "//testing/base/public:gunit",
        "//testing/base/public:gunit_main",
    ],
)

################################## Benchmarks ##################################

EMSCRIPTEN_BENCHMARK_LINKOPTS = [
    "-s ASSERTIONS=1",
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
    "-s EXIT_RUNTIME=1",
    "-s ALLOW_MEMORY_GROWTH=0",
]

cc_binary(
    name = "latency_bench",
    srcs = ["bench/latency.cc"],
    linkopts = select({
        ":emscripten": EMSCRIPTEN_BENCHMARK_LINKOPTS,
        "//conditions:default": [],
    }),
    tags = ["benchmark"],
    deps = [
        ":pthreadpool",
        # note: internal version is incompatible
        "//third_party/benchmark",
    ],
)

cc_binary(
    name = "throughput_bench",
    srcs = ["bench/throughput.cc"],
    linkopts = select({
        ":emscripten": EMSCRIPTEN_BENCHMARK_LINKOPTS,
        "//conditions:default": [],
    }),
    tags = ["benchmark"],
    deps = [
        ":pthreadpool",
        # note: internal version is incompatible
        "//third_party/benchmark",
    ],
)

############################# Build configurations #############################

# Synchronize workers using pthreads condition variable.
config_setting(
    name = "pthreadpool_sync_primitive_explicit_condvar",
    define_values = {"pthreadpool_sync_primitive": "condvar"},
)

# Synchronize workers using futex.
config_setting(
    name = "pthreadpool_sync_primitive_explicit_futex",
    define_values = {"pthreadpool_sync_primitive": "futex"},
)

config_setting(
    name = "optimized_build",
    values = {
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "emscripten",
    flag_values = {
        "//tools/cpp:cc_target_os": "emscripten",
    },
)

config_setting(
    name = "linux_aarch64",
    flag_values = {"//tools/cpp:cc_target_os": "linux-google"},
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "chromiumos_arm64",
    flag_values = {"//tools/cpp:cc_target_os": "chromiumos"},
    values = {"cpu": "arm"},
)

config_setting(
    name = "chromiumos_armv7",
    flag_values = {"//tools/cpp:cc_target_os": "chromiumos"},
    values = {"cpu": "armeabi-v7a"},
)

config_setting(
    name = "android_armv7",
    flag_values = {"//tools/cpp:cc_target_os": "android"},
    values = {"android_cpu": "armeabi-v7a"},
)

config_setting(
    name = "android_arm64",
    flag_values = {"//tools/cpp:cc_target_os": "android"},
    values = {"android_cpu": "arm64-v8a"},
)
