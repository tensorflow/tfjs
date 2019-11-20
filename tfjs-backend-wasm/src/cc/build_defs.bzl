def tfjs_unit_test(name, srcs, deps = []):
    """Unit test binary based on Google Test.
    Args:
      name: The name of the test target to define.
      srcs: The list of source and header files.
      deps: The list of additional libraries to be linked. Google Test library
            (with main() function) is always added as a dependency and does not
            need to be explicitly specified.
    """

    native.cc_test(
        name = name,
        srcs = srcs,
        linkstatic = True,
        deps = [
            "@com_google_googletest//:gtest_main",
        ] + deps
    )

def tfjs_cc_library(name, srcs = [], hdrs = [], deps = []):
    """CC library for TensorFlow.js
    Args:
      name: The name of the TensorFlow.js cc library to define.
      srcs: The list of source files.
      hdrs: The list of header files.
      deps: The list of additional libraries to be linked.
    """

    native.cc_library(
      name = name,
      linkstatic = True,
      linkopts = [
        "-s ALLOW_MEMORY_GROWTH=0",
        "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
        # "-s DISABLE_EXCEPTION_CATCHING=1",
        "-s FILESYSTEM=0",
        # "-s EXIT_RUNTIME=0",
        "-s EXPORTED_FUNCTIONS='[\"_malloc\", \"_free\"]'",
        "-s EXTRA_EXPORTED_RUNTIME_METHODS='[\"cwrap\"]'",
        "-s ENVIRONMENT=web",
        "-s MODULARIZE=1",
        "-s EXPORT_NAME=WasmBackendModule",
        "-s MALLOC=emmalloc",
        "-s SAFE_HEAP=1",
        "-s ASSERTIONS=2",
        "-s TOTAL_MEMORY=536870912",
        "-s STACK_OVERFLOW_CHECK=2",
        "-s DEMANGLE_SUPPORT=1",
        "-s WARN_UNALIGNED=1",
        "-s SAFE_HEAP_LOG=1",
        "-s ALIASING_FUNCTION_POINTERS=0",
        "-s DISABLE_EXCEPTION_CATCHING=0",
        "-s EXCEPTION_DEBUG=1",
      ],
      srcs = srcs,
      hdrs = hdrs,
      deps = deps,
    )
