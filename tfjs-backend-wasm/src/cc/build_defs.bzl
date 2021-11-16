def tfjs_unit_test(name, srcs, deps = [], **kwargs):
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
        ] + deps,
        **kwargs
    )

def tfjs_cc_library(name, **kwargs):
    """CC library for TensorFlow.js
    Args:
      name: The name of the TensorFlow.js cc library to define.
    """

    native.cc_library(
        name = name,
        linkstatic = True,
        **kwargs
    )
