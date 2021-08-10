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

load("@npm//@bazel/concatjs:index.bzl", "karma_web_test")

def _make_karma_config_impl(ctx):
    output_file_path = ctx.label.name + ".js"
    output_file = ctx.actions.declare_file(output_file_path)
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.config_file,
        substitutions = {
            "TEMPLATE_browser": ctx.attr.browser,
        },
    )
    return [DefaultInfo(files = depset([output_file]))]

_make_karma_config = rule(
    implementation = _make_karma_config_impl,
    attrs = {
        "browser": attr.string(
            mandatory = True,
            doc = "The browser to run",
        ),
        "template": attr.label(
            default = Label("@//tools:karma_template.conf.js"),
            allow_single_file = True,
            doc = "The karma config template to expand",
        ),
    },
    outputs = {"config_file": "%{name}.js"},
)

def tfjs_web_test(name, ci = True, **kwargs):
    tags = kwargs.pop("tags", [])
    browsers = kwargs.pop("browsers", [
        "bs_chrome_mac",
        "bs_firefox_mac",
        "bs_safari_mac",
        "bs_ios_11",
        "bs_android_9",
        "win_10_chrome",
    ])

    # Browsers that should always run in presubmit checks All browsers are run
    # in nightly, but only the ones listed here run for each PR.
    presubmit_browsers = kwargs.pop(
        "presubmit_browsers",
        [browsers[0]] if len(browsers) > 0 else [],
    )

    size = kwargs.pop("size", "large")
    timeout = kwargs.pop("timeout", "long")

    # For local testing
    # NOTE: If karma_template.conf.js is changed such that it affects the tests
    # outside of choosing which browsers they run on, it may need to be added
    # here.
    karma_web_test(
        size = size,
        timeout = timeout,
        name = name,
        tags = ["native"] + tags,
        **kwargs
    )

    # Create a 'karma_web_test' target for each browser
    for browser in browsers:
        config_file = "{}_config_{}".format(name, browser)
        _make_karma_config(
            name = config_file,
            browser = browser,
        )

        additional_tags = []
        if ci:
            # Tag to be run in nightly.
            additional_tags.append("nightly")
            if browser in presubmit_browsers:
                # Tag to also be run in PR presubmit tests.
                additional_tags.append("ci")

        karma_web_test(
            size = size,
            timeout = timeout,
            name = "browserstack_{}_{}".format(browser, name),
            config_file = config_file,
            peer_deps = [
                "@npm//karma",
                "@npm//karma-jasmine",
                "@npm//karma-requirejs",
                "@npm//karma-sourcemap-loader",
                "@npm//requirejs",
                # The above dependencies are the default when 'peer_deps' is not
                # specified. They are manually spefied here so we can append
                # karma-browserstack-launcher.
                "@npm//karma-browserstack-launcher",
            ],
            tags = tags + additional_tags,
            **kwargs
        )
