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

PEER_DEPS = [
    "@npm//karma",
    "@npm//karma-jasmine",
    "@npm//karma-requirejs",
    "@npm//karma-sourcemap-loader",
    "@npm//requirejs",
    "@npm//@bazel/concatjs",
    # The above dependencies are the default when 'peer_deps' is not
    # specified. They are manually spefied here so we can append
    # extra dependencies.
    "@npm//karma-jasmine-html-reporter",
    "@npm//karma-jasmine-order-reporter",
]

# @//:grep flag greps for a set of tests to run
GrepProvider = provider(fields = ["grep"])

def _grep_flag_impl(ctx):
    return GrepProvider(grep = ctx.build_setting_value)

grep_flag = rule(
    implementation = _grep_flag_impl,
    build_setting = config.string(flag = True),
)

# @//:headless flag runs tests headlessly. Defaults to true.
HeadlessProvider = provider(fields = ["headless"])

def _headless_flag_impl(ctx):
    return HeadlessProvider(headless = ctx.build_setting_value)

headless_flag = rule(
    implementation = _headless_flag_impl,
    build_setting = config.bool(flag = True),
)

def _make_karma_config_impl(ctx):
    grep = ctx.attr._grep[GrepProvider].grep
    headless = ctx.attr.headless[HeadlessProvider].headless
    output_file_path = ctx.label.name + ".js"
    output_file = ctx.actions.declare_file(output_file_path)
    args = ctx.attr.args
    if grep:
        args = args + ["--grep=" + grep]

    seed = ctx.attr.seed
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.config_file,
        substitutions = {
            "TEMPLATE_args": str(args),
            "TEMPLATE_browser": ctx.attr.browser,
            "TEMPLATE_headless": "true" if headless else "false",
            "TEMPLATE_jasmine_random": "false" if seed else "true",
            "TEMPLATE_jasmine_seed": seed if seed else "undefined",
        },
    )
    return [DefaultInfo(files = depset([output_file]))]

_make_karma_config = rule(
    implementation = _make_karma_config_impl,
    attrs = {
        "args": attr.string_list(
            # TODO(mattsoulanille): Make this a dict instead of a list
            doc = """Args to pass through to the client.

            They appear in '__karma__.config.args'.
            """,
        ),
        "browser": attr.string(
            default = "",
            doc = "The browser to run",
        ),
        "headless": attr.label(
            default = "@//:headless",
            doc = """Whether to run chrome tests headlessly.

            Defaults to true on most platforms. Note that not all browsers
            support headless mode. Check //tools/karma_template.conf.js for
            more details.
            """,
        ),
        "seed": attr.string(
            default = "",
            doc = """Use this seed for test order.

            If not specified or empty, use a random seed every time.
            """,
        ),
        "template": attr.label(
            default = Label("@//tools:karma_template.conf.js"),
            allow_single_file = True,
            doc = "The karma config template to expand",
        ),
        "_grep": attr.label(default = "@//:grep"),
    },
    outputs = {"config_file": "%{name}.js"},
)

def tfjs_web_test(name, ci = True, args = [], **kwargs):
    tags = kwargs.pop("tags", [])
    local_browser = kwargs.pop("local_browser", "")
    seed = kwargs.pop("seed", "")
    headless = kwargs.pop("headless", True)

    browsers = kwargs.pop("browsers", [
        "bs_chrome_mac",
        "bs_firefox_mac",
        "bs_safari_mac",
        "bs_ios_12",
        "bs_android_10",
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
    config_file = "{}_config".format(name)
    _make_karma_config(
        name = config_file,
        args = args,
        browser = local_browser,
        seed = seed,
    )

    karma_web_test(
        size = size,
        timeout = timeout,
        name = name,
        config_file = config_file,
        configuration_env_vars = [] if headless else ["DISPLAY"],
        peer_deps = PEER_DEPS,
        tags = ["native", "no-remote-exec"] + tags,
        **kwargs
    )

    # Create a 'karma_web_test' target for each browser
    for browser in browsers:
        config_file = "{}_config_{}".format(name, browser)
        _make_karma_config(
            name = config_file,
            browser = browser,
            args = args,
            seed = seed,
        )

        additional_tags = ["no-remote-exec"]
        if ci:
            # Tag to be run in nightly.
            additional_tags.append("nightly")
            if browser in presubmit_browsers:
                # Tag to also be run in PR presubmit tests.
                additional_tags.append("ci")

        karma_web_test(
            size = size,
            timeout = timeout,
            name = "{}_{}".format(browser, name),
            config_file = config_file,
            peer_deps = PEER_DEPS + ["@npm//karma-browserstack-launcher"],
            tags = tags + additional_tags,
            **kwargs
        )
