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
        "template": attr.label(
            default = Label("@//tools:karma_template.conf.js"),
            allow_single_file = True,
            doc = "The karma config template to expand",
        ),
        "browser": attr.string(
            mandatory = True,
            doc = "The browser to run",
        )
    },
    outputs = {"config_file": "%{name}.js"}
)

def tfjs_web_test(name, ci=True, **kwargs):
    tags = kwargs.pop("tags", [])
    browsers = kwargs.pop("browsers", [
        "bs_chrome_mac",
        "bs_firefox_mac",
        "bs_safari_mac",
        "bs_ios_11",
        "bs_android_9",
        "win_10_chrome",
    ])

    # For local testing
    # NOTE: If karma_template.conf.js is changed such that it affects the tests
    # outside of choosing which browsers they run on, it may need to be added
    # here.
    karma_web_test(
        name = name,
        tags = ["native"] + tags,
        **kwargs,
    )

    # If the target is marked as not for CI, don't create CI targets
    if not ci:
        return

    # Create a 'karma_web_test' target for each browser
    for browser in browsers:
        config_file = "{}_config_{}".format(name, browser)
        _make_karma_config(
            name = config_file,
            browser = browser,
        )        

        karma_web_test(
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
            tags = ["ci"] + tags, # Tag to be run in ci
            **kwargs,
        )
