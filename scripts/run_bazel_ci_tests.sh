set -e
yarn bazel test --action_env=BROWSERSTACK_USERNAME=$BROWSERSTACK_USERNAME --action_env=BROWSERSTACK_KEY=$BROWSERSTACK_KEY --config=ci --flaky_test_attempts=3 --test_output=all `./node_modules/.bin/bazel query --output label 'attr("tags", "ci", ...)'`
