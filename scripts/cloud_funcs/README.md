This directory contains the following Google Cloud Functions.

### `trigger_nightly`
Programatically triggers a Cloud Build on master. This function is called by the Cloud Scheduler around 4am "America/New York" time every day (configurable via the Cloud Scheduler UI).
You can also trigger the function manually via the Cloud UI.

Command to re-deploy:
```sh
gcloud functions deploy nightly_tfjs \
  --runtime nodejs8 \
  --trigger-topic nightly_tfjs
```

If a build was triggered by nightly, there is a substitution variable `_NIGHTLY=true`.
You can forward the substitution as the `NIGHTLY` environment variable so the scripts can use it, by specifying `env: ['NIGHTLY=$_NIGHTLY']` in `cloudbuild.yml`. E.g. `integration_tests/benchmarks/benchmark_cloud.sh` uses the `NIGHTLY` bit to always run on nightly.
