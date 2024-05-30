This directory contains the following Google Cloud Functions.

### `trigger_nightly`

Programmatically triggers a Cloud Build on master. This function is called by the Cloud Scheduler around 4am "America/New York" time every day (configurable via the Cloud Scheduler UI).
You can also trigger the function manually via the Cloud UI.

Command to re-deploy:

```sh
gcloud functions deploy nightly_tfjs \
  --runtime nodejs14 \
  --trigger-topic nightly_tfjs
```

If a build was triggered by nightly, there is a substitution variable `_NIGHTLY=true`.
You can forward the substitution as the `NIGHTLY` environment variable so the scripts can use it, by specifying `env: ['NIGHTLY=$_NIGHTLY']` in `cloudbuild.yml`. E.g. `integration_tests/benchmarks/benchmark_cloud.sh` uses the `NIGHTLY` bit to always run on nightly.

### `send_email`

Sends an email and a chat message with the nightly build status. Every build sends a message to the `cloud-builds` topic with its build information. The `send_email` function is subscribed to that topic and ignores all builds (e.g. builds triggered by pull requests) **except** for the nightly build and sends an email to an internal mailing list with its build status around 4:40am.

Command to re-deploy:

```sh
gcloud functions deploy send_email \
  --runtime nodejs14 \
  --stage-bucket learnjs-174218_cloudbuild \
  --trigger-topic cloud-builds \
  --set-env-vars MAILGUN_API_KEY="[API_KEY_HERE]",HANGOUTS_URL="[URL_HERE]"
```

### `sync_reactnative`

Makes a request to browserStack to sync the current build of the tfjs-react-native integration app to browserstack. The app itself is stored in a GCP bucket. This needs to be done at least once every 30 days and is triggered via cloud scheduler via the `sync_reactnative` topic.
Currently set to run weekly on Thursdays at 3AM.

Command to re-deploy:

```sh
gcloud functions deploy sync_reactnative \
  --runtime nodejs14 \
  --trigger-topic sync_reactnative \
  --set-env-vars HANGOUTS_URL="[URL_HERE]",BOTS_HANGOUTS_URL="[URL_HERE]"
```

### The pipeline

The pipeline looks like this:

1. At 4am, Cloud Scheduler writes to `nightly_tfjs` topic
2. That triggers the `nightly_tfjs` function, which starts a build programmatically
3. That build runs and writes its status to `cloud-builds` topic
4. That triggers the `send_email` function, which sends email and chat with the build status.
