This directory contains the following Google Cloud Function.

### `trigger_nightly`
Programatically triggers a Cloud Build on master. This function is called by the Cloud Scheduler at 3am EST every day (configurable via the Cloud Scheduler UI).
You can also trigger the function manually via the Cloud UI.

Command to re-deploy:
```sh
gcloud functions deploy converter_pip_nightly_test \
  --runtime nodejs8 \
  --trigger-topic nightly
```

### The pipeline

The pipeline looks like this:

1) At 3am, Cloud Scheduler writes to `nightly` topic
2) That triggers the `nightly` function, which starts a build programatically
3) That build runs and writes its status to `cloud-builds` topic
4) That triggers the `send_email` function, which sends email and chat with the build status.
