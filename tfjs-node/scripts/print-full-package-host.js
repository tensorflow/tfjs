// This file generate and print out the full path of the pre-compiled addon in
// GCP bucket. It is used by bash script when uploading pre-compiled addon to
// GCP bucket.
console.log(
  require('../package.json').binary.host.split('.com/')[1] +
  '/napi-v' +
  process.versions.napi +
  '/' +
  require('../package.json').version + '/');
