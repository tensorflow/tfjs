:: Download TensorFlow GPU library and compile native node addon
node scripts/install.js gpu download
call yarn
call yarn build-addon-from-source
:: Compress and upload the GPU node addon to GCP bucket.
for /f %%i in ('node scripts/get-addon-name.js') do set PACKAGE_NAME=%%i
:: Update addon tarball name to GPU
set PACKAGE_NAME=%PACKAGE_NAME:CPU=GPU%
for /f %%i in ('node -p "process.versions.napi"') do set NAPI_VERSION=%%i
tar -czvf %PACKAGE_NAME% -C lib napi-v%NAPI_VERSION%/tfjs_binding.node
for /f %%i in ('node scripts/print-full-package-host') do set PACKAGE_HOST=%%i
gsutil cp %PACKAGE_NAME% gs://%PACKAGE_HOST%
