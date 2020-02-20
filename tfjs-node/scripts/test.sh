echo "=====Build tfjs.====="
cd ../tfjs && yarn && yarn build

echo "=====Start testing.====="
cd ../tfjs-node && yarn && yarn build && ts-node src/run_tests.ts
