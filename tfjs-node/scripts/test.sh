echo "=====Build tfjs-core.====="
cd ../tfjs-core && yarn && yarn build

echo "=====Build tfjs-converter.====="
cd ../tfjs-converter && yarn && yarn build

echo "=====Build tfjs-layers.====="
cd ../tfjs-layers && yarn && yarn build

echo "=====Build tfjs-data.====="
cd ../tfjs-data && yarn && yarn build

echo "=====Build tfjs.====="
cd ../tfjs && yarn && yarn build

echo "=====Start testing.====="
cd ../tfjs-node && yarn && yarn build && ts-node src/run_tests.ts
