## Starter project in TypeScript

This is a starter project demonstrating how to use
[deeplearn.js](https://deeplearn.js) in a TypeScript environment.

Before we start, see [`main.ts`](./main.ts) for the example code. It sums a 1D
array with a scalar (taking advantage of broadcasting) and outputs the result
in the console. Feel free to also check [`package.json`](./package.json) for the
scripts we will be using.

> NOTE: This setup uses [browserify](http://browserify.org/) as a bundler.
Feel free to change the setup to use your preferred bundler
([WebPack](https://webpack.github.io/), [Rollup](https://rollupjs.org/), ...).

We start with preparing the dev environment:

```bash
$ yarn prep # Installs node modules and prepares the dev environment.
```

To interactively develop with fast edit-refresh cycle (~200-400ms):

```bash
$ yarn watch
>> 1275567 bytes written to dist/bundle.js (0.58 seconds) at 10:18:10 AM
```

Then visit `index.html` in the browser and open the console. You should see:

```
Float32Array(3) [3, 4, 5]
Float32Array(3) [3, 4, 5]
Float32Array(3) [3, 4, 5]
```

To produce a non-minified bundle for production:

```
$ yarn build
```

Stores the output in `dist/bundle.js`.

To produce a minified bundle for production:

```
$ yarn deploy
```

Stores the output in `dist/bundle.min.js`.
