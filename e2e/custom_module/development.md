# A few notes on the custom_module subfolders

<!-- because json does not support comments -->

In each folders package.json there is an entry like:

`"make-custom-tfjs-modules": "node ./node_modules/@tensorflow/tfjs/dist/tools/custom_module/cli.js --config app_tfjs_config.json",`

This would normally look like:

`"make-custom-tfjs-modules": "npx tfjs-custom-module --config app_tfjs_config.json",`

However when yarn is installing a dependency specified with `link://` it does
not properly symlink `bin` scripts in the dependencies.

This can be fixed by changing `link://` to (the potentially more standard)
`file:..` (it also works when installing from npm). However that would
potentially introduce skew in the lock files as the paths generated after
running yarn are have parts local to the specific file system. We also have
other scripts that look to replace `link` dependencies with npm ones.

This file is just to document that quirk.

