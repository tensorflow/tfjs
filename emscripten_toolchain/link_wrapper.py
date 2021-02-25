#!/usr/bin/env python
"""wrapper around emcc link step.

This wrapper currently serves the following purposes.

1. Ensures we always link to file with .js extension. The upstream default
   it to link to an llvm bitcode file which is never (AFAICT) want to do that.

2. When building with --config=wasm the final output is multiple files, usually
   at least one .js and one .wasm file. Since the cc_binary link step only
   allows a single output, we must tar up the outputs into a single file.

3. Add quotes around arguments that need them in the response file to work
   around a bazel quirk.
"""

from __future__ import print_function

import os
import subprocess
import sys

# Only argument should be @path/to/parameter/file
assert sys.argv[1][0] == '@'
param_filename = sys.argv[1][1:]
param_file_args = [l.strip() for l in open(param_filename, 'r').readlines()]

output_index = param_file_args.index('-o') + 1
orig_output = js_output = param_file_args[output_index]
outdir = os.path.dirname(orig_output)

# google3-only(TODO(b/139440956): Default to False once the bug is fixed)
replace_response_file = any(' ' in a for a in param_file_args)

if not os.path.splitext(orig_output)[1]:
  js_output = orig_output + '.js'
  param_file_args[output_index] = js_output
  replace_response_file = True

# Re-write response file if needed.
if replace_response_file:
  new_param_filename = param_filename + '.modified'
  with open(new_param_filename, 'w') as f:
    for param in param_file_args:
      if ' ' in param:
        f.write('"%s"' % param)
      else:
        f.write(param)
      f.write('\n')
  sys.argv[1] = '@' + new_param_filename

emcc_py = os.path.join(os.environ['EMSCRIPTEN'], 'emcc.py')
rtn = subprocess.call(['python3', emcc_py] + sys.argv[1:])
if rtn != 0:
  sys.exit(1)

js_name = os.path.basename(js_output)
base_name = os.path.splitext(js_name)[0]

files = []
extensions = [
    '.js',
    '.wasm',
    '.wasm.map',
    '.js.mem',
    '.fetch.js',
    '.worker.js',
    '.data',
    '.js.symbols',
    '.wasm.debug.wasm'
]

for ext in extensions:
  filename = base_name + ext
  if os.path.exists(os.path.join(outdir, filename)):
    files.append(filename)

wasm_base = os.path.join(outdir, base_name + '.wasm')
if os.path.exists(wasm_base + '.debug.wasm') and os.path.exists(wasm_base):
  # If we have a .wasm.debug.wasm file and a .wasm file, we need to rewrite the
  # section in the .wasm file that refers to it. The path that's in there
  # is the blaze output path; we want it to be just the filename.

  llvm_objcopy = os.path.join(
      os.environ['EMSCRIPTEN'], 'llvm-bin/llvm-objcopy')
  # First, check to make sure the .wasm file has the header that needs to be
  # rewritten.
  rtn = subprocess.call([
      llvm_objcopy,
      '--dump-section=external_debug_info=/dev/null',
      wasm_base], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if rtn == 0:
    # If llvm-objcopy did not return an error, the external_debug_info section
    # must exist, so we're good to continue.

    # Next we need to convert length of the filename to LEB128.
    # Start by converting the length of the filename to a bit string.
    bit_string = '{0:b}'.format(len(base_name + '.wasm.debug.wasm'))

    # Pad the bit string with 0s so that its length is a multiple of 7.
    while len(bit_string) % 7 != 0:
      bit_string = '0' + bit_string

    # Break up our bit string into chunks of 7.
    # We do this backwards because the final format is little-endian.
    final_bytes = bytearray()
    for i in reversed(range(0, len(bit_string), 7)):
      binary_part = bit_string[i:i + 7]
      if i != 0:
        # Every chunk except the last one needs to be prepended with '1'.
        # The length of each chunk is 7, so that one has an implicit '0'.
        binary_part = '1' + binary_part
      final_bytes.append(int(binary_part, 2))
    # Finally, add the actual filename.
    final_bytes.extend(base_name + '.wasm.debug.wasm')

    # Write our length + filename bytes to a temp file.
    with open('debugsection.tmp', 'wb+') as f:
      f.write(final_bytes)
      f.close()

    # First delete the old section.
    subprocess.check_call([
        llvm_objcopy,
        wasm_base,
        '--remove-section=external_debug_info'])
    # Rewrite section with the new size and filename from the temp file.
    subprocess.check_call([
        llvm_objcopy,
        wasm_base,
        '--add-section=external_debug_info=debugsection.tmp'])

# If we have more than one output file then create tarball
if len(files) > 1:
  cmd = ['tar', 'cf', 'tmp.tar'] + files
  subprocess.check_call(cmd, cwd=outdir)
  os.rename(os.path.join(outdir, 'tmp.tar'), orig_output)
elif len(files) == 1:
  # Otherwise, if only have a single output than move it to the expected name
  if files[0] != os.path.basename(orig_output):
    os.rename(os.path.join(outdir, files[0]), orig_output)
else:
  print('emcc.py did not appear to output any known files!')
  sys.exit(1)

sys.exit(0)
