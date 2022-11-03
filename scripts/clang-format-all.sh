set -eEuo pipefail
cd `git rev-parse --show-toplevel`
find . -type d -name node_modules -prune -o \( -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" \) -exec parallel ./node_modules/.bin/clang-format -i --style='file' -- {} \+
