if [[ "$NIGHTLY" = true || "$RELEASE" = true ]]; then
  yarn test-bundle-size
fi
