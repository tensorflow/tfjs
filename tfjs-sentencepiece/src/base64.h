#ifndef TFJS_SENTENCEPIECE_BASE64_H_
#define TFJS_SENTENCEPIECE_BASE64_H_

#include <stdexcept>
#include <string>

namespace base64 {

std::string Decode(const std::string& input) {
  static constexpr unsigned char kDecodingTable[] = {
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63, 52, 53, 54, 55, 56, 57,
      58, 59, 60, 61, 64, 64, 64, 64, 64, 64, 64, 0,  1,  2,  3,  4,  5,  6,
      7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      25, 64, 64, 64, 64, 64, 64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64};

  size_t in_len = input.size();
  if (in_len % 4 != 0) {
    throw std::invalid_argument(
        "Base64: input data size is not a multiple of 4");
  }

  size_t out_len = in_len / 4 * 3;
  if (input[in_len - 1] == '=') out_len--;
  if (input[in_len - 2] == '=') out_len--;

  std::string out;
  out.resize(out_len);

  for (size_t i = 0, j = 0; i < in_len;) {
    uint32_t a = input[i] == '=' ? 0 & i++
                                 : kDecodingTable[static_cast<int>(input[i++])];
    uint32_t b = input[i] == '=' ? 0 & i++
                                 : kDecodingTable[static_cast<int>(input[i++])];
    uint32_t c = input[i] == '=' ? 0 & i++
                                 : kDecodingTable[static_cast<int>(input[i++])];
    uint32_t d = input[i] == '=' ? 0 & i++
                                 : kDecodingTable[static_cast<int>(input[i++])];

    uint32_t triple = (a << 3 * 6) + (b << 2 * 6) + (c << 1 * 6) + (d << 0 * 6);

    if (j < out_len) out[j++] = (triple >> 2 * 8) & 0xFF;
    if (j < out_len) out[j++] = (triple >> 1 * 8) & 0xFF;
    if (j < out_len) out[j++] = (triple >> 0 * 8) & 0xFF;
  }

  return out;
}

}  // namespace base64

#endif
