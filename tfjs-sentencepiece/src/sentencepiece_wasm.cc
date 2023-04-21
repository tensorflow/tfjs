#include <emscripten.h>
#include <emscripten/bind.h>

#include "tensorflow_text/core/kernels/sentencepiece/optimized_decoder.h"
#include "tensorflow_text/core/kernels/sentencepiece/optimized_encoder.h"

using namespace emscripten;

namespace tensorflow::text::sentencepiece {

EMSCRIPTEN_BINDINGS(sentencepiece) {
  value_array<std::vector<int>>("IntVector").element(emscripten::index<0>());

  enum_<EncoderResultType>("EncoderResultType")
      .value("SUCCESS", EncoderResultType::SUCCESS)
      .value("WRONG_CONFIG", EncoderResultType::WRONG_CONFIG);

  value_object<EncoderResult>("EncoderResult")
      .field("type", &EncoderResult::type)
      .field("codes", &EncoderResult::codes)
      .field("offsets", &EncoderResult::offsets);

  enum_<DecoderResultType>("DecoderResultType")
      .value("SUCCESS", DecoderResultType::SUCCESS)
      .value("WRONG_CONFIG", DecoderResultType::WRONG_CONFIG)
      .value("INVALID_INPUT", DecoderResultType::INVALID_INPUT);

  value_object<DecoderResult>("DecoderResult")
      .field("decoded", &DecoderResult::decoded)
      .field("type", &DecoderResult::type);

  function("NormalizeString", &NormalizeString);
  function("EncodeString", &EncodeString, allow_raw_pointers());
  function("DecodeString", &DecodeString, allow_raw_pointers());
}

}  // namespace tensorflow::text::sentencepiece
