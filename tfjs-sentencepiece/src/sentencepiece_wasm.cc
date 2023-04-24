#include <emscripten.h>
#include <emscripten/bind.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "base64.h"
// #include "sentencepiece_processor.h"
#include "tensorflow_text/core/kernels/sentencepiece/encoder_config_generated.h"
#include "tensorflow_text/core/kernels/sentencepiece/optimized_decoder.h"
#include "tensorflow_text/core/kernels/sentencepiece/optimized_encoder.h"

#include "flatbuffers/table.h"
#include "flatbuffers/verifier.h"

namespace tfjs::sentencepiece {

using tensorflow::text::sentencepiece::DecoderResult;
using tensorflow::text::sentencepiece::DecoderResultType;
using tensorflow::text::sentencepiece::EncoderResult;
using tensorflow::text::sentencepiece::EncoderResultType;

static std::unordered_map<std::string, std::string> sp_model_pool;

std::string RegisterModel(const std::string& model_base64) {
  std::string key = std::to_string(std::hash<std::string>()(model_base64));

  auto it = sp_model_pool.find(key);
  if (it != sp_model_pool.end()) {
    return key;
  }

  std::string model = base64::Decode(model_base64);
  sp_model_pool.insert({key, std::move(model)});
  return key;
}

namespace {

using namespace tensorflow::text::sentencepiece;
using namespace std;
using namespace flatbuffers;

void DebugModel(const std::string& model_base64) {
  std::string model = base64::Decode(model_base64);

  const EncoderConfig* config = GetEncoderConfig(model.c_str());

  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(model.c_str()), model.size());

  cout << "Verify:: " << (config->Verify(verifier) ? "SUCCESS" : "FAIL")
       << endl;
  cout << config->start_code() << " " << config->end_code() << endl;
  cout << config->normalized_replacements()->size() << endl;
}

}  // namespace

EncoderResult EncodeString(const std::string& string,
                           const std::string& config_key, bool add_bos,
                           bool add_eos, bool reverse) {
  auto it = sp_model_pool.find(config_key);
  if (it == sp_model_pool.end()) {
    auto error_message = "EncodeString error: Model not found";
    std::cout << error_message << std::endl;
    throw std::invalid_argument(error_message);
  }

  return tensorflow::text::sentencepiece::EncodeString(
      string, it->second.c_str(), add_bos, add_eos, reverse);
}

DecoderResult DecodeString(const int* encoded, size_t encoded_size,
                           const std::string& config_key) {
  auto it = sp_model_pool.find(config_key);
  if (it == sp_model_pool.end()) {
    auto error_message = "EncodeString error: Model not found";
    std::cout << error_message << std::endl;
    throw std::invalid_argument(error_message);
  }

  std::vector<int> encoded_vec(encoded, encoded + encoded_size);
  return tensorflow::text::sentencepiece::DecodeString(encoded_vec,
                                                       it->second.c_str());
}

EMSCRIPTEN_BINDINGS(tfjs_sentencepiece) {
  emscripten::function("RegisterModel", &RegisterModel);
  emscripten::value_array<std::vector<int>>("IntVector")
      .element(emscripten::index<0>());

  emscripten::enum_<EncoderResultType>("EncoderResultType")
      .value("SUCCESS", EncoderResultType::SUCCESS)
      .value("WRONG_CONFIG", EncoderResultType::WRONG_CONFIG);

  emscripten::value_object<EncoderResult>("EncoderResult")
      .field("type", &EncoderResult::type)
      .field("codes", &EncoderResult::codes)
      .field("offsets", &EncoderResult::offsets);

  emscripten::enum_<DecoderResultType>("DecoderResultType")
      .value("SUCCESS", DecoderResultType::SUCCESS)
      .value("WRONG_CONFIG", DecoderResultType::WRONG_CONFIG)
      .value("INVALID_INPUT", DecoderResultType::INVALID_INPUT);

  emscripten::value_object<DecoderResult>("DecoderResult")
      .field("decoded", &DecoderResult::decoded)
      .field("type", &DecoderResult::type);

  emscripten::function("EncodeString", &EncodeString);
  emscripten::function("DecodeString", &DecodeString,
                       emscripten::allow_raw_pointers());

  emscripten::function("DebugModel", &DebugModel);
}

// EMSCRIPTEN_BINDINGS(sentencepiece) {

// }

}  // namespace tfjs::sentencepiece
