#include <emscripten.h>
#include <emscripten/bind.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "base64.h"
#include "sentencepiece_processor.h"
#include "tensorflow_text/core/kernels/sentencepiece/encoder_config_generated.h"
#include "tensorflow_text/core/kernels/sentencepiece/optimized_decoder.h"
#include "tensorflow_text/core/kernels/sentencepiece/optimized_encoder.h"

#include "flatbuffers/table.h"
#include "flatbuffers/verifier.h"

namespace tfjs::sentencepiece {

namespace {

template <typename Status>
void AssertOk(Status status) {
  assert(status.ok());
}

class SentencePieceModel {
 public:
  SentencePieceModel() = delete;

  SentencePieceModel(const SentencePieceModel&) = delete;
  SentencePieceModel& operator=(const SentencePieceModel&) = delete;

  explicit SentencePieceModel(const std::string& serialized_model_proto) {
    processor.LoadFromSerializedProto(serialized_model_proto);
  }

  void SetOptions(bool add_bos, bool add_eos, bool reverse) {
    std::string options;
    if (add_bos) options += options.empty() ? "bos" : ":bos";
    if (add_eos) options += options.empty() ? "eos" : ":eos";
    if (reverse) options += options.empty() ? "reverse" : ":reverse";
    AssertOk(processor.SetEncodeExtraOptions(options));
    AssertOk(processor.SetDecodeExtraOptions(options));
  }

  ::sentencepiece::SentencePieceProcessor processor;
};

std::unordered_map<std::string, std::unique_ptr<SentencePieceModel>>
    sp_model_pool;

std::string RegisterModel(const std::string& serialized_proto) {
  std::string key = std::to_string(std::hash<std::string>()(serialized_proto));

  auto it = sp_model_pool.find(key);
  if (it != sp_model_pool.end()) {
    return key;
  }

  sp_model_pool.insert(
      {key, std::make_unique<SentencePieceModel>(serialized_proto)});
  return key;
}

std::string RegisterModelBase64(const std::string& serialized_proto_base64) {
  return RegisterModel(base64::Decode(serialized_proto_base64));
}

struct EncodeStringResult {
  std::vector<int> values_flat;
  std::vector<int> splits_flat;
};

// REQUIRES:
// - Attr `out_type`: int32 type
// - Attr `Tsplits`: int64/int32 type
// - Attr `return_nbest`: false
// - Input `nbest_size`: [0]
// - Input `alpha`: 1
EncodeStringResult EncodeString(
    const std::string& model_key,
    const std::vector<std::string>& input_values_flat, bool add_bos,
    bool add_eos, bool reverse) {
  std::unique_ptr<SentencePieceModel>& model = sp_model_pool.at(model_key);
  model->SetOptions(add_bos, add_eos, reverse);

  std::vector<std::vector<int>> tokens(input_values_flat.size());
  for (size_t i = 0; i < input_values_flat.size(); ++i) {
    AssertOk(model->processor.Encode(input_values_flat[i], &tokens[i]));
  }

  int total_tokens = 0;
  for (const auto& tokens_row : tokens) {
    total_tokens += tokens_row.size();
  }
  int splits_size = tokens.size() + 1;
  EncodeStringResult result{
      .values_flat = std::vector<int>(total_tokens),
      .splits_flat = std::vector<int>(splits_size),
  };

  result.splits_flat[0] = 0;
  for (int i = 0, row = 0; row < tokens.size(); ++row) {
    for (int col = 0; col < tokens[row].size(); ++col, ++i) {
      result.values_flat[i] = tokens[row][col];
    }
    result.splits_flat[row + 1] = i;
  }
  return result;
}

// REQUIRES:
// - Attr `out_type`: int32 type
// - Attr `Tsplits`: int64/int32 type
std::vector<std::string> DecodeString(const std::string& model_key,
                                      const std::vector<int>& values_flat,
                                      const std::vector<int>& splits_flat,
                                      bool add_bos, bool add_eos,
                                      bool reverse) {
  std::unique_ptr<SentencePieceModel>& model = sp_model_pool.at(model_key);
  model->SetOptions(add_bos, add_eos, reverse);

  int num_of_sentences = static_cast<int>(splits_flat.size()) - 1;

  std::vector<std::string> output_flat;
  for (int i = 0; i < num_of_sentences; ++i) {
    std::vector<int> pieces(&values_flat[splits_flat[i]],
                            &values_flat[splits_flat[i + 1]]);
    std::string output_flat_str;
    AssertOk(model->processor.Decode(pieces, &output_flat_str));
    output_flat.push_back(std::move(output_flat_str));
  }
  return output_flat;
}

}  // namespace

EMSCRIPTEN_BINDINGS(tfjs_sentencepiece) {
  emscripten::register_vector<int>("VectorInt");
  emscripten::register_vector<std::string>("VectorString");

  emscripten::value_object<EncodeStringResult>("EncodeStringResult")
      .field("valuesFlat", &EncodeStringResult::values_flat)
      .field("splitsFlat", &EncodeStringResult::splits_flat);

  emscripten::function("RegisterModel", &RegisterModel);
  emscripten::function("RegisterModelBase64", &RegisterModelBase64);
  emscripten::function("EncodeString", &EncodeString);
  emscripten::function("DecodeString", &DecodeString);
}

}  // namespace tfjs::sentencepiece
