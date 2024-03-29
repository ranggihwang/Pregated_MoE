#include "src/fastertransformer/models/t5/T5Encoder.h"
#include "src/fastertransformer/models/t5/T5EncoderWeight.h"
#include <vector>
#include "T5_config.h"

namespace ft = fastertransformer;
using namespace std;

int main() {
    ft::T5EncoderWeight<float> t5_encoder_weights(
        head_num, size_per_head, d_model, inter_size, vocab_size, num_layer,
        num_bucket_or_max_seq_len, tensor_para_size, tensor_para_rank,
        pipeline_para_size, pipeline_para_rank, t5_with_bias_para,
        use_gated_activation_para, pe_type, prompt_learning_type,
        prompt_learning_pair, ia3_num_tasks, adapter_inter_size);
}