#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <limits>

// ==========================================
// 1. CONFIG STRUCT (llama2.c format)
// ==========================================
struct Config {
    int dim;        // 288
    int hidden_dim; // 768
    int n_layers;   // 6
    int n_heads;    // 6
    int n_kv_heads; // 6
    int vocab_size; // 32000
    int seq_len;    // 256
};

// ==========================================
// 2. FP32 TENSOR
// ==========================================
class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;
    Tensor(std::vector<int> shape) : shape(shape) {
        size_t total_size = 1;
        for (int dim : shape) total_size *= dim;
        data.resize(total_size, 0.0f);
    }

    size_t size() const { return data.size(); }
    inline float& operator()(int r, int c) { return data[r * shape[1] + c]; }
    inline const float& operator()(int r, int c) const { return data[r * shape[1] + c]; }
};

// ==========================================
// 3. INT8 QUANTIZED TENSOR & AVX2 MATMUL
// ==========================================
struct QuantizedTensorINT8 {
    std::vector<int8_t> data;
    std::vector<int> shape; // [out_dim x in_dim]
    float scale;

    QuantizedTensorINT8() = default;

    // llama2.c stores matrices row-major as [out_dim, in_dim].
    // Quantize in that same layout; do NOT transpose the checkpoint weights.
    void transpose_and_quantize(const float* raw_data, int out_dim, int in_dim) {
        shape = {out_dim, in_dim};
        size_t total_size = static_cast<size_t>(out_dim) * in_dim;
        data.resize(total_size);

        float max_val = 0.0f;
        for (size_t i = 0; i < total_size; ++i) {
            max_val = std::max(max_val, std::abs(raw_data[i]));
        }

        scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;

        for (int row = 0; row < out_dim; ++row) {
            for (int col = 0; col < in_dim; ++col) {
                float val = raw_data[static_cast<size_t>(row) * in_dim + col];
                float scaled = std::round(val * inv_scale);
                int clamped = std::clamp(static_cast<int>(scaled), -128, 127);
                data[row * in_dim + col] = static_cast<int8_t>(clamped);
            }
        }
    }

    void quantize_direct(const float* fp32_data, int rows, int cols) {
        shape = {rows, cols};
        size_t total_size = static_cast<size_t>(rows) * cols;
        data.resize(total_size);

        float max_val = 0.0f;
        for (size_t i = 0; i < total_size; ++i) {
            max_val = std::max(max_val, std::abs(fp32_data[i]));
        }

        scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;

        for (size_t i = 0; i < total_size; ++i) {
            float scaled = std::round(fp32_data[i] * inv_scale);
            int clamped = std::clamp(static_cast<int>(scaled), -128, 127);
            data[i] = static_cast<int8_t>(clamped);
        }
    }

    static inline float _mm256_reduce_add_ps(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        return _mm_cvtss_f32(lo);
    }

    static Tensor matmul(const Tensor& A, const QuantizedTensorINT8& B) {
        const int M = A.shape[0];
        const int K = A.shape[1];
        const int N = B.shape[0];

        Tensor C({M, N});
        const float scale = B.scale;

        #pragma omp parallel for collapse(2) schedule(static) if(static_cast<long long>(M) * N >= 4096)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                __m256 accum_vec = _mm256_setzero_ps();
                int k = 0;
                const int8_t* b_row = B.data.data() + static_cast<size_t>(j) * K;
                const float* a_row = A.data.data() + static_cast<size_t>(i) * K;

                for (; k <= K - 8; k += 8) {
                    const __m256 a_vec = _mm256_loadu_ps(a_row + k);
                    int64_t b_raw;
                    std::memcpy(&b_raw, b_row + k, sizeof(b_raw));
                    const __m128i b_int8 = _mm_cvtsi64_si128(b_raw);
                    const __m256i b_int32 = _mm256_cvtepi8_epi32(b_int8);
                    const __m256 b_fp32 = _mm256_cvtepi32_ps(b_int32);
                    accum_vec = _mm256_fmadd_ps(a_vec, b_fp32, accum_vec);
                }

                float dot = _mm256_reduce_add_ps(accum_vec);
                for (; k < K; ++k) dot += a_row[k] * static_cast<float>(b_row[k]);
                C.data[static_cast<size_t>(i) * N + j] = dot * scale;
            }
        }
        return C;
    }
};

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void softmax(float* row, int size) {
    float max_val = row[0];
    for (int i = 1; i < size; ++i) if (row[i] > max_val) max_val = row[i];
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }
    for (int i = 0; i < size; ++i) row[i] /= sum;
}

// ==========================================
// 4. KV CACHE
// ==========================================
struct KVCache {
    int max_seq_len;
    int d_model;
    int current_pos = 0;
    Tensor k_cache, v_cache;

    KVCache() = default;
    KVCache(int max_seq_len, int d_model)
        : max_seq_len(max_seq_len), d_model(d_model),
          k_cache({max_seq_len, d_model}), v_cache({max_seq_len, d_model}) {}

    bool append(const Tensor& new_k, const Tensor& new_v) {
        int num_tokens = new_k.shape[0];
        if (current_pos + num_tokens > max_seq_len) return false;
        for (int i = 0; i < num_tokens; ++i) {
            for (int d = 0; d < d_model; ++d) {
                k_cache(current_pos + i, d) = new_k(i, d);
                v_cache(current_pos + i, d) = new_v(i, d);
            }
        }
        current_pos += num_tokens;
        return true;
    }
};

// ==========================================
// 5. RMSNORM & ROPE
// ==========================================
class RMSNorm {
public:
    int dim;
    float eps;
    std::vector<float> weight;

    RMSNorm(int dim, float eps = 1e-5f) : dim(dim), eps(eps), weight(dim, 1.0f) {}

    Tensor forward(const Tensor& X) const {
        int seq_len = X.shape[0];
        Tensor Y({seq_len, dim});
        for (int i = 0; i < seq_len; ++i) {
            float sum_sq = 0.0f;
            for (int d = 0; d < dim; ++d) sum_sq += X(i, d) * X(i, d);
            float inv_rms = 1.0f / std::sqrt((sum_sq / dim) + eps);
            for (int d = 0; d < dim; ++d) Y(i, d) = X(i, d) * inv_rms * weight[d];
        }
        return Y;
    }
};

void apply_rope(Tensor& vec, int pos, int head_dim) {
    int num_heads = vec.shape[1] / head_dim;
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / std::pow(10000.0f, static_cast<float>(i * 2) / head_dim);
            float val = pos * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);

            int idx0 = h * head_dim + i;
            int idx1 = h * head_dim + i + head_dim / 2;

            float v0 = vec(0, idx0);
            float v1 = vec(0, idx1);

            vec(0, idx0) = v0 * fcr - v1 * fci;
            vec(0, idx1) = v0 * fci + v1 * fcr;
        }
    }
}

// ==========================================
// 6. TRANSFORMER BLOCK & LAYERS
// ==========================================
class QuantizedMultiHeadAttention {
public:
    int d_model, num_heads, head_dim;
    QuantizedTensorINT8 W_q, W_k, W_v, W_o;

    QuantizedMultiHeadAttention(int d_model, int num_heads)
        : d_model(d_model), num_heads(num_heads), head_dim(d_model / num_heads) {}

    Tensor forward(const Tensor& X, KVCache& cache, int start_pos) {
        int seq_len = X.shape[0];

        Tensor Q = QuantizedTensorINT8::matmul(X, W_q);
        Tensor new_K = QuantizedTensorINT8::matmul(X, W_k);
        Tensor new_V = QuantizedTensorINT8::matmul(X, W_v);

        for (int i = 0; i < seq_len; ++i) {
            Tensor q_token({1, d_model}), k_token({1, d_model});
            for (int d = 0; d < d_model; ++d) {
                q_token(0, d) = Q(i, d);
                k_token(0, d) = new_K(i, d);
            }

            apply_rope(q_token, start_pos + i, head_dim);
            apply_rope(k_token, start_pos + i, head_dim);

            for (int d = 0; d < d_model; ++d) {
                Q(i, d) = q_token(0, d);
                new_K(i, d) = k_token(0, d);
            }
        }

        if (!cache.append(new_K, new_V)) throw std::runtime_error("KV cache overflow");

        int total_cached_len = cache.current_pos;
        Tensor output({seq_len, d_model});
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < num_heads; ++h) {
            int head_offset = h * head_dim;
            Tensor scores({seq_len, total_cached_len});

            for (int i = 0; i < seq_len; ++i) {
                int absolute_q_pos = (total_cached_len - seq_len) + i;
                for (int j = 0; j < total_cached_len; ++j) {
                    if (j > absolute_q_pos) { scores(i, j) = -1e9f; continue; }
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        dot += Q(i, head_offset + d) * cache.k_cache(j, head_offset + d);
                    scores(i, j) = dot * scale;
                }
                softmax(&scores.data[i * total_cached_len], total_cached_len);
            }

            for (int i = 0; i < seq_len; ++i) {
                for (int d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (int j = 0; j < total_cached_len; ++j)
                        val += scores(i, j) * cache.v_cache(j, head_offset + d);
                    output(i, head_offset + d) = val;
                }
            }
        }
        return QuantizedTensorINT8::matmul(output, W_o);
    }
};

class QuantizedSwiGLUFFN {
public:
    int d_model, hidden_dim;
    QuantizedTensorINT8 W_gate, W_up, W_down;

    QuantizedSwiGLUFFN(int d_model, int hidden_dim) : d_model(d_model), hidden_dim(hidden_dim) {}

    Tensor forward(const Tensor& X) const {
        int seq_len = X.shape[0];
        Tensor gate = QuantizedTensorINT8::matmul(X, W_gate);
        Tensor up   = QuantizedTensorINT8::matmul(X, W_up);

        Tensor intermediate({seq_len, hidden_dim});
        for (int i = 0; i < seq_len; ++i)
            for (int d = 0; d < hidden_dim; ++d)
                intermediate(i, d) = silu(gate(i, d)) * up(i, d);

        return QuantizedTensorINT8::matmul(intermediate, W_down);
    }
};

class QuantizedTransformerBlock {
public:
    RMSNorm attn_norm;
    QuantizedMultiHeadAttention attention;
    RMSNorm ffn_norm;
    QuantizedSwiGLUFFN ffn;

    QuantizedTransformerBlock(int d_model, int num_heads, int hidden_dim)
        : attn_norm(d_model), attention(d_model, num_heads),
          ffn_norm(d_model), ffn(d_model, hidden_dim) {}

    Tensor forward(const Tensor& X, KVCache& cache, int start_pos) {
        int seq_len = X.shape[0], d_model = X.shape[1];
        Tensor norm_x1 = attn_norm.forward(X);
        Tensor attn_out = attention.forward(norm_x1, cache, start_pos);

        Tensor x_res1({seq_len, d_model});
        for (size_t i = 0; i < X.size(); ++i) x_res1.data[i] = X.data[i] + attn_out.data[i];

        Tensor norm_x2 = ffn_norm.forward(x_res1);
        Tensor ffn_out = ffn.forward(norm_x2);

        Tensor x_out({seq_len, d_model});
        for (size_t i = 0; i < x_res1.size(); ++i) x_out.data[i] = x_res1.data[i] + ffn_out.data[i];

        return x_out;
    }
};

// ==========================================
// 7. BINARY TOKENIZER PARSER
// ==========================================
class Tokenizer {
public:
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;

    bool load(const std::string& path, int vocab_size) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        uint32_t max_token_len = 0;
        if (!file.read(reinterpret_cast<char*>(&max_token_len), sizeof(max_token_len)) ||
            vocab_size <= 0) return false;

        vocab.clear(); vocab_scores.clear();
        vocab.reserve(vocab_size); vocab_scores.reserve(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            float score = 0.0f;
            int32_t len = 0;
            if (!file.read(reinterpret_cast<char*>(&score), sizeof(score)) ||
                !file.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
            if (len < 0 || static_cast<uint32_t>(len) > max_token_len) return false;
            std::string token(static_cast<size_t>(len), '\0');
            if (len > 0 && !file.read(token.data(), len)) return false;
            vocab_scores.push_back(score);
            vocab.push_back(std::move(token));
        }
        return true;
    }

    std::string decode(int token_id) const {
        if (token_id < 0 || token_id >= static_cast<int>(vocab.size())) return "";
        std::string s = vocab[token_id];
        if (s.rfind("\xE2\x96\x81", 0) == 0) {
            s = " " + s.substr(3);
        }
        return s;
    }
};

// ==========================================
// 8. MODEL LOADER & ENGINE
// ==========================================
class Stories15MEngine {
public:
    Config config;
    int actual_vocab_size;
    bool shared_weights;
    Tensor tok_embeddings;
    std::vector<QuantizedTransformerBlock> layers;
    RMSNorm final_norm;
    QuantizedTensorINT8 lm_head;
    std::vector<KVCache> caches;

    Stories15MEngine() : final_norm(1) {}

    bool load_model(const std::string& model_path) {
        std::ifstream file(model_path, std::ios::binary);
        if (!file.is_open()) return false;

        if (!file.read(reinterpret_cast<char*>(&config), sizeof(Config))) return false;
        
        actual_vocab_size = std::abs(config.vocab_size);
        shared_weights = config.vocab_size > 0;

        auto read_exact = [&](char* dst, size_t bytes) -> bool {
            file.read(dst, static_cast<std::streamsize>(bytes));
            return file.good();
        };

        std::cout << "Loading Model Architecture: Dim=" << config.dim 
                  << " Layers=" << config.n_layers 
                  << " Heads=" << config.n_heads 
                  << " Vocab=" << actual_vocab_size 
                  << " SharedWeights=" << (shared_weights ? "Yes" : "No") << "\n";

        final_norm = RMSNorm(config.dim);
        tok_embeddings = Tensor({actual_vocab_size, config.dim});
        if (!read_exact(reinterpret_cast<char*>(tok_embeddings.data.data()), tok_embeddings.size() * sizeof(float))) return false;

        auto read_fp32_vec = [&](size_t size) {
            std::vector<float> buf(size);
            if (!read_exact(reinterpret_cast<char*>(buf.data()), size * sizeof(float)))
                throw std::runtime_error("Truncated file read");
            return buf;
        };

        for (int l = 0; l < config.n_layers; ++l) {
            layers.emplace_back(config.dim, config.n_heads, config.hidden_dim);
            caches.emplace_back(config.seq_len, config.dim);
            auto& layer = layers.back();

            if (!read_exact(reinterpret_cast<char*>(layer.attn_norm.weight.data()), config.dim * sizeof(float))) return false;

            auto q_w = read_fp32_vec(static_cast<size_t>(config.dim) * config.dim);
            layer.attention.W_q.transpose_and_quantize(q_w.data(), config.dim, config.dim);

            auto k_w = read_fp32_vec(static_cast<size_t>(config.dim) * config.dim);
            layer.attention.W_k.transpose_and_quantize(k_w.data(), config.dim, config.dim);

            auto v_w = read_fp32_vec(static_cast<size_t>(config.dim) * config.dim);
            layer.attention.W_v.transpose_and_quantize(v_w.data(), config.dim, config.dim);

            auto o_w = read_fp32_vec(static_cast<size_t>(config.dim) * config.dim);
            layer.attention.W_o.transpose_and_quantize(o_w.data(), config.dim, config.dim);

            if (!read_exact(reinterpret_cast<char*>(layer.ffn_norm.weight.data()), config.dim * sizeof(float))) return false;

            auto gate_w = read_fp32_vec(static_cast<size_t>(config.hidden_dim) * config.dim);
            auto down_w = read_fp32_vec(static_cast<size_t>(config.dim) * config.hidden_dim);
            auto up_w   = read_fp32_vec(static_cast<size_t>(config.hidden_dim) * config.dim);

            layer.ffn.W_gate.transpose_and_quantize(gate_w.data(), config.hidden_dim, config.dim);
            layer.ffn.W_down.transpose_and_quantize(down_w.data(), config.dim, config.hidden_dim);
            layer.ffn.W_up.transpose_and_quantize(up_w.data(), config.hidden_dim, config.dim);
        }

        if (!read_exact(reinterpret_cast<char*>(final_norm.weight.data()), config.dim * sizeof(float))) return false;

        // The standard llama2.c checkpoint does not store RoPE tables.
        // Therefore, do not seek past bytes here: an untied output head follows
        // final_norm immediately in that format.
        if (!shared_weights) {
            auto lm_head_w = read_fp32_vec(static_cast<size_t>(actual_vocab_size) * config.dim);
            lm_head.transpose_and_quantize(lm_head_w.data(), actual_vocab_size, config.dim);
        } else {
            lm_head.quantize_direct(tok_embeddings.data.data(), actual_vocab_size, config.dim);
        }

        std::cout << "Model loaded and INT8 quantized successfully!\n";
        return true;
    }

    std::vector<float> forward(const std::vector<int>& token_ids, int start_pos) {
        int seq_len = static_cast<int>(token_ids.size());
        Tensor X({seq_len, config.dim});
        for (int i = 0; i < seq_len; ++i) {
            int token_id = token_ids[i];
            for (int d = 0; d < config.dim; ++d) X(i, d) = tok_embeddings(token_id, d);
        }

        for (int l = 0; l < config.n_layers; ++l) {
            X = layers[l].forward(X, caches[l], start_pos);
        }

        X = final_norm.forward(X);

        Tensor last_hidden({1, config.dim});
        for (int d = 0; d < config.dim; ++d) last_hidden(0, d) = X(seq_len - 1, d);

        Tensor logits_tensor = QuantizedTensorINT8::matmul(last_hidden, lm_head);
        return logits_tensor.data;
    }
};

// ==========================================
// 9. MAIN EXECUTION
// ==========================================
int main() {
    try {
        Tokenizer tokenizer;
        Stories15MEngine model;

        std::cout << "==========================================================" << std::endl;
        std::cout << " INT8 QUANTIZED TINY STORIES ENGINE (stories15M.bin)     " << std::endl;
        std::cout << "==========================================================" << std::endl;

        if (!model.load_model("stories15M.bin")) {
            std::cerr << "Ensure 'stories15M.bin' is in the working directory.\n";
            return 1;
        }

        if (!tokenizer.load("tokenizer.bin", model.actual_vocab_size)) {
            std::cerr << "Ensure 'tokenizer.bin' is in the working directory.\n";
            return 1;
        }

        // Canonical LLaMA-2 tokenizer IDs for: "<s> Once upon a time"
        // This assumes tokenizer.bin uses the standard LLaMA-2 32k vocabulary.
        std::vector<int> prompt = {1, 9038, 526, 263, 931}; 
        int max_tokens = 40;
        int pos = 0;
        int generated_tokens = 0;

        std::cout << "\nPrompt: ";
        for (int t : prompt) std::cout << tokenizer.decode(t);
        std::cout << "\nGenerated: " << std::flush;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Prefill Phase
        std::vector<float> logits = model.forward(prompt, pos);
        pos += prompt.size();

        int next_token = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        if (next_token != 1 && next_token != 2) {
            std::cout << tokenizer.decode(next_token) << std::flush;
            ++generated_tokens;
        }

        // Autoregressive Decoding Loop
        while (generated_tokens < max_tokens && pos < model.config.seq_len) {
            if (next_token == 1 || next_token == 2) break;
            logits = model.forward({next_token}, pos);
            ++pos;
            next_token = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
            if (next_token == 1 || next_token == 2) break;
            std::cout << tokenizer.decode(next_token) << std::flush;
            ++generated_tokens;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "\n\n==========================================================" << std::endl;
        std::cout << "Inference completed in " << duration << "s (";
        std::cout << (duration > 0.0 ? generated_tokens / duration : 0.0) << " generated tok/s; "
                  << generated_tokens << " tokens)\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal inference error: " << e.what() << "\n";
        return 1;
    }
}