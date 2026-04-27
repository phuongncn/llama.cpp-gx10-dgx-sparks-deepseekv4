#include "models.h"

// DeepSeek V4 Flash forward pass.
//
// Architecture highlights (all 43 layers):
//   Attention (non-absorbed GQA, n_kv_heads=1):
//     Q: wq_a [n_embd→1024] → q_norm → wq_b [1024→n_head*512=32768]
//     K: wkv  [n_embd→512] → kv_norm → c_kv[512] (K_nope=[448], K_rope=[64])
//     V: c_kv[:64] (first 64 dims of the normed KV latent)
//     Output: wo_a [n_head*v_head=4096→8192] → wo_b [8192→4096]
//   FFN: shared SwiGLU (n_embd=4096 → n_ff_exp=2048 → n_embd)
//        + top-6 routed MoE experts operating on shared intermediate (2048-dim input):
//          w1 (combined gate+up) [2048→2048, split at 1024] → silu(gate)*up=[1024] → w2 [1024→4096]

llm_build_deepseekv4flash::llm_build_deepseekv4flash(
        const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head_k   = hparams.n_embd_head_k_full; // 512
    const int64_t n_embd_head_v   = hparams.n_embd_head_v_full; // 64
    const int64_t n_embd_head_qk_rope = hparams.n_rot();        // 64
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope; // 448

    const int64_t kv_lora_rank = hparams.n_lora_kv; // 512 = c_kv latent dim
    const int64_t n_ff_exp     = hparams.n_ff_exp;  // 2048 = shared expert intermediate dim
    const int64_t n_ff_down    = n_ff_exp / 2;       // 1024 = routed expert intermediate dim (after combined split)
    const int64_t n_expert     = hparams.n_expert;  // 256
    const int64_t n_expert_used = hparams.n_expert_used; // 6

    // kq_scale: 1/sqrt(head_dim) with YaRN adjustments (same as DeepSeek V2)
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    // standard KV-cache (not MLA-absorbed)
    auto * inp_attn_kv = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // ── pre-attention RMS norm ───────────────────────────────────────
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // ── Query path: Q = wq_b(q_norm(wq_a(x))) ───────────────────────
        ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
        cb(q, "q_lora", il);
        q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q, "q_lora_norm", il);
        q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
        cb(q, "q_full", il);

        // reshape: [n_head*n_embd_head_k, n_tokens] → [n_embd_head_k, n_head, n_tokens]
        q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);

        // split Q into nope and rope parts
        ggml_tensor * q_nope = ggml_view_3d(
            ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
        cb(q_nope, "q_nope", il);

        ggml_tensor * q_pe = ggml_view_3d(
            ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));
        cb(q_pe, "q_pe", il);

        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "q_pe_rope", il);

        // Qcur: [n_embd_head_k, n_head, n_tokens]
        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(Qcur, "Qcur", il);

        // ── KV path: c_kv = kv_norm(wkv(x)) [kv_lora_rank, n_tokens] ───
        ggml_tensor * c_kv = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
        cb(c_kv, "c_kv_raw", il);
        c_kv = build_norm(c_kv, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(c_kv, "c_kv", il);

        // K: c_kv split as k_nope=[448] + k_rope_raw=[64]
        ggml_tensor * k_nope = ggml_view_3d(
            ctx0, c_kv, n_embd_head_qk_nope, 1, n_tokens,
            ggml_row_size(c_kv->type, kv_lora_rank),
            ggml_row_size(c_kv->type, kv_lora_rank), 0);
        cb(k_nope, "k_nope", il);

        ggml_tensor * k_rope_raw = ggml_view_3d(
            ctx0, c_kv, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(c_kv->type, kv_lora_rank),
            ggml_row_size(c_kv->type, kv_lora_rank),
            ggml_row_size(c_kv->type, n_embd_head_qk_nope));
        cb(k_rope_raw, "k_rope_raw", il);

        // Single-head K with rope applied, for KV cache
        ggml_tensor * k_nope_1h = ggml_cont(ctx0, k_nope);
        ggml_tensor * k_rope_1h = ggml_cont(ctx0, k_rope_raw);
        k_rope_1h = ggml_rope_ext(ctx0, k_rope_1h, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        ggml_tensor * Kcur = ggml_concat(ctx0, k_nope_1h, k_rope_1h, 0); // [512, 1, n_tokens]
        cb(Kcur, "Kcur", il);

        // V = c_kv[:n_embd_head_v] = first 64 dims of KV latent, single KV head
        ggml_tensor * Vcur = ggml_view_3d(
            ctx0, c_kv, n_embd_head_v, 1, n_tokens,
            ggml_row_size(c_kv->type, kv_lora_rank),
            ggml_row_size(c_kv->type, kv_lora_rank), 0);
        Vcur = ggml_cont(ctx0, Vcur);
        cb(Vcur, "Vcur", il);

        // ── Attention (GQA: 64 Q heads, 1 KV head) ──────────────────────
        // wo_a stored in layer.wo, wo_b stored in layer.wv_b
        cur = build_attn(inp_attn_kv,
            model.layers[il].wo,   // wo_a [n_head*v_head=4096 → 2*n_embd=8192]
            nullptr,
            Qcur, Kcur, Vcur,
            nullptr, nullptr, nullptr,
            kq_scale, il);
        cb(cur, "attn_out_a", il);

        // Second half of factored output: wo_b [2*n_embd=8192 → n_embd=4096]
        cur = ggml_mul_mat(ctx0, model.layers[il].wv_b, cur);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // ── Residual + FFN ───────────────────────────────────────────────
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // At the last layer inp_out_ids may reduce n_tokens → n_outputs; use actual dim.
        const int64_t n_tok = cur->ne[1];

        // ── Shared expert (SwiGLU on full n_embd=4096 input) ────────────
        ggml_tensor * sh_gate = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_shexp, cur);
        sh_gate = ggml_silu(ctx0, sh_gate);              // [n_ff_exp, n_tokens]
        ggml_tensor * sh_up = ggml_mul_mat(ctx0, model.layers[il].ffn_up_shexp, cur);
        ggml_tensor * shared_inter = ggml_mul(ctx0, sh_gate, sh_up); // [n_ff_exp=2048, n_tokens]
        cb(shared_inter, "ffn_sh_inter", il);
        ggml_tensor * shared_out = ggml_mul_mat(ctx0, model.layers[il].ffn_down_shexp, shared_inter);
        cb(shared_out, "ffn_sh_out", il);

        ggml_tensor * ffn_out;
        if (model.layers[il].ffn_gate_exps) {
            // ── Routed MoE experts on shared_inter (2048-dim input) ─────
            // Routing: uses the full 4096-dim FFN norm output
            ggml_tensor * router_logits = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_inp, cur);
            cb(router_logits, "ffn_moe_logits", il);
            router_logits = ggml_soft_max(ctx0, router_logits); // [n_expert=256, n_tokens]
            cb(router_logits, "ffn_moe_probs", il);

            ggml_tensor * selected = ggml_argsort_top_k(ctx0, router_logits, n_expert_used);
            cb(selected, "ffn_moe_topk", il); // [6, n_tokens]

            // Expert weights (normalized)
            ggml_tensor * probs_3d = ggml_reshape_3d(ctx0, router_logits, 1, n_expert, n_tok);
            ggml_tensor * weights = ggml_get_rows(ctx0, probs_3d, selected); // [1, 6, n_tok]
            cb(weights, "ffn_moe_weights", il);

            if (hparams.expert_weights_norm) {
                weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tok);
                ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
                weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5f, INFINITY);
                weights = ggml_div(ctx0, weights, weights_sum);
                weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tok);
                cb(weights, "ffn_moe_weights_norm", il);
            }
            if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
                weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
                cb(weights, "ffn_moe_weights_scaled", il);
            }

            ggml_build_forward_expand(gf, weights);

            // Expert computation: input is shared_inter [n_ff_exp=2048, n_tokens]
            // w1 (ffn_gate_exps) is stored as combined gate+up: output [2048] = [gate:1024, up:1024]
            ggml_tensor * expert_in = ggml_reshape_3d(ctx0, shared_inter, n_ff_exp, 1, n_tok);

            ggml_tensor * gate_up = ggml_mul_mat_id(ctx0, model.layers[il].ffn_gate_exps, expert_in, selected);
            cb(gate_up, "ffn_moe_gate_up", il); // [n_ff_exp=2048, n_expert_used=6, n_tokens]

            // Split combined output at n_ff_down=1024: gate=[:1024], up=[1024:]
            ggml_tensor * moe_gate = ggml_view_3d(ctx0, gate_up,
                n_ff_down, n_expert_used, n_tok,
                gate_up->nb[1], gate_up->nb[2], 0);
            cb(moe_gate, "ffn_moe_gate", il);

            ggml_tensor * moe_up = ggml_view_3d(ctx0, gate_up,
                n_ff_down, n_expert_used, n_tok,
                gate_up->nb[1], gate_up->nb[2],
                (size_t)n_ff_down * gate_up->nb[0]);
            cb(moe_up, "ffn_moe_up", il);

            moe_gate = ggml_silu(ctx0, moe_gate);
            ggml_tensor * moe_inter = ggml_mul(ctx0, moe_gate, moe_up); // [1024, 6, n_tokens]
            moe_inter = ggml_cont(ctx0, moe_inter);
            cb(moe_inter, "ffn_moe_inter", il);

            // Down projection: [n_ff_down=1024 → n_embd=4096]
            ggml_tensor * moe_down = ggml_mul_mat_id(ctx0, model.layers[il].ffn_down_exps, moe_inter, selected);
            cb(moe_down, "ffn_moe_down", il); // [n_embd=4096, 6, n_tokens]

            // Apply expert weights
            moe_down = ggml_mul(ctx0, moe_down, weights); // [4096, 6, n_tokens]
            cb(moe_down, "ffn_moe_weighted", il);

            ggml_build_forward_expand(gf, moe_down);

            // Reduce over experts by summing views
            ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
            for (int e = 0; e < (int)hparams.n_expert_used; ++e) {
                cur_experts[e] = ggml_view_2d(ctx0, moe_down, n_embd, n_tok,
                                              moe_down->nb[2], (size_t)e * moe_down->nb[1]);
                ggml_build_forward_expand(gf, cur_experts[e]);
            }

            ggml_tensor * moe_out = cur_experts[0];
            for (int e = 1; e < (int)hparams.n_expert_used; ++e) {
                moe_out = ggml_add(ctx0, moe_out, cur_experts[e]);
                ggml_build_forward_expand(gf, moe_out);
            }
            cb(moe_out, "ffn_moe_out", il);

            // Total FFN = shared + routed
            ffn_out = ggml_add(ctx0, shared_out, moe_out);
        } else {
            // Phase-1 fallback: shared expert only
            ffn_out = shared_out;
        }
        cb(ffn_out, "ffn_out", il);

        cur = ggml_add(ctx0, ffn_out, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
