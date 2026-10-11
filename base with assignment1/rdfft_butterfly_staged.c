/* ========================================================================
 * rdfft_butterfly_staged.c — 共轭对蝶形 (v2: 逐级连续旋转因子表)
 *
 * 与原 rdfft_butterfly_compute.c 完全相同的共轭对蝶形公式与半复数布局,
 * 唯一区别: 旋转因子改为逐级连续表 (generate_rdfft_twiddle_factors_staged),
 * 阶段内偏移 = M/2 - 1, 蝶形内 vle32 连续加载替代 vlse32 strided 加载。
 * ======================================================================== */
#include <riscv_vector.h>

void rdfft_butterfly_staged(float* Y, const float* W_re, const float* W_im,
                            int N, int half_step, int base) {

    int M = half_step;

    /* Case 1: k = 0 — DC 蝶形 (纯实数加减) */
    {
        float E_re = Y[base];
        float O_re = Y[base + M];

        Y[base]     = E_re + O_re;
        Y[base + M] = E_re - O_re;
    }

    /* Case 2: k = M/2 — Nyquist 蝶形 (仅 M >= 2) */
    if (M >= 2) {
        int k = M / 2;

        float E_re = Y[base + k];
        float O_re = Y[base + M + k];

        Y[base + k]         = E_re;
        Y[base + 2 * M - k] = -O_re;
    }

    /* Case 3: k = 1 .. M/2-1 — 常规共轭对蝶形 (RVV 向量化) */
    int reg_start = 1;
    int reg_limit = M / 2;

    if (reg_start < reg_limit) {
        /* 逐级表的本阶段起始偏移 (各阶段表长前缀和 = M/2 - 1) */
        const float* Wr = W_re + (M / 2 - 1);
        const float* Wi = W_im + (M / 2 - 1);

        int k_idx = reg_start;

        while (k_idx < reg_limit) {
            size_t vl = __riscv_vsetvl_e32m1(reg_limit - k_idx);

            /* --- 4 路数据拉取 --- */
            vfloat32m1_t v_E_re = __riscv_vle32_v_f32m1(&Y[base + k_idx], vl);

            float* E_im_ptr = &Y[base + M - k_idx - (int)vl + 1];
            vfloat32m1_t v_E_im_raw = __riscv_vle32_v_f32m1(E_im_ptr, vl);

            vfloat32m1_t v_O_re = __riscv_vle32_v_f32m1(&Y[base + M + k_idx], vl);

            float* O_im_ptr = &Y[base + 2 * M - k_idx - (int)vl + 1];
            vfloat32m1_t v_O_im_raw = __riscv_vle32_v_f32m1(O_im_ptr, vl);

            /* --- 倒序翻转虚部 --- */
            vuint32m1_t v_idx     = __riscv_vid_v_u32m1(vl);
            vuint32m1_t v_rev_idx = __riscv_vrsub_vx_u32m1(v_idx, vl - 1, vl);

            vfloat32m1_t v_E_im = __riscv_vrgather_vv_f32m1(v_E_im_raw, v_rev_idx, vl);
            vfloat32m1_t v_O_im = __riscv_vrgather_vv_f32m1(v_O_im_raw, v_rev_idx, vl);

            /* --- 旋转因子: 逐级表连续加载 (v2 优化, 替代 vlse32) --- */
            vfloat32m1_t v_W_re = __riscv_vle32_v_f32m1(&Wr[k_idx], vl);
            vfloat32m1_t v_W_im = __riscv_vle32_v_f32m1(&Wi[k_idx], vl);

            /* --- T = W * O (复数乘法) --- */
            vfloat32m1_t v_T_re = __riscv_vfmul_vv_f32m1(v_W_re, v_O_re, vl);
            v_T_re = __riscv_vfnmsac_vv_f32m1(v_T_re, v_W_im, v_O_im, vl);

            vfloat32m1_t v_T_im = __riscv_vfmul_vv_f32m1(v_W_re, v_O_im, vl);
            v_T_im = __riscv_vfmacc_vv_f32m1(v_T_im, v_W_im, v_O_re, vl);

            /* --- 共轭对蝶形输出: 4 组公式 --- */
            vfloat32m1_t v_out_1 = __riscv_vfadd_vv_f32m1(v_E_re, v_T_re, vl);
            vfloat32m1_t v_out_2 = __riscv_vfadd_vv_f32m1(v_E_im, v_T_im, vl);
            vfloat32m1_t v_out_3 = __riscv_vfsub_vv_f32m1(v_E_re, v_T_re, vl);
            vfloat32m1_t v_out_4 = __riscv_vfsub_vv_f32m1(v_T_im, v_E_im, vl);

            /* --- 4 路写回 (2 前向 + 2 倒序) --- */
            __riscv_vse32_v_f32m1(&Y[base + k_idx], v_out_1, vl);

            vfloat32m1_t v_out_3_rev = __riscv_vrgather_vv_f32m1(v_out_3, v_rev_idx, vl);
            __riscv_vse32_v_f32m1(&Y[base + M - k_idx - (int)vl + 1], v_out_3_rev, vl);

            __riscv_vse32_v_f32m1(&Y[base + M + k_idx], v_out_4, vl);

            vfloat32m1_t v_out_2_rev = __riscv_vrgather_vv_f32m1(v_out_2, v_rev_idx, vl);
            __riscv_vse32_v_f32m1(&Y[base + 2 * M - k_idx - (int)vl + 1], v_out_2_rev, vl);

            k_idx += (int)vl;
        }
    }
}
