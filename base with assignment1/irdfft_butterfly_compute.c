/* ========================================================================
 * irdfft_butterfly_compute.c — 逆向 irdFFT 共轭对蝶形算子 (RVV, DIF 结构)
 *
 * 与正变换 rdfft_butterfly_compute.c (DIT) 逐级互逆：
 *   正变换: 位反转(时域实数) -> 蝶形 M 从小到大, 逐级构建半复数配对
 *   逆变换: 蝶形 M 从大到小(消耗上一级的配对布局) -> 位反转 -> 除以 N
 *
 * 为什么不用任务书"先位反转+共轭旋转因子"的 DIT 写法:
 *   位反转会打乱半复数频谱的共轭对布局, 使蝶形读到的 E/O 不再是合法配对,
 *   结果整体错误(已在 N=4 上手工推演证实)。DIF 结构才是该隐式编码的
 *   正确逆运算, 与纯标量版 irdfft-main.c 的结构一致, 两者互为交叉验证。
 *
 * 核心公式 (含每级放大 2 倍, log2(N) 级共放大 N 倍, 最后统一除以 N):
 *   2*E[k] = X[k] + conj(X[M-k])
 *   2*O[k] = (X[k] - conj(X[M-k])) * conj(W_{2M}^k)
 * 其中 conj(W) 由与正变换共用的 twiddle 表取共轭得到:
 *   conj(W) = W_re - j*W_im
 *
 * 半复数布局 (组 [base, base+2M) 内):
 *   Re(X[k]) -> Y[base+k],      Im(X[k]) -> Y[base+2M-k]
 *   Re(X[M-k]) -> Y[base+M-k],  Im(X[M-k]) -> Y[base+M+k]
 * 输出 E 块 [base, base+M): Re@base+k, Im@base+M-k
 * 输出 O 块 [base+M, base+2M): Re@base+M+k, Im@base+2M-k
 *
 * @param Y         数据数组 (N 个 float, 半复数格式, 原地计算)
 * @param W_re      逐级连续旋转因子表 (与正变换共用, 存的是正变换因子,
 *                  本阶段偏移 = M/2 - 1, 蝶形内连续 vle32 加载)
 * @param W_im      旋转因子虚部表 (同上)
 * @param N         总点数
 * @param half_step 当前蝶形跨度 M
 * @param base      当前组起始索引
 * ======================================================================== */
#include <riscv_vector.h>

void irdfft_butterfly_compute(float* Y, const float* W_re, const float* W_im,
                              int N, int half_step, int base) {

    int M = half_step;
    (void)N;   /* N 仅在旧版 strided 表寻址中使用, v2 逐级表不再需要 */

    /* ==========================================================
     * Case 1: k = 0 — DC (W=1)
     * 2*E[0] = X[0] + X[M],  2*O[0] = X[0] - X[M]
     * ========================================================== */
    {
        float X0 = Y[base];
        float XM = Y[base + M];

        Y[base]     = X0 + XM;
        Y[base + M] = X0 - XM;
    }

    /* ==========================================================
     * Case 2: k = M/2 — Nyquist (仅 M >= 2)
     * 2*E[M/2] = 2*Re(X[M/2]),  2*O[M/2] = -2*Im(X[M/2])
     * (正变换此处置 -O_re, 故这里 O = -Im 与之互逆)
     * ========================================================== */
    if (M >= 2) {
        int k = M / 2;

        float X_re = Y[base + k];
        float X_im = Y[base + 2 * M - k];

        Y[base + k]         = 2.0f * X_re;
        Y[base + 2 * M - k] = -2.0f * X_im;
    }

    /* ==========================================================
     * Case 3: k = 1 .. M/2-1 — 常规共轭对蝶形 (RVV 向量化)
     * ========================================================== */
    int reg_start = 1;
    int reg_limit = M / 2;

    if (reg_start < reg_limit) {
        int k_idx = reg_start;

        while (k_idx < reg_limit) {
            size_t vl = __riscv_vsetvl_e32m1(reg_limit - k_idx);

            /* --- 4 路数据拉取 (先全部加载, 再写回, 保证原地正确) ---
             * a = Re(X[k])  正序;  d = Im(X[M-k]) 正序
             * b = Im(X[k])  倒序;  c = Re(X[M-k]) 倒序 */
            vfloat32m1_t v_a = __riscv_vle32_v_f32m1(&Y[base + k_idx], vl);
            vfloat32m1_t v_d = __riscv_vle32_v_f32m1(&Y[base + M + k_idx], vl);

            float* b_ptr = &Y[base + 2 * M - k_idx - (int)vl + 1];
            vfloat32m1_t v_b_raw = __riscv_vle32_v_f32m1(b_ptr, vl);

            float* c_ptr = &Y[base + M - k_idx - (int)vl + 1];
            vfloat32m1_t v_c_raw = __riscv_vle32_v_f32m1(c_ptr, vl);

            /* --- 倒序翻转 b, c 与正序的 a, d 对齐 --- */
            vuint32m1_t v_idx     = __riscv_vid_v_u32m1(vl);
            vuint32m1_t v_rev_idx = __riscv_vrsub_vx_u32m1(v_idx, vl - 1, vl);

            vfloat32m1_t v_b = __riscv_vrgather_vv_f32m1(v_b_raw, v_rev_idx, vl);
            vfloat32m1_t v_c = __riscv_vrgather_vv_f32m1(v_c_raw, v_rev_idx, vl);

            /* --- 旋转因子: 逐级连续表 (v2 优化, 阶段偏移 = M/2-1, 替代 vlse32) --- */
            vfloat32m1_t v_W_re = __riscv_vle32_v_f32m1(&W_re[(M / 2 - 1) + k_idx], vl);
            vfloat32m1_t v_W_im = __riscv_vle32_v_f32m1(&W_im[(M / 2 - 1) + k_idx], vl);

            /* --- 2*E[k] = X[k] + conj(X[M-k]) = (a+c) + j(b-d) --- */
            vfloat32m1_t v_ek_re = __riscv_vfadd_vv_f32m1(v_a, v_c, vl);
            vfloat32m1_t v_ek_im = __riscv_vfsub_vv_f32m1(v_b, v_d, vl);

            /* --- 2*O[k] = (X[k] - conj(X[M-k])) * conj(W)
             * diff = (a-c) + j(b+d), conj(W) = W_re - j*W_im
             * re = diff_re*W_re + diff_im*W_im  -> vfmacc
             * im = diff_im*W_re - diff_re*W_im  -> vfnmsac
             * 注意 RVV 语义: vfmsac 是 vd = vs1*vs2 - vd, 会得到相反符号,
             * 所以这里必须用 vfnmsac (vd = -vs1*vs2 + vd)。 */
            vfloat32m1_t v_diff_re = __riscv_vfsub_vv_f32m1(v_a, v_c, vl);
            vfloat32m1_t v_diff_im = __riscv_vfadd_vv_f32m1(v_b, v_d, vl);

            vfloat32m1_t v_ok_re = __riscv_vfmul_vv_f32m1(v_W_re, v_diff_re, vl);
            v_ok_re = __riscv_vfmacc_vv_f32m1(v_ok_re, v_W_im, v_diff_im, vl);

            vfloat32m1_t v_ok_im = __riscv_vfmul_vv_f32m1(v_W_re, v_diff_im, vl);
            v_ok_im = __riscv_vfnmsac_vv_f32m1(v_ok_im, v_W_im, v_diff_re, vl);

            /* --- 4 路写回 ---
             * E 块: Re 正序 @base+k,  Im 倒序 @base+M-k
             * O 块: Re 正序 @base+M+k, Im 倒序 @base+2M-k */
            __riscv_vse32_v_f32m1(&Y[base + k_idx], v_ek_re, vl);

            vfloat32m1_t v_ek_im_rev = __riscv_vrgather_vv_f32m1(v_ek_im, v_rev_idx, vl);
            __riscv_vse32_v_f32m1(&Y[base + M - k_idx - (int)vl + 1], v_ek_im_rev, vl);

            __riscv_vse32_v_f32m1(&Y[base + M + k_idx], v_ok_re, vl);

            vfloat32m1_t v_ok_im_rev = __riscv_vrgather_vv_f32m1(v_ok_im, v_rev_idx, vl);
            __riscv_vse32_v_f32m1(&Y[base + 2 * M - k_idx - (int)vl + 1], v_ok_im_rev, vl);

            k_idx += (int)vl;
        }
    }
}

