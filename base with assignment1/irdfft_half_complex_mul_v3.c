/* ========================================================================
 * irdfft_half_complex_mul_v3.c — 半复数频域乘 (RVV, v3 原地版)
 *
 * 计算 B = A .* B (原地写回 B, 省去输出缓冲 C):
 *   - 每对 k / N-k 的 4 个输入先全部加载, 2 个结果算完后才写回,
 *     写回地址与读取地址重叠但同迭代内先读后写, 原地安全
 *   - 输出保持隐式共轭对称布局 (Re@k, Im@N-k), irdFFT 可直接解析
 *   - LMUL 提升到 m2 (v3 微调)
 *
 * 数学与 irdfft_half_complex_mul.c (A, B, C 三缓冲版) 逐位一致:
 *   reZ = A_re*B_re - A_im*B_im
 *   imZ = A_re*B_im + A_im*B_re
 * ======================================================================== */
#include <riscv_vector.h>

void half_complex_mul_v3(const float *A, float *B, int N) {
    int n_half = N / 2;

    /* DC 与 Nyquist: 纯实数标量乘法, 原地安全 */
    B[0]      = A[0] * B[0];
    B[n_half] = A[n_half] * B[n_half];

    int k_idx = 1;
    while (k_idx < n_half) {
        size_t vl = __riscv_vsetvl_e32m2(n_half - k_idx);

        /* 1. 先全部加载 (4 路), 保证原地正确 */
        vfloat32m2_t vA_re = __riscv_vle32_v_f32m2(&A[k_idx], vl);
        vfloat32m2_t vB_re = __riscv_vle32_v_f32m2(&B[k_idx], vl);

        int base_im = N - k_idx - (int)vl + 1;
        vfloat32m2_t vA_im_raw = __riscv_vle32_v_f32m2(&A[base_im], vl);
        vfloat32m2_t vB_im_raw = __riscv_vle32_v_f32m2(&B[base_im], vl);

        /* 2. 寄存器内镜像翻转虚部, 与实部对齐 */
        vuint32m2_t v_idx     = __riscv_vid_v_u32m2(vl);
        vuint32m2_t v_rev_idx = __riscv_vrsub_vx_u32m2(v_idx, vl - 1, vl);

        vfloat32m2_t vA_im = __riscv_vrgather_vv_f32m2(vA_im_raw, v_rev_idx, vl);
        vfloat32m2_t vB_im = __riscv_vrgather_vv_f32m2(vB_im_raw, v_rev_idx, vl);

        /* 3. 复数乘法 */
        vfloat32m2_t v_reZ = __riscv_vfmul_vv_f32m2(vA_re, vB_re, vl);
        v_reZ = __riscv_vfnmsac_vv_f32m2(v_reZ, vA_im, vB_im, vl);

        vfloat32m2_t v_imZ = __riscv_vfmul_vv_f32m2(vA_re, vB_im, vl);
        v_imZ = __riscv_vfmacc_vv_f32m2(v_imZ, vA_im, vB_re, vl);

        /* 4. 写回 B (实部正序, 虚部镜像) */
        __riscv_vse32_v_f32m2(&B[k_idx], v_reZ, vl);

        vfloat32m2_t v_imZ_rev = __riscv_vrgather_vv_f32m2(v_imZ, v_rev_idx, vl);
        __riscv_vse32_v_f32m2(&B[base_im], v_imZ_rev, vl);

        k_idx += (int)vl;
    }
}
