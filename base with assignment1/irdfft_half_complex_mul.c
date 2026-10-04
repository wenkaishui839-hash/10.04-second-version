/* ========================================================================
 * irdfft_half_complex_mul.c — 半复数格式逐元素复数乘法 (RVV)
 *
 * 半复数格式 (正序实部、倒序虚部): 
 *   C[0]      = Re(F[0])    直流分量, 纯实数
 *   C[N/2]    = Re(F[N/2])  奈奎斯特分量, 纯实数
 *   C[k]      = Re(F[k])    (k = 1..N/2-1)
 *   C[N-k]    = Im(F[k])    (虚部按频率倒序存储)
 *
 * 计算 Z = A .* B, 用于 rdFFT 频域做圆周卷积
 * (时域卷积与频域逐点相乘等价)。
 *
 * RVV 实现要点:
 *   - DC/Nyquist 特判为标量实数乘法;
 *   - 实部正序连续加载, 虚部按倒序地址整块加载后,
 *     用 vrgather 在寄存器内翻转, 与实部对齐;
 *   - reZ = A_re*B_re - A_im*B_im -> vfnmsac (vd = -vs1*vs2 + vd)
 *     imZ = A_re*B_im + A_im*B_re -> vfmacc  (vd = +vs1*vs2 + vd)
 *   - 虚部结果重新倒序后写回 C[N-k] 位置。
 * ======================================================================== */
#include <riscv_vector.h>

void half_complex_mul_rvv(const float* restrict A, const float* restrict B,
                          float* restrict C, int N) {
    int n_half = N / 2;

    /* ===== 标量特殊频点 k=0 与 k=N/2 ===== */
    C[0]      = A[0] * B[0];
    C[n_half] = A[n_half] * B[n_half];

    int k_idx = 1;
    int reg_limit = n_half;

    while (k_idx < reg_limit) {
        size_t vl = __riscv_vsetvl_e32m1(reg_limit - k_idx);

        /* 1. 正序加载实部 A_re[k], B_re[k] */
        vfloat32m1_t vA_re = __riscv_vle32_v_f32m1(&A[k_idx], vl);
        vfloat32m1_t vB_re = __riscv_vle32_v_f32m1(&B[k_idx], vl);

        /* 2. 倒序虚部原始内存地址, 整块连续加载 */
        int base_im = N - k_idx - (int)vl + 1;
        vfloat32m1_t vA_im_raw = __riscv_vle32_v_f32m1(&A[base_im], vl);
        vfloat32m1_t vB_im_raw = __riscv_vle32_v_f32m1(&B[base_im], vl);

        /* 3. 生成倒序索引, 寄存器内部翻转虚部, 与实部对齐 */
        vuint32m1_t v_idx     = __riscv_vid_v_u32m1(vl);
        vuint32m1_t v_rev_idx = __riscv_vrsub_vx_u32m1(v_idx, vl - 1, vl);

        vfloat32m1_t vA_im = __riscv_vrgather_vv_f32m1(vA_im_raw, v_rev_idx, vl);
        vfloat32m1_t vB_im = __riscv_vrgather_vv_f32m1(vB_im_raw, v_rev_idx, vl);

        /* 4. 复数乘法 Z = A * B */
        vfloat32m1_t v_reZ = __riscv_vfmul_vv_f32m1(vA_re, vB_re, vl);
        v_reZ = __riscv_vfnmsac_vv_f32m1(v_reZ, vA_im, vB_im, vl);

        vfloat32m1_t v_imZ = __riscv_vfmul_vv_f32m1(vA_re, vB_im, vl);
        v_imZ = __riscv_vfmacc_vv_f32m1(v_imZ, vA_im, vB_re, vl);

        /* 5. 写回半复数格式: 实部正序连续写回, 虚部重新倒序写回 */
        __riscv_vse32_v_f32m1(&C[k_idx], v_reZ, vl);

        vfloat32m1_t v_imZ_rev = __riscv_vrgather_vv_f32m1(v_imZ, v_rev_idx, vl);
        __riscv_vse32_v_f32m1(&C[base_im], v_imZ_rev, vl);

        k_idx += (int)vl;
    }
}
