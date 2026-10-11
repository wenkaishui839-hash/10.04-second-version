/* ========================================================================
 * irdfft_butterfly_v3.c — 逆向小阶段 codelet 特化 (v3, DIF 结构不变)
 *
 * 与正变换 rdfft_butterfly_v3.c 逐级互逆, M=8/4/2/1 四级直写硬展开:
 *   - 逆向旋转因子 conj(W_{2M}^k) = (wr, +ws), (wr,ws) 同为编译期常量
 *   - 每对蝶形仍是 4 读 4 写原地闭环, 无表加载、无循环分支
 *
 * 各阶段按 k 分组 (组内各 k 索引集合互不相交):
 *   M=8:  k=0 (16g,16g+8) s/d; k=4: Y[+4]=2*, Y[+12]=-2*; k=1..3 4R4W
 *   M=4:  k=0 (8g,8g+4) s/d;   k=2: Y[+2]=2*, Y[+6]=-2*;   k=1 4R4W
 *   M=2:  k=0 (4g,4g+2) s/d;   k=1: Y[+1]=2*, Y[+3]=-2*
 *   M=1:  相邻对 s/d
 *
 * 4R4W 公式 (与 irdfft_butterfly_compute 通用版逐位等价):
 *   a = Re(X[k]), c = Re(X[M-k]), d = Im(X[M-k]), b = Im(X[k])
 *   E: e_re = a+c -> +k,  e_im = b-d -> +M-k
 *   O: o_re = wr*(a-c) - ws*(b+d) -> +M+k
 *      o_im = ws*(a-c) + wr*(b+d) -> +2M-k
 * ======================================================================== */
#include <riscv_vector.h>

#define CODELET_SQRT2_2   0.70710678118654752440f
#define CODELET_COS_PI_8  0.92387953251128675613f
#define CODELET_SIN_PI_8  0.38268343236508977173f
#define CODELET_COS_3PI_8 0.38268343236508977173f
#define CODELET_SIN_3PI_8 0.92387953251128675613f

/* ---- M=1: 相邻对 s/d ---- */
void irdfft_stage_M1(float *Y, int N) {
    int i = 0;
    while (i < N / 2) {
        size_t vl = __riscv_vsetvl_e32m2(N / 2 - i);
        vfloat32m2_t ve = __riscv_vlse32_v_f32m2(&Y[2 * i], 8, vl);
        vfloat32m2_t vo = __riscv_vlse32_v_f32m2(&Y[2 * i + 1], 8, vl);
        __riscv_vsse32_v_f32m2(&Y[2 * i], 8, __riscv_vfadd_vv_f32m2(ve, vo, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[2 * i + 1], 8, __riscv_vfsub_vv_f32m2(ve, vo, vl), vl);
        i += (int)vl;
    }
}

/* ---- M=2 ---- */
void irdfft_stage_M2(float *Y, int N) {
    int i = 0;
    while (i < N / 4) {
        size_t vl = __riscv_vsetvl_e32m2(N / 4 - i);
        vfloat32m2_t ve = __riscv_vlse32_v_f32m2(&Y[4 * i], 16, vl);
        vfloat32m2_t vo = __riscv_vlse32_v_f32m2(&Y[4 * i + 2], 16, vl);
        __riscv_vsse32_v_f32m2(&Y[4 * i], 16, __riscv_vfadd_vv_f32m2(ve, vo, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[4 * i + 2], 16, __riscv_vfsub_vv_f32m2(ve, vo, vl), vl);
        i += (int)vl;
    }
    i = 0;
    while (i < N / 4) {
        size_t vl = __riscv_vsetvl_e32m2(N / 4 - i);
        vfloat32m2_t v1 = __riscv_vlse32_v_f32m2(&Y[4 * i + 1], 16, vl);
        vfloat32m2_t v3 = __riscv_vlse32_v_f32m2(&Y[4 * i + 3], 16, vl);
        __riscv_vsse32_v_f32m2(&Y[4 * i + 1], 16,
            __riscv_vfmul_vf_f32m2(v1, 2.0f, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[4 * i + 3], 16,
            __riscv_vfmul_vf_f32m2(v3, -2.0f, vl), vl);
        i += (int)vl;
    }
}

/* ---- M=4 ----
 * k=1: conj(w) = e^{+jπ/4} = (c, +c), c = √2/2
 *   e_re = a+c, e_im = b-d, o_re = c(dr-di), o_im = c(dr+di)
 *   dr = a-c, di = b+d ---- */
void irdfft_stage_M4(float *Y, int N) {
    int i = 0;
    while (i < N / 8) {
        size_t vl = __riscv_vsetvl_e32m2(N / 8 - i);
        vfloat32m2_t ve = __riscv_vlse32_v_f32m2(&Y[8 * i], 32, vl);
        vfloat32m2_t vo = __riscv_vlse32_v_f32m2(&Y[8 * i + 4], 32, vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i], 32, __riscv_vfadd_vv_f32m2(ve, vo, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 4], 32, __riscv_vfsub_vv_f32m2(ve, vo, vl), vl);
        i += (int)vl;
    }
    i = 0;
    while (i < N / 8) {
        size_t vl = __riscv_vsetvl_e32m2(N / 8 - i);
        vfloat32m2_t v2 = __riscv_vlse32_v_f32m2(&Y[8 * i + 2], 32, vl);
        vfloat32m2_t v6 = __riscv_vlse32_v_f32m2(&Y[8 * i + 6], 32, vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 2], 32,
            __riscv_vfmul_vf_f32m2(v2, 2.0f, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 6], 32,
            __riscv_vfmul_vf_f32m2(v6, -2.0f, vl), vl);
        i += (int)vl;
    }
    i = 0;
    while (i < N / 8) {
        size_t vl = __riscv_vsetvl_e32m2(N / 8 - i);
        vfloat32m2_t v_a = __riscv_vlse32_v_f32m2(&Y[8 * i + 1], 32, vl);
        vfloat32m2_t v_c = __riscv_vlse32_v_f32m2(&Y[8 * i + 3], 32, vl);
        vfloat32m2_t v_d = __riscv_vlse32_v_f32m2(&Y[8 * i + 5], 32, vl);
        vfloat32m2_t v_b = __riscv_vlse32_v_f32m2(&Y[8 * i + 7], 32, vl);
        vfloat32m2_t v_dr = __riscv_vfsub_vv_f32m2(v_a, v_c, vl);
        vfloat32m2_t v_di = __riscv_vfadd_vv_f32m2(v_b, v_d, vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 1], 32,
            __riscv_vfadd_vv_f32m2(v_a, v_c, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 3], 32,
            __riscv_vfsub_vv_f32m2(v_b, v_d, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 5], 32,
            __riscv_vfmul_vf_f32m2(__riscv_vfsub_vv_f32m2(v_dr, v_di, vl),
                                   CODELET_SQRT2_2, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 7], 32,
            __riscv_vfmul_vf_f32m2(__riscv_vfadd_vv_f32m2(v_dr, v_di, vl),
                                   CODELET_SQRT2_2, vl), vl);
        i += (int)vl;
    }
}

/* ---- M=8 ----
 * k=1..3: conj(w) = e^{+jπk/8} = (wr, +ws)
 *   o_re = wr*(a-c) - ws*(b+d) -> +M+k
 *   o_im = ws*(a-c) + wr*(b+d) -> +2M-k ---- */
void irdfft_stage_M8(float *Y, int N) {
    const int G = N / 16;
    int i = 0;
    while (i < G) {
        size_t vl = __riscv_vsetvl_e32m2(G - i);
        vfloat32m2_t ve = __riscv_vlse32_v_f32m2(&Y[16 * i], 64, vl);
        vfloat32m2_t vo = __riscv_vlse32_v_f32m2(&Y[16 * i + 8], 64, vl);
        __riscv_vsse32_v_f32m2(&Y[16 * i], 64, __riscv_vfadd_vv_f32m2(ve, vo, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[16 * i + 8], 64, __riscv_vfsub_vv_f32m2(ve, vo, vl), vl);
        i += (int)vl;
    }
    i = 0;
    while (i < G) {
        size_t vl = __riscv_vsetvl_e32m2(G - i);
        vfloat32m2_t v4  = __riscv_vlse32_v_f32m2(&Y[16 * i + 4], 64, vl);
        vfloat32m2_t v12 = __riscv_vlse32_v_f32m2(&Y[16 * i + 12], 64, vl);
        __riscv_vsse32_v_f32m2(&Y[16 * i + 4], 64,
            __riscv_vfmul_vf_f32m2(v4, 2.0f, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[16 * i + 12], 64,
            __riscv_vfmul_vf_f32m2(v12, -2.0f, vl), vl);
        i += (int)vl;
    }
    for (int k = 1; k <= 3; k++) {
        float wr, ws;
        if (k == 1)      { wr = CODELET_COS_PI_8;  ws = CODELET_SIN_PI_8; }
        else if (k == 2) { wr = CODELET_SQRT2_2;   ws = CODELET_SQRT2_2; }
        else             { wr = CODELET_COS_3PI_8; ws = CODELET_SIN_3PI_8; }
        i = 0;
        while (i < G) {
            size_t vl = __riscv_vsetvl_e32m2(G - i);
            float *a_p = &Y[16 * i + k];
            float *c_p = &Y[16 * i + 8 - k];
            float *d_p = &Y[16 * i + 8 + k];
            float *b_p = &Y[16 * i + 16 - k];
            vfloat32m2_t v_a = __riscv_vlse32_v_f32m2(a_p, 64, vl);
            vfloat32m2_t v_c = __riscv_vlse32_v_f32m2(c_p, 64, vl);
            vfloat32m2_t v_d = __riscv_vlse32_v_f32m2(d_p, 64, vl);
            vfloat32m2_t v_b = __riscv_vlse32_v_f32m2(b_p, 64, vl);
            vfloat32m2_t v_dr = __riscv_vfsub_vv_f32m2(v_a, v_c, vl);
            vfloat32m2_t v_di = __riscv_vfadd_vv_f32m2(v_b, v_d, vl);
            __riscv_vsse32_v_f32m2(a_p, 64, __riscv_vfadd_vv_f32m2(v_a, v_c, vl), vl);
            __riscv_vsse32_v_f32m2(c_p, 64, __riscv_vfsub_vv_f32m2(v_b, v_d, vl), vl);
            vfloat32m2_t v_o_re = __riscv_vfmul_vf_f32m2(v_di, -ws, vl);
            v_o_re = __riscv_vfmacc_vf_f32m2(v_o_re, wr, v_dr, vl);
            vfloat32m2_t v_o_im = __riscv_vfmul_vf_f32m2(v_dr, ws, vl);
            v_o_im = __riscv_vfmacc_vf_f32m2(v_o_im, wr, v_di, vl);
            __riscv_vsse32_v_f32m2(d_p, 64, v_o_re, vl);
            __riscv_vsse32_v_f32m2(b_p, 64, v_o_im, vl);
            i += (int)vl;
        }
    }
}
