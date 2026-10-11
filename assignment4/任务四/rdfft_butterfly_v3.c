/* ========================================================================
 * rdfft_butterfly_v3.c — 小阶段 codelet 特化 (v3, 原论文共轭对结构不变)
 *
 * M=1/2/4/8 四级蝶形直写硬展开:
 *   - 每级旋转因子角度 θ = πk/M, 与 N 无关, 全部是编译期常量
 *     (M=1/2 无旋转因子; M=4 仅 √2/2; M=8 仅 π/8、π/4、3π/8 三组)
 *   - 零旋转因子表加载、零 vsetvl 循环分支, 跨组向量化
 *     (vlse32 步进取各组同位元素, 组间无数据依赖)
 *   - 数学与通用 rdfft_butterfly_compute 逐位等价,
 *     每对蝶形仍是 4 读 4 写纯实数原地闭环 (原论文 Proposition 1)
 *
 * 各阶段按 k 分组处理 (组内各 k 的索引集合互不相交, 顺序无关):
 *   M=1:  (2g, 2g+1) 对 — s/d 纯实数加减
 *   M=2:  k=0 (4g,4g+2) s/d;  k=1: Y[+1] 不变, Y[+3] = -Y[+3]
 *   M=4:  k=0 (8g,8g+4) s/d;  k=2: Y[+2] 不变, Y[+6] = -Y[+6];
 *         k=1: 4R4W, w = e^{-jπ/4}
 *   M=8:  k=0 (16g,16g+8) s/d; k=4: Y[+4] 不变, Y[+12] = -Y[+12];
 *         k=1..3: 4R4W, w = e^{-jπk/8}
 * ======================================================================== */
#include <riscv_vector.h>

#define CODELET_SQRT2_2   0.70710678118654752440f /* cos(π/4) = sin(π/4) */
#define CODELET_COS_PI_8  0.92387953251128675613f /* cos(π/8)           */
#define CODELET_SIN_PI_8  0.38268343236508977173f /* sin(π/8)           */
#define CODELET_COS_3PI_8 0.38268343236508977173f /* cos(3π/8)          */
#define CODELET_SIN_3PI_8 0.92387953251128675613f /* sin(3π/8)          */

/* ---- M=1: Y[2g],Y[2g+1] = s,d (纯实数加减, 无旋转因子) ---- */
void rdfft_stage_M1(float *Y, int N) {
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

/* ---- M=2 ----
 * k=0: 对 (4g, 4g+2): Y[4g]=s, Y[4g+2]=d
 * k=1 (=M/2): Y[4g+1] 不变, Y[4g+3] = -Y[4g+3] ---- */
void rdfft_stage_M2(float *Y, int N) {
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
        vfloat32m2_t v = __riscv_vlse32_v_f32m2(&Y[4 * i + 3], 16, vl);
        __riscv_vsse32_v_f32m2(&Y[4 * i + 3], 16, __riscv_vfneg_v_f32m2(v, vl), vl);
        i += (int)vl;
    }
}

/* ---- M=4 ----
 * k=0: 对 (8g, 8g+4) s/d
 * k=2 (=M/2): Y[8g+2] 不变, Y[8g+6] = -Y[8g+6]
 * k=1: w = e^{-jπ/4} = (c, -c), c = √2/2
 *   t_re = c(o_re + o_im),  t_im = c(o_im - o_re)
 *   Y[+1] = e_re + t_re,  Y[+3] = e_re - t_re,
 *   Y[+5] = t_im - e_im,  Y[+7] = e_im + t_im ---- */
void rdfft_stage_M4(float *Y, int N) {
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
        vfloat32m2_t v = __riscv_vlse32_v_f32m2(&Y[8 * i + 6], 32, vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 6], 32, __riscv_vfneg_v_f32m2(v, vl), vl);
        i += (int)vl;
    }
    i = 0;
    while (i < N / 8) {
        size_t vl = __riscv_vsetvl_e32m2(N / 8 - i);
        vfloat32m2_t v_e_re = __riscv_vlse32_v_f32m2(&Y[8 * i + 1], 32, vl);
        vfloat32m2_t v_e_im = __riscv_vlse32_v_f32m2(&Y[8 * i + 3], 32, vl);
        vfloat32m2_t v_o_re = __riscv_vlse32_v_f32m2(&Y[8 * i + 5], 32, vl);
        vfloat32m2_t v_o_im = __riscv_vlse32_v_f32m2(&Y[8 * i + 7], 32, vl);
        vfloat32m2_t v_t_re = __riscv_vfmul_vf_f32m2(
            __riscv_vfadd_vv_f32m2(v_o_re, v_o_im, vl), CODELET_SQRT2_2, vl);
        vfloat32m2_t v_t_im = __riscv_vfmul_vf_f32m2(
            __riscv_vfsub_vv_f32m2(v_o_im, v_o_re, vl), CODELET_SQRT2_2, vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 1], 32, __riscv_vfadd_vv_f32m2(v_e_re, v_t_re, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 3], 32, __riscv_vfsub_vv_f32m2(v_e_re, v_t_re, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 5], 32, __riscv_vfsub_vv_f32m2(v_t_im, v_e_im, vl), vl);
        __riscv_vsse32_v_f32m2(&Y[8 * i + 7], 32, __riscv_vfadd_vv_f32m2(v_e_im, v_t_im, vl), vl);
        i += (int)vl;
    }
}

/* ---- M=8 ----
 * k=0: 对 (16g, 16g+8) s/d
 * k=4 (=M/2): Y[16g+4] 不变, Y[16g+12] = -Y[16g+12]
 * k=1..3: w = e^{-jπk/8} = (wr, -ws), (wr,ws)=(cos,sin)(πk/8)
 *   t_re = wr*o_re + ws*o_im,  t_im = wr*o_im - ws*o_re
 *   Y[+k] = e_re+t_re,  Y[+M-k] = e_re-t_re,
 *   Y[+M+k] = t_im-e_im, Y[+2M-k] = e_im+t_im ---- */
void rdfft_stage_M8(float *Y, int N) {
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
        vfloat32m2_t v = __riscv_vlse32_v_f32m2(&Y[16 * i + 12], 64, vl);
        __riscv_vsse32_v_f32m2(&Y[16 * i + 12], 64, __riscv_vfneg_v_f32m2(v, vl), vl);
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
            float *e_re_p = &Y[16 * i + k];
            float *e_im_p = &Y[16 * i + 8 - k];
            float *o_re_p = &Y[16 * i + 8 + k];
            float *o_im_p = &Y[16 * i + 16 - k];
            vfloat32m2_t v_e_re = __riscv_vlse32_v_f32m2(e_re_p, 64, vl);
            vfloat32m2_t v_e_im = __riscv_vlse32_v_f32m2(e_im_p, 64, vl);
            vfloat32m2_t v_o_re = __riscv_vlse32_v_f32m2(o_re_p, 64, vl);
            vfloat32m2_t v_o_im = __riscv_vlse32_v_f32m2(o_im_p, 64, vl);
            vfloat32m2_t v_t_re = __riscv_vfmul_vf_f32m2(v_o_im, ws, vl);
            v_t_re = __riscv_vfmacc_vf_f32m2(v_t_re, wr, v_o_re, vl);
            vfloat32m2_t v_t_im = __riscv_vfmul_vf_f32m2(v_o_im, wr, vl);
            v_t_im = __riscv_vfnmsac_vf_f32m2(v_t_im, ws, v_o_re, vl);
            __riscv_vsse32_v_f32m2(e_re_p, 64, __riscv_vfadd_vv_f32m2(v_e_re, v_t_re, vl), vl);
            __riscv_vsse32_v_f32m2(e_im_p, 64, __riscv_vfsub_vv_f32m2(v_e_re, v_t_re, vl), vl);
            __riscv_vsse32_v_f32m2(o_re_p, 64, __riscv_vfsub_vv_f32m2(v_t_im, v_e_im, vl), vl);
            __riscv_vsse32_v_f32m2(o_im_p, 64, __riscv_vfadd_vv_f32m2(v_e_im, v_t_im, vl), vl);
            i += (int)vl;
        }
    }
}
