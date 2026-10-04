#define _USE_MATH_DEFINES
#include <math.h>
void generate_rdfft_twiddle_factors(float* W_re, float* W_im, int N) {
    int max_k = N / 2;
    for (int k = 0; k < max_k; k++) {
        double angle = -2.0 * M_PI * (double)k / (double)N;
        W_re[k] = (float)cos(angle);
        W_im[k] = (float)sin(angle);
    }
}