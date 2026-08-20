#include <arm_neon.h>
void schedforge_neon_matmul(const float* a, const float* b, float* c) {
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column + 4 <= 4; column += 4) {
      float32x4_t acc = vdupq_n_f32(0.0f);
      for (int kk = 0; kk < 4; ++kk) {
        acc = vfmaq_n_f32(acc, vld1q_f32(b + kk * 4 + column), a[row * 4 + kk]);
      }
      vst1q_f32(c + row * 4 + column, acc);
    }
  }
}
