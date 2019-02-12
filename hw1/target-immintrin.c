/*
    Please include compiler name below (you may also include any other modules
you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You
can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3 -std=c99
LDLIBS = -lrt

*/
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 128
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

// Multiplies two blocks of size four.
__attribute__((hot)) static void miniblock_mult(double *A, double *B, double *restrict C, int len_A, int len_B) {

  __m256d fill_register(int increment) { return _mm256_load_pd(A + len_A * increment); }

  void fill_block(int increment) {
    __m256d temp = _mm256_mul_pd(fill_register(0), _mm256_broadcast_sd(B + increment * len_B));
    temp = _mm256_fmadd_pd(fill_register(1), _mm256_broadcast_sd(B + increment * len_B + 1), temp);
    temp = _mm256_fmadd_pd(fill_register(2), _mm256_broadcast_sd(B + increment * len_B + 2), temp);
    temp = _mm256_fmadd_pd(fill_register(3), _mm256_broadcast_sd(B + increment * len_B + 3), temp);
    _mm256_store_pd(C + increment * 4, _mm256_add_pd(_mm256_load_pd(C + increment * 4), temp));
  }

  //#pragma forceinline
  {
    fill_block(0);
    fill_block(1);
    fill_block(2);
    fill_block(3);
  }
}

// Add size four block to target matrix.
inline static void insertblock(double *temp, double *restrict target, int len_A, int leftover_row, int leftover_column) {
  for (int x = 0; x < leftover_column; ++x) {
    for (int i = 0; i < leftover_row; ++i) {
      target[i + len_A * x] += temp[i + 4 * x];
    }
  }
}

// performs dgemms on smaller rectangular subblocks
double static A_temp[BLOCK_SIZE * BLOCK_SIZE * sizeof(double)] __attribute__((aligned(64)));

__attribute__((optimize("unroll-loops"))) inline static void block_operation(int len_A, int len_B, int len_C, int M, int N, int K, double *A, double *B, double *restrict C) {
  for (int k = 0; k < K; k += 4)
    for (int m = 0; m < M; m += 4) {

      int extra_col = min(4, M - m);
      int extra_row = min(4, K - k);

      for (int i = 0; i < 4; ++i) {

        if (i >= extra_row) {
          for (int j = 0; j < 4; ++j) {
            A_temp[k * 4 + m * BLOCK_SIZE + j + 4 * i] = 0;
          }
        } else {
          if (extra_col == 4) {
            _mm256_store_pd(A_temp + k * 4 + m * BLOCK_SIZE + 4 * i, _mm256_loadu_pd(m + A + (k + i) * len_A));
          } else {
            for (int t = 0; t < 4; ++t) {
              if (t < extra_col) {
                A_temp[k * 4 + m * BLOCK_SIZE + t + 4 * i] = A[(m + t) + (k + i) * len_A];
              } else {
                A_temp[k * 4 + m * BLOCK_SIZE + t + 4 * i] = 0;
              }
            }
          }
        }
      }
    }

  for (int n = 0; n < N; n += 4)
    for (int m = 0; m < M; m += 4) {
      double C_temp[16] = {0};
      for (int k = 0; k < K; k += 4) {
        miniblock_mult(A_temp + 4 * k + m * BLOCK_SIZE, B + k + n * len_B, C_temp, 4, len_B);
      }
      int extra_row = min(4, M - m);
      int extra_col = min(4, N - n);

      insertblock(C_temp, m + C + len_C * n, len_C, extra_row, extra_col);
    }
}

// A, B, and C are square matrices of dimension len_A and stored in column-major
// format.
void square_dgemm(int len_A, double *A, double *B, double *C) {
  for (int n = 0; n < len_A; n += BLOCK_SIZE)
    for (int m = 0; m < len_A; m += BLOCK_SIZE)
      for (int k = 0; k < len_A; k += BLOCK_SIZE) {
        // Fix dimensions in edge cases
        int M = min(BLOCK_SIZE, len_A - m);
        int N = min(BLOCK_SIZE, len_A - n);
        int K = min(BLOCK_SIZE, len_A - k);
        block_operation(len_A, len_A, len_A, M, N, K, m + A + k * len_A, k + B + n * len_A, m + C + n * len_A);
      }
}
