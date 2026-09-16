/* SPDX-License-Identifier: GPL-2.0 */
/*
 * nap.h - Neural Adaptive Predictor cpuidle governor (fixed-point, arm64/GKI)
 *
 * Independent fixed-point implementation written to the same behavioural
 * spec as firelzrd/nap v0.5.0 (MLP 8->8 trunk + ordinal survival head,
 * confidence-gated deepest-state selection, online SGD refinement). Not a
 * line-for-line port: upstream is x86-only (SSE2/AVX2 float intrinsics,
 * kernel_fpu_begin()/asm/fpu/api.h) which cannot build on arm64 at all, and
 * cpuidle_select() runs from do_idle() with IRQs disabled where
 * may_use_simd() is false on arm64 (kernel_neon_begin() would BUG_ON()), so
 * a NEON port isn't viable either. Everything here is plain integer/Q16.16
 * fixed-point arithmetic - no FPU, no SIMD, bit-identical across CPUs.
 */
#ifndef _NAP_H
#define _NAP_H

#include <linux/types.h>
#include <linux/cpuidle.h>

/* ----------------------------------------------------------------
 * Fixed-point format: Q16.16 signed 32-bit, with s64 intermediates
 * for multiply to avoid overflow.
 * ---------------------------------------------------------------- */
#define NAP_FP_SHIFT		16
#define NAP_FP_ONE		(1 << NAP_FP_SHIFT)

static inline s32 nap_fp_from_int(s32 x)
{
	return x << NAP_FP_SHIFT;
}

/* Saturating Q16.16 multiply. */
static inline s32 nap_fp_mul(s32 a, s32 b)
{
	s64 r = ((s64)a * (s64)b) >> NAP_FP_SHIFT;

	if (r > S32_MAX)
		return S32_MAX;
	if (r < S32_MIN)
		return S32_MIN;
	return (s32)r;
}

/* ----------------------------------------------------------------
 * Neural network dimensions
 * ---------------------------------------------------------------- */
#define NAP_INPUT_SIZE		8
#define NAP_HIDDEN_SIZE		8
#define NAP_NUM_CUTS		(CPUIDLE_STATE_MAX - 1)
#define NAP_HISTORY_SIZE	8

/*
 * Weights for the 8->8 ReLU MLP trunk feeding a shared linear score
 * s = w_out . h + b_out, which is the input to a proportional-odds
 * ordinal survival head: for boundary k, q_k = sigmoid(s - thr_ord[k]).
 * All values are Q16.16 fixed point.
 */
struct nap_weights {
	s32 w1[NAP_INPUT_SIZE][NAP_HIDDEN_SIZE];
	s32 b1[NAP_HIDDEN_SIZE];
	s32 w_out[NAP_HIDDEN_SIZE];
	s32 b_out;
	s32 thr_ord[NAP_NUM_CUTS];
};

struct nap_stats {
	u64 total_selects;
	u64 total_residency_ns;
	u64 overshoot_count;
	u64 learn_count;
};

struct nap_cpu_data {
	/* Ring buffer of recent idle durations, log2(ns) in Q16.16 */
	s32 log_history[NAP_HISTORY_SIZE];
	int hist_idx;
	int hist_count;

	/* Cached shallowest valid state (latency_req gated) */
	int cached_min_state;
	s64 cached_min_state_latency;
	unsigned long cached_min_state_jiffies;

	/* select() -> reflect() handoff */
	int last_selected_idx;
	bool short_circuited;

	/* Per-state log2(target_residency_ns), Q16.16, filled by nap_enable() */
	s32 log2_tres[CPUIDLE_STATE_MAX];

	/* Forward-pass scratch, reused by the learning step */
	s32 hidden[NAP_HIDDEN_SIZE];
	s32 features[NAP_INPUT_SIZE];
	s32 score;

	struct nap_weights weights;

	/* Online learning */
	unsigned int learning_rate_millths;	/* /1000, e.g. 1 = 0.001 */
	unsigned int conf_millths;		/* decision confidence, /1000 */
	int learn_interval;
	int learn_counter;
	bool reset_pending;

	struct nap_stats stats;
};

#endif /* _NAP_H */
