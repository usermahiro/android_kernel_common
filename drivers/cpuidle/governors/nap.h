/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _NAP_H
#define _NAP_H

#include <linux/types.h>
#include <linux/cpuidle.h>
#include <linux/ktime.h>

/* ponytail: Q16.16 fixed-point replaces x86 SSE2/AVX2 kernel_fpu_begin() (unsafe in arm64 do_idle with IRQs off); add NEON when cpuidle_select runs in preemptible/SIMD-allowed context. */
#define NAP_FP_SHIFT			16
#define NAP_FP_ONE			(1 << NAP_FP_SHIFT)

#define NAP_INPUT_SIZE			8
#define NAP_HIDDEN_SIZE			8
#define NAP_NUM_CUTS			(CPUIDLE_STATE_MAX - 1)
#define NAP_HISTORY_SIZE		8

#define NAP_POLL_LIMIT_MIN_NS		1000ULL
#define NAP_POLL_LIMIT_MARGIN_NS	1000ULL
#define NAP_MIN_STATE_REFRESH_JIFFIES	HZ

static inline s32 nap_fp_from_int(s32 x)
{
	return x << NAP_FP_SHIFT;
}

static inline s32 nap_fp_clamp(s32 v, s32 lo, s32 hi)
{
	if (v < lo)
		return lo;
	if (v > hi)
		return hi;
	return v;
}

static inline s32 nap_fp_mul(s32 a, s32 b)
{
	s64 r = ((s64)a * (s64)b) >> NAP_FP_SHIFT;

	if (r > S32_MAX)
		return S32_MAX;
	if (r < S32_MIN)
		return S32_MIN;
	return (s32)r;
}

struct nap_weights {
	s32 w_h1[NAP_INPUT_SIZE][NAP_HIDDEN_SIZE];
	s32 b_h1[NAP_HIDDEN_SIZE];
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
	u64 history[NAP_HISTORY_SIZE];
	s32 log_history[NAP_HISTORY_SIZE];
	int hist_idx;
	int hist_count;

	u64 prev_idle_exit;
	s64 last_predicted_ns;
	s64 last_prediction_error;

	bool short_circuited;
	int cached_min_state;
	s64 cached_min_state_latency;
	unsigned long cached_min_state_jiffies;

	unsigned long last_learn_jiffies;
	unsigned int learn_jiffies_min;

	int last_selected_idx;

	s32 nn_output;
	s32 hidden_out[NAP_HIDDEN_SIZE];
	s32 features[NAP_INPUT_SIZE];

	s32 learn_d_out;
	s32 learn_lr;
	s32 learn_d_hid[NAP_HIDDEN_SIZE];

	s32 log2_tres[CPUIDLE_STATE_MAX];
	s32 bin_count[CPUIDLE_STATE_MAX];

	bool needs_learn;
	bool have_sample;
	u64 learn_actual_ns;

	struct nap_weights weights;

	unsigned int learning_rate_millths;
	unsigned int max_grad_norm_millths;
	unsigned int conf_millths;
	int learn_interval;
	int learn_counter;
	bool reset_pending;

	struct nap_stats stats;
};

DECLARE_PER_CPU(struct nap_cpu_data, nap_data);

#endif /* _NAP_H */
