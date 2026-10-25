// SPDX-License-Identifier: GPL-2.0
#include <linux/cpuidle.h>
#include <linux/cpu.h>
#include <linux/cpumask.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kobject.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/percpu.h>
#include <linux/pm_qos.h>
#include <linux/sched/clock.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/tick.h>

#include "nap.h"
#include "gov.h"

#define CPUIDLE_NAP_PROGNAME		"Nap CPUIdle Governor"
#define CPUIDLE_NAP_AUTHOR		"Masahito Suzuki"
#define CPUIDLE_NAP_VERSION		"0.5.1"

#define NAP_DEFAULT_LR_MILLTHS		1
#define NAP_DEFAULT_INTERVAL		4
#define NAP_DEFAULT_CLAMP_MILLTHS	1000
#define NAP_DEFAULT_CONF_MILLTHS	500

#define NAP_FLOOR_WIN			256
#define NAP_PRIOR_K			16
#define NAP_PRNG_SEED			42u

#define NAP_LOG2_A			93118
#define NAP_LOG2_B			(-37831)
#define NAP_LOG2_C			10249

#define NAP_SIGMOID_LUT_SIZE		256
#define NAP_SIGMOID_RANGE_FP		(8 << NAP_FP_SHIFT)

static const s32 nap_sigmoid_lut[NAP_SIGMOID_LUT_SIZE + 1] = {
	22, 23, 25, 27, 28, 30, 32, 34,
	36, 39, 41, 44, 47, 50, 53, 56,
	60, 64, 68, 72, 77, 82, 87, 92,
	98, 105, 111, 119, 126, 134, 143, 152,
	162, 172, 184, 195, 208, 221, 236, 251,
	267, 284, 302, 321, 342, 364, 387, 412,
	439, 467, 497, 528, 562, 598, 636, 677,
	720, 766, 815, 867, 922, 980, 1042, 1109,
	1179, 1253, 1333, 1417, 1506, 1601, 1701, 1808,
	1921, 2041, 2168, 2303, 2446, 2598, 2758, 2928,
	3108, 3298, 3500, 3713, 3938, 4176, 4427, 4692,
	4971, 5266, 5577, 5904, 6249, 6611, 6992, 7392,
	7812, 8252, 8714, 9197, 9702, 10230, 10782, 11357,
	11955, 12579, 13226, 13898, 14595, 15316, 16062, 16832,
	17625, 18442, 19282, 20143, 21025, 21928, 22849, 23788,
	24743, 25712, 26695, 27689, 28693, 29705, 30723, 31744,
	32768, 33792, 34813, 35831, 36843, 37847, 38841, 39824,
	40793, 41748, 42687, 43608, 44511, 45393, 46254, 47094,
	47911, 48704, 49474, 50220, 50941, 51638, 52310, 52957,
	53581, 54179, 54754, 55306, 55834, 56339, 56822, 57284,
	57724, 58144, 58544, 58925, 59287, 59632, 59959, 60270,
	60565, 60844, 61109, 61360, 61598, 61823, 62036, 62238,
	62428, 62608, 62778, 62938, 63090, 63233, 63368, 63495,
	63615, 63728, 63835, 63935, 64030, 64119, 64203, 64283,
	64357, 64427, 64494, 64556, 64614, 64669, 64721, 64770,
	64816, 64859, 64900, 64938, 64974, 65008, 65039, 65069,
	65097, 65124, 65149, 65172, 65194, 65215, 65234, 65252,
	65269, 65285, 65300, 65315, 65328, 65341, 65352, 65364,
	65374, 65384, 65393, 65402, 65410, 65417, 65425, 65431,
	65438, 65444, 65449, 65454, 65459, 65464, 65468, 65472,
	65476, 65480, 65483, 65486, 65489, 65492, 65495, 65497,
	65500, 65502, 65504, 65506, 65508, 65509, 65511, 65513,
	65514,
};

DEFINE_PER_CPU(struct nap_cpu_data, nap_data);
static struct cpuidle_driver *nap_cached_drv;

static s32 nap_fp_log2_u64(u64 v)
{
	int n;
	u64 mant;
	s32 x, t, frac_log;

	if (unlikely(v == 0))
		return 0;

	n = fls64(v) - 1;
	if (n >= NAP_FP_SHIFT)
		mant = v >> (n - NAP_FP_SHIFT);
	else
		mant = v << (NAP_FP_SHIFT - n);

	x = (s32)mant - NAP_FP_ONE;
	t = nap_fp_mul(x, NAP_LOG2_C) + NAP_LOG2_B;
	t = nap_fp_mul(x, t) + NAP_LOG2_A;
	frac_log = nap_fp_mul(x, t);

	return (n << NAP_FP_SHIFT) + frac_log;
}

static s32 nap_fp_sigmoid(s32 x)
{
	u32 u, idx, frac;
	s32 y0, y1;

	if (x <= -NAP_SIGMOID_RANGE_FP)
		return nap_sigmoid_lut[0];
	if (x >= NAP_SIGMOID_RANGE_FP)
		return nap_sigmoid_lut[NAP_SIGMOID_LUT_SIZE];

	u = (u32)(x + NAP_SIGMOID_RANGE_FP);
	idx = u >> 12;
	frac = (u & 0xFFFu) << 4;

	y0 = nap_sigmoid_lut[idx];
	y1 = nap_sigmoid_lut[idx + 1];
	return y0 + (((y1 - y0) * (s32)frac) >> NAP_FP_SHIFT);
}

#ifdef CONFIG_CPU_IDLE_GOV_NAP_SELFTEST
static int __init nap_selftest(void)
{
	int fails = 0;
	s32 v, prev;
	u64 val;
	int i;

	v = nap_fp_log2_u64(1ULL << 20);
	if (v != nap_fp_from_int(20))
		fails++;

	v = nap_fp_sigmoid(0);
	if (v != NAP_FP_ONE / 2)
		fails++;

	if (nap_fp_sigmoid(-NAP_SIGMOID_RANGE_FP) > NAP_FP_ONE / 100 ||
	    nap_fp_sigmoid(NAP_SIGMOID_RANGE_FP) < NAP_FP_ONE - NAP_FP_ONE / 100)
		fails++;

	prev = -1;
	for (i = -800; i <= 800; i += 4) {
		v = nap_fp_sigmoid(i * (NAP_FP_ONE / 100));
		if (v < prev) {
			fails++;
			break;
		}
		prev = v;
	}

	if (nap_fp_mul(S32_MAX, nap_fp_from_int(2)) != S32_MAX)
		fails++;

	prev = S32_MIN;
	for (val = 1; val < (1ULL << 40); val <<= 1) {
		v = nap_fp_log2_u64(val | (val >> 2));
		if (v < prev) {
			fails++;
			break;
		}
		prev = v;
	}

	if (fails)
		pr_err("nap selftest: %d check(s) FAILED\n", fails);
	else
		pr_info("nap selftest: all fixed-point math checks passed\n");

	return 0;
}
#else
static inline int nap_selftest(void) { return 0; }
#endif

static inline s32 nap_prng_fp(u32 *state)
{
	*state = *state * 1664525u + 1013904223u;
	return (s32)*state >> 15;
}

static void nap_init_weights(struct nap_weights *w)
{
	u32 rng = NAP_PRNG_SEED;
	const s32 scale_h1 = 40132;
	const s32 scale_out = 655;
	int i, j;

	for (i = 0; i < NAP_INPUT_SIZE; i++)
		for (j = 0; j < NAP_HIDDEN_SIZE; j++)
			w->w_h1[i][j] = nap_fp_mul(nap_prng_fp(&rng), scale_h1);

	memset(w->b_h1, 0, sizeof(w->b_h1));

	for (j = 0; j < NAP_HIDDEN_SIZE; j++)
		w->w_out[j] = nap_fp_mul(nap_prng_fp(&rng), scale_out);

	w->b_out = 0;

	for (i = 0; i < NAP_INPUT_SIZE; i++)
		w->w_h1[i][0] = 0;
	w->w_h1[0][0] = NAP_FP_ONE;
	w->b_h1[0] = 0;
	w->w_out[0] = NAP_FP_ONE;
}

static void nap_init_log2_tres(struct nap_cpu_data *d,
			       struct cpuidle_driver *drv)
{
	int i;

	for (i = 0; i < drv->state_count && i < CPUIDLE_STATE_MAX; i++)
		d->log2_tres[i] = nap_fp_log2_u64(
			max_t(u64, drv->states[i].target_residency_ns, 1));

	for (i = 1; i < drv->state_count && i <= NAP_NUM_CUTS; i++)
		d->weights.thr_ord[i - 1] = d->log2_tres[i];
}

struct logring_stats {
	s32 avg;
	s32 min;
	s32 max;
};

static void logring_compute(const struct nap_cpu_data *d,
			    struct logring_stats *s)
{
	int i, n = d->hist_count;
	s64 sum;
	s32 val;

	if (n <= 0) {
		*s = (struct logring_stats){ 0 };
		return;
	}

	sum = d->log_history[0];
	s->min = d->log_history[0];
	s->max = d->log_history[0];

	for (i = 1; i < n; i++) {
		val = d->log_history[i];
		sum += val;
		if (val < s->min)
			s->min = val;
		if (val > s->max)
			s->max = val;
	}

	s->avg = (s32)div_s64(sum, n);
}

static void nap_extract_features(struct cpuidle_driver *drv,
				 struct cpuidle_device *dev,
				 struct nap_cpu_data *d,
				 u64 sleep_length_ns,
				 s64 latency_req)
{
	struct logring_stats lr;
	u64 busy_ns = local_clock() - d->prev_idle_exit;
	s64 err_us = div_s64(d->last_prediction_error, 1000);
	u64 abs_err_us = (err_us >= 0) ? (u64)err_us : (u64)(-err_us);
	s32 log_err;
	int prev;

	d->features[0] = nap_fp_log2_u64(max_t(u64, sleep_length_ns, 1));
	d->features[1] = nap_fp_log2_u64(max_t(u64, dev->last_residency_ns, 1));
	d->features[6] = nap_fp_log2_u64(max_t(u64, busy_ns, 1));

	log_err = nap_fp_log2_u64(abs_err_us + 1);
	d->features[5] = (err_us < 0) ? -log_err : log_err;

	prev = (d->hist_idx - 1 + NAP_HISTORY_SIZE) % NAP_HISTORY_SIZE;
	d->log_history[prev] = d->features[1];

	logring_compute(d, &lr);
	d->features[2] = lr.avg;
	d->features[3] = lr.min;
	d->features[4] = lr.max;

	if (drv->state_count > 0) {
		u64 deepest_lat = drv->states[drv->state_count - 1].exit_latency_ns;

		if (latency_req < PM_QOS_LATENCY_ANY_NS && deepest_lat > 0)
			d->features[7] = nap_fp_log2_u64(max_t(s64, latency_req, 1)) -
					 nap_fp_log2_u64(deepest_lat);
		else
			d->features[7] = 0;
	} else {
		d->features[7] = 0;
	}

	d->last_predicted_ns = (s64)sleep_length_ns;
}

static void nap_nn_forward(struct nap_cpu_data *d)
{
	int i, j;
	s64 acc;

	for (i = 0; i < NAP_HIDDEN_SIZE; i++) {
		acc = d->weights.b_h1[i];
		for (j = 0; j < NAP_INPUT_SIZE; j++)
			acc += ((s64)d->weights.w_h1[j][i] * d->features[j]) >> NAP_FP_SHIFT;
		if (acc > S32_MAX)
			acc = S32_MAX;
		d->hidden_out[i] = acc > 0 ? (s32)acc : 0;
	}

	acc = d->weights.b_out;
	for (i = 0; i < NAP_HIDDEN_SIZE; i++)
		acc += ((s64)d->weights.w_out[i] * d->hidden_out[i]) >> NAP_FP_SHIFT;
	if (acc > S32_MAX)
		acc = S32_MAX;
	if (acc < S32_MIN)
		acc = S32_MIN;
	d->nn_output = (s32)acc;
}

static void nap_nn_learn(struct nap_cpu_data *d)
{
	s32 d_out = d->learn_d_out;
	s32 lr = d->learn_lr;
	s32 clamp_val = ((s64)d->max_grad_norm_millths << NAP_FP_SHIFT) / 1000;
	int i, j;

	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		if (d->hidden_out[j] > 0)
			d->learn_d_hid[j] = nap_fp_mul(d->weights.w_out[j], d_out);
		else
			d->learn_d_hid[j] = 0;
	}

	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		s32 grad = nap_fp_clamp(nap_fp_mul(d->hidden_out[j], d_out),
					-clamp_val, clamp_val);
		d->weights.w_out[j] -= nap_fp_mul(lr, grad);
	}

	d->weights.b_out -= nap_fp_mul(lr, nap_fp_clamp(d_out, -clamp_val, clamp_val));

	for (i = 0; i < NAP_INPUT_SIZE; i++) {
		s32 feat = d->features[i];

		for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
			s32 grad;

			if (!d->learn_d_hid[j])
				continue;
			grad = nap_fp_clamp(nap_fp_mul(feat, d->learn_d_hid[j]),
					    -clamp_val, clamp_val);
			d->weights.w_h1[i][j] -= nap_fp_mul(lr, grad);
		}
	}

	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		s32 grad;

		if (!d->learn_d_hid[j])
			continue;
		grad = nap_fp_clamp(d->learn_d_hid[j], -clamp_val, clamp_val);
		d->weights.b_h1[j] -= nap_fp_mul(lr, grad);
	}
}

static int nap_fp_select(struct cpuidle_driver *drv,
			 struct cpuidle_device *dev,
			 struct nap_cpu_data *d,
			 u64 sleep_length_ns,
			 s64 latency_req)
{
	int state_count = min_t(int, drv->state_count, CPUIDLE_STATE_MAX);

	if (unlikely(d->reset_pending)) {
		nap_init_weights(&d->weights);
		nap_init_log2_tres(d, drv);
		memset(d->bin_count, 0, sizeof(d->bin_count));
		d->have_sample = false;
		d->stats.learn_count = 0;
		d->needs_learn = false;
		d->reset_pending = false;
	}

	if (d->have_sample) {
		int k, label_bin = 0;

		if (d->needs_learn) {
			s32 base_lr = ((s64)d->learning_rate_millths << NAP_FP_SHIFT) / 1000;
			s32 clamp_val = ((s64)d->max_grad_norm_millths << NAP_FP_SHIFT) / 1000;
			s32 s = d->nn_output;
			s32 g = 0;

			for (k = 1; k < state_count; k++) {
				s32 th = d->weights.thr_ord[k - 1];
				s32 q = nap_fp_sigmoid(s - th);
				s32 y = (d->learn_actual_ns >=
					 drv->states[k].target_residency_ns)
					? NAP_FP_ONE : 0;
				s32 err = q - y;
				s32 lo = d->log2_tres[k] - nap_fp_from_int(6);
				s32 hi = d->log2_tres[k] + nap_fp_from_int(6);
				s32 step = nap_fp_clamp(nap_fp_mul(base_lr, err),
							-clamp_val, clamp_val);

				g += err;
				d->weights.thr_ord[k - 1] = nap_fp_clamp(th + step, lo, hi);
			}
			d->learn_d_out = g;
			d->learn_lr = base_lr;
			d->stats.learn_count++;
			nap_nn_learn(d);
			d->needs_learn = false;
		}

		for (k = 1; k < state_count; k++)
			if (d->learn_actual_ns >= drv->states[k].target_residency_ns)
				label_bin = k;
		for (k = 0; k < state_count; k++)
			d->bin_count[k] = (s32)(((s64)d->bin_count[k] * (NAP_FLOOR_WIN - 1)) /
						NAP_FLOOR_WIN);
		d->bin_count[label_bin] += NAP_FP_ONE;

		d->have_sample = false;
	}

	nap_extract_features(drv, dev, d, sleep_length_ns, latency_req);
	nap_nn_forward(d);

	{
		s32 conf = ((s64)d->conf_millths << NAP_FP_SHIFT) / 1000;
		s32 s = d->nn_output;
		s32 sleep_log2 = d->features[0];
		s32 suffix[CPUIDLE_STATE_MAX];
		s64 total = 0;
		s32 qmin = NAP_FP_ONE;
		int k, m = 0, idx = 0;

		for (k = 0; k < state_count; k++)
			total += d->bin_count[k];

		suffix[state_count - 1] = d->bin_count[state_count - 1];
		for (k = state_count - 2; k >= 0; k--)
			suffix[k] = suffix[k + 1] + d->bin_count[k];

		for (k = 1; k < state_count; k++) {
			s32 q_nn = nap_fp_sigmoid(s - d->weights.thr_ord[k - 1]);
			s64 num = ((s64)NAP_PRIOR_K * ((s64)q_nn << NAP_FP_SHIFT)) +
				  ((s64)suffix[k] << NAP_FP_SHIFT);
			s64 den = ((s64)NAP_PRIOR_K << NAP_FP_SHIFT) + total;
			s32 q = (s32)div64_s64(num, den);

			if (d->log2_tres[k] > sleep_log2)
				q = 0;
			if (q < qmin)
				qmin = q;
			q = qmin;

			if (q >= conf)
				m = k;
			else
				break;
		}

		for (k = m; k >= 1; k--) {
			if (dev->states_usage[k].disable)
				continue;
			if (drv->states[k].exit_latency_ns > latency_req)
				continue;
			idx = k;
			break;
		}
		return idx;
	}
}

static void nap_history_update(struct nap_cpu_data *d, u64 measured_ns)
{
	d->history[d->hist_idx] = measured_ns;
	d->hist_idx = (d->hist_idx + 1) % NAP_HISTORY_SIZE;
	if (d->hist_count < NAP_HISTORY_SIZE)
		d->hist_count++;
}

static void nap_update_external_signals(struct nap_cpu_data *d)
{
	d->prev_idle_exit = local_clock();
}

static int nap_find_min_valid_state(struct cpuidle_driver *drv,
				    struct cpuidle_device *dev,
				    s64 latency_req)
{
	int i;

	for (i = 1; i < drv->state_count; i++) {
		if (dev->states_usage[i].disable)
			continue;
		if (drv->states[i].exit_latency_ns > latency_req)
			continue;
		return i;
	}
	return 0;
}

static inline int nap_get_min_valid_state(struct nap_cpu_data *d,
					  struct cpuidle_driver *drv,
					  struct cpuidle_device *dev,
					  s64 latency_req)
{
	if (unlikely(latency_req != d->cached_min_state_latency ||
		     time_after(jiffies,
				d->cached_min_state_jiffies +
				NAP_MIN_STATE_REFRESH_JIFFIES))) {
		d->cached_min_state = nap_find_min_valid_state(drv, dev,
							       latency_req);
		d->cached_min_state_latency = latency_req;
		d->cached_min_state_jiffies = jiffies;
	}
	return d->cached_min_state;
}

static inline u64 nap_compute_poll_limit(u64 sleep_length_ns,
					 u64 min_state_target_ns)
{
	u64 budget = sleep_length_ns + NAP_POLL_LIMIT_MARGIN_NS;

	return clamp_t(u64, budget,
		       NAP_POLL_LIMIT_MIN_NS,
		       min_state_target_ns);
}

static int nap_select(struct cpuidle_driver *drv,
		      struct cpuidle_device *dev,
		      bool *stop_tick)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	s64 latency_req;
	ktime_t delta_tick;
	u64 sleep_length_ns;
	int idx, min_state;

	if (unlikely(drv->state_count <= 1))
		return 0;

	latency_req = cpuidle_governor_latency_req(dev->cpu);
	sleep_length_ns = ktime_to_ns(tick_nohz_get_sleep_length(&delta_tick));
	min_state = nap_get_min_valid_state(d, drv, dev, latency_req);

	if (min_state == 0 ||
	    sleep_length_ns < drv->states[min_state].target_residency_ns) {
		if (min_state > 0)
			dev->poll_limit_ns = nap_compute_poll_limit(
				sleep_length_ns,
				drv->states[min_state].target_residency_ns);
		else
			dev->poll_limit_ns = max_t(u64, sleep_length_ns,
						   NAP_POLL_LIMIT_MIN_NS);

		*stop_tick = false;
		d->last_selected_idx = 0;
		d->short_circuited = true;
		d->stats.total_selects++;
		return 0;
	}

	d->short_circuited = false;
	idx = nap_fp_select(drv, dev, d, sleep_length_ns, latency_req);

	*stop_tick = sleep_length_ns >= TICK_NSEC &&
		     drv->states[idx].target_residency_ns >
		     RESIDENCY_THRESHOLD_NS;

	d->last_selected_idx = idx;
	d->stats.total_selects++;

	return idx;
}

static void nap_reflect(struct cpuidle_device *dev, int index)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct cpuidle_driver *drv = cpuidle_get_cpu_driver(dev);
	u64 measured_ns = dev->last_residency_ns;

	dev->last_state_idx = index;

	if (unlikely(!drv))
		return;

	if (d->short_circuited) {
		d->stats.total_residency_ns += measured_ns;
		return;
	}

	nap_history_update(d, measured_ns);

	d->last_prediction_error = d->last_predicted_ns - (s64)measured_ns;
	nap_update_external_signals(d);

	d->learn_actual_ns = measured_ns;
	d->have_sample = true;

	if (++d->learn_counter >= d->learn_interval &&
	    time_after_eq(jiffies,
			  d->last_learn_jiffies + d->learn_jiffies_min)) {
		d->learn_counter = 0;
		d->last_learn_jiffies = jiffies;
		d->needs_learn = true;
	}

	d->stats.total_residency_ns += measured_ns;
	if (index > 0 && measured_ns < drv->states[index].target_residency_ns)
		d->stats.overshoot_count++;
}

static int nap_enable(struct cpuidle_driver *drv,
		      struct cpuidle_device *dev)
{
	struct nap_cpu_data *d = per_cpu_ptr(&nap_data, dev->cpu);

	memset(d, 0, sizeof(*d));

	WRITE_ONCE(nap_cached_drv, drv);
	d->learning_rate_millths = NAP_DEFAULT_LR_MILLTHS;
	d->learn_interval = NAP_DEFAULT_INTERVAL;
	d->max_grad_norm_millths = NAP_DEFAULT_CLAMP_MILLTHS;
	d->conf_millths = NAP_DEFAULT_CONF_MILLTHS;

	d->cached_min_state_latency = S64_MIN;
	d->cached_min_state_jiffies = jiffies - NAP_MIN_STATE_REFRESH_JIFFIES;
	d->learn_jiffies_min = 1;

	d->reset_pending = true;

	return 0;
}

static void nap_disable(struct cpuidle_driver *drv,
			struct cpuidle_device *dev)
{
	WRITE_ONCE(nap_cached_drv, NULL);
}

static ssize_t stats_show(struct kobject *kobj,
			  struct kobj_attribute *attr, char *buf)
{
	int cpu, len = 0;
	u64 total_sel = 0, total_res = 0, total_under = 0, total_learn = 0;

	for_each_online_cpu(cpu) {
		struct nap_cpu_data *d = &per_cpu(nap_data, cpu);

		total_sel   += d->stats.total_selects;
		total_res   += d->stats.total_residency_ns;
		total_under += d->stats.overshoot_count;
		total_learn += d->stats.learn_count;
	}

	len += sysfs_emit_at(buf, len, "total_selects: %llu\n", total_sel);
	len += sysfs_emit_at(buf, len, "total_residency_ms: %llu\n",
			     div_u64(total_res, NSEC_PER_MSEC));
	len += sysfs_emit_at(buf, len, "overshoot_count: %llu\n", total_under);
	len += sysfs_emit_at(buf, len, "overshoot_rate_permil: %llu\n",
			     total_sel ? div64_u64(total_under * 1000, total_sel) : 0);
	len += sysfs_emit_at(buf, len, "learn_count: %llu\n", total_learn);
	return len;
}

static ssize_t learning_rate_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	int cpu = cpumask_first(cpu_online_mask);

	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).learning_rate_millths);
}

static ssize_t learning_rate_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val > 100)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learning_rate_millths = val;

	return count;
}

static ssize_t learn_interval_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	int cpu = cpumask_first(cpu_online_mask);

	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%d\n",
			  per_cpu(nap_data, cpu).learn_interval);
}

static ssize_t learn_interval_store(struct kobject *kobj,
				    struct kobj_attribute *attr,
				    const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val > 10000)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learn_interval = val;

	return count;
}

static ssize_t reset_weights_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	cpumask_var_t mask;
	int cpu;

	if (!READ_ONCE(nap_cached_drv))
		return -ENODEV;

	if (sysfs_streq(buf, "all")) {
		for_each_online_cpu(cpu)
			per_cpu(nap_data, cpu).reset_pending = true;
		pr_info("nap: weight reset scheduled for all CPUs\n");
		return count;
	}

	if (!alloc_cpumask_var(&mask, GFP_KERNEL))
		return -ENOMEM;

	if (cpulist_parse(buf, mask)) {
		free_cpumask_var(mask);
		return -EINVAL;
	}

	for_each_cpu_and(cpu, mask, cpu_online_mask)
		per_cpu(nap_data, cpu).reset_pending = true;

	pr_info("nap: weight reset scheduled for CPUs %*pbl\n",
		cpumask_pr_args(mask));
	free_cpumask_var(mask);
	return count;
}

static ssize_t reset_stats_store(struct kobject *kobj,
				 struct kobj_attribute *attr,
				 const char *buf, size_t count)
{
	int cpu;

	for_each_online_cpu(cpu)
		memset(&per_cpu(nap_data, cpu).stats, 0,
		       sizeof(struct nap_stats));

	return count;
}

static ssize_t confidence_show(struct kobject *kobj,
			       struct kobj_attribute *attr, char *buf)
{
	int cpu = cpumask_first(cpu_online_mask);

	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).conf_millths);
}

static ssize_t confidence_store(struct kobject *kobj,
				struct kobj_attribute *attr,
				const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val >= 1000)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).conf_millths = val;

	return count;
}

static ssize_t version_show(struct kobject *kobj,
			    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%s\n", CPUIDLE_NAP_VERSION);
}

static ssize_t simd_show(struct kobject *kobj,
			 struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "fixed-point\n");
}

static struct kobj_attribute version_attr        = __ATTR_RO(version);
static struct kobj_attribute simd_attr           = __ATTR_RO(simd);
static struct kobj_attribute stats_attr          = __ATTR_RO(stats);
static struct kobj_attribute learning_rate_attr  = __ATTR_RW(learning_rate);
static struct kobj_attribute learn_interval_attr = __ATTR_RW(learn_interval);
static struct kobj_attribute confidence_attr     = __ATTR_RW(confidence);
static struct kobj_attribute reset_weights_attr  = __ATTR_WO(reset_weights);
static struct kobj_attribute reset_stats_attr    = __ATTR_WO(reset_stats);

static struct attribute *nap_attrs[] = {
	&version_attr.attr,
	&simd_attr.attr,
	&stats_attr.attr,
	&learning_rate_attr.attr,
	&learn_interval_attr.attr,
	&confidence_attr.attr,
	&reset_weights_attr.attr,
	&reset_stats_attr.attr,
	NULL,
};

static const struct attribute_group nap_attr_group = {
	.attrs = nap_attrs,
};

static struct kobject *cpuidle_kobj;

static int __init nap_sysfs_init(void)
{
	struct device *dev_root;
	int ret;

	dev_root = bus_get_dev_root(&cpu_subsys);
	if (!dev_root)
		return -ENODEV;

	cpuidle_kobj = kobject_create_and_add("nap", &dev_root->kobj);
	put_device(dev_root);
	if (!cpuidle_kobj)
		return -ENOMEM;

	ret = sysfs_create_group(cpuidle_kobj, &nap_attr_group);
	if (ret) {
		kobject_put(cpuidle_kobj);
		cpuidle_kobj = NULL;
	}
	return ret;
}

static void __init nap_sysfs_exit(void)
{
	if (cpuidle_kobj) {
		sysfs_remove_group(cpuidle_kobj, &nap_attr_group);
		kobject_put(cpuidle_kobj);
		cpuidle_kobj = NULL;
	}
}

static struct cpuidle_governor nap_governor = {
	.name    = "nap",
	.rating  = 26,
	.enable  = nap_enable,
	.disable = nap_disable,
	.select  = nap_select,
	.reflect = nap_reflect,
};

static int __init nap_init(void)
{
	int ret;

	nap_selftest();

	ret = nap_sysfs_init();
	if (ret)
		pr_warn("nap: sysfs init failed: %d (continuing without sysfs)\n", ret);

	ret = cpuidle_register_governor(&nap_governor);
	if (ret) {
		pr_err("nap: register_governor failed: %d\n", ret);
		nap_sysfs_exit();
		return ret;
	}

	pr_info("%s v%s by %s registered (rating=%u)\n",
		CPUIDLE_NAP_PROGNAME, CPUIDLE_NAP_VERSION,
		CPUIDLE_NAP_AUTHOR, nap_governor.rating);
	return 0;
}
postcore_initcall(nap_init);
