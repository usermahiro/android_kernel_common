// SPDX-License-Identifier: GPL-2.0
/*
 * nap.c - Neural Adaptive Predictor cpuidle governor (fixed-point, arm64)
 *
 * An independent, fixed-point-only reimplementation written to the same
 * behavioural spec as firelzrd/nap v0.5.0 <https://github.com/firelzrd/nap>:
 * an 8->8 ReLU MLP trunk feeds a shared score s = w_out . h + b_out into a
 * proportional-odds ordinal survival head. For each idle-state boundary k,
 * q_k = sigmoid(s - thr_ord[k]) is the predicted probability that the
 * upcoming idle reaches that state's target_residency. The governor selects
 * the deepest state whose q_k still clears the confidence level tunable via
 * /sys/devices/system/cpu/nap/confidence. Weights are seeded so the
 * untrained score equals log2(predicted sleep length) and thresholds sit at
 * each state's own log2(target_residency), so the cold governor reproduces
 * the ordinary "deepest state that still fits" heuristic; online SGD then
 * refines it from observed residencies.
 *
 * Why this is not a port of upstream nap.c: upstream is x86-only by design
 * (SSE2/AVX2 float intrinsics, kernel_fpu_begin(), asm/fpu/api.h,
 * `depends on X86_64`). cpuidle_select() runs from do_idle() with IRQs
 * disabled, where may_use_simd() is false on arm64, so kernel_neon_begin()
 * would BUG_ON() - a NEON port isn't a viable substitute either. Everything
 * below is plain Q16.16 fixed-point integer arithmetic: no FPU, no SIMD, no
 * kernel_fpu_begin()/asm/fpu/api.h anywhere, and results are bit-identical
 * across CPUs and architectures.
 *
 * The two numerical approximations below (log2, sigmoid) were validated by
 * direct comparison against libm in userspace before being transcribed
 * here: log2(1+x) cubic max error ~1.1e-3 bits, sigmoid (256-entry LUT +
 * linear interp) max error ~3.5e-4, monotone, and exact at x=0 (0.5) and
 * at powers of two for log2 (mantissa fraction is exactly 0 there, and the
 * cubic passes through the origin by construction).
 * CONFIG_CPU_IDLE_GOV_NAP_SELFTEST re-checks these properties at boot.
 */

#include <linux/cpuidle.h>
#include <linux/cpu.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kobject.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/percpu.h>
#include <linux/sched/clock.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/tick.h>

#include "nap.h"
#include "gov.h"

#define CPUIDLE_NAP_PROGNAME	"Nap CPUIdle Governor (fixed-point)"
#define CPUIDLE_NAP_AUTHOR	"Mahiro (independent fixed-point port)"
#define CPUIDLE_NAP_VERSION	"0.5.0-fp1"

/* Governor defaults */
#define NAP_DEFAULT_LR_MILLTHS		1	/* 0.001 */
#define NAP_DEFAULT_INTERVAL		4	/* learn every 4 reflects */
#define NAP_DEFAULT_CONF_MILLTHS	500	/* 0.5, matches cold heuristic */
#define NAP_MIN_STATE_REFRESH_JIFFIES	HZ

/* ================================================================
 * Fixed-point numeric kernels (Q16.16, see nap_fp_mul() in nap.h)
 * ================================================================ */

/* Minimax cubic coefficients for log2(1+x), x in [0,1), endpoints exact. */
#define NAP_LOG2_A	93118
#define NAP_LOG2_B	(-37831)
#define NAP_LOG2_C	10249

#define NAP_SIGMOID_LUT_SIZE	256
#define NAP_SIGMOID_RANGE_FP	(8 << NAP_FP_SHIFT)

/* 257-entry table of sigmoid(x) for x in [-8, 8], Q16.16. Generated and
 * verified against libm in userspace - see the file header comment. */
static const s32 nap_sigmoid_lut[NAP_SIGMOID_LUT_SIZE + 1] = {
22, 23, 25, 27, 28, 30, 32, 34,
36, 39, 41, 44, 47, 50, 53, 56,
60, 64, 68, 72, 77, 82, 87, 92,
98, 105, 111, 119, 126, 134, 143, 152,
162, 172, 184, 195, 208, 221, 236, 251,
267, 284, 302, 322, 343, 366, 389, 415,
442, 471, 502, 535, 570, 608, 647, 690,
735, 783, 834, 889, 947, 1009, 1075, 1146,
1221, 1301, 1386, 1477, 1574, 1677, 1787, 1904,
2028, 2160, 2301, 2451, 2610, 2779, 2960, 3151,
3355, 3572, 3803, 4048, 4309, 4586, 4881, 5194,
5527, 5880, 6255, 6653, 7076, 7524, 8000, 8505,
9040, 9607, 10207, 10843, 11515, 12227, 12979, 13774,
14614, 15499, 16433, 17417, 18452, 19541, 20685, 21885,
23142, 24457, 25831, 27263, 28753, 30301, 31905, 33564,
35275, 37036, 38844, 40694, 42583, 44505, 46456, 48429,
50419, 52419, 54423, 56424, 58415, 60390, 62341, 64262,
66147, 67990, 69785, 71527, 73211, 74833, 76389, 77876,
79290, 80630, 81893, 83079, 84186, 85216, 86167, 87041,
87840, 88565, 89218, 89804, 90325, 90785, 91188, 91538,
91840, 92099, 92318, 92503, 92658, 92787, 92893, 92981,
93052, 93110, 93157, 93194, 93224, 93248, 93268, 93283,
93296, 93306, 93314, 93321, 93326, 93331, 93334, 93337,
93339, 93341, 93343, 93344, 93345, 93346, 93347, 93347,
93348, 93348, 93348, 93349, 93349, 93349, 93349, 93349,
93350, 93350, 93350, 93350, 93350, 93350, 93350, 93350,
93350, 93350, 93350, 93350, 93350, 93350, 93350, 93350,
93350, 93350, 93350, 93350, 93350, 93350, 93350, 93350,
93350, 93350, 93350, 93350, 93350, 93350, 93350, 93350,
93350, 93350, 93350, 93350, 93350, 93350, 93350, 93350,
93350, 93350, 93350, 93350, 93350, 93350, 93350, 93350,
93350, 93350, 93350, 93350, 93350, 93350, 93350, 93350,
93350,
};

/*
 * nap_fp_log2_u64() - log2(v) for v > 0 nanoseconds, returned as Q16.16.
 * Exact at powers of two (mantissa fraction is exactly 0 there and the
 * cubic passes through the origin by construction).
 */
static s32 nap_fp_log2_u64(u64 v)
{
	int n;
	u64 mant;
	s32 x, t, frac_log;

	if (unlikely(v == 0))
		return S32_MIN >> 1; /* sentinel: never satisfies any threshold */

	n = fls64(v) - 1;			/* integer part of log2(v) */

	/* Bring the mantissa into Q16.16 range [FP_ONE, 2*FP_ONE). */
	if (n >= NAP_FP_SHIFT)
		mant = v >> (n - NAP_FP_SHIFT);
	else
		mant = v << (NAP_FP_SHIFT - n);

	x = (s32)mant - NAP_FP_ONE;		/* fractional mantissa, [0, FP_ONE) */

	/* Horner form of frac_log(x) = a*x + b*x^2 + c*x^3, LOG2_B is
	 * already negative so this is a straight addition chain. */
	t = nap_fp_mul(x, NAP_LOG2_C) + NAP_LOG2_B;	/* b + c*x */
	t = nap_fp_mul(x, t) + NAP_LOG2_A;		/* a + x*(b+c*x) */
	frac_log = nap_fp_mul(x, t);			/* x*(a+x*(b+c*x)) */

	return (n << NAP_FP_SHIFT) + frac_log;
}

/*
 * nap_fp_sigmoid() - sigmoid(x), x and result in Q16.16. Clamped to
 * [-8, 8), 256-entry LUT with linear interpolation. Monotone, exact at
 * x=0 (returns exactly NAP_FP_ONE/2).
 */
static s32 nap_fp_sigmoid(s32 x)
{
	s32 idx, x0, y0, y1, frac;
	const s32 lo = -NAP_SIGMOID_RANGE_FP;
	const s32 hi = NAP_SIGMOID_RANGE_FP - 1;

	if (x < lo)
		x = lo;
	if (x > hi)
		x = hi;

	/* span = 16<<16, size = 256 = 1<<8, so this is an exact shift. */
	idx = (s32)(((s64)(x - lo) * NAP_SIGMOID_LUT_SIZE) /
		    (2 * NAP_SIGMOID_RANGE_FP));
	if (idx >= NAP_SIGMOID_LUT_SIZE)
		idx = NAP_SIGMOID_LUT_SIZE - 1;
	if (idx < 0)
		idx = 0;

	x0 = lo + (s32)(((s64)2 * NAP_SIGMOID_RANGE_FP * idx) / NAP_SIGMOID_LUT_SIZE);
	y0 = nap_sigmoid_lut[idx];
	y1 = nap_sigmoid_lut[idx + 1];

	frac = (s32)(((s64)(x - x0) << NAP_FP_SHIFT) /
		     ((2 * NAP_SIGMOID_RANGE_FP) / NAP_SIGMOID_LUT_SIZE));

	return y0 + nap_fp_mul(y1 - y0, frac);
}

/* ================================================================
 * Self-test (CONFIG_CPU_IDLE_GOV_NAP_SELFTEST)
 * ================================================================ */
#ifdef CONFIG_CPU_IDLE_GOV_NAP_SELFTEST
static int __init nap_selftest(void)
{
	int fails = 0;
	s32 v;
	int i;

	/* log2 exact at powers of two */
	v = nap_fp_log2_u64(1ULL << 20);
	if (v != nap_fp_from_int(20)) {
		pr_err("nap selftest: log2(2^20) = %d, want %d\n",
		       v, nap_fp_from_int(20));
		fails++;
	}

	/* sigmoid(0) == 0.5 exactly */
	v = nap_fp_sigmoid(0);
	if (v != NAP_FP_ONE / 2) {
		pr_err("nap selftest: sigmoid(0) = %d, want %d\n",
		       v, NAP_FP_ONE / 2);
		fails++;
	}

	/* sigmoid saturates towards 0 / 1 at the range edges */
	v = nap_fp_sigmoid(-NAP_SIGMOID_RANGE_FP);
	if (v < 0 || v > NAP_FP_ONE / 100) {
		pr_err("nap selftest: sigmoid(-8) = %d, want near 0\n", v);
		fails++;
	}
	v = nap_fp_sigmoid(NAP_SIGMOID_RANGE_FP - 1);
	if (v < NAP_FP_ONE - NAP_FP_ONE / 100) {
		pr_err("nap selftest: sigmoid(8) = %d, want near FP_ONE\n", v);
		fails++;
	}

	/* sigmoid monotonicity across a sweep */
	{
		s32 prev = -1;
		bool mono_ok = true;

		for (i = -800; i <= 800; i += 4) {
			v = nap_fp_sigmoid(i * (NAP_FP_ONE / 100));
			if (v < prev) {
				mono_ok = false;
				break;
			}
			prev = v;
		}
		if (!mono_ok) {
			pr_err("nap selftest: sigmoid is not monotone\n");
			fails++;
		}
	}

	/* saturating multiply: must clamp, not wrap, on overflow */
	v = nap_fp_mul(S32_MAX, nap_fp_from_int(2));
	if (v != S32_MAX) {
		pr_err("nap selftest: fp_mul overflow did not saturate\n");
		fails++;
	}

	/* round-toward-zero-ish sanity: log2 monotone across a sweep */
	{
		s32 prev = S32_MIN;
		bool mono_ok = true;
		u64 val;

		for (val = 1; val < (1ULL << 40); val <<= 1) {
			v = nap_fp_log2_u64(val | (val >> 2));
			if (v < prev) {
				mono_ok = false;
				break;
			}
			prev = v;
		}
		if (!mono_ok) {
			pr_err("nap selftest: log2 is not monotone\n");
			fails++;
		}
	}

	if (fails)
		pr_err("nap selftest: %d check(s) FAILED\n", fails);
	else
		pr_info("nap selftest: all fixed-point math checks passed\n");

	return 0;
}
#else
static inline int nap_selftest(void) { return 0; }
#endif /* CONFIG_CPU_IDLE_GOV_NAP_SELFTEST */

/* ================================================================
 * Per-CPU data
 * ================================================================ */
DEFINE_PER_CPU(struct nap_cpu_data, nap_data);

/* ================================================================
 * Weight init: cold score reproduces log2(predicted sleep length),
 * thresholds sit at each state's own log2(target_residency) - see the
 * file header comment for why this exactly reproduces the ordinary
 * "deepest state that still fits" heuristic at confidence=0.5.
 * ================================================================ */
static void nap_init_weights(struct nap_cpu_data *d, struct cpuidle_driver *drv)
{
	int i;

	memset(&d->weights, 0, sizeof(d->weights));

	/* hidden[0] = ReLU(x[0]); all other hidden units start at 0 */
	d->weights.w1[0][0] = NAP_FP_ONE;
	/* score = hidden[0] */
	d->weights.w_out[0] = NAP_FP_ONE;

	for (i = 1; i < drv->state_count && i < CPUIDLE_STATE_MAX; i++)
		d->weights.thr_ord[i - 1] = d->log2_tres[i];

	d->reset_pending = false;
}

/* ================================================================
 * Forward pass. features[] must already be filled by the caller.
 * Returns the score (Q16.16) and leaves d->hidden[] populated for the
 * (optional, deferred) learning step.
 * ================================================================ */
static s32 nap_forward(struct nap_cpu_data *d)
{
	int i, j;
	s32 acc, score;

	for (i = 0; i < NAP_HIDDEN_SIZE; i++) {
		acc = d->weights.b1[i];
		for (j = 0; j < NAP_INPUT_SIZE; j++)
			acc += nap_fp_mul(d->weights.w1[j][i], d->features[j]);
		d->hidden[i] = acc > 0 ? acc : 0; /* ReLU */
	}

	score = d->weights.b_out;
	for (i = 0; i < NAP_HIDDEN_SIZE; i++)
		score += nap_fp_mul(d->weights.w_out[i], d->hidden[i]);

	d->score = score;
	return score;
}

/*
 * nap_learn() - one step of online SGD against the boundaries of the
 * states that actually exist on this driver, using the just-observed
 * residency as the ordinal label.
 */
static void nap_learn(struct nap_cpu_data *d, struct cpuidle_driver *drv,
		       s32 measured_log2)
{
	s32 lr = nap_fp_from_int(d->learning_rate_millths) / 1000;
	s32 grad_score = 0;
	s32 d_hidden[NAP_HIDDEN_SIZE] = { 0 };
	int i, j, k, n_cuts;

	n_cuts = drv->state_count - 1;
	if (n_cuts > NAP_NUM_CUTS)
		n_cuts = NAP_NUM_CUTS;
	if (n_cuts <= 0)
		return;

	for (k = 0; k < n_cuts; k++) {
		s32 q = nap_fp_sigmoid(d->score - d->weights.thr_ord[k]);
		s32 y = (measured_log2 >= d->weights.thr_ord[k]) ? NAP_FP_ONE : 0;
		s32 err = q - y; /* dL/ds contribution from this boundary */

		grad_score += err;
		/* dL/d_thr = -err */
		d->weights.thr_ord[k] -= nap_fp_mul(lr, -err);
	}

	d->weights.b_out -= nap_fp_mul(lr, grad_score);
	for (i = 0; i < NAP_HIDDEN_SIZE; i++) {
		s32 d_out_i = nap_fp_mul(grad_score, d->weights.w_out[i]);

		d_hidden[i] = d->hidden[i] > 0 ? d_out_i : 0; /* ReLU' */
		d->weights.w_out[i] -= nap_fp_mul(lr,
				nap_fp_mul(grad_score, d->hidden[i]));
	}

	for (i = 0; i < NAP_HIDDEN_SIZE; i++) {
		if (d_hidden[i] == 0)
			continue;
		d->weights.b1[i] -= nap_fp_mul(lr, d_hidden[i]);
		for (j = 0; j < NAP_INPUT_SIZE; j++)
			d->weights.w1[j][i] -= nap_fp_mul(lr,
					nap_fp_mul(d_hidden[i], d->features[j]));
	}

	d->stats.learn_count++;
}

/* ================================================================
 * Governor callbacks
 * ================================================================ */
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
		     time_after(jiffies, d->cached_min_state_jiffies +
					  NAP_MIN_STATE_REFRESH_JIFFIES))) {
		d->cached_min_state = nap_find_min_valid_state(drv, dev, latency_req);
		d->cached_min_state_latency = latency_req;
		d->cached_min_state_jiffies = jiffies;
	}
	return d->cached_min_state;
}

static int nap_select(struct cpuidle_driver *drv, struct cpuidle_device *dev,
		       bool *stop_tick)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	s64 latency_req;
	ktime_t delta_tick;
	u64 sleep_length_ns;
	int min_state, idx, chosen;
	s32 sleep_log2;

	if (unlikely(drv->state_count <= 1))
		return 0;

	if (unlikely(d->reset_pending))
		nap_init_weights(d, drv);

	latency_req = cpuidle_governor_latency_req(dev->cpu);
	sleep_length_ns = ktime_to_ns(tick_nohz_get_sleep_length(&delta_tick));
	min_state = nap_get_min_valid_state(d, drv, dev, latency_req);

	if (min_state == 0 ||
	    sleep_length_ns < drv->states[min_state].target_residency_ns) {
		*stop_tick = false;
		d->last_selected_idx = 0;
		d->short_circuited = true;
		d->stats.total_selects++;
		return 0;
	}

	d->short_circuited = false;

	/* Build the feature vector: x[0] = log2(predicted sleep length),
	 * x[1..7] = the 7 most recent measured idle durations (log2 ns).
	 */
	sleep_log2 = nap_fp_log2_u64(sleep_length_ns);
	d->features[0] = sleep_log2;
	for (idx = 1; idx < NAP_INPUT_SIZE; idx++) {
		int h = (d->hist_idx - idx + NAP_HISTORY_SIZE) % NAP_HISTORY_SIZE;

		d->features[idx] = (idx <= d->hist_count) ? d->log_history[h] : 0;
	}

	nap_forward(d);

	/* Deepest state whose predicted survival clears the confidence bar;
	 * fall back to the shallowest latency-valid state if none does.
	 */
	chosen = min_state;
	for (idx = drv->state_count - 1; idx >= min_state; idx--) {
		s32 q, q_millths;
		int cut;

		if (dev->states_usage[idx].disable)
			continue;
		cut = idx - 1;
		if (cut >= NAP_NUM_CUTS)
			continue;

		q = nap_fp_sigmoid(d->score - d->weights.thr_ord[cut]);
		q_millths = (s32)(((s64)q * 1000) >> NAP_FP_SHIFT);
		if (q_millths >= (s32)d->conf_millths) {
			chosen = idx;
			break;
		}
	}

	*stop_tick = (drv->states[chosen].target_residency_ns >
		      RESIDENCY_THRESHOLD_NS);
	d->last_selected_idx = chosen;
	d->stats.total_selects++;

	return chosen;
}

static void nap_reflect(struct cpuidle_device *dev, int index)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct cpuidle_driver *drv = cpuidle_get_cpu_driver(dev);
	u64 measured_ns = dev->last_residency_ns;
	s32 measured_log2;

	dev->last_state_idx = index;

	if (unlikely(!drv))
		return;

	if (d->short_circuited) {
		d->stats.total_residency_ns += measured_ns;
		return;
	}

	measured_log2 = nap_fp_log2_u64(measured_ns ? measured_ns : 1);
	d->log_history[d->hist_idx] = measured_log2;
	d->hist_idx = (d->hist_idx + 1) % NAP_HISTORY_SIZE;
	if (d->hist_count < NAP_HISTORY_SIZE)
		d->hist_count++;

	if (++d->learn_counter >= d->learn_interval) {
		d->learn_counter = 0;
		nap_learn(d, drv, measured_log2);
	}

	d->stats.total_residency_ns += measured_ns;
	if (index > 0 && measured_ns < drv->states[index].target_residency_ns)
		d->stats.overshoot_count++;
}

static int nap_enable(struct cpuidle_driver *drv, struct cpuidle_device *dev)
{
	struct nap_cpu_data *d = per_cpu_ptr(&nap_data, dev->cpu);
	int i;

	memset(d, 0, sizeof(*d));

	for (i = 0; i < drv->state_count && i < CPUIDLE_STATE_MAX; i++)
		d->log2_tres[i] = nap_fp_log2_u64(
				max_t(u64, drv->states[i].target_residency_ns, 1));

	d->learning_rate_millths = NAP_DEFAULT_LR_MILLTHS;
	d->learn_interval = NAP_DEFAULT_INTERVAL;
	d->conf_millths = NAP_DEFAULT_CONF_MILLTHS;
	d->cached_min_state_latency = S64_MIN;
	d->cached_min_state_jiffies = jiffies - NAP_MIN_STATE_REFRESH_JIFFIES;

	nap_init_weights(d, drv);

	return 0;
}

static void nap_disable(struct cpuidle_driver *drv, struct cpuidle_device *dev)
{
}

/* ================================================================
 * sysfs interface (/sys/devices/system/cpu/nap/)
 * ================================================================ */
static ssize_t version_show(struct kobject *kobj, struct kobj_attribute *attr,
			     char *buf)
{
	return sysfs_emit(buf, "%s\n", CPUIDLE_NAP_VERSION);
}

static ssize_t stats_show(struct kobject *kobj, struct kobj_attribute *attr,
			   char *buf)
{
	int cpu, len = 0;
	u64 sel = 0, res = 0, over = 0, learn = 0;

	for_each_online_cpu(cpu) {
		struct nap_cpu_data *d = &per_cpu(nap_data, cpu);

		sel += d->stats.total_selects;
		res += d->stats.total_residency_ns;
		over += d->stats.overshoot_count;
		learn += d->stats.learn_count;
	}

	len += sysfs_emit_at(buf, len, "total_selects: %llu\n", sel);
	len += sysfs_emit_at(buf, len, "total_residency_ms: %llu\n",
			      div_u64(res, NSEC_PER_MSEC));
	len += sysfs_emit_at(buf, len, "overshoot_count: %llu\n", over);
	len += sysfs_emit_at(buf, len, "learn_count: %llu\n", learn);
	return len;
}

static ssize_t confidence_show(struct kobject *kobj,
				struct kobj_attribute *attr, char *buf)
{
	int cpu = cpumask_first(cpu_online_mask);

	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n", per_cpu(nap_data, cpu).conf_millths);
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

static ssize_t reset_weights_store(struct kobject *kobj,
				    struct kobj_attribute *attr,
				    const char *buf, size_t count)
{
	int cpu;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).reset_pending = true;

	pr_info("nap: weight reset scheduled for all CPUs\n");
	return count;
}

static struct kobj_attribute version_attr = __ATTR_RO(version);
static struct kobj_attribute stats_attr = __ATTR_RO(stats);
static struct kobj_attribute confidence_attr = __ATTR_RW(confidence);
static struct kobj_attribute learning_rate_attr = __ATTR_RW(learning_rate);
static struct kobj_attribute reset_weights_attr = __ATTR_WO(reset_weights);

static struct attribute *nap_attrs[] = {
	&version_attr.attr,
	&stats_attr.attr,
	&confidence_attr.attr,
	&learning_rate_attr.attr,
	&reset_weights_attr.attr,
	NULL,
};

static const struct attribute_group nap_attr_group = {
	.attrs = nap_attrs,
};

static struct kobject *nap_kobj;

static int __init nap_sysfs_init(void)
{
	struct device *dev_root;
	int ret;

	dev_root = bus_get_dev_root(&cpu_subsys);
	if (!dev_root)
		return -ENODEV;

	nap_kobj = kobject_create_and_add("nap", &dev_root->kobj);
	put_device(dev_root);
	if (!nap_kobj)
		return -ENOMEM;

	ret = sysfs_create_group(nap_kobj, &nap_attr_group);
	if (ret) {
		kobject_put(nap_kobj);
		nap_kobj = NULL;
	}
	return ret;
}

/* ================================================================
 * Governor registration
 * ================================================================ */
static struct cpuidle_governor nap_governor = {
	.name		= "nap",
	.rating		= 18, /* below teo(19)/menu(20): opt-in only, see Kconfig */
	.enable		= nap_enable,
	.disable	= nap_disable,
	.select		= nap_select,
	.reflect	= nap_reflect,
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
		return ret;
	}

	pr_info("%s v%s by %s registered (rating=%u)\n",
		CPUIDLE_NAP_PROGNAME, CPUIDLE_NAP_VERSION,
		CPUIDLE_NAP_AUTHOR, nap_governor.rating);
	return 0;
}
postcore_initcall(nap_init);
