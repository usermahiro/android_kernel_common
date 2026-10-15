// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (C) 2020 Oplus. All rights reserved.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/mm.h>
#include <linux/sched/mm.h>
#include <linux/module.h>
#include <linux/gfp.h>
#include <linux/kernel_stat.h>
#include <linux/pagemap.h>
#include <linux/page_ref.h>
#include <linux/init.h>
#include <linux/highmem.h>
#include <linux/vmpressure.h>
#include <linux/vmstat.h>
#include <linux/file.h>
#include <linux/writeback.h>
#include <linux/blkdev.h>
#include <linux/mm_inline.h>
#include <linux/backing-dev.h>
#include <linux/rmap.h>
#include <linux/topology.h>
#include <linux/cpu.h>
#include <linux/cpuset.h>
#include <linux/compaction.h>
#include <linux/notifier.h>
#include <linux/rwsem.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/freezer.h>
#include <linux/memcontrol.h>
#include <linux/delayacct.h>
#include <linux/sysctl.h>
#include <linux/oom.h>
#include <linux/prefetch.h>
#include <linux/printk.h>
#include <linux/dax.h>
#include <asm/tlbflush.h>
#include <asm/div64.h>
#include <linux/swapops.h>
#include <linux/balloon_compaction.h>
#include <linux/swap.h>
#include <linux/swapfile.h>
#include <linux/proc_fs.h>
#include <linux/time.h>
#include <linux/timer.h>
#include <linux/kernel.h>
#include <linux/namei.h>
#include <linux/fs.h>
#include <linux/statfs.h>
#include <linux/f2fs_fs.h>
#include <linux/magic.h>
#include <scsi/scsi_device.h>
#include <linux/notifier.h>
#include <linux/profile.h>
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 4, 0)
#include <linux/pagewalk.h>
#endif
#include "nandswap.h"
#include <linux/pgtable.h>
#include <linux/huge_mm.h>
#include <linux/mmap_lock.h>
#include <linux/page-flags.h>
#include <linux/sched/signal.h>
#include <linux/sched/task.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 4, 0)
#define walk_page_range_hook(mm, start, end, walk, rp) \
	walk_page_range(mm, start, end, walk, rp)
#else
#define walk_page_range_hook(mm, start, end, walk, rp) \
	walk_page_range(start, end, walk)
#endif


/*
 * 6.12 port helpers
 *
 * swap_address_space() lives in mm/swap.h (private) in 6.12, so mirror it here.
 * swapper_spaces[] is a non-static global in mm/swap_state.c.
 */
extern struct address_space *swapper_spaces[];
#ifndef SWAP_ADDRESS_SPACE_SHIFT
#define SWAP_ADDRESS_SPACE_SHIFT	14
#endif
static inline struct address_space *ns_swap_address_space(swp_entry_t entry)
{
	return &swapper_spaces[swp_type(entry)]
			      [swp_offset(entry) >> SWAP_ADDRESS_SPACE_SHIFT];
}

/*
 * isolate_lru_page() is gone in 6.12; folio_isolate_lru() is the replacement.
 * VERIFY in mm/vmscan.c: if folio_isolate_lru() returns 0 on success (int),
 * keep "!"; if it returns bool true on success, drop the "!".
 */
static inline bool ns_isolate_folio(struct folio *folio)
{
	return !folio_isolate_lru(folio);
}

#define RD_SIZE 128
#define DEBUG_TIME_INFO 0

#define NS_QUOTA_DAY_64G	5242880		/* 5G kB */
#define NS_QUOTA_DAY_128G	10485760	/* 10G kB */
#define NS_WRITE_UPPER_64G	31457280	/* 30G kB */
#define NS_WRITE_UPPER_128G	52428800	/* 50G kB */
#define NS_DATA_THRESHOLD_64G	1179648		/* 4.5G blks */
#define NS_DATA_THRESHOLD_128G	1835008		/* 7G blks */
#define NS_DATA_64G		9175040		/* 35G blks */
#define NS_DATA_128G		18350080	/* 70G blks */

#define NS_CHECK_INTERVAL	3600		/* 1 hour */
#define NS_CHECK_COUNT		12		/* at least half day */

enum {
	NS_NORMAL	= 0,			/* function is normal */
	NS_CAPACITY	= (1 << 0),		/* data capacity is not enough */
	NS_FRAGMENT	= (1 << 1),		/* frag_score is too high */
	NS_UNDISCARD	= (1 << 2),		/* undiscard_score is too high */
	NS_DEV_LIFE	= (1 << 3),		/* device life is not enough */
	NS_QUOTA	= (1 << 4),		/* nandswap write more than quota */
	NS_CHECK_ERROR	= (1 << 5),
};

struct ns_reclaim_param {
	struct vm_area_struct *vma;
	int nr_scanned;
	int nr_to_reclaim;
	int nr_reclaimed;
	int type;
	long nr_act;
	long nr_inact;
};

struct reclaim_data {
	pid_t pid;
	int prev_adj;
};

struct reclaim_info {
	struct reclaim_data rd[RD_SIZE];
	int i_idx;
	int o_idx;
	int count;
};

struct ns_swap_ratio {
	struct vm_area_struct *vma;
	unsigned long nand;
	unsigned long ram;
	unsigned long type;
};

struct ns_stat_info {
	bool life_protect;
	bool swap_limit;
	bool dev_life_end;
	struct tm tm;
	struct timer_list timer;
	struct work_struct work;
	long quota_today;
	long write_lifelong;
	long quota;
	long write_upper;
	long swap_total;
	long swap_out;
	long swap_in;
	long swap_out_pg_act;
	long swap_out_pg_inact;
	unsigned int wake_cnt;
	unsigned int frag_score;
	unsigned int undiscard_score;
	unsigned int swap_out_cnt;
	unsigned int swap_in_cnt;
	unsigned long write_total;
	unsigned long data_avail;
	unsigned long data_threshold;
	unsigned long fn_status;
	pid_t pid;
	unsigned long swap_ratio;
	unsigned long nand_type;
};

struct ns_task_struct {
	pid_t pid;
	int state;
	int type;
	int retry;
	spinlock_t lock;
	unsigned long timeout;
	struct kref kref;
};

static RADIX_TREE(ns_tree, GFP_ATOMIC);
static DEFINE_SPINLOCK(ns_tree_lock);
static struct ns_stat_info nsi;
static struct proc_dir_entry *ns_proc_root;
static struct proc_dir_entry *ns_proc_root_vnd;

struct task_struct *nswapoutd;
struct task_struct *nswapdropd;
struct task_struct *nswapind;
static struct reclaim_info out_info;
static struct reclaim_info drop_info;
static struct reclaim_info in_info;
static DEFINE_SPINLOCK(rd_lock);

bool nandswap_enable __read_mostly;

static void ns_free_task(struct kref *kref)
{
	struct ns_task_struct *ntask;

	ntask = container_of(kref, struct ns_task_struct, kref);
	kfree(ntask);
}

static inline void ns_get_task(struct ns_task_struct *ntask)
{
	kref_get(&ntask->kref);
}

static inline void ns_put_task(struct ns_task_struct *ntask)
{
	kref_put(&ntask->kref, ns_free_task);
}

static int process_notifier(struct notifier_block *self,
			unsigned long cmd, void *v)
{
	struct task_struct *task = v;
	struct ns_task_struct *ntask = NULL;

	if (!task)
		return NOTIFY_OK;

	spin_lock(&ns_tree_lock);
	ntask = radix_tree_lookup(&ns_tree, task->pid);
	if (ntask) {
		radix_tree_delete(&ns_tree, task->pid);
		ns_put_task(ntask);
	}
	spin_unlock(&ns_tree_lock);

	return NOTIFY_OK;
}

static struct notifier_block process_notifier_block = {
	.notifier_call	= process_notifier,
};

static inline struct ns_task_struct *ns_task_insert(pid_t pid, int type)
{
	struct ns_task_struct *ntask = NULL;

	ntask = kzalloc(sizeof(struct ns_task_struct),
				GFP_KERNEL);
	if (!ntask)
		goto out;

	ntask->pid = pid;
	ntask->state = NS_OUT_STANDBY;
	spin_lock_init(&ntask->lock);
	ntask->type = type;
	ntask->timeout = jiffies;
	ntask->retry = 0;
	kref_init(&ntask->kref);

	spin_lock(&ns_tree_lock);
	if (radix_tree_insert(&ns_tree, pid, ntask)) {
		ns_put_task(ntask);
		ntask = NULL;
	}
	if (ntask)
		ns_get_task(ntask);
	spin_unlock(&ns_tree_lock);

out:
	return ntask;
}

static void ns_life_protect_update(void)
{
	nsi.fn_status = (nsi.data_avail < nsi.data_threshold
				? NS_CAPACITY : NS_NORMAL) |
			(nsi.frag_score > 80 ? NS_FRAGMENT : NS_NORMAL) |
			(nsi.undiscard_score > 80 ? NS_UNDISCARD : NS_NORMAL) |
			(nsi.dev_life_end ? NS_DEV_LIFE : NS_NORMAL) |
			(nsi.swap_out - nsi.swap_total > nsi.quota_today
				? NS_QUOTA : NS_NORMAL);

	if (likely(nsi.swap_limit))
		nsi.life_protect = nsi.fn_status;
	else
		nsi.life_protect = false;
}

static inline void ns_data_check(void)
{
	struct path kpath;
	struct kstatfs st;

	if (kern_path("/data", LOOKUP_FOLLOW, &kpath))
		goto out;
	if (vfs_statfs(&kpath, &st))
		goto out;

	if (unlikely(nsi.quota == 0))
		if (st.f_blocks > NS_DATA_128G) {
			nsi.write_upper = NS_WRITE_UPPER_128G;
			nsi.quota_today = nsi.quota = NS_QUOTA_DAY_128G;
			nsi.data_threshold = NS_DATA_THRESHOLD_128G;
		}
	nsi.data_avail = st.f_bavail;

out:
	;
}

static void ns_life_ctrl_update(bool timer)
{
	unsigned long events[NR_VM_EVENT_ITEMS];
	long write_inc;
	struct tm tm;
	struct timespec64 ts;

	ns_data_check();
	all_vm_events(events);
	if (unlikely(events[PGPGOUT]/2 < nsi.write_total))
		nsi.write_total = 0;

	if (!timer)
		goto out;

	ktime_get_real_ts64(&ts);
	time64_to_tm(ts.tv_sec - sys_tz.tz_minuteswest * 60, 0, &tm);

	if (tm.tm_mday != nsi.tm.tm_mday && tm.tm_hour > 2
		&& nsi.wake_cnt >= NS_CHECK_COUNT) {
		nsi.tm = tm;
		write_inc = events[PGPGOUT]/2 - nsi.write_total;
		nsi.write_total = events[PGPGOUT]/2;
		nsi.swap_total = nsi.swap_out;
		nsi.write_lifelong = nsi.write_lifelong - write_inc +
					nsi.write_upper;
		nsi.quota_today = nsi.write_lifelong > 0
					? nsi.quota
					: nsi.quota + nsi.write_lifelong;
		nsi.wake_cnt = 0;
	}
out:
	ns_life_protect_update();
}

static void ns_life_ctrl_work(struct work_struct *work)
{
	nsi.wake_cnt++;
	ns_life_ctrl_update(true);
}

static void ns_life_ctrl_timer(struct timer_list *t)
{
	schedule_work(&nsi.work);
	nsi.timer.expires = jiffies + NS_CHECK_INTERVAL * HZ;
	add_timer(&nsi.timer);
}

static inline void ns_life_ctrl_init(void)
{
	unsigned long events[NR_VM_EVENT_ITEMS];
	struct timespec64 ts;

	memset(&nsi, 0, sizeof(nsi));
	nsi.life_protect = false;
	nsi.swap_limit = true;
	nsi.dev_life_end = false;

	all_vm_events(events);
	nsi.write_total = events[PGPGOUT]/2;

	ktime_get_real_ts64(&ts);
	time64_to_tm(ts.tv_sec - sys_tz.tz_minuteswest * 60, 0, &nsi.tm);

	INIT_WORK(&nsi.work, ns_life_ctrl_work);
	timer_setup(&nsi.timer, ns_life_ctrl_timer, 0);
	nsi.timer.expires = jiffies + NS_CHECK_INTERVAL * HZ;
	add_timer(&nsi.timer);
}

static void enqueue_reclaim_data(pid_t nr, struct reclaim_info *info)
{
	int idx;
	struct task_struct *waken_task;

	spin_lock(&rd_lock);
	if (info->count < RD_SIZE) {
		info->count++;
		idx = info->i_idx++ % RD_SIZE;
		info->rd[idx].pid = nr;
	}
	spin_unlock(&rd_lock);
	WARN_ON(info->count > RD_SIZE || info->count < 0);

	waken_task = (info == &out_info ? nswapoutd : nswapind);
	if (!waken_task)
		return;
	/* wake_up_process() is a no-op if the task is already running */
	wake_up_process(waken_task);
	if (nswapdropd)
		wake_up_process(nswapdropd);
}

static bool dequeue_reclaim_data(struct reclaim_data *data, struct reclaim_info *info)
{
	int idx;
	bool has_data = false;

	spin_lock(&rd_lock);
	if (info->count > 0) {
		has_data = true;
		info->count--;
		idx = info->o_idx++ % RD_SIZE;
		*data = info->rd[idx];
	}
	spin_unlock(&rd_lock);
	WARN_ON(info->count > RD_SIZE || info->count < 0);

	return has_data;
}

static void ns_state_check(int cur_adj, struct task_struct *task, int type)
{
	int uid = task_uid(task).val;
	struct ns_task_struct *ntask = NULL;

	if (!nandswap_enable)
		return;

	if (uid % AID_USER_OFFSET < AID_APP)
		return;

	spin_lock(&ns_tree_lock);
	ntask = radix_tree_lookup(&ns_tree, task->pid);
	if (ntask)
		ns_get_task(ntask);
	spin_unlock(&ns_tree_lock);

	if (cur_adj > 0) {
		if (!ntask)
			ntask = ns_task_insert(task->pid, type);
		if (!ntask)
			return;

		if (!time_after_eq(jiffies, ntask->timeout)) {
			ns_put_task(ntask);
			return;
		}

		spin_lock(&ntask->lock);
		ntask->timeout = jiffies + 2*HZ;

		if (ntask->state == NS_OUT_STANDBY) {
			ntask->state = NS_OUT_QUEUE;
			enqueue_reclaim_data(ntask->pid, &out_info);
		}
		spin_unlock(&ntask->lock);
	} else if (cur_adj == 0 && ntask) {
		spin_lock(&ntask->lock);
		ntask->timeout = jiffies + 2*HZ;

		if (ntask->state == NS_OUT_QUEUE) {
			ntask->state = NS_OUT_STANDBY;
		} else if ((ntask->state == NS_OUT_CACHE) ||
			   (ntask->state == NS_OUT_DONE)) {
			ntask->state = NS_IN_QUEUE;
			enqueue_reclaim_data(task->pid, &in_info);
		}
		spin_unlock(&ntask->lock);
	}

	if (ntask)
		ns_put_task(ntask);
}

static ssize_t swapin_anon(struct task_struct *task)
{
	struct mm_struct *mm;
	struct vm_area_struct *vma;
	struct vma_iterator vmi;
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	struct mm_walk walk = {};
#else
	const struct mm_walk_ops walk = {
		.pmd_entry = swapin_walk_pmd_entry,
	};
#endif
	struct ns_task_struct *ntask = NULL;
	int task_anon = 0, task_swap = 0, err = 0;

retry:
	mm = get_task_mm(task);
	if (!mm)
		goto out;

	task_anon = get_mm_counter(mm, MM_ANONPAGES);
	task_swap = get_mm_counter(mm, MM_SWAPENTS);

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	walk.mm = mm;
	walk.pmd_entry = swapin_walk_pmd_entry;
#endif

	mmap_write_lock(mm);

	vma_iter_init(&vmi, mm, 0);
	for_each_vma(vmi, vma) {
		if (vma->vm_flags & VM_NANDSWAP)
			continue;

		if (is_vm_hugetlb_page(vma))
			continue;

		if (vma->vm_file)
			continue;

		if (vma->vm_flags & VM_LOCKED)
			continue;

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
		walk.private = vma;
#endif
		err = walk_page_range_hook(mm, vma->vm_start, vma->vm_end,
					   &walk, vma);
		if (err == -1)
			break;
		vm_flags_set(vma, VM_NANDSWAP);
	}

	flush_tlb_mm(mm);
	mmap_write_unlock(mm);
	mmput(mm);
	if (err) {
		err = 0;
		goto retry;
	}
out:
	lru_add_drain();

#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	pr_info("in %s (pid %d) (size %d-%d)\n",
		task->comm, task->pid, task_anon, task_swap);
#endif

	spin_lock(&ns_tree_lock);
	ntask = radix_tree_lookup(&ns_tree, task->pid);
	if (ntask)
		ns_get_task(ntask);
	spin_unlock(&ns_tree_lock);
	if (!ntask)
		return 0;

	spin_lock(&ntask->lock);
	WARN_ON(ntask->state != NS_IN_QUEUE);
	ntask->state = NS_OUT_STANDBY;
	spin_unlock(&ntask->lock);
	ns_put_task(ntask);

	nsi.swap_in_cnt++;

	return 0;
}

static int swapind_fn(void *p)
{
	struct reclaim_data data;
	struct task_struct *task;

	set_freezable();
	for ( ; ; ) {
		while (!pm_freezing && dequeue_reclaim_data(&data, &in_info)) {
			rcu_read_lock();
			task = find_task_by_vpid(data.pid);
			if (!task) {
				rcu_read_unlock();
				continue;
			}

			get_task_struct(task);
			rcu_read_unlock();
			swapin_anon(task);
			put_task_struct(task);
		}

		set_current_state(TASK_INTERRUPTIBLE | TASK_FREEZABLE);
		schedule();

		if (kthread_should_stop())
			break;
	}

	return 0;
}

static int ns_reclaim_pte(pmd_t *pmd, unsigned long addr,
				unsigned long end, struct mm_walk *walk)
{
	struct ns_reclaim_param *rp = walk->private;
	struct vm_area_struct *vma = rp->vma;
	pte_t *start_pte, *pte, ptent;
	spinlock_t *ptl;
	struct folio *folio;
	LIST_HEAD(page_list);
	int isolated;
	int reclaimed = 0;
	int reclaim_type = rp->type;

	if (!rp->nr_to_reclaim)
		return 0;

	split_huge_pmd(vma, pmd, addr);
cont:
	isolated = 0;
	/* returns NULL if the pmd is huge/none/changed under us */
	start_pte = pte_offset_map_lock(vma->vm_mm, pmd, addr, &ptl);
	if (!start_pte)
		return 0;
	pte = start_pte;
	for (; addr != end; pte++, addr += PAGE_SIZE) {
		ptent = ptep_get(pte);
		if (!pte_present(ptent))
			continue;

		folio = vm_normal_folio(vma, addr, ptent);
		if (!folio || folio_test_large(folio))
			continue;

		/*
		 * About 11% of pages have more than 1 map_count.
		 * Only taking care of mapcount == 1 is good enough.
		 */

		if (folio_test_active(folio)) {
			rp->nr_act++;
			if (reclaim_type == NS_TYPE_NAND_ACT)
				continue;
		} else
			rp->nr_inact++;

		if (!ns_isolate_folio(folio))
			continue;

		if (folio_test_anon(folio) && !folio_test_swapbacked(folio)) {
			folio_putback_lru(folio);
			continue;
		}

		list_add(&folio->lru, &page_list);
		node_stat_mod_folio(folio, NR_ISOLATED_ANON +
				    folio_is_file_lru(folio),
				    folio_nr_pages(folio));
		isolated++;
		rp->nr_scanned++;

		if ((isolated >= SWAP_CLUSTER_MAX) || !rp->nr_to_reclaim) {
			addr += PAGE_SIZE;
			break;
		}
	}
	pte_unmap_unlock(start_pte, ptl);

	if (reclaim_type == NS_TYPE_NAND_ACT || reclaim_type == NS_TYPE_NAND_ALL)
		reclaimed = nswap_reclaim_page_list(&page_list, vma, true);

	rp->nr_reclaimed += reclaimed;
	rp->nr_to_reclaim -= reclaimed;
	if (rp->nr_to_reclaim < 0)
		rp->nr_to_reclaim = 0;

	if (rp->nr_to_reclaim && (addr != end))
		goto cont;

	cond_resched();

	return 0;
}

static ssize_t reclaim_anon(struct task_struct *task)
{
	struct mm_struct *mm;
	struct vm_area_struct *vma;
	struct vma_iterator vmi;
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	struct mm_walk reclaim_walk = {};
#else
	const struct mm_walk_ops reclaim_walk = {
		.pmd_entry = ns_reclaim_pte,
	};
#endif
	struct ns_reclaim_param rp;
	struct ns_task_struct *ntask = NULL;

#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	int task_anon = 0, task_swap = 0;
	int a_task_anon = 0, a_task_swap = 0;
#endif

	spin_lock(&ns_tree_lock);
	ntask = radix_tree_lookup(&ns_tree, task->pid);
	if (ntask)
		ns_get_task(ntask);
	spin_unlock(&ns_tree_lock);
	if (!ntask)
		return 0;

	spin_lock(&ntask->lock);
	if (ntask->state != NS_OUT_QUEUE) {
#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
		pr_info("exit reclaim\n");
#endif
		spin_unlock(&ntask->lock);
		goto out;
	}
	ntask->state = NS_OUT_CACHE;
	spin_unlock(&ntask->lock);

	mm = get_task_mm(task);
	if (!mm)
		goto out;

#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	task_anon = get_mm_counter(mm, MM_ANONPAGES);
	task_swap = get_mm_counter(mm, MM_SWAPENTS);
#endif

	rp.nr_act = 0;
	rp.nr_inact = 0;
	rp.nr_to_reclaim = INT_MAX;
	rp.nr_reclaimed = 0;
	rp.nr_scanned = 0;
	rp.type = ntask->type;

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	reclaim_walk.mm = mm;
	reclaim_walk.private = &rp;
	reclaim_walk.pmd_entry = ns_reclaim_pte;
#endif

	mmap_write_lock(mm);
	vma_iter_init(&vmi, mm, 0);
	for_each_vma(vmi, vma) {
		if (is_vm_hugetlb_page(vma))
			continue;

		if (vma->vm_file)
			continue;

		if (vma->vm_flags & VM_LOCKED)
			continue;

		if (ntask->state != NS_OUT_CACHE)
			break;

		rp.vma = vma;
		walk_page_range_hook(mm, vma->vm_start, vma->vm_end,
				     &reclaim_walk, &rp);
		vm_flags_clear(vma, VM_NANDSWAP);
		if (!rp.nr_to_reclaim)
			break;
	}

	flush_tlb_mm(mm);
	mmap_write_unlock(mm);
#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	a_task_anon = get_mm_counter(mm, MM_ANONPAGES);
	a_task_swap = get_mm_counter(mm, MM_SWAPENTS);
#endif
	mmput(mm);
#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	pr_info("out %s (pid %d) (size %d-%d to %d-%d) scan %d cache %d active %d inactive %d\n",
		task->comm, task->pid, task_anon, task_swap,
		a_task_anon, a_task_swap,
		rp.nr_scanned, rp.nr_reclaimed, rp.nr_act, rp.nr_inact);
#endif
	nsi.swap_out_cnt++;
	nsi.swap_out += rp.nr_reclaimed * 4;
	nsi.swap_out_pg_act += rp.nr_act;
	nsi.swap_out_pg_inact += rp.nr_inact;
	ns_life_protect_update();
out:
	ns_put_task(ntask);
	return 0;
}

static int swapoutd_fn(void *p)
{
	struct reclaim_data data;
	struct task_struct *task;

	set_freezable();
	for ( ; ; ) {
		while (!pm_freezing && dequeue_reclaim_data(&data, &out_info)) {
			rcu_read_lock();
			task = find_task_by_vpid(data.pid);
			if (!task) {
				rcu_read_unlock();
				continue;
			}

			get_task_struct(task);
			rcu_read_unlock();

			do {
				msleep(30);
			} while (nswapind && task_is_running(nswapind));

			reclaim_anon(task);
			enqueue_reclaim_data(task->pid, &drop_info);
			put_task_struct(task);
		}

		ns_life_ctrl_update(false);
		set_current_state(TASK_INTERRUPTIBLE | TASK_FREEZABLE);
		schedule();

		if (kthread_should_stop())
			break;
	}

	return 0;
}

static int drop_swapcache_pte(pmd_t *pmd, unsigned long start,
				unsigned long end, struct mm_walk *walk)
{
	struct ns_reclaim_param *rp = walk->private;
	struct vm_area_struct *vma = rp->vma;
	unsigned long index;
	LIST_HEAD(page_list);
	int reclaimed = 0;

	for (index = start; index != end; index += PAGE_SIZE) {
		pte_t pte, *orig_pte;
		spinlock_t *ptl;
		swp_entry_t entry;
		struct folio *folio;

		/* NULL: huge/none/bad pmd, nothing to do for this range */
		orig_pte = pte_offset_map_lock(vma->vm_mm, pmd, index, &ptl);
		if (!orig_pte)
			break;
		pte = ptep_get(orig_pte);
		pte_unmap_unlock(orig_pte, ptl);

		if (pte_present(pte) || pte_none(pte))
			continue;

		entry = pte_to_swp_entry(pte);
		if (unlikely(non_swap_entry(entry)))
			continue;

		folio = filemap_get_folio(ns_swap_address_space(entry),
					  swp_offset(entry));
		if (IS_ERR_OR_NULL(folio))
			continue;
		folio_put(folio);

		if (folio_ref_count(folio) > 1 || folio_test_large(folio))
			continue;
		if (!ns_isolate_folio(folio))
			continue;

		if (folio_test_anon(folio) && !folio_test_swapbacked(folio)) {
			folio_putback_lru(folio);
			continue;
		}

		list_add(&folio->lru, &page_list);
		node_stat_mod_folio(folio, NR_ISOLATED_ANON +
				    folio_is_file_lru(folio),
				    folio_nr_pages(folio));
		rp->nr_scanned++;
	}

	reclaimed = nswap_reclaim_page_list(&page_list, vma, false);
	rp->nr_reclaimed += reclaimed;
	rp->nr_to_reclaim -= reclaimed;

	return 0;
}

static bool drop_swapcache_task(struct task_struct *task)
{
	struct mm_struct *mm;
	struct vm_area_struct *vma;
	struct vma_iterator vmi;
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	struct mm_walk reclaim_walk = {};
#else
	const struct mm_walk_ops reclaim_walk = {
		.pmd_entry = drop_swapcache_pte,
	};
#endif
	struct ns_reclaim_param rp;
	struct ns_task_struct *ntask = NULL;
	bool retry = false;

#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	int task_anon = 0, task_swap = 0;
	int a_task_anon = 0, a_task_swap = 0;
#endif

	spin_lock(&ns_tree_lock);
	ntask = radix_tree_lookup(&ns_tree, task->pid);
	if (ntask)
		ns_get_task(ntask);
	spin_unlock(&ns_tree_lock);
	if (!ntask)
		return retry;

	spin_lock(&ntask->lock);
	if (ntask->state != NS_OUT_CACHE) {
#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
		pr_info("exit dropcache\n");
#endif
		spin_unlock(&ntask->lock);
		goto out;
	}
	ntask->state = NS_OUT_DONE;
	spin_unlock(&ntask->lock);

	mm = get_task_mm(task);
	if (!mm)
		goto out;

#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	task_anon = get_mm_counter(mm, MM_ANONPAGES);
	task_swap = get_mm_counter(mm, MM_SWAPENTS);
#endif

	rp.nr_to_reclaim = INT_MAX;
	rp.nr_reclaimed = 0;
	rp.nr_scanned = 0;

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	reclaim_walk.mm = mm;
	reclaim_walk.private = &rp;
	reclaim_walk.pmd_entry = drop_swapcache_pte;
#endif

	mmap_read_lock(mm);
	vma_iter_init(&vmi, mm, 0);
	for_each_vma(vmi, vma) {
		if (is_vm_hugetlb_page(vma))
			continue;

		if (vma->vm_file)
			continue;

		if (vma->vm_flags & VM_LOCKED)
			continue;

		if (ntask->state != NS_OUT_DONE)
			break;

		rp.vma = vma;
		walk_page_range_hook(mm, vma->vm_start, vma->vm_end,
				     &reclaim_walk, &rp);
		if (!rp.nr_to_reclaim)
			break;
	}

	flush_tlb_mm(mm);
	mmap_read_unlock(mm);
#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	a_task_anon = get_mm_counter(mm, MM_ANONPAGES);
	a_task_swap = get_mm_counter(mm, MM_SWAPENTS);
#endif
	mmput(mm);
#ifdef CONFIG_OPLUS_NANDSWAP_DEBUG
	pr_info("cache %s (pid %d) (size %d-%d to %d-%d) scan %d reclaim %d\n",
		task->comm, task->pid, task_anon, task_swap,
		a_task_anon, a_task_swap, rp.nr_scanned, rp.nr_reclaimed);
#endif
	if (rp.nr_reclaimed < rp.nr_scanned * 9 / 10 && ntask->retry < 3) {
		spin_lock(&ntask->lock);
		ntask->state = NS_OUT_CACHE;
		spin_unlock(&ntask->lock);
		ntask->retry++;
		retry = true;
	} else {
		ntask->retry = 0;
		retry = false;
	}
out:
	ns_put_task(ntask);

	return retry;
}

static int swapdropd_fn(void *p)
{
	struct reclaim_data data;
	struct task_struct *task;
	bool retry;

	set_freezable();
	for ( ; ; ) {
		while (!pm_freezing && dequeue_reclaim_data(&data, &drop_info)) {
			rcu_read_lock();
			task = find_task_by_vpid(data.pid);
			if (!task) {
				rcu_read_unlock();
				continue;
			}

			get_task_struct(task);
			rcu_read_unlock();

			do {
				msleep(30);
			} while (nswapind && task_is_running(nswapind));

			retry = drop_swapcache_task(task);
			put_task_struct(task);

			if (retry) {
				enqueue_reclaim_data(data.pid, &drop_info);
				msleep_interruptible(1000);
			}
		}

		set_current_state(TASK_INTERRUPTIBLE | TASK_FREEZABLE);
		schedule();

		if (kthread_should_stop())
			break;
	}

	return 0;
}

static inline unsigned long nandswap_type(void)
{
	unsigned long type;

	for (type = 0; type < MAX_SWAPFILES; type++)
		if (swap_info[type] && (swap_info[type]->flags & SWP_NANDSWAP) &&
			(swap_info[type]->flags & SWP_USED))
			break;

	return type;
}

static int swap_ratio_pte(pmd_t *pmd, unsigned long start,
				unsigned long end, struct mm_walk *walk)
{
	unsigned long index;
	struct ns_swap_ratio *nsr = walk->private;
	struct vm_area_struct *vma = nsr->vma;

	for (index = start; index != end; index += PAGE_SIZE) {
		pte_t pte, *orig_pte;
		spinlock_t *ptl;
		unsigned long type;

		orig_pte = pte_offset_map_lock(vma->vm_mm, pmd, index, &ptl);
		if (!orig_pte)
			return 0;
		pte = ptep_get(orig_pte);
		pte_unmap_unlock(orig_pte, ptl);

		if (pte_present(pte) || pte_none(pte))
			continue;

		type = swp_type(pte_to_swp_entry(pte));
		if (type == nsr->type)
			nsr->nand++;
		else
			nsr->ram++;
	}

	return 0;
}

static unsigned long swap_ratio_task(struct task_struct *task, unsigned long type)
{
	struct mm_struct *mm;
	struct vm_area_struct *vma;
	struct vma_iterator vmi;
	struct ns_swap_ratio nsr = {};
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	struct mm_walk walk = {};
#else
	const struct mm_walk_ops walk = {
		.pmd_entry = swap_ratio_pte,
	};
#endif

	mm = get_task_mm(task);
	if (!mm)
		goto out;

	nsr.nand = 0;
	nsr.ram = 0;
	nsr.type = type;

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	walk.mm = mm;
	walk.pmd_entry = swap_ratio_pte;
	walk.private = &nsr;
#endif

	mmap_read_lock(mm);
	vma_iter_init(&vmi, mm, 0);
	for_each_vma(vmi, vma) {
		if (is_vm_hugetlb_page(vma))
			continue;

		if (vma->vm_file)
			continue;

		if (vma->vm_flags & VM_LOCKED)
			continue;

		nsr.vma = vma;
		walk_page_range_hook(mm, vma->vm_start, vma->vm_end,
				     &walk, &nsr);
	}

	flush_tlb_mm(mm);
	mmap_read_unlock(mm);
	mmput(mm);
out:
	return nsr.nand + nsr.ram ? nsr.nand * 100 / (nsr.nand + nsr.ram) : 0;
}

static int swap_ctl_seq_show(struct seq_file *seq, void *p)
{
	seq_printf(seq, "%d %lu\n", nsi.pid, nsi.swap_ratio);
	return 0;
}

static int swap_ctl_open(struct inode *inode, struct file *file)
{
	return single_open(file, swap_ctl_seq_show, NULL);
}

static ssize_t swap_ctl_write(struct file *filp, const char __user *ubuf,
		size_t cnt, loff_t *ppos)
{
	char kbuf[16];
	char *act_str, *token;
	long value;
	int type;
	pid_t task_pid;
	struct task_struct *task;
	unsigned long nand_type;

	if (cnt > 16)
		return -EINVAL;

	memset(kbuf, 0, sizeof(kbuf));
	if (copy_from_user(kbuf, ubuf, cnt))
		return -EFAULT;
	kbuf[15] = '\0';

	act_str = strstrip(kbuf);
	token = strsep(&act_str, " 	");
	if (!token || kstrtol(token, 10, &value)) {
		pr_err("NandSwap: write [%s] pid format is invalid.\n", kbuf);
		return -EINVAL;
	}

	if (value <= 0 || value > INT_MAX) {
		pr_err("NandSwap: write [%s] is invalid.\n", kbuf);
		return -EINVAL;
	}

	task_pid = (pid_t)value;

	if (!act_str) {
		pr_err("NandSwap: write [%s] do not set reclaim type.\n", kbuf);
		return -EINVAL;
	}

	act_str = strstrip(act_str);
	if (kstrtoint(act_str, 0, &type))
		return -EINVAL;

	if (type >= NS_TYPE_END || type < 0)
		return -EINVAL;

	rcu_read_lock();
	task = find_task_by_vpid(task_pid);
	if (!task) {
		rcu_read_unlock();
		pr_err("NandSwap: can not find task of pid:%d\n", task_pid);
		return -ESRCH;
	}

	if (task != task->group_leader)
		task = task->group_leader;
	get_task_struct(task);
	rcu_read_unlock();

	if (type == NS_TYPE_FG)
		ns_state_check(0, task, type);
	else if ((type == NS_TYPE_NAND_ACT || type == NS_TYPE_NAND_ALL)
		&& nandswap_enable)
		ns_state_check(task->signal->oom_score_adj, task, type);
	else if (type == NS_TYPE_DROP) {
		nand_type = nandswap_type();
		if (nand_type < MAX_SWAPFILES)
			drop_swapcache_task(task);
	} else if (type == NS_TYPE_RATIO) {
		nand_type = nandswap_type();
		if (nand_type < MAX_SWAPFILES) {
			nsi.pid = task_pid;
			nsi.swap_ratio = swap_ratio_task(task, nand_type);
		}
	}

	put_task_struct(task);

	return cnt;
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
static const struct proc_ops swap_ctl_fops = {
	.proc_open = swap_ctl_open,
	.proc_read = seq_read,
	.proc_write = swap_ctl_write,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};
#else
static const struct file_operations swap_ctl_fops = {
	.open = swap_ctl_open,
	.read = seq_read,
	.write = swap_ctl_write,
	.llseek = seq_lseek,
	.release = single_release,
};
#endif

static int fn_enable_seq_show(struct seq_file *seq, void *p)
{
	seq_printf(seq, "%d\n", nandswap_enable);
	return 0;
}

static int fn_enable_open(struct inode *inode, struct file *file)
{
	return single_open(file, fn_enable_seq_show, NULL);
}

static ssize_t fn_enable_write(struct file *filp, const char __user *ubuf,
		size_t cnt, loff_t *ppos)
{
	char kbuf[8] = { 0 };
	ssize_t buf_size;

	if (!cnt)
		return 0;

	buf_size = min(cnt, (size_t)(sizeof(kbuf)-1));
	if (copy_from_user(kbuf, ubuf, buf_size))
		return -EFAULT;

	if (kbuf[0] == '0')
		nandswap_enable = false;
	else
		nandswap_enable = true;

	ns_data_check();
	ns_life_protect_update();

	return cnt;
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
static const struct proc_ops fn_enable_fops = {
	.proc_open = fn_enable_open,
	.proc_read = seq_read,
	.proc_write = fn_enable_write,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};
#else
static const struct file_operations fn_enable_fops = {
	.open = fn_enable_open,
	.read = seq_read,
	.write = fn_enable_write,
	.llseek = seq_lseek,
	.release = single_release,
};
#endif

static ssize_t swap_limit_write(struct file *filp, const char __user *ubuf,
	size_t cnt, loff_t *ppos)
{
	char kbuf[8] = { 0 };
	ssize_t buf_size;

	if (!cnt)
		return 0;

	buf_size = min(cnt, (size_t)(sizeof(kbuf)-1));
	if (copy_from_user(kbuf, ubuf, buf_size))
		return -EFAULT;

	if (kbuf[0] == '0')
		nsi.swap_limit = false;
	else
		nsi.swap_limit = true;

	return cnt;
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
static const struct proc_ops swap_limit_fops = {
	.proc_write = swap_limit_write,
	.proc_lseek = noop_llseek,
};
#else
static const struct file_operations swap_limit_fops = {
	.write = swap_limit_write,
};
#endif

static ssize_t dev_life_write(struct file *filp, const char __user *ubuf,
		size_t cnt, loff_t *ppos)
{
	char kbuf[8] = { 0 };
	ssize_t buf_size;

	if (!cnt)
		return 0;

	buf_size = min(cnt, (size_t)(sizeof(kbuf)-1));
	if (copy_from_user(kbuf, ubuf, buf_size))
		return -EFAULT;

	if (kbuf[0] == '0')
		nsi.dev_life_end = false;
	else
		nsi.dev_life_end = true;

	ns_life_protect_update();

	return cnt;
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
static const struct proc_ops dev_life_fops = {
	.proc_write = dev_life_write,
	.proc_lseek = noop_llseek,
};
#else
static const struct file_operations dev_life_fops = {
	.write = dev_life_write,
};
#endif

static int life_protect_seq_show(struct seq_file *seq, void *offset)
{
	seq_printf(seq, "%d\n", nsi.life_protect);
	return 0;
}

static int life_protect_open(struct inode *inode, struct file *file)
{
	return single_open(file, life_protect_seq_show, NULL);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
static const struct proc_ops life_protect_fops = {
	.proc_open = life_protect_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};
#else
static const struct file_operations life_protect_fops = {
	.open = life_protect_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};
#endif

static int stat_info_seq_show(struct seq_file *seq, void *offset)
{
	unsigned long events[NR_VM_EVENT_ITEMS];

	all_vm_events(events);

	seq_printf(seq, "ns_fn_enable: %d\n"
			"ns_fn_status: %lld\n"
			"ns_life_protect: %d\n"
			"ns_swap_limit: %d\n"
			"ns_write_lifelong: %lld\n"
			"ns_quota_today: %lld\n"
			"ns_swap_out: %u\n"
			"ns_swap_in: %u\n"
			"ns_pg_out: %lld\n"
			"ns_pg_in: %lld\n"
			"ns_pg_out_act: %lld\n"
			"ns_pg_out_inact: %lld\n"
			"pswpout: %llu\n"
			"pswpin: %llu\n"
			"ns_dev_life_end: %u\n",
			nandswap_enable,
			(long long)nsi.fn_status,
			nsi.life_protect,
			nsi.swap_limit,
			(long long)nsi.write_lifelong,
			(long long)nsi.quota_today,
			nsi.swap_out_cnt,
			nsi.swap_in_cnt,
			(long long)nsi.swap_out,
			(long long)nsi.swap_in,
			(long long)nsi.swap_out_pg_act,
			(long long)nsi.swap_out_pg_inact,
			(unsigned long long)events[PSWPOUT],
			(unsigned long long)events[PSWPIN],
			nsi.dev_life_end);
	return 0;
}

static int stat_info_open(struct inode *inode, struct file *file)
{
	return single_open(file, stat_info_seq_show, NULL);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
static const struct proc_ops stat_info_fops = {
	.proc_open = stat_info_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};
#else
static const struct file_operations stat_info_fops = {
	.open = stat_info_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};
#endif

static inline void ns_create_proc(struct proc_dir_entry *parent)
{
	if (parent) {
		proc_create("swap_ctl", 0666, parent,
				&swap_ctl_fops);
		proc_create("fn_enable", 0666, parent,
				&fn_enable_fops);
		proc_create("swap_limit", 0222, parent,
				&swap_limit_fops);
		proc_create("dev_life", 0222, parent,
				&dev_life_fops);
		proc_create("life_protect", 0444, parent,
				&life_protect_fops);
		proc_create("stat_info", 0444, parent,
				&stat_info_fops);
	}
}

static inline void ns_create_proc_dir(void)
{
	ns_proc_root = proc_mkdir("nandswap", NULL);
	ns_create_proc(ns_proc_root);
	ns_proc_root_vnd = proc_mkdir("nandswap_vnd", NULL);
	ns_create_proc(ns_proc_root_vnd);
}

static inline void ns_remove_proc(struct proc_dir_entry *parent)
{
	if (parent) {
		remove_proc_entry("stat_info", parent);
		remove_proc_entry("life_protect", parent);
		remove_proc_entry("dev_life", parent);
		remove_proc_entry("swap_limit", parent);
		remove_proc_entry("fn_enable", parent);
		remove_proc_entry("swap_ctl", parent);
		if (parent == ns_proc_root)
			remove_proc_entry("nandswap", NULL);
		else
			remove_proc_entry("nandswap_vnd", NULL);
		parent = NULL;
	}
}

static inline void ns_remove_proc_dir(void)
{
	ns_remove_proc(ns_proc_root);
	ns_remove_proc(ns_proc_root_vnd);
}

static void nandswap_stop(void)
{
	if (nswapoutd) {
		kthread_stop(nswapoutd);
		nswapoutd = NULL;
	}
	if (nswapdropd) {
		kthread_stop(nswapdropd);
		nswapdropd = NULL;
	}
	if (nswapind) {
		kthread_stop(nswapind);
		nswapind = NULL;
	}
}

static int __init nandswap_init(void)
{
	nswapoutd = kthread_run(swapoutd_fn, 0, "nswapoutd");
	if (IS_ERR(nswapoutd)) {
		pr_err("NandSwap: [%s] Failed to start nswapoutd\n", __func__);
		nswapoutd = NULL;
	}

	nswapdropd = kthread_run(swapdropd_fn, 0, "nswapdropd");
	if (IS_ERR(nswapdropd)) {
		pr_err("NandSwap: [%s] Failed to start nswapdropd\n", __func__);
		nswapdropd = NULL;
	}

	nswapind = kthread_run(swapind_fn, 0, "nswapind");
	if (IS_ERR(nswapind)) {
		pr_err("NandSwap: [%s] Failed to start nswapind\n", __func__);
		nswapind = NULL;
	}

	profile_event_register(PROFILE_TASK_EXIT, &process_notifier_block);
	ns_life_ctrl_init();
	ns_create_proc_dir();

	return 0;
}

static void __exit nandswap_exit(void)
{
	ns_remove_proc_dir();
	profile_event_unregister(PROFILE_TASK_EXIT, &process_notifier_block);
	nandswap_stop();
}

module_init(nandswap_init);
module_exit(nandswap_exit);
MODULE_LICENSE("GPL v2");
