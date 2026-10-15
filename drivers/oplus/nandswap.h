/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Copyright (C) 2020 Oplus. All rights reserved.
 */

#ifndef _NAND_SWAP_H
#define _NAND_SWAP_H

#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/pagewalk.h>

/*
 * 6.12: bit 39 (0x8000000000) is VM_ALLOW_ANY_UNCACHED on arm64, so the old
 * value collides. VERIFY this bit is unused in include/linux/mm.h of your tree
 * (look at the VM_*_BIT defines) before relying on it.
 */
#define VM_NANDSWAP_BIT	45
#define VM_NANDSWAP	(1UL << VM_NANDSWAP_BIT)	/* swapin mark */

#define AID_APP		10000		/* first app user */
#define AID_USER_OFFSET	100000		/* offset for uid ranges for each user */

enum {
	NS_OUT_STANDBY,
	NS_OUT_QUEUE,
	NS_OUT_CACHE,
	NS_OUT_DONE,
	NS_IN_QUEUE,
};

enum {
	NS_TYPE_FG,
	NS_TYPE_NAND_ACT,
	NS_TYPE_NAND_ALL,
	NS_TYPE_DROP,
	NS_TYPE_RATIO,
	NS_TYPE_END
};

extern struct task_struct *nswapoutd;
extern int swapin_walk_pmd_entry(pmd_t *pmd, unsigned long start,
				 unsigned long end, struct mm_walk *walk);
extern unsigned long nswap_reclaim_page_list(struct list_head *page_list,
					     struct vm_area_struct *vma, bool scan);

static inline bool current_is_nswapoutd(void)
{
	return current == nswapoutd;
}
#endif /* _NAND_SWAP_H */
