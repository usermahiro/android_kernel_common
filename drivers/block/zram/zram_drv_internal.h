/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef _ZRAM_DRV_INTERNAL_H_
#define _ZRAM_DRV_INTERNAL_H_

#include <linux/spinlock.h>
#include <linux/bitops.h>
#include "zram_drv.h"

#define zram_slot_lock(zram, index) \
	spin_lock(&(zram)->table[(index)].lock)

#define zram_slot_unlock(zram, index) \
	spin_unlock(&(zram)->table[(index)].lock)

#define zram_slot_trylock(zram, index) \
	spin_trylock(&(zram)->table[(index)].lock)

#define init_done(zram) ((zram)->disksize)

#define dev_to_zram(dev) ((struct zram *)dev_to_disk(dev)->private_data)

#define zram_get_handle(zram, index) ((zram)->table[(index)].handle)

#define zram_set_handle(zram, index, handle_val) \
	((zram)->table[(index)].handle = (handle_val))

#define zram_get_element(zram, index) ((zram)->table[(index)].element)

#define zram_set_element(zram, index, elem) \
	((zram)->table[(index)].element = (elem))

#define zram_test_flag(zram, index, flag) \
	((zram)->table[(index)].flags & BIT(flag))

#define zram_set_flag(zram, index, flag) \
	((zram)->table[(index)].flags |= BIT(flag))

#define zram_clear_flag(zram, index, flag) \
	((zram)->table[(index)].flags &= ~BIT(flag))

#define zram_get_obj_size(zram, index) \
	((zram)->table[(index)].flags & (BIT(ZRAM_FLAG_SHIFT) - 1))

#define zram_set_obj_size(zram, index, size) do {\
	unsigned int flags = (zram)->table[(index)].flags >> ZRAM_FLAG_SHIFT; \
	(zram)->table[(index)].flags = (flags << ZRAM_FLAG_SHIFT) | (size); \
} while (0)

#endif /* _ZRAM_DRV_INTERNAL_H_ */
