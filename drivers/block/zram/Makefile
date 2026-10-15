# SPDX-License-Identifier: GPL-2.0-only

zram-y	:=	zcomp.o zram_drv.o

zram-$(CONFIG_ZRAM_BACKEND_LZO)		+= backend_lzorle.o backend_lzo.o
zram-$(CONFIG_ZRAM_BACKEND_LZ4)		+= backend_lz4.o
zram-$(CONFIG_ZRAM_BACKEND_LZ4HC)	+= backend_lz4hc.o
zram-$(CONFIG_ZRAM_BACKEND_ZSTD)	+= backend_zstd.o
zram-$(CONFIG_ZRAM_BACKEND_DEFLATE)	+= backend_deflate.o
zram-$(CONFIG_ZRAM_BACKEND_842)		+= backend_842.o

obj-$(CONFIG_ZRAM)	+=	zram.o
zram-$(CONFIG_HYBRIDSWAP) += hybridswap/hybridmain.o
zram-$(CONFIG_HYBRIDSWAP_SWAPD) += hybridswap/hybridswapd.o
zram-$(CONFIG_HYBRIDSWAP_CORE) += hybridswap/hybridswap.o

CFLAGS_hybridswap/hybridswap.o := -Wno-error=format
CFLAGS_hybridswap/hybridmain.o := -Wno-error=format
CFLAGS_hybridswap/hybridswapd.o := -Wno-error=format
zram-$(CONFIG_ZRAM_BACKEND_ZSTDN)	+= backend_zstdn.o
