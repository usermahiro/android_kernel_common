// SPDX-License-Identifier: GPL-2.0-or-later

#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/zstd.h>

#include "backend_zstdn.h"

struct zstdn_params {
	zstd_parameters cprm;
	bool is_oplus;
};

struct zstdn_ctx {
	zstd_cctx *cctx;
	zstd_dctx *dctx;
	void *cctx_mem;
	void *dctx_mem;
};

static void zstdn_release_params(struct zcomp_params *params)
{
	kfree(params->drv_data);
	params->drv_data = NULL;
}

static int zstdn_setup_common(struct zcomp_params *params, bool is_oplus)
{
	struct zstdn_params *zp;

	zp = kzalloc(sizeof(*zp), GFP_KERNEL);
	if (!zp)
		return -ENOMEM;

	zp->is_oplus = is_oplus;
	params->drv_data = zp;

	if (params->level == ZCOMP_PARAM_NO_LEVEL)
		params->level = is_oplus ? -3 : -1;

	zp->cprm = zstd_get_params(params->level, PAGE_SIZE);

	if (zp->cprm.cParams.windowLog > 12)
		zp->cprm.cParams.windowLog = 12;

	if (is_oplus) {
		if (zp->cprm.cParams.chainLog > 12)
			zp->cprm.cParams.chainLog = 12;
		if (zp->cprm.cParams.hashLog > 12)
			zp->cprm.cParams.hashLog = 12;
		if (zp->cprm.cParams.minMatch < 4)
			zp->cprm.cParams.minMatch = 4;
	} else {
		if (zp->cprm.cParams.chainLog > 14)
			zp->cprm.cParams.chainLog = 14;
		if (zp->cprm.cParams.hashLog > 14)
			zp->cprm.cParams.hashLog = 14;
	}

	return 0;
}

static int zstdn_setup_params(struct zcomp_params *params)
{
	return zstdn_setup_common(params, false);
}

static int zstdn_o_setup_params(struct zcomp_params *params)
{
	return zstdn_setup_common(params, true);
}

static void zstdn_destroy(struct zcomp_ctx *ctx)
{
	struct zstdn_ctx *zctx = ctx->context;

	if (!zctx)
		return;

	vfree(zctx->cctx_mem);
	vfree(zctx->dctx_mem);
	kfree(zctx);
	ctx->context = NULL;
}

static int zstdn_create(struct zcomp_params *params, struct zcomp_ctx *ctx)
{
	struct zstdn_params *zp = params->drv_data;
	struct zstdn_ctx *zctx;
	size_t sz;

	if (!zp || params->dict_sz)
		return -EINVAL;

	zctx = kzalloc(sizeof(*zctx), GFP_KERNEL);
	if (!zctx)
		return -ENOMEM;

	ctx->context = zctx;

	sz = zstd_cctx_workspace_bound(&zp->cprm.cParams);
	zctx->cctx_mem = vzalloc(sz);
	if (!zctx->cctx_mem)
		goto error;

	zctx->cctx = zstd_init_cctx(zctx->cctx_mem, sz);
	if (!zctx->cctx)
		goto error;

	sz = zstd_dctx_workspace_bound();
	zctx->dctx_mem = vzalloc(sz);
	if (!zctx->dctx_mem)
		goto error;

	zctx->dctx = zstd_init_dctx(zctx->dctx_mem, sz);
	if (!zctx->dctx)
		goto error;

	return 0;

error:
	zstdn_destroy(ctx);
	return -ENOMEM;
}

static int zstdn_compress(struct zcomp_params *params, struct zcomp_ctx *ctx,
			  struct zcomp_req *req)
{
	struct zstdn_params *zp = params->drv_data;
	struct zstdn_ctx *zctx = ctx->context;
	size_t ret;

	ret = zstd_compress_cctx(zctx->cctx, req->dst, req->dst_len,
				 req->src, req->src_len, &zp->cprm);
	if (zstd_is_error(ret))
		return -EINVAL;

	req->dst_len = ret;
	return 0;
}

static int zstdn_decompress(struct zcomp_params *params, struct zcomp_ctx *ctx,
			    struct zcomp_req *req)
{
	struct zstdn_ctx *zctx = ctx->context;
	size_t ret;

	ret = zstd_decompress_dctx(zctx->dctx, req->dst, req->dst_len,
				   req->src, req->src_len);
	if (zstd_is_error(ret))
		return -EINVAL;

	return 0;
}

const struct zcomp_ops backend_zstdn = {
	.compress	= zstdn_compress,
	.decompress	= zstdn_decompress,
	.create_ctx	= zstdn_create,
	.destroy_ctx	= zstdn_destroy,
	.setup_params	= zstdn_setup_params,
	.release_params	= zstdn_release_params,
	.name		= "zstdn",
};

const struct zcomp_ops backend_zstdn_o = {
	.compress	= zstdn_compress,
	.decompress	= zstdn_decompress,
	.create_ctx	= zstdn_create,
	.destroy_ctx	= zstdn_destroy,
	.setup_params	= zstdn_o_setup_params,
	.release_params	= zstdn_release_params,
	.name		= "zstdn_o",
};
