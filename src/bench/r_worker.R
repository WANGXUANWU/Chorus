# One isolated R method run on a case: grpreg (group MCP), sparsegl, SAM, biglasso.
# Usage: Rscript r_worker.R CASE_DIR METHOD OUT_DIR [JSON_OPTIONS]
# Timing stages match py_worker.py: load, warmup (package load), fit (paths for all
# responses plus validation selection), predict. The parent process samples memory.
args <- commandArgs(trailingOnly = TRUE)
here <- normalizePath(file.path(dirname(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value = TRUE))), '..', '..'))
.libPaths(c(file.path(here, '.rlib'), normalizePath(file.path(here, '..', 'VLDB_Batched_SAM', '.rlib')), .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
case <- args[1]; method <- args[2]; out <- args[3]
opts <- if (length(args) > 3) fromJSON(args[4]) else list()
dir.create(out, recursive = TRUE, showWarnings = FALSE)
stage <- function(s) write_json(list(stage = s, time = as.numeric(Sys.time())), file.path(out, 'stage.json'), auto_unbox = TRUE)
el <- function() proc.time()[['elapsed']]
t_all <- el()
res <- list(method = method)
readmat <- function(kind, name, meta) {
  n <- meta[[paste0('n_', name)]]
  k <- switch(kind, X = meta$D, y = meta$q, xraw = meta$p_sam)
  con <- file(file.path(case, paste0(kind, '_', name, '.f64')), 'rb')
  on.exit(close(con))
  matrix(readBin(con, 'double', n * k), n, k)
}
writebin <- function(m, f) { con <- file(file.path(out, f), 'wb'); writeBin(as.double(m), con); close(con) }
res <- tryCatch({
  stage('load'); t0 <- el()
  meta <- fromJSON(file.path(case, 'meta.json'))
  groups <- meta$groups; G <- nrow(groups); sizes <- groups[, 2] - groups[, 1]
  K <- if (!is.null(meta$n_lambda)) meta$n_lambda else 30
  ratio <- if (!is.null(meta$lambda_min_ratio)) meta$lambda_min_ratio else 0.02
  factors <- exp(seq(log(1), log(ratio), length.out = K))
  yt <- readmat('y', 'train', meta); yv <- readmat('y', 'val', meta); yte_n <- meta$n_test
  if (!is.null(opts$resp_range)) { rr <- (opts$resp_range[1] + 1):opts$resp_range[2]; yt <- yt[, rr, drop = FALSE]; yv <- yv[, rr, drop = FALSE] }
  if (!is.null(opts$resp_list)) { rr <- as.integer(opts$resp_list) + 1L; yt <- yt[, rr, drop = FALSE]; yv <- yv[, rr, drop = FALSE] }
  q <- ncol(yt)
  disk <- isTRUE(opts$disk)
  if (method == 'SAM') {
    xt <- readmat('xraw', 'train', meta); xv <- readmat('xraw', 'val', meta); xte <- readmat('xraw', 'test', meta)
    # SAM places spline knots at training quantiles and fails when ties make them coincide
    # (for example rainfall, mostly zero); a deterministic jitter of 1e-6 standard deviations
    # separates tied values without changing the data in any material way.
    # Only columns in which one value covers more than 5% of training rows are jittered.
    set.seed(2027)
    s <- apply(xt, 2, sd); s[!is.finite(s) | s == 0] <- 1
    tied <- apply(xt, 2, function(v) max(table(v)) / length(v) > 0.05)
    s[!tied] <- 0
    jit <- function(x) x + sweep(matrix(runif(length(x), -1, 1), nrow(x)), 2, 1e-6 * s, '*')
    xt <- jit(xt); xv <- jit(xv); xte <- jit(xte)
    res_tied <- sum(tied)
  } else if (!(method == 'biglasso' && disk)) {
    xt <- readmat('X', 'train', meta); xv <- readmat('X', 'val', meta); xte <- readmat('X', 'test', meta)
  } else {
    xv <- readmat('X', 'val', meta); xte <- readmat('X', 'test', meta)
  }
  load_seconds <- el() - t0
  stage('warmup'); t0 <- el()
  pkg <- switch(method, grpreg = 'grpreg', sparsegl = 'sparsegl', SAM = 'SAM', biglasso = 'biglasso')
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  warmup_seconds <- el() - t0
  stage('fit'); t0 <- el()
  grp <- rep(seq_len(G), sizes)
  pred <- matrix(0, yte_n, q); entry <- matrix(K, G, q); coef <- NULL
  valpath <- matrix(NA_real_, K, q); support <- matrix(0L, G, q)
  save_vp <- isTRUE(opts$save_val_pred)
  vres <- if (save_vp) array(0, c(nrow(yv), K, q)) else NULL   # validation residuals along the path
  # test predictions and active groups at every level (matched-size comparisons, amendment 22)
  save_tp <- isTRUE(opts$save_test_path)
  if (save_tp) { tpath <- array(0, c(yte_n, K, q)); apath <- array(0L, c(G, K, q)) }
  fillK <- function(P) if (ncol(P) < K) cbind(P, matrix(P[, ncol(P)], nrow(P), K - ncol(P))) else P
  first_nz <- function(B) {   # B: D x K coefficients -> entry index per group (0-based)
    nzg <- apply(B, 2, function(b) tapply(abs(b), grp, max) > 0)
    e <- apply(nzg, 1, function(z) if (any(z)) which(z)[1] - 1 else K)
    as.integer(e)
  }
  if (method %in% c('grpreg', 'sparsegl')) {
    coef <- matrix(0, meta$D, q)
    n <- nrow(xt)
    for (r in seq_len(q)) {
      if (method == 'grpreg') {
        corr <- crossprod(xt, yt[, r]) / n
        lmax <- max(tapply(corr^2, grp, sum)^.5 / sqrt(sizes))
        fit <- grpreg(xt, yt[, r], group = grp, penalty = 'grMCP', family = 'gaussian',
                      lambda = factors * lmax, eps = 1e-6, max.iter = 1e6, warn = FALSE)
        B <- as.matrix(fit$beta[-1, , drop = FALSE]); a0 <- fit$beta[1, ]
      } else {
        fit <- sparsegl(xt, yt[, r], group = grp, family = 'gaussian', nlambda = K,
                        lambda.factor = ratio, asparse = 0.05, intercept = FALSE,
                        standardize = FALSE, eps = 1e-6, maxit = 3e8)
        B <- as.matrix(fit$beta); a0 <- rep(0, ncol(B))
      }
      if (ncol(B) < K) B <- cbind(B, matrix(B[, ncol(B)], nrow(B), K - ncol(B)))
      if (length(a0) < K) a0 <- c(a0, rep(a0[length(a0)], K - length(a0)))
      pv <- xv %*% B + matrix(a0, nrow(xv), K, byrow = TRUE)
      mse <- colMeans((yv[, r] - pv)^2); valpath[, r] <- mse
      if (save_vp) vres[, , r] <- yv[, r] - pv
      if (save_tp) {
        tpath[, , r] <- xte %*% B + matrix(a0, yte_n, K, byrow = TRUE)
        apath[, , r] <- apply(B, 2, function(b) as.integer(tapply(abs(b), grp, max) > 0))
      }
      kb <- which.min(mse)
      coef[, r] <- B[, kb]
      pred[, r] <- xte %*% B[, kb] + a0[kb]
      entry[, r] <- first_nz(B)
      support[, r] <- as.integer(tapply(abs(B[, kb]), grp, max) > 0)
    }
  } else if (method == 'SAM') {
    pdeg <- if (!is.null(opts$sam_p)) as.integer(opts$sam_p) else 3L
    for (r in seq_len(q)) {
      fit <- samQL(xt, yt[, r], p = pdeg, nlambda = K, lambda.min.ratio = ratio)
      pv <- predict(fit, xv)$values
      kk <- ncol(pv)
      mse <- colMeans((yv[, r] - pv)^2); valpath[seq_len(kk), r] <- mse
      if (save_vp) vres[, , r] <- yv[, r] - fillK(pv)
      kb <- which.min(mse)
      pred[, r] <- predict(fit, xte)$values[, kb]
      W <- fit$w                                 # (p_sam * pdeg) x K
      fg <- rep(seq_len(meta$p_sam), each = pdeg)
      nzf <- apply(W, 2, function(w) tapply(abs(w), fg, max) > 0)   # p_sam x K
      # SAM groups are raw features; map to the case groups of the same feature
      tf <- meta$term_features + 1
      for (g in seq_len(G)) {
        z <- nzf[tf[g], ]
        entry[g, r] <- if (any(z)) which(z)[1] - 1 else K
        support[g, r] <- as.integer(nzf[tf[g], kb])
      }
      if (save_tp) {
        tpath[, , r] <- fillK(predict(fit, xte)$values)
        apath[, , r] <- fillK(nzf * 1L)[tf, , drop = FALSE]
      }
    }
  } else if (method == 'biglasso') {
    if (disk) {
      xb <- attach.big.matrix(file.path(case, 'X_train_big.desc'))
    } else {
      xb <- as.big.matrix(xt)
    }
    # the multi-task family needs two or more responses; one response is a plain lasso
    fit <- biglasso(xb, if (q > 1) yt else yt[, 1], family = if (q > 1) 'mgaussian' else 'gaussian',
                    nlambda = K, lambda.min = ratio, screen = 'Adaptive', ncores = 1, eps = 1e-6, max.iter = 1e6)
    kk <- length(fit$lambda)
    coefs <- if (q > 1) fit$beta else list(fit$beta)   # list of q, each (D + 1) x kk
    mse_k <- matrix(sapply(seq_len(kk), function(k) {
      P <- matrix(sapply(seq_len(q), function(r) coefs[[r]][1, k] + as.vector(xv %*% coefs[[r]][-1, k])), ncol = q)
      colMeans((yv - P)^2)
    }), nrow = q)                                 # q x kk
    valpath[seq_len(kk), ] <- t(mse_k)
    if (save_vp) for (k in seq_len(K)) { kk2 <- min(k, kk)
      vres[, k, ] <- yv - matrix(sapply(seq_len(q), function(r) coefs[[r]][1, kk2] + as.vector(xv %*% coefs[[r]][-1, kk2])), ncol = q) }
    if (save_tp) for (k in seq_len(K)) { kk2 <- min(k, kk)
      for (r in seq_len(q)) { b <- as.vector(coefs[[r]][, kk2])
        tpath[, k, r] <- b[1] + as.vector(xte %*% b[-1])
        apath[, k, r] <- as.integer(tapply(abs(b[-1]), grp, max) > 0) } }
    kb <- which.min(colMeans(mse_k))              # one shared level for the joint model
    coef <- matrix(0, meta$D, q)
    for (r in seq_len(q)) {
      B <- as.matrix(coefs[[r]][-1, , drop = FALSE])
      coef[, r] <- B[, kb]
      pred[, r] <- coefs[[r]][1, kb] + as.vector(xte %*% B[, kb])
      entry[, r] <- first_nz(B)
      support[, r] <- as.integer(tapply(abs(B[, kb]), grp, max) > 0)
    }
  }
  fit_seconds <- el() - t0
  stage('serialize')
  writebin(pred, 'pred.f64'); writebin(entry, 'entry.f64'); writebin(valpath, 'val_path.f64')
  writebin(support, 'support.f64')
  if (save_vp) { con <- file(file.path(out, 'val_resid.f32'), 'wb'); writeBin(as.numeric(vres), con, size = 4); close(con) }
  if (save_tp) {
    con <- file(file.path(out, 'test_path.f32'), 'wb'); writeBin(as.numeric(tpath), con, size = 4); close(con)
    con <- file(file.path(out, 'active_path.u8'), 'wb'); writeBin(as.raw(apath), con); close(con)
  }
  if (!is.null(coef)) writebin(coef, 'coef.f64')
  list(method = method, status = 'ok', load_seconds = load_seconds, warmup_seconds = warmup_seconds,
       fit_seconds = fit_seconds, predict_seconds = 0, version = as.character(packageVersion(pkg)),
       q = q, G = G, K = K, n_val = nrow(yv), has_coef = !is.null(coef), sam_p = if (method == 'SAM') pdeg else NULL)
}, error = function(e) list(method = method, status = 'error', error = conditionMessage(e)))
res$worker_seconds <- el() - t_all
write_json(res, file.path(out, 'result.json'), auto_unbox = TRUE, pretty = TRUE, digits = 16)
stage('done')
cat(toJSON(res, auto_unbox = TRUE), '\n')
