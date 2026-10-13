#!/usr/bin/env Rscript
# Fit GammaGateR on one cell x marker intensity table and write a raw-scale
# threshold per marker. The threshold is the largest raw intensity whose
# posterior is still below PROB, so the pipeline rule `X > threshold` matches
# posterior >= PROB.
#
# Usage:
#   Rscript scripts/gammagater_fit.R \
#     --input   output/gammagater_input.csv \
#     --summary output/gating_thresholds_gammagater.csv \
#     --slide   SLIDE-045 \
#     --prob    0.5 \
#     --boundaries output/gammagater_boundaries.json   # optional

# ~/.Rprofile prepends ~/R/.../4.6. Packages there are built for R 4.6
# (data.table needs R_duplicateAsResizable) and will not load under this
# env's R 4.4. Keep the search path on the env library only.
.libPaths(file.path(R.home(), "library"))

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(GammaGateR)
})

parse_args <- function(args) {
  out <- list(prob = "0.5", slide = "SLIDE-045", boundaries = "")
  i <- 1L
  while (i <= length(args)) {
    key <- sub("^--", "", args[[i]])
    if (i == length(args) || startsWith(args[[i + 1L]], "--")) {
      stop("Missing value for --", key)
    }
    out[[key]] <- args[[i + 1L]]
    i <- i + 2L
  }
  if (is.null(out$input) || is.null(out$summary)) {
    stop("Required: --input and --summary")
  }
  out
}

norm_marker <- function(x) {
  mu <- mean(x[is.finite(x) & x > 0], na.rm = TRUE)
  if (!is.finite(mu) || mu <= 0) mu <- 1
  list(mu = mu, y = log10(1 + x / mu))
}

boundary_list <- function(markers, path) {
  if (!nzchar(path) || !file.exists(path)) {
    return(NULL)
  }
  spec <- fromJSON(path, simplifyVector = FALSE)
  if (length(spec) == 0) {
    return(NULL)
  }
  lapply(markers, function(m) {
    b <- spec[[m]]
    if (is.null(b)) {
      return(NULL)
    }
    mat <- matrix(as.numeric(unlist(b)), nrow = 2, ncol = 2, byrow = TRUE)
    if (any(!is.finite(mat)) || any(mat < 0 | mat > 1)) {
      stop("Quantile boundary for ", m, " must be a 2x2 matrix with values in [0, 1]")
    }
    mat
  })
}

component_mode <- function(tab, comp) {
  if (is.null(tab) || !all(c("alpha", "beta") %in% rownames(tab))) {
    return(NA_real_)
  }
  if (!comp %in% colnames(tab)) {
    return(NA_real_)
  }
  a <- tab["alpha", comp]
  b <- tab["beta", comp]
  if (!is.finite(a) || !is.finite(b)) {
    return(NA_real_)
  }
  (a - 1) * b
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
prob_cut <- as.numeric(args$prob)
if (!is.finite(prob_cut) || prob_cut <= 0 || prob_cut >= 1) {
  stop("--prob must be between 0 and 1")
}

raw <- as.data.frame(fread(args$input))
if (!"cell_id" %in% names(raw)) {
  stop("Input CSV must contain a cell_id column")
}
markers <- setdiff(names(raw), "cell_id")
if (length(markers) == 0) {
  stop("Input CSV has no marker columns")
}

cat(sprintf("Cells: %s   Markers: %d   Slide: %s\n",
            format(nrow(raw), big.mark = ","), length(markers), args$slide))

normed <- data.frame(row.names = seq_len(nrow(raw)))
norm_mean <- setNames(numeric(length(markers)), markers)
for (m in markers) {
  x <- as.numeric(raw[[m]])
  x[!is.finite(x)] <- NA_real_
  tr <- norm_marker(x)
  norm_mean[[m]] <- tr$mu
  normed[[m]] <- tr$y
  raw[[m]] <- x
}

qbounds <- boundary_list(markers, args$boundaries)
if (is.null(qbounds)) {
  cat("Quantile boundaries: none (unconstrained fit)\n")
} else {
  constrained <- markers[vapply(qbounds, Negate(is.null), logical(1))]
  cat("Quantile boundaries on:", paste(constrained, collapse = ", "), "\n")
}

fit <- groupGammaGateR(
  normed,
  slide = rep(args$slide, nrow(normed)),
  qboundaryMarkers = qbounds,
  n.cores = 1
)
slide_fit <- fit[[args$slide]]
if (is.null(slide_fit)) {
  stop("GammaGateR returned no fit for slide ", args$slide)
}

align_names <- function(obj, markers) {
  if (is.null(obj)) {
    return(obj)
  }
  if (is.data.frame(obj)) {
    missing <- setdiff(markers, names(obj))
    if (length(missing) > 0) {
      stop("Fit columns missing markers: ", paste(missing, collapse = ", "))
    }
    return(obj)
  }
  if (is.list(obj) && is.null(names(obj)) && length(obj) == length(markers)) {
    names(obj) <- markers
  }
  obj
}
slide_fit$expressionZ <- align_names(slide_fit$expressionZ, markers)
slide_fit$params <- align_names(slide_fit$params, markers)
slide_fit$fit <- align_names(slide_fit$fit, markers)

rows <- vector("list", length(markers))
for (i in seq_along(markers)) {
  m <- markers[[i]]
  z <- slide_fit$expressionZ[[m]]
  x <- raw[[m]]
  ok <- is.finite(z) & is.finite(x)
  pos <- ok & z >= prob_cut
  neg <- ok & z < prob_cut

  ft <- slide_fit$fit[[m]]
  converged <- if (is.list(ft) && !is.null(ft$convergence)) isTRUE(ft$convergence) else FALSE
  message <- if (is.list(ft) && !is.null(ft$message)) as.character(ft$message) else ""

  pt <- slide_fit$params[[m]]
  tab <- NULL
  if (is.matrix(pt)) {
    tab <- pt
  } else if (is.list(pt) && length(pt) >= 1) {
    tab <- pt[[1]]
  }
  lambda_pos <- if (!is.null(tab) && "lambda" %in% rownames(tab) && "comp2" %in% colnames(tab)) {
    tab["lambda", "comp2"]
  } else {
    NA_real_
  }

  if (!converged || !any(pos)) {
    thr <- NA_real_
  } else if (!any(neg)) {
    thr <- 0
  } else {
    thr <- max(x[neg])
  }

  rows[[i]] <- data.frame(
    marker = m,
    threshold = thr,
    prob_cut = prob_cut,
    positive_rate = if (any(ok)) mean(pos[ok]) else NA_real_,
    converged = converged,
    lambda_pos = as.numeric(lambda_pos),
    mode_neg = component_mode(tab, "comp1"),
    mode_pos = component_mode(tab, "comp2"),
    norm_mean = norm_mean[[m]],
    n_cells = sum(ok),
    message = message,
    stringsAsFactors = FALSE
  )
  cat(sprintf("  %-8s  converged=%-5s  threshold=%s  positive=%s\n",
              m, converged,
              if (is.finite(thr)) sprintf("%.4g", thr) else "NA",
              if (is.finite(rows[[i]]$positive_rate)) sprintf("%.3f", rows[[i]]$positive_rate) else "NA"))
}

summary <- rbindlist(rows)
fwrite(summary, args$summary)
cat("Wrote ", args$summary, "\n", sep = "")
