# ============================================================================
#   LONGITUDINAL K‑MEANS PF‑TRAJECTORY CLUSTERING  (MIMIC)  –  K = 2 … 6
#   • PF ratios kept in **mmHg** (no unit conversion).
#   • **No imputation** – missing values remain NA and are accepted by {kml}.
#   • Evaluation metrics now NA‑robust:
#       – Average **Gower silhouette** (via cluster::daisy)  
#       – **Adjusted WSS** (each squared deviation scaled by # observed dims)  
#       – (Optional) Gap statistic based on the adjusted WSS
#   • Artefacts: class sizes, PCA/MDS (complete rows only), mean trajectories.
# ============================================================================

# ── 0 · PACKAGES ────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(dplyr);          library(tidyr);       library(purrr)
  library(readr);          library(lubridate);   library(longitudinalData)
  library(kml);            library(cluster);     library(ggplot2)
  library(tibble)
})

# ── 1 · CONFIG ──────────────────────────────────────────────────────────────
run_date <- format(Sys.Date(), "%Y-%m-%d")
root_dir <- file.path("MIMIC_longitudinal_kmeans")   # top‑level output folder
cfg <- list(
  csv_pf     = "Data/mimic_pf_daily.csv",
  csv_stay   = "Data/mimic_stays.csv",
  admin_max  = 14,                   # observation window (days)
  K_values   = 2:6,                  # K to explore
  nb_redraw  = 20,                   # kml redraws / stabilisations
  run_gap    = FALSE                 # set TRUE to compute Gap statistic
)
main_dir <- file.path(root_dir, "main_analysis")
add_dir  <- file.path(root_dir, "additional_analysis")
dir.create(main_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(add_dir , recursive = TRUE, showWarnings = FALSE)

# ── 2 · DATA ────────────────────────────────────────────────────────────────
parse_datetime <- function(x){
  x <- trimws(as.character(x)); x[x==""] <- NA
  suppressWarnings(as.POSIXct(x, format = "%d/%m/%Y %H:%M", tz = "UTC"))
}

pf_raw <- read_csv(cfg$csv_pf, show_col_types = FALSE) %>%
  transmute(stay_id,
            day_period = day_period,
            pf_ratio_avg = if_else(pf_ratio_avg > 470, NA_real_, pf_ratio_avg)) %>%
  filter(day_period <= cfg$admin_max)

# build stay × day matrix (mmHg, NO imputation)
build_pf_matrix <- function(pf_tbl, admin_days){
  all_ids <- sort(unique(pf_tbl$stay_id))
  mat <- matrix(NA_real_, nrow = length(all_ids), ncol = admin_days + 1,
                dimnames = list(all_ids, paste0("d",0:admin_days)))
  pf_tbl %>%
    mutate(col_idx = paste0("d", day_period)) %>%
    split(.$stay_id) %>%
    walk(function(df){
      sid <- as.character(df$stay_id[1])
      mat[sid, df$col_idx] <<- df$pf_ratio_avg
    })
  mat
}

mat_pf <- build_pf_matrix(pf_raw, cfg$admin_max)
# drop all‑NA columns (days never measured in any patient)
mat_pf <- mat_pf[, !apply(mat_pf, 2, function(x) all(is.na(x))), drop = FALSE]
mat_pf_scaled <- scale(mat_pf)          # z‑score scaling (col‑wise)

# ── 3 · LONGITUDINAL K‑MEANS ────────────────────────────────────────────────
traj <- cld(mat_pf_scaled)              # matrix → LongData object
set.seed(2025)
# run kml across all requested K in one go
kml(traj, nbClusters = cfg$K_values, nbRedrawing = cfg$nb_redraw, toPlot = "none")

get_cluster_mat <- function(traj, Ks){
  cls <- sapply(Ks, function(k) getClusters(traj, k))
  cls <- as.matrix(cls)
  colnames(cls) <- paste0("K", Ks)
  rownames(cls) <- rownames(mat_pf_scaled)
  cls
}
cluster_mat <- get_cluster_mat(traj, cfg$K_values)

# ── 4 · EVALUATION METRICS ─────────────────────────────────────────────────
calc_wss <- function(X, cl){
  uniq <- sort(unique(cl))
  wss  <- 0
  for(k in uniq){
    idx <- which(cl == k)
    if(length(idx) < 1) next
    sub <- X[idx, , drop = FALSE]
    centroid <- colMeans(sub, na.rm = TRUE)
    for(i in seq_len(nrow(sub))){
      row <- sub[i, ]
      dif <- (row - centroid)^2
      obs <- !is.na(dif)
      if(any(obs)) wss <- wss + sum(dif[obs]) / sum(obs)
    }
  }
  wss
}

metrics <- map_dfr(cfg$K_values, function(K){
  cl <- cluster_mat[, paste0("K", K)]
  valid_idx <- which(!is.na(cl) & cl > 0)
  cl_int <- as.integer(factor(cl[valid_idx]))   # relabel to 1…K
  xmat   <- mat_pf_scaled[valid_idx, , drop = FALSE]
  
  # ---- Gower silhouette ----------------------------------------------------
  sil <- tryCatch({
    if(length(unique(cl_int)) < 2) NA_real_ else {
      dist_gow <- daisy(xmat, metric = "gower")
      mean(silhouette(cl_int, dist_gow)[, 3])
    }
  }, error = function(e) NA_real_)
  
  # ---- Adjusted WSS --------------------------------------------------------
  wss <- calc_wss(xmat, cl_int)
  
  tibble(K = K, Silhouette = sil, WSS = wss)
})

write_csv(metrics, file.path(main_dir, "EvalMetrics_AllK.csv"))

# ── 4·A  Optional Gap statistic (disabled by default) ----------------------
if(cfg$run_gap){
  gap_tbl <- map_dfr(cfg$K_values, function(K){
    cl <- cluster_mat[, paste0("K", K)]
    valid_idx <- which(!is.na(cl) & cl > 0)
    xmat   <- mat_pf_scaled[valid_idx, , drop = FALSE]
    wss_obs <- calc_wss(xmat, as.integer(factor(cl[valid_idx])))
    
    # reference distribution (uniform within each column’s range)
    ref_wss <- replicate(100, {
      ref <- apply(xmat, 2, function(col){
        rng <- range(col, na.rm = TRUE)
        runif(nrow(xmat), rng[1], rng[2])
      })
      calc_wss(ref, sample(1:K, nrow(ref), replace = TRUE))
    })
    tibble(K = K, Gap = mean(log(ref_wss)) - log(wss_obs))
  })
  write_csv(gap_tbl, file.path(main_dir, "GapStatistic_AllK.csv"))
  metrics <- left_join(metrics, gap_tbl, by = "K")
}

# ---- plot summary ---------------------------------------------------------
png(file.path(main_dir, "Eval_Criteria_vs_K.png"), 1200, 600, res = 150)
par(mfrow = c(1, if(cfg$run_gap) 3 else 2), mar = c(4,4,3,1))
with(metrics, plot(K, Silhouette, type="b", pch=19, col="steelblue",
                   main="Gower silhouette", xlab="K", ylab="Average sil."))
with(metrics, plot(K, WSS, type="b", pch=19, col="tomato",
                   main="Adjusted WSS (↓)", xlab="K", ylab="WSS"))
if(cfg$run_gap){
  with(metrics, plot(K, Gap, type="b", pch=19, col="purple",
                     main="Gap statistic", xlab="K", ylab="Gap"))
}
dev.off()

# ── 5 · PER‑K ARTEFACTS ────────────────────────────────────────────────────
for(K in cfg$K_values){
  cl <- cluster_mat[, paste0("K", K)]
  dest <- if(K == 4) main_dir else add_dir
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  
  # 5·A  class sizes --------------------------------------------------------
  write_csv(as_tibble(table(cluster = cl), .name_repair = "minimal") %>% rename(N=n),
            file.path(dest, sprintf("ClassSizes_K%02d.csv", K)))
  
  # 5·B  PCA & MDS (complete rows with valid cluster) ----------------------
  comp_idx <- which(complete.cases(mat_pf_scaled) & !is.na(cl))
  if(length(comp_idx) > 10 && length(unique(cl[comp_idx])) > 0){
    comp_mat <- mat_pf_scaled[comp_idx, ]
    comp_cl  <- factor(cl[comp_idx])
    
    # PCA
    pca <- prcomp(comp_mat, center=FALSE, scale.=FALSE)$x[,1:2]
    ggsave(file.path(dest, sprintf("PCA2D_K%02d.png", K)),
           ggplot(data.frame(PC1=pca[,1], PC2=pca[,2], cluster=comp_cl),
                  aes(PC1, PC2, colour=cluster))+
             geom_point(size=.8, alpha=.7)+theme_minimal()+
             labs(title=paste("PCA (complete cases) – K", K)),
           width=6, height=5, dpi=150)
    
    # MDS only if >1 distinct clusters in comp set
    if(length(unique(comp_cl)) > 1){
      mds <- cmdscale(dist(comp_mat), k=2)
      ggsave(file.path(dest, sprintf("MDS_K%02d.png", K)),
             ggplot(data.frame(Dim1=mds[,1], Dim2=mds[,2], cluster=comp_cl),
                    aes(Dim1, Dim2, colour=cluster))+
               geom_point(size=.8, alpha=.7)+theme_minimal()+
               labs(title=paste("MDS (complete cases) – K", K)),
             width=6, height=5, dpi=150)
    }
  }
  
  # 5·C  Mean trajectories --------------------------------------------------
  traj_df <- as.data.frame(mat_pf) %>%
    rownames_to_column("stay_id") %>%
    mutate(cluster = factor(cl[stay_id])) %>%
    pivot_longer(cols = starts_with("d"), names_to = "day", values_to = "pf_ratio") %>%
    mutate(day = as.integer(sub("d", "", day)))
  
  mean_traj <- traj_df %>%
    group_by(cluster, day) %>%
    summarise(mean_pf = mean(pf_ratio, na.rm = TRUE), .groups = "drop") %>%
    filter(is.finite(mean_pf))
  
  if(nrow(mean_traj) > 0){
    ggsave(file.path(dest, sprintf("Trajectories_K%02d.png", K)),
           ggplot(mean_traj, aes(day, mean_pf, colour=cluster))+
             geom_line(linewidth=1)+geom_point()+theme_minimal()+
             labs(title=paste("Mean PF Trajectories – K", K),
                  x="Day", y="PF Ratio (mmHg)"),
           width=8, height=4.5, dpi=150)
  }
}

cat("✅  All outputs saved in", root_dir, "\n")