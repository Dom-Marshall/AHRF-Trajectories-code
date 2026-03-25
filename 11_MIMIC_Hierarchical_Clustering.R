###############################################################################
#   CONSENSUS PF-TRAJECTORY CLUSTERING  (MIMIC)  –  K = 2 … 6
#   Outputs split into “main_analysis” (manuscript) and
#   “additional_analysis” (supplement).
###############################################################################

# ── 0 · PACKAGES  ────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(dplyr);  library(tidyr);  library(purrr);  library(readr)
  library(tibble);  library(lubridate);  library(cluster);  library(parallel)
  library(ggplot2);    library(scatterplot3d);  library(pheatmap)
})

# ── 1 · CONFIG  ─────────────────────────────────────────────────────────────
run_date <- format(Sys.Date(), "%Y-%m-%d")
root_dir <- file.path("MIMIC_hierarchical_clustering")   # top-level folder
cfg <- list(
  csv_pf    = "Data/mimic_pf_daily.csv",
  csv_stay  = "Data/mimic_stays.csv",
  admin_max = 14,
  K_values  = 2:6,
  reps      = 100,
  prop_sub  = 0.70,
  parallel_type = "PSOCK"
)
main_dir <- file.path(root_dir, "main_analysis")
add_dir  <- file.path(root_dir, "additional_analysis")
dir.create(main_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(add_dir , recursive = TRUE, showWarnings = FALSE)

conv <- 7.50062
unit_conv <- function(x,f,t){
  if (f==t) x else if (f=="mmHg"&&t=="kPa") x/conv else if (f=="kPa"&&t=="mmHg") x*conv
  else stop("unit combo")
}
parse_datetime <- function(datetime_str) {
  datetime_str <- as.character(datetime_str)
  datetime_str <- trimws(datetime_str)
  datetime_str[datetime_str == ""] <- NA
  suppressWarnings(as.POSIXct(datetime_str, format = "%d/%m/%Y %H:%M", tz = "UTC"))
}

# ── 2 · DATA  ───────────────────────────────────────────────────────────────
pf_raw <- read_csv(cfg$csv_pf, show_col_types = FALSE) %>%
  transmute(stay_id, day_period = day_period,
            pf_ratio_avg = if_else(pf_ratio_avg > 470, NA_real_, pf_ratio_avg)) %>%
  filter(day_period <= cfg$admin_max)

stay_raw <- read_csv(cfg$csv_stay, show_col_types = FALSE) %>%
  mutate(
    timezero = parse_datetime(timezero),
    outtime  = parse_datetime(outtime)
  ) %>%
  # Keep rows with valid timezero only
  filter(!is.na(timezero)) %>%
  # Filter out outtime earlier than timezero, or NA outtimes if needed
  mutate(
    raw_tte = as.numeric(difftime(coalesce(outtime, timezero), timezero, units = "days")),
    time_order_issue = !is.na(outtime) & outtime < timezero
  ) %>%
  filter(!time_order_issue & raw_tte >= 0) %>%
  # Assign event status
  mutate(
    status = case_when(
      icu_mort == 1 ~ 1L,
      !is.na(outtime) ~ 2L,
      TRUE ~ 0L
    ),
    event = if_else(raw_tte > cfg$admin_max, 0L, status),
    tte = pmin(raw_tte, cfg$admin_max)
  ) %>%
  select(stay_id, tte, event)


# --- build stay × day matrix (mmHg → kPa) ------------------------------
build_pf_matrix <- function(pf_tbl, cr_tbl, admin_days){
  long <- pf_tbl %>% left_join(cr_tbl,"stay_id") %>%
    arrange(stay_id, day_period)
  all_ids <- unique(long$stay_id)
  fill_one <- function(sid){
    sub <- long %>% filter(stay_id==sid)
    out <- tibble(stay_id = sid,
                  day_period = 0:admin_days,
                  pf = NA_real_)
    out$pf[out$day_period %in% sub$day_period] <- sub$pf_ratio_avg
    # LOCF up to tte; fill death/discharge rules beyond
    tte <- sub$tte[1]; evt <- sub$event[1]
    last <- NA_real_
    for (d in 0:admin_days){
      idx <- d+1
      if(is.na(out$pf[idx])) out$pf[idx] <- last
      if(d > tte){
        out$pf[idx] <- if(evt==1) 0 else if(evt==2){
          if(!is.na(last)) ifelse(last>250,350,last) else NA_real_
        } else out$pf[idx]
      }
      last <- out$pf[idx]
    }
    out
  }
  pf_long <- map_dfr(all_ids, fill_one)
  pf_long <- pf_long %>%
    group_by(day_period) %>%
    mutate(pf = if_else(is.na(pf), median(pf,na.rm=TRUE), pf)) %>%
    ungroup()
  mat <- pf_long %>% pivot_wider(names_from = day_period,
                                 names_prefix="d", values_from = pf) %>%
    column_to_rownames("stay_id") %>% as.matrix()
  mat <- unit_conv(mat,"mmHg","kPa")
  mat
}

mat_pf <- build_pf_matrix(pf_raw, stay_raw, cfg$admin_max)

# ── 3 · CONSENSUS FUNCTIONS  ──────────────────────────────────────────
run_one_kmeans <- function(mat_pf, idx, K){
  km <- kmeans(mat_pf[idx,,drop=FALSE], centers=K, nstart=10)
  list(idx=idx, cl=km$cluster)
}
run_consensus_K <- function(K, mat_pf, reps, prop_sub, parallel_type, n_workers){
  n <- nrow(mat_pf); co <- cnt <- matrix(0L,n,n)
  if(is.null(n_workers)) n_workers <- max(1, detectCores()-1)
  cl <- if(parallel_type=="FORK") makeCluster(n_workers, type="FORK")
  else                       makeCluster(n_workers, type="PSOCK")
  clusterEvalQ(cl, {library(stats)})
  clusterExport(cl, varlist=c("mat_pf","run_one_kmeans","K","prop_sub"), envir=environment())
  clusterSetRNGStream(cl, 12345)
  seq_reps <- parLapply(cl, seq_len(reps), function(i){
    idx <- sample.int(n, floor(prop_sub*n)); run_one_kmeans(mat_pf, idx, K)
  })
  stopCluster(cl)
  for(res in seq_reps){
    idx <- res$idx; clus <- res$cl
    cnt[idx,idx] <- cnt[idx,idx]+1L
    same <- which(outer(clus,clus,"=="), arr.ind=TRUE)
    co[cbind(idx[same[,1]], idx[same[,2]])] <- co[cbind(idx[same[,1]], idx[same[,2]])]+1L
  }
  consensus <- co/cnt; diag(consensus) <- 1; consensus[is.na(consensus)] <- 0
  rownames(consensus) <- colnames(consensus) <- rownames(mat_pf)
  consensus
}
cdf_area <- function(cmat){
  v <- cmat[upper.tri(cmat)]
  xs <- sort(v); ys <- ecdf(v)(xs)
  sum((ys[-1]+ys[-length(ys)])*diff(xs)/2)
}
sil_score <- function(cmat,K){
  pam(as.dist(1-cmat), k=K, diss=TRUE)$silinfo$avg.width
}

# ── 4 · CONSENSUS OVER K  ─────────────────────────────────────────────
mat_pf_scaled <- scale(mat_pf)
results <- map(cfg$K_values, function(K){
  cat("Consensus for K",K,"\n")
  cm <- run_consensus_K(K, mat_pf_scaled, cfg$reps, cfg$prop_sub,
                        cfg$parallel_type, cfg$n_workers)
  list(K=K, consensus=cm,
       CDF_area=cdf_area(cm),
       Silhouette=sil_score(cm,K))
})

# ── 5 · CORE METRICS & Δ-CDF/PAC  ────────────────────────────────────
metrics <- map_dfr(results, function(x){
  up <- x$consensus[upper.tri(x$consensus)]
  tibble(K=x$K,
         CDF_area=x$CDF_area,
         Silhouette=x$Silhouette,
         PAC = mean(up>0.1 & up<0.9))
}) %>% arrange(K) %>%
  mutate(Delta_CDF = CDF_area - lag(CDF_area, default=first(CDF_area)))

write_csv(metrics, file.path(main_dir,"Consensus_Metrics.csv"))

# Optional axis controls: set to NULL for “auto”
axis_ctrl <- list(
  x_lim = NULL,                   # e.g. c(2, 6)
  y_lim_CDF = c(0.5,0.9),               # e.g. c(0.70, 1.00)
  y_lim_dCDF = NULL,              # e.g. c(0.00, 0.10)
  y_lim_sil = NULL,               # e.g. c(0.00, 1.00)
  y_lim_pac = NULL                # e.g. c(0.00, 0.50)
)

png(file.path(main_dir,"Consensus_Metrics_AllK.png"),1400,1400,res=150)
par(mfrow=c(2,2), mar=c(4,4,3,1))
with(metrics, plot(K, CDF_area, type="b", pch=19, col="steelblue",
                   main="CDF Area vs K", xlab="K", ylab="Area",
                   xlim=axis_ctrl$x_lim, ylim=axis_ctrl$y_lim_CDF))
with(metrics, plot(K, Delta_CDF, type="b", pch=19, col="darkgreen",
                   main="Delta-CDF vs K", xlab="K", ylab="Delta Area",
                   xlim=axis_ctrl$x_lim, ylim=axis_ctrl$y_lim_dCDF))
with(metrics, plot(K, Silhouette, type="b", pch=19, col="tomato",
                   main="Average Silhouette", xlab="K", ylab="Width",
                   xlim=axis_ctrl$x_lim, ylim=axis_ctrl$y_lim_sil))
with(metrics, plot(K, PAC, type="b", pch=19, col="purple",
                   main="PAC", xlab="K", ylab="PAC",
                   xlim=axis_ctrl$x_lim, ylim=axis_ctrl$y_lim_pac))
dev.off()

## 5·C  Heat-map for primary K #########################################
PRIMARY_K <- 4
prim_cons <- results[[which(map_int(results,"K")==PRIMARY_K)]]$consensus

# ensure pheatmap is loaded
suppressPackageStartupMessages(requireNamespace("pheatmap", quietly = TRUE))

png(file.path(main_dir,"ConsensusHeatmap_K04.png"),1200,1200,res=150)
pheatmap::pheatmap(
  prim_cons,
  show_rownames = FALSE, show_colnames = FALSE,
  color = colorRampPalette(c("white","steelblue","black"))(100))
dev.off()

## 5·D  Patient-level cluster membership CSV ###########################
set.seed(2025)
km4 <- kmeans(mat_pf_scaled, centers = PRIMARY_K, nstart = 20)
write_csv(
  tibble(stay_id = as.numeric(names(km4$cluster)),
         cluster = km4$cluster),
  file.path(main_dir,"ClusterMembership_K04.csv")
)

cat("✅  Manuscript key outputs saved in", main_dir, "\n")


# ════════════════════════════════════════════════════════════════════════
# 6 · PER-K CLUSTER ARTEFACTS  (class sizes, MDS, PCA, PF trajectories)
#     – K = 4   → main_analysis
#     – others → additional_analysis
# ════════════════════════════════════════════════════════════════════════

library(ggplot2)

for (res in results) {
  K <- res$K
  cmat <- res$consensus
  dest <- if (K == 4) main_dir else add_dir
  
  # ---- k-means on full scaled matrix to get assignments
  set.seed(123 + K)
  km <- kmeans(mat_pf_scaled, centers = K, nstart = 20)
  cl_vec <- km$cluster          # integer vector
  names(cl_vec) <- rownames(mat_pf_scaled)
  
  # ---- 6·A  class-size table ------------------------------------------
  size_tbl <- as_tibble(table(cluster = cl_vec), .name_repair = "minimal") %>%
    rename(N = n)
  write_csv(size_tbl, file.path(dest, sprintf("ClassSizes_K%02d.csv", K)))
  
  # ---- 6·B  2-D PCA scatter -------------------------------------------
  pca <- prcomp(mat_pf_scaled, center = FALSE, scale. = FALSE)$x[,1:2]
  pca_df <- data.frame(PC1 = pca[,1], PC2 = pca[,2],
                       cluster = factor(cl_vec))
  p_pca <- ggplot(pca_df, aes(PC1, PC2, colour = cluster))+
    geom_point(size = .8)+
    theme_minimal()+
    labs(title = paste("PCA (PC1 vs PC2)  –  K =", K))
  ggsave(file.path(dest, sprintf("PCA2D_K%02d.png", K)),
         p_pca, width = 6, height = 5, dpi = 150)
  
  # ---- 6·C  MDS scatter (cmdscale on 1–consensus) ----------------------
  mds <- cmdscale(as.dist(1 - cmat), k = 2)
  mds_df <- data.frame(Dim1 = mds[,1], Dim2 = mds[,2],
                       cluster = factor(cl_vec))
  p_mds <- ggplot(mds_df, aes(Dim1, Dim2, colour = cluster))+
    geom_point(size = .8)+
    theme_minimal()+
    labs(title = paste("MDS of Consensus  –  K =", K))
  ggsave(file.path(dest, sprintf("MDS_K%02d.png", K)),
         p_mds, width = 6, height = 5, dpi = 150)
  
  # ---- 6·D  Mean PF trajectory plot ------------------------------------
  long_df <- as.data.frame(mat_pf) %>%
    rownames_to_column("stay_id") %>%
    mutate(cluster = factor(cl_vec[stay_id])) %>%
    pivot_longer(cols = starts_with("d"), names_to = "day",
                 values_to = "pf_kPa") %>%
    mutate(day = as.integer(sub("d", "", day)))
  
  traj_df <- long_df %>% group_by(cluster, day) %>%
    summarise(mean_pf = mean(pf_kPa), .groups = "drop")
  
  p_traj <- ggplot(traj_df, aes(day, mean_pf, colour = cluster))+
    geom_line(linewidth = 1)+
    theme_bw()+
    labs(title = paste("Mean PF trajectory  –  K =", K),
         x = "Day (0–14)", y = "PF (kPa)")
  ggsave(file.path(dest, sprintf("MeanPF_K%02d.png", K)),
         p_traj, width = 6, height = 4, dpi = 150)
}

cat("Per-K class tables & plots saved\n")
