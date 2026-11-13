suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(greybox)
  library(ggplot2)
})

script_arg <- commandArgs(trailingOnly = FALSE)
source_path <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
script_path <- if (!is.null(source_path)) source_path else sub("^--file=", "", script_arg[grepl("^--file=", script_arg)])
module_dir <- normalizePath(dirname(script_path), mustWork = TRUE)
project_root <- dirname(dirname(dirname(module_dir)))
output_root <- file.path(project_root, 'Results_check', 'deduplication_results')

plot_mcb_unified <- function(result, metric, dataset_label, horizon) {
  mean_rank <- result$mean
  method_names <- names(mean_rank)
  intervals <- as.matrix(result$interval)
  if (nrow(intervals) == 2 && ncol(intervals) == length(mean_rank)) {
    intervals <- t(intervals)
  }
  rownames(intervals) <- method_names

  frame <- data.frame(
    method = method_names,
    mean_rank = as.numeric(mean_rank),
    lower = intervals[, 1],
    upper = intervals[, 2],
    stringsAsFactors = FALSE
  )
  best_name <- frame$method[which.min(frame$mean_rank)]
  not_significant <- setNames(rep(FALSE, length(method_names)), method_names)
  if (!is.null(result$methods) && is.matrix(result$methods)) {
    if (best_name %in% colnames(result$methods)) {
      values <- as.logical(result$methods[, best_name])
      names(values) <- rownames(result$methods)
      not_significant[names(values)] <- values
    } else if (best_name %in% rownames(result$methods)) {
      values <- as.logical(result$methods[best_name, ])
      names(values) <- colnames(result$methods)
      not_significant[names(values)] <- values
    }
  }
  not_significant[best_name] <- TRUE
  frame$group <- ifelse(
    frame$method == best_name,
    "Best",
    ifelse(not_significant[frame$method], "No significant difference", "Significantly worse")
  )
  frame <- frame %>% arrange(mean_rank)
  frame$label_colour <- ifelse(
    frame$group == "Best",
    "#B44745",
    ifelse(frame$group == "No significant difference", "#3675A3", "#333333")
  )
  # Publication label: LN (Leaf node) replaces BO (Bottom only).
  # Keep the original method keys for all statistical comparisons above.
  display_method <- sub("^BO(?=-|_|$)", "LN", frame$method, perl = TRUE)
  frame$plain_label <- paste0(display_method, " (", sprintf("%.3f", floor(frame$mean_rank * 1000 + 0.5) / 1000), ")")
  frame$label <- factor(
    frame$plain_label,
    levels = frame$plain_label
  )
  best_row <- frame %>% filter(group == "Best")
  x_min <- min(frame$lower, na.rm = TRUE)
  x_max <- max(frame$upper, na.rm = TRUE)
  x_span <- max(x_max - x_min, 1)
  x_pad <- 0.04 * x_span

  plot <- ggplot(frame, aes(y = label)) +
    annotate(
      "rect",
      xmin = best_row$lower[1], xmax = best_row$upper[1],
      ymin = -Inf, ymax = Inf, fill = "#D9D9D9", alpha = 0.85
    ) +
    geom_segment(
      aes(x = lower, xend = upper, yend = label, colour = group),
      linewidth = 0.85, alpha = 0.95
    ) +
    geom_point(aes(x = lower, colour = group), size = 1.6) +
    geom_point(aes(x = upper, colour = group), size = 1.6) +
    geom_point(aes(x = mean_rank, colour = group), size = 2.4) +
    scale_colour_manual(
      values = c(
        "Best" = "#B44745",
        "No significant difference" = "#3675A3",
        "Significantly worse" = "#333333"
      ),
      breaks = c("Best", "No significant difference", "Significantly worse"),
      name = NULL,
      guide = guide_legend(nrow = 1, byrow = TRUE)
    ) +
    coord_cartesian(xlim = c(x_min - x_pad, x_max + x_pad), clip = "on") +
    labs(
      title = paste("MCB Test on", metric),
      subtitle = if (is.null(horizon)) paste0(dataset_label, " · Pooled horizons (legacy)") else paste0(dataset_label, " · Forecast horizon = ", horizon),
      x = "Mean rank (lower is better)",
      y = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 17, margin = margin(b = 5)),
      plot.subtitle = element_text(hjust = 0.5, size = 12, margin = margin(b = 8)),
      axis.title.x = element_text(size = 13, margin = margin(t = 8)),
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 14, hjust = 1, margin = margin(r = 6)),
      axis.line = element_blank(),
      axis.ticks = element_line(linewidth = 0.65),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.90),
      panel.grid.major.x = element_line(colour = "grey86", linewidth = 0.50),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.background = element_rect(fill = "grey98", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(9, 12, 7, 10),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.key = element_blank(),
      legend.key.width = grid::unit(1.2, "lines"),
      legend.spacing.x = grid::unit(0.35, "lines"),
      legend.margin = margin(t = 4)
    )

  list(plot = plot, label_colours = frame$label_colour)
}

# Keep the original layout (labels outside the panel) and apply one colour per
# standard ggplot axis label. Editing the axis text grob avoids the PDF clipping
# problems caused by HTML/ggtext while preserving the original visual design.
colour_y_axis_labels <- function(plot, label_colours) {
  grob <- ggplotGrob(plot)
  # Center the single-row legend across the complete figure, including labels.
  legend_index <- which(grob$layout$name == "guide-box-bottom")
  grob$layout$l[legend_index] <- 1
  grob$layout$r[legend_index] <- ncol(grob)

  axis_index <- which(grob$layout$name == "axis-l")
  if (length(axis_index) != 1) {
    stop("Could not identify the left axis grob")
  }

  axis_table <- grob$grobs[[axis_index]]$children$axis
  title_index <- which(vapply(axis_table$grobs, inherits, logical(1), "titleGrob"))
  if (length(title_index) != 1) {
    stop("Could not identify the y-axis label grob")
  }

  title_grob <- axis_table$grobs[[title_index]]
  text_index <- which(vapply(title_grob$children, inherits, logical(1), "text"))
  if (length(text_index) != 1) {
    stop("Could not identify the y-axis text child")
  }

  text_grob <- title_grob$children[[text_index]]
  if (length(text_grob$label) != length(label_colours)) {
    stop("Axis-label and colour counts do not match")
  }
  # A single textGrob with vectorised colours is not rendered reliably by all
  # PDF devices. Split it into one standard textGrob per label instead.
  label_grobs <- lapply(seq_along(text_grob$label), function(index) {
    grid::textGrob(
      label = text_grob$label[index],
      x = text_grob$x[index],
      y = text_grob$y[index],
      just = text_grob$just,
      hjust = text_grob$hjust,
      vjust = text_grob$vjust,
      rot = text_grob$rot,
      check.overlap = text_grob$check.overlap,
      name = paste0("mcb-axis-label-", index),
      gp = grid::gpar(
        fontsize = text_grob$gp$fontsize,
        col = label_colours[index],
        fontfamily = text_grob$gp$fontfamily,
        lineheight = text_grob$gp$lineheight,
        font = unname(text_grob$gp$font)
      )
    )
  })
  title_grob$children[[text_index]] <- grid::gTree(
    children = do.call(grid::gList, label_grobs),
    name = text_grob$name
  )
  axis_table$grobs[[title_index]] <- title_grob
  grob$grobs[[axis_index]]$children$axis <- axis_table
  grob
}

metadata_columns <- c(
  "node", "unique_id", "series", "level", "Level",
  "h", "horizon", "dataset"
)

dataset_labels <- c(
  "gef2017" = "GEF2017",
  "ausgrid2013" = "AusGrid2013"
)

save_mcb_figure <- function(raw, dataset, dataset_dir, metric, horizon, suffix) {
  horizon_raw <- if (is.null(horizon)) raw else raw %>% filter(.data$horizon == .env$horizon)
  score_matrix <- horizon_raw %>%
    select(-any_of(metadata_columns)) %>%
    select(where(is.numeric)) %>%
    as.matrix()
  storage.mode(score_matrix) <- "double"
  score_matrix <- score_matrix[complete.cases(score_matrix), , drop = FALSE]

  if (nrow(score_matrix) == 0) {
    stop(paste("No complete MCB observations for", dataset, metric, "horizon", horizon))
  }

  result <- rmcb(score_matrix, level = 0.95, outplot = "none")
  figure_parts <- plot_mcb_unified(
    result,
    metric,
    dataset_labels[[dataset]],
    horizon
  )
  figure <- colour_y_axis_labels(
    figure_parts$plot,
    figure_parts$label_colours
  )

  pdf_path <- file.path(dataset_dir, paste0("MCB_", metric, suffix, ".pdf"))
  png_path <- file.path(dataset_dir, paste0("MCB_", metric, suffix, ".png"))
  ggsave(
    pdf_path,
    figure,
    width = 6.4,
    height = 5.2,
    units = "in"
  )
  # Rasterise the verified vector output so PDF and PNG remain identical.
  status <- system2(
    "pdftoppm",
    c(
      "-png", "-singlefile", "-r", "180",
      pdf_path,
      tools::file_path_sans_ext(png_path)
    ),
    stdout = FALSE,
    stderr = FALSE
  )
  if (!identical(status, 0L)) {
    stop(paste("pdftoppm failed for", pdf_path))
  }
}

render_all_mcb <- function() {
for (dataset in c("gef2017", "ausgrid2013")) {
  dataset_dir <- file.path(output_root, dataset)
  for (metric in c("MASE", "RMSSE")) {
    raw <- read_csv(
      file.path(dataset_dir, paste0("all_per-node_", tolower(metric), ".csv")),
      show_col_types = FALSE
    )
    horizons <- sort(unique(raw$horizon))
    if (dataset == "gef2017") {
      for (horizon in horizons) {
        save_mcb_figure(
          raw, dataset, dataset_dir, metric, horizon,
          paste0("_h", horizon)
        )
      }
    } else {
      # AusGrid has one forecast horizon; retain its established filenames.
      save_mcb_figure(raw, dataset, dataset_dir, metric, horizons[[1]], "")
    }
  }
}

}

if (sys.nframe() == 0) {
  pdf(file = NULL)
  render_all_mcb()
  invisible(dev.off())
}
