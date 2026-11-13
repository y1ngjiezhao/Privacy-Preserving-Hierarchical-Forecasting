# Compatibility entry point: render current deduplicated results only.
script_args <- commandArgs(FALSE)
this_script <- normalizePath(sub('^--file=', '', grep('^--file=', script_args, value=TRUE)))
source(file.path(dirname(this_script), '..', 'deduplication_results', 'render_mcb_figures.R'))
pdf(file = NULL)
render_all_mcb()
invisible(dev.off())
