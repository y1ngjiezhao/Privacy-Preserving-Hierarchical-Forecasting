# Compatibility entry point: use the shared, verified publication layout.
args <- commandArgs(FALSE)
script <- normalizePath(sub('^--file=', '', grep('^--file=', args, value=TRUE)))
source(file.path(dirname(script), 'render_mcb_figures.R'))
