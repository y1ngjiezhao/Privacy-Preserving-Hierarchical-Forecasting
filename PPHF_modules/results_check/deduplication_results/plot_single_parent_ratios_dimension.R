# ggplot2 vector geometry + native LaTeX mathematical labels.
# Requires R/ggplot2, pdflatex, and pdftoppm. Formulas remain in the caption.
suppressPackageStartupMessages(library(ggplot2))
# Support source(), Rscript, and running selected lines in RStudio.
resolve_script <- function() {
  filename <- 'plot_single_parent_ratios_dimension.R'
  frames <- sys.frames()
  sourced <- unlist(lapply(frames, function(f) {
    if (!is.null(f$ofile)) as.character(f$ofile) else character()
  }), use.names=FALSE)
  cli <- sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE))
  editor <- character()
  if (requireNamespace('rstudioapi', quietly=TRUE) && rstudioapi::isAvailable()) {
    editor <- tryCatch(rstudioapi::getSourceEditorContext()$path,
                       error=function(e) character())
  }
  candidates <- c(rev(sourced), cli, editor,
                  file.path(getwd(), filename),
                  file.path(getwd(), 'PPHF_modules', 'results_check', 'deduplication_results', filename))
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  candidates <- candidates[basename(candidates)==filename & file.exists(candidates)]
  if (!length(candidates)) {
    stop('Cannot locate this script. Save it and run source("/full/path/',
         filename, '") or set the working directory to its folder.')
  }
  normalizePath(candidates[[1]], mustWork=TRUE)
}
script <- resolve_script()
project_root <- dirname(dirname(dirname(dirname(script))))
out <- file.path(project_root, 'Results_check', 'deduplication_results', 'ratio_illustration')
dir.create(out, showWarnings=FALSE, recursive=TRUE)
if (!dir.exists(out)) stop('Cannot create output directory: ', out)
stem <- file.path(out,'single_parent_ratios_dimension_bold')
W <- 690; H <- 330
# Extra room below the axis accommodates its tick labels and downward braces.
line_y <- c(forecast_base=285,forecast_bu=230,gap=230,axis=185,error_base=85,error_bu=140)
X <- function(v) 24 + 5*v
gray <- '#737373'; red <- '#FF0000'; ink <- '#000000'; blue <- '#0000FF'
dimensions <- data.frame(
  a=c(0,0,100,80,80),b=c(120,100,120,120,100),
  y=unname(line_y[c('forecast_base','forecast_bu','gap','error_base','error_bu')]),
  color=c(gray,gray,blue,red,red))
ticks <- data.frame(x=X(c(0,80,100,120)))
# One continuous guide per distinct endpoint and side avoids overdrawing shared
# endpoints, while connecting every nested brace to its value on the axis.
endpoints <- rbind(data.frame(value=dimensions$a,y=dimensions$y),
                   data.frame(value=dimensions$b,y=dimensions$y))
endpoints$side <- ifelse(endpoints$y > line_y[['axis']], 'above', 'below')
guides <- aggregate(y ~ value + side,endpoints,function(v)
  if (v[1] > line_y[['axis']]) max(v) else min(v))
stopifnot(all(vapply(seq_len(nrow(endpoints)),function(i) {
  e <- endpoints[i,]
  g <- guides[guides$value==e$value & guides$side==e$side,]
  nrow(g)==1 && abs(g$y-line_y[['axis']]) >= abs(e$y-line_y[['axis']])
},logical(1))))
p <- ggplot() +
  geom_segment(data=guides,aes(x=X(value),xend=X(value),y=line_y[['axis']],yend=y),
               colour='#000000',linetype='dashed',linewidth=0.25) +
  annotate('segment',x=X(0),xend=X(129),y=line_y[['axis']],yend=line_y[['axis']],colour=ink,linewidth=0.4,
           arrow=grid::arrow(length=grid::unit(0.11,'cm'),type='closed')) +
  geom_segment(data=ticks,aes(x=x,xend=x,y=line_y[['axis']]-4,yend=line_y[['axis']]+4),linewidth=0.35) +
  annotate('rect',xmin=24,xmax=366,ymin=57,ymax=134,
           fill='white',colour='black',linewidth=0.3) +
  scale_colour_identity() +
  scale_x_continuous(limits=c(0,W),expand=expansion(mult=0)) +
  scale_y_continuous(limits=c(0,H),expand=expansion(mult=0)) +
  theme_void() + theme(plot.margin=margin(0,0,0,0),plot.background=element_rect(fill='white',colour=NA))
background <- paste0(stem,'_lines.pdf')
if (capabilities('aqua')) {
  grDevices::quartz(type='pdf',file=background,width=W/72,height=H/72)
  print(p); invisible(dev.off())
} else {
  ggsave(background,p,width=W/72,height=H/72,device=cairo_pdf,bg='white')
}
# All variables below are typeset by TeX itself, including legend symbols.
labels <- data.frame(
  x=c(X(60),X(50),X(110),X(100),X(94),
      X(0),X(80),X(100),X(120),rep(40,3),rep(204,3)),
  y=c(305,250,250,62,117,164,164,164,164,119.5,95.5,71.5,119.5,95.5,71.5),
  align=c(rep('c',9),rep('l',6)),
  color=c('forecast','forecast','incoherence','error','error',rep('ink',4),
          'ink','ink','ink','incoherence','ink','ink'),
  math=c(
    r"($|\hat y_{p,t}|=120$)",
    r"($|\hat y^{\mathrm{fBU}}_{p,t}|=100$)",
    r"($\mathrm{inco.}_{p,t}=20$)",
    r"($|\hat y_{p,t}-y_{p,t}|=40$)",
    r"($|\hat y^{\mathrm{fBU}}_{p,t}-y_{p,t}|=20$)",
    '$0$',
    r"($y_{p,t}=80$)",
    r"($\hat y^{\mathrm{fBU}}_{p,t}=100$)",
    r"($\hat y_{p,t}=120$)",
    r"($y_{p,t}$: actual)",
    r"($\hat y_{p,t}$: base forecast)",
    r"($\hat y^{\mathrm{fBU}}_{p,t}$: fBU aggregated)",
    r"($\mathrm{inco.}_{p,t}$: incoherence)",
    r"($s^{\mathrm{F}}_{p,t}=110$: forecast scale)",
    r"($s^{\mathrm{E}}_{p,t}=30$: error scale)"))
tex <- c(
  r"(\documentclass[border=0pt]{standalone})",
  r"(\usepackage{graphicx,xcolor,amsmath})",
  r"(\definecolor{forecast}{HTML}{737373})",
  r"(\colorlet{error}{red})",
  r"(\colorlet{ink}{black})",
  r"(\colorlet{incoherence}{blue})",
  r"(\begin{document}\setlength{\unitlength}{1bp})",
  sprintf('\\begin{picture}(%d,%d)',W,H),
  sprintf('\\put(0,0){\\includegraphics[width=%dbp,height=%dbp]{\\detokenize{%s}}}',W,H,background))
# Use the actual Computer Modern overbrace/underbrace math glyphs.
# The two upper component braces meet at BU, below the full Base span.
for (i in seq_len(nrow(dimensions))) {
  d <- dimensions[i,]
  name <- if (i<=2) 'forecast' else if (i==3) 'incoherence' else 'error'
  brace <- if (i>=4) 'underbrace' else 'overbrace'
  tex <- c(tex,sprintf(
    '\\put(%.2f,%.2f){\\makebox(0,0){\\color{%s}\\fontsize{14}{17}\\selectfont$\\displaystyle\\%s{\\hspace{%.2fbp}}$}}',
    (X(d$a)+X(d$b))/2,d$y,name,brace,X(d$b)-X(d$a)))
}
for (i in seq_len(nrow(labels))) {
  a <- labels[i,]
  size <- if (i>=10) 12 else 14
  tex <- c(tex,sprintf(
    '\\put(%.2f,%.2f){\\makebox(0,0)[%s]{\\color{%s}\\fontsize{%d}{17}\\selectfont\\bfseries\\boldmath\\colorbox{white}{\\strut %s}}}',
    a$x,a$y,a$align,a$color,size,a$math))
}
tex <- c(tex,r"(\end{picture}\end{document})")
writeLines(tex,paste0(stem,'.tex'))
# GUI-launched RStudio may not inherit the terminal's PATH.
find_program <- function(name) {
  dirs <- c('/Library/TeX/texbin', '/opt/homebrew/bin', '/usr/local/bin',
            path.expand('~/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override'))
  candidates <- c(unname(Sys.which(name)), file.path(dirs, name))
  candidates <- candidates[nzchar(candidates)]
  candidates <- candidates[file.exists(candidates) & file.access(candidates, 1)==0]
  if (!length(candidates)) return('')
  candidates[[1]]
}
latex <- find_program('pdflatex')
raster <- find_program('pdftoppm')
if (!nzchar(latex)) stop('pdflatex not found. Install MacTeX/BasicTeX, then rerun the entire script.')
if (!nzchar(raster)) stop('pdftoppm not found. Install Poppler (brew install poppler), then rerun the entire script.')
message('Using pdflatex: ', latex, '\nUsing pdftoppm: ', raster)
status <- system2(latex,c('-interaction=nonstopmode','-halt-on-error',
                         paste0('-output-directory=',shQuote(out)),shQuote(paste0(stem,'.tex'))),
                  stdout=FALSE,stderr=FALSE)
if (status!=0) stop('LaTeX compilation failed; see ',stem,'.log')
status <- system2(raster,c('-png','-singlefile','-r','180',shQuote(paste0(stem,'.pdf')),shQuote(stem)),
                  stdout=FALSE,stderr=FALSE)
if (status!=0) stop('PDF preview rendering failed.')
cat('Saved:',stem,'\n')

# Keep a small outer margin so tight cropping never touches glyphs or arrows.
crop <- find_program('pdfcrop')
if (nzchar(crop)) {
  trimmed <- paste0(stem, '_trimmed')
  status <- system2(crop, c('--margins', '6', shQuote(paste0(stem, '.pdf')),
                            shQuote(paste0(trimmed, '.pdf'))), stdout=FALSE, stderr=FALSE)
  if (status != 0) stop('PDF crop failed.')
  status <- system2(raster, c('-png', '-singlefile', '-r', '180',
                              shQuote(paste0(trimmed, '.pdf')), shQuote(trimmed)),
                    stdout=FALSE, stderr=FALSE)
  if (status != 0) stop('Trimmed PDF preview rendering failed.')
}
