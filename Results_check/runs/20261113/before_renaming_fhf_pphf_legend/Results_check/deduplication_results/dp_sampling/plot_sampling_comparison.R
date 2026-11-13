#!/usr/bin/env Rscript
# Rebuild the three requested ggplot2 figures from freshly checked saved runs.
# First run sampling_results_check.py, then this file. Optional:
# --data-dir=PATH --output-dir=PATH
suppressPackageStartupMessages({library(ggplot2); library(readr)})
args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) substring(tail(hit, 1), nchar(prefix)+1) else default
}
source_path <- tryCatch(sys.frame(1)$ofile, error=function(e) NULL)
file_arg <- grep("^--file=", commandArgs(), value=TRUE)
script <- if (!is.null(source_path)) source_path else if (length(file_arg)) sub("^--file=", "", file_arg[1]) else NULL
root <- if (is.null(script)) getwd() else dirname(normalizePath(script))
while (!dir.exists(file.path(root,"PPHF_modules")) && dirname(root)!=root) root <- dirname(root)
data_dir <- arg("data-dir",file.path(root,"Results_check/deduplication_results/dp_sampling"))
figure_dir <- arg("output-dir",file.path(data_dir,"figures"))
dir.create(figure_dir,recursive=TRUE,showWarnings=FALSE)
read_table <- function(name) as.data.frame(read_csv(file.path(data_dir,name),show_col_types=FALSE,progress=FALSE))
wide <- read_table("sampling_100rounds_by_scope.csv")
examples <- read_table("forecast_examples_72h.csv")
wide$dp_enabled <- tolower(as.character(wide$dp_enabled))=="true"
run_order <- c("non_dp_q1","dp_q1p0","dp_q0p5","dp_q0p1")
run_labels <- c("Non-DP | q = 1.0","DP | q = 1.0","DP | q = 0.5","DP | q = 0.1")
names(run_labels) <- run_order
scope_order <- c("Node average","L1","L2","L3")
scope_labels <- c("Node average (n = 360)","L1 (n = 1)","L2 (n = 60)","L3 (n = 299)")
stopifnot(nrow(wide)==16, setequal(wide$condition,run_order),all(wide$num_rounds==100))
for (i in seq_along(scope_order)) stopifnot(all(wide$n_nodes[wide$scope==scope_order[i]]==c(360,1,60,299)[i]))
wide <- wide[order(match(wide$scope,scope_order),match(wide$condition,run_order)),]
round_half_up <- function(x) {
  y <- abs(x)*1000
  result <- sign(x)*floor(y+.5+8*.Machine$double.eps*pmax(1,y))/1000
  result[!is.na(result)&result==0] <- 0
  result
}
fmt <- function(x) {
  out <- trimws(formatC(round_half_up(x),format="f",digits=3,decimal.mark="."))
  out[is.na(x)] <- NA_character_
  out
}
stopifnot(identical(fmt(c(1.2345,-1.2345,0)),c("1.235","-1.235","0.000")))
presentation <- wide[c("scope","condition","n_nodes","MAE","RMSE","MASE","RMSSE","epsilon")]
for (name in c("MAE","RMSE","MASE","RMSSE","epsilon")) presentation[[name]] <- fmt(presentation[[name]])
write_csv(presentation,file.path(data_dir,"sampling_100rounds_by_scope_3dp.csv"))
palette <- c(non_dp_q1="#777777",dp_q1p0="#3675A3",dp_q0p5="#009E73",dp_q0p1="#B44745")
metric_shapes <- c(MASE=16,RMSSE=17)
factors <- function(x) {
  x$condition <- factor(x$condition,levels=run_order)
  x$scope <- factor(x$scope,levels=scope_order,labels=scope_labels)
  x
}
long <- function(x) do.call(rbind,lapply(c("MASE","RMSSE"),function(metric) {
  x$metric <- factor(metric,levels=c("MASE","RMSSE"));x$value <- x[[metric]];x
}))
colour_scale <- function() scale_colour_manual(values=palette,breaks=run_order,
  labels=unname(run_labels),name="Setting",drop=FALSE)
shape_scale <- function() scale_shape_manual(values=metric_shapes,name="Metric")
legends <- function(privacy=FALSE) guides(
  colour=guide_legend(nrow=1,order=1,override.aes=if(privacy)
    list(shape=c(NA,16,16,16),linetype=c("dashed","blank","blank","blank"),size=3.1,linewidth=.6)
    else list(shape=16,size=3.1)),
  shape=guide_legend(nrow=1,order=2,override.aes=list(colour="#333333",size=3.1,linetype=0)))
theme_mcb <- theme_classic(base_size=12,base_family="sans")+
  theme(plot.title=element_text(hjust=.5,face="bold",size=17,margin=margin(b=5)),
    plot.subtitle=element_text(hjust=.5,size=12,margin=margin(b=8)),
    plot.title.position="plot",axis.title.x=element_text(size=13,margin=margin(t=8)),
    axis.title.y=element_text(size=13,margin=margin(r=8)),
    axis.text.x=element_text(size=11,colour="#333333"),
    axis.text.y=element_text(size=12,colour="#333333",hjust=1,margin=margin(r=6)),
    axis.line=element_blank(),axis.ticks=element_line(linewidth=.65),
    panel.border=element_rect(colour="black",fill=NA,linewidth=.9),
    panel.grid.major.x=element_line(colour="grey86",linewidth=.5),
    panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),
    panel.background=element_rect(fill="grey98",colour=NA),
    plot.background=element_rect(fill="white",colour=NA),
    strip.background=element_blank(),strip.text=element_text(size=13,face="bold",margin=margin(b=8,t=8)),
    panel.spacing=grid::unit(1.5,"lines"),plot.margin=margin(12,15,10,12),
    legend.position="bottom",legend.direction="horizontal",legend.box="vertical",
    legend.box.just="center",legend.background=element_blank(),legend.key=element_blank(),
    legend.title=element_text(size=11,face="bold"),legend.text=element_text(size=11),
    legend.spacing.y=grid::unit(3,"pt"),legend.key.width=grid::unit(1.2,"lines"))
theme_set(theme_mcb)
save <- function(plot,name,width,height) {
  type <- if (isTRUE(capabilities("aqua"))) "quartz" else getOption("bitmapType","cairo")
  ggsave(file.path(figure_dir,paste0(name,".png")),plot,width=width,height=height,dpi=300,bg="white",type=type)
  ggsave(file.path(figure_dir,paste0(name,".pdf")),plot,width=width,height=height,
         device=grDevices::pdf,useDingbats=FALSE,bg="white")
}

d <- wide
d$row_index <- (match(d$scope,scope_order)-1)*4+match(d$condition,run_order)
d$row_label <- paste0(unname(run_labels[d$condition]),
  "\nMASE ",fmt(d$MASE)," | RMSSE ",fmt(d$RMSSE))
axis_labels <- d$row_label[match(1:16,d$row_index)]
d <- factors(long(d));d$y_position <- d$row_index+ifelse(d$metric=="MASE",.14,-.14)
p1 <- ggplot(d,aes(x=value,y=y_position,colour=condition,shape=metric))+
  geom_point(size=3.1)+facet_wrap(~scope,scales="free",ncol=2)+
  colour_scale()+shape_scale()+legends()+
  scale_y_continuous(breaks=1:16,labels=axis_labels,expand=expansion(add=.5))+
  scale_x_continuous(expand=expansion(mult=c(.08,.08)),breaks=scales::breaks_pretty(n=4),labels=fmt)+
  labs(title=NULL,subtitle=NULL,
       x="Mean scaled error",y=NULL,caption=NULL)+
  theme(axis.text.y=element_text(size=11,lineheight=1.1),legend.text=element_text(size=12))
save(p1,"01_MASE_RMSSE_node_average_L1_L2_L3",13,8)

d <- factors(long(wide[wide$dp_enabled,]))
reference <- factors(long(wide[!wide$dp_enabled,]))
labels <- d[d$metric=="MASE",]
labels$value <- pmax(labels$MASE,labels$RMSSE)
labels$text <- paste0("q = ",format(labels$sampling_rate,nsmall=1),"\nepsilon = ",fmt(labels$epsilon))
labels$hjust <- ifelse(labels$epsilon<10,0,ifelse(labels$epsilon>60,1,.5))
p2 <- ggplot(d,aes(x=epsilon,y=value,colour=condition,shape=metric))+
  geom_hline(data=reference,aes(yintercept=value,colour=condition),linetype="dashed",linewidth=.6,
             show.legend=c(colour=TRUE,shape=FALSE))+
  geom_point(size=3.1,show.legend=TRUE)+
  geom_text(data=labels,aes(label=text,hjust=hjust),vjust=-.65,lineheight=1.1,size=4.2,show.legend=FALSE)+
  facet_wrap(~scope,scales="free_y",ncol=2)+colour_scale()+shape_scale()+legends(TRUE)+
  scale_x_continuous(limits=c(0,87),breaks=c(0,25,50,75),expand=expansion(mult=c(.02,.02)),labels=fmt)+
  scale_y_continuous(expand=expansion(mult=c(.15,.5)),labels=fmt)+
  labs(title=NULL,
       subtitle=NULL,
       x="epsilon",y="Mean scaled error",caption=NULL)+
  theme(axis.text.x=element_text(size=14),axis.text.y=element_text(size=14),
        axis.title.x=element_text(size=16),axis.title.y=element_text(size=16),
        strip.text=element_text(size=16,face="bold"),
        legend.title=element_text(size=14,face="bold"),legend.text=element_text(size=14),
        legend.key.height=grid::unit(1.1,"lines"),legend.spacing.y=grid::unit(5,"pt"))
save(p2,"02_privacy_utility_MASE_RMSSE_by_scope",11.8,8)

examples$condition <- wide$condition[match(examples$run_key,wide$run_key)]
stopifnot(!anyNA(examples$condition),setequal(unique(examples$level),1:3))
examples$ds <- as.POSIXct(examples$ds,tz="UTC")
examples$panel <- paste0("L",examples$level," | ",examples$unique_id)
panels <- unique(examples[order(examples$level),"panel"])
examples$panel <- factor(examples$panel,levels=panels)
actual <- unique(examples[c("ds","y_true","panel")]);actual$condition <- "observed"
names(actual)[names(actual)=="y_true"] <- "value"
predicted <- examples[c("ds","base","panel","condition")];names(predicted)[2] <- "value"
lines <- rbind(predicted,actual[names(predicted)])
lines$condition <- factor(lines$condition,levels=c("observed",run_order))
line_palette <- c(observed="#111111",non_dp_q1="#B44745",dp_q1p0="#3675A3",dp_q0p5="#777777",dp_q0p1="#333333")
line_types <- c(observed="solid",non_dp_q1="dashed",dp_q1p0="solid",dp_q0p5="longdash",dp_q0p1="dotdash")
p3 <- ggplot(lines,aes(x=ds,y=value,colour=condition,linetype=condition))+
  geom_line(linewidth=.5,alpha=.9)+facet_wrap(~panel,ncol=1,scales="free_y")+
  scale_colour_manual(values=line_palette,labels=c("Observed",unname(run_labels)),name=NULL)+
  scale_linetype_manual(values=line_types,labels=c("Observed",unname(run_labels)),name=NULL)+
  scale_x_datetime(date_breaks="1 day",date_labels="%d %b",timezone="UTC",expand=expansion(mult=c(.01,.01)))+
  scale_y_continuous(labels=fmt)+
  labs(title=NULL,
       subtitle=NULL,
       x="Original dataset timestamps",y="Electricity consumption (kWh per half-hour)",caption=NULL)+
  theme(legend.text=element_text(size=10),strip.text=element_text(hjust=0),legend.box="horizontal")+
  guides(colour=guide_legend(nrow=1),linetype=guide_legend(nrow=1))
save(p3,"03_forecast_examples_L1_L2_L3_72h",12.5,8)
capture.output(sessionInfo(),file=file.path(data_dir,"R_session_info.txt"))
message("Saved three PNG/PDF pairs to ",figure_dir)
