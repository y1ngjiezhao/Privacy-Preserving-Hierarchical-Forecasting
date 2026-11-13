setwd("C:/Users/n12553263/yjzR/GEFCOM2017")
install.packages("remotes")
remotes::install_github("camroach87/gefcom2017data")
library(gefcom2017data)
gef2017 <- gefcom
file_path = 'C:/Users/n12553263/yjzR/GEFCOM2017/Gef2017.csv'
write.csv(gef2017, file_path, row.names = FALSE)
