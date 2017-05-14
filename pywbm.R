#! /usr/bin/Rscript
library(ggplot2)
data <- read.table("temp.csv", sep=',', header=TRUE)

plot <- ggplot(data, aes(x=x, y=y, fill=p_abs)) + geom_tile() + coord_fixed()
ggsave('plot.pdf')