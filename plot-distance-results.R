suppressPackageStartupMessages(library(pacman))
suppressPackageStartupMessages(p_load(ggplot2))
suppressPackageStartupMessages(p_load(reshape2))
suppressPackageStartupMessages(p_load(ggbeeswarm))
suppressPackageStartupMessages(p_load(scales))

## Plot the results of using various registration approaches in
## different datasets

## more brca_kp_distances.csv 
## Read in results collated by Aidan from OHSU BrCa dataset,
## which include alignments to R1.
## These include:
## OHSU Aligned: alignment provided by OHSU
## Unaligned: distance without alignment (these have limited precision -- get from elsewhere)
## Rigid alignment on DAPI: rigid alignment that we performed
## Non-rigid alignment on channel 2
## Non-rigid alignment on composite: exclude this is NA
tbl <- read.table("brca_kp_distances.csv", sep=",", header=TRUE)
tbl <- tbl[, !(colnames(tbl) == "Non.rigid.alignment.on.composite")]
tbl <- tbl[, colnames(tbl) %in% c("Moving", "OHSU.Aligned", "Non.rigid.alignment.on.channel.2")]
colnames(tbl) <- c("Moving", "Linear; Keypoint\n(OHSU; DAPI)", "Non-linear; Intensity\n(AF)")
tbl <- melt(tbl)
colnames(tbl) <- c("Moving", "Strategy", "Distance")
tbl$Target <- "R1"
flag <- grepl(tbl$Strategy, pattern="Non-linear")
tbl$Channel <- "DAPI"
tbl$Channel[flag] <- "AF"
tbl$Dataset <- "BrCa (OHSU)"
for(col in c("Moving", "Target", "Channel", "Strategy", "Dataset")) {
    tbl[,col] <- as.character(tbl[,col])
}
tbl1 <- tbl

tbl <- read.table("c1_brca_rigid_alignments", sep=",", header=TRUE)
tbl <- tbl[, !(colnames(tbl) == "target")]
colnames(tbl) <- c("Moving", "Linear; Keypoint", "Unaligned")
tbl <- melt(tbl)
colnames(tbl) <- c("Moving", "Strategy", "Distance")
tbl$Target <- "R1"
tbl$Channel <- "DAPI"
tbl$Strategy <- paste0(tbl$Strategy, "\n(DAPI)")
tbl$Dataset <- "BrCa (OHSU)"
for(col in c("Moving", "Target", "Channel", "Strategy", "Dataset")) {
    tbl[,col] <- as.character(tbl[,col])
}
tbl2 <- tbl

tbl <- read.table("c2_brca_rigid_alignments", sep=",", header=TRUE)
tbl <- tbl[, !(colnames(tbl) == "target")]
colnames(tbl) <- c("Moving", "Linear; Keypoint", "Unaligned")
tbl <- melt(tbl)
colnames(tbl) <- c("Moving", "Strategy", "Distance")
tbl$Target <- "R2"
tbl$Channel <- "AF"
tbl$Strategy <- paste0(tbl$Strategy, "\n(AF)")
tbl$Dataset <- "BrCa (OHSU)"
for(col in c("Moving", "Target", "Channel", "Strategy", "Dataset")) {
    tbl[,col] <- as.character(tbl[,col])
}
tbl3 <- tbl

tbl <- data.frame("Moving" = 2:11, "Target" = 1,
                   "Distance" = c(0.000185, 0.000168, 0.000219, 0.000161, 0.000208, 0.000209, 0.000195, 0.000205, 0.000251, 0.000174))
tbl <- tbl[, c("Moving", "Distance")]
tbl$Strategy <- "Non-linear; Intensity\n(AF)."
colnames(tbl) <- c("Moving", "Distance", "Strategy")
tbl$Target <- "R1"
tbl$Channel <- "AF"
tbl$Dataset <- "Tonsil (OHSU)"
for(col in c("Moving", "Target", "Channel", "Strategy", "Dataset")) {
    tbl[,col] <- as.character(tbl[,col])
}
tbl4 <- tbl

tbl <- rbind(tbl1, tbl2, tbl3, tbl4)

levels <- c("Unaligned\n(DAPI)", "Linear; Keypoint\n(OHSU; DAPI)", "Linear; Keypoint\n(DAPI)",
            "Unaligned\n(AF)", "Linear; Keypoint\n(AF)", "Non-linear; Intensity\n(AF)", "Non-linear; Intensity\n(AF).")

tbl$Strategy <- factor(tbl$Strategy, levels = levels)

tmp <- tbl
tmp$Strategy <- as.character(tmp$Strategy)
tmp$Strategy <- gsub(tmp$Strategy, pattern="\n", replacement=" ")

write.table(tmp, file = "distance-results-tbl.tsv", sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

g <- ggplot(data = tbl, aes(x = Strategy, y = Distance, colour = Dataset))
g <- g + scale_y_continuous(trans='log2', labels = scientific)
g <- g + geom_beeswarm(size = 3)
g <- g + theme(axis.text=element_text(size=20), axis.title=element_text(size=20),
               axis.text.x = element_text(angle = 45, hjust = 1),
               legend.text=element_text(size=20), legend.title=element_text(size=20))

png("registration-results.png", width = 480 * 2)
print(g)
d <- dev.off()
