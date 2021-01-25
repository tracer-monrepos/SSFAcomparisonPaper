#written by Ivan Calandra, 2020-11-19

#Output current version of RStudio to a text file for reporting purposes.

vers <- as.character(RStudio.Version()$version)
writeLines(c(vers, "\n"), "R_analysis/scripts/SSFA_0_RStudioVersion.txt")
