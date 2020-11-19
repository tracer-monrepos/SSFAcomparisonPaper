Summary statistics on SSFA datasets
================
Ivan Calandra
2020-11-19 13:40:21

-   [Goal of the script](#goal-of-the-script)
-   [Load packages](#load-packages)
-   [Read in data](#read-in-data)
    -   [Get names, path and information of input
        file](#get-names-path-and-information-of-input-file)
    -   [Read in Rbin file](#read-in-rbin-file)
-   [Summary statistics](#summary-statistics)
    -   [Create function to compute the statistics at
        once](#create-function-to-compute-the-statistics-at-once)
    -   [Define grouping and numerical variables to
        use](#define-grouping-and-numerical-variables-to-use)
    -   [Compute summary statistics](#compute-summary-statistics)
-   [Write results to XLSX](#write-results-to-xlsx)
-   [sessionInfo() and RStudio
    version](#sessioninfo-and-rstudio-version)

------------------------------------------------------------------------

# Goal of the script

This script computes standard descriptive statistics for each group.  
The groups are based on:

-   Diet (Guinea Pigs and Sheeps datasets)  
-   Material and before/after experiment (Lithics dataset)

It computes the following statistics:

-   sample size (n = `length`)  
-   smallest value (`min`)  
-   largest value (`max`)
-   mean  
-   median  
-   standard deviation (`sd`)

``` r
dir_in  <- "R_analysis/derived_data"
dir_out <- "R_analysis/summary_stats"
```

Input Rbin data file must be located in
“\~/R\_analysis/derived\_data”.  
Summary statistics table will be saved in
“\~/R\_analysis/summary\_stats”.

The knit directory for this script is the project directory.

------------------------------------------------------------------------

# Load packages

``` r
library(openxlsx)
library(R.utils)
library(tools)
library(doBy)
library(tidyverse)
```

------------------------------------------------------------------------

# Read in data

## Get names, path and information of input file

``` r
info_in <- list.files(dir_in, pattern = "\\.Rbin$", full.names = TRUE) %>% 
           md5sum()
```

The checksum (MD5 hashes) of the loaded file is:

                   files                         checksum
    1 SSFA_all_data.Rbin 55465bdc7308d20cdff2d3d1bdeea63d

## Read in Rbin file

``` r
all_data <- loadObject(names(info_in))
str(all_data)
```

    'data.frame':   280 obs. of  13 variables:
     $ Dataset     : chr  "GuineaPigs" "GuineaPigs" "GuineaPigs" "GuineaPigs" ...
     $ Name        : chr  "capor_2CC4B1_txP4_#1_1_100xL_1" "capor_2CC4B1_txP4_#1_1_100xL_1" "capor_2CC4B1_txP4_#1_1_100xL_2" "capor_2CC4B1_txP4_#1_1_100xL_2" ...
     $ Software    : chr  "ConfoMap" "Toothfrax" "ConfoMap" "Toothfrax" ...
     $ Diet        : chr  "Dry Lucerne" "Dry Lucerne" "Dry Lucerne" "Dry Lucerne" ...
     $ Treatment   : chr  NA NA NA NA ...
     $ Before.after: Factor w/ 2 levels "Before","After": NA NA NA NA NA NA NA NA NA NA ...
     $ epLsar      : num  0.00196 0.00147 0.00366 0.00269 0.00314 ...
     $ R²          : num  0.998 0.999 0.999 1 0.999 ...
     $ Asfc        : num  14.3 12.9 12.9 12 13.7 ...
     $ Smfc        : num  0.415 0.119 0.441 0.119 0.441 ...
     $ HAsfc9      : num  0.164 0.182 0.171 0.159 0.131 ...
     $ HAsfc81     : num  0.368 0.337 0.417 0.382 0.352 ...
     $ NewEplsar   : num  0.0184 NA 0.0189 NA 0.0187 ...
     - attr(*, "comment")= Named chr [1:8] "<no unit>" "<no unit>" "<no unit>" "<no unit>" ...
      ..- attr(*, "names")= chr [1:8] "epLsar" "NewEplsar" "R²" "Asfc" ...

------------------------------------------------------------------------

# Summary statistics

## Create function to compute the statistics at once

``` r
nminmaxmeanmedsd <- function(x){
    y <- x[!is.na(x)]     # Exclude NAs
    n_test <- length(y)   # Sample size (n)
    min_test <- min(y)    # Minimum
    max_test <- max(y)    # Maximum
    mean_test <- mean(y)  # Mean
    med_test <- median(y) # Median
    sd_test <- sd(y)      # Standard deviation
    out <- c(n_test, min_test, max_test, mean_test, med_test, sd_test) # Concatenate
    names(out) <- c("n", "min", "max", "mean", "median", "sd")         # Name values
    return(out)                                                        # Object to return
}
```

## Define grouping and numerical variables to use

``` r
# Define grouping variables
grp <- c("Dataset", "Software", "Diet", "Treatment", "Before.after")
```

The following grouping variables will be used:

    Dataset
    Software
    Diet
    Treatment
    Before.after

All numerical variables will be used:

    epLsar
    R²
    Asfc
    Smfc
    HAsfc9
    HAsfc81
    NewEplsar

## Compute summary statistics

``` r
all_stats <- as.formula(paste(".~", paste(grp, collapse = "+"))) %>% 
             summaryBy(data = all_data, FUN = nminmaxmeanmedsd)
```

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

    Warning in min(y): no non-missing arguments to min; returning Inf

    Warning in max(y): no non-missing arguments to max; returning -Inf

The warnings are due to the missing values for NewEplsar - Toothfrax.
NewEplsar can only be computed in ConfoMap.

``` r
str(all_stats)
```

    'data.frame':   30 obs. of  47 variables:
     $ Dataset         : chr  "GuineaPigs" "GuineaPigs" "GuineaPigs" "GuineaPigs" ...
     $ Software        : chr  "ConfoMap" "ConfoMap" "ConfoMap" "Toothfrax" ...
     $ Diet            : chr  "Dry Bamboo" "Dry Grass" "Dry Lucerne" "Dry Bamboo" ...
     $ Treatment       : chr  NA NA NA NA ...
     $ Before.after    : Factor w/ 2 levels "Before","After": NA NA NA NA NA NA 1 2 1 2 ...
     $ epLsar.n        : num  24 22 24 24 22 24 4 4 4 4 ...
     $ epLsar.min      : num  0.002936 0.000322 0.001195 0.003345 0.000234 ...
     $ epLsar.max      : num  0.00731 0.00534 0.00435 0.0066 0.00438 ...
     $ epLsar.mean     : num  0.00511 0.00186 0.00258 0.00503 0.00183 ...
     $ epLsar.median   : num  0.00522 0.0016 0.00251 0.00502 0.00167 ...
     $ epLsar.sd       : num  0.001063 0.001116 0.000891 0.000987 0.00105 ...
     $ R².n            : num  24 22 24 24 22 24 4 4 4 4 ...
     $ R².min          : num  0.998 0.998 0.998 0.997 0.996 ...
     $ R².max          : num  1 0.999 0.999 1 1 ...
     $ R².mean         : num  0.999 0.999 0.999 0.999 0.999 ...
     $ R².median       : num  0.999 0.999 0.999 0.999 0.999 ...
     $ R².sd           : num  0.000514 0.00028 0.000449 0.000613 0.000832 ...
     $ Asfc.n          : num  24 22 24 24 22 24 4 4 4 4 ...
     $ Asfc.min        : num  10.54 10.13 8.95 9.47 9.14 ...
     $ Asfc.max        : num  32.3 24 19.6 31.1 22.4 ...
     $ Asfc.mean       : num  17.4 14.8 13.2 16.4 13.7 ...
     $ Asfc.median     : num  16.3 13.7 13.2 15.2 12.4 ...
     $ Asfc.sd         : num  6.1 3.85 2.71 5.91 3.62 ...
     $ Smfc.n          : num  24 22 24 24 22 24 4 4 4 4 ...
     $ Smfc.min        : num  0.229 0.161 0.216 0.119 0.053 ...
     $ Smfc.max        : num  0.527 0.63 0.496 0.119 0.119 ...
     $ Smfc.mean       : num  0.357 0.379 0.375 0.119 0.11 ...
     $ Smfc.median     : num  0.358 0.38 0.38 0.119 0.119 ...
     $ Smfc.sd         : num  0.0606 0.1072 0.0843 0 0.0233 ...
     $ HAsfc9.n        : num  24 22 24 24 22 24 4 4 4 4 ...
     $ HAsfc9.min      : num  0.125 0.152 0.121 0.113 0.145 ...
     $ HAsfc9.max      : num  0.289 0.561 0.429 0.274 0.489 ...
     $ HAsfc9.mean     : num  0.208 0.311 0.239 0.191 0.287 ...
     $ HAsfc9.median   : num  0.21 0.289 0.205 0.197 0.254 ...
     $ HAsfc9.sd       : num  0.0484 0.1393 0.0913 0.0458 0.1202 ...
     $ HAsfc81.n       : num  24 22 24 24 22 24 4 4 4 4 ...
     $ HAsfc81.min     : num  0.251 0.278 0.309 0.255 0.223 ...
     $ HAsfc81.max     : num  0.45 0.614 0.856 0.428 0.569 ...
     $ HAsfc81.mean    : num  0.36 0.44 0.449 0.331 0.406 ...
     $ HAsfc81.median  : num  0.369 0.426 0.41 0.338 0.42 ...
     $ HAsfc81.sd      : num  0.0489 0.0966 0.1219 0.0464 0.0922 ...
     $ NewEplsar.n     : num  24 22 24 0 0 0 4 4 4 4 ...
     $ NewEplsar.min   : num  0.0188 0.0173 0.0181 Inf Inf ...
     $ NewEplsar.max   : num  0.0207 0.0198 0.0192 -Inf -Inf ...
     $ NewEplsar.mean  : num  0.0198 0.018 0.0185 NaN NaN ...
     $ NewEplsar.median: num  0.0198 0.018 0.0185 NA NA ...
     $ NewEplsar.sd    : num  0.000474 0.000626 0.000349 NA NA ...

------------------------------------------------------------------------

# Write results to XLSX

``` r
xlsx_md5 <- paste0(dir_out, "/SSFA_summary_stats.xlsx") %T>% 
            write.xlsx(list(summary_stats = all_stats), file = .) %>% 
            md5sum()
```

The checksum (MD5 hashes) of the exported file is:

                        files                         checksum
    1 SSFA_summary_stats.xlsx aeaff0fd6daa923692991db35ac2f4ba

------------------------------------------------------------------------

# sessionInfo() and RStudio version

``` r
sessionInfo()
```

    R version 4.0.3 (2020-10-10)
    Platform: x86_64-w64-mingw32/x64 (64-bit)
    Running under: Windows 10 x64 (build 18362)

    Matrix products: default

    locale:
    [1] LC_COLLATE=French_France.1252  LC_CTYPE=French_France.1252   
    [3] LC_MONETARY=French_France.1252 LC_NUMERIC=C                  
    [5] LC_TIME=French_France.1252    

    attached base packages:
    [1] tools     stats     graphics  grDevices utils     datasets  methods  
    [8] base     

    other attached packages:
     [1] forcats_0.5.0     stringr_1.4.0     dplyr_1.0.2       purrr_0.3.4      
     [5] readr_1.4.0       tidyr_1.1.2       tibble_3.0.4      ggplot2_3.3.2    
     [9] tidyverse_1.3.0   doBy_4.6.8        R.utils_2.10.1    R.oo_1.24.0      
    [13] R.methodsS3_1.8.1 openxlsx_4.2.3   

    loaded via a namespace (and not attached):
     [1] tidyselect_1.1.0  xfun_0.19         haven_2.3.1       lattice_0.20-41  
     [5] colorspace_2.0-0  vctrs_0.3.4       generics_0.1.0    htmltools_0.5.0  
     [9] yaml_2.2.1        rlang_0.4.8       pillar_1.4.6      withr_2.3.0      
    [13] glue_1.4.2        DBI_1.1.0         dbplyr_2.0.0      modelr_0.1.8     
    [17] readxl_1.3.1      lifecycle_0.2.0   munsell_0.5.0     gtable_0.3.0     
    [21] cellranger_1.1.0  rvest_0.3.6       zip_2.1.1         evaluate_0.14    
    [25] knitr_1.30        fansi_0.4.1       broom_0.7.2       Rcpp_1.0.5       
    [29] scales_1.1.1      backports_1.2.0   jsonlite_1.7.1    fs_1.5.0         
    [33] Deriv_4.1.1       hms_0.5.3         digest_0.6.27     stringi_1.5.3    
    [37] grid_4.0.3        rprojroot_2.0.2   cli_2.1.0         magrittr_1.5     
    [41] crayon_1.3.4      pkgconfig_2.0.3   MASS_7.3-53       ellipsis_0.3.1   
    [45] Matrix_1.2-18     xml2_1.3.2        reprex_0.3.0      lubridate_1.7.9.2
    [49] rstudioapi_0.13   assertthat_0.2.1  rmarkdown_2.5     httr_1.4.2       
    [53] R6_2.5.0          compiler_4.0.3   

RStudio version 1.4.1043.

------------------------------------------------------------------------

END OF SCRIPT
