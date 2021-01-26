Summary statistics on SSFA datasets
================
Ivan Calandra
2021-01-26 10:03:52

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
    1 SSFA_all_data.Rbin d2fc242595873d663b12beb7633acb0a

## Read in Rbin file

``` r
all_data <- loadObject(names(info_in))
str(all_data)
```

    'data.frame':   280 obs. of  13 variables:
     $ Dataset     : chr  "GuineaPigs" "GuineaPigs" "GuineaPigs" "GuineaPigs" ...
     $ Name        : chr  "capor_2CC4B1_txP4_#1_1_100xL_1" "capor_2CC4B1_txP4_#1_1_100xL_1" "capor_2CC4B1_txP4_#1_1_100xL_2" "capor_2CC4B1_txP4_#1_1_100xL_2" ...
     $ Software    : chr  "ConfoMap" "Toothfrax" "ConfoMap" "Toothfrax" ...
     $ Diet        : chr  "Dry lucerne" "Dry lucerne" "Dry lucerne" "Dry lucerne" ...
     $ Treatment   : Factor w/ 4 levels "Control","RubDirt",..: NA NA NA NA NA NA NA NA NA NA ...
     $ Before.after: Factor w/ 2 levels "Before","After": NA NA NA NA NA NA NA NA NA NA ...
     $ epLsar      : num  0.00207 0.00147 0.00381 0.00269 0.00327 ...
     $ R²          : num  0.998 0.999 0.998 1 0.998 ...
     $ Asfc        : num  10.8 12.9 10 12 10.5 ...
     $ Smfc        : num  0.448 0.119 0.591 0.119 0.591 ...
     $ HAsfc9      : num  0.181 0.182 0.19 0.159 0.114 ...
     $ HAsfc81     : num  0.365 0.337 0.407 0.382 0.363 ...
     $ NewEplsar   : num  0.0185 NA 0.019 NA 0.0188 ...
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
     $ Diet            : chr  "Dry bamboo" "Dry grass" "Dry lucerne" "Dry bamboo" ...
     $ Treatment       : Factor w/ 4 levels "Control","RubDirt",..: NA NA NA NA NA NA 1 1 2 2 ...
     $ Before.after    : Factor w/ 2 levels "Before","After": NA NA NA NA NA NA 1 2 1 2 ...
     $ epLsar.n        : num  24 22 24 24 22 24 3 3 4 4 ...
     $ epLsar.min      : num  0.003078 0.000347 0.001256 0.003345 0.000234 ...
     $ epLsar.max      : num  0.00752 0.00553 0.00455 0.0066 0.00438 ...
     $ epLsar.mean     : num  0.00531 0.00195 0.0027 0.00503 0.00183 ...
     $ epLsar.median   : num  0.00542 0.00166 0.00263 0.00502 0.00167 ...
     $ epLsar.sd       : num  0.001087 0.001148 0.000921 0.000987 0.00105 ...
     $ R².n            : num  24 22 24 24 22 24 3 3 4 4 ...
     $ R².min          : num  0.998 0.997 0.998 0.997 0.996 ...
     $ R².max          : num  0.999 0.999 0.999 1 1 ...
     $ R².mean         : num  0.998 0.998 0.998 0.999 0.999 ...
     $ R².median       : num  0.998 0.998 0.998 0.999 0.999 ...
     $ R².sd           : num  0.00034 0.000423 0.000254 0.000613 0.000832 ...
     $ Asfc.n          : num  24 22 24 24 22 24 3 3 4 4 ...
     $ Asfc.min        : num  6.75 5.93 5.52 9.47 9.14 ...
     $ Asfc.max        : num  28 18.9 15.7 31.1 22.4 ...
     $ Asfc.mean       : num  13.44 11.06 9.89 16.43 13.74 ...
     $ Asfc.median     : num  11.97 9.51 10.01 15.21 12.41 ...
     $ Asfc.sd         : num  5.71 3.62 2.4 5.91 3.62 ...
     $ Smfc.n          : num  24 22 24 24 22 24 3 3 4 4 ...
     $ Smfc.min        : num  0.401 0.231 0.34 0.119 0.053 ...
     $ Smfc.max        : num  0.738 0.698 0.661 0.119 0.119 ...
     $ Smfc.mean       : num  0.539 0.508 0.527 0.119 0.11 ...
     $ Smfc.median     : num  0.575 0.545 0.575 0.119 0.119 ...
     $ Smfc.sd         : num  0.0958 0.1274 0.1024 0 0.0233 ...
     $ HAsfc9.n        : num  24 22 24 24 22 24 3 3 4 4 ...
     $ HAsfc9.min      : num  0.1189 0.1612 0.0983 0.1127 0.1446 ...
     $ HAsfc9.max      : num  0.31 0.496 0.402 0.274 0.489 ...
     $ HAsfc9.mean     : num  0.217 0.301 0.242 0.191 0.287 ...
     $ HAsfc9.median   : num  0.212 0.284 0.21 0.197 0.254 ...
     $ HAsfc9.sd       : num  0.0503 0.1188 0.0911 0.0458 0.1202 ...
     $ HAsfc81.n       : num  24 22 24 24 22 24 3 3 4 4 ...
     $ HAsfc81.min     : num  0.291 0.268 0.318 0.255 0.223 ...
     $ HAsfc81.max     : num  0.486 0.599 0.925 0.428 0.569 ...
     $ HAsfc81.mean    : num  0.383 0.433 0.449 0.331 0.406 ...
     $ HAsfc81.median  : num  0.387 0.44 0.421 0.338 0.42 ...
     $ HAsfc81.sd      : num  0.0542 0.0804 0.1311 0.0464 0.0922 ...
     $ NewEplsar.n     : num  24 22 24 0 0 0 3 3 4 4 ...
     $ NewEplsar.min   : num  0.0188 0.0173 0.0181 Inf Inf ...
     $ NewEplsar.max   : num  0.0208 0.0199 0.0193 -Inf -Inf ...
     $ NewEplsar.mean  : num  0.0199 0.018 0.0186 NaN NaN ...
     $ NewEplsar.median: num  0.0199 0.0181 0.0186 NA NA ...
     $ NewEplsar.sd    : num  0.000485 0.000653 0.000361 NA NA ...

------------------------------------------------------------------------

# Write results to XLSX

``` r
write.xlsx(list(summary_stats = all_stats), 
           file = paste0(dir_out, "/SSFA_summary_stats.xlsx"))
```

------------------------------------------------------------------------

# sessionInfo() and RStudio version

``` r
sessionInfo()
```

    R version 4.0.3 (2020-10-10)
    Platform: x86_64-w64-mingw32/x64 (64-bit)
    Running under: Windows 10 x64 (build 19041)

    Matrix products: default

    locale:
    [1] LC_COLLATE=French_France.1252  LC_CTYPE=French_France.1252   
    [3] LC_MONETARY=French_France.1252 LC_NUMERIC=C                  
    [5] LC_TIME=French_France.1252    

    attached base packages:
    [1] tools     stats     graphics  grDevices datasets  utils     methods  
    [8] base     

    other attached packages:
     [1] forcats_0.5.0     stringr_1.4.0     dplyr_1.0.3       purrr_0.3.4      
     [5] readr_1.4.0       tidyr_1.1.2       tibble_3.0.5      ggplot2_3.3.3    
     [9] tidyverse_1.3.0   doBy_4.6.8        R.utils_2.10.1    R.oo_1.24.0      
    [13] R.methodsS3_1.8.1 openxlsx_4.2.3   

    loaded via a namespace (and not attached):
     [1] tidyselect_1.1.0  xfun_0.20         haven_2.3.1       lattice_0.20-41  
     [5] colorspace_2.0-0  vctrs_0.3.6       generics_0.1.0    htmltools_0.5.1.1
     [9] yaml_2.2.1        rlang_0.4.10      pillar_1.4.7      withr_2.4.0      
    [13] glue_1.4.2        DBI_1.1.1         dbplyr_2.0.0      readxl_1.3.1     
    [17] modelr_0.1.8      lifecycle_0.2.0   cellranger_1.1.0  munsell_0.5.0    
    [21] gtable_0.3.0      rvest_0.3.6       zip_2.1.1         evaluate_0.14    
    [25] knitr_1.30        fansi_0.4.2       broom_0.7.3       Rcpp_1.0.6       
    [29] renv_0.12.5       scales_1.1.1      backports_1.2.1   jsonlite_1.7.2   
    [33] fs_1.5.0          Deriv_4.1.2       hms_1.0.0         digest_0.6.27    
    [37] stringi_1.5.3     grid_4.0.3        rprojroot_2.0.2   cli_2.2.0        
    [41] magrittr_2.0.1    crayon_1.3.4      pkgconfig_2.0.3   ellipsis_0.3.1   
    [45] MASS_7.3-53       Matrix_1.3-2      xml2_1.3.2        reprex_0.3.0     
    [49] lubridate_1.7.9.2 rstudioapi_0.13   assertthat_0.2.1  rmarkdown_2.6    
    [53] httr_1.4.2        R6_2.5.0          compiler_4.0.3   

RStudio version 1.4.1103.

------------------------------------------------------------------------

END OF SCRIPT
