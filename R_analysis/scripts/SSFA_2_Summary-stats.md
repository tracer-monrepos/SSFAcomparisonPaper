Summary statistics on SSFA datasets
================
Ivan Calandra
2021-12-01 10:51:14

-   [Goal of the script](#goal-of-the-script)
-   [Load packages](#load-packages)
-   [Read in data](#read-in-data)
    -   [Get name and path of input
        file](#get-name-and-path-of-input-file)
    -   [Read in Rbin file](#read-in-rbin-file)
-   [Summary statistics](#summary-statistics)
    -   [Create function to compute the statistics at
        once](#create-function-to-compute-the-statistics-at-once)
    -   [Define grouping and numerical variables to
        use](#define-grouping-and-numerical-variables-to-use)
    -   [Exclude surfaces with NMP >=
        20%](#exclude-surfaces-with-nmp--20)
    -   [Compute summary statistics](#compute-summary-statistics)
-   [Write results to XLSX](#write-results-to-xlsx)
-   [sessionInfo() and RStudio
    version](#sessioninfo-and-rstudio-version)
-   [Cite R packages used](#cite-r-packages-used)

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

Input Rbin data file must be located in “\~/R_analysis/derived_data”.  
Summary statistics table will be saved in “\~/R_analysis/summary_stats”.

The knit directory for this script is the project directory.

------------------------------------------------------------------------

# Load packages

``` r
pack_to_load <- c("openxlsx", "R.utils", "doBy", "tidyverse")
sapply(pack_to_load, library, character.only = TRUE, logical.return = TRUE) 
```

     openxlsx   R.utils      doBy tidyverse 
         TRUE      TRUE      TRUE      TRUE 

------------------------------------------------------------------------

# Read in data

## Get name and path of input file

``` r
info_in <- list.files(dir_in, pattern = "\\.Rbin$", full.names = TRUE)
info_in
```

    [1] "R_analysis/derived_data/SSFA_all_data.Rbin"

## Read in Rbin file

``` r
all_data <- loadObject(info_in)
str(all_data)
```

    'data.frame':   284 obs. of  15 variables:
     $ Dataset     : Factor w/ 3 levels "GuineaPigs","Sheeps",..: 1 1 1 1 1 1 1 1 1 1 ...
     $ Name        : Factor w/ 142 levels "capor_2CC4B1_txP4_#1_1_100xL_1",..: 1 1 2 2 3 3 4 4 5 5 ...
     $ Software    : Factor w/ 2 levels "ConfoMap","Toothfrax": 1 2 1 2 1 2 1 2 1 2 ...
     $ Diet        : Factor w/ 7 levels "Dry bamboo","Dry grass",..: 3 3 3 3 3 3 3 3 3 3 ...
     $ Treatment   : Factor w/ 4 levels "Control","RubDirt",..: NA NA NA NA NA NA NA NA NA NA ...
     $ Before.after: Factor w/ 2 levels "Before","After": NA NA NA NA NA NA NA NA NA NA ...
     $ NMP         : num  1.896 1.896 1.308 1.308 0.806 ...
     $ NMP_cat     : Ord.factor w/ 4 levels "0-5%"<"5-10%"<..: 1 1 1 1 1 1 1 1 1 1 ...
     $ epLsar      : num  0.00196 0.00147 0.00366 0.00269 0.00314 ...
     $ Rsquared    : num  0.997 0.999 0.998 1 0.997 ...
     $ Asfc        : num  16 12.9 14.1 12 15.1 ...
     $ Smfc        : num  0.33 0.119 0.35 0.119 0.33 ...
     $ HAsfc9      : num  0.179 0.182 0.136 0.159 0.131 ...
     $ HAsfc81     : num  0.391 0.337 0.443 0.382 0.357 ...
     $ NewEplsar   : num  0.0184 NA 0.0189 NA 0.0187 ...
     - attr(*, "comment")= Named chr [1:9] "%" "<no unit>" "<no unit>" "<no unit>" ...
      ..- attr(*, "names")= chr [1:9] "NMP" "epLsar" "NewEplsar" "Rsquared" ...

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
grp <- c("Dataset", "Diet", "Treatment", "Before.after", "NMP_cat", "Software")
```

The following grouping variables will be used:

    Dataset
    Diet
    Treatment
    Before.after
    NMP_cat
    Software

All numerical variables will be used:

    NMP
    epLsar
    Rsquared
    Asfc
    Smfc
    HAsfc9
    HAsfc81
    NewEplsar

## Exclude surfaces with NMP \>= 20%

``` r
data_nmp0_20 <- filter(all_data, NMP_cat != "20-100%")
str(data_nmp0_20)
```

    'data.frame':   278 obs. of  15 variables:
     $ Dataset     : Factor w/ 3 levels "GuineaPigs","Sheeps",..: 1 1 1 1 1 1 1 1 1 1 ...
     $ Name        : Factor w/ 142 levels "capor_2CC4B1_txP4_#1_1_100xL_1",..: 1 1 2 2 3 3 4 4 5 5 ...
     $ Software    : Factor w/ 2 levels "ConfoMap","Toothfrax": 1 2 1 2 1 2 1 2 1 2 ...
     $ Diet        : Factor w/ 7 levels "Dry bamboo","Dry grass",..: 3 3 3 3 3 3 3 3 3 3 ...
     $ Treatment   : Factor w/ 4 levels "Control","RubDirt",..: NA NA NA NA NA NA NA NA NA NA ...
     $ Before.after: Factor w/ 2 levels "Before","After": NA NA NA NA NA NA NA NA NA NA ...
     $ NMP         : num  1.896 1.896 1.308 1.308 0.806 ...
     $ NMP_cat     : Ord.factor w/ 4 levels "0-5%"<"5-10%"<..: 1 1 1 1 1 1 1 1 1 1 ...
     $ epLsar      : num  0.00196 0.00147 0.00366 0.00269 0.00314 ...
     $ Rsquared    : num  0.997 0.999 0.998 1 0.997 ...
     $ Asfc        : num  16 12.9 14.1 12 15.1 ...
     $ Smfc        : num  0.33 0.119 0.35 0.119 0.33 ...
     $ HAsfc9      : num  0.179 0.182 0.136 0.159 0.131 ...
     $ HAsfc81     : num  0.391 0.337 0.443 0.382 0.357 ...
     $ NewEplsar   : num  0.0184 NA 0.0189 NA 0.0187 ...
     - attr(*, "comment")= Named chr [1:9] "%" "<no unit>" "<no unit>" "<no unit>" ...
      ..- attr(*, "names")= chr [1:9] "NMP" "epLsar" "NewEplsar" "Rsquared" ...

Surfaces with more than 20% NMP are likely the result of issues during
acquisition and must be excluded. While surfaces having more than 10%
(or 5%) NMP can also be problematic, they are still included in the
summary statistics. Nevertheless, the summary statistics will be
calculated for each NMP category (i.e. 0-5, 5-10 and 10-20% NMP).

6 surfaces have been filtered out, i.e. 3 surfaces for each software
package.

## Compute summary statistics

``` r
# All data
all_stats <- as.formula(paste(".~", paste(grp, collapse = "+"))) %>% 
             summaryBy(data = data_nmp0_20, FUN = nminmaxmeanmedsd)
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

    'data.frame':   42 obs. of  54 variables:
     $ Dataset         : Factor w/ 3 levels "GuineaPigs","Sheeps",..: 1 1 1 1 1 1 1 1 1 1 ...
     $ Diet            : Factor w/ 7 levels "Dry bamboo","Dry grass",..: 1 1 1 1 1 1 2 2 2 2 ...
     $ Treatment       : Factor w/ 4 levels "Control","RubDirt",..: NA NA NA NA NA NA NA NA NA NA ...
     $ Before.after    : Factor w/ 2 levels "Before","After": NA NA NA NA NA NA NA NA NA NA ...
     $ NMP_cat         : Ord.factor w/ 4 levels "0-5%"<"5-10%"<..: 1 1 2 2 3 3 1 1 2 2 ...
     $ Software        : Factor w/ 2 levels "ConfoMap","Toothfrax": 1 2 1 2 1 2 1 2 1 2 ...
     $ NMP.n           : num  17 17 4 4 3 3 8 8 8 8 ...
     $ NMP.min         : num  0.717 0.717 5.053 5.053 10.47 ...
     $ NMP.max         : num  4.02 4.02 7.84 7.84 11.32 ...
     $ NMP.mean        : num  2.31 2.31 6.12 6.12 11.01 ...
     $ NMP.median      : num  2.19 2.19 5.8 5.8 11.22 ...
     $ NMP.sd          : num  0.85 0.85 1.23 1.23 0.467 ...
     $ epLsar.n        : num  17 17 4 4 3 3 8 8 8 8 ...
     $ epLsar.min      : num  0.00344 0.00334 0.00367 0.00345 0.00294 ...
     $ epLsar.max      : num  0.00731 0.0066 0.00561 0.00538 0.00533 ...
     $ epLsar.mean     : num  0.00542 0.00534 0.00455 0.00431 0.00407 ...
     $ epLsar.median   : num  0.00538 0.0053 0.00446 0.00421 0.00395 ...
     $ epLsar.sd       : num  0.00093 0.000866 0.001016 0.001011 0.001202 ...
     $ Rsquared.n      : num  17 17 4 4 3 3 8 8 8 8 ...
     $ Rsquared.min    : num  0.993 0.997 0.993 0.999 0.995 ...
     $ Rsquared.max    : num  0.998 1 0.997 1 0.997 ...
     $ Rsquared.mean   : num  0.995 0.999 0.996 0.999 0.997 ...
     $ Rsquared.median : num  0.995 0.999 0.996 0.999 0.997 ...
     $ Rsquared.sd     : num  0.001456 0.000688 0.001563 0.000187 0.001063 ...
     $ Asfc.n          : num  17 17 4 4 3 3 8 8 8 8 ...
     $ Asfc.min        : num  11.54 9.47 24.84 19.34 30.15 ...
     $ Asfc.max        : num  26.4 20.9 38.6 31.1 32.9 ...
     $ Asfc.mean       : num  16.4 13.4 28.8 22.8 31.7 ...
     $ Asfc.median     : num  14.5 12 25.8 20.4 32.1 ...
     $ Asfc.sd         : num  4.52 3.39 6.57 5.55 1.39 ...
     $ Smfc.n          : num  17 17 4 4 3 3 8 8 8 8 ...
     $ Smfc.min        : num  0.234 0.119 0.33 0.119 0.371 ...
     $ Smfc.max        : num  0.33 0.119 0.523 0.119 0.392 ...
     $ Smfc.mean       : num  0.291 0.119 0.399 0.119 0.378 ...
     $ Smfc.median     : num  0.295 0.119 0.371 0.119 0.371 ...
     $ Smfc.sd         : num  0.0248 0 0.085 0 0.0126 ...
     $ HAsfc9.n        : num  17 17 4 4 3 3 8 8 8 8 ...
     $ HAsfc9.min      : num  0.128 0.113 0.189 0.184 0.131 ...
     $ HAsfc9.max      : num  0.284 0.274 0.27 0.246 0.26 ...
     $ HAsfc9.mean     : num  0.205 0.187 0.235 0.213 0.211 ...
     $ HAsfc9.median   : num  0.196 0.171 0.241 0.211 0.242 ...
     $ HAsfc9.sd       : num  0.0502 0.0497 0.0338 0.0257 0.0697 ...
     $ HAsfc81.n       : num  17 17 4 4 3 3 8 8 8 8 ...
     $ HAsfc81.min     : num  0.284 0.255 0.325 0.273 0.317 ...
     $ HAsfc81.max     : num  0.426 0.428 0.441 0.357 0.41 ...
     $ HAsfc81.mean    : num  0.362 0.334 0.395 0.334 0.367 ...
     $ HAsfc81.median  : num  0.356 0.338 0.407 0.352 0.375 ...
     $ HAsfc81.sd      : num  0.0457 0.0512 0.053 0.0407 0.0467 ...
     $ NewEplsar.n     : num  17 0 4 0 3 0 8 0 8 0 ...
     $ NewEplsar.min   : num  0.0191 Inf 0.0191 Inf 0.0188 ...
     $ NewEplsar.max   : num  0.0207 -Inf 0.02 -Inf 0.0198 ...
     $ NewEplsar.mean  : num  0.0199 NaN 0.0195 NaN 0.0193 ...
     $ NewEplsar.median: num  0.0198 NA 0.0195 NA 0.0192 ...
     $ NewEplsar.sd    : num  0.000394 NA 0.000498 NA 0.000538 ...

------------------------------------------------------------------------

# Write results to XLSX

``` r
write.xlsx(list(summary_stats = all_stats), 
           file = paste0(dir_out, "/SSFA_summary-stats.xlsx"))
```

    Error in saveWorkbook(wb, file = file, overwrite = overwrite): File already exists!

------------------------------------------------------------------------

# sessionInfo() and RStudio version

``` r
sessionInfo()
```

    R version 4.1.2 (2021-11-01)
    Platform: x86_64-w64-mingw32/x64 (64-bit)
    Running under: Windows 10 x64 (build 19043)

    Matrix products: default

    locale:
    [1] LC_COLLATE=English_United Kingdom.1252 
    [2] LC_CTYPE=English_United Kingdom.1252   
    [3] LC_MONETARY=English_United Kingdom.1252
    [4] LC_NUMERIC=C                           
    [5] LC_TIME=English_United Kingdom.1252    

    attached base packages:
    [1] stats     graphics  grDevices datasets  utils     methods   base     

    other attached packages:
     [1] forcats_0.5.1     stringr_1.4.0     dplyr_1.0.7       purrr_0.3.4      
     [5] readr_2.1.0       tidyr_1.1.4       tibble_3.1.6      ggplot2_3.3.5    
     [9] tidyverse_1.3.1   doBy_4.6.11       R.utils_2.11.0    R.oo_1.24.0      
    [13] R.methodsS3_1.8.1 openxlsx_4.2.4   

    loaded via a namespace (and not attached):
     [1] Rcpp_1.0.7           lubridate_1.8.0      curry_0.1.1         
     [4] lattice_0.20-45      assertthat_0.2.1     rprojroot_2.0.2     
     [7] digest_0.6.28        utf8_1.2.2           R6_2.5.1            
    [10] cellranger_1.1.0     backports_1.4.0      reprex_2.0.1        
    [13] evaluate_0.14        httr_1.4.2           pillar_1.6.4        
    [16] rlang_0.4.12         readxl_1.3.1         rstudioapi_0.13     
    [19] jquerylib_0.1.4      Matrix_1.3-4         rmarkdown_2.11      
    [22] munsell_0.5.0        broom_0.7.10         compiler_4.1.2      
    [25] Deriv_4.1.3          modelr_0.1.8         xfun_0.28           
    [28] pkgconfig_2.0.3      microbenchmark_1.4.9 htmltools_0.5.2     
    [31] tidyselect_1.1.1     fansi_0.5.0          withr_2.4.2         
    [34] crayon_1.4.2         tzdb_0.2.0           dbplyr_2.1.1        
    [37] MASS_7.3-54          grid_4.1.2           jsonlite_1.7.2      
    [40] gtable_0.3.0         lifecycle_1.0.1      DBI_1.1.1           
    [43] magrittr_2.0.1       scales_1.1.1         zip_2.2.0           
    [46] cli_3.1.0            stringi_1.7.6        renv_0.14.0         
    [49] fs_1.5.0             xml2_1.3.2           ellipsis_0.3.2      
    [52] generics_0.1.1       vctrs_0.3.8          tools_4.1.2         
    [55] glue_1.5.0           hms_1.1.1            fastmap_1.1.0       
    [58] yaml_2.2.1           colorspace_2.0-2     rvest_1.0.2         
    [61] knitr_1.36           haven_2.4.3         

RStudio version 2021.9.1.372.

------------------------------------------------------------------------

# Cite R packages used

    openxlsx 
    Philipp Schauberger and Alexander Walker (2021). openxlsx: Read, Write and Edit xlsx Files. R package version 4.2.4. https://CRAN.R-project.org/package=openxlsx 
     
    R.utils 
    Henrik Bengtsson (2021). R.utils: Various Programming Utilities. R package version 2.11.0. https://CRAN.R-project.org/package=R.utils 
     
    doBy 
    Søren Højsgaard and Ulrich Halekoh (2021). doBy: Groupwise Statistics, LSmeans, Linear Contrasts, Utilities. R package version 4.6.11. https://CRAN.R-project.org/package=doBy 
     
    tidyverse 
    Wickham et al., (2019). Welcome to the tidyverse. Journal of Open Source Software, 4(43), 1686, https://doi.org/10.21105/joss.01686 
     

------------------------------------------------------------------------

END OF SCRIPT
