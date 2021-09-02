Summary statistics on SSFA datasets
================
Ivan Calandra
2021-09-02 09:27:34

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
    -   [Exclude surfaces with NMP &gt; 20% (NMP\_cat =
        “20-100%”)](#exclude-surfaces-with-nmp--20-nmp_cat--20-100)
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

Input Rbin data file must be located in
“\~/R\_analysis/derived\_data”.  
Summary statistics table will be saved in
“\~/R\_analysis/summary\_stats”.

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
     $ Dataset     : chr  "GuineaPigs" "GuineaPigs" "GuineaPigs" "GuineaPigs" ...
     $ Name        : chr  "capor_2CC4B1_txP4_#1_1_100xL_1" "capor_2CC4B1_txP4_#1_1_100xL_1" "capor_2CC4B1_txP4_#1_1_100xL_2" "capor_2CC4B1_txP4_#1_1_100xL_2" ...
     $ Software    : chr  "ConfoMap" "Toothfrax" "ConfoMap" "Toothfrax" ...
     $ Diet        : chr  "Dry lucerne" "Dry lucerne" "Dry lucerne" "Dry lucerne" ...
     $ Treatment   : Factor w/ 4 levels "Control","RubDirt",..: NA NA NA NA NA NA NA NA NA NA ...
     $ Before.after: Factor w/ 2 levels "Before","After": NA NA NA NA NA NA NA NA NA NA ...
     $ NMP         : num  1.896 1.896 1.308 1.308 0.806 ...
     $ NMP_cat     : Ord.factor w/ 4 levels "0-5%"<"5-10%"<..: 1 1 1 1 1 1 1 1 1 1 ...
     $ epLsar      : num  0.00196 0.00147 0.00366 0.00269 0.00314 ...
     $ R²          : num  0.997 0.999 0.998 1 0.997 ...
     $ Asfc        : num  16 12.9 14.1 12 15.1 ...
     $ Smfc        : num  0.33 0.119 0.35 0.119 0.33 ...
     $ HAsfc9      : num  0.179 0.182 0.136 0.159 0.131 ...
     $ HAsfc81     : num  0.391 0.337 0.443 0.382 0.357 ...
     $ NewEplsar   : num  0.0184 NA 0.0189 NA 0.0187 ...
     - attr(*, "comment")= Named chr [1:9] "%" "<no unit>" "<no unit>" "<no unit>" ...
      ..- attr(*, "names")= chr [1:9] "NMP" "epLsar" "NewEplsar" "R²" ...

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
grp <- c("Dataset", "Software", "Diet", "Treatment", "Before.after", "NMP_cat")
```

The following grouping variables will be used:

    Dataset
    Software
    Diet
    Treatment
    Before.after
    NMP_cat

All numerical variables will be used:

    NMP
    epLsar
    R²
    Asfc
    Smfc
    HAsfc9
    HAsfc81
    NewEplsar

## Exclude surfaces with NMP &gt; 20% (NMP\_cat = “20-100%”)

``` r
data_nmp0_20 <- filter(all_data, NMP_cat != "20-100%")
```

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
     $ Dataset         : chr  "GuineaPigs" "GuineaPigs" "GuineaPigs" "GuineaPigs" ...
     $ Software        : chr  "ConfoMap" "ConfoMap" "ConfoMap" "ConfoMap" ...
     $ Diet            : chr  "Dry bamboo" "Dry bamboo" "Dry bamboo" "Dry grass" ...
     $ Treatment       : Factor w/ 4 levels "Control","RubDirt",..: NA NA NA NA NA NA NA NA NA NA ...
     $ Before.after    : Factor w/ 2 levels "Before","After": NA NA NA NA NA NA NA NA NA NA ...
     $ NMP_cat         : Ord.factor w/ 4 levels "0-5%"<"5-10%"<..: 1 2 3 1 2 3 1 2 1 2 ...
     $ NMP.n           : num  17 4 3 8 8 6 22 2 17 4 ...
     $ NMP.min         : num  0.717 5.053 10.47 0.815 5.545 ...
     $ NMP.max         : num  4.02 7.84 11.32 4.2 9.64 ...
     $ NMP.mean        : num  2.31 6.12 11.01 2.47 6.99 ...
     $ NMP.median      : num  2.19 5.8 11.22 2.4 6.62 ...
     $ NMP.sd          : num  0.85 1.23 0.467 1.194 1.374 ...
     $ epLsar.n        : num  17 4 3 8 8 6 22 2 17 4 ...
     $ epLsar.min      : num  0.003443 0.00367 0.002936 0.00131 0.000534 ...
     $ epLsar.max      : num  0.00731 0.00561 0.00533 0.00534 0.00323 ...
     $ epLsar.mean     : num  0.00542 0.00455 0.00407 0.00212 0.00189 ...
     $ epLsar.median   : num  0.00538 0.00446 0.00395 0.00169 0.00191 ...
     $ epLsar.sd       : num  0.00093 0.001016 0.001202 0.001326 0.000977 ...
     $ R².n            : num  17 4 3 8 8 6 22 2 17 4 ...
     $ R².min          : num  0.993 0.993 0.995 0.988 0.995 ...
     $ R².max          : num  0.998 0.997 0.997 0.998 0.998 ...
     $ R².mean         : num  0.995 0.996 0.997 0.996 0.997 ...
     $ R².median       : num  0.995 0.996 0.997 0.997 0.997 ...
     $ R².sd           : num  0.00146 0.00156 0.00106 0.00337 0.00114 ...
     $ Asfc.n          : num  17 4 3 8 8 6 22 2 17 4 ...
     $ Asfc.min        : num  11.5 24.8 30.2 11.6 12.5 ...
     $ Asfc.max        : num  26.4 38.6 32.9 20.9 23.9 ...
     $ Asfc.mean       : num  16.4 28.8 31.7 14.7 18.3 ...
     $ Asfc.median     : num  14.5 25.8 32.1 14 18.8 ...
     $ Asfc.sd         : num  4.52 6.57 1.39 3.21 4.53 ...
     $ Smfc.n          : num  17 4 3 8 8 6 22 2 17 4 ...
     $ Smfc.min        : num  0.234 0.33 0.371 0.166 0.248 ...
     $ Smfc.max        : num  0.33 0.523 0.392 0.466 0.523 ...
     $ Smfc.mean       : num  0.291 0.399 0.378 0.3 0.326 ...
     $ Smfc.median     : num  0.295 0.371 0.371 0.312 0.321 ...
     $ Smfc.sd         : num  0.0248 0.085 0.0126 0.0992 0.0913 ...
     $ HAsfc9.n        : num  17 4 3 8 8 6 22 2 17 4 ...
     $ HAsfc9.min      : num  0.128 0.189 0.131 0.157 0.175 ...
     $ HAsfc9.max      : num  0.284 0.27 0.26 0.607 0.526 ...
     $ HAsfc9.mean     : num  0.205 0.235 0.211 0.383 0.334 ...
     $ HAsfc9.median   : num  0.196 0.241 0.242 0.426 0.335 ...
     $ HAsfc9.sd       : num  0.0502 0.0338 0.0697 0.191 0.1349 ...
     $ HAsfc81.n       : num  17 4 3 8 8 6 22 2 17 4 ...
     $ HAsfc81.min     : num  0.284 0.325 0.317 0.312 0.287 ...
     $ HAsfc81.max     : num  0.426 0.441 0.41 0.652 0.507 ...
     $ HAsfc81.mean    : num  0.362 0.395 0.367 0.502 0.45 ...
     $ HAsfc81.median  : num  0.356 0.407 0.375 0.553 0.478 ...
     $ HAsfc81.sd      : num  0.0457 0.053 0.0467 0.1386 0.0754 ...
     $ NewEplsar.n     : num  17 4 3 8 8 6 22 2 0 0 ...
     $ NewEplsar.min   : num  0.0191 0.0191 0.0188 0.0174 0.0174 ...
     $ NewEplsar.max   : num  0.0207 0.02 0.0198 0.0198 0.019 ...
     $ NewEplsar.mean  : num  0.0199 0.0195 0.0193 0.0181 0.018 ...
     $ NewEplsar.median: num  0.0198 0.0195 0.0192 0.018 0.018 ...
     $ NewEplsar.sd    : num  0.000394 0.000498 0.000538 0.000798 0.0005 ...

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

    R version 4.1.1 (2021-08-10)
    Platform: x86_64-w64-mingw32/x64 (64-bit)
    Running under: Windows 10 x64 (build 19041)

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
     [5] readr_2.0.1       tidyr_1.1.3       tibble_3.1.4      ggplot2_3.3.5    
     [9] tidyverse_1.3.1   doBy_4.6.11       R.utils_2.10.1    R.oo_1.24.0      
    [13] R.methodsS3_1.8.1 openxlsx_4.2.4   

    loaded via a namespace (and not attached):
     [1] Rcpp_1.0.7           lubridate_1.7.10     curry_0.1.1         
     [4] lattice_0.20-44      assertthat_0.2.1     rprojroot_2.0.2     
     [7] digest_0.6.27        utf8_1.2.2           R6_2.5.1            
    [10] cellranger_1.1.0     backports_1.2.1      reprex_2.0.1        
    [13] evaluate_0.14        httr_1.4.2           pillar_1.6.2        
    [16] rlang_0.4.11         readxl_1.3.1         rstudioapi_0.13     
    [19] Matrix_1.3-4         rmarkdown_2.10       munsell_0.5.0       
    [22] broom_0.7.9          compiler_4.1.1       Deriv_4.1.3         
    [25] modelr_0.1.8         xfun_0.25            pkgconfig_2.0.3     
    [28] microbenchmark_1.4-7 htmltools_0.5.2      tidyselect_1.1.1    
    [31] fansi_0.5.0          crayon_1.4.1         tzdb_0.1.2          
    [34] dbplyr_2.1.1         withr_2.4.2          MASS_7.3-54         
    [37] grid_4.1.1           jsonlite_1.7.2       gtable_0.3.0        
    [40] lifecycle_1.0.0      DBI_1.1.1            magrittr_2.0.1      
    [43] scales_1.1.1         zip_2.2.0            cli_3.0.1           
    [46] stringi_1.7.4        renv_0.14.0          fs_1.5.0            
    [49] xml2_1.3.2           ellipsis_0.3.2       generics_0.1.0      
    [52] vctrs_0.3.8          tools_4.1.1          glue_1.4.2          
    [55] hms_1.1.0            fastmap_1.1.0        yaml_2.2.1          
    [58] colorspace_2.0-2     rvest_1.0.1          knitr_1.33          
    [61] haven_2.4.3         

RStudio version 1.4.1717.

------------------------------------------------------------------------

# Cite R packages used

    openxlsx 
    Philipp Schauberger and Alexander Walker (2021). openxlsx: Read, Write and Edit xlsx Files. R package version 4.2.4. https://CRAN.R-project.org/package=openxlsx 
     
    R.utils 
    Henrik Bengtsson (2020). R.utils: Various Programming Utilities. R package version 2.10.1. https://CRAN.R-project.org/package=R.utils 
     
    doBy 
    Søren Højsgaard and Ulrich Halekoh (2021). doBy: Groupwise Statistics, LSmeans, Linear Contrasts, Utilities. R package version 4.6.11. https://CRAN.R-project.org/package=doBy 
     
    tidyverse 
    Wickham et al., (2019). Welcome to the tidyverse. Journal of Open Source Software, 4(43), 1686, https://doi.org/10.21105/joss.01686 
     

------------------------------------------------------------------------

END OF SCRIPT
