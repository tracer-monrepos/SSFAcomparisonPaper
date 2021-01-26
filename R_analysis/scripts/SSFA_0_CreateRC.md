Create Research Compendium - SSFAcomparisonPaper
================
Ivan Calandra
2021-01-26 15:14:44

-   [Goal of the script](#goal-of-the-script)
-   [Prerequisites](#prerequisites)
-   [Preparations](#preparations)
-   [Create the research compendium](#create-the-research-compendium)
    -   [Load packages](#load-packages)
    -   [Create compendium](#create-compendium)
    -   [Create CC-BY, CC-0 or MIT
        license](#create-cc-by-cc-0-or-mit-license)
    -   [Create README.Rmd file](#create-readme.rmd-file)
    -   [Create a folders](#create-a-folders)
    -   [Delete file ‘NAMESPACE’](#delete-file-namespace)
-   [Before running the analyses](#before-running-the-analyses)
-   [After running the analyses](#after-running-the-analyses)
    -   [Packages](#packages)
    -   [Create Binder badge](#create-binder-badge)
    -   [renv](#renv)
-   [sessionInfo() and RStudio
    version](#sessioninfo-and-rstudio-version)

------------------------------------------------------------------------

# Goal of the script

Create and set up a research compendium for the SSFA comparison paper
using the R package `rrtools`.  
For details on rrtools, see Ben Marwick’s [GitHub
repository](https://github.com/benmarwick/rrtools).

Note that this script is there only to show the steps taken to create
the research compendium and is not part of the analysis per se. For this
reason, the code is not evaluated (`knitr::opts_chunk$set(eval=FALSE)`).

The knit directory for this script is the project directory.

------------------------------------------------------------------------

# Prerequisites

This script requires that you have a GitHub account and that you have
connected RStudio, Git and GitHub. For details on how to do it, check
[Happy Git](https://happygitwithr.com/).

------------------------------------------------------------------------

# Preparations

Before running this script, the first step is to [create a repository on
GitHub and to download it to
RStudio](https://happygitwithr.com/new-github-first.html). In this case,
the repository is called “SSFAcomparisonPaper”.  
Finally, open the RStudio project created.

------------------------------------------------------------------------

# Create the research compendium

## Load packages

``` r
library(rrtools)
library(usethis)
library(renv)
library(holepunch)
```

## Create compendium

``` r
rrtools::use_compendium(getwd())
```

A new project has opened in a new session.  
Edit the fields “Title”, “Author” and “Description” in the `DESCRIPTION`
file.

## Create CC-BY, CC-0 or MIT license

A project can have only one license. Run one line only:

``` r
#usethis::use_ccby_license()
#usethis::use_cc0_license()
usethis::use_mit_license(copyright_holder = "Ivan Calandra")
```

## Create README.Rmd file

``` r
rrtools::use_readme_rmd()
```

Edit the `README.Rmd` file as needed.  
Make sure you render (knit) it to create the `README.md` file.

## Create a folders

Create a folder ‘R\_analysis’ and subfolders to contain raw data,
derived data, plots, statistics and scripts. Also create a folder for
the Python analysis:

``` r
dir.create("R_analysis", showWarnings = FALSE)
dir.create("R_analysis/raw_data", showWarnings = FALSE)
dir.create("R_analysis/derived_data", showWarnings = FALSE)
dir.create("R_analysis/plots", showWarnings = FALSE)
dir.create("R_analysis/summary_stats", showWarnings = FALSE)
dir.create("R_analysis/scripts", showWarnings = FALSE)
dir.create("Python_analysis", showWarnings = FALSE)
```

Note that the folders cannot be pushed to GitHub as long as they are
empty.

## Delete file ‘NAMESPACE’

``` r
file.remove("NAMESPACE")
```

------------------------------------------------------------------------

# Before running the analyses

After the creation of this research compendium, I have moved the raw,
input data files to `"~/R_analysis/raw_data"` (as read-only files) and
the R scripts to `"~/R_analysis/scripts"`.

------------------------------------------------------------------------

# After running the analyses

## Packages

I have run this command to add the dependencies to the DESCRIPTION file.

``` r
rrtools::add_dependencies_to_description()
```

## Create Binder badge

``` r
holepunch::write_dockerfile(maintainer = "Ivan Calandra")
holepunch::generate_badge()
```

Now paste to the README.Rmd, where you want the badge to appear.

## renv

Save the state of the project library using the `renv` package.

``` r
renv::init()
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
    [1] stats     graphics  grDevices datasets  utils     methods   base     

    other attached packages:
    [1] holepunch_0.1.28.9000 renv_0.12.5           usethis_2.0.0        
    [4] rrtools_0.1.0        

    loaded via a namespace (and not attached):
     [1] Rcpp_1.0.6        cliapp_0.1.1      compiler_4.0.3    git2r_0.28.0     
     [5] prettyunits_1.1.1 tools_4.0.3       progress_1.2.2    digest_0.6.27    
     [9] lubridate_1.7.9.2 jsonlite_1.7.2    evaluate_0.14     lifecycle_0.2.0  
    [13] pkgconfig_2.0.3   rlang_0.4.10      cli_2.2.0         rstudioapi_0.13  
    [17] curl_4.3          yaml_2.2.1        xfun_0.20         httr_1.4.2       
    [21] xml2_1.3.2        withr_2.4.0       stringr_1.4.0     knitr_1.30       
    [25] generics_0.1.0    desc_1.2.0        fs_1.5.0          vctrs_0.3.6      
    [29] hms_1.0.0         rprojroot_2.0.2   glue_1.4.2        here_1.0.1       
    [33] R6_2.5.0          gh_1.2.0          fansi_0.4.2       rmarkdown_2.6    
    [37] bookdown_0.21     purrr_0.3.4       selectr_0.4-2     magrittr_2.0.1   
    [41] clisymbols_1.2.0  htmltools_0.5.1.1 ellipsis_0.3.1    assertthat_0.2.1 
    [45] stringi_1.5.3     crayon_1.3.4     

RStudio version 1.4.1103.

END OF SCRIPT
