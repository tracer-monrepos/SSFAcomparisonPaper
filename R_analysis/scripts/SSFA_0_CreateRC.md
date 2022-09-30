Create Research Compendium - SSFAcomparisonPaper
================
Ivan Calandra
2022-09-30 11:05:14

-   <a href="#goal-of-the-script" id="toc-goal-of-the-script">Goal of the
    script</a>
-   <a href="#prerequisites" id="toc-prerequisites">Prerequisites</a>
-   <a href="#preparations" id="toc-preparations">Preparations</a>
-   <a href="#create-the-research-compendium"
    id="toc-create-the-research-compendium">Create the research
    compendium</a>
    -   <a href="#load-packages" id="toc-load-packages">Load packages</a>
    -   <a href="#create-compendium" id="toc-create-compendium">Create
        compendium</a>
    -   <a href="#create-cc-by-cc-0-or-mit-license"
        id="toc-create-cc-by-cc-0-or-mit-license">Create CC-BY, CC-0 or MIT
        license</a>
    -   <a href="#create-readmermd-file" id="toc-create-readmermd-file">Create
        README.Rmd file</a>
    -   <a href="#create-a-folders" id="toc-create-a-folders">Create a
        folders</a>
    -   <a href="#delete-file-namespace" id="toc-delete-file-namespace">Delete
        file ‘NAMESPACE’</a>
-   <a href="#before-running-the-analyses"
    id="toc-before-running-the-analyses">Before running the analyses</a>
-   <a href="#after-running-the-analyses"
    id="toc-after-running-the-analyses">After running the analyses</a>
    -   <a href="#packages" id="toc-packages">Packages</a>
    -   <a href="#renv" id="toc-renv">renv</a>
-   <a href="#sessioninfo-and-rstudio-version"
    id="toc-sessioninfo-and-rstudio-version">sessionInfo() and RStudio
    version</a>

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

Create a folder ‘R_analysis’ and subfolders to contain raw data, derived
data, plots, statistics and scripts. Also create a folder for the Python
analysis:

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

    R version 4.1.2 (2021-11-01)
    Platform: x86_64-w64-mingw32/x64 (64-bit)
    Running under: Windows 10 x64 (build 19043)

    Matrix products: default

    locale:
    [1] LC_COLLATE=English_United States.1252 
    [2] LC_CTYPE=English_United States.1252   
    [3] LC_MONETARY=English_United States.1252
    [4] LC_NUMERIC=C                          
    [5] LC_TIME=English_United States.1252    

    attached base packages:
    [1] stats     graphics  grDevices datasets  utils     methods   base     

    other attached packages:
    [1] holepunch_0.1.28.9000 renv_0.14.0           usethis_2.1.3        
    [4] rrtools_0.1.5        

    loaded via a namespace (and not attached):
     [1] cliapp_0.1.1      compiler_4.1.2    git2r_0.29.0      prettyunits_1.1.1
     [5] tools_4.1.2       progress_1.2.2    digest_0.6.28     lubridate_1.8.0  
     [9] jsonlite_1.7.2    evaluate_0.14     lifecycle_1.0.1   pkgconfig_2.0.3  
    [13] rlang_0.4.12      cli_3.1.0         rstudioapi_0.13   curl_4.3.2       
    [17] yaml_2.2.1        xfun_0.28         fastmap_1.1.0     httr_1.4.2       
    [21] xml2_1.3.2        withr_2.4.2       stringr_1.4.0     knitr_1.36       
    [25] generics_0.1.1    desc_1.4.0        fs_1.5.0          vctrs_0.3.8      
    [29] hms_1.1.1         rprojroot_2.0.2   glue_1.5.0        here_1.0.1       
    [33] R6_2.5.1          gh_1.3.0          fansi_0.5.0       rmarkdown_2.11   
    [37] bookdown_0.24     purrr_0.3.4       selectr_0.4-2     magrittr_2.0.1   
    [41] clisymbols_1.2.0  htmltools_0.5.2   ellipsis_0.3.2    stringi_1.7.6    
    [45] crayon_1.4.2     

RStudio version 2021.9.1.372.

END OF SCRIPT
