README
================
Ivan Calandra
2022-12-13 16:21:52

-   <a href="#ssfacomparisonpaper"
    id="toc-ssfacomparisonpaper">SSFAcomparisonPaper</a>
-   <a href="#how-to-cite" id="toc-how-to-cite">How to cite</a>
-   <a href="#contents" id="toc-contents">Contents</a>
-   <a href="#how-to-download-and-run-locally"
    id="toc-how-to-download-and-run-locally">How to download and run
    locally</a>
-   <a href="#licenses" id="toc-licenses">Licenses</a>
-   <a href="#contributions" id="toc-contributions">Contributions</a>

<!-- README.md is generated from README.Rmd. Please edit that file -->

# SSFAcomparisonPaper

This repository contains the data and code for the paper:

> Calandra I, Bob K, Merceron G, Blateyron F, Hildebrandt A,
> Schulz-Kornas E, Souron A & Winkler DE (2022). *Surface texture
> analysis in Toothfrax and MountainsMap® SSFA module: Different
> software packages, different results?* Peer Community Journal 2: e77.
> <https://doi.org/10.24072/pcjournal.204>

[![DOI](https://zenodo.org/badge/DOI/10.24072/pcjournal.204.svg)](https://doi.org/10.24072/pcjournal.204)

# How to cite

Please cite this compendium as:

> Calandra I, Bob K, Merceron G, Blateyron F, Hildebrandt A,
> Schulz-Kornas E, Souron A & Winkler DE (2022). Compendium of code and
> data for *Surface texture analysis in Toothfrax and MountainsMap® SSFA
> module: Different software packages, different results?* Online at
> <https://doi.org/10.5281/zenodo.4439450>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4439450.svg)](https://doi.org/10.5281/zenodo.4439450)

# Contents

This [README.md](/README.md) file has been created by rendering the
[README.Rmd](/README.Rmd) file.

The [DESCRIPTION](/DESCRIPTION) file contains information about the
version, author, license and packages. For details on the license, see
the [LICENSE.md](/LICENSE.md) and [LICENSE](/LICENSE) files.

The [SSFAcomparisonPaper.Rproj](/SSFAcomparisonPaper.Rproj) file is the
RStudio project file.

The [R_analysis](/R_analysis) directory contains all files related to
the R analysis. It is composed of the following folders:

-   [:file_folder: derived_data](/R_analysis/derived_data): output data
    generated during the analysis (script
    [SSFA_1\_Import.Rmd](/R_analysis/scripts/SSFA_1_Import.Rmd)).  
-   [:file_folder: plots](/R_analysis/plots): plots generated during the
    analyses (script
    [SSFA_3\_Plots.Rmd](/R_analysis/scripts/SSFA_3_Plots.Rmd)).  
-   [:file_folder: raw_data](/R_analysis/raw_data): input data used in
    the analyses (script
    [SSFA_1\_Import.Rmd](/R_analysis/scripts/SSFA_1_Import.Rmd)).  
-   [:file_folder: scripts](/R_analysis/scripts): scripts used to run
    the analyses. See below for details.  
-   [:file_folder: summary_stats](/R_analysis/summary_stats): summary
    statistics generated during the analyses (script
    [SSFA_2\_Summary-stats.Rmd](/R_analysis/scripts/SSFA_2_Summary-stats.Rmd)).

The [scripts](/R_analysis/scripts) directory contains the following
files:

-   [SSFA_0\_RStudioVersion.R](/R_analysis/scripts/SSFA_0_RStudioVersion.R):
    script to get the used version of RStudio and write it to a TXT file
    ([SSFA_0\_RStudioVersion.txt](/R_analysis/scripts/SSFA_0_RStudioVersion.txt)).
    This is necessary as the function `RStudio.Version()` can only be
    run in an interactive session (see [this
    post](https://community.rstudio.com/t/rstudio-version-not-found-on-knit/8088)
    for details).
-   [SSFA_0\_CreateRC.Rmd](/R_analysis/scripts/SSFA_0_CreateRC.Rmd):
    script used to create this research compendium - it is not part of
    the analysis *per se* and is not meant to be run again. Rendered to
    [SSFA_0\_CreateRC.html](/R_analysis/scripts/SSFA_0_CreateRC.html)
    and [SSFA_0\_CreateRC.md](/R_analysis/scripts/SSFA_0_CreateRC.md).  
-   [SSFA_1\_Import.Rmd](/R_analysis/scripts/SSFA_1_Import.Rmd): script
    to import the raw, input data. Rendered to
    [SSFA_1\_Import.md](/R_analysis/scripts/SSFA_1_Import.md) and
    [SSFA_1\_Import.html](/R_analysis/scripts/SSFA_1_Import.html).  
-   [SSFA_2\_Summary-stats.Rmd](/R_analysis/scripts/SSFA_2_Summary-stats.Rmd):
    script to compute group-wise summary statistics. Rendered to
    [SSFA_2\_Summary-stats.md](/R_analysis/scripts/SSFA_2_Summary-stats.md)
    and
    [SSFA_2\_Summary-stats.html](/R_analysis/scripts/SSFA_2_Summary-stats.html).  
-   [SSFA_3\_Plots.Rmd](/R_analysis/scripts/SSFA_3_Plots.Rmd): script to
    produce plots for each SSFA variable. Rendered to
    [SSFA_3\_Plots.md](/R_analysis/scripts/SSFA_3_Plots.md) and
    [SSFA_3\_Plots.html](/R_analysis/scripts/SSFA_3_Plots.html).  
-   [SSFA_3\_Plots_files](/R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/):
    contains PNG files of the plots; used in the
    [SSFA_3\_Plots.md](/R_analysis/scripts/SSFA_3_Plots.md) file.

The [Python_analysis](/Python_analysis) directory contains all files
related to the Python analysis. It is composed of the following folders
and files:

-   [:file_folder: code](/Python_analysis/code): notebooks, custom
    Python library and associated files. See below for details.  
-   [:file_folder: derived_data](/Python_analysis/derived_data): output
    of the pre-processing and of the 3 analyses notebooks. See below for
    details.
-   [:file_folder: plots](/Python_analysis/plots): plots of the 3
    models, saved as PDF files. See below for details.  
-   [requirements.txt](/Python_analysis/requirements.txt): list of used
    Python packages and their specific versions.  
-   [RUN_DOCKER.md](/Python_analysis/RUN_DOCKER.md): description of how
    to run the Docker image of the Python analysis.

The [code](/Python_analysis/code) folder contains the following files:

-   `*.ipynb` files: notebooks, rendered to HTML (`*.html`) and MD
    (`*.md`) files.  
-   The sub-folders (`Statistical_Model_*_files/`) contain the PNG
    images of the rendered MD files.  
-   [plotting_lib.py](/Python_analysis/code/plotting_lib.py): custom
    Python library with mainly functions for plotting and other
    auxiliary functions.

The [derived_data](/Python_analysis/derived_data) folder contains the
following files, organized in sub-folders for each model:

-   [`*.dat` files](/Python_analysis/derived_data/preprocessing):
    pre-processed raw data, used as input to the models.  
-   `*.pkl` files: Serialized model object and traces of the statistical
    models.  
-   `*.npy` files (two-factor model only): parameter samples for some
    model parameters from the run of the two factor model for
    *epLsar*.  
-   `*_hdi_*.csv`: 95% high probability density intervals of the
    contrasts in table form and CSV format.  
-   `*_summary_*.csv`: summary of the results in table form and CSV
    format.

The [plots](/Python_analysis/plots) folder contains the following files,
organized in sub-folders for each model:

-   `*_contrast_*.pdf` files: contrast plots of ConfoMap vs. Toothfrax
    from the three-factor model.  
-   `*_posterior_b_*.pdf` files: distribution of posteriors.  
-   `*_posterior_forest_*.pdf` files: plots of model parameter
    distributions and their HDIs, effective sample sizes (ess) and r_hat
    statistic.  
-   `*_posterior_pair_*.pdf` files: plots of joint distributions of
    model parameters.  
-   `*_posterior_parallel_*.pdf` files: plots of sampled posterior
    points. Used for checking sampling reliability.  
-   `*_prior_posterior_*.pdf` files: prior and posterior distributions
    of the model parameters.  
-   `*_prior_predictive_*.pdf` files: prior-predictive plots of the
    surface parameters.  
-   `*_prior_posterior_predictive_*.pdf` files: prior and
    posterior-predictive plots of the surface parameters.  
-   `*_trace_*.pdf` files: trace plots.  
-   `*_treatment_diff_*.pdf` files: Contrast plots of treatment pairs
    that differ between the software packages.  
-   `*_treatment_pairs_*.pdf` files: Contrast plots of treatment pairs
    for ConfoMap and Toothfrax from the two-factor and NewEplsar models.

Note that the HTML files are not rendered nicely on GitHub; you need to
download them and open them with your browser. Use the MD files to view
on GitHub. However, MD files do not have all functionalities of HTML
files (numbered sections, floating table of content). I therefore
recommend using the HTML files.  
To download an HTML file from GitHub, first display the “raw” file and
then save it as HTML.

Alternatively, use [GitHub & BitBucket HTML
Preview](https://htmlpreview.github.io/) to render it directly.  
Here are direct links to display the files directly in your browser:

-   [SSFA_0\_CreateRC.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/R_analysis/scripts/SSFA_0_CreateRC.html)
-   [SSFA_1\_Import.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/R_analysis/scripts/SSFA_1_Import.html)  
-   [SSFA_2\_Summary-stats.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/R_analysis/scripts/SSFA_2_Summary-stats.html)  
-   [SSFA_3\_Plots.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/R_analysis/scripts/SSFA_3_Plots.html)  
-   [Preprocessing_filter_strong.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/Python_analysis/code/Preprocessing_filter_strong.html)
-   [Statistical_Model_ThreeFactor_filter_strong.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/Python_analysis/code/Statistical_Model_ThreeFactor_filter_strong.html)  
-   [Statistical_Model_TwoFactor_filter_strong.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/Python_analysis/code/Statistical_Model_TwoFactor_filter_strong.html)  
-   [Statistical_Model_NewEplsar_filter_strong.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/Python_analysis/code/Statistical_Model_NewEplsar_filter_strong.html)

The [renv.lock](/renv.lock) file is the lockfile describing the state of
the R project’s library. It is associated to the [activation
script](/renv/activate.R) and the R project’s library. All these files
have been created using the package
[renv](https://rstudio.github.io/renv/index.html).

The [Dockerfile](/Dockerfile) contains the instruction to build the
Docker image for the Python analysis. See section [How to run in your
browser or download and run
locally](#how-to-run-in-your-browser-or-download-and-run-locally) for
details.

See the section [Contributions](#contributions) for details on
[CONDUCT.md](/CONDUCT.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

# How to download and run locally

This research compendium has been developed using the statistical
programming languages R and Python. To work with the compendium, you
will need to install on your computer the [R
software](https://cloud.r-project.org/) and [RStudio
Desktop](https://rstudio.com/products/rstudio/download/) for the R
analysis, and [Python
3.8.5](https://www.python.org/downloads/release/python-385/) and the
packages listed in [requirements.txt](/Python_analysis/requirements.txt)
for the Python analysis.

To work locally with the R analysis, either from the ZIP archive or from
cloning the GitHub repository to your computer:

-   open the [SSFAcomparisonPaper.Rproj](/SSFAcomparisonPaper.Rproj)
    file in RStudio; this takes some time the first time.  
-   run `renv::status()` and then `renv::restore()` to restore the state
    of your project from [renv.lock](/renv.lock). Make sure that the
    package `devtools` is installed to be able to install packages from
    source.

Using the package `renv` implies that installing, removing and updating
packages is done within the project. In other words, all the packages
that you install/update while in a project using `renv` will not be
available in any other project. If you want to globally
install/remove/update packages, make sure you close the project first.

For the Python analysis, the easiest way is to use the Docker image
hosted on [Zenodo](https://doi.org/10.5281/zenodo.4302091). The detailed
instructions are given in
[RUN_DOCKER.md](/Python_analysis/RUN_DOCKER.md).

You can also download the compendium as [a ZIP
archive](https://github.com/tracer-monrepos/SSFAcomparisonPaper/archive/master.zip).  
Alternatively, if you use GitHub, you can [fork and
clone](https://happygitwithr.com/fork-and-clone.html) the repository to
your account. See also the [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Licenses

**Data, text and figures :** [CC BY-SA
4.0](https://creativecommons.org/licenses/by-sa/4.0/)  
**Code :** See the [DESCRIPTION](/DESCRIPTION),
[LICENSE.md](/LICENSE.md) and [LICENSE](/LICENSE) files.

# Contributions

We welcome contributions from everyone. Before you get started, please
see our [contributor guidelines](/CONTRIBUTING.md). Please note that
this project is released with a [Contributor Code of
Conduct](/CONDUCT.md). By participating in this project you agree to
abide by its terms.
