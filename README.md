README
================
Ivan Calandra
2020-11-30 15:09:05

-   [SSFAcomparisonPaper](#ssfacomparisonpaper)
-   [How to cite](#how-to-cite)
-   [Contents](#contents)
-   [How to run in your browser or download and run
    locally](#how-to-run-in-your-browser-or-download-and-run-locally)
-   [Licenses](#licenses)
-   [Contributions](#contributions)

<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- badges: start -->

[![Launch Rstudio
Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tracer-monrepos/SSFAcomparisonPaper/master?urlpath=rstudio)

<!-- badges: end -->

# SSFAcomparisonPaper

This repository contains the data and code for our open-access paper:

> Calandra I, Bob K, Merceron G, Hildebrandt A, Schulz-Kornas E, Souron
> A & Winkler DE (submitted). *A note of caution: MountainsMap® SSFA
> module produces different results than Toothfrax*. Wear
> <https://doi.org/xxx/xxx>

# How to cite

Please cite this compendium as:

> Calandra I, Bob K, Merceron G, Hildebrandt A, Schulz-Kornas E, Souron
> A & Winkler DE (2020). Compendium of code and data for *A note of
> caution: MountainsMap® SSFA module produces different results than
> Toothfrax*. Accessed 30 Nov 2020. Online at <https://doi.org/xxx/xxx>

# Contents

This [README.md](/README.md) file has been created by rendering the
[README.Rmd](/README.Rmd) file.

The [DESCRIPTION](/DESCRIPTION) file contains information about the
version, author, license and packages. For details on the license, see
the [LICENSE.md](/LICENSE.md) and [LICENSE](/LICENSE) files.

The [SSFAcomparisonPaper.Rproj](/SSFAcomparisonPaper.Rproj) file is the
RStudio project file.

The [R\_analysis](/R_analysis) directory contains all files related to
the R analysis. It is composed of the following folders:

-   [:file\_folder: derived\_data](/R_analysis/derived_data): output
    data generated during the analysis (script
    [SSFA\_1\_Import.Rmd](/R_analysis/scripts/SSFA_1_Import.Rmd)).  
-   [:file\_folder: plots](/R_analysis/plots): plots generated during
    the analyses (script
    [SSFA\_3\_Plots.Rmd](/R_analysis/scripts/SSFA_3_Plots.Rmd)).  
-   [:file\_folder: raw\_data](/R_analysis/raw_data): input data used in
    the analyses (script
    [SSFA\_1\_Import.Rmd](/R_analysis/scripts/SSFA_1_Import.Rmd)).  
-   [:file\_folder: scripts](/R_analysis/scripts): scripts used to run
    the analyses. See below for details.  
-   [:file\_folder: summary\_stats](/R_analysis/summary_stats): summary
    statistics generated during the analyses (script
    [SSFA\_2\_Summary-stats.Rmd](/R_analysis/scripts/SSFA_2_Summary-stats.Rmd)).

The [scripts](/R_analysis/scripts) directory contains the following
files:

-   [SSFA\_0\_RStudioVersion.R](/R_analysis/scripts/SSFA_0_RStudioVersion.R):
    script to get the used version of RStudio and write it to a TXT file
    ([RStudioVersion.txt](/R_analysis/scripts/RStudioVersion.txt)). This
    is necessary as the function `RStudio.Version()` can only be run in
    an interactive session (see [this
    post](https://community.rstudio.com/t/rstudio-version-not-found-on-knit/8088)
    for details).
-   [SSFA\_0\_CreateRC.Rmd](/R_analysis/scripts/SSFA_0_CreateRC.Rmd):
    script used to create this research compendium - it is not part of
    the analysis *per se* and is not meant to be run again. Rendered to
    [SSFA\_0\_CreateRC.md](/R_analysis/scripts/SSFA_0_CreateRC.md).  
-   [SSFA\_1\_Import.Rmd](/R_analysis/scripts/SSFA_1_Import.Rmd): script
    to import the raw, input data. Rendered to
    [SSFA\_1\_Import.md](/R_analysis/scripts/SSFA_1_Import.md) and
    [SSFA\_1\_Import.html](/R_analysis/scripts/SSFA_1_Import.html).  
-   [SSFA\_2\_Summary-stats.Rmd](/R_analysis/scripts/SSFA_2_Summary-stats.Rmd):
    script to compute group-wise summary statistics. Rendered to
    [SSFA\_2\_Summary-stats.md](/R_analysis/scripts/SSFA_2_Summary-stats.md)
    and
    [SSFA\_2\_Summary-stats.html](/R_analysis/scripts/SSFA_2_Summary-stats.html).  
-   [SSFA\_3\_Plots.Rmd](/R_analysis/scripts/SSFA_3_Plots.Rmd): script
    to produce plots for each SSFA variable. Rendered to
    [SSFA\_3\_Plots.md](/R_analysis/scripts/SSFA_3_Plots.md) and
    [SSFA\_3\_Plots.html](/R_analysis/scripts/SSFA_3_Plots.html).  
-   [SSFA\_3\_Plots\_files](/R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/):
    contains PNG files of the plots; used in the
    [SSFA\_3\_Plots.md](/R_analysis/scripts/SSFA_3_Plots.md) file.

The [Python\_analysis](/Python_analysis) directory contains all files
related to the Python analysis. It is composed of the following folders
and files:

-   [:file\_folder: code](/Python_analysis/code): notebooks, custom
    Python library and associated files. See below for details.  
-   [:file\_folder: derived\_data](/Python_analysis/derived_data):
    output of the pre-processing and of the 3 analyses notebooks. See
    below for details.
-   [:file\_folder: plots](/Python_analysis/plots): plots of the 3
    models, saved as PDF files. See below for details.  
-   [requirements.txt](/Python_analysis/requirements.txt): list of used
    Python packages and their specific versions.  
-   [RUN\_DOCKER.md](/Python_analysis/RUN_DOCKER.md): description of how
    to run the Docker image of the Python analysis.

The [code](/Python_analysis/code) folder contains the following files:

-   [plotting\_lib.py](/Python_analysis/code/plotting_lib.py): custom
    Python library with mainly functions for plotting and other
    auxiliary functions.  
-   [Preprocessing.ipynb](/Python_analysis/code/Preprocessing.ipynb):
    notebook of the pre-processing of the raw data. Rendered to
    [Preprocessing.html](/Python_analysis/code/Preprocessing.html) and
    [Preprocessing.md](/Python_analysis/code/Preprocessing.md).  
-   [Statistical\_Model\_ThreeFactor.ipynb](/Python_analysis/code/Statistical_Model_ThreeFactor.ipynb):
    notebook of the three-factor model. Rendered to
    [Statistical\_Model\_ThreeFactor.html](/Python_analysis/code/Statistical_Model_ThreeFactor.html)
    and
    [Statistical\_Model\_ThreeFactor.md](/Python_analysis/code/Statistical_Model_ThreeFactor.md).  
-   [Statistical\_Model\_TwoFactor.ipynb](/Python_analysis/code/Statistical_Model_TwoFactor.ipynb):
    notebook of the two-factor model. Rendered to
    [Statistical\_Model\_TwoFactor.html](/Python_analysis/code/Statistical_Model_TwoFactor.html)
    and
    [Statistical\_Model\_TwoFactor.md](/Python_analysis/code/Statistical_Model_TwoFactor.md).  
-   [Statistical\_Model\_NewEplsar.ipynb](/Python_analysis/code/Statistical_Model_NewEplsar.ipynb):
    notebook of the NewEplsar model Rendered to
    [Statistical\_Model\_NewEplsar.html](/Python_analysis/code/Statistical_Model_NewEplsar.html)
    and
    [Statistical\_Model\_NewEplsar.md](/Python_analysis/code/Statistical_Model_NewEplsar.md).  
-   The sub-folders (Statistical\_Model\_\*\_files) contains the PNG
    images of the rendered MD files.

The [derived\_data](/Python_analysis/derived_data) folder contains the
following files, organized in sub-folders for each model:

-   [preprocessed.dat](/Python_analysis/derived_data/preprocessed.dat):
    pre-processed raw data, used as input to the models.  
-   `*.pkl` files: Serialized model object and traces of the statistical
    models.  
-   `*.npy` files (two-factor model only): parameter samples for some
    model parameters from the run of the two factor model for
    *epLsar*.  
-   `hdi_*.csv`: 95% high probability density intervals of the contrasts
    in table form and CSV format.  
-   `summary.csv`: summary of the results in table form and CSV format.

The [plots](/Python_analysis/plots) folder contains the following files,
organized in sub-folders for each model:

-   `contrast_*.pdf` files: contrast plots of ConfoMap vs. Toothfrax
    from the three-factor model.  
-   `posterior_b_*.pdf` files: distribution of posteriors.  
-   `posterior_forest_*.pdf` files: plots of model parameter
    distributions and their HDIs, effective sample sizes (ess) and
    r\_hat statistic.  
-   `posterior_pair_*.pdf` files: plots of joint distributions of model
    parameters.  
-   `posterior_parallel_*.pdf` files: plots of sampled posterior points.
    Used for checking sampling reliability.  
-   `prior_posterior_*.pdf` files: prior and posterior distributions of
    the model parameters.  
-   `prior_predictive_*.pdf` files: prior-predictive plots of the
    surface parameters.  
-   `prior_posterior_predictive_*.pdf` files: prior and
    posterior-predictive plots of the surface parameters.  
-   `trace_*.pdf` files: trace plots.  
-   `treatment_pairs_*.pdf` files: Contrast plots of treatment pairs for
    ConfoMap and Toothfrax from the two-factor and NewEplsar models.

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

-   [SSFA\_1\_Import.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/R_analysis/scripts/SSFA_1_Import.html)  
-   [SSFA\_2\_Summary-stats.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/R_analysis/scripts/SSFA_2_Summary-stats.html)  
-   [SSFA\_3\_Plots.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/R_analysis/scripts/SSFA_3_Plots.html)  
-   [Preprocessing.html](http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/Python_analysis/code/Preprocessing.html)
-   [Statistical\_Model\_ThreeFactor.html](/http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/Python_analysis/code/Statistical_Model_ThreeFactor.html)  
-   [Statistical\_Model\_TwoFactor.html](/http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/Python_analysis/code/Statistical_Model_TwoFactor.html)  
-   [Statistical\_Model\_NewEplsar.html](/http://htmlpreview.github.io/?https://github.com/tracer-monrepos/SSFAcomparisonPaper/blob/master/Python_analysis/code/Statistical_Model_NewEplsar.html)

The [renv.lock](/renv.lock) file is the lockfile describing the state of
the R project’s library. It is associated to the [activation
script](/renv/activate.R) and the R project’s library. All these files
have been created using the package
[renv](https://rstudio.github.io/renv/index.html).

The [.binder](/.binder) directory contains the Docker image for the R
analysis, while [Dockerfile](/Dockerfile) contains the instruction to
build the Docker image for the Python analysis. See section [How to run
in your browser or download and run
locally](#how-to-run-in-your-browser-or-download-and-run-locally)) for
details.

See the section [Contributions](#contributions) for details on
[CONDUCT.md](/CONDUCT.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

# How to run in your browser or download and run locally

This research compendium has been developed using the statistical
programming languages R and Pÿthon. To work with the compendium, you
will need to install on your computer the [R
software](https://cloud.r-project.org/) and [RStudio
Desktop](https://rstudio.com/products/rstudio/download/), and
[Python]()…

The simplest way to explore the R analysis is to open an instance of
RStudio in your browser using [binder](https://mybinder.org/), either by
clicking on the “launch binder” badge at the top of this README, or by
following [this
link](https://mybinder.org/v2/gh/tracer-monrepos/SSFAcomparisonPaper/master?urlpath=rstudio).
Binder will have the compendium files ready to work with. Binder uses
Docker images to ensure a consistent and reproducible computational
environment. These Docker images can also be used locally.

If you want to work locally with the R analysis, either from the ZIP
archive or from cloning the GitHub repository to your computer:

-   open the `SSFAcomparisonPaper.Rproj` file in RStudio; this takes
    some time the first time.  
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
hosted on [Zenodo]()…

You can also download the compendium as [a ZIP
archive](https://github.com/tracer-monrepos/SSFAcomparisonPaper/archive/master.zip).  
Alternatively, if you use GitHub, you can [fork and
clone](https://happygitwithr.com/fork-and-clone.html) the repository to
your account. See also the [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Licenses

**Text and figures :**
[CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/)  
**Code :** See the [DESCRIPTION](/DESCRIPTION),
[LICENSE.md](/LICENSE.md) and [LICENSE](/LICENSE) files.  
**Data :** [CC-0](http://creativecommons.org/publicdomain/zero/1.0/)
attribution requested in reuse

# Contributions

We welcome contributions from everyone. Before you get started, please
see our [contributor guidelines](/CONTRIBUTING.md). Please note that
this project is released with a [Contributor Code of
Conduct](/CONDUCT.md). By participating in this project you agree to
abide by its terms.
