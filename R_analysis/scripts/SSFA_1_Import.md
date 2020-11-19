Import SSFA datasets
================
Ivan Calandra
2020-11-19 13:37:11

-   [Goal of the script](#goal-of-the-script)
-   [Load packages](#load-packages)
-   [Get names, path and information of all
    files](#get-names-path-and-information-of-all-files)
-   [Guinea pigs](#guinea-pigs)
    -   [ConfoMap](#confomap)
        -   [Read in data](#read-in-data)
        -   [Exclude repeated rows and select
            columns](#exclude-repeated-rows-and-select-columns)
        -   [Add headers](#add-headers)
        -   [Extract units](#extract-units)
        -   [Convert to numeric and add column about
            software](#convert-to-numeric-and-add-column-about-software)
    -   [Toothfrax](#toothfrax)
        -   [Read in data](#read-in-data-1)
        -   [Select columns, edit headers and add column about
            software](#select-columns-edit-headers-and-add-column-about-software)
        -   [Edit column “Name”](#edit-column-name)
        -   [Get units](#get-units)
    -   [Merge datasets](#merge-datasets)
        -   [Merge](#merge)
        -   [Add column about diet](#add-column-about-diet)
        -   [Add column about dataset and re-order
            columns](#add-column-about-dataset-and-re-order-columns)
        -   [Combine both sets of units](#combine-both-sets-of-units)
        -   [Check the result](#check-the-result)
-   [Sheeps](#sheeps)
    -   [ConfoMap](#confomap-1)
        -   [Read in data](#read-in-data-2)
        -   [Exclude repeated rows and select
            columns](#exclude-repeated-rows-and-select-columns-1)
        -   [Add headers](#add-headers-1)
        -   [Extract units](#extract-units-1)
        -   [Convert to numeric and add column about
            software](#convert-to-numeric-and-add-column-about-software-1)
    -   [Toothfrax](#toothfrax-1)
        -   [Read in data](#read-in-data-3)
        -   [Select columns, edit headers and add column about
            software](#select-columns-edit-headers-and-add-column-about-software-1)
        -   [Edit column “Name”](#edit-column-name-1)
        -   [Get units](#get-units-1)
    -   [Merge datasets](#merge-datasets-1)
        -   [Merge](#merge-1)
        -   [Add column about diet](#add-column-about-diet-1)
        -   [Add column about dataset and re-order
            columns](#add-column-about-dataset-and-re-order-columns-1)
        -   [Add units](#add-units)
        -   [Check the result](#check-the-result-1)
-   [Lithics](#lithics)
    -   [ConfoMap](#confomap-2)
        -   [Read in data](#read-in-data-4)
        -   [Exclude repeated rows and select
            columns](#exclude-repeated-rows-and-select-columns-2)
        -   [Add headers](#add-headers-2)
        -   [Extract units](#extract-units-2)
        -   [Convert to numeric and add column about
            software](#convert-to-numeric-and-add-column-about-software-2)
    -   [Toothfrax](#toothfrax-2)
        -   [Read in data](#read-in-data-5)
        -   [Select columns, edit headers and add column about
            software](#select-columns-edit-headers-and-add-column-about-software-2)
        -   [Edit column “Name”](#edit-column-name-2)
        -   [Get units](#get-units-2)
    -   [Merge datasets](#merge-datasets-2)
        -   [Merge](#merge-2)
        -   [Exclude FLT3-8\_Area1 (“before” wrongly acquired
            )](#exclude-flt3-8_area1-before-wrongly-acquired)
        -   [Add columns about before/after and
            treatment](#add-columns-about-beforeafter-and-treatment)
        -   [Add column about dataset and re-order
            columns](#add-column-about-dataset-and-re-order-columns-2)
        -   [Add units](#add-units-1)
        -   [Check the result](#check-the-result-2)
-   [Combine datasets](#combine-datasets)
    -   [Merge datasets](#merge-datasets-3)
    -   [Combine units](#combine-units)
    -   [Check the result](#check-the-result-3)
-   [Save data](#save-data)
    -   [Create file names](#create-file-names)
    -   [Write to XLSX and Rbin](#write-to-xlsx-and-rbin)
    -   [Show files information](#show-files-information)
-   [sessionInfo() and RStudio
    version](#sessioninfo-and-rstudio-version)

------------------------------------------------------------------------

# Goal of the script

This script formats the output of the resulting files from applying
SSFA. The script will:

1.  Read in the original files  
2.  Format the data  
3.  Write XLSX-files and save R objects ready for further analysis in R

``` r
dir_in  <- "R_analysis/raw_data"
dir_out <- "R_analysis/derived_data"
```

Raw data must be located in “\~/R\_analysis/raw\_data”.  
Formatted data will be saved in “\~/R\_analysis/derived\_data”.

The knit directory for this script is the project directory.

------------------------------------------------------------------------

# Load packages

``` r
library(openxlsx)
library(R.utils)
library(tools)
library(tidyverse)
```

------------------------------------------------------------------------

# Get names, path and information of all files

``` r
info_in <- list.files(dir_in, pattern = "\\.csv|xlsx$", full.names = TRUE) %>% 
           md5sum()
```

The checksum (MD5 hashes) of the imported files are:

                                       files                         checksum
    1 SSFA-ConfoMap_GuineaPigs_NMPfilled.csv 110b0de85f4b3999c927193bb4cf2209
    2    SSFA-ConfoMap_Lithics_NMPfilled.csv 51faaaec543ed4c0642b3f81b8fbfe7b
    3     SSFA-ConfoMap_Sheeps_NMPfilled.csv 23542e3e9aad38f9217f53b2306e788f
    4         SSFA-Toothfrax_GuineaPigs.xlsx 59684695c021e541699ea075a8c597a3
    5            SSFA-Toothfrax_Lithics.xlsx 5f9c641b05212b7565e79578d3a4eeac
    6             SSFA-Toothfrax_Sheeps.xlsx 90be4f96fe6178125f579b5b4993bdd5

------------------------------------------------------------------------

# Guinea pigs

## ConfoMap

### Read in data

``` r
# Extract file name for ConfoMap analysis on Guinea Pigs, and read in file
confoGP <- names(info_in) %>% 
           .[grepl("ConfoMap", .) & grepl("GuineaPigs", .)] %>% 
           read.csv(header = FALSE, na.strings = "*****")
```

### Exclude repeated rows and select columns

``` r
confoGP_keep_col  <- c(12, 46:52) # Define columns to keep
confoGP_keep_rows <- which(confoGP[[1]] != "#") # Define rows to keep
confoGP_keep      <- confoGP[confoGP_keep_rows, confoGP_keep_col] # Subset rows and columns
```

### Add headers

``` r
# Get headers from 2nd row
colnames(confoGP_keep) <- confoGP[2, confoGP_keep_col] %>% 
  
                          # Convert to valid names
                          make.names() %>% 
  
                          # Delete repeated periods
                          gsub("\\.+", "\\.", x = .) %>% 
  
                          # Delete periods at the end of the names
                          gsub("\\.$", "", x = .) %>%
  
                          # Keep only part after the last period
                          gsub("^([A-Za-z0-9]+\\.)+", "", x = .) 
```

### Extract units

``` r
confoGP_units <- unlist(confoGP[3, confoGP_keep_col[-1]]) # Extract unit line for considered columns
names(confoGP_units) <- colnames(confoGP_keep)[-1]        # Get names associated to the units
```

### Convert to numeric and add column about software

``` r
confoGP_keep <- type_convert(confoGP_keep) %>% 
                mutate(Software = "ConfoMap")
```

## Toothfrax

### Read in data

``` r
# Extract file name for Toothfrax analysis on Guinea Pigs
toothfraxGP <- names(info_in) %>% 
               .[grepl("Toothfrax", .) & grepl("GuineaPigs", .)] %>% 

               # Read in file
               read.xlsx(check.names = TRUE)
```

### Select columns, edit headers and add column about software

``` r
toothfraxGP_keep <- select(toothfraxGP, Name = Filename, epLsar = epLsar1.80µm, 
                           R² = Rsquared, Asfc = Asfc1om, Smfc = LineStart, 
                           HAsfc9 = X3x3HAsfc, HAsfc81 = X9x9HAsfc) %>% 
                    mutate(Software = "Toothfrax")
```

### Edit column “Name”

``` r
# Remove ".sur" at the end of the name and delete everything from the beginning to " --- "
toothfraxGP_keep[["Name"]] <- gsub("(\\.sur$)|(^([_A-Za-z0-9-]+) --- )", "", 
                                   toothfraxGP_keep[["Name"]])
```

### Get units

``` r
# Check if all units are identical
if (length(unique(toothfraxGP[["Units"]])) != 1) {
  warning(paste("The studiables have different units for the SSFA parameters.",
                "Only the unit for the first studiable will be kept.",
                "Please check in the original file.", sep = "\n"))
}

# Get units from the column "Units"
toothfraxGP_units <- toothfraxGP[["Units"]][1]

# Add names to it
names(toothfraxGP_units) <- "Toothfrax_units" 
```

## Merge datasets

### Merge

``` r
GP_keep <- merge(confoGP_keep, toothfraxGP_keep, all = TRUE)
```

### Add column about diet

``` r
GP_keep[grep("2CC4", GP_keep[["Name"]]), "Diet"] <- "Dry Lucerne"
GP_keep[grep("2CC5", GP_keep[["Name"]]), "Diet"] <- "Dry Grass"
GP_keep[grep("2CC6", GP_keep[["Name"]]), "Diet"] <- "Dry Bamboo"
```

### Add column about dataset and re-order columns

``` r
GP_final <- GP_keep %>% 
            mutate(Dataset = "GuineaPigs") %>% 
            select(Dataset, Name, Diet, Software, everything())
```

### Combine both sets of units

``` r
GP_units <- c(confoGP_units, toothfraxGP_units) 
```

### Check the result

``` r
str(GP_final)
```

    'data.frame':   140 obs. of  11 variables:
     $ Dataset  : chr  "GuineaPigs" "GuineaPigs" "GuineaPigs" "GuineaPigs" ...
     $ Name     : chr  "capor_2CC4B1_txP4_#1_1_100xL_1" "capor_2CC4B1_txP4_#1_1_100xL_1" "capor_2CC4B1_txP4_#1_1_100xL_2" "capor_2CC4B1_txP4_#1_1_100xL_2" ...
     $ Diet     : chr  "Dry Lucerne" "Dry Lucerne" "Dry Lucerne" "Dry Lucerne" ...
     $ Software : chr  "Toothfrax" "ConfoMap" "Toothfrax" "ConfoMap" ...
     $ epLsar   : num  0.00147 0.00196 0.00269 0.00366 0.0028 ...
     $ R²       : num  0.999 0.998 1 0.999 0.999 ...
     $ Asfc     : num  12.9 14.3 12 12.9 12.7 ...
     $ Smfc     : num  0.119 0.415 0.119 0.441 0.119 ...
     $ HAsfc9   : num  0.182 0.164 0.159 0.171 0.117 ...
     $ HAsfc81  : num  0.337 0.368 0.382 0.417 0.313 ...
     $ NewEplsar: num  NA 0.0184 NA 0.0189 NA ...

``` r
head(GP_final)
```

         Dataset                           Name        Diet  Software      epLsar
    1 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_1 Dry Lucerne Toothfrax 0.001471000
    2 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_1 Dry Lucerne  ConfoMap 0.001960011
    3 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_2 Dry Lucerne Toothfrax 0.002693000
    4 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_2 Dry Lucerne  ConfoMap 0.003662312
    5 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_3 Dry Lucerne Toothfrax 0.002797000
    6 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_3 Dry Lucerne  ConfoMap 0.003140386
             R²     Asfc      Smfc    HAsfc9   HAsfc81  NewEplsar
    1 0.9993430 12.92579 0.1192190 0.1819939 0.3368939         NA
    2 0.9983703 14.29323 0.4153938 0.1644618 0.3681239 0.01842431
    3 0.9995140 11.99982 0.1192190 0.1586045 0.3818615         NA
    4 0.9991123 12.90555 0.4408179 0.1713614 0.4172341 0.01888866
    5 0.9992400 12.71147 0.1192190 0.1170325 0.3130522         NA
    6 0.9989348 13.66607 0.4408179 0.1305287 0.3521687 0.01870314

------------------------------------------------------------------------

# Sheeps

Note that for compactness, comments to explain the code are given only
in the section about [Guinea Pigs](#guinea-pigs).

## ConfoMap

### Read in data

``` r
confoSheep <- names(info_in) %>% 
              .[grepl("ConfoMap", .) & grepl("Sheeps", .)] %>% 
              read.csv(header = FALSE, na.strings = "*****")
```

### Exclude repeated rows and select columns

``` r
confoSheep_keep_col  <- c(12, 43:49)
confoSheep_keep_rows <- which(confoSheep[[1]] != "#")  
confoSheep_keep      <- confoSheep[confoSheep_keep_rows, confoSheep_keep_col]
```

### Add headers

``` r
colnames(confoSheep_keep) <- confoSheep[2, confoSheep_keep_col] %>% 
                             make.names() %>% 
                             gsub("\\.+", "\\.", x = .) %>% 
                             gsub("\\.$", "", x = .) %>%
                             gsub("^([A-Za-z0-9]+\\.)+", "", x = .)
```

### Extract units

``` r
confoSheep_units <- unlist(confoSheep[3, confoSheep_keep_col[-1]]) %>% 
                    gsub("Â", "", .) 
names(confoSheep_units) <- colnames(confoSheep_keep)[-1] 
```

### Convert to numeric and add column about software

``` r
confoSheep_keep <- type_convert(confoSheep_keep) %>% 
                   mutate(Software = "ConfoMap")
```

## Toothfrax

### Read in data

``` r
toothfraxSheep <- names(info_in) %>% 
                  .[grepl("Toothfrax", .) & grepl("Sheeps", .)] %>% 
                  read.xlsx(check.names = TRUE)
```

### Select columns, edit headers and add column about software

``` r
toothfraxSheep_keep <- select(toothfraxSheep, Name = Filename, epLsar = epLsar1.80µm, 
                              R² = Rsquared, Asfc = Asfc1om, Smfc = LineStart, 
                              HAsfc9 = X3x3HAsfc, HAsfc81 = X9x9HAsfc) %>% 
                       mutate(Software = "Toothfrax")
```

### Edit column “Name”

``` r
toothfraxSheep_keep[["Name"]] <- gsub("(\\.sur$)|(^([_A-Za-z0-9-]+) --- )", "", 
                                      toothfraxSheep_keep[["Name"]]) 
```

### Get units

``` r
if (length(unique(toothfraxSheep[["Units"]])) != 1) {
  warning(paste("The studiables have different units for the SSFA parameters.",
                "Only the unit for the first studiable will be kept.",
                "Please check in the original file.", sep = "\n"))
} 
toothfraxSheep_units <- toothfraxSheep[["Units"]][1] 
names(toothfraxSheep_units) <- "Toothfrax_units" 
```

## Merge datasets

### Merge

``` r
sheep_keep <- merge(confoSheep_keep, toothfraxSheep_keep, all = TRUE)
```

### Add column about diet

``` r
sheep_keep[grep("L5", sheep_keep[["Name"]]), "Diet"] <- "Clover"
sheep_keep[grep("L6", sheep_keep[["Name"]]), "Diet"] <- "Clover+Dust"
sheep_keep[grep("L7", sheep_keep[["Name"]]), "Diet"] <- "Grass"
sheep_keep[grep("L8", sheep_keep[["Name"]]), "Diet"] <- "Grass+Dust"
```

### Add column about dataset and re-order columns

``` r
sheep_final <- sheep_keep %>% 
               mutate(Dataset = "Sheeps") %>% 
               select(Dataset, Name, Diet, Software, everything())
```

### Add units

``` r
sheep_units <- c(confoSheep_units, toothfraxSheep_units) 
```

### Check the result

``` r
str(sheep_final)
```

    'data.frame':   80 obs. of  11 variables:
     $ Dataset  : chr  "Sheeps" "Sheeps" "Sheeps" "Sheeps" ...
     $ Name     : chr  "L5_Ovis_10098_lm2_sin" "L5_Ovis_10098_lm2_sin" "L5_Ovis_11723_lm2_sin" "L5_Ovis_11723_lm2_sin" ...
     $ Diet     : chr  "Clover" "Clover" "Clover" "Clover" ...
     $ Software : chr  "Toothfrax" "ConfoMap" "ConfoMap" "Toothfrax" ...
     $ epLsar   : num  0.00192 0.0025 0.00121 0.00142 0.00243 ...
     $ R²       : num  1 0.997 0.999 1 1 ...
     $ Asfc     : num  9.56 9.31 2.15 2.23 5.88 ...
     $ Smfc     : num  2.7 24.16 4.11 1.2 1.2 ...
     $ HAsfc9   : num  1.33 1.328 0.236 0.248 0.483 ...
     $ HAsfc81  : num  2.057 2.143 0.726 0.736 1.845 ...
     $ NewEplsar: num  NA 0.0187 0.0172 NA NA ...

``` r
head(sheep_final)
```

      Dataset                  Name   Diet  Software      epLsar        R²     Asfc
    1  Sheeps L5_Ovis_10098_lm2_sin Clover Toothfrax 0.001922000 0.9995440 9.562123
    2  Sheeps L5_Ovis_10098_lm2_sin Clover  ConfoMap 0.002497369 0.9967923 9.308211
    3  Sheeps L5_Ovis_11723_lm2_sin Clover  ConfoMap 0.001207926 0.9989390 2.150905
    4  Sheeps L5_Ovis_11723_lm2_sin Clover Toothfrax 0.001416000 0.9998590 2.230088
    5  Sheeps L5_Ovis_20939_lm2_sin Clover Toothfrax 0.002431000 0.9996860 5.877414
    6  Sheeps L5_Ovis_20939_lm2_sin Clover  ConfoMap 0.002741181 0.9981801 5.621359
           Smfc    HAsfc9   HAsfc81  NewEplsar
    1  2.695842 1.3296145 2.0574152         NA
    2 24.155831 1.3283692 2.1434048 0.01865613
    3  4.106717 0.2358310 0.7257756 0.01715131
    4  1.198152 0.2476403 0.7357331         NA
    5  1.198152 0.4831631 1.8453318         NA
    6  9.251129 0.5678206 1.9202353 0.01843511

------------------------------------------------------------------------

# Lithics

Note that for compactness, comments to explain the code are given only
in the section about [Guinea Pigs](#guinea-pigs).

## ConfoMap

### Read in data

``` r
confoLith <- names(info_in) %>% 
             .[grepl("ConfoMap", .) & grepl("Lithics", .)] %>% 
             read.csv(header = FALSE, na.strings = "*****")
```

### Exclude repeated rows and select columns

``` r
confoLith_keep_col  <- c(12, 47:53) 
confoLith_keep_rows <- which(confoLith[[1]] != "#")
confoLith_keep      <- confoLith[confoLith_keep_rows, confoLith_keep_col] 
```

### Add headers

``` r
colnames(confoLith_keep) <- confoLith[2, confoLith_keep_col] %>% 
                            make.names() %>% 
                            gsub("\\.+", "\\.", x = .) %>% 
                            gsub("\\.$", "", x = .) %>%
                            gsub("^([A-Za-z0-9]+\\.)+", "", x = .)
```

### Extract units

``` r
confoLith_units <- unlist(confoLith[3, confoLith_keep_col[-1]]) 
names(confoLith_units) <- colnames(confoLith_keep)[-1]
```

### Convert to numeric and add column about software

``` r
confoLith_keep <- type_convert(confoLith_keep) %>% 
                  mutate(Software = "ConfoMap")
```

## Toothfrax

### Read in data

``` r
toothfraxLith <- names(info_in) %>% 
                 .[grepl("Toothfrax", .) & grepl("Lithics", .)] %>% 
                 read.xlsx(check.names = TRUE)
```

### Select columns, edit headers and add column about software

``` r
toothfraxLith_keep <- select(toothfraxLith, Name = Filename, epLsar = epLsar1.80µm, 
                             R² = Rsquared, Asfc = Asfc1om, Smfc = LineStart, 
                             HAsfc9 = X3x3HAsfc, HAsfc81 = X9x9HAsfc) %>% 
                      mutate(Software = "Toothfrax")
```

### Edit column “Name”

``` r
toothfraxLith_keep[["Name"]] <- gsub("(\\.sur$)|(^([_A-Za-z0-9-]+) --- )", "", 
                                     toothfraxLith_keep[["Name"]])
```

### Get units

``` r
if (length(unique(toothfraxLith[["Units"]])) != 1) {
  warning(paste("The studiables have different units for the SSFA parameters.",
                "Only the unit for the first studiable will be kept.",
                "Please check in the original file.", sep = "\n"))
}  
toothfraxLith_units <- toothfraxLith[["Units"]][1]
names(toothfraxLith_units) <- "Toothfrax_units"
```

## Merge datasets

### Merge

``` r
lith_keep <- merge(confoLith_keep, toothfraxLith_keep, all = TRUE)
```

### Exclude FLT3-8\_Area1 (“before” wrongly acquired )

``` r
FLT3_8_Area1 <- grep("^FLT3-8_.+_Area1", lith_keep[["Name"]])
lith_keep    <- lith_keep[-FLT3_8_Area1, ]
```

### Add columns about before/after and treatment

``` r
lith_keep[grep("LSM_",  lith_keep[["Name"]]), "Before.after"] <- "Before"
lith_keep[grep("LSM2_", lith_keep[["Name"]]), "Before.after"] <- "After"
lith_keep[["Before.after"]] <- factor(lith_keep[["Before.after"]], levels = c("Before", "After"))

lith_keep[grep("FLT3-8|QTZ3-2",   lith_keep[["Name"]]), "Treatment"] <- "Control"
lith_keep[grep("FLT3-9|QTZ3-5",   lith_keep[["Name"]]), "Treatment"] <- "BrushNoDirt"
lith_keep[grep("FLT3-13|QTZ3-3",  lith_keep[["Name"]]), "Treatment"] <- "RubDirt"
lith_keep[grep("FLT3-10|QTZ3-13", lith_keep[["Name"]]), "Treatment"] <- "BrushDirt"
```

### Add column about dataset and re-order columns

``` r
lith_final <- lith_keep %>% 
              mutate(Dataset = "Lithics") %>% 
              select(Dataset, Name, Treatment, Before.after, Software, everything())
```

### Add units

``` r
lith_units <- c(confoLith_units, toothfraxLith_units) 
```

### Check the result

``` r
str(lith_final)
```

    'data.frame':   60 obs. of  12 variables:
     $ Dataset     : chr  "Lithics" "Lithics" "Lithics" "Lithics" ...
     $ Name        : chr  "FLT3-10_LSM_50x-0.95_20190328_Area1_Topo" "FLT3-10_LSM_50x-0.95_20190328_Area1_Topo" "FLT3-10_LSM_50x-0.95_20190328_Area2_Topo" "FLT3-10_LSM_50x-0.95_20190328_Area2_Topo" ...
     $ Treatment   : chr  "BrushDirt" "BrushDirt" "BrushDirt" "BrushDirt" ...
     $ Before.after: Factor w/ 2 levels "Before","After": 1 1 1 1 2 2 2 2 1 1 ...
     $ Software    : chr  "ConfoMap" "Toothfrax" "Toothfrax" "ConfoMap" ...
     $ epLsar      : num  0.00109 0.00133 0.0015 0.00169 0.00144 ...
     $ R²          : num  0.998 0.999 0.999 0.998 0.998 ...
     $ Asfc        : num  42.5 42.5 39.3 38.9 45.4 ...
     $ Smfc        : num  0.0983 0.0145 0.0145 0.0983 0.0983 ...
     $ HAsfc9      : num  0.1155 0.0846 0.1142 0.1448 0.1066 ...
     $ HAsfc81     : num  0.182 0.148 0.161 0.192 0.159 ...
     $ NewEplsar   : num  0.0172 NA NA 0.0174 0.0171 ...

``` r
head(lith_final)
```

      Dataset                                      Name Treatment Before.after
    1 Lithics  FLT3-10_LSM_50x-0.95_20190328_Area1_Topo BrushDirt       Before
    2 Lithics  FLT3-10_LSM_50x-0.95_20190328_Area1_Topo BrushDirt       Before
    3 Lithics  FLT3-10_LSM_50x-0.95_20190328_Area2_Topo BrushDirt       Before
    4 Lithics  FLT3-10_LSM_50x-0.95_20190328_Area2_Topo BrushDirt       Before
    5 Lithics FLT3-10_LSM2_50x-0.95_20190731_Area1_Topo BrushDirt        After
    6 Lithics FLT3-10_LSM2_50x-0.95_20190731_Area1_Topo BrushDirt        After
       Software      epLsar        R²     Asfc       Smfc     HAsfc9   HAsfc81
    1  ConfoMap 0.001086565 0.9983105 42.47754 0.09827368 0.11549472 0.1817285
    2 Toothfrax 0.001330000 0.9993130 42.52394 0.01451400 0.08457134 0.1481977
    3 Toothfrax 0.001502000 0.9994600 39.32644 0.01451400 0.11421294 0.1611069
    4  ConfoMap 0.001686946 0.9981817 38.86831 0.09827368 0.14477286 0.1915275
    5  ConfoMap 0.001438974 0.9982742 45.35370 0.09827368 0.10656510 0.1592675
    6 Toothfrax 0.001594000 0.9994940 45.69748 0.01451400 0.07897466 0.1266690
       NewEplsar
    1 0.01720891
    2         NA
    3         NA
    4 0.01737361
    5 0.01708277
    6         NA

------------------------------------------------------------------------

# Combine datasets

## Merge datasets

``` r
# Merge the first two datasets
all_data <- merge(GP_final, sheep_final, all = TRUE) %>% 
  
            # Merge the first two with the third dataset
            merge(lith_final, all = TRUE) %>% 
  
            # Re-order columns
            select(Dataset, Name, Software, Diet, Treatment, Before.after, everything()) %>% 
  
            # Re-order rows
            arrange(Dataset, Name, Software, Diet, Treatment, Before.after)
```

## Combine units

``` r
# Test if units are identical across datasets
if (all(sapply(list(GP_units, sheep_units), FUN = identical, lith_units)) == TRUE) {
  
  # If TRUE, use only lith_units
  # Combine into a data.frame for export
  units_table <- data.frame(variable = names(lith_units), units = lith_units) 
  # Add as comment
  comment(all_data) <- lith_units
  
  # If FALSE, use all units
} else {
  
  # Combine into a data.frame for export
  units_table <- data.frame(variable = names(lith_units), units_GP = GP_units, 
                            units_sheep = sheep_units, units_lith = lith_units) 
  
  # Combine all units into a single vector
  units_all <- c(GP_units, sheep_units, lith_units)
  
  # Rename the vector to reflect the dataset
  names(units_all) <- length(GP_units) %>% 
                      rep(c("GP", "sheep", "lith"), each = .) %>% 
                      paste(names(x), sep=".")
  
  # Add as comment
  comment(all_data) <- units_all
}
```

Type `comment(all_data)` to check the units of the parameters.

## Check the result

``` r
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

``` r
head(all_data)
```

         Dataset                           Name  Software        Diet Treatment
    1 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_1  ConfoMap Dry Lucerne      <NA>
    2 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_1 Toothfrax Dry Lucerne      <NA>
    3 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_2  ConfoMap Dry Lucerne      <NA>
    4 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_2 Toothfrax Dry Lucerne      <NA>
    5 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_3  ConfoMap Dry Lucerne      <NA>
    6 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_3 Toothfrax Dry Lucerne      <NA>
      Before.after      epLsar        R²     Asfc      Smfc    HAsfc9   HAsfc81
    1         <NA> 0.001960011 0.9983703 14.29323 0.4153938 0.1644618 0.3681239
    2         <NA> 0.001471000 0.9993430 12.92579 0.1192190 0.1819939 0.3368939
    3         <NA> 0.003662312 0.9991123 12.90555 0.4408179 0.1713614 0.4172341
    4         <NA> 0.002693000 0.9995140 11.99982 0.1192190 0.1586045 0.3818615
    5         <NA> 0.003140386 0.9989348 13.66607 0.4408179 0.1305287 0.3521687
    6         <NA> 0.002797000 0.9992400 12.71147 0.1192190 0.1170325 0.3130522
       NewEplsar
    1 0.01842431
    2         NA
    3 0.01888866
    4         NA
    5 0.01870314
    6         NA

------------------------------------------------------------------------

# Save data

## Create file names

``` r
all_xlsx <- paste0(dir_out, "/SSFA_all_data.xlsx")
all_rbin <- paste0(dir_out, "/SSFA_all_data.Rbin")
```

## Write to XLSX and Rbin

``` r
write.xlsx(list(data = all_data, units = units_table), file = all_xlsx) 
saveObject(all_data, file = all_rbin) 
```

Rbin files (e.g. `SSFA_all_data.Rbin`) can be easily read into an R
object (e.g. `rbin_data`) using the following code:

``` r
library(R.utils)
rbin_data <- loadObject("SSFA_all_data.Rbin")
```

## Show files information

``` r
info_out <- c(all_xlsx, all_rbin) %>%
            md5sum()
```

The checksum (MD5 hashes) of the exported files are:

                   files                         checksum
    1 SSFA_all_data.xlsx d0ccdc8387aac3732551c8a4d659a7bd
    2 SSFA_all_data.Rbin 55465bdc7308d20cdff2d3d1bdeea63d

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
     [9] tidyverse_1.3.0   R.utils_2.10.1    R.oo_1.24.0       R.methodsS3_1.8.1
    [13] openxlsx_4.2.3   

    loaded via a namespace (and not attached):
     [1] tidyselect_1.1.0  xfun_0.19         haven_2.3.1       colorspace_2.0-0 
     [5] vctrs_0.3.4       generics_0.1.0    htmltools_0.5.0   yaml_2.2.1       
     [9] rlang_0.4.8       pillar_1.4.6      withr_2.3.0       glue_1.4.2       
    [13] DBI_1.1.0         dbplyr_2.0.0      modelr_0.1.8      readxl_1.3.1     
    [17] lifecycle_0.2.0   munsell_0.5.0     gtable_0.3.0      cellranger_1.1.0 
    [21] rvest_0.3.6       zip_2.1.1         evaluate_0.14     knitr_1.30       
    [25] fansi_0.4.1       broom_0.7.2       Rcpp_1.0.5        backports_1.2.0  
    [29] scales_1.1.1      jsonlite_1.7.1    fs_1.5.0          hms_0.5.3        
    [33] digest_0.6.27     stringi_1.5.3     grid_4.0.3        rprojroot_2.0.2  
    [37] cli_2.1.0         magrittr_1.5      crayon_1.3.4      pkgconfig_2.0.3  
    [41] ellipsis_0.3.1    xml2_1.3.2        reprex_0.3.0      lubridate_1.7.9.2
    [45] rstudioapi_0.13   assertthat_0.2.1  rmarkdown_2.5     httr_1.4.2       
    [49] R6_2.5.0          compiler_4.0.3   

RStudio version 1.4.1043.

------------------------------------------------------------------------

END OF SCRIPT
