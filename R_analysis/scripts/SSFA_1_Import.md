Import SSFA datasets
================
Ivan Calandra
2021-12-01 10:24:55

-   [Goal of the script](#goal-of-the-script)
-   [Load packages](#load-packages)
-   [Get names and path all files](#get-names-and-path-all-files)
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
        -   [Add columns about before/after and
            treatment](#add-columns-about-beforeafter-and-treatment)
        -   [Add column about dataset and re-order
            columns](#add-column-about-dataset-and-re-order-columns-2)
        -   [Add units](#add-units-1)
        -   [Check the result](#check-the-result-2)
-   [Combine datasets](#combine-datasets)
    -   [Merge datasets](#merge-datasets-3)
    -   [Combine units](#combine-units)
    -   [Fill in NMP values for Toothfrax
        samples](#fill-in-nmp-values-for-toothfrax-samples)
    -   [Add column for NMP categories](#add-column-for-nmp-categories)
    -   [Check the result](#check-the-result-3)
-   [Save data](#save-data)
    -   [Create file names](#create-file-names)
    -   [Write to XLSX and Rbin](#write-to-xlsx-and-rbin)
-   [sessionInfo() and RStudio
    version](#sessioninfo-and-rstudio-version)
-   [Cite R packages used](#cite-r-packages-used)

------------------------------------------------------------------------

# Goal of the script

This script formats the output of the resulting files from applying
SSFA. The script will:

1.  Read in the original files  
2.  Format the data  
3.  Write XLSX-files and save R objects ready for further analysis in R

``` r
dir_out <- "R_analysis/derived_data"
dir_in  <- "R_analysis/raw_data"
```

Raw data must be located in “\~/R_analysis/raw_data”.  
Formatted data will be saved in “\~/R_analysis/derived_data”.

The knit directory for this script is the project directory.

------------------------------------------------------------------------

# Load packages

``` r
pack_to_load <- c("openxlsx", "R.utils", "tidyverse")
sapply(pack_to_load, library, character.only = TRUE, logical.return = TRUE) 
```

     openxlsx   R.utils tidyverse 
         TRUE      TRUE      TRUE 

------------------------------------------------------------------------

# Get names and path all files

``` r
info_in <- list.files(dir_in, pattern = "\\_v8.2.9767.csv$", full.names = TRUE)
info_in
```

    [1] "R_analysis/raw_data/SSFA-ConfoMap_GuineaPigs_NMPfilled_v8.2.9767.csv"
    [2] "R_analysis/raw_data/SSFA-ConfoMap_Lithics_NMPfilled_v8.2.9767.csv"   
    [3] "R_analysis/raw_data/SSFA-ConfoMap_Sheeps_NMPfilled_v8.2.9767.csv"    
    [4] "R_analysis/raw_data/SSFA-Toothfrax_GuineaPigs_v8.2.9767.csv"         
    [5] "R_analysis/raw_data/SSFA-Toothfrax_Lithics_v8.2.9767.csv"            
    [6] "R_analysis/raw_data/SSFA-Toothfrax_Sheeps_v8.2.9767.csv"             

------------------------------------------------------------------------

# Guinea pigs

## ConfoMap

### Read in data

``` r
# Extract file name for ConfoMap analysis on Guinea Pigs, and read in file
confoGP <- info_in %>% 
           .[grepl("ConfoMap", .) & grepl("GuineaPigs", .)] %>% 
           read.csv(header = FALSE, na.strings = "*****")
```

### Exclude repeated rows and select columns

``` r
confoGP_keep_col  <- c(12, 34, 50:56) # Define columns to keep
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

# Edit name for NMP
colnames(confoGP_keep)[c(2, 5)] <- c("NMP", "Rsquared")
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
toothfraxGP <- info_in %>% 
               .[grepl("Toothfrax", .) & grepl("GuineaPigs", .)] %>% 

               # Read in file
               read.csv(fileEncoding = "UTF-16")
```

### Select columns, edit headers and add column about software

``` r
toothfraxGP_keep <- select(toothfraxGP, Name = Filename, epLsar = epLsar1.80µm, 
                           Rsquared = Rsquared, Asfc = Asfc1om, Smfc = LineStart, 
                           HAsfc9 = X3x3HAsfc, HAsfc81 = X9x9HAsfc) %>% 
                    mutate(Software = "Toothfrax")
```

### Edit column “Name”

``` r
# Remove the extension (".sur") from the name and keep only the part of the name after " --- "
toothfraxGP_keep[["Name"]] <- gsub("_v8.2.9767\\.sur$", "", toothfraxGP_keep[["Name"]]) %>% 
                              strsplit(" --- ") %>% 
                              sapply(`[`, 2)
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
GP_keep[grep("2CC4", GP_keep[["Name"]]), "Diet"] <- "Dry lucerne"
GP_keep[grep("2CC5", GP_keep[["Name"]]), "Diet"] <- "Dry grass"
GP_keep[grep("2CC6", GP_keep[["Name"]]), "Diet"] <- "Dry bamboo"
```

### Add column about dataset and re-order columns

``` r
GP_final <- GP_keep %>% 
            mutate(Dataset = "GuineaPigs") %>% 
            mutate_if(is.character, as.factor) %>% 
            select(Dataset, Name, Diet, Software, NMP, everything())
```

### Combine both sets of units

``` r
GP_units <- c(confoGP_units, toothfraxGP_units) 
```

### Check the result

``` r
str(GP_final)
```

    'data.frame':   140 obs. of  12 variables:
     $ Dataset  : Factor w/ 1 level "GuineaPigs": 1 1 1 1 1 1 1 1 1 1 ...
     $ Name     : Factor w/ 70 levels "capor_2CC4B1_txP4_#1_1_100xL_1",..: 1 1 2 2 3 3 4 4 5 5 ...
     $ Diet     : Factor w/ 3 levels "Dry bamboo","Dry grass",..: 3 3 3 3 3 3 3 3 3 3 ...
     $ Software : Factor w/ 2 levels "ConfoMap","Toothfrax": 2 1 2 1 2 1 2 1 1 2 ...
     $ NMP      : num  NA 1.9 NA 1.31 NA ...
     $ epLsar   : num  0.00147 0.00196 0.00269 0.00366 0.0028 ...
     $ Rsquared : num  0.999 0.997 1 0.998 0.999 ...
     $ Asfc     : num  12.9 16 12 14.1 12.7 ...
     $ Smfc     : num  0.119 0.33 0.119 0.35 0.119 ...
     $ HAsfc9   : num  0.182 0.179 0.159 0.136 0.117 ...
     $ HAsfc81  : num  0.337 0.391 0.382 0.443 0.313 ...
     $ NewEplsar: num  NA 0.0184 NA 0.0189 NA ...

``` r
head(GP_final)
```

         Dataset                           Name        Diet  Software      NMP
    1 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_1 Dry lucerne Toothfrax       NA
    2 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_1 Dry lucerne  ConfoMap 1.896275
    3 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_2 Dry lucerne Toothfrax       NA
    4 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_2 Dry lucerne  ConfoMap 1.307524
    5 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_3 Dry lucerne Toothfrax       NA
    6 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_3 Dry lucerne  ConfoMap 0.806428
           epLsar  Rsquared     Asfc      Smfc    HAsfc9   HAsfc81  NewEplsar
    1 0.001471000 0.9993430 12.92579 0.1192190 0.1819939 0.3368939         NA
    2 0.001960011 0.9967182 16.00717 0.3303686 0.1785900 0.3908931 0.01842431
    3 0.002693000 0.9995140 11.99982 0.1192190 0.1586045 0.3818615         NA
    4 0.003662312 0.9976931 14.05932 0.3498752 0.1360639 0.4434925 0.01888866
    5 0.002797000 0.9992400 12.71147 0.1192190 0.1170325 0.3130522         NA
    6 0.003140386 0.9973672 15.12322 0.3303686 0.1306295 0.3566666 0.01870314

------------------------------------------------------------------------

# Sheeps

Note that for compactness, comments to explain the code are given only
in the section about [Guinea Pigs](#guinea-pigs).

## ConfoMap

### Read in data

``` r
confoSheep <- info_in %>% 
              .[grepl("ConfoMap", .) & grepl("Sheeps", .)] %>% 
              read.csv(header = FALSE, na.strings = "*****")
```

### Exclude repeated rows and select columns

``` r
confoSheep_keep_col  <- c(12, 30, 46:52)
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
colnames(confoSheep_keep)[c(2, 5)] <- c("NMP", "Rsquared")
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
toothfraxSheep <- info_in %>% 
                  .[grepl("Toothfrax", .) & grepl("Sheeps", .)] %>% 
                  read.csv(fileEncoding = "UTF-16")
```

### Select columns, edit headers and add column about software

``` r
toothfraxSheep_keep <- select(toothfraxSheep, Name = Filename, epLsar = epLsar1.80µm, 
                              Rsquared = Rsquared, Asfc = Asfc1om, Smfc = LineStart, 
                              HAsfc9 = X3x3HAsfc, HAsfc81 = X9x9HAsfc) %>% 
                       mutate(Software = "Toothfrax")
```

### Edit column “Name”

``` r
toothfraxSheep_keep[["Name"]] <- gsub("\\.sur$", "", toothfraxSheep_keep[["Name"]]) %>% 
                                 strsplit(" --- ") %>% 
                                 sapply(`[`, 2)
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
sheep_keep[grep("L6", sheep_keep[["Name"]]), "Diet"] <- "Clover+dust"
sheep_keep[grep("L7", sheep_keep[["Name"]]), "Diet"] <- "Grass"
sheep_keep[grep("L8", sheep_keep[["Name"]]), "Diet"] <- "Grass+dust"
```

### Add column about dataset and re-order columns

``` r
sheep_final <- sheep_keep %>% 
               mutate(Dataset = "Sheeps") %>% 
               mutate_if(is.character, as.factor) %>% 
               select(Dataset, Name, Diet, Software, NMP, everything())
```

### Add units

``` r
sheep_units <- c(confoSheep_units, toothfraxSheep_units) 
```

### Check the result

``` r
str(sheep_final)
```

    'data.frame':   80 obs. of  12 variables:
     $ Dataset  : Factor w/ 1 level "Sheeps": 1 1 1 1 1 1 1 1 1 1 ...
     $ Name     : Factor w/ 40 levels "L5_Ovis_10098_lm2_sin",..: 1 1 2 2 3 3 4 4 5 5 ...
     $ Diet     : Factor w/ 4 levels "Clover","Clover+dust",..: 1 1 1 1 1 1 1 1 1 1 ...
     $ Software : Factor w/ 2 levels "ConfoMap","Toothfrax": 2 1 1 2 2 1 2 1 2 1 ...
     $ NMP      : num  NA 0 0 NA NA 0 NA 0 NA 0 ...
     $ epLsar   : num  0.00192 0.0025 0.00121 0.00142 0.00243 ...
     $ Rsquared : num  1 0.999 0.999 1 1 ...
     $ Asfc     : num  9.56 10.74 2.3 2.23 5.88 ...
     $ Smfc     : num  2.7 12.29 3.37 1.2 1.2 ...
     $ HAsfc9   : num  1.33 1.44 0.288 0.248 0.483 ...
     $ HAsfc81  : num  2.057 2.348 0.813 0.736 1.845 ...
     $ NewEplsar: num  NA 0.0187 0.0172 NA NA ...

``` r
head(sheep_final)
```

      Dataset                  Name   Diet  Software NMP      epLsar  Rsquared
    1  Sheeps L5_Ovis_10098_lm2_sin Clover Toothfrax  NA 0.001922000 0.9995440
    2  Sheeps L5_Ovis_10098_lm2_sin Clover  ConfoMap   0 0.002497369 0.9986401
    3  Sheeps L5_Ovis_11723_lm2_sin Clover  ConfoMap   0 0.001207926 0.9987381
    4  Sheeps L5_Ovis_11723_lm2_sin Clover Toothfrax  NA 0.001416000 0.9998590
    5  Sheeps L5_Ovis_20939_lm2_sin Clover Toothfrax  NA 0.002431000 0.9996860
    6  Sheeps L5_Ovis_20939_lm2_sin Clover  ConfoMap   0 0.002741181 0.9973668
           Asfc      Smfc    HAsfc9   HAsfc81  NewEplsar
    1  9.562123  2.695842 1.3296145 2.0574152         NA
    2 10.737686 12.285733 1.4401817 2.3481407 0.01865613
    3  2.300115  3.374298 0.2884126 0.8130148 0.01715131
    4  2.230088  1.198152 0.2476403 0.7357331         NA
    5  5.877414  1.198152 0.4831631 1.8453318         NA
    6  6.665880  2.720491 0.5665925 2.0633539 0.01843511

------------------------------------------------------------------------

# Lithics

Note that for compactness, comments to explain the code are given only
in the section about [Guinea Pigs](#guinea-pigs).

## ConfoMap

### Read in data

``` r
confoLith <- info_in %>% 
             .[grepl("ConfoMap", .) & grepl("Lithics", .)] %>% 
             read.csv(header = FALSE, na.strings = "*****")
```

### Exclude repeated rows and select columns

``` r
confoLith_keep_col  <- c(4, 39:46) 
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
colnames(confoLith_keep)[c(2, 5)] <- c("NMP", "Rsquared")
```

### Extract units

``` r
confoLith_units <- unlist(confoLith[3, confoLith_keep_col[-1]]) %>%
                   gsub("ï¿½mï¿½", "µm²", .)
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
toothfraxLith <- info_in %>% 
                 .[grepl("Toothfrax", .) & grepl("Lithics", .)] %>% 
                 read.csv(fileEncoding = "UTF-16")
```

### Select columns, edit headers and add column about software

``` r
toothfraxLith_keep <- select(toothfraxLith, Name = Filename, epLsar = epLsar1.80µm, 
                             Rsquared = Rsquared, Asfc = Asfc1om, Smfc = LineStart, 
                             HAsfc9 = X3x3HAsfc, HAsfc81 = X9x9HAsfc) %>% 
                      mutate(Software = "Toothfrax")
```

### Edit column “Name”

``` r
toothfraxLith_keep[["Name"]] <- gsub("\\.sur$", "", toothfraxLith_keep[["Name"]]) %>% 
                                strsplit(" --- ") %>% 
                                sapply(`[`, 2)
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

### Add columns about before/after and treatment

``` r
lith_keep[grep("LSM_",  lith_keep[["Name"]]), "Before.after"] <- "Before"
lith_keep[grep("LSM2_", lith_keep[["Name"]]), "Before.after"] <- "After"
lith_keep[["Before.after"]] <- factor(lith_keep[["Before.after"]], levels = c("Before", "After"))

lith_keep[grep("FLT3-8|QTZ3-2",   lith_keep[["Name"]]), "Treatment"] <- "Control"
lith_keep[grep("FLT3-9|QTZ3-5",   lith_keep[["Name"]]), "Treatment"] <- "BrushNoDirt"
lith_keep[grep("FLT3-13|QTZ3-3",  lith_keep[["Name"]]), "Treatment"] <- "RubDirt"
lith_keep[grep("FLT3-10|QTZ3-13", lith_keep[["Name"]]), "Treatment"] <- "BrushDirt"
lith_keep[["Treatment"]] <- factor(lith_keep[["Treatment"]], 
                                   levels = c("Control", "RubDirt", "BrushNoDirt", "BrushDirt"))
```

### Add column about dataset and re-order columns

``` r
lith_final <- lith_keep %>% 
              mutate(Dataset = "Lithics") %>% 
              mutate_if(is.character, as.factor) %>% 
              select(Dataset, Name, Treatment, Before.after, Software, NMP, everything())
```

### Add units

``` r
lith_units <- c(confoLith_units, toothfraxLith_units) 
```

### Check the result

``` r
str(lith_final)
```

    'data.frame':   64 obs. of  13 variables:
     $ Dataset     : Factor w/ 1 level "Lithics": 1 1 1 1 1 1 1 1 1 1 ...
     $ Name        : Factor w/ 32 levels "FLT3-10_LSM_50x-0.95_20190328_Area1_Topo",..: 1 1 2 2 3 3 4 4 5 5 ...
     $ Treatment   : Factor w/ 4 levels "Control","RubDirt",..: 4 4 4 4 4 4 4 4 2 2 ...
     $ Before.after: Factor w/ 2 levels "Before","After": 1 1 1 1 2 2 2 2 1 1 ...
     $ Software    : Factor w/ 2 levels "ConfoMap","Toothfrax": 2 1 2 1 1 2 2 1 1 2 ...
     $ NMP         : num  NA 1.77 NA 1.92 2.21 ...
     $ epLsar      : num  0.00269 0.00282 0.00221 0.00253 0.00268 ...
     $ Rsquared    : num  1 0.999 1 0.999 0.999 ...
     $ Asfc        : num  37.2 44.7 36.5 43.7 45.7 ...
     $ Smfc        : num  0.0145 0.1005 0.0145 0.1005 0.1005 ...
     $ HAsfc9      : num  0.0819 0.0766 0.0881 0.0999 0.0925 ...
     $ HAsfc81     : num  0.168 0.237 0.22 0.261 0.189 ...
     $ NewEplsar   : num  NA 0.0165 NA 0.017 0.0166 ...

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
       Software      NMP      epLsar  Rsquared     Asfc      Smfc     HAsfc9
    1 Toothfrax       NA 0.002693000 0.9998420 37.18201 0.0145140 0.08192972
    2  ConfoMap 1.766336 0.002819223 0.9989832 44.67407 0.1004648 0.07663490
    3 Toothfrax       NA 0.002211000 0.9997200 36.54107 0.0145140 0.08806302
    4  ConfoMap 1.922810 0.002529152 0.9991157 43.74264 0.1004648 0.09994383
    5  ConfoMap 2.211463 0.002683912 0.9986472 45.72611 0.1004648 0.09252342
    6 Toothfrax       NA 0.002752000 0.9998380 38.80595 0.0145140 0.09561877
        HAsfc81  NewEplsar
    1 0.1676768         NA
    2 0.2368160 0.01654116
    3 0.2196744         NA
    4 0.2608172 0.01702859
    5 0.1887984 0.01664714
    6 0.1460229         NA

------------------------------------------------------------------------

# Combine datasets

## Merge datasets

``` r
# Merge the first two datasets
all_data <- merge(GP_final, sheep_final, all = TRUE) %>% 
  
            # Merge the first two with the third dataset
            merge(lith_final, all = TRUE) %>% 
  
            # Re-order columns
            select(Dataset, Name, Software, Diet, Treatment, Before.after, NMP, everything()) %>% 
  
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

## Fill in NMP values for Toothfrax samples

The NMP values were computed in the ConfoMap templates used to (1)
pre-process acquired surfaces, (2) export the pre-processed surfaces for
the SSFA analysis in Toothfrax and (3) run the SSFA analysis in ConfoMap
on the pre-processed surfaces.  
NMP values are therefore not available in the Toothfrax datasets.
However, the same pre-processed surfaces were used in both ConfoMap and
Toothfrax, so the NMP values calculated in ConfoMap also apply to the
Toothfrax dataset.  
Thus, here we copy the values from ConfoMap rows to Toothfrax rows.

``` r
all_data <- fill(all_data, NMP)
```

## Add column for NMP categories

Here we define four ranges of non-measured points (NMP):

-   \< 5% NMP: “0-5%”  
-   ≥ 5% and \< 10% NMP: “5-10%”  
-   ≥ 10% and \< 20% NMP: “10-20%”  
-   ≥ 20% NMP: “20-100%” (these surfaces were not acquired correctly and
    should not be used)

``` r
# Create new column and fill it
all_data[all_data$NMP < 5                      , "NMP_cat"] <- "0-5%"
all_data[all_data$NMP >= 5  & all_data$NMP < 10, "NMP_cat"] <- "5-10%"
all_data[all_data$NMP >= 10 & all_data$NMP < 20, "NMP_cat"] <- "10-20%"
all_data[all_data$NMP >= 20                    , "NMP_cat"] <- "20-100%"

# Convert to ordered factor
all_data[["NMP_cat"]] <- factor(all_data[["NMP_cat"]], 
                                levels = c("0-5%", "5-10%", "10-20%", "20-100%"), ordered = TRUE)

# Re-order columns
all_data <- select(all_data, Dataset:NMP, NMP_cat, everything())
```

## Check the result

``` r
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

``` r
head(all_data)
```

         Dataset                           Name  Software        Diet Treatment
    1 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_1  ConfoMap Dry lucerne      <NA>
    2 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_1 Toothfrax Dry lucerne      <NA>
    3 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_2  ConfoMap Dry lucerne      <NA>
    4 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_2 Toothfrax Dry lucerne      <NA>
    5 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_3  ConfoMap Dry lucerne      <NA>
    6 GuineaPigs capor_2CC4B1_txP4_#1_1_100xL_3 Toothfrax Dry lucerne      <NA>
      Before.after      NMP NMP_cat      epLsar  Rsquared     Asfc      Smfc
    1         <NA> 1.896275    0-5% 0.001960011 0.9967182 16.00717 0.3303686
    2         <NA> 1.896275    0-5% 0.001471000 0.9993430 12.92579 0.1192190
    3         <NA> 1.307524    0-5% 0.003662312 0.9976931 14.05932 0.3498752
    4         <NA> 1.307524    0-5% 0.002693000 0.9995140 11.99982 0.1192190
    5         <NA> 0.806428    0-5% 0.003140386 0.9973672 15.12322 0.3303686
    6         <NA> 0.806428    0-5% 0.002797000 0.9992400 12.71147 0.1192190
         HAsfc9   HAsfc81  NewEplsar
    1 0.1785900 0.3908931 0.01842431
    2 0.1819939 0.3368939         NA
    3 0.1360639 0.4434925 0.01888866
    4 0.1586045 0.3818615         NA
    5 0.1306295 0.3566666 0.01870314
    6 0.1170325 0.3130522         NA

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
```

    Error in saveWorkbook(wb, file = file, overwrite = overwrite): File already exists!

``` r
saveObject(all_data, file = all_rbin) 
```

Rbin files (e.g. `SSFA_all_data.Rbin`) can be easily read into an R
object (e.g. `rbin_data`) using the following code:

``` r
library(R.utils)
rbin_data <- loadObject("SSFA_all_data.Rbin")
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
     [9] tidyverse_1.3.1   R.utils_2.11.0    R.oo_1.24.0       R.methodsS3_1.8.1
    [13] openxlsx_4.2.4   

    loaded via a namespace (and not attached):
     [1] tidyselect_1.1.1 xfun_0.28        haven_2.4.3      colorspace_2.0-2
     [5] vctrs_0.3.8      generics_0.1.1   htmltools_0.5.2  yaml_2.2.1      
     [9] utf8_1.2.2       rlang_0.4.12     jquerylib_0.1.4  pillar_1.6.4    
    [13] withr_2.4.2      glue_1.5.0       DBI_1.1.1        dbplyr_2.1.1    
    [17] readxl_1.3.1     modelr_0.1.8     lifecycle_1.0.1  cellranger_1.1.0
    [21] munsell_0.5.0    gtable_0.3.0     rvest_1.0.2      zip_2.2.0       
    [25] evaluate_0.14    knitr_1.36       tzdb_0.2.0       fastmap_1.1.0   
    [29] fansi_0.5.0      broom_0.7.10     Rcpp_1.0.7       renv_0.14.0     
    [33] backports_1.4.0  scales_1.1.1     jsonlite_1.7.2   fs_1.5.0        
    [37] hms_1.1.1        digest_0.6.28    stringi_1.7.6    rprojroot_2.0.2 
    [41] grid_4.1.2       cli_3.1.0        tools_4.1.2      magrittr_2.0.1  
    [45] crayon_1.4.2     pkgconfig_2.0.3  ellipsis_0.3.2   xml2_1.3.2      
    [49] reprex_2.0.1     lubridate_1.8.0  rstudioapi_0.13  assertthat_0.2.1
    [53] rmarkdown_2.11   httr_1.4.2       R6_2.5.1         compiler_4.1.2  

RStudio version 2021.9.1.372.

------------------------------------------------------------------------

# Cite R packages used

    openxlsx 
    Philipp Schauberger and Alexander Walker (2021). openxlsx: Read, Write and Edit xlsx Files. R package version 4.2.4. https://CRAN.R-project.org/package=openxlsx 
     
    R.utils 
    Henrik Bengtsson (2021). R.utils: Various Programming Utilities. R package version 2.11.0. https://CRAN.R-project.org/package=R.utils 
     
    tidyverse 
    Wickham et al., (2019). Welcome to the tidyverse. Journal of Open Source Software, 4(43), 1686, https://doi.org/10.21105/joss.01686 
     

------------------------------------------------------------------------

END OF SCRIPT
