Checksums
================
Ivan Calandra
2021-01-26 09:47:57

-   [Goal of the script](#goal-of-the-script)
-   [Load packages](#load-packages)
-   [Get names, path and MD5 checksums of all
    files](#get-names-path-and-md5-checksums-of-all-files)
    -   [R analysis](#r-analysis)
    -   [Python analysis](#python-analysis)
-   [sessionInfo() and RStudio
    version](#sessioninfo-and-rstudio-version)

------------------------------------------------------------------------

# Goal of the script

This script calculates the MD5 checksums of all files in the repository
to check for integrity and file versions.

------------------------------------------------------------------------

# Load packages

``` r
library(tools)
library(tidyverse)
```

------------------------------------------------------------------------

# Get names, path and MD5 checksums of all files

## R analysis

``` r
check_R <- list.files(path = "R_analysis", full.names = TRUE, recursive = TRUE) %>% 
           md5sum()
```

The MD5 checksums of the files created during the R analysis are:

    [1] R_analysis/derived_data/SSFA_all_data.Rbin
    d2fc242595873d663b12beb7633acb0a 

    [2] R_analysis/derived_data/SSFA_all_data.xlsx
    1a90206aa7c2daeb8c4054f3fc464b26 

    [3] R_analysis/plots/SSFA_GuineaPigs_plot.pdf
    f4e2b5e2327ea1a1ac7e207e6655f56d 

    [4] R_analysis/plots/SSFA_GuineaPigs_plot_Smfc.pdf
    d54f897c0ab7b6c38018f4f73b83d45a 

    [5] R_analysis/plots/SSFA_Lithics_plot.pdf
    772422eacdccb3f245eea686b270d59e 

    [6] R_analysis/plots/SSFA_Lithics_plot_Smfc.pdf
    01d18e1319268773c49c76f0c9eb2044 

    [7] R_analysis/plots/SSFA_Sheeps_plot.pdf
    84d12ff53a664c224c3f837e000e589d 

    [8] R_analysis/plots/SSFA_Sheeps_plot_Smfc.pdf
    65e746b88996db7cf086e212b62cb88c 

    [9] R_analysis/raw_data/SSFA-ConfoMap_GuineaPigs_NMPfilled.csv
    bb465ffcd50b3c2038e2c29eb403acfe 

    [10] R_analysis/raw_data/SSFA-ConfoMap_Lithics_NMPfilled.csv
    cd8819fa914a7f2a0a25d7a2b86913b9 

    [11] R_analysis/raw_data/SSFA-ConfoMap_Sheeps_NMPfilled.csv
    e010ebe147f86c23a2ccd8fe9f507d3c 

    [12] R_analysis/raw_data/SSFA-Toothfrax_GuineaPigs.xlsx
    59684695c021e541699ea075a8c597a3 

    [13] R_analysis/raw_data/SSFA-Toothfrax_Lithics.xlsx
    5f9c641b05212b7565e79578d3a4eeac 

    [14] R_analysis/raw_data/SSFA-Toothfrax_Sheeps.xlsx
    90be4f96fe6178125f579b5b4993bdd5 

    [15] R_analysis/scripts/SSFA_0_CreateRC.html
    3ddaf6e51bdd9d0376e4c47e89dcd7c3 

    [16] R_analysis/scripts/SSFA_0_CreateRC.md
    241cb867b1d2a26740a9cb1ea8c97eda 

    [17] R_analysis/scripts/SSFA_0_CreateRC.Rmd
    77760192c91b47d074001ab77b63365b 

    [18] R_analysis/scripts/SSFA_0_RStudioVersion.R
    ef459250d4a8f0247ec6b027382798be 

    [19] R_analysis/scripts/SSFA_0_RStudioVersion.txt
    797456f01eb4b0a915eee4b2dde6e778 

    [20] R_analysis/scripts/SSFA_1_Import.html
    f480d3d3324393f0d016da40b5766668 

    [21] R_analysis/scripts/SSFA_1_Import.md
    da07b82e3f7a44f2052e47c6a4f1515b 

    [22] R_analysis/scripts/SSFA_1_Import.Rmd
    540b2cc1cd9e3ca4d84b03958aa6a2a2 

    [23] R_analysis/scripts/SSFA_2_Summary-stats.html
    74edb5a092487e20459ccb05fbffffb6 

    [24] R_analysis/scripts/SSFA_2_Summary-stats.md
    93b92c42844e78814af8c2648bf7cd6c 

    [25] R_analysis/scripts/SSFA_2_Summary-stats.Rmd
    8ed776201cb4097b3be17b2ebe9eb57f 

    [26] R_analysis/scripts/SSFA_3_Plots.html
    13514a2174f9c1013f7909b5cff3dff7 

    [27] R_analysis/scripts/SSFA_3_Plots.md
    4001658dd4780778257ae56743903b71 

    [28] R_analysis/scripts/SSFA_3_Plots.Rmd
    dca4509d82f0b7929185b7b4958382b0 

    [29] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-10-1.png
    631dc9a7ef7c552d8922c250c16303ef 

    [30] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-11-1.png
    0471fe6338fda1876a05162be099d1e9 

    [31] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-12-1.png
    3ec45c68c487bb8dafddef4e8c8d18f7 

    [32] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-12-2.png
    ec7e7751b860b645c81b112649306503 

    [33] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-12-3.png
    729f9bdcdd32af8b811f0cf60479012e 

    [34] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-9-1.png
    1f1785fcdf2d98752418ffa49ce95bfa 

    [35] R_analysis/summary_stats/SSFA_summary_stats.xlsx
    aeaff0fd6daa923692991db35ac2f4ba 

## Python analysis

``` r
check_Python <- list.files(path = "Python_analysis", full.names = TRUE, recursive = TRUE) %>% 
                md5sum()
```

The MD5 checksums of the files created during the Python analysis are:

    [1] Python_analysis/code/plotting_lib.py
    c05a4787dac1e76a298ea3975ac8bd90 

    [2] Python_analysis/code/Preprocessing.html
    b1b45c06be5aaf463582f46716c323c8 

    [3] Python_analysis/code/Preprocessing.ipynb
    e0c2056f7f506ad5c2b43c784ea1441e 

    [4] Python_analysis/code/Preprocessing.md
    da0f269f9f343fa970ab84612f353619 

    [5] Python_analysis/code/Preprocessing_files/Preprocessing_33_0.png
    ad57bd0524dc2a624733723f710fe90e 

    [6] Python_analysis/code/Preprocessing_files/Preprocessing_33_1.png
    e47c192390d4cf51784e56ca1df735fe 

    [7] Python_analysis/code/Preprocessing_files/Preprocessing_33_2.png
    644e6bf694d03b6949c1c53e4aa5d9bd 

    [8] Python_analysis/code/Preprocessing_files/Preprocessing_33_3.png
    e6a53d03e38ce1e9570f11e68530eb9d 

    [9] Python_analysis/code/Preprocessing_files/Preprocessing_33_4.png
    4bb2fcab3dd54cb4f2dd3969551fbb98 

    [10] Python_analysis/code/Preprocessing_files/Preprocessing_33_5.png
    342a5fa2b56345c5ec30388441523fdb 

    [11] Python_analysis/code/Preprocessing_files/Preprocessing_45_0.png
    c8620e3365b6ed25af7e857cbef82a62 

    [12] Python_analysis/code/Preprocessing_files/Preprocessing_45_1.png
    b99e338fec69bb707cc9be6e355a1e75 

    [13] Python_analysis/code/Preprocessing_files/Preprocessing_45_2.png
    75d9a94c6e2fd8608bb93344b0a8917a 

    [14] Python_analysis/code/Preprocessing_files/Preprocessing_45_3.png
    c73c5e937afcea1c345ab16e0b9d1a39 

    [15] Python_analysis/code/Preprocessing_files/Preprocessing_45_4.png
    39a678979071edfc2b6c4cc3ac72b931 

    [16] Python_analysis/code/Preprocessing_files/Preprocessing_45_5.png
    67f9c2ef18d6f1b0942fb72928620396 

    [17] Python_analysis/code/Statistical_Model_NewEplsar.html
    75b24709455127285037f058e6a4e355 

    [18] Python_analysis/code/Statistical_Model_NewEplsar.ipynb
    5b1739d4ece56d7ab6a91b0e65d39d91 

    [19] Python_analysis/code/Statistical_Model_NewEplsar.md
    7531f360cc6d7c9bdeec7dff3ca804a1 

    [20] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_33_0.png
    02d518a1a16e7413f0df9585b08eb368 

    [21] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_35_0.png
    2832e76b0cd627d0a48c9b91511f47ea 

    [22] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_37_0.png
    67bf1db2ea3ac5f374c691098b19abce 

    [23] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_38_0.png
    4d17aff9c65bfec16fb0b6a30b4e3fef 

    [24] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_39_0.png
    0d274800919df562266a264b30001ad2 

    [25] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_54_0.svg
    89b703d39e14b7d81e18d31abface5c7 

    [26] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_57_0.png
    573c7d0d17815abd9e58b3968d497510 

    [27] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_65_0.png
    d875154bb42dd54cb9a4e8e4ea29b7cf 

    [28] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_66_0.png
    a9e78f1c0f4257898719760572d062bd 

    [29] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_67_0.png
    cda47211c4e88b2ec8c100c8f202e8b3 

    [30] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_70_0.png
    9b7a5103545b91ffbc5812d39a31bdc5 

    [31] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_73_0.png
    35a00dd379343a148ec1e301b114ee82 

    [32] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_78_0.png
    fa2d1052539b500c82b0baf038feb39d 

    [33] Python_analysis/code/Statistical_Model_ThreeFactor.html
    03bfedfc50e079b2f273c39ea6a0332c 

    [34] Python_analysis/code/Statistical_Model_ThreeFactor.ipynb
    685fb51389107c010a61a135477b063d 

    [35] Python_analysis/code/Statistical_Model_ThreeFactor.md
    dc1fd1f84d0fb00ea77be9b85bc78718 

    [36] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_101_0.png
    b8568ab5856f6055facefba220afc2b6 

    [37] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_1.png
    039bbe3b647ab73a91edf9e4d148b284 

    [38] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_2.png
    e7b8335e7a61d887138fb9a7f661713c 

    [39] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_3.png
    d1e89069381b7c0f6b7e7b1cddc4bcb6 

    [40] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_4.png
    b61a4256b0e316af1d9a358273913f5e 

    [41] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_110_0.png
    003ef235a241790d9ae3c8cf41313b1c 

    [42] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_111_0.png
    167b5d1aa1b320a8890aaa1ac99fedaa 

    [43] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_114_0.png
    564deb8ae4b42d411f4acd68325e4623 

    [44] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_117_0.png
    af315b28812a611dc19b6c575ba171de 

    [45] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_119_0.png
    0c56a1418e3cea31adb4d1796e46e7cb 

    [46] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_120_0.png
    57213cb2cce5ba449acc063f726fef35 

    [47] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_125_0.svg
    4012461c927fff5ec15cc766556bbdcd 

    [48] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_128_0.png
    363c4a75a5f42d29c28c88967d77e626 

    [49] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_137_0.svg
    d5002c411a97e2e5395cc465ca673681 

    [50] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_140_0.png
    6a29a8c633cd46c60a7c4c0e2ed5575c 

    [51] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_1.png
    865557a3f7a35c797ab10ed56fc374bc 

    [52] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_2.png
    f5a6baabdb2b637da40faa4191c99eb0 

    [53] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_3.png
    a84516182e592c66563ef57a4412f807 

    [54] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_4.png
    9e1cd518e3b279a9bcd184a1ba21088b 

    [55] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_149_0.png
    aca93fb781fdc9e6b902681abf8d0de4 

    [56] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_150_0.png
    32e93e43736df7995edeb9aa9d9dbd85 

    [57] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_153_0.png
    79e45eadf719f2168e71290ec10e8381 

    [58] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_156_0.png
    04ae85012e989d8ef5d305163512233c 

    [59] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_158_0.png
    38f2f42df29f962321662c67acc48d27 

    [60] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_159_0.png
    bd49e52417d85ed2455041ffde7673af 

    [61] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_164_0.svg
    709ae72a4bb7858c1f2a060eecce8f9a 

    [62] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_167_0.png
    e24e874a951efc16e71a5370b94c91d3 

    [63] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_1.png
    8d79c2b27847a89c70d9bc5ee515828c 

    [64] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_2.png
    21ca481f25b6b9683a8bd68c3d49da13 

    [65] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_3.png
    bfc68481b058f7e8f521ba97a1750e0c 

    [66] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_4.png
    7e35349cc4cc25801349a06704cec515 

    [67] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_176_0.png
    4949e161b113f2c46610e8f1be707745 

    [68] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_177_0.png
    385857f35130cab031cb0fcf1917b49a 

    [69] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_180_0.png
    70b79e6a0573f3a21a38ae8775b1110d 

    [70] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_183_0.png
    60b01c4106fa2d53abd4e5b106f0da44 

    [71] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_185_0.png
    69d62765e76099e2081bd0faee437dee 

    [72] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_186_0.png
    ab4c7d941547c32a2ed3f950004eedd0 

    [73] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_188_0.png
    a8bcef1a055b07cc60148372b45c4c45 

    [74] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_189_0.png
    8b1e68dc9d112fc477627922ba313f71 

    [75] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_190_0.png
    57213cb2cce5ba449acc063f726fef35 

    [76] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_191_0.png
    bd49e52417d85ed2455041ffde7673af 

    [77] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_192_0.png
    ab4c7d941547c32a2ed3f950004eedd0 

    [78] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_0.png
    7085086ed305de95830868468f133d80 

    [79] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_1.png
    9c1b09534ba48530257d1c742cca5727 

    [80] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_2.png
    09cc1e1dfa586debdeea2051ccfa733c 

    [81] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_3.png
    bb329997d01dbcc9d4a6b29e79de6244 

    [82] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_4.png
    8d96d809520cfb9b0f276007d65315eb 

    [83] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_5.png
    b44a470862390769a35bd9649a5562e1 

    [84] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_45_0.svg
    40f2f79959ecd005349e87a939145195 

    [85] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_48_0.png
    02e38ffe7f693212c55d6e4fffd4ba28 

    [86] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_1.png
    4bc6f74515bf1f9fd7d684431dd10100 

    [87] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_2.png
    5ea44f0c56c27a2d54dfb96e1b10a475 

    [88] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_3.png
    7bbcbf33ea79216c42be6ae5ee8087d2 

    [89] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_4.png
    12f0f23b5d57288605498f3e7bd5cf51 

    [90] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_57_0.png
    9f54d51b81f1a66fe412bec5a8026fa1 

    [91] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_58_0.png
    7d1ba4ad1b65c958c4561fef962f22dd 

    [92] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_61_0.png
    fcabb6625c63627f0d28cf2f8ec2142b 

    [93] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_64_0.png
    09d8cdcd99a6d9b17a06af37d3fff7fe 

    [94] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_66_0.png
    a0ee6838773cc3b630037fb6897b56eb 

    [95] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_67_0.png
    a8bcef1a055b07cc60148372b45c4c45 

    [96] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_72_0.svg
    3c558f19f00cb276153afc6cd0f3f6ac 

    [97] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_75_0.png
    4e249423920c6be4389a5536e7afb339 

    [98] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_1.png
    d4026fa9f30061e0f690cbd240764ae7 

    [99] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_2.png
    73ee65bda29495ed655cea0c5d8bbcef 

    [100] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_3.png
    f1c421385366ca5740c694ef064408b0 

    [101] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_4.png
    423a0f278f7aa21bce0f429813a0e354 

    [102] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_83_0.png
    44df44bcfe1f716dcb82d9846156a2f0 

    [103] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_84_0.png
    9e64a8acf444e3358a98cb05e0c82f4c 

    [104] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_87_0.png
    6053cc2ff6094dff1b0e275bc35fdb28 

    [105] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_90_0.png
    0d3a8afd37a51de50bfc0228b3e5e854 

    [106] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_92_0.png
    b8fe9f405bdb37a22c6dba21f61d6706 

    [107] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_93_0.png
    8b1e68dc9d112fc477627922ba313f71 

    [108] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_98_0.svg
    4585f0a092a89efd3171a48725856e35 

    [109] Python_analysis/code/Statistical_Model_TwoFactor.html
    4ea093f62cfaa9c14eaf2f1d6fb6ee09 

    [110] Python_analysis/code/Statistical_Model_TwoFactor.ipynb
    2e1975ca9fc414169cba4ffdf21537e4 

    [111] Python_analysis/code/Statistical_Model_TwoFactor.md
    07597fab2570370fc48d24b50fdd79a9 

    [112] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_1.png
    730034fdf48a56db85beadb0199be9f1 

    [113] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_2.png
    864e5d0045e2c7a0d119681bb29c7b58 

    [114] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_3.png
    55c8760bc38da3265e65ba82fa98f8d9 

    [115] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_4.png
    60aa979348de14037920fdf5244190e1 

    [116] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_108_0.png
    de987b62a95bf218930e48b242ccb2de 

    [117] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_109_0.png
    1004e48eb84277322de52fd31ba1d2df 

    [118] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_112_0.png
    444bc8b2c82c00c037683841963c64fc 

    [119] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_113_0.png
    02f271cfb23d86c70cfa744a13d681d4 

    [120] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_114_0.png
    24815c1f9875c8e34dea2e109c295a60 

    [121] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_117_0.png
    ebae425a377a37fb2faa9a7e686dcb8d 

    [122] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_119_0.png
    7fca6807545156267a06d5f4a30c7157 

    [123] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_120_0.png
    e706b8bf759f7cc90a1c5c3fb1a55e41 

    [124] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_127_0.svg
    192a6c49c9dde3abb04044d868f2b324 

    [125] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_130_0.png
    a1e01c7b53718b7cf5037fd325346796 

    [126] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_139_0.svg
    36b7a3f88438ab752d6e0d216a04a4a4 

    [127] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_142_0.png
    9c84a00ab1e712b67bb627d7fdf8f064 

    [128] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_1.png
    5091a9de009a0035a4e6bd32a7592f6d 

    [129] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_2.png
    68d6c6e162598bd8b7becfe59aceef77 

    [130] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_3.png
    8c0367f0588ded7bea3554f0430e708a 

    [131] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_4.png
    5a77df6034a9e01e056eb0f8c799fa6b 

    [132] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_151_0.png
    ee5857d7994e34af5ec8c621f8e378e6 

    [133] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_152_0.png
    a50e3ead50a8d49efb78e405effe5088 

    [134] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_155_0.png
    412fdd1bcfd5b8e75b347aac27f6555c 

    [135] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_156_0.png
    9cf4817e8d9fd481ce09926604cbfea5 

    [136] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_157_0.png
    531888fe87a5fe0dd14ee05020e03383 

    [137] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_160_0.png
    2361be7d70aff814c4cedd8873a30a6f 

    [138] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_162_0.png
    a9ec92e9579ed4387c9449b376fec0d0 

    [139] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_163_0.png
    93211249e62e2189731bfcffd13be5e2 

    [140] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_164_0.png
    e706b8bf759f7cc90a1c5c3fb1a55e41 

    [141] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_170_0.svg
    3fbbb91f74e6d78d1594db03155fcc7a 

    [142] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_171_0.svg
    3fbbb91f74e6d78d1594db03155fcc7a 

    [143] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_173_0.png
    7289cff0aa017b5091fcb5f44dee0a6b 

    [144] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_174_0.png
    7289cff0aa017b5091fcb5f44dee0a6b 

    [145] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_181_1.png
    05abb5591c5b1c8b6df4c61ba2d0c4d1 

    [146] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_181_2.png
    91a75bc96bd312286f12df775e31c2c6 

    [147] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_181_3.png
    a5cf47a1d6ff7b5b2f2b4939741ee7b4 

    [148] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_181_4.png
    76e3d9598b0e7564b97037d7dc5e413a 

    [149] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_0.png
    01c59b72e21a011d23b6471df8a01c35 

    [150] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_1.png
    05abb5591c5b1c8b6df4c61ba2d0c4d1 

    [151] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_2.png
    91a75bc96bd312286f12df775e31c2c6 

    [152] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_3.png
    a5cf47a1d6ff7b5b2f2b4939741ee7b4 

    [153] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_4.png
    76e3d9598b0e7564b97037d7dc5e413a 

    [154] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_183_0.png
    eef6473c37e4197676cecfd941550a8b 

    [155] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_184_0.png
    eef6473c37e4197676cecfd941550a8b 

    [156] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_186_0.png
    6b1e5fa007a553aa0668922282c03767 

    [157] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_187_0.png
    1895ada6f064e5b9d7ba2074c2dd09c6 

    [158] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_188_0.png
    1895ada6f064e5b9d7ba2074c2dd09c6 

    [159] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_190_0.png
    aac8aee7ce73537c387da2c99bc49ad5 

    [160] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_191_0.png
    be7bbf93882dc0e15e1d25b94d11b6d0 

    [161] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_192_0.png
    be7bbf93882dc0e15e1d25b94d11b6d0 

    [162] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_193_0.png
    d9f1b7da55acc0a8568ef7c16a59e280 

    [163] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_194_0.png
    811db30e453c23a639120f14bac1f251 

    [164] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_195_0.png
    811db30e453c23a639120f14bac1f251 

    [165] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_202_0.png
    4ea81314ffcd2e516f4c5af9b2794f5d 

    [166] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_203_0.png
    4ea81314ffcd2e516f4c5af9b2794f5d 

    [167] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_204_0.png
    4730a18002a7b2de3b8f1e154288e1d0 

    [168] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_205_0.png
    4730a18002a7b2de3b8f1e154288e1d0 

    [169] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_36_0.svg
    f28a373441cc0f4de9155896bbaec351 

    [170] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_39_0.png
    67250837bac4b48b8a319626716e32a6 

    [171] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_1.png
    116a4652e9a3f9ec661bf1073921dd72 

    [172] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_2.png
    f519dcf3fabcf7062fbb55c06b3dc191 

    [173] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_3.png
    3f3cd008a1ee38b3156ddc63ffd227a8 

    [174] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_4.png
    29162d9a83413ed5b3f18e81da1828aa 

    [175] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_50_0.png
    25be9a3341e5d25fde01607f947e8d0f 

    [176] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_51_0.png
    27054240f43530a3a69cb8aa5a908ca9 

    [177] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_54_0.png
    183288d44422280493f1273222a23749 

    [178] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_56_0.png
    1d984a5c6aa4c2f4fd22250e3758b5ea 

    [179] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_57_0.png
    79ae91775327f48e2dfdd30f90b375f4 

    [180] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_59_0.png
    b4d037d67812e0730ea130a51719bcb0 

    [181] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_66_0.svg
    3a69704d6e663470394bde94bfca2570 

    [182] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_69_0.png
    b95f0d8653d6d9fc116cc278bf2c1d28 

    [183] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_1.png
    ba96d91136f3bf4f52a4f192cb4605e3 

    [184] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_2.png
    b2e87ea6167487dbf6eb2fabe852bfa2 

    [185] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_3.png
    b692e8890bbd8fa18d421159ae17f817 

    [186] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_4.png
    074cc02a67702ecffe89526119b3ba6c 

    [187] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_77_0.png
    e1a31fa64db56f30631e577ecbc362c8 

    [188] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_78_0.png
    e94a3099c25106e7cecb7be7d2a73514 

    [189] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_81_0.png
    e53826a552d53ea91094f1783342f845 

    [190] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_84_0.png
    11c0db232fdfbb7f1f13514d6fca51e0 

    [191] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_85_0.png
    99816ed37f5d085bb638ded87793a0ea 

    [192] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_86_0.png
    1562640185861ae74e6366d495398e74 

    [193] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_88_0.png
    d9350aee8c5187c1220aa6aacb44259d 

    [194] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_89_0.png
    a91960977a9ec5560077312112ea1237 

    [195] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_96_0.svg
    0079aa65ad91b1e19f2d7f7a94fcaa2e 

    [196] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_99_0.png
    1ab478e8b966144c22c7065d2d807e3c 

    [197] Python_analysis/derived_data/preprocessing/preprocessed.dat
    c9eaaa1f93ecb2d341328d09bc983339 

    [198] Python_analysis/derived_data/statistical_model_neweplsar/hdi_NewEplsar.csv
    7e2d110b909aa763d4dc8b421ff1441f 

    [199] Python_analysis/derived_data/statistical_model_neweplsar/model_NewEplsar.pkl
    503fd7d0c6994b7917722b6b4c9ca8f8 

    [200] Python_analysis/derived_data/statistical_model_neweplsar/summary.csv
    c521ce7accd7e44b00bf57ef31388cc7 

    [201] Python_analysis/derived_data/statistical_model_three_factors/model_Asfc.pkl
    7602e4b3bb5c868121d53a026b763803 

    [202] Python_analysis/derived_data/statistical_model_three_factors/model_epLsar.pkl
    8febed3b0a20ea7f5cef5330583d6b5c 

    [203] Python_analysis/derived_data/statistical_model_three_factors/model_HAsfc81.pkl
    2fdfd35ef779c9f9b18b1400ad0617db 

    [204] Python_analysis/derived_data/statistical_model_three_factors/model_HAsfc9.pkl
    6c0d38712ed08704c7b3a5d4fa85ff79 

    [205] Python_analysis/derived_data/statistical_model_three_factors/model_R².pkl
    d93bd57ffdf9d520d60e17e5041f39dc 

    [206] Python_analysis/derived_data/statistical_model_two_factors/epLsar_oldb1.npy
    737a78722c98b0c7faf594bdf11be7b2 

    [207] Python_analysis/derived_data/statistical_model_two_factors/epLsar_oldb2.npy
    340215f91559437ef9b3482c2eaaecbe 

    [208] Python_analysis/derived_data/statistical_model_two_factors/epLsar_oldM12.npy
    cc3f180eae26fbed13cbe4a86f1cf8f8 

    [209] Python_analysis/derived_data/statistical_model_two_factors/hdi_Asfc.csv
    6ad7a158ee3cfed820fcf37cbd919875 

    [210] Python_analysis/derived_data/statistical_model_two_factors/hdi_epLsar.csv
    54849347dd3cb4ea5b7fd5f3898ed6c2 

    [211] Python_analysis/derived_data/statistical_model_two_factors/hdi_HAsfc81.csv
    c6a5158beb69e6d7182d1bba343d5fed 

    [212] Python_analysis/derived_data/statistical_model_two_factors/hdi_HAsfc9.csv
    4160b55e42fd5e138d035c1c7626c5d0 

    [213] Python_analysis/derived_data/statistical_model_two_factors/hdi_R².csv
    c5b30c9e2eb24d73ede6b25d5178e1da 

    [214] Python_analysis/derived_data/statistical_model_two_factors/model_Asfc.pkl
    12c01ec16790433f79042d5c4408cdc3 

    [215] Python_analysis/derived_data/statistical_model_two_factors/model_epLsar.pkl
    11011c02a1a3a3b5511f4c1400e0dd08 

    [216] Python_analysis/derived_data/statistical_model_two_factors/model_HAsfc81.pkl
    55fffab111f710d60b40e291d16b3eed 

    [217] Python_analysis/derived_data/statistical_model_two_factors/model_HAsfc9.pkl
    695aa9a7a8cc13bae45818740a032062 

    [218] Python_analysis/derived_data/statistical_model_two_factors/model_R².pkl
    f6353c1e2f401a3a2ef77c9a428407d3 

    [219] Python_analysis/derived_data/statistical_model_two_factors/summary.csv
    7679aa6201b888a8c542d73866fc5a9d 

    [220] Python_analysis/plots/statistical_model_neweplsar/posterior_b_NewEplsar.pdf
    05617ed15b44e0bc1ecb6f87edafeec7 

    [221] Python_analysis/plots/statistical_model_neweplsar/posterior_forest_NewEplsar.pdf
    eba4dc7a87c2aa69d050b599a56ff74a 

    [222] Python_analysis/plots/statistical_model_neweplsar/prior_posterior_predicitive_NewEplsar.pdf
    6fda126a665eb44a83fd1087bb004d70 

    [223] Python_analysis/plots/statistical_model_neweplsar/prior_predicitive_NewEplsar.pdf
    d0e7a4d9837ac48eb4229ec5a262dec6 

    [224] Python_analysis/plots/statistical_model_neweplsar/trace_NewEplsar.pdf
    7c7034d25254a9973a102272878f281f 

    [225] Python_analysis/plots/statistical_model_neweplsar/treatment_pairs_NewEplsar.pdf
    006e320f4360d1f6427857d2cbfc6ba2 

    [226] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_Asfc.pdf
    6bc2dcad7b0457a3f79a4b0c92ff289d 

    [227] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_epLsar.pdf
    f7eb02d654c16da62fa814d7cc798a40 

    [228] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_HAsfc81.pdf
    2f04fa9a868a0e4e5e1122540c3f7dd8 

    [229] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_HAsfc9.pdf
    e9ea1fadfe56d3bcc5e706e8b3fa7e0a 

    [230] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_R².pdf
    482c73a550a083a37b3bcecf52ae4b52 

    [231] Python_analysis/plots/statistical_model_three_factors/posterior_b_Asfc.pdf
    077170dd26c54977f9ab0186ed072595 

    [232] Python_analysis/plots/statistical_model_three_factors/posterior_b_epLsar.pdf
    5407de6f6e7ebd4373b189673f82f714 

    [233] Python_analysis/plots/statistical_model_three_factors/posterior_b_HAsfc81.pdf
    2857db606ea44d7aa86024ea82c11ff2 

    [234] Python_analysis/plots/statistical_model_three_factors/posterior_b_HAsfc9.pdf
    16a9a989cc68832a553bd23cdb656598 

    [235] Python_analysis/plots/statistical_model_three_factors/posterior_b_R².pdf
    1bcd5abc8fcb498f311ae7cad6fe1569 

    [236] Python_analysis/plots/statistical_model_three_factors/posterior_forest_Asfc.pdf
    accf9a1090b5db075ca39ce635357ed3 

    [237] Python_analysis/plots/statistical_model_three_factors/posterior_forest_epLsar.pdf
    e40913d7187b8a7b713ddff113ad84cd 

    [238] Python_analysis/plots/statistical_model_three_factors/posterior_forest_HAsfc81.pdf
    b745b4d8f2751f466d6c72e7b804b9b1 

    [239] Python_analysis/plots/statistical_model_three_factors/posterior_forest_HAsfc9.pdf
    aec67ed1fa2b36748bac588a3119001c 

    [240] Python_analysis/plots/statistical_model_three_factors/posterior_forest_R².pdf
    5720d3d7b803087d61af42a39d657938 

    [241] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_Asfc.pdf
    ffccf431bb30564fc5f38fe7f2af9785 

    [242] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_epLsar.pdf
    cab7b70953f68e1996b372a243dc6791 

    [243] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_HAsfc81.pdf
    a545e48aa9276ac40c3eb7770802bcdf 

    [244] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_HAsfc9.pdf
    8e0c5687f5860f56b80517fe87cd0d88 

    [245] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_R².pdf
    a14ce21687be4d52ed760a8dd35de06f 

    [246] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_Asfc.pdf
    0a990ed04c281d7fca61fd6f52a1d97d 

    [247] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_epLsar.pdf
    83320178b10efe1dfbb5e4bf312aeed3 

    [248] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_HAsfc81.pdf
    a29f0b77bfa6c6418b62d9d948e39da5 

    [249] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_HAsfc9.pdf
    47ffea61dc6a7e6ab1daf3d53ce1b27f 

    [250] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_R².pdf
    eee80c3e99f745e9ea38953088aefd9c 

    [251] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_Asfc.pdf
    e8a6591f6a31a3e7b0a10fecfeae1325 

    [252] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_epLsar.pdf
    f928054ea6af2ce12815f25633ea0ef7 

    [253] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_HAsfc81.pdf
    597e71ea347751d67af1b22392c66366 

    [254] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_HAsfc9.pdf
    52e46bc6ae690587399dacce94014ad7 

    [255] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_R².pdf
    f385508fe31c016b046d6b79ac01ffde 

    [256] Python_analysis/plots/statistical_model_three_factors/prior_posterior_Asfc_b1.pdf
    87c3b6fa09d04c75fcca1984ab184dec 

    [257] Python_analysis/plots/statistical_model_three_factors/prior_posterior_epLsar_b1.pdf
    550f8155afc6c1ffc37555fee95c244a 

    [258] Python_analysis/plots/statistical_model_three_factors/prior_posterior_HAsfc81_b1.pdf
    f00a0ff86adc22bbbee77193c96bc956 

    [259] Python_analysis/plots/statistical_model_three_factors/prior_posterior_HAsfc9_b1.pdf
    9309c67a06c0363e895093044f033b82 

    [260] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_Asfc.pdf
    9ce4345047ef3ca0540df5d03adeca75 

    [261] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_epLsar.pdf
    7026defe0642977988bfaa18fd88054f 

    [262] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_HAsfc81.pdf
    c2a536242f35e875f77653655f136a03 

    [263] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_HAsfc9.pdf
    8d6f1d98cb1275c640bc4f1bfe15af1b 

    [264] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_R².pdf
    70db7b402033819839f198da597805ed 

    [265] Python_analysis/plots/statistical_model_three_factors/prior_posterior_R²_b1.pdf
    8f77e0425fde17b7465fd0f942663bef 

    [266] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_Asfc.pdf
    66d764d3cb22ee36b7c5b8a78d6293da 

    [267] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_epLsar.pdf
    cdaa665dff47ba05cce6f80382db92cd 

    [268] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_HAsfc81.pdf
    30fa5e520aaa230c6c878872c70de639 

    [269] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_HAsfc9.pdf
    9ad7a56eb7b17d4c558d175d5ac50c15 

    [270] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_R².pdf
    a1aacfb8e2b13411028f29992115ea13 

    [271] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_Smfc.pdf
    ddbdd27e68f7969adf10cc19dad22546 

    [272] Python_analysis/plots/statistical_model_three_factors/trace_Asfc.pdf
    175ff9b201e36ecfecc40c2e708ed5e6 

    [273] Python_analysis/plots/statistical_model_three_factors/trace_epLsar.pdf
    3196f9b083efe4a81776d3d9151ede8d 

    [274] Python_analysis/plots/statistical_model_three_factors/trace_HAsfc81.pdf
    2e23e4e5a59881fee879f85af3635ce8 

    [275] Python_analysis/plots/statistical_model_three_factors/trace_HAsfc9.pdf
    e20bd23a68e4de9b29c20bcf120a010f 

    [276] Python_analysis/plots/statistical_model_three_factors/trace_R².pdf
    ed06312a36e1e88b790f3615e8f5cc88 

    [277] Python_analysis/plots/statistical_model_two_factors/posterior_b_Asfc.pdf
    57f5555c5e990e95168cb1bb233cb83f 

    [278] Python_analysis/plots/statistical_model_two_factors/posterior_b_HAsfc81.pdf
    ee354d82e5086f87c8fcb41a14cb540d 

    [279] Python_analysis/plots/statistical_model_two_factors/posterior_b_HAsfc9.pdf
    df8b6ca22ec9069ed40d440813b69068 

    [280] Python_analysis/plots/statistical_model_two_factors/posterior_b_R².pdf
    d1d31c2a58c278cdfb7df5d7b180fbbb 

    [281] Python_analysis/plots/statistical_model_two_factors/posterior_forest_Asfc.pdf
    7dfeced112052acb28011d1d5d65c8c3 

    [282] Python_analysis/plots/statistical_model_two_factors/posterior_forest_epLsar.pdf
    2a69ffc24cd6d05f1a10059f28d2b155 

    [283] Python_analysis/plots/statistical_model_two_factors/posterior_forest_HAsfc81.pdf
    020a62f3500b10d456e2a619df0ddfc4 

    [284] Python_analysis/plots/statistical_model_two_factors/posterior_forest_HAsfc9.pdf
    a0626b220d001aa0ed6850f45bcfc459 

    [285] Python_analysis/plots/statistical_model_two_factors/posterior_forest_R².pdf
    8ce603708b2cfcb1dae473f177a3615f 

    [286] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_Asfc.pdf
    74a72c5dfb97af8a6eb008bf8e546d84 

    [287] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_epLsar.pdf
    3b5d98ccf2046d7c1167caca0d6bf45f 

    [288] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_HAsfc81.pdf
    b2df2020a53964828e8c00278fe63010 

    [289] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_HAsfc9.pdf
    61ee2b0eaeee2ecb0c5074ebaa350844 

    [290] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_R².pdf
    97b1abbe73333fb89c27b2a8c727f991 

    [291] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_Asfc.pdf
    ce41b77040e53ef1e537e1b1ffece2bf 

    [292] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_epLsar.pdf
    a836006d1a489e59ee8387bca84ba897 

    [293] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_HAsfc81.pdf
    55777095b2f37a731e7bfbf63dafcf0e 

    [294] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_HAsfc9.pdf
    3ee8fb96b82be96bad6b77bef09cc74c 

    [295] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_R².pdf
    3eb681ff20744d1272b08ee7d81efa99 

    [296] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_Asfc.pdf
    30ce3822da17a11748c4dfd9998828d4 

    [297] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_epLsar.pdf
    411fa0aae5167e0b47e5a23f3c4b443e 

    [298] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_HAsfc81.pdf
    935099975df60971f3e3a9e33ed9ca30 

    [299] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_HAsfc9.pdf
    625a6c41faf90603865afb63a98ec8b2 

    [300] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_R².pdf
    b5d6f7df266e5048661de02cb7d929b6 

    [301] Python_analysis/plots/statistical_model_two_factors/prior_posterior_Asfc_b1.pdf
    a2c361d7b0e4471196a0fb6c5fa43864 

    [302] Python_analysis/plots/statistical_model_two_factors/prior_posterior_HAsfc81_b1.pdf
    99aed20d7d92266cf4e89f56e6a0bb68 

    [303] Python_analysis/plots/statistical_model_two_factors/prior_posterior_HAsfc9_b1.pdf
    df75d58421bd302d073acef0de6ed3a9 

    [304] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_Asfc.pdf
    5da31500dc237b6de6f5db0a4e188dc0 

    [305] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_epLsar.pdf
    1b851a028aa1f731a2b82a581f771b6b 

    [306] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_HAsfc81.pdf
    8c5fa8ef42aa7c2541ae61e85ec5fb0b 

    [307] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_HAsfc9.pdf
    15b197fd5e0140674189a5cf48f4ca5b 

    [308] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_R².pdf
    98de9a78158432ca4736c92ce9893432 

    [309] Python_analysis/plots/statistical_model_two_factors/prior_posterior_R²_b1.pdf
    af03b52b492e41b9da225d6583fa2a09 

    [310] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_Asfc.pdf
    8fad4db1927517cfd22342b5fa85111f 

    [311] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_epLsar.pdf
    665dd686970bbfc4475cc7975dfbe7c1 

    [312] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_HAsfc81.pdf
    6874e4b64fde807959a2920b3794fda9 

    [313] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_HAsfc9.pdf
    f0ef04550b6ff29c485f5c4e8216cf3a 

    [314] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_R².pdf
    bc29cd86643615d58ce64192dbdfa11e 

    [315] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_Smfc.pdf
    ab3d170797ca29b4356b616e1f005063 

    [316] Python_analysis/plots/statistical_model_two_factors/trace_Asfc.pdf
    ef3579357b6a6632ae20af03344e5e21 

    [317] Python_analysis/plots/statistical_model_two_factors/trace_epLsar.pdf
    3942e9430a0f5f15c6b1aab1f7b4c6bb 

    [318] Python_analysis/plots/statistical_model_two_factors/trace_HAsfc81.pdf
    49d853cd83104f0520e24e3d1a976762 

    [319] Python_analysis/plots/statistical_model_two_factors/trace_HAsfc9.pdf
    a9c8b7d264ec82eab63b6b9a3c03584e 

    [320] Python_analysis/plots/statistical_model_two_factors/trace_R².pdf
    42dcebb6b69141f155e387544b5dd9fe 

    [321] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_Asfc.pdf
    2c4617e648e649556d53cff8a2fc9a08 

    [322] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_epLsar.pdf
    33206f933bba1226398f30bde33a3c05 

    [323] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_HAsfc81.pdf
    37bee1ef79adf9c9c5375c655b62232c 

    [324] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_HAsfc9.pdf
    a811438bf5625a82a58b63a8a958c86a 

    [325] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_R².pdf
    6b29ac8f8c709818e135e20c97f0f0c2 

    [326] Python_analysis/requirements.txt
    29c0a7eb1ebc42a17ab8a84be5c70be6 

    [327] Python_analysis/RUN_DOCKER.md
    87fe28d39c1f5c8198c3f801359dc0a1 

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
    [1] forcats_0.5.0   stringr_1.4.0   dplyr_1.0.3     purrr_0.3.4    
    [5] readr_1.4.0     tidyr_1.1.2     tibble_3.0.5    ggplot2_3.3.3  
    [9] tidyverse_1.3.0

    loaded via a namespace (and not attached):
     [1] Rcpp_1.0.6        cellranger_1.1.0  pillar_1.4.7      compiler_4.0.3   
     [5] dbplyr_2.0.0      digest_0.6.27     lubridate_1.7.9.2 jsonlite_1.7.2   
     [9] evaluate_0.14     lifecycle_0.2.0   gtable_0.3.0      pkgconfig_2.0.3  
    [13] rlang_0.4.10      reprex_0.3.0      cli_2.2.0         rstudioapi_0.13  
    [17] DBI_1.1.1         yaml_2.2.1        haven_2.3.1       xfun_0.20        
    [21] withr_2.4.0       xml2_1.3.2        httr_1.4.2        knitr_1.30       
    [25] fs_1.5.0          hms_1.0.0         generics_0.1.0    vctrs_0.3.6      
    [29] grid_4.0.3        tidyselect_1.1.0  glue_1.4.2        R6_2.5.0         
    [33] fansi_0.4.2       readxl_1.3.1      rmarkdown_2.6     modelr_0.1.8     
    [37] magrittr_2.0.1    backports_1.2.1   scales_1.1.1      ellipsis_0.3.1   
    [41] htmltools_0.5.1.1 rvest_0.3.6       assertthat_0.2.1  colorspace_2.0-0 
    [45] renv_0.12.5       stringi_1.5.3     munsell_0.5.0     broom_0.7.3      
    [49] crayon_1.3.4     

RStudio version 1.4.1103.

------------------------------------------------------------------------

END OF SCRIPT
