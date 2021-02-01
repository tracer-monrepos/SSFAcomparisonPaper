Checksums
================
Ivan Calandra
2021-02-01 15:32:12

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
    d9b331642bfd0d192e4eff5808b2a30f 

    [4] R_analysis/plots/SSFA_GuineaPigs_plot_Smfc.pdf
    6dd874a394b14e4fb32da06b78fbe216 

    [5] R_analysis/plots/SSFA_Lithics_plot.pdf
    0cd70b6f51de5f4647e7637d93bfc21c 

    [6] R_analysis/plots/SSFA_Lithics_plot_Smfc.pdf
    d15e77beb1d19071bbd70bf9870e9236 

    [7] R_analysis/plots/SSFA_Sheeps_plot.pdf
    c1ae6475b30ab3eefcdc8e948ae7aa3e 

    [8] R_analysis/plots/SSFA_Sheeps_plot_Smfc.pdf
    6820be39c42d15cf5703023baed5893c 

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
    0501f089c0253cefaafff41470154a69 

    [16] R_analysis/scripts/SSFA_0_CreateRC.md
    248ae7ce39cdd50c69aa4a6391024aea 

    [17] R_analysis/scripts/SSFA_0_CreateRC.Rmd
    0af686b90f935a9c863704626fd9715e 

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
    96e694260630703d768ea0ef0c5a7ec3 

    [24] R_analysis/scripts/SSFA_2_Summary-stats.md
    63b445cbf6e2453f9d7727d7765a5c01 

    [25] R_analysis/scripts/SSFA_2_Summary-stats.Rmd
    ba69228f04f99e7429b21ca807a54d83 

    [26] R_analysis/scripts/SSFA_3_Plots.html
    8c79d06138e1a11d347d9741e8acbe72 

    [27] R_analysis/scripts/SSFA_3_Plots.md
    9ce4a35e75243284718e8e9b1f966bb7 

    [28] R_analysis/scripts/SSFA_3_Plots.Rmd
    56f7a870ea7d32624e047e7b2c72d6c6 

    [29] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-10-1.png
    00d816db5fd35a3372d61629220d0c0b 

    [30] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-11-1.png
    05d0e6f5805ee8ea1f3c1a3049708f1e 

    [31] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-12-1.png
    6b3aed3e2ee5bf04ea006e2e115ca2f2 

    [32] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-12-2.png
    ec7e7751b860b645c81b112649306503 

    [33] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-12-3.png
    729f9bdcdd32af8b811f0cf60479012e 

    [34] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-13-1.png
    d4971a4c0b4f5f1272b0963b1b2b8a7d 

    [35] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-13-2.png
    6a6f8cba978425d082caf6a489c84ea7 

    [36] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-13-3.png
    093a3ae46853ccf5653affe2ea283161 

    [37] R_analysis/scripts/SSFA_3_Plots_files/figure-gfm/unnamed-chunk-9-1.png
    1f1785fcdf2d98752418ffa49ce95bfa 

    [38] R_analysis/summary_stats/SSFA_summary_stats.xlsx
    de723c7e41e776035992c001caab11c6 

## Python analysis

``` r
check_Python <- list.files(path = "Python_analysis", full.names = TRUE, recursive = TRUE) %>% 
                md5sum()
```

The MD5 checksums of the files created during the Python analysis are:

    [1] Python_analysis/code/plotting_lib.py
    c05a4787dac1e76a298ea3975ac8bd90 

    [2] Python_analysis/code/Preprocessing.html
    f1f24365a0ffd5f26dc75ef0b7274805 

    [3] Python_analysis/code/Preprocessing.ipynb
    21326b6ba61c40a7ca1e62253efbfd85 

    [4] Python_analysis/code/Preprocessing.md
    341551e198a98791ac30c60ebaf04033 

    [5] Python_analysis/code/Preprocessing_files/Preprocessing_33_0.png
    89021367f68fb5cbba528011691d2b19 

    [6] Python_analysis/code/Preprocessing_files/Preprocessing_33_1.png
    4a4cd07784b529686956b83598a82499 

    [7] Python_analysis/code/Preprocessing_files/Preprocessing_33_2.png
    4b61170299c3e02582740b179eb6aff4 

    [8] Python_analysis/code/Preprocessing_files/Preprocessing_33_3.png
    ba1129eaa6c680e7af4fda04bec6e523 

    [9] Python_analysis/code/Preprocessing_files/Preprocessing_33_4.png
    05bfad1b3debafcffe1cc694a89fd2bf 

    [10] Python_analysis/code/Preprocessing_files/Preprocessing_33_5.png
    20732a5637038c03e4370b014415c6ba 

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
    5047819d2157fe8cfd1309fe554a1934 

    [18] Python_analysis/code/Statistical_Model_NewEplsar.ipynb
    af01184c5a59aae586a6f3e0348b2bfc 

    [19] Python_analysis/code/Statistical_Model_NewEplsar.md
    332a94b2b6a5981bb50bf6d8eebc89e5 

    [20] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_33_0.png
    d565ace50267423d909f7fe9c12d2909 

    [21] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_35_0.png
    a3bcbb687672d2996cc5fb6ebc011100 

    [22] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_37_0.png
    e949218c7a079bbfca1f4130cd5b61f4 

    [23] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_38_0.png
    e0ef6b9d2f75685aa6055a66072e7460 

    [24] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_39_0.png
    4615c20ea6607683f2049c4ced3d8020 

    [25] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_54_0.svg
    fff184a7c8c897e936c4f5663df2cf88 

    [26] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_57_0.png
    0281842b445201efef0a64c7a3cfd01f 

    [27] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_65_0.png
    0c1da8cdc5d56d1c28ed9b0ba8e130c4 

    [28] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_66_0.png
    77a2b8bf8cea38bf19964c1550624f07 

    [29] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_67_0.png
    a9aaa108fd4f31ea58129e34e6976079 

    [30] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_70_0.png
    e6e630a035101335a3cc8d8cb59d17d3 

    [31] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_73_0.png
    7dcaba7483c73308f67d382ae7e09248 

    [32] Python_analysis/code/Statistical_Model_NewEplsar_files/Statistical_Model_NewEplsar_78_0.png
    ec26645b9c3344f687cefb7a3e5f2acd 

    [33] Python_analysis/code/Statistical_Model_ThreeFactor.html
    49ef0096c890ee8a65fb4cb8a7e0b17a 

    [34] Python_analysis/code/Statistical_Model_ThreeFactor.ipynb
    524140488f2966caf85843221526a4f5 

    [35] Python_analysis/code/Statistical_Model_ThreeFactor.md
    9800b7208d1e56f6bf862a5c07e8aefc 

    [36] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_101_0.png
    93373468166e8d8bddf2b7256ba56b72 

    [37] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_1.png
    48458979a2ec2453cc41a587f1d67205 

    [38] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_2.png
    b37ae8a2562e605ddcd91303fd31abbe 

    [39] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_3.png
    cfb290897bbc3c4a007af4b123b60700 

    [40] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_4.png
    5cd3205ee932aca078aa29a1eb135140 

    [41] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_110_0.png
    ae4a5d4b6a5afaad56686d52fabd452a 

    [42] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_111_0.png
    fabceaae9b5c007d95a3552a81dd7085 

    [43] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_114_0.png
    894d0589f32f3dc4d23711d184c7ec86 

    [44] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_117_0.png
    4f5a191b790ee50bc4b73ed179db23c0 

    [45] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_119_0.png
    9ae56a60716feb07c47f5c1622dbdbac 

    [46] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_120_0.png
    81476d67ab0404fca113647b7c673378 

    [47] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_125_0.svg
    14de0f93137f317e06596f723d3308c7 

    [48] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_128_0.png
    be71cee9058bc9931189b5a0598a2426 

    [49] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_137_0.svg
    3da5ca7f4ba1dc3b85fe70f89350c898 

    [50] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_140_0.png
    23a776ab79fbc1cd1e01e1fc1d24462c 

    [51] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_1.png
    5d38da4e962fdf235760bcb269f432fd 

    [52] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_2.png
    867059774f7cb4c46c67e4d18d33036d 

    [53] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_3.png
    736f00cba9b30ec71e7f92da7e872b9e 

    [54] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_4.png
    09ecde1fd1ba47af93722bd680bc30be 

    [55] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_149_0.png
    00f974017b3ee7edc195344b1bfd200c 

    [56] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_150_0.png
    0b61064bc8a18987c334bc6eb7c0d415 

    [57] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_153_0.png
    2b7c30c588d2f79cc98cb6d29db8a8a0 

    [58] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_156_0.png
    3fd9eeda3209045d4e0dde00acb9087c 

    [59] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_158_0.png
    12d958a9b13dd3b2e95121c3a913d927 

    [60] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_159_0.png
    a9898a579ca3e4e5d3afa3e31c4373ef 

    [61] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_164_0.svg
    367c940b1aa96deaafabddd1a52b2d6d 

    [62] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_167_0.png
    cb83480128e21fc65c33c9594c6719e7 

    [63] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_1.png
    d59389e0ee20c0ab307a44280fdf88a5 

    [64] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_2.png
    f59cb36dec839a753a388a1e42456ece 

    [65] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_3.png
    b0de872b175fba453989f4f17b207495 

    [66] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_4.png
    1a72c5a2c8647131003baeabc5b42ddc 

    [67] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_176_0.png
    7943b3673de56bf97adb74bec632b4ce 

    [68] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_177_0.png
    0b0e0ef2a376f35fc62590cd5c0cde48 

    [69] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_180_0.png
    ea3ab0815a8b120dacd8e57da8cf6e9c 

    [70] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_183_0.png
    594300409222061e2ce3834ee86ca16f 

    [71] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_185_0.png
    e43412c9a55fd9524d7437c8143287fb 

    [72] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_186_0.png
    4e50a138b2bf6f64c3f22387cb48922f 

    [73] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_188_0.png
    0095bac5503a600bdfeee9fbe7af241b 

    [74] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_189_0.png
    8900c1cf6e5561b7ef94733a409c80e3 

    [75] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_190_0.png
    81476d67ab0404fca113647b7c673378 

    [76] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_191_0.png
    a9898a579ca3e4e5d3afa3e31c4373ef 

    [77] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_192_0.png
    4e50a138b2bf6f64c3f22387cb48922f 

    [78] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_0.png
    43afa0cb375655b615cb7e5eb92d30c6 

    [79] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_1.png
    313583d12821d1c14806a19d6fd02da0 

    [80] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_2.png
    79c7f6d8873cc01352da6c0d3a30708f 

    [81] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_3.png
    69f199a5e8a19b19d68a5d32dbbde125 

    [82] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_4.png
    0fe32be66585713885462ba1b7457dae 

    [83] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_5.png
    2b43cd7b87cf3d9c0a934248bf9eaa4a 

    [84] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_45_0.svg
    22d25010370647f748cf9577f4680425 

    [85] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_48_0.png
    01202a3f4a95270ef92c440f37f13ec3 

    [86] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_1.png
    d0698029e2adf83d8ba7f611382baf47 

    [87] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_2.png
    db0e8361c08969bc21eacb646d11b745 

    [88] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_3.png
    4bc125554caa4ba9444f6d25bbc74546 

    [89] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_4.png
    f5006df4429af13f97dfde4bb421bb4f 

    [90] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_57_0.png
    17bcfeaccbd64fc0c1fefcf7687d5a56 

    [91] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_58_0.png
    4114ffa06ae22d155e97a83b039a7270 

    [92] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_61_0.png
    c833202a8c67df30406a603a90f935fd 

    [93] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_64_0.png
    bb8ab3b8709a9d348956d2f034f1dd25 

    [94] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_66_0.png
    06363bcc71cc078806fc2def21a53b39 

    [95] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_67_0.png
    0095bac5503a600bdfeee9fbe7af241b 

    [96] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_72_0.svg
    6ff435a30be811f87396a93a57513bfc 

    [97] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_75_0.png
    c1175937cf6874df6c05bb25eea8d586 

    [98] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_1.png
    a02eab9c985c4adcd962d666b76219ec 

    [99] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_2.png
    42b91a055916cd06cd6da469d55bd752 

    [100] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_3.png
    617234317e891a0691d46ce5f861e6de 

    [101] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_4.png
    10aabe87e68117f95084cad10f360275 

    [102] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_83_0.png
    c7f66c3a14f58cd7cea0b4ce96659c06 

    [103] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_84_0.png
    7696ed16e0264397bb50b288560fda8e 

    [104] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_87_0.png
    bc65cd98b8ea3f91c95010500b094ae8 

    [105] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_90_0.png
    7213f381f0445ecdc5c43a65c31d8996 

    [106] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_92_0.png
    12d7e8b128316ac2b57ddb14be28d08b 

    [107] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_93_0.png
    8900c1cf6e5561b7ef94733a409c80e3 

    [108] Python_analysis/code/Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_98_0.svg
    57791e6397a58023428b7cb3d6f513e1 

    [109] Python_analysis/code/Statistical_Model_TwoFactor.html
    446ab283dccd29197fac5e2893129a5f 

    [110] Python_analysis/code/Statistical_Model_TwoFactor.ipynb
    6c341911a08d0676e3c4077719752911 

    [111] Python_analysis/code/Statistical_Model_TwoFactor.md
    816e5e2637256fcac44a9c22be6a68cb 

    [112] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_1.png
    4fd0627e36605b0376abf4554c5529f5 

    [113] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_2.png
    2d6c2d727c8dda61ff5b61f506ab504f 

    [114] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_3.png
    b4da5725751b4c1d8d76a4a2a8950cdb 

    [115] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_4.png
    5b592b22ec7030391536841a15fcb3e9 

    [116] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_108_0.png
    d92894d20dd157f30b4e87fdc8d76bb6 

    [117] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_109_0.png
    777e63a063c2ff4557acd3eb5f760109 

    [118] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_112_0.png
    c42a52d490868cf238f79d693992725b 

    [119] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_113_0.png
    1f2550ba351c49cd6c43cc5bc7ae9690 

    [120] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_114_0.png
    d0a3080b12a786a17f57aeaa18fff149 

    [121] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_117_0.png
    a519a363083839aa6ee23c072caecbb4 

    [122] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_119_0.png
    df07d2e2a5780301124bcb73d6e99063 

    [123] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_120_0.png
    f323319c321e68de5a9725c9e2ff95f8 

    [124] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_127_0.svg
    3ad0003d1eeeda959ca996b784a4c5b5 

    [125] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_130_0.png
    a6b5cafb2c6a84852352d495506dd59e 

    [126] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_139_0.svg
    078a93dd6f9370ea12fffb7220e0cf3b 

    [127] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_142_0.png
    89f8fc1e2507cb6c1e9fbf7e30ad39aa 

    [128] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_1.png
    f44e2c2155e379d12fb515a7f8c37041 

    [129] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_2.png
    162901c6d1e7a3316a107ac20edfa65c 

    [130] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_3.png
    9bccb4856c816e6d18568324af7412ef 

    [131] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_4.png
    29fb522b1d42e877dc8f9d93b9dd67ac 

    [132] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_151_0.png
    924c12348ba807b2cce56b5163bd406e 

    [133] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_152_0.png
    c34236d50f9919426d64228e382f7a84 

    [134] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_155_0.png
    c5fd53046377db4d25e947b152cdec33 

    [135] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_156_0.png
    c792b8d1db0c20087a140b7c9b5d4c24 

    [136] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_157_0.png
    cd871335d7224a65462f138550d24c53 

    [137] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_160_0.png
    94cefaa8183a827ba0e954c549eeb32d 

    [138] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_162_0.png
    28df8ed70f8aeb6ad4817426c3834c37 

    [139] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_163_0.png
    1945cdf4b304f235d44407b3b404f92b 

    [140] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_164_0.png
    e706b8bf759f7cc90a1c5c3fb1a55e41 

    [141] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_170_0.svg
    5da61a47c7c558b70fb1d6374f0f62d9 

    [142] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_171_0.svg
    3fbbb91f74e6d78d1594db03155fcc7a 

    [143] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_173_0.png
    533d0b1f71ca21c60515a94bcc42a3ee 

    [144] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_174_0.png
    7289cff0aa017b5091fcb5f44dee0a6b 

    [145] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_181_1.png
    856881ba1286b302007630c156ab4bb7 

    [146] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_181_2.png
    27d7a3b21677103e91d728ad4cdf4e62 

    [147] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_181_3.png
    849f0a34fed987f6f6e585dce8d5ee91 

    [148] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_181_4.png
    0c31ee518b5e4701456e6ff701ce141e 

    [149] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_0.png
    28e8a13d68b4a0cca222030f1be42b6b 

    [150] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_1.png
    05abb5591c5b1c8b6df4c61ba2d0c4d1 

    [151] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_2.png
    91a75bc96bd312286f12df775e31c2c6 

    [152] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_3.png
    a5cf47a1d6ff7b5b2f2b4939741ee7b4 

    [153] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_4.png
    76e3d9598b0e7564b97037d7dc5e413a 

    [154] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_183_0.png
    50f288322e7b451688bd1007284c2e40 

    [155] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_184_0.png
    eef6473c37e4197676cecfd941550a8b 

    [156] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_186_0.png
    715c7c27239bf1774425ba36833e67d1 

    [157] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_187_0.png
    4c649f9f5ac49215b1a25c29f7a45aa9 

    [158] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_188_0.png
    1895ada6f064e5b9d7ba2074c2dd09c6 

    [159] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_190_0.png
    1c7b4e299b72ebb668b5a3839720a362 

    [160] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_191_0.png
    5bb51928bf77b4cccd6e02b3ae9dbd58 

    [161] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_192_0.png
    be7bbf93882dc0e15e1d25b94d11b6d0 

    [162] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_193_0.png
    b4745825a5c4a82a811c1a8667580863 

    [163] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_194_0.png
    4789e3b60436d68f79f35caf20046c2a 

    [164] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_195_0.png
    811db30e453c23a639120f14bac1f251 

    [165] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_202_0.png
    f4333097af66c003984134af4b73f18f 

    [166] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_203_0.png
    4ea81314ffcd2e516f4c5af9b2794f5d 

    [167] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_204_0.png
    bdd1b8f0fcf7bd5646617800b9692b60 

    [168] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_205_0.png
    4730a18002a7b2de3b8f1e154288e1d0 

    [169] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_36_0.svg
    2002091251590d0ede792773e1592e12 

    [170] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_39_0.png
    e0d6ecbe7880de3ce1b539da6f7f78e7 

    [171] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_1.png
    e758e575679372d8519992981f98edc2 

    [172] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_2.png
    a9aa4d86c3703c260dde889ad17964fa 

    [173] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_3.png
    24e88fd2451f8e39d96f89d540e9111d 

    [174] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_4.png
    ad4fcf841b1f9300c9cbc450241d56aa 

    [175] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_50_0.png
    1974d67767cb8566662a810322d9d5fb 

    [176] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_51_0.png
    459c48f43a44909334c19e905f2220fd 

    [177] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_54_0.png
    056a9761e54ddd3444835aeb3c291979 

    [178] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_56_0.png
    78df30bdccaf71aff6156c1edaf0c569 

    [179] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_57_0.png
    864dda3aff0f22fdc87e64557aec0fd3 

    [180] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_59_0.png
    03ff421bcecb319cc4bb281754e83a61 

    [181] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_66_0.svg
    8fe6b030339b44a0e1a357519614e74d 

    [182] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_69_0.png
    a5367695c6a4ec45638028ecd418e2f7 

    [183] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_1.png
    e605772c60d8196e761579abb1e69516 

    [184] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_2.png
    c8ed8feb5a2475e054b1968fa89d5811 

    [185] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_3.png
    470d52e0fedb84981adad3b483461054 

    [186] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_4.png
    b8091b1074ff39b623c32820612135d2 

    [187] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_77_0.png
    950b2f96fcc5db8f0b0815e6b7e0c348 

    [188] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_78_0.png
    42aa4fd7763b9d22cf0908ffee3fa893 

    [189] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_81_0.png
    a211140747372501817d699ee28fe7ba 

    [190] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_84_0.png
    7dd94d63fa6e88faf3bbbbaad2165c7d 

    [191] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_85_0.png
    b339b03731b1f6b5262ec4fc37c1efdf 

    [192] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_86_0.png
    7afd39704acc9b2167cffc2f985cab63 

    [193] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_88_0.png
    c7ed5828621c031905256ba62b1939f1 

    [194] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_89_0.png
    a91960977a9ec5560077312112ea1237 

    [195] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_89_1.png
    eb49302195fd326e62abebc6caf7d9c9 

    [196] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_96_0.svg
    9ca7a4e3e76db2258023b36bb754ca81 

    [197] Python_analysis/code/Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_99_0.png
    367e3aafa8bd255fb2f33a1b5b818939 

    [198] Python_analysis/derived_data/preprocessing/preprocessed.dat
    98f2839d4e2d27da5b510a4b861b12fc 

    [199] Python_analysis/derived_data/statistical_model_neweplsar/hdi_NewEplsar.csv
    ae110bfadc06deeba14be715434c7acb 

    [200] Python_analysis/derived_data/statistical_model_neweplsar/model_NewEplsar.pkl
    c41d1c5e0e1a698cdec5de5ac3692d6e 

    [201] Python_analysis/derived_data/statistical_model_neweplsar/summary.csv
    da1a67e41060efefff83b3e4307e4fcc 

    [202] Python_analysis/derived_data/statistical_model_three_factors/model_Asfc.pkl
    351de1ef5f41f5427cce5dbb03a7c2b4 

    [203] Python_analysis/derived_data/statistical_model_three_factors/model_epLsar.pkl
    b8fbcd2f80171ea9481d9565f9768674 

    [204] Python_analysis/derived_data/statistical_model_three_factors/model_HAsfc81.pkl
    836298e10202cfe7ed0b1e76dafc9f95 

    [205] Python_analysis/derived_data/statistical_model_three_factors/model_HAsfc9.pkl
    495a04fcf889cbf8486d984d66add339 

    [206] Python_analysis/derived_data/statistical_model_three_factors/model_R².pkl
    c8efaea7aa08f1d29141d6472eaaa944 

    [207] Python_analysis/derived_data/statistical_model_two_factors/epLsar_oldb1.npy
    de4555692046cb8f1e66f8998bc22a42 

    [208] Python_analysis/derived_data/statistical_model_two_factors/epLsar_oldb2.npy
    b14c8c3dee54afb9965510e5b357e036 

    [209] Python_analysis/derived_data/statistical_model_two_factors/epLsar_oldM12.npy
    bb022d3d4bd1bcf8cfb8bfc5ef2b757d 

    [210] Python_analysis/derived_data/statistical_model_two_factors/hdi_Asfc.csv
    848d4ed4ddd79a3b691e3377bcc32e6f 

    [211] Python_analysis/derived_data/statistical_model_two_factors/hdi_epLsar.csv
    0abe646b9a79093ee7f0abd5357c19f6 

    [212] Python_analysis/derived_data/statistical_model_two_factors/hdi_HAsfc81.csv
    e895ed166fd2d6830acfb6db38684ecc 

    [213] Python_analysis/derived_data/statistical_model_two_factors/hdi_HAsfc9.csv
    c899d9c42e14f9c7c0152719363902eb 

    [214] Python_analysis/derived_data/statistical_model_two_factors/hdi_R².csv
    5aa9514b7a92d496440188f94f06ef3d 

    [215] Python_analysis/derived_data/statistical_model_two_factors/model_Asfc.pkl
    a5e6f63d2e4be146fd1c40f92d6b3bcb 

    [216] Python_analysis/derived_data/statistical_model_two_factors/model_epLsar.pkl
    e802602607d32f8faea66278cb01ff0f 

    [217] Python_analysis/derived_data/statistical_model_two_factors/model_HAsfc81.pkl
    a91534c420d17826e5965889249975c9 

    [218] Python_analysis/derived_data/statistical_model_two_factors/model_HAsfc9.pkl
    033b56270676ac18baad676b85719a66 

    [219] Python_analysis/derived_data/statistical_model_two_factors/model_R².pkl
    e072f271043d03d1a49cf8c417b46783 

    [220] Python_analysis/derived_data/statistical_model_two_factors/summary.csv
    8b13dabf4960dce96264526a0903ac46 

    [221] Python_analysis/plots/statistical_model_neweplsar/posterior_b_NewEplsar.pdf
    d0a6beb40d5edec5965fed1429a98a43 

    [222] Python_analysis/plots/statistical_model_neweplsar/posterior_forest_NewEplsar.pdf
    16a85623298160b9c0e9f2b77c8498f0 

    [223] Python_analysis/plots/statistical_model_neweplsar/prior_posterior_predicitive_NewEplsar.pdf
    e0669551f6f60998f3b865479ce22d48 

    [224] Python_analysis/plots/statistical_model_neweplsar/prior_predicitive_NewEplsar.pdf
    262d36e5d6e2c84683eea13718487cf3 

    [225] Python_analysis/plots/statistical_model_neweplsar/trace_NewEplsar.pdf
    7740257d34a35e163d8d81369058fdbf 

    [226] Python_analysis/plots/statistical_model_neweplsar/treatment_pairs_NewEplsar.pdf
    622e800235729d3e9edad6e84530d81b 

    [227] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_Asfc.pdf
    e1bd42d3ad98e0f37face30253355d1b 

    [228] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_epLsar.pdf
    9c0af50922a1d668bf3f895f5c9f5e45 

    [229] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_HAsfc81.pdf
    ce2dd3eeec58eeb0958bde992744c743 

    [230] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_HAsfc9.pdf
    313f989da596d7ff44a0ba16ab9d6dcf 

    [231] Python_analysis/plots/statistical_model_three_factors/contrast_ConfoMap_Toothfrax_R².pdf
    2c566881a8c61cebb89f119b50f546e5 

    [232] Python_analysis/plots/statistical_model_three_factors/posterior_b_Asfc.pdf
    4d5aa63c660493b65f88de847215bad6 

    [233] Python_analysis/plots/statistical_model_three_factors/posterior_b_epLsar.pdf
    6d9405c07264ff17720b558649dc55e0 

    [234] Python_analysis/plots/statistical_model_three_factors/posterior_b_HAsfc81.pdf
    0e589da62e05e88703eb6f55bdf7f97d 

    [235] Python_analysis/plots/statistical_model_three_factors/posterior_b_HAsfc9.pdf
    c249a41ecbf80b564c0b1b5b132c8569 

    [236] Python_analysis/plots/statistical_model_three_factors/posterior_b_R².pdf
    3d6adeca22dea30c442338847f10116e 

    [237] Python_analysis/plots/statistical_model_three_factors/posterior_forest_Asfc.pdf
    0acb31b9ebd7e186dea426955a3ff904 

    [238] Python_analysis/plots/statistical_model_three_factors/posterior_forest_epLsar.pdf
    bf03468ae975aa22943368340e573fbe 

    [239] Python_analysis/plots/statistical_model_three_factors/posterior_forest_HAsfc81.pdf
    0568a295161ced37efdf0f4cdd21fcce 

    [240] Python_analysis/plots/statistical_model_three_factors/posterior_forest_HAsfc9.pdf
    2074a36e5baf9683fbd97cd48d8fe121 

    [241] Python_analysis/plots/statistical_model_three_factors/posterior_forest_R².pdf
    2bbd3693bdddd34bf7310c61770830d0 

    [242] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_Asfc.pdf
    bbdcd8cea3b4800083cfd85e0bea34ed 

    [243] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_epLsar.pdf
    84afb4d9c0c7f25279391d7a7ed2f684 

    [244] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_HAsfc81.pdf
    2243860cea1ae4d220b64162918bb601 

    [245] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_HAsfc9.pdf
    1503a39c24dae97143d23c11c6248f33 

    [246] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b1_R².pdf
    ea6fc1462ca02b4605a3ab88ad719ccc 

    [247] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_Asfc.pdf
    d10c2161700d1f6ba162b5222b2e2edb 

    [248] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_epLsar.pdf
    f451fbaad9b79b0503658102f172d4d4 

    [249] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_HAsfc81.pdf
    615c2d21f2305110e151f7478cd541be 

    [250] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_HAsfc9.pdf
    41d117d9708233320158261a5eb0826f 

    [251] Python_analysis/plots/statistical_model_three_factors/posterior_pair_b2_R².pdf
    fd5a5f717f78b83aa1908ffaf2001070 

    [252] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_Asfc.pdf
    50ed9bf28f1e051b251cfd59d51fef03 

    [253] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_epLsar.pdf
    fdcb0d3e5e233ad8b35a5052c9e8fc91 

    [254] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_HAsfc81.pdf
    ed3695e1e3913636d416b05232af2f57 

    [255] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_HAsfc9.pdf
    180461985c1eeb4a48f2c40f7c911768 

    [256] Python_analysis/plots/statistical_model_three_factors/posterior_parallel_R².pdf
    171af86dde3445d74b58955111786044 

    [257] Python_analysis/plots/statistical_model_three_factors/prior_posterior_Asfc_b1.pdf
    67b299edc60bbd1ff81ed31b2242b72c 

    [258] Python_analysis/plots/statistical_model_three_factors/prior_posterior_epLsar_b1.pdf
    89b6ea7716c54e5fae7ed123cee3c905 

    [259] Python_analysis/plots/statistical_model_three_factors/prior_posterior_HAsfc81_b1.pdf
    a96b10cc0117b54040e19f614cd692da 

    [260] Python_analysis/plots/statistical_model_three_factors/prior_posterior_HAsfc9_b1.pdf
    e52ac5b856daa78d3eac89837698e5fe 

    [261] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_Asfc.pdf
    d46c413ea1fa775e8139c2a75e0c0d36 

    [262] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_epLsar.pdf
    b8d7b7e9af85140149cda9e3a508c77e 

    [263] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_HAsfc81.pdf
    2d6f0a1fa57166a3603ac369967cbee4 

    [264] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_HAsfc9.pdf
    af260265214b5afc2cdea0375c487ddd 

    [265] Python_analysis/plots/statistical_model_three_factors/prior_posterior_predicitive_R².pdf
    0c55418c143c009e23b84c7eee89e5b8 

    [266] Python_analysis/plots/statistical_model_three_factors/prior_posterior_R²_b1.pdf
    bcb53e3bc9bd9ddadc5c0e5a9ddb42b8 

    [267] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_Asfc.pdf
    0471feb9bba4069d344eb6aea8ed178d 

    [268] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_epLsar.pdf
    068c5b599be61708a784e23d0834ac25 

    [269] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_HAsfc81.pdf
    6970ac22ed2cd108f750230962a88755 

    [270] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_HAsfc9.pdf
    4065f61bb71123048c4b188e60bb9656 

    [271] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_R².pdf
    713f0ab29ed5ddd404cef2ac491a59bb 

    [272] Python_analysis/plots/statistical_model_three_factors/prior_predicitive_Smfc.pdf
    085f4fb338500cb7f8db40041364cdbf 

    [273] Python_analysis/plots/statistical_model_three_factors/trace_Asfc.pdf
    9836be04169910e9e531abd5cfb55733 

    [274] Python_analysis/plots/statistical_model_three_factors/trace_epLsar.pdf
    91b9749cf79636b16653c28e22eec597 

    [275] Python_analysis/plots/statistical_model_three_factors/trace_HAsfc81.pdf
    e50eacf5a06c07f2fc4bbc291506a58f 

    [276] Python_analysis/plots/statistical_model_three_factors/trace_HAsfc9.pdf
    dbd9e70d921619e540461018706694f2 

    [277] Python_analysis/plots/statistical_model_three_factors/trace_R².pdf
    11758dea3297b0fc830ec8e02e473331 

    [278] Python_analysis/plots/statistical_model_two_factors/posterior_b_Asfc.pdf
    9df6135c37115e87dbc4a4d33faa62d8 

    [279] Python_analysis/plots/statistical_model_two_factors/posterior_b_HAsfc81.pdf
    9d09d1f3417b4e8c0f374f9deacae74c 

    [280] Python_analysis/plots/statistical_model_two_factors/posterior_b_HAsfc9.pdf
    2655f4451c349a7cdeacab1d5fcd9af9 

    [281] Python_analysis/plots/statistical_model_two_factors/posterior_b_R².pdf
    cc863ec87018eb668fc10e34d54d1873 

    [282] Python_analysis/plots/statistical_model_two_factors/posterior_forest_Asfc.pdf
    623b3d0b110e69dc5a7c283e1673eb4f 

    [283] Python_analysis/plots/statistical_model_two_factors/posterior_forest_epLsar.pdf
    40f528acb6dd74252b6d6a0da1173ae8 

    [284] Python_analysis/plots/statistical_model_two_factors/posterior_forest_HAsfc81.pdf
    264df21b622cc956fef7a49f2e4cb546 

    [285] Python_analysis/plots/statistical_model_two_factors/posterior_forest_HAsfc9.pdf
    904cb932d81b40bd5e28794123eab511 

    [286] Python_analysis/plots/statistical_model_two_factors/posterior_forest_R².pdf
    c13eb8ab63ae031264f53c98f8a3fa5a 

    [287] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_Asfc.pdf
    a0eea9c853c56d68270da28573aff919 

    [288] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_epLsar.pdf
    729fcb7a6672fdbecaaa67f6426222f7 

    [289] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_HAsfc81.pdf
    7b1d746e1616dd3693d714978e63a448 

    [290] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_HAsfc9.pdf
    7d5043faf9b30e2bc7d616b98fefe080 

    [291] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b1_R².pdf
    72770fc88a69e095fd57699c13fdabda 

    [292] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_Asfc.pdf
    88f53d256a7f28d8257648916f6e5e09 

    [293] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_epLsar.pdf
    260ff32d99e4bcad942157d0ea5d1d6a 

    [294] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_HAsfc81.pdf
    927d8080aa1c5b14d2e5071ef10a3a1a 

    [295] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_HAsfc9.pdf
    3fb2db786be23f8525510bb695a7b652 

    [296] Python_analysis/plots/statistical_model_two_factors/posterior_pair_b2_R².pdf
    2854fb9f739a99626abcf58f969e677c 

    [297] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_Asfc.pdf
    049e1f9946b4c871ba3653c403546914 

    [298] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_epLsar.pdf
    08a85eb71742ac8a9dc02fba1854f898 

    [299] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_HAsfc81.pdf
    cd4ef036a32ff6e315b50ff3215e30c9 

    [300] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_HAsfc9.pdf
    de8df856072490d2976e17cf55754df8 

    [301] Python_analysis/plots/statistical_model_two_factors/posterior_parallel_R².pdf
    3994a46f7a895fbf45a7b9965a8b583a 

    [302] Python_analysis/plots/statistical_model_two_factors/prior_posterior_Asfc_b1.pdf
    d888a9f2bb2386a0fe246a11a3205a88 

    [303] Python_analysis/plots/statistical_model_two_factors/prior_posterior_HAsfc81_b1.pdf
    ab3d75e623452e0b90b1b62efc95010a 

    [304] Python_analysis/plots/statistical_model_two_factors/prior_posterior_HAsfc9_b1.pdf
    993ddc6116e532f1c6c3225bdc16e5d1 

    [305] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_Asfc.pdf
    f17c93f85b0d7ad60ba36ce66306f06f 

    [306] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_epLsar.pdf
    9cc523136a7c4f92aede7e7d99f18a6a 

    [307] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_HAsfc81.pdf
    3c480ef23d2949c7799494b46c19053b 

    [308] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_HAsfc9.pdf
    bba421ca85d7d88ef684d1d59e76f063 

    [309] Python_analysis/plots/statistical_model_two_factors/prior_posterior_predicitive_R².pdf
    5ef1bf660b581b74e324418f47c01a98 

    [310] Python_analysis/plots/statistical_model_two_factors/prior_posterior_R²_b1.pdf
    a4fd0a8ccfc562e85699d144071609fe 

    [311] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_Asfc.pdf
    37483f673b04fe3e983be7c177264cb4 

    [312] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_epLsar.pdf
    eaee7ce372c0678469d86b697319e871 

    [313] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_HAsfc81.pdf
    b1bf319eff5160b1c44e3e61b434bc23 

    [314] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_HAsfc9.pdf
    e03f3beddd6139d17f8d98f275a2f41b 

    [315] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_R².pdf
    69253dab85b1206af09ee0db7146659d 

    [316] Python_analysis/plots/statistical_model_two_factors/prior_predicitive_Smfc.pdf
    af4f24516e43525582df48a4039ed0b5 

    [317] Python_analysis/plots/statistical_model_two_factors/trace_Asfc.pdf
    e92b1c77454d8093c3a0c85fe2cf8f2f 

    [318] Python_analysis/plots/statistical_model_two_factors/trace_epLsar.pdf
    31c5970fb1987f4071507720741d2206 

    [319] Python_analysis/plots/statistical_model_two_factors/trace_HAsfc81.pdf
    91d4d61665e319e7eee1157fd48c02f5 

    [320] Python_analysis/plots/statistical_model_two_factors/trace_HAsfc9.pdf
    d7d841b086d3d40168b93b80cff75b4b 

    [321] Python_analysis/plots/statistical_model_two_factors/trace_R².pdf
    22e05da4504ff2c5016968f768599648 

    [322] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_Asfc.pdf
    f4f22c377287e488950e338e135c8c60 

    [323] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_epLsar.pdf
    e805f3a29ee6d1f9022ba6a053eb8a6e 

    [324] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_HAsfc81.pdf
    b778c1eb0304750b755a390e750e426c 

    [325] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_HAsfc9.pdf
    ff9d4aaf0f06415ec25c751c4ab3ffad 

    [326] Python_analysis/plots/statistical_model_two_factors/treatment_pairs_R².pdf
    6af26ab351fc32907821b86cac297251 

    [327] Python_analysis/requirements.txt
    29c0a7eb1ebc42a17ab8a84be5c70be6 

    [328] Python_analysis/RUN_DOCKER.md
    92a91b9310eb72122b471b659b371742 

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
