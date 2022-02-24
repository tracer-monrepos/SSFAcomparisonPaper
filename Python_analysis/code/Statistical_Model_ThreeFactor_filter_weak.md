# Analysis for SSFA project: Three factor  model
# Filtered weakly by < 20% NMP

## Table of contents
1. [Used packages](#imports)
1. [Global settings](#settings)
1. [Load data](#load)
1. [Explore data](#exploration)
1. [Model specification](#model)
1. [Inference](#inference)
   1. [epLsar](#epLsar)
   1. [R²](#r)
   1. [Asfc](#Asfc)
   1. [Smfc](#Smfc)
   1. [HAsfc9](#HAsfc9)
1. [Summary](#summary)

## Used packages <a name="imports"></a>


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle
import arviz as az
import pymc3 as pm
```


```python
from matplotlib.colors import to_rgb
```


```python
import scipy.stats as stats 
```


```python
from IPython.display import display
```


```python
import matplotlib as mpl
```


```python
%load_ext autoreload
%autoreload 2
```


```python
import plotting_lib
```

## Global settings <a name="settings"></a>

#### Output


```python
writeOut = True
outPathPlots = "../plots/statistical_model_three_factors_filter_weak/"
outPathData = "../derived_data/statistical_model_three_factors_filter_weak/"
```

#### Plotting


```python
widthMM = 190 
widthInch = widthMM / 25.4
ratio = 0.66666
heigthInch = ratio*widthInch

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
sns.set_style("ticks")

dpi = 300
```


```python
sizes = [SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE]
```

#### Computing


```python
numSamples = 1000
numCores = 4
numTune = 1000
numPredSamples = 2000
random_seed=36514535
target_accept = 0.99
```

## Load data <a name="load"></a>


```python
datafile = "../derived_data/preprocessing/preprocessed_filter_weak.dat"
```


```python
with open(datafile, "rb") as f:
    x1,x2,x3,df,dataZ,dictMeanStd,dictTreatment,dictSoftware = pickle.load(f)    
```

Show that everything is correct:


```python
display(pd.DataFrame.from_dict({'x1':x1,'x2':x2,'x3':x3}))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>115</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5</td>
      <td>116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>116</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>117</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>273</th>
      <td>1</td>
      <td>9</td>
      <td>52</td>
    </tr>
    <tr>
      <th>274</th>
      <td>0</td>
      <td>9</td>
      <td>53</td>
    </tr>
    <tr>
      <th>275</th>
      <td>1</td>
      <td>9</td>
      <td>53</td>
    </tr>
    <tr>
      <th>276</th>
      <td>0</td>
      <td>9</td>
      <td>54</td>
    </tr>
    <tr>
      <th>277</th>
      <td>1</td>
      <td>9</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
<p>278 rows × 3 columns</p>
</div>


x1 indicates the software used, x2 indicates the treatment applied and x3 the sample.


```python
for surfaceParam,(mean,std) in dictMeanStd.items():
    print("Surface parameter {} has mean {} and standard deviation {}".format(surfaceParam,mean,std))
```

    Surface parameter epLsar has mean 0.0032388209205305753 and standard deviation 0.0019378273835719989
    Surface parameter Rsquared has mean 0.9974096825435252 and standard deviation 0.007283582118542012
    Surface parameter Asfc has mean 14.919474245449283 and standard deviation 12.47068676838922
    Surface parameter Smfc has mean 1.155270960424856 and standard deviation 7.13503174525663
    Surface parameter HAsfc9 has mean 0.44593694325514915 and standard deviation 0.7912033512620836
    Surface parameter HAsfc81 has mean 0.9300206156734742 and standard deviation 2.3638534390774013



```python
for k,v in sorted(dictTreatment.items(), key=lambda x: x[0]):    
    print("Number {} encodes treatment {}".format(k,v))
```

    Number 0 encodes treatment BrushDirt
    Number 1 encodes treatment BrushNoDirt
    Number 2 encodes treatment Clover
    Number 3 encodes treatment Clover+dust
    Number 4 encodes treatment Control
    Number 5 encodes treatment Dry bamboo
    Number 6 encodes treatment Dry grass
    Number 7 encodes treatment Dry lucerne
    Number 8 encodes treatment Grass
    Number 9 encodes treatment Grass+dust
    Number 10 encodes treatment RubDirt



```python
for k,v in dictSoftware.items():
    print("Number {} encodes software {}".format(k,v))
```

    Number 0 encodes software ConfoMap
    Number 1 encodes software Toothfrax



```python
display(dataZ)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>TreatmentNumber</th>
      <th>SoftwareNumber</th>
      <th>DatasetNumber</th>
      <th>NameNumber</th>
      <th>epLsar_z</th>
      <th>Rsquared_z</th>
      <th>Asfc_z</th>
      <th>Smfc_z</th>
      <th>HAsfc9_z</th>
      <th>HAsfc81_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>115</td>
      <td>0.608031</td>
      <td>0.081494</td>
      <td>-0.261684</td>
      <td>-0.120632</td>
      <td>-0.391977</td>
      <td>-0.239736</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>115</td>
      <td>0.764866</td>
      <td>0.295228</td>
      <td>-0.368764</td>
      <td>-0.145206</td>
      <td>-0.392397</td>
      <td>-0.240365</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>116</td>
      <td>1.355641</td>
      <td>-0.166422</td>
      <td>0.043912</td>
      <td>-0.120632</td>
      <td>-0.346351</td>
      <td>-0.268091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>116</td>
      <td>1.350574</td>
      <td>0.282460</td>
      <td>-0.137943</td>
      <td>-0.145206</td>
      <td>-0.349727</td>
      <td>-0.282929</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
      <td>0.930308</td>
      <td>-0.359987</td>
      <td>-0.137793</td>
      <td>-0.120632</td>
      <td>-0.233444</td>
      <td>-0.221925</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>273</th>
      <td>273</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>52</td>
      <td>0.611602</td>
      <td>0.267769</td>
      <td>-1.050437</td>
      <td>0.035516</td>
      <td>0.493708</td>
      <td>0.076860</td>
    </tr>
    <tr>
      <th>274</th>
      <td>274</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>53</td>
      <td>0.084569</td>
      <td>0.197735</td>
      <td>-0.966638</td>
      <td>-0.093723</td>
      <td>0.242115</td>
      <td>0.257597</td>
    </tr>
    <tr>
      <th>275</th>
      <td>275</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>53</td>
      <td>-0.051512</td>
      <td>0.319804</td>
      <td>-0.975181</td>
      <td>-0.143224</td>
      <td>0.644288</td>
      <td>0.381453</td>
    </tr>
    <tr>
      <th>276</th>
      <td>276</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>54</td>
      <td>-1.041990</td>
      <td>0.284041</td>
      <td>-1.077552</td>
      <td>0.011489</td>
      <td>-0.095103</td>
      <td>-0.053253</td>
    </tr>
    <tr>
      <th>277</th>
      <td>277</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>54</td>
      <td>-1.308590</td>
      <td>0.336005</td>
      <td>-1.081522</td>
      <td>-0.104672</td>
      <td>-0.126169</td>
      <td>-0.038105</td>
    </tr>
  </tbody>
</table>
<p>278 rows × 11 columns</p>
</div>



```python
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset</th>
      <th>Name</th>
      <th>Software</th>
      <th>Diet</th>
      <th>Treatment</th>
      <th>Before.after</th>
      <th>NMP</th>
      <th>NMP_cat</th>
      <th>epLsar</th>
      <th>Rsquared</th>
      <th>Asfc</th>
      <th>Smfc</th>
      <th>HAsfc9</th>
      <th>HAsfc81</th>
      <th>NewEplsar</th>
      <th>TreatmentNumber</th>
      <th>SoftwareNumber</th>
      <th>DatasetNumber</th>
      <th>NameNumber</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_1</td>
      <td>ConfoMap</td>
      <td>Dry bamboo</td>
      <td>Dry bamboo</td>
      <td>NaN</td>
      <td>0.717312</td>
      <td>0-5%</td>
      <td>0.004417</td>
      <td>0.998003</td>
      <td>11.656095</td>
      <td>0.294557</td>
      <td>0.135803</td>
      <td>0.363319</td>
      <td>0.019460</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>115</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_1</td>
      <td>Toothfrax</td>
      <td>Dry bamboo</td>
      <td>Dry bamboo</td>
      <td>NaN</td>
      <td>0.717312</td>
      <td>0-5%</td>
      <td>0.004721</td>
      <td>0.999560</td>
      <td>10.320730</td>
      <td>0.119219</td>
      <td>0.135471</td>
      <td>0.361833</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_2</td>
      <td>ConfoMap</td>
      <td>Dry bamboo</td>
      <td>Dry bamboo</td>
      <td>NaN</td>
      <td>1.674215</td>
      <td>0-5%</td>
      <td>0.005866</td>
      <td>0.996198</td>
      <td>15.467083</td>
      <td>0.294557</td>
      <td>0.171903</td>
      <td>0.296292</td>
      <td>0.020079</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_2</td>
      <td>Toothfrax</td>
      <td>Dry bamboo</td>
      <td>Dry bamboo</td>
      <td>NaN</td>
      <td>1.674215</td>
      <td>0-5%</td>
      <td>0.005856</td>
      <td>0.999467</td>
      <td>13.199232</td>
      <td>0.119219</td>
      <td>0.169232</td>
      <td>0.261217</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>116</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_3</td>
      <td>ConfoMap</td>
      <td>Dry bamboo</td>
      <td>Dry bamboo</td>
      <td>NaN</td>
      <td>1.760409</td>
      <td>0-5%</td>
      <td>0.005042</td>
      <td>0.994788</td>
      <td>13.201101</td>
      <td>0.294557</td>
      <td>0.261235</td>
      <td>0.405422</td>
      <td>0.019722</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>273</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90730-lm2sin-a</td>
      <td>Toothfrax</td>
      <td>Grass+dust</td>
      <td>Grass+dust</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0-5%</td>
      <td>0.004424</td>
      <td>0.999360</td>
      <td>1.819802</td>
      <td>1.408678</td>
      <td>0.836560</td>
      <td>1.111706</td>
      <td>NaN</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>52</td>
    </tr>
    <tr>
      <th>274</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90764-lm2sin-a</td>
      <td>ConfoMap</td>
      <td>Grass+dust</td>
      <td>Grass+dust</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0-5%</td>
      <td>0.003403</td>
      <td>0.998850</td>
      <td>2.864831</td>
      <td>0.486556</td>
      <td>0.637499</td>
      <td>1.538943</td>
      <td>0.018978</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>53</td>
    </tr>
    <tr>
      <th>275</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90764-lm2sin-a</td>
      <td>Toothfrax</td>
      <td>Grass+dust</td>
      <td>Grass+dust</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0-5%</td>
      <td>0.003139</td>
      <td>0.999739</td>
      <td>2.758297</td>
      <td>0.133366</td>
      <td>0.955699</td>
      <td>1.831721</td>
      <td>NaN</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>53</td>
    </tr>
    <tr>
      <th>276</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90814-lm2sin-a</td>
      <td>ConfoMap</td>
      <td>Grass+dust</td>
      <td>Grass+dust</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0-5%</td>
      <td>0.001220</td>
      <td>0.999479</td>
      <td>1.481662</td>
      <td>1.237247</td>
      <td>0.370691</td>
      <td>0.804138</td>
      <td>0.017498</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>54</td>
    </tr>
    <tr>
      <th>277</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90814-lm2sin-a</td>
      <td>Toothfrax</td>
      <td>Grass+dust</td>
      <td>Grass+dust</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0-5%</td>
      <td>0.000703</td>
      <td>0.999857</td>
      <td>1.432148</td>
      <td>0.408433</td>
      <td>0.346111</td>
      <td>0.839946</td>
      <td>NaN</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
<p>278 rows × 19 columns</p>
</div>


#### Contrasts
Prepare dicts for contrasts that are of interest later.


```python
x1contrast_dict = {'{}_{}'.format(dictSoftware[0],dictSoftware[1]):[1,-1]}
x1contrast_dict
```




    {'ConfoMap_Toothfrax': [1, -1]}



## Explore data <a name="exploration"></a>

#### Compute raw differences


```python
dfRawDiff = dataZ.groupby(['DatasetNumber','TreatmentNumber','NameNumber']).agg(list).applymap(lambda l : l[0]-l[1]).reset_index()
```


```python
dfRawDiff
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DatasetNumber</th>
      <th>TreatmentNumber</th>
      <th>NameNumber</th>
      <th>index</th>
      <th>SoftwareNumber</th>
      <th>epLsar_z</th>
      <th>Rsquared_z</th>
      <th>Asfc_z</th>
      <th>Smfc_z</th>
      <th>HAsfc9_z</th>
      <th>HAsfc81_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>115</td>
      <td>-1</td>
      <td>-1</td>
      <td>-0.156835</td>
      <td>-0.213734</td>
      <td>0.107080</td>
      <td>0.024574</td>
      <td>0.000420</td>
      <td>0.000629</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>5</td>
      <td>116</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.005067</td>
      <td>-0.448881</td>
      <td>0.181855</td>
      <td>0.024574</td>
      <td>0.003376</td>
      <td>0.014838</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5</td>
      <td>117</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.137060</td>
      <td>-0.615262</td>
      <td>0.171673</td>
      <td>0.024574</td>
      <td>0.055406</td>
      <td>0.020671</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>5</td>
      <td>118</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.363489</td>
      <td>-0.380505</td>
      <td>0.206489</td>
      <td>0.029593</td>
      <td>0.018505</td>
      <td>-0.035897</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>119</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.135927</td>
      <td>-0.441161</td>
      <td>0.431840</td>
      <td>0.029593</td>
      <td>0.051468</td>
      <td>0.035179</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>134</th>
      <td>2</td>
      <td>9</td>
      <td>50</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.565233</td>
      <td>-0.166711</td>
      <td>0.141982</td>
      <td>0.016303</td>
      <td>0.055662</td>
      <td>0.046721</td>
    </tr>
    <tr>
      <th>135</th>
      <td>2</td>
      <td>9</td>
      <td>51</td>
      <td>-1</td>
      <td>-1</td>
      <td>1.694319</td>
      <td>-0.156497</td>
      <td>0.021729</td>
      <td>0.016303</td>
      <td>0.446416</td>
      <td>0.141461</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2</td>
      <td>9</td>
      <td>52</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2.109816</td>
      <td>-0.348137</td>
      <td>0.000667</td>
      <td>0.390193</td>
      <td>-0.010465</td>
      <td>0.004987</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2</td>
      <td>9</td>
      <td>53</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.136080</td>
      <td>-0.122069</td>
      <td>0.008543</td>
      <td>0.049501</td>
      <td>-0.402172</td>
      <td>-0.123856</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2</td>
      <td>9</td>
      <td>54</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.266600</td>
      <td>-0.051963</td>
      <td>0.003970</td>
      <td>0.116161</td>
      <td>0.031066</td>
      <td>-0.015148</td>
    </tr>
  </tbody>
</table>
<p>139 rows × 11 columns</p>
</div>



#### Show difference between software in raw data


```python
variablesList = ['epLsar','Rsquared','Asfc','Smfc','HAsfc9']
```


```python
for var in variablesList:
    fig, axes = plt.subplots(1, 2,figsize=(12,6))
    fig.suptitle('{}'.format(var))

    
    sns.stripplot(data=dataZ,x='NameNumber',y='{}_z'.format(var),hue='SoftwareNumber',ax=axes[0]);
    axes[0].set_title('Raw data')

    sns.histplot(data=dfRawDiff,x='{}_z'.format(var),ax=axes[1]);
    axes[1].set_title('Aggegrated difference by software'.format(var))

    plt.show()
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_37_0.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_37_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_37_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_37_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_37_4.png)
    


## Model specification <a name="model"></a>


```python
class ThreeFactorModel(pm.Model):
    
    """
    Compute params of priors and hyperpriors.
    """
    def getParams(self,x1,x2,x3,y):
        # get lengths
        Nx1Lvl = np.unique(x1).size
        Nx2Lvl = np.unique(x2).size
        Nx3Lvl = np.unique(x3).size        
        
        dims = (Nx1Lvl, Nx2Lvl, Nx3Lvl)
        
        ### get standard deviations
        
        # convert to pandas dataframe to use their logic
        df = pd.DataFrame.from_dict({'x1':x1,'x2':x2,'x3':x3,'y':y})
        
        s1 = df.groupby('x1').std()['y'].max()
        s2 = df.groupby('x2').std()['y'].max()
        s3 = df.groupby('x3').std()['y'].max()
        
        stdSingle = (s1, s2, s3)
        
        return (dims, stdSingle)
    
    def printParams(self,x1,x2,x3,y):
        dims, stdSingle = self.getParams(x1,x2,x3,y)
        Nx1Lvl, Nx2Lvl, Nx3Lvl = dims
        s1, s2, s3 = stdSingle        
        
        print("The number of levels of the x variables are {}".format(dims))
        print("The standard deviations used for the beta priors are {}".format(stdSingle))
            
    def __init__(self,name,x1,x2,x3,y,model=None):
        
        # call super's init first, passing model and name
        super().__init__(name, model)
        
        # get parameter of hyperpriors
        dims, stdSingle = self.getParams(x1,x2,x3,y)
        Nx1Lvl, Nx2Lvl, Nx3Lvl = dims
        s1, s2, s3 = stdSingle        
        
        ### hyperpriors ### 
        # observation hyperpriors
        lamY = 1/30.
        muGamma = 0.5
        sigmaGamma = 2.
        
        # prediction hyperpriors
        sigma0 = pm.HalfNormal('sigma0',sd=1)
        sigma1 = pm.HalfNormal('sigma1',sd=s1, shape=Nx1Lvl)
        sigma2 = pm.HalfNormal('sigma2',sd=s2, shape=Nx2Lvl)
        sigma3 = pm.HalfNormal('sigma3',sd=s3, shape=Nx3Lvl)
                
        mu_b0 = pm.Normal('mu_b0', mu=0., sd=1)
        mu_b1 = pm.Normal('mu_b1', mu=0., sd=1, shape=Nx1Lvl)
        mu_b2 = pm.Normal('mu_b2', mu=0., sd=1, shape=Nx2Lvl)
        mu_b3 = pm.Normal('mu_b3', mu=0., sd=1, shape=Nx3Lvl)       
                                       
        ### priors ### 
        # observation priors        
        nuY = pm.Exponential('nuY',lam=lamY)
        sigmaY = pm.Gamma('sigmaY',mu=muGamma, sigma=sigmaGamma)
        
        # prediction priors
        b0_dist = pm.Normal('b0_dist', mu=0, sd=1)
        b0 = pm.Deterministic("b0", mu_b0 + b0_dist * sigma0)
       
        b1_dist = pm.Normal('b1_dist', mu=0, sd=1)
        b1 = pm.Deterministic("b1", mu_b1 + b1_dist * sigma1)
        
        b2_dist = pm.Normal('b2_dist', mu=0, sd=1)
        b2 = pm.Deterministic("b2", mu_b2 + b2_dist * sigma2)
        
        b3_dist = pm.Normal('b3_dist', mu=0, sd=1)
        b3 = pm.Deterministic("b3", mu_b3 + b3_dist * sigma3)
             
        #### prediction ###      
        mu = pm.Deterministic('mu',b0 + b1[x1] + b2[x2] +  b3[x3])

                                
        ### observation ### 
        y = pm.StudentT('y',nu = nuY, mu=mu, sd=sigmaY, observed=y)
```

## Inference <a name="inference"></a>

### epLsar <a name="epLsar"></a>


```python
with pm.Model() as model:
    epLsarModel = ThreeFactorModel('epLsar',x1,x2,x3,dataZ.epLsar_z.values)
```

#### Verify model settings


```python
epLsarModel.printParams(x1,x2,x3,dataZ.epLsar_z.values)
```

    The number of levels of the x variables are (2, 11, 139)
    The standard deviations used for the beta priors are (1.0181033807478441, 1.4891257020651163, 1.4918655146500768)



```python
try:
    graph_epLsar = pm.model_to_graphviz(epLsarModel)    
except:
    graph_epLsar = "Could not make graph"
graph_epLsar
```




    
![svg](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_45_0.svg)
    



#### Check prior choice


```python
with epLsarModel as model:
    prior_pred_epLsar = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_epLsar,dataZ.epLsar_z.values,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_48_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with epLsarModel as model:
    trace_epLsar = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=target_accept,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [epLsar_b3_dist, epLsar_b2_dist, epLsar_b1_dist, epLsar_b0_dist, epLsar_sigmaY, epLsar_nuY, epLsar_mu_b3, epLsar_mu_b2, epLsar_mu_b1, epLsar_mu_b0, epLsar_sigma3, epLsar_sigma2, epLsar_sigma1, epLsar_sigma0]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 09:48<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 590 seconds.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
with epLsarModel as model:
    if writeOut:
        with open(outPathData + 'model_{}.pkl'.format('epLsar'), 'wb') as buff:
            pickle.dump({'model':epLsarModel, 'trace': trace_epLsar}, buff)
```

#### Check sampling


```python
with epLsarModel as model:
    dataTrace_epLsar = az.from_pymc3(trace=trace_epLsar)
```


```python
pm.summary(dataTrace_epLsar,hdi_prob=0.95).round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_2.5%</th>
      <th>hdi_97.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>epLsar_mu_b0</th>
      <td>-0.04</td>
      <td>0.79</td>
      <td>-1.50</td>
      <td>1.64</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4578.0</td>
      <td>2208.0</td>
      <td>4588.0</td>
      <td>3218.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[0]</th>
      <td>0.02</td>
      <td>0.68</td>
      <td>-1.42</td>
      <td>1.25</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3638.0</td>
      <td>2339.0</td>
      <td>3636.0</td>
      <td>3182.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[1]</th>
      <td>-0.04</td>
      <td>0.68</td>
      <td>-1.41</td>
      <td>1.22</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3590.0</td>
      <td>2011.0</td>
      <td>3596.0</td>
      <td>2887.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[0]</th>
      <td>-0.14</td>
      <td>0.53</td>
      <td>-1.16</td>
      <td>0.89</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1993.0</td>
      <td>1993.0</td>
      <td>1996.0</td>
      <td>2449.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[1]</th>
      <td>-0.18</td>
      <td>0.52</td>
      <td>-1.22</td>
      <td>0.81</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1963.0</td>
      <td>1825.0</td>
      <td>1958.0</td>
      <td>2574.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>epLsar_mu[273]</th>
      <td>0.18</td>
      <td>0.78</td>
      <td>-1.65</td>
      <td>0.90</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>283.0</td>
      <td>283.0</td>
      <td>555.0</td>
      <td>965.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>epLsar_mu[274]</th>
      <td>0.05</td>
      <td>0.11</td>
      <td>-0.17</td>
      <td>0.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4555.0</td>
      <td>2784.0</td>
      <td>4392.0</td>
      <td>3415.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu[275]</th>
      <td>-0.01</td>
      <td>0.11</td>
      <td>-0.22</td>
      <td>0.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4624.0</td>
      <td>2478.0</td>
      <td>4478.0</td>
      <td>3058.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu[276]</th>
      <td>-1.11</td>
      <td>0.14</td>
      <td>-1.38</td>
      <td>-0.87</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4435.0</td>
      <td>4420.0</td>
      <td>4210.0</td>
      <td>3317.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu[277]</th>
      <td>-1.18</td>
      <td>0.14</td>
      <td>-1.43</td>
      <td>-0.92</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4418.0</td>
      <td>4415.0</td>
      <td>4205.0</td>
      <td>3404.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>743 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_epLsar,dataTrace_epLsar,'epLsar')
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_56_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_56_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_56_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_56_4.png)
    



```python
with epLsarModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_57_0.png)
    



```python
with epLsarModel as model:
    plotting_lib.pm.energyplot(trace_epLsar)
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_58_0.png)
    


#### Posterior predictive distribution


```python
with epLsarModel as model:
    posterior_pred_epLsar = pm.sample_posterior_predictive(trace_epLsar,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/sampling.py:1708: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      "samples parameter is smaller than nchains times ndraws, some draws "




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:01<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_epLsar,posterior_pred_epLsar,dataZ.epLsar_z.values,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_61_0.png)
    


#### Compare prior and posterior for model parameters


```python
with epLsarModel as model:
    pm_data_epLsar = az.from_pymc3(trace=trace_epLsar,prior=prior_pred_epLsar,posterior_predictive=posterior_pred_epLsar)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable epLsar_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_64_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_66_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_67_0.png)
    


### Rsquared<a name="r"></a>


```python
with pm.Model() as model:
    RsquaredModel = ThreeFactorModel('Rsquared',x1,x2,x3,dataZ["Rsquared_z"].values)
```

#### Verify model settings


```python
RsquaredModel.printParams(x1,x2,x3,dataZ["Rsquared_z"].values)
```

    The number of levels of the x variables are (2, 11, 139)
    The standard deviations used for the beta priors are (1.364226512794052, 3.833663350609328, 8.30397573287845)



```python
pm.model_to_graphviz(RsquaredModel)
```




    
![svg](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_72_0.svg)
    



#### Check prior choice


```python
with RsquaredModel as model:
    prior_pred_Rsquared = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Rsquared,dataZ["Rsquared_z"].values,'Rsquared')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_75_0.png)
    


#### Sampling


```python
with RsquaredModel as model:
    trace_Rsquared = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=target_accept,random_seed=random_seed)
    #fit_Rsquared = pm.fit(random_seed=random_seed)
    #trace_Rsquared = fit_Rsquared.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [Rsquared_b3_dist, Rsquared_b2_dist, Rsquared_b1_dist, Rsquared_b0_dist, Rsquared_sigmaY, Rsquared_nuY, Rsquared_mu_b3, Rsquared_mu_b2, Rsquared_mu_b1, Rsquared_mu_b0, Rsquared_sigma3, Rsquared_sigma2, Rsquared_sigma1, Rsquared_sigma0]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 13:46<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 827 seconds.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
with RsquaredModel as model:
    if writeOut:
        with open(outPathData + 'model_{}.pkl'.format('Rsquared'), 'wb') as buff:
            pickle.dump({'model': RsquaredModel, 'trace': trace_Rsquared}, buff)
```

#### Check sampling


```python
with RsquaredModel as model:
    dataTrace_Rsquared = az.from_pymc3(trace=trace_Rsquared)
```


```python
pm.summary(dataTrace_Rsquared,hdi_prob=0.95).round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_2.5%</th>
      <th>hdi_97.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rsquared_mu_b0</th>
      <td>0.05</td>
      <td>0.82</td>
      <td>-1.58</td>
      <td>1.64</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3512.0</td>
      <td>2594.0</td>
      <td>3532.0</td>
      <td>3278.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b1[0]</th>
      <td>-0.08</td>
      <td>0.73</td>
      <td>-1.44</td>
      <td>1.38</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2577.0</td>
      <td>2577.0</td>
      <td>2580.0</td>
      <td>2991.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b1[1]</th>
      <td>0.11</td>
      <td>0.73</td>
      <td>-1.31</td>
      <td>1.57</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2693.0</td>
      <td>2356.0</td>
      <td>2682.0</td>
      <td>2826.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b2[0]</th>
      <td>0.08</td>
      <td>0.53</td>
      <td>-0.98</td>
      <td>1.16</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1410.0</td>
      <td>1410.0</td>
      <td>1396.0</td>
      <td>1961.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b2[1]</th>
      <td>-0.11</td>
      <td>0.51</td>
      <td>-1.04</td>
      <td>0.98</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>1221.0</td>
      <td>1221.0</td>
      <td>1228.0</td>
      <td>1654.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Rsquared_mu[273]</th>
      <td>0.21</td>
      <td>0.09</td>
      <td>0.05</td>
      <td>0.40</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4436.0</td>
      <td>3566.0</td>
      <td>4200.0</td>
      <td>3064.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[274]</th>
      <td>0.14</td>
      <td>0.09</td>
      <td>-0.05</td>
      <td>0.31</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4024.0</td>
      <td>2720.0</td>
      <td>4023.0</td>
      <td>2677.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[275]</th>
      <td>0.38</td>
      <td>0.09</td>
      <td>0.20</td>
      <td>0.56</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4009.0</td>
      <td>3705.0</td>
      <td>3956.0</td>
      <td>2511.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[276]</th>
      <td>0.19</td>
      <td>0.11</td>
      <td>-0.01</td>
      <td>0.41</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4329.0</td>
      <td>2580.0</td>
      <td>4159.0</td>
      <td>3037.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[277]</th>
      <td>0.43</td>
      <td>0.11</td>
      <td>0.22</td>
      <td>0.64</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4296.0</td>
      <td>3506.0</td>
      <td>4111.0</td>
      <td>2970.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>743 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,dataTrace_Rsquared,trace_Rsquared,'Rsquared')
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/data/io_pymc3.py:89: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      FutureWarning,
    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/data/io_pymc3.py:89: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      FutureWarning,
    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,
    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/data/io_pymc3.py:89: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      FutureWarning,
    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/data/io_pymc3.py:89: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      FutureWarning,



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_82_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_82_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_82_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_82_4.png)
    



```python
with RsquaredModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Rsquared,'Rsquared')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_83_0.png)
    



```python
with RsquaredModel as model:
    plotting_lib.pm.energyplot(trace_Rsquared)
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_84_0.png)
    


#### Posterior predictive distribution


```python
with RsquaredModel as model:
    posterior_pred_Rsquared = pm.sample_posterior_predictive(trace_Rsquared,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/sampling.py:1708: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      "samples parameter is smaller than nchains times ndraws, some draws "




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:01<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Rsquared,posterior_pred_Rsquared,dataZ["Rsquared_z"].values,'Rsquared')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_87_0.png)
    


#### Compare prior and posterior for model parameters


```python
with RsquaredModel as model:
    pm_data_Rsquared = az.from_pymc3(trace=trace_Rsquared,prior=prior_pred_Rsquared,posterior_predictive=posterior_pred_Rsquared)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable Rsquared_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'Rsquared')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_90_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'Rsquared')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_92_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_Rsquared,'Rsquared')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_93_0.png)
    


### Asfc  <a name="Asfc"></a>


```python
with pm.Model() as model:
    AsfcModel = ThreeFactorModel('Asfc',x1,x2,x3,dataZ["Asfc_z"].values)
```

#### Verify model settings


```python
AsfcModel.printParams(x1,x2,x3,dataZ["Asfc_z"].values)
```

    The number of levels of the x variables are (2, 11, 139)
    The standard deviations used for the beta priors are (1.0834504037573414, 1.7347208877727194, 0.46758648719539714)



```python
pm.model_to_graphviz(AsfcModel)
```




    
![svg](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_98_0.svg)
    



#### Check prior choice


```python
with AsfcModel as model:
    prior_pred_Asfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Asfc,dataZ["Asfc_z"].values,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_101_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with AsfcModel as model:
    trace_Asfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=target_accept,random_seed=random_seed)
    #fit_Asfc = pm.fit(random_seed=random_seed)
    #trace_Asfc = fit_Asfc.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [Asfc_b3_dist, Asfc_b2_dist, Asfc_b1_dist, Asfc_b0_dist, Asfc_sigmaY, Asfc_nuY, Asfc_mu_b3, Asfc_mu_b2, Asfc_mu_b1, Asfc_mu_b0, Asfc_sigma3, Asfc_sigma2, Asfc_sigma1, Asfc_sigma0]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 11:08<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 668 seconds.
    The rhat statistic is larger than 1.05 for some parameters. This indicates slight problems during sampling.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
with AsfcModel as model:
    if writeOut:
        with open(outPathData + 'model_{}.pkl'.format('Asfc'), 'wb') as buff:
            pickle.dump({'model': AsfcModel, 'trace': trace_Asfc}, buff)
```

#### Check sampling


```python
with AsfcModel as model:
    dataTrace_Asfc = az.from_pymc3(trace=trace_Asfc)
```


```python
pm.summary(dataTrace_Asfc,hdi_prob=0.95).round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_2.5%</th>
      <th>hdi_97.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Asfc_mu_b0</th>
      <td>0.05</td>
      <td>0.83</td>
      <td>-1.55</td>
      <td>1.72</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>2909.0</td>
      <td>2255.0</td>
      <td>2905.0</td>
      <td>2923.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[0]</th>
      <td>0.12</td>
      <td>0.71</td>
      <td>-1.35</td>
      <td>1.45</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2599.0</td>
      <td>2599.0</td>
      <td>2600.0</td>
      <td>3001.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[1]</th>
      <td>-0.07</td>
      <td>0.71</td>
      <td>-1.55</td>
      <td>1.28</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2755.0</td>
      <td>2673.0</td>
      <td>2754.0</td>
      <td>2993.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[0]</th>
      <td>1.09</td>
      <td>0.58</td>
      <td>0.02</td>
      <td>2.26</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1841.0</td>
      <td>1841.0</td>
      <td>1844.0</td>
      <td>2312.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[1]</th>
      <td>0.87</td>
      <td>0.55</td>
      <td>-0.20</td>
      <td>1.97</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>1376.0</td>
      <td>1376.0</td>
      <td>1380.0</td>
      <td>1877.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Asfc_mu[273]</th>
      <td>-1.16</td>
      <td>0.10</td>
      <td>-1.35</td>
      <td>-0.96</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4235.0</td>
      <td>4235.0</td>
      <td>4229.0</td>
      <td>3084.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[274]</th>
      <td>-0.86</td>
      <td>0.10</td>
      <td>-1.06</td>
      <td>-0.68</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4135.0</td>
      <td>4017.0</td>
      <td>4147.0</td>
      <td>2803.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[275]</th>
      <td>-1.08</td>
      <td>0.10</td>
      <td>-1.28</td>
      <td>-0.90</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4093.0</td>
      <td>4048.0</td>
      <td>4115.0</td>
      <td>2683.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[276]</th>
      <td>-0.97</td>
      <td>0.10</td>
      <td>-1.16</td>
      <td>-0.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4176.0</td>
      <td>4161.0</td>
      <td>4171.0</td>
      <td>2895.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[277]</th>
      <td>-1.19</td>
      <td>0.10</td>
      <td>-1.38</td>
      <td>-1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4128.0</td>
      <td>4117.0</td>
      <td>4124.0</td>
      <td>2964.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>743 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,dataTrace_Asfc,'Asfc')
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_109_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_109_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_109_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_109_4.png)
    



```python
with AsfcModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_110_0.png)
    



```python
with AsfcModel as model:
    plotting_lib.pm.energyplot(trace_Asfc)
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_111_0.png)
    


#### Posterior predictive distribution


```python
with AsfcModel as model:
    posterior_pred_Asfc = pm.sample_posterior_predictive(trace_Asfc,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/sampling.py:1708: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      "samples parameter is smaller than nchains times ndraws, some draws "




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:01<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Asfc,posterior_pred_Asfc,dataZ["Asfc_z"].values,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_114_0.png)
    


#### Compare prior and posterior for model parameters


```python
with AsfcModel as model:
    pm_data_Asfc = az.from_pymc3(trace=trace_Asfc,prior=prior_pred_Asfc,posterior_predictive=posterior_pred_Asfc)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable Asfc_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_117_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_119_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_120_0.png)
    


### 	Smfc  <a name="Smfc"></a>


```python
with pm.Model() as model:
    SmfcModel = ThreeFactorModel('Smfc',x1,x2,x3,dataZ.Smfc_z.values)
```

#### Verify model settings


```python
SmfcModel.printParams(x1,x2,x3,dataZ.Smfc_z.values)
```

    The number of levels of the x variables are (2, 11, 139)
    The standard deviations used for the beta priors are (1.2903046708375803, 3.554940942907768, 10.68509404141528)



```python
pm.model_to_graphviz(SmfcModel)
```




    
![svg](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_125_0.svg)
    



#### Check prior choice


```python
with SmfcModel as model:
    prior_pred_Smfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Smfc,dataZ.Smfc_z.values,'Smfc')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_128_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with SmfcModel as model:
    trace_Smfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.8,random_seed=random_seed)
    #fit_Smfc = pm.fit(random_seed=random_seed)
    #trace_Smfc = fit_Smfc.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [Smfc_b3_dist, Smfc_b2_dist, Smfc_b1_dist, Smfc_b0_dist, Smfc_sigmaY, Smfc_nuY, Smfc_mu_b3, Smfc_mu_b2, Smfc_mu_b1, Smfc_mu_b0, Smfc_sigma3, Smfc_sigma2, Smfc_sigma1, Smfc_sigma0]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='799' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  9.99% [799/8000 00:06<01:00 Sampling 4 chains, 0 divergences]
</div>




    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    ~/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/sampling.py in _mp_sample(draws, tune, step, chains, cores, chain, random_seed, start, progressbar, trace, model, callback, discard_tuned_samples, mp_ctx, pickle_backend, **kwargs)
       1485             with sampler:
    -> 1486                 for draw in sampler:
       1487                     trace = traces[draw.chain - chain]


    ~/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/parallel_sampling.py in __iter__(self)
        491         while self._active:
    --> 492             draw = ProcessAdapter.recv_draw(self._active)
        493             proc, is_last, draw, tuning, stats, warns = draw


    ~/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/parallel_sampling.py in recv_draw(processes, timeout)
        351         pipes = [proc._msg_pipe for proc in processes]
    --> 352         ready = multiprocessing.connection.wait(pipes)
        353         if not ready:


    /usr/lib/python3.7/multiprocessing/connection.py in wait(object_list, timeout)
        919             while True:
    --> 920                 ready = selector.select(timeout)
        921                 if ready:


    /usr/lib/python3.7/selectors.py in select(self, timeout)
        414         try:
    --> 415             fd_event_list = self._selector.poll(timeout)
        416         except InterruptedError:


    KeyboardInterrupt: 

    
    During handling of the above exception, another exception occurred:


    ValueError                                Traceback (most recent call last)

    <ipython-input-85-755e53d074df> in <module>
          1 with SmfcModel as model:
    ----> 2     trace_Smfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.8,random_seed=random_seed)
          3     #fit_Smfc = pm.fit(random_seed=random_seed)
          4     #trace_Smfc = fit_Smfc.sample(draws=numSamples)


    ~/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/sampling.py in sample(draws, step, init, n_init, start, trace, chain_idx, chains, cores, tune, progressbar, model, random_seed, discard_tuned_samples, compute_convergence_checks, callback, return_inferencedata, idata_kwargs, mp_ctx, pickle_backend, **kwargs)
        543         _print_step_hierarchy(step)
        544         try:
    --> 545             trace = _mp_sample(**sample_args, **parallel_args)
        546         except pickle.PickleError:
        547             _log.warning("Could not pickle model, sampling singlethreaded.")


    ~/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/sampling.py in _mp_sample(draws, tune, step, chains, cores, chain, random_seed, start, progressbar, trace, model, callback, discard_tuned_samples, mp_ctx, pickle_backend, **kwargs)
       1510     except KeyboardInterrupt:
       1511         if discard_tuned_samples:
    -> 1512             traces, length = _choose_chains(traces, tune)
       1513         else:
       1514             traces, length = _choose_chains(traces, 0)


    ~/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/sampling.py in _choose_chains(traces, tune)
       1528     lengths = [max(0, len(trace) - tune) for trace in traces]
       1529     if not sum(lengths):
    -> 1530         raise ValueError("Not enough samples to build a trace.")
       1531 
       1532     idxs = np.argsort(lengths)[::-1]


    ValueError: Not enough samples to build a trace.


Analysis stopped here because sampling did not converge.
As the plot shows, some data points are very far away from the others, which would require the analysis to be based on more heavy-tailed distributions.

### HAsfc9 <a name="HAsfc9"></a>


```python
with pm.Model() as model:
    HAsfc9Model = ThreeFactorModel('HAsfc9',x1,x2,x3,dataZ["HAsfc9_z"].values)
```

#### Verify model settings


```python
HAsfc9Model.printParams(x1,x2,x3,dataZ["HAsfc9_z"].values)
```

    The number of levels of the x variables are (2, 11, 139)
    The standard deviations used for the beta priors are (1.2742199511468122, 3.5115369692183998, 5.975437122957168)



```python
pm.model_to_graphviz(HAsfc9Model)
```




    
![svg](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_137_0.svg)
    



#### Check prior choice


```python
with HAsfc9Model as model:
    prior_pred_HAsfc9 = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc9,dataZ["HAsfc9_z"].values,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_140_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with HAsfc9Model as model:
    trace_HAsfc9 = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=target_accept,random_seed=random_seed)
    #fit_HAsfc9 = pm.fit(random_seed=random_seed)
    #trace_HAsfc9 = fit_HAsfc9.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [HAsfc9_b3_dist, HAsfc9_b2_dist, HAsfc9_b1_dist, HAsfc9_b0_dist, HAsfc9_sigmaY, HAsfc9_nuY, HAsfc9_mu_b3, HAsfc9_mu_b2, HAsfc9_mu_b1, HAsfc9_mu_b0, HAsfc9_sigma3, HAsfc9_sigma2, HAsfc9_sigma1, HAsfc9_sigma0]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 1:23:33<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 5014 seconds.
    The rhat statistic is larger than 1.05 for some parameters. This indicates slight problems during sampling.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
with HAsfc9Model as model:
    if writeOut:
        with open(outPathData + 'model_{}.pkl'.format('HAsfc9'), 'wb') as buff:
            pickle.dump({'model': HAsfc9Model, 'trace': trace_HAsfc9}, buff)
```

#### Check sampling


```python
with HAsfc9Model as model:
    dataTrace_HAsfc9 = az.from_pymc3(trace=trace_HAsfc9)
```


```python
pm.summary(dataTrace_HAsfc9,hdi_prob=0.95).round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_2.5%</th>
      <th>hdi_97.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HAsfc9_mu_b0</th>
      <td>-0.02</td>
      <td>0.83</td>
      <td>-1.68</td>
      <td>1.58</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4930.0</td>
      <td>1763.0</td>
      <td>4905.0</td>
      <td>3032.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[0]</th>
      <td>0.01</td>
      <td>0.71</td>
      <td>-1.38</td>
      <td>1.45</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4889.0</td>
      <td>2383.0</td>
      <td>4870.0</td>
      <td>3394.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[1]</th>
      <td>-0.01</td>
      <td>0.71</td>
      <td>-1.34</td>
      <td>1.51</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4937.0</td>
      <td>2152.0</td>
      <td>4934.0</td>
      <td>3077.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[0]</th>
      <td>-0.26</td>
      <td>0.53</td>
      <td>-1.37</td>
      <td>0.74</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1869.0</td>
      <td>1869.0</td>
      <td>1857.0</td>
      <td>2582.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[1]</th>
      <td>0.07</td>
      <td>0.52</td>
      <td>-0.99</td>
      <td>1.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1892.0</td>
      <td>1892.0</td>
      <td>1902.0</td>
      <td>2461.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[273]</th>
      <td>0.47</td>
      <td>0.03</td>
      <td>0.42</td>
      <td>0.52</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3427.0</td>
      <td>3302.0</td>
      <td>3426.0</td>
      <td>3592.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[274]</th>
      <td>0.46</td>
      <td>0.22</td>
      <td>0.20</td>
      <td>0.73</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>159.0</td>
      <td>159.0</td>
      <td>248.0</td>
      <td>2818.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[275]</th>
      <td>0.42</td>
      <td>0.22</td>
      <td>0.16</td>
      <td>0.69</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>159.0</td>
      <td>159.0</td>
      <td>247.0</td>
      <td>2869.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[276]</th>
      <td>-0.09</td>
      <td>0.02</td>
      <td>-0.12</td>
      <td>-0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3060.0</td>
      <td>2461.0</td>
      <td>3936.0</td>
      <td>3005.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[277]</th>
      <td>-0.13</td>
      <td>0.02</td>
      <td>-0.16</td>
      <td>-0.10</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3008.0</td>
      <td>2507.0</td>
      <td>3864.0</td>
      <td>3304.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>743 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,dataTrace_HAsfc9,'HAsfc9')
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_148_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_148_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_148_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_148_4.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_149_0.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.pm.energyplot(trace_HAsfc9)
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_150_0.png)
    


#### Posterior predictive distribution


```python
with HAsfc9Model as model:
    posterior_pred_HAsfc9 = pm.sample_posterior_predictive(trace_HAsfc9,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/pymc3/sampling.py:1708: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      "samples parameter is smaller than nchains times ndraws, some draws "




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:01<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc9,posterior_pred_HAsfc9,dataZ["HAsfc9_z"].values,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_153_0.png)
    


#### Compare prior and posterior for model parameters


```python
with HAsfc9Model as model:
    pm_data_HAsfc9 = az.from_pymc3(trace=trace_HAsfc9,prior=prior_pred_HAsfc9,posterior_predictive=posterior_pred_HAsfc9)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable HAsfc9_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_156_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_158_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_159_0.png)
    


## Summary <a name="summary"></a>
The contrast plots between the two software packages are shown below again for each variable except Smfc


```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_161_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_Rsquared,'Rsquared')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_162_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_163_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_filter_weak_files/Statistical_Model_ThreeFactor_filter_weak_164_0.png)
    


### Write out


```python
!jupyter nbconvert --to html Statistical_Model_ThreeFactor_filter_weak.ipynb
```

    [NbConvertApp] Converting notebook Statistical_Model_ThreeFactor_filter_weak.ipynb to html
    [NbConvertApp] Writing 7303955 bytes to Statistical_Model_ThreeFactor_filter_weak.html



```python
!jupyter nbconvert --to markdown Statistical_Model_ThreeFactor_filter_weak.ipynb
```

    [NbConvertApp] Converting notebook Statistical_Model_ThreeFactor_filter_weak.ipynb to markdown
    [NbConvertApp] Support files will be in Statistical_Model_ThreeFactor_filter_weak_files/
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_filter_weak_files
    [NbConvertApp] Writing 69184 bytes to Statistical_Model_ThreeFactor_filter_weak.md



```python

































```
