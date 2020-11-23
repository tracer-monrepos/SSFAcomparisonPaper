# Analysis for SSFA project: Three factor  model

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
   1. [HAsfc81](#HAsfc81)   
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

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.



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
outPathPlots = "../plots/statistical_model_three_factors/"
outPathData = "../derived_data/statistical_model_three_factors/"
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
numCores = 10
numTune = 1000
numPredSamples = 2000
random_seed=36514535
target_accept = 0.99
```

## Load data <a name="load"></a>


```python
datafile = "../derived_data/preprocessing/preprocessed.dat"
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
      <td>116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5</td>
      <td>117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>117</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>118</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>275</th>
      <td>1</td>
      <td>9</td>
      <td>51</td>
    </tr>
    <tr>
      <th>276</th>
      <td>0</td>
      <td>9</td>
      <td>52</td>
    </tr>
    <tr>
      <th>277</th>
      <td>1</td>
      <td>9</td>
      <td>52</td>
    </tr>
    <tr>
      <th>278</th>
      <td>0</td>
      <td>9</td>
      <td>53</td>
    </tr>
    <tr>
      <th>279</th>
      <td>1</td>
      <td>9</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
<p>280 rows × 3 columns</p>
</div>


x1 indicates the software used, x2 indicates the treatment applied and x3 the sample.


```python
for surfaceParam,(mean,std) in dictMeanStd.items():
    print("Surface parameter {} has mean {} and standard deviation {}".format(surfaceParam,mean,std))
```

    Surface parameter epLsar has mean 0.0028221512571428575 and standard deviation 0.0019000504797019124
    Surface parameter R² has mean 0.998765273042857 and standard deviation 0.0015328558023807836
    Surface parameter Asfc has mean 18.13129049385357 and standard deviation 16.348381888991312
    Surface parameter Smfc has mean 4.938492967075 and standard deviation 38.106353908569346
    Surface parameter HAsfc9 has mean 0.3085220006979256 and standard deviation 0.2211140516395699
    Surface parameter HAsfc81 has mean 0.5639598041398011 and standard deviation 0.40668126467296095



```python
for k,v in dictTreatment.items():
    print("Number {} encodes treatment {}".format(k,v))
```

    Number 5 encodes treatment Dry Bamboo
    Number 6 encodes treatment Dry Grass
    Number 7 encodes treatment Dry Lucerne
    Number 0 encodes treatment BrushDirt
    Number 1 encodes treatment BrushNoDirt
    Number 4 encodes treatment Control
    Number 10 encodes treatment RubDirt
    Number 2 encodes treatment Clover
    Number 3 encodes treatment Clover+Dust
    Number 8 encodes treatment Grass
    Number 9 encodes treatment Grass+Dust



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
      <th>R²_z</th>
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
      <td>116</td>
      <td>0.839414</td>
      <td>-0.017104</td>
      <td>-0.448128</td>
      <td>-0.120476</td>
      <td>-0.806963</td>
      <td>-0.512247</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>116</td>
      <td>0.999368</td>
      <td>0.518462</td>
      <td>-0.477757</td>
      <td>-0.126469</td>
      <td>-0.782634</td>
      <td>-0.497016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
      <td>1.601888</td>
      <td>-0.509024</td>
      <td>-0.276513</td>
      <td>-0.119325</td>
      <td>-0.584158</td>
      <td>-0.662886</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>117</td>
      <td>1.596720</td>
      <td>0.457791</td>
      <td>-0.301685</td>
      <td>-0.126469</td>
      <td>-0.629947</td>
      <td>-0.744422</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>118</td>
      <td>1.168099</td>
      <td>-0.221668</td>
      <td>-0.393502</td>
      <td>-0.121498</td>
      <td>-0.269712</td>
      <td>-0.370958</td>
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
      <th>275</th>
      <td>275</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>51</td>
      <td>0.843056</td>
      <td>0.387986</td>
      <td>-0.997743</td>
      <td>-0.092631</td>
      <td>2.388080</td>
      <td>1.346868</td>
    </tr>
    <tr>
      <th>276</th>
      <td>276</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>52</td>
      <td>0.305544</td>
      <td>-0.791837</td>
      <td>-0.967607</td>
      <td>-0.104937</td>
      <td>2.014963</td>
      <td>2.573677</td>
    </tr>
    <tr>
      <th>277</th>
      <td>277</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>52</td>
      <td>0.166758</td>
      <td>0.635237</td>
      <td>-0.940337</td>
      <td>-0.126098</td>
      <td>2.926894</td>
      <td>3.117333</td>
    </tr>
    <tr>
      <th>278</th>
      <td>278</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>53</td>
      <td>-0.843412</td>
      <td>0.042974</td>
      <td>-1.022523</td>
      <td>-0.085082</td>
      <td>0.280534</td>
      <td>0.543577</td>
    </tr>
    <tr>
      <th>279</th>
      <td>279</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>53</td>
      <td>-1.115313</td>
      <td>0.712218</td>
      <td>-1.021455</td>
      <td>-0.118879</td>
      <td>0.169999</td>
      <td>0.678630</td>
    </tr>
  </tbody>
</table>
<p>280 rows × 11 columns</p>
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
      <th>epLsar</th>
      <th>R²</th>
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
      <td>Dry Bamboo</td>
      <td>Dry Bamboo</td>
      <td>NaN</td>
      <td>0.004417</td>
      <td>0.998739</td>
      <td>10.805118</td>
      <td>0.347586</td>
      <td>0.130091</td>
      <td>0.355639</td>
      <td>0.019460</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_1</td>
      <td>Toothfrax</td>
      <td>Dry Bamboo</td>
      <td>Dry Bamboo</td>
      <td>NaN</td>
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
      <td>116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_2</td>
      <td>ConfoMap</td>
      <td>Dry Bamboo</td>
      <td>Dry Bamboo</td>
      <td>NaN</td>
      <td>0.005866</td>
      <td>0.997985</td>
      <td>13.610750</td>
      <td>0.391436</td>
      <td>0.179356</td>
      <td>0.294377</td>
      <td>0.020079</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_2</td>
      <td>Toothfrax</td>
      <td>Dry Bamboo</td>
      <td>Dry Bamboo</td>
      <td>NaN</td>
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
      <td>117</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GuineaPigs</td>
      <td>capor_2CC6B1_txP4_#1_1_100xL_3</td>
      <td>ConfoMap</td>
      <td>Dry Bamboo</td>
      <td>Dry Bamboo</td>
      <td>NaN</td>
      <td>0.005042</td>
      <td>0.998425</td>
      <td>11.698166</td>
      <td>0.308648</td>
      <td>0.248885</td>
      <td>0.413098</td>
      <td>0.019722</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>118</td>
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
    </tr>
    <tr>
      <th>275</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90730-lm2sin-a</td>
      <td>Toothfrax</td>
      <td>Grass+Dust</td>
      <td>Grass+Dust</td>
      <td>NaN</td>
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
      <td>51</td>
    </tr>
    <tr>
      <th>276</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90764-lm2sin-a</td>
      <td>ConfoMap</td>
      <td>Grass+Dust</td>
      <td>Grass+Dust</td>
      <td>NaN</td>
      <td>0.003403</td>
      <td>0.997552</td>
      <td>2.312486</td>
      <td>0.939718</td>
      <td>0.754059</td>
      <td>1.610626</td>
      <td>0.018978</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>52</td>
    </tr>
    <tr>
      <th>277</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90764-lm2sin-a</td>
      <td>Toothfrax</td>
      <td>Grass+Dust</td>
      <td>Grass+Dust</td>
      <td>NaN</td>
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
      <td>52</td>
    </tr>
    <tr>
      <th>278</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90814-lm2sin-a</td>
      <td>ConfoMap</td>
      <td>Grass+Dust</td>
      <td>Grass+Dust</td>
      <td>NaN</td>
      <td>0.001220</td>
      <td>0.998831</td>
      <td>1.414701</td>
      <td>1.696316</td>
      <td>0.370552</td>
      <td>0.785022</td>
      <td>0.017498</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>53</td>
    </tr>
    <tr>
      <th>279</th>
      <td>Sheeps</td>
      <td>L8-Ovis-90814-lm2sin-a</td>
      <td>Toothfrax</td>
      <td>Grass+Dust</td>
      <td>Grass+Dust</td>
      <td>NaN</td>
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
      <td>53</td>
    </tr>
  </tbody>
</table>
<p>280 rows × 17 columns</p>
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
      <th>R²_z</th>
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
      <td>116</td>
      <td>-1</td>
      <td>-1</td>
      <td>-0.159954</td>
      <td>-0.535566</td>
      <td>0.029629</td>
      <td>0.005993</td>
      <td>-0.024329</td>
      <td>-0.015230</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>5</td>
      <td>117</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.005168</td>
      <td>-0.966815</td>
      <td>0.025172</td>
      <td>0.007144</td>
      <td>0.045788</td>
      <td>0.081537</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5</td>
      <td>118</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.139785</td>
      <td>-0.550288</td>
      <td>0.039022</td>
      <td>0.004971</td>
      <td>0.142403</td>
      <td>0.139025</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>5</td>
      <td>119</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.370716</td>
      <td>-0.763882</td>
      <td>0.041183</td>
      <td>0.007772</td>
      <td>0.113397</td>
      <td>-0.223817</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>120</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.138629</td>
      <td>-0.150835</td>
      <td>0.074677</td>
      <td>0.007772</td>
      <td>0.179692</td>
      <td>0.075084</td>
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
      <th>135</th>
      <td>2</td>
      <td>9</td>
      <td>49</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.576471</td>
      <td>-0.505284</td>
      <td>-0.008495</td>
      <td>0.004560</td>
      <td>0.011585</td>
      <td>0.007728</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2</td>
      <td>9</td>
      <td>50</td>
      <td>-1</td>
      <td>-1</td>
      <td>1.728006</td>
      <td>-1.316577</td>
      <td>-0.005035</td>
      <td>0.005599</td>
      <td>0.579285</td>
      <td>0.167894</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2</td>
      <td>9</td>
      <td>51</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2.151764</td>
      <td>-2.659380</td>
      <td>-0.008650</td>
      <td>0.510897</td>
      <td>0.095887</td>
      <td>0.165743</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2</td>
      <td>9</td>
      <td>52</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.138786</td>
      <td>-1.427074</td>
      <td>-0.027269</td>
      <td>0.021161</td>
      <td>-0.911931</td>
      <td>-0.543656</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2</td>
      <td>9</td>
      <td>53</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.271901</td>
      <td>-0.669244</td>
      <td>-0.001067</td>
      <td>0.033797</td>
      <td>0.110535</td>
      <td>-0.135053</td>
    </tr>
  </tbody>
</table>
<p>140 rows × 11 columns</p>
</div>



#### Show difference between software in raw data


```python
variablesList = ['epLsar','R²','Asfc','Smfc','HAsfc9','HAsfc81']
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


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_0.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_1.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_2.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_3.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_4.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_37_5.png)
    


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

    The number of levels of the x variables are (2, 11, 140)
    The standard deviations used for the beta priors are (1.0034852973454325, 1.2901096312620055, 1.5215268338889083)



```python
try:
    graph_epLsar = pm.model_to_graphviz(epLsarModel)    
except:
    graph_epLsar = "Could not make graph"
graph_epLsar
```




    
![svg](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_45_0.svg)
    



#### Check prior choice


```python
with epLsarModel as model:
    prior_pred_epLsar = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_epLsar,dataZ.epLsar_z.values,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_48_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with epLsarModel as model:
    trace_epLsar = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=target_accept,random_seed=random_seed)
    #fit_epLsar = pm.fit(random_seed=random_seed)
    #trace_epLsar = fit_epLsar.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
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
  <progress value='20000' class='' max='20000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [20000/20000 18:55<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 1_000 draw iterations (10_000 + 10_000 draws total) took 1137 seconds.
    The number of effective samples is smaller than 10% for some parameters.



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
      <td>-0.06</td>
      <td>0.82</td>
      <td>-1.67</td>
      <td>1.48</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>8287.0</td>
      <td>5027.0</td>
      <td>8296.0</td>
      <td>7497.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[0]</th>
      <td>0.00</td>
      <td>0.70</td>
      <td>-1.34</td>
      <td>1.39</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>8365.0</td>
      <td>5752.0</td>
      <td>8342.0</td>
      <td>7640.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[1]</th>
      <td>-0.06</td>
      <td>0.69</td>
      <td>-1.37</td>
      <td>1.32</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>8990.0</td>
      <td>5503.0</td>
      <td>8990.0</td>
      <td>7313.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[0]</th>
      <td>-0.26</td>
      <td>0.53</td>
      <td>-1.32</td>
      <td>0.75</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>4827.0</td>
      <td>4811.0</td>
      <td>4780.0</td>
      <td>6328.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[1]</th>
      <td>-0.50</td>
      <td>0.53</td>
      <td>-1.49</td>
      <td>0.57</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4168.0</td>
      <td>4149.0</td>
      <td>4156.0</td>
      <td>5540.0</td>
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
      <th>epLsar_mu[275]</th>
      <td>0.43</td>
      <td>0.76</td>
      <td>-1.42</td>
      <td>1.13</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>710.0</td>
      <td>710.0</td>
      <td>1342.0</td>
      <td>2187.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[276]</th>
      <td>0.27</td>
      <td>0.12</td>
      <td>0.04</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>11139.0</td>
      <td>8611.0</td>
      <td>10983.0</td>
      <td>7449.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[277]</th>
      <td>0.21</td>
      <td>0.12</td>
      <td>-0.03</td>
      <td>0.43</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>11102.0</td>
      <td>7679.0</td>
      <td>10946.0</td>
      <td>7539.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[278]</th>
      <td>-0.92</td>
      <td>0.13</td>
      <td>-1.17</td>
      <td>-0.65</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10940.0</td>
      <td>10940.0</td>
      <td>11017.0</td>
      <td>7556.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[279]</th>
      <td>-0.98</td>
      <td>0.13</td>
      <td>-1.25</td>
      <td>-0.73</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10936.0</td>
      <td>10936.0</td>
      <td>11000.0</td>
      <td>7728.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>748 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_epLsar,dataTrace_epLsar,'epLsar')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_1.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_2.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_3.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_56_4.png)
    



```python
with epLsarModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_57_0.png)
    



```python
with epLsarModel as model:
    plotting_lib.pm.energyplot(trace_epLsar)
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_58_0.png)
    


#### Posterior predictive distribution


```python
with epLsarModel as model:
    posterior_pred_epLsar = pm.sample_posterior_predictive(trace_epLsar,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/.local/lib/python3.8/site-packages/pymc3/sampling.py:1707: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      warnings.warn(




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
  100.00% [2000/2000 00:02<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_epLsar,posterior_pred_epLsar,dataZ.epLsar_z.values,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_61_0.png)
    


#### Compare prior and posterior for model parameters


```python
with epLsarModel as model:
    pm_data_epLsar = az.from_pymc3(trace=trace_epLsar,prior=prior_pred_epLsar,posterior_predictive=posterior_pred_epLsar)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable epLsar_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_64_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_66_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_67_0.png)
    


### R²<a name="r"></a>


```python
with pm.Model() as model:
    RsquaredModel = ThreeFactorModel('R²',x1,x2,x3,dataZ["R²_z"].values)
```

#### Verify model settings


```python
RsquaredModel.printParams(x1,x2,x3,dataZ["R²_z"].values)
```

    The number of levels of the x variables are (2, 11, 140)
    The standard deviations used for the beta priors are (1.0112351887037512, 2.890846695679254, 6.372712717079752)



```python
pm.model_to_graphviz(RsquaredModel)
```




    
![svg](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_72_0.svg)
    



#### Check prior choice


```python
with RsquaredModel as model:
    prior_pred_Rsquared = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Rsquared,dataZ["R²_z"].values,'R²')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_75_0.png)
    


#### Sampling


```python
with RsquaredModel as model:
    trace_Rsquared = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=target_accept,random_seed=random_seed)
    #fit_Rsquared = pm.fit(random_seed=random_seed)
    #trace_Rsquared = fit_Rsquared.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
    NUTS: [R²_b3_dist, R²_b2_dist, R²_b1_dist, R²_b0_dist, R²_sigmaY, R²_nuY, R²_mu_b3, R²_mu_b2, R²_mu_b1, R²_mu_b0, R²_sigma3, R²_sigma2, R²_sigma1, R²_sigma0]




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
  <progress value='20000' class='' max='20000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [20000/20000 10:34<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 1_000 draw iterations (10_000 + 10_000 draws total) took 636 seconds.
    The number of effective samples is smaller than 10% for some parameters.



```python
with RsquaredModel as model:
    if writeOut:
        with open(outPathData + 'model_{}.pkl'.format('R²'), 'wb') as buff:
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
      <th>R²_mu_b0</th>
      <td>0.02</td>
      <td>0.80</td>
      <td>-1.54</td>
      <td>1.60</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>7488.0</td>
      <td>6433.0</td>
      <td>7488.0</td>
      <td>7567.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>R²_mu_b1[0]</th>
      <td>-0.21</td>
      <td>0.69</td>
      <td>-1.60</td>
      <td>1.12</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>6630.0</td>
      <td>6234.0</td>
      <td>6640.0</td>
      <td>7314.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>R²_mu_b1[1]</th>
      <td>0.25</td>
      <td>0.69</td>
      <td>-1.09</td>
      <td>1.59</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>6754.0</td>
      <td>6464.0</td>
      <td>6756.0</td>
      <td>6869.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>R²_mu_b2[0]</th>
      <td>0.02</td>
      <td>0.52</td>
      <td>-0.97</td>
      <td>1.06</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2703.0</td>
      <td>2703.0</td>
      <td>2701.0</td>
      <td>5022.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>R²_mu_b2[1]</th>
      <td>0.02</td>
      <td>0.52</td>
      <td>-1.02</td>
      <td>1.03</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2798.0</td>
      <td>2798.0</td>
      <td>2787.0</td>
      <td>4860.0</td>
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
      <th>R²_mu[275]</th>
      <td>0.04</td>
      <td>0.67</td>
      <td>-1.74</td>
      <td>0.77</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>1631.0</td>
      <td>1535.0</td>
      <td>3108.0</td>
      <td>3027.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>R²_mu[276]</th>
      <td>-0.32</td>
      <td>0.38</td>
      <td>-1.00</td>
      <td>0.32</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>8948.0</td>
      <td>6818.0</td>
      <td>10701.0</td>
      <td>7802.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>R²_mu[277]</th>
      <td>0.22</td>
      <td>0.38</td>
      <td>-0.46</td>
      <td>0.86</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>8920.0</td>
      <td>6761.0</td>
      <td>10664.0</td>
      <td>8104.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>R²_mu[278]</th>
      <td>0.10</td>
      <td>0.15</td>
      <td>-0.20</td>
      <td>0.42</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10768.0</td>
      <td>4635.0</td>
      <td>10676.0</td>
      <td>6613.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>R²_mu[279]</th>
      <td>0.64</td>
      <td>0.16</td>
      <td>0.34</td>
      <td>0.96</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10634.0</td>
      <td>10634.0</td>
      <td>10524.0</td>
      <td>6410.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>748 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,dataTrace_Rsquared,trace_Rsquared,'R²')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/data/io_pymc3.py:85: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      warnings.warn(
    /home/bob/.local/lib/python3.8/site-packages/arviz/data/io_pymc3.py:85: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      warnings.warn(
    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(
    /home/bob/.local/lib/python3.8/site-packages/arviz/data/io_pymc3.py:85: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      warnings.warn(
    /home/bob/.local/lib/python3.8/site-packages/arviz/data/io_pymc3.py:85: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      warnings.warn(



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_1.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_2.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_3.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_82_4.png)
    



```python
with RsquaredModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Rsquared,'R²')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_83_0.png)
    



```python
with RsquaredModel as model:
    plotting_lib.pm.energyplot(trace_Rsquared)
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_84_0.png)
    


#### Posterior predictive distribution


```python
with RsquaredModel as model:
    posterior_pred_Rsquared = pm.sample_posterior_predictive(trace_Rsquared,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/.local/lib/python3.8/site-packages/pymc3/sampling.py:1707: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      warnings.warn(




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
  100.00% [2000/2000 00:02<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Rsquared,posterior_pred_Rsquared,dataZ["R²_z"].values,'R²')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_87_0.png)
    


#### Compare prior and posterior for model parameters


```python
with RsquaredModel as model:
    pm_data_Rsquared = az.from_pymc3(trace=trace_Rsquared,prior=prior_pred_Rsquared,posterior_predictive=posterior_pred_Rsquared)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable R²_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'R²')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_90_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'R²')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_92_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_Rsquared,'R²')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_93_0.png)
    


### Asfc  <a name="Asfc"></a>


```python
with pm.Model() as model:
    AsfcModel = ThreeFactorModel('Asfc',x1,x2,x3,dataZ["Asfc_z"].values)
```

#### Verify model settings


```python
AsfcModel.printParams(x1,x2,x3,dataZ["Asfc_z"].values)
```

    The number of levels of the x variables are (2, 11, 140)
    The standard deviations used for the beta priors are (1.0422213301584404, 0.6910746316178924, 0.4801483731544936)



```python
pm.model_to_graphviz(AsfcModel)
```




    
![svg](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_98_0.svg)
    



#### Check prior choice


```python
with AsfcModel as model:
    prior_pred_Asfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Asfc,dataZ["Asfc_z"].values,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_101_0.png)
    


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
    Multiprocess sampling (10 chains in 10 jobs)
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
  <progress value='20000' class='' max='20000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [20000/20000 1:28:26<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 1_000 draw iterations (10_000 + 10_000 draws total) took 5307 seconds.
    The number of effective samples is smaller than 10% for some parameters.



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
      <td>0.06</td>
      <td>0.82</td>
      <td>-1.51</td>
      <td>1.69</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>7447.0</td>
      <td>5776.0</td>
      <td>7446.0</td>
      <td>7394.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[0]</th>
      <td>0.03</td>
      <td>0.70</td>
      <td>-1.33</td>
      <td>1.43</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>7109.0</td>
      <td>5649.0</td>
      <td>7115.0</td>
      <td>7132.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[1]</th>
      <td>0.01</td>
      <td>0.69</td>
      <td>-1.32</td>
      <td>1.39</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>7552.0</td>
      <td>5825.0</td>
      <td>7558.0</td>
      <td>7622.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[0]</th>
      <td>1.02</td>
      <td>0.54</td>
      <td>-0.09</td>
      <td>2.03</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3499.0</td>
      <td>3361.0</td>
      <td>3432.0</td>
      <td>5699.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[1]</th>
      <td>1.17</td>
      <td>0.56</td>
      <td>0.04</td>
      <td>2.24</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2981.0</td>
      <td>2907.0</td>
      <td>2952.0</td>
      <td>4442.0</td>
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
      <th>Asfc_mu[275]</th>
      <td>-1.02</td>
      <td>0.03</td>
      <td>-1.07</td>
      <td>-0.97</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9456.0</td>
      <td>9456.0</td>
      <td>9569.0</td>
      <td>6577.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[276]</th>
      <td>-0.94</td>
      <td>0.03</td>
      <td>-1.00</td>
      <td>-0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>11371.0</td>
      <td>11055.0</td>
      <td>9957.0</td>
      <td>6840.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[277]</th>
      <td>-0.97</td>
      <td>0.03</td>
      <td>-1.02</td>
      <td>-0.91</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>11345.0</td>
      <td>11109.0</td>
      <td>10011.0</td>
      <td>6933.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[278]</th>
      <td>-1.01</td>
      <td>0.02</td>
      <td>-1.05</td>
      <td>-0.96</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10217.0</td>
      <td>10198.0</td>
      <td>10062.0</td>
      <td>7580.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[279]</th>
      <td>-1.04</td>
      <td>0.02</td>
      <td>-1.08</td>
      <td>-0.99</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10185.0</td>
      <td>10165.0</td>
      <td>10023.0</td>
      <td>7435.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>748 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,dataTrace_Asfc,'Asfc')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_1.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_2.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_3.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_109_4.png)
    



```python
with AsfcModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_110_0.png)
    



```python
with AsfcModel as model:
    plotting_lib.pm.energyplot(trace_Asfc)
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_111_0.png)
    


#### Posterior predictive distribution


```python
with AsfcModel as model:
    posterior_pred_Asfc = pm.sample_posterior_predictive(trace_Asfc,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/.local/lib/python3.8/site-packages/pymc3/sampling.py:1707: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      warnings.warn(




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
  100.00% [2000/2000 00:02<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Asfc,posterior_pred_Asfc,dataZ["Asfc_z"].values,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_114_0.png)
    


#### Compare prior and posterior for model parameters


```python
with AsfcModel as model:
    pm_data_Asfc = az.from_pymc3(trace=trace_Asfc,prior=prior_pred_Asfc,posterior_predictive=posterior_pred_Asfc)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable Asfc_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_117_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_119_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_120_0.png)
    


### 	Smfc  <a name="Smfc"></a>


```python
with pm.Model() as model:
    SmfcModel = ThreeFactorModel('Smfc',x1,x2,x3,dataZ.Smfc_z.values)
```

#### Verify model settings


```python
SmfcModel.printParams(x1,x2,x3,dataZ.Smfc_z.values)
```

    The number of levels of the x variables are (2, 11, 140)
    The standard deviations used for the beta priors are (1.3943421938332237, 3.387697874270125, 9.05131085356785)



```python
pm.model_to_graphviz(SmfcModel)
```




    
![svg](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_125_0.svg)
    



#### Check prior choice


```python
with SmfcModel as model:
    prior_pred_Smfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Smfc,dataZ.Smfc_z.values,'Smfc')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_128_0.png)
    


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
    Multiprocess sampling (10 chains in 10 jobs)
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
  <progress value='2157' class='' max='20000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  10.79% [2157/20000 17:51<2:27:45 Sampling 10 chains, 0 divergences]
</div>




    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    ~/.local/lib/python3.8/site-packages/pymc3/sampling.py in _mp_sample(draws, tune, step, chains, cores, chain, random_seed, start, progressbar, trace, model, callback, discard_tuned_samples, mp_ctx, pickle_backend, **kwargs)
       1485             with sampler:
    -> 1486                 for draw in sampler:
       1487                     trace = traces[draw.chain - chain]


    ~/.local/lib/python3.8/site-packages/pymc3/parallel_sampling.py in __iter__(self)
        491         while self._active:
    --> 492             draw = ProcessAdapter.recv_draw(self._active)
        493             proc, is_last, draw, tuning, stats, warns = draw


    ~/.local/lib/python3.8/site-packages/pymc3/parallel_sampling.py in recv_draw(processes, timeout)
        351         pipes = [proc._msg_pipe for proc in processes]
    --> 352         ready = multiprocessing.connection.wait(pipes)
        353         if not ready:


    /usr/lib/python3.8/multiprocessing/connection.py in wait(object_list, timeout)
        930             while True:
    --> 931                 ready = selector.select(timeout)
        932                 if ready:


    /usr/lib/python3.8/selectors.py in select(self, timeout)
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


    ~/.local/lib/python3.8/site-packages/pymc3/sampling.py in sample(draws, step, init, n_init, start, trace, chain_idx, chains, cores, tune, progressbar, model, random_seed, discard_tuned_samples, compute_convergence_checks, callback, return_inferencedata, idata_kwargs, mp_ctx, pickle_backend, **kwargs)
        543         _print_step_hierarchy(step)
        544         try:
    --> 545             trace = _mp_sample(**sample_args, **parallel_args)
        546         except pickle.PickleError:
        547             _log.warning("Could not pickle model, sampling singlethreaded.")


    ~/.local/lib/python3.8/site-packages/pymc3/sampling.py in _mp_sample(draws, tune, step, chains, cores, chain, random_seed, start, progressbar, trace, model, callback, discard_tuned_samples, mp_ctx, pickle_backend, **kwargs)
       1510     except KeyboardInterrupt:
       1511         if discard_tuned_samples:
    -> 1512             traces, length = _choose_chains(traces, tune)
       1513         else:
       1514             traces, length = _choose_chains(traces, 0)


    ~/.local/lib/python3.8/site-packages/pymc3/sampling.py in _choose_chains(traces, tune)
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

    The number of levels of the x variables are (2, 11, 140)
    The standard deviations used for the beta priors are (1.0299664523718457, 2.0653529113379627, 2.5833026311707217)



```python
pm.model_to_graphviz(HAsfc9Model)
```




    
![svg](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_137_0.svg)
    



#### Check prior choice


```python
with HAsfc9Model as model:
    prior_pred_HAsfc9 = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc9,dataZ["HAsfc9_z"].values,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_140_0.png)
    


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
    Multiprocess sampling (10 chains in 10 jobs)
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
  <progress value='20000' class='' max='20000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [20000/20000 53:56<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 1_000 draw iterations (10_000 + 10_000 draws total) took 3237 seconds.
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
      <td>0.02</td>
      <td>0.82</td>
      <td>-1.57</td>
      <td>1.63</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>7718.0</td>
      <td>4985.0</td>
      <td>7730.0</td>
      <td>7316.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[0]</th>
      <td>0.04</td>
      <td>0.69</td>
      <td>-1.36</td>
      <td>1.36</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>7548.0</td>
      <td>5733.0</td>
      <td>7518.0</td>
      <td>7411.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[1]</th>
      <td>-0.03</td>
      <td>0.70</td>
      <td>-1.42</td>
      <td>1.30</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>7414.0</td>
      <td>5426.0</td>
      <td>7407.0</td>
      <td>7311.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[0]</th>
      <td>-0.48</td>
      <td>0.52</td>
      <td>-1.50</td>
      <td>0.56</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>4597.0</td>
      <td>4597.0</td>
      <td>4582.0</td>
      <td>6172.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[1]</th>
      <td>-0.26</td>
      <td>0.52</td>
      <td>-1.28</td>
      <td>0.73</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3897.0</td>
      <td>3897.0</td>
      <td>3880.0</td>
      <td>5847.0</td>
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
      <th>HAsfc9_mu[275]</th>
      <td>2.39</td>
      <td>0.04</td>
      <td>2.31</td>
      <td>2.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10050.0</td>
      <td>10050.0</td>
      <td>10017.0</td>
      <td>7890.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[276]</th>
      <td>2.17</td>
      <td>0.36</td>
      <td>1.84</td>
      <td>3.08</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>761.0</td>
      <td>711.0</td>
      <td>1589.0</td>
      <td>2185.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[277]</th>
      <td>2.08</td>
      <td>0.36</td>
      <td>1.75</td>
      <td>2.99</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>761.0</td>
      <td>709.0</td>
      <td>1588.0</td>
      <td>2159.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[278]</th>
      <td>0.27</td>
      <td>0.04</td>
      <td>0.18</td>
      <td>0.36</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9874.0</td>
      <td>9874.0</td>
      <td>9900.0</td>
      <td>8275.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[279]</th>
      <td>0.18</td>
      <td>0.04</td>
      <td>0.10</td>
      <td>0.27</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9767.0</td>
      <td>9666.0</td>
      <td>9773.0</td>
      <td>8522.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>748 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,dataTrace_HAsfc9,'HAsfc9')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_1.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_2.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_3.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_148_4.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_149_0.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.pm.energyplot(trace_HAsfc9)
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_150_0.png)
    


#### Posterior predictive distribution


```python
with HAsfc9Model as model:
    posterior_pred_HAsfc9 = pm.sample_posterior_predictive(trace_HAsfc9,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/.local/lib/python3.8/site-packages/pymc3/sampling.py:1707: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      warnings.warn(




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
  100.00% [2000/2000 00:02<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc9,posterior_pred_HAsfc9,dataZ["HAsfc9_z"].values,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_153_0.png)
    


#### Compare prior and posterior for model parameters


```python
with HAsfc9Model as model:
    pm_data_HAsfc9 = az.from_pymc3(trace=trace_HAsfc9,prior=prior_pred_HAsfc9,posterior_predictive=posterior_pred_HAsfc9)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable HAsfc9_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_156_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_158_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_159_0.png)
    


### HAsfc81 <a name="HAsfc81"></a>


```python
with pm.Model() as model:
    HAsfc81Model = ThreeFactorModel('HAsfc81',x1,x2,x3,dataZ["HAsfc81_z"].values)
```

#### Verify model settings


```python
HAsfc81Model.printParams(x1,x2,x3,dataZ["HAsfc81_z"].values)
```

    The number of levels of the x variables are (2, 11, 140)
    The standard deviations used for the beta priors are (1.0035538697180921, 1.674345831114272, 1.8608701769937184)



```python
pm.model_to_graphviz(HAsfc81Model)
```




    
![svg](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_164_0.svg)
    



#### Check prior choice


```python
with HAsfc81Model as model:
    prior_pred_HAsfc81 = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc81,dataZ["HAsfc81_z"].values,'HAsfc81')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_167_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with HAsfc81Model as model:
    trace_HAsfc81 = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=target_accept,random_seed=random_seed)
    #fit_HAsfc81 = pm.fit(random_seed=random_seed)
    #trace_HAsfc81 = fit_HAsfc81.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
    NUTS: [HAsfc81_b3_dist, HAsfc81_b2_dist, HAsfc81_b1_dist, HAsfc81_b0_dist, HAsfc81_sigmaY, HAsfc81_nuY, HAsfc81_mu_b3, HAsfc81_mu_b2, HAsfc81_mu_b1, HAsfc81_mu_b0, HAsfc81_sigma3, HAsfc81_sigma2, HAsfc81_sigma1, HAsfc81_sigma0]




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
  <progress value='20000' class='' max='20000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [20000/20000 1:11:03<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 1_000 draw iterations (10_000 + 10_000 draws total) took 4265 seconds.
    The number of effective samples is smaller than 10% for some parameters.



```python
with HAsfc81Model as model:
    if writeOut:
        with open(outPathData + 'model_{}.pkl'.format('HAsfc81'), 'wb') as buff:
            pickle.dump({'model': HAsfc81Model, 'trace': trace_HAsfc81}, buff)
```

#### Check sampling


```python
with HAsfc81Model as model:
    dataTrace_HAsfc81 = az.from_pymc3(trace=trace_HAsfc81)
```


```python
pm.summary(dataTrace_HAsfc81,hdi_prob=0.95).round(2)
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
      <th>HAsfc81_mu_b0</th>
      <td>0.03</td>
      <td>0.83</td>
      <td>-1.60</td>
      <td>1.62</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>6328.0</td>
      <td>5593.0</td>
      <td>6339.0</td>
      <td>7044.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu_b1[0]</th>
      <td>0.03</td>
      <td>0.70</td>
      <td>-1.35</td>
      <td>1.38</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>7201.0</td>
      <td>5956.0</td>
      <td>7198.0</td>
      <td>7184.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu_b1[1]</th>
      <td>-0.03</td>
      <td>0.69</td>
      <td>-1.36</td>
      <td>1.35</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5990.0</td>
      <td>5990.0</td>
      <td>6007.0</td>
      <td>6910.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu_b2[0]</th>
      <td>-0.55</td>
      <td>0.54</td>
      <td>-1.69</td>
      <td>0.47</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3153.0</td>
      <td>3153.0</td>
      <td>3150.0</td>
      <td>4960.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu_b2[1]</th>
      <td>-0.42</td>
      <td>0.54</td>
      <td>-1.54</td>
      <td>0.58</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3787.0</td>
      <td>3787.0</td>
      <td>3774.0</td>
      <td>5995.0</td>
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
      <th>HAsfc81_mu[275]</th>
      <td>1.39</td>
      <td>0.05</td>
      <td>1.29</td>
      <td>1.49</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9779.0</td>
      <td>9779.0</td>
      <td>9781.0</td>
      <td>7388.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu[276]</th>
      <td>2.70</td>
      <td>0.26</td>
      <td>2.43</td>
      <td>3.26</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>1242.0</td>
      <td>1169.0</td>
      <td>2138.0</td>
      <td>4486.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu[277]</th>
      <td>2.62</td>
      <td>0.26</td>
      <td>2.36</td>
      <td>3.18</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>1241.0</td>
      <td>1166.0</td>
      <td>2111.0</td>
      <td>4217.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu[278]</th>
      <td>0.65</td>
      <td>0.10</td>
      <td>0.48</td>
      <td>0.82</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7362.0</td>
      <td>7321.0</td>
      <td>7854.0</td>
      <td>8632.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu[279]</th>
      <td>0.57</td>
      <td>0.10</td>
      <td>0.40</td>
      <td>0.74</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7379.0</td>
      <td>7313.0</td>
      <td>7925.0</td>
      <td>8249.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>748 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc81,dataTrace_HAsfc81,'HAsfc81')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_1.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_2.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_3.png)
    



    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_175_4.png)
    



```python
with HAsfc81Model as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc81,'HAsfc81')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_176_0.png)
    



```python
with HAsfc81Model as model:
    plotting_lib.pm.energyplot(trace_HAsfc81)
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_177_0.png)
    


#### Posterior predictive distribution


```python
with HAsfc81Model as model:
    posterior_pred_HAsfc81 = pm.sample_posterior_predictive(trace_HAsfc81,samples=numPredSamples,random_seed=random_seed)
```

    /home/bob/.local/lib/python3.8/site-packages/pymc3/sampling.py:1707: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      warnings.warn(




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
  100.00% [2000/2000 00:02<00:00]
</div>




```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc81,posterior_pred_HAsfc81,dataZ["HAsfc81_z"].values,'HAsfc81')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_180_0.png)
    


#### Compare prior and posterior for model parameters


```python
with HAsfc81Model as model:
    pm_data_HAsfc81 = az.from_pymc3(trace=trace_HAsfc81,prior=prior_pred_HAsfc81,posterior_predictive=posterior_pred_HAsfc81)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable HAsfc81_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc81,'HAsfc81')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_183_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc81,'HAsfc81')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_185_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_HAsfc81,'HAsfc81')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_186_0.png)
    


## Summary <a name="summary"></a>
The contrast plots between the two software packages are shown below again for each variable except Smfc


```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_epLsar,'epLsar')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_188_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_Rsquared,'R²')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_189_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_Asfc,'Asfc')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_190_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_191_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,x1contrast_dict,trace_HAsfc81,'HAsfc81')
```


    
![png](Statistical_Model_ThreeFactor_files/Statistical_Model_ThreeFactor_192_0.png)
    


### Write out


```python
!jupyter nbconvert --to html Statistical_Model_ThreeFactor.ipynb
```

    [NbConvertApp] Converting notebook Statistical_Model_ThreeFactor.ipynb to html
    [NbConvertApp] Writing 9742523 bytes to Statistical_Model_ThreeFactor.html



```python
!jupyter nbconvert --to markdown Statistical_Model_ThreeFactor.ipynb
```

    [NbConvertApp] Converting notebook Statistical_Model_ThreeFactor.ipynb to markdown
    [NbConvertApp] Support files will be in Statistical_Model_ThreeFactor_files/
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Making directory Statistical_Model_ThreeFactor_files
    [NbConvertApp] Writing 81991 bytes to Statistical_Model_ThreeFactor.md



```python

































```
