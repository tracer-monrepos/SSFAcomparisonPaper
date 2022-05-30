# Analysis for SSFA project: Three factor  model
# Filtered strongly by < 5% NMP

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
outPathPlots = "../plots/statistical_model_three_factors_filter_strong/"
outPathData = "../derived_data/statistical_model_three_factors_filter_strong/"
prefix = "ThreeFactor_filter_strong"
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
datafile = "../derived_data/preprocessing/preprocessed_filter_strong.dat"
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
      <td>98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5</td>
      <td>99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>100</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>225</th>
      <td>1</td>
      <td>9</td>
      <td>52</td>
    </tr>
    <tr>
      <th>226</th>
      <td>0</td>
      <td>9</td>
      <td>53</td>
    </tr>
    <tr>
      <th>227</th>
      <td>1</td>
      <td>9</td>
      <td>53</td>
    </tr>
    <tr>
      <th>228</th>
      <td>0</td>
      <td>9</td>
      <td>54</td>
    </tr>
    <tr>
      <th>229</th>
      <td>1</td>
      <td>9</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 3 columns</p>
</div>


x1 indicates the software used, x2 indicates the treatment applied and x3 the sample.


```python
for surfaceParam,(mean,std) in dictMeanStd.items():
    print("Surface parameter {} has mean {} and standard deviation {}".format(surfaceParam,mean,std))
```

    Surface parameter epLsar has mean 0.0033908522378621737 and standard deviation 0.001989767613698366
    Surface parameter Rsquared has mean 0.9973621855273914 and standard deviation 0.007969957191822462
    Surface parameter Asfc has mean 13.800529685456086 and standard deviation 12.970336908999228
    Surface parameter Smfc has mean 1.350346875497435 and standard deviation 7.832927757231397
    Surface parameter HAsfc9 has mean 0.4835258022408901 and standard deviation 0.8643524891543747
    Surface parameter HAsfc81 has mean 1.0354777484507558 and standard deviation 2.586788670504154



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
      <td>98</td>
      <td>0.515753</td>
      <td>0.080435</td>
      <td>-0.165334</td>
      <td>-0.134789</td>
      <td>-0.402292</td>
      <td>-0.259843</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>98</td>
      <td>0.668494</td>
      <td>0.275762</td>
      <td>-0.268289</td>
      <td>-0.157173</td>
      <td>-0.402677</td>
      <td>-0.260418</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>1.243847</td>
      <td>-0.146130</td>
      <td>0.128490</td>
      <td>-0.134789</td>
      <td>-0.360528</td>
      <td>-0.285754</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>99</td>
      <td>1.238912</td>
      <td>0.264094</td>
      <td>-0.046359</td>
      <td>-0.157173</td>
      <td>-0.363618</td>
      <td>-0.299313</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>100</td>
      <td>0.829617</td>
      <td>-0.323025</td>
      <td>-0.046215</td>
      <td>-0.134789</td>
      <td>-0.257176</td>
      <td>-0.243567</td>
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
      <th>225</th>
      <td>225</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>52</td>
      <td>0.519230</td>
      <td>0.250668</td>
      <td>-0.923702</td>
      <td>0.007447</td>
      <td>0.408438</td>
      <td>0.029468</td>
    </tr>
    <tr>
      <th>226</th>
      <td>226</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>53</td>
      <td>0.005954</td>
      <td>0.186666</td>
      <td>-0.843131</td>
      <td>-0.110277</td>
      <td>0.178138</td>
      <td>0.194629</td>
    </tr>
    <tr>
      <th>227</th>
      <td>227</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>53</td>
      <td>-0.126574</td>
      <td>0.298222</td>
      <td>-0.851345</td>
      <td>-0.155367</td>
      <td>0.546274</td>
      <td>0.307811</td>
    </tr>
    <tr>
      <th>228</th>
      <td>228</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>54</td>
      <td>-1.091196</td>
      <td>0.265539</td>
      <td>-0.949772</td>
      <td>-0.014439</td>
      <td>-0.130543</td>
      <td>-0.089431</td>
    </tr>
    <tr>
      <th>229</th>
      <td>229</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>54</td>
      <td>-1.350837</td>
      <td>0.313027</td>
      <td>-0.953590</td>
      <td>-0.120251</td>
      <td>-0.158980</td>
      <td>-0.075589</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 11 columns</p>
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
      <td>98</td>
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
      <td>98</td>
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
      <td>99</td>
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
      <td>99</td>
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
      <td>100</td>
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
      <th>225</th>
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
      <th>226</th>
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
      <th>227</th>
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
      <th>228</th>
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
      <th>229</th>
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
<p>230 rows × 19 columns</p>
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
      <td>98</td>
      <td>-1</td>
      <td>-1</td>
      <td>-0.152741</td>
      <td>-0.195327</td>
      <td>0.102955</td>
      <td>0.022385</td>
      <td>0.000385</td>
      <td>0.000575</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>5</td>
      <td>99</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.004935</td>
      <td>-0.410224</td>
      <td>0.174849</td>
      <td>0.022385</td>
      <td>0.003090</td>
      <td>0.013559</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5</td>
      <td>100</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.133482</td>
      <td>-0.562275</td>
      <td>0.165059</td>
      <td>0.022385</td>
      <td>0.050717</td>
      <td>0.018889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>5</td>
      <td>101</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.354001</td>
      <td>-0.347736</td>
      <td>0.198535</td>
      <td>0.026957</td>
      <td>0.016939</td>
      <td>-0.032803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>102</td>
      <td>-1</td>
      <td>-1</td>
      <td>-0.024548</td>
      <td>-0.495939</td>
      <td>0.191016</td>
      <td>0.020288</td>
      <td>-0.043304</td>
      <td>0.020445</td>
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
      <th>110</th>
      <td>2</td>
      <td>9</td>
      <td>50</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.550478</td>
      <td>-0.152354</td>
      <td>0.136512</td>
      <td>0.014851</td>
      <td>0.050951</td>
      <td>0.042694</td>
    </tr>
    <tr>
      <th>111</th>
      <td>2</td>
      <td>9</td>
      <td>51</td>
      <td>-1</td>
      <td>-1</td>
      <td>1.650091</td>
      <td>-0.143020</td>
      <td>0.020892</td>
      <td>0.014851</td>
      <td>0.408637</td>
      <td>0.129269</td>
    </tr>
    <tr>
      <th>112</th>
      <td>2</td>
      <td>9</td>
      <td>52</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2.054742</td>
      <td>-0.318156</td>
      <td>0.000641</td>
      <td>0.355427</td>
      <td>-0.009580</td>
      <td>0.004557</td>
    </tr>
    <tr>
      <th>113</th>
      <td>2</td>
      <td>9</td>
      <td>53</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.132528</td>
      <td>-0.111556</td>
      <td>0.008214</td>
      <td>0.045090</td>
      <td>-0.368137</td>
      <td>-0.113182</td>
    </tr>
    <tr>
      <th>114</th>
      <td>2</td>
      <td>9</td>
      <td>54</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.259641</td>
      <td>-0.047488</td>
      <td>0.003818</td>
      <td>0.105812</td>
      <td>0.028437</td>
      <td>-0.013843</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 11 columns</p>
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


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_37_0.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_37_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_37_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_37_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_37_4.png)
    


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

    The number of levels of the x variables are (2, 11, 115)
    The standard deviations used for the beta priors are (1.022692176080412, 1.4502540614172976, 1.4529223548483703)



```python
try:
    graph_epLsar = pm.model_to_graphviz(epLsarModel)    
except:
    graph_epLsar = "Could not make graph"
graph_epLsar
```




    
![svg](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_45_0.svg)
    



#### Check prior choice


```python
with epLsarModel as model:
    prior_pred_epLsar = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,\
                                 prior_pred_epLsar,dataZ.epLsar_z.values,'epLsar',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_48_0.png)
    


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
  100.00% [8000/8000 09:35<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 576 seconds.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
with epLsarModel as model:
    if writeOut:
        with open(outPathData + '{}_model_{}.pkl'.format(prefix,'epLsar'), 'wb') as buff:
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
      <td>0.81</td>
      <td>-1.65</td>
      <td>1.54</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>2667.0</td>
      <td>1940.0</td>
      <td>2695.0</td>
      <td>2645.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[0]</th>
      <td>0.01</td>
      <td>0.69</td>
      <td>-1.35</td>
      <td>1.34</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3259.0</td>
      <td>2306.0</td>
      <td>3262.0</td>
      <td>2709.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[1]</th>
      <td>-0.03</td>
      <td>0.68</td>
      <td>-1.36</td>
      <td>1.28</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3443.0</td>
      <td>2153.0</td>
      <td>3443.0</td>
      <td>2375.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[0]</th>
      <td>-0.17</td>
      <td>0.55</td>
      <td>-1.23</td>
      <td>0.94</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2468.0</td>
      <td>2249.0</td>
      <td>2484.0</td>
      <td>2258.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[1]</th>
      <td>-0.21</td>
      <td>0.53</td>
      <td>-1.26</td>
      <td>0.80</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1904.0</td>
      <td>1904.0</td>
      <td>1912.0</td>
      <td>2121.0</td>
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
      <th>epLsar_mu[225]</th>
      <td>0.10</td>
      <td>0.75</td>
      <td>-1.67</td>
      <td>0.82</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>349.0</td>
      <td>349.0</td>
      <td>622.0</td>
      <td>1067.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>epLsar_mu[226]</th>
      <td>-0.03</td>
      <td>0.11</td>
      <td>-0.24</td>
      <td>0.19</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3510.0</td>
      <td>2451.0</td>
      <td>3624.0</td>
      <td>3055.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu[227]</th>
      <td>-0.09</td>
      <td>0.11</td>
      <td>-0.29</td>
      <td>0.14</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3658.0</td>
      <td>2605.0</td>
      <td>3767.0</td>
      <td>2667.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu[228]</th>
      <td>-1.16</td>
      <td>0.14</td>
      <td>-1.41</td>
      <td>-0.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3939.0</td>
      <td>3939.0</td>
      <td>4215.0</td>
      <td>2818.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>epLsar_mu[229]</th>
      <td>-1.22</td>
      <td>0.14</td>
      <td>-1.48</td>
      <td>-0.96</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3989.0</td>
      <td>3989.0</td>
      <td>4255.0</td>
      <td>2849.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>623 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,\
                             trace_epLsar,dataTrace_epLsar,'epLsar',prefix)
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_56_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_56_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_56_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_56_4.png)
    



```python
with epLsarModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,\
                             trace_epLsar,'epLsar',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_57_0.png)
    



```python
with epLsarModel as model:
    plotting_lib.pm.energyplot(trace_epLsar)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_58_0.png)
    


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
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,\
                                          prior_pred_epLsar,posterior_pred_epLsar,dataZ.epLsar_z.values,'epLsar',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_61_0.png)
    


#### Compare prior and posterior for model parameters


```python
with epLsarModel as model:
    pm_data_epLsar = az.from_pymc3(trace=trace_epLsar,prior=prior_pred_epLsar,posterior_predictive=posterior_pred_epLsar)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable epLsar_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,\
                                 pm_data_epLsar,'epLsar',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_64_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,\
                           pm_data_epLsar,'epLsar',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_66_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,\
                          x1contrast_dict,trace_epLsar,'epLsar',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_67_0.png)
    


### Rsquared<a name="r"></a>


```python
with pm.Model() as model:
    RsquaredModel = ThreeFactorModel('Rsquared',x1,x2,x3,dataZ["Rsquared_z"].values)
```

#### Verify model settings


```python
RsquaredModel.printParams(x1,x2,x3,dataZ["Rsquared_z"].values)
```

    The number of levels of the x variables are (2, 11, 115)
    The standard deviations used for the beta priors are (1.3690772711592387, 3.503507127699258, 7.588834883938683)



```python
pm.model_to_graphviz(RsquaredModel)
```




    
![svg](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_72_0.svg)
    



#### Check prior choice


```python
with RsquaredModel as model:
    prior_pred_Rsquared = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,\
                                 prior_pred_Rsquared,dataZ["Rsquared_z"].values,'Rsquared',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_75_0.png)
    


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
  100.00% [8000/8000 18:29<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 1109 seconds.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
with RsquaredModel as model:
    if writeOut:
        with open(outPathData + '{}_model_{}.pkl'.format(prefix,'Rsquared'), 'wb') as buff:
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
      <td>0.84</td>
      <td>-1.54</td>
      <td>1.74</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>3295.0</td>
      <td>1812.0</td>
      <td>3302.0</td>
      <td>2786.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b1[0]</th>
      <td>-0.06</td>
      <td>0.73</td>
      <td>-1.48</td>
      <td>1.40</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2825.0</td>
      <td>1979.0</td>
      <td>2808.0</td>
      <td>2487.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b1[1]</th>
      <td>0.06</td>
      <td>0.73</td>
      <td>-1.34</td>
      <td>1.51</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2880.0</td>
      <td>2004.0</td>
      <td>2866.0</td>
      <td>2494.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b2[0]</th>
      <td>0.09</td>
      <td>0.54</td>
      <td>-0.96</td>
      <td>1.18</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1721.0</td>
      <td>1721.0</td>
      <td>1723.0</td>
      <td>2236.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b2[1]</th>
      <td>-0.10</td>
      <td>0.51</td>
      <td>-1.10</td>
      <td>0.88</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1578.0</td>
      <td>1578.0</td>
      <td>1572.0</td>
      <td>2142.0</td>
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
      <th>Rsquared_mu[225]</th>
      <td>0.18</td>
      <td>0.09</td>
      <td>-0.00</td>
      <td>0.35</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4224.0</td>
      <td>3376.0</td>
      <td>4145.0</td>
      <td>2678.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[226]</th>
      <td>0.15</td>
      <td>0.07</td>
      <td>-0.00</td>
      <td>0.28</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3966.0</td>
      <td>3432.0</td>
      <td>4157.0</td>
      <td>2986.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[227]</th>
      <td>0.33</td>
      <td>0.07</td>
      <td>0.21</td>
      <td>0.49</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3992.0</td>
      <td>3992.0</td>
      <td>4175.0</td>
      <td>2921.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[228]</th>
      <td>0.20</td>
      <td>0.09</td>
      <td>0.04</td>
      <td>0.38</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4466.0</td>
      <td>3224.0</td>
      <td>4199.0</td>
      <td>3003.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[229]</th>
      <td>0.38</td>
      <td>0.09</td>
      <td>0.21</td>
      <td>0.55</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4374.0</td>
      <td>3913.0</td>
      <td>4141.0</td>
      <td>2890.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>623 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,\
                             dataTrace_Rsquared,trace_Rsquared,'Rsquared',prefix)
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



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_82_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_82_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_82_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_82_4.png)
    



```python
with RsquaredModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Rsquared,'Rsquared',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_83_0.png)
    



```python
with RsquaredModel as model:
    plotting_lib.pm.energyplot(trace_Rsquared)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_84_0.png)
    


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
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,\
                                          prior_pred_Rsquared,posterior_pred_Rsquared,dataZ["Rsquared_z"].values,'Rsquared',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_87_0.png)
    


#### Compare prior and posterior for model parameters


```python
with RsquaredModel as model:
    pm_data_Rsquared = az.from_pymc3(trace=trace_Rsquared,prior=prior_pred_Rsquared,posterior_predictive=posterior_pred_Rsquared)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable Rsquared_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,\
                                 outPathPlots,dictMeanStd,pm_data_Rsquared,'Rsquared',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_90_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'Rsquared',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_92_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,\
                          x1contrast_dict,trace_Rsquared,'Rsquared',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_93_0.png)
    


### Asfc  <a name="Asfc"></a>


```python
with pm.Model() as model:
    AsfcModel = ThreeFactorModel('Asfc',x1,x2,x3,dataZ["Asfc_z"].values)
```

#### Verify model settings


```python
AsfcModel.printParams(x1,x2,x3,dataZ["Asfc_z"].values)
```

    The number of levels of the x variables are (2, 11, 115)
    The standard deviations used for the beta priors are (1.0817297140638054, 1.6678950572969222, 0.44957387459221787)



```python
pm.model_to_graphviz(AsfcModel)
```




    
![svg](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_98_0.svg)
    



#### Check prior choice


```python
with AsfcModel as model:
    prior_pred_Asfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,\
                                 outPathPlots,df,dictMeanStd,prior_pred_Asfc,dataZ["Asfc_z"].values,'Asfc',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_101_0.png)
    


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
  100.00% [8000/8000 10:33<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 634 seconds.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
with AsfcModel as model:
    if writeOut:
        with open(outPathData + '{}_model_{}.pkl'.format(prefix,'Asfc'), 'wb') as buff:
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
      <td>0.82</td>
      <td>-1.65</td>
      <td>1.56</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>2550.0</td>
      <td>2402.0</td>
      <td>2544.0</td>
      <td>2779.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[0]</th>
      <td>0.09</td>
      <td>0.70</td>
      <td>-1.29</td>
      <td>1.45</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3058.0</td>
      <td>2386.0</td>
      <td>3058.0</td>
      <td>3085.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[1]</th>
      <td>-0.08</td>
      <td>0.69</td>
      <td>-1.43</td>
      <td>1.29</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2890.0</td>
      <td>2530.0</td>
      <td>2891.0</td>
      <td>3163.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[0]</th>
      <td>1.05</td>
      <td>0.59</td>
      <td>-0.20</td>
      <td>2.08</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>1569.0</td>
      <td>1569.0</td>
      <td>1580.0</td>
      <td>2218.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[1]</th>
      <td>0.91</td>
      <td>0.54</td>
      <td>-0.13</td>
      <td>1.92</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1496.0</td>
      <td>1496.0</td>
      <td>1506.0</td>
      <td>2389.0</td>
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
      <th>Asfc_mu[225]</th>
      <td>-1.01</td>
      <td>0.09</td>
      <td>-1.20</td>
      <td>-0.84</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3922.0</td>
      <td>3920.0</td>
      <td>3924.0</td>
      <td>2855.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[226]</th>
      <td>-0.76</td>
      <td>0.09</td>
      <td>-0.93</td>
      <td>-0.58</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3856.0</td>
      <td>3848.0</td>
      <td>3855.0</td>
      <td>2852.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[227]</th>
      <td>-0.94</td>
      <td>0.09</td>
      <td>-1.11</td>
      <td>-0.76</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3861.0</td>
      <td>3855.0</td>
      <td>3866.0</td>
      <td>2900.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[228]</th>
      <td>-0.86</td>
      <td>0.09</td>
      <td>-1.04</td>
      <td>-0.68</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3964.0</td>
      <td>3963.0</td>
      <td>3964.0</td>
      <td>3000.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[229]</th>
      <td>-1.04</td>
      <td>0.09</td>
      <td>-1.23</td>
      <td>-0.87</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4010.0</td>
      <td>4010.0</td>
      <td>4024.0</td>
      <td>2897.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>623 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,dataTrace_Asfc,'Asfc',prefix)
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_109_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_109_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_109_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_109_4.png)
    



```python
with AsfcModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,'Asfc',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_110_0.png)
    



```python
with AsfcModel as model:
    plotting_lib.pm.energyplot(trace_Asfc)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_111_0.png)
    


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
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,\
                                          dictMeanStd,prior_pred_Asfc,\
                                          posterior_pred_Asfc,dataZ["Asfc_z"].values,'Asfc',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_114_0.png)
    


#### Compare prior and posterior for model parameters


```python
with AsfcModel as model:
    pm_data_Asfc = az.from_pymc3(trace=trace_Asfc,prior=prior_pred_Asfc,posterior_predictive=posterior_pred_Asfc)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable Asfc_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,\
                                 sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_117_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_119_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,\
                          x1contrast_dict,trace_Asfc,'Asfc',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_120_0.png)
    


### 	Smfc  <a name="Smfc"></a>


```python
with pm.Model() as model:
    SmfcModel = ThreeFactorModel('Smfc',x1,x2,x3,dataZ.Smfc_z.values)
```

#### Verify model settings


```python
SmfcModel.printParams(x1,x2,x3,dataZ.Smfc_z.values)
```

    The number of levels of the x variables are (2, 11, 115)
    The standard deviations used for the beta priors are (1.2914215627674575, 3.238203806583398, 9.733076513589282)



```python
pm.model_to_graphviz(SmfcModel)
```




    
![svg](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_125_0.svg)
    



#### Check prior choice


```python
with SmfcModel as model:
    prior_pred_Smfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,\
                                 df,dictMeanStd,prior_pred_Smfc,dataZ.Smfc_z.values,'Smfc',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_128_0.png)
    


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
  <progress value='1178' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  14.72% [1178/8000 08:28<49:07 Sampling 4 chains, 0 divergences]
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

    <ipython-input-91-755e53d074df> in <module>
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

    The number of levels of the x variables are (2, 11, 115)
    The standard deviations used for the beta priors are (1.2761156280868398, 3.214359712024942, 5.469742884138536)



```python
pm.model_to_graphviz(HAsfc9Model)
```




    
![svg](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_137_0.svg)
    



#### Check prior choice


```python
with HAsfc9Model as model:
    prior_pred_HAsfc9 = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,\
                                 writeOut,outPathPlots,df,dictMeanStd,\
                                 prior_pred_HAsfc9,dataZ["HAsfc9_z"].values,'HAsfc9',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_140_0.png)
    


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
  100.00% [8000/8000 1:20:42<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 4843 seconds.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
with HAsfc9Model as model:
    if writeOut:
        with open(outPathData + '{}_model_{}.pkl'.format(prefix,'HAsfc9'), 'wb') as buff:
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
      <td>-0.00</td>
      <td>0.82</td>
      <td>-1.68</td>
      <td>1.55</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3782.0</td>
      <td>2036.0</td>
      <td>3779.0</td>
      <td>2689.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[0]</th>
      <td>0.01</td>
      <td>0.69</td>
      <td>-1.33</td>
      <td>1.34</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4279.0</td>
      <td>2197.0</td>
      <td>4390.0</td>
      <td>3117.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[1]</th>
      <td>-0.02</td>
      <td>0.70</td>
      <td>-1.38</td>
      <td>1.35</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4028.0</td>
      <td>2107.0</td>
      <td>4063.0</td>
      <td>2662.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[0]</th>
      <td>-0.26</td>
      <td>0.53</td>
      <td>-1.35</td>
      <td>0.72</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2143.0</td>
      <td>2124.0</td>
      <td>2150.0</td>
      <td>2808.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[1]</th>
      <td>0.07</td>
      <td>0.50</td>
      <td>-0.90</td>
      <td>1.07</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>2457.0</td>
      <td>2386.0</td>
      <td>2444.0</td>
      <td>2855.0</td>
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
      <th>HAsfc9_mu[225]</th>
      <td>0.39</td>
      <td>0.02</td>
      <td>0.34</td>
      <td>0.43</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4042.0</td>
      <td>4035.0</td>
      <td>3933.0</td>
      <td>3252.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[226]</th>
      <td>0.36</td>
      <td>0.20</td>
      <td>0.13</td>
      <td>0.63</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>288.0</td>
      <td>288.0</td>
      <td>455.0</td>
      <td>3073.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[227]</th>
      <td>0.33</td>
      <td>0.20</td>
      <td>0.10</td>
      <td>0.59</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>288.0</td>
      <td>288.0</td>
      <td>451.0</td>
      <td>3108.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[228]</th>
      <td>-0.13</td>
      <td>0.02</td>
      <td>-0.16</td>
      <td>-0.10</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5325.0</td>
      <td>2741.0</td>
      <td>3907.0</td>
      <td>3274.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[229]</th>
      <td>-0.16</td>
      <td>0.02</td>
      <td>-0.19</td>
      <td>-0.13</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5295.0</td>
      <td>2994.0</td>
      <td>3852.0</td>
      <td>3338.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>623 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,\
                             outPathPlots,trace_HAsfc9,dataTrace_HAsfc9,'HAsfc9',prefix)
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_148_1.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_148_2.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_148_3.png)
    



    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_148_4.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,'HAsfc9',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_149_0.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.pm.energyplot(trace_HAsfc9)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_150_0.png)
    


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
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,\
                                          outPathPlots,df,dictMeanStd,prior_pred_HAsfc9,posterior_pred_HAsfc9,\
                                          dataZ["HAsfc9_z"].values,'HAsfc9',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_153_0.png)
    


#### Compare prior and posterior for model parameters


```python
with HAsfc9Model as model:
    pm_data_HAsfc9 = az.from_pymc3(trace=trace_HAsfc9,prior=prior_pred_HAsfc9,posterior_predictive=posterior_pred_HAsfc9)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable HAsfc9_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,\
                                 dictMeanStd,pm_data_HAsfc9,'HAsfc9',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_156_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_158_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd\
                          ,x1contrast_dict,trace_HAsfc9,'HAsfc9',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_159_0.png)
    


## Summary <a name="summary"></a>
The contrast plots between the two software packages are shown below again for each variable except Smfc


```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,\
                          x1contrast_dict,trace_epLsar,'epLsar',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_161_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,\
                          x1contrast_dict,trace_Rsquared,'Rsquared',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_162_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,\
                          x1contrast_dict,trace_Asfc,'Asfc',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_163_0.png)
    



```python
plotting_lib.plotContrast(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,\
                          x1contrast_dict,trace_HAsfc9,'HAsfc9',prefix)
```


    
![png](Statistical_Model_ThreeFactor_filter_strong_files/Statistical_Model_ThreeFactor_filter_strong_164_0.png)
    


### Write out


```python
!jupyter nbconvert --to html Statistical_Model_ThreeFactor_filter_strong.ipynb
```


```python
!jupyter nbconvert --to markdown Statistical_Model_ThreeFactor_filter_strong.ipynb
```


```python

































```
