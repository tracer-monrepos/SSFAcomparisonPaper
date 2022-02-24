# Analysis for SSFA project: Two factor model
# Filtered strongly by < 20% NMP

## Table of contents
1. [Used packages](#imports)
1. [Global settings](#settings)
1. [Load data](#load)
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
outPathPlots = "../plots/statistical_model_two_factors_filter_weak/"
outPathData = "../derived_data/statistical_model_two_factors_filter_weak/"
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
random_seed=36534535
target_accept = 0.99
```

## Load data <a name="load"></a>


```python
datafile = "../derived_data/preprocessing/preprocessed_filter_weak.dat"
```


```python
with open(datafile, "rb") as f:
    x1,x2,_,df,dataZ,dictMeanStd,dictTreatment,dictSoftware = pickle.load(f)    
```

Show that everything is correct:


```python
display(pd.DataFrame.from_dict({'x1':x1,'x2':x2}))
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>273</th>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>274</th>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>275</th>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>276</th>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>277</th>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>278 rows × 2 columns</p>
</div>


x1 indicates the software used, x2 indicates the treatment applied.


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


## Model specification <a name="model"></a>


```python
class TwoFactorModel(pm.Model):
    
    """
    Compute params of priors and hyperpriors.
    """
    def getParams(self,x1,x2,y):
        # get lengths
        Nx1Lvl = np.unique(x1).size
        Nx2Lvl = np.unique(x2).size
                        
        dims = (Nx1Lvl, Nx2Lvl)
        
        ### get standard deviations
        
        # convert to pandas dataframe to use their logic
        df = pd.DataFrame.from_dict({'x1':x1,'x2':x2,'y':y})
        
        s1 = df.groupby('x1').std()['y'].max()
        s2 = df.groupby('x2').std()['y'].max()
                
        stdSingle = (s1, s2)
        
        prefac = 0.05
        s12 = prefac * np.linalg.norm([s1,s2])
        
        stdMulti = (s12)
        
        return (dims, stdSingle, stdMulti)
    
    def printParams(self,x1,x2,y):
        dims, stdSingle, stdMulti = self.getParams(x1,x2,y)
        Nx1Lvl, Nx2Lvl = dims
        s1, s2 = stdSingle
        s12 = stdMulti
        
        print("The number of levels of the x variables are {}".format(dims))
        print("The standard deviations used for the beta priors are  {}".format(stdSingle))
        print("The standard deviations used for the M12 priors are {}".format(stdMulti))
    
    def __init__(self,name,x1,x2,y,model=None):
        
        # call super's init first, passing model and name
        super().__init__(name, model)
        
        # get parameter of hyperpriors
        dims, stdSingle, stdMulti = self.getParams(x1,x2,y)
        Nx1Lvl, Nx2Lvl = dims
        s1, s2 = stdSingle
        s12 = stdMulti
        
        ### hyperpriors ### 
        # observation hyperpriors
        lamY = 1/30.
        muGamma = 0.5
        sigmaGamma = 2.
        
        # prediction hyperpriors
        sigma0 = pm.HalfNormal('sigma0',sd=1)
        sigma1 = pm.HalfNormal('sigma1',sd=s1, shape=Nx1Lvl)       
        sigma2 = pm.HalfNormal('sigma2',sd=s2, shape=Nx2Lvl)
        beta2 = (np.sqrt(6)*sigma2)/(np.pi)
        
        mu_b0 = pm.Normal('mu_b0', mu=0., sd=1)
        mu_b1 = pm.Normal('mu_b1', mu=0., sd=1, shape=Nx1Lvl)        
        mu_b2 = pm.Normal('mu_b2', mu=0., sd=1, shape=Nx2Lvl)       
        
        sigma12 = pm.HalfNormal('sigma12',sd=s12)
                               
        ### priors ### 
        # observation priors        
        nuY = pm.Exponential('nuY',lam=lamY)
        sigmaY = pm.Gamma('sigmaY',mu=muGamma, sigma=sigmaGamma)
        
        # prediction priors
        b0_dist = pm.Normal('b0_dist', mu=0, sd=1)
        b0 = pm.Deterministic("b0", mu_b0 + b0_dist * sigma0)
       
        b1_dist = pm.Normal('b1_dist', mu=0, sd=1)
        b1 = pm.Deterministic("b1", mu_b1 + b1_dist * sigma1)
                        
        b2_beta = pm.HalfNormal('b2_beta', sd=beta2, shape=Nx2Lvl)
        b2_dist = pm.Gumbel('b2_dist', mu=0, beta=1)
        b2 = pm.Deterministic("b2", mu_b2 + b2_beta * b2_dist)
        
        mu_M12 = pm.Normal('mu_M12', mu=0., sd=1, shape=[Nx1Lvl, Nx2Lvl])
        M12_dist = pm.Normal('M12_dist', mu=0, sd=1)
        M12 = pm.Deterministic("M12", mu_M12 + M12_dist * sigma12)
        
        #### prediction ###         
        mu = pm.Deterministic('mu',b0 + b1[x1]+ b2[x2] +  M12[x1,x2] )
                                        
        ### observation ### 
        y = pm.StudentT('y',nu = nuY, mu=mu, sd=sigmaY, observed=y)
```

## Inference <a name="inference"></a>

### epLsar <a name="epLsar"></a>


```python
with pm.Model() as model:
    epLsarModel = TwoFactorModel('epLsar',x1,x2,dataZ.epLsar_z.values)
```

#### Verify model settings


```python
epLsarModel.printParams(x1,x2,dataZ.epLsar_z.values)
```

    The number of levels of the x variables are (2, 11)
    The standard deviations used for the beta priors are  (1.0181033807478441, 1.4891257020651163)
    The standard deviations used for the M12 priors are 0.09019464854470463



```python
try:
    graph_epLsar = pm.model_to_graphviz(epLsarModel)    
except:
    graph_epLsar = "Could not make graph"
graph_epLsar
```




    
![svg](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_36_0.svg)
    



#### Check prior choice


```python
with epLsarModel as model:
    prior_pred_epLsar = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_epLsar,dataZ.epLsar_z.values,'epLsar')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_39_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with epLsarModel as model:
    trace_epLsar = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
    #fit_epLsar = pm.fit(random_seed=random_seed)
    #trace_epLsar = fit_epLsar.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [epLsar_M12_dist, epLsar_mu_M12, epLsar_b2_dist, epLsar_b2_beta, epLsar_b1_dist, epLsar_b0_dist, epLsar_sigmaY, epLsar_nuY, epLsar_sigma12, epLsar_mu_b2, epLsar_mu_b1, epLsar_mu_b0, epLsar_sigma2, epLsar_sigma1, epLsar_sigma0]




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
  100.00% [8000/8000 02:57<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 178 seconds.



```python
with epLsarModel as model:
    if writeOut:
        with open(outPathData + 'model_{}.pkl'.format('epLsar'), 'wb') as buff:
            pickle.dump({'model':epLsarModel, 'trace': trace_epLsar}, buff)            
```

##### Save for later comparison


```python
if writeOut:
    np.save('../derived_data/statistical_model_two_factors_filter_strong/epLsar_oldb1', trace_epLsar['epLsar_b1'])
    np.save('../derived_data/statistical_model_two_factors_filter_strong/epLsar_oldb2', trace_epLsar['epLsar_b2'])
    np.save('../derived_data/statistical_model_two_factors_filter_strong/epLsar_oldM12', trace_epLsar['epLsar_M12'])
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
      <td>-0.05</td>
      <td>0.82</td>
      <td>-1.65</td>
      <td>1.54</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>6470.0</td>
      <td>1562.0</td>
      <td>6473.0</td>
      <td>2593.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[0]</th>
      <td>0.04</td>
      <td>0.73</td>
      <td>-1.46</td>
      <td>1.38</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5416.0</td>
      <td>1540.0</td>
      <td>5398.0</td>
      <td>2587.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[1]</th>
      <td>-0.07</td>
      <td>0.74</td>
      <td>-1.55</td>
      <td>1.36</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5127.0</td>
      <td>1465.0</td>
      <td>5139.0</td>
      <td>2939.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[0]</th>
      <td>-0.15</td>
      <td>0.66</td>
      <td>-1.45</td>
      <td>1.12</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>6447.0</td>
      <td>2150.0</td>
      <td>6450.0</td>
      <td>3472.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[1]</th>
      <td>-0.12</td>
      <td>0.69</td>
      <td>-1.52</td>
      <td>1.18</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5453.0</td>
      <td>2235.0</td>
      <td>5448.0</td>
      <td>2856.0</td>
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
      <th>epLsar_mu[273]</th>
      <td>0.45</td>
      <td>0.28</td>
      <td>-0.11</td>
      <td>0.98</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3996.0</td>
      <td>3777.0</td>
      <td>4001.0</td>
      <td>3551.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[274]</th>
      <td>0.47</td>
      <td>0.32</td>
      <td>-0.13</td>
      <td>1.13</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4040.0</td>
      <td>3810.0</td>
      <td>4033.0</td>
      <td>3712.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[275]</th>
      <td>0.45</td>
      <td>0.28</td>
      <td>-0.11</td>
      <td>0.98</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3996.0</td>
      <td>3777.0</td>
      <td>4001.0</td>
      <td>3551.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[276]</th>
      <td>0.47</td>
      <td>0.32</td>
      <td>-0.13</td>
      <td>1.13</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4040.0</td>
      <td>3810.0</td>
      <td>4033.0</td>
      <td>3712.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[277]</th>
      <td>0.45</td>
      <td>0.28</td>
      <td>-0.11</td>
      <td>0.98</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3996.0</td>
      <td>3777.0</td>
      <td>4001.0</td>
      <td>3551.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>382 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_epLsar,dataTrace_epLsar,'epLsar')
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_49_1.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_49_2.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_49_3.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_49_4.png)
    



```python
with epLsarModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_epLsar,'epLsar')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_50_0.png)
    



```python
with epLsarModel as model:
    plotting_lib.pm.energyplot(trace_epLsar)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_51_0.png)
    


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


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_54_0.png)
    


#### Level plots


```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_epLsar,'epLsar',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_56_0.png)
    



```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_epLsar,'epLsar',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_57_0.png)
    


#### Posterior and contrasts


```python
df_hdi_epLsar = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_epLsar,'epLsar',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_59_0.png)
    



```python
df_hdi_epLsar
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
      <th>Treatment_i</th>
      <th>Treatment_j</th>
      <th>hdi_ConfoMap_2.5%</th>
      <th>hdi_ConfoMap_97.5%</th>
      <th>isSignificant_on_ConfoMap</th>
      <th>hdi_Toothfrax_2.5%</th>
      <th>hdi_Toothfrax_97.5%</th>
      <th>isSignificant_on_Toothfrax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dry grass</td>
      <td>Dry bamboo</td>
      <td>-0.004152</td>
      <td>-0.002594</td>
      <td>True</td>
      <td>-0.003996</td>
      <td>-0.002397</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dry lucerne</td>
      <td>Dry bamboo</td>
      <td>-0.003326</td>
      <td>-0.001788</td>
      <td>True</td>
      <td>-0.003509</td>
      <td>-0.002022</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dry lucerne</td>
      <td>Dry grass</td>
      <td>0.000057</td>
      <td>0.001571</td>
      <td>True</td>
      <td>-0.000323</td>
      <td>0.001235</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BrushNoDirt</td>
      <td>BrushDirt</td>
      <td>-0.001377</td>
      <td>0.002078</td>
      <td>False</td>
      <td>-0.001797</td>
      <td>0.001283</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Control</td>
      <td>BrushDirt</td>
      <td>-0.001051</td>
      <td>0.002062</td>
      <td>False</td>
      <td>-0.000649</td>
      <td>0.002677</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Control</td>
      <td>BrushNoDirt</td>
      <td>-0.001703</td>
      <td>0.001898</td>
      <td>False</td>
      <td>-0.000599</td>
      <td>0.002795</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RubDirt</td>
      <td>BrushDirt</td>
      <td>-0.000716</td>
      <td>0.002956</td>
      <td>False</td>
      <td>-0.001682</td>
      <td>0.001413</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RubDirt</td>
      <td>BrushNoDirt</td>
      <td>-0.001393</td>
      <td>0.002726</td>
      <td>False</td>
      <td>-0.001327</td>
      <td>0.001686</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>-0.001339</td>
      <td>0.002509</td>
      <td>False</td>
      <td>-0.002646</td>
      <td>0.000660</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clover+dust</td>
      <td>Clover</td>
      <td>-0.001123</td>
      <td>0.001394</td>
      <td>False</td>
      <td>-0.000920</td>
      <td>0.001698</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Grass</td>
      <td>Clover</td>
      <td>0.000052</td>
      <td>0.003036</td>
      <td>True</td>
      <td>0.000386</td>
      <td>0.003713</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+dust</td>
      <td>-0.000229</td>
      <td>0.002855</td>
      <td>False</td>
      <td>0.000049</td>
      <td>0.003324</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Grass+dust</td>
      <td>Clover</td>
      <td>0.000332</td>
      <td>0.003297</td>
      <td>True</td>
      <td>0.000620</td>
      <td>0.003434</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grass+dust</td>
      <td>Clover+dust</td>
      <td>0.000167</td>
      <td>0.003241</td>
      <td>True</td>
      <td>0.000285</td>
      <td>0.003140</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Grass+dust</td>
      <td>Grass</td>
      <td>-0.001428</td>
      <td>0.002001</td>
      <td>False</td>
      <td>-0.001697</td>
      <td>0.001666</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
plotting_lib.plotTreatmentPosteriorDiff(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_epLsar,'epLsar',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_61_0.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_61_1.png)
    



```python
if writeOut:
    df_hdi_epLsar.to_csv(outPathData+ 'hdi_{}.csv'.format('epLsar'))
```

### R²<a name="r"></a>


```python
with pm.Model() as model:
    RsquaredModel = TwoFactorModel('Rsquared',x1,x2,dataZ["Rsquared_z"].values)
```

#### Verify model settings


```python
RsquaredModel.printParams(x1,x2,dataZ["Rsquared_z"].values)
```

    The number of levels of the x variables are (2, 11)
    The standard deviations used for the beta priors are  (1.364226512794052, 3.833663350609328)
    The standard deviations used for the M12 priors are 0.20345815702507086



```python
pm.model_to_graphviz(RsquaredModel)
```




    
![svg](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_67_0.svg)
    



#### Check prior choice


```python
with RsquaredModel as model:
    prior_pred_Rsquared = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Rsquared,dataZ["Rsquared_z"].values,'Rsquared')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_70_0.png)
    


#### Sampling


```python
with RsquaredModel as model:
    trace_Rsquared = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [Rsquared_M12_dist, Rsquared_mu_M12, Rsquared_b2_dist, Rsquared_b2_beta, Rsquared_b1_dist, Rsquared_b0_dist, Rsquared_sigmaY, Rsquared_nuY, Rsquared_sigma12, Rsquared_mu_b2, Rsquared_mu_b1, Rsquared_mu_b0, Rsquared_sigma2, Rsquared_sigma1, Rsquared_sigma0]




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
  100.00% [8000/8000 27:08<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 1629 seconds.



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
      <td>0.09</td>
      <td>0.83</td>
      <td>-1.56</td>
      <td>1.69</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5090.0</td>
      <td>2109.0</td>
      <td>5099.0</td>
      <td>2890.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b1[0]</th>
      <td>-0.06</td>
      <td>0.74</td>
      <td>-1.53</td>
      <td>1.34</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3681.0</td>
      <td>2539.0</td>
      <td>3669.0</td>
      <td>3353.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b1[1]</th>
      <td>0.10</td>
      <td>0.72</td>
      <td>-1.36</td>
      <td>1.46</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4328.0</td>
      <td>2934.0</td>
      <td>4336.0</td>
      <td>3441.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b2[0]</th>
      <td>0.05</td>
      <td>0.67</td>
      <td>-1.25</td>
      <td>1.34</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3253.0</td>
      <td>2383.0</td>
      <td>3270.0</td>
      <td>3303.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu_b2[1]</th>
      <td>0.02</td>
      <td>0.67</td>
      <td>-1.31</td>
      <td>1.26</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3147.0</td>
      <td>2682.0</td>
      <td>3155.0</td>
      <td>2829.0</td>
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
      <td>0.30</td>
      <td>0.02</td>
      <td>0.26</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4108.0</td>
      <td>4092.0</td>
      <td>4103.0</td>
      <td>3710.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[274]</th>
      <td>0.22</td>
      <td>0.03</td>
      <td>0.16</td>
      <td>0.27</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4006.0</td>
      <td>3960.0</td>
      <td>4006.0</td>
      <td>3711.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[275]</th>
      <td>0.30</td>
      <td>0.02</td>
      <td>0.26</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4108.0</td>
      <td>4092.0</td>
      <td>4103.0</td>
      <td>3710.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[276]</th>
      <td>0.22</td>
      <td>0.03</td>
      <td>0.16</td>
      <td>0.27</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4006.0</td>
      <td>3960.0</td>
      <td>4006.0</td>
      <td>3711.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Rsquared_mu[277]</th>
      <td>0.30</td>
      <td>0.02</td>
      <td>0.26</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4108.0</td>
      <td>4092.0</td>
      <td>4103.0</td>
      <td>3710.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>382 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Rsquared,dataTrace_Rsquared,'Rsquared')
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_77_1.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_77_2.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_77_3.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_77_4.png)
    



```python
with RsquaredModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Rsquared,'Rsquared')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_78_0.png)
    



```python
with RsquaredModel as model:
    plotting_lib.pm.energyplot(trace_Rsquared)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_79_0.png)
    


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


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_82_0.png)
    


#### Compare prior and posterior for model parameters


```python
with RsquaredModel as model:
    pm_data_Rsquared = az.from_pymc3(trace=trace_Rsquared,prior=prior_pred_Rsquared,posterior_predictive=posterior_pred_Rsquared)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable Rsquared_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'Rsquared')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_85_0.png)
    



```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Rsquared,'Rsquared',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_86_0.png)
    



```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Rsquared,'Rsquared',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_87_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'Rsquared')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_89_0.png)
    



```python
df_hdi_R = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Rsquared,'Rsquared',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_90_0.png)
    



```python
df_hdi_R
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
      <th>Treatment_i</th>
      <th>Treatment_j</th>
      <th>hdi_ConfoMap_2.5%</th>
      <th>hdi_ConfoMap_97.5%</th>
      <th>isSignificant_on_ConfoMap</th>
      <th>hdi_Toothfrax_2.5%</th>
      <th>hdi_Toothfrax_97.5%</th>
      <th>isSignificant_on_Toothfrax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dry grass</td>
      <td>Dry bamboo</td>
      <td>0.000659</td>
      <td>0.001970</td>
      <td>True</td>
      <td>-0.000351</td>
      <td>0.000253</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dry lucerne</td>
      <td>Dry bamboo</td>
      <td>0.000067</td>
      <td>0.001447</td>
      <td>True</td>
      <td>-0.000407</td>
      <td>0.000144</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dry lucerne</td>
      <td>Dry grass</td>
      <td>-0.001082</td>
      <td>0.000049</td>
      <td>False</td>
      <td>-0.000384</td>
      <td>0.000276</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BrushNoDirt</td>
      <td>BrushDirt</td>
      <td>-0.001244</td>
      <td>0.000215</td>
      <td>False</td>
      <td>-0.000965</td>
      <td>0.000147</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Control</td>
      <td>BrushDirt</td>
      <td>-0.001290</td>
      <td>0.000753</td>
      <td>False</td>
      <td>-0.000634</td>
      <td>0.000352</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Control</td>
      <td>BrushNoDirt</td>
      <td>-0.000712</td>
      <td>0.001349</td>
      <td>False</td>
      <td>-0.000288</td>
      <td>0.000781</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RubDirt</td>
      <td>BrushDirt</td>
      <td>-0.001629</td>
      <td>0.000051</td>
      <td>False</td>
      <td>-0.001153</td>
      <td>0.000118</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RubDirt</td>
      <td>BrushNoDirt</td>
      <td>-0.001106</td>
      <td>0.000574</td>
      <td>False</td>
      <td>-0.000755</td>
      <td>0.000541</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>-0.001619</td>
      <td>0.000571</td>
      <td>False</td>
      <td>-0.000955</td>
      <td>0.000241</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clover+dust</td>
      <td>Clover</td>
      <td>-0.000472</td>
      <td>0.000480</td>
      <td>False</td>
      <td>-0.000234</td>
      <td>0.000504</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Grass</td>
      <td>Clover</td>
      <td>-0.000724</td>
      <td>0.000501</td>
      <td>False</td>
      <td>-0.000771</td>
      <td>0.000095</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+dust</td>
      <td>-0.000710</td>
      <td>0.000560</td>
      <td>False</td>
      <td>-0.000889</td>
      <td>-0.000103</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Grass+dust</td>
      <td>Clover</td>
      <td>-0.000485</td>
      <td>0.000552</td>
      <td>False</td>
      <td>-0.000475</td>
      <td>0.000304</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grass+dust</td>
      <td>Clover+dust</td>
      <td>-0.000508</td>
      <td>0.000541</td>
      <td>False</td>
      <td>-0.000566</td>
      <td>0.000135</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Grass+dust</td>
      <td>Grass</td>
      <td>-0.000535</td>
      <td>0.000797</td>
      <td>False</td>
      <td>-0.000193</td>
      <td>0.000637</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
plotting_lib.plotTreatmentPosteriorDiff(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Rsquared,'Rsquared',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_92_0.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_92_1.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_92_2.png)
    



```python
if writeOut:
    df_hdi_R.to_csv(outPathData+ 'hdi_{}.csv'.format('Rsquared'))
```

### Asfc  <a name="Asfc"></a>


```python
with pm.Model() as model:
    AsfcModel = TwoFactorModel('Asfc',x1,x2,dataZ["Asfc_z"].values)
```

#### Verify model settings


```python
AsfcModel.printParams(x1,x2,dataZ["Asfc_z"].values)
```

    The number of levels of the x variables are (2, 11)
    The standard deviations used for the beta priors are  (1.0834504037573414, 1.7347208877727194)
    The standard deviations used for the M12 priors are 0.10226340176080734



```python
pm.model_to_graphviz(AsfcModel)
```




    
![svg](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_98_0.svg)
    



#### Check prior choice


```python
with AsfcModel as model:
    prior_pred_Asfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Asfc,dataZ["Asfc_z"].values,'Asfc')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_101_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with AsfcModel as model:
    trace_Asfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [Asfc_M12_dist, Asfc_mu_M12, Asfc_b2_dist, Asfc_b2_beta, Asfc_b1_dist, Asfc_b0_dist, Asfc_sigmaY, Asfc_nuY, Asfc_sigma12, Asfc_mu_b2, Asfc_mu_b1, Asfc_mu_b0, Asfc_sigma2, Asfc_sigma1, Asfc_sigma0]




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
  100.00% [8000/8000 09:03<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 543 seconds.
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
      <td>-0.06</td>
      <td>0.81</td>
      <td>-1.65</td>
      <td>1.48</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4064.0</td>
      <td>2319.0</td>
      <td>4058.0</td>
      <td>3075.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[0]</th>
      <td>0.07</td>
      <td>0.73</td>
      <td>-1.39</td>
      <td>1.51</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3340.0</td>
      <td>2543.0</td>
      <td>3338.0</td>
      <td>2695.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[1]</th>
      <td>-0.11</td>
      <td>0.73</td>
      <td>-1.50</td>
      <td>1.32</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3625.0</td>
      <td>2612.0</td>
      <td>3630.0</td>
      <td>3292.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[0]</th>
      <td>1.20</td>
      <td>0.76</td>
      <td>-0.26</td>
      <td>2.66</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>2517.0</td>
      <td>2517.0</td>
      <td>2652.0</td>
      <td>2405.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[1]</th>
      <td>0.94</td>
      <td>0.73</td>
      <td>-0.42</td>
      <td>2.39</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3625.0</td>
      <td>3328.0</td>
      <td>3631.0</td>
      <td>3427.0</td>
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
      <td>-1.05</td>
      <td>0.06</td>
      <td>-1.17</td>
      <td>-0.94</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4027.0</td>
      <td>4000.0</td>
      <td>4022.0</td>
      <td>3506.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[274]</th>
      <td>-1.05</td>
      <td>0.06</td>
      <td>-1.16</td>
      <td>-0.92</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4220.0</td>
      <td>4220.0</td>
      <td>4219.0</td>
      <td>4065.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[275]</th>
      <td>-1.05</td>
      <td>0.06</td>
      <td>-1.17</td>
      <td>-0.94</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4027.0</td>
      <td>4000.0</td>
      <td>4022.0</td>
      <td>3506.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[276]</th>
      <td>-1.05</td>
      <td>0.06</td>
      <td>-1.16</td>
      <td>-0.92</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4220.0</td>
      <td>4220.0</td>
      <td>4219.0</td>
      <td>4065.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[277]</th>
      <td>-1.05</td>
      <td>0.06</td>
      <td>-1.17</td>
      <td>-0.94</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4027.0</td>
      <td>4000.0</td>
      <td>4022.0</td>
      <td>3506.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>382 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,dataTrace_Asfc,'Asfc')
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_109_1.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_109_2.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_109_3.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_109_4.png)
    



```python
with AsfcModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,'Asfc')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_110_0.png)
    



```python
with AsfcModel as model:
    plotting_lib.pm.energyplot(trace_Asfc)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_111_0.png)
    


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


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_114_0.png)
    



```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Asfc,'Asfc',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_115_0.png)
    



```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Asfc,'Asfc',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_116_0.png)
    


#### Compare prior and posterior for model parameters


```python
with AsfcModel as model:
    pm_data_Asfc = az.from_pymc3(trace=trace_Asfc,prior=prior_pred_Asfc,posterior_predictive=posterior_pred_Asfc)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable Asfc_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_119_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_121_0.png)
    



```python
df_hdi_Asfc = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Asfc,'Asfc',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_122_0.png)
    



```python
df_hdi_Asfc
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
      <th>Treatment_i</th>
      <th>Treatment_j</th>
      <th>hdi_ConfoMap_2.5%</th>
      <th>hdi_ConfoMap_97.5%</th>
      <th>isSignificant_on_ConfoMap</th>
      <th>hdi_Toothfrax_2.5%</th>
      <th>hdi_Toothfrax_97.5%</th>
      <th>isSignificant_on_Toothfrax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dry grass</td>
      <td>Dry bamboo</td>
      <td>-3.916452</td>
      <td>2.795973</td>
      <td>False</td>
      <td>-4.296147</td>
      <td>1.377498</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dry lucerne</td>
      <td>Dry bamboo</td>
      <td>-4.260454</td>
      <td>1.899807</td>
      <td>False</td>
      <td>-4.704579</td>
      <td>0.667324</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dry lucerne</td>
      <td>Dry grass</td>
      <td>-2.615826</td>
      <td>1.717622</td>
      <td>False</td>
      <td>-2.195671</td>
      <td>1.343382</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BrushNoDirt</td>
      <td>BrushDirt</td>
      <td>-15.010884</td>
      <td>3.380206</td>
      <td>False</td>
      <td>-13.605707</td>
      <td>4.419980</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Control</td>
      <td>BrushDirt</td>
      <td>-35.731852</td>
      <td>4.275047</td>
      <td>False</td>
      <td>-31.670087</td>
      <td>3.675433</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Control</td>
      <td>BrushNoDirt</td>
      <td>-34.009450</td>
      <td>14.409722</td>
      <td>False</td>
      <td>-30.388298</td>
      <td>13.091720</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RubDirt</td>
      <td>BrushDirt</td>
      <td>-42.757337</td>
      <td>-32.042331</td>
      <td>True</td>
      <td>-35.972142</td>
      <td>-23.722769</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RubDirt</td>
      <td>BrushNoDirt</td>
      <td>-42.529481</td>
      <td>-21.978611</td>
      <td>True</td>
      <td>-36.265074</td>
      <td>-14.506282</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>-43.365126</td>
      <td>-0.792099</td>
      <td>True</td>
      <td>-36.453023</td>
      <td>2.421650</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clover+dust</td>
      <td>Clover</td>
      <td>-4.429319</td>
      <td>2.363512</td>
      <td>False</td>
      <td>-4.325436</td>
      <td>2.025695</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Grass</td>
      <td>Clover</td>
      <td>-5.539381</td>
      <td>1.270810</td>
      <td>False</td>
      <td>-5.413694</td>
      <td>0.927516</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+dust</td>
      <td>-3.510605</td>
      <td>1.055924</td>
      <td>False</td>
      <td>-3.020075</td>
      <td>1.235952</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Grass+dust</td>
      <td>Clover</td>
      <td>-6.448750</td>
      <td>0.127469</td>
      <td>False</td>
      <td>-6.012465</td>
      <td>0.356649</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grass+dust</td>
      <td>Clover+dust</td>
      <td>-4.200499</td>
      <td>0.142961</td>
      <td>False</td>
      <td>-3.783693</td>
      <td>0.366821</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Grass+dust</td>
      <td>Grass</td>
      <td>-3.274366</td>
      <td>1.239253</td>
      <td>False</td>
      <td>-2.811040</td>
      <td>1.448578</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
plotting_lib.plotTreatmentPosteriorDiff(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Asfc,'Asfc',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_124_0.png)
    



```python
if writeOut:
    df_hdi_Asfc.to_csv(outPathData+ 'hdi_{}.csv'.format('Asfc'))
```

### 	Smfc  <a name="Smfc"></a>


```python
with pm.Model() as model:
    SmfcModel = TwoFactorModel('Smfc',x1,x2,dataZ.Smfc_z.values)
```

#### Verify model settings


```python
SmfcModel.printParams(x1,x2,dataZ.Smfc_z.values)
```

    The number of levels of the x variables are (2, 11)
    The standard deviations used for the beta priors are  (1.2903046708375803, 3.554940942907768)
    The standard deviations used for the M12 priors are 0.18909317313924404



```python
pm.model_to_graphviz(SmfcModel)
```




    
![svg](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_130_0.svg)
    



#### Check prior choice


```python
with SmfcModel as model:
    prior_pred_Smfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Smfc,dataZ.Smfc_z.values,'Smfc')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_133_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with SmfcModel as model:
    trace_Smfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [Smfc_M12_dist, Smfc_mu_M12, Smfc_b2_dist, Smfc_b2_beta, Smfc_b1_dist, Smfc_b0_dist, Smfc_sigmaY, Smfc_nuY, Smfc_sigma12, Smfc_mu_b2, Smfc_mu_b1, Smfc_mu_b0, Smfc_sigma2, Smfc_sigma1, Smfc_sigma0]




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
  <progress value='961' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  12.01% [961/8000 03:51<28:18 Sampling 4 chains, 0 divergences]
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

    <ipython-input-93-4de91012caf3> in <module>
          1 with SmfcModel as model:
    ----> 2     trace_Smfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
    

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
    HAsfc9Model = TwoFactorModel('HAsfc9',x1,x2,dataZ["HAsfc9_z"].values)
```

#### Verify model settings


```python
HAsfc9Model.printParams(x1,x2,dataZ["HAsfc9_z"].values)
```

    The number of levels of the x variables are (2, 11)
    The standard deviations used for the beta priors are  (1.2742199511468122, 3.5115369692183998)
    The standard deviations used for the M12 priors are 0.18677880213027476



```python
pm.model_to_graphviz(HAsfc9Model)
```




    
![svg](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_142_0.svg)
    



#### Check prior choice


```python
with HAsfc9Model as model:
    prior_pred_HAsfc9 = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc9,dataZ["HAsfc9_z"].values,'HAsfc9')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_145_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with HAsfc9Model as model:
    trace_HAsfc9 = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [HAsfc9_M12_dist, HAsfc9_mu_M12, HAsfc9_b2_dist, HAsfc9_b2_beta, HAsfc9_b1_dist, HAsfc9_b0_dist, HAsfc9_sigmaY, HAsfc9_nuY, HAsfc9_sigma12, HAsfc9_mu_b2, HAsfc9_mu_b1, HAsfc9_mu_b0, HAsfc9_sigma2, HAsfc9_sigma1, HAsfc9_sigma0]




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
  100.00% [8000/8000 14:33<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 874 seconds.



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
      <td>-0.06</td>
      <td>0.84</td>
      <td>-1.81</td>
      <td>1.49</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4513.0</td>
      <td>1991.0</td>
      <td>4527.0</td>
      <td>2921.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[0]</th>
      <td>-0.01</td>
      <td>0.74</td>
      <td>-1.39</td>
      <td>1.48</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3933.0</td>
      <td>2226.0</td>
      <td>3929.0</td>
      <td>2942.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[1]</th>
      <td>-0.04</td>
      <td>0.74</td>
      <td>-1.51</td>
      <td>1.42</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4126.0</td>
      <td>2501.0</td>
      <td>4125.0</td>
      <td>3332.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[0]</th>
      <td>-0.13</td>
      <td>0.66</td>
      <td>-1.46</td>
      <td>1.15</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3202.0</td>
      <td>2835.0</td>
      <td>3195.0</td>
      <td>3082.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[1]</th>
      <td>-0.07</td>
      <td>0.67</td>
      <td>-1.38</td>
      <td>1.21</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3094.0</td>
      <td>2469.0</td>
      <td>3090.0</td>
      <td>3108.0</td>
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
      <th>HAsfc9_mu[273]</th>
      <td>-0.18</td>
      <td>0.06</td>
      <td>-0.28</td>
      <td>-0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4055.0</td>
      <td>3970.0</td>
      <td>4041.0</td>
      <td>3683.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[274]</th>
      <td>-0.15</td>
      <td>0.07</td>
      <td>-0.29</td>
      <td>-0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3918.0</td>
      <td>3610.0</td>
      <td>3927.0</td>
      <td>3528.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[275]</th>
      <td>-0.18</td>
      <td>0.06</td>
      <td>-0.28</td>
      <td>-0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4055.0</td>
      <td>3970.0</td>
      <td>4041.0</td>
      <td>3683.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[276]</th>
      <td>-0.15</td>
      <td>0.07</td>
      <td>-0.29</td>
      <td>-0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3918.0</td>
      <td>3610.0</td>
      <td>3927.0</td>
      <td>3528.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[277]</th>
      <td>-0.18</td>
      <td>0.06</td>
      <td>-0.28</td>
      <td>-0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4055.0</td>
      <td>3970.0</td>
      <td>4041.0</td>
      <td>3683.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>382 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,dataTrace_HAsfc9,'HAsfc9')
```

    /home/bob/Documents/Projekt_Neuwied/SSFA/ssfa-env/lib/python3.7/site-packages/arviz/plots/backends/matplotlib/pairplot.py:216: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      UserWarning,



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_153_1.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_153_2.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_153_3.png)
    



    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_153_4.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_154_0.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.pm.energyplot(trace_HAsfc9)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_155_0.png)
    


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


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_158_0.png)
    



```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc9,'HAsfc9',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_159_0.png)
    



```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc9,'HAsfc9',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_160_0.png)
    


#### Compare prior and posterior for model parameters


```python
with HAsfc9Model as model:
    pm_data_HAsfc9 = az.from_pymc3(trace=trace_HAsfc9,prior=prior_pred_HAsfc9,posterior_predictive=posterior_pred_HAsfc9)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable HAsfc9_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_163_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_165_0.png)
    



```python
df_hdi_HAsfc9 = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc9,'HAsfc9',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_filter_weak_files/Statistical_Model_TwoFactor_filter_weak_166_0.png)
    



```python
df_hdi_HAsfc9
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
      <th>Treatment_i</th>
      <th>Treatment_j</th>
      <th>hdi_ConfoMap_2.5%</th>
      <th>hdi_ConfoMap_97.5%</th>
      <th>isSignificant_on_ConfoMap</th>
      <th>hdi_Toothfrax_2.5%</th>
      <th>hdi_Toothfrax_97.5%</th>
      <th>isSignificant_on_Toothfrax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dry grass</td>
      <td>Dry bamboo</td>
      <td>-0.010268</td>
      <td>0.143011</td>
      <td>False</td>
      <td>-0.010625</td>
      <td>0.126663</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dry lucerne</td>
      <td>Dry bamboo</td>
      <td>-0.049642</td>
      <td>0.073424</td>
      <td>False</td>
      <td>-0.043776</td>
      <td>0.064924</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dry lucerne</td>
      <td>Dry grass</td>
      <td>-0.133646</td>
      <td>0.031096</td>
      <td>False</td>
      <td>-0.122211</td>
      <td>0.028118</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BrushNoDirt</td>
      <td>BrushDirt</td>
      <td>-0.125128</td>
      <td>0.238569</td>
      <td>False</td>
      <td>-0.140860</td>
      <td>0.229987</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Control</td>
      <td>BrushDirt</td>
      <td>-0.020567</td>
      <td>0.415332</td>
      <td>False</td>
      <td>-0.066013</td>
      <td>0.448612</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Control</td>
      <td>BrushNoDirt</td>
      <td>-0.120627</td>
      <td>0.368658</td>
      <td>False</td>
      <td>-0.139664</td>
      <td>0.449018</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RubDirt</td>
      <td>BrushDirt</td>
      <td>-0.106453</td>
      <td>0.178762</td>
      <td>False</td>
      <td>-0.142743</td>
      <td>0.149182</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RubDirt</td>
      <td>BrushNoDirt</td>
      <td>-0.209790</td>
      <td>0.142115</td>
      <td>False</td>
      <td>-0.242017</td>
      <td>0.174436</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>-0.368504</td>
      <td>0.063584</td>
      <td>False</td>
      <td>-0.482366</td>
      <td>0.066704</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clover+dust</td>
      <td>Clover</td>
      <td>-0.132341</td>
      <td>0.158945</td>
      <td>False</td>
      <td>-0.086217</td>
      <td>0.173111</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Grass</td>
      <td>Clover</td>
      <td>-0.286952</td>
      <td>0.065272</td>
      <td>False</td>
      <td>-0.128695</td>
      <td>0.125433</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+dust</td>
      <td>-0.281473</td>
      <td>0.032155</td>
      <td>False</td>
      <td>-0.168734</td>
      <td>0.086427</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Grass+dust</td>
      <td>Clover</td>
      <td>-0.319844</td>
      <td>0.007649</td>
      <td>False</td>
      <td>-0.216295</td>
      <td>0.043416</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grass+dust</td>
      <td>Clover+dust</td>
      <td>-0.300752</td>
      <td>-0.021022</td>
      <td>True</td>
      <td>-0.245888</td>
      <td>-0.001685</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Grass+dust</td>
      <td>Grass</td>
      <td>-0.223490</td>
      <td>0.131237</td>
      <td>False</td>
      <td>-0.206777</td>
      <td>0.039294</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
if writeOut:
    df_hdi_HAsfc9.to_csv(outPathData+ 'hdi_{}.csv'.format('HAsfc9'))
```


```python
plotting_lib.plotTreatmentPosteriorDiff(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc9,'HAsfc9',x1,x2)
```

## Summary<a name="summary"></a>

Set the surface parameters for every treatment dataframe:


```python
df_hdi_Asfc["SurfaceParameter"] = "Asfc"
df_hdi_HAsfc9["SurfaceParameter"] = "HAsfc9"
df_hdi_R["SurfaceParameter"] = "R²"
df_hdi_epLsar["SurfaceParameter"] = "epLsar"
```


```python
df_hdi_total = pd.concat([df_hdi_epLsar,df_hdi_R,df_hdi_Asfc,df_hdi_HAsfc9],ignore_index=True)
```

Show the treatment pairs and surface parameters where the softwares differ


```python
df_summary = df_hdi_total[df_hdi_total.isSignificant_on_ConfoMap != df_hdi_total.isSignificant_on_Toothfrax][["Treatment_i","Treatment_j","SurfaceParameter","isSignificant_on_ConfoMap","isSignificant_on_Toothfrax","hdi_ConfoMap_2.5%","hdi_ConfoMap_97.5%","hdi_Toothfrax_2.5%","hdi_Toothfrax_97.5%"]]
df_summary
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
      <th>Treatment_i</th>
      <th>Treatment_j</th>
      <th>SurfaceParameter</th>
      <th>isSignificant_on_ConfoMap</th>
      <th>isSignificant_on_Toothfrax</th>
      <th>hdi_ConfoMap_2.5%</th>
      <th>hdi_ConfoMap_97.5%</th>
      <th>hdi_Toothfrax_2.5%</th>
      <th>hdi_Toothfrax_97.5%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Dry lucerne</td>
      <td>Dry grass</td>
      <td>epLsar</td>
      <td>True</td>
      <td>False</td>
      <td>0.000057</td>
      <td>0.001571</td>
      <td>-0.000323</td>
      <td>0.001235</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+dust</td>
      <td>epLsar</td>
      <td>False</td>
      <td>True</td>
      <td>-0.000229</td>
      <td>0.002855</td>
      <td>0.000049</td>
      <td>0.003324</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Dry grass</td>
      <td>Dry bamboo</td>
      <td>R²</td>
      <td>True</td>
      <td>False</td>
      <td>0.000659</td>
      <td>0.001970</td>
      <td>-0.000351</td>
      <td>0.000253</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Dry lucerne</td>
      <td>Dry bamboo</td>
      <td>R²</td>
      <td>True</td>
      <td>False</td>
      <td>0.000067</td>
      <td>0.001447</td>
      <td>-0.000407</td>
      <td>0.000144</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Grass</td>
      <td>Clover+dust</td>
      <td>R²</td>
      <td>False</td>
      <td>True</td>
      <td>-0.000710</td>
      <td>0.000560</td>
      <td>-0.000889</td>
      <td>-0.000103</td>
    </tr>
    <tr>
      <th>38</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>Asfc</td>
      <td>True</td>
      <td>False</td>
      <td>-43.365126</td>
      <td>-0.792099</td>
      <td>-36.453023</td>
      <td>2.421650</td>
    </tr>
  </tbody>
</table>
</div>




```python
if writeOut:
    df_summary.to_csv(outPathData+ 'summary_filter_weak.csv')
```

### Write out


```python
!jupyter nbconvert --to html Statistical_Model_TwoFactor_filter_weak.ipynb
```


```python
!jupyter nbconvert --to markdown Statistical_Model_TwoFactor_filter_weak.ipynb
```


```python

































```
