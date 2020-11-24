# Analysis for SSFA project: Two factor model

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
   1. [HAsfc81](#HAsfc81)       

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
outPathPlots = "../plots/statistical_model_two_factors/"
outPathData = "../derived_data/statistical_model_two_factors/"
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
numSamples = 500
numCores = 10
numTune = 1000
numPredSamples = 2000
random_seed=3651435
```

## Load data <a name="load"></a>


```python
datafile = "../derived_data/preprocessing/preprocessed.dat"
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
    <tr>
      <th>278</th>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>279</th>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>280 rows × 2 columns</p>
</div>


x1 indicates the software used, x2 indicates the treatment applied.


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
    The standard deviations used for the beta priors are  (1.0034852973454325, 1.2901096312620055)
    The standard deviations used for the M12 priors are 0.08172156390242785



```python
try:
    graph_epLsar = pm.model_to_graphviz(epLsarModel)    
except:
    graph_epLsar = "Could not make graph"
graph_epLsar
```




    
![svg](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_36_0.svg)
    



#### Check prior choice


```python
with epLsarModel as model:
    prior_pred_epLsar = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_epLsar,dataZ.epLsar_z.values,'epLsar')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_39_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with epLsarModel as model:
    trace_epLsar = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.95,random_seed=random_seed)
    #fit_epLsar = pm.fit(random_seed=random_seed)
    #trace_epLsar = fit_epLsar.sample(draws=numSamples)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
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
  <progress value='15000' class='' max='15000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [15000/15000 03:11<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 500 draw iterations (10_000 + 5_000 draws total) took 193 seconds.



```python
with epLsarModel as model:
    if writeOut:
        with open(outPathData + 'model_{}.pkl'.format('epLsar'), 'wb') as buff:
            pickle.dump({'model':epLsarModel, 'trace': trace_epLsar}, buff)            
```

##### Save for later comparison


```python
if writeOut:
    np.save('../derived_data/statistical_model_two_factors/epLsar_oldb1', trace_epLsar['epLsar_b1'])
    np.save('../derived_data/statistical_model_two_factors/epLsar_oldb2', trace_epLsar['epLsar_b2'])
    np.save('../derived_data/statistical_model_two_factors/epLsar_oldM12', trace_epLsar['epLsar_M12'])
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
      <td>-0.09</td>
      <td>0.79</td>
      <td>-1.70</td>
      <td>1.44</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5425.0</td>
      <td>2884.0</td>
      <td>5427.0</td>
      <td>3762.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[0]</th>
      <td>-0.03</td>
      <td>0.70</td>
      <td>-1.39</td>
      <td>1.39</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4860.0</td>
      <td>2747.0</td>
      <td>4867.0</td>
      <td>3934.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b1[1]</th>
      <td>-0.07</td>
      <td>0.71</td>
      <td>-1.53</td>
      <td>1.25</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5257.0</td>
      <td>2785.0</td>
      <td>5261.0</td>
      <td>3982.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[0]</th>
      <td>-0.25</td>
      <td>0.68</td>
      <td>-1.59</td>
      <td>1.06</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4407.0</td>
      <td>3292.0</td>
      <td>4401.0</td>
      <td>4000.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu_b2[1]</th>
      <td>-0.36</td>
      <td>0.67</td>
      <td>-1.70</td>
      <td>0.93</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4184.0</td>
      <td>3665.0</td>
      <td>4184.0</td>
      <td>3778.0</td>
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
      <td>0.68</td>
      <td>0.27</td>
      <td>0.15</td>
      <td>1.19</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5046.0</td>
      <td>4965.0</td>
      <td>5046.0</td>
      <td>4460.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[276]</th>
      <td>0.72</td>
      <td>0.31</td>
      <td>0.12</td>
      <td>1.32</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4876.0</td>
      <td>4562.0</td>
      <td>4894.0</td>
      <td>3579.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[277]</th>
      <td>0.68</td>
      <td>0.27</td>
      <td>0.15</td>
      <td>1.19</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5046.0</td>
      <td>4965.0</td>
      <td>5046.0</td>
      <td>4460.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[278]</th>
      <td>0.72</td>
      <td>0.31</td>
      <td>0.12</td>
      <td>1.32</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4876.0</td>
      <td>4562.0</td>
      <td>4894.0</td>
      <td>3579.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>epLsar_mu[279]</th>
      <td>0.68</td>
      <td>0.27</td>
      <td>0.15</td>
      <td>1.19</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5046.0</td>
      <td>4965.0</td>
      <td>5046.0</td>
      <td>4460.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>384 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_epLsar,dataTrace_epLsar,'epLsar')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_1.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_2.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_3.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_49_4.png)
    



```python
with epLsarModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_epLsar,'epLsar')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_50_0.png)
    



```python
with epLsarModel as model:
    plotting_lib.pm.energyplot(trace_epLsar)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_51_0.png)
    


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


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_54_0.png)
    


#### Level plots


```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_epLsar,'epLsar',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_56_0.png)
    



```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_epLsar,'epLsar',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_57_0.png)
    


#### Posterior and contrasts


```python
df_hdi_epLsar = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_epLsar,'epLsar',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_59_0.png)
    



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
      <td>Dry Grass</td>
      <td>Dry Bamboo</td>
      <td>-0.004067</td>
      <td>-0.002744</td>
      <td>True</td>
      <td>-0.003976</td>
      <td>-0.002647</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dry Lucerne</td>
      <td>Dry Bamboo</td>
      <td>-0.003291</td>
      <td>-0.001971</td>
      <td>True</td>
      <td>-0.003465</td>
      <td>-0.002210</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dry Lucerne</td>
      <td>Dry Grass</td>
      <td>0.000215</td>
      <td>0.001528</td>
      <td>True</td>
      <td>-0.000166</td>
      <td>0.001127</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BrushNoDirt</td>
      <td>BrushDirt</td>
      <td>-0.001322</td>
      <td>0.001007</td>
      <td>False</td>
      <td>-0.001513</td>
      <td>0.000565</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Control</td>
      <td>BrushDirt</td>
      <td>-0.002129</td>
      <td>0.000235</td>
      <td>False</td>
      <td>-0.001954</td>
      <td>0.000213</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Control</td>
      <td>BrushNoDirt</td>
      <td>-0.001726</td>
      <td>0.000381</td>
      <td>False</td>
      <td>-0.001382</td>
      <td>0.000666</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RubDirt</td>
      <td>BrushDirt</td>
      <td>-0.001713</td>
      <td>0.000537</td>
      <td>False</td>
      <td>-0.001373</td>
      <td>0.000730</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RubDirt</td>
      <td>BrushNoDirt</td>
      <td>-0.001426</td>
      <td>0.000616</td>
      <td>False</td>
      <td>-0.000906</td>
      <td>0.001069</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>-0.000705</td>
      <td>0.001380</td>
      <td>False</td>
      <td>-0.000561</td>
      <td>0.001490</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clover+Dust</td>
      <td>Clover</td>
      <td>-0.001075</td>
      <td>0.001140</td>
      <td>False</td>
      <td>-0.000765</td>
      <td>0.001629</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Grass</td>
      <td>Clover</td>
      <td>-0.000194</td>
      <td>0.002930</td>
      <td>False</td>
      <td>0.000277</td>
      <td>0.003761</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+Dust</td>
      <td>-0.000132</td>
      <td>0.003035</td>
      <td>False</td>
      <td>-0.000093</td>
      <td>0.003438</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Grass+Dust</td>
      <td>Clover</td>
      <td>0.000526</td>
      <td>0.003260</td>
      <td>True</td>
      <td>0.000854</td>
      <td>0.003494</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grass+Dust</td>
      <td>Clover+Dust</td>
      <td>0.000522</td>
      <td>0.003389</td>
      <td>True</td>
      <td>0.000470</td>
      <td>0.003078</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Grass+Dust</td>
      <td>Grass</td>
      <td>-0.001254</td>
      <td>0.002331</td>
      <td>False</td>
      <td>-0.001810</td>
      <td>0.001876</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
if writeOut:
    df_hdi_epLsar.to_csv(outPathData+ 'hdi_{}.csv'.format('epLsar'))
```

### R²<a name="r"></a>


```python
with pm.Model() as model:
    RsquaredModel = TwoFactorModel('R²',x1,x2,dataZ["R²_z"].values)
```

#### Verify model settings


```python
RsquaredModel.printParams(x1,x2,dataZ["R²_z"].values)
```

    The number of levels of the x variables are (2, 11)
    The standard deviations used for the beta priors are  (1.0112351887037512, 2.890846695679254)
    The standard deviations used for the M12 priors are 0.1531305915288677



```python
pm.model_to_graphviz(RsquaredModel)
```




    
![svg](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_66_0.svg)
    



#### Check prior choice


```python
with RsquaredModel as model:
    prior_pred_Rsquared = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Rsquared,dataZ["R²_z"].values,'R²')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_69_0.png)
    


#### Sampling


```python
with RsquaredModel as model:
    trace_Rsquared = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
    NUTS: [R²_M12_dist, R²_mu_M12, R²_b2_dist, R²_b2_beta, R²_b1_dist, R²_b0_dist, R²_sigmaY, R²_nuY, R²_sigma12, R²_mu_b2, R²_mu_b1, R²_mu_b0, R²_sigma2, R²_sigma1, R²_sigma0]




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
  <progress value='15000' class='' max='15000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [15000/15000 12:10<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 500 draw iterations (10_000 + 5_000 draws total) took 732 seconds.



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
      <td>0.06</td>
      <td>0.81</td>
      <td>-1.52</td>
      <td>1.64</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5502.0</td>
      <td>2696.0</td>
      <td>5518.0</td>
      <td>3915.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>R²_mu_b1[0]</th>
      <td>-0.21</td>
      <td>0.72</td>
      <td>-1.61</td>
      <td>1.24</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4648.0</td>
      <td>3068.0</td>
      <td>4644.0</td>
      <td>3840.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>R²_mu_b1[1]</th>
      <td>0.27</td>
      <td>0.72</td>
      <td>-1.10</td>
      <td>1.74</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5066.0</td>
      <td>3627.0</td>
      <td>5073.0</td>
      <td>4140.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>R²_mu_b2[0]</th>
      <td>-0.01</td>
      <td>0.67</td>
      <td>-1.34</td>
      <td>1.25</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>3977.0</td>
      <td>2832.0</td>
      <td>3960.0</td>
      <td>3601.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>R²_mu_b2[1]</th>
      <td>-0.03</td>
      <td>0.66</td>
      <td>-1.30</td>
      <td>1.25</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4154.0</td>
      <td>3450.0</td>
      <td>4155.0</td>
      <td>3974.0</td>
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
      <th>R²_mu[275]</th>
      <td>0.53</td>
      <td>0.07</td>
      <td>0.39</td>
      <td>0.68</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4891.0</td>
      <td>4891.0</td>
      <td>4951.0</td>
      <td>4145.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>R²_mu[276]</th>
      <td>-0.01</td>
      <td>0.12</td>
      <td>-0.23</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4383.0</td>
      <td>2619.0</td>
      <td>4827.0</td>
      <td>4134.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>R²_mu[277]</th>
      <td>0.53</td>
      <td>0.07</td>
      <td>0.39</td>
      <td>0.68</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4891.0</td>
      <td>4891.0</td>
      <td>4951.0</td>
      <td>4145.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>R²_mu[278]</th>
      <td>-0.01</td>
      <td>0.12</td>
      <td>-0.23</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4383.0</td>
      <td>2619.0</td>
      <td>4827.0</td>
      <td>4134.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>R²_mu[279]</th>
      <td>0.53</td>
      <td>0.07</td>
      <td>0.39</td>
      <td>0.68</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4891.0</td>
      <td>4891.0</td>
      <td>4951.0</td>
      <td>4145.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>384 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Rsquared,dataTrace_Rsquared,'R²')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_1.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_2.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_3.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_76_4.png)
    



```python
with RsquaredModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Rsquared,'R²')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_77_0.png)
    



```python
with RsquaredModel as model:
    plotting_lib.pm.energyplot(trace_Rsquared)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_78_0.png)
    


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


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_81_0.png)
    


#### Compare prior and posterior for model parameters


```python
with RsquaredModel as model:
    pm_data_Rsquared = az.from_pymc3(trace=trace_Rsquared,prior=prior_pred_Rsquared,posterior_predictive=posterior_pred_Rsquared)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable R²_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'R²')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_84_0.png)
    



```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Rsquared,'R²',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_85_0.png)
    



```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Rsquared,'R²',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_86_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Rsquared,'R²')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_88_0.png)
    



```python
df_hdi_R = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Rsquared,'R²',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_89_0.png)
    



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
      <td>Dry Grass</td>
      <td>Dry Bamboo</td>
      <td>-0.000170</td>
      <td>0.000357</td>
      <td>False</td>
      <td>-0.000320</td>
      <td>0.000205</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dry Lucerne</td>
      <td>Dry Bamboo</td>
      <td>-0.000111</td>
      <td>0.000435</td>
      <td>False</td>
      <td>-0.000353</td>
      <td>0.000114</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dry Lucerne</td>
      <td>Dry Grass</td>
      <td>-0.000182</td>
      <td>0.000285</td>
      <td>False</td>
      <td>-0.000338</td>
      <td>0.000222</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BrushNoDirt</td>
      <td>BrushDirt</td>
      <td>-0.000799</td>
      <td>0.000192</td>
      <td>False</td>
      <td>-0.000403</td>
      <td>0.000604</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Control</td>
      <td>BrushDirt</td>
      <td>-0.000806</td>
      <td>0.000270</td>
      <td>False</td>
      <td>-0.000464</td>
      <td>0.000426</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Control</td>
      <td>BrushNoDirt</td>
      <td>-0.000439</td>
      <td>0.000502</td>
      <td>False</td>
      <td>-0.000690</td>
      <td>0.000398</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RubDirt</td>
      <td>BrushDirt</td>
      <td>-0.000495</td>
      <td>0.000548</td>
      <td>False</td>
      <td>-0.000228</td>
      <td>0.000622</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RubDirt</td>
      <td>BrushNoDirt</td>
      <td>-0.000164</td>
      <td>0.000742</td>
      <td>False</td>
      <td>-0.000409</td>
      <td>0.000594</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>-0.000214</td>
      <td>0.000759</td>
      <td>False</td>
      <td>-0.000243</td>
      <td>0.000693</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clover+Dust</td>
      <td>Clover</td>
      <td>-0.000156</td>
      <td>0.000715</td>
      <td>False</td>
      <td>-0.000183</td>
      <td>0.000480</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Grass</td>
      <td>Clover</td>
      <td>-0.000479</td>
      <td>0.000628</td>
      <td>False</td>
      <td>-0.000705</td>
      <td>0.000042</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+Dust</td>
      <td>-0.000742</td>
      <td>0.000319</td>
      <td>False</td>
      <td>-0.000812</td>
      <td>-0.000138</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Grass+Dust</td>
      <td>Clover</td>
      <td>-0.000138</td>
      <td>0.000808</td>
      <td>False</td>
      <td>-0.000443</td>
      <td>0.000248</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grass+Dust</td>
      <td>Clover+Dust</td>
      <td>-0.000456</td>
      <td>0.000426</td>
      <td>False</td>
      <td>-0.000563</td>
      <td>0.000042</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Grass+Dust</td>
      <td>Grass</td>
      <td>-0.000380</td>
      <td>0.000764</td>
      <td>False</td>
      <td>-0.000120</td>
      <td>0.000575</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
if writeOut:
    df_hdi_R.to_csv(outPathData+ 'hdi_{}.csv'.format('R²'))
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
    The standard deviations used for the beta priors are  (1.0422213301584404, 0.6910746316178924)
    The standard deviations used for the M12 priors are 0.06252618346546977



```python
pm.model_to_graphviz(AsfcModel)
```




    
![svg](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_96_0.svg)
    



#### Check prior choice


```python
with AsfcModel as model:
    prior_pred_Asfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Asfc,dataZ["Asfc_z"].values,'Asfc')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_99_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with AsfcModel as model:
    trace_Asfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
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
  <progress value='15000' class='' max='15000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [15000/15000 15:30<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 500 draw iterations (10_000 + 5_000 draws total) took 931 seconds.



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
      <td>0.02</td>
      <td>0.80</td>
      <td>-1.56</td>
      <td>1.56</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>6414.0</td>
      <td>2566.0</td>
      <td>6418.0</td>
      <td>3790.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[0]</th>
      <td>0.00</td>
      <td>0.73</td>
      <td>-1.45</td>
      <td>1.44</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5801.0</td>
      <td>2387.0</td>
      <td>5814.0</td>
      <td>3868.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b1[1]</th>
      <td>0.02</td>
      <td>0.72</td>
      <td>-1.42</td>
      <td>1.41</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5774.0</td>
      <td>2489.0</td>
      <td>5858.0</td>
      <td>3660.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[0]</th>
      <td>0.79</td>
      <td>0.67</td>
      <td>-0.49</td>
      <td>2.13</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5568.0</td>
      <td>4274.0</td>
      <td>5588.0</td>
      <td>3081.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu_b2[1]</th>
      <td>0.82</td>
      <td>0.68</td>
      <td>-0.47</td>
      <td>2.17</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5994.0</td>
      <td>4271.0</td>
      <td>5950.0</td>
      <td>3934.0</td>
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
      <td>-0.99</td>
      <td>0.06</td>
      <td>-1.10</td>
      <td>-0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4965.0</td>
      <td>4965.0</td>
      <td>4968.0</td>
      <td>4536.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[276]</th>
      <td>-1.00</td>
      <td>0.06</td>
      <td>-1.11</td>
      <td>-0.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4854.0</td>
      <td>4854.0</td>
      <td>4859.0</td>
      <td>4733.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[277]</th>
      <td>-0.99</td>
      <td>0.06</td>
      <td>-1.10</td>
      <td>-0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4965.0</td>
      <td>4965.0</td>
      <td>4968.0</td>
      <td>4536.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[278]</th>
      <td>-1.00</td>
      <td>0.06</td>
      <td>-1.11</td>
      <td>-0.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4854.0</td>
      <td>4854.0</td>
      <td>4859.0</td>
      <td>4733.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Asfc_mu[279]</th>
      <td>-0.99</td>
      <td>0.06</td>
      <td>-1.10</td>
      <td>-0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4965.0</td>
      <td>4965.0</td>
      <td>4968.0</td>
      <td>4536.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>384 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,dataTrace_Asfc,'Asfc')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_1.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_2.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_3.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_107_4.png)
    



```python
with AsfcModel as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_Asfc,'Asfc')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_108_0.png)
    



```python
with AsfcModel as model:
    plotting_lib.pm.energyplot(trace_Asfc)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_109_0.png)
    


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


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_112_0.png)
    



```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Asfc,'Asfc',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_113_0.png)
    



```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Asfc,'Asfc',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_114_0.png)
    


#### Compare prior and posterior for model parameters


```python
with AsfcModel as model:
    pm_data_Asfc = az.from_pymc3(trace=trace_Asfc,prior=prior_pred_Asfc,posterior_predictive=posterior_pred_Asfc)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable Asfc_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_117_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_Asfc,'Asfc')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_119_0.png)
    



```python
df_hdi_Asfc = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Asfc,'Asfc',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_120_0.png)
    



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
      <td>Dry Grass</td>
      <td>Dry Bamboo</td>
      <td>-4.590184</td>
      <td>1.147175</td>
      <td>False</td>
      <td>-4.558174</td>
      <td>0.815497</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dry Lucerne</td>
      <td>Dry Bamboo</td>
      <td>-5.494453</td>
      <td>-0.013526</td>
      <td>True</td>
      <td>-5.244830</td>
      <td>-0.130161</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dry Lucerne</td>
      <td>Dry Grass</td>
      <td>-3.003669</td>
      <td>1.160887</td>
      <td>False</td>
      <td>-2.904640</td>
      <td>1.154366</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BrushNoDirt</td>
      <td>BrushDirt</td>
      <td>-4.739801</td>
      <td>5.943660</td>
      <td>False</td>
      <td>-3.249920</td>
      <td>6.365292</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Control</td>
      <td>BrushDirt</td>
      <td>1.157339</td>
      <td>11.554010</td>
      <td>True</td>
      <td>1.327801</td>
      <td>14.288748</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Control</td>
      <td>BrushNoDirt</td>
      <td>-0.023134</td>
      <td>12.236542</td>
      <td>False</td>
      <td>-0.655977</td>
      <td>13.044664</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RubDirt</td>
      <td>BrushDirt</td>
      <td>0.315135</td>
      <td>9.771061</td>
      <td>True</td>
      <td>1.263154</td>
      <td>13.157769</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RubDirt</td>
      <td>BrushNoDirt</td>
      <td>-0.902446</td>
      <td>10.179298</td>
      <td>False</td>
      <td>-0.778604</td>
      <td>11.638509</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>-6.656884</td>
      <td>3.955110</td>
      <td>False</td>
      <td>-8.631530</td>
      <td>6.598749</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clover+Dust</td>
      <td>Clover</td>
      <td>-4.401580</td>
      <td>1.531404</td>
      <td>False</td>
      <td>-4.469708</td>
      <td>1.591586</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Grass</td>
      <td>Clover</td>
      <td>-5.547873</td>
      <td>0.620996</td>
      <td>False</td>
      <td>-5.618248</td>
      <td>0.604946</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+Dust</td>
      <td>-3.465540</td>
      <td>1.510666</td>
      <td>False</td>
      <td>-3.286469</td>
      <td>1.643290</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Grass+Dust</td>
      <td>Clover</td>
      <td>-6.182086</td>
      <td>-0.207515</td>
      <td>True</td>
      <td>-6.128564</td>
      <td>0.017583</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grass+Dust</td>
      <td>Clover+Dust</td>
      <td>-3.967283</td>
      <td>0.884150</td>
      <td>False</td>
      <td>-4.158386</td>
      <td>0.756972</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Grass+Dust</td>
      <td>Grass</td>
      <td>-3.168229</td>
      <td>1.958645</td>
      <td>False</td>
      <td>-3.075500</td>
      <td>1.888422</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




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
    The standard deviations used for the beta priors are  (1.3943421938332237, 3.387697874270125)
    The standard deviations used for the M12 priors are 0.18317127941381783



```python
pm.model_to_graphviz(SmfcModel)
```




    
![svg](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_127_0.svg)
    



#### Check prior choice


```python
with SmfcModel as model:
    prior_pred_Smfc = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_Smfc,dataZ.Smfc_z.values,'Smfc')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_130_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with SmfcModel as model:
    trace_Smfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
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
  <progress value='2127' class='' max='15000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  14.18% [2127/15000 12:41<1:16:46 Sampling 10 chains, 0 divergences]
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

    <ipython-input-91-4de91012caf3> in <module>
          1 with SmfcModel as model:
    ----> 2     trace_Smfc = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
    

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
    HAsfc9Model = TwoFactorModel('HAsfc9',x1,x2,dataZ["HAsfc9_z"].values)
```

#### Verify model settings


```python
HAsfc9Model.printParams(x1,x2,dataZ["HAsfc9_z"].values)
```

    The number of levels of the x variables are (2, 11)
    The standard deviations used for the beta priors are  (1.0299664523718457, 2.0653529113379627)
    The standard deviations used for the M12 priors are 0.11539620380870036



```python
pm.model_to_graphviz(HAsfc9Model)
```




    
![svg](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_139_0.svg)
    



#### Check prior choice


```python
with HAsfc9Model as model:
    prior_pred_HAsfc9 = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc9,dataZ["HAsfc9_z"].values,'HAsfc9')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_142_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with HAsfc9Model as model:
    trace_HAsfc9 = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
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
  <progress value='15000' class='' max='15000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [15000/15000 06:32<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 500 draw iterations (10_000 + 5_000 draws total) took 394 seconds.



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
      <td>-0.07</td>
      <td>0.83</td>
      <td>-1.66</td>
      <td>1.58</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5152.0</td>
      <td>2742.0</td>
      <td>5154.0</td>
      <td>3850.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[0]</th>
      <td>0.01</td>
      <td>0.72</td>
      <td>-1.38</td>
      <td>1.49</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4806.0</td>
      <td>2997.0</td>
      <td>4806.0</td>
      <td>3874.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b1[1]</th>
      <td>-0.08</td>
      <td>0.73</td>
      <td>-1.46</td>
      <td>1.42</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4874.0</td>
      <td>2965.0</td>
      <td>4873.0</td>
      <td>4131.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[0]</th>
      <td>-0.30</td>
      <td>0.66</td>
      <td>-1.64</td>
      <td>0.95</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4119.0</td>
      <td>3526.0</td>
      <td>4134.0</td>
      <td>3967.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu_b2[1]</th>
      <td>-0.17</td>
      <td>0.68</td>
      <td>-1.47</td>
      <td>1.20</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4664.0</td>
      <td>3355.0</td>
      <td>4667.0</td>
      <td>4027.0</td>
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
      <th>HAsfc9_mu[275]</th>
      <td>0.01</td>
      <td>0.20</td>
      <td>-0.37</td>
      <td>0.43</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5151.0</td>
      <td>4276.0</td>
      <td>5148.0</td>
      <td>4522.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[276]</th>
      <td>-0.03</td>
      <td>0.21</td>
      <td>-0.44</td>
      <td>0.39</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4911.0</td>
      <td>3439.0</td>
      <td>4923.0</td>
      <td>4077.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[277]</th>
      <td>0.01</td>
      <td>0.20</td>
      <td>-0.37</td>
      <td>0.43</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5151.0</td>
      <td>4276.0</td>
      <td>5148.0</td>
      <td>4522.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[278]</th>
      <td>-0.03</td>
      <td>0.21</td>
      <td>-0.44</td>
      <td>0.39</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4911.0</td>
      <td>3439.0</td>
      <td>4923.0</td>
      <td>4077.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc9_mu[279]</th>
      <td>0.01</td>
      <td>0.20</td>
      <td>-0.37</td>
      <td>0.43</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5151.0</td>
      <td>4276.0</td>
      <td>5148.0</td>
      <td>4522.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>384 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,dataTrace_HAsfc9,'HAsfc9')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_1.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_2.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_3.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_150_4.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_151_0.png)
    



```python
with HAsfc9Model as model:
    plotting_lib.pm.energyplot(trace_HAsfc9)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_152_0.png)
    


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


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_155_0.png)
    



```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc9,'HAsfc9',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_156_0.png)
    



```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc9,'HAsfc9',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_157_0.png)
    


#### Compare prior and posterior for model parameters


```python
with HAsfc9Model as model:
    pm_data_HAsfc9 = az.from_pymc3(trace=trace_HAsfc9,prior=prior_pred_HAsfc9,posterior_predictive=posterior_pred_HAsfc9)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable HAsfc9_y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_160_0.png)
    


#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc9,'HAsfc9')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_162_0.png)
    



```python
df_hdi_HAsfc9 = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc9,'HAsfc9',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_163_0.png)
    



```python
df_hdi_Asfc = plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_Asfc,'Asfc',x1,x2)
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_164_0.png)
    



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
      <td>Dry Grass</td>
      <td>Dry Bamboo</td>
      <td>0.004741</td>
      <td>0.154269</td>
      <td>True</td>
      <td>0.009964</td>
      <td>0.146980</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dry Lucerne</td>
      <td>Dry Bamboo</td>
      <td>-0.039222</td>
      <td>0.077737</td>
      <td>False</td>
      <td>-0.035389</td>
      <td>0.075596</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dry Lucerne</td>
      <td>Dry Grass</td>
      <td>-0.135338</td>
      <td>0.023288</td>
      <td>False</td>
      <td>-0.125951</td>
      <td>0.018912</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BrushNoDirt</td>
      <td>BrushDirt</td>
      <td>-0.154694</td>
      <td>0.173059</td>
      <td>False</td>
      <td>-0.051301</td>
      <td>0.218016</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Control</td>
      <td>BrushDirt</td>
      <td>-0.096697</td>
      <td>0.183108</td>
      <td>False</td>
      <td>-0.014656</td>
      <td>0.238499</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Control</td>
      <td>BrushNoDirt</td>
      <td>-0.129790</td>
      <td>0.194543</td>
      <td>False</td>
      <td>-0.115211</td>
      <td>0.189870</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RubDirt</td>
      <td>BrushDirt</td>
      <td>-0.173911</td>
      <td>0.089835</td>
      <td>False</td>
      <td>-0.088172</td>
      <td>0.144856</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RubDirt</td>
      <td>BrushNoDirt</td>
      <td>-0.217105</td>
      <td>0.097660</td>
      <td>False</td>
      <td>-0.194993</td>
      <td>0.099310</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RubDirt</td>
      <td>Control</td>
      <td>-0.224113</td>
      <td>0.034596</td>
      <td>False</td>
      <td>-0.220132</td>
      <td>0.051093</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clover+Dust</td>
      <td>Clover</td>
      <td>-0.174753</td>
      <td>0.114866</td>
      <td>False</td>
      <td>-0.082380</td>
      <td>0.151813</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Grass</td>
      <td>Clover</td>
      <td>-0.202863</td>
      <td>0.096800</td>
      <td>False</td>
      <td>-0.121873</td>
      <td>0.115497</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Grass</td>
      <td>Clover+Dust</td>
      <td>-0.138433</td>
      <td>0.091202</td>
      <td>False</td>
      <td>-0.148223</td>
      <td>0.078917</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Grass+Dust</td>
      <td>Clover</td>
      <td>-0.290836</td>
      <td>0.004972</td>
      <td>False</td>
      <td>-0.205483</td>
      <td>0.039646</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grass+Dust</td>
      <td>Clover+Dust</td>
      <td>-0.218805</td>
      <td>0.016317</td>
      <td>False</td>
      <td>-0.229822</td>
      <td>0.000143</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Grass+Dust</td>
      <td>Grass</td>
      <td>-0.211966</td>
      <td>0.043802</td>
      <td>False</td>
      <td>-0.211792</td>
      <td>0.033222</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
if writeOut:
    df_hdi_HAsfc9.to_csv(outPathData+ 'hdi_{}.csv'.format('HAsfc9'))
```

### HAsfc81 <a name="HAsfc81"></a>


```python
with pm.Model() as model:
    HAsfc81Model = TwoFactorModel('HAsfc81',x1,x2,dataZ["HAsfc81_z"].values)
```

#### Verify model settings


```python
HAsfc81Model.printParams(x1,x2,dataZ["HAsfc81_z"].values)
```

    The number of levels of the x variables are (2, 11)
    The standard deviations used for the beta priors are  (1.0035538697180921, 1.674345831114272)
    The standard deviations used for the M12 priors are 0.09760320603847883



```python
pm.model_to_graphviz(HAsfc81Model)
```




    
![svg](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_171_0.svg)
    



#### Check prior choice


```python
with HAsfc81Model as model:
    prior_pred_HAsfc81 = pm.sample_prior_predictive(samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc81,dataZ["HAsfc81_z"].values,'HAsfc81')
```


    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_174_0.png)
    


Prior choice is as intended: Broad over the data range.

#### Sampling


```python
with HAsfc81Model as model:
    trace_HAsfc81 = pm.sample(numSamples,cores=numCores,tune=numTune,max_treedepth=20, init='auto',target_accept=0.99,random_seed=random_seed)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (10 chains in 10 jobs)
    NUTS: [HAsfc81_M12_dist, HAsfc81_mu_M12, HAsfc81_b2_dist, HAsfc81_b2_beta, HAsfc81_b1_dist, HAsfc81_b0_dist, HAsfc81_sigmaY, HAsfc81_nuY, HAsfc81_sigma12, HAsfc81_mu_b2, HAsfc81_mu_b1, HAsfc81_mu_b0, HAsfc81_sigma2, HAsfc81_sigma1, HAsfc81_sigma0]




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
  <progress value='15000' class='' max='15000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [15000/15000 10:41<00:00 Sampling 10 chains, 0 divergences]
</div>



    Sampling 10 chains for 1_000 tune and 500 draw iterations (10_000 + 5_000 draws total) took 642 seconds.



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
      <td>-0.08</td>
      <td>0.79</td>
      <td>-1.63</td>
      <td>1.47</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5680.0</td>
      <td>2624.0</td>
      <td>5669.0</td>
      <td>3994.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu_b1[0]</th>
      <td>-0.04</td>
      <td>0.71</td>
      <td>-1.38</td>
      <td>1.38</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5284.0</td>
      <td>2779.0</td>
      <td>5277.0</td>
      <td>3962.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu_b1[1]</th>
      <td>-0.01</td>
      <td>0.70</td>
      <td>-1.44</td>
      <td>1.33</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>5321.0</td>
      <td>2812.0</td>
      <td>5321.0</td>
      <td>3952.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu_b2[0]</th>
      <td>-0.41</td>
      <td>0.66</td>
      <td>-1.70</td>
      <td>0.88</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4284.0</td>
      <td>3170.0</td>
      <td>4276.0</td>
      <td>3847.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu_b2[1]</th>
      <td>-0.40</td>
      <td>0.68</td>
      <td>-1.70</td>
      <td>0.96</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>4457.0</td>
      <td>3595.0</td>
      <td>4455.0</td>
      <td>3814.0</td>
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
      <td>-0.10</td>
      <td>0.18</td>
      <td>-0.40</td>
      <td>0.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4839.0</td>
      <td>4614.0</td>
      <td>4936.0</td>
      <td>4783.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu[276]</th>
      <td>-0.12</td>
      <td>0.15</td>
      <td>-0.41</td>
      <td>0.16</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5236.0</td>
      <td>4918.0</td>
      <td>5285.0</td>
      <td>4728.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu[277]</th>
      <td>-0.10</td>
      <td>0.18</td>
      <td>-0.40</td>
      <td>0.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4839.0</td>
      <td>4614.0</td>
      <td>4936.0</td>
      <td>4783.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu[278]</th>
      <td>-0.12</td>
      <td>0.15</td>
      <td>-0.41</td>
      <td>0.16</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5236.0</td>
      <td>4918.0</td>
      <td>5285.0</td>
      <td>4728.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>HAsfc81_mu[279]</th>
      <td>-0.10</td>
      <td>0.18</td>
      <td>-0.40</td>
      <td>0.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4839.0</td>
      <td>4614.0</td>
      <td>4936.0</td>
      <td>4783.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>384 rows × 11 columns</p>
</div>




```python
plotting_lib.plotDiagnostics(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc81,dataTrace_HAsfc81,'HAsfc81')
```

    /home/bob/.local/lib/python3.8/site-packages/arviz/plots/backends/matplotlib/pairplot.py:212: UserWarning: rcParams['plot.max_subplots'] (40) is smaller than the number of resulting pair plots with these variables, generating only a 8x8 grid
      warnings.warn(



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_1.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_2.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_3.png)
    



    
![png](Statistical_Model_TwoFactor_files/Statistical_Model_TwoFactor_182_4.png)
    



```python
with HAsfc81Model as model:
    plotting_lib.plotTracesB(widthInch,heigthInch,dpi,writeOut,outPathPlots,trace_HAsfc81,'HAsfc81')
```


```python
with HAsfc81Model as model:
    plotting_lib.pm.energyplot(trace_HAsfc81)
```

#### Posterior predictive distribution


```python
with HAsfc81Model as model:
    posterior_pred_HAsfc81 = pm.sample_posterior_predictive(trace_HAsfc81,samples=numPredSamples,random_seed=random_seed)
```


```python
plotting_lib.plotPriorPosteriorPredictive(widthInch,heigthInch,dpi,writeOut,outPathPlots,df,dictMeanStd,prior_pred_HAsfc81,posterior_pred_HAsfc81,dataZ["HAsfc81_z"].values,'HAsfc81')
```


```python
plotting_lib.plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc81,'HAsfc81',x1,x2)
```

#### Compare prior and posterior for model parameters


```python
with HAsfc81Model as model:
    pm_data_HAsfc81 = az.from_pymc3(trace=trace_HAsfc81,prior=prior_pred_HAsfc81,posterior_predictive=posterior_pred_HAsfc81)
```


```python
plotting_lib.plotPriorPosteriorB(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc81,'HAsfc81')
```


```python
plotting_lib.plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc81,'HAsfc81',x1,x2)
```

#### Posterior and contrasts


```python
plotting_lib.plotPosterior(widthInch,heigthInch,dpi,writeOut,outPathPlots,dictMeanStd,pm_data_HAsfc81,'HAsfc81')
```


```python
plotting_lib.plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,outPathPlots,dictMeanStd,dictTreatment,dictSoftware,trace_HAsfc81,'HAsfc81',x1,x2)
```

#### Bimodal distribution in contrast plots
For e.g. the pair Clover+Dust vs. Clover shows an unexpected bimodal distribution of the contrast.
We now examine the traces carefully to exclude errors in the sampling:  
Get the traces on ConfoMap of the interactions


```python
m12_clover = trace_HAsfc81['HAsfc81_M12'][:,0,2]
m12_cloverDust = trace_HAsfc81['HAsfc81_M12'][:,0,3]
diff_m12 = m12_cloverDust-m12_clover
```

Get the traces of the treatments:


```python
b2_clover = trace_HAsfc81['HAsfc81_b2'][:,2]
b2_cloverDust = trace_HAsfc81['HAsfc81_b2'][:,3]
diff_b2 = b2_cloverDust-b2_clover
```

Look at all the pairs


```python
sns.pairplot(data=pd.DataFrame.from_dict(
    {'m12_clover':m12_clover,'m12_cloverDust':m12_cloverDust,'diff_m12':diff_m12,
     'b2_clover':b2_clover,'b2_cloverDust':b2_cloverDust,'diff_b2':diff_b2
    }
));
```

Let's have that of the differences again


```python
sns.jointplot(x=diff_b2,y=diff_m12);
```

We see two sets that are distributed along parallel lines in the scatter plot.
This means that the model estimates two subsets of possible differences.  
However, when looking at the raw data at 'analysis/plots/SSFA_Sheeps_plot.pdf' one can see that the distribution of values for HAsfc81 on Clover (Sheep) measured by ConfoMap appear to have a bimodal distribution.  
Thus, in combination with the chosen uninformative priors, the model correctly describes the distribution as bimodal.  
In summary, we see no issue with the modeling and sampling.

### Write out


```python
!jupyter nbconvert --to html Statistical_Model_TwoFactor.ipynb
```


```python
!jupyter nbconvert --to markdown Statistical_Model_TwoFactor.ipynb
```


```python

































```
