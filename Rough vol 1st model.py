
# coding: utf-8

# In[370]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)



class FractionalBM:
    def __init__(self,
                 h,
                 days,timestep=1):
        
        self.h = h
        self.days = days
        self.cholesky_matrix = None
        self.fbm = None
        self.timestep = timestep
        
        
    def generate(self):
        
        nb_points = int(self.days/self.timestep)
        covar_matrix = np.zeros((nb_points,
                                 nb_points))
        
        scale_range = 0.0001+(1/252)*np.arange(nb_points)
        #scale_range = np.arange(1,self.days+1,self.timestep)
        for j,t_j in enumerate(scale_range):
            for i,t_i in enumerate(scale_range):
                covar_matrix[i][j] = 0.5*(t_i**(2*self.h) + t_j**(2*self.h) -np.abs(t_i-t_j)**(2*self.h))
        lower_cholesky_matrix = linalg.cholesky(covar_matrix , 
                                                      lower = True)
        
        self.cholesky_matrix = lower_cholesky_matrix
        
        random_normal_variables = np.random.normal(0,1,nb_points)
        
        fractional_bm = np.dot(lower_cholesky_matrix,
                              random_normal_variables)
        
        self.fbm = fractional_bm
        return fractional_bm
    
    def rough_vol(self,nu,sigma0=0.1):
        return sigma0*np.exp(nu*self.fbm)
    
    def plot_rough_vol(self,nu,sigma0):
        y = self.rough_vol(nu,sigma0)
        x = np.arange(0,self.days,self.timestep)
        plt.plot(x,y)
        plt.show()
        
    
    
def generate_corr_brownian(corr,dt):

    cov = [[dt, 0],
           [0, dt]]
    
    mean = [0, 0]
    (dWt_1,dWt_2) = np.random.multivariate_normal(mean = mean,
                                                  cov = cov)

    return corr*dWt_1 + np.sqrt(1-corr**2)*dWt_2



# ## Rough Vol model


class Configuration:
    def __init__(self, NumberOfScenarios, TimeStep):
        self.NumberOfScenarios=NumberOfScenarios 
        self.TimeStep = TimeStep

class OptionTrade:
    def __init__(self, stock_price, strike_price, risk_free_rate,time_to_maturity):
        self.stock_price=stock_price
        self.strike_price=strike_price
        self.risk_free_rate = risk_free_rate
        self.time_to_maturity = time_to_maturity

class RoughVol:
    def __init__(self, Configuration, H, nu, T_max,sigma0=0.1):
        self.Configuration = Configuration
        self.H = H
        self.nu = nu
        self.sigma0 = sigma0
        self.t_max = T_max
    def SimulateRoughVol(self):
        Timestep = self.Configuration.TimeStep
        fbm = FractionalBM(h = self.H, days = self.t_max, timestep=Timestep)
        fractional_bm = fbm.generate()
        self.fbm_simul = fractional_bm
        vol = fbm.rough_vol(nu=self.nu, sigma0=self.sigma0)
        return vol
        
    
class Model_stochVol:
    def __init__(self, Configuration, H, nu, corr, sigma0=0.1):
        self.Configuration = Configuration
        self.H = H
        self.nu = nu
        self.corr = corr
        self.sigma0 = sigma0
    #simulate risk factors using GBM stochastic differential equation
    def SimulateStockPrices(self, stock_price0, T_max=250):
        simulation_df = pd.DataFrame()
        # for this example, we only are concerned with one time step as it’s an European option
        roughVol = RoughVol(self.Configuration, self.H, self.nu, T_max)
        timestep = self.Configuration.TimeStep
        nb_points = int(T_max/timestep)
        dt = timestep
        for scenarioNumber in range(self.Configuration.NumberOfScenarios):
            prices = []
            Rvol = roughVol.SimulateRoughVol()
            uncertainty = Rvol[0]*np.sqrt(dt)*np.random.normal(0,1)

            price = stock_price0 * np.exp(uncertainty)
            prices.append(price)
            for y in range(nb_points-1):
                uncertainty = Rvol[y+1]*np.sqrt(dt)*np.random.normal(0,1)
                price = prices[y] * np.exp(uncertainty)
                prices.append(price)
            simulation_df[scenarioNumber] = prices
            
        return simulation_df
    
class OptionTradePayoffPricer:
    def CalculatePrice(self, trade, df_prices_per_scenario, configuration):
        pay_offs = 0
        total_scenarios = configuration.NumberOfScenarios
        for i in range(total_scenarios):
            price = df_prices_per_scenario[i].values.tolist()[-1]
            pay_offs=pay_offs+max(price - trade.strike_price,0)
                
        discounted_price = (np.exp(-1.0*trade.risk_free_rate * (trade.time_to_maturity/365))*pay_offs)
        result = discounted_price/total_scenarios
        return result
    
class MonteCarloEngineSimulator:
    
    #instationate with configuration and the model
    def __init__(self, configuration, model):
        self.configuration = configuration
        self.model = model
        
    #simulate trade and calculate price    
    def Simulate(self, stock_price0, T_max=250, display=True):
        df_prices_per_scenario = self.model.SimulateStockPrices(stock_price0, T_max=T_max)
        #print(df_prices_per_scenario)
        nb_scenarios = self.configuration.NumberOfScenarios
        timestep = self.configuration.TimeStep
        #plot scenarios
        if display:
            plot_scenario_paths(df_prices_per_scenario, nb_scenarios, timestep)
        return df_prices_per_scenario

    


# In[432]:


def plot_scenario_paths(df_prices_per_scenario, nb_scenarios, timestep):
    plt.figure(figsize=(14,8))
    for i in range(nb_scenarios):            
        plt.plot((df_prices_per_scenario.index)*timestep, df_prices_per_scenario[i])            

    plt.ylabel('Stock Value')
    plt.xlabel('Timestep')
    plt.show()


# In[433]:


from scipy.stats import norm

def implied_volatility(Price,Stock,Strike,Time,Rf):
    P = float(Price)
    S = float(Stock)
    E = float(Strike)
    T = float(Time)
    r = float(Rf)
    sigma = 0.01
    while sigma < 1:
        d_1 = float(float((np.log(S/E)+(r+(sigma**2)/2)*T))/float((sigma*(np.sqrt(T)))))
        d_2 = float(float((np.log(S/E)+(r-(sigma**2)/2)*T))/float((sigma*(np.sqrt(T)))))
        P_implied = float(S*norm.cdf(d_1) - E*np.exp(-r*(T/365))*norm.cdf(d_2))
        if P-(P_implied) < 0.001:
            return sigma
        sigma +=0.001
    return np.nan


# In[481]:


class RSV_model:
    
    #instationate with configuration and the model
    def __init__(self, stock_price = 200,Rf = 0.05,nb_scenarios = 1000,
                             timestep_days = 1, sigma0=0.1, H = 0.04,nu = 1,corr = -0.4):
        self.stock_price = stock_price
        self.Rf = Rf
        #self.nb_scenarios = nb_scenarios
        self.sigma0 = sigma0
        self.H = H
        self.nu = nu
        self.corr = corr
        self.configuration = Configuration(nb_scenarios, timestep_days) # config
    def simulate_paths(self, T_max=250, display=True):    
        model = Model_stochVol(self.configuration, self.H, self.nu, self.corr, self.sigma0)
        simulator = MonteCarloEngineSimulator(self.configuration, model)
        df_paths = simulator.Simulate(self.stock_price, T_max=T_max, display=display)
        self.df_paths = df_paths
    def compute_option_price(self, strike, maturity_days):
        trade = OptionTrade(self.stock_price, strike, self.Rf, maturity_days) # trade
        tradePricer = OptionTradePayoffPricer()
        df_prices_per_scenario_mat = self.df_paths.loc[:int(maturity_days/self.configuration.TimeStep)]
        option_price = tradePricer.CalculatePrice(trade, df_prices_per_scenario_mat, self.configuration)

        print("option price with Monte carlo method : ",option_price)        
        return option_price



# ## Use of the model to simulate implied volatilities for the date 20180108 for different maturities and strikes : simulated volatility surface




def find_sigma0(df_iv_groupBy,min_maturity):
    df_min_mat = df_iv_groupBy.loc[min_maturity].reset_index()
    sigma0 = df_min_mat["IV_ATM"].unique()[0]
    return sigma0

def compute_iv_roughVol_fixedDate(df_iv_groupBy, index_dic_param = 2, tenor_min=0,tenor_max = 1):
    tenor_list_unique = sorted(df_iv_groupBy.index.get_level_values(0).unique())
    df_iv_groupBy["iv_rough_vol_"+str(index_dic_param)] = np.nan
    H = dic_params[index_dic_param]["H"]
    nu = dic_params[index_dic_param]["nu"]
    corr = dic_params[index_dic_param]["corr"]
    spot = df_iv_groupBy["spot"].values[0]
    print("spot :",spot)
    #sigma0 = find_sigma0(df_iv_groupBy,tenor_list_unique[0])
    sigma0 = 0.15
    rsv_model = RSV_model(stock_price=spot, Rf=0.05, nb_scenarios=1000,
    timestep_days=1, sigma0=sigma0, H = H, nu=nu, corr=corr)
    T_max = 252
    rsv_model.simulate_paths(T_max=T_max,display=False)
    nb_options = 0
    for tenor in tenor_list_unique:
        if (tenor>=tenor_min) and (tenor<=tenor_max):
            strike_list = df_iv_groupBy.loc[tenor].index
            for strike in strike_list:
                maturity_days = int(tenor*365)+1
                print("maturity in days : "+str(maturity_days)+" , strike : "+str(strike))
                option_price = rsv_model.compute_option_price(strike = strike, maturity_days = maturity_days)
                iv = implied_volatility(option_price,spot,strike,maturity_days,Rf=0.05)
                print("implied vol with BS formula inversion : ",iv)
                df_iv_groupBy.loc[(tenor,strike),"iv_rough_vol_"+str(index_dic_param)] = iv
                nb_options+=1
        elif tenor>=tenor_max:
            break;
    print("total number of options priced :",nb_options)
    return df_iv_groupBy


# In[700]:

def read_data(date,index="SPY",indicator="impliedV"):
    path = os.path.join("data",indicator,index,date+"_"+index+"~market__"+indicator+".csv")
    df = pd.read_csv(path)
    return(df)


def get_iv_atm(expiry,df):
    df1 = df[df.expiry == expiry]
    df_bis = df1.drop_duplicates("expiry")
    df_bis["strike"] = df_bis["fwd"]
    df_bis["midImpliedV"] = np.nan
    df1 = df1.append(df_bis,ignore_index=True)
    df1.sort_values("strike",inplace=True)
    df1["midImpliedV"] = df1["midImpliedV"].interpolate()
    return(df1[df1["strike"] == df1["fwd"]]["midImpliedV"].values[0])


def add_iv_atm_col(df):
    def change_iv_atm(r,expiry,v):
        if r["expiry"]==expiry:
            r["IV_ATM"]=v
        return r
    # compute of IV_ATM of each expiry and add it to the column df["IV_ATM"]
    df["IV_ATM"] = np.nan
    for expiry in df.expiry.unique():
        v = get_iv_atm(expiry,df)
        df = df.apply(lambda r: change_iv_atm(r,expiry,v), axis=1)
    return df

def read_process_options_data(date):
    df_iv = read_data(date)
    df_iv["expiry"] = df_iv["expiry"].astype("str")
    df_iv["expiry"] = pd.to_datetime(df_iv["expiry"],format="%Y-%m-%d")
    
    df_iv = add_iv_atm_col(df_iv)
    df_iv["Moneyness"] = np.log(df_iv["strike"]/df_iv["fwd"])/(np.sqrt(df_iv["tenor"])*df_iv["IV_ATM"])
    
    df_iv_groupBy = df_iv.groupby(["tenor","strike","expiry"]).mean()
    df_iv_groupBy.reset_index("expiry",inplace=True)
    return df_iv_groupBy

if __name__ == '__main__':

    df_iv_groupBy = read_process_options_data("20180108")
    
    dic_params={0:{"H":0.04,"nu":0.2,"corr":0},1:{"H":0.04,"nu":0.04,"corr":0}, 2:{"H":0.1,"nu":0.001,"corr":0},3:{"H":0.33,"nu":0.001,"corr":0},4:{"H":0.3,"nu":0.04,"corr":0},5:{"H":0.05,"nu":0.04,"corr":0},
        6:{"H":0.13,"nu":0.2,"corr":0}}
    df_iv_groupBy = compute_iv_roughVol_fixedDate(df_iv_groupBy,index_dic_param=6,tenor_max=0.3)


