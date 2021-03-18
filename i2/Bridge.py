import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import os

from multiprocessing import Pool, cpu_count

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# Notes:
#      DilatedCNN if there is belief that there are different granularities of patterns
#      TRMF & PCA for cross-sectional residualization
#      VAE
#
#      There is a change in the diff (East-West) around 2017.  People are going one direction but not returning the same way?
#
#
#      (I)   Run on Daily Data to establish a baseline daily expectation
#      (II)  Run on Complete Data and identify on each day if it is within bounds of the daily estimate
#      (III) Day-of-week
#      (IV)  Time of day     => Pct-diff
#      (V)   Year-over-year
#
# Residual adjustments
#
# Additional vars:  Rain/Sunlight/Traffic/
#
#      METHODS:
#           (a) Identify if data needs to be log-transformed
#           (b) Box-Cox Transform to stabilize variance


def prophetFor(args):
    (dfTrain, dfPred, predVar, outputDir, label, holidayDF) = args
    granularity = "daily"
    if "daily" not in label:
        granularity = "intraday"
    fwdPredHorPeriods = len(dfPred)
    df2 = dfTrain
    df2['ds'] = df2['Date']
    df2['y'] = df2[predVar]
    maxDt = df2['ds'].max()

    ensure_dir(outputDir)
    ensure_dir(outputDir + predVar)
    ensure_dir(outputDir + predVar + "/prophet/")
    ensure_dir(outputDir + predVar + "/prophet/season/")
    ensure_dir(outputDir + predVar + "/prophet/forecast/")

    # m1 = Prophet(mcmc_samples=500)
    # m1 = Prophet(growth='logistic')
    m1 = None
    if "daily" in label:
        m1 = Prophet(holidays=holidayDF)
    else:
        m1 = Prophet(changepoint_prior_scale=0.01, holidays=holidayDF)

    success = False
    #try:
    if True:
        m1.fit(df2[['ds', 'y']])

        # Predict
        future=None
        if "daily" in label:
            future = m1.make_future_dataframe(periods=fwdPredHorPeriods)
        else:
            future = m1.make_future_dataframe(periods=fwdPredHorPeriods, freq='H')

        forecastAll = m1.predict(future)
        forecast = forecastAll[forecastAll['ds']>maxDt]

        dfPred['yhat'] = list(forecast['yhat'])
        dfPred['resid'] = dfPred['yhat'] - dfPred[predVar]

        dfAll = dfTrain.append(dfPred)
        dfAll['yhat'] = list(forecastAll['yhat'])
        dfAll['resid'] = dfAll['yhat'] - dfAll[predVar]
        dfAll['absResid'] = np.where(dfAll['resid']<0, -1.0*dfAll['resid'], dfAll['resid'])
        dfAll['pctResid'] = dfAll['resid'] / dfAll[predVar]
        dfAll['absPctResid'] = dfAll['absResid'] / dfAll[predVar]
        dfAll['zResid'] = (dfAll['resid'] - dfAll['resid'].mean()) / dfAll['resid'].std()

        doy = forecast['ds'].map(lambda x: x.dayofyear)
        seasonal = forecast['yearly']
        seasDF = pd.DataFrame({'doy': doy, 'season': seasonal})

        seasDF.to_csv(outputDir + predVar + "/prophet/season/" + label + "_seasonalSignature.csv", index=False, header=True)
        dfPred.to_csv(outputDir + predVar + "/prophet/forecast/" + label + "_pred.csv", index=False, header=True)
        dfAll.to_csv(outputDir + predVar + "/prophet/forecast/" + label + "_all.csv", index=False, header=True)

        # Plot time-series of residuals
        #dfAll = dfAll.set_index('Date')
        for r in ['resid', 'absResid', 'pctResid', 'absPctResid', 'zResid']:
            plt.plot(dfAll['Date'], dfAll[r])
            plt.savefig(outputDir + predVar + "/prophet/forecast/" + label + "_" + r + "_pred.png")
            plt.close()

        success = True
    #except:
    #    pass

    if success:
        try:

            m1.plot_components(forecastAll)
            outputFileP = outputDir + predVar + "/prophet/season/" + label + ".png"
            plt.savefig(outputFileP)
            plt.close()

            m1.plot(forecastAll)
            outputFileP = outputDir + predVar + "/prophet/forecast/" + label + ".png"
            plt.savefig(outputFileP)
            plt.close()

        except:
            pass





class BridgeEDA:
    def __init__(self, baseDir, inputFile="Fremont_Bridge_Bicycle_Counter.csv"):
        self.rawDir = baseDir+"data/raw/"
        self.derivedDir = baseDir+"data/derived/"
        ensure_dir(baseDir+"data/")
        ensure_dir(self.rawDir)
        ensure_dir(self.derivedDir)
        self.inputFile = self.rawDir+inputFile
        self.df = pd.read_csv(self.inputFile, sep=',', parse_dates=['Date'])
        allCols = list(self.df.columns)
        for c in allCols:
            newC = c.replace(" ", "_")
            if c!=newC:
                self.df[newC] = self.df[c]
                del self.df[c]

        # (1) Ensure the dates are unique (and can thus be used as an index)
        uniqueDateSet = self.df['Date'].unique()
        self.uniqueDates = (len(uniqueDateSet)==len(self.df))

        # Additional variables
        self.df['eastDiffWest'] = self.df['Fremont_Bridge_East_Sidewalk'] - self.df['Fremont_Bridge_West_Sidewalk']
        self.df['eastPct'] = self.df['Fremont_Bridge_East_Sidewalk'] / self.df['Fremont_Bridge_Total']
        self.df['westPct'] = self.df['Fremont_Bridge_West_Sidewalk'] / self.df['Fremont_Bridge_Total']
        self.df['imbalancePct'] = self.df['eastDiffWest'] / self.df['Fremont_Bridge_Total']

        # Identify unique independent-variable columns
        colSet = set(self.df.columns)
        indColSet = colSet.difference(set(['Date']))
        self.indColList = list(indColSet)
        self.indColList.sort()

        # Add YYYYMMDD and HH columns for simplicity...
        self.df['yyyymmdd'] = self.df['Date'].dt.normalize()
        self.df['hh'] = self.df['Date'].dt.hour

        self.holidayDF = self.genHolidayDF()



    def genHolidayDF(self):
        holiday2DateList = dict()
        holiday2DateList['mlk'] = ['2012-01-16', '2013-01-21', '2014-01-20', '2015-01-19', '2016-01-18', '2017-01-16', '2018-01-15', '2019-01-21']
        holiday2DateList['presidents'] = ['2012-02-20', '2013-02-18', '2014-02-17', '2015-02-16', '2016-02-15', '2017-02-20', '2018-02-19', '2019-02-18']
        holiday2DateList['memorial'] = ['2012-05-28', '2013-05-27', '2014-05-26', '2015-05-25', '2016-05-30', '2017-05-29', '2018-05-28', '2019-05-27']
        holiday2DateList['labor'] = ['2012-09-03', '2013-09-02', '2014-09-01', '2015-09-07', '2016-09-05', '2017-09-04', '2018-09-03', '2019-09-02']
        holiday2DateList['thanks'] = ['2012-11-22', '2013-11-28', '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23', '2018-11-22', '2019-11-28']

        holidayList = []
        for year in ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"]:
            nyStr = year+"-01-01"
            july4Str = year+"-07-04"
            christmasStr = year+"-12-25"

            holidayList.append({'holiday': 'newYear', 'ds': pd.to_datetime(nyStr), 'lower_window':-1, 'upper_window':1})
            holidayList.append({'holiday': 'july4', 'ds': pd.to_datetime(christmasStr), 'lower_window':-1, 'upper_window':1})
            holidayList.append({'holiday': 'christmas', 'ds': pd.to_datetime(july4Str), 'lower_window':-1, 'upper_window':0})

        for hol in holiday2DateList:
            for dStr in holiday2DateList[hol]:
                if ("thanks" in hol) or ("labor" in hol):
                    holidayList.append({'holiday': hol, 'ds': pd.to_datetime(dStr), 'lower_window': -1, 'upper_window': 1})
                else:
                    holidayList.append({'holiday': hol, 'ds': pd.to_datetime(dStr), 'lower_window': 0, 'upper_window': 0})

        holidayDF = pd.DataFrame(holidayList)
        return holidayDF



    def summaryStats(self, dfi, interval, label):

        # (1) Report the quantiles of the independent variables
        quantiles = list(np.arange(0, 100, 10))
        quantiles[0] = 5
        quantiles.append(95)
        quantiles = [n / 100.0 for n in quantiles]
        quantileDF = self.df[self.indColList].quantile(quantiles)

        # (2) Report the total moments of the independent variables
        momentsDF = self.df[self.indColList].apply(["mean", "std", "skew", "kurtosis"])

        # (3) Generate rolling moments
        rollingDict = dict()
        rollingDict['mean'] = self.df[self.indColList].rolling(interval).mean()
        rollingDict['std'] = self.df[self.indColList].rolling(interval).std()
        rollingDict['skew'] = self.df[self.indColList].rolling(interval).skew()
        rollingDict['kurtosis'] = self.df[self.indColList].rolling(interval).kurt()

        ensure_dir(self.derivedDir+"/summaryStats/")
        quantileDF.to_csv(self.derivedDir+"/summaryStats/"+label+"Quantiles.csv", index=False, header=True)

        for dist in rollingDict:
            currOutputDir = self.derivedDir + "/summaryStats/"+dist+"/"
            ensure_dir(currOutputDir)
            for col in self.indColList:
                rollingDict[dist][[col]].plot()
                plt.savefig(currOutputDir+label+"_"+col+".png")
                plt.close()

        return (quantileDF, rollingDict)



    def forecast(self, dfi, fwdHor, granularity="daily"):
        df2 = dfi.reset_index()
        # Cross-validation training to identify error characteristics
        dates = list(df2['Date'].unique())
        dates.sort()

        # Full Period
        eIdx = -1*(fwdHor+1)
        trainSD = dates[0]
        trainED = dates[eIdx]
        predSD = dates[eIdx+1]
        predED = dates[-1]

        trainFilter = (df2['Date']>=trainSD) & (df2['Date']<=trainED)
        predFilter = (df2['Date']>=predSD) & (df2['Date']<=predED)

        cpus = int(cpu_count())
        ret_list1 = []
        with Pool(cpus) as p:
            ret_list1 = p.map(prophetFor, [(df2[trainFilter], df2[predFilter], predVar, self.derivedDir, granularity+"Pred"+str(fwdHor), self.holidayDF) for predVar in self.indColList])



    # Generate forecasts aggregate by day to set a target for intraday forecasts
    def predByDay(self):
        df = self.df.set_index('Date').copy(deep=True)
        df = df[self.indColList].resample('D').apply(sum)                  # Resample by day  [Index is daily]

        (quantileDF, rollingDict) = self.summaryStats(df, 28, "daily")     # Daily rolling summary stats across history of 28-days

        # Daily idiosyncratic behavior
        #self.dailyIdiosyncraticBehavior(df)

        for fwdHor in [28, 60, 90, 183, 365]:
            self.forecast(df, fwdHor, "daily")


    # Generate forecasts aggregate by day to set a target for intraday forecasts
    def predAll(self):
        df = self.df.set_index('Date').copy(deep=True)

        #(quantileDF, rollingDict) = self.summaryStats(df, 28, "intraday")     # Daily rolling summary stats across history of 28-days

        # Daily idiosyncratic behavior
        #self.dailyIdiosyncraticBehavior(df)

        for fwdHor in [28, 60, 90, 183, 365]:
            self.forecast(df, fwdHor*24, "intraday")









# docker run --name=Bridge -d -v /mnt/prod:/mnt/prod falkon:1.0 python3 /exeDir/Bridge.py
# docker run --name=Bridge --rm -v /mnt/prod:/mnt/prod falkon:1.0 python3 /exeDir/Bridge.py

baseDir = "/mnt/prod/signals/school/"
bo = BridgeEDA(baseDir)
bo.predByDay()
bo.predAll()





'''
East => Northbound
West => Southbound
1) Summary Stats [Daily]
    (a) Total:  
        Quantile analysis:  Range doesn't seem logarithmic
        Time-series of 7-day average:  Traffic in peak-months seems to be growing.  Troughs seem stable.
    (b) Diff/imbalancePct:   
        Quantile analysis:  Negative tailed (More traffic "southbound" than "northbound").  Would expect roughly symmetric
        Time-series of 7-day average:  Starting in roughly 2017, we see steady "downside" diffs (increasingly West-traffic > East-traffic)
            East_Sidewalk vs. West_Sidewalk plots:  Driven by slightly lower peak East and much higher West traffic.
            Diff Skew:  Strong negative skew
            
            
2) Seasonality:
    (a) Total:
        2017 sees a strong increase in total traffic
        Strong weekly seasonality
        Peak yearly seasonality in May-August.
        Bi-modal (2-peaks) in intraday seasonality:  (8AM and 4PM)
    (b) Diff:
        2016 represents the start of a strong move bias towards Southbound (West) traffic
        Bias seems to occur predominately during peak yearly-seasonality (May-August)
        The bias seems heaviest during the weekday
        Morning has positive bias (net people going Northbound [East] downhill)
        Bias seems driven by the afternoon rush (net people going Southbound [West] uphill) (5PM)


3) Modeling [Total]:
    (a) Anomaly-detection:  Fit most time period (forecast 28-days):
            Visually, review the forecast => We see most of the upside-outliers come during summer months.  Thus we might want to conditionalize more explicitly against summer.
            Review PctResid               => Spikes here seem focused on "small-denominator" periods (winter)
            zScore
            [x] Adjust for holidays 
            
    (b) Forecast out 365 days

    (c) Proper approach would be to run backtest fitting and forecasting forward to look for anomalies.
    
    
4) Model Observational    
    (a) Standardized resids (Z-Score) at the 2.5% indicates that most of the under-forecasts appear to be in March, April, and May
    (b) Standardized resids (Z-Score) at the 97.5% indicates that most of the over-forecasts appear to be in Feburary, May, June-Sept

    
'''



df = pd.read_csv(baseDir+"data/derived/Fremont_Bridge_Total/prophet/forecast/dailyPred28_all.csv", parse_dates=['Date'])
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

negTailDF = df[df['zResid']<df['zResid'].quantile(0.025)].copy(deep=True)
negTailDF.groupby('month').size()
negTailDF.groupby(['year', 'month']).size()

posTailDF = df[df['zResid']>df['zResid'].quantile(0.975)].copy(deep=True)
posTailDF.groupby('month').size()
posTailDF.groupby(['year', 'month']).size()
