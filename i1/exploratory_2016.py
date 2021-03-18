import os
import codecs
from optparse import OptionParser
import copy

from datetime import datetime

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as ts
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt



class FileLoader:
    def __init__(self, inputFile):
        self.inputFile = inputFile
        self.df = None
        self.notes = ""

    def parseDatetime_YYYYMMDD(self, val):
        return datetime.strptime(str(int(val)), "%Y%m%d")

    def parseDatetime_YYYYMMDD_time(self, val):
        return datetime.strptime(str(val), "%Y-%m-%d %H:%M:%S")

    def readDaily(self):
        self.df = pd.read_csv(self.inputFile, index_col=False)
        self.df['dt'] = self.df.apply(lambda row: self.parseDatetime_YYYYMMDD(row['date']), axis=1)
        return self.df

    def genAlignedRet(self, df, var='spy_close_price', label='fwdSpyRet'):
        # (1) Ensure the dates are unique
        df2 = df.groupby('date').count().reset_index()
        numberDuplicates = len(df2[df2[var]>1])
        notes = "\n\nFUNCTION: FileLoader.genAlignedSPYRet\n"
        notes += "\tNumber of duplicate dates: "+str(numberDuplicates)+"\n"
        notes += "\tWe don't have a trading calendar so it's difficult to identify missing trading days.  We will assume days are all present.\n"

        df = df.sort(['date'], ascending=[1])
        df[label] = (df[var].shift(-1) - df[var]) / df[var]
        self.notes += notes
        print(notes)
        return df.dropna()




class ErrDisc:
    def __init__(self, outputDir):
        self.errorDict = dict()
        self.notes = ""
        self.outputDir = outputDir

    def idGrossErrors(self, df):
        # (1) df.describe()
        #          We see positive outliers in both "signal" as well as "spy_close_price"
        notes = "\n\nFUNCTION:  ErrDisc.idGrossErrors\n"
        notes += "\tdf.describe() reveals three obvious outliers in spy_close_price:\n"
        notes += "\tManual inspection appears to indicate that these are likely typos given the consistency of \"incorrectness\"\n"
        notes += "\tAs such, we will correct these items manually given our hypothesis of \"typos\"\n"
        manualCorrections = dict()
        manualCorrections[618.95] = 168.95
        manualCorrections[619.33] = 169.33
        manualCorrections[710.31] = 170.31
        for k in manualCorrections:
            notes += "\t\tUpdating spy_close_price: ["+str(k)+" to "+str(manualCorrections[k])+"]\n"
            idx = df[df['spy_close_price']==k].index[0]
            df.set_value(idx, 'spy_close_price', manualCorrections[k])

        notes += "\n\tdf.describe() reveals two obvious outliers in signal:\n"
        notes += "\tManual inspection appears to indicate that these are likely typos given the consistency of \"incorrectness\"\n"
        notes += "\tA third outlier appears as a negative value.  We should probably correct to be positive given all other values are positive and same magnitude\n"
        notes += "\tAs such, we will correct these items manually given our hypothesis of \"typos\"\n"
        updateIndices = list(df[(df['signal']>10) | (df['signal']<2)].index)
        for idx in updateIndices:
            val = df.iloc[[idx]]['signal'].values[0]
            notes += "\t\tUpdating signal: ["+str(val)+" to "
            if val>10:
                val /= 100.0
            elif val<0:
                val *= -1.0
            elif val<2:
                while val<1.0:
                    val *= 10.0
            notes += str(val)+"]\n"
            df.set_value(idx, 'signal', val)

        notes += "\n\tLooking at df.describe(), there doesn't appear to be a need to winsorize or otherwise truncate outliers\n"
        notes += "\n\tHowever, there does appear to be a trend...\n"
        print(notes)
        self.notes += notes
        print(df[['signal', 'spy_close_price']].describe())
        print("\n")
        return df


    def standardize(self, dfi, col):
        df = dfi.copy(deep=True)
        df[col] = (df[col]-df[col].mean())/df[col].std()
        return df


    def genARIMAResid(self, dfi, indVar):
        notes = "\n\nFUNCTION:  ErrDisc.genARIMAResid\n"
        df = dfi[[indVar, 'dt']].copy(deep=True)
        dfAll = dfi.copy(deep=True)
        dfAll = dfAll.set_index('dt')

        # (1) Find order of differencing (roughly)
        sm.graphics.tsa.plot_acf(df[indVar].dropna(), lags=25)
        plt.savefig(self.outputDir+indVar+"_0Diff_acf.png")
        plt.close()
        sm.graphics.tsa.plot_acf(np.diff(df[indVar].dropna()), lags=25)
        plt.savefig(self.outputDir+indVar+"_1Diff_acf.png")
        plt.close()
        notes += "\tWe see that for variable ["+indVar+"] we should probably model the first-differences\n"

        # (2) Identify the number of lags (AR) and error-correction (MA)
        sm.graphics.tsa.plot_pacf(np.diff(df[indVar].dropna(), n=1), lags=25)
        plt.savefig(self.outputDir+indVar+"_1Diff_pacf.png")
        plt.close()
        notes += "\tPACF lag-1 is negative => No need for AR, need to correct the diff\n"
        notes += "\tACF is negative => MA until cutoff [1]\n"

        # (3) Fit (p, d, q) => (AR, DIFF, MA)
        df = df.set_index('dt')
        fit1 = sm.tsa.ARIMA(df[[indVar]].dropna(), (0,1,1)).fit()  
        notes += "\tFit ARIMA(0,1,1) for "+indVar+"\n"

        resids = pd.DataFrame(fit1.resid)
        resids[indVar+'_ARIMAresid'] = fit1.resid
        #resids[indVar+'_ARIMAresid'] = resids[indVar+'_ARIMAresid'].shift(-1)
        residMerge = pd.merge(resids, dfAll, left_index=True, right_index=True)
        residMerge = residMerge.drop([residMerge.columns[0]], axis=1)

        notes += "\tLooks like this model is overkill.  Simple difference is sufficient\n"
        self.notes += notes
        print(notes)
        return residMerge




class ExploratoryAnalysis:
    def __init__(self):
        self.notes = ""

    def corrAnalysis(self, df, varList):
        notes = "\n\nFUNCTION: ExploratoryAnalysis.corrAnalysis\n"
        notes += "\tWe see high correlation between contemporaneous [signal, spy_close_price], but low correlation with future return\n"
        notes += "\t\t(a) Maybe this is indicative of a cointegrated pair?\n"
        notes += "\t\t(b) Maybe relationships only exist in the tails [or middle]?\n"
        self.notes += notes
        print(notes)
        print(df[varList].corr())

    def cointAnalysis(self, df, indVar, depVar):
        notes = "\n\nFUNCTION: ExploratoryAnalysis.cointAnalysis\n"



    def quantileReg(self, df, indVar, depVar, outputFile, quantiles=10):
        notes = "\n\nFUNCTION: ExploratoryAnalysis.quantileReg\n"
        regDF = self.quantileAndClean(df[[indVar, depVar]], depVar, outputFile=outputFile)
        notes += "\tWe notice that there is an insignificant (but increasing relationship between signal[T-1, T] and SPY[T, T+1])\n"
        notes += "\tPotentially useful is the fact that the top decile shows a positive relationship significantly different from an OLS through the entire data\n"
        print(notes)
        self.notes += notes
        return regDF


    # Fit Quantile
    def fit_model(self, q, depVar, indVar, df):
        mod = smf.quantreg(depVar+" ~ "+indVar, df)
        res = mod.fit(q=q)
        dfCount = [len(res.fittedvalues)]
        return [q, res.params['Intercept'], res.params[indVar]] + res.conf_int().ix[indVar].tolist() + dfCount



    # -1 lag => align current data against "future" data
    # Let's do pairwise regressions so we reduce the impact of NaNs across covariates
    def quantileAndClean(self, df, depVar, lag=0, outputFile=None):
        quantiles = np.arange(0.05, 1.0, 0.05)
        allModels = pd.DataFrame()

        cols = df.columns.values.tolist()
        df2 = df.copy(deep=True)
        if lag!=0:
            df2[depVar] = df2[depVar].shift(lag)
        for col in cols:
            if col==depVar:
                continue
            df4 = df2[[depVar, col]].copy(deep=True)
            df4 = df4.replace([np.inf, -np.inf], np.nan)
            models = [self.fit_model(x, depVar, col, df4) for x in quantiles]
            models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub', 'count'])
            models['depVar'] = depVar
            models['indVar'] = col

            if outputFile is not None:
                self.ensure_dir(outputFile)
                #plt.rc('text', usetex=True)
                ols = smf.ols(depVar+' ~ '+col, df4).fit()
                ols_ci = ols.conf_int().ix[col].tolist()
                olsDict = dict(a = ols.params['Intercept'],
                               b = ols.params[col],
                               lb = ols_ci[0],
                               ub = ols_ci[1])

                n = models.shape[0]
                p1 = plt.plot(models.q, models.b, color='black', label='Quantile Reg.')
                p2 = plt.plot(models.q, models.ub, linestyle='dotted', color='black')
                p3 = plt.plot(models.q, models.lb, linestyle='dotted', color='black')
                p4 = plt.plot(models.q, [olsDict['b']] * n, color='red', label='OLS')
                p5 = plt.plot(models.q, [olsDict['lb']] * n, linestyle='dotted', color='red')
                p6 = plt.plot(models.q, [olsDict['ub']] * n, linestyle='dotted', color='red')
                plt.ylabel(r'\beta_\mbox{'+col+'}')
                plt.xlabel('Quantiles of the conditional '+depVar+' distribution')
                plt.legend()
                plt.savefig(outputFile+depVar+"_vs_"+col+".png")
                plt.close()
                #plt.savefig(pp, format='pdf')
                #plt.close()

            allModels = allModels.append(models, ignore_index=True)
        return allModels


    def cointegration_test(self, dfInput, indVar, depVar):
        dfSubset = dfInput[[indVar, depVar]].copy(deep=True)
        dfSubset = dfSubset.replace([np.inf, -np.inf], np.nan)
        dfSubset = dfSubset.dropna()

        # Step 1: regress on variable on the other 
        # Step 2: obtain the residual (ols_result.resid)
        # Step 3: apply Augmented Dickey-Fuller test to see whether 
        #        the residual is unit root    
        ols_result = sm.OLS(dfSubset[depVar], dfSubset[indVar]).fit()   # depVar, indVar
        params = ols_result.params
        std = np.std(ols_result.resid)
        mean = np.mean(ols_result.resid)
        adfResult = ts.adfuller(ols_result.resid)
        return (mean, std, adfResult, params, ols_result.resid)



    def ensure_dir(self, f):
        d = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)




class SuperLearner:
    def __init__(self, learnerDict, indVarList, depVar):
        self.colOrder = copy.deepcopy(indVarList)
        self.colOrder.append(depVar)
        self.learnerDict = copy.deepcopy(learnerDict)
        self.cvLearners = copy.deepcopy(learnerDict)
        self.metaColOrder = []
        self.baseModelOrder = []
        for learnerName in sorted(self.learnerDict['base'].keys()):
            self.metaColOrder.append(learnerName)
            self.baseModelOrder.append(learnerName)
        self.metaColOrder.append(depVar)


    def cvFit(self, df, indVarList, depVar, folds=10):
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # (1) Fit and store base models over all data
        for bm in self.learnerDict['base']:
            self.learnerDict['base'][bm].fit(df[indVarList], df[depVar])

        # (2) Generate Z matrix
        skf = StratifiedKFold(df[depVar], n_folds=folds)
        currFold = 0
        ZDF = None
        for train_index, test_index in skf:
            trainDF = df.iloc[train_index]
            testDF = df.iloc[test_index]    

            # Force to iterate over each base model in the same order for each fold
            currZDF = None
            for probBM in self.baseModelOrder:
                self.cvLearners['base'][probBM].fit(trainDF[indVarList], trainDF[depVar])
                preds = self.cvLearners['base'][probBM].predict(testDF[indVarList])

                if currZDF is None:
                    currZDF = trainDF[[depVar]].copy(deep=True)
                currZDF[probBM] = preds

            if ZDF is None:
                ZDF = currZDF[self.metaColOrder].copy(deep=True)
            else:
                ZDF = ZDF[self.metaColOrder].append(currZDF[self.metaColOrder], ignore_index=True)

            currFold += 1


        # (3) Fit metalearner
        for ml in list(self.learnerDict['meta']):
            self.model2File['meta'][ml].fit(ZDF[self.baseModelOrder], ZDF[depVar])


from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
class GenModels:
    def __init__(self):
        self.blah = 1

    def genGBM(self):
        classifiers = dict()
        classifiers['GBM'] = GradientBoostingRegressor()
        for nfeat in [100, 500, 1000]:
            for mdepth in [3, 5, 10, 20]:
                classifiers["GBM_"+str(nfeat)+"iter_"+str(mdepth)+"mdepth"] = GradientBoostingRegressor(n_estimators=nfeat, max_depth=mdepth)
        return classifiers

    def genLR(self):
        classifiers = dict()
        classifiers['LR'] = LogisticRegression()
        C_range = np.logspace(-2, 1, 4)  # Regularizer
        penal = ['l1', 'l2']
        for c in C_range:
            c_s = str(int(c*100))
            for p in penal:
                classifiers['LR_'+c_s+"C_"+p+"Penalty"] = LogisticRegression(penalty=p, C=c)
        return classifiers

    def genSVC(self):
        classifiers = dict()
        classifiers['SVC_2_1'] = SVR(gamma=2, C=1, probability=True)
        C_range= np.logspace(0, 3, 4)
        gamma_range = np.logspace(-2, 3, 6)
        for c in C_range:
            c_s = str(int(c*100))
            for g in gamma_range:
                g_s = str(int(g*100))
                classifiers["SVC_"+g_s+"Gamma_"+c_s+"C"] = SVR(gamma=g, C=c, probability=True)
        return classifiers

    def genAda(self):
        classifiers = dict()
        for estimators in [10, 25, 50, 100, 200]:
            classifiers['Ada_'+str(estimators)+"Est"] = AdaBoostRegressor(n_estimators=estimators)
        return classifiers

    def genET(self):
        classifiers = dict()
        classifiers['ET'] = ExtraTreesRegressor(n_estimators=1000)
        for maxDepth in [5, 10, 25, 50, 100]:
            for maxFeat in ["auto", "sqrt", "log2"]:
                for crit in ["gini", "entropy"]:
                    classifiers["ET_"+str(maxDepth)+"Depth_"+str(maxFeat)+"maxFeat_"+crit+"Crit"] = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, max_depth=maxDepth, max_features=maxFeat, criterion=crit)
        return classifiers

    def genRF(self):
        classifiers = dict()
        classifiers['RF'] = RandomForestRegressor(n_estimators=1000)
        for maxDepth in [10, 100]:
            for maxFeat in ["auto", "sqrt", "log2"]:
                for crit in ["gini", "entropy"]:
                    classifiers["RF_"+str(maxDepth)+"Depth_"+str(maxFeat)+"maxFeat_"+crit+"Crit"] = RandomForestRegressor(n_estimators=1000, n_jobs=-1, max_depth=maxDepth, max_features=maxFeat, criterion=crit)
        return classifiers










# exec(open("C:\\Dev\\Phishing\\Python\\Python\\cubist\\exploratory.py").read())

if __name__ == "__main__":
    usage = "usage: %prog [options] arg1 arg2"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--baseInput", dest="baseInput", default="C:\\Users\\chngan\\Downloads\\cubist\\data\\input\\data.csv",
                      help="base directory containing data")
    parser.add_option("-o", "--baseOutput", dest="baseOutput", default="C:\\Users\\chngan\\Downloads\\cubist\\data\\output\\",
                      help="Base output directory")

    (options, args) = parser.parse_args()
    inputFile = options.baseInput
    outputDir = options.baseOutput

    flo = FileLoader(inputFile)
    rawDF = flo.readDaily()

    edo = ErrDisc(outputDir)
    errCorr = edo.idGrossErrors(rawDF)
    cleanedDF = flo.genAlignedRet(errCorr, 'spy_close_price', 'fwdSpyRet')





    #####################
    # (1) PREPROCESSING #
    #####################

    # Different forms of preprocessing
    preprocessedDF = dict()
    
    # Although signal is non-stationary, ARIMA looks like overkill.  
    # Simple difference (or return) will suffice
    arimaResidDF = edo.genARIMAResid(cleanedDF, 'signal')
    arimaResidSPDF = edo.genARIMAResid(cleanedDF, 'spy_close_price')
    cleanedDF = flo.genAlignedRet(cleanedDF, 'signal', 'signalRet')
    cleanedDF['signalRet'] = cleanedDF['signalRet'].shift(1)

    pxDF = cleanedDF[['date', 'dt', 'signal', 'spy_close_price']].copy(deep=True)                 # Represents contemporaneous pricing information
    retDF = cleanedDF[['date', 'dt', 'signalRet', 'fwdSpyRet']].dropna().copy(deep=True)          # Signal returns reflect day [T-1, T] returns.  fwdSpyRet reflects [T, T+1] returns

    #plt.scatter(retDF.signalRet, retDF.fwdSpyRet)
    #plt.show()
    #
    retDF[['signalRet']].hist(bins=25)
    plt.savefig(outputDir+"signalRetHist.png")
    plt.close()
    retDF[['fwdSpyRet']].hist(bins=25)
    plt.savefig(outputDir+"fwdSpyRetHist.png")
    plt.close()
    plt.show()





    ############################
    # (2) Exploratory Analysis #
    ############################

    # (a) Let's analyze if signal returns are predictive
    eao = ExploratoryAnalysis()
    eao.quantileReg(retDF, 'signalRet', 'fwdSpyRet', outputDir+"ret_quantileReg.png", quantiles=20)


    # (b) Let's analyze if price signals are cointegrated
    (mean, std, adfResult, params, resids) = eao.cointegration_test(pxDF, 'signal', 'spy_close_price')
    adf = adfResult[0]
    pval = adfResult[1]
    paramOutput = params[0]
    pxDF['resids'] = resids
    print("COINTEGRATION ANALYSIS indicates these series are cointegrated with spread (mu, sigma) => (" + str(mean) + ", "+str(std)+")")
    print("Test statistic of " + str(adf) + " is smaller than the 5% critical value of -2.87")
    print("PVAL: "+str(pval))




    # (c) Now that we know there might be a cointegrated strategy, let's focus on breadth of search to see if there's anything we can do with returns
    #     (i) Y ~ X
    #     (ii) Various other learners
    #     (iii) Ensemble different learners

#    gmo = GenModels()
#    learnerDict = dict()
#    learnerDict['base'] = dict()
#    learnerDict['meta'] = dict()
#    learnerDict['base']['GNB'] = GaussianNB()
#    learnerDict['base'].update(self.genModel.genGBM())
#    learnerDict['base'].update(self.genModel.genLR())
#    learnerDict['base'].update(self.genModel.genSVC())
#    learnerDict['base'].update(self.genModel.genRF())
#    learnerDict['base'].update(self.genModel.genAda())
#    learnerDict['base'].update(self.genModel.genET())

    modelDict = dict()
    for j in range(1, 10):
        numLags = j
        indVars = []
        currRetDF = retDF.copy(deep=True)
        for i in range(0, numLags):
            currRetDF['signalRet_'+str(i)] = list(currRetDF['signalRet'].shift(i))
            indVars.append('signalRet_'+str(i))
        ols = smf.ols('fwdSpyRet ~ '+"+".join(indVars), currRetDF).fit()
        olsDF = pd.DataFrame(ols.params)
        olsDF['lb'] = ols.conf_int()[0]
        olsDF['ub'] = ols.conf_int()[1]
        modelDict[j] = olsDF.copy(deep=True)
        
#        slo = SuperLearner(learnerDict, indVars, 'fwdSpyRet')
#        slo.cvFit(currRetDF, indVars, 'fwdSpyRet', folds=10)
#        slo.pred(currRetDF, indVars, 'fwdSpyRet')

    print("Odd Behavior where the most recent signal is not useful, but subsequent lags ARE???\n")



    #     (ii) Maybe the skew of the signal can be used as a conditional variable
    skewDF = pd.rolling_skew(retDF[['signalRet', 'fwdSpyRet']], 100)
    varDF = pd.rolling_var(retDF[['signalRet', 'fwdSpyRet']], 100)
    retAugDF = retDF.copy(deep=True)
    retAugDF['signalSkew'] = list(skewDF['signalRet'])
    retAugDF['fwdSpySkew'] = list(skewDF['fwdSpyRet'])
    retAugDF['signalVar'] = list(varDF['signalRet'])
    retAugDF['fwdSpyVar'] = list(varDF['fwdSpyRet'])
    retAugDF = retAugDF.dropna()
    





    eao = ExploratoryAnalysis()
    for type in preprocessedDF:
        print("\n\n#############################################################")
        print("EXPLORATORY ANALYSIS: ["+type.upper()+"]")
        eao.corrAnalysis(preprocessedDF[type], ['signal', 'spy_close_price', 'fwdSpyRet'])







    # TODO:
    #   (1)  Identify any errors in the data, flag them with a note, and suggest a corrected value or if advisable, you may choose to ignore them for purposes of your analysis.  
    #        Please explain what types of analysis you did to identify the errors, and provide any assumptions/intuition/formulas/scripts you may have used to help you find them (any language or pseudocode is fine).


    # (1) Preprocessing signal to generate a df [dfRaw => dfPP]
    #     (a) Normalize, standardize, winsorize, residualize, log-transform, raw
    #         Residualize can be done with ARIMA, GARCH, RNN, etc...
    #     (b) To be able to stack multiple transforms
    #
    # (2) Generate models (dfPP => dfPP_Model)  [Incorporate information about dependent variable]
    #     (a) Arbitrary time-horizons
    #
    # (3) Use input DF (raw, normalized, standardized, winsorized) to search for signals
    #     (a) Is there a signal in the rolling skew/kurtosis?
    #     (b) Are series' cointegrated?  Correlated?
    #     (c) Quantile Regression to see if there are regions where signal is stronger
    #     (d) SuperLearner applied to variety of transforms (residualizations)
    #     (e) Backwards (does S&P predict the signal?)
    #
    
    # (II)
    #    Overnight Gap vs. Intraday returns
    #    Prior to close (15-minutes)
