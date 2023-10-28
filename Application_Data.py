#importing the libaries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#importing application_data dataset
df=pd.read_csv(r"C:\Users\YASH\Desktop\pandas\application_data.csv")

# Identify Missing Data and Deal with it Appropriately
a=df.isnull().sum()/len(df)*100

df.info()

#droping the column having missing value greater than 40 %
df=df.drop(['OWN_CAR_AGE','EXT_SOURCE_1','APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG','COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI','FONDKAPREMONT_MODE','HOUSETYPE_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE'],axis=1)

#filling missing value for numerical column using medianfunction
df['AMT_ANNUITY']=df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median())
df['AMT_GOODS_PRICE']=df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())
df['CNT_FAM_MEMBERS']=df['CNT_FAM_MEMBERS'].fillna(df['CNT_FAM_MEMBERS'].median())
df['EXT_SOURCE_2']=df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median())
df['EXT_SOURCE_3']=df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median())
df['OBS_30_CNT_SOCIAL_CIRCLE']=df['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(df['OBS_30_CNT_SOCIAL_CIRCLE'].median())
df['DEF_30_CNT_SOCIAL_CIRCLE']=df['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(df['DEF_30_CNT_SOCIAL_CIRCLE'].median())
df['OBS_60_CNT_SOCIAL_CIRCLE']=df['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(df['OBS_60_CNT_SOCIAL_CIRCLE'].median())
df['DEF_60_CNT_SOCIAL_CIRCLE']=df['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(df['DEF_60_CNT_SOCIAL_CIRCLE'].median())
df['DAYS_LAST_PHONE_CHANGE']=df['DAYS_LAST_PHONE_CHANGE'].fillna(df['DAYS_LAST_PHONE_CHANGE'].median())
df['AMT_REQ_CREDIT_BUREAU_HOUR']=df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(df['AMT_REQ_CREDIT_BUREAU_HOUR'].median())
df['AMT_REQ_CREDIT_BUREAU_DAY']=df['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(df['AMT_REQ_CREDIT_BUREAU_DAY'].median())
df['AMT_REQ_CREDIT_BUREAU_WEEK']=df['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(df['AMT_REQ_CREDIT_BUREAU_WEEK'].median())
df['AMT_REQ_CREDIT_BUREAU_MON']=df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(df['AMT_REQ_CREDIT_BUREAU_MON'].median())
df['AMT_REQ_CREDIT_BUREAU_QRT']=df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(df['AMT_REQ_CREDIT_BUREAU_QRT'].median())
df['AMT_REQ_CREDIT_BUREAU_YEAR']=df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(df['AMT_REQ_CREDIT_BUREAU_YEAR'].median())

#filling missing value for categorial column using mode function
df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].fillna(df['NAME_TYPE_SUITE'].mode()[0])
df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].fillna(df['OCCUPATION_TYPE'].mode()[0])

#droping unwanted column 
df=df.drop(['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','CNT_FAM_MEMBERS','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21','EXT_SOURCE_2','EXT_SOURCE_3'],axis=1)

#converting '-ve' values to '+ve' values 
df['DAYS_BIRTH']=df['DAYS_BIRTH'].abs()
df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].abs()
df['DAYS_REGISTRATION']=df['DAYS_REGISTRATION'].abs()
df['DAYS_ID_PUBLISH']=df['DAYS_ID_PUBLISH'].abs()
df['DAYS_LAST_PHONE_CHANGE']=df['DAYS_LAST_PHONE_CHANGE'].abs()

# Identify Outliers in the Dataset
b=df.describe()
sns.boxplot(y='CNT_CHILDREN',data=df)
sns.boxplot(y='AMT_INCOME_TOTAL',data=df)
sns.boxplot(y='AMT_CREDIT',data=df)
sns.boxplot(y='AMT_ANNUITY',data=df)
sns.boxplot(y='DAYS_EMPLOYED',data=df)
sns.boxplot(y='DAYS_REGISTRATION',data=df)

#data imbalance
df['TARGET'].value_counts()

repayers=df.loc[df['TARGET']==0]
defaulters=df.loc[df['TARGET']==1]
round(len(repayers)/len(defaulters),2)
sns.countplot(data=df,x='TARGET')
plt.ylabel('count of repayers and defaulters')
plt.title('Imbalance')
plt.xlabel('loan repayment')
plt.xticks([0,1],['repayers','defaulters'])

#Perform Univariate, Segmented Univariate, and Bivariate
bins=[0,100000,200000,300000,400000,500000,10000000000]
slot=['<100000','100000-200000','200000-300000','300000-400000','400000-500000','500000>']
df['AMT_INCOME_TOTAL']=pd.cut(df['AMT_INCOME_TOTAL'], bins,labels=slot)
df['AMT_CREDIT']=pd.cut(df['AMT_CREDIT'], bins,labels=slot)

#Univariate analysis
sns.countplot(x='NAME_CONTRACT_TYPE',hue='TARGET',data=df)
sns.countplot(x='NAME_INCOME_TYPE',hue='TARGET',data=df)
plt.xticks(rotation=45)
sns.countplot(x='AMT_CREDIT',hue='TARGET',data=df)
plt.xticks(rotation=45)
sns.countplot(x='AMT_INCOME_TOTAL',hue='TARGET',data=df)
plt.xticks(rotation=45)
sns.countplot(x='CODE_GENDER',hue='TARGET',data=df)
sns.countplot(x='FLAG_OWN_CAR',hue='TARGET',data=df)

sns.boxplot(y='AMT_ANNUITY',x='TARGET',data=df)
sns.boxplot(y='AMT_GOODS_PRICE',x='TARGET',data=df)
sns.boxplot(y='DAYS_BIRTH',x='TARGET',data=df)
sns.boxplot(y='DAYS_EMPLOYED',x='TARGET',data=df)
sns.boxplot(y='DAYS_LAST_PHONE_CHANGE',x='TARGET',data=df)
sns.boxplot(y='DAYS_ID_PUBLISH',x='TARGET',data=df)

#Bivariate analysis 
#For target 0
sns.boxplot(data=defaulters,x='NAME_EDUCATION_TYPE',y='AMT_CREDIT',hue='NAME_FAMILY_STATUS')
plt.xticks(rotation=45)
plt.title('credit amount vs education status')

sns.boxplot(data=defaulters,x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL',hue='NAME_FAMILY_STATUS',orient='v')
plt.yscale('log')
plt.xticks(rotation=45)
plt.title('income amount vs education status')

#For target 1
sns.boxplot(data=repayers,x='NAME_EDUCATION_TYPE',y='AMT_CREDIT',hue='NAME_FAMILY_STATUS')
plt.xticks(rotation=45)
plt.title('credit amount vs education status')

sns.boxplot(data=repayers,x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL',hue='NAME_FAMILY_STATUS',orient='v')
plt.yscale('log')
plt.xticks(rotation=45)
plt.title('income amount vs education status')


#Top Correlations for Different Scenarios
sns.heatmap(data=df)

#for target 0
cor = defaulters.corr()
cordf = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))
cordf = cordf.unstack().reset_index()
cordf.columns = ['Var1', 'Var2', 'Correlation']
cordf.dropna(subset = ['Correlation'], inplace = True)
cordf['Correlation'] = round(cordf['Correlation'], 2)
cordf['Correlation'] = abs(cordf['Correlation'])
corr0=cordf.sort_values(by = 'Correlation', ascending = False).head(10)

#for target 1
corr = repayers.corr()
corrdf = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corrdf = corrdf.unstack().reset_index()
corrdf.columns = ['Var1', 'Var2', 'Correlation']
corrdf.dropna(subset = ['Correlation'], inplace = True)
corrdf['Correlation'] = round(corrdf['Correlation'], 2)
corrdf['Correlation'] = abs(corrdf['Correlation'])
corrdf.sort_values(by = 'Correlation', ascending = False).head(10)
corr1=corrdf.sort_values(by = 'Correlation', ascending = False).head(10)


#importing previous_application dataset
fd=pd.read_csv(r'C:\Users\YASH\Desktop\pandas\previous_application.csv')

# Identify Missing Data and Deal with it Appropriately
d=fd.isnull().sum()/len(fd)*100

#droping the column having missing value around or greater than 40%
fd=fd.drop(['AMT_DOWN_PAYMENT','RATE_DOWN_PAYMENT','RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED','NAME_TYPE_SUITE','DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL'],axis=1)

#filling missing value for categorial column with mode function
fd['PRODUCT_COMBINATION']=fd['PRODUCT_COMBINATION'].fillna(fd['PRODUCT_COMBINATION'].mode()[0])

#filling missing value for numerical column with median function
fd['AMT_ANNUITY']=fd['AMT_ANNUITY'].fillna(fd['AMT_ANNUITY'].median())
fd['AMT_GOODS_PRICE']=fd['AMT_GOODS_PRICE'].fillna(fd['AMT_GOODS_PRICE'].median())
fd['CNT_PAYMENT']=fd['CNT_PAYMENT'].fillna(fd['CNT_PAYMENT'].median())

#converting '-ve' values to '+ve' values
fd['DAYS_DECISION']=fd['DAYS_DECISION'].abs()
fd['SELLERPLACE_AREA']=fd['SELLERPLACE_AREA'].abs()

#Identify Outliers in the Dataset
e=fd.describe()
sns.boxplot(data=fd,y='AMT_ANNUITY')
sns.boxplot(data=fd,y='AMT_CREDIT')
sns.boxplot(data=fd,y='SELLERPLACE_AREA')
sns.boxplot(data=fd,y='AMT_APPLICATION')
sns.boxplot(data=fd,y='AMT_GOODS_PRICE')
sns.boxplot(data=fd,y='CNT_PAYMENT')
sns.boxplot(data=fd,y='DAYS_DECISION')

#Perform Univariate, Segmented Univariate, and Bivariate
#univariate analysis
sns.countplot(data=fd,x='NAME_CONTRACT_TYPE')
sns.countplot(data=fd,x='NAME_CLIENT_TYPE')
sns.countplot(data=fd,x='NAME_SELLER_INDUSTRY')
plt.xticks(rotation=45)

#bivariate analysis
sns.countplot(data=fd,x='NAME_CLIENT_TYPE',hue='NAME_CONTRACT_STATUS')
sns.countplot(data=fd,x='NAME_CONTRACT_TYPE',hue='NAME_CONTRACT_STATUS')
sns.countplot(data=fd,x='NAME_YIELD_GROUP',hue='NAME_CONTRACT_STATUS')
sns.countplot(data=fd,x='NAME_PRODUCT_TYPE',hue='NAME_CONTRACT_STATUS')
sns.countplot(data=fd,x='CHANNEL_TYPE',hue='NAME_CONTRACT_STATUS')
plt.xticks(rotation=45)


# Merging the Application dataset with previous appliaction dataset
comb = pd.merge(left=df,right=fd,how='inner',on='SK_ID_CURR',suffixes='_x')


comb = comb.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',
                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',
                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',
                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',
                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',
                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)


comb.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 
              'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
              'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',
              'HOUR_APPR_PROCESS_START_PREV', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)
f=comb.columns

#univariate analysis
sns.countplot(data=comb,y='NAME_CASH_LOAN_PURPOSE',hue='NAME_CONTRACT_STATUS')
plt.xscale('log')

sns.countplot(data=comb,y='NAME_CASH_LOAN_PURPOSE',hue='TARGET')
plt.xscale('log')

#Bivariate analysis
sns.boxplot(data=comb,x='NAME_CASH_LOAN_PURPOSE',y='AMT_CREDIT_PREV',hue='NAME_INCOME_TYPE')
plt.xticks(rotation=90)
plt.yscale('log')
plt.title('Prev Credit amount vs Loan Purpose')


plt.xticks(rotation=90)
sns.barplot(data =comb, y='AMT_CREDIT_PREV',hue='TARGET',x='NAME_HOUSING_TYPE')
plt.title('Prev Credit amount vs Housing type')








