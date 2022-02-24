import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the train data set and test data set
df_train = pd.read_csv(r'E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\train.csv')
df_test = pd.read_csv(r'E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\test.csv')

################################# Exploratory Data analysis ###########################################
#name of the columns
df_train.columns

####
# Index(['Severity', 'Safety_Score', 'Days_Since_Inspection',
#        'Total_Safety_Complaints', 'Control_Metric', 'Turbulence_In_gforces',
#        'Cabin_Temperature', 'Accident_Type_Code', 'Max_Elevation',
#        'Violations', 'Adverse_Weather_Metric', 'Accident_ID']

#check for null values in any column
df_train.isnull().sum()  ##no null value found

#Target variable Severity analysis
#check for unique values
df_train['Severity'].unique()
# array(['Minor_Damage_And_Injuries', 'Significant_Damage_And_Fatalities',
#        'Significant_Damage_And_Serious_Injuries',
#        'Highly_Fatal_And_Damaging'], dtype=object)

#check for balance of data - group by and check counts
df_target_var = df_train.groupby('Severity').size().reset_index(name='counts')
df_target_var.columns

# Pie Chart
# Create a list of colors
# colors = ["#E13F29", "#D69A80", "#D63B59", "#AE5552", "#CB5C3B", "#EB8076", "#96624E"]

# Create a pie chart
plt.clf()
plt.pie(
    # using data total)arrests
    df_target_var['counts'],
    # with the labels being officer names
    labels=df_target_var['Severity'],
    # with no shadows
    shadow=False,
    # # with colors
    # colors=colors,
    # with one slide exploded out
    explode=(0, 0, 0, 0),
    # with the start angle at 90%
    startangle=90,
    # with the percent listed as a fraction
    autopct='%1.1f%%',
    )

# View the plot drop above
plt.axis('equal')

# View the plot
plt.tight_layout()
plt.show()
###  Pie chart shows that data is not that much imbalanced so weight adjustment is not needed

#EDA -  Safety_Score
df_train['Safety_Score'].describe()
plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Safety_Score'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Safety_Score'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Safety_Score'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Safety_Score'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
## there is a safety score for which there are fatalaties

plt.hist(df_train['Safety_Score']) #seems normally disctributed

#EDA - Days_Since_Inspection
df_train['Days_Since_Inspection'].describe()
plt.hist(df_train['Days_Since_Inspection']) #seems normally distributed
plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Days_Since_Inspection'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Days_Since_Inspection'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Days_Since_Inspection'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Days_Since_Inspection'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#seems relation with Days_Since_Inspection for a particular range

#EDA Total_Safety_Complaints
df_train['Total_Safety_Complaints'].describe()
plt.hist(df_train['Total_Safety_Complaints']) #not normally distributed
plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Total_Safety_Complaints'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Total_Safety_Complaints'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Total_Safety_Complaints'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Total_Safety_Complaints'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#There is high co-relation with Highly_Fatal_And_Damaging

#EDA Control_Metric
df_train['Control_Metric'].describe()
plt.hist(df_train['Control_Metric']) #seems normally distributed
plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Control_Metric'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Control_Metric'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Control_Metric'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Control_Metric'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#There is high co-relation with Highly_Fatal_And_Damaging

#EAD  Turbulence_In_gforces
df_train['Turbulence_In_gforces'].describe()
plt.hist(df_train['Turbulence_In_gforces']) #seems normally distributed
# appropriate scaling to see corelation
df_train['Turbulence_In_gforces_trans'] = df_train['Turbulence_In_gforces'].apply(lambda x: x*100)

plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Turbulence_In_gforces_trans'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Turbulence_In_gforces_trans'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Turbulence_In_gforces_trans'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Turbulence_In_gforces_trans'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#There is high co-relation with Highly_Fatal_And_Damaging  and for a specific range

#EDA  Cabin_Temperature
df_train['Cabin_Temperature'].describe()
plt.hist(df_train['Cabin_Temperature']) #seems skwed distributed

plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Cabin_Temperature'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Cabin_Temperature'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Cabin_Temperature'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Cabin_Temperature'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#Highly and only co-rleated to Highly_Fatal_And_Damaging

#EDA  Accident_Type_Code
df_train['Accident_Type_Code'].describe()
plt.hist(df_train['Accident_Type_Code'])

plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Accident_Type_Code'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Accident_Type_Code'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Accident_Type_Code'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Accident_Type_Code'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#This field  to be used as categories - OneHotEncoder

#EDA - Max_Elevation

df_train['Max_Elevation'].describe()
plt.hist(df_train['Max_Elevation']) #normal distributed

plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Max_Elevation'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Max_Elevation'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Max_Elevation'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Max_Elevation'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#hughliy correlated to Highly_Fatal_And_Damaging . large values need to scale down

#EDA Violations
df_train['Violations'].describe()
plt.hist(df_train['Violations']) #normal distributed

plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Violations'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Violations'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Violations'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Violations'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#Categorical variable - One Hot Encoding needed

#EDA - Adverse_Weather_Metric
df_train['Adverse_Weather_Metric'].describe()
plt.hist(df_train['Adverse_Weather_Metric']) #skewed distributed

plt.clf()
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df_train[df_train['Severity']=='Minor_Damage_And_Injuries']['Adverse_Weather_Metric'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Fatalities']['Adverse_Weather_Metric'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Significant_Damage_And_Serious_Injuries']['Adverse_Weather_Metric'],bins=bins,alpha =0.8)
plt.hist(df_train[df_train['Severity']=='Highly_Fatal_And_Damaging']['Adverse_Weather_Metric'],bins=bins,alpha =0.8)
plt.legend(('Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries','Highly_Fatal_And_Damaging'))
plt.show()
#corelatuon with all types . smaller values

##################################### DATA PREPROCESSING #################################################

