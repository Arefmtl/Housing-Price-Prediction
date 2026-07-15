##adding libraris  

# import seaborn library for ploting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder #for encoding string data 1hot is better
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression



##reading and shffling data (pandas, mathplotlib)

housing =pd.read_csv(r"C:\Users\ASUS\Desktop\code\visionacademy\housing.csv")
#housing =pd.read_csv(r"C:\Users\ASUS\Desktop\code\visionacademy\housing.csv",header = 0,sep = ",", names=['name0','name1',...])
#housing_shuffled = housing.sample(frac=1, random_state=42)
#housing.head() == housin[0:6]
housing.info()
#housing.columns
#housing['ocean_proximity'].unique() 
#housing['ocean_proximity'].value_counts()
#housing[housing['ocean_proximity']=='ISLAND']
#housing[[housing['population'],[housing['ocean_proximity']=='ISLAND']]       
#housing.describe()

#plotting 


#housing.hist(bins = 50, figsize = (20, 20))
#plt.show()

# train_set && test set

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

train_set.shape

train_set.head()

data = train_set.copy()

data.head()
data.plot(kind = "scatter",x="longitude",s=data["population"]/15,label="population", y = "latitude",
          c=data['median_house_value'], cmap=plt.get_cmap('jet'),figsize=(15, 12), alpha=0.2)
data.shape



#standard correlation coefficient[-1, 1] 
#HAMBASTEGIYE DO MEYAR

corr_matrix = data.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


features = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(data[features], figsize=(20, 15))
plt.show()


data = train_set.copy()

data.head()
data.plot(kind = "scatter",x="median_income", y = 'median_house_value', figsize=(15, 10), alpha=0.5)
data.shape




#makong useful data 
data['total_rooms_per_households'] = data['total_rooms'] / data['households']
data['total_bedrooms_per_total_rooms'] = data['total_bedrooms'] / data['total_rooms']
data['population_per_households'] = data['population']/data['households']

data.head()
#ploting new corr plot
corr_matrix = data.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
features = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(data[features], figsize=(20, 15))
plt.show()

# cleaning data


df = train_set.copy()
df_label = df['median_house_value'].copy()
df = df.drop("median_house_value", axis=1)

df_num = df.drop("ocean_proximity", axis=1 )
df.info()

# Option 1
# df_num = df_num.dropna(subset=['total_bedrooms'])

# Option 2
#df_num = df_num.drop('total_bedrooms', axis=1)
#median = df_num['total_rooms'].median()  # Calculate median of 'total_rooms'
#df_num['total_rooms'].fillna(median, inplace=True)  # Fill missing values in 'total_rooms' with the calculated median
#option 3 #imputing data ~ #option 2
imputer = SimpleImputer(missing_values=np.nan, strategy="median")  # or strategy="most_frequent"
imputer.fit(df_num)
X = imputer.transform(df_num)
df_num_impute_tr = pd.DataFrame(X, columns=df_num.columns)
df_num_impute_tr.info()
df.info()
df_num_impute_tr.head()
#add custom attribute to df(useful data)
#makong useful data 
room_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6

class add_makedup_attribute(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        bedrooms_per_room = X[:, bedroom_ix] / X[:, room_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

custom = add_makedup_attribute()
data_custom_tr_tmp = custom.transform(df_num_impute_tr.values)
data_custom_tr = pd.DataFrame(data_custom_tr_tmp)

# Define 'columns' variable
columns = df_num.columns.tolist()
columns.extend(["rooms_per_household", "population_per_household", "bedrooms_per_room"])

data_custom_tr.columns = columns
data_custom_tr.head(10)

   

#feature scaling standardizaion and normalization [0, 1]#(for neural network)



# Assuming data_custom_tr is your DataFrame containing numerical features

# Initialize the StandardScaler
feature_scaler = StandardScaler()

# Scale the numerical features
data_num_scaled_tr = pd.DataFrame(feature_scaler.fit_transform(data_custom_tr.values), columns=data_custom_tr.columns)

# Display the scaled DataFrame
data_num_scaled_tr.head()


#labeling encodin or  onehot encoder # text to num algorithm

#label encoding problem make relation between encoded nums
encoder = LabelEncoder()

data_cat = df["ocean_proximity"]
data_cat_encoded = encoder.fit_transform(data_cat)
data_cat_encoded = pd.DataFrame(data_cat_encoded, columns=['ocean_proximity'])
data_cat_encoded.head()

#1hot encodinng alternative for last labeling model
#if coulmns not so many otherweise lbel encode
        


from sklearn.preprocessing import OneHotEncoder

# Initialize OneHotEncoder
encoder1hot = OneHotEncoder(sparse=False)

# Transform categorical variable 'ocean_proximity' using OneHotEncoder
data_cat_1hot_tmp = encoder1hot.fit_transform(df[['ocean_proximity']])

# Convert the transformed data into a DataFrame
data_cat_1hot = pd.DataFrame(data_cat_1hot_tmp)

# Set column names for the one-hot encoded features
feature_names_out = encoder1hot.get_feature_names_out(input_features=['ocean_proximity'])
data_cat_1hot.columns = feature_names_out

# Concatenate the one-hot encoded features with the scaled numerical features
final = pd.concat([data_num_scaled_tr, data_cat_1hot], axis=1)

# Display the first 10 rows of the final DataFrame
final.head(10)


num_pipeline = Pipeline([
    ('selector', add_makedup_attribute(num_attrs)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_attrs =["ocean_proximity"]
cat_pipeline = Pipeline([('selector', add_makedup_attribute(cat_attrs)),
                        ('one_hot_encoder', OneHotEncoder(sparse=False)),
                        ])

full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', numpipeline),
                                              ('cat_pipeline', catpipeline),
                                              ])

housing_prepared =full_pipeline.fit_transform(df)
housing_prepared_df = pd.DataFrame(housing_prepared, cloumns=['longitude', 'latittude', 'housinh_median_age', 'total_rooms',
                                                              'total_bedrooms', 'population', 'households', 'median_income',
                                                              'rooms_per_household', 'population_per_household', 'bedrooms_per_rooms',
                                                              'prox_<1h ocean', 'prox_inland', 'prox_island', 'prox_near bay', 'prox_near ocean'])
housing_prepared_df.head(10) 




lin_reg = LinearRegresion()
lin_reg.fit(housing_prepared_df, df_label)


