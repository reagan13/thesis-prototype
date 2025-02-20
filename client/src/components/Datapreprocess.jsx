import React, { useState, useEffect } from "react";
import { useData } from "../context/DataContext";

const Datapreprocess = () => {
  const { isSidebarCollapsed } = useData();
  const [codeTexts, setCodeTexts] = useState([
    `
#title PIP 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#title Load 
df = pd.read_csv('data.csv')

#title Data Cleaning - Handling Missing Values
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Data Transformation - Scaling Features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Categorical Data Transformation - Encoding Categorical Variables
categorical_features = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first')
encoded_categorical_data = encoder.fit_transform(df[categorical_features]).toarray()

# Combine the processed data
processed_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_features))
processed_df[numerical_features] = df[numerical_features]

# Train-Test Split
X = processed_df.drop('target', axis=1)
y = processed_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the first few rows of the processed data
print(X_train.head())
  `,
    `
#title PIP INSTALL
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#title Load the dataset
df = pd.read_csv('data.csv')

#title Data Cleaning - Handling Missing Values
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Data Transformation - Scaling Features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Categorical Data Transformation - Encoding Categorical Variables
categorical_features = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first')
encoded_categorical_data = encoder.fit_transform(df[categorical_features]).toarray()

# Combine the processed data
processed_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_features))
processed_df[numerical_features] = df[numerical_features]

# Train-Test Split
X = processed_df.drop('target', axis=1)
y = processed_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the first few rows of the processed data
print(X_train.head())
  `,
    `
#title PIP INSTALL
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#title Load the dataset
df = pd.read_csv('data.csv')

#title Data Cleaning - Handling Missing Values
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Data Transformation - Scaling Features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Categorical Data Transformation - Encoding Categorical Variables
categorical_features = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first')
encoded_categorical_data = encoder.fit_transform(df[categorical_features]).toarray()

# Combine the processed data
processed_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_features))
processed_df[numerical_features] = df[numerical_features]

# Train-Test Split
X = processed_df.drop('target', axis=1)
y = processed_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the first few rows of the processed data
print(X_train.head())
  `
  ]);

  const [nameTexts, setNameTexts] = useState([]);

  useEffect(() => {
    const newNameTexts = codeTexts.map(codeText => {
      const hashtagMatches = codeText.match(/#title\s+([A-Za-z0-9_ ]+)/g);
      return hashtagMatches ? hashtagMatches.map(tag => tag.replace(/#title\s+/, "").trim()).join(" / ") : "";
    });
    setNameTexts(newNameTexts);
  }, [codeTexts]);

  const cleanedCodeTexts = codeTexts.map(codeText => codeText.replace(/#title\s+([A-Za-z0-9_ ]+)/g, ""));

  return (
    <div
      className="bg-white h-[670px] w-[1230px] overflow-y-auto p-4"
      style={{
        width: isSidebarCollapsed ? "1400px" : "1200px",
        marginLeft: "30px",
        transition: "all 0.3s ease",
        scrollbarWidth: "none", msOverflowStyle: "none" 
      }}
    >
      <div
        className="mb-10 mt-[20px] ml-[40px] h-[610px] p-6 bg-white rounded-lg shadow-lg outline outline-2 outline-black outline-offset-2"
        style={{ width: isSidebarCollapsed ? "1300px" : "1100px", transition: "all 0.3s ease" }}
      >
        <div className="text-2xl font-bold mb-2 text-black text-center pb-[20px]">
          Code Flow
        </div>
        <div className="bg-white rounded-lg shadow-lg outline outline-2 outline-black outline-offset-2 p-4 h-[500px] overflow-y-auto">
          <p>Data preprocessing is a crucial step in preparing data for analysis...</p>
        </div>
      </div>
      {cleanedCodeTexts.map((codeText, index) => (
        <div
          key={index}
          className="mt-[20px] ml-[40px] h-[610px] p-6 bg-white rounded-lg shadow-lg outline outline-2 outline-black outline-offset-2"
          style={{ width: isSidebarCollapsed ? "1300px" : "1100px", transition: "all 0.3s ease" }}
        >
          <div 
            className="text-2xl font-bold mb-2 text-black text-center pb-[20px] mb-3 h-[30px]"
            style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
          >
            {nameTexts[index] || "#title"}
          </div>
          <div className="bg-white rounded-lg shadow-lg outline outline-2 outline-black outline-offset-2 p-4 h-[500px] overflow-y-auto" style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}>
            <pre className="overflow-auto">{codeText}</pre>
          </div>
        </div>
      ))}
    </div>  
  );
};

export default Datapreprocess;