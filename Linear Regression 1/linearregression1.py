import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Problem 1
# a) Import the data
grade_data = pd.read_csv("Grade Point Average Data.csv")

# b) Plot ACT against GPA
plt.figure(figsize=(10, 6))
plt.scatter(grade_data['X'], grade_data['Y'])
plt.xlabel('ACT TEST SCORES')
plt.ylabel('GPA')
plt.title('ACT Scores vs GPA')

# Fit linear regression
X = sm.add_constant(grade_data['X'])
model = sm.OLS(grade_data['Y'], X).fit()
plt.plot(grade_data['X'], model.predict(X), 'r-')
plt.show()

# c) Calculate correlation
correlation = grade_data['X'].corr(grade_data['Y'])
print(f"Correlation between ACT and GPA: {correlation}")

# d) Build regression model
#model = sm.OLS(grade_data['Y'], X).fit()
print("\nRegression Model Summary:")
print(model.summary())

# Problem 2
# Load the uswages dataset
uswages = pd.read_csv("uswages.csv")

# a) Number of observations
print(f"Number of observations: {len(uswages)}")

# b) Calculate mean and median
summary_stats = uswages.describe()
print("\nSummary Statistics:")
print(summary_stats)

# c) Calculate correlations and plot
correlation_matrix = uswages[['wage', 'educ', 'exper']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot wage vs education
plt.figure(figsize=(10, 6))
plt.scatter(uswages['educ'], uswages['wage'])
plt.xlabel('Years of Education')
plt.ylabel('Wages')
plt.title('Education vs Wages')
plt.show()

# Plot wage vs experience
plt.figure(figsize=(10, 6))
plt.scatter(uswages['exper'], uswages['wage'])
plt.xlabel('Years of Experience')
plt.ylabel('Wages')
plt.title('Experience vs Wages')
plt.show()

# d) Race wage difference
race_correlation = uswages['wage'].corr(uswages['race'])
print(f"\nCorrelation between wage and race: {race_correlation}")

# e) Regression model with education
X_educ = sm.add_constant(uswages['educ'])
model_educ = sm.OLS(uswages['wage'], X_educ).fit()
print("\nEducation Regression Model Summary:")
print(model_educ.summary())

# f) Regression model with experience
X_exper = sm.add_constant(uswages['exper'])
model_exper = sm.OLS(uswages['wage'], X_exper).fit()
print("\nExperience Regression Model Summary:")
print(model_exper.summary()) 