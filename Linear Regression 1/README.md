# Linear Regression Analysis

This repository contains Python code for performing linear regression analysis on two datasets:
1. Grade Point Average (GPA) vs ACT Scores
2. US Wages dataset analysis

## Prerequisites

To run this code, you'll need the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels

You can install these packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

## Required Data Files

The code requires two CSV files to be present in the same directory:
1. `Grade Point Average Data.csv` - Contains ACT scores and GPA data
2. `uswages.csv` - Contains wage data with education and experience information

## Running the Code

Simply run the Python script:
```bash
python linearregression1.py
```

## Analysis Results

### Problem 1: ACT Scores vs GPA Analysis

The code performs the following analyses:
1. Creates a scatter plot of ACT scores against GPA
2. Fits a linear regression line to the data
3. Calculates the correlation between ACT scores and GPA
4. Provides a detailed regression model summary

#### Key Findings:
- The correlation between ACT scores and GPA is 0.269, indicating a weak positive relationship
- The regression model shows:
  - R-squared of 0.073, meaning ACT scores explain only about 7.3% of the variance in GPA
  - The coefficient for ACT scores (X) is 0.0388, suggesting that for each point increase in ACT score, GPA increases by approximately 0.039 points
  - The relationship is statistically significant (p-value = 0.003)
  - The model has 120 observations

### Problem 2: US Wages Analysis

The analysis includes:
1. Basic statistics of the dataset
2. Correlation analysis between wages, education, and experience
3. Scatter plots showing relationships between:
   - Wages vs Education
   - Wages vs Experience
4. Analysis of wage differences by race
5. Two separate regression models:
   - Wages vs Education
   - Wages vs Experience

#### Key Findings:
1. Dataset Overview:
   - 2000 observations in the dataset
   - Average wage: $608.12
   - Average education: 13.11 years
   - Average experience: 18.41 years

2. Correlation Analysis:
   - Wages and Education: 0.248 (moderate positive correlation)
   - Wages and Experience: 0.183 (weak positive correlation)
   - Education and Experience: -0.302 (moderate negative correlation)
   - Wages and Race: -0.096 (very weak negative correlation)

3. Education Regression Model:
   - R-squared: 0.062 (education explains 6.2% of wage variance)
   - Each additional year of education is associated with a $38.01 increase in wages
   - Highly statistically significant (p-value < 0.001)

4. Experience Regression Model:
   - R-squared: 0.034 (experience explains 3.4% of wage variance)
   - Each additional year of experience is associated with a $6.30 increase in wages
   - Highly statistically significant (p-value < 0.001)

#### Conclusions:
1. ACT Scores and GPA:
   - There is a weak but statistically significant relationship between ACT scores and GPA
   - ACT scores alone are not a strong predictor of college GPA

2. Wage Analysis:
   - Both education and experience have positive effects on wages
   - Education has a stronger impact on wages than experience
   - The relationship between race and wages shows a very weak negative correlation
   - The models suggest that human capital (education and experience) plays a significant role in determining wages
   - There is a negative correlation between education and experience, suggesting that people with more education tend to have less work experience

## Output
Correlation between ACT and GPA: 0.26948180326626364

Regression Model Summary:
                            OLS Regression Results
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.073
Model:                            OLS   Adj. R-squared:                  0.065
Method:                 Least Squares   F-statistic:                     9.240
Date:                Tue, 27 May 2025   Prob (F-statistic):            0.00292
Time:                        18:29:29   Log-Likelihood:                -112.50
No. Observations:                 120   AIC:                             229.0
Df Residuals:                     118   BIC:                             234.6
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.1140      0.321      6.588      0.000       1.479       2.750
X              0.0388      0.013      3.040      0.003       0.014       0.064
==============================================================================
Omnibus:                       26.969   Durbin-Watson:                   1.831
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               47.360
Skew:                          -0.994   Prob(JB):                     5.20e-11
Kurtosis:                       5.349   Cond. No.                         142.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Number of observations: 2000

Summary Statistics:
              wage         educ        exper         race       smsa           ne           mw           so          we           pt
count  2000.000000  2000.000000  2000.000000  2000.000000  2000.0000  2000.000000  2000.000000  2000.000000  2000.00000  2000.000000   
mean    608.117865    13.111000    18.410500     0.078000     0.7560     0.229000     0.248500     0.312500     0.21000     0.092500   
std     459.832629     3.004196    13.375778     0.268239     0.4296     0.420294     0.432251     0.463628     0.40741     0.289803   
min      50.390000     0.000000    -2.000000     0.000000     0.0000     0.000000     0.000000     0.000000     0.00000     0.000000   
25%     308.640000    12.000000     8.000000     0.000000     1.0000     0.000000     0.000000     0.000000     0.00000     0.000000   
50%     522.320000    12.000000    15.000000     0.000000     1.0000     0.000000     0.000000     0.000000     0.00000     0.000000   
75%     783.480000    16.000000    27.000000     0.000000     1.0000     0.000000     0.000000     1.000000     0.00000     0.000000   
max    7716.050000    18.000000    59.000000     1.000000     1.0000     1.000000     1.000000     1.000000     1.00000     1.000000   

Correlation Matrix:
           wage      educ     exper
wage   1.000000  0.248336  0.183201
educ   0.248336  1.000000 -0.302479
exper  0.183201 -0.302479  1.000000

Correlation between wage and race: -0.09622038659128118

Education Regression Model Summary:
                            OLS Regression Results
==============================================================================
Dep. Variable:                   wage   R-squared:                       0.062
Model:                            OLS   Adj. R-squared:                  0.061
Method:                 Least Squares   F-statistic:                     131.3
Date:                Tue, 27 May 2025   Prob (F-statistic):           1.72e-29
Time:                        18:29:32   Log-Likelihood:                -15035.
No. Observations:                2000   AIC:                         3.007e+04
Df Residuals:                    1998   BIC:                         3.009e+04
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        109.7538     44.616      2.460      0.014      22.254     197.253
educ          38.0111      3.317     11.459      0.000      31.506      44.516
==============================================================================
Omnibus:                     1980.408   Durbin-Watson:                   1.910
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           231041.874
Skew:                           4.419   Prob(JB):                         0.00
Kurtosis:                      54.908   Cond. No.                         60.6
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Experience Regression Model Summary:
                            OLS Regression Results
==============================================================================
Dep. Variable:                   wage   R-squared:                       0.034
Model:                            OLS   Adj. R-squared:                  0.033
Method:                 Least Squares   F-statistic:                     69.39
Date:                Tue, 27 May 2025   Prob (F-statistic):           1.48e-16
Time:                        18:29:32   Log-Likelihood:                -15065.
No. Observations:                2000   AIC:                         3.013e+04
Df Residuals:                    1998   BIC:                         3.015e+04
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        492.1669     17.204     28.607      0.000     458.427     525.907
exper          6.2981      0.756      8.330      0.000       4.815       7.781
==============================================================================
Omnibus:                     1741.557   Durbin-Watson:                   1.898
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           112147.928
Skew:                           3.749   Prob(JB):                         0.00
Kurtosis:                      38.910   Cond. No.                         38.8
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The code generates:
1. Multiple scatter plots saved as PNG files
2. Statistical summaries printed to the console
3. Regression model summaries with coefficients and statistical significance

## Note

Make sure both data files are in the same directory as the Python script before running the code. 