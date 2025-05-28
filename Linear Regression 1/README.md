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

The code generates:
1. Multiple scatter plots saved as PNG files
2. Statistical summaries printed to the console
3. Regression model summaries with coefficients and statistical significance

## Note

Make sure both data files are in the same directory as the Python script before running the code. 