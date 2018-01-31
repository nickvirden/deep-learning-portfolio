# All-in
- Just use all the variables
- Prior Knowledge; OR
- You have to; OR
- Preparing for Backward Elimination

# Backward Elimination
- **Step 1:** Select a significance level to stay in the model (e.g. SL = 0.05)
- **Step 2:** Fit the full model with all possible preditors
- **Step 3:** Consider the predictor with the highest P-value
-- If P > SL, go to **Step 4**, otherwise you're done!
- **Step 4:** Remove the predictor
- **Step 5:** Fit the model without this variable*
-- You need to rebuild the entire model because your coefficients will be different
- **Step 6:** Return to **Step 3**

# Forward Selection
- **Step 1:** Select a significance level to enter in the model (e.g. SL = 0.05)
- **Step 2:** Fit all simple regression models y ~ x(n)
- **Step 3:** Select the simple regression model with the lowest P-value
- **Step 4:** Keep this variable and fit all possible models with one extra predictor added to the ones you already have
--> Construct all possible linear regressions with two variables where one of those two variables is whatever you select.
- **Step 5:** Consider the predictor with the lowest P-value
--> If P < SL, return to **Step 3**, otherwise, you're done!

# Bidirectional Elimination
- **Step 1:** Select a significance level to enter and to stay in the model (e.g. SL Enter = 0.05, SL Stay = 0.05)
- **Step 2:** Perform the next step of a Forward Selection
--> A new variable must have P < SL Enter to enter
- **Step 3:** Perform all steps of Backward Elimination
--> The old variables must have P < SL Stay to stay
--> If you have 6 variables, you see how many you can eliminate
- **Step 4:** No new variables can enter and no old variables can exit
Your model is now ready!

# All Possible Models / Score Comparison
- **Step 1:** Select a criterion for goodness of fit (e.g. X^2, R^2, Akaike criterion)
- **Step 2:** Construct all possible regression models - there are 2^n - 1 total combinations
- **Step 3:** Select the model with the best criterion