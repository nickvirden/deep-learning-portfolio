# Random Forest Intuition
- **Step 1:** Pick a random K data points from the Training set
- **Step 2:** Build a Decision Tree associated to the K data point
- **Step 3:** Choose the number Ntree of trees you want to build
- **Step 4:** Repeat Steps 1 & 2
- **Step 4:** For a new data point, make each one of your Ntree trees predict the value of Y for the data point in question
- **Step 5:** Assign the new data point the average across all of the predicted Y values