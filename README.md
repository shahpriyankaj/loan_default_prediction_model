# Load Default Prediction Model

### Problem Statement
The challenge is to build a model that predicts at application time whether an applicant will become bad loan or not. The data consists of a set of variables that describe the requested loan and the applicant's credit attributes (annual income, debt-to-income ratio, number of credit inquiries in the last 6 months, etc.).

### Objective
Perform exploratory data analysis (EDA) on a mixed-feature dataset and build a binary classification neural network model using PyTorch.

### Data Dictionary
- annual_inc: The annual income provided by the borrower during application.
- bc_util: Ratio of total current balance to high credit/credit limit for all bankcard accounts.
- desc: Loan description provided by the borrower
- dti: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations excluding mortgage and the requested loan divided by the borrower's self-reported monthly income.
- emp_length: Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
- home_ownership: The home ownership status provided by the borrower during registration. Our values are: RENT OWN MORTGAGE OTHER.
- id: A unique assigned ID for the loan listing.
- inq_last_6mths: The number of inquiries by creditors during the past 6 months.
- int_rate: Interest Rate on the loan
- loan_amnt: The listed amount of the loan applied for by the borrower.
- member_id: A unique assigned Id for the borrower member.
- mths_since_last_major_derog: Months since most recent 90-day or worse rating
- mths_since_recent_inq: Months since most recent inquiry.
- percent_bc_gt_75: Percentage of all bankcard accounts > 75% of limit.
- purpose: A category provided by the borrower for the loan request.
- revol_util: Revolving line utilization rate or the amount of credit the borrower is using relative to all available revolving credit.
- term: The number of payments on the loan. Values are in months and can be either 36 or 60.
- tot_cur_bal: Total current balance of all accounts
- tot_hi_cred_lim: Total high credit/credit limit
- total_bc_limit: Total bankcard high credit/credit limit
- application approved_flag: Indicates if the loan application is approved or not
- internal_score: An third party vendor's risk score generated when the application is made
- bad_flag: Target variable indicates if the loan is eventually bad or not

### Project Structure
- data_preparation python notebook contains data loading and Exploratory Data Analysis (EDA).