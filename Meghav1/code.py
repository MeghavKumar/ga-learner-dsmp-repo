# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
#pd.read_csv(path, index_col=0, parse_dates=True)
bank =pd.read_csv(path)
df = bank
categorical_var =df.select_dtypes(include = 'object')
print(categorical_var)
numerical_var=df.select_dtypes(include = 'number')
print(numerical_var)


# code starts here






# code ends here


# --------------
# code starts here

bank.drop(['Loan_ID'] ,axis =1, inplace = True)
banks = bank.copy()
#banks.head()
print(banks.isnull().sum())
bank_mode=banks.mode()
banks=banks.fillna(bank_mode.T.squeeze())
banks.head()

print(banks)

#code ends here


# --------------
# Code starts here




avg_loan_amount=pd.pivot_table(banks, values= 'LoanAmount' , index =('Gender', 'Married', 'Self_Employed'),aggfunc= 'mean')



# code ends here



# --------------
# code starts here

# code starts here

# code for loan aprroved for self employed
loan_approved_se = banks.loc[(banks["Self_Employed"]=="Yes")  & (banks["Loan_Status"]=="Y"), ["Loan_Status"]].count()
print(loan_approved_se)

# code for loan approved for non self employed
loan_approved_nse = banks.loc[(banks["Self_Employed"]=="No")  & (banks["Loan_Status"]=="Y"), ["Loan_Status"]].count()
print(loan_approved_nse)

# percentage of loan approved for self employed
percentage_se = (loan_approved_se * 100 / 614)
percentage_se=percentage_se[0]
# print percentage of loan approved for self employed
print(percentage_se)

#percentage of loan for non self employed
percentage_nse = (loan_approved_nse * 100 / 614)
percentage_nse=percentage_nse[0]
#print percentage of loan for non self employed
print (percentage_nse)

# code ends here
# code ends here





# --------------
# code starts here
#loan_term =df['Loan_Amount_Term'].apply(lambda x :int(x)/12)
loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12 )


big_loan_term=len(loan_term[loan_term>=25])

print(big_loan_term)



# code ends here


# --------------
# code starts here

loan_groupby = banks.groupby(['Loan_Status'])

loan_groupby = loan_groupby[['ApplicantIncome', 'Credit_History']]

mean_values=loan_groupby.agg([np.mean])


# code ends here


