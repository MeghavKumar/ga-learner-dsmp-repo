# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(path)
loan_status = data['Loan_Status'].value_counts()

data.plot(kind = 'bar',stacked =True )



#Code starts here


# --------------
#Code starts here
property_and_loan = data.groupby(['Property_Area', 'Loan_Status']).size().unstack()
property_and_loan.plot(kind = 'bar', stacked = False  )
plt.xlabel('Property Area')
plt.ylabel('Loan Status')

plt.xticks(rotation=45)


# --------------
#Code starts here
education_and_loan = data.groupby(['Education','Loan_Status']).size().unstack()
education_and_loan.plot(kind = 'bar', stacked = True)
plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation = 45)



# --------------
#Code starts here

graduate= data[data['Education'] == 'Graduate']
#print(graduate)
not_graduate= data[data['Education'] == 'Not Graduate']

graduate['LoanAmount'].plot(kind='density', label='Graduate')

not_graduate['LoanAmount'].plot(kind='density' , label = 'Not Graduate' )
plt.show()










#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig ,(ax_1,ax_2,ax_3) = plt.subplots(nrows = 3 , ncols = 1)
ax_1.plot(kind = 'scatter' , stacked = True )
plt.xlabel('ApplicantIncomes')
plt.ylabel('LoanAmount')
fig.suptitle('Applicant Income')

ax_2.plot(kind = 'scatter' , stacked = True )
plt.xlabel('CoapplicantIncome')
plt.ylabel('LoanAmount')
fig.suptitle('CoapplicantIncome')

data['TotalIncome'] = data['ApplicantIncome']+data['CoapplicantIncome']
ax_3.plot(kind = 'scatter' , stacked = True )
plt.xlabel('TotalIncome')
plt.ylabel('LoanAmount')
fig.suptitle('TotalIncome')
plt.show()


