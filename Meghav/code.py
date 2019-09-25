# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
path
data= np.genfromtxt(path,delimiter=",", skip_header=1)
print(data)
print(type(data))
#New record
new_record=np.array([[50,  9,  4,  1,  0,  0, 40,  0]])
print(type(new_record))
census=np.concatenate((data,new_record))
print(census)

#Code starts here



# --------------
#Code starts here
age = np.array(census[ :,[0]])
print(age)
max_age=np.max(age)
print(max_age)
min_age=np.min(age)
print(min_age)
age_mean=np.mean(age)
age_std=np.std(age)
print(age_std)


# --------------
#Code starts here

#creating race array
#race = np.array(census[ :, [2]])
#print(race[15])
#race_0 = race[race==0]
#len_0 = race_0.size
#race_0.reshape((1,len_0))
race_0=census[census[ :,2] == 0]
print(race_0)
#race_1 = race[race==1]
race_1=census[census[ :,2] == 1]
print(race_1)
#race_2 = race[race==2]
race_2=census[census[ :,2] == 2]
print(race_2)
#race_3 = race[race==3]
race_3=census[census[ :,2] == 3]
print(race_3)
race_4=census[census[ :,2] == 4]
#race_4 = race[race==4]
print(race_4)

#finding lenth

#len_0 = race_0.size
len_0 = len(race_0)
print(len_0)
#len_1 = race_1.size
len_1 = len(race_1)
print(len_1)
#len_2 = race_2.size
len_2 = len(race_2)
print(len_2)
#len_3 = race_3.size
len_3 = len(race_3)
print(len_3)
#len_4 = race_4.size
len_4 = len(race_4)
print(len_4)

# min of race
minority_race1=min(len_0,len_1,len_2,len_3,len_4)
if minority_race1 == len_0:
    minority_race = 0
    print(minority_race)
elif minority_race1 == len_1:
    minority_race = 1
    print(minority_race)
elif minority_race1 == len_2:
    minority_race = 2
    print(minority_race)
elif minority_race1 == len_3:
    minority_race = 3
    print(minority_race)
else:
    minority_race = 4
    print(minority_race)







# --------------
#Code starts here
import numpy as np 
# creating senior citizen array
race_0=census[census[ :,2] == 0]
senior_citizens=census[census[ : ,0]>60]
print(type(senior_citizens))
print(senior_citizens)

#working hours of senior citizen

whs = np.sum(senior_citizens, axis = 0)
print(whs)
working_hours_sum =whs[6]
print(working_hours_sum)

senior_citizens_len = len(senior_citizens)

print(senior_citizens_len)

#Average working hours

avg_working_hours=(working_hours_sum/senior_citizens_len)
print(avg_working_hours)


# --------------
#Code starts here
senior_citizens=census[census[ : ,0]>60]
high = census[census[ : ,1 ]>10]
print(high)
low = census[census [:  ,1]<=10]
print(low)

# Average pay
mic = np.mean(high ,axis =0)
avg_pay_high =mic[7]
mic1 = np.mean(low ,axis =0)
avg_pay_low=mic1[7]
print(avg_pay_high)
print(avg_pay_low)
print(avg_pay_high>avg_pay_low)





