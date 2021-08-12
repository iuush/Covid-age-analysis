import xlrd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd

#Question 1(a)
print("\nQ1(i)")
s1=("Covid19IndiaData_30032020.xlsx")
wb = xlrd.open_workbook(s1) 
sheet = wb.sheet_by_index(0)
Age=[]
sum1=0
sheet.cell_value(0,0)
for i in range(sheet.ncols):
    if sheet.cell_value(0,i)=='Age':
        for j in range(1,sheet.nrows):
            Age.append(int(sheet.cell_value(j,i)))
            sum1+=int(sheet.cell_value(j,i))
mean=sum1/sheet.nrows

x=list(set(Age))
y=[]
print("X represents the age of the infected person")
var=0
exp=0
for i in range(len(x)):
    count=0
    for j in range(len(Age)):
        if x[i]==Age[j]:
            count+=1
    y.append(count)     
    y[i]=y[i]/len(Age)  #calculating p(X=i)
    var+=((x[i]-mean)**2)*y[i]  #var=E(X-E(X))^2
    #print("P(X = ",x[i],") is ",y[i])
    exp+=x[i]*y[i]  #E(X) = x*p(x)

y_pos = np.arange(len(x))

plt.bar(y_pos,y,align='center',alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('Probability')
plt.xlabel('X')
plt.title('Probability mass function Q1(i)')
plt.show()
print('There is sudden change in the graph at X = 38 ,which is the expected age of the infected person')
print("Expected age is ", exp)
print("variance is ", var)
print('Yes variance is very high as compared to the expected age because at age 38 there is a steep rise in the probability')

#Question 1(b)
print("\nQ1(ii)")
s1=("Covid19IndiaData_30032020.xlsx")
wb = xlrd.open_workbook(s1) 
sheet = wb.sheet_by_index(0)

rec=0
dead=0
agerec=[]
agedead=[]
sum1=0  #sum for recovered
sum2=0  #sum for dead
sheet.cell_value(0,0)
for i in range(sheet.nrows):
    if sheet.cell_value(i,7)=="Recovered":
        rec+=1
        agerec.append(int(sheet.cell_value(i,2)))
        sum1+=int(sheet.cell_value(i,2))
    elif sheet.cell_value(i,7)=="Dead":
        dead+=1
        agedead.append(int(sheet.cell_value(i,2)))
        sum2+=int(sheet.cell_value(i,2))
prec=rec/sheet.nrows    #mean for recovered  
pdead=dead/sheet.nrows  #mean for dead
print('We have supposed the person who recovered as Y = 1 and the person who is dead as Y = 0 ')
agerec1=list(set(agerec))
agedead1=list(set(agedead))
agerec1.sort()
agedead1.sort()
pagerec=[]  #for probability of recovered
pagedead=[] #for probability of dead

exprec=0
expdead=0
varrec=0
vardead=0
for i in range(len(agerec1)):
    count=0
    for j in range(len(agerec)):
        if agerec1[i]==agerec[j]:
            count+=1
    pagerec.append(count/rec)   #calculating p(X=i) for recovered 
    exprec+=agerec1[i]*count/rec #E(X) = x*p(x) for recovered
for i in range(len(agerec1)):
    varrec+=((agerec1[i]-exprec)**2)*pagerec[i] #var=E(X-E(X))^2 for recovered
    
y_pos = np.arange(len(pagerec))
#PMF for recovered patients
plt.bar(y_pos,pagerec,align='center',alpha=0.5)
plt.xticks(y_pos, agerec1)
plt.ylabel('Probability')
plt.xlabel('value of X')
plt.title('Probability mass function for Recovered patients Q1(ii)')
plt.show()
print("expectation of recovered person is ",exprec)
print("variance of recovered person is ",varrec)
for i in range(len(agedead1)):
    count=0
    for j in range(len(agedead)):
        if agedead1[i]==agedead[j]:
            count+=1
    pagedead.append(count/dead) #calculating p(X=i) for dead
    expdead+=agedead1[i]*count/dead #E(X) = x*p(x) for dead
for i in range(len(agedead1)):
    vardead+=((agedead1[i]-expdead)**2)*pagedead[i]   #var=E(X-E(X))^2 for dead

y_pos1 = np.arange(len(pagedead))
#PMF for dead patients
plt.bar(y_pos1,pagedead,align='center',alpha=0.5)
plt.xticks(y_pos1, agedead1)
plt.ylabel('Probability')
plt.xlabel('value of X')
plt.title('Probability mass function for Dead patients Q1(ii)')
plt.show()
print("expectation of dead person is ",expdead)
print("variance of dead person is ",vardead)
print("Expectations are different from case(i). We can say that old age people are at more risk of dying due to COVID-19")

#Question 1(c)
print("\nQ1(iii)")
s1=("Covid19IndiaData_30032020.xlsx")
wb = xlrd.open_workbook(s1) 
sheet = wb.sheet_by_index(0)
sheet.cell_value(0,0)
print('We have assumed Y = 0 for Gender Code = 0 and Y = 1 for Gender Code = 1')
a0=0    #people with gender 0
a1=0    #people with gender 1
a0age=[]#ages with gender 0
a1age=[]#ages with gender 1
sum0=0  #sum of ages for gender 0
sum1=0  #sum of ages for gender1
for i in range(1,sheet.nrows):
    if sheet.cell_value(i,3)==0:
        a0+=1
        a0age.append(int(sheet.cell_value(i,2)))
        sum0+=int(sheet.cell_value(i,2))
    else:
        a1+=1
        a1age.append(int(sheet.cell_value(i,2)))
        sum1+=int(sheet.cell_value(i,2))
a0age.sort()
a1age.sort()
x0=list(set(a0age))
x1=list(set(a1age))
p0=[]
p1=[]
ex0=0
ex1=0
for i in range(len(x0)):
    count=0
    for j in range(len(a0age)):
        if x0[i]==a0age[j]:
            count+=1
    p0.append(count/sheet.nrows)    #calculating p(X=i) for gender 0
    #print("P(X = ",x0[i]," | P(Y = 0) = ",count/sheet.nrows)
    ex0+=x0[i]*count/sheet.nrows #E(X) for gender 0

y_pos = np.arange(len(x0))
#PMF for gender 0
plt.bar(y_pos,p0,align='center',alpha=0.5)
plt.xticks(y_pos, x0)
plt.ylabel('Probability')
plt.xlabel('value of X')
plt.title('Conditional Probability mass function for gender 0')
plt.show()
print("expectation for Gender Code = 0 is ",ex0)

for i in range(len(x1)):
    count=0
    for j in range(len(a1age)):
        if x1[i]==a1age[j]:
            count+=1
    p1.append(count/sheet.nrows)    #calculating p(X=i) for gender 1
    #print("P(X = ",x1[i]," | P(Y = 0) = ",count/sheet.nrows)
    ex1+=x1[i]*count/sheet.nrows    #E(X) for gender 1

y_pos = np.arange(len(x1))
#PMF for gender 1
plt.bar(y_pos,p1,align='center',alpha=0.5)
plt.xticks(y_pos, x1)
plt.ylabel('Probability')
plt.xlabel('value of X')
plt.title('Conditional Probability mass function for gender 1')
plt.show()            
print("expectation of Gender Code = 1 is ",ex1)
print("As it can be seen the pmf are very different.Females have very high expectation as compared to males. Although the expected infection age remains same.")

#Ques2
df=pd.read_excel('linton_supp_tableS1_S2_8Feb2020.xlsx',sheet_name=0,skiprows=1)
data=df[df['Onset'].notnull()]
data=data[(data['ExposureL'].notnull()) | (data['ExposureType']=="Lives-works-studies in Wuhan")]
data['ExposureL']=data['ExposureL'].fillna(pd.to_datetime('01122019',format='%d%m%Y'))  #cleaning the data

def plotfind(data):
    l=list((data['Onset'] - data['ExposureL']).dt.days)
    X=[i for i in range(min(l),max(l))]
    PX=[l.count(i)/len(l) for i in range(min(l),max(l))]    #calculating P(X=i)
    plt.stem(X,PX,use_line_collection=True)
    plt.xlabel('No. of Days')
    plt.ylabel('Probability')
    plt.title('PMF OF INCUBATION PERIOD')
    plt.show()
    mean=0
    variance=0
    for i in range(len(X)-1):
        mean+=(X[i]*PX[i])
        variance+=((X[i]**2)*PX[i])
    print("Mean incubation period : ", mean)
    print("Variance : ",variance)

print("\nQ2(i)")
print("PMF OF INCUBATION PERIOD")
plotfind(data)
print("\nQ2(ii)")
print("PMF OF INCUBATION PERIOD EXCLUDING WUHAN RESIDENTS")
newdata=data[data['ExposureType']!="Lives-works-studies in Wuhan"]
plotfind(newdata)
print('On seeing the above graphs and incubation period one can say that for Wuhan residents the incubation period was very long and they did not show symptoms early while those who are from outside of Wuhan showed symptoms very early after infection. ')

#Q2(iii)
print("\nQ2(iii)")
df2=pd.read_excel('linton_supp_tableS1_S2_8Feb2020.xlsx',sheet_name=1,skiprows=1)
data2=df2[df2['Onset'].notnull() & df2['Hospitalization/Isolation'].notnull() & df2['Death'].notnull()]
listHO=list((data2['Hospitalization/Isolation'] - data2['Onset']).dt.days)
listXO=list((data2['Death'] - data2['Onset']).dt.days)
listXH=list((data2['Death'] - data2['Hospitalization/Isolation']).dt.days)

def plot(l):
    X=[i for i in range(1,31)]
    PX=[l.count(i)/len(l) for i in range(1,31)]
    plt.stem(X,PX,use_line_collection=True)
    plt.xlabel('No. of Days')
    plt.ylabel('Probability')
    plt.show()

print("PMF OF ONSET TO HOSPITALIZATION PERIOD")
plot(listHO)
print("PMF OF ONSET TO DEATH PERIOD")
plot(listXO)
print("PMF OF HOSPITALIZATION TO DEATH PERIOD")
plot(listXH)
print("PMF OF ONSET TO HOSPITALIZATION FOR SURVIVING PATIENTS")
data=df[df['Onset'].notnull() & df['DateHospitalizedIsolated'].notnull()]
l2=list((data['DateHospitalizedIsolated'] - data['Onset']).dt.days)
plot(l2)
print('''From the graph it is evident that those who survived were hospitalised early between 0-5 days of infection while those who are dead were hospitalised in 5-10 days after infection''')