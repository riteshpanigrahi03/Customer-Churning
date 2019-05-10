import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import pickle

l=[]
filename='pic.pkl'
with open(filename,'rb') as file:
    pickle_model=pickle.load(file)
#d=[[1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,2,35.50]]
x=int(input("enter gender\n 0-female\n1-male\n"))
l.append(x)

x=int(input("enter senior citzen 0-for yes 1-No\n"))
l.append(x)

x=int(input("Do the customer have any partner? 0-No 1-Yes\n"))
l.append(x)

x=int(input("Do the customer have dependents?? 0-No 1-Yes\n"))
l.append(x)

x=int(input("From how many months the customer is associated with us?\n"))
l.append(x)

x=int(input("Do the customer own a phone service? 0-No 1-Yes\n"))
l.append(x)

x=int(input("Do the customer own a Multiple Line Service service?\n0-No\n1-Yes\n"))
l.append(x)

x=int(input("What kind of Internet service do the customer own?\n 0-No internet Service \n 1-DSL \n2-Fibre Optic\n"))
l.append(x)

if(l[len(l)-1]==0):
    l.append(0)
    l.append(0)
    l.append(0)
    l.append(0)
    l.append(0)
    l.append(0)
else:

    x=int(input("Do the customer is enrolled for Online Security?\n0-No\n1-Yes\n"))
    l.append(x)

    x=int(input("Do the customer have online backup?\n 0-No\n1-Yes\n"))
    l.append(x)


    x=int(input("Device Protection?\n0-No\n1-yes\n"))
    l.append(x)

    x=int(input("Do the customer opted for TechSupport?\n0-No\n1-yes\n"))
    l.append(x)

    x=int(input("Do the person have subscribed for streaming Tv?\n0-No\n1-yes\n"))
    l.append(x)

    x=int(input("Do the person have subscribed for streaming Movies? \n0-No\n1-yes\n"))
    l.append(x)

x=int(input("What is the contract?\n0-Month-to-Month\n1-One year\n2-tw0 year"))
l.append(x)

x=int(input("Do the customer request for Paperless Bill?\n0-No\n1-Yes"))
l.append(x)

x=int(input("What kind of payment method does the customer follow?\n0-electronic check\n1-Mailed check\n3-bank transfer\n4-credit card\n"))
l.append(x)

x=float(input("Enter monthly charges of the customer\n"))
l.append(x)
print("\n\n\n")

d1=[]
d1.append(l)
result=pickle_model.predict(d1)
print(result)
if result==[1]:
    print("Yes,the customer will churn..")
else:
    print("No,the customer won't churn..")
