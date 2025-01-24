#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Taking input for marks in three subjects
subject1 = float(input("Enter marks for subject 1: "))
subject2 = float(input("Enter marks for subject 2: "))
subject3 = float(input("Enter marks for subject 3: "))

# Calculate the average of the marks
average = (subject1 + subject2 + subject3) / 3

# Determine the grade based on the average
if average >= 90:
    print("Grade: A")
elif 80 <= average < 90:
    print("Grade: B")
elif 70 <= average < 80:
    print("Grade: C")
else:
    print("Grade: Fail")


# In[ ]:




