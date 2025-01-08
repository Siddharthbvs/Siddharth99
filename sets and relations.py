#!/usr/bin/env python
# coding: utf-8

# In[1]:


s1 = {1,3,4,0,6,4,}
print(s1)
print(type(s1))


# In[2]:


lst1 = [1,8,9,52,0,6,9,8]
s2 = set(lst1)
s2


# In[5]:


s1 = {1,2,3,4,5}
s2 = {3,4,5,6,7}
s1 | s2


# In[6]:


s1 & s2


# In[7]:


s1.intersection(s2)


# In[8]:


s1 - s2


# In[9]:


s2 - s1


# In[17]:


#symmetric difference
s1={1,2,3,4,5,6}
s2={5,6,7,8,9,0}
s1.symmetric_difference(s2)


# In[14]:


s1={1,2,3,4,5,6}
s2={5,6,}
s1.issubset(s2)


# In[11]:


s2.issubset(s1)


# In[15]:


s1.issuperset(s2)


# In[16]:


s2.issuperset(s1)


# In[19]:


#strings
str1 = "Welcome aiml class"
str2 = 'we started with python'
str3 = '''This is alpha'''
print(type(str1))
print(type(str2))
print(type(str3))


# In[24]:


#slicing in strings
print(str1)
str1[5:10]


# In[31]:


print(str1)
str1.split()


# In[32]:


##join functoin
str1.join(str2)


# In[35]:


str4 = "Hello"
'  '.join(str4)


# In[37]:


#strip
str5 = "     Hello alpha?"
str5.strip()


# In[1]:


num = 6
if num %2 == 0:
    print("even")
else:
    print("odd")


# In[2]:


print("even")if num %2 == 0 else print ("odd")


# In[4]:


x = 10
result = "positive" if x > 0 else "Negative"
print(result)


# In[6]:


age = 18
category = "Adult" if age>=18 else "minor"
print(category)


# In[10]:


num = int(input("Enter a number: "))
result = "positive" if num > 0 else ("Negative" if num < 0 else "zero")
print(result)


# In[11]:


#print even numbers
l =[1,3,7,5,8,2]
[x for x in l if x%2 ==0]


# In[12]:


#print odd numbers
l =[1,3,7,5,8,2]
[x for x in l if x%2 !=0]


# In[18]:


#print updated salaries
#condition is given 20% hike for
sal = [40000,60000,70000,45000]
[(x*1.2  if  x<= 50000 else x) for x in sal]


# In[19]:


#dictionaey comprehension
d1 = {'Ram':[70,45,69,100], 'john':[56,34,76,88,99]}
d1


# In[22]:


{k:sum(v)/len(v) for k,v in d1.items()}


# In[28]:


def mean_value(given_list):
    total = sum(given_list)
    average_value = total/len(given_list)
    return average_value


# In[29]:


L = [1,2,3,4,5,6,7,8,9,10]
mean_value(L)


# In[49]:


#gretting
def greet(name):
    print(f"good morning {name}!")


# In[50]:


greet("Soumya")


# In[46]:


#variables numbers of arguments
def avg_value(*n):
    l = len(n)
    average = sum(n)/l
    return average


# In[48]:


avg_value(10,50,60,900,800)

