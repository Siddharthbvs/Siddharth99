#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#lambda function
greet = lambda name : print(f"Good Morning {name}!")
greet("Soumya")


# In[ ]:


#product with lambda function
product = lambda a,b,c : a*b*c
product(4,5,6)


# In[ ]:


#lambda function with list comprehension
even = lambda L : [x for x in L if x%2 == 0]
my_list = [100,51,9,58,5,6,4]
even(my_list)

