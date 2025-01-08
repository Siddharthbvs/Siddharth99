#!/usr/bin/env python
# coding: utf-8

# In[1]:


def sum_of_evens(n):
    total_sum = 0
    
    for i in range(1, n+1):
        if i % 2 == 0:
            total_sum += i
    
    return total_sum

n = int(input("Enter a positive integer: "))
result = sum_of_evens(n)
print(f"The sum of all even numbers between 1 and {n} is: {result}")


# In[ ]:




