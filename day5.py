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


#Ask the user to enter a positive integer n
n = int(input("Enter a positive integer n: "))

# for loop
print("\nUsing for loop:")
for i in range(1, n + 1):
    print(i)

#while loop
sum_numbers = 0
i = 1
while i <= n:
    sum_numbers += i
    i += 1

print("\nSum of numbers from 1 to", n, "is:", sum_numbers)


# In[ ]:


def calculate_square(n):
    return n * n
n = int(input("Enter a positive integer: "))
result = calculate_square(n)
print("\nThe square of", n, "is:", result)


# In[ ]:




