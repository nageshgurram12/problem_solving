#!/usr/bin/python

array = [1, 2, 1]

maxSpan = 0
for left in range(len(array)):
    span = 1
    right = len(array) - 1
    
    while array[left] != array[right]:
        right -= 1
        if right < 0:
            break
        
    span = right-left+1
    
    if span > maxSpan:
        maxSpan = span

print(maxSpan)