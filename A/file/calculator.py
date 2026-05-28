def c(i,op):
    return 1.2**(-i)* op ** (1.08)
print(c(1,2))
ans = c(1,2)
n = 1
# print(c(10,1))
for i in range(3,10,2):
    ans += c(i,1)
    n +=1
print(ans,n,ans/n)
print(c(3,2))
print((c(1,2)+c(3,1))/2)