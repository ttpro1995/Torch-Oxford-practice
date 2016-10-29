require 'torch'
data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}

cdata = data:clone()
cdata:narrow(2,1,1):fill(1)
y = data:narrow(2,1,1)

w_hat = torch.inverse(cdata:t()*cdata)*cdata:t()*y 

dataTest = torch.Tensor{
{6, 4},
{10, 5},
{14, 8},
{24,20}
}


one = torch.Tensor(dataTest:size(1),1):fill(1)

dataTest = one:cat(dataTest)

y_hat = dataTest*w_hat

print('weight')
print (w_hat)
print('prediction on dataTest')
print (y_hat)
