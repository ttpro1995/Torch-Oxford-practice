require 'nn'
require 'nngraph'
-- meow
-- z = x1 + x2*linear(x3)
-- let x = vector of dimension 3


x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Identity()()

l = nn.Linear(3,3)({x3})
mul = nn.CMulTable()({x2,l})
add = nn.CAddTable()({x1,mul})
m = nn.gModule({x1,x2,x3},{add})
