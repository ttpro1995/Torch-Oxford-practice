require 'torch'
require 'gnuplot'

t1 = torch.Tensor{1,4,3,2,1,5,9,7}
t2 = torch.Tensor{1,2,3,4,5,6,7,8}
n_value = t1:size(1)
r = torch.range(1,n_value)

gnuplot.plot({r,t1},
{r,t2})



