local function prac1()
  local t = torch.Tensor({{1,2,3},
                          {4,5,6},
                          {7,8,9}});
  print('whole t');
  print(t);
  local col = t:narrow(2,2,1); --extract the middle col
  print('the middle col');
  print(col);
  
end
prac1()
