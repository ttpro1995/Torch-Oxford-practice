require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  -- ...something here...
  local x = input
  local z = (x:gt(0)):double():cmul(x):cmul(x)
  self.output = z
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  -- ...something here..
  -- dloss/dx = gradInput
  -- dloss/dz = gradOutput
  -- dloss/dx = dz/dx * dloss/dz
  -- dz/dx = 2x
  local x = input
  local d = (x:gt(0):double())*2
  --local d = x:mul(2)
  self.gradInput = d:cmul(gradOutput)
  return self.gradInput
end

