require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local CharLMMinibatchLoader = require 'data.CharLMMinibatchLoader'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
require 'Embedding'                     -- class name is Embedding (not namespaced)
local model_utils=require 'model_utils'

opt = {}
opt.seed = 123
opt.datafile = 'train.t7'
opt.vocabfile = 'vocab.t7'
opt.batch_size =16
opt.seq_length =16 


-- preparation stuff:
torch.manualSeed(opt.seed)

loader = CharLMMinibatchLoader.create(
         opt.datafile, opt.vocabfile, opt.batch_size, opt.seq_length)
vocab_size = loader.vocab_size  -- the number of distinct characters
