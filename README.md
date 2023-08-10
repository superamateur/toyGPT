# toyGPT
Implementation of the tutorial video from  Andrej Karpathy

## what is does
Train a language model base on the transformer architecture.

## Step by step
- Download a text dataset, for example: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- build a text encoder to convert each entity (character, word, sentence) to a number or tensor
- encode the whole text data set, split them into train and test data
- split the training set into chunks of smaller text with size = block_size