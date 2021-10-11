import torch, torch.nn as nn

ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,}

class CharDecoderRNN(nn.Module):
    """
    Character level decoder rnn to generate words from embeddings
    """
    def __init__(self,
                 hidden_size = 10,
                 char_count = 28,
                 char_embedding_size = 64,
                 input_embedding_size = 80,
                 embedding_to_hidden_activation = 'relu',
                 num_layers=1,
                 dropout = 0,
                 bidirectional = False):
        """
        hidden_size: hidden size of the rnn
        char_count: number of characters in the encoder dictionary
        char_embedding_size: (also equal to input_size of rnn) embedding size of the IDs input to the decoder (we choose, so could be different from the size we used for the embeddings)
        input_embedding_size: the size of word embeddings (no choice, based on the embedding model). A fully connected
            layer from input_embedding_size -> hidden_size is used to
            create the initial rnn hidden state from a word embedding
        embedding_to_hidden_activation: the activation function applied to the result of 'embedding_to_hidden'
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        """
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # transforms the input word embedding into a hidden state for the rnn model
        self.embedding_to_hidden = nn.Linear(input_embedding_size, hidden_size)

        # activation function after the 'embedding_to_hidden' transformation
        self.activation_function = ACTS[embedding_to_hidden_activation]()
        
        # embedds the lists of IDs
        self.rnn_input_module = nn.Embedding(num_embeddings=char_count,
                                         embedding_dim=char_embedding_size)

        # rnn layer, takes the hidden state and the embedded lists of IDs as input
        self.rnn = nn.GRU(char_embedding_size,
                              hidden_size,
                              num_layers,
                              dropout=dropout,)
                              #bidirectional=bidirectional)

        # transforms the output of the rnn into probabilities
        self.rnn_output_module = nn.Linear(hidden_size, char_count, bias=True)

        # should we apply a sigmoid here?
        self.output_activation_function = nn.Softmax(dim=1)
        
    def forward(self, hidden, *packed_input, use_head = True):
        '''
        hidden: expects the (batch of) word embedding(s)
        packed_input: expects the (batch of) words in the form of lists of IDs
        use_head: indicates whether the hidden state goes through the embedding layer (we use it only at the first step, when the hidden state is the original word embedding)
        '''

        # --- transorms input if necessary ---
        # word embedding -> hidden state if hidden is a word embedding
        if use_head:
            #print(hidden.shape)
            hidden = self.embedding_to_hidden(hidden)
            hidden = self.activation_function(hidden).squeeze(0)
        #else:
        #    hidden = hidden.to(self.device)

        # --- GRU layer ---
        #packed_input = packed_input.to(self.device)
        batch_sizes = packed_input[0].batch_sizes

        # the IDs go through an embedding layer: 'char_count' characters, outputs embeddings of size 'char_embedding_size'
        rnn_input = self.rnn_input_module(*[p.to(self.device).data for p in packed_input])
        # to deal with batches
        rnn_input = nn.utils.rnn.PackedSequence(rnn_input, batch_sizes)
        # GRU layer(s): 'num_layers' layer(s) of size 'hidden_size'
        rnn_output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # apply a linear layer + softmax to get probabilities
        output = self.rnn_output_module(rnn_output.data)
        #output = self.output_activation_function(output)
        # to deal with batches
        decoded = nn.utils.rnn.PackedSequence(output, batch_sizes)
        
        return decoded, hidden
