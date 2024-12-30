__author__ = "Lilli Lewis"
__organization__ = "COSC420, University of Otago"
__email__ = "lewli942@student.otago.ac.nz"

# Most of this code is taken from the transformer.py file Lech provided us. I've adjusted the numbers and the network to suit my purposes.
# I've kept his comments just to keep myself familiar with what each line he wrote does. I've altered the network to increase the number
# of transformer layers, and to be able to switch between onehot or my own encoding. I altered the variables used in the layers to use variable
# names instead of direct values so that I can adjust the values in one place instead of several. But, most of this code is Lech's, and deleting
# his comments to clean up the code felt like I was pretending it was my code. 

#Control if I'm using my own embeddings from task 1or one-hot embeddings
useOneHot = False


#imports
from load_text import load_prideandprejudice
from tokeniser import Tokeniser
from transformer import *
import tensorflow as tf
import sys
import os
import pickle, gzip

# Load text for training
text = load_prideandprejudice(max_words=1000)

seq_len = 15     #Length of the input sequence to the transformer
vec_dim = 768    #Dimension of the embedding vectors
key_dim = 64
num_heads = 12
dff = 256

epochs = 3       #Number of epochs to train for

#Load in my tokenizer from part One
tokeniser = Tokeniser.load('tokenizerPt1.json')

# Convert text to token ids
print("Converting training text to tokens...")
ids = tokeniser.encode(text)


##################################################################################################### Lech's data generator, copied with his permission

class predictTextDataGenerator(tf.keras.utils.Sequence):
    
      def __init__(self, ids, seq_len,batch_size):
          '''
          Constructor for the data generator.  

          param ids: A list of integers representing the tokens in the training text.
          param seq_len: The length of the input and target sequences for the transformer model.
          param batch_size: The number of sequences in each batch for training          
          '''

          # Save all the training text and parameters of the data generator
          self.ids = ids          
          self.seq_len = seq_len
          self.batch_size = batch_size

          # Compute the number of samples - it's the length of the text minus the sequence length
          self.num_samples = len(self.ids)-seq_len-1
          # Run the on_epoch_end() method - which scrambles the data into the batchs
          # (this method will also be run during trainin at the end of each training epoch)
          self.on_epoch_end()

      def __len__(self):
          '''
          You must provide this method to tell the model how many batches there are in an epoch.

          returns The number of batches in an epoch.
          '''
          return self.num_samples // self.batch_size

      def __data_generation(self, list_IDs_temp):
          '''
          This method generates a batch of training data for the model.  It's called by the
          __getitem__() method which is called by the model during training.

          param list_IDs_temp: A list of integers representing the indexes of the training data
          to be included in the batch.
          returns A tuple of input and target output sequences for the model.
          '''

          # The input and target sequences are both of shape (batch_size, seq_len) and
          # are integer ids of the tokens (the transformer model will convert these to word vectors based
          # on the embedding you specify)
          X = np.zeros((self.batch_size, self.seq_len),dtype='int')
          y = np.zeros((self.batch_size, self.seq_len),dtype='int')

          # For each index in the list of indexes...
          for i, ID in enumerate(list_IDs_temp):
              #...get the sequence of tokens from the training of length seq_len starting at
              #index ID.  In this case the input sequence is the sequence spans the entire
              #length of seq_len, but you might also train on shorter sequences, padded with zeros.
              #makse_loss will included padded inputs/outputs.
              X[i,:seq_len] = self.ids[ID:ID+seq_len]
              #....and the sequence of target tokens, which is the sequence of tokens from the
              #training text of length seq_len starting at index ID+1 (offset by one, to match
              #the next word in the output to current word in the input)
              y[i,:seq_len] = self.ids[ID+1:ID+seq_len+1]


          return X, y

      def __getitem__(self, index):
          '''
          This method is called by the model during training to get a batch of training data.
          
          param index: The index of the batch to get.
          returns A tuple of input and target output sequences for the model.
          '''
          
          # Generate indexes of the batch
          list_IDs_temp = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]

          # Generate data
          X, y = self.__data_generation(list_IDs_temp)

          return X, y

      def on_epoch_end(self):
          '''
          This method is called at the end of each epoch of training.  It shuffles the data
          so that the batches are different in each epoch.
          '''
          
          # Shuffle the tokens
          self.list_IDs = np.arange(self.num_samples)
          np.random.shuffle(self.list_IDs)


################################################################################################################ End of Lech's data generator

#this was to fix an error 
############## Chat GPT ##############
class SerializableFixedEmbedding(FixedEmbedding):
    def get_config(self):
        base_config = super().get_config()
        return base_config
############## Chat GPT ##############


# Create a data generator
print("Loading data generator...")
train_data = predictTextDataGenerator(ids=ids, seq_len=seq_len, batch_size=32)

# Get the vocabulary sice of the tokeniser
vocab_size = tokeniser.vocab_size

# Fetch the (vocab_size, vec_dim)-shape embedding matrix from my tok2vec model
w = np.load('embeddingPt1.npy')

#decide to run with own embeddings or onehot embeddings
if useOneHot:
    vec_dim = vocab_size
else: 
    vec_dim = w.shape[1]



# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
   os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'task2') 
if useOneHot:
    net_save_name = save_name + '_oneHot_model.h5'
    history_save_name = save_name + '_oneHot_history.hist'
else:
    net_save_name = save_name + '_task1Embeddings_model.h5'
    history_save_name = save_name + '_task1Embeddings_history.hist'

# Create a new sequential model
print("Creating a model...")
model = tf.keras.models.Sequential()

# The first layer of the model is the embedding layer.  The fixed embedding is conveyed
# in the w argument passed in, which is a numpy array of shape (vocab_size, vec_dim).  You
# also need to specify the seq_len of your input sequence.  The input then is a tensorf of
# shape (num_examples, seq_len) of integers representing tokens from the vocabulary; the
# output is a (num_examples, seq_len, vec_dim) tensor of word vectors.
#if statement to determine first layer of model
if useOneHot:
    model.add(OneHotEmbedding(vocab_size, seq_len))
else:
    model.add(SerializableFixedEmbedding(w, seq_len))

# Positional endcoding is added to the embedding. This layer needs to know the vec_dim of
# the embedding space and the seq_len of the input sequence.  The input is a tensor of shape
# (num_examples, seq_len, vec_dim) of word vectors; the output is the same shape with positional 
# encoding added to the word vectors, of shape (num_examples, seq_len, vec_dim).
model.add(PositionalEncoding(vec_dim=vec_dim, seq_len=seq_len))

# The transformer layer is added to the model.  This layer needs to know the vec_dim of the
# word vectors, the key_dim of the key/value/query vectors used in the self-attention mechanism,
# the number of heads in the multi-head attention mechanism, and the dimension of the feed-forward
# network in the transformer layer.  The input is a tensor of shape (num_examples, seq_len, vec_dim)
# of word vectors with positional encoding added to them; the output is of shape (num_examples, seq_len, vec_dim).  You can have sever transformer layers in the model, just like you can have several dense or convolutional layers.
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=key_dim, num_heads=num_heads, dff=dff))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=key_dim, num_heads=num_heads, dff=dff))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=key_dim, num_heads=num_heads, dff=dff))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=key_dim, num_heads=num_heads, dff=dff))


# The final dense layer of the netowork is added.  This layer has a softmax activation function and
# outputs a tensor of shape (num_examples, seq_len, vocab_size) of probabilities of the next token in
# the sequence for each position in the input.  The input is a tensor of shape (num_examples, seq_len, vec_dim); the output is of shape (num_examples, seq_len, vocab_size).
model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

################################################################################################################ Start code directly from tensorflow

# Custom learning rate schedule for the transformer model - taken directly from
# https://www.tensorflow.org/text/tutorials/transformer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        #super().__init__() #Removed this line because it's not an inner class

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    #This function generated by chat GPT to overcome a get_config not implemented error
    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

################################################################################################################ End code directly from tensorflow


# This custom learning schedule (which varies the learning rate) is much better to use than
# a fixed learning rate. 
learning_rate = CustomSchedule(vec_dim, dff)
opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                epsilon=1e-9)    

# Complie the modle.  The masked_loss and maske_accuracy (provided at the top of this file) are
# better than the standard loss and accuracy functions because they ignore the padding tokens in
# your target outputs...so that, if your input and target output don't span the entire seq_len,
# they should be padded with zeros, and the loss and accuracy functions will ignore those zeros.
model.compile(optimizer=opt,
                loss=masked_loss,
                metrics=[masked_accuracy])

# Show the archtiecture of the model
model.summary()

# Train the model
train_info=model.fit(train_data, epochs=epochs)


# Save the model to file
print("Saving neural network to %s..." % net_save_name)
model.save(net_save_name)

# Save training history to file
history = train_info.history
with gzip.open(history_save_name, 'w') as f:
    pickle.dump(history, f)

# Test the model by generating text that follows this prompt
# prompt = "It is a truth universally acknowledged"
prompt = "She loved him but"

print(prompt, end='')
sys.stdout.flush()

# Encode prompt to tokens
tokens = tokeniser.encode(prompt)

for i in range(1,200):
    # Check if prompt is more than seq_len, if so, truncate, grabbing the
    # last seq_len tokens
    if len(tokens) >= seq_len:
        tokens = tokens[-seq_len:]
    # Index of the last token, which is going to be the 
    # index of the output stream that we are going to use for prediction
    j = len(tokens)-1

    # If the prompt is less than seq_len, pad it with zeros
    if len(tokens) < seq_len:
        x = np.concatenate([tokens,np.zeros((seq_len-len(tokens)),dtype='int')], axis=0)
    else:
        x = np.array(tokens)

    # Since the transformer expect input to be of shape (num_examples, seq_len), and
    # at this point x is just a vector of seq_len integers, we need to add a dimension
    # to change x to a tensor of shape (1, seq_len)     
    x = np.expand_dims(x,axis=0)

    # Compute output of the transformer
    y = model.predict(x,verbose=False)
    # The output will be of dmension (1, seq_len, vocab_size), but we are only interested in
    # the token that follow the prompt, at position j in the output stream.  
    # And so y[:,j,:] is a (1, vocab_size) tensor of probabilities of the next token in the sequence.
    # and we want to find the token with the highest probability.
    y = np.argmax(y[:,j,:])
    
    # Decode the token back to text
    t = tokeniser.decode(y)
    # Print it
    print(t, end='')
    sys.stdout.flush()
    # Apend the token (integer) to the prompot tokens
    tokens.append(y)

print("\n")


#run this with useSavedModel as false twice, once with useOneHot as true and once as false. then make sure those models were created. then run it twice more, 
# with useSavedModel as true and switching useOneHot. Just make sure it runs all four ways.