__author__ = "Lilli Lewis"
__organization__ = "COSC420, University of Otago"
__email__ = "lewli942@student.otago.ac.nz"

#imports
from load_text import load_prideandprejudice, load_warandpeace
from tokeniser import Tokeniser, plot_tok2vec
from transformer import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import os
import pickle, gzip

#Keep myself updated with what's going on, as I watch the program run
print("Loading text...") 
# Define text as first 1000 words of Pride and Prejudice
#text = load_prideandprejudice(max_words=1000)
#Define text as the entirety of Pride and Prejudice and War and Peace
text = load_prideandprejudice()
# text2 = load_warandpeace()
# text = text + ' ' + text2 #this will cause a bit of inaccuracy since a few words at the end of P&P will be associated with a few words at the beginning of W&P - not enough of an overlap to cause concern though, in my opnion
#Using both texts and a 1000 word vocab took up too much memory and wouldn't run at Owheo labs, so I'm only using P&P
#I tried using both texts and 500 word vocab, and each epoch took around 10 mins, so I quit and reran with just P&P

#control the if statement, if I want to use a saved tokenizer or not
useSavedTokenizer = True #using a saved tokenizer with a vocab_size of 1000 so I don't have to keep redoing it! 
#set vocab size to 1000
vocab_size = 1000

if (useSavedTokenizer):
    print("Using saved tokenizer: ")
    tokeniser = Tokeniser.load('tokenizerPt1.json')
else: 
    print("Creating a new tokenizer:")
    #instantiate a new tokeniser call with a maximum of 1000 tokens
    tokeniser = Tokeniser(vocab_size = vocab_size)
    #train tokeniser on selected text
    tokeniser.train(text, verbose = True)
    #save tokeniser
    tokeniser.save('tokenizerPt1.json')

#Keep myself updated with what's going on, as I watch the program run
print("Encoding the tokens:")
#Tokenize the text, so the words are represented as integers
ids = tokeniser.encode(text, verbose = True)

#Keep myself updated with what's going on, as I watch the program run
print("Preparing the data...")
#Start preparing the data to be used in the model
window_size = 5 #so there will be five words preceeding the target token and five following it
input = [] #empty input data array 
output = [] #empty output data array 
for i in range(len(ids)): 
    if (i-window_size)<0: #if we're at the beginning of ids and there won't be a full window on the 'left' side
        for j in range(window_size): #loop through and add the inputs to the input vector and their outputs on the 'right' side to the output vector
            j+=1
            input.append(ids[i])
            output.append(ids[i+j])
            j-=1
        if i!=0: #if we're not on the first id
            for jj in range(i): #loop through and add inputs and outputs on the 'left' side of the id
                input.append(ids[i])
                output.append(ids[jj])
    elif(i+window_size)>(len(ids)-1): #if we're at the end of ids and there won't be a full window on the 'right' side
        for k in range((len(ids)-1)-i):#loop through and add what we do have from the 'right' side
            k+=1
            input.append(ids[i])
            output.append(ids[i+k])
            k-=1
        if i!=(len(ids)-1):#if we're not on the last id
            for kk in range(window_size):#loop through and add the 'left' side of the window
                kk+=(i-window_size)
                input.append(ids[i])
                output.append(ids[kk])
                kk-=(i-window_size)
        else:#here we would be on the last id
            for klk in range(window_size): #loop through and add the 'left' side of the window 
                klk+=i
                input.append(ids[i])
                output.append(ids[klk-window_size])
                klk-=i
    else: #this is the normal case
        for l in range(window_size): #loop through and add on both sides of the window
            l+=1
            input.append(ids[i])
            output.append(ids[i-l])
            input.append(ids[i])
            output.append(ids[i+l])
            l-=1


#Keep myself updated with what's going on, as I watch the program run
print("Switching to one-hot...")
#switch input data to one-hot encoding
input = tf.one_hot(indices=input, depth=vocab_size) 

#convert output array into a numpy format
output_array = np.array(output)

#The code below regarding using a saved model is from example 3
useSavedModel = True

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
   os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'task1') 
net_save_name = save_name + '_model.h5'
history_save_name = save_name + '_model.hist'

if useSavedModel and os.path.isfile(net_save_name):

   # ***************************************************
   # * Loading previously trained neural network model *
   # ***************************************************

   # Load the model from file
   print("Loading network from %s..." % net_save_name)
   model = tf.keras.models.load_model(net_save_name)

   # Load the training history - since it should have been created right after
   # saving the model
   if os.path.isfile(history_save_name):
      with gzip.open(history_save_name) as f:
         history = pickle.load(f)
   else:
      history = []
else:
    #Keep myself updated with what's going on, as I watch the program run
    print("Building model...")
    # building the model
    model = Sequential() #sequential model
    model.add(Dense(768, activation= "softmax", use_bias = False)) #hidden dense layer with 768 neurons and 0 bias
    model.add(Dense(vocab_size, activation = "softmax"))#output layer with same number of neurons as vocab size

    #Keep myself updated with what's going on, as I watch the program run
    print("Compiling model...")
    #compile model
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    #Keep myself updated with what's going on, as I watch the program run
    print("Training model...")
    #train model
    train_info = model.fit(input, to_categorical(output_array, num_classes=vocab_size), epochs=10, batch_size=32)

    #also borrowed from example 3
    # Save the model to file
    print("Saving network to %s..." % net_save_name)
    model.save(net_save_name)

    # Save training history to file
    history = train_info.history
    with gzip.open(history_save_name, 'w') as f:
      pickle.dump(history, f)



#Access weights (I took this line from Stack Overflow)
first_layer_weights = model.layers[0].get_weights()[0]
#this line is to save the weights matrix, it's from chat GPT
print("Saving embeddings...")
np.save('embeddingPt1.npy', first_layer_weights)

#Keep myself updated with what's going on, as I watch the program run
print("Plotting!")
#Plot 
plot_tok2vec(first_layer_weights, tokeniser.word_index)


# All of these have a window of 2
# both texts & 1000 vocab takes 17:58 to build tokenizer and is too big to run
# P&P & 1000 vocab takes 2 mins to build tokenizer but like 3 mins per epoch 
# both texts & 500 vocab 9:44 to build tokenizer and like 10 mins per epoch
# P&P & 500 vocab takes 1:38 to build and like 1:15 per epoch 

# P&P with 1000 vocab, 10 epochs, window of 5 takes 2:48 to build tokenizer and ~10 mins per epoch 
# it looks the best and it is what is saved right now!!!!!

#Run and then switched usedSavedTokenizer to True and AFTER pressing run on this! THEN DON'T EDIT THIS FILE!




#after this is done running, check to make sure a model was saved, then run it again switching the useSavedModel flag to true and make sure it runs
#then go try to see if task2 works