# Simple-LSTM
A simple LSTM in Python that trains to predict the next word based on the input file.

An LSTM Neural Network is short for "Long Short Term Memory" network. It is very useful for creating neural networks that need to predict
things like the next words in sentences, where it can't just forget something it processed a long time ago. 

I created an LSTM from scratch using Python, meaning I used no libraries. In this implementation, I have the file ABC_Data.txt, which contains words in a 
short sequence. The LSTM is trained to predict, given one word in the sequence, the next word that should come after it. 

I train the small LSTM for about 1500 epochs, and it eventually is able to predict with high accuracy.
