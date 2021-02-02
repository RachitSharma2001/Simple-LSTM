'''
In this code, I have created an simple LSTM from scratch. It reads from
a text file ABC_data.txt, which shows it a sequence of four words, "ABCD" followed
by "BCDA" following by "CDAB" followed by "DABC". It then trains to predict,
given a word, what is the word that comes after it. As shown by the output, it
predicts successfully what words come after each of the strings. 
'''

import numpy as np

# Calculate sigmoid
def sigmoid(z_arr):
    for i in range(len(z_arr)):
        z_arr[i] = 1.0/(1.0+np.exp(-z_arr[i]))
        
    return z_arr

# Calculate tanh
def tanh(z_arr):
    for i in range(len(z_arr)):
        z_arr[i] = (np.exp(z_arr[i]) - np.exp(-z_arr[i])) / (np.exp(z_arr[i]) + np.exp(-z_arr[i]))
        
    return z_arr

# define important variables
input_size = 4
lamda = 0.1
num_epochs = 1500
num_w_weights = 4
num_u_weights = 4
char_array = ['A', 'B', 'C', 'D']

letter_to_index = {
    "A":0,
    "B":1,
    "C":2,
    "D":3,
    "E":4,
    "F":5,
    "G":6,
    "H":7,
    "I":8,
    "J":9,
    "K":10,
    "L":11,
    "M":12,
    "N":13,
    "O":14,
    "P":15,
    "Q":16,
    "R":17,
    "S":18,
    "T":19,
    "U":20,
    "V":21,
    "W":22,
    "X":23,
    "Y":24,
    "Z":25,
    }

# Initialize weights
w_f = [[(np.random.random_sample() * 2) - 1 for _ in range(num_w_weights)] for _ in range(num_w_weights)]
w_c = [[(np.random.random_sample() * 2) - 1 for _ in range(num_w_weights)] for _ in range(num_w_weights)]
w_i = [[(np.random.random_sample() * 2) - 1 for _ in range(num_w_weights)] for _ in range(num_w_weights)]
w_o = [[(np.random.random_sample() * 2) - 1 for _ in range(num_w_weights)] for _ in range(num_w_weights)]

u_f = [[(np.random.random_sample() * 2) - 1 for _ in range(num_u_weights)] for _ in range(num_u_weights)]
u_c = [[(np.random.random_sample() * 2) - 1 for _ in range(num_u_weights)] for _ in range(num_u_weights)]
u_i = [[(np.random.random_sample() * 2) - 1 for _ in range(num_u_weights)] for _ in range(num_u_weights)]
u_o = [[(np.random.random_sample() * 2) - 1 for _ in range(num_u_weights)] for _ in range(num_u_weights)]

test_arr_A = []
test_arr_B = []
test_arr_C = []
test_arr_D = []

# --------------------------- The Algorithm ---------------------------

for epoch in range(num_epochs+1):
    user_file = open("ABC_data.txt", "r")
    
    for line in user_file:
        input_x = [[0.0 for _ in range(input_size)] for _ in range(len(line))]
        for ind in range(len(line)-1):
            input_x[ind][letter_to_index[str(line[ind])]] = 1
            
        previous_H = [np.array([0.0 for _ in range(input_size)]) for _ in range(len(line))]
        cell_state = [np.array([0.0 for _ in range(input_size)]) for _ in range(len(line))]
        forget_mat = [np.array([0.0 for _ in range(input_size)]) for _ in range(len(line))]
        candidate_mat = [np.array([0.0 for _ in range(input_size)]) for _ in range(len(line))]
        ignore_mat = [np.array([0.0 for _ in range(input_size)]) for _ in range(len(line))]
        output_mat = [np.array([0.0 for _ in range(input_size)]) for _ in range(len(line))]

        if epoch == num_epochs:
            print 'The predicted string after {} is:'.format(line)
        for ind in range(len(line)-1):
            char = line[ind]
            
            x_t = input_x[ind]
            
            forget_mat[ind] = sigmoid(np.add(np.dot(w_f, previous_H[ind]), np.dot(u_f, x_t)))
            candidate_mat[ind] = tanh(np.add(np.dot(w_c, previous_H[ind]), np.dot(u_c, x_t)))
            ignore_mat[ind] = sigmoid(np.add(np.dot(w_i, previous_H[ind]), np.dot(u_i, x_t)))
            output_mat[ind] = sigmoid(np.add(np.dot(w_o, previous_H[ind]), np.dot(u_o, x_t)))
            
            cell_state[ind+1] = np.add(np.multiply(cell_state[ind], forget_mat[ind]), np.multiply(candidate_mat[ind], ignore_mat[ind]))
            previous_H[ind+1] = np.multiply(tanh(cell_state[ind+1]), output_mat[ind])

            if epoch == num_epochs:
                if char == 'A':
                    test_arr_A.append(forget_mat[ind])
                elif char == 'B':
                    test_arr_B.append(forget_mat[ind])
                elif char == 'C':
                    test_arr_C.append(forget_mat[ind])
                else:
                    test_arr_D.append(forget_mat[ind])
            
            if epoch == num_epochs:
                largest = previous_H[ind+1][0]
                largest_ind = 0
                for j in range(4):
                    if largest < previous_H[ind+1][j]:
                        largest = previous_H[ind+1][j]
                        largest_ind = j
                
                print char_array[largest_ind]

        if epoch == num_epochs:
            print

        d_out_t = [0.0 for _ in range(input_size)]
        d_c_t = [0.0 for _ in range(input_size)]
        d_cbar_t = [0.0 for _ in range(input_size)]
        d_i_t = [0.0 for _ in range(input_size)]
        d_f_t = [0.0 for _ in range(input_size)]
        d_o_t = [0.0 for _ in range(input_size)]
        d_x_t = [0.0 for _ in range(input_size)]
        change_out_t = [0.0 for _ in range(input_size)]
        change_gates = [[[0.0 for _ in range(input_size)] for _ in range(4)] for _ in range(len(line))]

        for t_fake in range(len(line)):
            t = len(line) - t_fake - 2
            if t < 0:
                continue
            char = line[0]
            if t_fake != 0:
                char = line[t+1]
            difference = [previous_H[t+1][i] for i in range(input_size)]
            difference[letter_to_index[str(char)]] -= 1
           
            d_out_t = np.add(difference, change_out_t)
            d_c_t = np.multiply(np.multiply(d_out_t, output_mat[t]), np.multiply(np.add(np.subtract(np.array([1.0 for _ in range(input_size)]), (tanh(candidate_mat[t])*tanh(candidate_mat[t]))), d_c_t), forget_mat[t+1]))
            d_cbar_t = np.multiply(np.multiply(d_c_t, ignore_mat[t]), np.subtract(np.array([1.0 for _ in range(input_size)]), candidate_mat[t]*candidate_mat[t]))
            d_i_t = np.multiply(np.multiply(d_c_t, candidate_mat[t]), np.multiply(ignore_mat[t], np.subtract(np.array([1.0 for _ in range(input_size)]), ignore_mat[t])))

            if t == 0:
                d_f_t = np.array([0.0 for _ in range(len(line)-1)])
            else:
                d_f_t = np.multiply(np.multiply(d_c_t, cell_state[t]), np.multiply(forget_mat[t], np.subtract(np.array([1.0 for _ in range(input_size)]), forget_mat[t])))

            d_o_t = np.multiply(np.multiply(d_out_t, tanh(cell_state[t+1])), np.multiply(output_mat[t], np.subtract(np.array([1.0 for _ in range(input_size)]), output_mat[t])))

            W = [np.array([0.0 for _ in range(input_size)]) for _ in range(4)]
            W[0] = w_c
            W[1] = w_i
            W[2] = w_f
            W[3] = w_o

            change_gates[t][0] = d_cbar_t
            change_gates[t][1] = d_i_t
            change_gates[t][2] = d_f_t
            change_gates[t][3] = d_o_t

            for i in range(input_size):
                change_out_t[i] = np.dot(np.transpose(W), change_gates[t])[i][0][0]

        change_U = [[[0.0 for _ in range(input_size)] for _ in range(input_size)] for _ in range(4)]
        change_W = [[[0.0 for _ in range(input_size)] for _ in range(input_size)] for _ in range(4)]

        for t in range(len(line)-1):
            change_U[0] = np.add(change_U[0], np.outer(np.transpose(change_gates[t][0]), input_x[t]))
            change_U[1] = np.add(change_U[1], np.outer(np.transpose(change_gates[t][1]), input_x[t]))
            change_U[2] = np.add(change_U[2], np.outer(np.transpose(change_gates[t][2]), input_x[t]))
            change_U[3] = np.add(change_U[3], np.outer(np.transpose(change_gates[t][3]), input_x[t]))

        for t in range(len(line)-2):
            change_W[0] = np.add(change_W[0], np.outer(np.transpose(change_gates[t+1][0]), output_mat[t]))
            change_W[1] = np.add(change_W[1], np.outer(np.transpose(change_gates[t+1][1]), output_mat[t]))
            change_W[2] = np.add(change_W[2], np.outer(np.transpose(change_gates[t+1][2]), output_mat[t]))
            change_W[3] = np.add(change_W[3], np.outer(np.transpose(change_gates[t+1][3]), output_mat[t]))
        
        for i in range(num_u_weights):
            for j in range(num_u_weights):
                u_c[i][j] -= lamda * change_U[0][i][j]
                u_i[i][j] -= lamda * change_U[1][i][j]
                u_f[i][j] -= lamda * change_U[2][i][j]
                u_o[i][j] -= lamda * change_U[3][i][j]
                
        for i in range(num_w_weights):
            for j in range(num_w_weights):
                w_c[i][j] -= lamda * change_W[0][i][j]
                w_i[i][j] -= lamda * change_W[1][i][j]
                w_f[i][j] -= lamda * change_W[2][i][j]
                w_o[i][j] -= lamda * change_W[3][i][j]
