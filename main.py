from __future__ import print_function
import argparse
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import time
import math  # Add this line to import the math module
from spiking_model_LIF import*
from N_cars_dataset import*
import pynvml # put this on package initialization




#init value for python script

parser=argparse.ArgumentParser()
parser.add_argument('--filenet', type=str, dest='filename_net')
parser.add_argument('--fileresult', type=str, default='result.txt', dest='filename_result')
parser.add_argument('--sample_time', type=float, default=1, dest='sample_time')
parser.add_argument('--sample_length', type=float, default=10, dest='sample_length')
parser.add_argument('--batch_size', type=int, default=40, dest='batch_size')
parser.add_argument('--lr', type=float, default=1e-3, dest='lr')
parser.add_argument('--lr_decay_epoch', type=int, default=20, dest='lr_decay')
parser.add_argument('--lr_decay_value', type=float, default=0.5, dest='lr_decay_value')
parser.add_argument('--lr_policy', type=int, default=0, dest='lr_policy') 
#policy 0: decreasing_step policy (default); 
#policy 1: warm_restart policy; 
#policy 2: linear decreasing warm_restart policy; 
#policy 3: triangular-based cyclical policy; 
#policy 4: decreasing triangular-based cyclical policy
#policy 5: triangular-based one-cycle policy
parser.add_argument('--threshold', type=float, default=0.4, dest='thresh')
parser.add_argument('--n_decay', type=float, default=0.2, dest='n_decay') #decay constant
parser.add_argument('--att_window', type=int, nargs=4, dest='att_window')
parser.add_argument('--weight_decay', type=float, default=0, dest='weight_decay') #L2regularizzation


args = parser.parse_args()
pynvml.nvmlInit()
handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)


# initialize spiking model and network
initialize_model(args.filename_net, args.thresh, args.n_decay, 2, args.batch_size, args.lr, kernel_init_f=[args.att_window[0], args.att_window[1]])


batch_size= args.batch_size


data_path_train =  './'  #todo: input your data path for train dataset if not write in train files (car_train.txt and background_train.txt)
data_path_test =  './'   #todo: input your data path for test dataset if not write in test files (car_test.txt and background_test.txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

samplingTime = args.sample_time
sampleLength = args.sample_length
filename_result = args.filename_result

# instantiate the train dataset and use the DataLoader function to give samples to the network 
trainingSet = IBMGestureDataset(datasetPath=data_path_train, 
									sampleFile_car  ='./N_cars/car_train.txt',
									sampleFile_background  ='./N_cars/background_train.txt',
									samplingTime=samplingTime,
									sampleLength=sampleLength,
									shift_x=args.att_window[2],
 									shift_y=args.att_window[3], 
									att_window=[args.att_window[0],	args.att_window[1]])

train_loader  = DataLoader(dataset=trainingSet, batch_size=batch_size, shuffle=True, num_workers=10)

# instantiate the test dataset and use the DataLoader function to give samples to the network 
testingSet = IBMGestureDataset(datasetPath=data_path_test, 
									sampleFile_car  ='./N_cars/car_test.txt',
									sampleFile_background  ='./N_cars/background_test.txt',
									samplingTime=samplingTime,
									sampleLength=sampleLength,
									shift_x=args.att_window[2],
 									shift_y=args.att_window[3], 
									att_window=[args.att_window[0],	args.att_window[1]])
test_loader = DataLoader(dataset=testingSet, batch_size=batch_size, shuffle=True, num_workers=10)


if (lr_policy==1): # warm_restart
    # ----uncomment this for warm_restart values for the T_max and T_mult will change to get different peaks----
    T_max = 100  # Maximum number of epochs before a reset
    eta_min = 1e-5  # Minimum learning rate
    base_lr = 1e-2  # Maximum learning rate
    T_mult = 1  # Multiplier to expand the cycle length after each restart
    num_epochs = 200 # Total number of epochs
    #
    def calculate_warm_restart_lr(epoch, T_max_initial, eta_min, base_lr, T_mult):
        T_i = T_max_initial
        t_cur = epoch
        for i in range(epoch):
            if t_cur < T_i:
                break
            t_cur -= T_i
            T_i *= T_mult
        lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * t_cur / T_i))
        return lr
    # ----uncomment this for warm_restart policy----

if (lr_policy==2): # linear decreasing warm_restart policy
    # ----uncomment this for linear decreasing warm_restart policy----
    num_epochs = 200
    cycle_length = 40  # Number of epochs in one learning rate cycle
    # Define the learning rate update function based on the one-cycle policy
    def update_decreasingWR_lr(optimizer, epoch, lr_max=1e-2, lr_min=1e-5, total_epochs=num_epochs, cycle_length=cycle_length):
        cycle = epoch // cycle_length
        cycle_epoch = epoch % cycle_length
        lr_trend_up = (lr_max - lr_min) * cycle_epoch / cycle_length
        lr_trend_down = lr_max - (lr_max - lr_min) * cycle_epoch / cycle_length
        new_lr = lr_min + (lr_trend_down if cycle % 2 else lr_trend_up)
        return new_lr
    # ----uncomment this for linear decreasing warm_restart policy----

if (lr_policy==3): # triangular-based cyclical policy
    # ----uncomment this for triangular-based cyclical policy----
    # Initialize parameters
    min_lr = 1e-5  # Min learning rate
    max_lr = 1e-2  # Max learning rate
    cycle_length = 25  # Length of each cycle (in epochs)
    num_cycles = 4  # Number of cycles
    #
    def update_cyclical_lr(epoch, min_lr, max_lr, cycle_length, num_cycles):
        # Calculate the total number of iterations
        total_iterations = num_cycles * cycle_length * 2
        # Calculate the position within the cycle
        cycle_position = epoch % (cycle_length * 2)
        #
        if cycle_position < cycle_length:
            # Increasing phase of the cycle
            lr = min_lr + (max_lr - min_lr) * cycle_position / cycle_length
        else:
            # Decreasing phase of the cycle
            lr = max_lr - (max_lr - min_lr) * (cycle_position - cycle_length) / cycle_length
        return lr
    # ----uncomment this for triangular-based cyclical policy----

if (lr_policy==4): # decreasing triangular-based cyclical policy
    # ----uncomment this for decreasing triangular-based cyclical policy----
    # Initialize parameters
    base_lr = 1e-5  # Min learning rate
    max_lr_start = 1e-2  # Start max learning rate
    stepsize = 50  # Half a cycle length
    num_epochs = 200  # Total number of epochs
    #
    def update_triangular_lr(epoch, base_lr, max_lr_start, stepsize, num_epochs):
        # Calculate the number of stages (cycles)
        num_stages = num_epochs // (2 * stepsize)
        # Determine the current stage
        stage = epoch // stepsize
        # Calculate the position within the current stage
        stage_progress = epoch % stepsize
        # Calculate the maximum LR for the current stage
        max_lr = max_lr_start / (2 ** stage)
        # 
        if stage_progress < stepsize // 2:
            lr = base_lr + (max_lr - base_lr) * stage_progress / (stepsize // 2)
        else:
            lr = max_lr - (max_lr - base_lr) * (stage_progress - stepsize // 2) / (stepsize // 2)
        return lr
    # ----uncomment this for decreasing triangular-based cyclical policy----

if (lr_policy==5): # triangular-based one-cycle policy
    # ----uncomment this for triangular-based one-cycle policy----
    num_epochs = 200
    # Define the learning rate update function with a steep end
    def update_one_cycle_lr(optimizer, epoch, lr_start=1e-5, lr_max=1e-2, lr_end=1e-6, steep_drop_epoch=180):
        if epoch <= steep_drop_epoch:
            # Increase or decrease the learning rate linearly
            lr = (lr_max if epoch <= steep_drop_epoch / 2 else lr_start) - (lr_max - lr_start) * abs(epoch / (steep_drop_epoch / 2) - 1)
        else:
            # Steep decrease to the final learning rate
            lr = lr_start - (lr_start - lr_end) * (epoch - steep_drop_epoch) / (num_epochs - steep_drop_epoch)
        return lr
    # ----uncomment this for triangular-based one-cycle policy----
        
   
# create and open the file to write the principle numerical results runtime 
f=open(filename_result, 'w')

# write the principal initialization information 
f.write('batch size: '+str(args.batch_size)+ ' sampling time: '+str(samplingTime)+ ' sampling length: '+str(sampleLength)+ ' filenet: '+str(args.filename_net)+ ' learning rate: '+str(args.lr)+ ' lr decay epoch: '+str(args.lr_decay)+ ' lr decay value: '+str(args.lr_decay_value)+ ' threashold: '+str(args.thresh)+ ' neuron decay constant: '+str(args.n_decay)+ ' attention window: '+str(args.att_window)+ ' weight decay(L2 reg): '+str(args.weight_decay)+'\n')

# define the network and load saved weights
snn = SCNN()
snn = putWeight(snn)# this part can be used to load the weigh of a previously trained network. 
snn.to(device)

# define criterion and optimizer
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False) #L2r

# ----guide for warm_restart----
# # run the train and test for num_epochs epochs
# for epoch in range(num_epochs):
#    # Simulate the learning rate update for each epoch and print the result
#     lr = calculate_warm_restart_lr(epoch, T_max, eta_min, base_lr, T_mult)

#    # Update optimizer's learning rate
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     best_acc_entire_image_test=0
#     running_loss = 0
#     start_time = time.time()
    
#     len_of_sample= len(trainingSet)
    
#     snn=snn.train()
#     correct_entire_image=0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
#     total_entire_image=0   # number of images to predict
# ----guide for warm_restart----

# ----guide for linear decreasing warm_restart----
# run the train and test for num_epochs epochs
# for epoch in range(num_epochs):
#     lr = update_decreasingWR_lr(optimizer, epoch)  # Update the learning rate

#     best_acc_entire_image_test=0
#     running_loss = 0
#     start_time = time.time()
    
#     len_of_sample= len(trainingSet)
    
#     snn=snn.train()
#     correct_entire_image=0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
#     total_entire_image=0   # number of images to predict
# ----guide for linear decreasing warm_restart----

# ----guide for triangular-based cyclical policy----
# Run the train and test for num_epochs epochs
# for epoch in range(num_epochs):
#     best_acc_entire_image_test = 0
#     running_loss = 0
#     start_time = time.time()
    
#     len_of_sample = len(trainingSet)
    
#     snn = snn.train()
#     correct_entire_image = 0 # number of correct decision after sampleLength/samplingTime predictions then choose the most predicted
#     total_entire_image = 0   # number of images to predict
    
#     # Update the learning rate using the cyclical policy
#     if args.lr_policy:
#         lr = update_cyclical_lr(epoch, min_lr, max_lr, cycle_length, num_cycles)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
# ----guide for triangular-based cyclical policy----

# ----guide for decreasing triangular-based cyclical policy----
# # Run the train and test for num_epochs epochs
# for epoch in range(num_epochs):
#     best_acc_entire_image_test = 0
#     running_loss = 0
#     start_time = time.time()
    
#     len_of_sample = len(trainingSet)
    
#     snn = snn.train()
#     correct_entire_image = 0 # number of correct decision after sampleLength/samplingTime predictions then choose the most predicted
#     total_entire_image = 0   # number of images to predict
    
#     # Update the learning rate using the triangular decreasing policy
#     if args.triangular_policy:
#         lr = update_triangular_lr(epoch, base_lr, max_lr_start, stepsize, num_epochs)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
# ----guide for decreasing triangular-based cyclical policy----

# ----guide for triangular-based one-cycle policy----
# # run the train and test for num_epochs epochs
# for epoch in range(num_epochs):
#     update_one_cycle_lr(optimizer, epoch+1)  # epoch+1 because your epochs seem to start from 1
        
#     best_acc_entire_image_test=0
#     running_loss = 0
#     start_time = time.time()
    
#     len_of_sample= len(trainingSet)
    
#     snn=snn.train()
#     correct_entire_image=0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
#     total_entire_image=0   # number of images to predict
# ----guide for triangular-based one-cycle policy----
    
# run the train and test for num_epochs epochs
for epoch in range(num_epochs):
    #
    if (lr_policy==1): # warm_restart
        lr = calculate_warm_restart_lr(epoch, T_max, eta_min, base_lr, T_mult)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif (lr_policy==2): # linear decrasing warm_restart
        lr = update_decreasingWR_lr(optimizer, epoch) 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr        
    elif (lr_policy==3):# triangular-based cyclical policy
        lr = update_cyclical_lr(epoch, min_lr, max_lr, cycle_length, num_cycles)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif (lr_policy==4):# triangular-based cyclical policy
        lr = update_triangular_lr(epoch, base_lr, max_lr_start, stepsize, num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif (lr_policy==5):# triangular-based one-cycle policy
        lr = update_one_cycle_lr(optimizer, epoch+1)  # epoch+1 because your epochs seem to start from 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        print("Default learning policy: decreasing step")
    #
    best_acc_entire_image_test=0
    running_loss = 0
    start_time = time.time()
    len_of_sample= len(trainingSet)
    snn=snn.train()
    correct_entire_image = 0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
    total_entire_image = 0   # number of images to predict
    #
    for i, (images, labels_,labels) in enumerate(train_loader,0):
        # run only for complete batches
        len_of_sample=len_of_sample-batch_size
        if len_of_sample >= 0:
            snn.zero_grad()
            optimizer.zero_grad()
            images = images.float().to(device)
            first=0
            ### print(snn) ### debug
	        # group outputs of the same image of length sampleLength and accumulate the prediction for every samplingTime
            for j in range (0, int(sampleLength/samplingTime)):
                outputs = snn(images[:,:,:,:,j])
                ### print(snn) ### debug
                if first==0:
                   _,accumulation=outputs.to(device).max(1)
                   first=first+1
                else:
                   _,predicted=outputs.max(1)
                   accumulation+=predicted
                
                loss = criterion(outputs, labels_[:,:,0,0,0].to(device))
                running_loss += loss.item()
                
                loss.backward()
            optimizer.step()
		
	        # see what is the most predicted class for the image
            accumulation[accumulation<(sampleLength/samplingTime)/2]=0
            accumulation[accumulation>=(sampleLength/samplingTime)/2]=1
           
	        # calculate accuracy on the image of length sampleLength
            total_entire_image += float(labels.size(0))
            correct_entire_image += float(accumulation.eq(labels.to(device)).sum().item())
            acc_entire_image_train=100*correct_entire_image/total_entire_image
        running_loss_last = running_loss
        if (i+1)%20 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Accuracy: %.5f'
                   %(epoch+1, num_epochs, i+1, len(trainingSet)//batch_size,running_loss, acc_entire_image_train))
            running_loss = 0
            print('Time elasped:', time.time()-start_time)
        if i== batch_size/2:
           # put this on the line that you want the power to be sensed
           measure0 = pynvml.nvmlDeviceGetPowerUsage(handle0)     
           f.write('power: '+str(measure0)+'\n')   
    end_time = time.time()
    print('Training duration for one epoch:', end_time-start_time)
    f.write('Training  duration for one epoch: '+str(end_time-start_time))

    correct = 0 # number of correct decision for each samplingTime 
    total = 0   # number of total samplingTime predictions 
    correct_entire_image=0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
    total_entire_image=0   # number of images of sampleLength length
    with torch.no_grad():
        snn=snn.eval()
        len_of_sample= len(testingSet)
        for batch_idx, (inputs, labels_, targets) in enumerate(test_loader,0):
            # run only for the complete batch size
            len_of_sample=len_of_sample-batch_size
            if len_of_sample >= 0:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                first=0
	            # group outputs of the same image of length sampleLength and accumulate the prediction for every samplingTime
                for j in range (0, int(sampleLength/samplingTime)):
                    outputs = snn(inputs[:,:,:,:,j])
                    if first==0:
                       _,accumulation=outputs.to(device).max(1)
                       first=first+1
                    else:
                       _,pre=outputs.max(1)
                       accumulation+=pre
                    
                    loss = criterion(outputs, labels_[:,:,0,0,0].to(device))
	                # calculate the prediction at every samplingTime without grouping them in an image of sampleLength length  
                    _, predicted = outputs.max(1)
                    total += float(targets.size(0))
                    correct += float(predicted.eq(targets.to(device)).sum().item())
		
                # see the most predicted class for the image of length sampleLength
                accumulation[accumulation<(sampleLength/samplingTime)/2]=0
                accumulation[accumulation>=(sampleLength/samplingTime)/2]=1
                
	            # calculate accuracy on the image of length sampleLength and at every samplingTime
                total_entire_image += float(targets.size(0))
                correct_entire_image += float(accumulation.eq(targets.to(device)).sum().item())
                acc_entire_image_test=100*correct_entire_image/total_entire_image
                if batch_idx %100 ==0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the sampling time streams: %.3f' % (100 * correct / total))
    print('Test Accuracy of the model on the entire test images: %.3f' % (acc_entire_image_test))
    acc = 100. * float(correct) / float(total)
    
    # every epoch save the results 
    if epoch % 1 == 0:
        print(acc)
        print('Saving results..')

        f.write('acc: '+str(acc)+' loss: '+str(running_loss_last)+' acc_train: '+str(acc_entire_image_train)+' acc_test: '+str(acc_entire_image_test)+' epoch: '+str(epoch)+'\n')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
	
	# save the network and the weights only if the accuracy on entire images is better than before 
        if epoch>=0 and best_acc_entire_image_test < acc_entire_image_test:
           print('Saving weights and network..')
           best_acc_entire_image_test=acc_entire_image_test
           if not os.path.isdir('checkpoint'):
              os.mkdir('checkpoint')
           torch.save(state, './checkpoint/ckpt' + str(args.att_window[0])+'_ceil' + '.t7')
           genLoihiParams(snn)
