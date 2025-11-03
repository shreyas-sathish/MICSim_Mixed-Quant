from Performance.src.CNN.Network_mixed import Network
import time

average_file_path = './average_files/VGG16/'

# Format: [k_h, k_w, C_in, C_out, H_in, W_in, s_h, s_w, pad, avg_file, shift_file, name, type, wbits, abits]
layer_list = layer_list = [
            # Block 1: Input 224x224
            [3, 3,   3,  64, 224, 224, 1, 1, 1, average_file_path + 'conv1_1.csv', average_file_path + 'conv1_1_shift.csv', 'conv1_1', 'Conv', 8, 8],
            [3, 3,  64,  64, 224, 224, 1, 1, 1, average_file_path + 'conv1_2.csv', average_file_path + 'conv1_2_shift.csv', 'conv1_2', 'Conv', 8, 8],
            # MaxPool: 224 -> 112
            
            # Block 2: Input 112x112
            [3, 3,  64, 128, 112, 112, 1, 1, 1, average_file_path + 'conv2_1.csv', average_file_path + 'conv2_1_shift.csv', 'conv2_1', 'Conv', 8, 8],
            [3, 3, 128, 128, 112, 112, 1, 1, 1, average_file_path + 'conv2_2.csv', average_file_path + 'conv2_2_shift.csv', 'conv2_2', 'Conv', 8, 8],
            # MaxPool: 112 -> 56
            
            # Block 3: Input 56x56
            [3, 3, 128, 256,  56,  56, 1, 1, 1, average_file_path + 'conv3_1.csv', average_file_path + 'conv3_1_shift.csv', 'conv3_1', 'Conv', 4, 8],
            [3, 3, 256, 256,  56,  56, 1, 1, 1, average_file_path + 'conv3_2.csv', average_file_path + 'conv3_2_shift.csv', 'conv3_2', 'Conv', 4, 8],
            [3, 3, 256, 256,  56,  56, 1, 1, 1, average_file_path + 'conv3_3.csv', average_file_path + 'conv3_3_shift.csv', 'conv3_3', 'Conv', 4, 8],
            # MaxPool: 56 -> 28
            
            # Block 4: Input 28x28
            [3, 3, 256, 512,  28,  28, 1, 1, 1, average_file_path + 'conv4_1.csv', average_file_path + 'conv4_1_shift.csv', 'conv4_1', 'Conv', 4, 8],
            [3, 3, 512, 512,  28,  28, 1, 1, 1, average_file_path + 'conv4_2.csv', average_file_path + 'conv4_2_shift.csv', 'conv4_2', 'Conv', 4, 8],
            [3, 3, 512, 512,  28,  28, 1, 1, 1, average_file_path + 'conv4_3.csv', average_file_path + 'conv4_3_shift.csv', 'conv4_3', 'Conv', 4, 8],
            # MaxPool: 28 -> 14
            
            # Block 5: Input 14x14
            [3, 3, 512, 512,  14,  14, 1, 1, 1, average_file_path + 'conv5_1.csv', average_file_path + 'conv5_1_shift.csv', 'conv5_1', 'Conv', 4, 8],
            [3, 3, 512, 512,  14,  14, 1, 1, 1, average_file_path + 'conv5_2.csv', average_file_path + 'conv5_2_shift.csv', 'conv5_2', 'Conv', 4, 8],
            [3, 3, 512, 512,  14,  14, 1, 1, 1, average_file_path + 'conv5_3.csv', average_file_path + 'conv5_3_shift.csv', 'conv5_3', 'Conv', 4, 8],
            # MaxPool: 14 -> 7
            
            # AdaptiveAvgPool: 7x7 -> 1x1
            
            # Classifier: Input 1x1
            [1, 1, 512, 512,   1,   1, 1, 1, 0, average_file_path + 'fc1.csv', average_file_path + 'fc1_shift.csv', 'fc1', 'FC', 8, 8],
            [1, 1, 512, 512,   1,   1, 1, 1, 0, average_file_path + 'fc2.csv', average_file_path + 'fc2_shift.csv', 'fc2', 'FC', 8, 8],
            [1, 1, 512,  10,   1,   1, 1, 1, 0, average_file_path + 'fc3_output.csv', average_file_path + 'fc3_output_shift.csv', 'fc3_output', 'FC', 8, 8],
        ]


def main_cnn():
    time_start = time.time()
    VGG16 = Network(layer_list)
    VGG16.Map()
    VGG16.Configure()
    VGG16.CalculateArea()
    VGG16.CalculatePerformance()
    time_end = time.time()
    time_sum = time_end - time_start
    print("-------------------- Simulation Performance --------------------")
    print("Total Run-time of NeuroSim: {:.2f} seconds".format(time_sum))
    print("-------------------- Simulation Performance --------------------")

main_cnn()