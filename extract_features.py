from torch.autograd import Variable
from PIL import Image
from natsort import natsorted
import torch
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def load_frame(frame_file):
    data = Image.open(frame_file)
    data = data.resize((340, 256), Image.ANTIALIAS)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert(data.max() <= 1.0)
    assert(data.min() >= -1.0)
    return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(
                os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
    return batch_data


def oversample_data(data):
    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5,
            data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]


def run(i3d, frequency, frames_dir, batch_size, sample_mode):
    assert(sample_mode in ['oversample', 'center_crop'])
    print("calling batchsize", batch_size)
    segment_size = 16

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224
        with torch.no_grad():
            b_data = Variable(b_data.cuda()).float()
            inp = {'frames': b_data}
            print("The shape of input batch data",b_data.shape)
            features = i3d(inp)
            print("shape of batch segments features",features.shape)
        return features.cpu().numpy()

    rgb_files = natsorted([i for i in os.listdir(frames_dir)])
    frame_cnt = len(rgb_files)
    
    print("number of frames in directory",frame_cnt)
    # Cut frames
    assert(frame_cnt > segment_size)
    max_starting_index_of_last_segment = frame_cnt - segment_size
    processable_frame_indexes = (max_starting_index_of_last_segment // frequency) * \
        frequency  # The start of last segment
    frame_indices = []  # Frames to segments
    for i in range(processable_frame_indexes // frequency + 1):
        frame_indices.append(
            [j for j in range(i * frequency, i * frequency + segment_size)])
    frame_indices = np.array(frame_indices)
    print(f"Frame indices: {frame_indices}")
    segment_num = frame_indices.shape[0]
    print(f"The number of segments are {segment_num}")
    print(f"Total time we have in original segments {(segment_num*segment_size)/25}")
    batch_num = int(np.ceil(segment_num / batch_size))   # segments to batches
    print(f"Total time we have in batched segments {(batch_num*segment_size)/25}")
    print(f"number of batches out of {segment_num} are : {batch_num}")
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)

    if sample_mode == 'over_sample':
        full_features = [[] for i in range(10)]
    else:
        full_features = [[]]

    for batch_id in range(batch_num):
        #Get scaled rgb pixel values of frames of a batch 
        batch_data = load_rgb_batch(
            frames_dir, rgb_files, frame_indices[batch_id])
        print(f"shape of batch {batch_id} is : {batch_data.shape}")
        if(sample_mode == 'oversample'):
            batch_data_ten_crop = oversample_data(batch_data)
            for i in range(10):
                assert(batch_data_ten_crop[i].shape[-2] == 224)
                assert(batch_data_ten_crop[i].shape[-3] == 224)
                print("Calling the shape of each cropped batch",batch_data_ten_crop[i].shape)
                temp = forward_batch(batch_data_ten_crop[i])
                print(f"calling the output shape of each cropped part data batch : {temp.shape}")
                full_features[i].append(temp)

        elif(sample_mode == 'center_crop'):
            batch_data = batch_data[:, :, 16:240, 58:282, :]
            assert(batch_data.shape[-2] == 224)
            assert(batch_data.shape[-3] == 224)
            temp = forward_batch(batch_data)
            full_features[0].append(temp)

    full_features = [np.concatenate(i, axis=0) for i in full_features]
    print(f"Calling the full features shape {len(full_features)} for each  batch with  batch size : {batch_num}")
    full_features = [np.expand_dims(i, axis=0) for i in full_features]
    print("Shape after expanding dimension",len(full_features))
    full_features = np.concatenate(full_features, axis=0)
    print("Calling features shape after concat:",len(full_features))
    full_features = full_features[:, :, :, 0, 0, 0]
    full_features = np.array(full_features).transpose([1, 0, 2])
    print("Calling features shape after transposing",full_features.shape)
    return full_features
