from torch.autograd import Variable
from PIL import Image
from natsort import natsorted
import torch
import numpy as np
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_extraction.log", mode='a')
    ]
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def load_frame(frame_file):
    try:
        data = Image.open(frame_file)
        data = data.resize((340, 256), Image.ANTIALIAS)
        data = np.array(data)
        data = data.astype(float)
        data = (data * 2 / 255) - 1
        assert data.max() <= 1.0
        assert data.min() >= -1.0
        return data
    except Exception as e:
        logging.error(f"Error loading frame {frame_file}: {e}")
        raise


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    logging.info("Loading RGB batch...")
    try:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
        for i in range(frame_indices.shape[0]):
            for j in range(frame_indices.shape[1]):
                batch_data[i, j, :, :, :] = load_frame(
                    os.path.join(frames_dir, rgb_files[frame_indices[i][j]])
                )
        logging.info("RGB batch loaded successfully.")
        return batch_data
    except Exception as e:
        logging.error(f"Error loading RGB batch: {e}")
        raise


def oversample_data(data):
    logging.info("Performing oversampling...")
    try:
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

        logging.info("Oversampling completed.")
        return [data_1, data_2, data_3, data_4, data_5,
                data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]
    except Exception as e:
        logging.error(f"Error in oversampling: {e}")
        raise


def run(i3d, classify, frames_dir, batch_size,frequency,segment_size, sample_mode, class_names=None, video_id="video"):
    logging.info(f"Starting feature extraction/classification for video: {video_id}")
    assert sample_mode in ['oversample', 'center_crop'], "Invalid sample mode."

    def forward_batch(b_data):
        try:
            logging.info(f"Running forward pass for batch with shape: {b_data.shape}")
            b_data = b_data.transpose([0, 4, 1, 2, 3])
            b_data = torch.from_numpy(b_data).cuda().float()
            with torch.no_grad():
                b_data = Variable(b_data)
                inp = {'frames': b_data}
                outputs = i3d(inp)
            logging.info("Forward pass completed.")
            return outputs
        except Exception as e:
            logging.error(f"Error during forward pass: {e}")
            raise

    rgb_files = natsorted([i for i in os.listdir(frames_dir)])
    frame_cnt = len(rgb_files)
    assert frame_cnt > segment_size, "Not enough frames for processing."
    logging.info(f"Found {frame_cnt} frames in {frames_dir}.")

    max_start_idx = frame_cnt - segment_size
    processable_frame_indexes = (max_start_idx // frequency) * frequency
    frame_indices = [
        [j for j in range(i * frequency, i * frequency + segment_size)]
        for i in range(processable_frame_indexes // frequency + 1)
    ]
    frame_indices = np.array(frame_indices)
    logging.info(f"Generated frame indices for {len(frame_indices)} segments.")

    batch_num = int(np.ceil(len(frame_indices) / batch_size))
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)
    logging.info(f"Split into {batch_num} batches with batch size {batch_size}.")

    results = []
    for batch_id in range(batch_num):
        logging.info(f"Processing batch {batch_id + 1}/{batch_num}")
        batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])
        if sample_mode == 'oversample':
            batch_data_ten_crop = oversample_data(batch_data)
            batch_results = []
            for crop in batch_data_ten_crop:
                outputs = forward_batch(crop[:, :, 16:240, 58:282, :])
                batch_results.append(outputs.cpu().numpy())
            results.append(np.stack(batch_results, axis=1))
        else:
            batch_data = batch_data[:, :, 16:240, 58:282, :]
            outputs = forward_batch(batch_data)
            results.append(outputs.cpu().numpy())
    results = np.concatenate(results, axis=0)
    print("calling the shape of results",results.shape)

    if classify:
        logging.info("Performing classification...")
        results = torch.tensor(results)
        results = torch.nn.functional.softmax(results, dim=1).cpu().numpy()
        print("Classified scores after softmax are",results)
        sorted_scores = np.argsort(results, axis=1)[:, ::-1]  # Top scores
        top_k = min(5, results.shape[1])

        classification_results = []
        for i, scores in enumerate(sorted_scores[:, :top_k]):
            segment_results = []
            for rank in range(top_k):
                # logging.info(f"scores: {scores}, type: {type(scores)}")
                # logging.info(f"rank: {rank}, type: {type(rank)}")
                logging.info(f"scores[rank]: {scores[rank]}, type: {type(scores[rank])}")
                
                # Extract the scalar index safely
                if isinstance(scores[rank], (np.ndarray, torch.Tensor)):
                    scalar_index = scores[rank].item() if scores[rank].size == 1 else scores[rank].flatten()[0]
                else:
                    scalar_index = scores[rank]
                
                # Ensure the index is within range
                if scalar_index < len(class_names):
                    label = class_names[scalar_index]
                    score = results[i, scalar_index]
                    segment_results.append({'label': label, 'score': float(score)})
                else:
                    logging.error(f"Index {scalar_index} is out of range for class_names with length {len(class_names)}.")
                    raise Exception  # Skip invalid indices
            classification_results.append(segment_results)


        save_path = f"{video_id}_classification.json"
        with open(save_path, "w") as f:
            json.dump(classification_results, f, indent=4)
        logging.info(f"Classification results saved to {save_path}")
        return classification_results
    else:
        logging.info("Performing feature extraction...")
        results = results[:, :, 0, 0, 0]  # Remove spatial dimensions
        save_path = f"{video_id}_features.npy"
        np.save(save_path, results)
        logging.info(f"Features saved to {save_path}, shape: {results.shape}")
        return results
