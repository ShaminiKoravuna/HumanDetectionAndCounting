import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.patches import Rectangle
import PIL.Image as Image
import shutil

# Modify the path where the dataset is located
dataset_folder = "./examples"
group = "train"

dataset_numpy_path = os.path.join(os.getcwd(), f'{dataset_folder}/numpy/{group}')
dataset_xml_path = os.path.join(os.getcwd(), f'{dataset_folder}/xml/{group}')

# Ensure the destination folders exist
def ensure_dir(dir_path):
    if not os.path.exists(dir_path)):
        os.makedirs(dir_path)

# Function to parse XML file and count bounding boxes
def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bbox_coordinates = []
    for member in root.findall('object'):
        class_name = member.find('name').text
        if class_name == 'person':  # Consider only 'person' class
            xmin = int(member.find('bndbox').find('xmin').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            bbox_coordinates.append((xmin, ymin, xmax, ymax))
    return bbox_coordinates

# Function to move files based on the number of bounding boxes
def segregate_and_move_files(xml_folder, numpy_folder, output_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            bboxes = parse_voc_xml(xml_path)
            count = len(bboxes)
            frame_name = xml_file.replace('.xml', '')
            person_folder = os.path.join(output_folder, f'person_{count}')
            ensure_dir(person_folder)
            
            # Move XML file
            shutil.move(xml_path, os.path.join(person_folder, xml_file))
            
            # Move corresponding numpy file
            numpy_file = f"{frame_name}.npy"
            numpy_path = os.path.join(numpy_folder, numpy_file)
            if os.path.exists(numpy_path):
                shutil.move(numpy_path, os.path.join(person_folder, numpy_file))

# Paths to the dataset folders
output_folder = './sorted_frames'

# Segregate and move files
segregate_and_move_files(dataset_xml_path, dataset_numpy_path, output_folder)

# Visualization function (optional)
def visualize_sample(xml_path, numpy_path):
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(root.find("size")[0].text)
    height = int(root.find("size")[1].text)
    channels = int(root.find("size")[2].text)
    bbox_coordinates = []
    for member in root.findall('object'):
        class_name = member.find('name').text
        xmin = int(member.find('bndbox').find('xmin').text)
        ymin = int(member.find('bndbox').find('ymin').text)
        xmax = int(member.find('bndbox').find('xmax').text)
        ymax = int(member.find('bndbox').find('ymax').text)
        bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

    # Create the Surface of Active Events (SAE)
    events = np.load(numpy_path)
    df_events = pd.DataFrame({'timestamp': events[:,0], 'x': events[:,1], 'y': events[:,2], 'polarity': events[:,3]})
    timestamps_vector = df_events['timestamp'].to_numpy()
    df_events_neg = df_events[df_events['polarity'] == 0]
    df_events_pos = df_events[df_events['polarity'] == 1]
    df_events_neg_remaining = df_events_neg.sort_values(by='timestamp').drop_duplicates(subset=['x', 'y'], keep='last', inplace=False)
    df_events_pos_remaining = df_events_pos.sort_values(by='timestamp').drop_duplicates(subset=['x', 'y'], keep='last', inplace=False)
    sae = np.zeros((width, height, 2), dtype='float32')
    time_limit = int(timestamps_vector[-1])
    time_interval = 40e3  # Define time interval
    t_init_0 = int(timestamps_vector[-1] - time_interval)
    df_events_neg_remaining_subset = df_events_neg_remaining[df_events_neg_remaining['timestamp'].isin(range(t_init_0, time_limit))]
    df_events_pos_remaining_subset = df_events_pos_remaining[df_events_pos_remaining['timestamp'].isin(range(t_init_0, time_limit))]
    x_neg = df_events_neg_remaining_subset['x'].to_numpy()
    y_neg = df_events_neg_remaining_subset['y'].to_numpy()
    t_neg = df_events_neg_remaining_subset['timestamp'].to_numpy()
    sae[x_neg, y_neg, 1] = (255 * ((t_neg - t_init_0) / time_interval)).astype(int)
    x_pos = df_events_pos_remaining_subset['x'].to_numpy()
    y_pos = df_events_pos_remaining_subset['y'].to_numpy()
    t_pos = df_events_pos_remaining_subset['timestamp'].to_numpy()
    sae[x_pos, y_pos, 0] = (255 * ((t_pos - t_init_0) / time_interval)).astype(int)
    im = Image.fromarray(0.5 * sae[:,:,0].T + 0.5 * sae[:,:,1].T).convert("L")
    
    # Plot SAE and bounding boxes
    sae_plot = plt.imshow(im, cmap="gray")
    ax = plt.gca()
    for bbox in bbox_coordinates:
        xmin, ymin, xmax, ymax = bbox[1], bbox[2], bbox[3], bbox[4]
        width_lbl = xmax - xmin
        height_lbl = ymax - ymin
        rect = Rectangle((xmin, ymin), width_lbl, height_lbl, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# Example of how to use the visualization function
sample_xml = os.path.join(output_folder, 'person_1', 'frame0002015.xml')
sample_numpy = os.path.join(output_folder, 'person_1', 'frame0002015.npy')
if os.path.exists(sample_xml) and os.path.exists(sample_numpy):
    visualize_sample(sample_xml, sample_numpy)
