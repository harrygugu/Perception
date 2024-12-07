import joblib
import networkx as nx
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.neighbors import BallTree
from superpoint import SuperPoint

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load preprocessed ball tree, codebook,and database
tree = joblib.load('ball_tree.pkl')
codebook = joblib.load('codebook.pkl')
#database_VLAD = joblib.load('database_VLAD.pkl')
database_name = joblib.load('database_name.pkl')
data_path = './images_subsample'
offset = 14 # There are 14 images remain the same at the start of exploration data

# Load superpoint model
def superpoint(image, model):
    image = torch.as_tensor(image).float() #Convert np array to torch tensor
    image = image.unsqueeze(0).unsqueeze(0).to(device) #Resize the input data for [H,W] to [1,1,H,W]
    model.eval()
    #_, inp = read_image(img, device)
    pred = model({'image': image}) #run model inference
    des = pred['descriptors'][0] #split the descriptor from the output
    des = torch.transpose(des, 0, 1)
    des = des.cpu().detach().numpy()
    des = des.astype(np.float64)
    return des
config = {}
model = SuperPoint(config).to(device)

def query(img, model, img_id = None):


    # Computer VLAD for every query images
    def get_VLAD(X, codebook):

        predictedLabels = codebook.predict(X)
        centroids = codebook.cluster_centers_
        labels = codebook.labels_
        k = codebook.n_clusters
    
        m,d = X.shape
        VLAD_feature = np.zeros([k,d])
        #computing the differences

        # for all the clusters (visual words)
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels == i) > 0:
                # add the diferences
                VLAD_feature[i] = np.sum(X[predictedLabels==i,:] - centroids[i],axis=0)
        

        VLAD_feature = VLAD_feature.flatten()
        # power normalization, also called square-rooting normalization
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))

        # L2 normalization
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)
        return VLAD_feature

    value_list = []
    #image = cv2.imread(os.path.join(data_path,f"{img_id}.jpg"),cv2.IMREAD_GRAYSCALE) / 255
    descriptor = superpoint(img,model)
    VLAD_query = get_VLAD(descriptor, codebook).reshape(1, -1)
    dist, index = tree.query(VLAD_query, 4)
    #print(index,img_id) # this is tree index not the image index
    # index is an array of array of 1
    for j in range(len(index[0])):
        value_name = database_name[index[0][j]]
        print(value_name)
        value_name = int(value_name.split('.')[0])
        value_list.append(value_name)
    return value_list

def create_and_save_graph(node_size=3, save_path=None):
    """
    Create and save a connected graph of the data directory images.
    """
    num_nodes = len(os.listdir(data_path)) // node_size  # Number of nodes
    #num_nodes = round(num_nodes+0.5)
    #print(num_nodes)

    # Initialize adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Initialize the graph
    #print("Drawing graph...")
    #graph = nx.Graph()

    for i in range(num_nodes):
        # Connect consecutive nodes
        if i < num_nodes - 1:
            adj_matrix[i][i + 1] = 1
            adj_matrix[i + 1][i] = 1

        # Select one image from each node (e.g., the first image)
        img_id = i * node_size + offset
        img_path = os.path.join(data_path, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            print(f"Image with ID {img_id} does not exist")
            continue
        
        # Load the image
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) / 255
        if img is None:
            print(f"Failed to load image with ID {img_id}")
            continue
        
        # Find the nearest neighbors within the current node and its neighbors
        neighbors = query(img, model)
        for neighbor in neighbors:
            # Calculate the node index of the neighbor
            neighbor_node = (neighbor - offset) // node_size # To be optimized
            #print(neighbor_node,neighbor)
            if neighbor > img_id + 2 or neighbor < img_id - 2:
                # Mark the adjacency in both directions
                adj_matrix[i][neighbor_node] = 1
                #adj_matrix[neighbor_node][i] = 1
        

    # # Save adjacency matrix as .npy file
    # adj_matrix_path = f"{save_path}_adj_matrix.npy"
    # np.save(adj_matrix_path, adj_matrix)
    # print(f"Adjacency matrix saved to {adj_matrix_path}")
    
    # # Plot the graph
    # plt.figure(figsize=(10, 8))
    # pos = nx.spring_layout(graph)  # Layout for visualization
    # nx.draw(graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10)
    # plt.title("Connected Graph of Data Directory Images")

    # # Save the plot as .png
    # graph_image_path = f"{save_path}.png"
    # plt.savefig(graph_image_path)
    # print(f"Graph saved to {graph_image_path}")
    # validation for symmetry adj_matrix
    #breakpoint()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if ((adj_matrix[i][j]) == 1 and  (adj_matrix[j][i]) == 0) or ((adj_matrix[i][j]) == 0 and  (adj_matrix[j][i]) == 1) == 1:
                adj_matrix[i][j] = 0
                adj_matrix[j][i] = 0
            

    print(len(os.listdir(data_path)))
    print(num_nodes)
    return adj_matrix

if __name__ == "__main__":
    # Compute
    adj_matrix = create_and_save_graph(node_size=3)
    joblib.dump(adj_matrix, 'adj_matrix.pkl')
    #adj_matrix = joblib.load('adj_matrix.pkl')
    print(adj_matrix[229,1204])

