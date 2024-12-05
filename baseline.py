# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
from superpoint import SuperPoint
from build_adj_matrix import query
import torch
import pygame
import cv2
import joblib
import numpy as np
import json
import os
import time
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted
import networkx as nx
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for reading exploration data
        self.save_dir = "./images_subsample/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")

        self.goal_id = None
        self.current_id = 14
        self.offset = 14 # remove idle frame at start of exploration data
        self.graph = None
        self.node_size = 3 
        self.num_nodes = (len(os.listdir(self.save_dir)) - self.offset) // self.node_size + 2 
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.database_name = joblib.load('database_name.pkl')
        self.tree = joblib.load('ball_tree.pkl')
        self.codebook = joblib.load('codebook.pkl')
        self.actions_file = "data/image_actions.json"
        self.path = None
        self.path_segments = None
        self.frames = None
        self.actions = None
        self.config = {}
        self.model = SuperPoint(self.config).to(self.device)
        self.count = 0
        self.tmp = 0
        self.time_buffer = 0 #a buffer stores the time the last function is excuted 
        self.debounce_delay = 0.25 #0.25s delay for debouncing a pressed button
        #self.adj_matrix_path = "graph_adj_matrix.npy"

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                # else:
                # #     If a key is pressed that is not mapped to an action, then display target images
                # #     self.show_target_images()
                #      self.display_next_best_view()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def superpoint(self, image):
        image = torch.as_tensor(image).float() #Convert np array to torch tensor
        image = image.unsqueeze(0).unsqueeze(0).to(self.device) #Resize the input data for [H,W] to [1,1,H,W]
        self.model.eval()
        #_, inp = read_image(img, device)
        pred = self.model({'image': image}) #run model inference
        des = pred['descriptors'][0] #split the descriptor from the output
        des = torch.transpose(des, 0, 1)
        des = des.cpu().detach().numpy()
        des = des.astype(np.float64)
        return des

    # Computer VLAD for every query images
    def get_VLAD(self, X):

        predictedLabels = self.codebook.predict(X)
        centroids = self.codebook.cluster_centers_
        labels = self.codebook.labels_
        k = self.codebook.n_clusters
    
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

    def query(self, img, img_id=None):

        #id_list = []
        #image = cv2.imread(os.path.join(data_path,f"{img_id}.jpg"),cv2.IMREAD_GRAYSCALE) / 255
        descriptor = self.superpoint(img)
        VLAD_query = self.get_VLAD(descriptor).reshape(1, -1)
        dist, index = self.tree.query(VLAD_query, 1)
        #print(index,img_id) # this is self.tree index not the image index
        # index is an array of array of 1
        for j in range(len(index[0])):
            id_name = self.database_name[index[0][j]]
            print(id_name)
            id_name = int(id_name.split('.')[0])
            #id_list.append(id_name)
        return id_name

    def query_actions(self, frames):
        # Load the actions JSON file
        with open(self.actions_file, 'r') as file:
            actions_data = json.load(file)

        actions = []
        for index in range(len(frames)):
            frame = frames[index]
            frame_next = frames[index + 1]
            if frame < frame_next:
                for i in range(5): # downsample dataset -> original dataset
                    action = actions_data.get(str(frame * 5 + i))
                    actions.append(action)


        return actions

    def find_path_segments(self, path):
        # Load the actions JSON file
        with open(self.actions_file, 'r') as file:
            actions_data = json.load(file)
        
        if not path:
            return []

        segments = []
        current_segment = [path[0]]

        for i in range(1, len(path)):
            if abs(path[i] - path[i - 1]) == 1:  # Check if the current number is consecutive
                current_segment.append(path[i])
            else:
                segments.append(current_segment)  # Add the completed segment
                current_segment = [path[i]]  # Start a new segment

        segments.append(current_segment)  # Add the last segment

        # Compute frames and corresponding controls for each segment
        frames = []
        actions = []
        for segment in segments:
            start_node = segment[0]
            end_node = segment[-1]
            # Generate frames in ascending or descending order based on segment direction
            if start_node <= end_node:
                start_frame = start_node * self.node_size + 1
                end_frame = (end_node + 1) * self.node_size
                frame_sequence = list(range(start_frame, end_frame + 1))
                frames.append(frame_sequence)
                index_sequence = list(range(start_frame * 5, end_frame * 5 + 1))
                action_sequence = []
                for i in index_sequence:
                    action_sequence.append(actions_data[str(i)].get("action"))
                actions.append(action_sequence)
            else:
                start_frame = (start_node + 1) * self.node_size 
                end_frame = end_node * self.node_size + 1
                frame_sequence = list(range(start_frame, end_frame - 1, -1))
                frames.append(frame_sequence)
                index_sequence = list(range(start_frame * 5, end_frame * 5 - 1, -1))
                action_sequence = []
                for i in index_sequence:
                    action = actions_data[str(i)].get("action")
                    if action == "Action.FORWARD":
                        reverse_action = "Action.BACKWARD"
                    elif action == "Action.BACKWARD":
                        reverse_action = "Action.FORWARD"
                    elif action == "Action.LEFT":
                        reverse_action = "Action.RIGHT"
                    elif action == "Action.RIGHT":
                        reverse_action = "Action.LEFT"
                    else:
                        reverse_action = "Action.IDLE"
                    action_sequence.append(reverse_action)
                actions.append(action_sequence)

        return segments, frames, actions
    
    def compute_shortest_path(self):
        """
        Compute the shortest path from the current frame to the target frame using the adjacency matrix.

        Parameters:
        adj_matrix_path (str): Path to the adjacency matrix saved as a .npy file.
        current_frame (int): The starting frame (node).
        target_frame (int): The destination frame (node).

        Returns:
        list: List of nodes representing the shortest path, or an empty list if no path exists.
        """
        # Load the adjacency matrix
        adj_matrix = joblib.load('adj_matrix.pkl')

        # Convert the adjacency matrix to a NetworkX graph
        
        graph = nx.from_numpy_array(adj_matrix)

        # Determine the nodes for the current and target frames
        current_node = (self.current_id - self.offset) // self.node_size 
        target_node = (self.goal_id - self.offset) // self.node_size 

        # Check if both nodes are in the graph
        if current_node not in graph or target_node not in graph:
            print(f"One of the nodes ({current_node} or {target_node}) is not in the graph.")
            return []

        # Compute the shortest path
        try:
            shortest_path = nx.shortest_path(graph, source=current_node, target=target_node)
            print(f"The shortest path from {current_node} to {target_node} is: {shortest_path}")
            return shortest_path
        except nx.NetworkXNoPath:
            print(f"No path exists between {current_node} and {target_node}.")
            return []  
    
    def display_multiple_images(self, window_name="Combined Images"):
        """Displays multiple images from the database in a single window."""
        #self.count += 1
        images = []
        img_names = []
        num_group = (len(self.path) // 5 + 1)
        #print(num_group)
        if self.count != self.tmp:
            if self.count < num_group-1:
                #print(num_group - self.count)
                for index in range(5):
                    path = self.save_dir + str(self.path[self.count*5+index]*3+self.offset) + ".jpg"
                    if os.path.exists(path):
                        img = cv2.imread(path)
                        images.append(img)
                        img_names.append(path.split('/')[-1].split('.')[0])
                    else:
                        print(f"Image with ID {index} does not exist")
                            # Combine images vertically or horizontally (adjust as needed)
                #print(len(images))
                combined_image = np.concatenate(images, axis=1)  
                if self.count != self.tmp:
                    print("image displayed: ", img_names)
                    self.tmp = self.count
                self.time_buffer = time.time()
                cv2.imshow(window_name, combined_image)
                cv2.waitKey(1)
            else:
                self.count -= 1

        # problem: the program either run action contrl or run display next best view(dnbv), can not switch back to action control once dnbv has been run. 

    def display_imgs_from_id(self, id, window_name):
        """
        Display images from database based on its ID using OpenCV
        """
        targets = []
        for i in id:
            # path = self.save_dir + str(i) + ".jpg"
            path = self.save_dir + str(i) + ".png"
            if os.path.exists(path):
                img = cv2.imread(path)
                targets.append(img)
            else:
                print(f"Image with ID {i} does not exist")
                blank_img = np.zeros_like(targets[0]) if targets else np.zeros((100, 100, 3), dtype=np.uint8)
                targets.append(blank_img)

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, '1st View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, '2nd View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, '3rd View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, '4th View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(window_name, concat_img)
        cv2.waitKey(1)

    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """

        # TODO: could you write this function in a smarter way to not simply display the image that closely 
        # matches the current FPV but the image that can efficiently help you reach the target?

        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        #id_list = self.query(self.fpv)
        #self.current_id = self.query(self.fpv)
        # Display the image 5 frames ahead of the neighbor, so that next best view is not exactly same as current FPV
        # self.display_img_from_id(index+3, f'Next Best View')
        # Display the images along several frames ahead of the neighbor, so that next best view is not exactly same as current FPV
        #for i in range(len(self.path) // 3):  # Assuming each group of 3 elements holds ID data
            #img_id = self.path[i] * 3 + 14 
        self.display_multiple_images()
        # Display the next best view id along with the goal id to understand how close/far we are from the goal
        #print(f'Next View ID: {[index[0]+10, index[1]+10, index[2]+10, index[3]+10]} || Goal ID: {self.goal}')
        #self.display_img_from_id(path[1], f'Possible Shortcut')

    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # Nothing to do here since exploration data has been provided
                pass
            
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
    
                if self.goal_id is None:
                    # Get the neighbor nearest to the front view of the target image and set it as goal
                    targets = self.get_target_images()
                    targets[0] = cv2.cvtColor(targets[0], cv2.COLOR_BGR2GRAY) / 255
                    self.goal_id = self.query(targets[0])
                    print(f'Goal ID: {self.goal_id}') # goal_id = 22336
                    # Compute at the begining only to save time
                    self.path = self.compute_shortest_path()
                    self.path_segments, self.frames, self.actions = self.find_path_segments(self.path)
                    #print(self.path)
                                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                #print(keys)
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q] and (time.time() - self.time_buffer) > self.debounce_delay:
                    self.count += 1
                    self.display_next_best_view()                    
                    print("Dispalying Next 5 Images")
                if keys[pygame.K_e] and (time.time() - self.time_buffer) > self.debounce_delay:
                    self.count -= 1
                    self.display_next_best_view()
                    print("Dispalying Last 5 Images")

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
