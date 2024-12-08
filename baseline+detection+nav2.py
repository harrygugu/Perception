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
import math

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

#Make sure change the path in line 59 and line 73 for subsample data and image action json. 


#Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        # self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        
        #Automated control
        self.automatic_mode = False
        self.paused = False
        self.current_segment_index = 0
        self.current_action_index = 0
  
        #Movement and position tracking
        self.action_queue = []
        self.movement_state = Action.IDLE
        self.position = np.array([0.0, 0.0])
        self.heading = 0.0
        self.last_action = Action.IDLE
        self.path = [self.position.copy()]
        self.map_surface = None 
        
        #Movement increments
        self.increments_per_full_rotation = 147
        self.rotation_per_action = 360 / self.increments_per_full_rotation
        self.movement_per_action = 0.1       
        
        super(KeyboardPlayerPyGame, self).__init__()
        
        #Variables for reading exploration data
        self.save_dir = "C:/Users/15463/vis_nav_player/data/images_subsample/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")

        self.goal_id = None
        self.current_id = 49 ####modify
        self.offset = 49 #remove idle frame at start of exploration data
        self.graph = None
        self.node_size = 3 
        self.num_nodes = (len(os.listdir(self.save_dir)) - self.offset) // self.node_size + 2 
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.database_name = joblib.load('database_name.pkl')
        self.tree = joblib.load('ball_tree.pkl')
        self.codebook = joblib.load('codebook.pkl')
        self.actions_file = "C:/Users/15463/Downloads/exploration_data (1)/image_actions.json"
        self.adj_matrix = joblib.load('adj_matrix.pkl')
        self.path = [self.position.copy()]
        self.path_segments = None
        self.frames = None
        self.actions = None
        self.node_path = None
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
        # self.last_act = Action.IDLE
        self.screen = None

        # Reset movement and position
        self.action_queue = []
        self.movement_state = Action.IDLE
        self.position = np.array([0.0, 0.0])
        self.heading = 0.0
        self.last_action = Action.IDLE
        self.path = [self.position.copy()]
        self.map_surface = None
        
        self.paused = False  # Reset the paused flag
        self.current_segment_index = 0
        self.current_action_index = 0
        
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
        
        # Store previous states
        prev_automatic_mode = self.automatic_mode
        prev_paused = self.paused
        
        if self.action_queue:
            action = self.action_queue.pop(0)
            self.last_action = action
            return action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    # Enable automatic mode
                    if not self.automatic_mode:
                        self.automatic_mode = True
                        if not prev_automatic_mode and self.automatic_mode:
                            logging.info("Automatic mode enabled. Resuming from last position.") 
                    else:
                        pass
                        # logging.info("Automatic mode is already enabled.") 
                    # self.automatic_mode = True
                    # self.current_segment_index = 0
                    # self.current_action_index = 0
                elif event.key == pygame.K_o:
                    # Disable automatic mode
                    if self.automatic_mode:
                        self.automatic_mode = False
                        logging.info("Automatic mode disabled.")
                        self.paused = False
                elif not self.automatic_mode:
                # Only process normal controls if not in automatic mode
                    if event.key in self.keymap:
                        action = self.keymap[event.key]
                        if action in [Action.LEFT, Action.RIGHT]:
                            self.enqueue_rotation_actions(action)
                            if self.action_queue:
                                action = self.action_queue.pop(0)
                                self.last_action = action
                                return action
                        elif action in [Action.FORWARD, Action.BACKWARD]:
                            self.movement_state = action
                        else:
                            self.last_action = action
                            return action
            elif event.type == pygame.KEYUP and not self.automatic_mode:
                if event.key in self.keymap:
                    action = self.keymap[event.key]
                    if action in [Action.FORWARD, Action.BACKWARD]:
                        if self.movement_state == action:
                            self.movement_state = Action.IDLE
                            
        # print("Checking automatic_mode:", self.automatic_mode)
        if self.automatic_mode and not self.paused:
            if self.automatic_mode != prev_automatic_mode or (prev_paused and not self.paused):    
                logging.info("Inside automatic_mode block")
                
        # Automatic action sequence logic
        # Check if we still have actions and not at the end of the segment
            if self.actions and self.current_segment_index < len(self.actions):
                current_actions = self.actions[self.current_segment_index]
                if self.current_action_index < len(current_actions):
                # Convert the action string (e.g., "Action.FORWARD") to an Action enum
                    action_str = current_actions[self.current_action_index]
                    action_name = action_str.split('.')[-1]
                    action = getattr(Action, action_name, Action.IDLE)
                    print("Automatic mode action:", action)
                    
                # Check for walls during automatic mode
                    if action in [Action.FORWARD, Action.BACKWARD] and self.wall_detected():
                        if not self.paused:
                            logging.warning("Wall detected! Pausing automatic mode.")
                        self.paused = True   
                    # Stop moving forward/backward if wall is detected
                        return Action.IDLE

                    self.current_action_index += 1
                    return action
                else:
                    # Move to next segment or stop
                    self.current_segment_index += 1
                    self.current_action_index = 0
                    return Action.IDLE
            else:
                #Revert to manual by choise
                if self.automatic_mode:
                    logging.info("No more actions. Automatic mode disabled.")
                self.automatic_mode = False
                return Action.IDLE
        else:
            if self.automatic_mode and self.paused and not self.wall_detected():
                if self.paused:
                    logging.info("Wall cleared! Resuming automatic mode.")
                self.paused = False
                        

            # If not automatic and no actions triggered:
        if self.movement_state == Action.FORWARD:
            if self.wall_detected():
                self.movement_state = Action.IDLE
                self.last_action = Action.IDLE
            else:
                self.last_action = self.movement_state
                return self.movement_state
        elif self.movement_state == Action.BACKWARD:
            self.last_action = self.movement_state
            return self.movement_state

        self.last_action = Action.IDLE
        return Action.IDLE
    
    def enqueue_rotation_actions(self, rotation_action):
        increments_per_45_degrees = int(self.increments_per_full_rotation * 45 / 360)
        self.action_queue.extend([rotation_action] * increments_per_45_degrees)
   

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
                    action = actions_data[str(i)].get("action")
                    if action == "Action.QUIT" or action == "Action.QUIT|CHECKIN":
                        action =="Action.IDLE"
                    action_sequence.append(action)
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
        
        graph = nx.from_numpy_array(self.adj_matrix)

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
        num_group = (len(self.node_path) // 5 + 1)
        #print(num_group)
        if self.count != self.tmp:
            if self.count < num_group:
                #print(num_group - self.count)
                for index in range(5):
                    try:
                        path = self.save_dir + str(self.node_path[self.count*5+index]*3+self.offset) + ".png"
                    except(IndexError):

                        pass

                    if os.path.exists(path):

                        img = cv2.imread(path)

                        images.append(img)

                        img_names.append(path.split('/')[-1].split('.')[0])

                    else:

                        print(f"Image with ID {index} does not exist")

                            # Combine images vertically or horizontally (adjust as needed)

                #print(len(images))

                combined_image = np.concatenate(images, axis=1)  

                if self.count != self.tmp and len(img_names) != 0:

                    print("image displayed: ", img_names)

                    self.tmp = self.count

                self.time_buffer = time.time()

                cv2.imshow(window_name, combined_image)

                cv2.waitKey(1)

            else:

                self.count -= 1

                print("Goal Reached!!")

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

#-------------------------------------------------------------------------------------------------------------- mapping + wall detection + ROI defined
    def update_position_heading(self):
        action = self.last_action
        if action == Action.FORWARD:
            rad = math.radians(self.heading)
            dx = self.movement_per_action * math.cos(rad)
            dy = self.movement_per_action * math.sin(rad)
            self.position += np.array([dx, dy])
        elif action == Action.BACKWARD:
            rad = math.radians(self.heading)
            dx = -self.movement_per_action * math.cos(rad)
            dy = -self.movement_per_action * math.sin(rad)
            self.position += np.array([dx, dy])
        elif action == Action.LEFT:
            self.heading = (self.heading + self.rotation_per_action) % 360
        elif action == Action.RIGHT:
            self.heading = (self.heading - self.rotation_per_action) % 360
        self.path.append(self.position.copy())
        

    def update_map(self):
        if self.map_surface is None:
            return
        self.map_surface.fill((255, 255, 255))  # White background
        scale = 5
        offset_x = self.map_surface.get_width() // 2
        offset_y = self.map_surface.get_height() // 2

        if len(self.path) > 1:
            points = [(int(pos[0] * scale + offset_x), int(offset_y - pos[1] * scale)) for pos in self.path]
            pygame.draw.lines(self.map_surface, (255, 0, 0), False, points, 2)

        x = int(self.position[0] * scale + offset_x)
        y = int(offset_y - self.position[1] * scale)
        pygame.draw.circle(self.map_surface, (0, 0, 255), (x, y), 5)
        

    def get_front_roi(self, image, width_percentage=0.2, height_percentage=0.1, vertical_offset_percentage=0.00):
        height, width = image.shape[:2]
        roi_width = int(width * width_percentage)
        roi_height = int(height * height_percentage)
        horizontal_offset = int((width - roi_width) / 2)
        vertical_offset = int(height * vertical_offset_percentage)
        x_start = horizontal_offset
        x_end = horizontal_offset + roi_width
        y_start = height - roi_height - vertical_offset
        y_end = height - vertical_offset
        x_start = max(0, x_start)
        x_end = min(width, x_end)
        y_start = max(0, y_start)
        y_end = min(height, y_end)
        roi = image[y_start:y_end, x_start:x_end]
        return roi

    def detect_walls_in_roi(self, roi, edge_density_threshold=0.03, line_count_threshold=1):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        median_intensity = np.median(blurred)
        lower_thresh = int(max(0, 0.66 * median_intensity))
        upper_thresh = int(min(255, 1.33 * median_intensity))
        edges = cv2.Canny(blurred, lower_thresh, upper_thresh)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels
        # logging.info(f"Edge density in ROI: {edge_density:.4f}")
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=70,
                                minLineLength=20, maxLineGap=20)
        num_lines = len(lines) if lines is not None else 0
        # logging.info(f"Number of Hough lines detected: {num_lines}")
        wall_detected = False
        if edge_density > edge_density_threshold or num_lines >= line_count_threshold:
            # logging.info("Wall detected based on edge density.")
            wall_detected = True
        # else:
        #     # logging.info("No wall detected.")
        return wall_detected, edges, lines


    def wall_detected(self):
        if self.fpv is None:
            return False
        width_percentage = 0.2
        height_percentage = 0.1
        vertical_offset_percentage = 0.00
        roi = self.get_front_roi(self.fpv, width_percentage, height_percentage, vertical_offset_percentage)
        wall_detected, _, _ = self.detect_walls_in_roi(roi)
        return wall_detected 
#--------------------------------------------------------------------------------------------------------------

    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv
        self.update_position_heading()
        
        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            map_width = 250
            self.map_surface = pygame.Surface((map_width, h))
            screen_width = w + map_width
            self.screen = pygame.display.set_mode((screen_width, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image

        self.update_map()
        
        detection_image = self.fpv.copy()
        
        # pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # Nothing to do here since exploration data has been provided
                pass
            
            # If in navigation stage
            elif self._state and self._state[1] == Phase.NAVIGATION:
                # Parameters for ROI
                width_percentage = 0.2
                height_percentage = 0.1
                vertical_offset_percentage = 0.00

                roi = self.get_front_roi(self.fpv, width_percentage, height_percentage, vertical_offset_percentage)
                wall, edges, lines = self.detect_walls_in_roi(roi)
                height, width = self.fpv.shape[:2]
                roi_width = int(width * width_percentage)
                roi_height = int(height * height_percentage)
                horizontal_offset = int((width - roi_width) / 2)
                vertical_offset = int(height * vertical_offset_percentage)

                x_start = horizontal_offset
                x_end = horizontal_offset + roi_width
                y_start = height - roi_height - vertical_offset
                y_end = height - vertical_offset

                cv2.rectangle(detection_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                roi_with_edges = cv2.addWeighted(roi, 0.7, edges_colored, 0.3, 0)
    
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(roi_with_edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

                detection_image[y_start:y_end, x_start:x_end] = roi_with_edges

                if wall:
                    cv2.putText(detection_image, 'Wall Ahead!', (50, y_start - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(detection_image, 'Path Clear', (50, y_start - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
                
                if self.goal_id is None:

                    img_list = []

                    goal_list = []

                    # Get the neighbor nearest to the front view of the target image and set it as goal

                    targets = self.get_target_images()

                    print(len(targets))

                    for i in range(len(targets)):

                        targets[i] = cv2.cvtColor(targets[i], cv2.COLOR_BGR2GRAY) / 255

                        self.goal_id = self.query(targets[i])

                        goal_path = self.save_dir + str(self.goal_id) + ".png"
                        self.node_path = self.compute_shortest_path()
                    
                        #print(self.path)

                        if os.path.exists(goal_path):

                            img = cv2.imread(goal_path)

                            img_list.append(img)

                            goal_list.append(self.goal_id)

                            print('path of Goal_ID: ', self.goal_id , ' has length of: ',len(self.node_path))

                        else:

                            print(f"Goal Image with ID {self.goal_id} does not exist")

                            # Combine images vertically or horizontally (adjust as needed)

                        

                    combined_goal_image = np.concatenate(img_list, axis=1)

                    cv2.imshow('targets',combined_goal_image)

                    cv2.waitKey(0)

                    # Ask user the path to go:

                    #decision = user keyboard inpui 1-4

                    while True:

                        print("Choose a goal (1-", len(goal_list), "):") 

                        try:

                            decision = int(input())  

                            if 1 <= decision <= len(goal_list):

                                break

                            else:

                                print("Invalid choice. Please enter a number between 1 and", len(goal_list))

                        except ValueError:

                            print("Invalid input. Please enter a number.")



                    # Compute at the begining only to save time

                    self.goal_id = goal_list[decision-1]

                    self.node_path = self.compute_shortest_path()
                    self.path_segments, self.frames, self.actions = self.find_path_segments(self.node_path) ##3
                    # display the final decison

                    print(f'Goal ID: {self.goal_id}') # goal_id = 2336
                    # Compute at the begining only to save time
                    self.node_path = self.compute_shortest_path()
                    self.path_segments, self.frames, self.actions = self.find_path_segments(self.node_path) ##3
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
        rgb = convert_opencv_img_to_pygame(detection_image)
        self.screen.blit(rgb, (0, 0))
        self.screen.blit(self.map_surface, (self.fpv.shape[1], 0))
        pygame.display.update()

                

                                

if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())