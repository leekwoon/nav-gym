# modified from:
# https://github.com/ignc-research/arena-tools
import cv2
import numpy as np


# create empty map with format given by height,width and initialize empty tree
def initialize_map(height, width, type="indoor"):
    if type == "outdoor":
        map = np.tile(1, [height, width])
        # map[slice(1, height-1), slice(1, width-1)] = 0
        map[slice(5, height-5), slice(5, width-5)] = 0
        return map
    else:
        return np.tile(1, [height, width])


def insert_root_node(map, tree):  # create root node in center of map
    root_node = [int(np.floor(map.shape[0]/2)),
                    int(np.floor(map.shape[1]/2))]
    map[root_node[0], root_node[1]] = 0
    tree.append(root_node)


# find nearest node according to L1 norm
def find_nearest_node(random_position, tree):
    nearest_node = []
    min_distance = np.inf
    for node in tree:
        distance = sum(np.abs(np.array(random_position)-np.array(node)))
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node


# insert new node into the map and tree
def insert_new_node(random_position, tree, map):
    map[random_position[0], random_position[1]] = 0
    tree.append(random_position)


def get_constellation(node1, node2):
    # there are two relevant constellations for the 2 nodes, which must be considered when creating the horizontal and vertical path
    # 1: lower left and upper right
    # 2: upper left and lower right
    # each of the 2 constellation have 2 permutations which must be considered as well
    constellation1 = {
        # x1>x2 and y1<y2
        "permutation1": node1[0] > node2[0] and node1[1] < node2[1],
        "permutation2": node1[0] < node2[0] and node1[1] > node2[1]}  # x1<x2 and y1>y2
    if constellation1["permutation1"] or constellation1["permutation2"]:
        return 1
    else:
        return 2


def create_path(node1, node2, corridor_radius, map):
    coin_flip = np.random.random()
    # x and y coordinates must be sorted for usage with range function
    x1, x2 = sorted([node1[0], node2[0]])
    y1, y2 = sorted([node1[1], node2[1]])
    if get_constellation(node1, node2) == 1:  # check which constellation
        # randomly determine the curvature of the path (right turn/left turn)
        if coin_flip >= 0.5:
            map[slice(x1-corridor_radius, x1+corridor_radius+1), range(y1 -
                                                                        corridor_radius, y2+1+corridor_radius, 1)] = 0  # horizontal path
            map[range(x1-corridor_radius, x2+1+corridor_radius, 1), slice(y1 -
                                                                            corridor_radius, y1+corridor_radius+1)] = 0  # vertical path
        else:
            map[slice(x2-corridor_radius, x2+corridor_radius+1), range(y1 -
                                                                        corridor_radius, y2+1+corridor_radius, 1)] = 0  # horizontal path
            map[range(x1-corridor_radius, x2+1+corridor_radius, 1), slice(y2 -
                                                                            corridor_radius, y2+corridor_radius+1)] = 0  # vertical path
    else:
        # randomly determine the curvature of the path (right turn/left turn)
        if coin_flip >= 0.5:
            map[slice(x1-corridor_radius, x1+corridor_radius+1), range(y1 -
                                                                        corridor_radius, y2+1+corridor_radius, 1)] = 0  # horizontal path
            map[range(x1-corridor_radius, x2+1+corridor_radius, 1), slice(y2 -
                                                                            corridor_radius, y2+corridor_radius+1)] = 0  # vertical path
        else:
            map[slice(x2-corridor_radius, x2+corridor_radius+1), range(y1 -
                                                                        corridor_radius, y2+1+corridor_radius, 1)] = 0  # horizontal path
            map[range(x1-corridor_radius, x2+1+corridor_radius, 1), slice(y1 -
              
                                                                            corridor_radius, y1+corridor_radius+1)] = 0  # vertical path
# sample position from map within boundary and leave tolerance for corridor width
def sample(map, corridor_radius):
    random_x = np.random.choice(
        range(corridor_radius+2, map.shape[0]-corridor_radius-1, 1))
    random_y = np.random.choice(
        range(corridor_radius+2, map.shape[1]-corridor_radius-1, 1))
    return [random_x, random_y]


def create_indoor_map(corridor_width, iterations):
    tree = []  # initialize empty tree
    map = initialize_map(height=100, width=100)
    insert_root_node(map, tree)
    for i in range(iterations):  # create as many paths/nodes as defined in iteration
        random_position = sample(map, corridor_width)
        # nearest node must be found before inserting the new node into the tree, else nearest node will be itself
        nearest_node = find_nearest_node(random_position, tree)
        insert_new_node(random_position, tree, map)
        create_path(random_position, nearest_node,
                            corridor_width, map)
    map = cv2.resize(
        map.astype(np.uint8),
        (1000, 1000),
        interpolation=cv2.INTER_NEAREST
    )
    map_data = np.zeros((1000, 1000), dtype='int8')
    map_data[map == 1] = 100 # occupied
    map_data = np.flipud(map_data)
    map_info = {
        'data': map_data,
        'origin': (0, 0),
        'resolution': 0.05,
        'width': 1000, 
        'height': 1000 
    }
    return map_info


def create_outdoor_map(obstacle_number, obstacle_width):
    obstacle_width = int(10 * obstacle_width)
    map = initialize_map(height=400, width=400, type="outdoor")
    for i in range(obstacle_number):
        random_position = sample(map, obstacle_width)
        map[slice(random_position[0]-obstacle_width, random_position[0]+obstacle_width+1),  # create 1 pixel obstacles with extra radius if specified
            slice(random_position[1]-obstacle_width, random_position[1]+obstacle_width+1)] = 1
    map_data = np.zeros((400, 400), dtype='int8')
    map_data[map == 1] = 100 # occupied
    map_data = np.flipud(map_data)
    map_info = {
        'data': map_data,
        'origin': (0, 0),
        'resolution': 0.05,
        'width': 400,
        'height': 400
    }
    return map_info


