NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3

maze1_config = {
    'start_room': 1,
    'end_room': 3,
    'exits': {
        1: {EAST: 2},
        2: {WEST: 1, EAST: 3}
    }
}

def Maze1(T, play):
    return abstract_maze_env(T, play, maze=maze1_config)

maze2_config = {
    'start_room': 1,
    'end_room': 5,
    'exits': {
        1: {EAST: 2},
        2: {WEST: 1, EAST: 3},
        3: {WEST: 2, EAST: 4},
        4: {WEST: 3, EAST: 5}
    }
}

def Maze2(T, play):
    return abstract_maze_env(T, play, maze=maze2_config)

maze3_config = {
    'start_room': 1,
    'end_room': 6,
    'exits': {
        1: {SOUTH: 2, EAST: 3},
        2: {NORTH: 1},
        3: {WEST: 1, EAST: 5, NORTH: 4, SOUTH: 6},
        4: {SOUTH: 3},
        5: {WEST: 3}
    }
}

def Maze3(T, play):
    return abstract_maze_env(T, play, maze=maze3_config)

maze4_config = {
    'start_room': 1,
    'end_room': 6,
    'exits': {
        1: {SOUTH: 5, EAST: 2},
        2: {WEST: 1, EAST: 3},
        3: {WEST: 2, SOUTH: 4},
        4: {WEST: 5, EAST: 6},
        5: {EAST: 4, NORTH: 1}
    }
}

def Maze4(T, play):
    return abstract_maze_env(T, play, maze=maze4_config)

maze5_config = {
    'start_room': 1,
    'end_room': 6,
    'exits': {
        1: {EAST: 2},
        2: {WEST: 1, EAST: 3},
        3: {WEST: 2, EAST: 4, NORTH: 1},
        4: {WEST: 3, EAST: 5, NORTH: 1, SOUTH: 1},
        5: {WEST: 1, EAST: 1, SOUTH: 1, NORTH: 6}
    }
}

def Maze5(T, play):
    return abstract_maze_env(T, play, maze=maze5_config)

def abstract_maze_env(T, play, maze):
    if len(play) == 0:
        reward = 0
        obs = maze['start_room']
        return [reward, obs]

    room, action = play[-2], play[-1]

    if (room < maze['start_room']) or (room > maze['end_room']):
        reward = 0
        obs = maze['start_room']
        return [reward, obs]

    if not(action in maze['exits'][room]):
        reward = 0
        obs = room
        return [reward, obs]

    next_room = maze['exits'][room][action]

    if next_room == maze['end_room']:
        reward = 1
        obs = maze['start_room']
        return [reward, obs]

    reward = 0
    obs = next_room
    return [reward, obs]