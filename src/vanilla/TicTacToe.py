from random import random

UNMARKED,AGENT,ENV = 0,1,2

def create_TicTacToe_env(who_goes_first):
    class E:
        def __init__(self):
            self.num_legal_actions = 9
            self.num_possible_obs = pow(2,18)
            self.fnc = lambda T, play: abstract_tictactoe_env(T, play, who_goes_first)

    return E

TicTacToe1 = create_TicTacToe_env(who_goes_first='agent')
TicTacToe2 = create_TicTacToe_env(who_goes_first='env')
TicTacToe3 = create_TicTacToe_env(who_goes_first='rand')

TL,TM,TR = 0,1,2
ML,MM,MR = 3,4,5
BL,BM,BR = 6,7,8
positions = [TL,TM,TR,ML,MM,MR,BL,BM,BR]

winning_rows = [
    [TL,TM,TR], [ML,MM,MR], [BL,BM,BR],
    [TL,ML,BL], [TM,MM,BM], [TR,MR,BR],
    [TL,MM,BR], [TR,MM,BL]
]

def abstract_tictactoe_env(T, play, who_goes_first):
    if len(play) == 0:
        board = new_board(who_goes_first)
        reward = 0
        obs = encode_board(board)
        return (reward, obs)

    obs, action = play[-2], play[-1]
    board = decode_board(obs)

    if not(action in positions) or (board[action] != UNMARKED):
        reward = 0
        return (reward, obs)

    board[action] = AGENT
    if detect_win(board, AGENT):
        board = new_board(who_goes_first)
        reward = 1
        obs = encode_board(board)
        return (reward, obs)

    if detect_draw(board):
        board = new_board(who_goes_first)
        reward = 0
        obs = encode_board(board)
        return (reward, obs)

    board[random_unmarked_pos(board)] = ENV
    if detect_win(board, ENV) or detect_draw(board):
        board = new_board(who_goes_first)
        reward = 0
        obs = encode_board(board)
        return (reward, obs)

    reward = 0
    obs = encode_board(board)
    return (reward, obs)

def encode_board(board):
    encode_dict = {UNMARKED: '00', AGENT: '01', ENV: '10'}
    binary_pairs = [encode_dict[board[x]] for x in positions]
    binary_str = ''.join(binary_pairs)
    binary_str = '0b' + binary_str
    return eval(binary_str)

def decode_board(code):
    decode_dict = {'00': UNMARKED, '01': AGENT, '10': ENV}
    binary_str = bin(code)[2:].zfill(18)
    board = {x:decode_dict[binary_str[2*x:2*x+2]] for x in positions}
    return board

def new_board(who_goes_first):
    board = {x: UNMARKED for x in positions}

    if who_goes_first == 'rand':
        who_goes_first = 'agent' if random()<.5 else 'env'

    if who_goes_first == 'env':
        pos = random_unmarked_pos(board)
        board[pos] = ENV
    return board

def random_unmarked_pos(board):
    while True:
        pos = int(random()*len(positions))
        if board[pos] == UNMARKED:
            return pos

def detect_win(board, who):
    for row in winning_rows:
        if all((board[x]==who) for x in row):
            return True
    return False

def detect_draw(board):
    for pos in positions:
        if board[pos] == UNMARKED:
            return False
    return True