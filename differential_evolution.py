import os
import sys
from subprocess import Popen, PIPE, STDOUT
import cPickle as pickle
import numpy as np
import re

weight_template = 'gen%03did%03d.pkl'

def main():
    weights_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights')
    ensure_folder_exists(weights_folder)
    generation = find_greatest_generation(weights_folder)
    pop_size = 200
    if generation == -1: #initialize population
        print 'Generating starting population of size %d' % (pop_size)
        generation = 0
        for i in range(pop_size):
            weights_file = weight_template % (generation, i)
            with open(os.path.join(weights_folder, weights_file), 'wb') as f:
                pickle.dump(gen_weights(), f)

    scores = np.zeros(pop_size)
    for botid in range(pop_size):
        weights_file = weight_template % (generation, botid)
        for i in range(50):
            scores[botid] += sim_game(bot1_opts=os.path.join(weights_folder, weights_file))
        print 'generation %03d\t\tbot %03d\t\tcumulative_score: %d' % (generation, botid, scores[botid])

    fitness = np.maximum(0, scores)
    fitness -= np.min(fitness)
    fitness /= np.sum(fitness)

    generation += 1
    new_pop = np.random.choice(pop_size, pop_size, p=fitness)
    # need more checks here to make sure population hasn't been decimated
    assert len(np.unique(new_pop)) > 5, 'only %d unique individuals selected for the next generation' % (len(np.unique(new_pop)))

    F = 1.0  # differntial weight [0, 2]
    CR = .5 # crossover probability [0, 1]

    for botid in range(pop_size):
        x = new_pop[botid]
        a = np.random.choice(new_pop)
        while x==a:
            a = np.random.choice(new_pop)
        b = np.random.choice(new_pop)
        while x==b or a==b:
            b = np.random.choice(new_pop)
        c = np.random.choice(new_pop)
        while x==c or a==c or b==c:
            c = np.random.choice(new_pop)
        with open(os.path.join(weights_folder, weight_template % (generation-1, x)), 'rb') as f:
            weights = pickle.load(f)
        with open(os.path.join(weights_folder, weight_template % (generation-1, a)), 'rb') as f:
            a_weights = pickle.load(f)
        with open(os.path.join(weights_folder, weight_template % (generation-1, b)), 'rb') as f:
            b_weights = pickle.load(f)
        with open(os.path.join(weights_folder, weight_template % (generation-1, c)), 'rb') as f:
            c_weights = pickle.load(f)

        weights_filter = np.random.rand(*weights['conv_weights'].shape) > CR
        weights['conv_weights'][weights_filter] = a_weights['conv_weights'][weights_filter] + F * (b_weights['conv_weights'][weights_filter] - c_weights['conv_weights'][weights_filter])
        weights_filter = np.random.rand(*weights['score_weights'].shape) > CR
        weights['score_weights'][weights_filter] = a_weights['score_weights'][weights_filter] + F * (b_weights['score_weights'][weights_filter] - c_weights['score_weights'][weights_filter])
        with open(os.path.join(weights_folder, weight_template % (generation, botid)), 'wb') as f:
            pickle.dump(weights, f)

def find_greatest_generation(weights_folder):
    starting_generation = -1
    for folder, file in file_list(weights_folder):
        match = re.match(r'gen(?P<generation>\d+)id(?P<id>\d+).pkl', file)
        if match:
            starting_generation = max(starting_generation, int(match.group('generation')))
    return starting_generation

def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def file_list(folder):
    return [(folder, filename) for filename in os.listdir(folder)]

def gen_weights():
    sigma = .1
    conv_layer_output_size = 5
    weights = {}
    weights['conv_weights'] = np.random.randn(9*3, conv_layer_output_size) * sigma
    weights['score_weights'] = np.random.randn(conv_layer_output_size*9+9, 1) * sigma
    return weights
        
def sim_game(bot1_name='rollout_bot', bot1_opts='', bot2_name='random_bot', bot2_opts=''):
    # Get robots who are fighting (player1, player2)
    bot1, bot2 = get_bots(bot1_name=bot1_name, bot1_opts=bot1_opts, bot2_name=bot2_name, bot2_opts=bot2_opts)
    # Simulate game init input
    send_init('1', bot1)
    send_init('2', bot2)
    round_num = 1
    move = 1
    field = ','.join(['0'] * 81)
    macroboard = ','.join(['-1'] * 9)
    #print_board(field, macroboard, round_num, '')
    while True:
        for bot_id, bot in [('1', bot1), ('2', bot2)]:
            # Wait for any key
            #raw_input()
            # Send inputs to bot
            move = send_update(bot, round_num, move, field, macroboard)
            # Update macroboard and game field
            field = update_field(field, move, str(bot_id))
            macroboard = update_macroboard(field, move)
            # Check for winner. If winner, exit.
            #print_board(field, macroboard, round_num, move)
            if is_winner(macroboard):
                bot1.kill()
                bot2.kill()
                return get_winner(macroboard)

            round_num += 1

def get_bots(bot1_name=None, bot1_opts='', bot2_name=None, bot2_opts=''):
    root = os.path.dirname(os.path.realpath(__file__))
    files = os.listdir(root)
    bots = [f for f in files
            if os.path.isdir(os.path.join(root, f)) and f != '.git']

    bot_list = '\n'.join(
        ['{}. {}'.format(i, bot) for i, bot in enumerate(bots)])
    if not bot1_name:
        bot1_name = bots[int(raw_input(
            'Choose Player 1:\n' + bot_list + '\n\n> '))]
    if not bot2_name:
        bot2_name = bots[int(raw_input(
            'Choose Player 2:\n' + bot_list + '\n\n> '))]

    bot1 = Popen(['python', 'main.py', bot1_opts],
                 cwd=os.path.join(root, bot1_name),
                 stdout=PIPE,
                 stdin=PIPE,
                 stderr=STDOUT)
    bot2 = Popen(['python', 'main.py', bot2_opts],
                 cwd=os.path.join(root, bot2_name),
                 stdout=PIPE,
                 stdin=PIPE,
                 stderr=STDOUT)

    return bot1, bot2


def send_init(bot_id, bot):
    init_input = (
        'settings timebank 10000\n'
        'settings time_per_move 500\n'
        'settings player_names player1,player2\n'
        'settings your_bot player{bot_id}\n'
        'settings your_botid {bot_id}\n'.format(bot_id=bot_id))

    bot.stdin.write(init_input)


def send_update(bot, round_num, move, field, macroboard):
    update_input = (
        'update game round {round}\n'
        'update game move {move}\n'
        'update game field {field}\n'
        'update game macroboard {macro}\n'
        'action move 10000\n'.format(
            round=round_num,
            move=move,
            field=field,
            macro=macroboard))

    bot.stdin.write(update_input)
    out = bot.stdout.readline().strip()
    #print 'bot output: ' + repr(out)
    if "Traceback" in out:
        t = bot.stdout.readline().strip()
        while t:
            print t
            t = bot.stdout.readline().strip()
        print "DONE"
    return out


def update_field(field, move, bot_id):
    col, row = move.split(' ')[1:3]
    arr = field.split(',')
    index = int(row) * 9 + int(col)
    if arr[index] != '0':
        raise RuntimeError(
            'Square {col} {row} already occupied by {occ}.'.format(
                col=col, row=row, occ=arr[index]))

    arr[index] = bot_id
    return ','.join(arr)


def update_macroboard(field, move):
    # break it up into small boards
    board = field.split(',')
    small_boards = []
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            sb = []
            sb.extend(board[r * 9 + c:r * 9 + c + 3])
            sb.extend(board[(r + 1) * 9 + c:(r + 1) * 9 + c + 3])
            sb.extend(board[(r + 2) * 9 + c:(r + 2) * 9 + c + 3])
            small_boards.append(sb)

    # determine macro board state
    def get_state(a):
        winopts = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [6, 4, 2]]

        winners = ('111', '222')
        for opt in winopts:
            val = a[opt[0]] + a[opt[1]] + a[opt[2]]
            if val in winners:
                return a[opt[0]]

        if '0' not in a:
            return '3'

        return '0'

    macroboard = [get_state(b) for b in small_boards]

    # modify macro board state based on availability of small board
    col, row = move.split(' ')[1:3]
    index = int(row) * 9 + int(col)
    boards = [
        [0, 3, 6, 27, 30, 33, 54, 57, 60],  # top-left
        [1, 4, 7, 28, 31, 34, 55, 58, 61],  # top-middle
        [2, 5, 8, 29, 32, 35, 56, 59, 62],  # top-right
        [9, 12, 15, 36, 39, 42, 63, 66, 69],  # middle-left
        [10, 13, 16, 37, 40, 43, 64, 67, 70],  # middle-middle
        [11, 14, 17, 38, 41, 44, 65, 68, 71],  # middle-right
        [18, 21, 24, 45, 48, 51, 72, 75, 78],  # bottom-left
        [19, 22, 25, 46, 49, 52, 73, 76, 79],  # bottom-middle
        [20, 23, 26, 47, 50, 53, 74, 77, 80]]  # bottom-right

    for i, b in enumerate(boards):
        if index in b:
            # If macro space available, update it to -1
            if macroboard[i] == '0':
                macroboard[i] = '-1'
                break
            else:  # If macro space not available, update all 0 to -1
                macroboard = ['-1' if m == '0' else m for m in macroboard]
                break

    return ','.join(macroboard)


def print_board(field, macroboard, round_num, move):
    field = field.replace('0', ' ')
    a = field.split(',')
    msg = ''
    for i in range(0, 81, 9):
        if not i % 27 and i > 0:
            msg += '---+---+---\n'

        msg += '|'.join([
            ''.join(a[i:i+3]),
            ''.join(a[i+3:i+6]),
            ''.join(a[i+6:i+9])]) + '\n'

    sys.stderr.write("\x1b[2J\x1b[H")  # clear screen
    msg += '\nRound {}\nmacroboard: {}\nfield: {}\nmove: {}\n'.format(
        round_num, macroboard, field, move)

    sys.stdout.write(msg)


def is_winner(macroboard):
    winopts = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [6, 4, 2]]

    m = macroboard.split(',')
    winners = ('111', '222')
    for opt in winopts:
        val = m[opt[0]] + m[opt[1]] + m[opt[2]]
        if val in winners:
            #print 'WINNER! Player {}'.format(m[opt[0]])
            return True
    if '-1' not in m:
        #print 'TIE!'
        return True

    return False

def get_winner(macroboard):
    winopts = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [6, 4, 2]]

    m = macroboard.split(',')
    winners = ('111', '222')
    scores = (0, 1, -1)
    for opt in winopts:
        val = m[opt[0]] + m[opt[1]] + m[opt[2]]
        if val in winners:
            return scores[int(m[opt[0]])]
    if '-1' not in m:
        return scores[0]
    return False


if __name__ == '__main__':
    main()