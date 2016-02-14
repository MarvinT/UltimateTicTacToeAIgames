
def parse_command(instr, bot, pos):
    if instr.startswith('action move'):
        time = int(instr.split(' ')[-1])
        x, y = bot.get_move(pos, time)
        return 'place_move %d %d\n' % (x, y)
    elif instr.startswith('update game field'):
        fstr = instr.split(' ')[-1]
        pos.parse_field(fstr)
    elif instr.startswith('update game macroboard'):
        mbstr = instr.split(' ')[-1]
        pos.parse_macroboard(mbstr)
    elif instr.startswith('update game move'):
        pos.nmove = int(instr.split(' ')[-1])
    elif instr.startswith('settings your_botid'):
        myid = int(instr.split(' ')[-1])
        bot.myid = myid
        bot.oppid = 1 if myid == 2 else 2
    elif instr.startswith('settings timebank'): 
        bot.timebank = int(instr.split(' ')[-1])
    elif instr.startswith('settings time_per_move'): 
        bot.time_per_move = int(instr.split(' ')[-1])
    return ''

if __name__ == '__main__':
    import sys
    from position import Position
    from randombot import RandomBot

    pos = Position()
    bot = RandomBot()
    
    while True:
        try:
            instr = raw_input()
        except Exception as e:
            sys.stderr.write('error reading input')
        outstr = parse_command(instr, bot, pos)
        sys.stdout.write(outstr)
        sys.stdout.flush()            
            