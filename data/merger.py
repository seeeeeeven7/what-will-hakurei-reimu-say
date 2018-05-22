from os import walk

with open('talk_in_game/all_withoutspace.txt', 'w', encoding = 'utf8') as outfile:
    f = []
    for (dirpath, dirnames, filenames) in walk('talk_in_game/seperate'):
        f.extend(filenames)
        for filename in filenames:
            lines = open('talk_in_game/seperate/' + filename, 'r', encoding = 'utf8').read().splitlines()
            for line in lines:
                h = 0
                s = ''
                for ch in list(line):
                    if ch == '<':
                        h += 1
                    else:
                        if ch == '>':
                            h -= 1
                        else:
                            if h == 0:
                                s += ch
                print(s, file = outfile)