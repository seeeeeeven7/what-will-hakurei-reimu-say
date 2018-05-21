from os import walk

with open('talk_in_game/all.txt', 'w', encoding = 'utf8') as outfile:
    f = []
    for (dirpath, dirnames, filenames) in walk('talk_in_game'):
        f.extend(filenames)
        for filename in filenames:
            if filename == 'all.txt':
                continue
            lines = open('talk_in_game' + '/' + filename, 'r', encoding = 'utf8').read().splitlines()
            for line in lines:
                print(' '.join(list(line)), file = outfile)