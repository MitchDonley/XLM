import pdb
f = open('data/para/en-fr.en.train', 'r')
fw = open('data/syn/en-fr.en.train.part1', 'w')
i = 0
for line in f:
    if i > 500000:
        break
    if len(line.strip().split()) > 0 and len(line.strip().split()) < 256:
        fw.write(line)
        i += 1
