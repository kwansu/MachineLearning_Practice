s = slice(3, 13, 2)

l = list(range(8))
sl = l.__getitem__(s)

print(sl)

#l[2:2] = range(-1,-20, -2)
l[2:8:2] = ['z', 'zz', 'zzz']
print(l)

del l[-3:]
print(l)

