import re

print(re.search('[0-9]*$','fdd545df'))
print(re.search('^[a-z]+', '1hello'))
print(re.match('\d+', '11fjk'))

g = re.match('(\w+) (\w+)', 'sg 354')
print(g.groups())
print(g.group(2))
f = re.match('(?P<Func>[a-zA-z0-9_]+)\((?P<arg>\w+)\)', 'exit(naga)')
print(f.groups())
print(re.sub('[0-9]', r'{}', 'sdj43kdmkf83jdn8 mk38,sd89m 8d89jnr803jm u3mnr39 997895 dj'))
print(re.sub('[a-zA-Z]',lambda g:chr(ord(g.group())-32),'hello world!1234'))