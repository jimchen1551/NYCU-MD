parameter = []
with open('parameter.txt') as f:
    rpara = f.readlines()
    for i in rpara:
        x = i.replace('\n', '')
        parameter.append(x.split(' '))
for i in range(3):
    for j in range(3):
        parameter[i][j] = parameter[i][j].split(',')
parameter[3][0] = parameter[3][0].split(',')