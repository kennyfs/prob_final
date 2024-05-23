data = []
for i in range(10):
    for j in range(10):
        d = f"{i+j:03}{2*i+4*j:03}{i:02}{j:02}"
        data.append([int(dd) for dd in d])
print(data)