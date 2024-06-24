total = 0
count = 1
f = open("base-joint.txt", "r")
for x in f:
    if x[0:18] == "for snr:  15 bler:":
        # print(count)
        number = float(x[20:-1])
        # print(number)
        total = total + number
        count = count + 1
        if count > 20:
            count= 1
            print(total/20)
            total = 0

f.close()
