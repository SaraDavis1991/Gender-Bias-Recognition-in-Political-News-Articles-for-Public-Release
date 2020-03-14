import random
women = ["Hillary Clinton", "Elizabeth Warren", "Sarah Palin", "Betsy Devos", "Alexandria Ocasio-Cortez" ]
men = ["Bernie Sanders", "Barrack Obama", "Joe Biden", "John McCain", "Donald Trump"]


######################FOLD 1#######################################
train1female = random.sample(women, k = 3)
train1male = random.sample(men, k =3)
train1 = train1female + train1male
val1female = random.sample(list(set(women) - set(train1female)), k =1)
val1male = random.sample(list(set(men) - set(train1male)), k =1)
val1 = val1female + val1male
test1female =  list(set(women) - set(train1female)-set(val1female))
test1male = list(set(men)-set(train1male)-set(val1male))
test1 = test1female + test1male

file1 = open("fold1.txt", 'w')
for i in range(len(train1)):
	file1.write(train1[i] + " 0\n")
for j in range(len(val1)):
	file1.write(val1[j] + " 1\n")
for k in range(len(test1)):
	file1.write(test1[k]+ " 2\n")
file1.close()


######################FOLD 2#######################################
test2female = random.sample(list (set(women) - set(test1female)), k = 1)
test2male = random.sample(list(set(men)- set(test1male)), k = 1)
test2 = test2female + test2male
val2female = random.sample(list(set(women) - set(test2female)- set(val1female)), k =1)
val2male = random.sample(list(set(men) - set(test2male)- set(val1male)), k =1)
val2 = val2female + val2male
train2female = random.sample(list(set(women)- set(test2female) - set(val2female)), k =3)
train2male = random.sample(list(set(men)-set(test2male)-set(val2male)), k = 3)
train2 = train2female + train2male


file2 = open("fold2.txt", 'w')
for i in range(len(train2)):
	file2.write(train2[i] + " 0\n")
for j in range(len(val2)):
	file2.write(val2[j] + " 1\n")
for k in range(len(test2)):
	file2.write(test2[k]+ " 2\n")
file2.close()

######################FOLD 3#######################################

test3female = random.sample(list (set(women) - set(test1female) - set(test2female)), k = 1)
test3male = random.sample(list(set(men)- set(test1male) - set(test2male)), k = 1)
test3 = test3female + test3male
val3female = random.sample(list(set(women) - set(test3female)- set(val1female) - set(val2female)), k =1)
val3male = random.sample(list(set(men) - set(test3male)- set(val1male)- set(val2male)), k =1)
val3 = val3female + val3male
train3female = random.sample(list(set(women)- set(test3female) - set(val3female)), k =3)
train3male = random.sample(list(set(men)-set(test3male)-set(val3male)), k = 3)
train3 = train3female + train3male

file3 = open("fold3.txt", 'w')
for i in range(len(train3)):
	file3.write(train3[i] + " 0\n")
for j in range(len(val3)):
	file3.write(val3[j] + " 1\n")
for k in range(len(test3)):
	file3.write(test3[k]+ " 2\n")
file3.close()


######################FOLD 4#######################################

test4female = random.sample(list (set(women) - set(test1female) - set(test2female)- set(test3female)), k = 1)
test4male = random.sample(list(set(men)- set(test1male) - set(test2male) - set(test3male)), k = 1)
test4 = test4female + test4male
val4female = random.sample(list(set(women) - set(test4female)- set(val1female) - set(val2female) - set(val3female)), k =1)
val4male = random.sample(list(set(men) - set(test4male)- set(val1male)- set(val2male)- set(val3male)), k =1)
val4 = val4female + val4male
train4female = random.sample(list(set(women)- set(test4female) - set(val4female)), k =3)
train4male = random.sample(list(set(men)-set(test4male)-set(val4male)), k = 3)
train4 = train4female + train4male

file4 = open("fold4.txt", 'w')
for i in range(len(train4)):
	file4.write(train4[i] + " 0\n")
for j in range(len(val4)):
	file4.write(val4[j] + " 1\n")
for k in range(len(test4)):
	file4.write(test4[k]+ " 2\n")
file4.close()


######################FOLD 5#######################################

test5female = random.sample(list (set(women) - set(test1female) - set(test2female)- set(test3female) - set(test4female)), k = 1)
test5male = random.sample(list(set(men)- set(test1male) - set(test2male) - set(test3male) - set(test4male)), k = 1)
test5 = test5female + test5male
val5female = random.sample(list(set(women) - set(test5female)- set(val1female) - set(val2female) - set(val3female) - set(val4female)), k =1)
val5male = random.sample(list(set(men) - set(test5male)- set(val1male)- set(val2male)- set(val3male) - set(val4male)), k =1)
val5 = val5female + val5male
train5female = random.sample(list(set(women)- set(test5female) - set(val5female)), k =3)
train5male = random.sample(list(set(men)-set(test5male)-set(val5male)), k = 3)
train5 = train5female + train5male

file5 = open("fold5.txt", 'w')
for i in range(len(train5)):
	file5.write(train5[i] + " 0\n")
for j in range(len(val5)):
	file5.write(val5[j] + " 1\n")
for k in range(len(test5)):
	file5.write(test5[k]+ " 2\n")
file5.close()

#print(train1, train2, train3, train4, train5)
#print(val1, val2, val3, val4, val5)
#print(test1, test2, test3, test4, test5)


