import os
import shutil

base_path = "/home/deepak/PycharmProjects/3DCNN/3DCNN-final/ds/"

new_path = "/home/deepak/Desktop/UCF"


i=0
j=0
img_count = 0

for f1 in os.listdir(base_path):
	
	i += 1
	j=0
	
	dir_path = new_path + "/" + f1
	print(dir_path)

	# Making new 101 folder
	try:
		os.makedirs(dir_path)
	except OSError:
		if not os.path.isdir(dir_path):
			raise

	sub_dir = os.path.join(base_path,f1)

	print(f1+ " ----> " +sub_dir)

	for f2 in os.listdir(sub_dir):
		
		img_count = 0
		count = 0
		
		print(f2)
		j+=1

		copy_sub_dir = os.path.join(dir_path, f2)

		try:
			os.makedirs(copy_sub_dir)
		except OSError:
			if not os.path.isdir(copy_sub_dir):
				raise

		sub_dir_2 = os.path.join(sub_dir, f2)
		copy_sub_dir_2 = sub_dir_2
		
		z = [ ]
		count = 0
		for f3 in os.listdir(copy_sub_dir_2):
			z.append(f3)
			img_count +=1
		
	
		rem = img_count % 10
		img_count = img_count - rem
		index = img_count / 10


		for i in range(0, img_count):
			shutil.copy(os.path.join(copy_sub_dir_2,z[i]), os.path.join(copy_sub_dir, z[i]))
			count +=1

		print("===> %d" %count)
