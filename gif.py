import os
import imageio

# png_dir = './Filter/'
# images = []
# for file_name in os.listdir(png_dir):
#     if file_name.endswith('.jpg'):
#         file_path = os.path.join(png_dir, file_name)
#         images.append(imageio.imread(file_path))
# imageio.mimsave('./results/FilterEven.gif', images)


png_dir = './first/'
images = []
names = []
for i in range(1, 21, 1):
    names.append("First_Rank_Even_"+str(i)+".jpg")
for file_name in names:
    if file_name.endswith('.jpg'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('./results/Fisrt_20.gif', images)


# png_dir = './except/'
# images = []
# names = []
# for i in range(0, 20, 1):
#     names.append("Except_First_Rank_"+str(i)+".jpg")
# for file_name in names:
#     if file_name.endswith('.jpg'):
#         file_path = os.path.join(png_dir, file_name)
#         images.append(imageio.imread(file_path))
# imageio.mimsave('./results/Except_20.gif', images)