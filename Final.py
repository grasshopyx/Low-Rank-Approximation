import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib
from numpy.random import beta
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import colorsys
import os

class svdClassRGB:
    def __init__(self, R, G, B):
        self.UR, self.SR, self.VR = np.linalg.svd(R, full_matrices=True)
        self.UG, self.SG, self.VG = np.linalg.svd(G, full_matrices=True)
        self.UB, self.SB, self.VB = np.linalg.svd(B, full_matrices=True)

    def Rank(self,i):
        u_i_R = self.UR[i].reshape(len(self.UR[i]), i)
        vh_i_R = self.VR[i].reshape(i, len(self.VR[i]))
        u_i_G = self.UG[i].reshape(len(self.UG[i]), i)
        vh_i_G = self.VG[i].reshape(i, len(self.VG[i]))
        u_i_B = self.UB[i].reshape(len(self.UB[i]), i)
        vh_i_B = self.VB[i].reshape(i, len(self.VB[i]))

        rank_i = []
        rank_i.append([self.SR[i] * u_i_R @ vh_i_R])  # row of u * col of v
        rank_i.append([self.SG[i] * u_i_G @ vh_i_G])
        rank_i.append([self.SB[i] * u_i_B @ vh_i_B])

class svdClassSingle:
    rank_store = {}  # store the calculated component, don't have to recalculate, to speed up

    def __init__(self, image):
        self.IM = image
        self.row = image.shape[0]
        self.col = image.shape[1]
        self.channel = image.shape[2]
        self.UR, self.SR, self.VR = np.linalg.svd(image[:, :, 0], full_matrices=True)
        self.UG, self.SG, self.VG = np.linalg.svd(image[:, :, 1], full_matrices=True)
        self.UB, self.SB, self.VB = np.linalg.svd(image[:, :, 2], full_matrices=True)
        self.rank_store[0] = self.Rank(0)
        # print("Rank Size:", self.rank_store[0].size)

    def Rank(self, i, Show=False, Save=False):
        u_i_R = self.UR[i].reshape(len(self.UR[i]), 1)  # u row
        vh_i_R = self.VR[i].reshape(1, len(self.VR[i]))  # v row
        # vh_i_R = self.VR[:, i].reshape(1, len(self.VR[i]))  # v col
        print("vh_i_R!!!!!!!!!!!!!!!!", vh_i_R)

        u_i_G = self.UG[i].reshape(len(self.UG[i]), 1)  # u row
        vh_i_G = self.VG[i].reshape(1, len(self.VG[i]))  # v row
        # vh_i_G = self.VG[:, i].reshape(1, len(self.VG[i])) # v col

        u_i_B = self.UB[i].reshape(len(self.UB[i]), 1)  # u row
        vh_i_B = self.VB[i].reshape(1, len(self.VB[i]))  # v row
        # vh_i_B = self.VB[:, i].reshape(1, len(self.VB[i]))  # v col

        RM = self.SR[i] * u_i_R @ vh_i_R
        GM = self.SG[i] * u_i_G @ vh_i_G
        BM = self.SB[i] * u_i_B @ vh_i_B
        rank_i = []
        # for i in range(self.IM.shape[0]):
        #      for j in range(self.IM.shape[1]):
        #          rank_i.append([int(RM[i][j]), int(GM[i][j]), int(BM[i][j])])  # RGB
        #          # rank_i.append([int(RM[i][j]), int(RM[i][j]), int(RM[i][j])])  # red channel to grey


        for row in range(self.IM.shape[0]):
            for col in range(self.IM.shape[1]):
                 rank_i.append([int(RM[row][col]), int(GM[row][col]), int(BM[row][col])])  # RGB

        arr_rank_i = np.asarray(rank_i).reshape(self.IM.shape[0], self.IM.shape[1], 3)
        # print(arr_rank_i.shape)
        print("Rank calculated")
        im = Image.fromarray(np.uint8(arr_rank_i))

        if Show:
            im.show()
        return arr_rank_i

    # def add_Rank(self, rk1, rk2):
    #     rank1 = np.asarray(rk1)
    #     rank1.reshape(self.row, self.col, self.channel)
    #     rank2 = np.asarray(rk2)
    #     rank2.reshape(self.row, self.col, self.channel)
    #     rank_added = []
    #     print("The rank1 shape is now:", rank1.shape)
    #     for row in range(rank1.shape[0]):
    #         for col in range(rank1.shape[1]):
    #             for channel in range(rank1.shape[2]):
    #                 rank_added.append(rank1[row, col, channel] + rank2[row, col, channel])
    #     return rank_added

    def FisrtRank(self, k, ShowFrist=False, Save=False):
        rank_sum = self.rank_store[0]
        for i in range(1, k+1):
            if i in self.rank_store:
                rank_sum = np.add(rank_sum, self.rank_store[i])
                # rank_sum = self.add_Rank(rank_sum, self.rank_store[i])
            else:
                self.rank_store[i] = self.Rank(i)
                rank_sum = np.add(rank_sum, self.rank_store[i])
                # rank_sum = self.add_Rank(rank_sum, self.rank_store[i])
        im = Image.fromarray(np.uint8(rank_sum))
        if ShowFrist:
            im.show()
        if Save:
            im.save(str(k) + ".jpg", "JPEG")
        return rank_sum


class SVD_Diag:
    rank_store = {}  # store the calculated component, don't have to recalculate, to speed up

    def __init__(self, image):
        self.IM = image
        self.row = image.shape[0]
        self.col = image.shape[1]
        self.channel = image.shape[2]
        self.UR, self.SR, self.VR = np.linalg.svd(image[:, :, 0], full_matrices=True)
        self.UG, self.SG, self.VG = np.linalg.svd(image[:, :, 1], full_matrices=True)
        self.UB, self.SB, self.VB = np.linalg.svd(image[:, :, 2], full_matrices=True)
        # self.rank_store[0] = self.Rank(0)
        # print("Rank Size:", self.rank_store[0].size)
        # print("@@@@@@@@@ S shape:", self.SR.shape)
        self.smaller = self.row if (self.row < self.col) else self.col

    def rgb_2_hsv(self, RGBimage):
        image_in_HSV = []
        for row in range(self.row):
            for col in range(self.col):
                image_in_HSV.append(colorsys.rgb_to_hsv(RGBimage[row,col,0], RGBimage[row,col,1], RGBimage[row,col,2]))
        HSV_arr = np.asarray(image_in_HSV).reshape(self.row, self.col, self.channel)
        return HSV_arr

    def hsv_2_rgb(self, HSVimage):
        image_in_RGB = []
        for row in range(self.row):
            for col in range(self.col):
                image_in_RGB.append(colorsys.rgb_to_hsv(HSVimage[0], HSVimage[1], HSVimage[2]))
        RGB_arr = np.asarray(image_in_RGB).reshape(self.row, self.col, self.channel)
        return RGB_arr

    def RGB_distribution(self):
        plt.style.use('bmh')
        data1 = self.SR
        data2 = self.SG
        data3 = self.SB

        def plot_beta_hist(data, CO):
            ax.hist(data, histtype="stepfilled",
                    bins=300, alpha=0.5, density=True, color=CO)

        fig, ax = plt.subplots()
        plot_beta_hist(data1, 'red')
        plot_beta_hist(data2, 'green')
        plot_beta_hist(data3, 'blue')
        ax.set_title("Theta Distribution of RGB Channels")

        plt.show()

    def distributionDraw(self):
        fig, ax = plt.subplots()

        # histogram our data with numpy
        data = self.SR
        n, bins = np.histogram(data, 100)

        # get the corners of the rectangles for the histogram
        left = np.array(bins[:-1])
        right = np.array(bins[1:])
        bottom = np.zeros(len(left))
        top = bottom + n


        # we need a (numrects x numsides x 2) numpy array for the path helper
        # function to build a compound path
        XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

        # get the Path object
        barpath = path.Path.make_compound_path_from_polys(XY)

        # make a patch out of it
        patch = patches.PathPatch(barpath)
        ax.add_patch(patch)

        # update the view limits
        ax.set_xlim(left[0], right[-1])
        ax.set_ylim(bottom.min(), top.max())
        plt.title("Theta Distribution of RGB Channels")
        plt.show()

    def RGB_dot(self, stop=-1, name=''):
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots()
        ax.plot(range(len(self.SR[:stop])), self.SR[:stop], 'r')
        ax.plot(range(len(self.SG[:stop])), self.SG[:stop], 'g')
        ax.plot(range(len(self.SB[:stop])), self.SB[:stop], 'b')
        ax.set_title('Singular Value Distribution of '+name)
        plt.show()
        return self.SR[:stop], self.SG[:stop], self.SB[:stop]

    def Rank(self, k, Show=False, Save=False):
        DiagR = np.zeros((self.row, self.col))
        DiagG = np.zeros((self.row, self.col))
        DiagB = np.zeros((self.row, self.col))
        if k <= self.row and k <= self.col:
            DiagR[0, 0] = self.SR[k]
            DiagG[0, 0] = self.SG[k]
            DiagB[0, 0] = self.SB[k]
        else:
            print("Error, i must smaller than the width and height!")


        RM = self.UR @ DiagR @ self.VR
        GM = self.UG @ DiagG @ self.VG
        BM = self.UB @ DiagB @ self.VB
        newIM = []
        for row in range(self.row):
            for col in range(self.col):
                newIM.append([RM[row, col], GM[row, col], BM[row, col]])
        arr_newIM = np.asarray(newIM)

        reshaped_im = arr_newIM.reshape(self.row, self.col, self.channel)
        print(reshaped_im.shape)

        im = Image.fromarray(np.uint8(reshaped_im))
        if Show:
            im.show()
        if Save:
            im.save("./results/" +"Rank_"+ str(k) + ".jpg", "JPEG")
        return arr_newIM

    def FisrtRank(self, k, ShowFrist=False, Save=False, SavePath='./first'):
        os.makedirs(SavePath, exist_ok=True)
        os.makedirs(SavePath)
        DiagR = np.zeros((self.row, self.col))
        DiagG = np.zeros((self.row, self.col))
        DiagB = np.zeros((self.row, self.col))
        if k <= self.row and k <= self.col:
            for i in range(k):
                DiagR[i, i] = self.SR[i]
                DiagG[i, i] = self.SG[i]
                DiagB[i, i] = self.SB[i]
        else:
            print("Error, k must smaller than the width and height!")


        RM = self.UR @ DiagR @ self.VR
        GM = self.UG @ DiagG @ self.VG
        BM = self.UB @ DiagB @ self.VB
        newIM = []
        for row in range(self.row):
            for col in range(self.col):
                newIM.append([RM[row, col], GM[row, col], BM[row, col]])
        arr_newIM = np.asarray(newIM)

        reshaped_im = arr_newIM.reshape(self.row, self.col, self.channel)
        print(reshaped_im.shape)

        im = Image.fromarray(np.uint8(reshaped_im))
        draw = ImageDraw.Draw(im)
        draw.text((20, 60), "FirstRank:"+str(k), fill=(255, 0, 0, 255))
        if ShowFrist:
            im.show()
        if Save:
            im.save(SavePath+str(k) + "First_Rank_" + ".jpg", "JPEG")
        return reshaped_im

    def LastRank(self, k, ShowFrist=False, Save=False, SavePath="./results/"):
        os.makedirs(SavePath, exist_ok=True)
        DiagR = np.zeros((self.row, self.col))
        DiagG = np.zeros((self.row, self.col))
        DiagB = np.zeros((self.row, self.col))

        smaller = self.row if (self.row < self.col) else self.col
        if k <= smaller:
            for i in range(k):
                DiagR[i, i] = self.SR[smaller - i -1]
                DiagG[i, i] = self.SG[smaller - i -1]
                DiagB[i, i] = self.SB[smaller - i -1]
        else:
            print("Error, k must smaller than the width and height!")


        RM = self.UR @ DiagR @ self.VR
        GM = self.UG @ DiagG @ self.VG
        BM = self.UB @ DiagB @ self.VB
        newIM = []
        for row in range(self.row):
            for col in range(self.col):
                newIM.append([RM[row, col], GM[row, col], BM[row, col]])
        arr_newIM = np.asarray(newIM)

        reshaped_im = arr_newIM.reshape(self.row, self.col, self.channel)
        print(reshaped_im.shape)

        im = Image.fromarray(np.uint8(reshaped_im))
        if ShowFrist:
            im.show()
        if Save:
            im.save(SavePath + "Last_Rank_" + str(k) + ".jpg", "JPEG")
        return arr_newIM

    def Filter(self, k, r_scale, g_scale, b_scale, ShowFrist=False, Save=False, SavePath="./Filter/"):
        os.makedirs(SavePath, exist_ok=True)
        DiagR = np.zeros((self.row, self.col))
        DiagG = np.zeros((self.row, self.col))
        DiagB = np.zeros((self.row, self.col))
        if k <= self.row and k <= self.col:
            for i in range(k):
                if (i < r_scale):
                    DiagR[i, i] = self.SR[i]
                if (i < g_scale):
                    DiagG[i, i] = self.SG[i]
                if (i < b_scale):
                    DiagB[i, i] = self.SB[i]
        else:
            print("Error, k must smaller than the width and height!")


        RM = self.UR @ DiagR @ self.VR
        GM = self.UG @ DiagG @ self.VG
        BM = self.UB @ DiagB @ self.VB
        newIM = []
        for row in range(self.row):
            for col in range(self.col):
                newIM.append([self.cut_one_digit(RM[row, col]), self.cut_one_digit(GM[row, col]), self.cut_one_digit(BM[row, col])])
        arr_newIM = np.asarray(newIM)

        reshaped_im = arr_newIM.reshape(self.row, self.col, self.channel)

        im = Image.fromarray(np.uint8(reshaped_im))
        draw = ImageDraw.Draw(im)
        # draw.rectangle(((0, 00), (100, 100)), fill="black")
        draw.text((20, 60), "Rank:("+str(r_scale)+","+str(g_scale)+","+str(b_scale)+")", fill=(255, 255, 255, 255))
        if ShowFrist:
            im.show()
        if Save:
            im.save(SavePath + str(r_scale)+"_"+ str(g_scale)+"_"+str(b_scale)+"_"+ "Filter_" + ".jpg", "JPEG")
        return arr_newIM

    def HSV(self, k, h_scale, s_scale, v_scale, ShowFrist=False, Save=False):

        HSVIM = self.rgb_2_hsv(self.IM)

        UH, SH, VH = np.linalg.svd(HSVIM[:, :, 0], full_matrices=True)
        US, SS, VS = np.linalg.svd(HSVIM[:, :, 0], full_matrices=True)
        UV, SV, VV = np.linalg.svd(HSVIM[:, :, 0], full_matrices=True)


        DiagR = np.zeros((self.row, self.col))
        DiagG = np.zeros((self.row, self.col))
        DiagB = np.zeros((self.row, self.col))
        if k <= self.row and k <= self.col:
            for i in range(k):
                if (i < h_scale):
                    DiagR[i, i] = SH[i]
                if (i < s_scale):
                    DiagG[i, i] = SS[i]
                if (i < v_scale):
                    DiagB[i, i] = SV[i]
        else:
            print("Error, k must smaller than the width and height!")


        RM = UH @ DiagR @ VH
        GM = US @ DiagG @ VS
        BM = UV @ DiagB @ VV
        newIM = []
        for row in range(self.row):
            for col in range(self.col):
                newIM.append([RM[row, col], GM[row, col], BM[row, col]])
        arr_newIM = np.asarray(newIM)

        reshaped_im = arr_newIM.reshape(self.row, self.col, self.channel)

        RGBIM = self.rgb_2_hsv(reshaped_im)

        im = Image.fromarray(np.uint8(RGBIM))
        draw = ImageDraw.Draw(im)
        # draw.rectangle(((0, 00), (100, 100)), fill="black")
        draw.text((20, 60), "Rank:("+str(h_scale)+","+str(s_scale)+","+str(v_scale)+")", fill=(255, 255, 255, 255))
        if ShowFrist:
            im.show()
        if Save:
            im.save("./HSV/" + str(h_scale)+"_"+ str(s_scale)+"_"+str(v_scale)+"_"+ "Filter_" + ".jpg", "JPEG")


        return arr_newIM

    def greater0_minus_row(self, digit):
        if digit < 0:
            return 0
        elif digit >= self.row:
            return self.row - 1
        else:
            return digit

    def greater0_minus_col(self, digit):
        if digit < 0:
            return 0
        elif digit >= self.col:
            return self.col - 1
        else:
            return digit

    def greater0_minus(self, digit, high, low=0):
        if digit < 0:
            return 0
        elif digit >= high:
            return high - 1
        else:
            return digit

    def even(self, imageOneChannel):
        for row in range(self.row):
            for col in range(self.col):
                sum = 0
                for i in range(9):
                    sum += imageOneChannel[self.greater0_minus_row(row-1+i//3), self.greater0_minus_col(col-1+i%3)]
                imageOneChannel[row,col] = sum//9
        return imageOneChannel

    def cut(self, imageOneChannel):
        for row in range(self.row):
            for col in range(self.col):
                if imageOneChannel[row, col] >= 255:
                    imageOneChannel[row, col] = 255
                elif imageOneChannel[row, col] < 0:
                    imageOneChannel[row, col] = 0
        return imageOneChannel

    def cut_one_digit(self, digit):
        if digit < 0:
            return 0
        elif digit >= 255:
            return 255
        else:
            return digit

    def FisrtRank_With_even(self, k, ShowFrist=False, Save=False, SavePath='./first/', FontColor=(255, 0, 0, 255)):
        os.makedirs(SavePath,exist_ok=True)
        print("FirstRankWithEven")
        DiagR = np.zeros((self.row, self.col))
        DiagG = np.zeros((self.row, self.col))
        DiagB = np.zeros((self.row, self.col))
        if k <= self.row and k <= self.col:
            for i in range(k):
                DiagR[i, i] = self.SR[i]
                DiagG[i, i] = self.SG[i]
                DiagB[i, i] = self.SB[i]
        else:
            print("Error, k must smaller than the width and height!")


        RM = self.UR @ DiagR @ self.VR
        GM = self.UG @ DiagG @ self.VG
        BM = self.UB @ DiagB @ self.VB

        # print("@@@@RM shape", RM.shape)
        RM_even = self.cut(RM)
        GM_even = self.cut(GM)
        BM_even = self.cut(BM)

        newIM = []
        for row in range(self.row):
            for col in range(self.col):
                newIM.append([RM_even[row, col], GM_even[row, col], BM_even[row, col]])
        arr_newIM = np.asarray(newIM)

        reshaped_im = arr_newIM.reshape(self.row, self.col, self.channel)
        print(reshaped_im.shape)

        im = Image.fromarray(np.uint8(reshaped_im))
        draw = ImageDraw.Draw(im)
        draw.text((20, 60), "Rank:"+str(k), fill=FontColor)
        if ShowFrist:
            im.show()
        if Save:
            im.save(SavePath + "First_Rank_Even_"+str(k) + ".jpg", "JPEG")
        return reshaped_im

    def ExceptFirstRank(self, k, ShowFrist=False, Save=False, SavePath="./except/"):
        os.makedirs(SavePath, exist_ok=True)
        DiagR = np.zeros((self.row, self.col))
        DiagG = np.zeros((self.row, self.col))
        DiagB = np.zeros((self.row, self.col))
        smaller = self.row if (self.row < self.col) else self.col
        if k <= smaller:
            for i in range(smaller):
                if i > k:
                    DiagR[i, i] = self.SR[i]
                    DiagG[i, i] = self.SG[i]
                    DiagB[i, i] = self.SB[i]
        else:
            print("Error, k must smaller than the width and height!")

        RM = self.UR @ DiagR @ self.VR
        GM = self.UG @ DiagG @ self.VG
        BM = self.UB @ DiagB @ self.VB
        newIM = []
        for row in range(self.row):
            for col in range(self.col):
                newIM.append([self.cut_one_digit(RM[row, col]), self.cut_one_digit(GM[row, col]), self.cut_one_digit(BM[row, col])])
        arr_newIM = np.asarray(newIM)

        reshaped_im = arr_newIM.reshape(self.row, self.col, self.channel)
        print(reshaped_im.shape)

        im = Image.fromarray(np.uint8(reshaped_im))
        draw = ImageDraw.Draw(im)
        draw.text((20, 60), "LastFromRank:"+str(k+1), fill=(255, 0, 0, 255))
        if ShowFrist:
            im.show()
        if Save:
            im.save(SavePath + "Except_First_Rank_" + str(k+1) + ".jpg", "JPEG")
        return reshaped_im

    def Mosaaic(self, scale=3, ShowFrist=False, Save=False, SavePath="./Mossaic/", FontColor=(255, 0, 0, 255), FileName=''):
        print("Now Dealing With Mossaic scale:", str(scale))
        os.makedirs(SavePath, exist_ok=True)
        MossaicImage = np.zeros((self.row, self.col, self.channel))
        eachRow_has_col = self.col//scale
        for row in range(self.row):
            for col in range(self.col):
                index = (row // scale) * eachRow_has_col + (col // scale)
                sumR = 0
                sumG = 0
                sumB = 0
                total =(scale * scale)
                for i in range(scale):
                    for j in range(scale):
                        sumR += self.IM[self.greater0_minus_row((row // scale)*scale + i), self.greater0_minus_col((col // scale)*scale + j), 0]
                        sumG += self.IM[self.greater0_minus_row((row // scale)*scale + i), self.greater0_minus_col((col // scale)*scale + j), 1]
                        sumB += self.IM[self.greater0_minus_row((row // scale)*scale + i), self.greater0_minus_col((col // scale)*scale + j), 2]
                MossaicImage[row, col, 0] = sumR//total
                MossaicImage[row, col, 1] = sumG//total
                MossaicImage[row, col, 2] = sumB//total

        MossaicArr = np.asarray(MossaicImage).reshape(self.row, self.col, self.channel)
        im = Image.fromarray(np.uint8(MossaicArr))
        draw = ImageDraw.Draw(im)
        draw.text((20, 60), "Rank:"+str(self.smaller//scale), fill=FontColor)
        if ShowFrist:
            im.show()
        if Save:
            im.save(SavePath + FileName +"Mossiac_Rank" + str(self.smaller//scale) + ".jpg", "JPEG")
        return MossaicArr


###################################
im = np.asarray(Image.open('NT.jpg'))
print(im.shape)
NT = SVD_Diag(im)

# NT.ExceptFirstRank(-1, True, True)
#
# _ = NT.Rank(0, True, True)
# ########Filter###########
# for i in np.arange(1,10,1):
#     for j in np.arange(1,10,1):
#         for k in np.arange(1,10,1):
#             _ = NT.Filter(20, i, j, k, False, True)
# #########################
# NT.HSV(20, 1, 1, 1, False, True)
# NT.LastRank(100, False, True)
# NT.FisrtRank_With_even(100, False, True)
#
# NT.distributionDraw()
# NT.RGB_distribution()
#
#
# for i in range(-1, 290, 1):
#     NT.ExceptFirstRank(i, False, True)
#
#
# for i in range(1, 290, 1):
#     NT.FisrtRank_With_even(i, False, True)
#
# NT.Mosaaic(5, False, True)
# NT.FisrtRank_With_even(60, False, True)
# _ = NT.RGB_dot(20, "Newton")

###################Distribution#########################

Mondrain = np.asarray(Image.open('grid_R.jpg'))
print(Mondrain.shape)
grid = SVD_Diag(Mondrain)

Picasso = np.asarray(Image.open('guernica_R.jpg'))
print(Picasso.shape)
Guernica = SVD_Diag(Picasso)

# ####################################################
# R1, G1, B1 = grid.RGB_dot(20, "Mondrain_grid")
# R2, G2, B2 = Guernica.RGB_dot(20, "Picasso_Guernica")
#
# plt.rcParams['axes.unicode_minus'] = False
# fig, ax = plt.subplots()
# ax.plot(range(len(R1)), R1, 'paleturquoise', label='Mondrain_grid')
# ax.plot(range(len(G1)), G1, 'paleturquoise')
# ax.plot(range(len(B1)), B1, 'paleturquoise')
# ax.plot(range(len(R2)), R2, 'plum', label='Picasso_Guernica')
# ax.plot(range(len(G2)), G2, 'plum')
# ax.plot(range(len(B2)), B2, 'plum')
# legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# ax.set_title('Singular Value Distribution of Both')
# plt.show()
# ####################################################
#
# for i in range(1, 51, 1):
#     grid.FisrtRank_With_even(i, False, True, "./Mondrain/", FontColor=(255, 255, 0, 255))
# for i in range(1, 51, 1):
#     Guernica.FisrtRank_With_even(i, False, True, "./Picasso/")
#
# Guernica.FisrtRank_With_even(200, False, True, "./Picasso/")
# Guernica.FisrtRank_With_even(100, False, True, "./Picasso/")


# grid.Mosaaic(50, False, True, FontColor=(255, 255, 0, 255))
# grid.Mosaaic(4, False, True, FontColor=(255, 255, 0, 255))
# for i in range(1, 101, 10):
#     grid.Mosaaic(i, False, True, FontColor=(255, 255, 0, 255), FileName='RBY')
# for i in range(1, 101, 10):
#     Guernica.Mosaaic(i, False, True, FontColor=(255, 255, 0, 255), FileName='Gurenica')

for i in [2, 4, 5, 10, 20, 40]:
    grid.Mosaaic(i, False, True, FontColor=(255, 255, 0, 255), FileName='RBY')
for i in [2, 4, 5, 10, 20, 40]:
    Guernica.Mosaaic(i, False, True, FontColor=(255, 255, 0, 255), FileName='Gurenica')


######################Merge############################
# im = np.asarray(Image.open('NTResized.jpg'))
# NT1 = SVD_Diag(im)
# merge_part_1 = NT1.FisrtRank(50, False, False)
# im = np.asarray(Image.open('Moon1.jpg'))
# Moon = SVD_Diag(im)
# merge_part_2 = Moon.FisrtRank(50, False, False)
# print("Image Shape of Merge Part 1:", merge_part_1.shape)
# print("Image Shape of Merge Part 2:", merge_part_2.shape)
# image_merged = []
# for i in range(merge_part_1.shape[0]):
#     for j in range(merge_part_1.shape[1]):
#         for k in range(merge_part_1.shape[2]):
#             image_merged.append((merge_part_1[i,j,k] + merge_part_2[i,j,k])/2)
# arr_merged_image = np.asarray(image_merged).reshape(merge_part_1.shape[0], merge_part_1.shape[1], merge_part_1.shape[2])
# im = Image.fromarray(np.uint8(arr_merged_image))
# im.show()
# im.save("./results/Merge" + ".jpg", "JPEG")
###################################################


# #####################################
# im = np.asarray(Image.open('Moon.jpg'))
# print(im.shape)
# print(im.shape[0])
# print(im.shape[1])
# print(im.shape[2])
# a = SVD_Diag(im)
# _ = a.Rank(0, True, True)
# _ = a.Rank(50, True, True)
# _ = a.LastRank(100, True, True)
###################filter
# for i in range(5, 290, 5):
#     _ = a.FisrtRank(i, False, True)




# ############################################################

# r = im[:, :, 0]
#
# # print(r.shape)
#
# rad_grey_image = []
# for i in range(im.shape[0]):
#      for j in range(im.shape[1]):
#          # rank_i.append([int(RM[i][j]), int(GM[i][j]), int(BM[i][j])])
#          v = r[i][j]
#          rad_grey_image.append([v, v, v])
#
# arr_rank_i = np.asarray(rad_grey_image).reshape(im.shape[0], im.shape[1], 3)
# print(arr_rank_i.shape)
# imNew = Image.fromarray(np.uint8(arr_rank_i))
# imNew.show()


############################################################
# UR, s, VR = np.linalg.svd(im[:, :, 0], full_matrices=True)
#
# u_i_R = UR[0].reshape(len(UR[0]), 1)
# vh_i_R = VR[0].reshape(1, len(VR[0]))
#
# RM =s[0] * u_i_R @ vh_i_R
#
# rank_0 = []
# for i in range(im.shape[0]):
#      for j in range(im.shape[1]):
#          rank_0.append([RM[i][j], RM[i][j], RM[i][j]])
#
# arr_rank_0 = np.asarray(rank_0).reshape(im.shape[0], im.shape[1], 3)
# imAfter = Image.fromarray(np.uint8(arr_rank_0))
# imAfter.show()

###############################
# ma = [[1, 2, 3]]
# marr = np.asarray(ma).reshape(3, 1)
# print(marr @ marr.T)





############################################
# im = Image.open('Moon.jpg')   #2580 * 2452
# # im = np.asarray(Image.open('Moon.jpg'))
# # print(im.size)
# # print(2580*2452*3)
# r, g, b = im.split()
# R = np.asarray(r)
# G = np.asarray(g)
# B = np.asarray(b)



