
import os, numpy, cv2, matplotlib.pyplot as plt

# General parameters
files         = os.listdir(r'./jpgs/allConfigs')
nFiles        = len(files)
nSampledFiles = 1000
nCopies       = 20
mu            = 0
sigma         = 1.0
plotting      = False

# Main function that stores nCopies copies of each image
def main():

    os.chdir('jpgs')

    itos = 0
    for file in files:
        itos += 1
        print str(int(float(itos)/nFiles*100)) + '% done'
        if file[-4:] == ".jpg":
            if float(file[7:-3]) <= int(nSampledFiles/2) or float(file[7:-3]) >= int(nFiles-nSampledFiles/2):
                img = cv2.imread('allConfigs/' + file)
                for i in xrange(nCopies):
                    if i == 0:
                        nZeros = 1
                    else:
                        nZeros = int(numpy.log10(nCopies)) - int(numpy.log10(i))

                    cv2.imwrite(file[0:-4] + '_' + '0'*nZeros + str(i) + '.jpg', addNoise(img))
    os.chdir('..')

# Add noise to an image
def addNoise(img):

    returnedImg = img + numpy.random.normal(mu, sigma, img.shape).astype('uint8')
    returnedImg[returnedImg<  0] = 0
    returnedImg[returnedImg>255] = 255

    if plotting:
        plt.figure()
        plt.imshow(returnedImg)
        plt.show()

    return returnedImg

# Main call
if __name__ == '__main__':
    main()