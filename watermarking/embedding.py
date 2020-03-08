import pywt # DWT
import math # SQRT
from watermarking import wavelet_diff

class Embedding:

    RED = 0
    GREEN = 1
    BLUE = 2
    KEY_FILENAME = "static/data/key.txt"

    # to get only single channel
    def getColorAt(self, color,image,col,row):
        return image[row][col][color]

    def getSingleColorImage(self, color, image):
        single_color_image = []
        
        for row in range(0,len(image) - 1):
            colors = []
            for col in range(0,len(image[row]) - 1):
                colors.append(self.getColorAt(color,image,col,row))
            single_color_image.append(colors)
        return single_color_image

    # to get wavelet difference
    def getWaveDiff(self, HL,LH):
        return abs(HL - LH)

    def getDescendingWaveletDiffs(self, HLs,LHs, numberOfResult):
        waveletDiffs = []
        for i in range(0, len(HLs)):
            for j in range(0,len(HLs)):
                waveletDiffs.append(
                    wavelet_diff.WaveletDiff(
                        self.getWaveDiff(HLs[i][j],LHs[i][j]),
                        j,
                        i
                    )
                )
        descended = sorted(waveletDiffs, key=lambda x: x.value, reverse=True)
        return descended[0:numberOfResult]

    # create key to get watermark from host on extraction
    def createKey(self, waveletDiffs, pxPerRow):
        file = open(self.KEY_FILENAME, "w")
        text = ''
        currentRowLength = 1
        for data in waveletDiffs:
            text += ("(" + str(data.x) + "," + str(data.y) + ")")
            if currentRowLength == pxPerRow:
                file.write(text + "\n")
                text = ''
                currentRowLength = 1
            else:
                currentRowLength += 1
        file.close()

    def embed(self, host, wm):
        # 1. Get part of image that is gonna be embedded with WM
        toBeEmbedded = self.getSingleColorImage(self.BLUE, host)

        # 2. DWT
        LL,(LH, HL, HH) = pywt.dwt2(toBeEmbedded, 'haar')

        # 3. Get wavelet diffs with descending sort. Only take 64 (as much as WM that is gonna be embedded)
        waveDiffs = self.getDescendingWaveletDiffs(HL, LH, numberOfResult=len(wm))

        # 4. Create key based on chosen wavelet diffs
        self.createKey(waveDiffs, math.sqrt(len(wm)))