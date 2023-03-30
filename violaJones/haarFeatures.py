import numpy as np

class Region:

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.pos: tuple[int, int] = (x, y)
        self.width: int = w
        self.height: int = h

    def intensity(self, ii, scale: float = 1.0) -> int:
        x, y = self.pos
        x1: int = int(x * scale)
        y1: int = int(y * scale)
        x2: int = x1 + int(self.width * scale) - 1
        y2: int = y1 + int(self.height * scale) - 1

        x1_pos: int = int(x1 > 0)
        y1_pos: int = int(y1 > 0)

        A: int = int(x1_pos * y1_pos * ii[x1 - 1, y1 - 1])
        B: int = int(y1_pos * ii[x2, y1 - 1])
        C: int = int(x1_pos * ii[x1 - 1, y2])
        D: int = int(ii[x2, y2])    

        return D - B - C + A

class HaarFeature:

    def __init__(self, positive_regions, negative_regions):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions

    def invert(self):
        return HaarFeature(self.negative_regions, self.positive_regions)

    def value(self, ii, scale: float = 1.0) -> float:
        sum_pos_reg = sum((r.intensity(ii, scale) for r in self.positive_regions))
        sum_neg_reg = sum((r.intensity(ii, scale) for r in self.negative_regions))

        return sum_neg_reg - sum_pos_reg

def build_features(
    width, height,
    shift: int = 1,
    min_w: int = 4, min_h: int = 4) -> np.ndarray[HaarFeature]:

    features = []
    for w in range(min_w, width + 1):
        for h in range(min_h, height + 1):
            for x in range(0, width - w, shift):
                for y in range(0, height - h, shift):

                    # c : current region
                    # r : right region
                    # rr: right-right region
                    # b : bottom region
                    # bb: bottom-bottom region
                    # br: bottom-right region
                    # ▓ : positive region
                    # ░ : negative region

                    c_reg: Region = Region(x, y, w, h)
                    r_reg: Region = Region(x + w, y, w, h)
                    rr_reg: Region = Region(x + 2 * w, y, w, h)
                    b_reg: Region = Region(x, y + h, w, h)
                    bb_reg: Region = Region(x, 2 * y, w, h)
                    br_reg: Region = Region(x + w, y + h, w, h)

                    # [Haar] 2 rectagles
                    # Horizontal ▓░
                    if x + w * 2 < width:
                        hf: HaarFeature = HaarFeature([c_reg], [r_reg])
                        features.append(hf)

                    # Vertical ▓
                    #          ░
                    if y + h * 2 < height:
                        hf: HaarFeature = HaarFeature([c_reg], [b_reg])
                        features.append(hf)

                    # [Haar] 3 rectagles 
                    # Horizontal ▓░▓
                    if x + w * 3 < width:
                        hf: HaarFeature = HaarFeature([c_reg, rr_reg], [r_reg])
                        features.append(hf)
                        features.append(hf.invert())
                    
                    # Vertical ▓
                    #          ░
                    #          ▓
                    if y + h * 3 < height:
                        hf: HaarFeature = HaarFeature([c_reg, bb_reg], [b_reg])
                        features.append(hf)

                    # [Haar] 4 rectagles ▓░
                    #                    ░▓
                    if x + w * 2 < width and y + h * 2 < height:
                        hf: HaarFeature = HaarFeature([c_reg, br_reg], [b_reg, r_reg])
                        features.append(hf)

    return np.array(features)
