class Region:

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.pos: tuple[int, int] = (x, y)
        self.width: int = w
        self.height: int = h

    @property
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

    def value(self, ii, scale: float = 1.0) -> float:
        sum_pos_reg = sum((r.intensity(ii, scale) for r in self.positive_regions))
        sum_neg_reg = sum((r.intensity(ii, scale) for r in self.negative_regions))

        return sum_neg_reg - sum_pos_reg

def build_features():
    pass