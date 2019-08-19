import numpy as np
import pydensecrf.densecrf as dcrf

d = dcrf.DenseCRF2D(640,480,5)
print(d)
Q = d.inference(5)
print(Q)


#prob_map = readt7()



