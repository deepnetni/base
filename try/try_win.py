from scipy.signal import get_window
import numpy as np


win = get_window("hann", 128, fftbins=True)
win = np.sqrt(win)
print(win)

np.savetxt("window128w64.txt", (win,), fmt="%.8f", delimiter=",")
