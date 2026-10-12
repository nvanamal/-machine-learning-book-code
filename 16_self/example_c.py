# SELF-SUPERVISED LEARNING | EXAMPLE C
# Next-sample prediction from a raw time signal
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
signal = np.sin(np.arange(700) * 0.08)
window, cutoff = 8, 500
# Inputs end at t-1; the generated target is the next sample t.
train_t = np.arange(window, cutoff)
test_t = np.arange(cutoff, len(signal))
def windows(times): return np.array([signal[t-window:t] for t in times])
model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
model.fit(windows(train_t), signal[train_t])
print("One-step MSE:", mean_squared_error(signal[test_t],
      model.predict(windows(test_t))))
# Rolling one-step test uses observed past values, not future samples.
