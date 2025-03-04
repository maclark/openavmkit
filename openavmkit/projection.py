import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.core.display import display

def project_trend(
    time_series: np.ndarray,
    time_index: int
):
  if len(time_series) < 2:
    return time_series[0]

  y = [i for i in range(0, len(time_series))]
  const = [1.0 for i in range(0, len(time_series))]
  x = pd.DataFrame(data={
    "slope": time_series,
    "intercept": const
  })

  model = sm.OLS(y, x, hasconst=False).fit()

  # given:
  # y = mx + b
  # y - b = mx
  # (y - b)/m = x

  # solve for x:
  return (time_index - model.params["intercept"])/model.params["slope"]
