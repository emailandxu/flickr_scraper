#%%
from pandas import DataFrame, read_json
import shutil
from pathlib import Path

#%%
df = read_json("contains_human.txt")
# %%

should_reserve = df[df["reserve"]]
# %%
len(should_reserve)
# %%
should_reserve.iloc[0]
# %%
should_reserve.path.apply( lambda path: shutil.copy(path,  "flickr-10k-valid/" + Path(path).name) )
# %%
