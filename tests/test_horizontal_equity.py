from IPython.core.display_functions import display

from openavmkit.data import SalesUniversePair, get_hydrated_sales_from_sup
from openavmkit.horizontal_equity_study import make_clusters, mark_horizontal_equity_clusters
from openavmkit.synthetic.basic import generate_basic


def test_clusters():
	print("")
	sd = generate_basic(100)

	sup = SalesUniversePair(sd.df_sales, sd.df_universe)
	df = get_hydrated_sales_from_sup(sup)

	df = mark_horizontal_equity_clusters(df, {})
	display(df["he_id"].value_counts())