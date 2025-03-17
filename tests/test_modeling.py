import os

from openavmkit.benchmark import run_models, MultiModelResults
from openavmkit.cleaning import clean_valid_sales
from openavmkit.data import enrich_time, _perform_canonical_split, get_hydrated_sales_from_sup, SalesUniversePair
from openavmkit.horizontal_equity_study import mark_horizontal_equity_clusters, \
	mark_horizontal_equity_clusters_per_model_group_sup
from openavmkit.modeling import _run_lars_sqft, DataSplit, run_lars
from openavmkit.ratio_study import run_and_write_ratio_study_breakdowns
from openavmkit.sales_scrutiny_study import SalesScrutinyStudy, run_sales_scrutiny
from openavmkit.synthetic.basic import generate_basic
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.settings import get_valuation_date, load_settings

from IPython.core.display import display

def test_run_lars_sqft():

	# Generate basic synthetic data
	data = generate_basic(100)
	data.df_universe.to_csv("data/zz-synthetic/universe.csv", index=False)
	data.df_sales.to_csv("data/zz-synthetic/sales.csv", index=False)
	sup = SalesUniversePair(data.df_sales, data.df_universe)

	df_universe = sup.universe
	df_sales = get_hydrated_sales_from_sup(sup)

	settings = {
		"analysis":{
			"impr_equity":{
				"fields_categorical":["bldg_type"],
				"fields_numeric":[
					"bldg_area_finished_sqft",
					"bldg_quality_num",
					"bldg_age_years"
				]
			}
		}
	}

	settings = load_settings("", settings)

	df_universe["model_group"] = "single_family"
	df_sales["model_group"] = "single_family"

	sup = SalesUniversePair(df_sales, df_universe)
	sup =	mark_horizontal_equity_clusters_per_model_group_sup(sup, settings, verbose=True)

	df_sales = get_hydrated_sales_from_sup(sup)
	df_universe = sup.universe

	# split df_sales into training and test sets
	df_test, df_train = _perform_canonical_split("single_family", df_sales, settings)
	test_keys = df_test["key"]
	train_keys = df_test["key"]

	locations = ["quadrant", "neighborhood"]

	ds = DataSplit(
		df_sales,
		df_universe,
		model_group="single_family",
		settings={},
		dep_var="sale_price",
		dep_var_test="sale_price",
		ind_vars=["bldg_area_finished_sqft", "land_area_sqft", "quadrant", "neighborhood"],
		categorical_vars=locations,
		interactions={},
		test_keys=test_keys,
		train_keys=train_keys
	)

	#_run_lars_sqft(ds, ["neighborhood"], verbose=True)
	run_lars(ds, ["neighborhood"], verbose=True)


def test_guilford_sales_scrutiny():
	# TODO: update this
	return True


def test_models_guilford():
	# TODO: update this
	return True


def test_models_synthetic():
	# TODO: update this
	return True
	# print("")
	# # set working directory to the library's root/tests/data:
	# os.chdir("data/zz-synthetic")
	#
	# # load the settings:
	# settings = load_settings()
	#
	# # load the data
	# sd = generate_basic(100)
	# df = sd.df
	#
	# df["assr_market_value"] = df["total_value"]
	# df["model_group"] = "residential_sf"
	#
	# # calculate horizontal equity cluster ID's
	# df = mark_horizontal_equity_clusters(df, settings)
	#
	# # run the predictive models
	# results = run_models(
	# 	df,
	# 	settings,
	# 	verbose=True,
	# 	save_params=True,
	# 	use_saved_params=True
	# )
	#
	# print(results.benchmark.print())
	# return True