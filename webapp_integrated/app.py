import os
import json
from collections import defaultdict

import requests
import pandas as pd
from flask import Flask, render_template, request
from dotenv import load_dotenv

from linear_programming import select_recipes_scipy  # your LP function

app = Flask(__name__)

# --- ENV / KEYS ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCHAPI_KEY = os.getenv("SEARCHAPI_KEY")

print("DEBUG GOOGLE_API_KEY set:", bool(GOOGLE_API_KEY))
print("DEBUG SEARCHAPI_KEY set:", bool(SEARCHAPI_KEY))

# --- DEFAULT LOCATION (fallback if geocoding fails) ---
DEFAULT_LOCATION = {
    "zip": "30332",
    "city": "Atlanta",
    "state": "GA",
    "lat": 33.7756,
    "lng": -84.3963,
}

# --- LOAD RECIPES NUTRITION DICT ---
try:
    # best case: you have a Python module with the dict
    from recipe_nutrition_data import recipes_nutrition
except ImportError:
    # fallback: load from JSON file
    with open("recipes_nutrition.json") as f:
        recipes_nutrition = json.load(f)

# --- LOAD MERGED PRICE / PRODUCT DATAFRAME ---
df = pd.read_csv("merged_recipe_price_nodupl_with_clusters.csv")

# Ensure PRICE_USED exists and is usable
if "PRICE_USED" not in df.columns:
    df["PRICE_USED"] = df.get("PRICE_CURRENT").fillna(df.get("PRICE_RETAIL"))
else:
    df["PRICE_USED"] = (
        df["PRICE_USED"]
        .fillna(df.get("PRICE_CURRENT"))
        .fillna(df.get("PRICE_RETAIL"))
    )

# Ensure match_score exists
if "match_score" not in df.columns:
    df["match_score"] = pd.Series([None] * len(df))
df["match_score"] = df["match_score"].fillna(df["match_score"].median())


# Optional: normalize item names for better search matches
NORMALIZATION_MAP = {
    "egg": "eggs",
    "banana": "bananas",
    "tomato": "tomatoes",
    "potato": "potatoes",
}

def normalize_query_item(item: str) -> str:
    item = item.strip().lower()
    return NORMALIZATION_MAP.get(item, item)

# ----------------------------------------------------------------------
# (OPTIONAL) ZIP → COORDINATES RESOLVER (placeholder)
# ----------------------------------------------------------------------
def resolve_zip_to_location(zip_code: str):
    zip_code = (zip_code or "").strip()
    if not zip_code or not GOOGLE_API_KEY:
        return DEFAULT_LOCATION

    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": zip_code, "key": GOOGLE_API_KEY},
            timeout=5,
        )
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            return DEFAULT_LOCATION

        result = data["results"][0]
        loc = result["geometry"]["location"]
        city = ""
        state = ""
        for comp in result.get("address_components", []):
            if "locality" in comp.get("types", []):
                city = comp.get("long_name", "")
            if "administrative_area_level_1" in comp.get("types", []):
                state = comp.get("short_name", "")

        return {
            "zip": zip_code,
            "city": city or "Area",
            "state": state,
            "lat": loc["lat"],
            "lng": loc["lng"],
        }
    except Exception:
        return DEFAULT_LOCATION
# ----------------------------------------------------------------------
# GOOGLE PLACES API: NEARBY GROCERY STORES
# ----------------------------------------------------------------------

def get_nearby_grocery_stores(location_info, radius_m=8000, max_results=20):
    """
    Use Google Places 'searchNearby' to find supermarkets/grocery stores
    near the given lat/lng.

    Returns: list of dicts with at least {name, place_id, address}
    """
    if not GOOGLE_API_KEY:
        # No key: nothing real we can do
        return []

    lat = location_info["lat"]
    lng = location_info["lng"]

    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        # Ask for only the fields we need (saves quota)
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.types,places.id",
    }
    body = {
        # "grocery_or_supermarket" is not valid in Places v1
        "includedTypes": ["supermarket", "grocery_store"],
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius_m,
            }
        },
    }

    try:
        print("DEBUG calling Places with lat/lng:", lat, lng)
        resp = requests.post(url, headers=headers, json=body, timeout=5)
        print("DEBUG Places status:", resp.status_code, resp.text[:300]) 
        resp.raise_for_status()
        data = resp.json()
        places = []
        for p in data.get("places", []):
            places.append(
                {
                    "name": p["displayName"]["text"],
                    "place_id": p["id"],
                    "address": p.get("formattedAddress", ""),
                    "types": p.get("types", []),
                }
            )
        return places
    except Exception as e:
        print("DEBUG Places error:", repr(e))        # Fail silently, just return empty list
        return []

# from functools import lru_cache

def search_retailer_item(engine: str, query: str, zip_code: str) -> bool:
    """
    Use SearchAPI.io to check if `engine` (walmart/target/google_shopping/etc.)
    returns ANY results for `query` in/around `zip_code`.

    Returns True if at least one product was found, else False.
    """
    if not SEARCHAPI_KEY:
        return False

    engine = engine.lower()
    query = normalize_query_item(query)
    zip_code = zip_code.strip()

    if not query or not zip_code:
        return False

    params = {
        "engine": engine,           # "walmart", "target", or "google_shopping"
        "q": query,
        "api_key": SEARCHAPI_KEY,
        "location": zip_code,       # SearchAPI supports location-based searches
    }

    try:
        resp = requests.get("https://www.searchapi.io/api/v1/search", params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return False

    # Heuristic: if any known result array is non-empty, consider the item available
    if isinstance(data, dict):
        for key in ("products", "items", "organic_results", "shopping_results"):
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                return True

    return False

# ----------------------------------------------------------------------
# GREEDY SET-COVER: FEWEST STORES TO COVER MOST ITEMS
# ----------------------------------------------------------------------

def choose_min_stores(stores_inventory, requested_items):
    """
    Greedy set-cover approximation.

    Args:
      stores_inventory: dict[store_name -> set(items)]
      requested_items: iterable of lowercased item strings

    Returns:
      chosen_stores: list of store names
      store_to_items: dict[store_name -> set(items covered by that store)]
      uncovered_items: set(items not covered by chosen stores)
      truly_unavailable: set(items not present in ANY store inventory)
    """
    requested_items = {i.strip().lower() for i in requested_items if i.strip()}
    requested_items = {i for i in requested_items if i}

    covered_somewhere = set()
    for inv in stores_inventory.values():
        covered_somewhere.update(inv)

    uncovered_items = requested_items.copy()
    chosen_stores = []
    store_to_items = defaultdict(set)

    while uncovered_items:
        best_store = None
        best_covers = set()

        for store, inv_items in stores_inventory.items():
            covers = inv_items & uncovered_items
            if len(covers) > len(best_covers):
                best_covers = covers
                best_store = store

        if best_store is None or not best_covers:
            # No store covers any remaining item
            break

        chosen_stores.append(best_store)
        store_to_items[best_store] = best_covers
        uncovered_items -= best_covers

    truly_unavailable = requested_items - covered_somewhere

    return chosen_stores, store_to_items, uncovered_items, truly_unavailable

def fetch_inventory_for_store(store, items, zip_code: str, cache: dict) -> set:
    """
    Uses SearchAPI.io to check which of `items` are available for this store.

    - Walmart  -> engine="walmart"
    - Target   -> engine="target"
    - Others   -> engine="google_shopping" (generic grocery search)

    store: dict with keys ['name', 'place_id', 'address', 'types']
    items: list of item strings (already lowercased)
    zip_code: user zip (string)
    cache: dict used to memoize (engine, zip, item) -> bool

    Returns:
        set of items available at this store.
    """
    name = store["name"].lower()
    available = set()

    # Decide which SearchAPI engine to use
    if "walmart" in name:
        engine = "walmart"
    elif "target" in name:
        engine = "target"
    else:
        # Fallback for Publix, Kroger, Trader Joe's, Sprouts, Whole Foods, etc.
        engine = "google_shopping"

    for item in items:
        key = (engine, zip_code, item)
        if key in cache:
            has_item = cache[key]
        else:
            has_item = search_retailer_item(engine, item, zip_code)
            cache[key] = has_item

        if has_item:
            available.add(item)

    return available


def get_stores_and_inventory(items, location_info, max_stores=5):
    """
    1. Use Google Places to get up to `max_stores` nearby grocery stores.
    2. For each store, use SearchAPI.io (via fetch_inventory_for_store)
       to determine which items this store has.
    """
    nearby = get_nearby_grocery_stores(location_info, max_results=max_stores)
    if not nearby:
        return {}

    items = [i.strip().lower() for i in items if i.strip()]
    stores_inventory = {}
    cache = {}  # (engine, zip, item) -> bool
    zip_code = location_info.get("zip", "").strip()

    for store in nearby:
        available_here = fetch_inventory_for_store(store, items, zip_code, cache)

        if not available_here:
            continue

        key = f"{store['name']} – {store['address']}"
        stores_inventory[key] = available_here

    return stores_inventory

import numpy as np

def calculate_distance_vectorized(lat1, lon1, lat2_series, lon2_series):
    R = 6371.0  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_series)
    dphi = np.radians(lat2_series - lat1)
    dlambda = np.radians(lon2_series - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def optimize_recipe_cost(
    recipe_name,
    df_prices,
    user_lat,
    user_lon,
    shipping_rate=0.5,
    min_match_score=80.0,
    lat_diff=5.0,
    lon_diff=5.0,
):
    mask = df_prices["title"].str.lower() == str(recipe_name).lower()
    recipe_df = df_prices[mask].copy()
    if recipe_df.empty:
        print(f"No recipe found in df for: {recipe_name}")
        return None, 0.0

    # match_score filter
    if "match_score" in recipe_df.columns:
        recipe_df = recipe_df[recipe_df["match_score"] >= float(min_match_score)].copy()
        if recipe_df.empty:
            print(f"No usable matches after match_score filter for '{recipe_name}'.")
            return None, 0.0

    # lat/lon window + distances
    if "latitude" in recipe_df.columns and "longitude" in recipe_df.columns:
        recipe_df = recipe_df[
            (recipe_df["latitude"] > user_lat - lat_diff)
            & (recipe_df["latitude"] < user_lat + lat_diff)
            & (recipe_df["longitude"] > user_lon - lon_diff)
            & (recipe_df["longitude"] < user_lon + lon_diff)
        ].copy()
        if recipe_df.empty:
            print(f"No relevant products near user for '{recipe_name}'.")
            return None, 0.0

        recipe_df["distance_km"] = calculate_distance_vectorized(
            user_lat,
            user_lon,
            recipe_df["latitude"],
            recipe_df["longitude"],
        )
    else:
        recipe_df["distance_km"] = 0.0

    recipe_df["distance_km"] = recipe_df["distance_km"].fillna(9999.0)

    if "PRICE_USED" not in recipe_df.columns:
        raise ValueError("PRICE_USED column missing in df_prices.")
    recipe_df["PRICE_USED"] = pd.to_numeric(recipe_df["PRICE_USED"], errors="coerce")
    recipe_df = recipe_df.dropna(subset=["PRICE_USED"]).copy()
    if recipe_df.empty:
        print(f"No valid PRICE_USED rows for '{recipe_name}'.")
        return None, 0.0

    # Ranking cost = price + per-item shipping approximation
    recipe_df["sort_cost"] = recipe_df["PRICE_USED"] + recipe_df["distance_km"] * float(shipping_rate)

    if "ingredient" not in recipe_df.columns:
        raise ValueError("ingredient column is required in df for per-ingredient selection.")

    chosen_rows = []
    for ing, group in recipe_df.groupby("ingredient"):
        g = group.copy()
        if "cluster" in g.columns and g["cluster"].nunique() > 1:
            dominant_cluster = g["cluster"].value_counts().idxmax()
            g = g[g["cluster"] == dominant_cluster].copy()

        best_idx = g["sort_cost"].idxmin()
        chosen_rows.append(g.loc[best_idx])

    if not chosen_rows:
        print(f"No ingredient-level choices for '{recipe_name}'.")
        return None, 0.0

    result = pd.DataFrame(chosen_rows)

    # product price
    product_total_price = float(result["PRICE_USED"].sum())

    # shipping per unique store
    if "SHIPPING_LOCATION" in result.columns:
        unique_stores = (
            result[["SHIPPING_LOCATION", "distance_km"]]
            .drop_duplicates(subset=["SHIPPING_LOCATION"])
            .copy()
        )
    else:
        unique_stores = result[["distance_km"]].copy()
        unique_stores["SHIPPING_LOCATION"] = "Unknown"
        unique_stores = unique_stores.drop_duplicates()

    basket_shipping_cost = float(
        (unique_stores["distance_km"] * float(shipping_rate)).sum()
    )

    final_total_cost = product_total_price + basket_shipping_cost

    cols_to_show = [
        "ingredient",
        "PRODUCT_NAME",
        "BRAND",
        "PRICE_USED",
        "distance_km",
        "cluster",
        "SHIPPING_LOCATION",
    ]
    cols_to_show = [c for c in cols_to_show if c in result.columns]
    result_display = result[cols_to_show].rename(columns={"PRICE_USED": "Product_Price"})

    print(f"--- Optimization for '{recipe_name}' ---")
    print(f"Items Count: {len(result_display)}")
    print(f"Product Cost:   ${product_total_price:.2f}")
    print(f"Shipping Cost:  ${basket_shipping_cost:.2f}")
    print(f"TOTAL COST:     ${final_total_cost:.2f}")

    return result_display, final_total_cost

# ----------------------------------------------------------------------
# FLASK ROUTES
# ----------------------------------------------------------------------

def plan_day_with_cost_and_shipping(
    recipes_nutrition,
    df_prices,
    location_info,
    kcal_target=2200,
    mins=None,
    min_recipes=4,
    kcal_tol_pct=0.05,
    max_servings_per_recipe=1.0,
    shipping_rate=0.5,
    min_match_score=80.0,
):
    mins = mins or {}

    lp_res = select_recipes_scipy(
        recipes_nutrition,
        kcal_target=kcal_target,
        mins=mins,
        kcal_tol_pct=kcal_tol_pct,
        min_recipes=min_recipes,
        max_servings_per_recipe=max_servings_per_recipe,
    )

    if lp_res["status"] != "Optimal":
        return {"status": lp_res["status"]}

    selection = lp_res["selection"]
    nutrition_totals = lp_res["totals"]

    user_lat = location_info["lat"]
    user_lon = location_info["lng"]

    daily_cost = 0.0
    recipe_cost_details = {}
    all_ingredients = []

    for recipe_name, servings in selection.items():
        result_df, cost_one_serving = optimize_recipe_cost(
            recipe_name,
            df_prices,
            user_lat,
            user_lon,
            shipping_rate=shipping_rate,
            min_match_score=min_match_score,
        )
        if result_df is None:
            continue

        total_cost_for_servings = cost_one_serving * servings
        daily_cost += total_cost_for_servings

        recipe_cost_details[recipe_name] = {
            "servings": servings,
            "cost_per_serving_with_shipping": cost_one_serving,
            "total_cost": total_cost_for_servings,
            "ingredients_table": result_df,
        }

        if "ingredient" in result_df.columns:
            all_ingredients.extend(result_df["ingredient"].tolist())

    # Deduplicate + sort ingredients for passing into Grocery Optimizer
    ingredients_unique = sorted({i.strip().lower() for i in all_ingredients if i and isinstance(i, str)})
    ingredients_text = "\n".join(ingredients_unique)

    return {
        "status": "Optimal",
        "selection": selection,
        "nutrition_totals": nutrition_totals,
        "daily_cost": daily_cost,
        "recipe_cost_details": recipe_cost_details,
        "location": location_info,
        "ingredients_list": ingredients_unique,
        "ingredients_text": ingredients_text,
    }

@app.route("/recipe-plan", methods=["GET", "POST"])
def recipe_plan():
    if request.method == "POST":
        zip_code = request.form.get("zip_code", "").strip()
        use_nutrition_lp = request.form.get("use_nutrition_lp") == "1"

        manual_recipes_raw = request.form.get("manual_recipes", "")

        # Always need location
        location_info = resolve_zip_to_location(zip_code)

        if use_nutrition_lp:
            # ---- MODE A: Nutrition-based LP (existing behavior) ----
            kcal_raw = request.form.get("kcal_target", "2200")
            protein_raw = request.form.get("protein_min", "100")
            fat_raw = request.form.get("fat_min", "50")
            carbs_raw = request.form.get("carbs_min", "200")
            min_recipes_raw = request.form.get("min_recipes", "4")

            try:
                kcal_target = float(kcal_raw)
                protein_min = float(protein_raw)
                fat_min = float(fat_raw)
                carbs_min = float(carbs_raw)
                min_recipes = int(min_recipes_raw)
            except ValueError:
                error = "Please enter numeric values for calories, macros, and min recipes."
                return render_template(
                    "recipe_plan.html",
                    error=error,
                    prev_zip=zip_code,
                    prev_kcal=kcal_raw,
                    prev_protein=protein_raw,
                    prev_fat=fat_raw,
                    prev_carbs=carbs_raw,
                    prev_min_recipes=min_recipes_raw,
                    prev_manual_recipes=manual_recipes_raw,
                )

            mins = {"protein": protein_min, "fat": fat_min, "carbs": carbs_min}

            plan_res = plan_day_with_cost_and_shipping(
                recipes_nutrition=recipes_nutrition,
                df_prices=df,
                location_info=location_info,
                kcal_target=kcal_target,
                mins=mins,
                min_recipes=min_recipes,
            )

        else:
            # ---- MODE B: manual recipe selection (no LP) ----
            # Parse manual recipes
            recipe_names = [
                line.strip()
                for line in manual_recipes_raw.splitlines()
                if line.strip()
            ]

            if not recipe_names:
                error = "Please enter at least one recipe name when nutrition LP is disabled."
                return render_template(
                    "recipe_plan.html",
                    error=error,
                    prev_zip=zip_code,
                    prev_manual_recipes=manual_recipes_raw,
                )

            # Build a fake 'selection' dict: 1 serving each
            selection = {name: 1.0 for name in recipe_names}

            # We'll reuse the cost + ingredient logic from plan_day_with_cost_and_shipping,
            # but bypass the LP and nutrition totals.
            user_lat = location_info["lat"]
            user_lon = location_info["lng"]

            daily_cost = 0.0
            recipe_cost_details = {}
            all_ingredients = []

            for recipe_name, servings in selection.items():
                result_df, cost_one_serving = optimize_recipe_cost(
                    recipe_name,
                    df,
                    user_lat,
                    user_lon,
                    shipping_rate=0.5,
                    min_match_score=80.0,
                )
                if result_df is None:
                    continue

                total_cost_for_servings = cost_one_serving * servings
                daily_cost += total_cost_for_servings

                recipe_cost_details[recipe_name] = {
                    "servings": servings,
                    "cost_per_serving_with_shipping": cost_one_serving,
                    "total_cost": total_cost_for_servings,
                    "ingredients_table": result_df,
                }

                if "ingredient" in result_df.columns:
                    all_ingredients.extend(result_df["ingredient"].tolist())

            ingredients_unique = sorted(
                {i.strip().lower() for i in all_ingredients if i and isinstance(i, str)}
            )
            ingredients_text = "\n".join(ingredients_unique)

            # Fake nutrition totals (we're not using LP)
            nutrition_totals = {
                "kcal": float("nan"),
                "protein": float("nan"),
                "fat": float("nan"),
                "carbs": float("nan"),
            }

            plan_res = {
                "status": "Manual",
                "selection": selection,
                "nutrition_totals": nutrition_totals,
                "daily_cost": daily_cost,
                "recipe_cost_details": recipe_cost_details,
                "location": location_info,
                "ingredients_list": ingredients_unique,
                "ingredients_text": ingredients_text,
            }

        # In both modes, if we reach here we have a plan_res dict
        if not plan_res.get("recipe_cost_details"):
            error = "No recipes could be cost-optimized. Check recipe names or match settings."
            return render_template(
                "recipe_plan.html",
                error=error,
                prev_zip=zip_code,
                prev_manual_recipes=manual_recipes_raw,
            )

        return render_template(
            "recipe_plan_results.html",
            plan=plan_res,
            zip_code=location_info["zip"],
        )

    # GET
    return render_template(
        "recipe_plan.html",
        prev_zip="30332",
        prev_kcal="2200",
        prev_protein="100",
        prev_fat="50",
        prev_carbs="200",
        prev_min_recipes="4",
        prev_manual_recipes="",
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        zip_code = request.form.get("zip_code", "").strip()
        raw_items = request.form.get("items", "")
        max_stores_raw = request.form.get("max_stores", "5")

        # Parse items
        items = [line.strip().lower() for line in raw_items.splitlines() if line.strip()]

        # Parse max_stores safely (default 5, clamp 1–30)
        try:
            max_stores = int(max_stores_raw)
            if max_stores < 1:
                max_stores = 1
            if max_stores > 30:
                max_stores = 30
        except ValueError:
            max_stores = 5

        if not items:
            error = "Please enter at least one grocery item."
            return render_template(
                "index.html",
                error=error,
                prev_zip=zip_code,
                prev_items=raw_items,
                prev_max_stores=max_stores,
            )

        # Resolve ZIP -> lat/lng (or default)
        location_info = resolve_zip_to_location(zip_code)

        # Get store inventories near that ZIP (now using real APIs in your helper)
        stores_inventory = get_stores_and_inventory(items, location_info, max_stores=max_stores)

        if not stores_inventory:
            error = (
                "No stores with matching items were found near this PIN/ZIP, "
                "or store data is unavailable right now."
            )
            return render_template(
                "index.html",
                error=error,
                prev_zip=zip_code,
                prev_items=raw_items,
                prev_max_stores=max_stores,
            )

        # Choose minimal stores
        chosen_stores, store_to_items, uncovered_items, truly_unavailable = choose_min_stores(
            stores_inventory, items
        )

        return render_template(
            "results.html",
            zip_code=location_info["zip"],
            location_info=location_info,
            items=items,
            chosen_stores=chosen_stores,
            store_to_items=store_to_items,
            uncovered_items=uncovered_items,
            truly_unavailable=truly_unavailable,
            max_stores=max_stores,
            total_candidate_stores=len(stores_inventory),
        )

    # GET
    return render_template("index.html")

if __name__ == "__main__":
    # Run from a terminal so you can see any errors printed
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True)
