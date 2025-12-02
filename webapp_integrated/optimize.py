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
):
    """
    1) Use LP to choose recipes and servings (nutrition + diversity).
    2) For each chosen recipe, call your shipping-aware optimize_recipe_cost.
    """
    mins = mins or {}

    # --- Stage 1: LP to pick recipes ---
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

    selection = lp_res["selection"]        # {recipe_name: servings}
    nutrition_totals = lp_res["totals"]

    # --- Stage 2: per-recipe cost optimization (with distance/shipping) ---
    user_lat = location_info["lat"]
    user_lon = location_info["lng"]

    daily_cost = 0.0
    recipe_cost_details = {}

    for recipe_name, servings in selection.items():
        result_df, total_cost_one_serving = optimize_recipe_cost(
            recipe_name,
            df_prices,
            user_lat,
            user_lon,
            shipping_rate=shipping_rate,
        )

        if result_df is None:
            continue

        total_cost_for_servings = total_cost_one_serving * servings
        daily_cost += total_cost_for_servings

        recipe_cost_details[recipe_name] = {
            "servings": servings,
            "cost_per_serving_with_shipping": total_cost_one_serving,
            "total_cost": total_cost_for_servings,
            "ingredients_table": result_df,
        }

    return {
        "status": "Optimal",
        "selection": selection,
        "nutrition_totals": nutrition_totals,
        "daily_cost": daily_cost,
        "recipe_cost_details": recipe_cost_details,
        "location": location_info,
    }
