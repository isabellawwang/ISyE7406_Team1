import numpy as np
from scipy.optimize import linprog

def select_recipes_scipy(
    recipes,
    kcal_target=2200,
    mins=None,
    kcal_tol_pct=0.05,
    min_recipes=None,
    max_servings_per_recipe=1.0,  # <= 1.0 makes min_recipes exact
):
    mins = mins or {}

    names = list(recipes.keys())
    kcal = np.array([recipes[r]["kcal_total"] for r in names])
    prot = np.array([recipes[r].get("protein_total_g", 0) for r in names])
    fat  = np.array([recipes[r].get("fat_total_g", 0) for r in names])
    carb = np.array([recipes[r].get("carb_total_g", 0) for r in names])

    n = len(names)
    # Decision vars: fractions/servings of each recipe
    bounds = [(0, max_servings_per_recipe)] * n

    # Constraints
    A = []
    b = []

    tol = kcal_target * kcal_tol_pct
    # kcal lower bound → -kcal ≤ -(kcal_target - tol)  -> kcal·x ≥ kcal_target - tol
    A.append(-kcal)
    b.append(-(kcal_target - tol))
    # kcal upper bound → kcal ≤ kcal_target + tol
    A.append(kcal)
    b.append(kcal_target + tol)

    # Macro minimums
    if "protein" in mins:
        A.append(-prot)
        b.append(-mins["protein"])
    if "fat" in mins:
        A.append(-fat)
        b.append(-mins["fat"])
    if "carbs" in mins:
        A.append(-carb)
        b.append(-mins["carbs"])

    # At least N different recipes (only exact if max_servings_per_recipe <= 1.0)
    if min_recipes is not None:
        # sum(x_i) >= min_recipes  <=>  -sum(x_i) <= -min_recipes
        A.append(-np.ones(n))
        b.append(-float(min_recipes))

    # Objective: minimize total servings
    c = np.ones(n)

    A_ub = np.vstack(A)
    b_ub = np.array(b)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return {"status": res.message, "selection": {}, "totals": {}}

    x = res.x
    totals = {
        "kcal": float(kcal.dot(x)),
        "protein": float(prot.dot(x)),
        "fat": float(fat.dot(x)),
        "carbs": float(carb.dot(x)),
    }
    selection = {
        names[i]: round(x[i], 3)
        for i in range(n)
        if x[i] > 1e-6
    }
    return {"status": "Optimal", "selection": selection, "totals": totals}
