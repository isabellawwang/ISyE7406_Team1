# 🥗 Smart Meal & Grocery Optimization Web App
**Nutrition-aware recipe planning, product-level cost optimization, and minimal-store grocery routing.**

Built with **Flask**, **SciPy**, **Google Places API**, **SearchAPI.io**, and Walmart product data.

Note: To run the app, walmart dataset is needed with clusters as "merged_recipe_price_nodupl_with_clusters.csv" within the same directory as app.py

---

## 🚀 Overview

This web application helps users:

### **1️⃣ Plan a full day of meals**
- Pick recipes automatically using **Linear Programming** (LP)
- OR manually enter recipe names
- Optimize for:
  - calories
  - protein
  - fat
  - carbs
  - minimum number of recipes
- Select cheapest + nearest products for each ingredient (price + shipping distance)

### **2️⃣ Build a Smart Shopping List**
- Extract all required ingredients across recipes
- Auto-send ingredients to the store optimizer

### **3️⃣ Find the minimal set of nearby stores**
- Identify grocery stores near the user’s ZIP
- Check which items each store carries using SearchAPI.io
- Solve a greedy **set cover** optimization to choose the smallest store set
- Output uncovered items and stores carrying each item

---

## 🧠 Key Features

### ✔ Nutrition-based recipe selection (LP)
Uses **SciPy’s linear programming solver** to choose recipes that meet nutrient targets.

### ✔ Product-level cost optimization
For each ingredient:
- Filter candidate products
- Compute distance from user ZIP (Haversine)
- Rank using: `PRICE + shipping_rate * distance_km`
- Keep the dominant similarity cluster (if available)
- Pick the cheapest valid product

### ✔ Automatic ingredient generation
Outputs a clean, deduplicated ingredient list.

### ✔ Minimal store selection
Uses:
- **Google Places API v1** → find nearby supermarkets
- **SearchAPI.io** → check availability
- **Greedy set cover algorithm** → pick minimal stores covering most items

### ✔ Dual workflow
- **Nutrition LP Mode**: app chooses recipes automatically
- **Manual Recipe Mode**: user enters recipe names

---

## 📂 Repository Structure

```text
.
├── app.py                   # Main Flask application
├── linear_programming.py    # LP solver for recipe selection
├── optimize.py              # Product-level cost optimizer
├── recipes_nutrition.json   # Nutrition data for all recipes
├── templates/               # HTML templates
│   ├── base.html
│   ├── index.html           # Store optimizer
│   ├── recipe_plan.html     # Recipe planner
│   └── recipe_plan_results.html
└── __pycache__/

⚙️ Installation
1. Clone the repo

2. Create a conda environment
conda create -n mealopt python=3.10
conda activate mealopt

3. Install dependencies
pip install flask scipy pandas numpy python-dotenv requests


If SciPy fails via pip:

conda install scipy

🔑 API Keys Required

Create a .env file in the project root:

GOOGLE_API_KEY=your_google_key_here
SEARCHAPI_KEY=your_searchapi_key_here


The app uses:

Google Places API for store discovery

Google Geocoding API for ZIP → coordinates

SearchAPI.io for item availability (Walmart, Target, Google Shopping)

▶️ Running the App
python app.py


Visit:
👉 http://127.0.0.1:5000/recipe-plan


(start here!)

🛠 How It Works (Full Pipeline)
Step 1: Recipe Planning

Users choose:

ZIP code

calorie target

protein/fat/carbs

minimum number of recipes

OR enter recipe names manually.

Then the app:

Runs LP (select_recipes_scipy) — if nutrition mode is enabled

Computes the best product per ingredient (optimize_recipe_cost)

Aggregates a unique ingredient list

Step 2: Store Optimization

The recipe-planner page sends ingredients → Grocery Trip Optimizer.

Then:

Google Places finds nearby grocery stores

SearchAPI checks which store has which item

A greedy set cover algorithm picks the minimal set of stores

The app outputs:

stores to visit

store → items mapping

uncovered & unavailable items
