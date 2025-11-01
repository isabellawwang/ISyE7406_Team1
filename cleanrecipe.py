import pandas as pd
import numpy as np
import re
import ast

data = pd.read_csv("recipe.csv")
data = data.dropna()
data = data.drop(columns = ['directions', 'link', 'source'])

# Sample data out with 2000 rows instead of 2000000
middle = len(data)//100
df_qtr1 = data.iloc[:middle]
print(len(df_qtr1)) #2231 #22311

df_qtr1["ingredients"] = df_qtr1["ingredients"].apply(ast.literal_eval)
df_qtr1["NER"] = df_qtr1["NER"].apply(ast.literal_eval)

# 1. Indexing method?
## removing keywords from the description
def kw_remove(row):
    description = row["ingredients"]
    ingreds = row["NER"]
    removed = []
    for desc in description:
        p = desc
        for product in ingreds:
            p = re.sub(rf'\b{re.escape(product)}\b', '', p, flags = re.IGNORECASE)
        p = re.sub(r'\s*,\s*', ', ', p)
        p = re.sub(r',\s*,', ',', p)
        p = re.sub(r'\s+', ' ', p).strip()
        p = re.sub(r'^,|,$', '', p).strip()
        removed.append(p)
    return removed

df_qtr1["Description"] = df_qtr1.apply(kw_remove, axis = 1)

## map ingredient to its measurement description
df_qtr1["ner_to_ingred"] = df_qtr1.apply(lambda row: dict(zip(row["NER"], row["Description"])), axis = 1)

# 1.1 Complete breakdown with regex
df2 = df_qtr1
rows = []
for i, row in df2.iterrows():
    title = row["title"]
    for ing, des in row["ner_to_ingred"].items():
        rows.append({"title": title, "ingredient": ing, "direction": des})
df2 = pd.DataFrame(rows)

def collapse(map):
    map = str(map).lower()
    first_match = re.search(r'\(([\d\s\/\.]+)\s*(oz|ounce|ounces|lb|pound|cups?|c\.|tsp|teaspoons?|tbsp|tablespoons?)', map)
    if first_match:
        numb = first_match.group(1).strip()
        unit = first_match.group(2).replace(".", "").lower()
    else: 
        next_match = re.search(r'([\d\s\/\.]+)\s*(oz|ounce|ounces|lb|pound|cups?|c\.|tsp|teaspoons?|tbsp|tablespoons?)', map)
        if next_match:
            numb = next_match.group(1).strip()
            unit = next_match.group(2).replace(".", "").lower()
        else:
            return pd.Series({"amount": None, "unit": None})
    '''
    match = re.match(r'([\d\s\/\.]+)', map)
    measures = match.group(1).strip() if match else None
    '''
    try:
        parts = numb.split()
        if len(parts) == 2:
            value = float(parts[0])+eval(parts[1])
        else:
            value = eval(numb)
    except Exception:
        value = None
    #unit_match = re.search(r'(c\.|cup|cups|tsp\.|tbsp\.|tablespoon|teaspoon|oz\.|ounce|lb\.|pound)', map, re.IGNORECASE)
    #unit = unit_match.group(1).replace('.', '').lower() if unit_match else None

    return pd.Series({"amount": value, "unit": unit})

df2[["amount", "unit"]] = df2["direction"].apply(collapse)

'''
# 2. Regex method
def breakdown(measure, ingred_name):
    result = {}
    for num, name in zip(measure, ingred_name):
        pattern = (
            r"^([\d\s\/\.\(\)a-zA-Z-]+?)"
        r"\s*(?:of\s+)?"
        + re.escape(name)
        )
        match = re.search(pattern, num, re.IGNORECASE)
        if match:
            amt = match.group(1).strip()
            result[name] = amt
        else:
            result[name] = None
    return result

df_qtr1["ingredient_dict"] = df_qtr1.apply(
    lambda row: breakdown(row["ingredients"], row["NER"]), axis = 1
)

# Drop recipes with None in the failed Regex sequence
def isnone(dict):
    return any(v is None for v in dict.values())

df_qtr1 = df_qtr1[~df_qtr1["ingredient_dict"].apply(isnone)]
'''

#output stats and sample data

print(len(df_qtr1)) #1511 #15046
print(df_qtr1.head())

# Export CSV
df_qtr1.to_csv("cleanrecipe.csv", index = False)
df2.to_csv("recipe2.csv", index = False)