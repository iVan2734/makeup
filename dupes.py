import pandas as pd

def get_predictions_from_url(brand, name, high_csv="high_end.csv", low_csv="low_end.csv", topn=10):
    brand = brand.lower()
    name = name.lower()
    highEnd = pd.read_csv(high_csv)
    lowEnd = pd.read_csv(low_csv)

    # Find product in high end
    category = ""
    shade = ""
    ingredients = []
    for i in range(len(highEnd)):
        currName = str(highEnd.iloc[i]["name"]).lower()
        currBrand = str(highEnd.iloc[i]["brand"]).lower()
        if currName == name and currBrand == brand:
            category = str(highEnd.iloc[i]["category"]).lower()
            ingredients = str(highEnd.iloc[i]["ingredients"]).split(", ")
            shade = str(highEnd.iloc[i]["shade"]).lower()
            break
    if not ingredients:
        return []

    matchingList = []
    for i in range(len(lowEnd)):
        currCategory = str(lowEnd.iloc[i]["category"]).lower()
        if currCategory != category:
            continue
        currIngredients = str(lowEnd.iloc[i]["ingredients"]).split(", ")
        n = len(ingredients)
        m = len(currIngredients)
        k = 0
        for x in ingredients:
            if x in currIngredients:
                k += 1
        if m > 0 and n > 0:
            matching = ((k/m+k/n)/2)*k/(k+0.5)*0.98+0.2
        else:
            matching = 0.0
        currShade = str(lowEnd.iloc[i]["shade"]).lower()
        if currShade == shade and len(currShade) > 0:
            matching += 0.1
        matching = min(matching, 0.99)
        temp = [matching, i]
        matchingList.append(temp)

    matchingList.sort(key=lambda x: x[0], reverse=True)
    topn=min(topn, len(matchingList))
    result = []
    for match, idx in matchingList[:topn]:
        row = lowEnd.iloc[idx]
        result.append({
            "id": int(row["ID"]),
            "name": row["name"],
            "brand": row["brand"],
            "category": row["category"],
            "description": row["description"],
            "ingredients": row["ingredients"],
            "price_rsd": float(row["price_rsd"]) if pd.notna(row["price_rsd"]) else 0.0,
            "shade": row["shade"],
            "match_probability": round(float(match), 4)
        })
    return result

# Only runs for manual commandline usage, not Flask imports
if __name__ == "__main__":
    def demo():
        brand = input('Brand: ').strip()
        name = input('Name: ').strip()
        results = get_predictions_from_url(brand, name)
        print(results)
    demo()
