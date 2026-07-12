from langchain_core.tools import tool


# --- Mock Food Database ---
FOOD_DB = {
    "tofu": {"calories": 76, "protein": 8, "carbs": 2, "fat": 5, "vitamins": ["B1", "Iron", "Calcium"]},
    "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "vitamins": ["B6", "Niacin"]},
    "lentils": {"calories": 116, "protein": 9, "carbs": 20, "fat": 0.4, "vitamins": ["Folate", "Iron"]},
    "broccoli": {"calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4, "vitamins": ["C", "K"]},
}


@tool
def lookup_food_db(food_name: str):
    """Look up the exact macro and vitamin profile for a specific food."""
    food_name = food_name.lower().strip()
    if food_name in FOOD_DB:
        data = FOOD_DB
        return f"✅ {food_name.capitalize()} profile: {data}"
    return f"❌ '{food_name}' not found. Try: tofu, chicken breast, lentils, or broccoli."


@tool
def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str):
    """Calculate Basal Metabolic Rate (BMR) using the Mifflin-St Jeor equation."""
    if gender.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    return f"🫀 Your estimated BMR is **{int(bmr)} kcal/day**."


@tool
def generate_meal_plan(target_calories: int, restrictions: str = ""):
    """
    Generate a simple meal plan based on calorie target and dietary restrictions.
    """
    meals = {
        "breakfast": "Oatmeal with berries and walnuts",
        "lunch": "Grilled chicken salad with olive oil",
        "dinner": "Baked salmon with quinoa and asparagus"
    }

    if "vegetarian" in restrictions.lower():
        meals["lunch"] = "Hearty lentil soup with sourdough"
        meals["dinner"] = "Tofu stir-fry with brown rice"
        meals["dinner"] = "Chickpea coconut curry with rice"

    plan = f"🍽️ Sample {target_calories} kcal plan ({restrictions if restrictions else 'Standard'}):\n"
    for meal, food in meals.items():
        plan += f"  • {meal.capitalize()}: {food}\n"
    plan += "*(Adjust portions to hit your exact macros.)*"
    return plan

# Export all tools (rag_retrieve will be added in graph.py)
nutritional_tools = [lookup_food_db, calculate_bmr, generate_meal_plan]

