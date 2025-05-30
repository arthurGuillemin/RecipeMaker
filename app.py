import gradio as gr
from transformers import pipeline
from PIL import Image
import torch
from torchvision import transforms

# MODELES
INGREDIENT_MODEL_ID = "stchakman/Fridge_Items_Model"
RECIPE_MODEL_ID = "flax-community/t5-recipe-generation"

# PIPELINES
ingredient_classifier = pipeline(
    "image-classification",
    model=INGREDIENT_MODEL_ID,
    device=0 if torch.cuda.is_available() else -1,
    top_k=4
)

recipe_generator = pipeline(
    "text2text-generation",
    model=RECIPE_MODEL_ID,
    device=0 if torch.cuda.is_available() else -1
)

# AUGMENTATION
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# FONCTION PRINCIPALE
def generate_recipe(image: Image.Image):
    yield "🔄 Traitement de l'image... Veuillez patienter."

    # Augmentation
    image_aug = augment(image)

    # Classification
    results = ingredient_classifier(image_aug)
    ingredients = [res["label"] for res in results]
    ingredient_str = ", ".join(ingredients)

    yield f"🥕 Ingrédients détectés : {ingredient_str}\n\n🍳 Génération de la recette..."
    prompt = f"Ingredients: {ingredient_str}. Recipe:"
    recipe = recipe_generator(prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]
    yield f"### 🥕 Ingrédients détectés :\n{ingredient_str}\n\n### 🍽️ Recette générée :\n{recipe}"

# INTERFACE
interface = gr.Interface(
    fn=generate_recipe,
    inputs=gr.Image(type="pil", label="📷 Image de vos ingrédients"),
    outputs=gr.Markdown(),
    title="🥕 Générateur de Recettes 🧑‍🍳",
    description="Dépose une image d'ingrédients pour obtenir une recette automatiquement générée à partir d'un modèle IA.",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(share=True)
