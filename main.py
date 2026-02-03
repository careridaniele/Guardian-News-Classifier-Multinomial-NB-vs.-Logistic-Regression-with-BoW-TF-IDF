from Modules.text_preprocessor import Text_Preprocessor
from Model import Model
from joblib import load

def token_extractor(token_list, lemma:bool=True):
    if lemma: return [[token[0] for token in text] for text in token_list]
    else: return [[token[1] for token in text] for text in token_list]

text = ["""
The Burger Renaissance: How an American Icon Evolved from Fast Food to Fine Dining

By Gemini

NEW YORK — It is perhaps the most recognizable silhouette in the culinary world: a seasoned patty of ground beef nestled between two halves of a soft bun, often adorned with a melting slice of cheese.

For decades, the hamburger was the undisputed king of convenience—cheap, fast, and mass-produced. But in recent years, this symbol of Americana has undergone a dramatic transformation. No longer just a drive-thru staple, the humble hamburger has graduated to the white tablecloths of high-end gastronomy, sparking a global obsession with the "perfect patty."

### From the Docks to the Drive-Thru
To understand the modern burger craze, one must look back to its murky origins. While culinary historians debate the exact moment of invention, the lineage traces back to the "Hamburg steak" brought to America by German immigrants in the 19th century.

However, it was the rapid industrialization of the United States that truly forged the burger. By the mid-20th century, pioneers like McDonald’s and White Castle had streamlined the cooking process, turning the burger into a standardized product of efficiency. It became the fuel of the post-war economic boom—predictable, affordable, and distinctly American.

### The Gourmet Turn
The script flipped in the early 2000s. Chefs began to ask a simple question: What happens if we treat a burger like a steak?

The result was the "gourmet burger" movement. Suddenly, mystery meat was replaced with custom blends of Brisket, Short Rib, and Chuck. The flimsy sesame seed bun was ousted by buttery Brioche or Potato rolls.

"The burger is the ultimate blank canvas," says Marcus Vance, a food critic based in Chicago. "You can keep it simple with just onions and cheese, or you can elevate it with foie gras, truffle aioli, and aged cheddar. It bridges the gap between high-brow and low-brow dining."

Today, the trend has bifurcated. On one side, there are the maximalist creations topping $25 in bistros. On the other, there is the resurgence of the "Smash Burger"—a technique where the beef is smashed flat on a searing hot griddle to create a crispy, caramelized crust (the Maillard reaction) that focuses purely on texture and beefy flavor.

### The Plant-Based Future
Perhaps the most significant shift in the burger’s history is the one currently unfolding: the removal of the cow.

With environmental concerns rising, companies like Beyond Meat and Impossible Foods have utilized food science to engineer plant-based patties that bleed, sizzle, and taste remarkably like beef. These aren't the dry veggie pucks of the 1990s; they are designed to satisfy carnivores.

Whether made of Wagyu beef or pea protein, the hamburger remains resilient. It has survived health crazes, economic downturns, and changing trends to remain the world’s favorite comfort food. As it continues to evolve, one thing is certain: the burger is here to stay.
"""]

preprocessor = Text_Preprocessor(text, model="en_core_web_sm")
preprocessor.process()
token_list = preprocessor.get_token()

token = token_extractor(token_list)
lemma = token_extractor(token_list)

model = load("Model/MNB-TFIDF-TOKEN-60000.joblib")
print(model.use(token))


