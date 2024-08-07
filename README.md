# Semantics or spelling? Probing contextual word embeddings with orthographic noise. 
Code and data related to "Semantics or spelling? Probing contextual word embeddings with orthographic noise" by Jacob A. Matthews, John R. Starr, and Marten van Schijndel.

Model-specific data is contained in the "outputs" folder, while the inputs generated from Wikitext are contained in the "inputs" folder. 

If you'd like to run our experiment with a different seed or with other models, simply modify those values (`RNG`, `MODELS`) in `__main__.py`. Note that if there is a `cleaned.csv` in "inputs" when running this script, it will not generate it again. To regenerate it (i.e. with a new seed), rename or remove this file.

In addition, there are a few useful data formatting and visualization functions in the `generate_figures.ipynb` notebook. 
