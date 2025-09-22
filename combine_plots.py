import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

# Zmień folder na plots_2000steps
input_folder = "plots_2000steps"
output_filename = "combined_plots.png"

# Wyszukaj wszystkie pliki PNG w folderze
image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')]
image_paths.sort()  # Sortowanie, aby zachować stałą kolejność

if not image_paths:
    print(f"Error: No .png files found in {input_folder}")
    exit()

num_images = len(image_paths)
rows = math.ceil(math.sqrt(num_images))
cols = math.ceil(num_images / rows)

# Utwórz figurę i siatkę subplotów
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

# Jeśli jest tylko jeden obraz, axes nie jest siatką, więc musimy go dostosować
if num_images == 1:
    axes = [[axes]]  # Tworzymy siatkę z jednym elementem
elif rows == 1:
    axes = [axes]

# Układaj obrazy w siatce
for i, ax in enumerate(axes.flat):
    if i < num_images:
        img = mpimg.imread(image_paths[i])
        ax.imshow(img)
        ax.set_title(os.path.basename(image_paths[i]))
        ax.axis('off')  # Ukryj osie
    else:
        ax.axis('off')  # Ukryj puste sub-wykresy

# Popraw odstępy między wykresami
plt.tight_layout()

# Zapisz wynikowy obraz
plt.savefig(output_filename, bbox_inches='tight', dpi=300)

print(f"Wszystkie wykresy ({num_images}) połączono i zapisano w pliku: {output_filename}")