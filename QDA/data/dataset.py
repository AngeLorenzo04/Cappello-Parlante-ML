import numpy as np

# --- DATASET REALE DI HOGWARTS ---
import numpy as np

# --- DATASET DI HOGWARTS AMPLIATO (200 PERSONAGGI) ---
data_hogwarts = [
    # [Nome, Coraggio, Ambizione, Casa (0:G, 1:S, 2:H)]
    
    # --- GRIFONDORO (67 Personaggi) ---
    ["Harry Potter", 98, 45, 0], ["Ron Weasley", 85, 25, 0], ["Hermione Granger", 88, 55, 0],
    ["Neville Paciock", 92, 15, 0], ["Albus Silente", 99, 60, 0], ["Minerva McGonagall", 94, 30, 0],
    ["Sirius Black", 96, 35, 0], ["Remus Lupin", 90, 20, 0], ["Ginny Weasley", 89, 40, 0],
    ["Fred Weasley", 87, 45, 0], ["George Weasley", 87, 45, 0], ["Lily Potter", 95, 20, 0],
    ["James Potter", 96, 40, 0], ["Rubeus Hagrid", 97, 10, 0], ["Molly Weasley", 93, 15, 0],
    ["Arthur Weasley", 85, 10, 0], ["Bill Weasley", 88, 30, 0], ["Charlie Weasley", 90, 20, 0],
    ["Seamus Finnigan", 75, 30, 0], ["Dean Thomas", 80, 25, 0], ["Lavender Brown", 72, 35, 0],
    ["Parvati Patil", 70, 40, 0], ["Colin Creevey", 88, 10, 0], ["Alicia Spinnet", 82, 35, 0],
    ["Katie Bell", 81, 30, 0], ["Angelina Johnson", 86, 50, 0], ["Oliver Wood", 88, 65, 0],
    ["Lee Jordan", 78, 45, 0], ["Godric Grifondoro", 100, 50, 0], ["Frank Longbottom", 95, 20, 0],
    ["Alice Longbottom", 94, 25, 0], ["Augusta Longbottom", 90, 45, 0], ["Cormac McLaggen", 75, 75, 0],
    ["Romilda Vane", 65, 60, 0], ["Demelza Robins", 78, 20, 0], ["Ritchie Coote", 74, 15, 0],
    ["Jimmy Peakes", 76, 10, 0], ["Euan Abercrombie", 70, 20, 0], ["Aberforth Silente", 92, 15, 0],
    ["Nicholas de Mimsy-Porpington", 70, 55, 0], ["Dobby (Onorario)", 98, 5, 0], ["Fawkes (Onorario)", 100, 0, 0],
    ["Andromeda Tonks", 85, 40, 0], ["Ted Lupin", 82, 25, 0], ["Percy Weasley", 70, 90, 0], # Outlier coraggioso ma ambizioso
    ["Grif S. 01", 85, 30, 0], ["Grif S. 02", 78, 22, 0], ["Grif S. 03", 91, 18, 0], ["Grif S. 04", 88, 42, 0],
    ["Grif S. 05", 82, 28, 0], ["Grif S. 06", 95, 35, 0], ["Grif S. 07", 79, 15, 0], ["Grif S. 08", 84, 48, 0],
    ["Grif S. 09", 87, 33, 0], ["Grif S. 10", 93, 25, 0], ["Grif S. 11", 80, 50, 0], ["Grif S. 12", 74, 20, 0],
    ["Grif S. 13", 89, 38, 0], ["Grif S. 14", 96, 44, 0], ["Grif S. 15", 72, 30, 0], ["Grif S. 16", 81, 25, 0],
    ["Grif S. 17", 85, 45, 0], ["Grif S. 18", 90, 10, 0], ["Grif S. 19", 77, 15, 0], ["Grif S. 20", 83, 35, 0],
    ["Grif S. 21", 98, 20, 0], ["Grif S. 22", 86, 40, 0],

    # --- SERPEVERDE (67 Personaggi) ---
    ["Lord Voldemort", 50, 100, 1], ["Severus Piton", 90, 85, 1], ["Draco Malfoy", 35, 92, 1],
    ["Bellatrix Lestrange", 80, 95, 1], ["Lucius Malfoy", 30, 90, 1], ["Narcissa Malfoy", 40, 85, 1],
    ["Horace Lumacorno", 45, 80, 1], ["Regulus Black", 85, 75, 1], ["Pansy Parkinson", 25, 80, 1],
    ["Blaise Zabini", 30, 85, 1], ["Dolores Umbridge", 10, 98, 1], ["Salazar Serpeverde", 60, 100, 1],
    ["Rodolphus Lestrange", 50, 90, 1], ["Vincent Crabbe", 20, 70, 1], ["Gregory Goyle", 20, 70, 1],
    ["Marcus Flint", 40, 80, 1], ["Terence Higgs", 35, 75, 1], ["Adrian Pucey", 40, 70, 1],
    ["Millicent Bulstrode", 30, 65, 1], ["Phineas Nigellus Black", 20, 90, 1], ["Tom Riddle Sr", 10, 60, 1],
    ["Daphne Greengrass", 45, 75, 1], ["Astoria Greengrass", 50, 70, 1], ["Theodore Nott", 40, 88, 1],
    ["Tracey Davis", 35, 70, 1], ["Gilderoy Lockhart", 30, 95, 1], ["Scabior", 60, 70, 1],
    ["Fenrir Greyback", 85, 80, 1], ["Corban Yaxley", 55, 90, 1], ["Antonin Dolohov", 80, 85, 1],
    ["Thorfinn Rowle", 75, 80, 1], ["Alecto Carrow", 40, 90, 1], ["Amycus Carrow", 40, 90, 1],
    ["Walburga Black", 20, 95, 1], ["Orion Black", 30, 85, 1], ["Cygnus Black", 35, 80, 1],
    ["Druella Rosier", 25, 85, 1], ["Evan Rosier", 70, 80, 1], ["Wilkes", 65, 75, 1],
    ["Barone Sanguinario", 50, 85, 1], ["Serp S. 01", 30, 95, 1], ["Serp S. 02", 45, 82, 1],
    ["Serp S. 03", 20, 78, 1], ["Serp S. 04", 55, 88, 1], ["Serp S. 05", 40, 92, 1], ["Serp S. 06", 35, 85, 1],
    ["Serp S. 07", 25, 70, 1], ["Serp S. 08", 50, 98, 1], ["Serp S. 09", 15, 88, 1], ["Serp S. 10", 42, 75, 1],
    ["Serp S. 11", 38, 80, 1], ["Serp S. 12", 28, 93, 1], ["Serp S. 13", 60, 85, 1], ["Serp S. 14", 48, 77, 1],
    ["Serp S. 15", 33, 89, 1], ["Serp S. 16", 22, 96, 1], ["Serp S. 17", 52, 81, 1], ["Serp S. 18", 41, 74, 1],
    ["Serp S. 19", 18, 86, 1], ["Serp S. 20", 36, 90, 1], ["Serp S. 21", 58, 83, 1], ["Serp S. 22", 44, 79, 1],
    ["Serp S. 23", 26, 91, 1], ["Serp S. 24", 39, 87, 1], ["Serp S. 25", 54, 94, 1], ["Serp S. 26", 47, 72, 1],
    ["Serp S. 27", 31, 84, 1],

    # --- TASSOROSSO (66 Personaggi) ---
    ["Cedric Diggory", 85, 50, 2], ["Nymphadora Tonks", 90, 30, 2], ["Pomona Sprite", 70, 40, 2],
    ["Newt Scamander", 80, 20, 2], ["Helga Tassorosso", 75, 50, 2], ["Hannah Abbott", 55, 40, 2],
    ["Susan Bones", 60, 35, 2], ["Ernie Macmillan", 65, 55, 2], ["Justin Finch-Fletchley", 55, 45, 2],
    ["Zacharias Smith", 40, 60, 2], ["Leanne", 50, 30, 2], ["Silvanus Kettleburn", 85, 20, 2],
    ["Teddy Lupin", 80, 30, 2], ["Theseus Scamander", 90, 50, 2], ["Bridget Wenlock", 40, 60, 2],
    ["Grogan Stump", 50, 70, 2], ["Artemisia Lufkin", 60, 75, 2], ["Dugald McPhail", 55, 50, 2],
    ["Fat Friar", 40, 30, 2], ["Newton Scamander", 82, 25, 2], ["Wayne Hopkins", 52, 38, 2],
    ["Megan Jones", 58, 42, 2], ["Eloise Midgen", 45, 20, 2], ["Eleanor Branstone", 50, 30, 2],
    ["Owen Cauldwell", 48, 35, 2], ["Laura Madley", 53, 40, 2], ["Kevin Whitby", 56, 44, 2],
    ["Rose Zeller", 51, 32, 2], ["Hepzibah Smith", 30, 75, 2], ["Hengist of Woodcroft", 65, 45, 2],
    ["Tass S. 01", 62, 48, 2], ["Tass S. 02", 55, 52, 2], ["Tass S. 03", 48, 30, 2], ["Tass S. 04", 70, 55, 2],
    ["Tass S. 05", 58, 42, 2], ["Tass S. 06", 65, 38, 2], ["Tass S. 07", 52, 60, 2], ["Tass S. 08", 44, 35, 2],
    ["Tass S. 09", 68, 50, 2], ["Tass S. 10", 61, 41, 2], ["Tass S. 11", 54, 65, 2], ["Tass S. 12", 49, 28, 2],
    ["Tass S. 13", 63, 44, 2], ["Tass S. 14", 57, 58, 2], ["Tass S. 15", 46, 33, 2], ["Tass S. 16", 66, 49, 2],
    ["Tass S. 17", 59, 36, 2], ["Tass S. 18", 53, 62, 2], ["Tass S. 19", 42, 31, 2], ["Tass S. 20", 67, 45, 2],
    ["Tass S. 21", 60, 53, 2], ["Tass S. 22", 51, 39, 2], ["Tass S. 23", 45, 56, 2], ["Tass S. 24", 64, 40, 2],
    ["Tass S. 25", 56, 47, 2], ["Tass S. 26", 43, 34, 2], ["Tass S. 27", 69, 51, 2], ["Tass S. 28", 50, 43, 2],
    ["Tass S. 29", 55, 66, 2], ["Tass S. 30", 47, 29, 2], ["Tass S. 31", 61, 46, 2], ["Tass S. 32", 52, 37, 2],
    ["Tass S. 33", 58, 59, 2], ["Tass S. 34", 44, 32, 2], ["Tass S. 35", 65, 42, 2], ["Tass S. 36", 54, 50, 2]
]
nomi_case = ["Grifondoro", "Serpeverde", "Tassorosso"]

def train_test_split(X, y, names, train_ratio=0.8):
    """Mischia e divide il dataset in train e test set."""
    # Creiamo gli indici e mischiamoli
    indices = np.arange(len(y))
    np.random.seed(42) # Per coerenza nei test
    np.random.shuffle(indices)
    
    # Applichiamo lo shuffle
    X, y, names = X[indices], y[indices], names[indices]
    
    # Calcoliamo il punto di taglio
    split = int(len(y) * train_ratio)
    
    return (X[:split], y[:split], names[:split]), (X[split:], y[split:], names[split:])

def get_prepared_data():
    """Estrae nomi, X normalizzata con bias e y."""
    nomi = np.array([row[0] for row in data_hogwarts])
    X = np.array([row[1:3] for row in data_hogwarts], dtype=float)
    y = np.array([row[3] for row in data_hogwarts])
    
    # 1. Normalizzazione (0-1)
    X /= 100.0
    
    # 2. Assorbimento Bias (Aggiunta colonna di 1 a sinistra)
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack((ones, X))
    
    return X_aug, y, nomi, nomi_case

