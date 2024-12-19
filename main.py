import cv2
import os
import numpy as np

# Fonction pour comparer deux images en utilisant SIFT
def matchingImg(img1, img2):
    sift = cv2.SIFT_create()

    keypointsA, descriptorsA = sift.detectAndCompute(img1, None)
    keypointsB, descriptorsB = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptorsA, descriptorsB, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    return good_matches

# Fonction pour comparer une image test avec un ensemble d'images classées par compagnie et retourne les 3 images qui correspondent le plus
def compare_match(imgTest, avions_images):
    resultats = {}

    for avion, images in avions_images.items():
        total_matches = 0
        scores_images = []
        for image_path in images:
            imgAvion = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            good_matches = matchingImg(imgTest, imgAvion)
            total_matches += len(good_matches)
            scores_images.append((image_path, len(good_matches)))
        
        resultats[avion] = (total_matches, sorted(scores_images, key=lambda x: x[1], reverse=True)[:3])

    return resultats

# Fonction pour trouver la compagnie ayant le meilleur score de correspondances
def findBestCompagnie(resultats, seuil_minimum=70):
    meilleur_compagnie = None
    meilleur_score = 0
    meilleures_images = []

    for avion, (score, top_images) in resultats.items():
        if score > meilleur_score and score >= seuil_minimum:
            meilleur_compagnie = avion
            meilleur_score = score
            meilleures_images = top_images

    return meilleur_compagnie, meilleur_score, meilleures_images

# Fonction pour lister toutes les images dans un dossier    
def list_test(test_folder):
    return [f for f in os.listdir(test_folder)]

# Fonction pour demander à l'utilisateur de choisir quelle image tester
def getChoice(test_folder):
    print("Choisissez une option :")
    print("1. Choisir une image parmi le dossier test")
    print("2. Entrer le chemin d'une image personnelle")
    
    choice = input("Votre choix (1/2) : ")
    if choice == '1':
        test_images = list_test(test_folder)
        if not test_images:
            print("Aucune image trouvee dans le dossier de test.")
            return None
        
        print("Images disponibles dans le dossier de test :")
        for i, img in enumerate(test_images):
            print(i + 1, img)
        
        img_choice = int(input("Entrez le numero de l'image : ")) - 1
        if 0 <= img_choice < len(test_images):
            return os.path.join(test_folder, test_images[img_choice])
        else:
            print("Choix invalide.")
            return None
    elif choice == '2':
        custom_path = input("Entrez le chemin complet de votre image : ")
        if os.path.isfile(custom_path):
            return custom_path
        else:
            print("Chemin invalide. Veuillez verifier le chemin d'acces.")
            return None
    else:
        print("Choix invalide.")
        return None

def main():
    avions = {
        "Airfrance": ["images/Airfrance/Airfrance1.jpg", "images/Airfrance/Airfrance2.jpg", "images/Airfrance/Airfrance3.jpg", "images/Airfrance/Airfrance4.jpg", "images/Airfrance/Airfrance5.jpg", "images/Airfrance/Airfrance6.jpg", "images/Airfrance/Airfrance7.jpg", "images/Airfrance/Airfrance8.jpg", "images/Airfrance/Airfrance9.png"],
        "Klm": ["images/Klm/Klm1.jpg", "images/Klm/Klm2.jpg", "images/Klm/Klm3.jpg", "images/Klm/Klm4.jpg", "images/Klm/Klm5.jpg", "images/Klm/Klm6.jpg", "images/Klm/Klm8.jpg", "images/Klm/Klm9.jpg"],
        "Easyjet": ["images/Easyjet/Easyjet1.jpg", "images/Easyjet/Easyjet2.jpg", "images/Easyjet/Easyjet3.jpg", "images/Easyjet/Easyjet4.png", "images/Easyjet/Easyjet5.jpg", "images/Easyjet/Easyjet6.jpg", "images/Easyjet/Easyjet7.jpg", "images/Easyjet/Easyjet8.jpg", "images/Easyjet/Easyjet9.png"],
        "Koreanair": ["images/Koreanair/Koreanair1.png", "images/Koreanair/Koreanair2.jpg", "images/Koreanair/Koreanair3.jpg", "images/Koreanair/Koreanair4.jpg", "images/Koreanair/Koreanair5.jpg", "images/Koreanair/Koreanair6.jpg", "images/Koreanair/Koreanair7.jpg", "images/Koreanair/Koreanair8.jpg", "images/Koreanair/Koreanair9.jpg"],
        "Ryanair": ["images/Ryanair/Ryanair1.png", "images/Ryanair/Ryanair2.jpg", "images/Ryanair/Ryanair3.jpg", "images/Ryanair/Ryanair4.jpg", "images/Ryanair/Ryanair5.jpg", "images/Ryanair/Ryanair6.jpg", "images/Ryanair/Ryanair7.jpg", "images/Ryanair/Ryanair8.png", "images/Ryanair/Ryanair9.jpg"]
    }

    test_folder = "images/test"  
    image_path = getChoice(test_folder)
    
    if image_path is None:
        print("Aucune image valide n'a ete selectionnee.")
        return
    
    print("\nTest de l'image :", image_path)

    imgTest = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if imgTest is None:
        print("Erreur : Impossible de lire l'image.")
        return

    resultats = compare_match(imgTest, avions)
    meilleur_compagnie, meilleur_score, top_images = findBestCompagnie(resultats)

    if meilleur_compagnie:
        print("La compagnie aerienne la plus proche semble etre :", meilleur_compagnie, " avec un score de", meilleur_score)
        
        for img_path, score in top_images:
            img = cv2.imread(img_path)
            cv2.imshow(img_path, img)

        cv2.imshow("Image d'origine", imgTest)
        cv2.waitKey(0)
    else:
        print("Aucune compagnie aerienne ne semble correspondre, ou le score est trop bas.")

main()