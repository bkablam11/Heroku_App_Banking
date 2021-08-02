> ### Descriptif
Un responsable d’une banque souhaite ``réduire le nombre de clients qui quittent leurs services de carte de crédit``. Mettre en place un modèle de Machine Learning capable de ``prédire les départs des clients``.La variable cible est ``Attrition_Flag.``
Description du jeu de donnée :

1. CLIENTNUM Numéro client. Identifiant unique du client titulaire du compte
2. Attrition_Flag Variable cible (activité client) - si le compte est fermé, 1 sinon 0
3. Customer_Age Âge du client en années
4. Gender Sexe du client (M = Homme, F = Femme)
5. Dependent_count Nombre de personnes à charge
6. Education_Level Niveau d'éducation
7. Marital_Status situation maritale
8. Income_Category Catégorie de revenu annuel
9. Card_Category Type de carte
10. Months_on_book Période de relation avec la banque
11. Total_Relationship_Count Nombre total de produits détenus par le client
12. Months_Inactive_12_mon Nombre de mois d'inactivité au cours des 12 derniers mois
13. Contacts_Count_12_mon Nombre de contacts au cours des 12 derniers mois
14. Credit_Limit Limite de crédit sur la carte de crédit
15. Total_Revolving_Bal Solde renouvelable total sur la carte de crédit
16. Avg_Open_To_Buy Ligne de crédit ouverte à l'achat (moyenne des 12 derniers mois)
17. Total_Amt_Chng_Q4_Q1 Changement du montant de la transaction (T4 par rapport au T1)
18. Total_Trans_Amt Montant total de la transaction (12 derniers mois)
19. Total_Trans_Ct Nombre total de transactions (12 derniers mois)
20. Total_Ct_Chng_Q4_Q1 Changement du nombre de transactions (T4 par rapport au T1)
21. Avg_Utilization_Ratio Taux d'utilisation moyen de la carte

> ### Premières Observations

- Apparament la **colonne** ``CLIENTNUM`` n'a pas vraiment d'impact sur la Variable ``Attrition_Flag``
- La variable **cible** ``Attrition_Flag`` doit être codifié en 1: Existing Customer et 0: Attrited Customer	
- Les variables **quantitatives** sont: ``Customer_Age``, ``Dependent_count``, ``Total_Relationship_Count``, (``Months_Inactive_12_mon``, ``Contacts_Count_12_mon ``),``Credit_Limit``, ``Total_Revolving_Bal ``,``Avg_Open_To_Buy``, `` Total_Amt_Chng_Q4_Q1 ``,``Total_Trans_Amt``, ``Total_Trans_Ct```,`Total_Ct_Chng_Q4_Q1``,``Avg_Utilization_Ratio``, ``Months_on_book``
- Les variables **qualitatives** sont: ``Gender``, ``Education_Level``, ``Marital_Status``, ``Income_Category``, ``Card_Category``
- La variable ``Income_Category`` doit être subdiviser en classe, 1: 60K-80K etc

Nous avons donc suivant les lignes : 
- **Education_Level**(1519 valeurs manquantes)
- **Marital_Status** (749 valeurs manquantes)
- **Income_Category** (1112 valeurs manquantes)

Aussi suivant les colonnes.

Unknown est considérée comme les données manquantes.
Supprimer les données manquantes par lignes et par la suite par colonnes pour avoir plus de données

>## Techniques de Codification

A. *With gradation* 
1. **Modality of Attrition_Flag**:
  - Existing Customer == 1 
  - Attrited Customer == 0

2. **Modality of Education_Level**:
      - Uneducated == Pas scolarisé = 0  
      - College == université  = 1
      - High School = lycée  = 2
      - Graduate = diplômé = 3
      - Post-Graduate = Études supérieures = 4 
      - Doctorate = doctorat = 5

3. **Modality of Income_Category**:
    - Less than $40K == 0
    
    - $40K à $60K == 1   
    
    - $80K à $120K == 2   
    
    - $60K à $80K  ==3     
    
    - $120K à plus ==4  
    
4. **Modality of Card_Category**:
    - Blue == 0
    - Silver == 1
    - Gold == 2
    - Platinum == 3

B. *Without gradation (get_dummy)* 
   - Modality of Gender
   - Modality of Marital_Status

``Avg_Utilization_Ratio`` doit etre ``entiere``